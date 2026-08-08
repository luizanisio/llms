# Unfreeze Progressivo e Modo Fusão — Guia Prático

Guia de configuração das estratégias de **progressão de capacidade** (descongelamento
progressivo) e **execução contínua** (modo fusão) do framework de treinamento.

**Público:** pesquisadores configurando novos treinamentos. Este guia cobre as decisões de
design, os dois modos de execução, exemplos YAML completos e checklists de verificação.

**Arquivos-chave do framework:**
- [`treinar_unsloth_fusao.py`](file:///mnt/d/wsl_dev/llms/src/treinar_unsloth_fusao.py) — `ConfigFusao`, validador, dataset fundido, grupos de gating, `FusaoTrainer`, callback de marcadores, guarda de hash do resume.
- [`treinar_unsloth.py`](file:///mnt/d/wsl_dev/llms/src/treinar_unsloth.py) — `_aplicar_unfreeze_parcial()` (congelamento real), `_aplicar_regime_fusao()` (regime único do run fundido), `_train_fundido()` (orquestração).
- [`treinar_unsloth_pipeline.py`](file:///mnt/d/wsl_dev/llms/src/treinar_unsloth_pipeline.py) — `EtapaCurriculum` (campos `unfreeze_layers_from` / `unfreeze_layers_pct`), parse dual, whitelist `_CHAVES_ETAPA_VALIDAS`.
- [`treinar_model_loader.py`](file:///mnt/d/wsl_dev/llms/src/treinar_model_loader.py) — `ModelLoader` (carregamento base, LoRA, merge).

---

## 1. Qual modo usar? (Fluxograma de decisão)

```
Precisa misturar LoRA e Full FT no mesmo run?
  ├─ SIM → SEGMENTADO (único modo que suporta transição de regime)
  └─ NÃO
       Quer medir/controlar cada etapa isoladamente?
       (eval separado, pace_loss adaptativo, checkpoints por etapa)
         ├─ SIM → SEGMENTADO
         └─ NÃO
              Quer CL/PT SEM custo de fronteira?
              (scheduler contínuo, momentos Adam preservados, LR discriminativo emergente)
                ├─ SIM → FUNDIDO (fusao.ativo: true)
                └─ NÃO → SEGMENTADO (mais simples, padrão)
```

**Resumo da regra prática:**
- **SEGMENTADO** → quando cada etapa é um experimento controlável (eval, pacing, regimes mistos).
- **FUNDIDO** → quando o objetivo é continuidade total (CL e/ou PT sem descontinuidades).

---

## 2. Comparação detalhada: SEGMENTADO vs. FUNDIDO

| Aspecto | **SEGMENTADO** (padrão) | **FUNDIDO** (`fusao.ativo: true`) |
|---|---|---|
| Execução | Um `trainer.train()` por etapa | UM único `trainer.train()` para todas as etapas |
| Fronteiras entre etapas | **Reais:** otimizador e scheduler reiniciam | **Virtuais:** apenas marcadores de log/gráfico; nada reinicia |
| Scheduler | Um cosine por etapa (reinicia no pico a cada fronteira) | Um único cosine do pico a ~0 no run inteiro |
| Momentos Adam | Zerados a cada fronteira | Contínuos; grupos "adormecidos" acumulam mesmo com LR=0 |
| `unfreeze_layers_from` | **Congelamento real** (`requires_grad=False`) — só `tipo: "full"` | **Gating de LR** (nada congela) — funciona em `lora` e `full` |
| Parâmetros por etapa | `tipo`, `learning_rate`, `batch_size`, `max_seq_length`, `warmup_steps` permitidos | **Proibidos** nas etapas (são globais em `fusao`/`treinamento`) |
| `pace_loss` / `pace_epochs_max` | Suportados (pacing adaptativo) | **Incompatíveis** (dataset é pré-materializado) |
| Eval | Por etapa + global | Somente global |
| Melhor checkpoint | Melhor da etapa, selecionado a cada etapa | Melhor global do run inteiro |
| Resume | `curriculum_state.json` + checkpoints HF | Checkpoint HF padrão + guarda de hash da fusão |

---

## 3. `unfreeze_layers_from` — a chave de configuração

### 3.1 Os dois formatos aceitos

```yaml
unfreeze_layers_from: 21     # absoluto: treina blocos 21..N-1 (+ norm/cabeça)
unfreeze_layers_from: 75%    # percentual: congela/adormece os primeiros 75% dos blocos
```

O percentual é resolvido em runtime contra `model.config.num_hidden_layers` — portanto, o
**mesmo YAML funciona em modelos de tamanhos diferentes**. No Qwen 2.5 (tanto 1.5B quanto 7B,
ambos com 28 blocos), `75%` resolve para o bloco 21 → 7 camadas finais ativas.

> **Nota YAML:** `unfreeze_layers_from: 75%` sem aspas é lido como string `"75%"` pelo
> `yaml.safe_load` (o `%` impede a interpretação numérica). Aspas são opcionais.

### 3.2 No modo SEGMENTADO → congelamento real

Aplica-se **apenas** a etapas `tipo: "full"` (em LoRA a capacidade é controlada pelo rank;
o validador rejeita a combinação antes de alocar GPU).

**Como funciona:**
- Blocos com índice < corte ficam com `requires_grad=False`: não recebem gradiente nem entram
  no otimizador.
- Transição entre etapas full consecutivas é ***function-preserving*** (nenhum peso muda na
  fronteira; apenas mais blocos passam a ter `requires_grad=True`).
- `0` ou `0%` = full tradicional, tudo treinável.

**Política de congelamento:**

| Grupo de parâmetros | Regra |
|---|---|
| Bloco `layers.N.*` | Treinável se `N >= from_layer` |
| `embed_tokens` e `lm_head` | Treináveis **apenas** se `from_layer == 0` |
| Demais float (norm final, etc.) | Sempre treináveis |
| Parâmetros quantizados (int4/int8) | Intocados — não suportam gradiente |

**Tied embeddings (Qwen 1.5B):** `embed_tokens` e `lm_head` compartilham um único tensor
(`tie_word_embeddings: true`). No 7B são tensores separados (untied). A política congela
ambos até o corte 0 nos dois casos, mantendo a progressão de capacidade comparável:

| Corte | Qwen 2.5 1.5B (tied) | Qwen 2.5 7B (untied) |
|---|---|---|
| `75%` | ~21,2% dos params | ~21,4% dos params |
| `50%` | ~42,4% | ~42,8% |
| `25%` | ~63,7% | ~64,3% |
| `0%` | 100% | 100% |

> O gradiente flui normalmente **através** da cabeça congelada para os blocos superiores —
> isso é esperado e não impede o treinamento.

**Recomendações para o modo segmentado:**
- `warmup_steps: 100` por etapa (cada etapa reconstrói o otimizador com momentos zerados; a
  rampa reconstrói o pré-condicionador antes de passos grandes).
- `max_grad_norm: 0.3` (aperta o clip para estabilizar os primeiros steps após cada
  descongelamento).
- `nbits: 16` em todas as etapas (Full FT não suporta quantização).

### 3.3 No modo FUNDIDO → gating de LR

Válido com `fusao.tipo: "lora"` **e** `"full"`.

**Como funciona:**
- **Nada é congelado.** Todos os parâmetros treináveis entram no otimizador no step 0.
- São organizados em **grupos por faixa de camadas** (derivados dos cortes distintos declarados;
  cabeça+norm sempre no primeiro grupo).
- Cada grupo tem uma LR controlada por duas funções:

  ```
  LR_grupo(t) = learning_rate × cosine_global(t) × gate(t)
  ```

  onde `gate(t)` vale 0 até o início do span que declara o corte, sobe linearmente por
  `warmup_grupo_steps` e satura em 1.

**Consequências práticas:**

1. **LR discriminativo emergente:** grupos que acordam tarde rampam até o valor **já decaído**
   do cosine → LR naturalmente menor para camadas liberadas tarde, sem precisar configurar LR
   por etapa. No smoke test d_mini_fusao, o grupo `gate_0%` satura em ~2.42e-06 — praticamente
   o 3e-06 que o d_mini_uf configurava manualmente na última etapa.

2. **Momentos Adam aquecidos:** grupos adormecidos acumulam `exp_avg`/`exp_avg_sq` (LR=0 ⇒ o
   peso não se move e não sofre weight decay, mas o pré-condicionador "aquece"). Ao acordar,
   o grupo não parte de estatísticas zeradas.

3. **Em `lora`:** o gating agrupa os **adapters** por índice de camada; a base segue congelada
   como sempre. **Em `full`:** agrupa os pesos plenos; `embed_tokens` (tied) pertence ao grupo
   do corte `0` (se nenhum span declarar `0`, embeddings não treinam no run).

**Regras de validação:**
- Os cortes devem ser **monotônicos não crescentes** ao longo dos spans (ex.: 75% → 50% → 25% → 0%).
- A **primeira etapa** deve declarar um corte (define o grupo ativo desde o step 0).
- "Re-congelar" (subir o percentual) **não é suportado**.

**Custo:** o backward computa gradientes de **todos** os parâmetros desde o step 0 (LR=0 não
economiza compute). Irrelevante em LoRA; aceitável em FF no 1.5B.

### 3.4 Por que o LR zero simula o congelamento — e onde a simulação difere

No modo segmentado com `tipo: "full"`, o congelamento é **real**: `requires_grad=False` exclui os parâmetros do grafo computacional, de modo que nenhum gradiente é calculado, nenhum estado de otimizador é alocado e nenhuma atualização de peso ocorre — o parâmetro é tratado como constante pelo autograd. Na transição entre etapas, novos parâmetros são descongelados e um **novo otimizador** é instanciado, o que implica momentos Adam zerados e a necessidade de warmup para reconstruir o pré-condicionador. No modo fundido, o mecanismo é diferente: todos os parâmetros treináveis entram no otimizador desde o step 0, mas o gate multiplica a learning rate do grupo por zero (`gate(t)=0`), o que anula a atualização `θ ← θ − lr·m̂/√v̂` sem alterar o cálculo de `m̂` e `v̂`. Isso significa que o backward ainda computa gradientes para esses parâmetros (custo de compute idêntico ao full sem freeze), os momentos do Adam acumulam estatísticas mesmo durante o "sono", e ao acordar o grupo já possui um pré-condicionador aquecido — eliminando a descontinuidade de fronteira que o segmentado introduz. Em LoRA, a distinção é mais sutil: no segmentado, o congelamento real de camadas não se aplica (o validador rejeita `unfreeze_layers_from` em etapas LoRA, pois a capacidade já é restrita pelo rank dos adapters); no fundido, o gating de LR agrupa os **adapters** por índice de camada do modelo base, permitindo o descongelamento progressivo dos adapters sem tocar na base — algo que o modo segmentado não suporta com LoRA. A consequência prática é que o modo fundido sacrifica economia de memória (todos os estados de otimizador existem desde o início) em troca de continuidade total na otimização.

---

## 4. O bloco `fusao` — referência de configuração

O bloco `fusao` é irmão de `divisao`, dentro de `curriculum`:

```yaml
curriculum:
  fusao:
    ativo: true               # liga o modo fundido (ausente/false = segmentado)
    tipo: "lora"              # regime ÚNICO do run: "lora" | "full"
    learning_rate: 2e-05      # pico do cosine global (LoRA típico 2e-5; FF típico 5e-6)
    warmup_grupo_steps: 100   # rampa de cada grupo de camadas ao acordar (gating)
    seed_shuffle: 3407        # shuffle intra-span reprodutível (afeta resume)
  divisao:
  - arquivo: .../divisao.csv
    dataset_filtro: {"dificuldade": "facil"}
    alias: "fácil"
    pace_epochs: 2            # ÉPOCAS VIRTUAIS: o span entra 2× no stream,
                              # com shuffle independente em cada repetição
    unfreeze_layers_from: 75% # opcional: ativa gating (ver §3.3)
  - arquivo: .../divisao.csv
    alias: "completo"
    pace_epochs: 2
    unfreeze_layers_from: 0%
```

### 4.1 Tabela de parâmetros

| Chave | Efeito | Observações |
|---|---|---|
| `fusao.ativo` | Liga o executor fundido | Sem ele, cada etapa tem um treino independente |
| `fusao.tipo` | Regime único do run | Misturar LoRA e FF exige modo segmentado |
| `fusao.learning_rate` | Pico do cosine único | Sobrepõe `treinamento.learning_rate`; **proibido** nas etapas |
| `fusao.warmup_grupo_steps` | Rampa dos grupos no gating | Sem efeito se nenhum span declara unfreeze; ajustar ao nº de steps do run |
| `fusao.seed_shuffle` | Ordem determinística do stream | Mudar entre queda e resume dispara a guarda de hash |
| `pace_epochs` (por span) | Multiplica o span | Total de instâncias = Σ(span × pace) |
| `treinamento.warmup_steps` | Warmup **GLOBAL** no início do run | É a rampa do primeiro grupo (cosine) |
| `treinamento.num_train_epochs` | **Ignorado** no fundido | O executor treina 1 época real sobre o stream; repetições vêm de `pace_epochs` |
| `treinamento.eval_steps` | Frequência do eval global | Recomendado `5%` a `10%` (~10-20 avaliações) para caracterizar os marcadores |
| `treinamento.nbits` | Precisão do run | Recomendado 16 para full; com 4 o log emite aviso |

### 4.2 Artefatos gerados pelo modo fundido

| Artefato | Descrição |
|---|---|
| `fusao_spans.json` | Metadados dos spans + hash de configuração (insumo de relatórios, callback e guarda de resume) |
| `treinamento_eventos.md` | Eventos `SPAN INICIADO (virtual)` ao cruzar cada fronteira |
| `treinamento_lr_grupos.png` | Curvas de LR por grupo — evidência visual do gating |
| `fusao_lr_grupos.jsonl` | Dados brutos de LR por grupo a cada step de log |
| Gráficos do treinamento | Linhas violeta = marcadores virtuais (derivados de `fusao_spans.json`) |

### 4.3 Checkpoint e resume

Funciona com o mecanismo padrão do HF (`resume_from_checkpoint: true`). O framework salva
`fusao_spans.json` com um hash SHA-256 da configuração efetiva (tipo, LR, warmup, seed,
batch efetivo, spans). No resume:

1. O hash é recomputado e comparado com o salvo.
2. Se divergente → o resume é **recusado** com mensagem clara (scheduler e grupos seriam
   inconsistentes com o estado salvo).
3. O dataloader retoma por **fast-forward determinístico** até a posição exata do stream
   (minutos, sem GPU).

> `curriculum_state.json` **NÃO** é usado no modo fundido.

---

## 5. Orçamento de VRAM

### 5.1 Modo segmentado (unfreeze real)

No segmentado, as primeiras etapas (com muitos blocos congelados) usam **menos VRAM** porque
gradientes e estados do otimizador existem apenas para os blocos descongelados. O pico ocorre
na última etapa (0% congelado, tudo treinável).

**Exemplo para Qwen 2.5 1.5B (Full FT 16 bits, `adamw_8bit`):**

| Corte | Params treináveis | Custo estático aprox. |
|---|---|---|
| `75%` | 327M (~21%) | ~4.5 GiB |
| `50%` | 655M (~42%) | ~6.0 GiB |
| `0%` | 1.54B (100%) | ~8.6 GiB |

Custo dinâmico (logits): `batch × seq × vocab(151936)` em bf16 + upcast fp32 da loss.
Com batch=1: seq 1024 → +0.87 GiB; seq 1536 → +1.30 GiB; seq 2048 → +1.74 GiB.

### 5.2 Modo fundido (gating de LR)

No fundido, **nada é congelado**: todos os parâmetros entram no otimizador desde o step 0.
O pico do segmentado (etapa 0%) vira o custo de **todo o run**.

- **LoRA fundido:** custo irrelevante do gating (só os adapters entram no otimizador).
- **Full FT fundido:** custo estático sustentado do primeiro ao último step.

**Se estourar VRAM:**
1. Reduza `max_seq_length` (reduz os logits).
2. Troque `fusao.tipo` para `"lora"` (custo do gating desprezível).
3. Use `adafactor` em vez de `adamw_8bit` (elimina os momentos por parâmetro).

---

## 6. Receitas completas (YAML copiável)

### 6.1 Full FT com unfreeze progressivo — SEGMENTADO

Caso de uso: CL alinhado à capacidade com etapas controláveis (D19/D20).

```yaml
curriculum:
  divisao:
  - arquivo: dados/divisao.csv
    dataset_filtro: {"dificuldade": "facil"}
    alias: "fácil-uf75pct"
    tipo: "full"
    unfreeze_layers_from: 75%  # blocos 21-27 (25% finais)
    pace_epochs: 2
    batch_size: 1
    learning_rate: 5e-06
  - arquivo: dados/divisao.csv
    dataset_filtro: {"dificuldade": "medio"}
    alias: "médio-uf50pct"
    tipo: "full"
    unfreeze_layers_from: 50%  # blocos 14-27 (50% finais)
    pace_epochs: 2
    batch_size: 1
    learning_rate: 5e-06
  - arquivo: dados/divisao.csv
    dataset_filtro: {"dificuldade": "dificil"}
    alias: "difícil-uf25pct"
    tipo: "full"
    unfreeze_layers_from: 25%  # blocos 7-27 (75% finais)
    pace_epochs: 2
    batch_size: 1
    learning_rate: 5e-06
  - arquivo: dados/divisao.csv
    alias: "completo-uf0pct"
    tipo: "full"
    unfreeze_layers_from: 0%   # tudo treinável (inclui embeddings/lm_head)
    pace_epochs: 2
    batch_size: 1
    learning_rate: 3e-06       # LR menor na capacidade plena
    warmup_steps: 10           # override por etapa (retorna ao global na seguinte)

treinamento:
  nbits: 16                    # Full FT: sempre 16 bits
  max_grad_norm: 0.3           # estabiliza steps pós-descongelamento
  warmup_steps: 100            # rampa após cada reset de otimizador
  lr_scheduler_type: cosine
  optim: adamw_8bit
  liger_kernel: true
  flash_attention_2: true
  full_com_sdpa: true          # todas as etapas são full → SDPA seguro

lora:
  r: 0                         # r=0 garante tipo_padrao="full"
```

**O que esperar nos logs:**
```
📋 Curriculum: 4 etapa(s) configurada(s)
   [0] alias='fácil-uf75pct', tipo=full, epochs=2, unfreeze_from=75%
   [1] alias='médio-uf50pct', tipo=full, epochs=2, unfreeze_from=50%
   [2] alias='difícil-uf25pct', tipo=full, epochs=2, unfreeze_from=25%
   [3] alias='completo-uf0pct', tipo=full, epochs=2, unfreeze_from=0%
```

No início de cada etapa:
```
🧊 Unfreeze 75% congelado (7 camadas finais treináveis ≈ 25%): blocos 21-27 de 28
ℹ️  Âncora de conhecimento pré-treinado: tensor único tied embed_tokens/lm_head
    permanece(m) CONGELADO(S) até unfreeze_layers_from: 0.
🧊 Unfreeze parcial: blocos >= 21 treináveis (327.586.304 params) |
    blocos congelados: 982.752.768 | embeddings+cabeça congelados: 233.373.696
🔓 Modo FULL: 327.586.304/1.543.714.304 parâmetros desbloqueados
```

**Checklist de verificação:**
- [ ] Contador `Modo FULL: X/Y` **cresce** etapa a etapa
- [ ] **Sem** log de `🔄 Descarregando e recarregando o modelo` (transição in-memory)
- [ ] Loss desce nos primeiros ~50 steps de cada etapa
- [ ] Sem warning de gradiente nulo

---

### 6.2 CL sem custo de fronteira, LoRA — FUNDIDO

Caso de uso: CL puro, sem progressão de capacidade (D21). Uma única curva de LR cosine.

```yaml
curriculum:
  fusao:
    ativo: true
    tipo: "lora"
    learning_rate: 2e-05
    seed_shuffle: 3407
  divisao:
  - arquivo: dados/divisao.csv
    dataset_filtro: {"dificuldade": "facil"}
    alias: "fácil"
    pace_epochs: 2             # cada span entra 2× no stream
  - arquivo: dados/divisao.csv
    dataset_filtro: {"dificuldade": "medio"}
    alias: "médio"
    pace_epochs: 2
  - arquivo: dados/divisao.csv
    alias: "completo"
    pace_epochs: 1

treinamento:
  eval_steps: 5%
  num_train_epochs: 1          # ignorado (1 época real sobre o stream)
  learning_rate: 2e-05         # sobreposto por fusao.learning_rate
  warmup_steps: 100

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

**Checklist de verificação:**
- [ ] Uma **única** curva de LR (cosine limpo, sem degraus)
- [ ] Marcadores virtuais nos gráficos (linhas violeta)
- [ ] Eventos `🪧 SPAN INICIADO (virtual)` no `treinamento_eventos.md`
- [ ] **Nenhum** log de reload, troca de etapa ou "Etapa X/N"
- [ ] Um único "melhor global" (todo o run compete pelo mesmo eval)

---

### 6.3 CL + PT fundidos, LoRA — FUNDIDO com gating

Caso de uso: CL + progressão de capacidade sem fronteiras (D22). Curvas de LR por grupo.

```yaml
curriculum:
  fusao:
    ativo: true
    tipo: "lora"
    learning_rate: 2e-05
    warmup_grupo_steps: 100    # rampa de cada grupo ao acordar
    seed_shuffle: 3407
  divisao:
  - arquivo: dados/divisao.csv
    dataset_filtro: {"dificuldade": "facil"}
    alias: "fácil"
    pace_epochs: 2
    unfreeze_layers_from: 75%  # G1: adapters das camadas 21-27
  - arquivo: dados/divisao.csv
    dataset_filtro: {"dificuldade": "medio"}
    alias: "médio"
    pace_epochs: 2
    unfreeze_layers_from: 50%  # G2 acorda: adapters das camadas 14-20
  - arquivo: dados/divisao.csv
    alias: "completo"
    pace_epochs: 1
    unfreeze_layers_from: 0%   # G4 acorda: adapters das camadas 0-6

treinamento:
  eval_steps: 5%
  warmup_steps: 100            # warmup global (rampa do G1)

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

**O que esperar nos logs:**
```
🧩 Grupos de gating (LR por faixa de camadas):
| Grupo     | Faixa de camadas | Parâmetros | Step de ativação | Rampa        |
|-----------|------------------|------------|------------------|--------------|
| gate_75%  | 21-27 (+cabeça)  | ...        | 0                | +100 steps   |
| gate_50%  | 14-20            | ...        | <step>           | +100 steps   |
| gate_0%   | 0-6              | ...        | <step>           | +100 steps   |
```

**Checklist de verificação:**
- [ ] Tabela `🧩 Grupos de gating` com o número correto de grupos e ativações
- [ ] `treinamento_lr_grupos.png` com curvas em degraus sob um único envelope cosine
- [ ] Nenhuma descontinuidade no loss nos marcadores virtuais
- [ ] Grupos tardios com LR de saturação menor que os primeiros (LR discriminativo emergente)

---

### 6.4 Full FT fundido com gating

Caso de uso: PT puro sem CL, ou CL + PT com regime full (D23 / smoke test d_mini_fusao).

```yaml
curriculum:
  fusao:
    ativo: true
    tipo: "full"
    learning_rate: 5e-06       # FF típico
    warmup_grupo_steps: 100    # ajustar para runs curtos (ver nota abaixo)
    seed_shuffle: 3407
  divisao:
  - arquivo: dados/divisao.csv
    dataset_filtro: {"dificuldade": "facil"}
    alias: "fácil-uf75pct"
    pace_epochs: 1
    unfreeze_layers_from: 75%
  - arquivo: dados/divisao.csv
    dataset_filtro: {"dificuldade": "medio"}
    alias: "médio-uf50pct"
    pace_epochs: 1
    unfreeze_layers_from: 50%
  - arquivo: dados/divisao.csv
    dataset_filtro: {"dificuldade": "dificil"}
    alias: "difícil-uf25pct"
    pace_epochs: 1
    unfreeze_layers_from: 25%
  - arquivo: dados/divisao.csv
    alias: "completo-uf0pct"
    pace_epochs: 1
    unfreeze_layers_from: 0%

treinamento:
  nbits: 16                    # obrigatório para full fundido
  eval_steps: 10%
  warmup_steps: 5
  max_grad_norm: 0.3
  optim: adamw_8bit            # ou adafactor para economizar VRAM
  liger_kernel: true
  flash_attention_2: true
  full_com_sdpa: true

lora:
  r: 16                        # declarado caso se queira trocar fusao.tipo para "lora"
```

> **Nota sobre `warmup_grupo_steps`:** em smoke tests com poucos optimizer steps (~50-60),
> use um valor proporcional (ex.: 3-5 steps) para que os gates de fato saturem dentro do run.
> O padrão de 100 é adequado para protocolos completos com milhares de steps.

**Checklist de verificação (full fundido):**
- [ ] Tabela `🧩 Grupos de gating` com 4 linhas (75%/50%/25%/0%)
- [ ] `treinamento_lr_grupos.png` com 4 curvas em degraus
- [ ] **Nenhum** log de reload ou troca de etapa
- [ ] VRAM estável no pico durante todo o run (sem o alívio das primeiras etapas do segmentado)

---

## 7. Erros de validação e como corrigi-los

| Mensagem (resumo) | Causa | Correção |
|---|---|---|
| "unfreeze_layers_from só é válido com tipo full" | Unfreeze em etapa LoRA no modo **segmentado** | Use `tipo: "full"` na etapa, ou ative a fusão (gating suporta LoRA) |
| "no modo fundido esses valores são globais" | `tipo`/`learning_rate`/`batch_size`/`max_seq_length`/`warmup_steps` declarados numa etapa com fusão ativa | Mover para `fusao`/`treinamento` |
| "pace_loss/pace_epochs_max incompatíveis com fusão" | Pacing adaptativo com dataset pré-materializado | Remover ou usar modo segmentado |
| "cortes devem ser monotônicos não crescentes" | Ex.: 50% depois 75% (re-congelar não é suportado) | Reordenar: cortes devem decrescer ao longo dos spans |
| "a primeira etapa deve declarar unfreeze_layers_from" | Gating sem grupo inicial definido | Declarar o corte no primeiro span (ex.: 75%) |
| "configuração de fusão mudou desde o checkpoint" | YAML editado entre queda e resume | Restaurar o YAML original ou apagar checkpoints |
| "chaves desconhecidas IGNORADAS" (aviso) | Typo em chave de etapa ou de `fusao` | Conferir grafia contra as chaves válidas listadas no aviso |
| "fusao.tipo deve ser 'lora' ou 'full'" | Valor inválido em `fusao.tipo` | Corrigir para `"lora"` ou `"full"` |
| "Modo FUSÃO 'lora' requer lora.r > 0" | `fusao.tipo: "lora"` com `lora.r: 0` | Definir `lora.r: 16` (ou outro rank > 0) |

---

## 8. Interpretação dos logs — exemplos concretos

### 8.1 Início do run fundido

```
[4/6] Modo FUSÃO: 4 etapa(s) como spans de um único treinamento (tipo=full)…
ℹ️  curriculum_state.json NÃO é usado no modo fundido
🔓 Regime FUSÃO 'full': 1.543.714.304/1.543.714.304 parâmetros treináveis
📚 Dataset fundido: 450 instâncias no stream, batch_efetivo=8 → ~56 optimizer steps
📊 Eval do modo fundido: somente GLOBAL (300 instâncias de validação)
```

### 8.2 Tabela de grupos de gating

```
🧩 Grupos de gating (LR por faixa de camadas):
| Grupo    | Faixa de camadas  | Parâmetros    | Step de ativação | Rampa      |
|----------|-------------------|---------------|------------------|------------|
| gate_75% | 21-27 (+cabeça)   | 327.587.840   | 0                | +3 steps   |
| gate_50% | 14-20             | 327.584.768   | 8                | +3 steps   |
| gate_25% | 7-13              | 327.584.768   | 18               | +3 steps   |
| gate_0%  | 0-6               | 561.055.232   | 28               | +3 steps   |
```

### 8.3 Marcadores virtuais durante o treinamento

```
🪧 SPAN INICIADO (virtual): 'médio-uf50pct' @ step 8 (corte=50)
🪧 SPAN INICIADO (virtual): 'difícil-uf25pct' @ step 18 (corte=25)
🪧 SPAN INICIADO (virtual): 'completo-uf0pct' @ step 28 (corte=0)
```

> O primeiro span não gera marcador (já está ativo desde o step 0).

### 8.4 Scheduler do gating

```
ℹ️  Scheduler do gating: LambdaLR = cosine global (pico 5e-06, warmup 5 steps) ×
    gate por grupo (rampa 3 steps) — o lr_scheduler_type 'cosine' do YAML não é usado.
```

---

## 9. Validação antes de runs longos

### 9.1 Dry-run com `--datasets`

```bash
python src/treinar_unsloth.py --datasets meu_config.yaml
```

Confirma o parse, a listagem das etapas/spans e o contrato de validação **sem alocar GPU**.

### 9.2 Smoke test rápido

Antes de submeter um protocolo completo à fila da H100, rode uma versão mini com:
- **Mesmo YAML**, apenas com `max_seq_length` e `pace_epochs` reduzidos
- Verificar os itens do checklist da receita correspondente
- Para testar o resume: matar o processo no meio e reexecutar → deve retomar sem erros

### 9.3 Validação específica do resume (modo fundido)

Para validar a guarda de hash: após uma interrupção, edite qualquer chave de `fusao` ou
`divisao` e reexecute → o resume deve ser **recusado** com "configuração de fusão mudou
desde o checkpoint".

---

## 10. Referência rápida: chaves aceitas por span no modo fundido

No modo fundido, cada item de `divisao` aceita **apenas** estas chaves:

| Chave | Obrigatória? | Descrição |
|---|---|---|
| `arquivo` | Sim | Caminho do CSV de divisão |
| `dataset_filtro` | Não | Filtro JSON inline sobre o dataset |
| `alias` | Sim | Nome descritivo do span |
| `pace_epochs` | Sim (≥ 1) | Quantas vezes o span entra no stream |
| `unfreeze_layers_from` | Não | Corte de gating (percentual ou absoluto) |

**Proibidas nos spans (são globais):** `tipo`, `learning_rate`, `batch_size`,
`max_seq_length`, `warmup_steps`, `pace_loss`, `pace_epochs_max`.