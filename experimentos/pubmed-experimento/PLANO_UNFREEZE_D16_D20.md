# Plano de Alteração — Descongelamento Progressivo (`unfreeze_layers_from`) + Protocolos D16–D20

**Escopo:** habilitar o parâmetro por etapa `unfreeze_layers_from: N` (válido apenas em `tipo: "full"`),
habilitar `warmup_steps` por etapa, e aplicar correções essenciais identificadas na revisão do código.

**Arquivos alterados:**
1. `treinar_unsloth_pipeline.py` — dataclass `EtapaCurriculum` + `construir_etapas()`
2. `treinar_unsloth.py` — `LLMsTrainer._aplicar_etapa_curriculum()`, novo método `_aplicar_unfreeze_parcial()`, `train()` e `_load_model()`

Nenhuma alteração é necessária em `treinar_unsloth_util.py` (as etapas são parseadas no pipeline;
`ConfigTreinamento.warmup_steps` já existe e já flui para o `SFTConfig` em `_build_trainer`, linha ~1834).

---

## Contexto arquitetural (por que a mudança é pequena)

- O loop de `train()` reconstrói o `SFTTrainer` **do zero a cada etapa** (`_build_trainer`), e o
  `Trainer` do HF cria o otimizador filtrando `p.requires_grad`. Portanto, basta ajustar
  `requires_grad` **antes** de `_build_trainer` — exatamente o que `_aplicar_etapa_curriculum` já faz
  para os modos `full`/`lora`. O unfreeze é um refinamento do ramo `full`.
- Entre etapas `full` consecutivas com o mesmo `nbits` (16), `precisa_recarregar` é `False`:
  o modelo **permanece em memória**, e a transição é *function-preserving* (nenhum peso muda na
  fronteira). É o comportamento desejado para D19/D20.

---

## 1. `treinar_unsloth_pipeline.py`

### 1.1 — `EtapaCurriculum`: dois campos novos

```python
@dataclass
class EtapaCurriculum:
    # ... campos existentes ...
    batch_size: int = 0            # 0 = usa valor global (treinamento.batch_size)
    dataset_filtro: Optional[Dict[str, Any]] = None

    # === NOVOS CAMPOS ===
    warmup_steps: int = -1         # -1 = usa valor global (treinamento.warmup_steps)
    unfreeze_layers_from: int = -1 # -1 = desativado. N >= 0 = treina apenas blocos transformer
                                   # com índice >= N (+ norm final e lm_head). N=0 equivale a
                                   # full completo (inclui embeddings). Válido apenas em tipo "full".
```

> Atenção: `warmup_steps` usa sentinela `-1` (e não `0`), porque `warmup_steps: 0` é um valor
> legítimo ("sem warmup nesta etapa").

### 1.2 — `construir_etapas()`: whitelist de chaves + parse + validação

Inserir a constante no topo do módulo (após os imports):

```python
# Chaves aceitas em cada item de curriculum.divisao (proteção contra typos e chaves mortas)
_CHAVES_ETAPA_VALIDAS = {
    "alias", "arquivo", "tipo", "pace_epochs", "pace_epochs_max", "pace_loss",
    "max_seq_length", "learning_rate", "batch_size", "dataset_filtro",
    "warmup_steps", "unfreeze_layers_from",
}
```

Dentro do `for i, item in enumerate(divisao_list):`, logo após validar que `item` é dict:

```python
        desconhecidas = set(item.keys()) - _CHAVES_ETAPA_VALIDAS
        if desconhecidas:
            logger.warning(
                f"⚠️  Etapa {i}: chaves desconhecidas IGNORADAS no YAML: {sorted(desconhecidas)}. "
                f"Verifique typos (chaves válidas: {sorted(_CHAVES_ETAPA_VALIDAS)})"
            )
```

Na construção do `EtapaCurriculum` (junto de `batch_size=...`):

```python
            warmup_steps=int(item.get("warmup_steps", -1)),
            unfreeze_layers_from=int(item.get("unfreeze_layers_from", -1)),
```

Após criar `etapa` e **antes** de `etapas.append(etapa)`:

```python
        # unfreeze_layers_from só faz sentido com todos os parâmetros base destravados (full)
        if etapa.unfreeze_layers_from >= 0 and etapa.tipo != "full":
            raise ValueError(
                f"Etapa '{alias}': 'unfreeze_layers_from' só é válido com tipo: \"full\" "
                f"(recebido tipo='{etapa.tipo or '(vazio)'}'). Em etapas LoRA a capacidade "
                f"já é controlada pelo rank do adaptador."
            )
```

Opcional (linha ~419, log das etapas): acrescentar o unfreeze ao `pace_info` para aparecer no
resumo inicial:

```python
        if e.unfreeze_layers_from >= 0:
            pace_info += f", unfreeze>={e.unfreeze_layers_from}"
```

---

## 2. `treinar_unsloth.py`

### 2.1 — Import de `re` e novo método `_aplicar_unfreeze_parcial`

Garantir `import re` no topo do arquivo (hoje não é importado).

Adicionar o método na classe `LLMsTrainer` (sugestão: imediatamente antes de
`_aplicar_etapa_curriculum`, linha ~2212):

```python
    # Regex para extrair o índice do bloco transformer do nome do parâmetro.
    # Cobre nomes como "model.layers.24.self_attn.q_proj.weight" e também o
    # prefixo PEFT "base_model.model.model.layers.24....".
    _RE_IDX_CAMADA = re.compile(r"\.layers\.(\d+)\.")

    def _aplicar_unfreeze_parcial(self, from_layer: int, alias: str) -> None:
        """Descongelamento progressivo: mantém treináveis apenas os blocos
        transformer com índice >= from_layer, além da norm final e do lm_head.

        Política:
        - bloco `layers.N.*`      → treinável se N >= from_layer
        - `embed_tokens`          → treinável apenas se from_layer == 0
        - demais float (norm final, lm_head não-tied) → sempre treináveis
        - parâmetros quantizados  → intocados (não suportam gradiente)

        NOTA (tied embeddings): no Qwen 2.5, `lm_head.weight` compartilha o tensor
        de `embed_tokens.weight` (tie_word_embeddings=true). Nesses modelos o par
        embeddings/cabeça é UM parâmetro e permanece congelado até from_layer==0.
        O gradiente continua fluindo através da cabeça congelada para os blocos
        superiores — isso é esperado e não impede o treinamento.
        """
        if self._lora_applied:
            raise ValueError(
                f"Etapa '{alias}': unfreeze_layers_from não é compatível com adaptadores "
                f"LoRA aplicados ao modelo. Use pipelines 100% 'full' para protocolos de "
                f"descongelamento progressivo (D19/D20)."
            )

        n_layers = getattr(self.model.config, "num_hidden_layers", None)
        if n_layers is not None and from_layer >= n_layers:
            logger.warning(
                f"⚠️  Etapa '{alias}': unfreeze_layers_from={from_layer} >= num_hidden_layers="
                f"{n_layers}. Nenhum bloco transformer será treinado (apenas norm/lm_head)."
            )

        stats = {"blocos_treinaveis": 0, "blocos_congelados": 0,
                 "embeddings": 0, "cabeca_norm": 0}
        for name, param in self.model.named_parameters():
            if param.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                continue  # quantizados: intocados
            m = self._RE_IDX_CAMADA.search(name)
            if m:
                treinavel = int(m.group(1)) >= from_layer
                stats["blocos_treinaveis" if treinavel else "blocos_congelados"] += param.numel()
            elif "embed_tokens" in name:
                treinavel = (from_layer <= 0)
                stats["embeddings"] += param.numel()
            else:
                treinavel = True  # norm final, lm_head (quando não-tied), etc.
                stats["cabeca_norm"] += param.numel()
            param.requires_grad = treinavel

        # Gradient checkpointing com camadas iniciais congeladas: garante que a
        # entrada dos blocos checkpointed exija grad (mesmo mecanismo usado pelo PEFT).
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        if getattr(self.model.config, "tie_word_embeddings", False) and from_layer > 0:
            logger.info(
                "ℹ️  Modelo com embeddings/cabeça compartilhados (tied): o tensor único "
                "embed_tokens/lm_head permanece CONGELADO até unfreeze_layers_from: 0."
            )
        logger.info(
            f"🧊 Unfreeze parcial (etapa '{alias}'): blocos >= {from_layer} treináveis "
            f"({stats['blocos_treinaveis']:,} params) | congelados: blocos "
            f"{stats['blocos_congelados']:,} + embeddings {stats['embeddings']:,} | "
            f"norm/cabeça: {stats['cabeca_norm']:,}"
        )
```

### 2.2 — Integração no ramo `full` de `_aplicar_etapa_curriculum` (linha ~2313)

Substituir o bloco:

```python
        if etapa.tipo == "full":
            # Full fine-tuning: desbloqueia todos os parâmetros float (base + LoRA se presente)
            # Parâmetros quantizados (int8/int4 via bitsandbytes) não suportam gradientes
            for param in self.model.parameters():
                if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    param.requires_grad = True
            n_total = sum(p.numel() for p in self.model.parameters())
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
```

por:

```python
        if etapa.tipo == "full":
            # Full fine-tuning: desbloqueia todos os parâmetros float (base + LoRA se presente)
            # Parâmetros quantizados (int8/int4 via bitsandbytes) não suportam gradientes
            for param in self.model.parameters():
                if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    param.requires_grad = True
            # Descongelamento progressivo: re-congela blocos abaixo do corte da etapa
            if getattr(etapa, "unfreeze_layers_from", -1) >= 0:
                self._aplicar_unfreeze_parcial(etapa.unfreeze_layers_from, etapa.alias)
            n_total = sum(p.numel() for p in self.model.parameters())
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
```

(As duas linhas de log logo abaixo já refletem o `n_train` correto sem mudanças.)

### 2.3 — Override de `warmup_steps` por etapa

No mesmo método, junto dos overrides existentes (após o bloco do `learning_rate`, linha ~2379):

```python
        # Override de warmup_steps se especificado (>= 0; -1 = usa global)
        if getattr(etapa, "warmup_steps", -1) >= 0:
            treino.warmup_steps = etapa.warmup_steps
            logger.info(f"🔥 Etapa '{etapa.alias}': warmup_steps={etapa.warmup_steps} (override por etapa)")
```

### 2.4 — Anti-vazamento em `train()` (linha ~2512)

O dict `global_defaults` restaura os globais antes de cada etapa, mas **não** inclui
`warmup_steps` — sem isso, o override de uma etapa vazaria para as seguintes.

```python
        global_defaults = {
            "learning_rate": treino_cfg.learning_rate,
            "batch_size": treino_cfg.batch_size,
            "epochs": treino_cfg.epochs,
            "max_seq_length": treino_cfg.max_seq_length,
            "warmup_steps": treino_cfg.warmup_steps,   # NOVO
        }
```

E na restauração dentro do loop (linha ~2530):

```python
            treino_cfg.warmup_steps = global_defaults["warmup_steps"]   # NOVO
```

---

## 3. Correções essenciais identificadas na revisão

### 3.1 — Chave morta `bits: 16` nas etapas (d5/d6) — **corrigida pela whitelist do item 1.2**

Os YAMLs `04_treinar_d5.yaml` e `04_treinar_d6.yaml` definem `bits: 16` na etapa FF, mas
`construir_etapas()` nunca lê essa chave — o valor efetivo vem de
`alvo_nbits = 16 if etapa.tipo == "full" else treinamento.nbits` em `_aplicar_etapa_curriculum`.
O comportamento está correto por coincidência (FF sempre roda em 16), mas a chave é silenciosamente
ignorada. Com a whitelist, ela passará a gerar warning. **Remover `bits: 16` dos YAMLs d5/d6.**
Isso também protege o novo `unfreeze_layers_from` contra typos (um typo hoje desativaria o
recurso sem nenhum aviso).

### 3.2 — Merge LoRA→base com modelo quantizado em 4 bits (linha ~2252) — **adicionar alerta**

No bloco de merge antes do reload, quando a etapa LoRA anterior rodou com `nbits: 4`,
`self.model.merge_adapter()` opera sobre camadas bitsandbytes: o PEFT dequantiza, soma o delta
e **requantiza** cada camada — introduzindo exatamente o erro de requantização que causa o
degrau de loss que não retorna ao patamar anterior nas transições LoRA→FF. Adicionar, logo
antes de `self.model.merge_adapter()`:

```python
                if nbits_memoria == 4:
                    logger.warning(
                        "⚠️  Merge LoRA→base com modelo em 4 bits: o merge dequantiza e "
                        "REQUANTIZA cada camada, introduzindo erro de quantização nos pesos "
                        "salvos (fonte do degrau de loss na transição de regime). "
                        "Para eliminar o efeito, rode as etapas LoRA com nbits: 16."
                    )
```

Não é um fix automático (mudar o fluxo de merge é invasivo); é um guarda-corpo para que o
fenômeno fique registrado no log dos experimentos. Nos protocolos D16–D20 abaixo o problema
desaparece por construção (`nbits: 16` global).

### 3.3 — `_load_model`: `os.listdir` sem guarda de existência (linha ~1422)

`_tem_full_local = any(... for f in os.listdir(lora_model_path))` levanta `FileNotFoundError`
se a pasta de saída sumir entre a detecção do tipo e o listdir (raro, mas possível em Slurm com
storage de rede). Trocar por:

```python
            _tem_full_local = os.path.isdir(lora_model_path) and any(
                f.endswith('.safetensors') and not f.startswith('adapter')
                for f in os.listdir(lora_model_path)
            )
```

### 3.4 — Cosmético: atribuição duplicada de `_tipo_etapa_atual`

Em `_aplicar_etapa_curriculum`, `self._tipo_etapa_atual = novo_tipo` é executado dentro do
bloco de reload (linha ~2296) e novamente após ele (linha ~2310). Inofensivo; pode remover a
segunda ocorrência ou manter com comentário. Sem impacto funcional.

---

## 4. Comportamento esperado após as mudanças (checklist de validação)

1. `--datasets` (dry-run) num YAML com `unfreeze_layers_from` deve listar as etapas com
   `unfreeze>=N` no log do curriculum, sem erro.
2. `unfreeze_layers_from` em etapa `tipo: "lora"` → `ValueError` na carga do YAML.
3. No início de cada etapa D19/D20, o log deve mostrar `🧊 Unfreeze parcial ...` e o
   contador `Modo FULL: X/Y parâmetros desbloqueados` deve **crescer** etapa a etapa
   (ex.: no Qwen 1.5B com 28 blocos, algo como ~25% → ~50% → ~75% → 100%).
4. Entre etapas full consecutivas **não** deve aparecer o log `🔄 ... Descarregando e
   recarregando o modelo` (nbits não muda) — a transição é em memória, function-preserving.
5. Chave desconhecida em qualquer etapa (ex.: o `bits: 16` antigo) → warning explícito.
6. `warmup_steps` por etapa aparece no log e volta ao global na etapa seguinte se omitido.

---

## 5. Protocolos D16–D20 — desenho e orçamento

Todos com `nbits: 16` (elimina a variável de quantização) e `warmup_steps: 100` (fronteiras
bem mitigadas — o custo medido de fronteira é o custo *intrínseco*, não o de warmup curto).
Qwen 2.5 1.5B tem **28 blocos** → cortes em quartos: 21 / 14 / 7 / 0.

Orçamento de instâncias (N = tamanho do conjunto completo; fácil+médio+difícil particionam N):

| Protocolo | Estrutura | Instâncias | Compara com | Pergunta |
|---|---|---|---|---|
| D16 | LoRA-16b, dados completos, 4 execuções seriais de 1 época | 4N | B | custo puro de fronteira |
| D17 | LoRA-16b + CL (fácil→médio→difícil→completo, 2 ép. cada) | 2N + 2N = 4N | D16, D7 | efeito da ordenação (sem 4-bit) |
| D18 | LoRA-16b + blocos aleatórios (b1→b2→b3→completo, 2 ép. cada) | 2N + 2N = 4N | D17 | ordenação vs. efeito de bloco/recência |
| D19 | FF + unfreeze (21→14→7→0) + CL, 2 ép. cada | 2N + 2N = 4N | D13, D20 | sinergia CL+PT (tese central) |
| D20 | FF + unfreeze (21→14→7→0), dados completos, 1 ép. cada | 4N | C, D19 | efeito do PT-unfreezing puro |

Contrastes fatoriais: **B vs D16** (fronteiras), **D16 vs D18** (blocos), **D18 vs D17**
(ordenação por dificuldade), **C vs D20** (unfreezing), **D20 vs D19** (CL dado unfreezing),
**D13 vs D19** (valor do PT somado ao CL).

> Ajuste os `pace_epochs` se o orçamento de referência de B for diferente de 4 épocas no seu
> grupo experimental — a aritmética acima assume B = 4 épocas sobre o conjunto completo.

### 5.1 — Pré-requisito do D18: CSV de divisão com blocos aleatórios

O D18 precisa de uma coluna `bloco` (∈ {b1, b2, b3}) no CSV de divisão, com atribuição
aleatória em **terços** (mesma granulometria de fácil/médio/difícil, para igualar instâncias
com o D17). Gerar uma vez com seed fixa:

```python
import pandas as pd
import numpy as np

SRC = "compara/analises_comparacao_pubmed (full)/divisoes/divisao_Professor_Qwen1_5B.csv"
DST = "compara/analises_comparacao_pubmed (full)/divisoes/divisao_Professor_Qwen1_5B_blocos.csv"

df = pd.read_csv(SRC)
rng = np.random.default_rng(3407)
df["bloco"] = rng.permutation(np.resize(np.array(["b1", "b2", "b3"]), len(df)))
df.to_csv(DST, index=False)
print(df["bloco"].value_counts())
```

Nota: como no filtro por `dificuldade` dos protocolos existentes, o filtro por `bloco` também
recorta o conjunto de validação da etapa — comportamento idêntico ao D17, portanto comparável.
O `eval_loss_global` (GlobalEvalCallback) segue avaliando o conjunto combinado.

### 5.2 — Observações operacionais

- **D19/D20 (unfreeze)**: validar os primeiros ~50 steps da etapa 1 observando se o loss
  desce e se não há warning de gradiente nulo. O `enable_input_require_grads()` do item 2.1
  cobre o caso gradient checkpointing + camadas iniciais congeladas, mas a validação visual
  nos primeiros steps é barata e conclusiva.
- **Tied embeddings**: no Qwen 1.5B, embeddings/lm_head são o mesmo tensor e ficam congelados
  nas etapas 1–3 do D19/D20 (mensagem `ℹ️` no log). Documentar na dissertação como decisão
  de desenho (âncora de conhecimento pré-treinado).
- **D16–D18 (LoRA 16 bits)**: sem o pacote flash-attn instalado, o fallback será `eager`
  (regra anti-NaN do SDPA+LoRA já existente no código) — VRAM maior que nos runs 4-bit;
  no 1.5B com seq 8192 na H100 cabe, mas monitorar o pico na primeira etapa.
- **`load_best_model_at_end`**: cada etapa termina carregando o melhor checkpoint *da etapa*.
  Isso vale para todos os protocolos D (comportamento herdado), então os contrastes seguem
  comparáveis entre si; apenas B difere (seleção única ao final). Registrar como nota
  metodológica.
