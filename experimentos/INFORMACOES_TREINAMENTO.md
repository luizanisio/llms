# Informações Técnicas dos Experimentos (SUMMA, SemClinBr e PubMed)

Anotações de referência para pesquisadores que utilizarem estes experimentos como base.
Registra decisões de implementação, configurações efetivas e comportamentos observados.

Os três experimentos compartilham a mesma grade de 31 protocolos treinados (B, B16, B16R8,
C, D1–D25, D1a, D1b) mais o zero-shot A, com a mesma topologia de etapas por protocolo.
O que muda entre eles é o modelo aluno, a janela de contexto, o tamanho dos dados e a
referência. Índices dos protocolos: `summa-experimento/04_treinar__resumo.md` e
`semclinbr-experimento/04_treinar__resumo.md`.

---

## 0. Visão geral

| Item | SUMMA | SemClinBr | PubMed |
|---|---|---|---|
| Modelo aluno | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct | Qwen2.5-1.5B-Instruct |
| Referência | Qwen 235B (professor) | Gold humano (`saida_semclinbr_gold.parquet`) | Gold do dataset (`saida_pubmed_prof.parquet`) |
| `max_seq_length` (treino) | 32.768 | 12.288 | 8.192 (B, B16, B16R8, C: 16.384) |
| `max_model_len` (inferência) | 38.912 | 12.288 | 8.192 |
| Precisão na inferência | bitsandbytes 4-bit | bitsandbytes 4-bit | bf16 nativo |
| Divisão treino/val/teste | 15.508 / 2.217 / 4.430 | 697 / 103 / 197 | 14.996 / 2.499 / 2.500 |
| Teste efetivo | **3.948** (`fold <= 10`) | **197** | **2.500** |
| Lote efetivo | 16 | 16 | 16 (8 nas etapas `full` dos mistos, ver §3) |
| `warmup_steps` | 5 (D16–D25: 100) | 5 em todos | 5 (D16–D25: 100) |
| `warmup_grupo_steps` (fusão) | 100 | 5 | 100 |
| Avaliação | Métricas automáticas + juiz LLM | Métricas automáticas + NER-F1 + baseline ClinicalNERpt | Métricas automáticas |

LR por etapa é igual nos três: etapas `lora` em `2e-05`, etapas `full` em `5e-06`, seed `3407`.

---

## 1. Precisão na inferência (vLLM)

**SUMMA e SemClinBr:** todos os YAMLs de extração (`05_extracao_*_teste.yaml`) carregam
os modelos via vLLM com quantização bitsandbytes 4-bit on-the-fly, independentemente da
precisão usada no treino:

```yaml
vllm:
  quantization: bitsandbytes
  load_format: bitsandbytes
```

| Protocolo | Treino (nbits) | Tipo | Inferência |
|---|---|---|---|
| **A** (zero-shot) | — | — | bitsandbytes 4-bit |
| **B** | 4 | LoRA | bitsandbytes 4-bit |
| **B16, B16R8** | 16 | LoRA | bitsandbytes 4-bit |
| **C** | 16 | Full FT | bitsandbytes 4-bit |
| **D1–D25, D1a, D1b** | 4 ou 16 | Misto | bitsandbytes 4-bit |

No SemClinBr, o Qwen2.5-7B em bf16 ocupa ~14 GB só de pesos. A quantização mantém
a mesma condição de inferência do SUMMA e deixa folga para o KV cache em batch com
`max_model_len: 12288`.

**PubMed (Qwen 1.5B):** não usa quantização na inferência. O Qwen2.5-1.5B-Instruct
ocupa ~3 GB em bf16, cabendo com folga na H100 80 GB mesmo com KV cache para
sequências de 8k tokens. Os YAMLs usam apenas `enforce_eager: true` (sem
`quantization` nem `load_format`), e o `dtype: auto` resulta em bf16 nativo.

### Implicação observada

A quantização 4-bit na inferência é uniforme entre protocolos, então os contrastes
*entre* protocolos são comparáveis. Mas o caminho até ela não é o mesmo para todos:

- **LoRA puro** (B, B16, B16R8, D7–D10, D16–D18, D21–D24): o adaptador é aplicado sobre
  os pesos originais do Qwen2.5, que o vLLM quantiza uma única vez.
- **Full e mistos** (C, D1–D6, D1a, D1b, D11–D15, D19, D20, D25): o modelo é salvo em
  bf16 (`config.json` sem `quantization_config`) e o vLLM quantiza os pesos já
  modificados. Nos mistos com etapas LoRA em 4 bits, esses pesos já passaram por NF4
  durante o treino (§2), então a quantização da inferência é a segunda.

Isso é uma limitação do setup de inferência e deve ser mencionado ao comparar LoRA vs
Full FT no SUMMA e no SemClinBr. No PubMed ela não existe na inferência, apenas no treino.

---

## 2. Recarga, merge e quantização nos pipelines com troca de regime

### Regra geral

Etapas `full` sempre rodam em 16 bits; etapas `lora` usam o `nbits` global do YAML.
O modelo é descarregado e recarregado quando a etapa seguinte exige outra precisão
(`treinar_unsloth.py`, `_aplicar_etapa_curriculum()`):

```python
alvo_nbits = 16 if etapa.tipo == "full" else self._yaml_config.treinamento.nbits
precisa_recarregar = alvo_nbits != nbits_memoria
```

A recarga também é forçada quando o tipo da etapa muda e o pacote `flash-attn` não está
instalado com `full_com_sdpa: true` (troca de implementação de atenção).

Antes de recarregar, o merge LoRA→base (`merge_adapter()`) só acontece se o pipeline
tem etapa `full` **e** o LoRA já está aplicado. Na prática:

- **LoRA→Full:** merge na fronteira, salvamento e recarga em 16 bits.
- **Full→LoRA:** **não há merge** (ainda não existe adaptador). Os pesos da etapa
  `full` são salvos em bf16 e recarregados com o `nbits` global. Com `nbits: 4`, os
  pesos treinados em full são quantizados para NF4 nesse ponto.
- **LoRA→LoRA com o mesmo `nbits`:** sem recarga. O adaptador permanece em VRAM e
  acumula treinamento entre etapas, sem reinicialização.

### Salvamento ao fim de cada etapa

`_save_model()` é chamado ao fim de toda etapa (duas vezes: antes e depois do eval
final). Em pipelines com etapa `full` e LoRA aplicado, ele faz
`merge_adapter() → save_pretrained() → unmerge_adapter()` e remove os arquivos do
adaptador, de modo que a pasta de saída sempre contém um modelo full auto-contido.
Pipelines LoRA puro salvam só o adaptador.

### Mapa por família de protocolo

A topologia é a mesma nos três experimentos.

| Família | Protocolos | Etapas | nbits das etapas LoRA | Na fronteira de regime | Artefato salvo |
|---|---|---|---|---|---|
| LoRA puro | B, D7, D8, D9, D10 | L (1, 3 ou 4) | 4 | — | adaptador |
| LoRA puro 16b | B16, B16R8, D16, D17, D18 | L (1 ou 4) | 16 | — | adaptador |
| Full puro | C, D13 | F (1 ou 4) | — | — | full bf16 |
| Full com unfreeze progressivo | D19, D20 | F ×4 (75→50→25→0%) | — | — | full bf16 |
| Full→LoRA | D1, D1a, D1b, D3, D5, D11 | F → L… | 4 | sem merge; pesos FF recarregados em NF4 | full bf16 (merge final sobre base NF4) |
| LoRA→Full | D2, D4, D6, D12, D14, D15 | L… → F… | 4 | merge em 4 bits + recarga em 16 bits | full bf16 |
| Fusão LoRA | D21, D22, D23 | 1 trainer, 4 spans | 16 | — | adaptador |
| Fusão LoRA 4b | D24 | 1 trainer, 10 spans | 4 | — | adaptador |
| Fusão Full | D25 | 1 trainer, 10 spans | — (16 forçado) | — | full bf16 |

No modo fusão (`curriculum.fusao.ativo: true`) não há fronteiras reais: o regime e a
precisão são únicos no run, e o `unfreeze_layers_from` de cada span vira gating de LR
por grupo de camadas, sem congelamento. Com `fusao.tipo: full`, o `nbits` é forçado a 16.

### Erro de quantização nos mistos com `nbits: 4`

Quando o merge ou o unmerge ocorre sobre um modelo em 4 bits (bitsandbytes NF4), o
PEFT dequantiza cada camada, soma (ou subtrai) o delta LoRA e requantiza, recriando o
`Params4bit` (`peft/tuners/lora/bnb.py`, `Linear4bit.merge/unmerge`). O framework
emite warning explícito no merge de fronteira.

- **LoRA→Full (D2, D4, D6, D12, D14, D15):** o merge de fronteira opera sobre a base
  NF4. As etapas `full` seguintes partem desses pesos, recarregados em 16 bits.
- **Full→LoRA (D1, D1a, D1b, D3, D5, D11):** não há merge na fronteira, mas os pesos
  treinados em full são quantizados para NF4 ao entrar nas etapas LoRA. O modelo
  final é o merge do LoRA sobre essa base NF4, salvo em bf16, e carrega o erro de
  quantização dos pesos FF.
- **Ambas as direções:** o par `merge_adapter()`/`unmerge_adapter()` feito por
  `_save_model()` ao fim de cada etapa LoRA requantiza a base NF4 em memória. A cada
  salvamento, a base sobre a qual o adaptador continua treinando sofre um pequeno
  arredondamento adicional. Não ocorre em pipelines LoRA puro, que não fazem merge.

Os protocolos sem etapa LoRA em 4 bits não têm essa variável por construção: D16–D18
(LoRA 16b segmentado), D21–D23 (fusão LoRA 16b), D19, D20 e D25 (full 16b). D24 é LoRA
4b fundido: não faz merge, mas treina sobre base NF4, como o B.

---

## 3. Lote efetivo e override de `batch_size` por etapa

`batch_size.efetivo` manda em `grad_batch_size`: `_aplicar_batch_size_auto`
(`src/treinar_unsloth_util.py`) recalcula `grad_batch_size = efetivo / (batch_size × n_gpus)`
e sobrescreve o valor escrito no YAML.

O override de `batch_size` por etapa **não** recalcula o acúmulo: a etapa troca o
`batch_size`, mas mantém o `grad_batch_size` global.

- **SUMMA e SemClinBr:** `batch_size` global é 1 e as etapas que declaram
  `batch_size: 1` repetem o mesmo valor. Todas as etapas rodam com lote efetivo 16.
- **PubMed:** `batch_size` global é 2. As etapas `full` de D1–D6, D1a, D1b, D11 e D12
  declaram `batch_size: 1` e rodam com lote efetivo 8 (ver
  `README_PUBMED_EXPERIMENTO.md`).

---

## 4. Conjuntos de teste

### SUMMA (3.948 documentos)

O filtro `fold: <=10` nas integras resulta em 3.948 documentos de teste, não 4.430.
Os 482 documentos restantes pertencem aos folds 11 (388) e 12 (94):

- **Fold 11:** documentos usados como holdout intermediário
- **Fold 12:** documentos de 2026 (out-of-time, posteriores ao cutoff de todos os modelos)

O mesmo `dataset_filtro: {"fold": "<=10"}` é aplicado na entrada dos YAMLs de treino,
então treino e validação também ficam restritos aos folds 1–10.

Os YAMLs de extração aplicam `dataset_filtro: {"fold": "<=10"}` nas integras e o filtro
externo `{"alvo": "teste"}` na divisão, de modo que todos os parquets de saída dos
modelos treinados contêm os 3.948 IDs.

Os YAMLs de comparação (`06_compara_*.yaml`) aplicam o filtro
`{"alvo": "teste", "fold": "<=10"}` na divisão CSV, garantindo que o zero-shot (A) e o
professor, que têm saídas completas (22k docs), sejam avaliados apenas sobre os mesmos
3.948 documentos. A coluna `fold` foi adicionada à divisão CSV via join com as integras
(`seq_documento_acordao` → `id`). Para o juiz LLM, o `07_avaliacao_llm.py` gera
`saida_qwen7b(a)_teste.parquet` e `saida_or_235b(prof)_teste.parquet` já filtrados.

### SemClinBr (197 documentos)

A divisão `dados/divisao_Gold_Qwen7B.csv` (gerada no passo 03) tem 997 dos 1.000
documentos do corpus. Os YAMLs de extração usam o filtro externo `{"alvo": "teste"}`, e
os de comparação aplicam o mesmo filtro ao zero-shot e ao gold (1.000 docs cada).

Com 697 documentos de treino, os protocolos granulares de 10 etapas (D11, D12, D24, D25)
ficam com ~70 instâncias por faixa. Os YAMLs rodam com uma seed (`3407`); a replicação
por seed fica para uma segunda rodada.

### PubMed (2.500 documentos)

O split oficial `test` do PubMed 20k RCT. Os YAMLs de extração filtram
`{"split": "test"}`, e os de comparação aplicam o mesmo filtro (via
`dados/pubmed-rct-20k.parquet`, campo `pmid`) ao zero-shot e ao gold, que têm 19.995
docs cada.

---

## 5. Avaliação

### 5.1 Métricas automáticas e ROPE (`06_compara_todos.yaml`)

As métricas automáticas medem similaridade com a referência, ou seja, fidelidade de
destilação, não qualidade. A ROPE é a margem de empate sobre os escores brutos,
calibrada pelas réplicas D1, D1a e D1b (`06_compara_d1ab.yaml` →
`estatisticas/00_rope_sugerido.md`).

| Experimento | `rope` (fallback) | `rope_por_campo` |
|---|---|---|
| SUMMA (recalibrada em 13/10/2026) | 0,0179 | Likert 0,0153 · MetaDados 0,0179 |
| SemClinBr | 0,0401 | (global) 0,0188 · Entidades 0,0162 · Relacoes 0,0401 |
| PubMed | 0,0058 | (global) 0,0030 · MetaDados 0,0058 · Sentencas 0,0022 |

O `rope_likert: 0.01` dos YAMLs de comparação é um placeholder. Ele só teria efeito se
`campos_parquet.avaliacao` apontasse para as notas do juiz, o que hoje não acontece. A
análise Likert é feita exclusivamente pelo `07_avaliacao_llm.py` (§5.2).

### 5.2 Juiz LLM (SUMMA)

- **Modelo:** GPT-5 em Azure, por duas implantações do mesmo modelo: `oa:gpt5:m:l` e
  `oa:gpt5-chat:m:l`. O padrão atual do `07_avaliacao_llm.py` é `gpt5-chat`.
  - `gpt5`: A, B, D1b, D8, D9, D18, D24
  - `gpt5-chat`: prof, C, D1, D1a, D7, D10, D16, D17, D21, D23
- **Reasoning:** médio (`m`). **Verbose:** low (`l`).
- **Passadas:** uma por protocolo. A variabilidade do juiz é estimada pelas réplicas.
- **ROPE Likert:** calibrada pelas réplicas D1, D1a e D1b (`ROPE_PROTOCOLOS`) com o
  `calibrar_rope()` de `src/realizar_avaliacoes.py`: para cada par de réplicas, a menor
  margem em que o par sai `equivalente` ao limiar 0,95; a ROPE é a maior delas. Último
  `--resumo` (14/10/2026): **0,0314** (3.948 itens, 3 pares). Como D1b foi avaliada por
  outra implantação, a ROPE inclui a diferença entre as implantações (o resumo emite
  aviso). O relatório repete a comparação com ROPE/2 e 2·ROPE.
  O valor antigo, 0,1579, vinha da divergência média entre os 3 avaliadores humanos e
  está desativado no script (`ROPE_LIKERT = None`).
- **Protocolos avaliados (3.948 itens cada):** prof, A, B, C, D1, D1a, D1b, D7, D8, D9,
  D10, D16, D17, D18, D21, D23, D24. D23 foi avaliado depois do último `--resumo` e
  ainda não aparece no `RESUMO_AVALIACAO_LLM.md`. A lista `PROTOCOLOS` do script é só
  o padrão de execução, não o conjunto avaliado.

> **Avaliação humana:** em andamento na pasta `avaliacao_llm_humana/`. Não entra
> no artigo do SAC; será reportada na dissertação.

### 5.3 NER (SemClinBr)

O SemClinBr não usa juiz LLM. A variável primária é o NER-F1 do `07_avaliar_ner.py`,
calculado por documento de teste e por protocolo sobre os mesmos parquets dos passos
02 e 05 (estatística: Friedman + Wilcoxon com Holm + tamanho de efeito). O
`08_baseline_clinicalnerpt.py` roda os modelos `pucpr/clinicalnerpt-*` (BioBERTpt) no
mesmo split de teste como baseline externo.

---

## 6. Estado dos protocolos

As taxas de JSON inválido referem-se ao parquet de teste de cada protocolo.

### SUMMA

| Protocolo | Treino | Extração teste | Observação |
|---|---|---|---|
| A | — | ✅ 22k (filtrar na comparação) | Zero-shot. `saida_qwen7b(a)_teste.parquet` (3.948) é gerado para o juiz |
| B, B16, B16R8 | ✅ | ✅ 3.948 | — |
| C | ✅ | ✅ 3.948 | Full FT 16b |
| D1–D10 | ✅ | ✅ 3.948 | D10 concluído em outra máquina |
| D1a, D1b | ✅ | ✅ 3.948 | Réplicas de calibração da ROPE (métricas e juiz) |
| D11–D25 | ✅ | ✅ 3.948 | — |
| Fold 12 (2026) | — | Pendente | Não entra no artigo SAC; planejado para a dissertação |

JSON inválido: os protocolos com etapa `full` segmentada concentram as falhas. C,
D1–D6, D1a, D1b e D11–D14 ficam entre 3,3% e 7,3% (máximo no D11, 7,29%), e D15 em 1,2%.
Os demais, incluindo os full sem fronteira de regime (D19, D20, D25), ficam ≤ 0,51%.

### SemClinBr

| Protocolo | Treino | Extração teste | Observação |
|---|---|---|---|
| A | — | ✅ 1.000 (filtrar na comparação) | Zero-shot |
| B, B16, B16R8, C | ✅ | ✅ 197 | — |
| D1–D25, D1a, D1b | ✅ | ✅ 197 | — |

JSON inválido ≤ 4,06% (máximo no D3). Cada documento vale 0,51% do teste.

### PubMed

| Protocolo | Treino | Extração teste | Observação |
|---|---|---|---|
| A | — | ✅ 19.995 (filtrar na comparação) | Zero-shot |
| B, B16, B16R8, C | ✅ | ✅ 2.500 | — |
| D1–D25, D1a, D1b | ✅ | ✅ 2.500 | — |

JSON inválido ≤ 0,2%, exceto D24 (1,12%).
