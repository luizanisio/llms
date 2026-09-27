# Informações Técnicas do Experimento SUMMA

Anotações de referência para pesquisadores que utilizarem este experimento como base.
Registra decisões de implementação, configurações efetivas e comportamentos observados.

---

## 1. Precisão na inferência (vLLM)

Todos os YAMLs de extração (`05_extracao_*_teste.yaml`) carregam os modelos via vLLM
com quantização bitsandbytes 4-bit on-the-fly, independentemente da precisão usada
no treino:

```yaml
vllm:
  quantization: bitsandbytes
  load_format: bitsandbytes
```

| Protocolo | Treino (nbits) | Tipo | Inferência |
|---|---|---|---|
| **A** (zero-shot) | — | — | bitsandbytes 4-bit |
| **B** | 4 | LoRA | bitsandbytes 4-bit |
| **B16** | 16 | LoRA | bitsandbytes 4-bit |
| **C** | 16 | Full FT | bitsandbytes 4-bit |
| **D1–D25** | 4 ou 16 | Misto | bitsandbytes 4-bit |

**PubMed (Qwen 1.5B):** não usa quantização na inferência — modelo pequeno o
suficiente para rodar em precisão nativa (`dtype: auto`).

**SemClinBr (Qwen 7B):** mesma configuração bitsandbytes do SUMMA.

### Implicação observada

A quantização 4-bit na inferência é uniforme entre protocolos, então os contrastes
*entre* protocolos são comparáveis. Porém, modelos com pesos base modificados (Full FT
e pipelines com merge) sofrem uma requantização adicional que não ocorre nos adaptadores
LoRA puros (cujos pesos base permanecem os originais do Qwen2.5-7B). Isso é uma
limitação do setup de inferência e deve ser mencionado ao comparar LoRA vs Full FT.

---

## 2. Comportamento do merge nos pipelines com troca de regime

### Regra geral

O merge LoRA→base (`merge_adapter()`) ocorre **apenas** quando há troca de `nbits`
entre etapas — ou seja, na fronteira lora↔full. Etapas LoRA consecutivas com o mesmo
`nbits` **não fazem merge**: o adaptador permanece em VRAM e acumula treinamento
in-place entre etapas.

A condição no código (`treinar_unsloth.py`, função `_aplicar_etapa_curriculum()`):

```python
precisa_recarregar = alvo_nbits != nbits_memoria
```

Quando `precisa_recarregar = False` (ex: lora 4-bit → lora 4-bit), o adaptador LoRA
aplicado na primeira etapa LoRA é simplesmente continuado nas etapas seguintes, sem
reinicialização dos pesos do adaptador.

### Mapa de merges por protocolo

| Protocolo | Etapas | Merges durante treino | Merge final |
|---|---|---|---|
| **B** (LoRA puro) | LoRA-completo | 0 | 0 — salva adapter separado |
| **C** (Full puro) | FF-completo | 0 | 0 — salva full diretamente |
| **D1** (FF→LoRA, 4b) | FF → LoRA → LoRA → LoRA | 1 na transição FF→LoRA | 1 — merge final (LoRA 4-bit nos pesos) |
| **D2** (LoRA→FF, 4b) | LoRA → LoRA → LoRA → FF | 1 na transição LoRA→FF | 0 — já é full |
| **D5** (FF→LoRA sem CL, 4b) | FF → LoRA | 1 na transição FF→LoRA | 1 — merge final |
| **D6** (LoRA→FF sem CL, 4b) | LoRA → FF | 1 na transição LoRA→FF | 0 — já é full |
| **D7** (LoRA com CL, 4b) | LoRA → LoRA → LoRA → LoRA | 0 | 0 — salva adapter |
| **D16–D18** (16-bit) | LoRA puro em 16b | 0 | 0 — salva adapter |
| **D19–D20** (unfreeze) | Full progressivo em 16b | 0 | 0 — salva full |

### Erro de requantização no merge com nbits=4

Quando o merge ocorre sobre um modelo em 4 bits (bitsandbytes NF4), o PEFT
dequantiza cada camada, soma o delta LoRA e requantiza — introduzindo erro de
quantização nos pesos salvos. O framework emite warning explícito nessa situação.

Comportamento específico por protocolo (D1–D6, `nbits: 4` global):

- **D1 (FF→LoRA):** a transição FF→LoRA não sofre requantização no merge intermediário
  (modelo estava em 16 bits). O merge **final** opera sobre pesos LoRA em 4 bits —
  erro de quantização é injetado nos pesos salvos para inferência.

- **D2 (LoRA→FF):** a transição LoRA→FF sofre requantização no merge intermediário
  (3 etapas LoRA em 4 bits). O modelo é recarregado em 16 bits imediatamente após —
  o erro é transitório na memória, mas os pesos salvos no disco carregam o delta
  quantizado. A etapa Full subsequente parte desses pesos.

Para eliminar esta variável confundidora por construção, os protocolos **D16–D18**
usam `nbits: 16` em todas as etapas LoRA, e os protocolos **D19–D25** usam Full FT
com descongelamento progressivo (sem merge de adaptadores em nenhuma etapa).

---

## 3. Conjunto de teste canônico (3.948 documentos)

O filtro `fold: <=10` nas integras resulta em 3.948 documentos de teste, não 4.430.
Os 482 documentos restantes pertencem aos folds 11 e 12:

- **Fold 11:** documentos usados como holdout intermediário
- **Fold 12:** documentos de 2026 (out-of-time, posteriores ao cutoff de todos os modelos)

Os YAMLs de extração já aplicam `dataset_filtro: {"fold": "<=10"}` nas integras, de
modo que todos os parquets de saída dos modelos treinados contêm os 3.948 IDs.

Os YAMLs de comparação (`06_compara_*.yaml`) aplicam o filtro
`{"alvo": "teste", "fold": "<=10"}` na divisão CSV, garantindo que o modelo
zero-shot (A) e o professor, que têm saídas completas (22k docs), sejam avaliados
apenas sobre os mesmos 3.948 documentos. A coluna `fold` foi adicionada à divisão
CSV via join com as integras (`seq_documento_acordao` → `id`).

---

## 4. Juiz LLM

- **Modelo:** GPT-5 (`oa:gpt5-chat:m:l`) — deployment Azure
- **Reasoning:** Médio (`m`)
- **Verbose:** Low (`l`)
- **Rodadas:** 3 (para cálculo de variabilidade inter-rodada)
- **ROPE Likert:** 0.1579 — calibrada pela divergência média entre as 3 rodadas do juiz
- **Protocolos avaliados:** B, C, D24, D25 (conforme `PROTOCOLOS` em `07_avaliacao_llm.py`)

O `rope_likert` nos YAMLs de comparação (`06_compara_*.yaml`) não é usado pela
comparação Likert — esse cálculo é feito exclusivamente pelo `07_avaliacao_llm.py`.

---

## 5. Estado dos protocolos (SUMMA)

| Protocolo | Treino | Extração teste | Observação |
|---|---|---|---|
| A | — | ✅ 22k (filtrar na comparação) | Zero-shot, parquet completo |
| B, B16, B16R8 | ✅ | ✅ 3.948 | — |
| C | ✅ | ✅ 3.948 | Full FT 16b |
| D1–D10 | ✅ | ✅ 3.948 | D10 concluído em outra máquina |
| D11 | ❌ Não executado | ❌ | Diretório de treino inexistente |
| D12–D25 | ✅ | ✅ 3.948 | Verificar individualmente |
| Fold 12 (2026) | — | Pendente | Out-of-time, análise confirmatória separada |
