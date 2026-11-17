# Informações Técnicas dos Experimentos (SUMMA, SemClinBr e PubMed)

Registro do que foi efetivamente configurado e executado: números de cada experimento,
decisões de implementação, comportamentos observados e estado de cada protocolo.

> O desenho (protocolos, perguntas de pesquisa, comparabilidade) está em
> [README_protocolos.md](./README_protocolos.md). Os hiperparâmetros de cada protocolo, lidos dos
> YAMLs e conferidos contra os `training_metrics.jsonl`, estão em
> [RESUMO_EXPERIMENTOS.md](./RESUMO_EXPERIMENTOS.md) (gerado por `tabela_experimentos.py`).

Os três experimentos compartilham a mesma grade de 31 protocolos treinados (B, B16, B16R8, C,
D1–D25, D1a, D1b) mais o zero-shot A, com a mesma topologia de etapas. Mudam o modelo aluno, a
janela de contexto, o tamanho dos dados e a referência.

---

## 0. Visão geral

| Item | SUMMA | SemClinBr | PubMed |
|---|---|---|---|
| Modelo aluno | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct | Qwen2.5-1.5B-Instruct |
| Referência | Qwen3-235B-A22B-2507 (professor, OpenRouter ZDR) | Gold humano (`saidas/saida_semclinbr_gold.parquet`) | Gold do dataset (`saidas/saida_pubmed_prof.parquet`) |
| Arquivo de divisão | `dados/divisao_Qwen235b_Qwen7b.csv` | `dados/divisao_Gold_Qwen7B.csv` | `dados/divisao_Professor_Qwen1_5B.csv` |
| Divisão treino / val / teste | 15.508 / 2.217 / 4.430 | 697 / 103 / 197 | 14.996 / 2.499 / 2.500 |
| Usado após filtros (§4) | **13.788 / 1.976 / 3.948** (`fold <= 10`) | 697 / 103 / **197** | 14.996 / 2.499 / **2.500** |
| `max_seq_length` (treino) | 32.768 | 12.288 | 8.192 (B, B16, B16R8, C: 16.384) |
| `max_model_len` (inferência) | 38.912 | 12.288 | 8.192 (zero-shot A: 16.384) |
| Precisão na inferência | bitsandbytes 4-bit | bitsandbytes 4-bit | bf16 nativo |
| Lote efetivo | 16 | 16 | 16 (8 nas etapas `full` dos mistos, §3) |
| `warmup_steps` | 5 (D16–D25: 100) | 5 em todos | 5 (D16–D25: 100) |
| `warmup_grupo_steps` (fusão) | 100 | 5 | 100 |
| GPU de treino | H200 NVL (todos) | H200 NVL (todos) | H100 80GB (cluster Slurm) e H200 NVL |
| Avaliação | Métricas automáticas + juiz LLM | Métricas automáticas + NER-F1 + baseline ClinicalNERpt | Métricas automáticas |

LR por etapa é igual nos três: etapas `lora` em `2e-05`, etapas `full` em `5e-06`, seed `3407`
(exceções: D15 estabiliza em `1e-06`; D19/D20 usam `3e-06` na última etapa — ver RESUMO §3).
A GPU de cada treino está registrada em `treinos/<modelo>(<protocolo>)/RESUMO_TREINO.md`.

**Warmup do SemClinBr.** O SemClinBr tem ~20× menos steps por protocolo. Com `warmup_steps: 100`,
o warmup de D16–D20 era maior que as etapas e o LR não chegava ao pico; D16–D25 foram retreinados
com `warmup_steps: 5` e `warmup_grupo_steps: 5`, o que preserva a mesma **fração** dos steps em
warmup que 100 ocupa no PubMed e no SUMMA. Tabela e justificativa:
`semclinbr-experimento/README_SEMCLIMBR_EXPERIMENTO.md` §2.4.

---

## 1. Precisão na inferência (vLLM)

**SUMMA e SemClinBr:** todos os YAMLs de extração (`05_extracao_*_teste.yaml`) carregam os
modelos via vLLM com quantização bitsandbytes 4-bit on-the-fly, independentemente da precisão
usada no treino:

```yaml
vllm:
  quantization: bitsandbytes
  load_format: bitsandbytes
```

No SemClinBr, o Qwen2.5-7B em bf16 ocupa ~14 GB só de pesos. A quantização mantém a mesma
condição de inferência do SUMMA e deixa folga para o KV cache com `max_model_len: 12288`.

**PubMed (Qwen 1.5B):** sem quantização na inferência. O modelo ocupa ~3 GB em bf16; os YAMLs usam
apenas `enforce_eager: true`, e o `dtype: auto` resulta em bf16 nativo.

### Implicação observada

A quantização 4-bit na inferência é uniforme entre protocolos, então os contrastes *entre*
protocolos são comparáveis. Mas o caminho até ela não é o mesmo para todos:

- **LoRA puro** (B, B16, B16R8, D7–D10, D16–D18, D21–D24): o adaptador é aplicado sobre os pesos
  originais do Qwen2.5, que o vLLM quantiza uma única vez.
- **Full e mistos** (C, D1–D6, D1a, D1b, D11–D15, D19, D20, D25): o modelo é salvo em bf16
  (`config.json` sem `quantization_config`) e o vLLM quantiza os pesos já modificados. Nos mistos
  com etapas LoRA em 4 bits, esses pesos já passaram por NF4 durante o treino (§2), então a
  quantização da inferência é a segunda.

É uma limitação do setup de inferência a declarar ao comparar LoRA vs Full FT no SUMMA e no
SemClinBr. No PubMed ela não existe na inferência, apenas no treino.

---

## 2. Recarga, merge e quantização nos pipelines com troca de regime

### Regra geral

Etapas `full` sempre rodam em 16 bits; etapas `lora` usam o `nbits` global do YAML. O modelo é
descarregado e recarregado quando a etapa seguinte exige outra precisão
(`treinar_unsloth.py`, `_aplicar_etapa_curriculum()`):

```python
alvo_nbits = 16 if etapa.tipo == "full" else self._yaml_config.treinamento.nbits
precisa_recarregar = alvo_nbits != nbits_memoria
```

A recarga também é forçada quando o tipo da etapa muda e o pacote `flash-attn` não está instalado
com `full_com_sdpa: true` (troca de implementação de atenção). Toda fronteira real reinicia o
otimizador Adam e o scheduler cosine; no modo fusão (`curriculum.fusao.ativo: true`) há um único
`trainer.train()`, e as fronteiras viram marcadores de log.

Antes de recarregar, o merge LoRA→base (`merge_adapter()`) só acontece se o pipeline tem etapa
`full` **e** o LoRA já está aplicado. Na prática:

- **LoRA→Full:** merge na fronteira, salvamento e recarga em 16 bits.
- **Full→LoRA:** **não há merge** (ainda não existe adaptador). Os pesos da etapa `full` são salvos
  em bf16 e recarregados com o `nbits` global. Com `nbits: 4`, os pesos treinados em full são
  quantizados para NF4 nesse ponto.
- **LoRA→LoRA com o mesmo `nbits`:** sem recarga. O adaptador permanece em VRAM e acumula
  treinamento entre etapas, sem reinicialização.

### Salvamento ao fim de cada etapa

`_save_model()` é chamado ao fim de toda etapa (duas vezes: antes e depois do eval final). Em
pipelines com etapa `full` e LoRA aplicado, faz `merge_adapter() → save_pretrained() →
unmerge_adapter()` e remove os arquivos do adaptador, de modo que a pasta de saída sempre contém um
modelo full auto-contido. Pipelines LoRA puro salvam só o adaptador.

### Mapa por família de protocolo

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

No modo fusão o regime e a precisão são únicos no run, e o `unfreeze_layers_from` de cada span
vira gating de LR por grupo de camadas, sem congelamento. Com `fusao.tipo: full`, o `nbits` é
forçado a 16.

### Erro de quantização nos mistos com `nbits: 4`

Quando o merge ou o unmerge ocorre sobre um modelo em 4 bits (bitsandbytes NF4), o PEFT
dequantiza cada camada, soma (ou subtrai) o delta LoRA e requantiza, recriando o `Params4bit`
(`peft/tuners/lora/bnb.py`, `Linear4bit.merge/unmerge`). O framework emite warning explícito no
merge de fronteira.

- **LoRA→Full (D2, D4, D6, D12, D14, D15):** o merge de fronteira opera sobre a base NF4. As
  etapas `full` seguintes partem desses pesos, recarregados em 16 bits.
- **Full→LoRA (D1, D1a, D1b, D3, D5, D11):** não há merge na fronteira, mas os pesos treinados em
  full são quantizados para NF4 ao entrar nas etapas LoRA. O modelo final é o merge do LoRA sobre
  essa base NF4, salvo em bf16, e carrega o erro de quantização dos pesos FF.
- **Ambas as direções:** o par `merge_adapter()`/`unmerge_adapter()` feito por `_save_model()` ao
  fim de cada etapa LoRA requantiza a base NF4 em memória. A cada salvamento, a base sobre a qual
  o adaptador continua treinando sofre um pequeno arredondamento adicional. Não ocorre em
  pipelines LoRA puro, que não fazem merge.

Os protocolos sem etapa LoRA em 4 bits não têm essa variável por construção: D16–D18, D21–D23,
D19, D20 e D25. D24 é LoRA 4b fundido: não faz merge, mas treina sobre base NF4, como o B.

---

## 3. Lote efetivo e override de `batch_size` por etapa

`batch_size.efetivo` manda em `grad_batch_size`: `_aplicar_batch_size_auto`
(`src/treinar_unsloth_util.py`) recalcula `grad_batch_size = efetivo / (batch_size × n_gpus)` e
sobrescreve o valor escrito no YAML.

O override de `batch_size` por etapa **não** recalcula o acúmulo: a etapa troca o `batch_size`,
mas mantém o `grad_batch_size` global.

- **SUMMA e SemClinBr:** `batch_size` global é 1 e as etapas que declaram `batch_size: 1` repetem o
  mesmo valor. Todas as etapas rodam com lote efetivo 16.
- **PubMed:** `batch_size` global é 2. As etapas `full` de D1–D6, D1a, D1b, D11 e D12 declaram
  `batch_size: 1` e rodam com lote efetivo 8 (confirmado no evento `train_begin` do
  `training_metrics.jsonl`: `effective_batch_size: 8`). O RESUMO_EXPERIMENTOS lista 16 como
  constante porque lê o valor global.

---

## 4. Conjuntos de teste e filtros por experimento

O mesmo recorte é aplicado nas três pontas: extração do teste (`05_*`), entrada do treino (`04_*`)
e comparação (`06_*`). A sintaxe dos filtros está em [README_filtros.md](./README_filtros.md).

| | SUMMA | SemClinBr | PubMed |
|---|---|---|---|
| Extração (`05_*` → `entrada.filtro`) | `dataset_filtro: {"fold": "<=10"}` nas íntegras + `filtro_externo` na divisão com `{"alvo": "teste"}` | `filtro_externo` na divisão com `{"alvo": "teste"}` | `dataset_filtro: {"split": "test"}` |
| Treino (`04_*` → `curriculum.entrada.dataset_filtro`) | `{"fold": "<=10"}` | — | — |
| Comparação (`06_*` → `configuracao_comparacao.filtro`) | divisão CSV, `campo_id: id`, `{"alvo": "teste", "fold": "<=10"}` | divisão CSV, `campo_id: id`, `{"alvo": "teste"}` | `dados/pubmed-rct-20k.parquet`, `campo_id: pmid`, `{"split": "test"}` |
| Saídas completas filtradas na comparação | A e professor (22.269 docs) | A e gold (1.000 docs) | A e gold (19.995 docs) |

### SUMMA (3.948 documentos de teste)

As íntegras (`dados/integras_experimento_summa_novos.parquet`) têm 22.471 documentos em 12 folds:
folds 1–11 com 2.000 cada e o fold 12 com 471. A divisão tem 22.155: ficaram de fora 202
documentos sem saída do professor e 114 em que o Qwen 7B zero-shot falhou na extração (o
`03_compara_q235_full.yaml` roda com `ignorar_erro_extracao: true`).

O filtro `fold <= 10` reduz o teste de 4.430 para 3.948 documentos. Os 482 restantes pertencem
aos folds 11 (388 > piloto qualificação) e 12 (94 > inéditos):

- **Fold 11:** holdout intermediário.
- **Fold 12:** acórdãos publicados em 2026, posteriores ao cutoff de todos os modelos (469 na
  divisão, de 471 nas íntegras). Reservado para a validação out-of-time, ainda não executada
  (`summa-experimento/README_SUMMA_EXPERIMENTO.md` §7). A coluna `alvo` não vale para ele: o
  filtro `fold <= 10` o removeu de treino, validação e teste em todos os protocolos.

A coluna `fold` foi adicionada à divisão CSV via join com as íntegras (`seq_documento_acordao` →
`id`). Para o juiz LLM, o `07_avaliacao_llm.py` gera `saida_qwen7b(a)_teste.parquet` e
`saida_or_235b(prof)_teste.parquet` já filtrados.

### SemClinBr (197 documentos de teste)

A divisão tem 997 dos 1.000 documentos do corpus (3 não anotados descartados). Com 697 documentos
de treino, os protocolos granulares de 10 etapas (D11, D12, D24, D25) ficam com ~70 instâncias por
faixa, e as duas primeiras etapas de D11 e D12 têm 10 e 18 steps (o warmup 5 ocupa 50% e 28%
delas — RESUMO §6). Os YAMLs rodam com uma seed (`3407`); a replicação por seed fica para uma
segunda rodada.

### PubMed (2.500 documentos de teste)

O split oficial `test` do PubMed 20k RCT. A divisão preserva os splits oficiais
(`dados/divisao_pubmed.csv`, colunas `id_arquivo` e `alvo`), passada como `arquivo_referencia`
no `03_compara_prof_full.yaml`.

### Faixas de dificuldade

Nos três experimentos, `dificuldade_int` vai de 1 a 10 (`facil` = 1–3, `medio` = 4–7,
`dificil` = 8–10), na proporção 30/40/30. O D18 usa uma cópia da divisão com a coluna `bloco`
(terços aleatórios, seed 3407), gerada por `gerar_divisao_blocos_d18.py` em cada experimento
(`dados/divisao_*_blocos.csv`).

---

## 5. Avaliação

### 5.1 Métricas automáticas e ROPE (`06_compara_todos.yaml`)

As métricas automáticas medem similaridade com a referência — fidelidade de destilação no SUMMA,
acerto contra o gold no PubMed e no SemClinBr. A ROPE é a margem de empate sobre os escores
brutos, calibrada pelas réplicas D1, D1a e D1b (`06_compara_d1ab.yaml` →
`estatisticas/00_rope_sugerido.md`; procedimento no
[README_PASSO_A_PASSO.md, etapa 06](./README_PASSO_A_PASSO.md#06--comparação-dos-protocolos)).

| Experimento | Campos × métricas | `rope` (fallback) | `rope_por_campo` | Verificação (2ª rodada) |
|---|---|---|---|---|
| SUMMA | Likert, MetaDados × BERTScore, SBERT médio, ROUGE-L | 0,0179 | Likert 0,0153 · MetaDados 0,0179 | ✅ 3/3 pares equivalentes nos 6 campos × métricas (execução completa, 15/11/2026) |
| SemClinBr | (global), Entidades, Relacoes × ROUGE-L, ROUGE-2 | 0,0401 | (global) 0,0188 · Entidades 0,0162 · Relacoes 0,0401 | ✅ 3/3 pares equivalentes em todos os campos |
| PubMed | (global), MetaDados, Sentencas × ROUGE-L | 0,0058 | (global) 0,0030 · MetaDados 0,0058 · Sentencas 0,0022 | ✅ 3/3 pares equivalentes em todos os campos |

A ROPE do SUMMA foi recalibrada em 13/10/2026: a 1ª rodada usou 0,0132 (Likert 0,0121) e deixou
D1 × D1b `incerto` nas três métricas; o `06_compara_todos.yaml` (relatório de 14/10) já usa os
valores sugeridos por ela. O valor de cada campo é fixado pelo BERTScore no par D1 × D1b; o
ROUGE-L (Likert 0,0096) e o SBERT médio (Likert 0,0146 · MetaDados 0,0167) exigiriam menos.

> **Réplica D1b.** As duas rodadas usaram a inferência do D1b feita após o retreino (H200,
> 13/10/2026). A cópia local de `saida/saida_qwen7b(d1b)_teste.parquet` era, até 15/11/2026, a
> inferência de 29/09 de um modelo D1b anterior ao retreino; foi substituída nessa data. A
> planilha da execução completa de 15/11 confere com os parquets locais de D1, D1a e D1b.
>
> **Bloco YAML do `00_rope_sugerido.md`.** Até 16/11/2026 ele listava `sbert` em vez de
> `sbert_medio`. Copiado literalmente, a análise ignorava o SBERT médio sem aviso (a coluna
> procurada era `*_sbert_F1`). Os dois problemas foram corrigidos no código: o nome sugerido e o
> aviso para métrica sem coluna. Foi o que aconteceu na execução parcial de 15/11 (só
> `--estatisticas`, 4 mapas em vez de 6): a saída é idêntica à de um YAML com `sbert`, reproduzida
> sobre o mesmo CSV. A execução completa, com
> `sbert_medio`, analisou as três métricas. Confira se há um mapa por campo × métrica.

O `rope_likert: 0.01` dos YAMLs de comparação é um placeholder. Só teria efeito se
`campos_parquet.avaliacao` apontasse para as notas do juiz, o que hoje não acontece: a análise
Likert é feita exclusivamente pelo `07_avaliacao_llm.py` (§5.2).

### 5.2 Juiz LLM (SUMMA)

O juiz dá nota Likert 1–4 a cada extração do teste (prompt
`avaliacao_llm_humana/prompt_juiz_llm.txt`). Extrações inválidas não vão ao juiz e recebem nota 1.
Uso do script, perfis e retomada: `summa-experimento/07_avaliacao_llm.md`.

- **Juiz principal:** GPT-5 em Azure (perfil `gpt5`, pasta `avaliacao_llm/`), reasoning médio e
  verbose low, em duas implantações do mesmo modelo (`oa:gpt5:m:l` e `oa:gpt5-chat:m:l`). O
  modelo usado em cada protocolo está na tabela de custos do `RESUMO_AVALIACAO_LLM.md` e no
  `config_juiz_{p}.yaml`.
- **Passadas:** uma por protocolo. A variabilidade do juiz é estimada pelas réplicas.
- **Cobertura:** base (GPT-5), prof, A, B, C e D1–D25, 3.948 itens cada, mais D1a e D1b para a
  calibração. B16 foi avaliado depois (09/11) e ainda não está na lista `PROTOCOLOS` do script
  nem no resumo; B16R8 não foi avaliado.
- **ROPE Likert:** calibrada pelas réplicas D1, D1a e D1b (`ROPE_PROTOCOLOS`) com o
  `calibrar_rope()` de `src/realizar_avaliacoes.py`: para cada par, a menor margem em que ele sai
  `equivalente` ao limiar 0,95; a ROPE é a maior delas. Último `--resumo` (15/11/2026):
  **0,0307**, com as três réplicas avaliadas pela mesma implantação (`oa:gpt5-chat:m:l`). O
  relatório repete a comparação com ROPE/2 e 2·ROPE. O valor antigo, 0,1579, vinha da divergência
  entre avaliadores humanos e está desativado (`ROPE_LIKERT = None`).
- **Reavaliação do D1b (15/11/2026):** até essa data, os três juízes tinham avaliado a inferência
  do D1b de 29/09, anterior ao retreino (ver §5.1). O D1b foi reavaliado nos três juízes com a
  inferência pós-retreino (3.732 itens válidos + 216 inválidos com nota 1). Resumos e recortes
  foram regenerados. ROPE antes → depois: GPT-5 0,0314 → 0,0307; sabia5 0,0405 → 0,0492; glm52
  0,0508 → 0,0398.
- **Recortes:** `--resumo --recorte <Qx> --protocolos ...` grava um resumo por questão de
  pesquisa em `avaliacao_llm/recortes/<recorte>/`, com a mesma ROPE. Os comandos estão no
  cabeçalho do `07_avaliacao_llm.py`. Recortes atuais do GPT-5: Q1, Q2a–Q2c, Q3a, Q3b, Q4a, Q4b,
  Q5, Q6a–Q6c e Melhores_D9_D22.

**Robustez ao juiz.** Dois juízes adicionais, em perfis isolados (pastas e cópias de parquet
próprias, sem tocar nas notas do GPT-5), verificam a sensibilidade das conclusões ao juiz. Cada
um tem a própria ROPE, calibrada nas réplicas avaliadas por ele:

| Perfil | Modelo | Protocolos avaliados | ROPE Likert | Último resumo |
|---|---|---|---|---|
| `sabia5` | `mt:S5-free:m:l` | os mesmos 30 do GPT-5 + D1a, D1b | 0,0492 | 15/11/2026 |
| `glm52` | `or:z-ai/glm-5.2:floor:m:l` | base, prof, A, D1, D9, D22 + D1a, D1b | 0,0398 | 15/11/2026 |

No glm52, D1 e D1a foram avaliados em 11/11 sem o sufixo `:m:l` (reasoning padrão do provedor),
e D1b em 15/11 com ele. Os três raciocinaram (≈1.780, 1.900 e 1.920 tokens de saída por item),
mas o resumo avisa que a ROPE inclui essa diferença de configuração.

**Validação do juiz contra humanos** (`avaliacao_llm_humana/`): 70 documentos estratificados ×
3 fontes (GPT-5, Qwen 235B, Qwen 7B), 3 rodadas por juiz candidato, analisados com
`realizar_avaliacoes.py` (κw ponderado, equivalência bayesiana das notas e da decisão adequado/
inadequado). A coleta humana via Label Studio está em andamento (um avaliador até agora). Não
entra no artigo do SAC; será reportada na dissertação.

### 5.3 NER (SemClinBr)

O SemClinBr não usa juiz LLM. A variável primária é o NER-F1 do `07_avaliar_ner.py`
(strict/lenient/flexible/relaxed, por documento de teste e por protocolo, sobre os mesmos parquets
dos passos 02 e 05; saída em `compara/ner/`). Estatística: Friedman + Wilcoxon com Holm + tamanho
de efeito. O `08_baseline_clinicalnerpt.py` roda os modelos `pucpr/clinicalnerpt-*` (BioBERTpt)
no mesmo split de teste como baseline externo (`saidas/avaliacao_ner/comparacao_baseline.*`).

---

## 6. Estado dos protocolos (15/11/2026)

Todos os 31 protocolos estão treinados, extraídos no teste e incluídos no `06_compara_todos.yaml`
nos três experimentos. Alguns treinos e extrações rodaram em outros servidores e só os artefatos
foram copiados para cá (pastas `treinos/` podem conter apenas métricas). A conferência
YAML × treino real do RESUMO_EXPERIMENTOS (§6) passa em todos: nenhum treino ficou com versão
antiga do YAML e o LR logado confere com o esperado.

As taxas de JSON inválido abaixo referem-se ao parquet de teste de cada protocolo.

| | SUMMA | SemClinBr | PubMed |
|---|---|---|---|
| Teste por protocolo | 3.948 | 197 | 2.500 |
| Zero-shot A | 22.269 (filtrado na comparação); `saida_qwen7b(a)_teste.parquet` com 3.948 para o juiz | 1.000 (filtrado na comparação) | 19.995 (filtrado na comparação) |
| JSON inválido | C, D1–D6, D1a, D1b, D11–D14: 3,3%–7,3% (máx. D11, 7,29%); D15: 1,2%; demais ≤ 0,51% (incluindo D19, D20, D25) | ≤ 4,06% (máx. D3; cada documento vale 0,51%) | ≤ 0,2%, exceto D24 (1,12%) |
| Comparação `06_compara_todos` | 14/10/2026 | 13/10/2026 (após retreino de D16–D25 com warmup 5) | 25/09/2026 |
| Avaliação adicional | Juiz LLM (§5.2) | NER-F1 + ClinicalNERpt (§5.3) | — |
| Pendente | recortes `prof-equivalentes` e `Q3a` do sabia5 (ROPE antiga); validação out-of-time no fold 12 | — | — |

No SUMMA, as falhas de JSON concentram-se nos protocolos com etapa `full` segmentada; os full sem
fronteira de regime (D19, D20, D25) ficam ≤ 0,51%.

Os arquivos `06_compara_todos_parcial.yaml` (SUMMA) e `06_compara_todos(parcial).yaml`
(SemClinBr) serviram aos relatórios incrementais durante o treinamento e não são mais necessários.
