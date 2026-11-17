# Passo a passo para replicar um experimento

Sequência de execução do pipeline, da preparação dos dados às figuras. **O número de cada etapa é o
prefixo dos YAMLs dela** (`02_*`, `03_*`, … `07_*`), em todos os experimentos.

Os comandos rodam de dentro da pasta do experimento e usam o **Summa** como exemplo. Para PubMed e
SemClinBr, os arquivos equivalentes estão na tabela abaixo; os detalhes de domínio (dataset, prompt,
gabarito) ficam no README de cada pasta.

> **Antes de começar:** ambiente (CUDA, PyTorch, vLLM, Flash-Attn) em [README.md](./README.md) ·
> desenho dos protocolos e perguntas de pesquisa em [README_protocolos.md](./README_protocolos.md) ·
> sintaxe dos filtros em [README_filtros.md](./README_filtros.md) · números e estado de cada
> experimento em [INFORMACOES_TREINAMENTO.md](./INFORMACOES_TREINAMENTO.md).

## Arquivos por etapa

| Etapa | Summa | PubMed | SemClinBr |
| :-- | :-- | :-- | :-- |
| 01 Dados | `util_ckan.py --config 01_extracao.yaml` | `python util_pubmed.py` | `python util_semclinbr.py` (corpus sob solicitação ao HAILab) |
| 02 Linha de base | `02_summa_235b.yaml` (professor), `02_summa_7b_hpc.yaml` (A), `02_summa_gpt5.yaml` (GPT-5) | `02_pubmed_rct_1_5b.yaml` (A) | `02_semclinbr_7b.yaml` (A) |
| Referência | `saida/saida_or_235b.parquet` (etapa 02) | `saidas/saida_pubmed_prof.parquet` (etapa 01) | `saidas/saida_semclinbr_gold.parquet` (etapa 01) |
| 03 Divisão | `03_compara_q235_full.yaml` | `03_compara_prof_full.yaml` | `03_compara_gold_full.yaml` |
| 04 Treino | `04_treinar_<p>.yaml` (31) | `04_treinar_<p>.yaml` (31) | `04_treinar_<p>.yaml` (31) |
| 05 Teste | `05_extracao_<p>_teste.yaml` | `05_extracao_<p>_teste.yaml` | `05_extracao_<p>_teste.yaml` |
| 06 Comparação | `06_compara_d1ab.yaml`, `06_compara_todos.yaml` | idem | idem |
| 07 Avaliação específica | `07_avaliacao_llm.py` (juiz LLM) | — | `07_avaliar_ner.py`, `08_baseline_clinicalnerpt.py` |
| Jobs Slurm | `job_treinar_*.sh`, `job_export_7b_testes.sh`, `job_compara_testes.sh` | `job_treinar_*.sh`, `job_export_1_5b_testes.sh`, `job_compara_testes.sh` | `job_treinar_*.sh`, `job_export_7b_testes.sh`, `job_compara_testes.sh` |

`<p>` é o protocolo (`b`, `b16`, `b16r8`, `c`, `d1`–`d25`, `d1a`, `d1b`). Os YAMLs declaram em
`misc.pastas_base` e `misc.pastas_modelos` os caminhos aceitos para o experimento e para os
modelos base; acrescente os da sua máquina.

---

## 01 — Preparação dos dados

Gera o parquet com os textos de entrada (e, no PubMed e no SemClinBr, o gabarito humano no formato
do framework).

```bash
# Summa: extrai os acórdãos do portal de dados abertos do STJ (CKAN)
python ../../src/util_ckan.py --config 01_extracao.yaml
# Só exibe o parquet final já gerado, sem reextrair
python ../../src/util_ckan.py --config 01_extracao.yaml --view
```

No PubMed (`util_pubmed.py`) e no SemClinBr (`util_semclinbr.py`) o script da pasta lê o dataset
original e grava `dados/*.parquet`, o prompt e `saidas/saida_*_{prof,gold}.parquet`.

---

## 02 — Linha de base e referência (sem filtro)

Extrai o corpus **inteiro** com o modelo base zero-shot (protocolo A) e, no Summa, com o professor
(que gera a referência) e com o GPT-5. Sem filtro: a etapa 03 precisa de todos os documentos para
estratificar a dificuldade.

```bash
python ../../src/util_vllm_batch.py --config 02_summa_235b.yaml   # professor (OpenRouter, ZDR)
python ../../src/util_vllm_batch.py --config 02_summa_7b_hpc.yaml # modelo base local (vLLM)
python ../../src/util_vllm_batch.py --config 02_summa_gpt5.yaml   # GPT-5 (Azure)
```

As chaves de API são lidas de variáveis de ambiente (lista no README do experimento). A extração é
retomável: rodar de novo refaz só os itens ausentes ou com erro.

---

## 03 — Comparação inicial: divisão e dificuldade

Compara o modelo base contra a referência e **gera o arquivo de divisão**: `alvo`
(treino/validação/teste) e `dificuldade` / `dificuldade_int` (1–10) de cada documento.

```bash
python ../../src/comparar_extracoes.py --config 03_compara_q235_full.yaml
```

Se já existe uma divisão a preservar (splits oficiais, ou uma divisão anterior), informe-a em
`execucao.divisao.arquivo_referencia`; sem ela, o script sorteia um split novo
([README_filtros.md §6](./README_filtros.md#6-o-csv-de-divisão-onde-nascem-dificuldade-e-alvo)).

> ⚠️ **Copiar a divisão para `dados/`.** O script grava em
> `compara/<saida.pasta>/divisoes/divisao_<base>_<modelo>.csv`, mas as etapas 04–06 leem só de
> `dados/`. Copie **antes** de apagar ou compactar a pasta de análise, e versione o arquivo: ele é a
> semente de reprodutibilidade de todo o resto.
> ```bash
> cp "compara/analises_comparacao_summa_q235 (full)/divisoes/divisao_Qwen235b_Qwen7b.csv" dados/
> cp "compara/analises_comparacao_pubmed (full)/divisoes/divisao_Professor_Qwen1_5B.csv" dados/   # PubMed
> cp "compara/analises_comparacao_semclinbr (full)/divisoes/divisao_Gold_Qwen7B.csv" dados/       # SemClinBr
> ```

**Summa — coluna `fold`.** Os filtros do Summa usam `fold` (folds 11 e 12 ficam fora do
experimento). Acrescente-a à divisão copiada:

```python
import pandas as pd
arq = "dados/divisao_Qwen235b_Qwen7b.csv"
div = pd.read_csv(arq)
folds = pd.read_parquet("dados/integras_experimento_summa_novos.parquet",
                        columns=["seq_documento_acordao", "fold"])
div = div.merge(folds.rename(columns={"seq_documento_acordao": "id"}), on="id", how="left")
div.to_csv(arq, index=False)
```

**D18 — blocos aleatórios.** Gere a cópia da divisão com a coluna `bloco`:

```bash
python gerar_divisao_blocos_d18.py   # grava dados/divisao_*_blocos.csv
```

---

## 04 — Treinamento

Cada protocolo tem um YAML pronto. Antes de treinar, confira o recorte de dados de cada etapa:

```bash
# Relatório de instâncias e filtros de cada etapa (dry-run, não treina)
python ../../src/treinar_unsloth.py --datasets 04_treinar_d1.yaml
```

```bash
python ../../src/treinar_unsloth.py --treinar 04_treinar_d1.yaml
```

Para treinos longos, use `tmux` (ver [README.md](./README.md#execução-em-background-tmux--nohup))
ou os jobs Slurm da pasta (`sbatch job_treinar_<grupo>.sh`, cada um com a lista de YAMLs em
`CONFIGS`). Com `resume_from_checkpoint: true`, rodar de novo retoma do último checkpoint. Para
retreinar do zero, renomeie a pasta de saída (preserva o treino antigo) ou use `--reset`
(apaga checkpoints e modelo).

Pelo menos um protocolo precisa de **3 réplicas** para calibrar a ROPE na etapa 06: nos
experimentos, `d1`, `d1a` e `d1b` (mesmos YAMLs, só a pasta de saída muda).

Cada treino grava em `treinos/<modelo>(<protocolo>)/` os pesos (ou o adaptador) e o
`RESUMO_TREINO.md`; em `treinamento/` ficam o `training_metrics.jsonl`, os relatórios e os
gráficos.

---

## 05 — Extração no conjunto de teste

```bash
python ../../src/util_vllm_batch.py --config 05_extracao_d1_teste.yaml
# ou todos em lote no cluster:
sbatch job_export_7b_testes.sh
```

Cada `05_extracao_<p>_teste.yaml` aponta para os pesos em `treinos/` e já restringe a entrada ao
teste (filtros de cada experimento em
[INFORMACOES_TREINAMENTO.md §4](./INFORMACOES_TREINAMENTO.md#4-conjuntos-de-teste-e-filtros-por-experimento)).
A saída é `saida/saida_qwen7b(<p>)_teste.parquet` no Summa e
`saidas/saida_<modelo>(<p>)_teste.parquet` nos demais.

---

## 06 — Comparação dos protocolos

Métricas automáticas (BERTScore, SBERT, ROUGE-L/2, Levenshtein, conforme o experimento) e análise
estatística por recorte — bayesiana como principal, frequentista exploratória. Primeiro calibra-se a
ROPE com as réplicas; depois compara-se tudo.

### 06a — Calibrar a ROPE (duas rodadas)

**1ª rodada.** Rode a comparação das réplicas com qualquer ROPE provisória > 0:

```bash
python ../../src/comparar_extracoes.py --config 06_compara_d1ab.yaml
```

Saída: `compara/analises_comparacao_<exp>_d1ab/estatisticas/00_rope_sugerido.md`, com o par mais
exigente por campo e o bloco YAML sugerido (a menor ROPE sob a qual **todos** os pares de réplicas
saem `equivalente` ao limiar 0,95).

**Transcrição.** Copie `rope` e `rope_por_campo` do bloco sugerido para
`estatistica.metricas_automaticas` do próprio `06_compara_d1ab.yaml`. Exemplo (SemClinBr):

```yaml
metricas_automaticas:
  rope: 0.0401
  rope_por_campo:
    (global): 0.0188
    Entidades: 0.0162
    Relacoes: 0.0401
```

Transcreva só `rope` e `rope_por_campo`. Relatórios gerados antes de 16/11/2026 listam em
`metricas` o nome `sbert` no lugar de `sbert_medio`. Hoje, uma métrica do YAML sem coluna nos
resultados gera um aviso `⚠️ <campo> × <métrica>: nenhuma coluna ...` no terminal e no relatório
bayesiano.

**2ª rodada — verificação de sanidade.** Refaça só a camada estatística:

```bash
python ../../src/comparar_extracoes.py --config 06_compara_d1ab.yaml --estatisticas
```

A calibração só está confirmada quando, em `bayesiana/analise_bayesiana.md`, **100% dos pares**
(`D1 × D1a`, `D1 × D1b`, `D1a × D1b`) saem `equivalente` em todos os campos e métricas. Se algum par
sair `incerto` ou com diferença, investigue antes de seguir. Confira também se há um mapa para cada
campo × métrica do YAML.

> ⚠️ **Dois cuidados no `06_compara_d1ab.yaml`:**
> 1. **Filtro do teste.** O `configuracao_comparacao.filtro` precisa restringir ao teste (a
>    referência tem o corpus inteiro e as réplicas só o teste) — ver
>    [README_filtros.md §5.1](./README_filtros.md#51-o-bloco-filtro).
> 2. **Nunca usar `"TODOS"` em `estatistica.protocolos`** com `calibracao_rope: true`. `"TODOS"`
>    inclui o protocolo virtual `★Prof` (referência = 1,0), e a calibração passaria a medir a
>    distância aluno–referência em vez do ruído entre réplicas, inflando a ROPE. Liste as réplicas:
>    ```yaml
>    estatistica:
>      calibracao_rope: true
>      protocolos:
>        Replicas_D1: [D1, D1a, D1b]
>    ```

### 06b — Comparar todos os protocolos

Copie o mesmo `rope` / `rope_por_campo` para `06_compara_todos.yaml` e rode:

```bash
python ../../src/comparar_extracoes.py --config 06_compara_todos.yaml
# ou, no cluster:
sbatch job_compara_testes.sh
```

Saída em `compara/analises_comparacao_<exp>_todos/`: planilhas, `relatorio_analise.md`,
`estatisticas/`, `bayesiana/` (heatmaps e relatório por recorte Q1–Q6) e `graficos/`. Mudanças só
nos recortes (`estatistica.protocolos`) ou na ROPE exigem apenas `--estatisticas`; só nos gráficos,
`--graficos`.

---

## 07 — Avaliações específicas de cada experimento

### Summa — juiz LLM sobre o teste

O juiz (GPT-5 por padrão) dá nota Likert 1–4 a cada extração dos 3.948 documentos de teste e grava
a coluna `avaliacao` no parquet do protocolo.

```bash
python 07_avaliacao_llm.py --protocolos d1 --tokens   # estima o custo antes
python 07_avaliacao_llm.py --protocolos d1 d1a d1b    # avalia (retoma se interrompido)
python 07_avaliacao_llm.py --resumo                   # RESUMO_AVALIACAO_LLM.md, ROPE calibrada nas réplicas
python 07_avaliacao_llm.py --resumo --recorte Q4b_unfreeze --protocolos b c d20 d19
python 07_avaliacao_llm.py --juiz sabia5 --protocolos d1 d1a d1b   # outro juiz, em pasta isolada
```

Parâmetros, perfis de juiz e retomada: `summa-experimento/07_avaliacao_llm.md`.

### SemClinBr — NER-F1 e baseline externo

```bash
python 07_avaliar_ner.py --config 07_avaliar_ner.yaml                         # compara/ner/
python 08_baseline_clinicalnerpt.py --config 08_baseline_clinicalnerpt.yaml   # saidas/avaliacao_ner/
```

---

## 08 — Validação do juiz e avaliação humana (Summa, opcional)

Mede se o juiz LLM concorda com especialistas, em 70 documentos estratificados × 3 fontes (GPT-5,
Qwen 235B, Qwen 7B). Fica em `summa-experimento/avaliacao_llm_humana/`. A coleta humana exige
aprovação do Comitê de Ética em Pesquisa (CEP).

```bash
cd avaliacao_llm_humana/

# Juízes LLM candidatos: monta a entrada (210 linhas) e roda cada juiz em 3 rodadas
python 01_gerar_entrada_juiz_llm.py
python ../../../src/util_vllm_batch.py --config 02_extracao_70.yaml   # ajustar modelo e saida_<juiz>_0N

# Avaliadores humanos (Label Studio): gera um JSON de tarefas por avaliador
python gerar_tarefas_label_studio.py
# após a anotação, converte a exportação do Label Studio para o formato do pipeline
cd ../avaliacao_humana/
python converter_label_studio_para_parquet.py --saida ../avaliacao_llm_humana <exportacao>.json --prefixo saida_humana
cd ../avaliacao_llm_humana/

# Concordância juiz × humano (κw, equivalência bayesiana das notas e da decisão)
python ../../../src/realizar_avaliacoes.py \
    --grupos saida_gpt:llm saida_sabia5:llm saida_humana:humano --bayes
```

Os comandos usados até aqui estão em `avaliacao_llm_humana/03_realizar_avaliacoes.md`, e a
configuração do Label Studio em `avaliacao_llm_humana/README_AVALIACAO.md`.

---

## 09 — Consolidação e figuras

```bash
# Na pasta experimentos/: confere os YAMLs e os treinos dos três experimentos
python tabela_experimentos.py          # gera RESUMO_EXPERIMENTOS.md

# Na pasta de cada experimento: copia os gráficos de treino para treinos/graficos/
python ../../src/treinar_consolidar_graficos.py
```

As figuras de comparação são geradas na etapa 06 (`compara/analises_comparacao_*/graficos/` e
`bayesiana/`) e as do juiz na etapa 07 (`avaliacao_llm/`).

---

## Checklist de integridade

1. A divisão em `dados/` é a que gerou os treinos (mesmo arquivo versionado, com as colunas extras
   do experimento).
2. `RESUMO_EXPERIMENTOS.md` §6 sem pendências: todo treino com a versão atual do YAML e LR logado
   igual ao esperado.
3. Todos os parquets de teste com o mesmo número de linhas (Summa 3.948 · PubMed 2.500 ·
   SemClinBr 197).
4. No `execucao.log` de cada comparação: `Filtro carregado: N IDs`, `IDs na origem (a processar): N`
   e `Total comparado: N` coincidem com o tamanho do teste.
5. Mesmo filtro e mesmo `ignorar_erro_extracao` (`false`) em todos os YAMLs de comparação
   confrontados.
6. Calibração da ROPE confirmada (100% dos pares de réplicas `equivalente`) antes de ler os recortes.

---

## Resumo visual do pipeline

```
01 Dados ──► 02 Linha de base + referência (corpus inteiro)
                       │
                       ▼
             03 Comparação inicial ──► divisão (alvo + dificuldade) ──► copiar para dados/
                       │
                       ▼
             04 Treinamento (31 protocolos) ──► 05 Extração no teste
                                                       │
                       ┌───────────────────────────────┘
                       ▼
             06a Calibrar ROPE (d1 × d1a × d1b) ──► 06b Comparar todos (recortes Q1–Q6)
                                                       │
                       ┌───────────────────────────────┤
                       ▼                               ▼
             07 Juiz LLM (Summa) /             08 Validação do juiz
                NER-F1 (SemClinBr)                × humanos (Summa)
                       │                               │
                       └───────────────┬───────────────┘
                                       ▼
                             09 Consolidação e figuras
```
