# Passo a passo para rodar um experimento

O passo a passo é didático e simplificado, cada etapa pode ser melhor entendida, incluindo seus parâmetros, em READMEs específicos.
É possível fazer o experimento com ou sem LLM-as-a-judge e com ou sem avaliadores humanos.

> 📖 **Para detalhes de ambiente (CUDA, vLLM, Flash-Attn), veja o [README.md](./README.md).**  
> 📖 **Para o design dos protocolos e perguntas de pesquisa, veja o [README_protocolos.md](./README_protocolos.md).**  
> 📖 **Para filtros de dados, veja o [README_filtros.md](./README_filtros.md).**

Os exemplos abaixo usam o experimento **Summa** como referência. Para outros experimentos (PubMed, SemClinBr, Puil-Mini), basta entrar na pasta correspondente e usar os YAMLs homônimos.

---

## 01 — Escolha do modelo base

Defina o modelo que será ajustado (aluno) e, opcionalmente, o modelo professor para destilação de conhecimento. Os modelos são referenciados nos YAMLs de configuração.

| Papel      | Exemplo usado no Summa                        |
| :--------- | :-------------------------------------------- |
| **Aluno**  | `Qwen2.5-7B-Instruct`                         |
| **Professor** | `Qwen3-235B-A22B-2507` (via OpenRouter, ZDR) |
| **Referência** | `GPT-5` (via Azure)                        |

> Nesta etapa não há script a executar — a escolha é feita nos arquivos YAML das etapas seguintes.

---

## 02 — Extração dos dados (prompt de entrada)

Extraia os textos-fonte de uma base de dados (CKAN, CSV, Parquet, etc.) com o script `util_ckan.py`. O YAML define filtros, colunas e formato de saída.

```bash
# Entrar na pasta do experimento
# Extrair dados do CKAN conforme configuração
python ../../src/util_ckan.py --config 01_extracao.yaml

# Apenas visualizar o que seria extraído (dry-run)
python ../../src/util_ckan.py --config 01_extracao.yaml --view
```

---

## 03 — Geração da linha de base (modelo base sem treinamento)

Gere as respostas do modelo base (zero-shot) e/ou do professor sobre os dados extraídos. Isso permite medir a performance antes de qualquer ajuste fino.

```bash
# Gerar saída do modelo professor (Qwen3-235B via OpenRouter)
python ../../src/util_vllm_batch.py --config 02_summa_235b.yaml

# Gerar saída do modelo base local (Qwen2.5-7B via vLLM)
python ../../src/util_vllm_batch.py --config 02_summa_7b_hpc.yaml

# Gerar saída do GPT-5 como referência (via Azure)
python ../../src/util_vllm_batch.py --config 02_summa_gpt5.yaml
```

---

## 04 — Comparar linha de base com dados de referência

Compare as saídas do modelo base com o gabarito do professor (ou dados raw). Essa comparação **gera o arquivo de divisão** (treino/teste/validação) e **classifica a dificuldade** de cada instância — informações que serão usadas no treinamento.

```bash
# Comparar modelo base (Qwen 7B) vs professor (Qwen3-235B)
python ../../src/comparar_extracoes.py --config 03_compara_q235_full.yaml

# Comparar modelo base vs GPT-5 (curadoria)
python ../../src/comparar_extracoes.py --config 03_compara_gpt5_curadoria.yaml
```

> **Saída importante:** o arquivo de divisão (`divisao_*.csv` ou `.parquet`) com as colunas `alvo` (treino/teste/validação) e `dificuldade_int` (1–9), que alimentará as etapas seguintes.
>
> ⚠️ **Atenção Crítica — Copiar para `dados/`:**  
> O script salva as divisões dentro da pasta de saída da comparação:  
> `compara/<pasta_saida_analises>/divisoes/divisao_<modelo>.csv` (ou `.parquet`).  
> **Você DEVE copiar esse arquivo para a pasta `dados/` do experimento antes de prosseguir:**
> ```bash
> # Exemplo no SemClinBr:
> cp "compara/analises_comparacao_semclinbr (full)/divisoes/divisao_Gold_Qwen7B.csv" dados/
> 
> # Exemplo no Summa:
> cp "compara/analises_comparacao_summa_full/divisoes/divisao_Qwen235b_Qwen7b.csv" dados/
> ```
> Todos os passos seguintes (05_extracao, 06_compara, 07_avaliar) procuram esse arquivo exclusivamente em `dados/`. Se ele não estiver lá, os filtros de partição falharão silenciosamente!
> 
> 🛑 **NUNCA apague ou compacte (`zip -r ... && rm -rf ...`) a pasta de análises da etapa 04 antes de garantir que a divisão foi copiada para `dados/`.**

---

## 05 — Preparar protocolos de treinamento

Crie os YAMLs de treinamento para cada protocolo desejado. Inclua pelo menos um protocolo executado **3 vezes** (réplicas) para calibração da ROPE (Region of Practical Equivalence).

| Tipo de protocolo            | Exemplo de YAML              |
| :--------------------------- | :--------------------------- |
| Baseline LoRA 4-bit          | `04_treinar_b.yaml`          |
| Baseline Full FT 16-bit      | `04_treinar_c.yaml`          |
| CL + Escalonamento (D1)      | `04_treinar_d1.yaml`         |
| Réplicas ROPE (D1a, D1b)     | `04_treinar_d1a.yaml`, `04_treinar_d1b.yaml` |

```bash
# Verificar o dataset de um YAML antes de treinar (relatório de divisão)
python ../../src/treinar_unsloth.py --datasets 04_treinar_b.yaml
```

> Nesta etapa o foco é **revisar e ajustar** os YAMLs. O treinamento em si ocorre no passo seguinte.

---

## 06 — Treinar os modelos

Execute o treinamento para cada protocolo. É recomendável usar `tmux` ou `nohup` para sessões longas.

```bash
# Treinar baseline LoRA (protocolo b)
python ../../src/treinar_unsloth.py --treinar 04_treinar_b.yaml

# Treinar baseline Full FT (protocolo c)
python ../../src/treinar_unsloth.py --treinar 04_treinar_c.yaml

# Treinar protocolo CL+PT D1 (e réplicas para ROPE)
python ../../src/treinar_unsloth.py --treinar 04_treinar_d1.yaml
python ../../src/treinar_unsloth.py --treinar 04_treinar_d1a.yaml
python ../../src/treinar_unsloth.py --treinar 04_treinar_d1b.yaml
```

> **Dica:** para rodar em background com `tmux`:
> ```bash
> source setup_tmux.sh
> tm new -s treino
> python ../../src/treinar_unsloth.py --treinar 04_treinar_d1.yaml
> # Ctrl+B, depois D para desanexar
> ```

---

## 07 — Extrair respostas com os modelos treinados (teste)

Após o treinamento, extraia as respostas do conjunto de teste usando os pesos ajustados.

```bash
# Extrair com protocolo b (LoRA)
python ../../src/util_vllm_batch.py --config 05_extracao_b_teste.yaml

# Extrair com protocolo c (Full FT)
python ../../src/util_vllm_batch.py --config 05_extracao_c_teste.yaml

# Extrair com protocolo D1
python ../../src/util_vllm_batch.py --config 05_extracao_d1_teste.yaml
```

> Cada YAML `05_extracao_*_teste.yaml` já aponta para os pesos treinados em `treinos/` e aplica o filtro `alvo: teste`.

---

## 08 — Calibrar a ROPE (comparar réplicas D1/D1a/D1b)

Compare as 3 execuções do mesmo protocolo para estimar a variabilidade natural não-determinística (hardware/framework) e calibrar a ROPE (*Region of Practical Equivalence*) por campo.

A calibração segue obrigatoriamente um **ciclo de verificação em duas rodadas**:

### 8.1 — Primeira rodada: Rodar calibração inicial
Execute a comparação das réplicas com qualquer valor provisório de ROPE (ex.: `rope: 0.01`):

```bash
# Comparar réplicas para estimar o ruído
python ../../src/comparar_extracoes.py --config 06_compara_d1ab.yaml
```

> **Saída gerada:** o arquivo `compara/analises_comparacao_*_d1ab/estatisticas/00_rope_sugerido.md` contendo a tabela de deltas observados e a sugestão analítica de ROPE global e por campo (`rope_por_campo`).

### 8.2 — Transcrever os valores sugeridos no próprio YAML
Abra `00_rope_sugerido.md`, copie o bloco YAML sugerido e cole na seção `estatistica.metricas_automaticas` do **próprio** `06_compara_d1ab.yaml`:

```yaml
metricas_automaticas:
  rope: 0.0210             # valor global sugerido
  rope_por_campo:
    (global): 0.0145
    Entidades: 0.0120
    Relacoes: 0.0210
  campos:
    - "(global)"
    - "Entidades"
    - "Relacoes"
  metricas:
    - rouge_l
    - rouge_2
```

### 8.3 — Segunda rodada: Verificação de Sanidade (Teste de Equivalência)
Rode novamente a análise estatística sobre o mesmo resultado (usando a flag `--estatisticas` para ser rápido e não recalcular as métricas pesadas de extração):

```bash
# Reexecutar apenas a camada estatística com a nova ROPE calculada
python ../../src/comparar_extracoes.py --config 06_compara_d1ab.yaml --estatisticas
```

> 🔍 **Verificação de Sanidade Obrigatória:**  
> Abra o relatório bayesiano (`bayesiana/analise_bayesiana.md`) e os relatórios estatísticos.  
> **A calibração só está confirmada quando 100% dos pares de réplicas (`D1 × D1a`, `D1 × D1b`, `D1a × D1b`) saírem classificados como `equivalentes` ao limiar de 0.95.**  
> Se algum par sair como "inconclusivo" ou indicar diferença real, a calibração falhou e deve ser investigada.

### 8.4 — Transcrever para a comparação geral
Após confirmar a equivalência de 100% das réplicas no passo 8.3, copie esse mesmo bloco `rope` e `rope_por_campo` para os YAMLs de comparação geral entre protocolos distintos (`06_compara_todos.yaml` e `06_compara_todos_parcial.yaml`).

> ⚠️ **Cuidados Críticos na Configuração da Calibração (`06_compara_d1ab.yaml`):**
> 1. **Filtro obrigatório do conjunto de teste (`alvo: teste`):**
>    - O gabarito humano/base geralmente contém todos os documentos do corpus (ex: 1.000 documentos), mas os modelos foram executados na etapa 07 **apenas sobre a partição de teste** (ex: 197 documentos).
>    - O bloco `configuracao_comparacao.filtro` **DEVE** apontar para `dados/divisao_*.csv` com `dataset_filtro: {"alvo": "teste"}`.
>    - *Se o arquivo de divisão não existir em `dados/` ou o filtro for omitido:* o comparador tentará avaliar todos os 1.000 documentos; para os 800 de treino/validação, o modelo não terá arquivo de saída, ficando marcado como `Inexistente` (com campo `pred` **vazio** e escore `0.0`), o que derruba drasticamente a média do modelo e distorce as estatísticas.
> 2. **NUNCA usar `"TODOS"` em `estatistica.protocolos`:**
>    - Em arquivos com `calibracao_rope: true`, declare explicitamente apenas as réplicas a comparar:
>      ```yaml
>      estatistica:
>        calibracao_rope: true
>        protocolos:
>          Replicas_D1:
>            - D1
>            - D1a
>            - D1b
>      ```
>    - *Por que não usar `"TODOS"`?* O valor `"TODOS"` instrui o pipeline a incluir o protocolo virtual `★Prof` (gabarito perfeito = 1.0). Como a calibração busca a maior discrepância ($|Δ|$) para garantir equivalência de 100% dos pares, se `★Prof` estiver no recorte, o script medirá a distância do aluno ao teto do professor em vez do ruído não-determinístico entre réplicas idênticas, gerando uma ROPE inflada próxima de `0.95`!

---

## 09 — Comparar todos os protocolos

Rode a comparação geral com todos os modelos treinados. Inclui métricas automáticas (BERTScore, SBERT, ROUGE-L/2, Levenshtein) e análise estatística bayesiana + frequentista.

```bash
# Comparação completa (todos os protocolos treinados)
python ../../src/comparar_extracoes.py --config 06_compara_todos.yaml

# Comparação parcial (apenas protocolos já prontos)
python ../../src/comparar_extracoes.py --config 06_compara_todos_parcial.yaml
```

> **Saídas:** pasta `compara/` com planilhas, gráficos, heatmaps bayesianos e relatórios `.md` por recorte (Q1–Q6).

---

## 10 — LLM-as-a-judge (opcional)

Use um modelo de referência (ex: GPT-5) como juiz para avaliar qualitativamente as extrações em escala Likert (1–4), com 3 rodadas para estabilidade.

```bash
# 1. Gerar o parquet de entrada do juiz (70 docs × 3 modelos = 210 linhas)
cd avaliacao_llm_humana/
python 01_gerar_entrada_juiz_llm.py

# 2. Rodar o juiz LLM (GPT-5 via Azure, 3 rodadas)
python ../../../src/util_vllm_batch.py --config 02_extracao_70.yaml

# 3. Consolidar e analisar as avaliações
python ../../../src/realizar_avaliacoes.py \
    --grupos saida_gpt:humano saida_nemo:llm saida_glm:llm saida_mmm3:llm saida_sabia:llm \
    --bayes
```

---

## 11 — Avaliação humana (opcional)

Configure avaliadores humanos via Label Studio para avaliação cega dos resultados. Requer aprovação do Comitê de Ética de Pesquisa (CEP).

```bash
# 1. Gerar as tarefas no formato Label Studio
cd avaliacao_llm_humana/
python gerar_tarefas_label_studio.py

# 2. Importar os JSONs gerados no Label Studio
#    Os arquivos ficam na pasta avaliacao_humana/:
#    - tarefas_avaliacao_label_studio_<Avaliador>.json

# 3. Após coleta, realizar análise das avaliações humanas
python ../../../src/realizar_avaliacoes.py \
    --grupos saida_gpt:humano saida_nemo:llm saida_glm:llm \
    --bayes
```

> ⚠️ **Atenção:** a utilização de avaliadores humanos requer submissão e aprovação pelo Comitê de Ética de Pesquisa (CEP) da instituição.

---

## 12 — Análise dos resultados e criação de figuras

Gere os resumos consolidados e as figuras finais para a dissertação/artigo.

```bash
# Gerar tabela-resumo de todos os experimentos (compara parâmetros entre Summa/PubMed/SemClinBr)
cd /caminho/para/llms/experimentos/
python tabela_experimentos.py

# Validar o ambiente de execução
python ../src2/teste_ambiente.py
```

> As figuras e gráficos são gerados automaticamente durante o passo 09 (comparação) e ficam nas pastas `compara/analises_comparacao_*/graficos/`.

---

## Resumo visual do pipeline

```
01 Escolha do modelo ──► 02 Extração dos dados ──► 03 Linha de base (zero-shot)
                                                          │
04 Comparar base vs referência ◄──────────────────────────┘
         │
         ▼ (gera divisão + dificuldade)
05 Preparar YAMLs de treinamento ──► 06 Treinar modelos
                                           │
07 Extrair com modelos treinados ◄─────────┘
         │
08 Calibrar ROPE ──► 09 Comparar todos os protocolos
                            │
                    ┌───────┴───────┐
                    ▼               ▼
          10 LLM-as-a-judge   11 Avaliação humana
                    │               │
                    └───────┬───────┘
                            ▼
                 12 Análise e figuras
```