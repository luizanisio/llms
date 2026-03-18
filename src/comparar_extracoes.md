# Comparação de Extrações de Documentos (Genérico via YAML)

> Documentação técnica para desenvolvedores e agentes de IA (Copilot).

## 1. Visão Geral e Propósito
O script `comparar_extracoes.py` é uma **ferramenta genérica e agnóstica** para comparar extrações de dados estruturados (JSON) provenientes de diferentes fontes (Modelos LLM, OCR, Parsers).

O objetivo é validar a qualidade de extração de "Modelos Candidatos" (Agentes/LLMs) contra um "Modelo Base" (Gabarito/Ground Truth/Referência), utilizando métricas textuais (ROUGE, BERTScore, Levenshtein, SBERT) adaptadas semanticamente para cada tipo de campo.

## 2. Arquitetura e Fluxo de Execução

O fluxo é controlado inteiramente por um arquivo de configuração YAML, eliminando a necessidade de alteração de código para novos experimentos.

### Fluxo de Dados:
1.  **Carga de Configuração**: O script lê o YAML (argumento posicional ou menu interativo) para definir caminhos, rótulos e regras.
2.  **Carga de Dados (`CargaDadosComparacao`)**:
    *   Lê os JSONs da pasta `modelo_base`.
    *   Lê os JSONs das pastas `modelos_comparacao`.
    *   Alinha os documentos pelo ID (padrão regex no nome do arquivo).
    *   Modelos marcados com `ativo: false` no YAML são ignorados completamente.
3.  **Configuração de Métricas**:
    *   Define quais métricas (BERTScore, ROUGE-L, ROUGE-1, ROUGE-2, Levenshtein, SBERT) se aplicam a quais campos.
    *   Em modo `teste_rapido`, BERTScore e SBERT são desativados e seus campos migram para ROUGE-L.
4.  **Processamento (`JsonAnaliseDataFrame`)**:
    *   Calcula métricas par-a-par (Base vs Modelo X) com suporte a paralelismo (`max_workers`).
    *   Gera vetores de similaridade e salva JSONs de análise individuais na subpasta `jsons/`.
5.  **Pós-Processamento e Saída**:
    *   **Excel**: Matriz colorida (heatmap) com os scores por documento e modelo.
    *   **CSV**: Dados brutos para análise externa.
    *   **Estatísticas**: Agregação (Mean/Std/Median) por modelo e métrica.
    *   **LLM-as-a-Judge (Opcional)**: Usa um LLM para decidir qual extração é melhor em casos complexos.
    *   **Gráficos**: Gera visualizações comparativas (barras agrupadas, status de erros).
6.  **Divisão Treino/Teste/Validação (`UtilJsonDivisoes`)**:
    *   Classifica cada documento em `fácil`, `médio` ou `difícil` com base nas métricas F1 globais e na **complexidade estrutural do ground truth** (número de chaves JSON).
    *   Gera arquivos `divisao_<modelo>.csv` e `.xlsx` na subpasta `divisoes/`.

## 3. Configuração (Arquivo YAML)

A estrutura do YAML é dividida em 4 seções principais:

### 3.1 `saida`
Controla onde os artefatos serão salvos e se deve reprocessar dados existentes.
```yaml
saida:
  pasta: "./saida_experimento_X"
  arquivo_base: "comparacao_resultados"  # Prefixo dos arquivos Excel e CSV gerados
  regerar_planilha_base: false  # Se true, força o recálculo mesmo que o Excel já exista
  linguagem_graficos: "pt"      # "pt" ou "en" (afeta rótulos dos gráficos)
```

> **Nota sobre caminhos**: Todos os caminhos definidos no YAML são resolvidos **relativamente à localização do próprio arquivo YAML**.
>
> **Organização de Arquivos**: O script estrutura a saída da seguinte forma:
> *   **Raiz (`saida.pasta`)**: Excel, CSV consolidado, gráficos de status e relatório de análise estatística.
> *   **Subpasta `jsons/`**: JSON de análise por documento, Relatórios Markdown e outros artefatos granulares.
> *   **Subpasta `graficos/`**: Imagens PNG dos gráficos comparativos.
> *   **Subpasta `divisoes/`**: CSV e Excel de divisão treino/teste/validação por modelo.

### 3.2 `execucao`
Flags de controle de fluxo.
```yaml
execucao:
  max_workers: 10
  teste_rapido: false        # Se true, desativa BERTScore e SBERT (usa apenas ROUGE/Levenshtein)
  gerar_graficos: true       # Gera/atualiza gráficos a partir do Excel existente
  llm_as_a_judge: false      # Executa avaliação qualitativa via LLM
  analise_estatistica: false # Executa testes de hipótese/comparação estatística profunda
  ignorar_erro_extracao: false # Se true, documentos com erro de extração são excluídos das métricas
  divisao:                   # Proporções para divisão treino/teste/validação (opcional)
    treino: 0.7
    teste: 0.2
    validacao: 0.1
```

### 3.3 Definição de Modelos
Define quem é a verdade (Base) e quem são os desafiantes.
```yaml
modelo_base:
  pasta: "./gabarito"
  rotulo: "Humano"
  familia: "Referência"  # Opcional, usado em estatísticas

modelos_comparacao:
  - pasta: "./modelo_A"
    rotulo: "GPT-4"
    familia: "OpenAI"  # Usado para agrupamento em gráficos/stats
    ativo: true        # Opcional (padrão true). Se false, ignora este modelo completamente.
```

> **Restrição de rótulos**: O rótulo do `modelo_base` deve ser único e diferente de todos os rótulos em `modelos_comparacao`. O script valida isso na carga e lança erro se houver duplicata.

### 3.4 `configuracao_comparacao` (Mapa de Métricas)
Mapeia campos do JSON para algoritmos de similaridade específicos. Campos especiais `(global)` e `(estrutura)` são calculados automaticamente.

```yaml
configuracao_comparacao:
  nivel_campos: 1          # 1 = apenas campos de nível raiz; 2 = inclui subníveis
  padronizar_simbolos: true
  rouge_stemmer: true
  nome_campo_id: "id"      # Nome do campo identificador nos JSONs
  campos:
    bertscore:   ["(global)", "textoLongo"]   # Textos longos/argumentativos/paráfrases
    rouge_l:     ["(global)", "listaOrdenada"] # Listas ou textos onde a ordem importa
    rouge_2:     ["termoCurto"]               # Textos curtos, termos técnicos, precisão n-gramas
    rouge_1:     ["(estrutura)"]              # Análise estrutural (nomes de campos presentes)
    levenshtein: ["codigo", "data"]           # Identificadores, datas, valores exatos
    sbert:       ["textoSemantico"]           # Similaridade semântica via Sentence-BERT (modelo padrão)
    sbert_pequeno: []                         # SBERT com modelo leve (mais rápido)
    sbert_medio:   []                         # SBERT com modelo intermediário
    sbert_grande:  []                         # SBERT com modelo grande (mais preciso)
  mascaras:                                   # Padrões regex para identificar arquivos por tipo
    extracao:      "^(\\d{12})\\.\\d+\\.\\d*\\.json$"
    tokens:        ".json"
    avaliacao:     ".avaliacao.json"
    observabilidade: ".obs.json"
```

> **Campos com subníveis**: Se o YAML definir `Temas.Ponto`, o script agrega todo o conteúdo da subchave `Ponto` — seja `Temas` uma lista de objetos ou um dicionário.

## 4. Divisão Treino/Teste/Validação e Classificação de Complexidade

A etapa de divisão (`UtilJsonDivisoes`) usa curriculum learning para classificar cada documento em três níveis de dificuldade e distribuí-los proporcionalmente entre os conjuntos de dados.

### Critério de Classificação de Dificuldade

A dificuldade é calculada por uma **pontuação composta** que combina as métricas F1 com a complexidade estrutural do ground truth:

```
soma_pontuacoes = Σ F1_globais - chaves_peso
```

*   **F1 globais**: Soma das colunas `(global)_*_F1` e `(estrutura)_*_F1` do JSON de análise. Scores mais baixos → documento mais difícil para o modelo.
*   **`chaves_peso`**: Peso normalizado (0–1) do número total de chaves (e subchaves recursivas) do JSON de ground truth. Documentos com mais chaves recebem peso maior, reduzindo a pontuação composta → classificados como mais difíceis.

### Colunas da Planilha de Divisão

| Coluna | Descrição |
|---|---|
| `id` | Identificador do documento |
| `nome_modelo` | Rótulo do modelo avaliado |
| `familia_modelo` | Família do modelo (quando disponível) |
| `chaves` | Número total de chaves (e subchaves) do JSON de ground truth |
| `chaves_peso` | Peso normalizado entre 0 e 1 (heatmap invertido: mais chaves = vermelho) |
| `(global)_*_F1` | Métricas F1 globais (heatmap padrão: valores baixos = vermelho) |
| `(estrutura)_*_F1` | Métricas F1 estruturais |
| `dificuldade` | `facil`, `medio` ou `dificil` (30%/40%/30%) |
| `dificuldade_int` | Valor de 1 (mais fácil) a 10 (mais difícil) |
| `alvo` | `treino`, `teste` ou `validacao` |

### Distribuição
*   **30%** dos documentos com menor pontuação → `dificil` (notas 8–10)
*   **40%** intermediários → `medio` (notas 4–7)
*   **30%** com maior pontuação → `facil` (notas 1–3)

Cada nível é então dividido independentemente nas proporções configuradas em `execucao.divisao` (padrão 70%/20%/10%), com semente fixa (`seed=42`) para reprodutibilidade.

## 5. Status de Implementação e Melhorias Recentes

### Carga dos dados
- [x] Agrupar campos com subníveis (`Temas.Ponto` agrega todas as ocorrências de `Ponto` dentro de `Temas`).

### Métricas suportadas
- [x] **BERTScore** — similaridade semântica a nível de tokens
- [x] **ROUGE-L / ROUGE-2 / ROUGE-1** — sobreposição de n-gramas
- [x] **Levenshtein** — distância de edição (coluna `_SIM`)
- [x] **SBERT** (pequeno, médio, grande) — similaridade semântica via Sentence-BERT

### Visualizações (Gráficos)
- [x] **Gráfico Global F1**: Comparativo de barras agrupadas por modelo e técnica (BERTScore, ROUGE, etc.).
- [x] **Gráfico por Campo**: Comparativo detalhado de desempenho (F1) para cada campo extraído, agrupado por métrica.
- [x] **Gráfico de Status**: Barras empilhadas com proporção de Sucesso/Erro/Inexistente por modelo.

### Correções de Métricas
- [x] **Pontuação de Erros de Extração**: Documentos com falha na extração (JSON inválido ou erro de parse) são contabilizados com score **0.0** em todas as métricas, garantindo que a taxa de falha impacte a média final.
- [x] **Gráfico Consolidado Levenshtein**: Gera gráficos comparativos para a métrica Levenshtein (coluna `_SIM`).

### Divisão e Complexidade
- [x] **Coluna `chaves`**: Número total de chaves (recursivo) do JSON de ground truth por documento.
- [x] **Coluna `chaves_peso`**: Peso normalizado (0–1) incorporado ao critério de dificuldade; documentos com mais chaves são classificados como mais difíceis.
- [x] **Heatmap invertido em `chaves_peso`**: Vermelho = muitas chaves (alta complexidade), Verde = poucas chaves.
