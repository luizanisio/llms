## Orquestração de Agentes-LLM para extração de metadados na geração do Espelho do Acórdão

Este projeto implementa e compara abordagens para a extração de metadados estruturados (Espelhos de Acórdãos) a partir de textos jurídicos do STJ. O experimento contrasta uma abordagem tradicional de prompt único ("Base") com uma arquitetura de múltiplos agentes especializados ("Orquestração"), avaliando os resultados através de métricas clássicas e LLM-as-a-Judge.

**📊 [Ver todos os diagramas de fluxo e arquitetura](README_MERMAID.md)**

### Como extrair os dados da origem "Dados abertos" do STJ com o "ckan"

O script principal para esta etapa é `ckan_extrair_espelhos.py`.

- **Configurações essenciais**:
  - A lista `DATASET_IDs` define os conjuntos de dados (turmas e seções) a serem baixados do portal de Dados Abertos do STJ.
  - O script consolida os dados baixados e os textos dos acórdãos em um arquivo Parquet (`espelhos_acordaos_consolidado_textos.parquet`) e CSVs auxiliares.

- **Saída esperada**:
  - Arquivos JSON originais na pasta `downloads_esp_stj`.
  - Arquivos consolidados para uso nos scripts de geração.
  - O script `converter_dados_abertos.py` deve ser executado posteriormente para converter os dados originais para o formato JSON padronizado (`espelhos_raw`), servindo como *Ground Truth* para as comparações.

### Geração do espelho com prompt base

Esta abordagem utiliza um único prompt complexo para extrair todos os campos de uma vez.

- **Como organizar os dados**:
  - Certifique-se de que o arquivo `espelhos_acordaos_consolidado_textos.parquet` foi gerado na etapa anterior.
  - Configure as variáveis de ambiente (chaves de API) no arquivo `.env`.

- **Como rodar a geração**:
  - Execute o script `gerar_espelho_sjr_base.py`.
  - Ele utiliza o prompt definido em `prompt_espelho_base.py` (`PROMPT_BASE_SJR_S3_JSON`).
  - Os resultados são salvos na pasta `saidas/espelhos_base/`.

### Geração do espelho com a orquestração de agentes

Esta abordagem divide a tarefa entre vários agentes especializados coordenados por um orquestrador.

- **Como organizar os dados**:
  - Utiliza a mesma base de dados consolidada em Parquet.

- **Como rodar a orquestração**:
  - Execute o script `agentes_gerar_espelhos.py`.
  - Este script instancia a classe `AgenteOrquestradorEspelho` (de `agentes_orquestrador.py`).
  - Os prompts específicos para cada agente estão em `prompt_espelho_agentes.py`.
  - Os resultados são salvos em pastas específicas por modelo (ex: `saidas/espelhos_agentes_gpt5/`).
  - **📊 [Ver diagrama detalhado do fluxo de orquestração](README_MERMAID.md#2-fluxo-de-orquestração-completo-sistema-de-agentes)**

**Pipeline de Execução:**

1. **ETAPA 1**: `AgenteCampos` - Identifica quais campos existem no acórdão
2. **ETAPA 1.5**: Revisão do `AgenteCampos` - Se não identificou campos, solicita revisão com instrução específica para conferir atentamente
3. **ETAPA 2**: `AgenteTeses` - Extrai as teses jurídicas (dependência primária)
4. **ETAPA 2.5**: `AgenteJurisprudenciasCitadas` - Extrai precedentes baseados nas teses extraídas
5. **ETAPA 3**: Execução Paralela - `AgenteNotas`, `AgenteInformacoesComplementares`, `AgenteTermosAuxiliares`, `AgenteTema` e `AgenteReferenciasLegislativas` rodam simultaneamente
6. **ETAPA 4**: `AgenteValidacaoFinal` - Consolida e valida todas as extrações
7. **ETAPA 5**: Loop de Revisão - Processa até 2 ciclos de revisões conforme necessário, reexecutando agentes com erros ou que precisam de ajustes
8. **Consolidação Final**: Monta o espelho final com todos os campos extraídos e metadados
9. **Verificação de Erros**: Apenas grava arquivos se não houver erros remanescentes, permitindo novas tentativas em caso de falha

### Avaliação LLM-as-a-judge

Utiliza um modelo avançado (GPT-5) para avaliar a qualidade semântica das extrações.

- **Como organizar os arquivos e rodar a avaliação**:
  - Execute `avaliacao_llm_as_a_judge.py`.
  - O script percorre as pastas de saída (Base e Agentes) e compara cada extração com o texto original do acórdão.
  - Calcula métricas de **Precision**, **Recall** e **F1-Score** baseadas na interpretação do LLM Juiz.
  - Gera arquivos `.avaliacao.json` junto aos arquivos extraídos.
  - **📊 [Ver diagrama do fluxo de avaliação](README_MERMAID.md#4-avaliação-llm-as-a-judge)**

### Geração de planilha de comparações

Realiza uma comparação técnica entre as extrações geradas e o *Ground Truth* (Dados Abertos).

- **Como organizar os arquivos**:
  - As pastas de saída das gerações (`espelhos_base_*`, `espelhos_agentes_*`) e a pasta de referência (`espelhos_raw`) devem estar populadas.

- **Como rodar a avaliação**:
  - Execute `comparar_extracoes.py`.
  - O script utiliza a classe `JsonAnaliseDataFrame` para aplicar métricas específicas para cada tipo de campo:
    - **BERTScore**: Para campos textuais longos e semânticos (ex: Teses).
    - **ROUGE-L/2**: Para sequências e frases (ex: Jurisprudência).
    - **Levenshtein**: Para campos exatos.

- **Dados que a planilha consolida**:
  - Gera relatórios comparativos que permitem visualizar a performance de cada modelo e abordagem (Base vs. Agentes) em relação aos dados oficiais.
  - **📊 [Ver diagramas de métricas e comparação](README_MERMAID.md#5-comparação-de-extrações-métricas-de-similaridade)**