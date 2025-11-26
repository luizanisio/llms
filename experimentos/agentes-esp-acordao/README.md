## Orquestração de Agentes-LLM para extração de metadados na geração do Espelho do Acórdão

Este projeto implementa e compara abordagens para a extração de metadados estruturados (Espelhos de Acórdãos) a partir de textos jurídicos do STJ. O experimento contrasta uma abordagem tradicional de prompt único ("Base") com uma arquitetura de múltiplos agentes especializados ("Orquestração"), avaliando os resultados através de métricas clássicas e LLM-as-a-Judge.

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
  - Este script instancia a classe `AgenteOrquestradorEspelho` (de `agentes_orquestrador.py`), que gerencia o fluxo:
    1. **AgenteCampos**: Identifica quais campos existem no acórdão.
    2. **AgenteTeses**: Extrai as teses jurídicas (dependência primária).
    3. **AgenteJurisprudenciasCitadas**: Extrai precedentes baseados nas teses.
    4. **Execução Paralela**: Agentes de Notas, ICE, TAP, Tema e Referências Legislativas rodam simultaneamente.
    5. **AgenteValidacaoFinal**: Consolida e revisa o JSON final.
  - Os prompts específicos para cada agente estão em `prompt_espelho_agentes.py`.
  - Os resultados são salvos em pastas específicas por modelo (ex: `saidas/espelhos_agentes_gpt5/`).

### Avaliação LLM-as-a-judge

Utiliza um modelo avançado (GPT-5) para avaliar a qualidade semântica das extrações.

- **Como organizar os arquivos e rodar a avaliação**:
  - Execute `avaliacao_llm_as_a_judge.py`.
  - O script percorre as pastas de saída (Base e Agentes) e compara cada extração com o texto original do acórdão.
  - Calcula métricas de **Precision**, **Recall** e **F1-Score** baseadas na interpretação do LLM Juiz.
  - Gera arquivos `.avaliacao.json` junto aos arquivos extraídos.

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