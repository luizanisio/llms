# -*- coding: utf-8 -*-
"""
Exemplo de YAML para o comparar_extracoes.py com explicações detalhadas.

Este arquivo contém exemplos comentados das configurações possíveis,
incluindo o uso de .parquet como entrada e o formato legado com pastas de JSONs.

Uso:
    from comparar_extracoes_exemplo import YAML_EXEMPLO_COMPARACAO
"""

# =============================================================================
# Exemplo completo com entrada via .parquet
# =============================================================================
YAML_EXEMPLO_PARQUET = '''
# Configuração para comparação de extrações usando arquivos .parquet como entrada.
# Executar com: python comparar_extracoes.py config_exemplo.yaml

# ---- Configuração Global de Saída ----
saida:
  pasta: "./compara/analises_comparacao"        # Pasta raiz para relatórios, gráficos e planilhas
  pasta_parquet: "./compara/"                    # (Obrigatório com .parquet) Pasta base para extração dos JSONs
                                                 # Os JSONs serão extraídos em: <pasta_parquet>/<nome_parquet_sem_ext>/
  arquivo_base: "comparacao_resultados"          # Nome base dos arquivos de saída (.xlsx, .csv)
  regerar_planilha_base: true                    # Se false, não regera planilha se já existir
  linguagem_graficos: "en"                       # "pt" ou "en" para legendas dos gráficos

# ---- Configurações de Execução ----
execucao:
  max_workers: 10                                # Número de threads para processamento paralelo
  teste_rapido: false                            # true = pula BERTScore/SBERT (move campos para ROUGE-L)
  gerar_graficos: true                           # Gera gráficos comparativos
  llm_as_a_judge: false                          # Executa avaliação LLM-as-a-Judge (requer API)
  analise_estatistica: true                      # Gera relatório estatístico
  ignorar_erro_extracao: true                    # true = ignora documentos com erro nos cálculos de métricas
  divisao:
    treino: 0.70                                 # Proporção para treino
    validacao: 0.10                              # Proporção para validação  
    teste: 0.20                                  # Proporção para teste

# ---- Modelo Base (Gabarito/Referência) ----
# Use 'arquivo:' para .parquet ou 'pasta:' para diretório de JSONs
modelo_base:
  arquivo: "./saida/saida_modelo_base.parquet"   # Caminho do arquivo .parquet
  rotulo: "ModeloBase"                           # Rótulo para identificação nos relatórios
  familia: "Modelo-Base"                         # Nome da família do modelo

# ---- Modelos Comparados ----
modelos_comparacao:
  - arquivo: "./saida/saida_modelo_a.parquet"    # Entrada via .parquet
    rotulo: "ModeloA"
    familia: "Modelo-A"

  - arquivo: "./saida/saida_modelo_b.parquet"
    rotulo: "ModeloB"
    familia: "Modelo-B"
    # ativo: false                               # Descomente para desabilitar este modelo

# ---- Configuração de Métricas e Campos ----
configuracao_comparacao:
  padronizar_simbolos: true                      # Normaliza símbolos antes de comparar
  rouge_stemmer: true                            # Usa stemmer no cálculo ROUGE

  # Campo de ID
  nome_campo_id: "id"                            # Nome interno do campo de ID
  rotulo_campo_id: "ID"                          # Rótulo de exibição

  # Mapeamento dos campos do .parquet
  # Define quais colunas do parquet correspondem aos dados esperados pelo pipeline.
  campos_parquet:
    id: "chave"                                  # (Obrigatório) Coluna com o ID do documento
    resposta: "resposta"                         # (Obrigatório) Coluna com o JSON da extração → {id}.json
    resumo_tokens: "resumo"                      # (Opcional) Coluna com JSON de tokens → {id}.tokens.json
    avaliacao: ""                                # (Opcional) Coluna com avaliação LLM → {id}.avaliacao.json
    erro: "erro"                                 # (Opcional) Coluna com mensagem de erro

  # Máscaras de arquivos (usadas para identificar os JSONs extraídos na pasta)
  mascaras:
    extracao: ".json"                            # Padrão dos arquivos de extração
    tokens: ".tokens.json"                       # Padrão dos arquivos de tokens
    avaliacao: ".avaliacao.json"                 # Padrão dos arquivos de avaliação
    observabilidade: ".obs.json"                 # Padrão dos arquivos de observabilidade

  # Mapeamento de campos para métricas
  # (Opcional) Sobrescrita de modelos HF para SBERT e BERTScore
  modelos:
    sbert:
      grande: "intfloat/multilingual-e5-base"           # Ex: SBERT grande personalizado
      # pequeno: "outro/modelo"                        # Omitidos usam o padrão
      medio: grande: "stjiris/bert-large-portuguese-cased-legal-mlm-mkd-nli-sts-v1" # Jurídico
    bertscore: grande: "stjiris/bert-large-portuguese-cased-legal-mlm-mkd-nli-sts-v1"    # Ex: BERTScore personalizado modelo STJ de Portugual

    # Tamanhos de mini-batch para otimização de memória no pré-cálculo (Padrão: 1024)
    bertscore_batch_size: 1024
    sbert_batch_size: 1024

  campos:
    bertscore:
      - "(global)"
      - "Resumo"
      - "Dispositivo"

    rouge_l:
      - "(global)"
      - "Materia"
      - "Resumo"

    levenshtein:
      - "Tipo"
      - "DataJulgamento"
'''

# =============================================================================
# Exemplo com entrada via pasta de JSONs (formato legado)
# =============================================================================
YAML_EXEMPLO_PASTA = '''
# Configuração usando diretórios de JSONs soltos (formato legado).
# Neste modo, NÃO é necessário definir 'pasta_parquet' nem 'campos_parquet'.

saida:
  pasta: "./compara/analises_comparacao"
  arquivo_base: "comparacao_resultados"
  regerar_planilha_base: true
  linguagem_graficos: "pt"

execucao:
  max_workers: 10
  teste_rapido: false
  gerar_graficos: true
  ignorar_erro_extracao: true
  divisao:
    treino: 0.70
    validacao: 0.10
    teste: 0.20

# Use 'pasta:' quando os JSONs já estão extraídos em diretórios
modelo_base:
  pasta: "./extraidos/modelo_base/"              # Diretório com {id}.json, {id}.tokens.json, etc.
  rotulo: "ModeloBase"
  familia: "Modelo-Base"

modelos_comparacao:
  - pasta: "./extraidos/modelo_a/"
    rotulo: "ModeloA"
    familia: "Modelo-A"

  - pasta: "./extraidos/modelo_b/"
    rotulo: "ModeloB"
    familia: "Modelo-B"

configuracao_comparacao:
  padronizar_simbolos: true
  rouge_stemmer: true

  nome_campo_id: "id"
  rotulo_campo_id: "ID"

  # Máscaras para identificar arquivos nos diretórios
  mascaras:
    extracao: "^(.+)\\\\.json$"                    # Regex: captura o ID do nome do arquivo
    tokens: ".tokens.json"
    avaliacao: ".avaliacao.json"
    observabilidade: ".obs.json"

  # (Opcional) Sobrescrita de modelos HF para SBERT e BERTScore
  modelos:
    sbert:
      grande: "intfloat/multilingual-e5-base"
    bertscore: "microsoft/deberta-xlarge-mnli"
    
    # Tamanhos de mini-batch para otimização de memória no pré-cálculo (Padrão: 1024)
    bertscore_batch_size: 1024
    sbert_batch_size: 1024

  campos:
    bertscore:
      - "(global)"
      - "Resumo"

    rouge_l:
      - "(global)"
      - "Materia"

    levenshtein:
      - "Tipo"
'''

# Exemplo padrão (usa parquet)
YAML_EXEMPLO_COMPARACAO = YAML_EXEMPLO_PARQUET