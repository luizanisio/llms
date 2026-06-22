# Comparação de Extrações de Documentos

## 🎯 Objetivo do Módulo (Visão Geral)
O script `comparar_extracoes.py` é uma **ferramenta genérica e agnóstica** projetada para automatizar o desafio de avaliar se um modelo LLM previu os dados que deveriam ter sido previstos. Ele avalia e compara extrações brutas de textos não-estruturados para JSON estruturados produzidas por diversos "Modelos Candidatos" em oposição a um "Modelo Base" de Referência (Ground Truth Humano), calculando um vasto array de métricas e consolidando os documentos em classes de dificuldade que darão insumos para gerar um Dataset de fine-tuning Curriculum Learning.

## 🚀 Funcionalidades Principais
- **Algoritmos de Ponta**: Suporta métricas clássicas de n-gram (ROUGE-1, ROUGE-2, ROUGE-L), similaridade semântica a nível de palavra baseada no transformer (BERTScore), similaridade semântica global de vetores sintéticos (Sentence-BERT), e distância exata via Levenshtein.
- **Aplicação Mapeada**: O YAML permite que você especifique exatamente em quais campos aplicará essas regras. Textos curtos e precisos vão com Levenshtein/ROUGE-2. Textos de resumos discursivos vão com BERTscore/SBERT. A agregação hierárquica lida muito bem com listas ordenadas.
- **Painéis Visuais Categóricos**: Analisa, gera subplanilhas de relatórios individuais por documento, exibe gráficos consolidando "Vitórias e Empates F1 Global F1 Estrutural" em agrupamentos visualmente ricos para pesquisadores avaliarem arquiteturas promissoras.
- **Divisão Baseada na Dificuldade**: Usando as pontuações compostas de taxa de F1 versus o volume descritivo original total detectado na Referência, a ferramenta gera matrizes e subdivide a população de documentos processados nas gavetas de "fácil/médio/difícil", distribuindo dados em proporções corretas de valid/teste/treino sob forte previsibilidade estratificada controlada pela SEED contínua.

## 🛠️ Como Executar (Início Rápido)
O script é agnóstico a comandos secundários e funciona via cardápio de entrada baseado nos seus arquivos YAML de experimento:

```bash
# Para rodar a análise completa (extração, comparação e relatórios) com um yaml específico:
python src/comparar_extracoes.py --config ./experimento_revisoes/meu_config_comparacao.yaml

# Para utilizar o console guiado (sem parâmetros informados):
python src/comparar_extracoes.py

# Para REGERAR apenas os gráficos, estatísticas ou a planilha base (sem refazer a análise pesada via NLP):
# (Exige que a análise principal já tenha sido processada previamente)
python src/comparar_extracoes.py --config ./experimento_revisoes/meu_config_comparacao.yaml --graficos --estatisticas --planilha
```

## ⚙️ Configuração (O Arquivo YAML)
Todas as manobras e experimentos de comparação prescindem da necessidade de entrar no código. Você declara suas intenções dentro do arquivo `.yaml` de configuração, com as seguintes chaves de destaque:

- **`misc`**: Configurações diversas do projeto.
  - `pastas_base`: (Opcional) Permite definir uma lista de diretórios base (absolutos ou relativos) que atua como prefixo para resolver todos os caminhos dinamicamente. O sistema tentará validar as pastas na ordem informada. Exemplo:
    ```yaml
    misc:
      pastas_base: 
        - /students/luiz.abatitucci/llms/experimentos/pubmed-experimento
        - /mnt/d/wsl_dev/llms/experimentos/pubmed-experimento
    ```
- **`saida`**: (Dicionário) Pasta raiz `.pasta` das avaliações e relatórios a serem gerados, além dos arquivos contendo as subdivisões no futuro de seu treinamento iterativo. Inclui também `pasta_parquet` (obrigatória quando se usa entrada `.parquet`) para definir a pasta base onde os JSONs extraídos serão armazenados.
- **`modelo_base`**: Configurações do modelo de referência (Ground Truth). Pode usar `pasta` (diretório com JSONs soltos) ou `arquivo` (caminho para um `.parquet`).
- **`modelos_comparacao`**: (Lista) Suas diversas frentes de modelos prevendo outputs do texto. Cada modelo pode usar `pasta` ou `arquivo` (.parquet). Pode usar `ativo: false` para desabilitar algum.
- **`execucao.divisao`**: Define as frações destinadas à criação de splits (ex: `{treino: 0.7, teste: 0.2, validacao: 0.1}`). As proporções formam os CSVs em `/divisoes/`.
- **`configuracao_comparacao.campos_parquet`**: Mapeamento das colunas de tabelas quando o modelo ou base são `.parquet` ou `.csv`. É aqui que fica o mapeamento original (`resposta`, `id`) e regras do dataset bruto:
  - `saida_json`: (Opcional, Padrão `true`). Se definido como `false`, o pipeline assumirá que o conteúdo extraído da coluna (texto puro) deve ser encapsulado no formato `{"resposta": "texto puro"}` ao gerar os `.json` da base. Dessa forma, você lida corretamente com saídas de LLM que não são estruturadas nativamente.
- **`configuracao_comparacao.campos`**: A engrenagem primordial. Dicionário declarativo dizendo em quais chaves folha do JSON de predição você irá aplicar `bertscore`, `rouge_1`, `levenshtein`, etc. Existe também `(global)` e `(estrutura)` que rodam implicitamente rastreando chaves e nós primários, de forma que eles criam seu escopo de avaliação macro para você sempre entender "Qual o F1 em relação a se acertaram ou não ao menos trazer a estrutura do campo".
  - **Comportamento de campos vazios:** Se nenhum campo for mapeado para uma métrica, o comparador **não** trará todos os campos automaticamente. Ao invés disso, ele avaliará estritamente os campos solicitados ou as métricas virtuais `(global)`/`(estrutura)` se configuradas para a técnica, gerando comparações puramente globais.
- **`configuracao_comparacao.modelos`**: (Opcional) Permite sobrescrever os modelos SBERT e BERTScore padrão por modelos HuggingFace personalizados. Exemplo:
  ```yaml
  modelos:
    sbert:
      grande: "intfloat/multilingual-e5-base"  # override individual do SBERT grande
      grande_alias: "E5-Base"                  # apelido que será exibido nos relatórios e gráficos
      pequeno: "stjiris/bert-large-portuguese-cased-legal-mlm-mkd-nli-sts-v1"
      pequeno_alias: "STJIris"                 # se omitido, o alias usa o final do path ('...-sts-v1')
    bertscore: "microsoft/deberta-xlarge-mnli" # modelo personalizado para BERTScore
    bertscore_alias: "DeBERTa"
  ```
  Se omitido, o comparador usa os modelos e aliases padrão: SBERT pequeno (`MiniLM`), SBERT médio (`MPNet`), SBERT grande (`E5-Large`), e BERTScore (`mBERT`).
- **`configuracao_comparacao.campos_parquet`**: Mapeamento das colunas do arquivo `.parquet` para os dados esperados pelo pipeline (ver seção abaixo).
- **`configuracao_comparacao.filtro`**: Permite definir um arquivo CSV ou Parquet e a coluna que servirá como filtro para a extração e carga. Apenas os IDs (da coluna especificada) que coincidirem com os dados da tabela serão avaliados. Você também pode utilizar o parâmetro `dataset_filtro` para aplicar queries dinâmicas do Pandas sobre essa base antes da extração dos IDs.
  ```yaml
  filtro: 
    arquivo: "./dados/integras_experimento_summa_novos.parquet" # Suporta CSV ou Parquet
    campo_id: "seq_documento_acordao"
    dataset_filtro: "fold <= 10" # (Opcional) Query Pandas aplicada sobre o arquivo
  ```
- **`campos_virtuais`**: (Opcional) Dicionário que permite combinar o conteúdo de múltiplas chaves do JSON em um novo campo "virtual", gerado em tempo de execução durante a carga. Ideal para métricas globais (como SBERT) ou de Prompt (LLM-as-a-judge) que precisam avaliar grandes blocos de texto agregados.
  ```yaml
  campos_virtuais:
    Likert:
      - Materia
      - Temas.Ponto
      - Resumo
  ```
  **Nota:** Campos virtuais são omitidos dinamicamente nas métricas `(global)` e `(estrutura)` para evitar inflar contagens estruturais ou duplicar textos que já existam nas chaves originais.

## 📦 Suporte a Entrada via Parquet

### Visão Geral
O pipeline suporta dois modos de entrada para os modelos (base e comparação):

| Modo | Chave YAML | Descrição |
|---|---|---|
| **Diretório** (legado) | `pasta:` | Diretório contendo arquivos `.json`, `.tokens.json`, etc. |
| **Parquet** (novo) | `arquivo:` | Arquivo `.parquet` consolidado com todas as extrações |

Quando a entrada é um `.parquet`, o pipeline automaticamente extrai os dados para um diretório de JSONs individuais antes de iniciar a comparação. A extração ocorre **uma única vez** e é cacheada — nas execuções seguintes, os dados já extraídos são reutilizados.

### Configuração de `campos_parquet`
Define o mapeamento entre as colunas do `.parquet` e os dados esperados:

```yaml
configuracao_comparacao:
  campos_parquet:
    id: "chave"             # (obrigatório) coluna com o ID do documento
    resposta: "resposta"    # (obrigatório) coluna com o JSON da extração → salvo como {id}.json
    resumo_tokens: "resumo" # (opcional) coluna com JSON de tokens → salvo como {id}.tokens.json
    avaliacao: ""           # (opcional) coluna com avaliação LLM → salvo como {id}.avaliacao.json
    erro: "erro"            # (opcional) coluna com mensagem de erro
```

### Pasta de Extração
A pasta onde os JSONs são extraídos é composta por:
```
<saida.pasta_parquet> / <nome_do_arquivo_parquet_sem_extensão> /
```
Exemplo: se `pasta_parquet: "./compara/"` e o arquivo é `saida_qwen7b.parquet`, os JSONs ficam em `./compara/saida_qwen7b/`.

### Mecanismo de Cache
Ao finalizar a extração, o sistema gera um arquivo `extracao_finalizada.md` na pasta de destino. Este arquivo funciona como um controle de cache:
- **Se existir:** a extração é pulada e o pipeline prossegue direto para a comparação.
- **Se removido:** uma nova extração é feita automaticamente na próxima execução.
- **Para forçar re-extração completa:** remova a pasta de destino inteira ou apenas o `extracao_finalizada.md`.

### Tratamento de Erros
- Registros com coluna `erro` preenchida no parquet são extraídos normalmente, mas com uma chave `"erro"` adicionada dentro do JSON gerado, permitindo que o fluxo existente os identifique e trate conforme a flag `ignorar_erro_extracao`.
- Registros com JSON inválido na coluna `resposta` geram um arquivo `{id}.json` com `{"erro": "JSON inválido na resposta: ..."}`, mantendo rastreabilidade completa.

## 🔄 Replicando Experimentos e Usando em Treino
- Os **Relatórios Analíticos**, CSVs globais e Gráficos residirão perenemente na infraestrutura em `<saida.pasta>/`.
- **Insumo para Treino**: A pasta gerada em `<saida.pasta>/divisoes` congraça o output analítico deste pipeline. Ali residirão os arquivos `divisao_<modelo>.csv` com identificações documentais (`id`), sub-anotações sobre quantificabilidade e a qual fração (teste, validacao, treino) aquele registro foi sorteado.
- Para realizar o seu Fine-Tuning interligado, você copia ou aponta este `.csv` na chave iterativa de pipeline em `treinar_unsloth.py` através da chave mãe `curriculum.divisao[x].arquivo:` .
