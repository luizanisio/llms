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

# Para REGERAR apenas os gráficos, estatísticas, análise bayesiana ou a planilha base
# (sem refazer a análise pesada via NLP; exige que a análise principal já tenha rodado)
python src/comparar_extracoes.py --config ./experimento_revisoes/meu_config_comparacao.yaml --graficos --estatisticas --bayesiana --planilha
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
- **`estatistica_bayesiana`**: (Opcional) Ativa a comparação bayesiana pareada entre protocolos — ver a seção dedicada abaixo. Também é aceita dentro de `configuracao_comparacao`.

## 🎲 Comparação Bayesiana Pareada (Opcional)

Camada **complementar** à análise estatística frequentista (Friedman, Wilcoxon + Holm, Nemenyi), implementada em `comparar_extracoes_baycomp.py` sobre o motor de `util_est_bayesiana.py` (teste de sinais bayesiano de Benavoli et al., 2017; `baycomp` como implementação de referência).

O que ela acrescenta é o que o teste de hipótese nula não consegue expressar: a probabilidade posterior de **equivalência prática**. "Equivalente" vira achado, e não falha em rejeitar H₀; "inconclusivo" é desfecho legítimo.

### Ativação

```yaml
estatistica_bayesiana:
  ativo: true
  eps: 0.05
  metricas_automaticas:
    rope: 0.01
    campos: ["(global)"]
    metricas: [bertscore, sbert_medio]
```

Sem a chave (ou com `ativo: false`) o pipeline se comporta exatamente como antes.

### Os dois alvos

| Alvo | Fonte | Modo | Margem | O que mede |
|---|---|---|---|---|
| **Likert do juiz LLM** (principal) | `{protocolo}_nota` dos arquivos `.avaliacao.json` | `proporcao` | `eps` sobre a posterior | qualidade percebida |
| **Métricas automáticas** (complementar) | colunas `{protocolo}_{campo}_{métrica}_F1` | `baycomp` | `rope` sobre os escores brutos | fidelidade ao modelo base |

Os rótulos coincidem sem que o significado coincida. No modo `proporcao`, equivalência é `P(|δ| ≤ ε)` — uma afirmação sobre **magnitude**, medida em proporção de documentos. No modo `baycomp`, é `P(a zona ROPE ser a maioritária)` — uma afirmação sobre **qual região concentra mais documentos**. O modo aparece na legenda de cada figura e no cabeçalho de cada seção do relatório.

⚠️ As métricas automáticas comparam cada protocolo com o **modelo base**, portanto medem fidelidade de destilação, **não qualidade**: um protocolo que reproduz fielmente um erro do modelo base é premiado por elas. Entram como triangulação da Likert, nunca como veredito.

### Dados do juiz LLM

A seção principal consome os mesmos dados que a análise frequentista já usa — não há mecanismo novo. As notas vêm dos arquivos `{id}.avaliacao.json`, que chegam por um de dois caminhos convergentes:

- **entrada `.parquet`/`.csv`**: preencha `configuracao_comparacao.campos_parquet.avaliacao` com o nome da coluna que contém o JSON da avaliação (o padrão é `""`, ou seja, desligado);
- **entrada por pasta**: deixe os arquivos `{id}.avaliacao.json` junto dos `{id}.json`, seguindo `configuracao_comparacao.mascaras.avaliacao`.

O JSON da avaliação precisa ter ao menos a chave `nota` (Likert). `precision`/`recall`/`f1`, `explicacao` e `metricas_por_campo` são opcionais e alimentam as abas de avaliação LLM do Excel. Sem notas para ao menos dois protocolos, a seção principal é omitida com aviso registrado no próprio relatório.

### Parâmetros

| Chave | Padrão | Efeito |
|---|---|---|
| `ativo` | — | única chave que decide se a etapa existe |
| `eps` | — | **obrigatório** para a Likert; margem sobre a posterior, em proporção de documentos |
| `origem_eps` | `""` | texto livre citado no relatório para justificar o ε |
| `limiar` | `0.80` | probabilidade mínima para classificar uma célula do heatmap |
| `limiar_equivalencia` | `0.95` | limiar do veredito, na curva de sensibilidade ao ε |
| `amostras` | `200000` | amostras da posterior |
| `semente` | `42` | reprodutibilidade |
| `incluir_base` | `false` | inclui o modelo base na matriz da Likert (ignorado quando `protocolos` é usado) |
| `protocolos` | todos | recorte(s) **e ordem** dos protocolos; lista simples ou `{nome: [protocolos]}` — ver abaixo |
| `metricas_automaticas.rope` | `0.0` | **obrigatória** (> 0) para a seção complementar |
| `metricas_automaticas.rope_sensibilidade` | `rope/2, rope, 2·rope` | valores da varredura |
| `metricas_automaticas.campos` | — | campos declarados em `configuracao_comparacao.campos` |
| `metricas_automaticas.metricas` | — | `bertscore`, `rouge_l`, `rouge_1`, `rouge_2`, `levenshtein`, `sbert_*` |

**O ε e a ROPE são pré-registrados.** Este pipeline não tem avaliadores humanos para calibrar o ε — isso pertence à Fase A (`realizar_avaliacoes.py --bayes`), de onde o valor deve ser trazido e fixado aqui. As curvas e varreduras de sensibilidade existem para demonstrar que a conclusão **não** depende de um número escolhido a dedo; lê-las e então adotar o valor que produz o resultado desejado é a versão bayesiana do *p-hacking*, e é detectável. Sem `eps` informado, a seção da Likert é omitida em vez de rodar com um valor de conveniência.

### Saída

Subpasta `bayesiana/` na pasta de saída, limpa a cada execução:

```
bayesiana/
├── analise_bayesiana.md                     relatório consolidado (tabelas + leitura descritiva)
├── bayes_<recorte>_likert_juiz.csv          matriz de relações (formato longo)
├── bayes_<recorte>_likert_juiz_heatmap.png
├── bayes_<recorte>_likert_juiz_curva_eps.png          P(equivalência) × ε, com o ε operacional marcado
├── bayes_<recorte>_likert_juiz_sensibilidade_limiar.csv
├── bayes_<recorte>_<campo>_<metrica>.csv
├── bayes_<recorte>_<campo>_<metrica>_heatmap.png
└── bayes_<recorte>_<campo>_<metrica>_sensibilidade_rope.csv
```

`<recorte>` é o nome declarado em `protocolos`, normalizado (`Q1_ajuste_fino` → `q1_ajuste_fino_`). Sem recortes nomeados o prefixo é vazio: `bayes_likert_juiz_heatmap.png`.

**Como ler o heatmap:** a cor comunica a categoria (verde = superior, azul = equivalente, vermelho = inferior, cinza = incerto), a intensidade comunica a magnitude da probabilidade posterior, e o número traz essa probabilidade explícita. `incerto` é categoria própria — ausência de evidência suficiente, não uma quarta relação. A diagonal é neutra porque `(Pi, Pi)` não é comparação.

**Dois limiares, dois usos:** `0.80` classifica o panorama do heatmap; `0.95` é a exigência do veredito na curva de ε. Um par pode aparecer `equivalente` no heatmap e não alcançar equivalência na curva — são perguntas com exigências diferentes, não uma inconsistência.

### Recortes de protocolos

16 protocolos geram 120 pares, e o heatmap deixa de ser legível bem antes disso. A chave `protocolos` recorta a comparação sem tocar no resto do YAML, em duas formas.

**Lista simples** — um recorte único, arquivos sem prefixo:

```yaml
estatistica_bayesiana:
  protocolos: ["A", "C", "D1", "D2"]   # alias ou rótulo, na ordem desejada
```

**Dicionário** — um recorte por questão de pesquisa, cada um com o seu conjunto de figuras, tabelas e seção no relatório:

```yaml
estatistica_bayesiana:
  protocolos:
    Q1_ajuste_fino:    ["A", "B", "C"]
    Q3_escalonamento:  ["D1", "D2", "D3", "D4"]
    Q6_fusao:          ["B", "C", "D21", "D22", "D23"]
```

O nome vira **prefixo dos arquivos** (`bayes_q1_ajuste_fino_likert_juiz_heatmap.png`) e título da seção. É o caminho recomendado para o YAML de panorama: uma única comparação processada rende várias figuras focadas, cada uma com os protocolos que aquela questão de fato contrasta.

```bash
# ajustar as listas e reexecutar SÓ esta etapa — não refaz a comparação pesada
python src/comparar_extracoes.py --config meu_config.yaml --bayesiana
```

Cada recorte faz **duas** coisas:

- **recorte** — só os protocolos listados entram na matriz;
- **ordem** — linhas e colunas saem na sequência declarada. Isso importa porque os heatmaps da Likert e das métricas automáticas são lidos lado a lado: com as linhas em ordens diferentes, a comparação visual induz leitura errada.

Detalhes de comportamento:

- aceita o **alias** (o nome que aparece nas figuras e tabelas) ou o **rótulo** do YAML, sem diferenciar maiúsculas nem espaços em volta; em caso de colisão entre o alias de um modelo e o rótulo de outro, o rótulo vence;
- nomes que não casam com nenhum modelo viram **aviso explícito**, no console e no relatório — um erro de digitação não pode encolher o heatmap em silêncio;
- um recorte que resolva menos de dois protocolos é **pulado com um aviso**, sem interromper os demais;
- o mesmo protocolo pode aparecer em vários recortes — é o caso normal quando uma baseline serve de referência para mais de uma questão;
- o **modelo base** só entra se você o listar (aí `incluir_base` é ignorado), e ainda assim apenas na Likert: nas métricas automáticas as colunas medem similaridade *com* a base, e base contra si mesma é 1,0 por construção;
- o pareamento por documento é **por recorte**: um protocolo com escores faltando não derruba documentos dos recortes em que não aparece;
- os recortes ficam registrados no cabeçalho de `analise_bayesiana.md`, para que as figuras continuem interpretáveis meses depois.

Omitida a chave, entram todos os modelos ativos, na ordem do YAML — e o módulo avisa acima de 8 protocolos.

**Sobre o volume de saída:** o número de arquivos é `recortes × campos × métricas`. O exemplo em `experimentos/summa-experimento/06_compara_todos_parcial.yaml` (6 recortes × 2 campos × 2 métricas, mais a Likert de cada recorte) gera 30 análises e ~97 arquivos. Vale dimensionar `campos`/`metricas` de `metricas_automaticas` com isso em mente — elas são triangulação, e raramente valem todas as combinações.

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
