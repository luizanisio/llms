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
# Para pular o fluxo interativo visual:
python src/comparar_extracoes.py ./experimento_revisoes/meu_config_comparacao.yaml

# Para utilizar o console guiado que lê quais são os metadados ativos na raiz do seu desenvolvimento:
python src/comparar_extracoes.py
```

## ⚙️ Configuração (O Arquivo YAML)
Todas as manobras e experimentos de comparação prescindem da necessidade de entrar no código. Você declara suas intenções dentro do arquivo `.yaml` de configuração, com as seguintes chaves de destaque:

- **`saida`**: (Dicionário) Pasta raiz `.pasta` das avaliações e relatórios a serem gerados, além dos arquivos contendo as subdivisões no futuro de seu treinamento iterativo.
- **`modelo_base`**: Configurações relativas a pasta local onde você dispôs os txts+jsons perfeitos de gabarito para ser tomado como padrão de colidência das predições, rotulando de "Humano". (Há um gabarito único a ser setado).
- **`modelos_comparacao`**: (Lista) Suas diversas frentes de modelos prevendo outputs do texto. Incluindo `gpt4`, `qwen_lora`, onde a tool irá varrer `pasta` carregando os JSON previstos de ids respectivos, excluindo do cálculo aqueles chumbados com erro e declarando notas baseadas nisto. Pode usar `ativo: false` para desabilitar algum.
- **`execucao.divisao`**: Define as frações destinadas à criação de splits (ex: `{treino: 0.7, teste: 0.2, validacao: 0.1}`). As proporções formam os CSVs em `/divisoes/`.
- **`configuracao_comparacao.campos`**: A engrenagem primordial. Dicionário declarativo dizendo em quais chaves folha do JSON de predição você irá aplicar `bertscore`, `rouge_1`, `levenshtein`, etc. Existe também `(global)` e `(estrutura)` que sempre rodam implicitamente rastreando chaves e nós primários, de forma que eles criam seu escopo de avaliação macro para você sempre entender "Qual o F1 em relação a se acertaram ou não ao menos trazer a estrutura do campo".

## 🔄 Replicando Experimentos e Usando em Treino
- Os **Relatórios Analíticos**, CSVs globais e Gráficos residirão perenemente na infraestrutura em `<saida.pasta>/`.
- **Insumo para Treino**: A pasta gerada em `<saida.pasta>/divisoes` congraça o output analítico deste pipeline. Ali residirão os arquivos `divisao_<modelo>.csv` com identificações documentais (`id`), sub-anotações sobre quantificabilidade e a qual fração (teste, validacao, treino) aquele registro foi sorteado.
- Para realizar o seu Fine-Tuning interligado, você copia ou aponta este `.csv` na chave iterativa de pipeline em `treinar_unsloth.py` através da chave mãe `curriculum.divisao[x].arquivo:` .
