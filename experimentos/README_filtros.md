# 🔍 Filtros de Dados: Extração, Treinamento e Comparação

Todo o pipeline (extração → treino → avaliação) opera sobre **um único dataset**, e o recorte de
dados usado em cada etapa é definido por filtros declarados no YAML. Este documento é a
referência da sintaxe desses filtros, de onde cada um age e das armadilhas conhecidas.

> **Sequência de execução:** [README_PASSO_A_PASSO.md](./README_PASSO_A_PASSO.md) ·
> **Filtros efetivamente usados em cada experimento:** [INFORMACOES_TREINAMENTO.md §4](./INFORMACOES_TREINAMENTO.md#4-conjuntos-de-teste-e-filtros-por-experimento)

---

## 1. Mapa Rápido: Onde Cada Filtro Age

| Etapa | Script | Chave no YAML | O que recorta |
| :-- | :-- | :-- | :-- |
| **02 / 05** — Extração (inferência) | [`util_vllm_batch.py`](../src/util_vllm_batch.py) | `entrada.filtro` | Quais documentos o modelo vai processar |
| **04** — Treinamento (entrada) | [`treinar_unsloth.py`](../src/treinar_unsloth.py) | `curriculum.entrada.dataset_filtro` | Universo de documentos disponível para todo o treino |
| **04** — Treinamento (etapa do currículo) | [`treinar_unsloth.py`](../src/treinar_unsloth.py) | `curriculum.divisao[].dataset_filtro` | Quais documentos entram **naquela etapa** do CL |
| **03 / 06** — Comparação | [`comparar_extracoes.py`](../src/comparar_extracoes.py) | `configuracao_comparacao.filtro` | Quais documentos entram nas métricas |
| **03 / 06** — Comparação (por modelo) | [`comparar_extracoes.py`](../src/comparar_extracoes.py) | `modelos_comparacao[].dataset_filtro` | Recorte extra aplicado a um parquet específico |

Todos usam a mesma função de baixo nível: `aplicar_filtro_dataset()` em
[`util_pandas.py`](../src/util_pandas.py).

---

## 2. Sintaxe do `dataset_filtro`

Duas formas, aceitas em qualquer ponto do pipeline (exceto onde indicado):

### Forma dicionário (recomendada)

```yaml
dataset_filtro: {"split": "test"}                       # igualdade
dataset_filtro: {"dificuldade": "!=facil"}              # diferente
dataset_filtro: {"dificuldade_int": "<=3"}              # menor ou igual
dataset_filtro: {"alvo": "teste", "fold": "<=10"}       # AND entre as colunas
```

Operadores reconhecidos no início da string: `==`, `=`, `!=`, `>=`, `<=`, `>`, `<`. Sem operador,
assume-se `==`. O valor é convertido automaticamente para o tipo da coluna (numérico quando a
coluna é numérica), então `{"dificuldade_int": "<=3"}` funciona mesmo com a coluna sendo `int`.

Se a coluna não existir no dataframe, o script **aborta com erro** listando as colunas
disponíveis — proteção contra filtros silenciosamente vazios por erro de digitação.

### Forma query (pandas)

```yaml
dataset_filtro: "fold <= 10 and dificuldade != 'facil'"
```

Repassada direto para `df.query()`. Mais expressiva (`or`, parênteses, `in`); erro de sintaxe ou
coluna inexistente também **abortam** com `ValueError`. Os YAMLs dos experimentos usam só a forma
dicionário.

> ⚠️ **Exceção:** em `entrada.filtro.dataset_filtro` da extração
> ([`util_vllm_batch.py`](../src/util_vllm_batch.py)), só a forma **dicionário** é aplicada;
> strings de query são ignoradas sem aviso.

---

## 3. Filtros na Extração (`02_*`, `05_*`)

O bloco `entrada.filtro` aceita três mecanismos, que podem ser combinados (o resultado é a
**interseção**):

```yaml
entrada:
  arquivo: "dados/integras_experimento_summa_novos.parquet"
  campo_chave: "seq_documento_acordao"
  campo_texto: "integra"
  filtro:
    # (a) filtro por coluna do próprio parquet de entrada
    dataset_filtro: {"fold": "<=10"}

    # (b) lista explícita de IDs vinda de um CSV
    arquivo: "saida/filtro_teste_fold_11.csv"
    campo_id: "id"

    # (c) IDs vindos de outro arquivo, filtrado antes
    filtro_externo:
      arquivo: "dados/divisao_Qwen235b_Qwen7b.csv"
      campo_id: "id"
      dataset_filtro: {"alvo": "teste"}
```

| Mecanismo | Quando usar |
| :-- | :-- |
| `dataset_filtro` | O recorte já existe como coluna no parquet de entrada (`split` do PubMed, `fold` do Summa) |
| `arquivo` + `campo_id` | Lista de IDs preparada à mão (amostras, casos de erro, subconjunto de debug) |
| `filtro_externo` | O recorte está no **CSV de divisão** — forma canônica de extrair "só o teste" quando o split foi definido pelo pipeline e não pelo dataset original |

- **Linha de base e referência (`02_*`)** rodam **sem filtro**: a dificuldade da etapa 03 precisa
  do corpus inteiro.
- **Modelos treinados (`05_*`)** inferem só sobre o teste, economizando GPU.

> 💡 A extração é **retomável**: ao reexecutar, itens já processados com sucesso são pulados e
> apenas os itens com erro ou ausentes são refeitos. Ampliar o filtro depois (ex.: de 100 para
> 2.500 IDs) reaproveita o que já existe no parquet de saída.

---

## 4. Filtros no Treinamento (`04_*`)

Dois níveis independentes, que se combinam por interseção.

### 4.1 `curriculum.entrada.dataset_filtro` — universo do experimento

```yaml
curriculum:
  entrada:
    dataframe: dados/integras_experimento_summa_novos.parquet
    dataframe_col: integra
    dataframe_id: seq_documento_acordao
    dataset_filtro: {"fold": "<=10"}
```

Aplica-se **apenas ao parquet de entrada**. Registros descartados aqui não disparam erros de
validação de pareamento com o CSV de divisão — o script apenas avisa quantos IDs da divisão foram
ignorados. É o recorte certo para "este experimento não enxerga os folds 11 e 12".

### 4.2 `curriculum.divisao[].dataset_filtro` — as etapas do currículo

Cada item da lista `divisao` é uma etapa de treinamento. O `dataset_filtro` da etapa é aplicado
**ao CSV de divisão daquela etapa**, e é assim que o eixo CL é implementado:

```yaml
  divisao:
  - arquivo: dados/divisao_Professor_Qwen1_5B.csv
    dataset_filtro: {"dificuldade": "facil"}
    alias: "fácil-FF"
    tipo: "full"
    pace_epochs: 2
  - arquivo: dados/divisao_Professor_Qwen1_5B.csv
    dataset_filtro: {"dificuldade": "medio"}
    alias: "médio"
    tipo: "lora"
    pace_epochs: 2
  - arquivo: dados/divisao_Professor_Qwen1_5B.csv
    dataset_filtro: {"dificuldade": "dificil"}
    alias: "difícil"
    tipo: "lora"
    pace_epochs: 2
  - arquivo: dados/divisao_Professor_Qwen1_5B.csv   # sem filtro = dataset completo
    alias: "completo"
    tipo: "lora"
    pace_epochs: 2
```

Variações usadas nos protocolos:

| Estratégia | Filtros por etapa | Protocolos |
| :-- | :-- | :-- |
| **Por etapas disjuntas** | `facil` → `medio` → `dificil` → *(sem filtro)* | d1, d2, d7, d13, d17… |
| **Acumulado (replay)** | `dificuldade_int <=3` → `<=7` → *(sem filtro)* | d3, d4, d8 |
| **Anti-CL** | *(sem filtro)* → `dificil` → `medio` → `facil` (d9) · `dificuldade_int >7` → `>3` → *(sem filtro)* (d10) | d9, d10 |
| **Granular** | `<=1` → `<=2` → … → `<=9` → *(sem filtro)* | d11, d12, d24, d25 |
| **Blocos aleatórios** | `{"bloco": "b1"}` → `b2` → `b3` → *(sem filtro)*, sobre `dados/divisao_*_blocos.csv` | d18 |

A coluna `bloco` não vem da etapa 03: é acrescentada por `gerar_divisao_blocos_d18.py` (terços
aleatórios, seed 3407), que grava `dados/divisao_*_blocos.csv` sem alterar o CSV original.

### 4.3 A coluna `alvo` é sempre respeitada

Independente do `dataset_filtro`, o treinamento só consome linhas com `alvo == "treino"`; a
validação usa `alvo == "validacao"`; e `alvo == "teste"` **nunca** entra no treino — fica
reservado para a avaliação final. O filtro da etapa recorta *dentro* do treino, não sobre o split.

> 🔒 **Regra de ouro:** `dataset_filtro` responde "quais documentos nesta etapa"; a coluna `alvo`
> responde "quais documentos posso usar para aprender". Nunca use `alvo` como filtro de etapa para
> tentar incluir teste no treino.

---

## 5. Filtros na Comparação (`03_*`, `06_*`)

### 5.1 O bloco `filtro`

```yaml
configuracao_comparacao:
  filtro:
    arquivo: "dados/pubmed-rct-20k.parquet"   # pode ser .parquet ou .csv
    campo_id: "pmid"
    dataset_filtro: {"split": "test"}
```

Lê o arquivo indicado, aplica o `dataset_filtro` **sobre esse arquivo** e extrai o conjunto de IDs
resultante. Esse conjunto é usado em **duas camadas**:

1. **Na extração dos JSONs** — cada parquet (inclusive o da referência) só gera arquivos para os
   IDs do filtro.
2. **Na carga da comparação** — os IDs da base são intersectados com o filtro, e só esse conjunto
   entra no laço de métricas.

A segunda camada é a garantia real: mesmo que a pasta de JSONs da referência contenha o corpus
inteiro (por ter sido gerada por outro YAML), apenas os IDs do filtro são avaliados. Consequência
prática: **usar uma referência extraída sobre todos os dados contra alunos extraídos só no teste
não distorce as métricas** — todas as métricas (BERTScore sem IDF e sem rescale, ROUGE,
Levenshtein, SBERT) são pareadas por documento, e os testes estatísticos operam sobre a mesma
matriz de IDs para todos os modelos.

Equivalente com CSV de divisão (Summa e SemClinBr):

```yaml
  filtro:
    arquivo: "dados/divisao_Qwen235b_Qwen7b.csv"
    campo_id: "id"
    dataset_filtro: {"alvo": "teste", "fold": "<=10"}   # SemClinBr: {"alvo": "teste"}
```

> ⚠️ `arquivo` **e** `campo_id` são obrigatórios. Um bloco `filtro` só com `dataset_filtro` é
> ignorado por completo, e a comparação roda sobre tudo que existir na pasta da base.

### 5.2 Filtro por modelo

```yaml
modelos_comparacao:
  - arquivo: "saidas/saida_pubmed_1_5b(d7)_teste.parquet"
    rotulo: "Qwen1.5B(d7)"
    dataset_filtro: {"erro": ""}    # recorte extra só deste parquet
```

Aplicado ao parquet daquele modelo antes da extração dos JSONs. Use com parcimônia: recortar um
modelo e não os outros quebra o pareamento e enviesa a comparação.

---

## 6. O CSV de Divisão: Onde Nascem `dificuldade` e `alvo`

O arquivo de divisão é **gerado pela comparação inicial** (`03_*`), não escrito à mão. Ao comparar
a referência contra o modelo base, [`util_json_divisoes.py`](../src/util_json_divisoes.py) grava em
`compara/<analise>/divisoes/` um CSV por modelo com todas as métricas por documento mais as
colunas de controle:

| Coluna | Conteúdo |
| :-- | :-- |
| `id` | Identificador do documento (migrado automaticamente para `id_arquivo` no treino) |
| `dificuldade` | `facil` · `medio` · `dificil` (30% / 40% / 30%) |
| `dificuldade_int` | 1 a 10 — `1–3` fácil, `4–7` médio, `8–10` difícil |
| `alvo` | `treino` · `validacao` · `teste` |

A dificuldade é derivada do **desempenho do modelo base contra a referência**, combinado com a
complexidade do documento (número de chaves-alvo, verbosidade): notas baixas = documento mais
difícil. A distribuição em `dificuldade_int` usa *ranking* dentro de cada faixa, garantindo
estratos equilibrados.

Colunas extras podem ser acrescentadas depois, desde que o CSV seja copiado de novo para `dados/`
(o Summa acrescenta `fold` por join com as íntegras; o D18 usa a cópia com `bloco`, §4.2).

### Configurando a geração

```yaml
execucao:
  ignorar_erro_extracao: false   # ver §7
  divisao:
    treino: 0.7        # opcional — padrão (0.7, 0.2, 0.1)
    teste: 0.2
    validacao: 0.1
    arquivo_referencia: "dados/divisao_pubmed.csv"
```

**`arquivo_referencia` é o item mais importante para replicação.** Se informado, o script não
sorteia novos alvos: faz merge da dificuldade recém-calculada com os alvos já existentes nesse
arquivo (que precisa ter as colunas de ID e `alvo`). Sem ele, cada nova comparação sorteia um split
diferente — e qualquer modelo treinado com o split antigo passa a ser avaliado sobre documentos que
viu no treino (**data leakage**). O PubMed usa `dados/divisao_pubmed.csv` (splits oficiais); Summa e
SemClinBr sortearam o split na primeira execução, e o CSV copiado para `dados/` é a semente de
reprodutibilidade.

---

## 7. Armadilhas Conhecidas

**A pasta de extração é cacheada pelo nome do parquet, não pelo filtro.** Os JSONs vão para
`<saida.pasta_parquet>/<nome_do_parquet>/`, e a reextração só ocorre se o parquet for mais novo que
o arquivo de controle `extracao_finalizada.md`. Dois YAMLs com filtros diferentes sobre o mesmo
parquet **compartilham a pasta**, e o conteúdo é o do último que rodou. Isso não afeta as métricas
(a interseção de IDs na carga protege), mas confunde a inspeção manual. Para forçar a reextração,
apague o `extracao_finalizada.md` da pasta.

**IDs são comparados como texto.** Todos os conjuntos de filtro passam por
`astype(str).str.strip()`. IDs numéricos lidos como `float` (`12345.0`) ou com zeros à esquerda não
casam com o nome do arquivo JSON. Se um filtro resultar em zero documentos comparados, verifique o
tipo da coluna antes de qualquer outra coisa.

**Sem o bloco `filtro`, uma base completa contamina a comparação.** Se o YAML usa a referência
extraída sobre todos os splits e não declara `filtro`, os documentos sem contraparte nos alunos
entram como `Inexistente` (campo `pred` vazio, escore 0) e derrubam as médias. Confira no
`execucao.log`: `Filtro carregado: N IDs`, `IDs na origem (a processar): N` e `Total comparado: N`
devem coincidir com o tamanho do teste.

**`ignorar_erro_extracao` muda o denominador.** Com `false`, documentos em que o modelo produziu
JSON inválido permanecem na comparação e penalizam aquele modelo; com `true`, saem das métricas — o
que infla o resultado de modelos instáveis. Todos os `06_compara_*` usam `false`; mantenha o mesmo
valor em todas as comparações colocadas lado a lado. Na etapa **03** o efeito é outro: o valor
decide se os documentos em que o modelo base falhou entram na divisão (`false`, como difíceis) ou
ficam fora dela (`true`). PubMed e SemClinBr geraram a divisão com `false`; o Summa, com `true`
(114 documentos ficaram fora).

**Divisão fora de `dados/`.** Os passos 04, 05 e 06 procuram o CSV de divisão em `dados/`. A etapa 03
o grava em `compara/<analise>/divisoes/`. Sem a cópia, a comparação só imprime
`⚠️ Aviso: Arquivo de filtro não encontrado` e segue **sem filtro** — com o efeito `Inexistente`
descrito acima.
