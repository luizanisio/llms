# 🔍 Filtros de Dados: Extração, Treinamento e Comparação

> 🛠️ Para setup de ambiente e execução, veja o [README.md](./README.md) · 🧭 Para o desenho dos protocolos (`b`, `c`, `d1`–`d25`), veja o [README_protocolos.md](./README_protocolos.md).

Todo o pipeline (extração → treino → avaliação) opera sobre **um único dataset**, e o recorte de dados usado em cada etapa é definido por filtros declarados no YAML. Este documento descreve a sintaxe desses filtros, onde cada um age e como combiná-los para replicar os experimentos ou montar novos.

---

## 1. Mapa Rápido: Onde Cada Filtro Age

| Etapa | Script | Chave no YAML | O que recorta |
| :-- | :-- | :-- | :-- |
| **02 / 05** — Extração (inferência) | [`util_vllm_batch.py`](../src/util_vllm_batch.py) | `entrada.filtro` | Quais documentos o modelo vai processar |
| **04** — Treinamento (entrada) | [`treinar_unsloth.py`](../src/treinar_unsloth.py) | `curriculum.entrada.dataset_filtro` | Universo de documentos disponível para todo o treino |
| **04** — Treinamento (etapa do currículo) | [`treinar_unsloth.py`](../src/treinar_unsloth.py) | `curriculum.divisao[].dataset_filtro` | Quais documentos entram **naquela etapa** do CL |
| **03 / 06** — Comparação | [`comparar_extracoes.py`](../src/comparar_extracoes.py) | `configuracao_comparacao.filtro` | Quais documentos entram nas métricas |
| **03 / 06** — Comparação (por modelo) | [`comparar_extracoes.py`](../src/comparar_extracoes.py) | `modelos_comparacao[].dataset_filtro` | Recorte extra aplicado a um parquet específico |

Todos usam a mesma função de baixo nível: `aplicar_filtro_dataset()` em [`util_pandas.py`](../src/util_pandas.py).

---

## 2. Sintaxe do `dataset_filtro`

Duas formas equivalentes, aceitas em qualquer ponto do pipeline (exceto onde indicado):

### Forma dicionário (recomendada)

```yaml
dataset_filtro: {"split": "test"}                       # igualdade
dataset_filtro: {"dificuldade": "!=facil"}              # diferente
dataset_filtro: {"dificuldade_int": "<=3"}              # menor ou igual
dataset_filtro: {"fold": ">10", "alvo": "treino"}       # AND entre as colunas
```

Operadores reconhecidos no início da string: `==`, `=`, `!=`, `>=`, `<=`, `>`, `<`. Sem operador, assume-se `==`. O valor é convertido automaticamente para o tipo da coluna (numérico quando a coluna é numérica), então `{"dificuldade_int": "<=3"}` funciona mesmo com a coluna sendo `int`.

Se a coluna não existir no dataframe, o script **aborta com erro** listando as colunas disponíveis — é uma proteção contra filtros silenciosamente vazios por erro de digitação.

### Forma query (pandas)

```yaml
dataset_filtro: "fold <= 10 and dificuldade != 'facil'"
```

Repassada direto para `df.query()`. Mais expressiva (`or`, parênteses, `in`), porém **falha em silêncio**: erro de sintaxe apenas emite um aviso e o dataframe volta sem filtro. Prefira a forma dicionário quando a integridade do recorte for crítica.

> ⚠️ **Exceção:** na extração ([`util_vllm_batch.py`](../src/util_vllm_batch.py)), o `dataset_filtro` só é aplicado quando é um **dicionário**. Strings de query são ignoradas ali.

---

## 3. Filtros na Extração (`02_*`, `05_*`)

O bloco `entrada.filtro` aceita três mecanismos, que podem ser combinados (o resultado é a **interseção**):

```yaml
entrada:
  arquivo: "dados/pubmed-rct-20k.parquet"
  campo_chave: "pmid"
  campo_texto: "article"
  filtro:
    # (a) filtro por coluna do próprio parquet de entrada
    dataset_filtro: {"split": "test"}

    # (b) lista explícita de IDs vinda de um CSV
    arquivo: "dados/filtro_ids.csv"
    campo_id: "pmid"

    # (c) IDs vindos de outro arquivo, filtrado antes
    filtro_externo:
      arquivo: "dados/divisao_Professor_Qwen1_5B.csv"
      campo_id: "id"
      dataset_filtro: {"alvo": "teste"}
```

| Mecanismo | Quando usar |
| :-- | :-- |
| `dataset_filtro` | O recorte já existe como coluna no parquet de entrada (ex.: `split` do PubMed) |
| `arquivo` + `campo_id` | Lista de IDs preparada à mão (amostras, casos de erro, subconjunto de debug) |
| `filtro_externo` | O recorte está no **CSV de divisão** — é a forma canônica de extrair "só o teste" quando o split foi definido pelo pipeline e não pelo dataset original |

**Casos reais no repositório:**

- `02_summa_235b.yaml` / `02_pubmed_rct_1_5b.yaml` — **sem filtro**: o professor extrai o corpus inteiro (é ele quem gera o gabarito e a métrica de dificuldade).
- `05_extracao_d13_teste.yaml` — `dataset_filtro: {"split": "test"}`: os modelos treinados só inferem sobre o teste, economizando GPU.

> 💡 A extração é **retomável**: ao reexecutar, itens já processados com sucesso são pulados e apenas os itens com erro ou ausentes são refeitos. Ampliar o filtro depois (ex.: de 100 para 2.500 IDs) reaproveita o que já existe no parquet de saída.

---

## 4. Filtros no Treinamento (`04_*`)

Dois níveis independentes, que se combinam por interseção.

### 4.1 `curriculum.entrada.dataset_filtro` — universo do experimento

```yaml
curriculum:
  entrada:
    dataframe: dados/pubmed-rct-20k.parquet
    dataframe_col: article
    dataframe_id: pmid
    dataset_filtro: {"fold": "!=12"}
```

Aplica-se **apenas ao parquet de entrada**. Registros descartados aqui não disparam erros de validação de pareamento com o CSV de divisão — o script apenas avisa quantos IDs da divisão foram ignorados. É o recorte certo para "este experimento não enxerga o fold 12".

### 4.2 `curriculum.divisao[].dataset_filtro` — as etapas do currículo

Cada item da lista `divisao` é uma etapa de treinamento. O `dataset_filtro` da etapa é aplicado **ao CSV de divisão daquela etapa**, e é assim que o eixo CL é implementado:

```yaml
  divisao:
  - arquivo: dados/divisao_Professor_Qwen1_5B.csv
    dataset_filtro: {"dificuldade": "facil"}
    alias: "fácil-FF"
    tipo: "full"
    pace_epochs: 2
  - arquivo: dados/divisao_Professor_Qwen1_5B.csv
    dataset_filtro: {"dificuldade": "medio"}
    alias: "médio-FF"
    tipo: "full"
    pace_epochs: 2
  - arquivo: dados/divisao_Professor_Qwen1_5B.csv
    dataset_filtro: {"dificuldade": "dificil"}
    alias: "difícil-FF"
    tipo: "full"
    pace_epochs: 2
  - arquivo: dados/divisao_Professor_Qwen1_5B.csv   # sem filtro = dataset completo
    alias: "completo-FF"
    tipo: "full"
    pace_epochs: 2
```

Variações usadas nos protocolos:

| Estratégia | Filtros por etapa | Protocolos |
| :-- | :-- | :-- |
| **Por etapas disjuntas** | `facil` → `medio` → `dificil` → *(sem filtro)* | d7, d13, d14… |
| **Acumulado (replay)** | `<=3` → `<=7` → *(sem filtro)* | d8, d10 |
| **Granular** | `<=1` → `<=2` → … → `<=9` → *(sem filtro)* | d11, d12 |

### 4.3 A coluna `alvo` é sempre respeitada

Independente do `dataset_filtro`, o treinamento só consome linhas com `alvo == "treino"`; a validação usa `alvo == "validacao"`; e `alvo == "teste"` **nunca** entra no treino — fica reservado para a avaliação final. O filtro da etapa recorta *dentro* do treino, não sobre o split.

> 🔒 **Regra de ouro:** `dataset_filtro` responde "quais documentos nesta etapa"; a coluna `alvo` responde "quais documentos posso usar para aprender". Nunca use `alvo` como filtro de etapa para tentar incluir teste no treino.

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

Lê o arquivo indicado, aplica o `dataset_filtro` **sobre esse arquivo** e extrai o conjunto de IDs resultante. Esse conjunto é então usado em **duas camadas**:

1. **Na extração dos JSONs** — cada parquet (inclusive o do professor) só gera arquivos para os IDs do filtro.
2. **Na carga da comparação** — os IDs da base são intersectados com o filtro, e só esse conjunto entra no laço de métricas.

A segunda camada é a garantia real: mesmo que a pasta de JSONs do professor contenha o corpus inteiro (por ter sido gerada por outro YAML), apenas os IDs do filtro são avaliados. Consequência prática: **usar um professor extraído sobre todos os dados contra alunos extraídos só no teste não distorce as métricas** — todas as métricas (BERTScore sem IDF e sem rescale, ROUGE, Levenshtein, SBERT) são pareadas por documento, e os testes estatísticos (Friedman/Wilcoxon/Nemenyi) operam sobre a mesma matriz de IDs para todos os modelos.

Equivalente com CSV de divisão (padrão do Summa):

```yaml
  filtro:
    arquivo: "dados/divisao_Qwen235b_Qwen7b.csv"
    campo_id: "id"
    dataset_filtro: {"alvo": "teste"}
```

> ⚠️ `arquivo` **e** `campo_id` são obrigatórios. Um bloco `filtro` só com `dataset_filtro` é ignorado por completo, e a comparação roda sobre tudo que existir na pasta da base.

### 5.2 Filtro por modelo

```yaml
modelos_comparacao:
  - arquivo: "saidas/saida_pubmed_1_5b(d7)_teste.parquet"
    rotulo: "Qwen1.5B(d7)"
    dataset_filtro: {"erro": ""}    # recorte extra só deste parquet
```

Aplicado ao parquet daquele modelo antes da extração dos JSONs. Use com parcimônia: recortar um modelo e não os outros quebra o pareamento e enviesa a comparação.

---

## 6. O CSV de Divisão: Onde Nascem `dificuldade` e `alvo`

O arquivo de divisão é **gerado pela comparação inicial** (`03_*`), não escrito à mão. Ao comparar o professor contra o modelo base, [`util_json_divisoes.py`](../src/util_json_divisoes.py) grava em `compara/<analise>/divisoes/` um CSV por modelo com todas as métricas por documento mais três colunas de controle:

| Coluna | Conteúdo |
| :-- | :-- |
| `id` | Identificador do documento (migrado automaticamente para `id_arquivo` no treino) |
| `dificuldade` | `facil` · `medio` · `dificil` |
| `dificuldade_int` | 1 a 10 — `1–3` fácil, `4–7` médio, `8–10` difícil |
| `alvo` | `treino` · `validacao` · `teste` |

A dificuldade é derivada do **desempenho do modelo base contra o gabarito do professor**, combinado com a complexidade do documento (número de chaves-alvo, verbosidade): notas baixas = documento mais difícil. A distribuição em `dificuldade_int` usa *ranking* dentro de cada faixa, garantindo estratos equilibrados.

### Configurando a geração

```yaml
execucao:
  divisao:
    treino: 0.7        # opcional — padrão (0.7, 0.2, 0.1)
    teste: 0.2
    validacao: 0.1
    arquivo_referencia: "dados/divisao_pubmed.csv"
```

**`arquivo_referencia` é o item mais importante para replicação.** Se informado, o script não sorteia novos alvos: faz merge da dificuldade recém-calculada com os alvos já existentes nesse arquivo (que precisa ter as colunas de ID e `alvo`). Sem ele, cada nova comparação sorteia um split diferente — e qualquer modelo treinado com o split antigo passa a ser avaliado sobre documentos que viu no treino (**data leakage**).

---

## 7. Receitas

### 7.1 Comparação inicial e geração da divisão de dificuldade

Objetivo: obter o CSV com `dificuldade` / `alvo` que alimentará todos os treinos.

```yaml
# 03_compara_prof_full.yaml — SEM bloco `filtro` (corpus completo)
execucao:
  divisao:
    arquivo_referencia: "dados/divisao_pubmed.csv"   # fixa o split, se já existir

modelo_base:
  arquivo: "saidas/saida_pubmed_prof.parquet"        # professor = gabarito
modelos_comparacao:
  - arquivo: "saidas/saida_pubmed_1_5b.parquet"      # modelo base zero-shot
    rotulo: "Qwen1.5B"
```

```bash
python ../../src/comparar_extracoes.py --config 03_compara_prof_full.yaml
cp "compara/analises_.../divisoes/divisao_Professor_Qwen1_5B.csv" dados/
```

Aqui o filtro é **ausente de propósito**: a dificuldade precisa ser calculada sobre todo o corpus, senão não há como estratificar treino e teste.

### 7.2 Avaliação dos modelos treinados sobre o teste

Objetivo: comparar N modelos treinados contra o professor, só no split reservado.

```yaml
# 06_compara_ablacoes.yaml
configuracao_comparacao:
  filtro:
    arquivo: "dados/pubmed-rct-20k.parquet"
    campo_id: "pmid"
    dataset_filtro: {"split": "test"}
```

Ou, quando o split vive no CSV de divisão:

```yaml
  filtro:
    arquivo: "dados/divisao_Professor_Qwen1_5B.csv"
    campo_id: "id"
    dataset_filtro: {"alvo": "teste"}
```

O professor pode continuar apontando para o parquet completo — o filtro cuida do recorte nas duas pontas.

### 7.3 Rodada rápida de validação (poucos documentos)

Para testar YAML, máscaras e campos sem gastar GPU/CPU:

```yaml
execucao:
  teste_rapido: true        # desativa BERTScore/SBERT, joga os campos para ROUGE-L
configuracao_comparacao:
  filtro:
    arquivo: "dados/filtro_10_docs.csv"
    campo_id: "pmid"
```

Gere o CSV de 10 IDs com um recorte do próprio arquivo de divisão. Aponte `saida.pasta` para uma pasta de análise distinta, para não sobrescrever os resultados oficiais.

### 7.4 Currículo granular por dificuldade

```yaml
  divisao:
  - {arquivo: dados/divisao.csv, dataset_filtro: {"dificuldade_int": "<=1"}, alias: "e1", tipo: "lora", pace_epochs: 1}
  - {arquivo: dados/divisao.csv, dataset_filtro: {"dificuldade_int": "<=2"}, alias: "e2", tipo: "lora", pace_epochs: 1}
  # … até <=9
  - {arquivo: dados/divisao.csv, alias: "completo", tipo: "lora", pace_epochs: 1}
```

### 7.5 Extrair somente o teste quando o dataset não tem coluna de split

```yaml
entrada:
  arquivo: "dados/integras_experimento_summa.parquet"
  campo_chave: "seq_documento_acordao"
  filtro:
    filtro_externo:
      arquivo: "dados/divisao_Qwen235b_Qwen7b.csv"
      campo_id: "id"
      dataset_filtro: {"alvo": "teste"}
```

---

## 8. Armadilhas Conhecidas

**A pasta de extração é cacheada pelo nome do parquet, não pelo filtro.** Os JSONs vão para `<saida.pasta_parquet>/<nome_do_parquet>/`, e a reextração só ocorre se o parquet for mais novo que o arquivo de controle `extracao_finalizada.md`. Dois YAMLs com filtros diferentes sobre o mesmo parquet **compartilham a pasta**, e o conteúdo é o do último que rodou. Isso não afeta as métricas (a interseção de IDs na carga protege), mas confunde a inspeção manual. Para forçar a reextração, apague o `extracao_finalizada.md` da pasta.

**IDs são comparados como texto.** Todos os conjuntos de filtro passam por `astype(str).str.strip()`. IDs numéricos lidos como `float` (`12345.0`) ou com zeros à esquerda não casam com o nome do arquivo JSON. Se um filtro resultar em zero documentos comparados, verifique o tipo da coluna antes de qualquer outra coisa.

**Sem o bloco `filtro`, uma base completa contamina a comparação.** Se o YAML usa o professor extraído sobre todos os splits e não declara `filtro`, os documentos sem contraparte nos alunos entram como `Inexistente` e derrubam as médias. Confira no log: `IDs na origem (a processar)` deve bater com o tamanho esperado do teste.

**`ignorar_erro_extracao` muda o denominador.** Com `false` (padrão dos experimentos), documentos em que o modelo produziu JSON inválido permanecem na comparação e penalizam aquele modelo. Com `true`, eles saem das métricas — o que infla o resultado de modelos instáveis. Mantenha o mesmo valor em todas as comparações que serão colocadas lado a lado.

**Filtro de query falha em silêncio.** `dataset_filtro: "coluna_inexistente > 5"` apenas imprime um aviso e segue sem filtrar. A forma dicionário levanta exceção — use-a em produção.

---

## 9. Checklist de Replicação

1. Extrair o professor **sem filtro** (corpus completo) — `02_*`.
2. Extrair o modelo base zero-shot **sem filtro** — `02_*`.
3. Rodar a comparação inicial **sem filtro**, com `execucao.divisao.arquivo_referencia` apontando para o split fixo (ou gerando-o na primeira vez) — `03_*`.
4. Copiar o CSV de `divisoes/` para `dados/` e **versioná-lo**: ele é a semente de reprodutibilidade de todo o resto.
5. Treinar usando `divisao[].dataset_filtro` sobre `dificuldade` / `dificuldade_int` — `04_*`.
6. Extrair cada modelo treinado com `entrada.filtro` restrito ao teste — `05_*`.
7. Comparar com `configuracao_comparacao.filtro` restrito ao teste, mantendo o mesmo filtro em **todos** os YAMLs de comparação que serão confrontados — `06_*`.
8. Conferir no `execucao.log` de cada comparação: `Filtro carregado: N IDs`, `IDs na origem (a processar): N` e `Total comparado: N` devem coincidir.
