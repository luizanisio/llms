# Guia de Uso: `util_ckan.py`

O script `util_ckan.py` foi projetado para facilitar a extração em lote e gerenciamento do acervo de jurisprudência (acórdãos) do STJ disponibilizado via CKAN. Ele lida automaticamente com downloads, caching e filtragem avançada de íntegras e espelhos.

Este guia rápido explica como utilizar o CLI para as extrações mais comuns e oferece uma visão geral para o uso em código através da classe base.

## 0. Entendendo o Mapeamento e Cruzamento (id_mapa)

Para relacionar as informações do acervo de **Espelhos** (metadados estruturados) com as **Íntegras** (textos completos), o script cria uma chave composta única chamada `id_mapa`.

**Composição do `id_mapa`:**
O `id_mapa` é gerado no formato `{numeroRegistro}.{dataPublicacao}.{tipoDeDecisao}`, onde a data de publicação é normalizada para o padrão `YYYYMMDD` e o tipo de decisão é padronizado (ex: ACORDAO, DECISAO). Exemplo: `202403674719.20250103.ACORDAO`.

**Como é feito o merge:**
Quando você realiza uma extração de Espelhos e define `incluir_integras: true`, o script:
1. Gera os dados do espelho e o `id_mapa` correspondente para cada registro.
2. Consulta o cache local das Íntegras pelo mesmo `id_mapa`.
3. Se houver correspondência, o texto integral é adicionado ao dataset final (junto com um campo booleano `tem_integra`).

**Dados Duplicados:**
Em alguns casos, o CKAN pode fornecer múltiplos registros que acabam gerando o mesmo `id_mapa` (por exemplo, republicações ou decisões corrigidas na mesma data), publicação de mais de uma decisão para o processo no mesmo dia. Quando o script detecta essa colisão durante a indexação, ele **não descarta silenciosamente** a informação. Em vez disso:
- Ele escolhe o primeiro registro.
- Registra as ocorrências adicionais como duplicatas.
- Salva relatórios (`mapa_integras_duplicados.json` ou `mapa_espelhos_duplicados.json`) no diretório de download. Isso permite que você investigue manualmente esse tipo de ocorrência no dataset original do Tribunal.
---

## 1. Como funciona o cache?

Todos os arquivos baixados pelo `util_ckan.py` ficam armazenados por padrão na pasta `downloads_stj/`.
Isso inclui:
- **`metadados_integras/`**: JSONs mensais com a lista de decisões (usados para montar o mapa).
- **`integras/`**: Arquivos ZIP contendo os textos dos documentos em `.txt`.
- **`espelhos/`**: JSONs brutos extraídos do dataset de espelhos do CKAN.
- **`mapa_integras.json` e `mapa_espelhos.json`**: Índices rápidos criados para acelerar extrações futuras sem ler milhares de arquivos.

Graças ao mapa e ao cache, ao rodar novamente uma extração para uma classe ou ano específico, os dados serão compilados em segundos, sem necessitar de downloads repetidos.

---

## 2. Uso via CLI (Command Line Interface)

O formato principal de uso para pesquisa é via linha de comando, referenciando um arquivo de configuração YAML:

```bash
python src/util_ckan.py --config config_extracao.yaml
```

### Estrutura do `config_extracao.yaml`

No YAML, você pode definir **blocos de extração**, combinando filtros variados para obter desde de anos completos até listas de processos muito específicos. 

#### Formatos de Saída e Resumo EDA

O campo `saida.arquivo` no arquivo de configuração aceita extensões `.parquet`, `.feather` ou `.csv`. O script detectará automaticamente a extensão e salvará no formato nativo correspondente. 
Além disso, toda vez que um dataset for exportado, um arquivo Markdown (`.md`) com o mesmo nome será gerado. Este arquivo contém um **resumo exploratório (EDA)** com as contagens das categorias dos campos que podem ser agrupados (como ministros, classes processuais, órgãos e ramos do direito).

#### Exemplo A: Extração Simples de Íntegras
Extrair as íntegras (textos) de todos os processos da classe "AREsp" e "REsp" nos anos de 2023 e 2024.

```yaml
atualizar_cache_e_mapas: true
download_dir: "downloads_stj"

extracoes:
  - tipo: "integras"
    incluir_texto: true
    filtros:
      anos:
        - "2023"
        - "2024"
      classes:
        - "AREsp"
        - "REsp"
    saida: "dados/acordaos_23_24.parquet"
```

#### Exemplo B: Extração de Espelhos
Extrair apenas os espelhos (metadados detalhados de julgamento, votação e classes), focando em uma data específica.

```yaml
atualizar_cache_e_mapas: true
download_dir: "downloads_stj"

extracoes:
  - tipo: "espelhos"
    incluir_integras: false   # Não cruza com mapa de integras
    incluir_ementas: true
    incluir_decisoes: true
    filtros:
      datas:
        - "2024-03-15"
      tipo_decisao: "acordao"
    saida: "dados/espelhos_15_marco.parquet"
```

#### Exemplo C: Extração Conjunta (Espelhos + Íntegras)
Cruzamento poderoso: puxa os metadados ricos do espelho e o texto longo da íntegra em uma única tabela. E filtrando dinamicamente por processos.

```yaml
atualizar_cache_e_mapas: false   # Usa só o que tem em cache local
download_dir: "downloads_stj"

extracoes:
  - tipo: "espelhos"
    incluir_integras: true       # Cruza os espelhos com o texto das íntegras
    filtros:
      processos:
        - "REsp 2274350"         # Por classe e número
        - "202403674719"         # Apenas o número do registro
        - "AREsp 2831077"
        - ["202403674719", "2025-01-03"] # Número de registro em uma publicacao
        - ["202403674719", "2025-01-03", "ACÓRDÃO"] # Número de registro em uma publicacao de um tipo
    saida: "dados/amostra_conjunta.parquet"
```

> **Dica**: O filtro `processos` permite desambiguação para casos onde o mesmo processo tem múltiplas publicações. Para usar no código, você pode passar as tuplas `("REsp 123", "2023-01-01")` ou `("REsp 123", "2023-01-01", "acordao")`. Pelo CLI, a desambiguação automática cruza o número/processo e traz todas as ocorrências mapeadas.

---

## 3. Uso Direto via Código (Classes base)

Para quem precisa inserir o CKAN numa pipeline customizada (como num notebook de análise), as classes `UtilCkanIntegra` e `UtilCkan` facilitam bastante:

### Extrair Apenas Íntegras
```python
from util_ckan import UtilCkanIntegra

# Inicializa o utilitário já com os filtros desejados
integra = UtilCkanIntegra(
    processos={"REsp 2274350", "202403674719"},
    anos={"2024"},
    download_dir="downloads_stj"
)

# Salva diretamente em um DataFrame Parquet
df_textos = integra.gerar_dataset_integras(
    caminho_saida="textos_filtrados.parquet", 
    incluir_texto=True
)

# Ou manipule em memória como um dicionário {id_mapa: texto}
dict_textos = integra.obter_integras()
```

### Extrair Espelhos (Opcionalmente com Íntegras)
```python
from util_ckan import UtilCkan

# Classe UtilCkan lida com espelhos
ckan = UtilCkan(
    classes={"HC", "RHC"},
    datas={"2023-11-17"},
    download_dir="downloads_stj"
)

# Traz os espelhos ricos + a flag 'tem_integra' e o texto se cruzar
df_completo = ckan.gerar_dataset_espelhos(
    incluir_integras=True, 
    caminho_saida="hc_rhc_completo.parquet"
)
```

## Dicas Rápidas e Pegadinhas

- **Sintaxe YAML**: No YAML de configuração, tanto faz usar o formato de bloco (`- item`) ou de lista embutida (`[item]`). Exemplo: `processos: ["AREsp 123", "REsp 456"]` funciona exatamente da mesma forma que colocar uma lista clássica um embaixo do outro.
- **Lógica OU (OR)**: Se você informar `processos` E `documentos` na mesma extração, a ferramenta traz os registros que baterem com **qualquer um dos dois**. Basta estar na lista de documentos ou na lista de processos para ser extraído.
- **Cuidado com o `tipo_decisao`!**: O STJ separa acórdãos de decisões monocráticas. Se você informar uma lista de processos específicos e também colocar `tipo_decisao: "acordao"`, qualquer decisão monocrática ("DECISÃO") desses processos será barrada pelo filtro e **não** será extraída. Se quiser todos os documentos daquele processo, remova o filtro de `tipo_decisao`.
- **Análise de Duplicatas**: Durante a geração dos mapas em cache, o script valida se os dados do CKAN não mandaram o mesmo `id_mapa` múltiplas vezes. Sempre que isso ocorre, um relatório automático é salvo em `downloads_stj/mapa_integras_duplicados.json` ou `mapa_espelhos_duplicados.json` detalhando a origem para ser validado.
- **Dedução Automática de Anos**: Se você passar uma lista de `processos` no YAML mas esquecer de passar os `anos`, o código irá inspecionar o mapa mais recente, deduzir em quais anos os processos caem, e baixar apenas os ZIPs pesados necessários para aquele texto.