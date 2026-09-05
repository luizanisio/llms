# Extração e Preparação do Corpus SemClinBr

## Sobre o Dataset SemClinBr

O projeto [SemClinBr](https://github.com/HAILab-PUCPR/SemClinBr) disponibiliza um corpus de narrativas clínicas em português anotadas semanticamente com conceitos clínicos. Além do corpus, o repositório oficial fornece recursos adicionais, como uma ferramenta de anotação de texto (em versão beta) e listas de abreviações médicas e termos de negação.

Para ter acesso ao corpus anotado, é necessário que a equipe de pesquisa preencha e assine um formulário de solicitação disponível no [repositório do projeto](https://github.com/HAILab-PUCPR/SemClinBr).

O artigo original associado à criação do dataset é o **SemClinBr - a multi-institutional and multi-specialty semantically annotated corpus for Portuguese clinical NLP tasks** ([DOI: 10.1186/s13326-022-00269-1](https://doi.org/10.1186/s13326-022-00269-1)). Para citações, utilize a seguinte referência:

## Preparação Local do Dataset

Este documento descreve de forma objetiva como a extração e o preparo do corpus foram realizados localmente utilizando o script `util_semclinbr.py`.

O corpus original "SemClinBr" (Oliveira et al., *J Biomed Semantics* 2022;13:13) distribui 1.000 notas clínicas em português como arquivos XML, um por documento, contendo o texto integral em `<TEXT>` e as anotações em `<TAGS>`/`<RELATIONS>` com offsets de caractere. O formato nativo não é consumível diretamente por um pipeline de extração estruturada com LLMs: os offsets não são previsíveis por um modelo autorregressivo, os ids das anotações são arbitrários, e não há divisão oficial de treino/teste.

Para converter o corpus em um dataset de extração estruturada e viabilizar a instanciação do framework CL+PT, o script `util_semclinbr.py` executa os seguintes passos (bloco `__main__`, executável com `python util_semclinbr.py`):

1. **Leitura dos XMLs originais:** `parse_semclinbr_xml` lê cada arquivo de `dados/SemClinBr-xml-public-v1`, recuperando o texto, as entidades (com `start`/`end`, `tag` e `abbr`) e as relações (`associated_with`, `negation_of`).

2. **Reancoragem dos offsets:** os offsets gravados no XML foram calculados sobre o texto original, em que cada quebra de linha ocupa dois caracteres (`\r\n`). A normalização de fim de linha é obrigatória para qualquer parser XML (XML 1.0 §2.11), então o texto lido chega com `\n` — um caractere a menos por quebra. `_mapa_offsets_crlf` converte os offsets para o espaço do texto normalizado. Sem essa conversão, todo span depois da primeira quebra fica deslocado, com o deslocamento crescendo a cada quebra anterior, e o F1 por span passa a medir ruído em vez de reconhecimento.

3. **Divisão em splits:** `definir_splits()` atribui 70% treino / 20% teste / 10% validação. A atribuição usa hash estável de `(seed, id)`, não a posição na lista ordenada, de modo que remover ou acrescentar documentos não reembaralha os demais. É estratificada por quartil de quantidade de entidades, para que as três partições tenham distribuição de complexidade comparável.

4. **Derivação do inventário de rótulos:** `inventario_tags(apenas_treino=True)` levanta os tipos semânticos (STY) presentes **somente no split de treino**, com a frequência de cada um. A ordem importa: splits primeiro, inventário depois. Derivar a lista do corpus inteiro vazaria para o prompt a existência de STYs que só ocorrem no teste — vazamento fraco, de metadado e não de rótulo por instância, mas gratuito de evitar.

5. **Criação do gabarito estruturado (target):** `xml_to_target_json` converte cada documento no JSON que a LLM deve produzir — entidades reordenadas por `(start, end)` e reindexadas em `1..n`, sem offsets. O campo `text` é reescrito a partir do offset, e não copiado do atributo `text` do XML: o atributo vem tokenizado pela ferramenta de anotação (`"35 , 7ºC"`, `"MÉDIA QUANTIDADE ."`, espaços duplos colapsados) e não ocorre literalmente na nota. Treinar contra ele contradiria a instrução do prompt ("*spans exactly as they appear*") e quebraria o round-trip do alinhamento em cerca de 4,5% das entidades.

6. **Exportação final:** `exportar()` grava em `dados/` o dataset consolidado `semclinbr.parquet` (colunas `id`, `texto`, `split`, `resposta`, `prompt` e as extras estruturais), a divisão base no formato `id_arquivo,alvo` que o framework espera em `arquivo_referencia`, o `prompt_semclinbr.txt` com o inventário já injetado e o `inventario_semclinbr.csv` com `rotulo,frequencia_treino`. O prompt é gravado junto com os dados de propósito: o inventário é derivado do corpus, então sem esse arquivo o experimento não é reprodutível.

7. **Gabarito no formato do framework:** o mesmo bloco grava `saidas/saida_semclinbr_gold.parquet` com as colunas `chave` e `resposta`, que é o `modelo_base` dos passos 03 e 06 e o *gold dataset* dos treinamentos do passo 04. É o análogo do `saida_pubmed_prof.parquet` do experimento PubMed, com a diferença de que aqui o gabarito é a anotação humana do corpus, não a saída de um modelo professor.

## Números do corpus

| | |
|---|---|
| Documentos | 1.000 (treino 700 / teste 201 / validação 99) |
| Entidades | 45.508 |
| Relações | 11.458 |
| Entidades por documento (mín / mediana / máx) | 0 / 43 / 212 |
| Rótulos distintos no corpus | 89 |
| Rótulos no prompt (cobertura 0,95 do treino) | 84 |

## Janela de contexto

Medida com o tokenizador do `Qwen2.5-7B-Instruct` sobre os 1.000 documentos:

| | média | p50 | p95 | p99 | máx |
|---|---|---|---|---|---|
| template do prompt (fixo) | 614 | | | | |
| prompt + texto | 918 | 878 | 1.244 | 1.423 | **2.425** |
| resposta (gabarito) | 1.717 | 1.636 | 3.536 | 4.633 | **7.581** |
| **total** | 2.635 | 2.532 | 4.809 | 5.948 | **10.006** |

A entrada é curta; quem consome a janela é a **saída**, por causa das listas de entidades. Cobertura do total: 8.192 → 99,8%; **12.288 → 100%**; 16.384 → 100% sem ganho.

Daí a escolha de **12.288** em treino (`treinamento.max_seq_length`, e por etapa) e em inferência (`vllm.max_model_len`), com `geracao.max_tokens: 8192` — o maior gabarito tem 7.581 tokens, e 2.425 + 8.192 = 10.617 cabe na janela. Com apenas 1.000 documentos, descartar qualquer um por truncamento custaria caro no desenho pareado, e `filtrar_max_seq_length: true` nos YAMLs de treino deve reportar zero exclusões.

## Teto do alinhamento

O gabarito não carrega offsets, então a avaliação os recupera com `alinhar_entidades`, em cascata: busca exata → tolerante a espaços → fuzzy (0,90) → global → falha. Esse procedimento tem um teto próprio, medido submetendo o **próprio gabarito** como se fosse a predição de um protocolo:

| | `f1_strict` |
|---|---|
| média | 0,992 |
| mediana | 1,000 |
| p25 | 1,000 |
| mínimo | 0,800 |
| documentos com F1 exatamente 1,0 | 75,1% |

Ou seja, o alinhamento não introduz viés relevante: um protocolo perfeito chegaria a ~0,99, não a 1,00. Vale reexecutar essa medição se `alinhar_entidades` for alterado — é o controle que separa erro do modelo de erro do instrumento.
