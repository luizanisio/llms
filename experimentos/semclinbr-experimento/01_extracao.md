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

2. **Reancoragem dos offsets:** o corpus é inconsistente na convenção de offsets — parte dos documentos grava offsets no espaço CRLF (texto com `\r\n`), parte já usa o espaço LF (pós-normalização XML 1.0 §2.11). `_escolher_espaco_offsets` pontua **todas** as anotações do documento nos dois espaços e fica com o que validar mais contra o atributo `text`, aplicando o mapeamento CRLF→LF (`_mapa_crlf_para_lf`) só quando ele ganha; empate mantém LF. Resultado: **421 documentos em CRLF**, 579 em LF. A pontuação precisa percorrer todas as anotações e usar a mesma comparação tolerante do `_spans_compativeis`: anotações que caem antes da primeira quebra de linha validam nos dois espaços e não discriminam, e o atributo `text` cru não casa nos documentos com entidades XML duplamente escapadas (`&amp;gt;`).

3. **Passada única de ajuste de posição:** resolvido o espaço, `resolver_span` fixa a posição de cada entidade numa só passada, sempre validando contra o atributo `text` do XML e parando na primeira correção que funciona — da nula à maior: **(a)** `exata`, o offset já casa (45 061 · 99,02%); **(b)** `trim` de whitespace nas bordas (espaço, tab, `\n`, `\r`), preservando pontuação, que pode ser legítima em contexto clínico (0 casos isolados); **(c)** `shift`, deslocando a borda esquerda e/ou direita em até **`raio_ajuste` caracteres** (421 · 0,93%), dominado por `(−1, 0)` — a borda esquerda um caractere adiantada, `"ORADA"` anotado para `"CORADA"`. O raio é parametrizável em `parse_semclinbr_xml(..., raio_ajuste=2)`, que é o padrão; `1` cobre só o erro de um caractere e `0` desliga o deslocamento. O `trim` nunca resgata sozinho, porque a comparação já ignora whitespace, mas atua em 43 spans *depois* do deslocamento (`"IRC "` → `"IRC"`).

4. **Descarte de entidades irrecuperáveis:** esgotados os candidatos, a entidade é **descartada** — removida de `doc.entidades`. Restam **26 (0,06%)** em 19 docs, todas de documentos cujo texto foi editado após a anotação (ex: doc 8965, atributo `"TRAUMATISMOS NAO ESPECIFICADOS"` contra o span `"TRAUMATISMOS MULTIPLOS NAO ESPECIFICADOS"`). A função `_spans_compativeis` é tolerante à tokenização da ferramenta de anotação (ex: `"35 , 7ºC"` vs `"35,7ºC"`) e a entidades XML (`&gt;`, `&lt;`, `&amp;`). As relações que referenciam entidades descartadas são também removidas. **Preservadas: 45 482 de 45 508 (99,94%)** e 11 458 relações; toda entidade preservada satisfaz `_spans_compativeis(texto[start:end], text)`.

5. **Contabilização de qualidade:** cada `Documento` retornado por `parse_semclinbr_xml` inclui `stats_parse` com contadores (`n_exatas`, `n_corrigidas_trim`, `n_corrigidas_shift`, `n_trim_aplicado`, `n_descartadas`, `n_total_xml`, `espaco_offsets`) e as listas completas das entidades ajustadas e descartadas. Na exportação, um relatório `.md` com o mesmo nome base do parquet é gerado automaticamente (`dados/semclinbr.md`, `saidas/saida_semclinbr_gold.md`), com o resumo, a distribuição dos deslocamentos aplicados e as duas listas item a item.

6. **Divisão treino/teste/validação — fora deste script:** quem define os alvos é o passo 03 (`03_compara_gold_full.yaml`), que calcula a dificuldade de cada documento e grava `dados/divisao_Gold_Qwen7B.csv` com `id`, `alvo`, `dificuldade` e `dificuldade_int`. Esse arquivo é a fonte única da divisão: todos os passos seguintes se apoiam nele e apenas nos ids que constam nele (997 dos 1.000 documentos — ver os três não anotados no README §8).

7. **Derivação do inventário de rótulos:** `inventario_tags(arquivo_divisao="dados/divisao_Gold_Qwen7B.csv", alvo="treino")` levanta os tipos semânticos (STY) presentes **somente nos documentos de treino**, com a frequência de cada um. Derivar a lista do corpus inteiro vazaria para o prompt a existência de STYs que só ocorrem no teste — vazamento fraco, de metadado e não de rótulo por instância, mas gratuito de evitar. Como o passo 03 depende deste parquet, na primeira execução a divisão ainda não existe: o inventário sai do corpus inteiro (89 rótulos em vez de 85) e o script **avisa que é preciso rodar de novo** depois do 03, para que o prompt fique só com os rótulos de treino.

8. **Criação do gabarito estruturado (target):** `xml_to_target_json` converte cada documento no JSON que a LLM deve produzir — entidades reordenadas por `(start, end)` e reindexadas em `1..n`, sem offsets. O campo `text` é reescrito a partir do offset, e não copiado do atributo `text` do XML: o atributo vem tokenizado pela ferramenta de anotação (`"35 , 7ºC"`, `"MÉDIA QUANTIDADE ."`, espaços duplos colapsados) e não ocorre literalmente na nota. Treinar contra ele contradiria a instrução do prompt ("*spans exactly as they appear*") e quebraria o round-trip do alinhamento em cerca de 4,5% das entidades.

9. **Exportação final:** `exportar()` grava em `dados/` o dataset consolidado `semclinbr.parquet` (colunas `id`, `texto`, `resposta`, `prompt` e as extras estruturais; a divisão não é coluna do parquet, vem do arquivo do passo 03), o `prompt_semclinbr.txt` com o inventário já injetado e o `inventario_semclinbr.csv`, cujo cabeçalho declara a origem da frequência (`frequencia_treino` ou `frequencia_corpus`). Junto com o parquet, é gerado `semclinbr.md` — o relatório de qualidade das anotações. O prompt é gravado junto com os dados de propósito: o inventário é derivado do corpus, então sem esse arquivo o experimento não é reprodutível.

10. **Gabarito no formato do framework:** o mesmo bloco grava `saidas/saida_semclinbr_gold.parquet` com as colunas `chave` e `resposta`, que é o `modelo_base` dos passos 03 e 06 e o *gold dataset* dos treinamentos do passo 04. É o análogo do `saida_pubmed_prof.parquet` do experimento PubMed, com a diferença de que aqui o gabarito é a anotação humana do corpus, não a saída de um modelo professor. Também gera `saida_semclinbr_gold.md` com o relatório de qualidade.

## Números do corpus

| | |
|---|---|
| Documentos no corpus | 1.000 |
| Documentos na divisão do passo 03 | 997 (treino 697 / teste 197 / validação 103) |
| Anotações no XML original | 45.508 |
| Entidades preservadas | 45.482 (99,94%) |
| — offset já exato | 45.061 (99,02%) |
| — corrigidas por deslocamento de borda | 421 (0,93%) |
| Entidades descartadas (offset irrecuperável) | 26 (0,06%) |
| Relações | 11.458 |
| Entidades por documento (mín / mediana / máx) | 0 / 43 / 212 |
| Rótulos distintos no corpus | 89 |
| Rótulos no prompt (derivados do treino) | 85 |

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
| média | 0,994 |
| mediana | 1,000 |
| p25 | 1,000 |
| mínimo | 0,800 |
| documentos com F1 exatamente 1,0 | 81,7% |

Ou seja, o alinhamento não introduz viés relevante: um protocolo perfeito chegaria a ~0,99, não a 1,00. Vale reexecutar essa medição sempre que `alinhar_entidades` ou o parse dos offsets for alterado — é o controle que separa erro do modelo de erro do instrumento.
