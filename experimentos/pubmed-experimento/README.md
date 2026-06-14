# PubMed 20k RCT — Instanciação Secundária do Framework CL+PT

Experimento de validação de portabilidade do framework de Curriculum Learning com
escalonamento de capacidade (CL+PT), focado na extração de informações de textos.

---

## Fontes dos dados

| Fonte | URL |
|---|---|
| Repositório oficial (GitHub) | https://github.com/Franck-Dernoncourt/pubmed-rct |
| Kaggle (CSV pré-processado) | https://www.kaggle.com/datasets/matthewjansen/pubmed-200k-rtc |
| Referência do paper | Dernoncourt & Lee, IJCNLP 2017 — [ACL Anthology](https://aclanthology.org/I17-2052/) |
| API de metadados (NCBI) | https://www.ncbi.nlm.nih.gov/books/NBK25499/ |

> O link aponta para ... kaggle ...-rtc corretamente, mas o nome do arquivo é pubmed-rct

Utilizar a versão **PubMed_20k_RCT** (sem substituição de números) para o experimento.
A versão `_numbers_replaced_with_at_sign` foi criada para experimentos de robustez
e não é relevante para este contexto.

---

## Formato nativo do dataset

O repositório GitHub distribui arquivos `.txt` (não CSV). O formato bruto é:

```
###9813759          ← separador: ### + PMID (sem título)
OBJECTIVE This study evaluated an [...]
METHODS Participants were 42 men [...]
RESULTS Intervention group subjects [...]
CONCLUSIONS This study has shown [...]
```

O Kaggle disponibiliza uma conversão em CSV com as colunas:
`abstract_id | line_id | abstract_text | line_number | total_lines | target`

O `abstract_id` é o PMID do artigo no PubMed. **Título não existe no dataset** —
deve ser buscado via API do NCBI pelo PMID (ver seção de extração abaixo).

---

## Enriquecimento via API do NCBI (Biopython)

O pacote `biopython` fornece acesso à API E-utilities do NCBI via `Bio.Entrez`.

```bash
pip install biopython
```

```python
from Bio import Entrez

Entrez.email = "seu@email.com"  # obrigatório pela política do NCBI

def enrich_abstract(pmid: str) -> dict:
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
    record = Entrez.read(handle)["PubmedArticle"][0]
    article = record["MedlineCitation"]["Article"]

    titulo = str(article.get("ArticleTitle", ""))

    journal = article.get("Journal", {}).get("Title", "")

    pub_date = article.get("Journal", {}) \
                      .get("JournalIssue", {}) \
                      .get("PubDate", {})
    data_pub = f"{pub_date.get('Year', '')}-{pub_date.get('Month', '')}".strip("-")

    keywords = [
        str(kw)
        for kw_list in record["MedlineCitation"].get("KeywordList", [])
        for kw in kw_list
    ]

    return {
        "pmid":            pmid,
        "titulo":          titulo,
        "journal":         journal,
        "data_publicacao": data_pub,
        "palavras_chave":  keywords,
    }
```

**Limite de taxa da API:** até 3 requisições/segundo sem chave; até 10/segundo com
chave NCBI (gratuita em https://www.ncbi.nlm.nih.gov/account/). Para processar
os ~20k abstracts, usar chave e adicionar `time.sleep(0.15)` entre chamadas.
Recomendado fazer o enriquecimento em lote e salvar em cache local (JSON ou Parquet)
para evitar re-consultas.

---

## Reconstrução do documento de entrada

O abstract reconstituído para o prompt segue estrutura padronizada — análoga ao
cabeçalho + corpo + rodapé dos acórdãos do STJ:

```
{titulo}

{abstract_text_concatenado_por_line_number}

{journal} | Published: {data_publicacao}
Keywords: {palavras_chave separadas por vírgula}
```

Esta reconstrução é inteiramente determinística — não envolve geração sintética.
Todos os campos vêm de anotações humanas (autores do artigo e indexadores do PubMed).

---

## Esquema JSON de saída (gabarito)

```json
{
  "titulo":          "string — extraído do cabeçalho",
  "data_publicacao": "string — formato YYYY-MM",
  "journal":         "string — nome do periódico",
  "palavras_chave":  ["lista", "de", "strings"],
  "background":      "string — seção BACKGROUND concatenada",
  "objective":       "string — seção OBJECTIVE concatenada",
  "methods":         "string — seção METHODS concatenada",
  "results":         "string — seção RESULTS concatenada",
  "conclusions":     "string — seção CONCLUSIONS concatenada"
}
```

### Heterogeneidade dos campos

| Campo | Tipo de tarefa | Análogo no SUMMA |
|---|---|---|
| `titulo` | extração exata do cabeçalho | `Materia` |
| `data_publicacao`, `journal` | extração de metadado pontual | `DataJulgamento`, `Tipo` |
| `palavras_chave` | extração de lista | `Normas`, `Jurisprudencia` |
| `background`, `objective` | extração de seção curta e conceitual | `Temas.Ponto`, `Temas.Decisao` |
| `methods`, `results` | extração de seção longa e densa em dados | `Temas.Argumentos`, `Temas.Explicacao` |
| `conclusions` | síntese interpretativa | `Resumo` |

A heterogeneidade dos campos justifica o Princípio 4 do framework (extração
simultânea de múltiplos campos em uma única inferência via saída estruturada).

---

## Filtragem do corpus

Aplicar os filtros abaixo antes de montar os splits de treino/validação/teste:

- **Reter** apenas abstracts com as 5 seções presentes e separadas
  (`BACKGROUND`, `OBJECTIVE`, `METHODS`, `RESULTS`, `CONCLUSIONS`).
- **Descartar** abstracts com seções fundidas (ex.: `METHODS AND RESULTS`) —
  ocorrem quando o journal usa headings não padronizados. A estrutura fundida
  impede o mapeamento determinístico para o esquema JSON.
- **Descartar** abstracts com `total_lines < 5` (instâncias degeneradas).
- Verificar se o PMID retorna metadados válidos na API antes de incluir no corpus.

Após filtragem, o corpus esperado é de aproximadamente **15–17k abstracts**
para treino, com splits de dev e test já definidos oficialmente no dataset.

---

## Processamento e Geração do Parquet

O script `util_pubmed.py` realiza o processamento do dataset original para o formato necessário para validação do framework. O processo envolve as seguintes etapas:

1. **Leitura dos CSVs originais**: Carrega as divisões oficiais do PubMed (`train`, `dev`, `test`).
2. **Enriquecimento (API NCBI)**: Busca em lote os metadados bibliográficos faltantes (`titulo`, `journal`, `data_publicacao`, `palavras_chave`) usando a API e-utilities, com cache local na pasta `.cache_pubmed` para evitar repetições.
3. **Reconstrução do Artigo**: Agrupa as sentenças por `abstract_id`, concatenando o texto e integrando os metadados (título no topo, texto corrido formatado, jornal, data e palavras-chave).
4. **Geração do Gabarito (JSON)**: Constrói a estrutura de dicionário de `response` com os campos extraídos deterministicamente, baseando-se nas marcações (target) do CSV.
5. **Exportação**: Gera o arquivo final `./dados/pubmed-rct-20k.parquet` contendo as colunas essenciais para treinamento e avaliação: `pmid`, `article`, `split` e `response`. Também gera listas de metadados (`lista_treino.csv`, `lista_validacao.csv`, `lista_teste.csv`) auxiliares para processamento de grupos de complexidade.

Este arquivo parquet padronizado servirá como base de dados validada consumida diretamente pelos scripts de Curriculum Learning.

---

## Estratificação de dificuldade (proxy $S_i$)

Mesma fórmula da instanciação principal (cap. 5 da dissertação), adaptada:

$$S_i = \sum_{m \in \mathcal{M}} F1_{i,m} - \hat{n}_i$$

onde $\hat{n}_i$ é o número total de linhas do abstract normalizado
($\hat{n}_i = (n_i - n_{min}) / (n_{max} - n_{min})$), disponível
diretamente na coluna `total_lines` do CSV.

Proxies de complexidade estrutural disponíveis nativamente:

| Proxy | Fonte | Direção |
|---|---|---|
| `total_lines` | coluna do CSV | mais linhas → mais difícil |
| Nº de seções presentes | contagem de `target` distintos | menos seções → mais fácil |
| Comprimento médio de `methods` + `results` | calculado | maior → mais difícil |
| F₁ do modelo base (Qwen 1.5B zero-shot) | inferência inicial | menor F₁ → mais difícil |

Partição por percentil idêntica à instanciação principal:
$P_{30}$ (Difícil, 30%) / $P_{30}$–$P_{70}$ (Médio, 40%) / $P_{70}$ (Fácil, 30%).

---

## Protocolos e modelo-alvo

Para a validação de portabilidade, rodar apenas os protocolos:

- **Protocolo A** — Qwen 2.5 1.5B Instruct, zero-shot (baseline sem treino)
- **Protocolo B** — Qwen 2.5 1.5B Instruct, QLoRA direto (sem currículo)
- **Protocolo D-best** — Qwen 2.5 1.5B Instruct, CL+PT na direção eleita como
  melhor na instanciação principal

O modelo 1.5B foi escolhido porque o domínio biomédico em inglês é genuinamente
difícil para modelos pequenos sem fine-tuning — garantindo espaço real de ganho
para o CL demonstrar efeito.

---

## Métricas de avaliação

Sem LLM-as-a-Judge nesta instanciação (gabarito determinístico, sem necessidade
de âncora humana). Métricas automáticas aplicadas campo a campo:

- **ROUGE-L** — campos de seção (background, methods, results, conclusions)
- **BERTScore F₁** — todos os campos textuais
- **Exact Match** — `data_publicacao`, `journal`
- **F1 de lista** — `palavras_chave` (comparação de conjuntos)

Análise estatística: Wilcoxon signed-rank bilateral A vs. B, A vs. D-best,
B vs. D-best. Tamanho de efeito $r = |z| / \sqrt{n}$.


## Alguns datasets considerados para um experimento com o framework

- **RAMS v1.0 (Ebner et al., 2020)**: Principal candidato para validação externa. Possui ~9.6k instâncias, 139 tipos de evento e 65 papéis, com gabarito 100% humano e argumentos *cross-sentence*.
- **PHEE (Sun et al., 2022)**: Excelente candidato complementar por apresentar um *schema* hierárquico em 2 níveis, que é estruturalmente muito análogo ao SUMMA.
- **CASIE (Satyapanich et al., 2020)**: Candidato secundário com foco no domínio de cibersegurança, descartado como validação principal devido ao volume menor que o RAMS.

## Alguns datasets analisados mas descartados e o motivo por não se adequarem ao framework

- **CoNLL-2003 (NER), BBC News Summary e Greene/UCD**: Descartados por possuírem um teto de desempenho já muito alto (*baseline* zero-shot muito forte), deixando pouco espaço para demonstrar o ganho diferencial do aprendizado curricular.
- **NestedClinBr, MLEE e WikiEvents**: Volume insuficiente de dados de treino para viabilizar os estágios curriculares e garantir testes estatísticos com poder adequado. (O *NestedClinBr* foi mantido apenas para estudos qualitativos).
- **EEMT e versões alternativas do RAMS**: O gabarito não é 100% humano (gerado sinteticamente por LLMs) ou os dados/argumentos encontram-se incompletos (e.g., `rams-short-generated-dataset`).
- **Few-NERD e CORD v2 text-only**: Formato de saída incompatível (nativamente BIO/BILOU com schema muito dinâmico) ou derivações informais sem *baselines* validados em cenários *text-only*.
- **n2c2 SDOH 2022**: Restrição de acesso que exige processo formal de credenciamento e aprovação institucional.
- **DAIC-WOZ, DocRED e MAVEN-Arg**: Escopo da tarefa divergente (classificação, extração de relações - RE) ou excessiva variabilidade do schema por instância (e.g., MAVEN-Arg possui mais de 600 papéis distintos, complicando a avaliação por campo).
- **OATS/ROAST (ABSA)**: A adequação exigiria geração sintética para completar alguns campos do gabarito, conflitando com a premissa de um *dataset* com avaliação 100% determinística e humana neste experimento.