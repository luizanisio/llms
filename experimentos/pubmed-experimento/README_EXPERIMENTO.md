# PubMed 20k RCT — Instanciação Secundária do Framework CL+PT

Experimento de validação de portabilidade do framework de Curriculum Learning com escalonamento de capacidade (CL+PT), focado na extração de informações de textos.

Diferente do SUMMA, o prompt desse experimento é mais simples, com menos detalhes do que o modelo deve fazer e como deve fazer, então espera-se que as saídas dos modelos sem fine-tuning sejam mais distantes do esperado.

---

## 🚀 Roteiro de Replicação (Passo a Passo)

Para replicar o experimento (exemplo para o protocolo D1 / treinamento do modelo 1.5B), siga os seguintes passos:

### 1. Configuração do Ambiente
> ⚠️ **Antes de começar:** Certifique-se de que o ambiente Conda e as dependências (PyTorch, vLLM, Flash-Attention, Liger-Kernel) estão configurados corretamente.  
👉 **Leia as instruções completas no arquivo central: [experimentos/README.md](../README.md)**

### 2. Preparação do Dataset
Certifique-se de ter baixado o dataset `PubMed_20k_RCT` (ver seção "Fontes dos dados" mais abaixo). O script `util_pubmed.py` processa os originais, enriquece os metadados faltantes (buscando via API do NCBI) e gera os parquets consolidados para o treinamento.
```bash
python util_pubmed.py
```
Isso gerará os arquivos prontos na pasta `dados/` (ex: `pubmed-rct-20k.parquet`).

### 3. Extração de Dados com Modelos Base (Baseline)
Antes de treinar, extraímos os dados usando o modelo original (zero-shot, Protocolo A) para gerar o baseline de comparação. O script de job submete a inferência no HPC utilizando a configuração do arquivo YAML.
```bash
sbatch job_export_1_5b.sh
# Internamente chama o util_vllm_batch.py com a config 02_pubmed_rct_1_5b.yaml
```
Também é possível rodar diretamente pela linha de comando em uma máquina com GPU. Ajuste os caminhos no arquivo yaml se necessário.

### 4. Treinamento
Os diferentes protocolos de treinamento (B, C, D1, D2, etc.) estão mapeados em arquivos YAML específicos. Submeta o job de treinamento desejado para o gerenciador Slurm.
*Exemplo para treinar o modelo do protocolo D1 (que internamente aponta para `04_treinar_d1.yaml`):*
```bash
sbatch job_treinar_d12.sh
```
*Dica: O script `job_treinar_*.sh` aloca a GPU e inicia o processo `treinar_unsloth.py`. Os logs podem ser acompanhados na pasta `saidas/`.*

### 5. Extração de Dados com Modelos Treinados
Após a conclusão do treinamento, os modelos ajustados (LoRA) estarão salvos em `treinos/`. Para extrair as informações do conjunto de teste com esses pesos treinados, usamos as configurações `05_extracao_*_teste.yaml`.
```bash
# Para enviar a fila do cluster Slurm:
sbatch job_export_1_5b_testes.sh

# Ou para execução manual interativa (ex: usando tmux ou diretamente na linha de comando):
bash manual_export_1_5b_testes.sh
```

### 6. Geração das Comparações dos Resultados
Por fim, comparamos as extrações geradas contra o gabarito. O processo lê os arquivos parquet de saída, aplica métricas automáticas (ROUGE-L, BERTScore, Exact Match, etc.) e compila as tabelas de resultados através do `06_compara_testes.yaml`.
```bash
sbatch job_compara_testes.sh
```

---

## 📚 Contexto Teórico e Metodologia

### Fontes dos dados

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

### Formato nativo do dataset

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
deve ser buscado via API do NCBI pelo PMID.

### Enriquecimento via API do NCBI (Biopython)

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

### Reconstrução do documento de entrada

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

### Esquema JSON de saída (gabarito)

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

#### Heterogeneidade dos campos

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

### Filtragem do corpus

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

### Estratificação de dificuldade (proxy $S_i$)

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

### Protocolos e modelo-alvo

O modelo-alvo é o **Qwen 2.5 1.5B Instruct**, escolhido porque o domínio biomédico
em inglês é genuinamente difícil para modelos pequenos sem fine-tuning — garantindo
espaço real de ganho para o CL demonstrar efeito.

O experimento conta com **13 protocolos** organizados em 5 camadas:

#### Perguntas de pesquisa

| Pergunta | Descrição |
|---|---|
| **Q1** | Efeito do ajuste fino: FT (qualquer variante) produz ganho sobre baseline zero-shot? |
| **Q2** | Efeito do CL: a progressão de dificuldade melhora sobre FT direto? |
| **Q3** | Direção do escalonamento: FF→LoRA vs LoRA→FF produz desempenhos distintos? |
| **Q4** | Direção do currículo: a ordem fácil→difícil importa vs difícil→fácil? |

#### Camada 1 — Baselines (sem CL, sem escalonamento)

| Proto | Modo | Etapas | Arquivo treino |
|---|---|---|---|
| **A** | Zero-shot (sem treino) | — | — |
| **b** | LoRA direto (dataset completo) | LoRA-completo | `04_treinar_b.yaml` |
| **c** | FF direto (dataset completo) | FF-completo | `04_treinar_c.yaml` |

#### Camada 2 — Experimentais (CL + escalonamento de capacidade)

| Proto | Pace | Direção | Etapas | Arquivo treino |
|---|---|---|---|---|
| **D1** | etapas | FF→LoRA | FF-fácil → LoRA-médio → LoRA-difícil → LoRA-completo | `04_treinar_d1.yaml` |
| **D2** | etapas | LoRA→FF | LoRA-fácil → LoRA-médio → LoRA-difícil → FF-completo | `04_treinar_d2.yaml` |
| **D3** | acumulado | FF→LoRA | FF-fácil → LoRA-(fácil+médio) → LoRA-tudo | `04_treinar_d3.yaml` |
| **D4** | acumulado | LoRA→FF | LoRA-fácil → LoRA-(fácil+médio) → FF-tudo | `04_treinar_d4.yaml` |

#### Camada 3 — Ablação: escalonamento sem CL

| Proto | Modo | Etapas | Arquivo treino |
|---|---|---|---|
| **D5** | FF→LoRA, sem progressão | FF-completo → LoRA-completo | `04_treinar_d5.yaml` |
| **D6** | LoRA→FF, sem progressão | LoRA-completo → FF-completo | `04_treinar_d6.yaml` |

#### Camada 4 — Ablação: CL sem escalonamento (LoRA-only)

| Proto | Pace | Etapas | Arquivo treino |
|---|---|---|---|
| **D7** | etapas | LoRA-fácil → LoRA-médio → LoRA-difícil → LoRA-completo | `04_treinar_d7.yaml` |
| **D8** | acumulado | LoRA-fácil → LoRA-(fácil+médio) → LoRA-tudo | `04_treinar_d8.yaml` |

#### Camada 5 — Ablação: anti-currículo (direção inversa, LoRA-only)

| Proto | Pace | Etapas | Arquivo treino |
|---|---|---|---|
| **D9** | etapas | LoRA-completo → LoRA-difícil → LoRA-médio → LoRA-fácil | `04_treinar_d9.yaml` |
| **D10** | acumulado | LoRA-(>7) → LoRA-(>3) → LoRA-tudo | `04_treinar_d10.yaml` |

#### Design fatorial

O design forma um fatorial quase completo em 3 dimensões:

|  | Sem escal. | FF→LoRA | LoRA→FF |
|---|---|---|---|
| **Sem CL** | b, c | D5 | D6 |
| **CL por etapas** | D7 | D1 | D2 |
| **CL acumulado** | D8 | D3 | D4 |
| **Anti-CL etapas** | D9 | — | — |
| **Anti-CL acumulado** | D10 | — | — |

#### Matriz de comparações por pergunta

**Q1 — Efeito do ajuste fino:**

| Comparação | Interpretação |
|---|---|
| A vs b | Efeito do LoRA direto |
| A vs c | Efeito do FF direto |
| A vs D1/D2 | Efeito do melhor protocolo CL |

**Q2 — Efeito do CL (decomposição):**

| Comparação | O que isola | Interpretação |
|---|---|---|
| b vs D7 | CL por etapas (LoRA) | CL puro melhora sobre FT direto? |
| b vs D8 | CL acumulado (LoRA) | CL acumulado melhora sobre FT direto? |
| b vs D1 | CL + escalonamento | Pacote completo vs FT direto |
| D7 vs D1 | Escalonamento sobre CL | O escalonamento adiciona valor ao CL? |
| D7 vs D8 | Pace por etapas vs acumulado | Qual pace funciona melhor? |

**Q3 — Direção do escalonamento:**

| Comparação | Contexto |
|---|---|
| D1 vs D2 | Com CL por etapas |
| D3 vs D4 | Com CL acumulado |
| D5 vs D6 | Sem CL (controle) |

**Q4 — Direção do currículo (anti-CL):**

| Comparação | O que isola | Interpretação |
|---|---|---|
| D7 vs D9 | Ordem no pacing por etapas | Fácil→difícil vs difícil→fácil |
| D8 vs D10 | Ordem no pacing acumulado | Fácil→difícil vs difícil→fácil |
| D9 vs b | Anti-CL vs baseline | Anti-CL pelo menos melhora sobre FT direto? |
| D10 vs b | Anti-CL acum vs baseline | Anti-CL acum melhora sobre FT direto? |

#### Arquivos de comparação

| Arquivo | Modelos incluídos | Propósito |
|---|---|---|
| `06_compara_experimentais.yaml` | A, b, c, D1, D2, D3, D4 | Q1 + Q2 + Q3 (experimento principal) |
| `06_compara_ablacoes.yaml` | A, b, c, D5, D6, D7, D8 | Decomposição CL vs escalonamento |
| `06_compara_ordem_cl.yaml` | A, b, D7, D8, D9, D10 | Q4 (efeito da direção do currículo) |
| `06_compara_ordem_pt.yaml` | A, b, c, D5, D6 | Q3 (efeito da direção do escalonamento) |
| `06_compara_todos.yaml` | A, b, c, D1–D10 | Panorama completo |

#### Controle de volume de treinamento entre estratégias de pacing

Todos os protocolos utilizam `pace_epochs=2` por estágio, resultando em volume total
de treinamento equivalente entre as duas estratégias de pacing:

| Protocolo | Estágios | Cálculo (% dataset × epochs) | Total |
|---|---|---|---|
| **b** (baseline) | 1 | 100% × 4ep | **4.0** dataset-eq |
| **D1** (etapas) | 4 | 30%×2 + 40%×2 + 30%×2 + 100%×2 | **4.0** dataset-eq |
| **D3** (acumulado) | 3 | 30%×2 + 70%×2 + 100%×2 | **4.0** dataset-eq |

Embora o volume total de tokens seja idêntico (4.0 dataset-equivalentes), a
**exposição por item** difere entre as estratégias:

| Dificuldade | Baseline b | Etapas (D1) | Acumulado (D3) |
|---|---|---|---|
| Fácil (30%) | 4 épocas | 4 épocas (2+2) | 6 épocas (2+2+2) |
| Médio (40%) | 4 épocas | 4 épocas (2+2) | 4 épocas (2+2) |
| Difícil (30%) | 4 épocas | 4 épocas (2+2) | 2 épocas (2) |

Na estratégia por etapas, a fase de consolidação (estágio "completo") garante que
todos os itens recebam exposição uniforme. Na estratégia acumulada, a exposição é
naturalmente decrescente com a dificuldade: itens fáceis são revisitados em todos
os estágios (replay), enquanto itens difíceis aparecem apenas no estágio final.

Esta distribuição diferenciada é uma propriedade intrínseca do pacing acumulado,
análoga ao *spaced repetition* — onde itens já dominados recebem mais reforço ao
longo do treinamento. Equalizar artificialmente a exposição por item entre as
estratégias exigiria alterar os epochs por estágio, o que confundiria o efeito
do tipo de pacing com o efeito de mais treinamento. A decisão de manter
`pace_epochs=2` uniforme preserva a comparabilidade em volume total de tokens
e permite que qualquer diferença de desempenho entre D1 e D3 (ou D7 e D8) seja
atribuída exclusivamente à estratégia de pacing.

### Métricas de avaliação

Sem LLM-as-a-Judge nesta instanciação (gabarito determinístico, sem necessidade
de âncora humana). Métricas automáticas aplicadas campo a campo:

- **ROUGE-L** — campos de seção (background, methods, results, conclusions)
- **BERTScore F₁** — todos os campos textuais
- **Exact Match** — `data_publicacao`, `journal`
- **F1 de lista** — `palavras_chave` (comparação de conjuntos)

Análise estatística: Wilcoxon signed-rank bilateral para todos os pares de
protocolos relevantes (A vs B, A vs D-best, B vs D1, D1 vs D2, etc.).
Tamanho de efeito $r = |z| / \sqrt{n}$.

### Alguns datasets considerados para um experimento com o framework

- **RAMS v1.0 (Ebner et al., 2020)**: Principal candidato para validação externa. Possui ~9.6k instâncias, 139 tipos de evento e 65 papéis, com gabarito 100% humano e argumentos *cross-sentence*.
- **PHEE (Sun et al., 2022)**: Excelente candidato complementar por apresentar um *schema* hierárquico em 2 níveis, que é estruturalmente muito análogo ao SUMMA.
- **CASIE (Satyapanich et al., 2020)**: Candidato secundário com foco no domínio de cibersegurança, descartado como validação principal devido ao volume menor que o RAMS.

### Alguns datasets analisados mas descartados e o motivo por não se adequarem ao framework

- **CoNLL-2003 (NER), BBC News Summary e Greene/UCD**: Descartados por possuírem um teto de desempenho já muito alto (*baseline* zero-shot muito forte), deixando pouco espaço para demonstrar o ganho diferencial do aprendizado curricular.
- **NestedClinBr, MLEE e WikiEvents**: Volume insuficiente de dados de treino para viabilizar os estágios curriculares e garantir testes estatísticos com poder adequado. (O *NestedClinBr* foi mantido apenas para estudos qualitativos).
- **EEMT e versões alternativas do RAMS**: O gabarito não é 100% humano (gerado sinteticamente por LLMs) ou os dados/argumentos encontram-se incompletos (e.g., `rams-short-generated-dataset`).
- **Few-NERD e CORD v2 text-only**: Formato de saída incompatível (nativamente BIO/BILOU com schema muito dinâmico) ou derivações informais sem *baselines* validados em cenários *text-only*.
- **n2c2 SDOH 2022**: Restrição de acesso que exige processo formal de credenciamento e aprovação institucional.
- **DAIC-WOZ, DocRED e MAVEN-Arg**: Escopo da tarefa divergente (classificação, extração de relações - RE) ou excessiva variabilidade do schema por instância (e.g., MAVEN-Arg possui mais de 600 papéis distintos, complicando a avaliação por campo).
- **OATS/ROAST (ABSA)**: A adequação exigiria geração sintética para completar alguns campos do gabarito, conflitando com a premissa de um *dataset* com avaliação 100% determinística e humana neste experimento.