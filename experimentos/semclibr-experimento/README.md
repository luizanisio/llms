# SemClinBr — prompt de extração e desenho de avaliação

Instanciação do framework CL+PT sobre o corpus **SemClinBr**
(Oliveira et al., *J Biomed Semantics* 2022;13:13 — 1.000 notas clínicas pt-br,
65.129 entidades, 11.263 relações, 100 STYs do UMLS + `Abbreviation` + `Negation`,
2 RTYs: `associated_with` e `negation_of`).

Experimento **isolado**: avaliação própria, comparação interna entre protocolos e
comparação externa contra os sistemas publicados sobre o mesmo corpus. 

---

## 1. Prompt

```text
Extract the structured clinical annotations from the clinical note below in tag <NOTE></NOTE>.

Use only the following semantic types as tags:
Body Location or Region; Body Part, Organ, or Organ Component; Organic Chemical;
Pharmacologic Substance; Quantitative Concept; Qualitative Concept; Temporal Concept;
Intellectual Product; Medical Device; Drug Delivery Device; Disease or Syndrome;
Finding; Injury or Poisoning; Sign or Symptom; Patient or Disabled Group;
Professional or Occupational Group; Population Group; Health Care Related Organization;
Laboratory or Test Result; Clinical Attribute; Diagnostic Procedure; Health Care Activity;
Therapeutic or Preventive Procedure; Abbreviation; Negation.

Return a valid JSON object matching the following schema exactly:
{
  "entities": [
    {
      "id": "integer — sequential, starting at 1, following the order of appearance in the text",
      "text": "string — the span exactly as it appears in the text",
      "tag": "string — semantic type of the span; multiple types separated by |",
      "abbr": "string — expanded form when the span is an abbreviation"
    }
  ],
  "relations": [
    {
      "annotation1": "integer — id of the source entity",
      "annotation2": "integer — id of the target entity",
      "reltype": "string — associated_with or negation_of"
    }
  ]
}

If no entity or relation is present, return an empty list. If a field does not apply, return an empty string. Do not hallucinate information.
Make sure to extract the spans exactly as they appear in the original text, preserving spelling, accentuation and casing, and to list them in the order in which they appear in the text.

<NOTE>
<<--TEXTO-->>
</NOTE>
```

### Por que o inventário de rótulos está presente

Os sistemas com que se compara — CRF de Souza et al. e BioBERTpt de Schneider
et al. — são *sequence labelers* supervisionados: conhecem o conjunto de rótulos
por construção. Um protocolo que precisasse adivinhar o inventário produziria um
número baixo por um motivo alheio à capacidade do modelo, inviabilizando a
comparação externa. Fornecer o vocabulário iguala essa condição.

**Escopo da lista.** Os 25 rótulos acima cobrem a maior parte das anotações.
O prompt real vai usar todos os rótulos do split de treino.

### Idioma

Instrução em inglês. O vocabulário de saída é integralmente inglês (nomes de STY
do UMLS, `associated_with`, `negation_of`); traduzir só a moldura criaria mistura
sem ganho. 

---

## 2. Decisões de projeto

| Questão | Decisão | Justificativa |
|---|---|---|
| Offsets (`start`/`end`) | **Fora do JSON**; resolvidos por pós-processamento determinístico | LLM autorregressiva não conta caracteres de forma confiável; a contagem viraria ruído de métrica, não sinal de extração |
| `id` das anotações | LLM gera **1..n** em ordem de aparição | Viabiliza o alinhamento por cursor; os ids originais (1259, 1260…) são arbitrários e não avaliáveis |
| Nomes dos campos | `tag`, `abbr`, `annotation1`, `annotation2`, `reltype` | Atributos nativos do XML — a conversão JSON→XML fica quase identidade |
| `tag` multi-rótulo | **String com `\|`**, como no XML | Evita divergência de formato entre gabarito e alvo |
| Canonicalização | `text` reescrito a partir do offset após alinhar | O span é a autoridade, não a cópia do modelo; garante round-trip exato |

---

## 3. Preparação do dataset (`CorpusSemClinBr`)

```python
from util_semclinbr import CorpusSemClinBr

corpus = CorpusSemClinBr("dados/semclinbr_xml", seed=42)
corpus.definir_splits()                  # 70 / 20 / 10, estratificado
corpus.inventario_tags(cobertura=0.95)   # rótulos derivados SÓ do treino
corpus.exportar("dados/")
```

### Dataset gerado

| Coluna | Conteúdo |
|---|---|
| `id` | nome do arquivo XML, sem extensão |
| `texto` | conteúdo de `<TEXT>` |
| `split` | `treino` (70%) / `teste` (20%) / `validacao` (10%) |
| `resposta` | gabarito JSON serializado |
| *extras* | `n_entidades`, `n_relacoes`, `n_rotulos_distintos`, `n_multirotulo`, `n_chars`, `n_tags_fora_do_prompt` |

As colunas extras alimentam o componente estrutural do proxy $S_i$ e podem ser
descartadas se não forem usadas.

### Arquivos de saída

| Arquivo | Papel |
|---|---|
| `semclinbr.parquet` (ou `.csv`) | o dataset acima |
| `divisao_Gold_Qwen7B.csv` | `id`, `alvo`, `dificuldade` — formato esperado pelo framework em `arquivo_referencia` |
| `prompt_semclinbr.txt` | prompt com o inventário já injetado |
| `inventario_semclinbr.csv` | `rotulo`, `frequencia_treino` |

O prompt é gravado junto com os dados de propósito: **o inventário é derivado do
corpus**, então sem esse arquivo o experimento não é reprodutível. 

### Splits

Determinísticos e **estáveis sob mudança do conjunto de arquivos**: a atribuição
usa hash de `(seed, id)`, não a posição na lista ordenada. Remover ou adicionar
documentos não reembaralha os demais — relevante se o corpus chegar em lotes ou
se algum XML for descartado por defeito de parsing.

Estratificados por quartil de quantidade de entidades, para que as três
partições tenham distribuição de complexidade comparável. Sem isso, o proxy
$S_i$ poderia estar medindo diferença entre partições em vez de dificuldade.
Alocação por maior resto, para que o arredondamento por estrato não desloque as
proporções globais.

**Divisão de Dificuldade:** A divisão de dificuldade é realizada no momento da comparação, configurada através do arquivo `03_compara_gold_full.yaml`. O script de comparação calcula a dificuldade de cada documento e faz o cruzamento (merge) utilizando o arquivo de referência (`dados/divisao_Gold_Qwen7B.csv`). Isso garante que a divisão original, já sorteada e estratificada, seja estritamente preservada. Sem essa referência explícita, o cálculo de dificuldade poderia reembaralhar as partições de treino e teste, causando vazamento de dados (data leakage).

---

## 4. Pipeline

```
XML original ──parse_semclinbr_xml──> Documento (com offsets)
                                        │
                                        ├─> xml_to_target_json ──> gabarito (ids 1..n, sem offsets)
                                        │                              │
                                        │                        prompt + gabarito
                                        │                              │
                                        │                         [SFT / inferência]
                                        │                              ▼
                                        │                     avaliar_documento(...)
                                        │                              │
                                        │            uma linha de métricas por (doc, protocolo, seed)
                                        │                              ▼
                                        │                    análise estatística pareada
                                        │
                                        └─> json_para_xml ──> saída no formato nativo do corpus
```

**Alinhamento.** A busca de cada entidade parte do `start` da anterior (não do
`end`), permitindo spans aninhados (`CURATIVO` dentro de `CURATIVO COM CARVÃO
ATIVADO`) sem quebrar a ordem. Cascata: exata → tolerante a espaços → fuzzy
(0,90) → global → falha.

**Duas taxas de robustez**, reportadas à parte para não virarem F1 zero
silencioso: *falha de parsing JSON* e *não-alinhamento* (span alucinado).

---

## 5. Comparação com o SemClinBr original

### 5.1 O que é comparável

| Sistema original | Métrica | Valor | Nossa métrica correspondente |
|---|---|---|---|
| Souza et al. (CRF) — Pharmacologic Substance | F1 exact | 0,84 | `avaliar_por_sty(modo="strict")` |
| Souza et al. (CRF) — Abbreviation | F1 exact | 0,71 | idem |
| Souza et al. (CRF) — SGR Disorder | F1 exact | 0,76 | `avaliar_por_sgr(modo="flexible")` |
| Souza et al. (CRF) — SGR Procedure | F1 exact | 0,70 | idem |
| BioBERTpt vs. CRF | Δ | +2,1 acc / +11,2 rec / +7,4 F1 | delta entre protocolos |
| Dalloux et al. — pista de negação | F1 | 92,63 | `avaliar_por_sty` no rótulo `Negation` |
| Dalloux et al. — escopo (parcial / exato) | F1 | 84,78 / 83,25 | `avaliar_relacoes` em `negation_of` |

### 5.2 O confundidor que precisa ser declarado

**Os números publicados não vêm do mesmo split.** Souza et al. usaram
"different fragments of our corpus and different annotation granularities";
o BioBERTpt tampouco documenta um split idêntico ao que você vai montar. Uma
tabela lado a lado é, portanto, **indicativa, não controlada** — e isso precisa
estar escrito na legenda, não escondido.

Duas formas de transformar isso em contraste real, em ordem de custo:

1. **Rodar o BioBERTpt no seu split de teste.** O modelo é público
   (`pucpr/biobertpt-clin` / `pucpr/biobertpt-all` no HuggingFace); um fine-tune
   de NER sobre o seu split de treino custa minutos de GPU e produz o baseline
   supervisionado **na mesma partição**, avaliado pelas mesmas quatro métricas.
   Isso converte a comparação externa em contraste pareado legítimo, elegível
   para Wilcoxon junto com os demais protocolos.
2. **Reimplementar o CRF** (sklearn-crfsuite, features do artigo de Souza et al.)
   como segundo baseline supervisionado no mesmo split. Mais barato ainda, e
   ancora o extremo inferior.

Recomendação: fazer (1). Com BioBERTpt-no-seu-split, a tabela de resultados passa
a ter um baseline de referência da literatura *dentro* do desenho pareado, e os
números publicados viram apenas contexto na discussão.

### 5.3 O IAA não é teto

Tentador usar IAA strict = 0,708 como teto de desempenho. O próprio artigo
desautoriza: cita Reidsma e Carletta ao afirmar que algoritmos de ML toleram
dados de baixa confiabilidade e que métricas de acordo são preditores fracos de
desempenho, e reproduz Roberts et al. — o IAA entre anotadores duplos não
fornece limite superior para o sistema, apenas indica quão difícil é a tarefa
de reconhecimento. Trate o IAA como **referência de dificuldade**, não como
teto; um protocolo que supere 0,708 em strict não está "acima do humano".

### 5.4 Segmentação gold/silver

O silver (387 docs, IAA ≤ 0,67) é ruído de rótulo conhecido. Decisão:

- **(a)** só gold no teste, gold+silver no treino — protege a validade da
  métrica final e ainda testa o CL sob ruído;
- **(b)** estratificar por gold/silver em todos os splits e reportar as faixas
  separadas — o contraste vira achado sobre robustez a ruído de rótulo.

Recomendação: **(a)** como principal, **(b)** como análise secundária.

---

## 6. Variável primária e análise inferencial

| Item | Definição |
|---|---|
| Unidade de análise | documento de teste (desenho pareado: todos os protocolos veem os mesmos documentos) |
| **Variável primária** | **F1 strict por documento** (`f1_strict`) |
| Complementares | lenient, flexible, relaxed, span exato/parcial, F1 de relações |
| Omnibus | Friedman sobre `f1_strict`, $k$ = nº de protocolos |
| Post-hoc | Wilcoxon signed-rank bilateral com Holm |
| Tamanho de efeito | $r = \lvert z \rvert / \sqrt{n}$ |
| Robustez | taxa de falha de parsing e taxa de não-alinhamento por protocolo |

Escolha do strict como primária: é a métrica mais exigente e a que menos depende
do mapeamento STY→SGR (que é uma escolha nossa, não do corpus). Declarar a
hierarquia antes de olhar os resultados; divergências entre strict e relaxed são
**achados** sobre onde o erro se concentra (fronteira de span vs. escolha de
rótulo), não inconsistências.

**ROPE (análise bayesiana).** a ROPE é ancorada na divergência entre
treinos distintos do protocolo D1.

---

## 7. Análise descritiva da performance dos protocolos

Template a repetir em cada experimento, para que os três capítulos de resultados
sejam lidos em paralelo mesmo sem contraste inferencial entre eles.

### 7.1 Tabela-âncora — desempenho por protocolo

Uma linha por protocolo (A, b, c, D1…, mais BioBERTpt-no-split se implementado):

| Coluna | Conteúdo |
|---|---|
| F1 strict | mediana [IQR] entre documentos |
| F1 lenient / flexible / relaxed | mediana [IQR] |
| Precisão / revocação strict | mediana | permite ver se o protocolo erra por omissão ou por excesso |
| F1 de relações | mediana [IQR] |
| Taxa de falha de parsing | % de documentos |
| Taxa de não-alinhamento | mediana entre documentos |
| Custo | horas de GPU, VRAM de pico |

Medianas e IQR, não médias — as distribuições de F1 por documento são assimétricas
e há massa em zero.

### 7.2 Desempenho por rótulo

O análogo, aqui, da "análise por campo" do SUMMA. Mediana de F1 por STY (25
rótulos) e por SGR (9 grupos), por protocolo. Duas leituras:

- **Perfil de erro**: quais rótulos o CL melhora e quais ele não toca. A Fig. 4
  do artigo dá a expectativa — `Pharmacologic Substance` e `Patient or Disabled
  Group` são fáceis (termos de token único, vocabulário pequeno); `Finding` e
  `Sign or Symptom` são difíceis (alta frequência, interpretações muito
  próximas). Se o seu ranking de dificuldade por rótulo reproduzir o do IAA
  humano, isso é evidência de que o modelo erra onde a tarefa é genuinamente
  ambígua, e não onde o treinamento falhou.
- **Comparação externa**: as quatro células com número publicado (Pharmacologic
  Substance, Abbreviation, SGR Disorder, SGR Procedure).

### 7.3 Desempenho por faixa de dificuldade

Repetir a tabela-âncora nos subconjuntos Fácil / Médio / Difícil do proxy $S_i$.
É onde o CL deveria aparecer, se aparecer: a hipótese do currículo prevê ganho
concentrado nas faixas difíceis. Ganho uniforme entre faixas é achado contra a
explicação curricular e a favor de um efeito genérico de mais treinamento.

### 7.4 Decomposição do erro

Diferença entre pares de métricas, por protocolo, para localizar o erro:

| Contraste | Interpreta |
|---|---|
| lenient − strict | erro de **fronteira de span** |
| flexible − strict | erro de **granularidade de rótulo** dentro do mesmo SGR |
| span exato − strict | quanto do erro é puramente de rotulagem, com span certo |
| relaxed − strict | erro total tolerável |

O artigo relata que o SemClinBr melhorou 16,9% de strict para lenient, contra
3,8–8,6% nos demais corpora — ou seja, os anotadores humanos tiveram dificuldade
específica com fronteiras de span. Se os seus protocolos reproduzirem um salto
dessa ordem, é a mesma dificuldade se manifestando; se não reproduzirem, vale
investigar se o alinhamento automático está normalizando fronteiras que os
humanos deixaram irregulares.

### 7.5 Viabilidade

Proporção de documentos com F1 strict acima de um piso declarado, com IC 95%
Wilson — o análogo do "mediana ≥ 3" do SUMMA. O piso precisa ser fixado *antes*
de ver os resultados; 0,70 é defensável por coincidir com o IAA strict do corpus
(referência de dificuldade, não teto — ver 5.3).

### 7.6 Leitura paralela entre experimentos

Sem teste inferencial cruzado. O que se compara é a **forma** dos resultados:
o ranking de protocolos se mantém nos três experimentos? O ganho do CL se
concentra na faixa difícil nos três? A ordem de escalonamento (FF→LoRA vs.
LoRA→FF) tem o mesmo sinal? Consistência de padrão em três domínios, três
idiomas e três esquemas de saída é evidência de generalidade do framework —
reportada como convergência descritiva, com a ressalva explícita de que não há
teste formal sustentando a comparação entre experimentos.

---

## 8. Pontos de atenção

1. **Direção de `negation_of`.** O artigo não fixa se `annotation1` é a pista ou
   o conceito negado. `auditar_direcao_relacoes()` resolve com um passe nos XMLs;
   ajuste as descrições no esquema se a convenção for a inversa.
2. **Tamanho do corpus.** 1.000 documentos, contra 22k do SUMMA e ~16k do PubMed.
   Com split 700/100/200 e 3 faixas, cada fase curricular fica com ~200
   instâncias — pouco para full fine-tuning, com alta variância entre seeds
   esperada. Planeje ≥3 seeds por protocolo e reporte desvio. Alternativa:
   currículo em granularidade de sentença, ao custo de quebrar as relações entre
   entidades de sentenças distintas.
3. **Acesso ao corpus.** Formulário de solicitação com termo de licença para uso
   científico e não comercial. Considere o prazo no cronograma.
4. **Sigilo.** A licença restringe redistribuição — não exponha texto integral em
   apêndices; use os exemplos publicados no artigo (Tabelas 1 e 4), CC BY 4.0.
5. **Tokenização.** Notas em CAIXA ALTA são comuns e fragmentam mais no
   tokenizador do Qwen. Se algum proxy de dificuldade usar contagem de tokens,
   isso vira viés; `n_chars` evita.
