# SemClinBr — prompt de extração e desenho de avaliação

Instanciação do framework CL+PT sobre o corpus **SemClinBr**
(Oliveira et al., *J Biomed Semantics* 2022;13:13 — 1.000 notas clínicas pt-br,
65.129 entidades, 11.263 relações, 100 STYs do UMLS + `Abbreviation` + `Negation`,
2 RTYs: `associated_with` e `negation_of`).

> Os números acima são os do artigo. O release público (`SemClinBr-xml-public-v1`)
> traz **45 508 anotações** nos 1.000 XMLs — é sobre esse total que as taxas de
> §2.1 são calculadas.

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
O prompt real (`dados/prompt_semclinbr.txt`) usa o inventário derivado do split
de treino: **84 rótulos** (`dados/inventario_semclinbr.csv`).

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

### 2.1. Limpeza de offsets do corpus

Os offsets de anotação do SemClinBr são inconsistentes. `parse_semclinbr_xml`
os corrige em **duas etapas**: primeiro decide em que espaço o documento gravou
os offsets, depois fixa a posição de cada entidade numa **passada única**
(`resolver_span`), sempre validando contra o atributo `text` do XML.

**Etapa 1 — espaço de offsets (por documento).** Parte do corpus grava offsets
sobre o texto CRLF original, parte sobre o texto LF (pós-normalização XML 1.0
§2.11). `_escolher_espaco_offsets` pontua **todas** as anotações do documento
nos dois espaços e fica com o que validar mais; empate mantém LF (conservador).
Resultado: **421 documentos em CRLF**, 579 em LF.

> Duas condições tornam a pontuação sensível e por isso ela percorre **todas**
> as anotações com a comparação tolerante de `_spans_compativeis`: anotações
> que caem antes da primeira quebra de linha validam nos dois espaços e não
> discriminam nada, e documentos com entidades XML duplamente escapadas
> (`&amp;gt;`) não casam contra o atributo `text` cru. Amostrar poucas
> anotações, ou comparar sem tolerância, escolhe LF indevidamente em 26
> documentos — 932 anotações perdidas.

**Etapa 2 — passada única de ajuste (`resolver_span`).** Para cada entidade,
da correção nula à maior, parando na primeira que valida:

| Ordem | Ajuste | O que corrige | Entidades |
|---|---|---|---|
| 1 | `exata` | offset já correto | 45 061 (99,02%) |
| 2 | `trim` | whitespace nas bordas do span | 0 (0,00%) |
| 3 | `shift` | borda deslocada em até **`raio_ajuste` caracteres** para a esquerda e/ou direita | 421 (0,93%) |
| 4 | `descartada` | nenhum candidato valida — entidade removida | 26 (0,06%) |

**Preservadas: 45 482 de 45 508 (99,94%)**, com `raio_ajuste=2` (o padrão).
As relações que referenciam entidades descartadas também são removidas
(11 458 relações preservadas).

O deslocamento é dominado por um padrão único: **387 casos de `(−1, 0)`**, a
borda esquerda um caractere adiantada (`"ORADA"` anotado para `"CORADA"`), mais
18 de `(−1, −1)`; o raio 2 acrescenta 16 casos (9 de `(−2, 0)`, 3 de `(−2, −2)`,
3 de `(−2, −1)`, 1 de `(−1, −2)`). O raio é parametrizável
(`parse_semclinbr_xml(..., raio_ajuste=2)`); `raio_ajuste=1` cobre só o erro de
um caractere e `0` desliga o ajuste. O relatório sempre imprime o raio efetivo.

O `trim` nunca resgata uma entidade sozinho (0 casos) porque a comparação com o
atributo `text` já ignora whitespace — mas ele **é** carregado: atua em 43 spans
*depois* do deslocamento, encostando a borda no token (`"IRC "` → `"IRC"`).

**Critério de validação e descarte:** o span resultante é comparado com o
atributo `text` do XML ignorando whitespace e entidades XML (`&gt;`, `&lt;`,
`&amp;`) — a ferramenta de anotação tokeniza o atributo ao salvar (`"35 , 7ºC"`
em vez de `"35,7ºC"`). As 26 descartadas restantes são irrecuperáveis de fato: o
texto foi editado após a anotação (ex: doc 8965, atributo `"TRAUMATISMOS NAO
ESPECIFICADOS"` contra o span `"TRAUMATISMOS MULTIPLOS NAO ESPECIFICADOS"`).

**Invariante garantida:** toda entidade preservada satisfaz
`_spans_compativeis(texto[start:end], text)`. O parquet de treinamento é gerado
a partir desses offsets já corrigidos — `xml_to_target_json` reescreve o campo
`text` a partir do span, então o gabarito carrega o texto real da nota.

**Relatório de qualidade:** cada exportação (parquet/CSV) gera um `.md` com o
mesmo nome base (`dados/semclinbr.md`, `saidas/saida_semclinbr_gold.md`) com o
resumo acima, a distribuição dos deslocamentos aplicados e a lista individual
das entidades ajustadas e descartadas.

### 2.2. Comparação com baseline externo (ClinicalNERpt)

Para contextualizar a performance dos nossos modelos (LLM generativa),
comparamos com os **modelos ClinicalNERpt** do HAILab-PUCPR
(`pucpr/clinicalnerpt-*` no HuggingFace), que são sequence labelers
(BertForTokenClassification) fine-tunados sobre o BioBERTpt com dados do
SemClinBr.

**Limitações importantes:**

1. **Cobertura parcial de STYs.** Cada `clinicalnerpt-*` cobre apenas **1 tipo
   de entidade** (ex: `MedicalDevice`, `DiseaseOrSyndrome`). Os 6 modelos
   disponíveis cobrem ~6 dos 84 STYs do nosso prompt. Entidades de STYs não
   cobertos (como `Sign or Symptom`, `Finding`) não têm predição do baseline.

2. **Split de treino desconhecido.** Schneider et al. não publicaram a divisão
   treino/teste usada para treinar os clinicalnerpt. Não há como garantir que
   nosso split de teste não vazou para o treino deles.

3. **Paradigma diferente.** Os clinicalnerpt são token classifiers IOB2
   (discriminativos, supervisionados); nossos modelos são LLMs generativas
   com prompt instruction-following. A comparação é informativa, não conclusiva.

**Decisão de agrupamento:** o relatório (`08_baseline_clinicalnerpt.py`)
é **agrupado por modelo baseline**. Para cada modelo:
- Lista os labels e acurácias do baseline
- Compara com os mesmos labels/acurácias dos nossos modelos
- Filtra **apenas os STYs que o modelo do HuggingFace possui**, garantindo
  comparação justa

**Split usado.** `08_baseline_clinicalnerpt.yaml` lê
`dados/divisao_Gold_Qwen7B.csv` — a fonte única de treino/teste/validação do
experimento (§3), a mesma dos treinos (`04_*`), das extrações (`05_*`), da
comparação (`06_*`) e da avaliação de NER (`07_*`).

**Limitação conhecida do script.** `rodar_modelo_ner` rotula toda entidade
detectada com `stys[0]`. Para os 3 modelos que cobrem mais de um STY
(`-medical` 2, `-chemical` 3, `-disorder` 7) as linhas dos demais STYs saem
com F1 0 por construção, não por erro do baseline — esses modelos foram
treinados com os STYs já fundidos numa classe e não os distinguem. Ler essas
linhas como agrupadas, ou comparar no nível do grupo.

---

## 3. Preparação do dataset (`CorpusSemClinBr`)

```python
from util_semclinbr import CorpusSemClinBr

corpus = CorpusSemClinBr("dados/SemClinBr-xml-public-v1")
corpus.inventario_tags(arquivo_divisao="dados/divisao_Gold_Qwen7B.csv")
corpus.exportar("dados/")
```

### Quem define treino / teste / validação

**Uma única fonte: `dados/divisao_Gold_Qwen7B.csv`, gerada pelo passo 03**
(`03_compara_gold_full.yaml`). A comparação calcula a dificuldade de cada
documento a partir das métricas do modelo base contra o gabarito e grava, num
só arquivo, as colunas `id`, `alvo` (`treino` / `teste` / `validacao`),
`dificuldade` e `dificuldade_int`. Todos os passos seguintes se apoiam nele, e
**apenas nos ids que constam nele**:

| Passo | Como consome a divisão |
|---|---|
| 04 — treino | `curriculum.divisao[].arquivo: dados/divisao_Gold_Qwen7B.csv`, com `dataset_filtro: {"dificuldade": ...}` por bloco curricular |
| 05 — extração no teste | `entrada.filtro.filtro_externo` → `arquivo: divisao_Gold_Qwen7B.csv`, `dataset_filtro: {"alvo": "teste"}` |
| 06 — comparação | `configuracao_comparacao.filtro` → mesmo arquivo, `dataset_filtro: {"alvo": "teste"}` |
| 07 — avaliação NER | `corpus.divisao` / `corpus.split` |
| 08 — baseline externo | `corpus.divisao` / `corpus.split` |

`util_semclinbr.py` **não divide o corpus** e a divisão **não é coluna do
parquet** — é sempre o arquivo do passo 03. Manter uma única fonte é o que
impede data leakage: duas divisões sorteadas independentemente sobre o mesmo
corpus divergem, e um filtro que use a divisão errada monta um "teste" com
documentos que o modelo viu no treino.

> **Ordem de execução.** O passo 03 precisa do parquet e do gabarito para
> calcular a dificuldade, então na primeira vez a divisão ainda não existe:
> `python util_semclinbr.py` → 02 (extração do modelo base) → 03 (**gera a
> divisão**) → `python util_semclinbr.py` de novo → 04 → 05 → 06 / 07 / 08.
> A segunda execução é o que faz o inventário de rótulos sair só do treino; o
> script avisa em voz alta quando roda sem a divisão (ver abaixo).

### Inventário de rótulos e a segunda execução

`inventario_tags(arquivo_divisao=..., alvo="treino")` deriva os rótulos do
prompt apenas dos documentos de treino, para não vazar a existência de STYs que
só ocorrem no teste. É vazamento fraco — metadado, não rótulo por instância —
mas evitá-lo é gratuito depois que o 03 rodou.

Quando o arquivo de divisão ainda não existe, o inventário sai do corpus
inteiro (89 rótulos em vez de 85) e `exportar()` imprime:

```
⚠️  ATENÇÃO — o prompt contém TODOS os rótulos do corpus (89), não apenas os do treino.
    O arquivo de divisão (dados/divisao_Gold_Qwen7B.csv) ainda não
    existe, então não há como saber quais documentos são de treino.
    O parquet está pronto e pode seguir para os passos 02 e 03.
    Depois que o 03 gerar a divisão, RODE ESTE SCRIPT DE NOVO para
    que o prompt fique só com os rótulos do treino.
```

Os 5 rótulos extras (`Amino Acid Sequence`, `Behavior`, `Fish`, `Regulation or
Law`, `Social Behavior`) têm frequência 1 cada, então o bootstrap serve
perfeitamente para rodar 02 e 03 — mas o prompt definitivo é o da segunda
execução.

### Dataset gerado

| Coluna | Conteúdo |
|---|---|
| `id` | nome do arquivo XML, sem extensão |
| `texto` | conteúdo de `<TEXT>` |
| `resposta` | gabarito JSON serializado, com os offsets já corrigidos (§2.1) |
| `prompt` | prompt com o inventário e o texto injetados (só com `incluir_prompt=True`) |
| *extras* | `n_entidades`, `n_relacoes`, `n_rotulos_distintos`, `n_multirotulo`, `n_chars`, `n_tags_fora_do_prompt` |

As colunas extras alimentam o componente estrutural do proxy $S_i$ e podem ser
descartadas se não forem usadas.

### Arquivos de saída

| Arquivo | Papel |
|---|---|
| `semclinbr.parquet` (ou `.csv`) | o dataset acima |
| `semclinbr.md` | relatório de qualidade das anotações (§2.1) |
| `prompt_semclinbr.txt` | prompt com o inventário já injetado |
| `inventario_semclinbr.csv` | `rotulo` + `frequencia_treino` (ou `frequencia_corpus`, no bootstrap — o cabeçalho declara a origem) |
| `saidas/saida_semclinbr_gold.parquet` | gabarito no formato do framework (`chave`, `resposta`, `erro`) — é o `modelo_base` dos passos 03 e 06 |
| `saidas/saida_semclinbr_gold.md` | o mesmo relatório de qualidade, junto do gabarito |

Nenhum arquivo de divisão é gerado aqui. O prompt é gravado junto com os dados
de propósito: **o inventário é derivado do corpus**, então sem esse arquivo o
experimento não é reprodutível.

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

Duas trilhas sobre as **mesmas saídas** (`saidas/*.parquet`):

**Trilha 1 — `06_compara_todos.yaml` (comum aos três experimentos).** Métricas
ROUGE-L e ROUGE-2 sobre os agregados virtuais `Entidades` (entities.text/tag/abbr —
alvo atômico), `Relacoes` (relations.reltype — alvo composto) e `(global)`; os ids
(`id`, `annotation1`, `annotation2`) ficam fora de propósito. É a trilha que ordena o
currículo (passo 03) e alimenta a **análise bayesiana por recorte** (análise
principal: `baycomp.CorrelatedTTest`, ROPE calibrada por campo via
`06_compara_d1ab.yaml`, limiar 0,95); Friedman/Wilcoxon/Nemenyi só exploratórios.

**Trilha 2 — `07_avaliar_ner.py` / `07_avaliar_ner.yaml` (específica do corpus).**
F1 de reconhecimento de entidades por documento, para comparabilidade com os
sistemas publicados:

| Item | Definição |
|---|---|
| Unidade de análise | documento de teste (desenho pareado: todos os protocolos veem os mesmos documentos) |
| **Variável primária da trilha** | **F1 strict por documento** (`f1_strict`) |
| Complementares | lenient, flexible, relaxed, span exato/parcial, F1 de relações |
| Estatística | Friedman + Wilcoxon (Holm) + $r = \lvert z \rvert / \sqrt{n}$ — **exploratória** (P6: o bayesiano da trilha 1 é a fonte dos vereditos) |
| Robustez | taxa de falha de parsing e taxa de não-alinhamento por protocolo |
| Piso de viabilidade | 0,70 (`piso_viabilidade`), fixado antes dos resultados |

Escolha do strict como primária: é a métrica mais exigente e a que menos depende
do mapeamento STY→SGR (que é uma escolha nossa, não do corpus). Declarar a
hierarquia antes de olhar os resultados; divergências entre strict e relaxed são
**achados** sobre onde o erro se concentra (fronteira de span vs. escolha de
rótulo), não inconsistências.

**ROPE (análise bayesiana).** A ROPE é ancorada na divergência entre os três
treinos do protocolo D1 (`d1`, `d1a`, `d1b`), por campo e por métrica — a menor
margem sob a qual todos os pares de réplicas saem equivalentes ao limiar de 0,95
(`00_rope_sugerido.md`), transcrita à mão para `rope_por_campo`. Ainda pendente
neste experimento (`rope: 0.01` é placeholder).

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
2. **Tamanho do corpus.** 997 documentos efetivos (ver item 6), contra 19,7k do
   SUMMA e 20k do PubMed. O número do SUMMA é o corpus **efetivo**: a divisão
   tem 22.155 linhas, mas todos os YAMLs de treino e de extração aplicam
   `dataset_filtro: {"fold": "<=10"}` em `curriculum.entrada`, reservando os
   folds 11 e 12 fora do experimento — restam 19.712 (13.788 / 1.976 / 3.948). Com split 697/103/197 (treino/validação/teste) e 3
   faixas, cada fase curricular fica com ~200 instâncias — pouco para full
   fine-tuning, com alta variância entre execuções esperada. A variância é medida
   pelo trio D1/D1a/D1b e absorvida pela ROPE calibrada (previsão declarada:
   margens mais largas e mais desfechos equivalentes/incertos). Alternativa
   (trabalho futuro): currículo em granularidade de sentença, ao custo de quebrar
   as relações entre entidades de sentenças distintas.
3. **Acesso ao corpus.** Formulário de solicitação com termo de licença para uso
   científico e não comercial. Considere o prazo no cronograma.
4. **Sigilo.** A licença restringe redistribuição — não exponha texto integral em
   apêndices; use os exemplos publicados no artigo (Tabelas 1 e 4), CC BY 4.0.
5. **Tokenização.** Notas em CAIXA ALTA são comuns e fragmentam mais no
   tokenizador do Qwen. Se algum proxy de dificuldade usar contagem de tokens,
   isso vira viés; `n_chars` evita.
6. **Instâncias não anotadas (IDs 8994, 8995, 8996).** Três documentos do corpus
   possuem `<TAGS>` e `<RELATIONS>` vazias nos XMLs originais, resultando em
   gabarito gold `{"entities": [], "relations": []}`. Contudo, seus textos
   contêm dezenas de entidades clínicas evidentes (e.g., "ACIDENTE VASCULAR
   CEREBRAL", "SEPTICEMIA", "GLASGOW 13", "CIRROSE HEPATICA", "PUPILAS
   ISOCÓRICAS", "TRAQUEOSTOMIZADA", termos farmacológicos e dispositivos).
   Trata-se de instâncias **não anotadas**, não de documentos genuinamente vazios.
   **Decisão conservadora:** descartar os três casos do pipeline de comparação e
   treinamento. O método `_filtro_origem` de `util_json_carga.py` já os exclui
   automaticamente ao detectar que todos os campos do gabarito são listas vazias,
   reduzindo o corpus efetivo de 1.000 para **997 instâncias** em
   `divisao_Gold_Qwen7B.csv`. Incluí-los introduziria ruído: o gabarito vazio
   puniria injustamente qualquer extração do modelo, distorcendo tanto as métricas
   de dificuldade ($S_i$) quanto o F1 por documento.
