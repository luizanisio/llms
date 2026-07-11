# Análise de Consistência — Juiz LLM × Label Studio × Prompt SUMMA

## 1. Juiz LLM vs. Template Label Studio (Avaliação LLM × Avaliação Humana)

### ✅ Elementos Consistentes

| Elemento | Juiz LLM | Label Studio | Status |
|---|---|---|---|
| **Critério único — Fidelidade** | Texto idêntico (L14 do prompt) | `instr_criterio` (L125 do template) | ✅ Idêntico |
| **Campos avaliados** (Matéria, Ponto, Argumentos, Doutrina, Conceitos, Resumo) | L16–22 | `instr_campos` (L127) | ✅ Idêntico |
| **Convenções** (Estilo, Siglas, Colegialidade, Páginas) | L24–28 | `instr_convencoes` (L129) | ✅ Idêntico |
| **Escala 1–4** com descritores | L30–35 | `instr_escala` (L131) | ✅ Idêntico |
| **Lista de problemas** (5 categorias) | L37–42 | `instr_problemas` (L133) | ✅ Idêntico |
| **Categorias nos Choices** (notas e problemas) | — | L158–173, L215–230, L272–287 | ✅ Consistente com a escala e a lista de problemas |
| **Hints dos Choices** | — | Resumos fiéis às definições do prompt | ✅ OK |

### ⚠️ Diferenças Menores (Esperadas / Aceitáveis)

| Diferença | Juiz LLM | Label Studio | Observação |
|---|---|---|---|
| **Formato da resposta** | JSON `{"nota":..., "problemas":[...]}` | Choices interativos + TextArea | Esperado: formatos diferentes para LLM vs. humano |
| **Justificativa** | Não solicitada (resposta mínima em JSON) | Campo `justificativa_colN` (TextArea, opcional) | ✅ OK — o template humano tem campo extra para justificativa, o que é bom para análise qualitativa |
| **Procedimento sequencial** | Descrito em `<Procedimento>` (L45–51) | Descrito em `instr_fluxo` (L137) com mecânica de "Extração concluída" | Adaptação de fluxo para interface interativa — OK |
| **Número de extrações** | Avalia 1 extração por chamada | Avalia 3 extrações em sequência | Design esperado — o LLM é chamado N vezes; o humano vê 3 de uma vez |
| **Obrigatoriedade de problemas** | "obrigatório indicar ao menos um para notas 1 e 2" | "obrigatório para notas 1 e 2, opcional para 3 e 4" | ✅ Mesmo conteúdo, redação ligeiramente mais curta no template |

### ❌ Inconsistências Encontradas

> [!NOTE]
> **Nenhuma inconsistência material foi encontrada entre o prompt do juiz LLM e o template do Label Studio.** Os dois estão muito bem alinhados — os textos das orientações são virtualmente idênticos, e as diferenças refletem apenas adaptações de formato (JSON programático vs. interface visual interativa).

---

## 2. Prompt SUMMA (Extração) vs. Orientações de Avaliação

Aqui a questão central é: **as orientações de avaliação dão ao avaliador (humano ou LLM) informação suficiente para não penalizar indevidamente o que o prompt SUMMA instrui, e para detectar quando a extração falha?**

### ✅ Instruções do SUMMA Refletidas na Avaliação

| Instrução SUMMA | Como aparece na avaliação | Status |
|---|---|---|
| **Foco no relator** — "Considere apenas as opiniões, posicionamentos e decisões da PessoaDeInteresse" (SUMMA L9) | "Considere apenas as opiniões, posicionamentos e decisões do relator (voto); afirmações de terceiros ou de citações não devem ter sido extraídas como se fossem do relator" | ✅ OK |
| **Não inventar** — "Nunca invente resposta" (SUMMA L5) | Problema `alucinacao` + critério de fidelidade | ✅ OK |
| **Siglas de normas** — "inclua a seguir sua sigla" (SUMMA L13) | Convenção explícita: "a extração acrescenta a sigla após o nome [...] não é alucinação" | ✅ OK |
| **Colegialidade** — "Nunca registre [...] mudou de entendimento [...] colegialidade" (SUMMA L21) | Convenção explícita: "a extração deliberadamente NÃO registra [...] colegialidade — não marque omissão por isso" | ✅ OK |
| **Paginação estrutural** — "Desconsidere as numerações de páginas constantes" + "utilize a estrutura descrita em EstruturaDocumentoFoco" (SUMMA L7) | Convenção: "referências de páginas (fls.) seguem a paginação estrutural interna [...] divergências de paginação não são erro factual" | ✅ OK |
| **Matéria** = questão de fundo (SUMMA L68–70) | Campo "Matéria: uma frase com a questão de fundo discutida no processo, nas palavras do relator" | ✅ OK |
| **Temas Tratados** incluem questões de ordem pública (SUMMA L117–119) | "questões de ordem pública tratadas no voto [...] devem sempre aparecer como temas" + lista explícita | ✅ OK |
| **Argumentos** = transcrições de frases do relator (SUMMA L142–148) | "transcrições das frases do relator que fundamentam sua posição no tema" | ✅ OK |
| **Doutrina citada** (SUMMA L204–218) | "obras e autores citados pelo relator como base doutrinária do tema" | ✅ OK |
| **Conceitos Relevantes** (SUMMA L220–232) | "definições apresentadas pelo relator relevantes para a análise do tema" | ✅ OK |
| **Resumo** sem nomes de partes, usando papel processual (SUMMA L264–265, L316–317) | "no Resumo, nomes de partes NÃO devem aparecer — o correto é o papel processual ('apelante', 'recorrido' etc.); não penalize o uso do papel no lugar do nome, penalize o inverso" | ✅ OK |
| **Estilo formal** — "Adote o estilo de escrita do Ministro Gilmar Mendes" (SUMMA L11) | Convenção: "o texto segue redação jurídica formal e detalhista; avalie o conteúdo, não o estilo" | ✅ OK |

### ⚠️ Campos do SUMMA **NÃO Avaliados** (Possíveis Lacunas)

> [!IMPORTANT]
> Os seguintes campos do prompt SUMMA **são extraídos** mas **NÃO são listados como campos avaliados** nas orientações de avaliação. Isso é intencional? Se sim, o avaliador simplesmente os ignora. Se não, há risco de não capturar problemas nesses campos.

| Campo SUMMA | Presente na avaliação? | Risco |
|---|---|---|
| **Dispositivo** — parte dispositiva do voto (SUMMA L72–74) | ❌ Não listado como campo avaliado | **Baixo** — se o resumo é bem feito, a decisão estará lá. Mas erros no dispositivo isolado não seriam detectados. |
| **Partes** — lista de partes (SUMMA L76–78) | ❌ Não listado como campo avaliado | **Baixo** — campo estrutural/metadata. |
| **Data do Julgamento** (SUMMA L80–82) | ❌ Não listado como campo avaliado | **Baixo** — campo de metadata. |
| **Preliminar** (Sim/Não por tema) (SUMMA L116) | ❌ Não avaliado explicitamente | **Baixo** — informação embutida nos temas, mas o avaliador não é instruído a verificar se a classificação de preliminar está correta. |
| **Fatos Apresentados** (SUMMA L130–139) | ❌ Não listado como campo avaliado | **Médio** — o SUMMA extrai fatos por tema; se a avaliação não olha para eles, erros factuais na extração de fatos podem não ser flagrados. |
| **Normas Aplicadas** (SUMMA L154–187) | ❌ Não listado como campo separado | **Médio** — normas são mencionadas nos exemplos de Resumo ("normas efetivamente aplicadas") mas não como campo avaliado independente. |
| **Jurisprudência Citada** (SUMMA L189–202) | ❌ Não listado como campo avaliado | **Médio** — similar ao caso de Normas. Jurisprudência é um componente importante da extração SUMMA. |
| **Decisão por Tema** (SUMMA L235–269) | ❌ Não listado como campo separado | **Baixo** — provavelmente coberto pelo Resumo. |
| **Explicação do Tema** (SUMMA L271–318) | ❌ Não listado como campo avaliado | **Baixo** — texto agregado, coberto pelo Resumo na avaliação. |

### ⚠️ Possível Problema: Citações e Argumentos de Terceiros

| Aspecto | SUMMA | Avaliação | Observação |
|---|---|---|---|
| **Citações** | SUMMA instrui que citações (`<CITACAO>`) e trechos entre aspas sejam usados para Jurisprudência e Doutrina, **mas NÃO para Fatos e Argumentos** | Avaliação diz "afirmações de terceiros ou de citações não devem ter sido extraídas como se fossem do relator" | ✅ Consistente, mas a avaliação é mais genérica. Se a extração incluir um trecho de doutrina (entre aspas) como Argumento, o avaliador deveria marcar `atribuicao_errada`. A instrução está lá, mas é sutil. |
| **Jurisprudência no SUMMA** | SUMMA instrui listar todas as jurisprudências do STJ/STF (SUMMA L22) | Avaliação não lista Jurisprudência como campo | ⚠️ Se a extração omite jurisprudência importante, o avaliador pode não detectar (não é campo avaliado). Pode estar coberto indiretamente por "Argumentos". |

---

## 3. Resumo Geral

### Consistência Juiz LLM ↔ Label Studio

> [!TIP]
> **Excelente alinhamento.** Os textos são praticamente idênticos. Não há risco de o avaliador humano receber instruções diferentes do juiz LLM. As diferenças são puramente de formato de interface.

### Cobertura do Prompt SUMMA nas Orientações de Avaliação

> [!WARNING]
> **A avaliação cobre os campos de mais alto valor informacional** (Matéria, Temas/Pontos, Argumentos, Doutrina, Conceitos, Resumo) e reflete corretamente as convenções do SUMMA (siglas, colegialidade, paginação, estilo, foco no relator).
> 
> **Porém, campos secundários do SUMMA não são avaliados individualmente:** Dispositivo, Partes, Data, Preliminar, Fatos, Normas, Jurisprudência, Decisão por Tema, Explicação. Isso é intencional, pois esses campos não serão avaliados na escala Likert.

