# Avaliação Likert por grupos e validação do juiz LLM

Consolida avaliações em escala Likert 1–4 produzidas por **grupos de avaliadores**
— rodadas de um juiz LLM ou especialistas humanos — e decide se o juiz LLM está
validado para aplicação em massa.

| Arquivo | Papel |
|---|---|
| `realizar_avaliacoes.py` | carga, estatística, relatórios e CLI (único ponto de entrada) |
| `realizar_avaliacoes_graficos.py` | apenas as figuras; não faz nenhuma conta |
| `realizar_avaliacoes_teste.py` | verifica as estatísticas contra `scikit-learn`, `statsmodels` e casos analíticos |
| `util_est_bayesiana.py` | camada fina sobre o `baycomp`; opcional, só com `--bayes` |
| `realizar_avaliacoes.md` | este documento |

---

## 1. Conceito central: grupo

Um **grupo** é um conjunto de avaliações independentes dos mesmos itens.

| Tipo | Composição | O que a concordância interna mede |
|---|---|---|
| `llm` | N rodadas do mesmo modelo juiz | estabilidade teste-reteste do instrumento |
| `humano` | N especialistas distintos | acordo inter-avaliadores |

A matemática é idêntica nos dois casos — muda a interpretação e os rótulos do
relatório. É isso que permite tratar a avaliação humana como mais um grupo, lado
a lado com os juízes LLM, sem nenhum caminho de código separado.

Cada grupo é resumido pela **nota mediana** dos seus avaliadores por
(documento, fonte).

---

## 2. Entrada

Uma pasta por avaliador, cada uma com `saida_juiz_llm.parquet`. As pastas de um
grupo são numeradas de `01` a `99`, com ou sem separador:

```
gpt5_01/  gpt5_02/  gpt5_03/      -> grupo `gpt5`     (3 rodadas do juiz)
sabia4_01/ sabia4_02/ sabia4_03/  -> grupo `sabia4`   (3 rodadas do juiz)
humanos01/ humanos02/ humanos03/  -> grupo `humanos`  (3 especialistas)
```

Esquema do parquet — **o mesmo para juiz LLM e para humanos**:

| coluna | conteúdo |
|---|---|
| `chave` | `<id_documento>_<fonte>` — o id do documento **não pode conter `_`** |
| `resposta` | JSON `{"nota": 1..4, "problemas": ["alucinacao", ...]}` |
| `resumo` | JSON opcional com metadados da chamada (`model`, tokens, `tempo`) |
| `erro` | string de erro; vazia quando a avaliação foi bem-sucedida |

Qualquer conversão (planilha dos especialistas → parquet) é feita antes, fora
destes scripts.

### Nomes de fonte divergentes

Se os grupos nomeiam a mesma fonte de formas diferentes — o juiz registra `a` e
o formulário humano registra `qwen7b` — normalize na chamada com
`--alias qwen7b=a`. Sem isso a interseção de fontes volta vazia e a execução
aborta com mensagem explícita.

---

## 3. Uso

```bash
# um grupo isolado: só a análise interna
python realizar_avaliacoes.py --grupos gpt5:llm

# dois juízes LLM e a avaliação humana: análise interna + validação
python realizar_avaliacoes.py --grupos gpt5:llm sabia4:llm humanos:humano \
    --alias qwen7b=a --saida analise

# com o panorama bayesiano complementar (ROPE calibrada entre os especialistas)
python realizar_avaliacoes.py --grupos gpt5:llm humanos:humano --bayes

# modo direto retrocompatível: cada pasta é uma rodada de um único grupo LLM
python realizar_avaliacoes.py --pastas saida_01 saida_02 saida_03
```

| Parâmetro | Efeito |
|---|---|
| `--base` | diretório que contém as pastas dos avaliadores (padrão `.`) |
| `--grupos NOME:TIPO ...` | grupos a analisar; **tipo obrigatório**, em `llm` ou `humano` |
| `--pastas ...` | modo direto, um único grupo `llm`; mutuamente exclusivo com `--grupos` |
| `--saida` | pasta de saída (padrão `analise_avaliacoes`) |
| `--escala MIN MAX` | limites da escala Likert (padrão `1 4`); expande sozinha se houver notas fora |
| `--alias ORIGEM=DESTINO ...` | normaliza nomes de fonte entre grupos |
| `--referencia GRUPO` | padrão de referência (padrão: primeiro grupo `humano`) |

Flags da camada bayesiana. `--bayes` liga só o panorama **complementar** (heatmaps e
forest plots); a ROPE e o limiar valem para os critérios 2 e 3 do gate com ou sem ele:

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `--bayes` | desligado | ativa o panorama complementar entre fontes e entre avaliadores |
| `--bayes-rope R` | calibrada | ROPE das notas. Por padrão, calibrada entre os especialistas; **obrigatória** quando não há 2+ especialistas para calibrar |
| `--bayes-rope-decisao R` | calibrada | idem para a taxa de adequação (decisão prática) |
| `--bayes-limiar P` | `0.95` | probabilidade mínima exigida para veredito e relatórios |

O teste é unificado no `baycomp.CorrelatedTTest` (analítico, determinístico, sem amostras nem sementes).
Uma ROPE informada pela flag sobrescreve a calibração e fica registrada como pré-registro manual.

Um grupo → análise interna. Dois ou mais → análise interna de cada um **mais** a
validação dos juízes contra a referência.

---

## 4. O gate de validação

Três critérios, **todos aferidos no agregado** — as fontes empilhadas numa única
série, sem estratificação. O juiz precisa passar nos três.

| # | Critério | Instrumento | Aprova se |
|---|---|---|---|
| 1 | Concordância ordinal com o humano | κw de Cohen ponderado | κw ≥ 0,60 |
| 2 | Ausência de viés sistemático | P(equivalência) na diferença média das notas | ≥ 0,95 |
| 3 | Decisão prática (nota ≥ piso) | P(equivalência) na taxa de adequação | ≥ 0,95 |

Concordância sozinha não basta: um juiz pode concordar razoavelmente e ainda
assim ser sistematicamente mais leniente, o que enviesaria toda a aplicação em
massa na mesma direção. Daí os critérios 2 e 3.

Os critérios 2 e 3 são **bayesianos** (`baycomp.CorrelatedTTest`): a probabilidade
posterior de a diferença **média** caber na ROPE. Diferentemente de falhar em
rejeitar H₀, uma posterior concentrada dentro da ROPE é evidência **a favor**
da equivalência. O limiar de 0,95 é único em todo o trabalho.

### Calibração da ROPE (Fronteira de Equivalência)

A margem ("ROPE") não é arbitrada. Ela é **calibrada dinamicamente** pela
divergência entre os próprios especialistas humanos de referência.

O algoritmo busca numericamente (Brent) a menor margem sob a qual **todos os
pares de especialistas (o controle negativo)** alcançam o limiar de 0,95 de
probabilidade de equivalência. É a fronteira matemática exata do ruído humano;
se o juiz LLM divergir da referência por menos do que isso, ele é indistinguível
do ruído inerente aos especialistas. A taxa de adequação (`nota ≥ piso`) é calibrada
da mesma forma, sobre as notas binarizadas.

É o mesmo critério da calibração das métricas automáticas, **sem piso e sem valor
padrão**:

- o valor calibrado é usado como sai;
- sem grupo humano de referência com 2+ especialistas, a calibração não é possível e
  as margens têm de vir de `--bayes-rope` e `--bayes-rope-decisao` (pré-registro
  manual) — sem elas, a validação é interrompida;
- se os especialistas forem idênticos em todos os itens, a calibração degenera a zero
  (o baycomp exige ROPE > 0) e a flag passa a ser obrigatória;
- no panorama complementar entre fontes de um grupo com um só avaliador, sem
  `--bayes-rope` a comparação é omitida com aviso.

### Os três status possíveis

| Status | Condição |
|---|---|
| **VALIDADO** | os três critérios atendidos |
| **VALIDADO COM RESSALVA** | critério 1 atendido, e equivalência (2 ou 3) **inconclusiva**. |
| **NÃO VALIDADO** | κw abaixo do corte **ou** viés relevante (a diferença quase certamente excede a ROPE em alguma direção). |

O status intermediário existe porque "incerto" é um desfecho bayesiano legítimo
quando a amostra não basta para afirmar equivalência nem para afirmar viés relevante.

A acurácia contra o humano é **reportada, não é critério** — não há limiar
teórico defensável para ela, e ela varia com a prevalência de itens adequados na
amostra.

### Referência não humana

Quando o grupo de referência é do tipo `llm` (nenhum grupo `humano` presente, ou
`--referencia` apontando para um LLM), o relatório muda: título vira
"Concordância entre juízes LLM (ensaio metodológico)", um alerta é inserido no
topo, os rótulos trocam para CONVERGENTE / NÃO CONVERGENTE e as limitações
registram que convergência entre modelos não constitui validação. Se houver mais
de um grupo `humano`, o script avisa e usa o primeiro — especialistas exportados
como grupos separados devem ser consolidados num único grupo.

---

## 5. Pareamento

| Nível | Regra |
|---|---|
| Dentro do grupo | descarte global pareado: documento com falha em qualquer célula fonte × avaliador sai de **todas** as fontes |
| Entre grupos | interseção estrita de **documentos e fontes**: só entra o que todos os grupos avaliaram |

O que fica de fora da interseção permanece na análise interna do seu grupo — o
Gold Set restringe a validação, não a análise dos documentos do juiz. A tabela de
cobertura mostra exatamente o que cada grupo perdeu, para que uma interseção
menor que a esperada não passe despercebida.

---

## 6. Saída

```
analise/
├── gpt5/          estatisticas.md, dados_longo.csv, notas_medianas.csv, 01..09_*.png
├── sabia4/        (mesma estrutura)
├── humanos/       (mesma estrutura)
├── validacao.md
├── notas_medianas_pareadas.csv       documento, fonte, uma coluna por grupo
├── tabela_concordancia.csv           κw, IC, P_o, viés (uma linha por juiz)
├── tabela_decisao_binaria.csv        κ binário, FP/FN, McNemar, acurácia
├── tabela_contrastes_grupos.csv      post-hoc Wilcoxon + Holm entre grupos
├── tabela_ranks_fontes.csv           rank médio das fontes por grupo
├── 01_confusao_<juiz>.png
├── 02_distribuicao_diferencas.png
└── 03_distribuicao_por_grupo.png
```

Com `--bayes`, acrescentam-se `<grupo>/10_bayes_fontes.png`,
`<grupo>/bayes_fontes.csv`, `04_bayes_grupos.png`, `bayes_grupos.csv` e
`tabela_bayes_juizes.csv`.

Com um único grupo, a análise interna é escrita direto na raiz de `--saida`.

### `<grupo>/estatisticas.md`

| Bloco | Serve para |
|---|---|
| Falhas e descarte pareado | robustez em produção; justificar o `n*` |
| Fleiss κw global + IC 95% | confiabilidade do grupo (κw ≥ 0,60) |
| Cohen κw par a par | diagnóstico de qual avaliador destoa |
| Instâncias ambíguas | itens com discordância > 2 pontos entre avaliadores |
| Friedman entre avaliadores | deriva de severidade (LLM) / avaliador mais rigoroso (humano) |
| Descritivas + IC Wilson | caracterizar cada fonte |
| Shapiro-Wilk → Friedman → Wilcoxon + Holm → *r* | comparação de desempenho entre as fontes |
| Viabilidade (proporção ≥ 3) | argumento de uso em produção |
| Rubrica, problemas, custo | descritivos complementares |

### `validacao.md`

| Bloco | Serve para |
|---|---|
| Veredito | resultado do gate por juiz, com o critério que falhou |
| Critérios | os três limiares e por que o IC não é critério |
| Cobertura | o que entrou e o que se perdeu na interseção |
| Evidência detalhada | duas tabelas — concordância/viés e decisão binária (com FP/FN) |
| Diferença de severidade | Friedman entre grupos + post-hoc Wilcoxon com Holm |
| Nota lateral | concordância juiz × juiz, fora do gate |
| Ordenação das fontes | ranks médios por grupo (descritivo) |
| Limitações | as ressalvas que precisam ir para o texto |

Figuras: matriz de confusão por juiz, distribuição das diferenças e distribuição
das notas por avaliador.

---

## 7. Camada bayesiana (opcional)

Os critérios 2 e 3 do gate já são bayesianos e rodam sempre; o que `--bayes`
acrescenta é o **panorama complementar** — heatmaps e forest plots entre as fontes
e entre os avaliadores. A quantidade central é a que o teste de hipótese nula não
produz: a probabilidade posterior de **equivalência prática**. "Não rejeitar H₀"
significa apenas que não se detectou diferença naquele *n*; uma posterior
concentrada dentro da ROPE é evidência positiva a favor da equivalência.

Toda a estatística vem do pacote **`baycomp`** (Benavoli et al., 2017), através
de `util_est_bayesiana.py`, que é uma camada fina: organiza as chamadas, monta a
matriz de todos os pares e desenha o heatmap. Nada é reimplementado.

### Um único teste: `CorrelatedTTest`

Todo o pipeline usa o `baycomp.CorrelatedTTest`, nas notas e nas métricas contínuas:

1. **Escopo.** Benavoli et al. (2017) o propõem para comparar dois algoritmos em **um
   único conjunto de dados com observações pareadas** — o desenho deste trabalho.
   O `SignTest` e o `SignedRankTest` foram propostos para coleções de conjuntos.
2. **O `SignTest` satura.** Numa escala de 4 pontos os empates passam de 50% com
   frequência; nenhuma zona lateral consegue superar a central e P(equivalência) → 1
   independentemente do desequilíbrio.
3. **Determinismo.** O cálculo é analítico (Student acumulada): sem Monte Carlo, sem
   amostras, sem semente.

A ROPE incide sobre a **diferença média** das notas pareadas.

### A ROPE é calibrada, não arbitrada

A margem vem do controle negativo (ver "Calibração da ROPE" na seção 4): a menor
margem sob a qual todos os pares de especialistas saem equivalentes ao limiar.
Sem piso e sem valor padrão — sem calibração possível, ela é declarada pela flag.

Como a ROPE pode decidir a classificação, as comparações entre protocolos
(`comparar_extracoes_baycomp.py` e `07_avaliacao_llm.py`) reportam a sensibilidade
com ROPE/2, ROPE e 2·ROPE.

⚠️ **A ROPE precisa ser > 0.** Com `rope = 0` o baycomp devolve apenas
`(p_esquerda, p_direita)`, sem a probabilidade de equivalência. O pipeline
recusa o valor com mensagem explícita.

### O que é produzido

| Onde | Conteúdo |
|---|---|
| `<grupo>/estatisticas.md` | relações par a par entre as **fontes**, síntese por fonte |
| `<grupo>/10_bayes_fontes.png` | heatmap fonte × fonte |
| `validacao.md` | leitura bayesiana dos critérios 2 e 3, por juiz; matriz entre avaliadores |
| `04_bayes_grupos.png` | heatmap grupo × grupo |
| `bayes_fontes.csv`, `bayes_grupos.csv`, `tabela_bayes_juizes.csv` | dados completos |

### Como ler o heatmap

Cor = categoria (verde superior, azul equivalente, vermelho inferior, cinza
incerto); número e intensidade = probabilidade posterior. A diagonal é neutra —
`(Pi, Pi)` não é comparação. `incerto` é categoria explícita: nenhuma das três
probabilidades alcançou o limiar. É desfecho legítimo.

Cada par é comparado **uma vez só**; a célula espelhada troca `p_esquerda` com
`p_direita`, o que garante simetria exata em vez de duas estimativas Monte Carlo
aproximadas.

### O veredito julga magnitude, não direção

| Leitura | Condição |
|---|---|
| **SEM VIÉS RELEVANTE** | equivalência nas notas **e** na decisão binária atinge o limiar |
| **VIÉS RELEVANTE** | é quase certo que a divergência **excede** a ROPE |
| **INCONCLUSIVO** | nenhuma das duas leituras se sustenta |

### Limitação assumida: média de notas ordinais

A diferença média de notas Likert é a diferença das médias — o pareamento não
contorna a objeção ordinal. A defesa: (i) a literatura de robustez (Norman, 2010;
Carifio & Perla, 2008); (ii) a ROPE ancorada na divergência entre especialistas, na
mesma unidade; (iii) as **contagens ordinais puras** (`Acima`, `Empate`, `Abaixo`)
reportadas ao lado de toda probabilidade — um número que não assume intervalos.

---

## 8. Convenções

| Item | Escolha |
|---|---|
| Pesos do κ | quadráticos |
| Faixas de interpretação | McHugh (2012) |
| IC do κw | bootstrap percentílico de **documentos** (2.000 réplicas, semente 42) |
| Critério 1 | valor **pontual** do κw; o IC é precisão, não critério |
| Correção múltipla | Holm-Bonferroni (contrastes entre fontes e entre grupos) |
| Critérios 2 e 3 do gate | P(equivalência) bayesiana ≥ 0,95 nas notas e na taxa de adequação |
| Grupo de referência | primeiro grupo `humano` |
| Empates no Wilcoxon | descartados (convenção `wilcox`) |
| Shapiro-Wilk | confirmatório; a opção não paramétrica é primariamente ordinal |
| McNemar | binomial exato com < 25 discordantes, qui-quadrado com correção acima |
| Binarização | `{1,2}` inadequado, `{3,4}` adequado |
| Posterior bayesiana | `baycomp.CorrelatedTTest` (Benavoli et al., 2017), o mesmo nas notas e nas métricas contínuas; analítico, sem amostras nem semente |
| ROPE na Likert | calibrada entre os especialistas (menor margem de equivalência de todos os pares), sem piso nem valor padrão; sem calibração, declarada pela flag |
| ROPE da taxa de adequação | idem, sobre as notas binarizadas |
| Limiar de classificação e de veredito | 0,95 (único) |

---

## 9. Observações para a dissertação

Pontos que precisam constar no texto ou aparecer na defesa.

**O gate combina concordância e equivalência.** O critério 1 é frequentista
(κw); os critérios 2 e 3 são bayesianos e afirmam o que o frequentista não
afirma: que a divergência entre juiz e humano **cabe** dentro de uma margem de
irrelevância prática. Onde um teste de hipótese escreveria "não se detectou viés
sistemático", o gate escreve "P(equivalência | ε) = 0,97" — uma afirmação
positiva, e mais forte. Wilcoxon e McNemar continuam nos relatórios como
evidência descritiva; se divergirem da leitura bayesiana, reporte as duas e
explique a divergência — não escolha a que favorece o resultado.

**Registre teste, ROPE e limiar antes de rodar.** Todos entram no rodapé de
reprodutibilidade dos relatórios. A ROPE da Likert é calibrada pelo controle
negativo (a divergência entre os especialistas), não escolhida — o que afasta a
suspeita de margem ajustada ao resultado. Quando vier da flag, declare-a como
pré-registro manual e justifique o valor.

**Reporte as contagens junto das probabilidades.** A diferença média de notas
ordinais é uma limitação assumida (seção 7); as contagens `Acima`/`Empate`/`Abaixo`
não assumem intervalos e permitem ao leitor ver o desequilíbrio entre as
laterais, por isso vão em toda tabela bayesiana.

**A análise é independente por conjunto de dados.** Não modela a variabilidade
compartilhada entre datasets nem faz atualização sequencial do conhecimento —
o resultado de um dataset não influencia o seguinte, o que é deliberado e
permite avaliar a **consistência** das relações entre protocolos. A limitação
precisa constar: modelos hierárquicos ou multinível, que estimariam efeitos
globais e específicos por dataset e avaliariam formalmente a heterogeneidade,
ficam como trabalho futuro.

**"Incerto" e "inconclusivo" são resultados, não lacunas.** Aparecem no heatmap
e nos vereditos quando os dados não sustentam nenhuma leitura ao limiar
adotado. Reporte-os como tais. A alternativa — afrouxar o limiar ou o ε até a
célula mudar de cor — troca uma resposta honesta por uma inventada.

**Likert é a métrica principal; o F1 é complementar, e isso precisa estar
escrito.** A Likert mede qualidade percebida, o F1 mede fidelidade de extração
contra o professor — um protocolo que reproduz fielmente um erro do professor é
premiado pelo F1 e punido pela Likert. Convergências e divergências entre as
duas são **achados**, não inconsistências, porque as métricas capturam
propriedades distintas. Mas a hierarquia tem de ser declarada antes dos
resultados: se o F1 favorecer uma conclusão e a Likert outra, é a Likert que
decide, e o texto deve ter dito isso antes de saber qual favorecia o quê.

**As "equivalências" da Likert e do F1 são a mesma afirmação, em unidades
diferentes.** Nas duas, equivalência é P(diferença média ∈ ROPE) pelo
`CorrelatedTTest`; mas a ROPE da Likert está em pontos da escala e é calibrada
pela divergência entre avaliadores, e a do F1 está na unidade da métrica e é
calibrada pela variação entre execuções do mesmo protocolo. O texto precisa dizer
em que unidade e com que controle negativo cada margem foi obtida.

**A ROPE pode determinar o resultado, e a sensibilidade é resultado principal.**
Com BERTScore entre protocolos destilados do mesmo professor, os escores são
comprimidos e a classificação vira função da ROPE; nas notas do juiz LLM, com
milhares de itens, a ROPE calibrada fica estreita e o mesmo pode ocorrer. Reporte
a matriz em pelo menos três valores (ROPE/2, ROPE, 2·ROPE) e mostre onde as
categorias trocam. E registre o controle negativo — réplicas do mesmo protocolo
saindo equivalentes é o que demonstra que a margem não foi escolhida para
produzir o resultado.

**A rodada única nos 4.000 se apoia na etapa de validação.** Não há evidência de
estabilidade teste-reteste dentro da etapa dos protocolos; ela vem inteiramente
dos 210 pares, e é por isso que aquela etapa usa a **pior** rodada do juiz, não
a média. Faça a ponte explícita no texto: sem ela, a rodada única parece
escolhida por economia.

**O critério é o valor pontual, e isso foi decidido a priori.** Exigir que o
limite inferior do IC alcançasse 0,60 reprovaria o juiz por tamanho de amostra,
não por falta de concordância — e não haveria como distinguir os dois casos. O IC
entra como medida de precisão. Registre a escolha antes de apresentar o
resultado, para não parecer definida depois de vê-lo.

**A validação é global, não por fonte.** O texto pode afirmar que o juiz
reproduz o julgamento humano *para o conjunto das fontes avaliadas* — que é
exatamente o uso pretendido. Não sustenta afirmar confiabilidade equivalente em
cada protocolo isoladamente. Uma frase nas limitações resolve.

**Por que não estratificar.** Dentro de um estrato as notas se concentram em
poucas categorias, a concordância esperada por acaso sobe e o κ despenca mesmo
com concordância observada acima de 0,90 — o paradoxo do Kappa (Feinstein &
Cicchetti, 1990). Estratificar produziria κ próximo de zero em fontes onde não há
discordância real. Se alguém perguntar por que não há análise por fonte, esta é a
resposta.

**Não rejeitar H₀ não é provar equivalência.** Nos critérios 2 e 3, a leitura
correta é "não há viés detectável neste tamanho de amostra". Equivalência formal
exigiria definir uma margem e um n maior. Não escreva "o juiz é equivalente ao
humano"; escreva "não se detectou viés sistemático".

**Se o κw sair moderado (0,60–0,79), o argumento prático é o critério 3.**
Concordância exata na escala de 4 pontos costuma ficar por volta de 50% mesmo
entre bons avaliadores; o que sustenta o uso em produção é a equivalência na
decisão adequado/inadequado, com a acurácia contra o humano. A matriz de confusão
tipicamente mostra que as discordâncias se concentram no limite 2↔3.

**Se o κw sair baixo com P_o alto**, a causa é efeito de teto da escala, não
instabilidade. Reporte P_o ao lado do κw sempre — é o que permite ao leitor
distinguir os dois casos sem precisar de coeficientes alternativos.

**Mediana ≡ moda com 3 avaliadores.** Os capítulos falam em mediana; as
anotações pós-qualificação falam em moda. Com 3 avaliações em escala de 4 pontos
as duas coincidem sempre que existe moda (dois valores iguais tornam a mediana
igual a eles), e no caso `{1,2,3}` a mediana devolve o valor central. É agregação
ordinal por voto majoritário, não média disfarçada. Vale uma nota de rodapé
harmonizando os dois termos.

**A escolha não-paramétrica é primariamente ordinal; o Shapiro-Wilk é
confirmatório.** O pipeline reporta o Shapiro das diferenças pareadas porque o
método o declara, mas a justificativa principal para Friedman/Wilcoxon é a
natureza ordinal da escala Likert (Stevens, 1946; Jamieson, 2004) — e ela vale
mesmo que o Shapiro não rejeitasse a normalidade. O texto deve apresentar as duas
razões nessa ordem.

**Os itens não são independentes no agregado.** Cada documento entra tantas
vezes quantas forem as fontes. É coerente com a unidade de julgamento — a
extração avaliada, não o documento — e o bootstrap reamostra documentos inteiros
justamente para preservar essa dependência. Vale a ressalva explícita.

**Por que o gate usa o p não corrigido.** A seção de severidade traz os
contrastes com Holm, mas o critério 2 lê o `p bruto`. A correção protege contra
falsos positivos; como aqui rejeitar H₀ é o resultado desfavorável ao juiz,
adotar o valor corrigido tornaria o gate mais permissivo — a correção trabalharia
a favor da conclusão que se quer testar. O `p Holm` fica na tabela para a leitura
conjunta dos três contrastes.

**Sobre diagramas de diferença crítica (Nemenyi).** O pipeline usa Friedman com
post-hoc Wilcoxon + Holm, não Nemenyi. Dois motivos, caso a pergunta apareça:
ranks médios dependem de quais avaliadores estão no pool comparado, e o Wilcoxon
fornece tamanho de efeito (*r*), que o Nemenyi não fornece. Além disso, a
diferença crítica escala com 1/√N — com milhares de blocos ela fica próxima de
zero e o diagrama deixa de ser inferencial. Se um diagrama for desejado para
manter simetria visual com o gráfico de acurácia entre modelos, ele deve ser
apresentado como descritivo.

**O `r` do Wilcoxon mede consistência, não magnitude.** Com empates
descartados, `r = |z|/√n′` responde "entre as discordâncias, quão consistente é
a direção?". Um juiz pode ter r = 0,87 ("grande") com diferença média de 0,2
ponto: viés pequeno porém quase sempre no mesmo sentido. A magnitude é lida em
`Média dif.`, confrontada com a margem — nunca no `r` sozinho. O relatório traz
essa explicação junto à tabela, mas o texto da dissertação deve repeti-la onde o
`r` for reportado.

**A concordância interna do grupo humano é pré-requisito.** Se os três
especialistas não concordarem entre si (κw < 0,60 no `humanos/estatisticas.md`),
o padrão de referência é frouxo e o gate perde sentido, qualquer que seja o
resultado do juiz. Verifique esse número antes de interpretar a validação.

**A concordância entre juízes LLM é nota lateral.** Dois juízes podem convergir
por compartilharem os mesmos vieses de pré-treinamento. Não é evidência de
validade — só o humano é o padrão.

**Se nenhum juiz for validado**, a aplicação em massa não fica impedida, mas
passa a ser reportada com ressalva explícita, e as conclusões sobre diferenças
entre protocolos precisam ser tratadas como dependentes do instrumento. Vale
decidir isso antes de rodar, não depois.

---

## 10. Taxonomia de problemas

A lista fechada espelha o template do Label Studio:

`alucinacao`, `omissao`, `erro_factual`, `atribuicao_errada`, `nao_consta_indev`.

A normalização tolera acentuação, caixa, separadores e escapes unicode
quebrados, e converte variantes conhecidas (`nao_consta` → `nao_consta_indev`,
`atribucao_errada` → `atribuicao_errada`). Rótulos que não colapsam em nenhuma
categoria aparecem na tabela de aderência marcados como fora da rubrica — sinal
de que o *prompt* ou a instrução aos avaliadores não restringiu efetivamente a
saída à lista.

---

## 11. Dependências e verificação das estatísticas

**Obrigatórias:** `pandas`, `numpy`, `scipy`, `pyarrow`, `scikit-learn`, `statsmodels`.

Os coeficientes com implementação consolidada em pacote são calculados por eles,
favorecendo a replicabilidade por terceiros:

| Estatística | Pacote |
|---|---|
| Kappa de Cohen ponderado | `sklearn.metrics.cohen_kappa_score` |
| Correção de Holm | `statsmodels.stats.multitest.multipletests` |
| Teste de McNemar | `statsmodels.stats.contingency_tables.mcnemar` |
| IC de Wilson | `statsmodels.stats.proportion.proportion_confint` |
| Friedman, Wilcoxon, Shapiro-Wilk | `scipy.stats` |
| Kappa de Fleiss **ponderado** | implementação interna — não há equivalente ponderado em pacote consolidado |
| Bootstrap com reamostragem de documentos | implementação interna — `scipy.stats.bootstrap` reamostra observações, não clusters |
| Comparação bayesiana pareada | `util_est_bayesiana` — método de Benavoli et al. (2017), `baycomp` como referência |
| Calibração da ROPE | percentil das diferenças pareadas entre execuções do mesmo protocolo |

Cada relatório registra, na seção "Origem das estatísticas" do rodapé, qual
pacote e versão calculou cada estatística.

**Verificação.** `realizar_avaliacoes_teste.py` contém as definições
operacionais (fórmulas internas) de cada estatística e verifica que coincidem
com os pacotes usados pelo pipeline (tolerância 1e-9). Também testa
propriedades analíticas conhecidas — concordância perfeita → κ = 1,
discordâncias simétricas → McNemar p = 1, Wilson dentro de [0, 1], IC
bootstrap contendo a estimativa pontual.

```bash
python realizar_avaliacoes_teste.py           # resumo legível + testes
python -m unittest realizar_avaliacoes_teste  # saída padrão
```

**Opcional:** `matplotlib` e `util_graficos` para as figuras. Sem
`util_graficos`, os gráficos herdados são pulados com aviso e toda a estatística
— inclusive as figuras da validação, feitas em matplotlib puro — segue
normalmente.

**Obrigatório para a validação:** `util_est_bayesiana.py` na mesma pasta (ou no
`PYTHONPATH`) e o pacote `baycomp` — os critérios 2 e 3 do gate são bayesianos.
Sem eles, a validação e `--bayes` abortam com mensagem explícita; a análise interna
de um grupo (sem validação nem `--bayes`) roda sem eles.

