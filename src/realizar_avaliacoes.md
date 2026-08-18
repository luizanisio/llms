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

# com a camada bayesiana (baycomp; ROPE = 0,5 = notas iguais)
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

Flags da camada bayesiana — **sem `--bayes` nenhuma delas é lida** e o pipeline
roda exatamente como antes:

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `--bayes` | desligado | ativa a etapa; é a única chave que decide se ela existe |
| `--bayes-rope R` | `0.5` | largura da ROPE sobre as notas; na escala inteira, "notas iguais". Deve ser > 0 |
| `--bayes-metodo M` | `sinais` | `sinais` (SignTest), `postos` (SignedRankTest) ou `t` (CorrelatedTTest) |
| `--bayes-limiar P` | `0.80` | probabilidade mínima para classificar uma célula do heatmap |
| `--bayes-limiar-veredito P` | `0.95` | probabilidade mínima para o veredito juiz × referência |
| `--bayes-amostras N` | `50000` | amostras da posterior; ignorado por `--bayes-metodo t` |
| `--bayes-semente S` | `42` | semente da amostragem |

Informar um `--bayes-*` sem `--bayes` é **erro**, não silêncio: quem escreveu
`--bayes-metodo t` esperava a seção no relatório, e ignorar a flag devolveria um
documento sem ela e sem nenhum aviso.

Um grupo → análise interna. Dois ou mais → análise interna de cada um **mais** a
validação dos juízes contra a referência.

---

## 4. O gate de validação

Três critérios, **todos aferidos no agregado** — as fontes empilhadas numa única
série, sem estratificação. O juiz precisa passar nos três.

| # | Critério | Estatística | Aprova se |
|---|---|---|---|
| 1 | Concordância ordinal com o humano | κw de Cohen ponderado | κw ≥ 0,60 |
| 2 | Ausência de viés sistemático | Wilcoxon bilateral pareado | p > 0,05 |
| 3 | Equivalência na decisão prática | McNemar sobre nota ≥ 3 | p > 0,05 |

Concordância sozinha não basta: um juiz pode concordar razoavelmente e ainda
assim ser sistematicamente mais leniente, o que enviesaria toda a aplicação em
massa na mesma direção. Daí os critérios 2 e 3.

### Os três status possíveis

| Status | Condição |
|---|---|
| **VALIDADO** | os três critérios atendidos |
| **VALIDADO COM RESSALVA** | critérios 1 e 3 atendidos; viés significativo (critério 2) porém com magnitude média abaixo da margem de relevância prática |
| **NÃO VALIDADO** | demais casos |

A **margem de relevância prática** é 0,5 desvio-padrão das notas do grupo de
referência — a regra da meia-DP de Norman, Sloan & Wyrwich (2003), que mostra
convergência empírica da diferença minimamente importante para ~0,5 DP através de
instrumentos e populações. A margem é derivada dos dados, não arbitrada. O status
intermediário operacionaliza o plano contingencial já declarado no método
("resultados reportados com a devida ressalva") e existe porque, com n grande, o
Wilcoxon detecta como significativas diferenças praticamente irrelevantes. É
classificação descritiva, não teste de equivalência (que exigiria TOST com margem
pré-registrada; Lakens, 2017).

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

Complementa o gate, **não o substitui**: κw, Wilcoxon e McNemar continuam
decidindo. O que a camada acrescenta é a quantidade que o teste de hipótese nula
não produz — a probabilidade posterior de **equivalência prática**. "Não rejeitar
H₀" significa apenas que não se detectou diferença naquele *n*; uma posterior
concentrada dentro da ROPE é evidência positiva a favor da equivalência.

Toda a estatística vem do pacote **`baycomp`** (Benavoli et al., 2017), através
de `util_est_bayesiana.py`, que é uma camada fina: organiza as chamadas, monta a
matriz de todos os pares e desenha o heatmap. Nada é reimplementado.

### Os três testes do baycomp

| Método | Classe | Quando |
|---|---|---|
| `sinais` *(padrão)* | `SignTest` | escala **ordinal**: conta direções, descarta magnitude |
| `postos` | `SignedRankTest` | usa a ordenação das magnitudes |
| `t` | `CorrelatedTTest` | escala **contínua**; analítico, sem Monte Carlo |

Esta etapa é integralmente ordinal (Likert 1–4), então o padrão é `sinais`. Os
outros ficam disponíveis por flag, mas não são o uso previsto aqui.

### A ROPE não é parâmetro livre

Na escala Likert inteira, `ROPE = 0,5` significa exatamente **notas iguais** — é
a tradução direta da escala, não uma margem arbitrada nem calibrada. Por isso não
há análise de sensibilidade a ela nesta etapa: alterá-la deixaria de representar
a escala.

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

### Limitação importante do teste de sinais

O `SignTest` responde **qual das três zonas concentra mais massa posterior**, e
não "quem é melhor, e por quanto". Com muitos empates — comum na Likert de 4
pontos —, a zona central vence quase sempre, e a camada tende a reportar
equivalência mesmo havendo desequilíbrio claro entre as laterais.

Na prática isso significa que **a leitura bayesiana pode divergir do gate
frequentista**: o Wilcoxon detecta viés sistemático a partir do desequilíbrio,
enquanto o teste de sinais vê a maioria de empates e conclui equivalência. Não é
contradição — são perguntas diferentes —, mas o texto precisa reportar as
contagens (`Acima`, `Empate`, `Abaixo`) ao lado das probabilidades, e o gate
continua sendo o que decide.

---

## 8. Convenções

| Item | Escolha |
|---|---|
| Pesos do κ | quadráticos |
| Faixas de interpretação | McHugh (2012) |
| IC do κw | bootstrap percentílico de **documentos** (2.000 réplicas, semente 42) |
| Critério 1 | valor **pontual** do κw; o IC é precisão, não critério |
| Correção múltipla | Holm-Bonferroni (contrastes entre fontes e entre grupos) |
| Critério 2 do gate | `p` **não corrigido** do Wilcoxon juiz × referência |
| Grupo de referência | primeiro grupo `humano` |
| Empates no Wilcoxon | descartados (convenção `wilcox`) |
| Margem de relevância prática | 0,5 DP das notas da referência (meia-DP) |
| Shapiro-Wilk | confirmatório; a opção não paramétrica é primariamente ordinal |
| McNemar | binomial exato com < 25 discordantes, qui-quadrado com correção acima |
| Binarização | `{1,2}` inadequado, `{3,4}` adequado |
| Posterior bayesiana | teste de sinais de Benavoli et al. (2017); 200.000 amostras, semente 42 |
| Modo na escala Likert | `proporcao` (ROPE = 0; as diferenças são inteiras) |
| Modo nos escores contínuos (F1) | `baycomp` (cálculo padrão do pacote; exige ROPE > 0) |
| ROPE na Likert | 0,5 — decorre de a escala ser inteira ("notas iguais") |
| Limiar de classificação / de veredito | 0,80 / 0,95 |

---

## 9. Observações para a dissertação

Pontos que precisam constar no texto ou aparecer na defesa.

**A camada bayesiana é complemento declarado, não substituição.** O gate
continua sendo κw + Wilcoxon + McNemar, e é isso que o texto deve dizer. A
leitura bayesiana entra para afirmar o que o frequentista não afirma: que a
divergência entre juiz e humano **cabe** dentro de uma margem de irrelevância
prática. Onde o relatório frequentista escreve "não se detectou viés
sistemático", o bayesiano pode escrever "P(equivalência | ε) = 0,97" — que é uma
afirmação positiva, e mais forte. Se as duas divergirem, reporte as duas e
explique a divergência; não escolha a que favorece o resultado.

**Registre método, ROPE, limiar, amostras e semente antes de rodar.** Todos
entram no rodapé de reprodutibilidade dos relatórios. Na Likert a ROPE não é
escolha — decorre da escala —, o que elimina de saída a suspeita de margem
ajustada ao resultado.

**Reporte as contagens junto das probabilidades.** O teste de sinais responde
qual zona concentra mais massa posterior; com muitos empates, a zona central
vence mesmo havendo desequilíbrio claro entre as laterais. As contagens
`Acima`/`Empate`/`Abaixo` são o que permite ao leitor enxergar esse
desequilíbrio, e por isso vão em toda tabela bayesiana.

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

**As duas "equivalências" não são a mesma afirmação.** Na Likert, equivalência é
P(|δ| ≤ ε) — magnitude do efeito medida em proporção de documentos. No F1, é
P(a zona ROPE ser a maioritária) — qual região concentra mais casos. O texto
precisa definir cada uma onde a reporta; usar a mesma palavra sem distinguir é o
erro mais fácil de cometer aqui, e o mais difícil de defender se apontado.

**A ROPE do F1 determina o resultado, e a sensibilidade é resultado principal.**
Com BERTScore entre protocolos destilados do mesmo professor, os escores são
comprimidos e a classificação vira função da ROPE. Reporte a matriz em pelo
menos três valores e mostre onde as categorias trocam. E registre o controle
negativo — o protocolo comparado consigo mesmo saindo equivalente é o que
demonstra que a margem não foi escolhida para produzir o resultado.

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

**Opcional, só com `--bayes`:** `util_est_bayesiana.py` na mesma pasta (ou no
`PYTHONPATH`). O `baycomp` é dispensável — o módulo cai no gerador próprio, que
é numericamente equivalente (divergência máxima verificada de 10⁻⁵, compatível
com ruído Monte Carlo) e, ao contrário do pacote, honra `prior`, `prior_place` e
semente. Sem o módulo, `--bayes` aborta com mensagem explícita; sem a flag, nada
disso é importado.

