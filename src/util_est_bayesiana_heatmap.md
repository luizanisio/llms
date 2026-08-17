# Heatmap de comparação bayesiana

Guia de leitura e uso da figura produzida por `heatmap_relacoes` /
`heatmap_comparacao`, em `util_est_bayesiana.py`.

A figura resume, numa única matriz, a comparação pareada de todos os pares de um
conjunto de protocolos (ou avaliadores, ou fontes). Cada célula responde a uma
pergunta sobre **um par**; a matriz inteira mostra o **padrão** — quem supera
vários, quem é predominantemente equivalente, e onde a evidência não fecha.

Onde é usada:

| Contexto | Entrada | Quem chama |
|---|---|---|
| Fase A — validação do juiz LLM | notas medianas por avaliador | `realizar_avaliacoes.py --bayes` |
| Fase B — comparação entre protocolos | Likert do juiz e métricas automáticas | `comparar_extracoes_baycomp.py` |

---

## 1. O que a figura afirma — e o que não afirma

Cada célula `(linha, coluna)` carrega três probabilidades posteriores, que somam
1, sobre a relação **da linha em relação à coluna**:

| quantidade | significado |
|---|---|
| `p_superior` | a linha supera a coluna |
| `p_equivalente` | as duas são praticamente equivalentes |
| `p_inferior` | a linha é superada pela coluna |

A convenção é sempre **maior é melhor**. Se o escore comparado for um erro (ou
qualquer métrica em que menor seja melhor), inverta os argumentos ao construir a
comparação — a figura não tem como detectar isso.

**A figura não produz ranking.** Com relações possivelmente intransitivas,
ordenar os protocolos por número de vitórias criaria uma ordem que os dados não
sustentam. A tabela auxiliar `resumo_relacoes` conta relações por protocolo
justamente para localizar padrões *sem* transformá-los em posição.

**A figura não é um teste de hipótese.** Não há p-valor, não há H₀ rejeitada. O
que se lê é probabilidade posterior: "dados os documentos observados, qual a
probabilidade de esta relação valer". Por isso `equivalente` é um achado
positivo, e não uma falha em detectar diferença.

---

## 2. Anatomia da figura

```
                    Comparação bayesiana entre protocolos — Likert
              métrica: Likert (juiz LLM) · análise principal        ← identificação
     cor = relação da linha em relação à coluna · número = prob.    ← como ler
        ε = 0,05 (proporção de documentos) · n = 4.000 ·            ← parâmetros
                  empates = 43% · limiar = 0,80

              A         B         C        D1
     A       ---      100,0%    100,0%   100,0%     ← linha: relação de A
     B     100,0%      ---       98,1%   100,0%
     C     100,0%     98,1%       ---    100,0%
    D1     100,0%    100,0%     100,0%     ---
                      comparado com

     ■ superior   ■ equivalente   ■ inferior   ■ incerto (< 0,80)
```

| elemento | o que comunica |
|---|---|
| **cor** | a **categoria** da relação (ver §3) |
| **intensidade** | a **magnitude** da probabilidade dominante |
| **número** | a mesma probabilidade, explícita, em percentual |
| **diagonal** | cinza-claro com `—`: `(Pi, Pi)` não é comparação |
| **`n/d`** | célula branca: o par não existe na matriz recebida |
| **subtítulo** | modo, margem, `n`, massa de empates e limiar |
| **legenda** | os quatro estados, com o limiar declarado |

### As cores

| estado | cor | código |
|---|---|---|
| superior | verde | `#2e7d4f` |
| equivalente | azul | `#2c6e91` |
| inferior | vermelho | `#b23a48` |
| incerto | cinza | `#8a8a8a` |

### A escala de intensidade

A saturação **não** começa em zero: começa em 1/3 — a probabilidade de uma
relação sorteada ao acaso entre as três. Sem isso, toda a faixa onde a leitura
de fato acontece (0,74 a 0,95) sairia visualmente quase idêntica.

As células `incerto` recebem uma escala deliberadamente mais lavada (no máximo
~45% de saturação, contra 100% das demais). É uma decisão de projeto: incerteza
não deve competir visualmente com as relações que a evidência sustenta.

### A cor do texto

Em células classificadas, o número fica branco sobre fundo escuro e escuro sobre
fundo claro.

Em células **`incerto`**, o número é colorido pela relação *dominante* — verde
escuro se `p_superior` lidera, vermelho escuro se `p_inferior`, azul escuro se
`p_equivalente`. É informação que a cor de fundo (cinza) descarta: a célula não
alcançou o limiar, mas a evidência ainda aponta para algum lado. Quando as duas
maiores probabilidades ficam a menos de 0,01 uma da outra, não há direção a
indicar e o texto volta ao cinza-escuro neutro.

### O subtítulo

Não é decoração. Sem ele os números da figura não se interpretam:

- **o modo** — sem ele, as duas "equivalências" se confundem (§4);
- **a margem** (ε ou ROPE) — é o que define a faixa central;
- **`n`** — governa a saturação de `P(direção)` (§6);
- **a massa de empates** — ε = 0,08 é apertado ou frouxo *dependendo de quanta
  massa sobra fora da zona de empate*. Com 70% de empates, δ fica limitado a
  ±0,30 e um ε de 0,08 já consome mais de um quarto da faixa disponível;
- **o limiar** — o corte que separou `incerto` do resto.

---

## 3. Como uma célula é classificada

```
classificação = a maior das três probabilidades,
                se ela for >= limiar E for única
              caso contrário: incerto
```

O limiar padrão é **0,80** (`LIMIAR_CLASSIFICACAO_PADRAO`).

`incerto` é **categoria própria**, não uma quarta relação: significa ausência de
evidência suficiente, e não a existência de algo intermediário entre superior,
equivalente e inferior. O número impresso continua sendo a probabilidade
dominante, e informa quão perto do limiar a evidência chegou.

Empates exatos entre as duas maiores resolvem para `incerto` — a leitura neutra.
A regra é deliberada: a figura nunca infere uma classificação que as
probabilidades não sustentam.

---

## 4. Os dois modos afirmam coisas diferentes

Esta é a distinção que mais custa caro se passar despercebida. As três
probabilidades podem sair da posterior por dois caminhos, e os rótulos coincidem
**sem que o significado coincida**.

| modo | `p_equivalente` significa | usado em |
|---|---|---|
| `proporcao` | `P(\|δ\| ≤ ε)` — a vantagem, **medida em proporção de documentos**, não passa de ε | escalas ordinais (Likert), com `rope = 0` |
| `baycomp` | `P(a zona ROPE ser a maioritária)` — cálculo padrão do pacote | escores contínuos (F1, BERTScore), com `rope > 0` |

O primeiro afirma algo sobre **magnitude do efeito**. O segundo, sobre **qual
região concentra mais documentos**.

A diferença é observável. Com 40% dos documentos acima, 35% dentro da ROPE e 25%
abaixo, o modo `baycomp` devolve `superior`, ainda que um terço dos documentos
esteja dentro da margem de irrelevância — enquanto o modo `proporcao`, com
ε = 0,20, olharia δ = 0,15 e diria `equivalente`. Nenhum está errado: respondem
perguntas diferentes.

Por isso o rótulo da faixa central **muda na legenda**: no modo `baycomp` ele
aparece como `ROPE maioritária`, não como `equivalente`. E por isso o modo é
gravado na matriz e impresso na figura — não existe padrão implícito seguro.

**Duas guardas no código**, que valem conhecer porque produzem erro em vez de
resultado silencioso:

- `modo="baycomp"` com `rope = 0` **levanta `ValueError`**. A zona central
  recolheria apenas os empates exatos e o triplet degeneraria em `(0, 1, 0)` —
  "equivalente" em todas as células, com aparência de resultado.
- `modo="proporcao"` com `ε = 0` emite **aviso**: a faixa de equivalência tem
  medida nula e nenhuma célula poderá ser classificada como equivalente.

---

## 5. Simetria: por construção, não por coincidência

Cada par **não ordenado** é amostrado uma única vez. A célula espelhada é
*derivada* trocando `p_inferior` por `p_superior` (e invertendo o sinal de δ e do
intervalo de credibilidade).

Consequências práticas:

- `P(Pi > Pj)` e `P(Pj < Pi)` são **o mesmo número**, não duas estimativas Monte
  Carlo que por acaso coincidem;
- o custo computacional cai pela metade;
- **a simetria da figura não serve como verificação de consistência dos dados.**
  Ela é garantida pela construção e apareceria mesmo se a entrada estivesse
  errada. Para verificar a estabilidade dos números, use `estabilidade_mc()` e
  `sensibilidade_prior()` em `ComparacaoPareada`.

---

## 6. Roteiro de leitura

1. **Leia o subtítulo primeiro.** Modo, margem, `n`, empates e limiar. Sem esses
   cinco números a matriz não se interpreta.
2. **Varra por linhas.** Uma linha predominantemente verde indica um protocolo
   que supera vários outros; predominantemente azul, um que é praticamente
   equivalente a vários.
3. **Localize o cinza.** São as comparações em que a evidência não fechou —
   desfecho legítimo, a ser reportado como tal.
4. **Olhe os números, não só as cores.** Duas células verdes com 82,1% e 99,9%
   dizem coisas bem diferentes.
5. **Vá à tabela par a par** para o que a figura não cabe: contagens de
   documentos, δ com intervalo de credibilidade e ε crítico.

### Armadilhas

**`P(direção)` satura com `n` grande.** Com milhares de documentos, a
probabilidade de dominância vai a 1 mesmo para diferenças triviais. Uma matriz
inteira de "100,0%" não significa que todos os protocolos sejam muito diferentes
entre si — significa que há dados suficientes para ter certeza da direção,
qualquer que seja a magnitude. **O conteúdo científico, nesse caso, está nas
contagens e no δ com intervalo**, não na probabilidade. A leitura descritiva
gerada em `analise_bayesiana.md` sinaliza os pares nessa situação.

**Direção confiável e magnitude trivial coexistem.** `P(A > B)` e `P(equiv.)`
podem ser ambas altas: A vence de forma confiável, por uma margem irrelevante.
Não é contradição — é exatamente a situação que o teste de hipótese nula não
consegue expressar, e a razão de reportar as duas quantidades.

**Dois limiares, dois usos.** O heatmap classifica a 0,80, adequado para
descrever o panorama. Vereditos (validação do juiz, contrastes pré-registrados)
exigem 0,95. Um par pode aparecer `equivalente` na figura e `inconclusivo` no
veredito: são perguntas com exigências diferentes, não uma inconsistência.

**No modo `baycomp`, a ROPE determina o resultado.** Com escores comprimidos
(caso típico de BERTScore e SBERT), uma ROPE pequena demais esvazia a zona
central e satura a figura em verde/vermelho; grande demais engole tudo e ela fica
azul. A transição entre os extremos é rápida. Reportar a matriz em três ROPEs
(`sensibilidade_margem`) é o mínimo defensável.

**Métricas de similaridade com um modelo de referência medem fidelidade, não
qualidade.** Quando a comparação é contra um professor/gabarito, um protocolo que
reproduz fielmente um erro do professor é premiado. Essas figuras entram como
triangulação, nunca como veredito — e a marcação `análise complementar` aparece
no topo delas.

**Nunca escolha a margem depois de ver a figura.** As curvas e varreduras de
sensibilidade existem para demonstrar que a conclusão **não** depende de um
número escolhido a dedo. Lê-las e então adotar o valor que produz o resultado
desejado é escolher a conclusão e inventar o critério depois — a versão bayesiana
do *p-hacking*, e detectável.

---

## 7. Como gerar

### Caminho de uma chamada

```python
from util_est_bayesiana import heatmap_comparacao

matriz, caminho = heatmap_comparacao(
    notas,                       # DataFrame: uma coluna por protocolo, linhas pareadas
    nomes=["A", "B", "C", "D1"], # recorte E ordem das linhas/colunas
    eps=0.05, rope=0.0,          # margem — ver §4
    limiar=0.80,
    modo="proporcao",
    metrica="Likert", papel="principal",
    nsamples=200_000, seed=42,
    rotacao_x=30,
    arquivo_saida="bayes_likert.png")
```

### Caminho em duas etapas

Útil quando a mesma matriz alimenta figura, tabelas e análises de sensibilidade —
que é o caso normal, porque reamostrar seria desperdício:

```python
from util_est_bayesiana import (matriz_relacoes, heatmap_relacoes,
                                resumo_relacoes, sensibilidade_limiar)

matriz = matriz_relacoes(notas, nomes=[...], eps=0.05, modo="proporcao",
                         metrica="Likert", papel="principal",
                         nsamples=200_000, seed=42)
heatmap_relacoes(matriz, arquivo_saida="bayes_likert.png")
resumo_relacoes(matriz)          # contagens por protocolo
sensibilidade_limiar(matriz)     # reclassifica sem reamostrar
```

### Parâmetros que mudam a leitura da figura

| parâmetro | efeito |
|---|---|
| `nomes` | recorte e **ordem** das linhas/colunas |
| `limiar` | corte de `incerto`; padrão vem de `matriz.attrs`, ou 0,80 |
| `metrica`, `papel` | identificam a figura fora de contexto, no topo |
| `referencia` | destaca um nome em negrito nos dois eixos e no subtítulo |
| `rotulo_entidade` | `"protocolo"`, `"avaliador"`, `"fonte"` — eixo Y e título |
| `casas` | decimais do percentual impresso (padrão 1) |
| `subtitulo` | substitui o subtítulo automático (use com cuidado: ele carrega os parâmetros) |
| `rotacao_x` | rotaciona os rótulos do eixo X; necessário com nomes longos |
| `eixo` | desenha num eixo existente, para compor painéis |

> **Trave a ordem com `nomes` sempre que for gerar mais de uma matriz.** Os
> heatmaps de métricas diferentes são feitos para serem lidos lado a lado; com as
> linhas em ordens diferentes, a comparação visual induz leitura errada. Em
> `comparar_extracoes_baycomp.py` isso é garantido pela chave `protocolos` do
> YAML, que define recorte **e** ordem.

O tamanho da figura é calculado a partir do número de protocolos e do
comprimento dos rótulos; `figsize` só é necessário para casos fora do comum.

---

## 8. O que a matriz traz além da figura

`matriz_relacoes` devolve um DataFrame em **formato longo**, uma linha por par
*ordenado*. A figura usa quatro dessas colunas; as demais são o que sustenta a
discussão no texto.

| coluna | conteúdo |
|---|---|
| `linha`, `coluna` | os dois protocolos do par |
| `p_inferior`, `p_equivalente`, `p_superior` | as três probabilidades |
| `classificacao`, `probabilidade` | o que a figura desenha |
| `p_dom` | `P(θ_esq > θ_dir)` — direção pura, **não usa margem** |
| `delta`, `ic_inf`, `ic_sup` | δ médio e intervalo de credibilidade 95% |
| `contagem_superior/empate/inferior` | documentos em cada zona |
| `eps_critico` | menor ε que estabeleceria equivalência — um **tamanho de efeito** |
| `n`, `massa_empate` | tamanho e massa da zona de empate |
| `eps`, `rope`, `modo`, `origem` | parâmetros da execução |

Os parâmetros globais ficam em `matriz.attrs` e são lidos pelo heatmap.
**O pandas descarta `attrs` em várias operações** — quem filtrar ou serializar a
matriz precisa recopiá-los, senão a figura perde o subtítulo.

---

## 9. Figuras e tabelas companheiras

| ferramenta | responde |
|---|---|
| `resumo_relacoes` | quantas vezes cada protocolo é superior/equivalente/inferior/incerto |
| `grafico_curva_sensibilidade_eps` | a qual ε mínimo cada par precisaria para ser equivalente |
| `sensibilidade_limiar` | quantas células mudam de categoria ao variar o limiar |
| `sensibilidade_margem` | idem, ao variar a margem (ε ou ROPE) |
| `tabela_convergencia` | onde duas métricas concordam, divergem ou não decidem |
| `calibrar_rope` + `controle_negativo` | ancorar a ROPE empiricamente e validá-la |

Sobre o custo, que é assimétrico e importa no planejamento: **o ε atua sobre a
posterior já amostrada** — variar o limiar ou percorrer a curva de ε sai
praticamente de graça. **A ROPE atua sobre os escores brutos** e muda as
contagens: cada valor exige nova amostragem.

Sobre `tabela_convergencia`: divergir **não é erro**. Métricas diferentes podem
capturar propriedades distintas do desempenho. O que não vale é deixar a
divergência implícita para o leitor descobrir comparando duas figuras a olho.

---

## 10. O que declarar ao reportar

- o **método**: teste de sinais bayesiano pareado (Benavoli et al., 2017, JMLR
  18:1-36), com o pacote `baycomp` como implementação de referência;
- que a amostragem da posterior foi **reimplementada** em `util_est_bayesiana`
  para expor as amostras e permitir análise de sensibilidade ao prior e
  reprodutibilidade, com equivalência numérica verificada;
- o **modo** (`proporcao` ou `baycomp`) e a **margem** correspondente, com a
  origem do valor — calibrado empiricamente ou pré-registrado;
- o **limiar de classificação** e, se diferente, o limiar dos vereditos;
- `nsamples` e `seed`;
- as **análises de sensibilidade** ao limiar e à margem;
- os pares em que `P(direção)` saturou, com as contagens e o δ correspondentes.

---

## Nota de implementação — por que matplotlib

Os requisitos originais desta figura apontavam para Plotly, pela interatividade e
pelo *hover* com as três probabilidades completas. A implementação usa
**matplotlib**, por três razões que continuam valendo:

1. o destino das figuras é **PNG embutido em relatórios Markdown e na
   dissertação** — um artefato estático, versionável e imprimível;
2. o *hover* seria o único lugar das três probabilidades completas, e elas
   precisam existir num artefato citável: ficam na matriz em CSV, ao lado da
   figura, junto com as contagens e o δ com intervalo;
3. o pipeline já gera todas as demais figuras em matplotlib, e uma segunda
   dependência gráfica não se pagaria.

A exigência de fundo — **categórica *e* quantitativa ao mesmo tempo**, e não um
gradiente contínuo de escala única — foi mantida integralmente:

```
categoria    → cor
probabilidade → intensidade + valor numérico
incerteza    → categoria explícita, visualmente contida
```
