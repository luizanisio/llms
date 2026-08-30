#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

util_est_bayesiana.py
=====================

Camada fina sobre o **baycomp** para comparar protocolos par a par.

Toda a estatística é do baycomp — este módulo só organiza as chamadas, monta a
matriz de todos os pares e desenha as figuras. Nada é reimplementado.

**Um único teste em todo o pipeline: `baycomp.CorrelatedTTest`.**

Motivos:

1. **Escopo.** Benavoli et al. (2017, JMLR 18:1-36) o propõem para comparar
   dois algoritmos em **um único conjunto de dados com observações pareadas** —
   exatamente o desenho das duas etapas deste trabalho. O `SignTest` e o
   `SignedRankTest` foram propostos para o outro problema (coleções de
   conjuntos) e permanecem apenas na demonstração, como justificativa.
2. **O `SignTest` satura.** Numa escala ordinal de 4 pontos os empates passam
   de 50% rotineiramente; nessa condição nenhuma zona lateral pode superar a
   central e P(equivalência) → 1 independentemente do desequilíbrio.
3. **Determinismo.** O cálculo é analítico (Student acumulada) — sem Monte
   Carlo, sem `nsamples`, sem semente a registrar.

O teste devolve a tripla ``(p_esquerda, p_rope, p_direita)``:

    p_esquerda  probabilidade de o PRIMEIRO protocolo ser melhor
    p_rope      probabilidade de a diferença média cair DENTRO da ROPE
    p_direita   probabilidade de o SEGUNDO ser melhor

⚠ A tripla só vem completa com ``rope > 0``. Com ``rope = 0`` o baycomp devolve
apenas dois valores; por isso a ROPE é obrigatória aqui. A ROPE incide sobre a
**diferença média** dos escores pareados — na Likert, calibrada pela
divergência média entre especialistas humanos (Etapa 1); nas métricas
contínuas, pela variação entre execuções do mesmo protocolo.

**Limiar único de decisão: 0,95** em todo o trabalho (gate, heatmap e
"Medindo as diferenças") — o valor que Benavoli et al. adotam para decisões
automáticas (§3.2). Consequência assumida: mais células `incerto` do que
haveria com 0,80; é desfecho legítimo, e o gráfico de diferenças mostra o
quanto faltou em cada caso.

**Limitação assumida, a declarar no texto:** a diferença média de notas Likert
É a diferença das médias — o pareamento não contorna a objeção ordinal. A
defesa: (i) literatura de robustez (Norman, 2010; Carifio & Perla, 2008);
(ii) ROPE ancorada na divergência entre especialistas, na mesma unidade;
(iii) as **contagens ordinais puras** (`d > 0`, `d == 0`, `d < 0`) reportadas
ao lado de toda probabilidade — um número que não assume intervalos.

Uso:
    from util_est_bayesiana import Comparacao, matriz_pares, heatmap
    from util_est_bayesiana import grafico_diferencas, sintese

    c = Comparacao(notas_A, notas_B, rope=0.32)      # ROPE calibrada
    print(c.probabilidades, c.classificacao, c.ic95)

    m = matriz_pares(escores, rope=0.01)
    heatmap(m, arquivo_saida="protocolos_heatmap.png")
    grafico_diferencas(m, arquivo_saida="protocolos_diferencas.png")
    print(sintese(m))

Requisitos: baycomp, numpy, pandas, scipy, matplotlib.
Autor: Luiz Anísio
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import baycomp

__all__ = ["RELACOES", "CORES", "LIMIAR_PADRAO", "NOME_TESTE",
           "Comparacao", "matriz_pares", "sintese", "verificar_transitividade",
           "heatmap", "grafico_diferencas", "dominante"]


#: nomes das três relações, na ordem em que o baycomp as devolve
RELACOES = ("esquerda", "rope", "direita")

#: nome do único teste do pipeline, para títulos e relatórios
NOME_TESTE = "baycomp.CorrelatedTTest"

#: cor de cada estado das figuras
CORES = {
    "superior": "#2e7d4f",     # verde  — a linha supera a coluna
    "equivalente": "#2c6e91",  # azul   — dentro da ROPE
    "inferior": "#b23a48",     # vermelho
    "incerto": "#8a8a8a",      # cinza  — nenhuma alcança o limiar
}

#: probabilidade mínima para classificar uma relação. Limiar ÚNICO em todo o
#: trabalho (Benavoli et al., 2017, §3.2) — gate, heatmap e forest plot.
LIMIAR_PADRAO = 0.95


class Comparacao:
    """Comparação bayesiana pareada de dois protocolos, via `baycomp.CorrelatedTTest`.

    Args:
        x: escores do primeiro protocolo.
        y: escores do segundo, **pareados** (posição i = mesmo documento).
        rope: largura da região de equivalência prática, sobre a **diferença
            média** dos escores. Obrigatória e > 0 — sem ela o baycomp devolve
            só duas probabilidades. Na Likert, calibre pela divergência média
            entre especialistas (Etapa 1); nas métricas contínuas, pela
            variação entre execuções.
        limiar: probabilidade mínima para classificar (padrão 0,95 — limiar
            único do trabalho).

    Atributos:
        probabilidades: dicionário com p_esquerda, p_rope, p_direita.
        classificacao: "superior", "equivalente", "inferior" ou "incerto",
            sempre do ponto de vista de `x`.
        probabilidade: a maior das três (a que a classificação usou).
        diferenca_media, variancia, gl: posterior de Student da diferença
            (x − y): média, variância corrigida e graus de liberdade.
        ic95: intervalo de credibilidade de 95% da diferença média.
        contagens: contagens ORDINAIS PURAS (d > 0, d == 0, d < 0) —
            independentes da ROPE, a âncora contra a objeção da escala ordinal.
    """

    def __init__(self, x, y, rope, limiar=LIMIAR_PADRAO):
        if rope <= 0:
            raise ValueError(
                "rope deve ser > 0: com rope = 0 o baycomp devolve apenas "
                "(p_esquerda, p_direita), sem a probabilidade de equivalência. "
                "Calibre a ROPE (Likert: divergência média entre especialistas; "
                "métricas contínuas: variação entre execuções).")

        self.x = np.asarray(x, dtype=float)     # baycomp exige array 1-D
        self.y = np.asarray(y, dtype=float)
        self.rope, self.limiar = float(rope), float(limiar)

        # O CorrelatedTTest é analítico (Student acumulada): determinístico,
        # sem nsamples nem semente. O baycomp trabalha com diff = y − x; aqui
        # tudo é reportado do ponto de vista de x (positivo = x melhor).
        posterior = baycomp.CorrelatedTTest(self.x, self.y, rope=self.rope)
        p_esq, p_rope, p_dir = posterior.probs()

        self.diferenca_media = float(-posterior.mean)   # média de (x − y)
        self.variancia = float(posterior.var)           # já com Nadeau-Bengio
        self.gl = float(posterior.df)

        self.probabilidades = {"p_esquerda": float(p_esq),
                               "p_rope": float(p_rope),
                               "p_direita": float(p_dir)}
        self.classificacao, self.probabilidade = self._classificar()

    def _classificar(self):
        """Categoria dominante, se ela alcançar o limiar; senão, 'incerto'."""
        valores = {"superior": self.probabilidades["p_esquerda"],
                   "equivalente": self.probabilidades["p_rope"],
                   "inferior": self.probabilidades["p_direita"]}
        categoria = max(valores, key=valores.get)
        maior = valores[categoria]
        return (categoria if maior >= self.limiar else "incerto"), maior

    @property
    def ic95(self) -> tuple:
        """Intervalo de credibilidade de 95% da diferença média (x − y)."""
        if self.variancia == 0:
            return (self.diferenca_media, self.diferenca_media)
        margem = stats.t.ppf(0.975, self.gl) * np.sqrt(self.variancia)
        return (self.diferenca_media - float(margem),
                self.diferenca_media + float(margem))

    @property
    def contagens(self) -> dict:
        """Contagens ORDINAIS PURAS: em quantos documentos x ganha, empata e perde.

        Independentes da ROPE de propósito: a ROPE incide sobre a diferença
        **média**, não sobre cada documento. Estas contagens são a âncora
        ordinal — o número reportado ao lado de toda probabilidade que não
        assume nada sobre os intervalos da escala.
        """
        d = self.x - self.y
        return {"x_melhor": int((d > 0).sum()),
                "empate": int((d == 0).sum()),
                "y_melhor": int((d < 0).sum())}

    def como_dict(self, nome_x="A", nome_y="B") -> dict:
        """Uma linha de resultado, pronta para virar tabela."""
        ic_inf, ic_sup = self.ic95
        return {"linha": nome_x, "coluna": nome_y, "n": len(self.x),
                "rope": self.rope,
                **self.probabilidades, **self.contagens,
                "diferenca_media": self.diferenca_media,
                "variancia": self.variancia, "gl": self.gl,
                "ic_inf": ic_inf, "ic_sup": ic_sup,
                "media_linha": float(np.mean(self.x)),
                "media_coluna": float(np.mean(self.y)),
                "classificacao": self.classificacao,
                "probabilidade": self.probabilidade}

    def grafico(self, nomes=("A", "B")):
        """Gráfico da posterior desenhado pelo próprio baycomp."""
        return baycomp.CorrelatedTTest.plot(self.x, self.y, rope=self.rope,
                                            names=nomes)

    def __repr__(self):
        p = self.probabilidades
        return (f"Comparacao(rope={self.rope:g}, n={len(self.x)}) "
                f"→ {self.classificacao} "
                f"[esq {p['p_esquerda']:.3f} · rope {p['p_rope']:.3f} · "
                f"dir {p['p_direita']:.3f}] · Δ={self.diferenca_media:+.4f}")


def matriz_pares(dados, rope, nomes=None, limiar=LIMIAR_PADRAO) -> pd.DataFrame:
    """Compara todos os pares de protocolos e devolve o resultado em formato longo.

    Args:
        dados: DataFrame (uma coluna por protocolo) ou dict {nome: escores}.
            As séries são pareadas: a linha i é o mesmo caso de teste.
        rope, limiar: ver ``Comparacao``.
        nomes: subconjunto e ORDEM dos protocolos. Trave esta ordem ao gerar
            mais de uma matriz para comparar, senão as figuras não se alinham.

    Returns:
        DataFrame com uma linha por par ordenado (i, j), i ≠ j, expondo a
        posterior completa (diferença média, variância, gl, IC 95%) e as
        contagens ordinais puras.

    Cada par é comparado uma vez só; a célula espelhada troca `p_esquerda` com
    `p_direita`, garantindo simetria exata.
    """
    nomes = list(nomes) if nomes is not None else list(
        dados.columns if hasattr(dados, "columns") else dados)
    if len(nomes) < 2:
        raise ValueError("são necessários ao menos dois protocolos")

    linhas = []
    for i, a in enumerate(nomes):
        for b in nomes[i + 1:]:
            c = Comparacao(dados[a], dados[b], rope=rope, limiar=limiar)
            direto = c.como_dict(a, b)
            linhas.append(direto)
            # célula espelhada: derivada, não recalculada
            espelho = dict(direto)
            espelho.update(
                linha=b, coluna=a,
                p_esquerda=direto["p_direita"], p_direita=direto["p_esquerda"],
                x_melhor=direto["y_melhor"], y_melhor=direto["x_melhor"],
                diferenca_media=-direto["diferenca_media"],
                ic_inf=-direto["ic_sup"], ic_sup=-direto["ic_inf"],
                media_linha=direto["media_coluna"],
                media_coluna=direto["media_linha"],
                classificacao={"superior": "inferior",
                               "inferior": "superior"}.get(
                                   direto["classificacao"],
                                   direto["classificacao"]))
            linhas.append(espelho)

    matriz = pd.DataFrame(linhas)
    matriz.attrs.update(nomes=nomes, rope=float(rope), limiar=float(limiar),
                        teste=NOME_TESTE,
                        n=int(matriz["n"].iloc[0]) if len(matriz) else 0)
    return matriz


def verificar_transitividade(matriz: pd.DataFrame) -> list:
    """Ciclos nas relações direcionais de superioridade.

    Constrói o grafo dirigido `a → b` para cada par classificado como
    `superior` e procura ciclos. Se houver, a leitura ordenada da tabela de
    síntese é inválida e isso precisa ser declarado no relatório.

    Returns:
        Lista de ciclos, cada um como lista de protocolos (ex.:
        ``[["A", "B", "C"]]`` significa A > B > C > A). Vazia = sem ciclos.
    """
    arestas = {}
    for _, linha in matriz.iterrows():
        if linha["classificacao"] == "superior":
            arestas.setdefault(linha["linha"], set()).add(linha["coluna"])

    ciclos, no_caminho, visitados = [], [], set()

    def _dfs(no):
        if no in no_caminho:
            ciclo = no_caminho[no_caminho.index(no):]
            if sorted(map(str, ciclo)) not in [sorted(map(str, c)) for c in ciclos]:
                ciclos.append(list(ciclo))
            return
        if no in visitados:
            return
        visitados.add(no)
        no_caminho.append(no)
        for destino in arestas.get(no, ()):  # noqa: B023
            _dfs(destino)
        no_caminho.pop()

    for origem in list(arestas):
        _dfs(origem)
    return ciclos


def sintese(matriz: pd.DataFrame) -> pd.DataFrame:
    """Síntese por protocolo: média e contagem de relações — "Contando as relações".

    Conta relações; **não ordena**, porque relações podem ser intransitivas.
    A verificação de transitividade acompanha a tabela em
    ``tabela.attrs["ciclos"]`` — se as relações direcionais formam ciclo, a
    leitura ordenada é inválida e isso precisa ser declarado.

    Par conceitual com "Medindo as diferenças": esta tabela é categórica
    (quantas vezes cada protocolo venceu); o forest plot é quantitativo
    (quanto separa cada par).
    """
    nomes = matriz.attrs.get("nomes") or sorted(matriz["linha"].unique())
    ordem = ["superior", "equivalente", "inferior", "incerto"]
    contagem = (matriz.groupby(["linha", "classificacao"]).size()
                .unstack(fill_value=0)
                .reindex(index=nomes, columns=ordem, fill_value=0))
    medias = (matriz.drop_duplicates("linha").set_index("linha")["media_linha"]
              .reindex(nomes))
    tabela = pd.DataFrame({"média": medias.round(4)}).join(contagem)
    tabela = tabela.rename(columns={"superior": "superior a",
                                    "equivalente": "equivalente a",
                                    "inferior": "inferior a"})
    tabela.index.name = "protocolo"
    tabela.columns.name = None
    tabela.attrs["ciclos"] = verificar_transitividade(matriz)
    return tabela


def _cor(classificacao, probabilidade):
    """Matiz pela categoria, saturação pela probabilidade (escala a partir de 1/3)."""
    from matplotlib.colors import to_rgb
    base = np.array(to_rgb(CORES[classificacao]))
    fracao = float(np.clip((probabilidade - 1 / 3) / (2 / 3), 0, 1))
    # o cinza do 'incerto' fica lavado: incerteza não deve competir visualmente
    mistura = (0.10 + 0.35 * fracao) if classificacao == "incerto" else (0.16 + 0.84 * fracao)
    return tuple(1 - mistura * (1 - base)), mistura


def dominante(linha) -> str:
    """Categoria com maior probabilidade posterior, mesmo quando não decide.

    Difere de `classificacao`, que devolve "incerto" quando nenhuma alcança o
    limiar. Aqui a resposta é sempre uma das três relações — é o que permite
    dizer *para onde* a evidência apontava num par indeciso.
    """
    return max((("superior", linha["p_esquerda"]),
                ("equivalente", linha["p_rope"]),
                ("inferior", linha["p_direita"])), key=lambda par: par[1])[0]


def _cor_texto(linha, mistura):
    """Cor do número impresso na célula.

    Nas células decididas, só contraste. No estado `incerto` o número herda a
    cor da categoria **dominante**: o fundo continua cinza, porque a evidência
    não alcançou o limiar, mas a cor diz para onde a posterior apontava. Sem
    isso, duas células cinzas de naturezas opostas — uma quase equivalente,
    outra quase superior — ficam visualmente idênticas.
    """
    if linha["classificacao"] != "incerto":
        return "white" if mistura > 0.62 else "#22333b"
    return CORES[dominante(linha)]


def heatmap(matriz, arquivo_saida=None, titulo=None, rotulo="protocolo",
            casas=1, dpi=150, figsize=None):
    """Heatmap das relações: cor = categoria, número = probabilidade posterior.

    Mantido para recortes maiores, onde "Medindo as diferenças" ficaria longo
    demais. A diagonal é neutra — (Pi, Pi) não é comparação. `incerto` é
    categoria explícita: nenhuma das três probabilidades alcançou o limiar —
    o MESMO limiar 0,95 do resto do trabalho.

    Returns:
        (figura, caminho) — `caminho` é None se nada foi gravado.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    nomes = matriz.attrs.get("nomes") or sorted(matriz["linha"].unique())
    limiar = matriz.attrs.get("limiar", LIMIAR_PADRAO)
    celulas = {(r["linha"], r["coluna"]): r for _, r in matriz.iterrows()}
    k = len(nomes)

    figura, eixo = plt.subplots(figsize=figsize or (1.4 * k + 3.5, 1.15 * k + 3.2))
    eixo.set_xlim(-0.5, k - 0.5)
    eixo.set_ylim(k - 0.5, -0.5)
    eixo.set_aspect("equal")
    eixo.set_xticks(range(k), nomes, fontsize=9, rotation=30, ha="right")
    eixo.set_yticks(range(k), nomes, fontsize=9)
    eixo.tick_params(length=0)
    eixo.grid(visible=False)
    for lado in ("top", "right", "bottom", "left"):
        eixo.spines[lado].set_visible(False)

    for i, a in enumerate(nomes):
        for j, b in enumerate(nomes):
            if i == j:
                cor, texto, cor_texto, peso = (0.93, 0.93, 0.93), "—", "#22333b", "normal"
            else:
                linha = celulas[(a, b)]
                cor, mistura = _cor(linha["classificacao"], linha["probabilidade"])
                texto = f"{100 * linha['probabilidade']:.{casas}f}%".replace(".", ",")
                cor_texto = _cor_texto(linha, mistura)
                # negrito reforça a pista de cor nas células indecisas
                peso = "bold" if linha["classificacao"] == "incerto" else "normal"
            eixo.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cor,
                                     edgecolor="white", linewidth=1.6))
            eixo.text(j, i, texto, ha="center", va="center", fontsize=9.5,
                      color=cor_texto, fontweight=peso)

    eixo.set_xlabel("comparado com", fontsize=10, labelpad=8)
    eixo.set_ylabel(rotulo, fontsize=10, labelpad=8)
    subtitulo = (f"{NOME_TESTE} · "
                 + (f"ROPE = {matriz.attrs.get('rope'):g} · "
                    f"n = {matriz.attrs.get('n')} · "
                    f"limiar = {limiar:.2f}").replace(".", ","))
    eixo.annotate(subtitulo, xy=(0.5, 1.0), xycoords="axes fraction",
                  xytext=(0, 10), textcoords="offset points",
                  ha="center", va="bottom", fontsize=8.5, color="0.35")
    if titulo:
        eixo.set_title(titulo, fontsize=12, pad=32)

    rotulos = {"superior": "superior", "equivalente": "equivalente",
               "inferior": "inferior",
               "incerto": f"incerto (< {limiar:.2f}) — a cor do número indica a "
                          f"categoria dominante".replace(".", ",")}
    eixo.legend(handles=[Patch(facecolor=CORES[c], label=r) for c, r in rotulos.items()],
                loc="upper center", bbox_to_anchor=(0.5, -0.12),
                ncol=4, frameon=False, fontsize=9, handlelength=1.2)

    caminho = None
    if arquivo_saida:
        figura.tight_layout()
        figura.savefig(arquivo_saida, dpi=dpi, bbox_inches="tight", facecolor="white")
        caminho = arquivo_saida
    return figura, caminho


def _pares_nao_ordenados(matriz: pd.DataFrame):
    """Itera uma linha por par NÃO ordenado, na ordem de `nomes`."""
    vistos = set()
    for _, linha in matriz.iterrows():
        chave = frozenset((linha["linha"], linha["coluna"]))
        if chave not in vistos:
            vistos.add(chave)
            yield linha


def grafico_diferencas(matriz, arquivo_saida=None, titulo=None,
                       rotulo_escala="diferença média", casas=2,
                       dpi=150, figsize=None):
    """"Medindo as diferenças" — gráfico de floresta (*forest plot*) dos pares.

    O gráfico principal da análise: mostra magnitude e incerteza juntas, e
    torna visível *por que* cada par foi classificado. Uma linha por par
    não ordenado, com:

    * ponto na **diferença média** (x − y, positivo = primeiro melhor);
    * barra do **IC 95%** (credibilidade, da posterior de Student);
    * faixa da **ROPE** ao fundo;
    * cor pela classificação (verde superior · azul equivalente · vermelho
      inferior · cinza incerto);
    * **P(equivalência)** anotada à direita.

    Legível até ~6 protocolos (15 linhas); acima disso use o `heatmap`.

    Returns:
        (figura, caminho) — `caminho` é None se nada foi gravado.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    pares = list(_pares_nao_ordenados(matriz))
    if not pares:
        raise ValueError("matriz vazia: nada a desenhar")

    rope = float(matriz.attrs.get("rope", 0))
    limiar = matriz.attrs.get("limiar", LIMIAR_PADRAO)
    k = len(pares)

    figura, eixo = plt.subplots(figsize=figsize or (8.6, max(3.6, 0.52 * k + 2.6)))

    # faixa da ROPE ao fundo
    eixo.axvspan(-rope, rope, color="#dfe9f0", zorder=0,
                 label=f"ROPE (± {rope:g})".replace(".", ","))
    eixo.axvline(0, color="0.6", linewidth=0.9, zorder=1)

    posicoes, nomes_pares = [], []
    for idx, linha in enumerate(pares):
        y = k - 1 - idx
        cor = CORES[linha["classificacao"]]
        eixo.plot([linha["ic_inf"], linha["ic_sup"]], [y, y],
                  color=cor, linewidth=2.4, solid_capstyle="round", zorder=3)
        eixo.plot(linha["diferenca_media"], y, "o", color=cor,
                  markersize=6.5, zorder=4)
        p_eq = f"{linha['p_rope']:.3f}".replace(".", ",")
        eixo.annotate(f"P(equiv.) = {p_eq}", xy=(1.005, y),
                      xycoords=("axes fraction", "data"),
                      va="center", ha="left", fontsize=8.5,
                      color=CORES[linha["classificacao"]]
                      if linha["classificacao"] != "incerto" else "0.35")
        posicoes.append(y)
        nomes_pares.append(f"{linha['linha']} − {linha['coluna']}")

    eixo.set_yticks(posicoes, nomes_pares, fontsize=9.5)
    eixo.set_ylim(-0.7, k - 0.3)
    eixo.set_xlabel(f"{rotulo_escala} (positivo = primeiro melhor) · "
                    f"ponto = média · barra = IC 95%", fontsize=9.5)
    eixo.grid(axis="x", color="0.92", linewidth=0.8)
    eixo.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        eixo.spines[lado].set_visible(False)

    subtitulo = (f"{NOME_TESTE} · "
                 + (f"ROPE = {rope:g} · n = {matriz.attrs.get('n')} · "
                    f"limiar = {limiar:.2f}").replace(".", ","))
    eixo.annotate(subtitulo, xy=(0.5, 1.0), xycoords="axes fraction",
                  xytext=(0, 8), textcoords="offset points",
                  ha="center", va="bottom", fontsize=8.5, color="0.35")
    eixo.set_title(titulo or "Medindo as diferenças (forest plot)",
                   fontsize=12, pad=30)

    legenda = [Patch(facecolor="#dfe9f0", label=f"ROPE (± {rope:g})".replace(".", ","))]
    legenda += [Patch(facecolor=CORES[c], label=c)
                for c in ("superior", "equivalente", "inferior", "incerto")]
    # deslocamento maior quando há poucos pares, para não sobrepor o xlabel
    legenda_y = -0.22 if k <= 2 else -0.16
    eixo.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.5, legenda_y),
                ncol=5, frameon=False, fontsize=8.5, handlelength=1.2)

    caminho = None
    if arquivo_saida:
        figura.tight_layout()
        figura.savefig(arquivo_saida, dpi=dpi, bbox_inches="tight",
                       facecolor="white")
        caminho = arquivo_saida
    return figura, caminho


if __name__ == "__main__":
    # Demonstração mínima com dados fictícios; a comparação SignTest ×
    # CorrelatedTTest (a justificativa da escolha) vive em demo_est_bayesiana.py.
    rng = np.random.default_rng(42)
    n = 300
    base = rng.normal(0, 1, n)
    efeitos = {"P1": 0.0, "P2": 0.05, "P3": 0.60, "P4": -0.45}

    likert = pd.DataFrame({p: np.clip(np.round(2.5 + base + v + rng.normal(0, 0.5, n)), 1, 4)
                           for p, v in efeitos.items()})
    m = matriz_pares(likert, rope=0.32)   # ROPE calibrada (exemplo)
    print(f"LIKERT ({NOME_TESTE}, ROPE = 0,32 calibrada, limiar = {LIMIAR_PADRAO})")
    print(m[["linha", "coluna", "p_esquerda", "p_rope", "p_direita",
             "diferenca_media", "ic_inf", "ic_sup",
             "classificacao"]].round(3).to_string(index=False))
    tabela = sintese(m)
    print("\n", tabela.to_string(), sep="")
    print("ciclos de transitividade:", tabela.attrs["ciclos"] or "nenhum")

    f1 = pd.DataFrame({p: np.clip(0.90 + 0.02 * (base + v) + rng.normal(0, 0.01, n), 0, 1)
                       for p, v in efeitos.items()})
    mf = matriz_pares(f1, rope=0.006)
    print(f"\nF1 ({NOME_TESTE}, ROPE = 0,006)")
    print(mf[["linha", "coluna", "p_esquerda", "p_rope", "p_direita",
              "classificacao"]].round(3).to_string(index=False))

    heatmap(m, "demo_likert_heatmap.png", titulo="Likert — heatmap")
    grafico_diferencas(m, "demo_likert_diferencas.png",
                       titulo="Likert — Medindo as diferenças (forest plot)")
    heatmap(mf, "demo_f1_heatmap.png", titulo="F1 — heatmap")
    grafico_diferencas(mf, "demo_f1_diferencas.png",
                       titulo="F1 — Medindo as diferenças (forest plot)")
    print("\nfiguras: demo_likert_heatmap.png, demo_likert_diferencas.png, "
          "demo_f1_heatmap.png, demo_f1_diferencas.png")
