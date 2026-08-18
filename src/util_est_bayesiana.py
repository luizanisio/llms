#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
util_est_bayesiana.py
=====================

Camada fina sobre o **baycomp** para comparar protocolos par a par.

Toda a estatística é do baycomp — este módulo só organiza as chamadas, monta a
matriz de todos os pares e desenha o heatmap. Nada é reimplementado.

Os três testes do pacote, e quando usar cada um:

    ┌──────────┬──────────────────────┬───────────────────────────────────────┐
    │ método   │ classe do baycomp    │ quando                                │
    ├──────────┼──────────────────────┼───────────────────────────────────────┤
    │ "sinais" │ SignTest             │ escala ORDINAL (Likert 1–4): só conta │
    │          │                      │ direções, ignora magnitude            │
    │ "postos" │ SignedRankTest       │ intermediário: usa a ordenação das    │
    │          │                      │ magnitudes                            │
    │ "t"      │ CorrelatedTTest      │ escala CONTÍNUA (BERTScore F1): usa   │
    │          │                      │ as magnitudes; cálculo analítico      │
    └──────────┴──────────────────────┴───────────────────────────────────────┘

Todos devolvem a mesma tripla ``(p_esquerda, p_rope, p_direita)``:

    p_esquerda  probabilidade de o PRIMEIRO protocolo ser melhor
    p_rope      probabilidade de a diferença ser praticamente irrelevante
    p_direita   probabilidade de o SEGUNDO ser melhor

⚠ A tripla só vem completa com ``rope > 0``. Com ``rope = 0`` o baycomp devolve
apenas dois valores; por isso a ROPE é obrigatória aqui.

Uso:
    from util_est_bayesiana import Comparacao, matriz_pares, heatmap

    # Likert 1–4: rope = 0,5 significa "notas iguais" (a escala é inteira)
    c = Comparacao(notas_A, notas_B, rope=0.5, metodo="sinais")
    print(c.probabilidades, c.classificacao)

    # F1: rope calibrada, teste t correlacionado
    m = matriz_pares(escores_f1, rope=0.006, metodo="t")
    heatmap(m, arquivo_saida="protocolos_f1.png")

Requisitos: baycomp, numpy, pandas, matplotlib.
Autor: Luiz Anísio
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import baycomp

__all__ = ["METODOS", "RELACOES", "CORES", "LIMIAR_PADRAO",
           "Comparacao", "matriz_pares", "resumo", "heatmap", "dominante"]


#: método → classe do baycomp
METODOS = {
    "sinais": baycomp.SignTest,
    "postos": baycomp.SignedRankTest,
    "t": baycomp.CorrelatedTTest,
}

#: nomes das três relações, na ordem em que o baycomp as devolve
RELACOES = ("esquerda", "rope", "direita")

#: cor de cada estado do heatmap
CORES = {
    "superior": "#2e7d4f",     # verde  — a linha supera a coluna
    "equivalente": "#2c6e91",  # azul   — dentro da ROPE
    "inferior": "#b23a48",     # vermelho
    "incerto": "#8a8a8a",      # cinza  — nenhuma alcança o limiar
}

#: probabilidade mínima para classificar uma relação
LIMIAR_PADRAO = 0.80


class Comparacao:
    """Comparação bayesiana pareada de dois protocolos, via baycomp.

    Args:
        x: escores do primeiro protocolo.
        y: escores do segundo, **pareados** (posição i = mesmo documento).
        rope: largura da região de equivalência prática, sobre os escores
            brutos. Obrigatória e > 0 — sem ela o baycomp devolve só duas
            probabilidades. Na Likert inteira, use 0,5 (equivale a "notas
            iguais"); no F1, calibre empiricamente.
        metodo: "sinais", "postos" ou "t" — ver o cabeçalho do módulo.
        limiar: probabilidade mínima para classificar (padrão 0,80).
        nsamples: amostras da posterior. Ignorado por "t", que é analítico.
        seed: semente. Idem.

    Atributos:
        probabilidades: dicionário com p_esquerda, p_rope, p_direita.
        classificacao: "superior", "equivalente", "inferior" ou "incerto",
            sempre do ponto de vista de `x`.
        probabilidade: a maior das três (a que a classificação usou).
    """

    def __init__(self, x, y, rope, metodo="sinais", limiar=LIMIAR_PADRAO,
                 nsamples=50_000, seed=42):
        if metodo not in METODOS:
            raise ValueError(f"método deve ser um de {tuple(METODOS)}")
        if rope <= 0:
            raise ValueError(
                "rope deve ser > 0: com rope = 0 o baycomp devolve apenas "
                "(p_esquerda, p_direita), sem a probabilidade de equivalência. "
                "Na escala Likert inteira use rope=0.5.")

        self.x = np.asarray(x, dtype=float)     # baycomp exige array 1-D
        self.y = np.asarray(y, dtype=float)
        self.rope, self.metodo, self.limiar = float(rope), metodo, float(limiar)

        teste = METODOS[metodo]
        if metodo == "t":
            # CorrelatedTTest é analítico (Student acumulada): não aceita
            # nsamples nem semente, e o resultado é determinístico
            p_esq, p_rope, p_dir = teste.probs(self.x, self.y, rope=self.rope)
        else:
            p_esq, p_rope, p_dir = teste.probs(
                self.x, self.y, rope=self.rope,
                nsamples=nsamples, random_state=seed)

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
    def diferenca_media(self):
        """Média de (x − y). Positiva = o primeiro protocolo é melhor."""
        return float(np.mean(self.x - self.y))

    @property
    def contagens(self):
        """Documentos em que x ganha, empata (dentro da ROPE) e perde."""
        d = self.x - self.y
        return {"x_melhor": int((d > self.rope).sum()),
                "empate": int((np.abs(d) <= self.rope).sum()),
                "y_melhor": int((d < -self.rope).sum())}

    def como_dict(self, nome_x="A", nome_y="B"):
        """Uma linha de resultado, pronta para virar tabela."""
        return {"linha": nome_x, "coluna": nome_y, "n": len(self.x),
                "metodo": self.metodo, "rope": self.rope,
                **self.probabilidades, **self.contagens,
                "diferenca_media": self.diferenca_media,
                "classificacao": self.classificacao,
                "probabilidade": self.probabilidade}

    def grafico(self, nomes=("A", "B")):
        """Gráfico da posterior desenhado pelo próprio baycomp."""
        teste = METODOS[self.metodo]
        return teste.plot(self.x, self.y, rope=self.rope, names=nomes)

    def __repr__(self):
        p = self.probabilidades
        return (f"Comparacao({self.metodo}, rope={self.rope:g}, n={len(self.x)}) "
                f"→ {self.classificacao} "
                f"[esq {p['p_esquerda']:.3f} · rope {p['p_rope']:.3f} · "
                f"dir {p['p_direita']:.3f}]")


def matriz_pares(dados, rope, metodo="sinais", nomes=None,
                 limiar=LIMIAR_PADRAO, nsamples=50_000, seed=42):
    """Compara todos os pares de protocolos e devolve o resultado em formato longo.

    Args:
        dados: DataFrame (uma coluna por protocolo) ou dict {nome: escores}.
            As séries são pareadas: a linha i é o mesmo caso de teste.
        rope, metodo, limiar, nsamples, seed: ver ``Comparacao``.
        nomes: subconjunto e ORDEM dos protocolos. Trave esta ordem ao gerar
            mais de uma matriz para comparar, senão as figuras não se alinham.

    Returns:
        DataFrame com uma linha por par ordenado (i, j), i ≠ j.

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
            c = Comparacao(dados[a], dados[b], rope=rope, metodo=metodo,
                           limiar=limiar, nsamples=nsamples, seed=seed)
            direto = c.como_dict(a, b)
            linhas.append(direto)
            # célula espelhada: derivada, não recalculada
            espelho = dict(direto)
            espelho.update(
                linha=b, coluna=a,
                p_esquerda=direto["p_direita"], p_direita=direto["p_esquerda"],
                x_melhor=direto["y_melhor"], y_melhor=direto["x_melhor"],
                diferenca_media=-direto["diferenca_media"],
                classificacao={"superior": "inferior",
                               "inferior": "superior"}.get(
                                   direto["classificacao"],
                                   direto["classificacao"]))
            linhas.append(espelho)

    matriz = pd.DataFrame(linhas)
    matriz.attrs.update(nomes=nomes, rope=float(rope), metodo=metodo,
                        limiar=float(limiar),
                        n=int(matriz["n"].iloc[0]) if len(matriz) else 0)
    return matriz


def resumo(matriz):
    """Quantas vezes cada protocolo é superior, equivalente, inferior ou incerto.

    Conta relações; não produz ranking, porque relações podem ser intransitivas.
    """
    nomes = matriz.attrs.get("nomes") or sorted(matriz["linha"].unique())
    ordem = ["superior", "equivalente", "inferior", "incerto"]
    tabela = (matriz.groupby(["linha", "classificacao"]).size()
              .unstack(fill_value=0)
              .reindex(index=nomes, columns=ordem, fill_value=0))
    tabela.index.name = "protocolo"
    tabela.columns.name = None
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

    A diagonal é neutra — (Pi, Pi) não é comparação. `incerto` é categoria
    explícita: nenhuma das três probabilidades alcançou o limiar.

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
    subtitulo = (f"baycomp · {matriz.attrs.get('metodo')} · "
                 f"ROPE = {matriz.attrs.get('rope'):g} · "
                 f"n = {matriz.attrs.get('n')} · "
                 f"limiar = {limiar:.2f}").replace(".", ",")
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


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 300
    base = rng.normal(0, 1, n)
    efeitos = {"P1": 0.0, "P2": 0.05, "P3": 0.60, "P4": -0.45}

    likert = pd.DataFrame({p: np.clip(np.round(2.5 + base + v + rng.normal(0, 0.5, n)), 1, 4)
                           for p, v in efeitos.items()})
    m = matriz_pares(likert, rope=0.5, metodo="sinais")
    print("LIKERT (SignTest, rope=0,5 = notas iguais)")
    print(m[["linha", "coluna", "p_esquerda", "p_rope", "p_direita",
             "classificacao"]].round(3).to_string(index=False))
    print("\n", resumo(m).to_string(), sep="")

    f1 = pd.DataFrame({p: np.clip(0.90 + 0.02 * (base + v) + rng.normal(0, 0.01, n), 0, 1)
                       for p, v in efeitos.items()})
    mf = matriz_pares(f1, rope=0.006, metodo="t")
    print("\nF1 (CorrelatedTTest, ROPE=0,006)")
    print(mf[["linha", "coluna", "p_esquerda", "p_rope", "p_direita",
              "classificacao"]].round(3).to_string(index=False))

    heatmap(m, "demo_likert.png", titulo="Likert — SignTest")
    heatmap(mf, "demo_f1.png", titulo="F1 — CorrelatedTTest")
    print("\nfiguras: demo_likert.png, demo_f1.png")