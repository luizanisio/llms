#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

realizar_avaliacoes_graficos.py
===============================

Camada de visualização de ``realizar_avaliacoes.py``. **Não faz nenhuma conta**:
recebe os DataFrames já calculados e desenha.

Duas fontes de gráficos:

* ``util_graficos.UtilGraficos`` — gráficos herdados do pipeline anterior
  (distribuições, boxplot, barras com IC). Se o pacote não estiver disponível,
  esses gráficos são pulados com aviso e a estatística segue normalmente.
* ``matplotlib`` direto — matriz de confusão e distribuição das diferenças, as
  duas figuras da validação.
* ``util_est_bayesiana`` — heatmap da comparação bayesiana, quando a etapa é
  ativada. O desenho mora lá porque a mesma figura serve à comparação entre
  protocolos de treinamento, que não passa por este pipeline.

Cada função recebe o dicionário de resultados e a pasta de saída, e devolve a
lista de arquivos gerados.

Autor: Luiz Anísio
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.extend(["../../src", "../../../src"])
try:
    from util_graficos import Cores, UtilGraficos
    UTIL_DISPONIVEL = True
except ImportError:  # o pipeline estatístico não depende deste pacote
    Cores = UtilGraficos = None
    UTIL_DISPONIVEL = False

try:
    import util_est_bayesiana as est
    BAYES_DISPONIVEL = True
except ImportError:  # só é exigido quando a etapa bayesiana é ativada
    est = None
    BAYES_DISPONIVEL = False

PISO_ADEQUACAO = 3

#: paleta dos gráficos matplotlib deste módulo
COR_PRIMARIA = "#2c6e91"
COR_SECUNDARIA = "#c26a3d"
COR_NEUTRA = "#8a8a8a"
MAPA_CALOR = LinearSegmentedColormap.from_list("azuis", ["#f5f8fa", "#2c6e91", "#153b4f"])

#: cores por avaliador (identidade), em ORDEM FIXA — atribuídas pela posição do
#: avaliador, nunca recicladas. Validada sobre fundo branco para barras lado a lado
#: (pares adjacentes): daltonismo ΔE ≥ 9,1 e visão normal ΔE ≥ 19,6 (OKLab ×100).
#: Verde-água, amarelo e rosa ficam abaixo de 3:1 de contraste com o fundo; por
#: isso os gráficos que usam esta paleta mantêm os valores escritos nas barras.
#: Para trocar as cores, revalide a nova ordem (CVD ≥ 8 e visão normal ≥ 15).
PALETA_AVALIADORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                      "#e87ba4", "#008300", "#6250d6", "#e34948"]
#: cor da referência (ex.: avaliação humana): cinza-tinta fora da paleta, para a
#: referência se destacar dos juízes (ΔE ≥ 20 de todas as cores da paleta)
COR_REFERENCIA = "#52514e"

FIG_DPI = 150


# =============================================================================
# Utilidades
# =============================================================================

def _sanitizar(texto: str) -> str:
    """Transforma um rótulo em nome de arquivo seguro."""
    texto = texto.replace("×", "x")
    texto = re.sub(r"[^0-9a-zA-Z_-]+", "_", texto).strip("_").lower()
    return texto or "sem_nome"


def _salvar(fig, caminho: str, gerados: list) -> None:
    """Grava a figura, fecha e registra o nome do arquivo."""
    fig.tight_layout()
    fig.savefig(caminho, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gerados.append(os.path.basename(caminho))


def _estilo(ax, titulo: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Estilo comum: sem molduras supérfluas, grade discreta no eixo de valores."""
    ax.set_title(titulo, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)


def _aviso_util(nome: str) -> None:
    print(f"  ⚠ util_graficos indisponível — gráfico '{nome}' não gerado.")


def _registrar(caminho, gerados: list) -> None:
    """Registra o retorno de UtilGraficos, que nem sempre devolve o caminho."""
    if caminho:
        gerados.append(os.path.basename(str(caminho)))


def _heatmap_matriz(tabela: pd.DataFrame, titulo: str, xlabel: str, ylabel: str,
                    caminho: str, gerados: list, percentual: bool = False) -> None:
    """Heatmap de uma matriz de contagens, com anotação em cada célula."""
    valores = tabela.to_numpy(dtype=float)
    total = valores.sum() or 1.0

    fig, ax = plt.subplots(figsize=(1.15 * tabela.shape[1] + 3.2,
                                    1.0 * tabela.shape[0] + 2.8))
    imagem = ax.imshow(valores, cmap=MAPA_CALOR, aspect="auto")
    limite = valores.max() * 0.55 if valores.max() else 1.0
    for i in range(tabela.shape[0]):
        for j in range(tabela.shape[1]):
            valor = valores[i, j]
            texto = f"{int(valor)}"
            if percentual:
                texto += f"\n{100 * valor / total:.1f}%".replace(".", ",")
            ax.text(j, i, texto, ha="center", va="center", fontsize=9,
                    color="white" if valor > limite else "#22333b")
    ax.set_xticks(range(tabela.shape[1]))
    ax.set_xticklabels(tabela.columns, fontsize=9)
    ax.set_yticks(range(tabela.shape[0]))
    ax.set_yticklabels(tabela.index, fontsize=9)
    ax.grid(visible=False)
    for lado in ("top", "right", "bottom", "left"):
        ax.spines[lado].set_visible(False)
    ax.set_title(titulo, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    barra = fig.colorbar(imagem, ax=ax, shrink=0.82)
    barra.set_label("Itens", fontsize=9)
    barra.outline.set_visible(False)
    _salvar(fig, caminho, gerados)


# =============================================================================
# Gráficos da análise interna de um grupo
# =============================================================================

def graficos_grupo(r: dict, saida: str) -> list:
    """Gera as figuras da análise interna de um grupo."""
    gerados = []
    rot = r["grupo"].rotulos
    _distribuicoes(r, saida, gerados, rot)
    _viabilidade(r, saida, gerados)
    _confusao_avaliadores(r, saida, gerados, rot)
    _ranks(r, saida, gerados)
    _complementares(r, saida, gerados)
    return sorted(set(gerados))


def _distribuicoes(r: dict, saida: str, gerados: list, rot: dict) -> None:
    """01–03: distribuição das notas por avaliador e da variável primária."""
    if not UTIL_DISPONIVEL:
        _aviso_util("distribuições")
        return
    df, pivo = r["df"], r["pivo"]
    grade = {
        f"{rot['avaliador']} {a}": {
            f: df[(df["avaliador"] == a) & (df["fonte"] == f) & df["nota"].notna()]["nota"].tolist()
            for f in r["fontes"]}
        for a in r["avaliadores"]}
    _registrar(UtilGraficos.grafico_grade_distribuicao(
        grade, categorias=r["categorias"], x="Nota Likert", y="% de itens",
        titulo_geral=f"Distribuição das notas — {rot['avaliadores']} (linhas) × fontes (colunas)",
        paleta_cores=Cores.RdYlGn,
        arquivo_saida=os.path.join(saida, "01_distribuicao_notas.png")), gerados)

    UtilGraficos.gerar_boxplot(
        {f: pivo[f].tolist() for f in pivo.columns},
        titulo="Nota mediana por fonte (variável primária)",
        ylabel="Nota Likert mediana", xlabel="Fonte",
        paleta_cores=Cores.Set2, rotacao_labels=0,
        nota=f"n = {len(pivo)} documentos pareados",
        arquivo_saida=os.path.join(saida, "02_boxplot_medianas.png"))
    gerados.append("02_boxplot_medianas.png")

    _registrar(UtilGraficos.grafico_grade_distribuicao(
        {"Nota mediana": {f: pivo[f].tolist() for f in pivo.columns}},
        categorias=r["categorias"], x="Nota Likert mediana", y="% de documentos",
        titulo_geral="Variável primária — distribuição da nota mediana por fonte",
        paleta_cores=Cores.RdYlGn,
        arquivo_saida=os.path.join(saida, "03_distribuicao_medianas.png")), gerados)


def _viabilidade(r: dict, saida: str, gerados: list) -> None:
    """04: proporção de documentos adequados com IC 95% Wilson."""
    if not UTIL_DISPONIVEL:
        _aviso_util("viabilidade")
        return
    descritivas = r["descritivas"]
    _registrar(UtilGraficos.gerar_barras_ic(
        categorias=list(descritivas.index),
        valores=descritivas[f"P(mediana ≥ {PISO_ADEQUACAO})"].tolist(),
        ic_inferior=descritivas["IC95 inf"].tolist(),
        ic_superior=descritivas["IC95 sup"].tolist(),
        titulo=f"Viabilidade — proporção com mediana ≥ {PISO_ADEQUACAO} (IC 95% Wilson)",
        ylabel="Proporção de documentos", xlabel="Fonte",
        linha_referencia=0.80, rotulo_referencia="Critério: LI do IC ≥ 0,80",
        ylim=(0, 1.12), paleta_cores=Cores.Blues,
        arquivo_saida=os.path.join(saida, "04_viabilidade.png")), gerados)


def _confusao_avaliadores(r: dict, saida: str, gerados: list, rot: dict) -> None:
    """05: matriz de confusão entre os dois primeiros avaliadores, todas as fontes."""
    avaliadores, categorias = r["avaliadores"], r["categorias"]
    if len(avaliadores) < 2:
        return
    a1, a2 = avaliadores[0], avaliadores[1]
    pivo = (r["df"].pivot_table(index=["documento", "fonte"], columns="avaliador",
                                values="nota", aggfunc="median")
            .reindex(columns=[a1, a2]).dropna())
    if pivo.empty:
        return
    tabela = pd.crosstab(pivo[a1].astype(int), pivo[a2].astype(int)) \
               .reindex(index=categorias, columns=categorias, fill_value=0)
    _heatmap_matriz(tabela,
                    f"Concordância entre {rot['avaliador'].lower()}s {a1} e {a2} "
                    "(todas as fontes)",
                    f"{rot['avaliador']} {a2}", f"{rot['avaliador']} {a1}",
                    os.path.join(saida, "05_confusao_avaliadores.png"), gerados)


def _ranks(r: dict, saida: str, gerados: list) -> None:
    """06: ranks médios do Friedman entre fontes."""
    if not UTIL_DISPONIVEL:
        _aviso_util("ranks")
        return
    _registrar(UtilGraficos.gerar_grafico_barras(
        r["comparacao"]["ranks_medios"].to_frame("Rank médio"),
        titulo="Ranks médios entre as fontes (1 = melhor desempenho)",
        ylabel="Rank médio", xlabel="Fonte",
        paleta_cores=Cores.Cividis, mostrar_valores=True,
        arquivo_saida=os.path.join(saida, "06_ranks_friedman.png")), gerados)


def _complementares(r: dict, saida: str, gerados: list) -> None:
    """07–09: problemas apontados, custo e taxa de falhas."""
    if not UTIL_DISPONIVEL:
        _aviso_util("problemas/custo/falhas")
        return
    if not r["problemas"].empty:
        _registrar(UtilGraficos.gerar_grafico_barras(
            r["problemas"].transpose(),
            titulo="Categorias de problema apontadas (todos os avaliadores)",
            ylabel="Ocorrências", xlabel="Fonte",
            paleta_cores=Cores.Dark2, mostrar_valores=False,
            arquivo_saida=os.path.join(saida, "07_problemas.png")), gerados)

    if not r["custos"].empty and "Total tokens" in r["custos"].columns:
        tokens = r["custos"].pivot(index="fonte", columns="avaliador", values="Total tokens")
        _registrar(UtilGraficos.gerar_grafico_barras(
            tokens.reindex([f for f in r["fontes"] if f in tokens.index]),
            titulo="Consumo total de tokens por fonte e avaliador",
            ylabel="Tokens", xlabel="Fonte", paleta_cores=Cores.PuBuGn,
            mostrar_valores=False,
            arquivo_saida=os.path.join(saida, "08_custo.png")), gerados)

    colunas_taxa = [c for c in r["falhas"].columns if c.endswith("taxa %")]
    if colunas_taxa and r["falhas"][colunas_taxa].to_numpy().sum() > 0:
        _registrar(UtilGraficos.gerar_grafico_barras(
            r["falhas"][colunas_taxa],
            titulo="Taxa de falhas por fonte e avaliador",
            ylabel="Falhas (%)", xlabel="Fonte",
            paleta_cores=Cores.Plasma, mostrar_valores=True,
            arquivo_saida=os.path.join(saida, "09_falhas.png")), gerados)


# =============================================================================
# Heatmap da comparação bayesiana
# =============================================================================

def grafico_bayes(matriz, caminho: str, titulo: str,
                  rotulo_entidade: str = "protocolo", **opcoes) -> list:
    """Heatmap das relações posteriores — desenho delegado ao `util_est_bayesiana`.

    A figura vive lá porque a mesma é usada pela comparação entre protocolos de
    treinamento, que não passa por este pipeline. Aqui fica só a ponte: nome do
    arquivo, rótulo do eixo e registro do que foi gerado — mantendo a regra do
    módulo: desenhar, nunca calcular.

    Returns:
        Lista com o nome do arquivo gerado, ou vazia se o módulo não estiver
        disponível ou a matriz vier vazia.
    """
    if not BAYES_DISPONIVEL:
        _aviso_util("heatmap bayesiano")
        return []
    if matriz is None or len(matriz) == 0:
        return []
    _, gerado = est.heatmap(matriz, arquivo_saida=caminho, titulo=titulo,
                            rotulo=rotulo_entidade, dpi=FIG_DPI, **opcoes)
    plt.close("all")
    return [os.path.basename(gerado)] if gerado else []


def grafico_diferencas_bayes(matriz, caminho: str, titulo: str, **opcoes) -> list:
    """"Medindo as diferenças" (forest plot) — desenho delegado ao `util_est_bayesiana`.

    Uma linha por par: ponto na diferença média, barra do IC 95%, faixa da
    ROPE ao fundo, cor pela classificação e P(equivalência) anotada. Mesma
    regra do heatmap: a figura vive no módulo bayesiano porque também serve à
    comparação entre protocolos; aqui fica só a ponte.

    Returns:
        Lista com o nome do arquivo gerado, ou vazia se o módulo não estiver
        disponível ou a matriz vier vazia.
    """
    if not BAYES_DISPONIVEL:
        _aviso_util("forest plot bayesiano")
        return []
    if matriz is None or len(matriz) == 0:
        return []
    _, gerado = est.grafico_diferencas(matriz, arquivo_saida=caminho,
                                       titulo=titulo, dpi=FIG_DPI, **opcoes)
    plt.close("all")
    return [os.path.basename(gerado)] if gerado else []


# =============================================================================
# Gráficos da validação (matplotlib)
# =============================================================================

def graficos_validacao(v: dict, saida: str) -> list:
    """Figuras da validação: confusão por juiz, diferenças e distribuição por grupo."""
    gerados = []
    for gate in v["gates"]:
        _heatmap_matriz(
            gate["confusao"],
            f"{gate['juiz']} × {gate['referencia']} — todas as fontes",
            gate["referencia"], gate["juiz"],
            os.path.join(saida, f"01_confusao_{_sanitizar(gate['juiz'])}.png"),
            gerados, percentual=True)
    _diferencas(v, saida, gerados)
    _distribuicao_grupos(v, saida, gerados)
    _decisao_rope_taxa(v, saida, gerados)
    _decisao_aprovacoes(v, saida, gerados)
    return sorted(set(gerados))


#: cores das quatro células da decisão binária (juiz × referência)
CORES_DECISAO = {
    "ambos aprovam": "#2e7d4f",
    "ambos reprovam": "#6b8fa3",
    "só o juiz aprova (FP)": "#b23a48",
    "só o juiz reprova (FN)": "#eda100",
}


def _decisao_rope_taxa(v: dict, saida: str, gerados: list) -> None:
    """08: critério 3 do gate — diferença da taxa de aprovação (juiz − referência).

    Uma linha por juiz: ponto na diferença da taxa de adequação (nota ≥ piso),
    barra do IC 95% da posterior, faixa da ROPE da taxa ao fundo e
    P(equiv. decisão) anotada. Cor pelo desfecho do critério: equivalente
    (P ≥ limiar), viés (P(fora da ROPE num sentido) ≥ limiar) ou inconclusivo.
    """
    gates = v["gates"]
    if not gates:
        return
    rope, limiar = float(v["rope_decisao"]), float(v["limiar"])
    k = len(gates)
    fig, ax = plt.subplots(figsize=(8.6, max(3.2, 0.55 * k + 2.4)))
    ax.axvspan(-rope, rope, color="#dfe9f0", zorder=0)
    ax.axvline(0, color="0.6", linewidth=0.9, zorder=1)

    cores = {"equivalente": "#2c6e91", "viés": "#b23a48", "inconclusivo": "#8a8a8a"}
    for idx, g in enumerate(gates):
        y = k - 1 - idx
        if g["p_equiv_decisao"] >= limiar:
            desfecho = "equivalente"
        elif max(g["p_decisao_maior"], g["p_decisao_menor"]) >= limiar:
            desfecho = "viés"
        else:
            desfecho = "inconclusivo"
        cor = cores[desfecho]
        ic_inf, ic_sup = g["ic_decisao"]
        ax.plot([ic_inf, ic_sup], [y, y], color=cor, linewidth=2.4,
                solid_capstyle="round", zorder=3)
        ax.plot(g["dif_taxa"], y, "o", color=cor, markersize=6.5, zorder=4)
        p_eq = f"{g['p_equiv_decisao']:.3f}".replace(".", ",")
        ax.annotate(f"P(equiv.) = {p_eq}", xy=(1.005, y),
                    xycoords=("axes fraction", "data"), va="center", ha="left",
                    fontsize=8.5, color=cor if desfecho != "inconclusivo" else "0.35")

    ax.set_yticks(range(k), [g["juiz"] for g in reversed(gates)], fontsize=9.5)
    ax.set_ylim(-0.7, k - 0.3)
    ax.grid(axis="x", color="0.92", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    referencia = gates[0]["referencia"]
    ax.set_xlabel(f"diferença na taxa de aprovação (juiz − {referencia}; "
                  f"positivo = juiz aprova mais) · ponto = média · barra = IC 95%",
                  fontsize=9.5)
    subtitulo = (f"aprovado = nota ≥ {PISO_ADEQUACAO} · ROPE (taxa) = ± {rope:.4f} · "
                 f"n = {gates[0]['n']} · limiar = {limiar:.2f}").replace(".", ",")
    ax.annotate(subtitulo, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 8),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8.5, color="0.35")
    ax.set_title("Critério 3 — decisão prática: taxa de aprovação × referência",
                 fontsize=12, pad=30)
    from matplotlib.patches import Patch
    legenda = [Patch(facecolor="#dfe9f0", label="ROPE (taxa)")]
    legenda += [Patch(facecolor=c, label=n) for n, c in cores.items()]
    ax.legend(handles=legenda, loc="upper center",
              bbox_to_anchor=(0.5, -0.22 if k <= 2 else -0.16),
              ncol=4, frameon=False, fontsize=8.5, handlelength=1.2)
    _salvar(fig, os.path.join(saida, "08_decisao_rope_taxa.png"), gerados)


def _decisao_aprovacoes(v: dict, saida: str, gerados: list) -> None:
    """09: de onde vem a diferença da taxa — as quatro células da decisão binária.

    Barras horizontais empilhadas (% dos itens pareados) por juiz: ambos
    aprovam, ambos reprovam, só o juiz aprova (falso positivo: aprova o que a
    referência reprovou) e só o juiz reprova (falso negativo). Mostra o que a
    diferença da taxa esconde: FP e FN em quantidades parecidas se anulam na
    taxa, mas continuam sendo discordâncias.
    """
    gates = v["gates"]
    if not gates:
        return
    k = len(gates)
    fig, ax = plt.subplots(figsize=(8.6, max(3.0, 0.55 * k + 2.2)))
    for idx, g in enumerate(gates):
        y = k - 1 - idx
        n = g["n"]
        partes = [("ambos aprovam", g["aprovam_ambos"]),
                  ("ambos reprovam", g["reprovam_ambos"]),
                  ("só o juiz aprova (FP)", g["b01"]),
                  ("só o juiz reprova (FN)", g["b10"])]
        esquerda = 0.0
        for nome, qtd in partes:
            pct = 100 * qtd / n if n else 0.0
            ax.barh(y, pct, left=esquerda, color=CORES_DECISAO[nome],
                    edgecolor="white", linewidth=0.8, height=0.62,
                    label=nome if idx == 0 else None)
            if pct >= 4:
                ax.text(esquerda + pct / 2, y, str(qtd), ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
            esquerda += pct
        discord = g["b01"] + g["b10"]
        ax.annotate(f"discordam: {discord} ({100 * discord / n:.1f}%)".replace(".", ","),
                    xy=(1.005, y), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=8.5, color="0.35")
    ax.set_yticks(range(k), [g["juiz"] for g in reversed(gates)], fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, k - 0.3)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    referencia = gates[0]["referencia"]
    ax.set_xlabel(f"% dos {gates[0]['n']} itens pareados · número na barra = itens",
                  fontsize=9.5)
    ax.set_title(f"Aprovações e reprovações: juiz × {referencia} "
                 f"(aprovado = nota ≥ {PISO_ADEQUACAO})", fontsize=12, pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16 if k > 2 else -0.24),
              ncol=4, frameon=False, fontsize=8.5, handlelength=1.2)
    _salvar(fig, os.path.join(saida, "09_decisao_aprovacoes.png"), gerados)


def _estilos_avaliadores(nomes: list, referencia=None) -> dict:
    """{nome: (cor, hachura, rótulo da legenda)} de cada avaliador.

    A referência recebe COR_REFERENCIA e o rótulo "(referência)"; os demais recebem
    PALETA_AVALIADORES na ordem em que aparecem, sem reciclar cores. Acima de 8
    avaliadores (além da referência), os excedentes repetem a cor com hachura,
    para continuarem distinguíveis.
    """
    estilos, slot = {}, 0
    for nome in nomes:
        if nome == referencia:
            estilos[nome] = (COR_REFERENCIA, None, f"{nome} (referência)")
            continue
        volta, posicao = divmod(slot, len(PALETA_AVALIADORES))
        estilos[nome] = (PALETA_AVALIADORES[posicao], "//" if volta else None, nome)
        slot += 1
    return estilos


def _distribuicao_grupos(v: dict, saida: str, gerados: list) -> None:
    """03: distribuição das notas atribuídas por cada grupo, nos itens pareados.

    Torna visível a severidade relativa: um avaliador mais leniente concentra
    massa nas categorias altas.
    """
    longo, categorias = v["longo"], v["categorias"]
    nomes = [n for n in longo.columns if n not in ("documento", "fonte")]
    if not nomes:
        return
    estilos = _estilos_avaliadores(nomes, v.get("referencia"))
    largura = 0.8 / len(nomes)

    fig, ax = plt.subplots(figsize=(1.7 * len(categorias) + 3.4, 4.8))
    base = np.arange(len(categorias))
    for i, nome in enumerate(nomes):
        serie = longo[nome].astype(int)
        percentuais = [100 * float((serie == c).mean()) for c in categorias]
        posicoes = base + i * largura - 0.4 + largura / 2
        cor, hachura, rotulo = estilos[nome]
        # Borda branca: separa barras vizinhas e dá cor à hachura (quando houver)
        ax.bar(posicoes, percentuais, width=largura * 0.9, color=cor,
               hatch=hachura, edgecolor="white", linewidth=0.6, label=rotulo)
        for x, pct in zip(posicoes, percentuais):
            if pct > 0.5:
                ax.text(x, pct + 1.0, f"{pct:.1f}".replace(".", ","),
                        ha="center", fontsize=8)
    if PISO_ADEQUACAO in categorias:
        ax.axvline(categorias.index(PISO_ADEQUACAO) - 0.5, color="#b23a48",
                   linestyle="--", linewidth=1.1, alpha=0.7)
        ax.plot([], [], color="#b23a48", linestyle="--",
                label=f"piso de adequação (≥ {PISO_ADEQUACAO})")
    ax.set_xticks(base)
    ax.set_xticklabels([f"Nota {c}" for c in categorias], fontsize=9.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.06)  # folga para os valores acima das barras
    # Legenda abaixo do eixo: dentro dele, colide com as barras mais altas
    ax.legend(frameon=False, fontsize=9, ncol=min(len(nomes) + 1, 4),
              loc="upper center", bbox_to_anchor=(0.5, -0.09))
    _estilo(ax, f"Distribuição das notas por avaliador "
                f"({len(longo)} itens pareados)", "", "% de itens")
    _salvar(fig, os.path.join(saida, "03_distribuicao_por_grupo.png"), gerados)


def _diferencas(v: dict, saida: str, gerados: list) -> None:
    """02: distribuição das diferenças de nota entre cada juiz e a referência."""
    gates = v["gates"]
    if not gates:
        return
    fig, eixos = plt.subplots(1, len(gates), figsize=(4.8 * len(gates), 4.2), squeeze=False)
    for eixo, gate in zip(eixos[0], gates):
        diferencas = np.asarray(gate["diferencas"], dtype=int)
        valores, contagens = np.unique(diferencas, return_counts=True)
        percentuais = 100 * contagens / contagens.sum()
        cores = [COR_NEUTRA if val == 0 else (COR_PRIMARIA if val > 0 else COR_SECUNDARIA)
                 for val in valores]
        eixo.bar(valores, percentuais, color=cores, width=0.62)
        for val, pct in zip(valores, percentuais):
            eixo.text(val, pct + 1.2, f"{pct:.1f}%".replace(".", ","),
                      ha="center", fontsize=8.5)
        eixo.set_xticks(valores)
        eixo.set_ylim(0, min(105, percentuais.max() * 1.25))
        _estilo(eixo, f"{gate['juiz']} × {gate['referencia']}",
                f"diferença ({gate['juiz']} − {gate['referencia']})", "% de itens")
    fig.suptitle("Direção e magnitude das discordâncias", fontsize=12.5, y=1.02)
    _salvar(fig, os.path.join(saida, "02_distribuicao_diferencas.png"), gerados)
