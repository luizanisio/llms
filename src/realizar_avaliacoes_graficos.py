#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

PISO_ADEQUACAO = 3

#: paleta dos gráficos matplotlib deste módulo
COR_PRIMARIA = "#2c6e91"
COR_SECUNDARIA = "#c26a3d"
COR_NEUTRA = "#8a8a8a"
MAPA_CALOR = LinearSegmentedColormap.from_list("azuis", ["#f5f8fa", "#2c6e91", "#153b4f"])

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
    return sorted(set(gerados))


def _distribuicao_grupos(v: dict, saida: str, gerados: list) -> None:
    """03: distribuição das notas atribuídas por cada grupo, nos itens pareados.

    Torna visível a severidade relativa: um avaliador mais leniente concentra
    massa nas categorias altas.
    """
    longo, categorias = v["longo"], v["categorias"]
    nomes = [n for n in longo.columns if n not in ("documento", "fonte")]
    if not nomes:
        return
    cores = [COR_PRIMARIA, COR_SECUNDARIA, "#3f7d54", COR_NEUTRA]
    largura = 0.8 / len(nomes)

    fig, ax = plt.subplots(figsize=(1.7 * len(categorias) + 3.4, 4.4))
    base = np.arange(len(categorias))
    for i, nome in enumerate(nomes):
        serie = longo[nome].astype(int)
        percentuais = [100 * float((serie == c).mean()) for c in categorias]
        posicoes = base + i * largura - 0.4 + largura / 2
        ax.bar(posicoes, percentuais, width=largura * 0.9,
               color=cores[i % len(cores)], label=nome)
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
    ax.legend(frameon=False, fontsize=9, ncol=min(len(nomes) + 1, 4))
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
