#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src
=====================

Consolida as avaliações do juiz LLM (Likert 1--4) executado em múltiplas rodadas
independentes, gera os gráficos comparativos e o relatório `estatisticas.md` com as
tabelas exigidas pelo arcabouço estatístico do experimento (Etapas 0--8 da receita
operacional).

Entrada
-------
Uma pasta por rodada, cada uma contendo `saida_juiz_llm.parquet`::

    ./saida_01/saida_juiz_llm.parquet   -> Rodada 1
    ./saida_02/saida_juiz_llm.parquet   -> Rodada 2
    ./saida_03/saida_juiz_llm.parquet   -> Rodada 3

Esquema esperado do parquet (colunas):
    chave     : '<id_documento>_<modelo>'  (ex.: '285434817_qwen7b')
    resumo    : JSON com metadados da chamada
                {"prompt_tokens", "completion_tokens", "total_tokens",
                 "finish_reason", "model", "tempo"}
    resposta  : JSON com o julgamento {"nota": 1..4, "problemas": ["alucinacao", ...]}
    erro      : string de erro (vazia/nula quando a avaliação foi bem-sucedida)

Saída (pasta `analise_llm/`)
----------------------------
    estatisticas.md                     relatório completo com tabelas e leituras
    dados_longo.csv                     base longa (documento x modelo x rodada)
    dados_notas_medianas.csv            matriz pareada documento x modelo (variável primária)
    01_grade_distribuicao_notas.png     grade rodadas (linhas) x modelos (colunas)
    02_boxplot_notas_medianas.png       dispersão da nota mediana por modelo
    03_distribuicao_notas_medianas.png  distribuição da variável primária
    04_viabilidade_producao.png         proporção mediana >= 3 com IC 95% Wilson
    05_concordancia_rodadas.png         Kappa ponderado par a par entre rodadas
    06_matriz_confusao_rodadas.png      estabilidade teste-reteste (R1 x R2)
    07_problemas_por_modelo.png         categorias de problema apontadas pelo juiz
    08_ranks_friedman.png               ranks médios do omnibus de Friedman
    09_custo_juiz.png                   tokens e tempo por rodada/modelo
    10_falhas_por_modelo.png            taxa de falhas de parsing (Etapa 0)

Uso
---
    python gerar_graficos_llm.py
    python gerar_graficos_llm.py --base ./resultados --saida ./analise_llm
    python gerar_graficos_llm.py --pastas saida_01 saida_02 saida_03 --escala 1 4
    python gerar_graficos_llm.py --pastas saida_01 saida_02 saida_03 
    python gerar_graficos_llm.py --pastas saida_b01 saida_b02 saida_b03 
    python gerar_graficos_llm.py --pastas saida_m01 saida_m02 saida_m03 

Requisitos: pandas, numpy, scipy, matplotlib, seaborn, pyarrow, util_graficos.py
"""

from __future__ import annotations

import sys
sys.path.extend(['../../src','../../../src'])
from util_graficos import Cores, UtilGraficos

import argparse
import json
import difflib
import os
import re
import sys
import unicodedata
from datetime import datetime
from itertools import combinations
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from util_graficos import Cores, UtilGraficos

# =============================================================================
# Configuração
# =============================================================================

PASTAS_PADRAO = ["saida_01", "saida_02", "saida_03"]
ARQUIVO_PARQUET = "saida_juiz_llm.parquet"
PASTA_SAIDA_PADRAO = "analise_llm"

#: separa `chave` em documento + modelo. O id do documento não contém '_'.
REGEX_CHAVE = re.compile(r"^(?P<documento>[^_]+)_(?P<modelo>.+)$")

#: ordem preferencial de exibição dos modelos/protocolos (os não listados vão ao final)
ORDEM_MODELOS = ["a", "base", "qwen7b", "b", "c", "d1", "d2", "d3", "d4",
                 "qwen235b", "gpt5", "sabia4"]

#: limiares do arcabouço estatístico
LIMIAR_KAPPA = 0.60        # Etapas 1.1/1.2/1.3 -- kappa_w >= 0,60
LIMIAR_ALFA = 0.05         # Etapas 2/3/4 -- nível de significância
PISO_ADEQUACAO = 3         # Etapa 6 -- nota mediana >= 3
LIMIAR_VIABILIDADE = 0.80  # Etapa 6 -- limite inferior do IC 95% Wilson

ESCALA_PADRAO = (1, 4)     # Likert de 4 pontos

#: taxonomia fechada de problemas prevista na rubrica do juiz. Rótulos fora desta
#: lista são preservados e sinalizados como desvio de aderência à rubrica.
CATEGORIAS_PROBLEMA = ["alucinacao", "erro_factual", "atribuicao_errada", "omissao"]

#: variantes conhecidas que devem ser unificadas antes da contagem
ALIASES_PROBLEMA = {
    "atribucao_errada": "atribuicao_errada",
    "atribuicao_incorreta": "atribuicao_errada",
    "erro_fatual": "erro_factual",
    "erro_factico": "erro_factual",
    "alucinacoes": "alucinacao",
    "omissoes": "omissao",
    "omissao_relevante": "omissao",
}

#: acima deste valor a concordância esperada por acaso domina o Kappa, produzindo
#: o "paradoxo do Kappa" (concordância observada alta com κ baixo)
LIMIAR_PE_PARADOXO = 0.85

#: rótulo da linha que agrega todas as combinações documento x fonte (nível instrumento)
ROTULO_INSTRUMENTO = "TODOS (instrumento)"


# =============================================================================
# 1. Carga dos dados
# =============================================================================

def _texto(valor) -> str:
    """Converte um valor em string limpa, tratando None/NaN/'nan' como vazio."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    texto = str(valor).strip()
    return "" if texto.lower() in {"nan", "none", "null", "<na>"} else texto


def _json_seguro(texto) -> dict:
    """Converte texto em dict tolerando ``None``, cercas markdown e JSON malformado."""
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return {}
    if isinstance(texto, dict):
        return texto
    bruto = str(texto).strip()
    if not bruto:
        return {}
    # remove cercas ```json ... ```
    if bruto.startswith("```"):
        bruto = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", bruto).strip()
    try:
        valor = json.loads(bruto)
        return valor if isinstance(valor, dict) else {}
    except json.JSONDecodeError:
        # última tentativa: primeiro objeto JSON balanceado do texto
        ini, fim = bruto.find("{"), bruto.rfind("}")
        if 0 <= ini < fim:
            try:
                valor = json.loads(bruto[ini:fim + 1])
                return valor if isinstance(valor, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def normalizar_problema(rotulo) -> str:
    """
    Normaliza um rótulo de problema para a taxonomia canônica.

    Trata quatro fontes de ruído observadas nas saídas do juiz:
    1. escapes unicode quebrados (`atribuiu00e7u00e3o_errada` → `atribuição_errada`);
    2. acentuação inconsistente (`atribuição_errada` → `atribuicao_errada`);
    3. variantes grafadas de forma diferente (`atribucao_errada`);
    4. separadores e caixa variáveis (`Erro Factual`, `erro-factual`).

    Rótulos que não colapsam em nenhuma categoria da rubrica são devolvidos
    normalizados (sem alias), para que apareçam no relatório de aderência.
    """
    texto = _texto(rotulo)
    if not texto:
        return ""

    # 1. escapes unicode que perderam a barra invertida em algum ponto do pipeline
    if re.search(r"u[0-9a-fA-F]{4}", texto):
        texto = re.sub(r"u([0-9a-fA-F]{4})",
                       lambda m: chr(int(m.group(1), 16)), texto)

    # 2 e 4. caixa, acentos e separadores
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    texto = re.sub(r"[^a-z0-9]+", "_", texto).strip("_")
    if not texto:
        return ""

    # 3. variantes conhecidas e, em seguida, aproximação com a taxonomia canônica
    texto = ALIASES_PROBLEMA.get(texto, texto)
    if texto not in CATEGORIAS_PROBLEMA:
        proximos = difflib.get_close_matches(texto, CATEGORIAS_PROBLEMA, n=1, cutoff=0.85)
        if proximos:
            texto = proximos[0]
    return texto


def _para_lista(valor) -> list:
    """Normaliza o campo `problemas` (lista, string ou nulo) para lista de strings."""
    if valor is None:
        return []
    if isinstance(valor, (list, tuple, np.ndarray, pd.Series)):
        return [str(v).strip() for v in valor if str(v).strip()]
    texto = str(valor).strip()
    return [texto] if texto else []


def carregar_rodadas(base: str,
                     pastas: Iterable[str],
                     leitor: Callable[[str], pd.DataFrame] = pd.read_parquet) -> pd.DataFrame:
    """
    Lê o parquet de cada pasta e devolve a base longa consolidada.

    Args:
        base: diretório raiz que contém as pastas das rodadas
        pastas: nomes das pastas, na ordem das rodadas (1, 2, 3, ...)
        leitor: função de leitura (injetável para testes)

    Returns:
        DataFrame com uma linha por (documento, modelo, rodada) e as colunas
        derivadas: nota, problemas, falha, motivo_falha, tokens e tempo.
    """
    registros = []
    for indice, pasta in enumerate(pastas, start=1):
        caminho = os.path.join(base, pasta, ARQUIVO_PARQUET)
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

        bruto = leitor(caminho)
        faltantes = {"chave", "resposta"} - set(bruto.columns)
        if faltantes:
            raise ValueError(f"{caminho}: colunas ausentes {sorted(faltantes)}")

        parcial = _normalizar_rodada(bruto, rodada=indice, origem=pasta)
        registros.append(parcial)
        print(f"  ✔ {pasta}: {len(parcial):,} avaliações "
              f"({parcial['falha'].sum():,} falhas)".replace(",", "."))

    return pd.concat(registros, ignore_index=True)


def _normalizar_rodada(bruto: pd.DataFrame, rodada: int, origem: str) -> pd.DataFrame:
    """Expande as colunas JSON de uma rodada para o formato analítico longo."""
    linhas = []
    for reg in bruto.to_dict("records"):
        chave = _texto(reg.get("chave"))
        match = REGEX_CHAVE.match(chave)
        documento = match.group("documento") if match else chave
        modelo = match.group("modelo") if match else "desconhecido"

        erro = _texto(reg.get("erro"))
        resposta = _json_seguro(reg.get("resposta"))
        meta = _json_seguro(reg.get("resumo"))

        nota_bruta = resposta.get("nota", None)
        try:
            nota = int(round(float(nota_bruta)))
        except (TypeError, ValueError):
            nota = None

        # Etapa 0: uma avaliação é falha se houve erro registrado, se a resposta
        # não pôde ser interpretada ou se a nota não veio no JSON.
        if erro:
            motivo = "erro_registrado"
        elif not resposta:
            motivo = "json_invalido"
        elif nota is None:
            motivo = "nota_ausente"
        else:
            motivo = ""

        linhas.append({
            "rodada": rodada,
            "origem": origem,
            "chave": chave,
            "documento": documento,
            "modelo": modelo,
            "nota": nota,
            "problemas_brutos": _para_lista(resposta.get("problemas")),
            "problemas": [p for p in
                          (normalizar_problema(x) for x in _para_lista(resposta.get("problemas")))
                          if p],
            "falha": bool(motivo),
            "motivo_falha": motivo,
            "erro": erro,
            "juiz": meta.get("model", ""),
            "finish_reason": meta.get("finish_reason", ""),
            "prompt_tokens": pd.to_numeric(meta.get("prompt_tokens"), errors="coerce"),
            "completion_tokens": pd.to_numeric(meta.get("completion_tokens"), errors="coerce"),
            "total_tokens": pd.to_numeric(meta.get("total_tokens"), errors="coerce"),
            "tempo": pd.to_numeric(meta.get("tempo"), errors="coerce"),
        })
    return pd.DataFrame(linhas)


def ordenar_modelos(modelos: Iterable[str]) -> list:
    """Ordena os modelos pela ordem preferencial; desconhecidos vão ao final, alfabéticos."""
    modelos = list(dict.fromkeys(modelos))
    conhecidos = [m for m in ORDEM_MODELOS if m in modelos]
    restantes = sorted(m for m in modelos if m not in conhecidos)
    return conhecidos + restantes


# =============================================================================
# 2. Etapa 0 -- falhas de parsing e descarte global pareado
# =============================================================================

def aplicar_descarte_global(df: pd.DataFrame, modelos: list, rodadas: list) -> tuple:
    """
    Aplica o descarte global pareado exigido por Friedman/Wilcoxon.

    Mantém apenas os documentos com nota válida em **todas** as combinações
    modelo x rodada. Documentos com falha em qualquer célula são removidos de
    todos os protocolos, preservando o pareamento estrito.

    Returns:
        (df_valido, docs_mantidos, docs_descartados)
    """
    esperado = len(modelos) * len(rodadas)
    validos = df[(~df["falha"]) & df["nota"].notna()]
    contagem = validos.groupby("documento")[["modelo", "rodada"]].apply(
        lambda g: len(set(zip(g["modelo"], g["rodada"])))
    )
    completos = set(contagem[contagem == esperado].index)
    todos = set(df["documento"])
    return (df[df["documento"].isin(completos)].copy(),
            sorted(completos),
            sorted(todos - completos))


def tabela_falhas(df: pd.DataFrame, modelos: list, rodadas: list) -> pd.DataFrame:
    """Taxa de falhas de parsing por modelo e rodada (métrica de robustez em produção)."""
    linhas = []
    for modelo in modelos:
        registro = {"modelo": modelo}
        sub_modelo = df[df["modelo"] == modelo]
        for rodada in rodadas:
            sub = sub_modelo[sub_modelo["rodada"] == rodada]
            n = len(sub)
            falhas = int(sub["falha"].sum())
            registro[f"R{rodada} (n)"] = n
            registro[f"R{rodada} falhas"] = falhas
            registro[f"R{rodada} taxa %"] = round(100.0 * falhas / n, 2) if n else np.nan
        n_total = len(sub_modelo)
        f_total = int(sub_modelo["falha"].sum())
        registro["Total falhas"] = f_total
        registro["Taxa global %"] = round(100.0 * f_total / n_total, 2) if n_total else np.nan
        linhas.append(registro)
    return pd.DataFrame(linhas).set_index("modelo")


# =============================================================================
# 3. Estatística
# =============================================================================

def _matriz_pesos(k: int, tipo: str = "quadratico") -> np.ndarray:
    """Matriz k x k de pesos de concordância (1 na diagonal, decrescente fora dela)."""
    i, j = np.indices((k, k))
    if k == 1:
        return np.ones((1, 1))
    if tipo == "linear":
        return 1.0 - np.abs(i - j) / (k - 1)
    return 1.0 - ((i - j) / (k - 1)) ** 2  # quadrático (padrão do arcabouço)


def concordancia_ponderada(notas: np.ndarray, categorias: list,
                           tipo_peso: str = "quadratico") -> dict:
    """
    Decompõe a concordância entre avaliadores em seus componentes.

    Devolve, além do Kappa de Fleiss ponderado, a concordância observada (`p_o`),
    a concordância esperada por acaso (`p_e`) e o **AC2 de Gwet**. O AC2 usa uma
    estimativa de acaso que não depende das distribuições marginais e por isso não
    sofre o "paradoxo do Kappa": quando as notas se concentram em uma categoria,
    `p_e` do Fleiss se aproxima de 1 e deprime κ mesmo com `p_o` altíssimo.

    Args:
        notas: matriz (n_itens x n_avaliadores) com valores nas categorias
        categorias: lista ordenada das categorias da escala (ex.: [1, 2, 3, 4])
        tipo_peso: 'quadratico' (padrão do arcabouço) ou 'linear'

    Returns:
        dict com p_o, p_e, kappa, ac2 e n (np.nan quando indefinido)
    """
    vazio = {"p_o": np.nan, "p_e": np.nan, "kappa": np.nan, "ac2": np.nan, "n": 0}
    notas = np.asarray(notas)
    if notas.ndim != 2 or notas.shape[0] < 2 or notas.shape[1] < 2:
        return vazio

    n, m = notas.shape
    q = len(categorias)
    indice = {c: i for i, c in enumerate(categorias)}

    # N[i, c] = quantos avaliadores atribuíram a categoria c ao item i
    contagens = np.zeros((n, q))
    for i in range(n):
        for valor in notas[i]:
            pos = indice.get(valor)
            if pos is not None:
                contagens[i, pos] += 1

    pesos = _matriz_pesos(q, tipo_peso)
    # concordância observada ponderada (exclui o pareamento do avaliador consigo mesmo)
    p_o = float((np.einsum("ia,ab,ib->i", contagens, pesos, contagens) - m).sum()
                / (n * m * (m - 1)))

    prop = contagens.sum(axis=0) / (n * m)
    p_e = float(prop @ pesos @ prop)                       # acaso à la Fleiss
    p_e_gwet = float(pesos.sum() / (q * (q - 1)) * (prop * (1 - prop)).sum())

    kappa = np.nan if np.isclose(p_e, 1.0) else (p_o - p_e) / (1.0 - p_e)
    ac2 = np.nan if np.isclose(p_e_gwet, 1.0) else (p_o - p_e_gwet) / (1.0 - p_e_gwet)
    return {"p_o": p_o, "p_e": p_e, "kappa": float(kappa), "ac2": float(ac2), "n": n}


def fleiss_kappa_ponderado(notas: np.ndarray, categorias: list,
                           tipo_peso: str = "quadratico") -> float:
    """
    Kappa de Fleiss ponderado (pesos quadráticos), generalização de Mielke et al.

    Args:
        notas: matriz (n_itens x n_avaliadores) com valores nas categorias
        categorias: lista ordenada das categorias da escala (ex.: [1, 2, 3, 4])
        tipo_peso: 'quadratico' (padrão do arcabouço) ou 'linear'

    Returns:
        kappa_w, ou np.nan quando indefinido (sem variabilidade ou dados insuficientes)
    """
    return concordancia_ponderada(notas, categorias, tipo_peso)["kappa"]


def cohen_kappa_ponderado(a: Iterable, b: Iterable, categorias: list,
                          tipo_peso: str = "quadratico") -> float:
    """Kappa de Cohen ponderado (2 avaliadores) com pesos quadráticos."""
    a, b = np.asarray(list(a)), np.asarray(list(b))
    if len(a) < 2 or len(a) != len(b):
        return np.nan

    k = len(categorias)
    indice = {c: i for i, c in enumerate(categorias)}
    observado = np.zeros((k, k))
    for x, y in zip(a, b):
        i, j = indice.get(x), indice.get(y)
        if i is not None and j is not None:
            observado[i, j] += 1

    total = observado.sum()
    if total == 0:
        return np.nan
    observado /= total
    esperado = np.outer(observado.sum(axis=1), observado.sum(axis=0))
    pesos = _matriz_pesos(k, tipo_peso)

    p_obs = float((pesos * observado).sum())
    p_esp = float((pesos * esperado).sum())
    if np.isclose(p_esp, 1.0):
        return np.nan
    return float((p_obs - p_esp) / (1.0 - p_esp))


def interpretar_kappa(valor: float) -> str:
    """Faixas de Landis & Koch (1977)."""
    if valor is None or np.isnan(valor):
        return "indefinido"
    if valor < 0.00:
        return "pior que o acaso"
    if valor < 0.20:
        return "leve"
    if valor < 0.40:
        return "razoável"
    if valor < 0.60:
        return "moderada"
    if valor < 0.80:
        return "substancial"
    return "quase perfeita"


def ic_wilson(sucessos: int, total: int, confianca: float = 0.95) -> tuple:
    """Intervalo de confiança de Wilson para uma proporção (Etapa 6)."""
    if total == 0:
        return (np.nan, np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - confianca) / 2)
    p = sucessos / total
    denom = 1 + z ** 2 / total
    centro = (p + z ** 2 / (2 * total)) / denom
    margem = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return (p, max(0.0, centro - margem), min(1.0, centro + margem))


def correcao_holm(p_valores: list) -> list:
    """Correção de Holm-Bonferroni (step-down), preservando a monotonicidade."""
    m = len(p_valores)
    if m == 0:
        return []
    ordem = np.argsort(p_valores)
    ajustados = np.empty(m, dtype=float)
    corrente = 0.0
    for posicao, indice in enumerate(ordem):
        corrente = max(corrente, (m - posicao) * p_valores[indice])
        ajustados[indice] = min(1.0, corrente)
    return ajustados.tolist()


def interpretar_efeito(r: float) -> str:
    """Faixas de Cohen para o tamanho de efeito r do Wilcoxon."""
    if r is None or np.isnan(r):
        return "indefinido"
    r = abs(r)
    if r < 0.10:
        return "desprezível"
    if r < 0.30:
        return "pequeno"
    if r < 0.50:
        return "médio"
    return "grande"


def wilcoxon_pareado(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Wilcoxon signed-rank bilateral com estatística z e tamanho de efeito r = |z|/sqrt(n').

    ``n'`` é o número de pares com diferença não nula, conforme o arcabouço.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    diferencas = x - y
    n_efetivo = int(np.sum(diferencas != 0))
    resultado = {"n": len(x), "n_efetivo": n_efetivo, "W": np.nan, "z": np.nan,
                 "p": np.nan, "r": np.nan, "mediana_dif": float(np.median(diferencas)),
                 "obs": ""}

    if n_efetivo < 1:
        resultado["obs"] = "todas as diferenças são nulas"
        return resultado
    try:
        teste = stats.wilcoxon(x, y, alternative="two-sided",
                               zero_method="wilcox", method="approx",
                               correction=False)
    except ValueError as exc:
        resultado["obs"] = f"não computável ({exc})"
        return resultado

    resultado["W"] = float(teste.statistic)
    resultado["p"] = float(teste.pvalue)
    z = getattr(teste, "zstatistic", None)
    if z is None:  # compatibilidade com versões antigas do SciPy
        z = stats.norm.isf(resultado["p"] / 2) * np.sign(resultado["mediana_dif"] or 1)
    resultado["z"] = float(z)
    resultado["r"] = float(abs(z) / np.sqrt(n_efetivo))
    return resultado


def shapiro_seguro(valores: np.ndarray, semente: int = 42) -> dict:
    """Shapiro-Wilk com amostragem quando n > 5000 (limite do teste)."""
    valores = np.asarray(valores, dtype=float)
    valores = valores[~np.isnan(valores)]
    if len(valores) < 3 or np.allclose(valores, valores[0]):
        return {"W": np.nan, "p": np.nan, "n": len(valores), "amostrado": False}
    amostrado = len(valores) > 5000
    if amostrado:
        valores = np.random.default_rng(semente).choice(valores, 5000, replace=False)
    W, p = stats.shapiro(valores)
    return {"W": float(W), "p": float(p), "n": len(valores), "amostrado": amostrado}


# =============================================================================
# 4. Blocos de análise
# =============================================================================

def notas_medianas(df: pd.DataFrame, modelos: list) -> pd.DataFrame:
    """
    Variável primária: nota mediana das rodadas por (documento, modelo).

    Com 3 rodadas em escala 1--4, a mediana é sempre um valor legítimo da escala.
    """
    pivo = (df.pivot_table(index="documento", columns="modelo",
                           values="nota", aggfunc="median")
              .reindex(columns=[m for m in modelos if m in df["modelo"].unique()])
              .dropna())
    return pivo


def matriz_por_rodada(df: pd.DataFrame, modelo: str, rodadas: list) -> pd.DataFrame:
    """Matriz documento x rodada das notas de um modelo (entrada do Fleiss teste-reteste)."""
    return (df[df["modelo"] == modelo]
            .pivot_table(index="documento", columns="rodada", values="nota", aggfunc="median")
            .reindex(columns=rodadas)
            .dropna())


def confiabilidade_teste_reteste(df: pd.DataFrame, modelos: list, rodadas: list,
                                 categorias: list) -> tuple:
    """
    Etapa 1.2 -- confiabilidade teste-reteste do juiz LLM.

    O critério do arcabouço (κw ≥ 0,60) recai sobre o **instrumento**, e por isso é
    avaliado na linha agregada, que empilha todas as combinações documento × fonte
    em uma única matriz de itens × rodadas. As linhas por fonte são diagnósticas:
    estratificar concentra as marginais dentro de cada estrato, inflando a
    concordância esperada por acaso e deprimindo κ artificialmente (paradoxo do
    Kappa) mesmo quando a concordância observada é praticamente a mesma.

    Returns:
        (tabela_fleiss_por_fonte, tabela_pares_cohen, dict_agregado)
    """
    matrizes = [(modelo, matriz_por_rodada(df, modelo, rodadas)) for modelo in modelos]
    matrizes = [(rotulo, m) for rotulo, m in matrizes if not m.empty]
    if not matrizes:
        return pd.DataFrame(), pd.DataFrame(), {}

    # instrumento: todas as combinações documento x fonte como itens independentes
    agregada = pd.concat([m for _, m in matrizes], ignore_index=True)
    alvos = matrizes + [(ROTULO_INSTRUMENTO, agregada)]

    linhas_fleiss, linhas_pares, agregado = [], [], {}
    for rotulo, matriz in alvos:
        valores = matriz.to_numpy().astype(int)
        comp = concordancia_ponderada(valores, categorias)
        kappa = comp["kappa"]
        concordancia_total = float(np.mean([len(set(linha)) == 1 for linha in valores]))
        amplitude_max = float(np.mean(valores.max(axis=1) - valores.min(axis=1) <= 1))
        # paradoxo: κ reprovado apesar de concordância observada alta, por marginais concentradas
        paradoxo = bool(kappa == kappa and kappa < LIMIAR_KAPPA
                        and comp["p_e"] >= LIMIAR_PE_PARADOXO)
        registro = {
            "modelo": rotulo,
            "n itens": len(matriz),
            "P_o (obs.)": round(comp["p_o"], 4),
            "P_e (acaso)": round(comp["p_e"], 4),
            "Fleiss κw": round(kappa, 4),
            "Gwet AC2": round(comp["ac2"], 4),
            "Interpretação": interpretar_kappa(kappa),
            "κw ≥ 0,60 (diag.)": "sim" if kappa >= LIMIAR_KAPPA else "não",
            "Paradoxo κ": "sim" if paradoxo else "não",
            "Concord. exata %": round(100 * concordancia_total, 2),
            "Amplitude ≤1 %": round(100 * amplitude_max, 2),
            "DP intra-item": round(float(np.mean(valores.std(axis=1, ddof=1))), 4),
        }
        if rotulo == ROTULO_INSTRUMENTO:
            agregado = dict(registro, aprovado=kappa >= LIMIAR_KAPPA)
        else:
            linhas_fleiss.append(registro)

        for r1, r2 in combinations(rodadas, 2):
            kappa_par = cohen_kappa_ponderado(matriz[r1], matriz[r2], categorias)
            linhas_pares.append({
                "modelo": rotulo,
                "par": f"R{r1}×R{r2}",
                "Cohen κw": round(kappa_par, 4),
                "Interpretação": interpretar_kappa(kappa_par),
                "Concord. exata %": round(100 * float((matriz[r1] == matriz[r2]).mean()), 2),
            })
    return pd.DataFrame(linhas_fleiss), pd.DataFrame(linhas_pares), agregado


def efeito_rodada(df: pd.DataFrame, modelos: list, rodadas: list) -> pd.DataFrame:
    """
    Verifica viés sistemático **entre rodadas** do juiz (deriva de severidade).

    A confiabilidade teste-reteste (κw) mede dispersão, mas não detecta uma rodada
    sistematicamente mais severa: se todas as notas caem 1 ponto na rodada 2, a
    concordância cai sem revelar a direção. Aqui um Friedman por modelo trata as
    rodadas como tratamentos e os documentos como blocos.
    """
    matrizes = [(m, matriz_por_rodada(df, m, rodadas)) for m in modelos]
    matrizes = [(r, m) for r, m in matrizes if not m.empty]
    if matrizes:
        matrizes.append((ROTULO_INSTRUMENTO,
                         pd.concat([m for _, m in matrizes], ignore_index=True)))

    linhas = []
    for modelo, matriz in matrizes:
        if len(matriz) < 2 or len(rodadas) < 3:
            continue
        registro = {"modelo": modelo, "n itens": len(matriz)}
        for rodada in rodadas:
            registro[f"Média R{rodada}"] = round(float(matriz[rodada].mean()), 4)
        registro["Amplitude médias"] = round(
            float(matriz.mean().max() - matriz.mean().min()), 4)
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                chi2, p = stats.friedmanchisquare(*[matriz[r].to_numpy() for r in rodadas])
            registro["χ²_F"] = round(float(chi2), 4)
            registro["p"] = float(p)
            registro["Deriva (p<0,05)"] = "sim" if p < LIMIAR_ALFA else "não"
        except ValueError:
            registro["χ²_F"] = np.nan
            registro["p"] = np.nan
            registro["Deriva (p<0,05)"] = "—"
        linhas.append(registro)
    return pd.DataFrame(linhas)


def descritivas_por_modelo(pivo: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas descritivas da variável primária por modelo."""
    linhas = []
    for modelo in pivo.columns:
        serie = pivo[modelo]
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        prop, li, ls = ic_wilson(int((serie >= PISO_ADEQUACAO).sum()), len(serie))
        linhas.append({
            "modelo": modelo,
            "n": len(serie),
            "Média": round(serie.mean(), 4),
            "Mediana": round(serie.median(), 2),
            "DP": round(serie.std(ddof=1), 4),
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(q3 - q1, 2),
            "Mín": int(serie.min()),
            "Máx": int(serie.max()),
            f"P(mediana ≥ {PISO_ADEQUACAO})": round(prop, 4),
            "IC95 inf": round(li, 4),
            "IC95 sup": round(ls, 4),
        })
    return pd.DataFrame(linhas).set_index("modelo")


def hipotese_principal(pivo: pd.DataFrame) -> dict:
    """
    Etapas 2--5: normalidade, Friedman omnibus, post-hoc Wilcoxon+Holm e tamanho de efeito.

    Convenção de ranks: rank 1 = melhor desempenho (maior nota). Assim, o protocolo
    eleito ``D_best`` é o de **menor** rank médio, conforme o arcabouço.
    """
    modelos = list(pivo.columns)
    k, n = len(modelos), len(pivo)
    saida = {"k": k, "n": n, "modelos": modelos}

    # --- Etapa 2: normalidade das diferenças pareadas -------------------------
    normalidade = []
    for m1, m2 in combinations(modelos, 2):
        teste = shapiro_seguro((pivo[m1] - pivo[m2]).to_numpy())
        normalidade.append({
            "par": f"{m1} − {m2}",
            "n": teste["n"],
            "W": round(teste["W"], 4) if not np.isnan(teste["W"]) else np.nan,
            "p": teste["p"],
            "Normal (p>0,05)": "sim" if (teste["p"] or 0) > LIMIAR_ALFA else "não",
            "Amostrado": "sim" if teste["amostrado"] else "não",
        })
    saida["normalidade"] = pd.DataFrame(normalidade)

    # --- Etapa 3: omnibus de Friedman ----------------------------------------
    # ranks por documento: nota negada para que rank 1 corresponda à melhor nota
    ranks = pivo.apply(lambda linha: stats.rankdata(-linha.to_numpy()), axis=1, result_type="expand")
    ranks.columns = modelos
    saida["ranks_medios"] = ranks.mean().sort_values().rename_axis("modelo")

    if k >= 3 and n >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            chi2, p_friedman = stats.friedmanchisquare(*[pivo[m].to_numpy() for m in modelos])
        saida["friedman"] = {
            "chi2": float(chi2),
            "gl": k - 1,
            "p": float(p_friedman),
            "kendall_w": float(chi2 / (n * (k - 1))) if n and k > 1 else np.nan,
            "rejeita_h0": bool(p_friedman < LIMIAR_ALFA),
        }
    else:
        saida["friedman"] = None

    saida["d_best"] = saida["ranks_medios"].index[0] if len(saida["ranks_medios"]) else None

    # --- Etapas 4 e 5: post-hoc Wilcoxon com Holm e tamanho de efeito ---------
    contrastes = []
    for m1, m2 in combinations(modelos, 2):
        res = wilcoxon_pareado(pivo[m1].to_numpy(), pivo[m2].to_numpy())
        res["contraste"] = f"{m1} vs {m2}"
        res["mediana_1"] = float(pivo[m1].median())
        res["mediana_2"] = float(pivo[m2].median())
        res["media_1"] = float(pivo[m1].mean())
        res["media_2"] = float(pivo[m2].mean())
        contrastes.append(res)

    if contrastes:
        brutos = [c["p"] if not np.isnan(c["p"]) else 1.0 for c in contrastes]
        for contraste, ajustado in zip(contrastes, correcao_holm(brutos)):
            contraste["p_holm"] = ajustado
            contraste["significativo"] = bool(ajustado < LIMIAR_ALFA)
            contraste["efeito"] = interpretar_efeito(contraste["r"])
    saida["contrastes"] = pd.DataFrame(contrastes)
    return saida


def auditar_rotulos_problema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audita a aderência do juiz à taxonomia fechada da rubrica.

    Cada rótulo bruto emitido pelo juiz é confrontado com sua forma canônica e
    marcado como dentro ou fora da rubrica. Rótulos fora da rubrica são um achado
    metodológico: indicam que o juiz não se limitou às categorias prescritas.
    """
    pares = []
    for brutos in df["problemas_brutos"]:
        for bruto in brutos:
            canonico = normalizar_problema(bruto)
            if canonico:
                pares.append((bruto, canonico))
    if not pares:
        return pd.DataFrame()

    tabela = (pd.DataFrame(pares, columns=["Rótulo emitido", "Categoria canônica"])
              .value_counts().reset_index(name="Ocorrências"))
    tabela["Na rubrica"] = np.where(
        tabela["Categoria canônica"].isin(CATEGORIAS_PROBLEMA), "sim", "não")
    tabela["Normalizado"] = np.where(
        tabela["Rótulo emitido"] == tabela["Categoria canônica"], "não", "sim")
    return tabela.sort_values(["Na rubrica", "Ocorrências"], ascending=[False, False])


def analisar_problemas(df: pd.DataFrame, modelos: list) -> tuple:
    """Frequência das categorias de problema apontadas pelo juiz (análise descritiva)."""
    explodido = df.explode("problemas")
    explodido = explodido[explodido["problemas"].notna() & (explodido["problemas"] != "")]
    if explodido.empty:
        return pd.DataFrame(), pd.DataFrame()

    contagem = (explodido.pivot_table(index="problemas", columns="modelo",
                                      values="documento", aggfunc="count")
                .reindex(columns=[m for m in modelos if m in explodido["modelo"].unique()])
                .fillna(0).astype(int))

    avaliacoes = df.groupby("modelo").size()
    taxa = pd.DataFrame({
        "modelo": modelos,
        "Avaliações": [int(avaliacoes.get(m, 0)) for m in modelos],
        "Com ≥1 problema": [int((df[df["modelo"] == m]["problemas"].str.len() > 0).sum())
                            for m in modelos],
    })
    taxa["Taxa %"] = (100 * taxa["Com ≥1 problema"] / taxa["Avaliações"]).round(2)
    taxa["Problemas/avaliação"] = [
        round(float(df[df["modelo"] == m]["problemas"].str.len().mean()), 3) for m in modelos
    ]
    return contagem, taxa.set_index("modelo")


def custo_juiz(df: pd.DataFrame, modelos: list, rodadas: list) -> pd.DataFrame:
    """Consumo de tokens e tempo do juiz por modelo e rodada (reprodutibilidade/custo)."""
    linhas = []
    for modelo in modelos:
        for rodada in rodadas:
            sub = df[(df["modelo"] == modelo) & (df["rodada"] == rodada)]
            if sub.empty:
                continue
            linhas.append({
                "modelo": modelo,
                "rodada": f"R{rodada}",
                "Avaliações": len(sub),
                "Prompt tokens (méd.)": round(sub["prompt_tokens"].mean(), 1),
                "Completion tokens (méd.)": round(sub["completion_tokens"].mean(), 1),
                "Total tokens": int(sub["total_tokens"].fillna(0).sum()),
                "Tempo méd. (s)": round(sub["tempo"].mean(), 3),
                "Tempo total (s)": round(sub["tempo"].fillna(0).sum(), 1),
            })
    return pd.DataFrame(linhas)


# =============================================================================
# 5. Gráficos
# =============================================================================

def gerar_graficos(df: pd.DataFrame, pivo: pd.DataFrame, modelos: list, rodadas: list,
                   categorias: list, analise: dict, fleiss: pd.DataFrame,
                   pares: pd.DataFrame, problemas: pd.DataFrame, falhas: pd.DataFrame,
                   custos: pd.DataFrame, saida: str, instrumento: dict = None) -> list:
    """Gera todos os gráficos da análise reaproveitando o pacote `util_graficos`."""
    gerados = []

    def _registrar(caminho):
        # rede de segurança: alguns métodos do util_graficos podem não devolver o
        # caminho, então a lista final é reconciliada com os arquivos em disco
        if caminho:
            gerados.append(os.path.basename(caminho))

    # 01 -- grade rodadas (linhas) x modelos (colunas): distribuição das notas Likert
    grade = {
        f"Rodada {r}": {m: df[(df["rodada"] == r) & (df["modelo"] == m) & df["nota"].notna()]["nota"].tolist()
                        for m in modelos}
        for r in rodadas
    }
    _registrar(UtilGraficos.grafico_grade_distribuicao(
        grade, categorias=categorias,
        x="Nota Likert", y="% de documentos",
        titulo_geral="Distribuição das notas do juiz LLM — rodadas (linhas) × modelos (colunas)",
        paleta_cores=Cores.RdYlGn,
        arquivo_saida=os.path.join(saida, "01_grade_distribuicao_notas.png")))

    # 02 -- boxplot da variável primária (gera .md com quartis/outliers automaticamente)
    UtilGraficos.gerar_boxplot(
        {m: pivo[m].tolist() for m in pivo.columns},
        titulo="Nota mediana das rodadas por modelo (variável primária)",
        ylabel="Nota Likert mediana", xlabel="Modelo",
        paleta_cores=Cores.Set2, rotacao_labels=0,
        nota=f"n = {len(pivo)} documentos pareados",
        arquivo_saida=os.path.join(saida, "02_boxplot_notas_medianas.png"))
    gerados.append("02_boxplot_notas_medianas.png")

    # 03 -- distribuição da variável primária
    _registrar(UtilGraficos.grafico_grade_distribuicao(
        {"Nota mediana (3 rodadas)": {m: pivo[m].tolist() for m in pivo.columns}},
        categorias=categorias, x="Nota Likert mediana", y="% de documentos",
        titulo_geral="Variável primária — distribuição da nota mediana por modelo",
        paleta_cores=Cores.RdYlGn,
        arquivo_saida=os.path.join(saida, "03_distribuicao_notas_medianas.png")))

    # 04 -- viabilidade de produção (Etapa 6)
    descritivas = descritivas_por_modelo(pivo)
    _registrar(UtilGraficos.gerar_barras_ic(
        categorias=list(descritivas.index),
        valores=descritivas[f"P(mediana ≥ {PISO_ADEQUACAO})"].tolist(),
        ic_inferior=descritivas["IC95 inf"].tolist(),
        ic_superior=descritivas["IC95 sup"].tolist(),
        titulo=f"Viabilidade de produção — proporção de documentos com mediana ≥ {PISO_ADEQUACAO} (IC 95% Wilson)",
        ylabel="Proporção de documentos", xlabel="Modelo",
        linha_referencia=LIMIAR_VIABILIDADE,
        rotulo_referencia=f"Critério: LI do IC ≥ {LIMIAR_VIABILIDADE:.2f}",
        ylim=(0, 1.12), paleta_cores=Cores.Blues,
        arquivo_saida=os.path.join(saida, "04_viabilidade_producao.png")))

    # 05 -- concordância entre rodadas (Etapa 1.2)
    if not pares.empty:
        matriz_kappa = pares.pivot(index="modelo", columns="par", values="Cohen κw")
        coluna_fleiss = (fleiss.set_index("modelo")["Fleiss κw"] if not fleiss.empty
                         else pd.Series(dtype=float))
        if instrumento:
            coluna_fleiss = pd.concat([coluna_fleiss, pd.Series(
                {ROTULO_INSTRUMENTO: instrumento["Fleiss κw"]})])
        if not coluna_fleiss.empty:
            matriz_kappa["Fleiss κw (3 rodadas)"] = coluna_fleiss
        ordem_heatmap = [m for m in modelos if m in matriz_kappa.index]
        if ROTULO_INSTRUMENTO in matriz_kappa.index:
            ordem_heatmap.append(ROTULO_INSTRUMENTO)
        _registrar(UtilGraficos.gerar_heatmap(
            matriz_kappa.reindex(ordem_heatmap),
            titulo="Confiabilidade teste-reteste do juiz LLM (Kappa ponderado quadrático)",
            vmin=0, vmax=1, fmt=".3f", rotulo_barra="κw",
            xlabel="Comparação entre rodadas", ylabel="Modelo",
            paleta_cores=Cores.RdYlGn,
            arquivo_saida=os.path.join(saida, "05_concordancia_rodadas.png")))

    # 06 -- estabilidade teste-reteste: matriz de confusão R1 x R2 (todos os modelos)
    if len(rodadas) >= 2:
        r1, r2 = rodadas[0], rodadas[1]
        confusao = pd.DataFrame(0, index=[f"R{r1}={c}" for c in categorias],
                                columns=[f"R{r2}={c}" for c in categorias])
        for modelo in modelos:
            matriz = matriz_por_rodada(df, modelo, [r1, r2])
            for a, b in zip(matriz[r1].astype(int), matriz[r2].astype(int)):
                if a in categorias and b in categorias:
                    confusao.loc[f"R{r1}={a}", f"R{r2}={b}"] += 1
        _registrar(UtilGraficos.gerar_heatmap(
            confusao, titulo=f"Estabilidade das notas entre as rodadas {r1} e {r2} (todos os modelos)",
            fmt=".0f", rotulo_barra="Documentos",
            xlabel=f"Rodada {r2}", ylabel=f"Rodada {r1}", paleta_cores=Cores.Blues,
            arquivo_saida=os.path.join(saida, "06_matriz_confusao_rodadas.png")))

    # 07 -- categorias de problema apontadas pelo juiz
    if not problemas.empty:
        _registrar(UtilGraficos.gerar_grafico_barras(
            problemas.transpose(),
            titulo="Categorias de problema apontadas pelo juiz LLM (todas as rodadas)",
            ylabel="Ocorrências", xlabel="Modelo",
            paleta_cores=Cores.Dark2, mostrar_valores=False,
            arquivo_saida=os.path.join(saida, "07_problemas_por_modelo.png")))

    # 08 -- ranks médios do Friedman
    ranks = analise["ranks_medios"]
    _registrar(UtilGraficos.gerar_grafico_barras(
        ranks.to_frame("Rank médio"),
        titulo="Ranks médios do omnibus de Friedman (1 = melhor desempenho)",
        ylabel="Rank médio", xlabel="Modelo",
        paleta_cores=Cores.Cividis, mostrar_valores=True,
        arquivo_saida=os.path.join(saida, "08_ranks_friedman.png")))

    # 09 -- custo do juiz (tokens e tempo)
    if not custos.empty:
        tokens = custos.pivot(index="modelo", columns="rodada", values="Total tokens")
        _registrar(UtilGraficos.gerar_grafico_barras(
            tokens.reindex([m for m in modelos if m in tokens.index]),
            titulo="Consumo total de tokens do juiz LLM por modelo e rodada",
            ylabel="Tokens", xlabel="Modelo", paleta_cores=Cores.PuBuGn,
            mostrar_valores=False,
            arquivo_saida=os.path.join(saida, "09_custo_juiz.png")))

    # 10 -- taxa de falhas de parsing (Etapa 0)
    colunas_taxa = [c for c in falhas.columns if c.endswith("taxa %")]
    if colunas_taxa:
        _registrar(UtilGraficos.gerar_grafico_barras(
            falhas[colunas_taxa],
            titulo="Etapa 0 — taxa de falhas de parsing por modelo e rodada",
            ylabel="Falhas (%)", xlabel="Modelo",
            paleta_cores=Cores.Plasma, mostrar_valores=True,
            arquivo_saida=os.path.join(saida, "10_falhas_por_modelo.png")))

    # reconcilia com o que realmente foi escrito em disco
    em_disco = sorted(f for f in os.listdir(saida) if f.lower().endswith(".png"))
    return em_disco or gerados


# =============================================================================
# 6. Relatório markdown
# =============================================================================

def _num(valor, casas: int = 4) -> str:
    """Formata um número no padrão pt-BR: vírgula decimal e ponto como separador de milhar."""
    if valor is None:
        return "—"
    if isinstance(valor, (bool, np.bool_)):
        return "sim" if valor else "não"
    if isinstance(valor, (int, np.integer, float, np.floating)):
        if isinstance(valor, (float, np.floating)) and np.isnan(valor):
            return "—"
        if casas <= 0:
            return f"{int(round(float(valor))):,}".replace(",", ".")
        return f"{float(valor):,.{casas}f}".replace(",", "|").replace(".", ",").replace("|", ".")
    return str(valor)


def _decimais(serie: pd.Series, maximo: int) -> int:
    """Menor número de casas decimais que representa toda a coluna sem perda."""
    valores = [float(v) for v in serie
               if isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_))
               and not (isinstance(v, float) and np.isnan(v))]
    if not valores:
        return 0
    for casas in range(0, maximo + 1):
        if all(abs(round(v, casas) - v) < 1e-9 for v in valores):
            return casas
    return maximo


def _md(df: pd.DataFrame, indice: bool = True, casas: int = 4) -> str:
    """
    Converte um DataFrame em tabela markdown com números no padrão pt-BR.

    Colunas cujos valores são todos inteiros são exibidas sem casas decimais,
    evitando ruído do tipo `400,0000` em contagens.
    """
    if df is None or df.empty:
        return "_Sem dados._"

    copia = df.reset_index() if indice else df.copy()
    copia = copia.apply(lambda col: col.map(
        lambda v, d=_decimais(col, casas): _num(v, d)))
    cabecalho = "| " + " | ".join(str(c) for c in copia.columns) + " |"
    separador = "|" + "|".join(["---"] * len(copia.columns)) + "|"
    linhas = ["| " + " | ".join(str(v) for v in reg) + " |"
              for reg in copia.itertuples(index=False)]
    return "\n".join([cabecalho, separador] + linhas)


def _p(valor: float) -> str:
    """Formata p-valores, com notação científica quando muito pequenos."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "—"
    if valor < 1e-4:
        return f"{valor:.2e}".replace(".", ",")
    return f"{valor:.4f}".replace(".", ",")


def _montar_sintese(c: dict) -> list:
    """Monta as linhas do sumário executivo a partir dos resultados já calculados."""
    itens = []
    analise = c["analise"]

    # instrumento (critério avaliado no agregado, conforme Etapa 1.2)
    inst = c.get("instrumento") or {}
    if inst:
        veredito = "aprovado" if inst.get("aprovado") else "reprovado"
        itens.append(f"**Instrumento:** {veredito} — Fleiss κw = "
                     f"{_num(inst['Fleiss κw'])} sobre {_num(inst['n itens'], 0)} itens "
                     f"(critério ≥ {_num(LIMIAR_KAPPA, 2)}; concordância observada "
                     f"P_o = {_num(inst['P_o (obs.)'])}).")
    paradoxais = ([] if c["fleiss"].empty
                  else c["fleiss"][c["fleiss"]["Paradoxo κ"] == "sim"]["modelo"].tolist())
    if paradoxais:
        itens.append(f"**Diagnóstico:** κw estratificado deprimido em {', '.join(paradoxais)} "
                     "por paradoxo do Kappa (efeito de teto da escala) — usar apenas como "
                     "diagnóstico, não como critério.")

    # hipótese principal
    friedman = analise.get("friedman")
    if friedman and not np.isnan(friedman["chi2"]):
        veredito = "rejeitada" if friedman["rejeita_h0"] else "não rejeitada"
        itens.append(f"**Hipótese principal:** H₀ {veredito} no Friedman "
                     f"(χ²_F = {_num(friedman['chi2'], 2)}, p = {_p(friedman['p'])}, "
                     f"W de Kendall = {_num(friedman['kendall_w'])}); "
                     f"melhor rank médio: `{analise['d_best']}`.")

    contrastes = analise.get("contrastes")
    if contrastes is not None and not contrastes.empty:
        n_sig = int(contrastes["significativo"].sum())
        itens.append(f"**Contrastes:** {n_sig} de {len(contrastes)} significativos após Holm; "
                     f"tamanhos de efeito r entre {_num(contrastes['r'].min(), 2)} e "
                     f"{_num(contrastes['r'].max(), 2)}.")

    # viabilidade
    viaveis = c["descritivas"][c["descritivas"]["IC95 inf"] >= LIMIAR_VIABILIDADE]
    if len(viaveis):
        prefixo = "apenas " if len(viaveis) == 1 else ""
        verbo = "atinge" if len(viaveis) == 1 else "atingem"
        itens.append(f"**Viabilidade de produção:** {prefixo}{', '.join(viaveis.index)} "
                     f"{verbo} o critério (LI do IC 95% ≥ {_num(LIMIAR_VIABILIDADE, 2)}).")
    else:
        itens.append("**Viabilidade de produção:** nenhum modelo atinge o critério "
                     f"(LI do IC 95% ≥ {_num(LIMIAR_VIABILIDADE, 2)}).")

    # qualidade dos dados
    if c["n_descartado"]:
        itens.append(f"**Dados:** {c['n_descartado']} de {c['n_bruto']} documentos "
                     "descartados pelo pareamento global.")
    else:
        itens.append(f"**Dados:** nenhuma falha de parsing; {c['n_valido']} documentos "
                     "pareados em todas as células.")

    if not c["auditoria"].empty:
        fora = c["auditoria"][c["auditoria"]["Na rubrica"] == "não"]["Ocorrências"].sum()
        norm = int((c["auditoria"]["Normalizado"] == "sim").sum())
        if fora or norm:
            plural_g = "grafia unificada" if norm == 1 else "grafias unificadas"
            plural_o = "ocorrência" if fora == 1 else "ocorrências"
            itens.append(f"**Rubrica:** {norm} {plural_g} na normalização e "
                         f"{int(fora)} {plural_o} fora da taxonomia prescrita.")
    return itens


def gerar_markdown(contexto: dict, caminho: str) -> str:
    """Escreve `estatisticas.md` com as tabelas e leituras das Etapas 0--8."""
    c = contexto
    analise = c["analise"]
    L = []

    L.append("# Análise estatística das avaliações do juiz LLM\n")
    L.append(f"Gerado em {datetime.now():%d/%m/%Y %H:%M} por `gerar_graficos_llm.py`.\n")
    L.append("| Parâmetro | Valor |")
    L.append("|---|---|")
    L.append(f"| Rodadas do juiz | {len(c['rodadas'])} ({', '.join(c['pastas'])}) |")
    L.append(f"| Modelos/protocolos avaliados (*k*) | {len(c['modelos'])} — {', '.join(c['modelos'])} |")
    L.append(f"| Escala Likert | {c['categorias'][0]}–{c['categorias'][-1]} |")
    L.append(f"| Modelo juiz | {c['juiz'] or 'não informado'} |")
    L.append(f"| Documentos brutos | {c['n_bruto']:,} |".replace(",", "."))
    L.append(f"| Documentos válidos pareados (*n\\**) | {c['n_valido']:,} |".replace(",", "."))
    L.append(f"| Documentos descartados | {c['n_descartado']:,} |".replace(",", "."))
    L.append(f"| Nível de significância (α) | {_num(LIMIAR_ALFA, 2)} |")
    L.append("")
    L.append("> **Variável primária:** nota Likert **mediana** das rodadas independentes, "
             "por documento e por protocolo. Com 3 rodadas em escala de 4 pontos, a mediana "
             "é necessariamente um valor legítimo da escala.\n")

    L.append("### Síntese dos achados\n")
    for item in c["sintese"]:
        L.append(f"- {item}")
    L.append("")

    # ---------------------------------------------------------------- Etapa 0
    L.append("---\n\n## Etapa 0 — Falhas de parsing e descarte global pareado\n")
    L.append("Critério de falha: erro registrado na chamada, JSON de resposta não "
             "interpretável ou ausência do campo `nota`. Documentos com falha em "
             "**qualquer** combinação modelo × rodada são removidos de **todos** os "
             "protocolos, preservando o pareamento estrito exigido por Friedman e Wilcoxon.\n")
    L.append(_md(c["falhas"]))
    L.append("")
    taxa_global = 100.0 * c["n_descartado"] / max(c["n_bruto"], 1)
    L.append(f"**n\\* = {c['n_valido']} documentos** completos em todas as "
             f"{len(c['modelos'])} × {len(c['rodadas'])} = "
             f"{len(c['modelos']) * len(c['rodadas'])} células. "
             f"Descarte global: {c['n_descartado']} documentos ({_num(taxa_global, 2)}%).\n")

    # -------------------------------------------------------------- Etapa 1.2
    L.append("---\n\n## Etapa 1.2 — Confiabilidade teste-reteste do juiz LLM\n")
    inst = c.get("instrumento") or {}
    if inst:
        L.append(f"### Critério do arcabouço: κw ≥ {_num(LIMIAR_KAPPA, 2)} (nível instrumento)\n")
        L.append("O arcabouço define a confiabilidade teste-reteste **do juiz**, no singular. "
                 "O critério é portanto avaliado sobre todas as combinações documento × fonte "
                 "empilhadas em uma única matriz de itens × rodadas.\n")
        L.append("| Métrica | Valor |")
        L.append("|---|---|")
        L.append(f"| Itens (documento × fonte) | {_num(inst['n itens'], 0)} |")
        L.append(f"| Avaliadores (rodadas) | {len(c['rodadas'])} |")
        L.append(f"| Concordância observada P_o | {_num(inst['P_o (obs.)'])} |")
        L.append(f"| Concordância esperada por acaso P_e | {_num(inst['P_e (acaso)'])} |")
        L.append(f"| **Fleiss κw** | **{_num(inst['Fleiss κw'])}** |")
        L.append(f"| Gwet AC2 (referência) | {_num(inst['Gwet AC2'])} |")
        L.append(f"| Interpretação | {inst['Interpretação']} |")
        L.append(f"| Concordância exata entre as 3 rodadas | {_num(inst['Concord. exata %'], 2)}% |")
        L.append(f"| Itens com amplitude ≤ 1 ponto | {_num(inst['Amplitude ≤1 %'], 2)}% |")
        L.append("")
        if inst.get("aprovado"):
            L.append(f"✅ **Instrumento aprovado** — κw = {_num(inst['Fleiss κw'])} ≥ "
                     f"{_num(LIMIAR_KAPPA, 2)}. A variável primária do juiz LLM pode ser "
                     "usada nos testes das Etapas 2–5 sem ressalva de confiabilidade.\n")
        else:
            L.append(f"⚠️ **Instrumento reprovado** — κw = {_num(inst['Fleiss κw'])} < "
                     f"{_num(LIMIAR_KAPPA, 2)}. Aciona-se o plano contingencial: os resultados "
                     "do juiz LLM devem ser reportados com ressalva, as conclusões da hipótese "
                     "principal ficam condicionadas a essa limitação e as métricas automáticas "
                     "(BERTScore F1) passam a variável substituta nos contrastes Q1–Q3.\n")

        k_inst = inst.get("Fleiss κw", np.nan)
        k_fontes = (c["fleiss"]["Fleiss κw"].dropna() if not c["fleiss"].empty
                    else pd.Series(dtype=float))
        # a explicação abaixo só se aplica quando o agregado de fato supera as fontes
        if len(k_fontes) and k_inst == k_inst and k_inst > k_fontes.max():
            L.append("#### Por que o κw agregado supera o de cada fonte isolada\n")
            pe_min = c["fleiss"]["P_e (acaso)"].min()
            pe_max = c["fleiss"]["P_e (acaso)"].max()
            po_min = c["fleiss"]["P_o (obs.)"].min()
            po_max = c["fleiss"]["P_o (obs.)"].max()
            k_min = c["fleiss"]["Fleiss κw"].min()
            k_max = c["fleiss"]["Fleiss κw"].max()
            L.append(f"A concordância **observada** praticamente não muda: {_num(po_min)}–"
                     f"{_num(po_max)} nas fontes isoladas contra {_num(inst['P_o (obs.)'])} no "
                     "agregado. O que muda é a concordância **esperada por acaso**, que cai de "
                     f"{_num(pe_min)}–{_num(pe_max)} para {_num(inst['P_e (acaso)'])}. Reunidas "
                     "as fontes, as notas passam a cobrir toda a extensão da escala, e dois "
                     "julgamentos aleatórios deixam de coincidir com tanta facilidade.\n")
            L.append(f"Como κ = (P_o − P_e)/(1 − P_e), o salto de {_num(k_min)}–{_num(k_max)} "
                     f"para {_num(inst['Fleiss κw'])} vem do **denominador**, não do numerador: "
                     "o termo (1 − P_e) volta a ter amplitude. Não se trata de o juiz ficar "
                     "mais consistente quando as fontes são reunidas — a estabilidade dele é a "
                     "mesma; o que se recupera é a capacidade do coeficiente de medi-la.\n")
            L.append("> **Frase pronta para a redação/apresentação:** a confiabilidade "
                     "teste-reteste do juiz foi estimada no nível do instrumento "
                     f"(κw = {_num(inst['Fleiss κw'])}). As estimativas estratificadas por "
                     f"fonte ({_num(k_min)}–{_num(k_max)}) são sistematicamente menores porque, "
                     "dentro de cada estrato, as notas se concentram em poucas categorias e a "
                     "concordância esperada por acaso se aproxima de 1, comprimindo o "
                     "coeficiente sem que a concordância observada se altere "
                     f"({_num(po_min)}–{_num(po_max)} nas fontes; "
                     f"{_num(inst['P_o (obs.)'])} no agregado). Trata-se do paradoxo do Kappa "
                     "descrito por Feinstein e Cicchetti (1990), e não de instabilidade do "
                     "instrumento.\n")
        L.append("**Ressalva de independência:** os itens do agregado são pares documento × "
                 "fonte, de modo que cada documento entra "
                 f"{len(c['modelos'])} vezes. As unidades não são, portanto, mutuamente "
                 "independentes. Isso é aceitável para um coeficiente de concordância, cuja "
                 "unidade de julgamento é a extração avaliada e não o acórdão; a ressalva é "
                 "registrada aqui para uso na discussão de limitações.\n")

    L.append(f"### Diagnóstico por fonte avaliada\n")
    L.append("Estratificação **diagnóstica**, não avaliativa. Dentro de cada estrato as notas "
             "se concentram em poucas categorias, o que eleva `P_e` e deprime κ mesmo com `P_o` "
             "praticamente idêntico entre as fontes — por isso o critério não é aplicado aqui. "
             "`Gwet AC2` usa uma estimativa de acaso independente das marginais e serve de "
             "contraprova.\n")
    L.append(_md(c["fleiss"], indice=False))
    L.append("")
    if not c["fleiss"].empty:
        baixos = c["fleiss"][c["fleiss"]["Fleiss κw"] < LIMIAR_KAPPA]["modelo"].tolist()
        paradoxais = c["fleiss"][c["fleiss"]["Paradoxo κ"] == "sim"]["modelo"].tolist()
        if baixos:
            L.append(f"κw estratificado abaixo de {_num(LIMIAR_KAPPA, 2)} em: "
                     f"{', '.join(baixos)}.")
        if paradoxais:
            linhas_par = c["fleiss"].set_index("modelo").loc[paradoxais]
            L.append("")
            L.append("#### 🔎 Paradoxo do Kappa detectado\n")
            L.append(f"Em {', '.join(paradoxais)} o κw estratificado ficou baixo **apesar de "
                     "concordância observada alta**, porque as notas se concentram no topo da "
                     "escala e a concordância esperada por acaso se aproxima de 1 "
                     "(Feinstein & Cicchetti, 1990; Gwet, 2008):\n")
            for modelo, linha in linhas_par.iterrows():
                L.append(f"- **{modelo}**: P_o = {_num(linha['P_o (obs.)'])}, "
                         f"P_e = {_num(linha['P_e (acaso)'])}, κw = {_num(linha['Fleiss κw'])}, "
                         f"**AC2 = {_num(linha['Gwet AC2'])}**, amplitude ≤1 em "
                         f"{_num(linha['Amplitude ≤1 %'], 2)}% dos itens.")
            L.append("")
            L.append("Leitura: a variação restante é o juiz hesitando entre duas categorias "
                     "adjacentes num modelo que opera no teto da escala de 4 pontos — efeito "
                     "de teto, não instabilidade. Vale registrar na seção de limitações que a "
                     "escala Likert de 4 pontos tem poder discriminativo reduzido nessa faixa.\n")
    L.append("### Concordância par a par entre rodadas (Cohen κw ponderado)\n")
    L.append(_md(c["pares"], indice=False))
    L.append("")

    L.append("### Deriva de severidade entre rodadas\n")
    L.append("Friedman por modelo, tratando as rodadas como tratamentos e os documentos como "
             "blocos. O κw mede dispersão, mas não detecta uma rodada sistematicamente mais "
             "severa; este teste cobre essa lacuna.\n")
    deriva = c["deriva"].copy()
    if not deriva.empty:
        deriva["p"] = deriva["p"].map(_p)
        L.append(_md(deriva, indice=False))
        L.append("")
        com_deriva = c["deriva"][c["deriva"]["Deriva (p<0,05)"] == "sim"]["modelo"].tolist()
        if com_deriva:
            L.append(f"⚠️ Deriva significativa entre rodadas em: {', '.join(com_deriva)}. "
                     "A mediana das três rodadas amortece o efeito, mas vale registrar a "
                     "ordem de execução como fonte de variação.\n")
        else:
            L.append("✅ Sem evidência de deriva sistemática de severidade entre as rodadas.\n")
    else:
        L.append("_Não computável (exige ao menos 3 rodadas)._\n")

    L.append("### Etapas 1.1 e 1.3 — validação humana (pendente)\n")
    L.append("Estas etapas exigem o *Gold Set* anotado pelos três especialistas, que não "
             "está contido nos parquets do juiz:\n")
    L.append("- **1.1** Concordância interna entre avaliadores humanos — Fleiss κw "
             f"ponderado (critério κw ≥ {_num(LIMIAR_KAPPA, 2)});")
    L.append("- **1.3** Concordância mediana humana ↔ mediana LLM — Cohen κw ponderado "
             f"(≥ {_num(LIMIAR_KAPPA, 2)}) + Wilcoxon bilateral (p > {_num(LIMIAR_ALFA, 2)}, ausência de "
             "viés sistemático).\n")
    L.append("As funções `fleiss_kappa_ponderado`, `cohen_kappa_ponderado` e "
             "`wilcoxon_pareado` deste script já implementam esses testes: basta fornecer a "
             "matriz de notas humanas para completar a Etapa 1.\n")

    # ---------------------------------------------------------- Descritivas
    L.append("---\n\n## Estatísticas descritivas da variável primária\n")
    L.append(_md(c["descritivas"]))
    L.append("")

    # ---------------------------------------------------------------- Etapa 2
    L.append("---\n\n## Etapa 2 — Normalidade (Shapiro-Wilk)\n")
    L.append("Aplicado às diferenças pareadas de cada contraste. A rejeição da normalidade "
             f"(p < {_num(LIMIAR_ALFA, 2)}) justifica formalmente a opção não paramétrica "
             "(Friedman/Wilcoxon).\n")
    normalidade = analise["normalidade"].copy()
    if not normalidade.empty:
        normalidade["p"] = normalidade["p"].map(_p)
    L.append(_md(normalidade, indice=False))
    L.append("")
    if not analise["normalidade"].empty:
        nao_normais = int((analise["normalidade"]["Normal (p>0,05)"] == "não").sum())
        L.append(f"Normalidade rejeitada em {nao_normais} de "
                 f"{len(analise['normalidade'])} contrastes — o uso de testes de posto "
                 "está justificado.\n")

    # ---------------------------------------------------------------- Etapa 3
    L.append("---\n\n## Etapa 3 — Omnibus de Friedman\n")
    L.append("**H₀:** as distribuições da nota Likert mediana são iguais para todos os "
             "protocolos. **H₁:** ao menos um protocolo difere.\n")
    friedman = analise["friedman"]
    if friedman:
        L.append("| Estatística | Valor |")
        L.append("|---|---|")
        L.append(f"| χ²_F | {_num(friedman['chi2'])} |")
        L.append(f"| Graus de liberdade | {friedman['gl']} |")
        L.append(f"| p-valor | {_p(friedman['p'])} |")
        L.append(f"| W de Kendall (tamanho de efeito) | {_num(friedman['kendall_w'])} |")
        L.append(f"| n (documentos pareados) | {analise['n']} |")
        L.append(f"| k (protocolos) | {analise['k']} |")
        L.append("")
        if np.isnan(friedman["chi2"]):
            decisao = ("**Omnibus indefinido** — não há variabilidade entre os protocolos "
                       "(todas as notas medianas empatadas). Verifique os dados de entrada.")
        elif friedman["rejeita_h0"]:
            decisao = ("**H₀ rejeitada** — há diferença entre protocolos; os contrastes "
                       "post-hoc estão autorizados.")
        else:
            decisao = ("**H₀ não rejeitada** — não há evidência de diferença entre protocolos; "
                       "os contrastes post-hoc a seguir têm caráter exploratório.")
        L.append(decisao + "\n")
    else:
        L.append("_Omnibus não computado: são necessários k ≥ 3 protocolos e n ≥ 2 documentos._\n")

    L.append("### Ranks médios (1 = melhor desempenho)\n")
    L.append(_md(analise["ranks_medios"].to_frame("Rank médio")))
    L.append("")
    if analise["d_best"]:
        L.append(f"**D_best = `{analise['d_best']}`** (menor rank médio).\n")

    # ------------------------------------------------------------ Etapas 4 e 5
    L.append("---\n\n## Etapas 4 e 5 — Contrastes post-hoc (Wilcoxon + Holm) e tamanho de efeito\n")
    L.append("Wilcoxon *signed-rank* bilateral em todos os pares, com correção de Holm "
             f"sobre o conjunto de {len(analise['contrastes'])} p-valores. Tamanho de efeito "
             "r = |z|/√n′, com n′ = pares de diferença não nula, interpretado pelas faixas "
             "de Cohen.\n")
    contrastes = analise["contrastes"]
    if not contrastes.empty:
        tabela = pd.DataFrame({
            "Contraste": contrastes["contraste"],
            "Mediana A": contrastes["mediana_1"].round(2),
            "Mediana B": contrastes["mediana_2"].round(2),
            "Média A": contrastes["media_1"].round(4),
            "Média B": contrastes["media_2"].round(4),
            "n′": contrastes["n_efetivo"],
            "W": contrastes["W"].round(1),
            "z": contrastes["z"].round(4),
            "p bruto": contrastes["p"].map(_p),
            "p Holm": contrastes["p_holm"].map(_p),
            "r": contrastes["r"].round(4),
            "Efeito": contrastes["efeito"],
            "Signif.": contrastes["significativo"].map({True: "sim", False: "não"}),
        })
        L.append(_md(tabela, indice=False))
        L.append("")
        significativos = contrastes[contrastes["significativo"]]
        L.append(f"Contrastes significativos após Holm: **{len(significativos)}** de "
                 f"{len(contrastes)}.\n")
        if not significativos.empty:
            L.append("Leitura dos contrastes significativos:\n")
            for _, linha in significativos.iterrows():
                vencedor = (linha["contraste"].split(" vs ")[0]
                            if linha["media_1"] > linha["media_2"]
                            else linha["contraste"].split(" vs ")[1])
                L.append(f"- `{linha['contraste']}`: p_Holm = {_p(linha['p_holm'])}, "
                         f"r = {_num(linha['r'], 3)} (efeito {linha['efeito']}); "
                         f"desempenho superior de **{vencedor}**.")
            L.append("")
    else:
        L.append("_Sem contrastes computáveis._\n")

    protocolos = [m for m in c["modelos"] if m.lower() in {"a", "b", "c", "d1", "d2",
                                                           "d3", "d4", "d5", "d6", "d7",
                                                           "d8", "d9", "d10"}]
    if protocolos:
        L.append("> **Mapeamento para as perguntas de pesquisa:** associe os contrastes acima "
                 "a Q1 (A vs. {B, C, D1, D2}), Q2 ({B, C} vs. {D1, D2}) e Q3 (D1 vs. D2).\n")
    else:
        L.append("> **Nota:** os rótulos avaliados nesta execução não correspondem aos "
                 "protocolos de treinamento (A, B, C, D1, D2), e sim a fontes de extração. "
                 "Os contrastes acima são, portanto, comparações entre fontes — o mapeamento "
                 "para Q1–Q3 só se aplica quando os protocolos treinados entrarem na "
                 "comparação.\n")
    if analise["n"] < 200:
        L.append(f"> **Poder estatístico:** com n = {analise['n']} documentos, contrastes de "
                 "efeito pequeno podem não ser detectados. Ausência de significância aqui "
                 "não é evidência de equivalência.\n")
    else:
        L.append("> Com testes bilaterais e n elevado, a relevância prática recai sobre o "
                 "tamanho de efeito r, não apenas sobre o p-valor.\n")

    # ---------------------------------------------------------------- Etapa 6
    L.append("---\n\n## Etapa 6 — Viabilidade de produção (descritivo)\n")
    L.append(f"Proporção de documentos com nota mediana ≥ {PISO_ADEQUACAO} (piso de "
             "adequação), com IC 95% de Wilson. Não constitui teste formal: proporções cujo "
             f"**limite inferior do IC ≥ {_num(LIMIAR_VIABILIDADE, 2)}** são interpretadas como "
             "evidência de viabilidade prática.\n")
    viabilidade = c["descritivas"][[f"P(mediana ≥ {PISO_ADEQUACAO})", "IC95 inf", "IC95 sup"]].copy()
    viabilidade["Viável (LI ≥ 0,80)"] = np.where(
        viabilidade["IC95 inf"] >= LIMIAR_VIABILIDADE, "sim", "não")
    L.append(_md(viabilidade))
    L.append("")

    # ------------------------------------------------------- Análises extras
    L.append("---\n\n## Análises descritivas adicionais (complementares à receita)\n")
    L.append("### Aderência do juiz à taxonomia da rubrica\n")
    L.append("Rótulos emitidos pelo juiz, sua forma canônica após normalização (acentos, "
             "escapes unicode quebrados e variantes de grafia) e se pertencem à taxonomia "
             "fechada prevista na rubrica.\n")
    L.append(_md(c["auditoria"], indice=False))
    L.append("")
    if not c["auditoria"].empty:
        fora = c["auditoria"][c["auditoria"]["Na rubrica"] == "não"]
        normalizados = c["auditoria"][c["auditoria"]["Normalizado"] == "sim"]
        if not normalizados.empty:
            L.append(f"{len(normalizados)} grafia(s) distinta(s) foram unificadas na "
                     "normalização; sem esse tratamento, a mesma categoria apareceria "
                     "fragmentada na contagem.\n")
        if not fora.empty:
            rotulos = ", ".join(f"`{r}`" for r in fora["Categoria canônica"].unique())
            L.append(f"⚠️ **Achado metodológico:** o juiz emitiu {int(fora['Ocorrências'].sum())} "
                     f"rótulo(s) fora da taxonomia prescrita ({rotulos}). Isso indica que o "
                     "*prompt* não restringiu efetivamente a saída à lista fechada de "
                     "categorias — vale reportar como limitação do instrumento e considerar "
                     "reforçar a restrição no *prompt* ou validar a saída por esquema.\n")
    L.append("### Categorias de problema apontadas pelo juiz (após normalização)\n")
    L.append(_md(c["problemas"]))
    L.append("")
    L.append(_md(c["taxa_problemas"]))
    L.append("")
    L.append("### Custo e desempenho do juiz LLM\n")
    L.append(_md(c["custos"], indice=False))
    L.append("")

    L.append("### Análises ainda não cobertas por este script\n")
    L.append("| Etapa | Análise | Insumo necessário |")
    L.append("|---|---|---|")
    L.append("| 1.1 | Concordância entre avaliadores humanos | notas dos 3 especialistas (Gold Set) |")
    L.append("| 1.3 | Validação humano ↔ LLM | mediana humana por documento |")
    L.append("| 7.1–7.3 | Robustez externa e *drift* (20 inéditos) | parquets do Conjunto Inéditos |")
    L.append("| 8 | Estratificação por dificuldade | rótulo fácil/médio/difícil por documento |")
    L.append("| 8 | Triangulação campo a campo | BERTScore, SBERT, ROUGE, Levenshtein |")
    L.append("")

    # ---------------------------------------------------------------- Figuras
    L.append("---\n\n## Figuras geradas\n")
    for arquivo in c["figuras"]:
        L.append(f"- `{arquivo}`")
    L.append("")
    L.append("---\n\n## Reprodutibilidade\n")
    L.append("| Item | Valor |")
    L.append("|---|---|")
    L.append(f"| Python | {sys.version.split()[0]} |")
    L.append(f"| pandas / numpy / scipy | {pd.__version__} / {np.__version__} / {__import__('scipy').__version__} |")
    L.append("| Pesos do Kappa | quadráticos |")
    L.append("| Correção múltipla | Holm-Bonferroni (step-down) |")
    L.append("| Semente para amostragem do Shapiro-Wilk | 42 |")
    L.append("")

    texto = "\n".join(L)
    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write(texto)
    return texto


# =============================================================================
# 7. Orquestração
# =============================================================================

def executar(base: str, pastas: list, saida: str, escala: tuple = ESCALA_PADRAO,
             leitor: Callable[[str], pd.DataFrame] = pd.read_parquet) -> dict:
    """Executa o pipeline completo: carga → descarte → estatística → gráficos → relatório."""
    os.makedirs(saida, exist_ok=True)

    print("→ Carregando rodadas...")
    df = carregar_rodadas(base, pastas, leitor=leitor)

    modelos = ordenar_modelos(df["modelo"].unique())
    rodadas = sorted(df["rodada"].unique())
    observadas = pd.to_numeric(df["nota"], errors="coerce").dropna()
    categorias = list(range(escala[0], escala[1] + 1))
    if len(observadas) and (observadas.min() < escala[0] or observadas.max() > escala[1]):
        print(f"  ⚠ Notas fora da escala {escala}: observado "
              f"[{observadas.min():.0f}, {observadas.max():.0f}] — escala ajustada.")
        categorias = list(range(int(min(escala[0], observadas.min())),
                                int(max(escala[1], observadas.max())) + 1))

    print(f"→ {len(modelos)} modelos, {len(rodadas)} rodadas, escala {categorias}")

    print("→ Etapa 0: descarte global pareado...")
    falhas = tabela_falhas(df, modelos, rodadas)
    n_bruto = df["documento"].nunique()
    df_valido, mantidos, descartados = aplicar_descarte_global(df, modelos, rodadas)
    print(f"  n* = {len(mantidos)} documentos ({len(descartados)} descartados)")

    if not len(mantidos):
        raise RuntimeError("Nenhum documento válido em todas as combinações modelo × rodada.")

    pivo = notas_medianas(df_valido, modelos)

    print("→ Etapa 1.2: confiabilidade teste-reteste...")
    fleiss, pares, instrumento = confiabilidade_teste_reteste(
        df_valido, modelos, rodadas, categorias)
    deriva = efeito_rodada(df_valido, modelos, rodadas)

    print("→ Etapas 2–5: normalidade, Friedman, Wilcoxon + Holm, tamanho de efeito...")
    analise = hipotese_principal(pivo)

    print("→ Etapas 6 e 8: viabilidade, problemas e custo...")
    descritivas = descritivas_por_modelo(pivo)
    problemas, taxa_problemas = analisar_problemas(df_valido, modelos)
    auditoria = auditar_rotulos_problema(df_valido)
    custos = custo_juiz(df_valido, modelos, rodadas)

    print("→ Gerando gráficos...")
    figuras = gerar_graficos(df_valido, pivo, modelos, rodadas, categorias, analise,
                             fleiss, pares, problemas, falhas, custos, saida,
                             instrumento=instrumento)

    # bases intermediárias, úteis para conferência e para as etapas ainda pendentes
    df_valido.assign(problemas=df_valido["problemas"].map(lambda p: "|".join(p))) \
             .to_csv(os.path.join(saida, "dados_longo.csv"), index=False, encoding="utf-8")
    pivo.to_csv(os.path.join(saida, "dados_notas_medianas.csv"), encoding="utf-8")

    contexto = {
        "pastas": list(pastas), "modelos": modelos, "rodadas": rodadas,
        "categorias": categorias, "juiz": next((j for j in df["juiz"].dropna().unique() if j), ""),
        "n_bruto": n_bruto, "n_valido": len(mantidos), "n_descartado": len(descartados),
        "falhas": falhas, "fleiss": fleiss, "pares": pares, "deriva": deriva,
        "instrumento": instrumento,
        "descritivas": descritivas, "analise": analise, "problemas": problemas,
        "taxa_problemas": taxa_problemas, "auditoria": auditoria,
        "custos": custos, "figuras": figuras,
    }
    contexto["sintese"] = _montar_sintese(contexto)

    print("→ Escrevendo estatisticas.md...")
    gerar_markdown(contexto, os.path.join(saida, "estatisticas.md"))
    print(f"✅ Concluído. Resultados em: {os.path.abspath(saida)}")
    return contexto


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Gera gráficos e estatísticas das avaliações do juiz LLM.")
    parser.add_argument("--base", default=".",
                        help="diretório que contém as pastas das rodadas (padrão: .)")
    parser.add_argument("--pastas", nargs="+", default=PASTAS_PADRAO,
                        help=f"pastas das rodadas, em ordem (padrão: {' '.join(PASTAS_PADRAO)})")
    parser.add_argument("--saida", default=PASTA_SAIDA_PADRAO,
                        help=f"pasta de saída (padrão: {PASTA_SAIDA_PADRAO})")
    parser.add_argument("--escala", nargs=2, type=int, default=list(ESCALA_PADRAO),
                        metavar=("MIN", "MAX"), help="limites da escala Likert (padrão: 1 4)")
    args = parser.parse_args(argv)

    executar(base=args.base, pastas=args.pastas, saida=args.saida,
             escala=tuple(args.escala))


if __name__ == "__main__":
    main()