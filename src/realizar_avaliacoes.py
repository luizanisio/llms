#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realizar_avaliacoes.py
======================

Consolida avaliações Likert (1--4) produzidas por **grupos de avaliadores** e
decide se um juiz LLM está validado para aplicação em massa.

Um *grupo* é um conjunto de avaliações independentes dos mesmos itens:

* grupo do tipo ``llm``     -> N rodadas do mesmo modelo juiz  (teste-reteste);
* grupo do tipo ``humano``  -> N especialistas distintos       (inter-avaliadores).

A matemática é a mesma nos dois casos; muda a interpretação e os rótulos.

Modos de execução
-----------------
1. **Um grupo** -> análise interna: confiabilidade global, severidade,
   descritivas, comparação entre as fontes avaliadas e viabilidade.
2. **Dois ou mais grupos** -> a análise interna de cada um **mais** a validação
   por concordância com o grupo humano de referência.

Gate de validação (tudo aferido no agregado, sem estratificação por fonte)
-------------------------------------------------------------------------
======  ======================  ==========================  =================
nº      critério                estatística                 aprova se
======  ======================  ==========================  =================
1       concordância ordinal    κw de Cohen (ponderado)     κw >= 0,60
2       ausência de viés        Wilcoxon bilateral          p > 0,05
3       decisão prática         McNemar (notas >= 3)        p > 0,05
======  ======================  ==========================  =================

O IC 95% do κw é reportado como medida de precisão, não como critério.

Camada bayesiana (opcional, `--bayes`)
--------------------------------------
Complementa o gate sem substituí-lo: acrescenta a probabilidade posterior de
**equivalência prática** — quantidade que o teste de hipótese nula não produz —
e um heatmap das relações entre fontes e entre avaliadores. Método: teste de
sinais bayesiano (Benavoli et al., 2017), via ``util_est_bayesiana``. Sem a
flag, nada disso é importado nem executado.

Entrada
-------
Uma pasta por avaliador, contendo ``saida_juiz_llm.parquet``::

    gpt5_01/, gpt5_02/, gpt5_03/          -> grupo `gpt5`    (3 rodadas do juiz)
    sabia4_01/, sabia4_02/, sabia4_03/    -> grupo `sabia4`  (3 rodadas do juiz)
    humanos01/, humanos02/, humanos03/    -> grupo `humanos` (3 especialistas)

Esquema do parquet (idêntico para juiz LLM e humanos):

============  =============================================================
coluna        conteúdo
============  =============================================================
``chave``     ``<id_documento>_<fonte>`` (o id do documento não contém ``_``)
``resposta``  JSON ``{"nota": 1..4, "problemas": ["alucinacao", ...]}``
``resumo``    JSON opcional com metadados da chamada (tokens, tempo, model)
``erro``      string de erro; vazia quando a avaliação foi bem-sucedida
============  =============================================================

Uso
---
::

    python realizar_avaliacoes.py --grupos gpt5:llm
    python realizar_avaliacoes.py --grupos gpt5:llm sabia4:llm humanos:humano \
        --alias qwen7b=a --saida analise
    python realizar_avaliacoes.py --grupos gpt5:llm humanos:humano \
        --bayes --bayes-eps 0.08
    python realizar_avaliacoes.py --pastas saida_01 saida_02 saida_03

Requisitos: pandas, numpy, scipy, pyarrow, scikit-learn, statsmodels; matplotlib e ``util_graficos`` para as
figuras (a estatística roda mesmo sem eles); ``util_est_bayesiana`` apenas quando
``--bayes`` é informado.

Autor: Luiz Anísio
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

import realizar_avaliacoes_graficos as viz

# Análise bayesiana: opcional. Só é exigida quando `--bayes` é informado; sem a
# flag, o pipeline roda exatamente como antes e não importa nada deste módulo.
try:
    import util_est_bayesiana as bayes
    BAYES_DISPONIVEL = True
except ImportError:
    bayes = None
    BAYES_DISPONIVEL = False

# -----------------------------------------------------------------------------
# Bibliotecas de referência
# -----------------------------------------------------------------------------
# Os coeficientes e testes com implementação consolidada em pacotes amplamente
# usados são calculados por eles, favorecendo a replicabilidade por terceiros.
# As definições operacionais (fórmulas internas) estão em
# `realizar_avaliacoes_teste.py`, que verifica que coincidem com os pacotes.

from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


def dependencias(bayes_ativo: bool = False) -> dict:
    """Origem efetiva de cada estatística nesta execução (vai ao relatório)."""
    import sklearn
    import statsmodels
    sk = sklearn.__version__
    sm = statsmodels.__version__
    extra = {}
    if bayes_ativo and BAYES_DISPONIVEL:
        extra["Comparação bayesiana pareada"] = (
            "baycomp (Benavoli et al., 2017), via util_est_bayesiana")
    return {
        "Kappa de Cohen ponderado": f"scikit-learn {sk}",
        "Correção de Holm": f"statsmodels {sm}",
        "Teste de McNemar": f"statsmodels {sm}",
        "IC de Wilson": f"statsmodels {sm}",
        "Kappa de Fleiss ponderado": "implementação interna (sem equivalente "
                                     "ponderado em pacote consolidado)",
        "Friedman, Wilcoxon, Shapiro-Wilk": f"scipy {__import__('scipy').__version__}",
        "Bootstrap (reamostragem de documentos)": "implementação interna",
    } | extra

# =============================================================================
# 1. Configuração
# =============================================================================

ARQUIVO_PARQUET = "saida_juiz_llm.parquet"
PASTA_SAIDA_PADRAO = "analise_avaliacoes"
PASTAS_PADRAO = ["saida_01", "saida_02", "saida_03"]

TIPOS_VALIDOS = ("llm", "humano")
ESCALA_PADRAO = (1, 4)

#: critérios de decisão
LIMIAR_KAPPA = 0.60         # κw mínimo, aferido no ponto (McHugh 2012: moderado)
LIMIAR_ALFA = 0.05          # nível de significância
PISO_ADEQUACAO = 3          # nota mínima considerada adequada
LIMIAR_VIABILIDADE = 0.80   # limite inferior do IC 95% Wilson para viabilidade

#: bootstrap do IC 95% do κw (precisão, não critério)
BOOTSTRAP_REPLICAS = 2000
SEMENTE = 42

#: ordem preferencial de exibição das fontes avaliadas
ORDEM_FONTES = ["a", "base", "qwen7b", "b", "c", "d1", "d2", "d3", "d4",
                "qwen235b", "gpt5", "sabia4"]

#: separa `chave` em documento + fonte (o id do documento não contém '_')
REGEX_CHAVE = re.compile(r"^(?P<documento>[^_]+)_(?P<fonte>.+)$")

#: taxonomia fechada de problemas prevista na rubrica (espelha o template do Label Studio)
CATEGORIAS_PROBLEMA = ["alucinacao", "erro_factual", "atribuicao_errada", "omissao",
                       "nao_consta_indev"]
ALIASES_PROBLEMA = {
    "atribucao_errada": "atribuicao_errada",
    "atribuicao_incorreta": "atribuicao_errada",
    "erro_fatual": "erro_factual",
    "erro_factico": "erro_factual",
    "alucinacoes": "alucinacao",
    "omissoes": "omissao",
    "omissao_relevante": "omissao",
    "nao_consta": "nao_consta_indev",
    "nao_consta_indevido": "nao_consta_indev",
    "nao_consta_indevidamente": "nao_consta_indev",
}

#: rótulos de relatório que dependem do tipo do grupo
ROTULOS = {
    "llm": {
        "avaliador": "Rodada", "plural": "Rodadas", "avaliadores": "rodadas",
        "artigo": "das", "sigla": "R",
        "concordancia": "Confiabilidade teste-reteste",
        "descricao": "o mesmo modelo avaliou os itens N vezes; a concordância "
                     "entre as colunas mede estabilidade do instrumento",
        "deriva": "deriva de severidade entre rodadas",
        "falha": "falha de parsing",
    },
    "humano": {
        "avaliador": "Avaliador", "plural": "Avaliadores", "avaliadores": "avaliadores",
        "artigo": "dos", "sigla": "A",
        "concordancia": "Concordância entre avaliadores",
        "descricao": "avaliadores distintos julgaram os mesmos itens; a "
                     "concordância entre as colunas mede acordo inter-avaliadores",
        "deriva": "severidade diferencial entre avaliadores",
        "falha": "item não avaliado",
    },
}


@dataclass
class Grupo:
    """Um grupo de avaliadores: nome, tipo e as pastas que o compõem."""
    nome: str
    tipo: str
    pastas: list = field(default_factory=list)

    @property
    def rotulos(self) -> dict:
        return ROTULOS[self.tipo]


#: padrões da etapa bayesiana (todos sobrescritíveis por flag)
#: A escala Likert é inteira: |diferença| <= 0,5 significa "notas iguais".
#: Não é margem arbitrada — é a tradução direta da escala, e substitui todo o
#: aparato de calibração de ε da versão anterior.
BAYES_ROPE_LIKERT = 0.5
BAYES_LIMIAR_PADRAO = 0.80       # classificação das células do heatmap
BAYES_VEREDITO_PADRAO = 0.95     # veredito juiz × referência
BAYES_AMOSTRAS_PADRAO = 50_000   # padrão do baycomp
BAYES_METODO_PADRAO = "sinais"   # SignTest: adequado a escala ordinal

#: nome da classe do baycomp por trás de cada método, para o relatório
_NOME_TESTE = {"sinais": "baycomp.SignTest",
               "postos": "baycomp.SignedRankTest",
               "t": "baycomp.CorrelatedTTest"}


@dataclass
class ConfigBayes:
    """Parâmetros repassados ao ``util_est_bayesiana`` (camada fina do baycomp).

    A etapa **só roda quando `ativo` é verdadeiro** — sem `--bayes` nada aqui é
    lido e o pipeline se comporta exatamente como antes.

    Esta etapa é integralmente **ordinal** (Likert 1–4), por isso o padrão é
    ``metodo="sinais"`` (``baycomp.SignTest``) com ``rope=0.5``. Os métodos
    "postos" e "t" existem para escores contínuos e ficam disponíveis por flag,
    mas não são o uso previsto aqui.
    """
    ativo: bool = False
    rope: float = BAYES_ROPE_LIKERT
    metodo: str = BAYES_METODO_PADRAO
    limiar: float = BAYES_LIMIAR_PADRAO
    limiar_veredito: float = BAYES_VEREDITO_PADRAO
    amostras: int = BAYES_AMOSTRAS_PADRAO
    semente: int = SEMENTE

    @property
    def kw(self) -> dict:
        """Argumentos comuns das chamadas ao módulo bayesiano."""
        return {"rope": self.rope, "metodo": self.metodo, "limiar": self.limiar,
                "nsamples": self.amostras, "seed": self.semente}


# =============================================================================
# 2. Utilidades de leitura e formatação
# =============================================================================

def _texto(valor) -> str:
    """Converte um valor em string limpa, tratando None/NaN/'nan' como vazio."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    texto = str(valor).strip()
    return "" if texto.lower() in {"nan", "none", "null", "<na>"} else texto


def _json_seguro(texto) -> dict:
    """Converte texto em dict tolerando None, cercas markdown e JSON malformado."""
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return {}
    if isinstance(texto, dict):
        return texto
    bruto = str(texto).strip()
    if not bruto:
        return {}
    if bruto.startswith("```"):
        bruto = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", bruto).strip()
    try:
        valor = json.loads(bruto)
        return valor if isinstance(valor, dict) else {}
    except json.JSONDecodeError:
        ini, fim = bruto.find("{"), bruto.rfind("}")
        if 0 <= ini < fim:
            try:
                valor = json.loads(bruto[ini:fim + 1])
                return valor if isinstance(valor, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def _para_lista(valor) -> list:
    """Normaliza o campo `problemas` (lista, string ou nulo) para lista de strings."""
    if valor is None:
        return []
    if isinstance(valor, (list, tuple, np.ndarray, pd.Series)):
        return [str(v).strip() for v in valor if str(v).strip()]
    texto = str(valor).strip()
    return [texto] if texto else []


def normalizar_problema(rotulo) -> str:
    """Reduz um rótulo de problema à taxonomia canônica da rubrica.

    Trata escapes unicode quebrados, acentuação, separadores, caixa e variantes
    de grafia. Rótulos que não colapsam em nenhuma categoria são devolvidos
    normalizados, para aparecerem no relatório de aderência à rubrica.
    """
    texto = _texto(rotulo)
    if not texto:
        return ""
    if re.search(r"u[0-9a-fA-F]{4}", texto):
        texto = re.sub(r"u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), texto)
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    texto = re.sub(r"[^a-z0-9]+", "_", texto).strip("_")
    if not texto:
        return ""
    texto = ALIASES_PROBLEMA.get(texto, texto)
    if texto not in CATEGORIAS_PROBLEMA:
        proximos = difflib.get_close_matches(texto, CATEGORIAS_PROBLEMA, n=1, cutoff=0.85)
        if proximos:
            texto = proximos[0]
    return texto


def _num(valor, casas: int = 4) -> str:
    """Formata número no padrão pt-BR (vírgula decimal, ponto de milhar)."""
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


def _p(valor) -> str:
    """Formata p-valores, com notação científica quando muito pequenos."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "—"
    if valor < 1e-4:
        return f"{valor:.2e}".replace(".", ",")
    return f"{valor:.4f}".replace(".", ",")


def _pct(valor, casas: int = 1) -> str:
    """Formata proporção 0–1 como percentual."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "—"
    return f"{_num(100 * float(valor), casas)}%"


def _decimais(serie, maximo: int) -> int:
    """Menor número de casas decimais que representa a coluna sem perda."""
    valores = [float(v) for v in serie
               if isinstance(v, (int, float, np.number))
               and not isinstance(v, (bool, np.bool_))
               and not (isinstance(v, float) and np.isnan(v))]
    if not valores:
        return 0
    for casas in range(0, maximo + 1):
        if all(abs(round(v, casas) - v) < 1e-9 for v in valores):
            return casas
    return maximo


def _md(df: pd.DataFrame, indice: bool = True, casas: int = 4) -> str:
    """Converte um DataFrame em tabela markdown com números no padrão pt-BR."""
    if df is None or len(df) == 0:
        return "_Sem dados._"
    copia = df.reset_index() if indice else df.copy()
    copia = copia.apply(lambda col: col.map(lambda v, d=_decimais(col, casas): _num(v, d)))
    cabecalho = "| " + " | ".join(str(c) for c in copia.columns) + " |"
    separador = "|" + "|".join(["---"] * len(copia.columns)) + "|"
    linhas = ["| " + " | ".join(str(v) for v in reg) + " |"
              for reg in copia.itertuples(index=False)]
    return "\n".join([cabecalho, separador] + linhas)


# =============================================================================
# 3. Carga dos dados
# =============================================================================

def expandir_grupo(base: str, prefixo: str) -> list:
    """Expande um prefixo em pastas numeradas de 01 a 99.

    Aceita os separadores usuais entre o prefixo e o número: ``gpt501``,
    ``gpt5_01`` e ``gpt5-01``. Só pastas que contenham o parquet esperado entram.
    """
    for separador in ("", "_", "-"):
        pastas = [f"{prefixo}{separador}{i:02d}" for i in range(1, 100)]
        pastas = [p for p in pastas if os.path.exists(os.path.join(base, p, ARQUIVO_PARQUET))]
        if pastas:
            return pastas
    raise FileNotFoundError(
        f"Nenhuma pasta encontrada para o grupo '{prefixo}' "
        f"(procurou {prefixo}01/{prefixo}_01/{prefixo}-01 .. 99 "
        f"em '{os.path.abspath(base)}')")


def carregar_grupo(base: str, grupo: Grupo, aliases: dict = None,
                   leitor: Callable[[str], pd.DataFrame] = pd.read_parquet) -> pd.DataFrame:
    """Lê o parquet de cada avaliador do grupo e devolve a base longa consolidada.

    Returns:
        DataFrame com uma linha por (documento, fonte, avaliador).
    """
    aliases = aliases or {}
    registros = []
    for indice, pasta in enumerate(grupo.pastas, start=1):
        caminho = os.path.join(base, pasta, ARQUIVO_PARQUET)
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
        bruto = leitor(caminho)
        faltantes = {"chave", "resposta"} - set(bruto.columns)
        if faltantes:
            raise ValueError(f"{caminho}: colunas ausentes {sorted(faltantes)}")
        parcial = _normalizar_avaliador(bruto, avaliador=indice, origem=pasta,
                                        grupo=grupo, aliases=aliases)
        registros.append(parcial)
        print(f"  ✔ {pasta}: {len(parcial):,} avaliações "
              f"({int(parcial['falha'].sum()):,} falhas)".replace(",", "."))
    return pd.concat(registros, ignore_index=True)


def _normalizar_avaliador(bruto: pd.DataFrame, avaliador: int, origem: str,
                          grupo: Grupo, aliases: dict) -> pd.DataFrame:
    """Expande as colunas JSON de um avaliador para o formato analítico longo."""
    linhas = []
    for reg in bruto.to_dict("records"):
        chave = _texto(reg.get("chave"))
        casado = REGEX_CHAVE.match(chave)
        documento = casado.group("documento") if casado else chave
        fonte = casado.group("fonte") if casado else "desconhecida"
        fonte = aliases.get(fonte, fonte)

        erro = _texto(reg.get("erro"))
        resposta = _json_seguro(reg.get("resposta"))
        meta = _json_seguro(reg.get("resumo"))

        try:
            nota = int(round(float(resposta.get("nota"))))
        except (TypeError, ValueError):
            nota = None

        if erro:
            motivo = "erro_registrado"
        elif not resposta:
            motivo = "json_invalido"
        elif nota is None:
            motivo = "nota_ausente"
        else:
            motivo = ""

        brutos = _para_lista(resposta.get("problemas"))
        linhas.append({
            "grupo": grupo.nome,
            "tipo": grupo.tipo,
            "avaliador": avaliador,
            "origem": origem,
            "documento": str(documento),
            "fonte": fonte,
            "nota": nota,
            "problemas_brutos": brutos,
            "problemas": [p for p in (normalizar_problema(x) for x in brutos) if p],
            "falha": bool(motivo),
            "motivo_falha": motivo,
            "erro": erro,
            "juiz": meta.get("model", ""),
            "prompt_tokens": pd.to_numeric(meta.get("prompt_tokens"), errors="coerce"),
            "completion_tokens": pd.to_numeric(meta.get("completion_tokens"), errors="coerce"),
            "total_tokens": pd.to_numeric(meta.get("total_tokens"), errors="coerce"),
            "tempo": pd.to_numeric(meta.get("tempo"), errors="coerce"),
        })
    return pd.DataFrame(linhas)


def ordenar_fontes(fontes: Iterable[str]) -> list:
    """Ordena as fontes pela ordem preferencial; desconhecidas ao final, alfabéticas."""
    fontes = list(dict.fromkeys(fontes))
    conhecidas = [f for f in ORDEM_FONTES if f in fontes]
    restantes = sorted(f for f in fontes if f not in conhecidas)
    return conhecidas + restantes


def definir_categorias(notas: Iterable, escala: tuple) -> list:
    """Lista de categorias da escala, expandida se houver notas fora dos limites."""
    valores = pd.to_numeric(pd.Series(list(notas)), errors="coerce").dropna()
    minimo, maximo = escala
    if len(valores):
        minimo = int(min(minimo, valores.min()))
        maximo = int(max(maximo, valores.max()))
    return list(range(minimo, maximo + 1))


# =============================================================================
# 4. Estatística — concordância
# =============================================================================

def _matriz_pesos(k: int) -> np.ndarray:
    """Matriz k x k de pesos quadráticos (1 na diagonal, decrescente fora dela)."""
    if k == 1:
        return np.ones((1, 1))
    i, j = np.indices((k, k))
    return 1.0 - ((i - j) / (k - 1)) ** 2


def fleiss_ponderado(notas: np.ndarray, categorias: list) -> dict:
    """Kappa de Fleiss ponderado para N avaliadores, com seus componentes.

    Args:
        notas: matriz (n_itens x n_avaliadores)
        categorias: categorias ordenadas da escala (ex.: [1, 2, 3, 4])

    Returns:
        dict com ``p_o`` (concordância observada), ``p_e`` (esperada por acaso),
        ``kappa`` e ``n``.
    """
    vazio = {"p_o": np.nan, "p_e": np.nan, "kappa": np.nan, "n": 0}
    notas = np.asarray(notas)
    if notas.ndim != 2 or notas.shape[0] < 2 or notas.shape[1] < 2:
        return vazio

    n, m = notas.shape
    q = len(categorias)
    indice = {c: i for i, c in enumerate(categorias)}

    contagens = np.zeros((n, q))
    for i in range(n):
        for valor in notas[i]:
            pos = indice.get(valor)
            if pos is not None:
                contagens[i, pos] += 1

    pesos = _matriz_pesos(q)
    p_o = float((np.einsum("ia,ab,ib->i", contagens, pesos, contagens) - m).sum()
                / (n * m * (m - 1)))
    prop = contagens.sum(axis=0) / (n * m)
    p_e = float(prop @ pesos @ prop)
    kappa = np.nan if np.isclose(p_e, 1.0) else (p_o - p_e) / (1.0 - p_e)
    return {"p_o": p_o, "p_e": p_e, "kappa": float(kappa), "n": n}


def cohen_ponderado(a: Iterable, b: Iterable, categorias: list) -> dict:
    """Kappa de Cohen ponderado (2 avaliadores) com pesos quadráticos.

    O coeficiente vem de ``sklearn.metrics.cohen_kappa_score``. A concordância
    observada (``p_o``) e a esperada por acaso (``p_e``) são calculadas aqui
    porque a API do sklearn não as expõe separadamente, e o relatório as
    reporta ao lado do κ.
    """
    a, b = np.asarray(list(a)), np.asarray(list(b))
    vazio = {"p_o": np.nan, "p_e": np.nan, "kappa": np.nan, "n": len(a)}
    if len(a) < 2 or len(a) != len(b):
        return vazio
    k = len(categorias)
    indice = {c: i for i, c in enumerate(categorias)}
    observado = np.zeros((k, k))
    for x, y in zip(a, b):
        i, j = indice.get(x), indice.get(y)
        if i is not None and j is not None:
            observado[i, j] += 1
    total = observado.sum()
    if total == 0:
        return vazio
    observado /= total
    esperado = np.outer(observado.sum(axis=1), observado.sum(axis=0))
    pesos = _matriz_pesos(k)
    p_o = float((pesos * observado).sum())
    p_e = float((pesos * esperado).sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        try:
            kappa = float(cohen_kappa_score(a, b, labels=list(categorias),
                                            weights="quadratic"))
        except (ValueError, ZeroDivisionError):
            kappa = np.nan
    return {"p_o": p_o, "p_e": p_e, "kappa": kappa, "n": len(a)}


def cohen_kappa(a: Iterable, b: Iterable, categorias: list) -> float:
    """Atalho: só o valor de κ ponderado de Cohen."""
    return cohen_ponderado(a, b, categorias)["kappa"]


def matriz_confusao(a: Iterable, b: Iterable, categorias: list,
                    rotulo_a: str = "A", rotulo_b: str = "B") -> pd.DataFrame:
    """Matriz de confusão k x k entre dois conjuntos de notas."""
    tabela = pd.DataFrame(0,
                          index=[f"{rotulo_a}={c}" for c in categorias],
                          columns=[f"{rotulo_b}={c}" for c in categorias])
    for x, y in zip(a, b):
        if x in categorias and y in categorias:
            tabela.loc[f"{rotulo_a}={x}", f"{rotulo_b}={y}"] += 1
    return tabela


def interpretar_kappa(valor: float) -> str:
    """Faixas de McHugh (2012), adotadas pelo arcabouço da pesquisa."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "indefinido"
    if valor < 0.21:
        return "nenhum"
    if valor < 0.40:
        return "mínimo"
    if valor < 0.60:
        return "fraco"
    if valor < 0.80:
        return "moderado"
    if valor <= 0.90:
        return "forte"
    return "quase perfeito"


def ic_bootstrap(dados: pd.DataFrame, estatistica: Callable[[pd.DataFrame], float],
                 replicas: int = None, semente: int = SEMENTE,
                 confianca: float = 0.95) -> tuple:
    """IC percentílico por bootstrap **de documentos** (respeita o agrupamento).

    Os itens são pares documento x fonte, que não são independentes entre si.
    Reamostrar documentos inteiros preserva essa estrutura.

    Args:
        dados: DataFrame com a coluna ``documento`` e as usadas pela estatística
        estatistica: função que recebe um DataFrame reamostrado e devolve um float
    """
    replicas = BOOTSTRAP_REPLICAS if replicas is None else replicas
    if dados is None or dados.empty:
        return (np.nan, np.nan)
    documentos = dados["documento"].to_numpy()
    unicos = np.unique(documentos)
    if len(unicos) < 3:
        return (np.nan, np.nan)

    posicoes = {doc: np.flatnonzero(documentos == doc) for doc in unicos}
    rng = np.random.default_rng(semente)
    valores = []
    for _ in range(replicas):
        sorteados = rng.choice(unicos, size=len(unicos), replace=True)
        indices = np.concatenate([posicoes[d] for d in sorteados])
        try:
            valor = estatistica(dados.iloc[indices])
        except Exception:
            valor = np.nan
        if valor == valor:
            valores.append(valor)
    if len(valores) < replicas * 0.5:
        return (np.nan, np.nan)
    alfa = (1 - confianca) / 2
    return (float(np.quantile(valores, alfa)), float(np.quantile(valores, 1 - alfa)))


# =============================================================================
# 5. Estatística — testes de hipótese
# =============================================================================

def ic_wilson(sucessos: int, total: int, confianca: float = 0.95) -> tuple:
    """Intervalo de confiança de Wilson para uma proporção.

    Calculado por ``statsmodels.stats.proportion.proportion_confint``.
    """
    if total == 0:
        return (np.nan, np.nan, np.nan)
    p = sucessos / total
    inferior, superior = proportion_confint(sucessos, total,
                                            alpha=1 - confianca, method="wilson")
    return (p, float(inferior), float(superior))


def correcao_holm(p_valores: Sequence[float]) -> list:
    """Correção de Holm-Bonferroni (step-down), preservando a monotonicidade.

    Calculada por ``statsmodels.stats.multitest.multipletests``.
    p-valores ausentes são tratados como 1,0 antes da correção.
    """
    m = len(p_valores)
    if m == 0:
        return []
    limpos = [1.0 if (v is None or v != v) else float(v) for v in p_valores]
    return [float(v) for v in multipletests(limpos, method="holm")[1]]


def interpretar_efeito(r: float) -> str:
    """Faixas de Cohen para o tamanho de efeito r do Wilcoxon."""
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "indefinido"
    r = abs(r)
    if r < 0.10:
        return "desprezível"
    if r < 0.30:
        return "pequeno"
    if r < 0.50:
        return "médio"
    return "grande"


def shapiro_seguro(valores, semente: int = SEMENTE) -> dict:
    """Shapiro-Wilk das diferenças pareadas, com amostragem quando n > 5000.

    Papel confirmatório: a opção não paramétrica decorre primariamente da
    natureza ordinal da escala Likert; o teste formaliza a verificação de
    normalidade citada no arcabouço metodológico.
    """
    valores = np.asarray(valores, dtype=float)
    valores = valores[~np.isnan(valores)]
    if len(valores) < 3 or np.allclose(valores, valores[0]):
        return {"W": np.nan, "p": np.nan, "n": len(valores)}
    if len(valores) > 5000:
        valores = np.random.default_rng(semente).choice(valores, 5000, replace=False)
    W, p = stats.shapiro(valores)
    return {"W": float(W), "p": float(p), "n": len(valores)}


def wilcoxon_pareado(x, y) -> dict:
    """Wilcoxon signed-rank bilateral com estatística z e efeito r = |z|/√n'.

    ``n'`` é o número de pares com diferença não nula. Com medianas Likert e
    alta concordância a maioria das diferenças é zero; os empates são
    descartados (convenção ``wilcox``), de modo que o teste responde "entre as
    discordâncias, elas são simétricas?".
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    diferencas = x - y
    n_efetivo = int(np.sum(diferencas != 0))
    saida = {"n": len(x), "n_efetivo": n_efetivo, "z": np.nan, "p": np.nan, "r": np.nan,
             "mediana_dif": float(np.median(diferencas)) if len(diferencas) else np.nan,
             "media_dif": float(np.mean(diferencas)) if len(diferencas) else np.nan}
    if n_efetivo < 1:
        saida["p"] = 1.0
        return saida
    try:
        teste = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox",
                               method="approx", correction=False)
    except ValueError:
        return saida
    saida["p"] = float(teste.pvalue)
    z = getattr(teste, "zstatistic", None)
    if z is None:
        z = stats.norm.isf(saida["p"] / 2)
    saida["z"] = float(z)
    saida["r"] = float(abs(z) / np.sqrt(n_efetivo))
    return saida


def mcnemar(a_binario, b_binario) -> dict:
    """Teste de McNemar para viés direcional em classificação binária pareada.

    Usa o teste binomial exato quando há poucos discordantes (< 25), que é o
    cenário esperado com um Gold Set de dezenas de itens; acima disso, aproximação
    qui-quadrado com correção de continuidade. Executado por
    ``statsmodels.stats.contingency_tables.mcnemar``.
    """
    a = np.asarray(list(a_binario)).astype(bool)
    b = np.asarray(list(b_binario)).astype(bool)
    b01 = int(np.sum(a & ~b))   # A adequado, B inadequado
    b10 = int(np.sum(~a & b))   # A inadequado, B adequado
    discordantes = b01 + b10
    saida = {"b01": b01, "b10": b10, "discordantes": discordantes,
             "p": np.nan, "metodo": "—"}
    if discordantes == 0:
        saida.update(p=1.0, metodo="sem discordâncias")
        return saida

    exato = discordantes < 25
    metodo = "binomial exato" if exato else "qui-quadrado com correção"
    tabela = [[int(np.sum(a & b)), b01], [b10, int(np.sum(~a & ~b))]]
    resultado = sm_mcnemar(tabela, exact=exato, correction=not exato)
    saida.update(p=float(resultado.pvalue), metodo=metodo)
    return saida


def metricas_binarias(referencia, teste) -> dict:
    """Acurácia, sensibilidade e especificidade tomando ``referencia`` como padrão."""
    ref = np.asarray(list(referencia)).astype(bool)
    tes = np.asarray(list(teste)).astype(bool)
    vp = int(np.sum(ref & tes))
    vn = int(np.sum(~ref & ~tes))
    fp = int(np.sum(~ref & tes))
    fn = int(np.sum(ref & ~tes))

    def _div(a, b):
        return float(a / b) if b else np.nan

    return {"VP": vp, "VN": vn, "FP": fp, "FN": fn,
            "acuracia": _div(vp + vn, len(ref)),
            "sensibilidade": _div(vp, vp + fn),
            "especificidade": _div(vn, vn + fp)}


# =============================================================================
# 6. Análise interna de um grupo
# =============================================================================

def aplicar_descarte_global(df: pd.DataFrame, fontes: list, avaliadores: list) -> tuple:
    """Descarte global pareado: mantém só documentos com nota válida em todas as células.

    Returns:
        (df_valido, docs_mantidos, docs_descartados)
    """
    esperado = len(fontes) * len(avaliadores)
    validos = df[(~df["falha"]) & df["nota"].notna()]
    contagem = validos.groupby("documento")[["fonte", "avaliador"]].apply(
        lambda g: len(set(zip(g["fonte"], g["avaliador"]))))
    completos = set(contagem[contagem == esperado].index)
    todos = set(df["documento"])
    return (df[df["documento"].isin(completos)].copy(),
            sorted(completos), sorted(todos - completos))


def tabela_falhas(df: pd.DataFrame, fontes: list, avaliadores: list, sigla: str) -> pd.DataFrame:
    """Taxa de falhas por fonte e avaliador."""
    linhas = []
    for fonte in fontes:
        registro = {"fonte": fonte}
        sub_fonte = df[df["fonte"] == fonte]
        for avaliador in avaliadores:
            sub = sub_fonte[sub_fonte["avaliador"] == avaliador]
            n, falhas = len(sub), int(sub["falha"].sum())
            registro[f"{sigla}{avaliador} (n)"] = n
            registro[f"{sigla}{avaliador} falhas"] = falhas
            registro[f"{sigla}{avaliador} taxa %"] = round(100.0 * falhas / n, 2) if n else np.nan
        n_total, f_total = len(sub_fonte), int(sub_fonte["falha"].sum())
        registro["Total falhas"] = f_total
        registro["Taxa global %"] = round(100.0 * f_total / n_total, 2) if n_total else np.nan
        linhas.append(registro)
    return pd.DataFrame(linhas).set_index("fonte")


def notas_medianas(df: pd.DataFrame, fontes: list) -> pd.DataFrame:
    """Variável primária: nota mediana dos avaliadores por (documento, fonte).

    Com 3 avaliadores em escala 1--4, a mediana coincide com a moda sempre que
    esta existe e é sempre um valor legítimo da escala.
    """
    return (df.pivot_table(index="documento", columns="fonte", values="nota", aggfunc="median")
              .reindex(columns=[f for f in fontes if f in set(df["fonte"])])
              .dropna())


def matriz_itens_avaliadores(df: pd.DataFrame, avaliadores: list) -> pd.DataFrame:
    """Matriz global de itens (documento × fonte) por avaliador.

    É esta matriz — e não uma por fonte — que responde ao critério: a
    concordância é aferida no instrumento como um todo.
    """
    matriz = (df.pivot_table(index=["documento", "fonte"], columns="avaliador",
                             values="nota", aggfunc="median")
                .reindex(columns=avaliadores)
                .dropna())
    return matriz.reset_index()


def concordancia_interna(df: pd.DataFrame, avaliadores: list, categorias: list) -> tuple:
    """Concordância global entre os avaliadores do grupo.

    Returns:
        (dict com o resultado global, DataFrame com os pares Cohen κw)
    """
    matriz = matriz_itens_avaliadores(df, avaliadores)
    if matriz.empty:
        return {}, pd.DataFrame()

    valores = matriz[avaliadores].to_numpy().astype(int)
    comp = fleiss_ponderado(valores, categorias)
    ic_inf, ic_sup = ic_bootstrap(
        matriz, lambda d: fleiss_ponderado(d[avaliadores].to_numpy().astype(int),
                                           categorias)["kappa"])

    global_ = {
        "n itens": len(matriz),
        "P_o": round(comp["p_o"], 4),
        "P_e": round(comp["p_e"], 4),
        "kappa": round(comp["kappa"], 4),
        "ic_inf": round(ic_inf, 4), "ic_sup": round(ic_sup, 4),
        "interpretacao": interpretar_kappa(comp["kappa"]),
        "exata": round(float(np.mean([len(set(l)) == 1 for l in valores])), 4),
        "amplitude_1": round(float(np.mean(valores.max(axis=1) - valores.min(axis=1) <= 1)), 4),
        "aprovado": bool(comp["kappa"] >= LIMIAR_KAPPA),
    }

    pares = [{"par": f"{a1} × {a2}",
              "Cohen κw": round(cohen_kappa(matriz[a1], matriz[a2], categorias), 4),
              "Concord. exata %": round(100 * float((matriz[a1] == matriz[a2]).mean()), 2)}
             for a1, a2 in combinations(avaliadores, 2)]
    return global_, pd.DataFrame(pares)


def itens_ambiguos(df: pd.DataFrame, avaliadores: list, limite: int = 2) -> pd.DataFrame:
    """Itens com alta discordância entre avaliadores (amplitude > ``limite`` pontos).

    Operacionaliza a identificação de instâncias ambíguas prevista no arcabouço:
    itens em que os avaliadores divergem em mais de 2 pontos na escala de 4.
    """
    matriz = matriz_itens_avaliadores(df, avaliadores)
    if matriz.empty:
        return pd.DataFrame()
    valores = matriz[avaliadores]
    amplitude = valores.max(axis=1) - valores.min(axis=1)
    ambiguos = matriz[amplitude > limite].copy()
    if ambiguos.empty:
        return pd.DataFrame()
    ambiguos["Amplitude"] = amplitude[amplitude > limite].astype(int)
    ambiguos = ambiguos.rename(columns={a: f"Nota {a}" for a in avaliadores})
    return ambiguos.sort_values("Amplitude", ascending=False)


def severidade_interna(df: pd.DataFrame, avaliadores: list) -> dict:
    """Friedman com avaliadores como tratamentos e itens como blocos.

    O κw mede dispersão, mas não detecta um avaliador sistematicamente mais
    severo: se todas as notas caem 1 ponto, a concordância cai sem revelar a
    direção. Este teste cobre essa lacuna.
    """
    matriz = matriz_itens_avaliadores(df, avaliadores)
    if len(matriz) < 2 or len(avaliadores) < 3:
        return {}
    medias = {a: round(float(matriz[a].mean()), 4) for a in avaliadores}
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            chi2, p = stats.friedmanchisquare(*[matriz[a].to_numpy() for a in avaliadores])
    except ValueError:
        return {"medias": medias, "chi2": np.nan, "p": np.nan, "significativo": False}
    return {"medias": medias, "n": len(matriz),
            "amplitude": round(max(medias.values()) - min(medias.values()), 4),
            "chi2": float(chi2), "gl": len(avaliadores) - 1, "p": float(p),
            "significativo": bool(p < LIMIAR_ALFA)}


def descritivas_por_fonte(pivo: pd.DataFrame) -> pd.DataFrame:
    """Descritivas da variável primária e viabilidade (proporção ≥ piso, IC Wilson)."""
    linhas = []
    for fonte in pivo.columns:
        serie = pivo[fonte]
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        prop, li, ls = ic_wilson(int((serie >= PISO_ADEQUACAO).sum()), len(serie))
        linhas.append({
            "fonte": fonte, "n": len(serie),
            "Média": round(serie.mean(), 4), "Mediana": round(serie.median(), 2),
            "DP": round(serie.std(ddof=1), 4),
            "Q1": round(q1, 2), "Q3": round(q3, 2),
            "Mín": int(serie.min()), "Máx": int(serie.max()),
            f"P(mediana ≥ {PISO_ADEQUACAO})": round(prop, 4),
            "IC95 inf": round(li, 4), "IC95 sup": round(ls, 4),
        })
    return pd.DataFrame(linhas).set_index("fonte")


def comparar_fontes(pivo: pd.DataFrame) -> dict:
    """Comparação de desempenho **entre as fontes** dentro de um grupo.

    Shapiro-Wilk das diferenças pareadas (confirmatório), Friedman omnibus,
    contrastes Wilcoxon com Holm e tamanho de efeito. A opção não paramétrica
    decorre primariamente da natureza ordinal da escala; o Shapiro formaliza a
    verificação. Convenção: rank 1 = melhor desempenho (maior nota).
    """
    fontes = list(pivo.columns)
    k, n = len(fontes), len(pivo)
    saida = {"k": k, "n": n, "fontes": fontes}

    normalidade = []
    for f1, f2 in combinations(fontes, 2):
        teste = shapiro_seguro((pivo[f1] - pivo[f2]).to_numpy())
        normalidade.append({
            "Par de fontes": f"{f1} − {f2}", "n": teste["n"],
            "W": round(teste["W"], 4) if teste["W"] == teste["W"] else np.nan,
            "p": teste["p"],
            "Normal (p>0,05)": "—" if teste["p"] != teste["p"]
                               else ("sim" if teste["p"] > LIMIAR_ALFA else "não"),
        })
    saida["normalidade"] = pd.DataFrame(normalidade)

    ranks = pivo.apply(lambda linha: stats.rankdata(-linha.to_numpy()), axis=1,
                       result_type="expand")
    ranks.columns = fontes
    saida["ranks_medios"] = ranks.mean().sort_values().rename_axis("fonte")
    saida["melhor"] = saida["ranks_medios"].index[0] if len(saida["ranks_medios"]) else None

    if k >= 3 and n >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            chi2, p_friedman = stats.friedmanchisquare(*[pivo[f].to_numpy() for f in fontes])
        saida["friedman"] = {"chi2": float(chi2), "gl": k - 1, "p": float(p_friedman),
                             "kendall_w": float(chi2 / (n * (k - 1))) if n and k > 1 else np.nan,
                             "rejeita_h0": bool(p_friedman < LIMIAR_ALFA)}
    else:
        saida["friedman"] = None

    contrastes = []
    for f1, f2 in combinations(fontes, 2):
        res = wilcoxon_pareado(pivo[f1].to_numpy(), pivo[f2].to_numpy())
        res.update(contraste=f"{f1} vs {f2}",
                   mediana_1=float(pivo[f1].median()), mediana_2=float(pivo[f2].median()),
                   media_1=float(pivo[f1].mean()), media_2=float(pivo[f2].mean()))
        contrastes.append(res)
    if contrastes:
        for contraste, p_holm in zip(contrastes, correcao_holm([c["p"] for c in contrastes])):
            contraste["p_holm"] = p_holm
            contraste["significativo"] = bool(p_holm < LIMIAR_ALFA)
            contraste["efeito"] = interpretar_efeito(contraste["r"])
    saida["contrastes"] = pd.DataFrame(contrastes)
    return saida


def auditar_rotulos_problema(df: pd.DataFrame) -> pd.DataFrame:
    """Audita a aderência dos avaliadores à taxonomia fechada da rubrica."""
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
    tabela["Na rubrica"] = np.where(tabela["Categoria canônica"].isin(CATEGORIAS_PROBLEMA),
                                    "sim", "não")
    return tabela.sort_values(["Na rubrica", "Ocorrências"], ascending=[False, False])


def analisar_problemas(df: pd.DataFrame, fontes: list) -> tuple:
    """Frequência das categorias de problema apontadas (descritivo)."""
    explodido = df.explode("problemas")
    explodido = explodido[explodido["problemas"].notna() & (explodido["problemas"] != "")]
    if explodido.empty:
        return pd.DataFrame(), pd.DataFrame()
    contagem = (explodido.pivot_table(index="problemas", columns="fonte",
                                      values="documento", aggfunc="count")
                .reindex(columns=[f for f in fontes if f in set(explodido["fonte"])])
                .fillna(0).astype(int))
    avaliacoes = df.groupby("fonte").size()
    taxa = pd.DataFrame({
        "fonte": fontes,
        "Avaliações": [int(avaliacoes.get(f, 0)) for f in fontes],
        "Com ≥1 problema": [int((df[df["fonte"] == f]["problemas"].str.len() > 0).sum())
                            for f in fontes],
    })
    taxa["Taxa %"] = (100 * taxa["Com ≥1 problema"] / taxa["Avaliações"]).round(2)
    return contagem, taxa.set_index("fonte")


def custo_avaliacao(df: pd.DataFrame, fontes: list, avaliadores: list, sigla: str) -> pd.DataFrame:
    """Consumo de tokens e tempo por fonte e avaliador (só se houver metadados)."""
    if df["total_tokens"].dropna().empty and df["tempo"].dropna().empty:
        return pd.DataFrame()
    linhas = []
    for fonte in fontes:
        for avaliador in avaliadores:
            sub = df[(df["fonte"] == fonte) & (df["avaliador"] == avaliador)]
            if sub.empty:
                continue
            linhas.append({
                "fonte": fonte, "avaliador": f"{sigla}{avaliador}", "Avaliações": len(sub),
                "Prompt tokens (méd.)": round(sub["prompt_tokens"].mean(), 1),
                "Completion tokens (méd.)": round(sub["completion_tokens"].mean(), 1),
                "Total tokens": int(sub["total_tokens"].fillna(0).sum()),
                "Tempo méd. (s)": round(sub["tempo"].mean(), 3),
            })
    return pd.DataFrame(linhas)


def analisar_grupo(base: str, grupo: Grupo, saida: str, escala: tuple,
                   aliases: dict = None,
                   leitor: Callable[[str], pd.DataFrame] = pd.read_parquet,
                   config_bayes: ConfigBayes = None) -> dict:
    """Executa a análise interna completa de um grupo e devolve o contexto de resultados."""
    os.makedirs(saida, exist_ok=True)
    rot = grupo.rotulos

    print(f"→ [{grupo.nome}] carregando {len(grupo.pastas)} {rot['avaliadores']}...")
    df = carregar_grupo(base, grupo, aliases=aliases, leitor=leitor)

    fontes = ordenar_fontes(df["fonte"].unique())
    avaliadores = sorted(df["avaliador"].unique())
    categorias = definir_categorias(df["nota"], escala)
    print(f"  {len(fontes)} fontes, {len(avaliadores)} {rot['avaliadores']}, escala {categorias}")

    falhas = tabela_falhas(df, fontes, avaliadores, rot["sigla"])
    n_bruto = df["documento"].nunique()
    df_valido, mantidos, descartados = aplicar_descarte_global(df, fontes, avaliadores)
    print(f"  descarte pareado: n* = {len(mantidos)} ({len(descartados)} descartados)")
    if not mantidos:
        raise RuntimeError(f"Grupo '{grupo.nome}': nenhum documento válido em todas as células.")

    pivo = notas_medianas(df_valido, fontes)
    interna, pares = concordancia_interna(df_valido, avaliadores, categorias)
    ambiguos = itens_ambiguos(df_valido, avaliadores)
    severidade = severidade_interna(df_valido, avaliadores)
    descritivas = descritivas_por_fonte(pivo)
    comparacao = comparar_fontes(pivo)
    problemas, taxa_problemas = analisar_problemas(df_valido, fontes)
    auditoria = auditar_rotulos_problema(df_valido)
    custos = custo_avaliacao(df_valido, fontes, avaliadores, rot["sigla"])

    resultado = {
        "grupo": grupo, "df": df_valido, "pivo": pivo,
        "fontes": fontes, "avaliadores": avaliadores, "categorias": categorias,
        "n_bruto": n_bruto, "n_valido": len(mantidos), "n_descartado": len(descartados),
        "falhas": falhas, "interna": interna, "pares": pares, "ambiguos": ambiguos,
        "severidade": severidade,
        "descritivas": descritivas, "comparacao": comparacao,
        "problemas": problemas, "taxa_problemas": taxa_problemas,
        "auditoria": auditoria, "custos": custos,
        "juiz": next((j for j in df["juiz"].dropna().unique() if j), ""),
        "saida": saida,
    }

    if config_bayes is not None and config_bayes.ativo:
        print(f"  comparação bayesiana entre as {len(fontes)} fontes...")
        resultado["bayes"] = analise_bayesiana_grupo(resultado, config_bayes)
        resultado["config_bayes"] = config_bayes

    print(f"  gerando figuras e relatório de {grupo.nome}...")
    resultado["figuras"] = viz.graficos_grupo(resultado, saida)
    resultado["figuras"] += [f for f in resultado.get("bayes", {}).get("figuras", [])
                             if f not in resultado["figuras"]]
    df_valido.assign(problemas=df_valido["problemas"].map("|".join)) \
             .drop(columns=["problemas_brutos"]) \
             .to_csv(os.path.join(saida, "dados_longo.csv"), index=False, encoding="utf-8")
    pivo.to_csv(os.path.join(saida, "notas_medianas.csv"), encoding="utf-8")
    if not ambiguos.empty:
        ambiguos.to_csv(os.path.join(saida, "itens_ambiguos.csv"),
                        index=False, encoding="utf-8")
    escrever_relatorio_grupo(resultado, os.path.join(saida, "estatisticas.md"))
    return resultado


# =============================================================================
# 6.5 Análise bayesiana (opcional — só com `--bayes`)
# =============================================================================
# Camada **complementar**, não substitutiva: Wilcoxon, McNemar, Friedman e κw
# continuam sendo o que decide o gate. A leitura bayesiana acrescenta o que o
# teste de hipótese nula não consegue expressar — a probabilidade posterior de
# equivalência prática — e trata "equivalente" como achado, não como falha em
# rejeitar H₀.
#
# Toda a estatística vem do baycomp, via `util_est_bayesiana`. Aqui só se
# organizam as chamadas e se formatam os resultados.


def matriz_bayesiana(dados: pd.DataFrame, colunas: list, cfg: ConfigBayes) -> pd.DataFrame:
    """Compara todos os pares das colunas indicadas."""
    return bayes.matriz_pares(dados, nomes=list(colunas), **cfg.kw)


def tabela_matriz_bayesiana(matriz: pd.DataFrame, rotulo: str = "Par") -> pd.DataFrame:
    """Formata a matriz para leitura: um par NÃO ordenado por linha."""
    vistos, linhas = set(), []
    for _, linha in matriz.iterrows():
        par = frozenset((linha["linha"], linha["coluna"]))
        if par in vistos:
            continue
        vistos.add(par)
        linhas.append({
            f"{rotulo} (A × B)": f"{linha['linha']} × {linha['coluna']}",
            "n": int(linha["n"]),
            "A melhor": int(linha["x_melhor"]),
            "Empate": int(linha["empate"]),
            "B melhor": int(linha["y_melhor"]),
            "Dif. média": round(float(linha["diferenca_media"]), 4),
            "P(A > B)": round(float(linha["p_esquerda"]), 4),
            "P(equiv.)": round(float(linha["p_rope"]), 4),
            "P(A < B)": round(float(linha["p_direita"]), 4),
            "Relação de A": linha["classificacao"],
        })
    return pd.DataFrame(linhas)


def analise_bayesiana_grupo(r: dict, cfg: ConfigBayes) -> dict:
    """Comparação bayesiana entre as fontes, dentro de um grupo.

    A unidade é a nota mediana por documento — a mesma variável primária do
    Friedman/Wilcoxon. O que muda é a pergunta: em vez de "há diferença
    detectável?", responde "qual a probabilidade de esta fonte superar aquela, e
    qual a de serem praticamente equivalentes?".
    """
    if len(r["fontes"]) < 2:
        return {}
    matriz = matriz_bayesiana(r["pivo"], r["fontes"], cfg)
    figuras = viz.grafico_bayes(
        matriz, os.path.join(r["saida"], "10_bayes_fontes.png"),
        titulo="Comparação bayesiana entre as fontes (nota mediana)",
        rotulo_entidade="fonte")
    matriz.to_csv(os.path.join(r["saida"], "bayes_fontes.csv"),
                  index=False, encoding="utf-8")
    return {"config": cfg, "matriz": matriz,
            "resumo": bayes.resumo(matriz),
            "tabela": tabela_matriz_bayesiana(matriz, "Fonte"),
            "figuras": figuras}


def bayes_par(longo: pd.DataFrame, juiz: str, referencia: str,
              cfg: ConfigBayes) -> dict:
    """Leitura bayesiana dos critérios 2 e 3 do gate, para um par (juiz, referência).

    Duas comparações pareadas sobre a mesma interseção:

    * **notas** — Likert 1–4, com ROPE = 0,5 ("notas iguais");
    * **decisão** — a binarização em `nota >= piso`, análogo bayesiano do
      McNemar. Também com ROPE = 0,5: sobre valores 0/1 as diferenças só valem
      0 ou ±1, e 0,5 separa exatamente "mesma decisão" de "decisão oposta".

    O veredito é **complementar** ao gate frequentista, nunca o substitui.
    """
    a = longo[juiz].astype(float).to_numpy()
    b = longo[referencia].astype(float).to_numpy()

    notas = bayes.Comparacao(a, b, **cfg.kw)
    decisao = bayes.Comparacao((a >= PISO_ADEQUACAO).astype(float),
                               (b >= PISO_ADEQUACAO).astype(float), **cfg.kw)

    p_equiv_notas = notas.probabilidades["p_rope"]
    p_equiv_decisao = decisao.probabilidades["p_rope"]
    limiar = cfg.limiar_veredito

    # O veredito julga MAGNITUDE contra a ROPE, nunca direção: um juiz pode ser
    # confiavelmente mais leniente por uma margem sem relevância prática.
    if p_equiv_notas >= limiar and p_equiv_decisao >= limiar:
        status = "SEM VIÉS RELEVANTE"
    elif (1 - p_equiv_notas) >= limiar or (1 - p_equiv_decisao) >= limiar:
        status = "VIÉS RELEVANTE"
    else:
        status = "INCONCLUSIVO"

    contagens = notas.contagens
    return {
        "juiz": juiz, "referencia": referencia, "n": len(a),
        "acima": contagens["x_melhor"], "empate": contagens["empate"],
        "abaixo": contagens["y_melhor"],
        "diferenca_media": notas.diferenca_media,
        "p_juiz_maior": notas.probabilidades["p_esquerda"],
        "p_equiv": p_equiv_notas,
        "p_juiz_menor": notas.probabilidades["p_direita"],
        "decisao_so_juiz": decisao.contagens["x_melhor"],
        "decisao_concordante": decisao.contagens["empate"],
        "decisao_so_referencia": decisao.contagens["y_melhor"],
        "decisao_p_equiv": p_equiv_decisao,
        "classificacao": notas.classificacao,
        "status": status,
    }


def analise_bayesiana_validacao(v: dict, cfg: ConfigBayes) -> dict:
    """Etapa bayesiana da validação: pares juiz × referência e matriz entre grupos."""
    longo, nomes = v["longo"], list(v["pivos"].keys())
    pares = [bayes_par(longo, juiz, v["referencia"], cfg) for juiz in v["juizes"]]

    matriz = None
    figuras = []
    if len(nomes) >= 2:
        matriz = matriz_bayesiana(longo[nomes], nomes, cfg)
        figuras = viz.grafico_bayes(
            matriz, os.path.join(v["saida"], "04_bayes_grupos.png"),
            titulo="Comparação bayesiana entre os avaliadores (nota mediana)",
            rotulo_entidade="avaliador")
        matriz.to_csv(os.path.join(v["saida"], "bayes_grupos.csv"),
                      index=False, encoding="utf-8")

    tabela = tabela_bayes_juizes(pares)
    tabela.to_csv(os.path.join(v["saida"], "tabela_bayes_juizes.csv"),
                  index=False, encoding="utf-8")
    return {"config": cfg, "pares": pares, "tabela": tabela, "matriz": matriz,
            "resumo": bayes.resumo(matriz) if matriz is not None else None,
            "tabela_matriz": (tabela_matriz_bayesiana(matriz, "Avaliador")
                              if matriz is not None else None),
            "figuras": figuras}


def tabela_bayes_juizes(pares: list) -> pd.DataFrame:
    """Uma linha por juiz: direção, equivalência nas notas e na decisão prática."""
    return pd.DataFrame([{
        "Juiz": p["juiz"], "n": p["n"],
        "Acima": p["acima"], "Empate": p["empate"], "Abaixo": p["abaixo"],
        "Dif. média": round(p["diferenca_media"], 4),
        "P(juiz > ref.)": round(p["p_juiz_maior"], 4),
        "P(equiv. notas)": round(p["p_equiv"], 4),
        "P(equiv. decisão)": round(p["decisao_p_equiv"], 4),
        "Leitura": p["status"],
    } for p in pares])


# =============================================================================
# 7. Validação por concordância entre grupos
# =============================================================================

def intersecar_grupos(resultados: dict) -> tuple:
    """Interseção estrita de **documentos e fontes** entre todos os grupos.

    A comparação exige pareamento estrito: um item só entra se todos os grupos o
    avaliaram. Itens exclusivos permanecem na análise interna do seu grupo.

    Returns:
        (pivos_recortados, documentos, fontes, tabela_cobertura)
    """
    docs = set.intersection(*(set(r["pivo"].index.astype(str)) for r in resultados.values()))
    fontes = set.intersection(*(set(r["pivo"].columns) for r in resultados.values()))
    documentos = sorted(docs)
    fontes_ordenadas = ordenar_fontes(fontes)

    cobertura = pd.DataFrame([{
        "grupo": nome,
        "tipo": r["grupo"].tipo,
        "documentos": len(r["pivo"]),
        "fontes": len(r["pivo"].columns),
        "docs fora da interseção": len(set(r["pivo"].index.astype(str)) - docs),
        "fontes fora da interseção": ", ".join(sorted(set(r["pivo"].columns) - fontes)) or "—",
    } for nome, r in resultados.items()]).set_index("grupo")

    if not documentos or not fontes_ordenadas:
        raise RuntimeError(
            "Interseção vazia entre os grupos.\n"
            f"  documentos em comum: {len(documentos)}\n"
            f"  fontes em comum: {len(fontes_ordenadas)}\n"
            "Verifique o formato da chave (<documento>_<fonte>) e o mapa --alias.")

    pivos = {nome: r["pivo"].copy().set_axis(r["pivo"].index.astype(str))
                     .loc[documentos, fontes_ordenadas]
             for nome, r in resultados.items()}
    return pivos, documentos, fontes_ordenadas, cobertura


def montar_longo(pivos: dict, fontes: list) -> pd.DataFrame:
    """Empilha os pivôs num formato longo: documento, fonte e uma coluna por grupo."""
    nomes = list(pivos.keys())
    linhas = []
    for fonte in fontes:
        bloco = {"documento": pivos[nomes[0]].index.astype(str), "fonte": fonte}
        bloco.update({nome: pivos[nome][fonte].to_numpy() for nome in nomes})
        linhas.append(pd.DataFrame(bloco))
    return pd.concat(linhas, ignore_index=True)


def avaliar_par(longo: pd.DataFrame, juiz: str, referencia: str,
                categorias: list, margem: float = np.nan) -> dict:
    """Aplica os três critérios do gate a um par (juiz, referência).

    Tudo no agregado — todas as combinações documento × fonte empilhadas numa
    única série. Sem estratificação por fonte.

    Resultados possíveis:

    * **VALIDADO** — os três critérios atendidos.
    * **VALIDADO COM RESSALVA** — critérios 1 e 3 atendidos e viés
      estatisticamente significativo (critério 2), porém com magnitude abaixo da
      ``margem`` de relevância prática (0,5 DP das notas da referência, regra da
      meia-DP de Norman, Sloan & Wyrwich, 2003). Classificação descritiva, não
      teste de equivalência.
    * **NÃO VALIDADO** — demais casos.
    """
    sub = longo[["documento", juiz, referencia]].rename(
        columns={juiz: "a", referencia: "b"})
    a = sub["a"].astype(int).to_numpy()
    b = sub["b"].astype(int).to_numpy()

    # critério 1 — concordância ordinal
    comp = cohen_ponderado(a, b, categorias)
    ic_inf, ic_sup = ic_bootstrap(
        sub, lambda d: cohen_kappa(d["a"].astype(int), d["b"].astype(int), categorias))

    # critério 2 — ausência de viés sistemático
    teste = wilcoxon_pareado(a, b)

    # critério 3 — decisão prática (binarização no piso de adequação)
    bin_a, bin_b = a >= PISO_ADEQUACAO, b >= PISO_ADEQUACAO
    mc = mcnemar(bin_a, bin_b)
    met = metricas_binarias(referencia=bin_b, teste=bin_a)

    criterios = {
        "concordancia": bool(comp["kappa"] == comp["kappa"] and comp["kappa"] >= LIMIAR_KAPPA),
        "sem_vies": bool(teste["p"] == teste["p"] and teste["p"] > LIMIAR_ALFA),
        "decisao": bool(mc["p"] == mc["p"] and mc["p"] > LIMIAR_ALFA),
    }
    vies_toleravel = bool(margem == margem and abs(teste["media_dif"]) < margem)
    if all(criterios.values()):
        status = "VALIDADO"
    elif criterios["concordancia"] and criterios["decisao"] and vies_toleravel:
        status = "VALIDADO COM RESSALVA"
    else:
        status = "NÃO VALIDADO"
    media_dif = teste["media_dif"]
    if abs(media_dif) < 1e-9:
        direcao = "—"
    else:
        direcao = (f"{juiz} mais leniente" if media_dif > 0
                   else f"{referencia} mais leniente")

    return {
        "juiz": juiz, "referencia": referencia, "n": len(a),
        "kappa": comp["kappa"], "ic_inf": ic_inf, "ic_sup": ic_sup,
        "p_o": comp["p_o"], "interpretacao": interpretar_kappa(comp["kappa"]),
        "exata": float(np.mean(a == b)), "amplitude_1": float(np.mean(np.abs(a - b) <= 1)),
        "media_dif": media_dif, "mediana_dif": teste["mediana_dif"],
        "n_efetivo": teste["n_efetivo"], "z": abs(teste["z"]), "p_wilcoxon": teste["p"],
        "r": teste["r"], "efeito": interpretar_efeito(teste["r"]), "direcao": direcao,
        "kappa_bin": cohen_kappa(bin_a.astype(int), bin_b.astype(int), [0, 1]),
        "p_mcnemar": mc["p"], "metodo_mcnemar": mc["metodo"],
        "discordantes": mc["discordantes"], "b01": mc["b01"], "b10": mc["b10"],
        "acuracia": met["acuracia"], "sensibilidade": met["sensibilidade"],
        "especificidade": met["especificidade"],
        "criterios": criterios, "status": status,
        "validado": status == "VALIDADO",
        "margem": margem, "vies_toleravel": vies_toleravel,
        "confusao": matriz_confusao(a, b, categorias, juiz, referencia),
        "diferencas": a - b,
    }


def comparar_grupos(longo: pd.DataFrame, nomes: list) -> dict:
    """Friedman entre os grupos, com post-hoc Wilcoxon + Holm.

    Grupos como tratamentos, itens (documento × fonte) como blocos. Responde
    "algum avaliador é sistematicamente mais severo?" e, no post-hoc, "quais
    diferem entre si". Convenção do rank: 1 = notas mais altas (mais leniente).

    Contextual, não integra o gate: o critério 2 lê o p **não corrigido** do par
    juiz × referência, que é a leitura mais exigente para o juiz.
    """
    matriz = longo[nomes]
    ranks = matriz.apply(lambda linha: stats.rankdata(-linha.to_numpy()), axis=1,
                         result_type="expand")
    ranks.columns = nomes

    saida = {"n": len(matriz), "friedman": None,
             "ranks": pd.DataFrame({
                 "Rank médio": ranks.mean().round(3),
                 "Nota média": matriz.mean().round(4),
                 "Nota mediana": matriz.median().round(2),
             }).rename_axis("grupo").sort_values("Rank médio")}

    if len(nomes) >= 3 and len(matriz) >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            chi2, p = stats.friedmanchisquare(*[matriz[n].to_numpy() for n in nomes])
        saida["friedman"] = {"chi2": float(chi2), "gl": len(nomes) - 1, "p": float(p),
                             "rejeita_h0": bool(p < LIMIAR_ALFA)}

    contrastes = []
    for a, b in combinations(nomes, 2):
        res = wilcoxon_pareado(matriz[a].to_numpy(), matriz[b].to_numpy())
        res.update(contraste=f"{a} vs {b}")
        contrastes.append(res)
    if contrastes:
        for contraste, p_holm in zip(contrastes, correcao_holm([c["p"] for c in contrastes])):
            contraste["p_holm"] = p_holm
            contraste["significativo"] = bool(p_holm < LIMIAR_ALFA)
            contraste["efeito"] = interpretar_efeito(contraste["r"])
    saida["contrastes"] = pd.DataFrame(contrastes)
    return saida


def tabela_ranks(pivos: dict, fontes: list) -> pd.DataFrame:
    """Ranks médios das fontes segundo cada grupo — descritivo, sem teste.

    Com poucas fontes na interseção, qualquer teste de correlação entre
    ordenações teria n insuficiente. A tabela serve apenas para mostrar se os
    grupos ordenam as fontes da mesma maneira.
    """
    ranks = {}
    for nome, pivo in pivos.items():
        r = pivo.apply(lambda linha: stats.rankdata(-linha.to_numpy()), axis=1,
                       result_type="expand")
        r.columns = pivo.columns
        ranks[nome] = r.mean().round(3)
    tabela = pd.DataFrame(ranks).reindex(fontes)
    tabela.index.name = "fonte"
    return tabela


def validar_juizes(resultados: dict, saida: str, escala: tuple,
                   referencia: str = None, config_bayes: ConfigBayes = None) -> dict:
    """Orquestra a validação dos juízes LLM contra o grupo humano de referência."""
    os.makedirs(saida, exist_ok=True)
    pivos, documentos, fontes, cobertura = intersecar_grupos(resultados)
    print(f"→ interseção: {len(documentos)} documentos × {len(fontes)} fontes "
          f"= {len(documentos) * len(fontes)} itens pareados")

    categorias = definir_categorias(
        np.concatenate([p.to_numpy().ravel() for p in pivos.values()]), escala)

    humanos = [n for n, r in resultados.items() if r["grupo"].tipo == "humano"]
    if referencia is None:
        referencia = humanos[0] if humanos else list(pivos.keys())[0]
    referencia_humana = resultados[referencia]["grupo"].tipo == "humano"
    print(f"  grupo de referência: {referencia} "
          f"({'humano' if referencia_humana else 'LLM'})")
    if not referencia_humana:
        print("  ⚠ ATENÇÃO: referência do tipo LLM — os resultados são um ensaio "
              "metodológico, não validação contra especialistas humanos.")
    if len(humanos) > 1:
        print(f"  ⚠ ATENÇÃO: {len(humanos)} grupos do tipo humano "
              f"({', '.join(humanos)}); usando '{referencia}' como referência. "
              "Se os especialistas foram exportados como grupos separados, "
              "consolide-os num único grupo (um avaliador por pasta).")

    longo = montar_longo(pivos, fontes)
    juizes = [n for n in pivos if n != referencia]

    # margem de relevância prática: 0,5 DP das notas da referência
    # (regra da meia-DP; Norman, Sloan & Wyrwich, 2003)
    dp_referencia = float(longo[referencia].std(ddof=1))
    margem = 0.5 * dp_referencia if dp_referencia == dp_referencia else np.nan

    gates = [avaliar_par(longo, juiz, referencia, categorias, margem=margem)
             for juiz in juizes]
    laterais = [avaliar_par(longo, a, b, categorias)
                for a, b in combinations(juizes, 2)]

    resultado = {
        "grupos": {n: r["grupo"] for n, r in resultados.items()},
        "pivos": pivos, "longo": longo, "cobertura": cobertura,
        "documentos": documentos, "fontes": fontes, "categorias": categorias,
        "referencia": referencia, "referencia_humana": referencia_humana,
        "grupos_humanos": humanos, "juizes": juizes,
        "dp_referencia": dp_referencia, "margem": margem,
        "gates": gates, "laterais": laterais,
        "entre_grupos": comparar_grupos(longo, list(pivos.keys())),
        "ranks": tabela_ranks(pivos, fontes), "saida": saida,
    }

    if config_bayes is not None and config_bayes.ativo:
        print("  comparação bayesiana entre os avaliadores...")
        resultado["bayes"] = analise_bayesiana_validacao(resultado, config_bayes)
        resultado["config_bayes"] = config_bayes

    print("  gerando figuras e relatório de validação...")
    resultado["figuras"] = viz.graficos_validacao(resultado, saida)
    resultado["figuras"] += [f for f in resultado.get("bayes", {}).get("figuras", [])
                             if f not in resultado["figuras"]]
    longo.to_csv(os.path.join(saida, "notas_medianas_pareadas.csv"),
                 index=False, encoding="utf-8")
    _tabela_concordancia(gates).to_csv(
        os.path.join(saida, "tabela_concordancia.csv"), index=False, encoding="utf-8")
    _tabela_decisao(gates).to_csv(
        os.path.join(saida, "tabela_decisao_binaria.csv"), index=False, encoding="utf-8")
    resultado["entre_grupos"]["contrastes"].to_csv(
        os.path.join(saida, "tabela_contrastes_grupos.csv"), index=False, encoding="utf-8")
    resultado["ranks"].to_csv(
        os.path.join(saida, "tabela_ranks_fontes.csv"), encoding="utf-8")
    escrever_relatorio_validacao(resultado, os.path.join(saida, "validacao.md"))

    icones = {"VALIDADO": "✅", "VALIDADO COM RESSALVA": "🟡", "NÃO VALIDADO": "❌"}
    for gate in gates:
        print(f"  {icones[gate['status']]} {gate['status']}: {gate['juiz']} "
              f"(κw = {gate['kappa']:.3f}, Wilcoxon p = {gate['p_wilcoxon']:.4f}, "
              f"McNemar p = {gate['p_mcnemar']:.4f}, "
              f"dif. média = {gate['media_dif']:+.3f})")
    return resultado


# =============================================================================
# 8. Relatório — análise interna do grupo
# =============================================================================

def escrever_relatorio_grupo(r: dict, caminho: str) -> str:
    """Escreve o `estatisticas.md` da análise interna de um grupo."""
    grupo, rot = r["grupo"], r["grupo"].rotulos
    interna = r["interna"]
    L = []

    L.append(f"# Análise do grupo `{grupo.nome}` ({grupo.tipo})\n")
    L.append(f"Gerado em {datetime.now():%d/%m/%Y %H:%M} por `realizar_avaliacoes.py`.\n")
    L.append("| Parâmetro | Valor |")
    L.append("|---|---|")
    L.append(f"| Tipo do grupo | {grupo.tipo} — {rot['descricao']} |")
    L.append(f"| {rot['plural']} | {len(r['avaliadores'])} ({', '.join(grupo.pastas)}) |")
    L.append(f"| Fontes avaliadas (*k*) | {len(r['fontes'])} — {', '.join(r['fontes'])} |")
    L.append(f"| Escala Likert | {r['categorias'][0]}–{r['categorias'][-1]} |")
    if grupo.tipo == "llm" and r["juiz"]:
        L.append(f"| Modelo juiz | {r['juiz']} |")
    L.append(f"| Documentos brutos | {_num(r['n_bruto'], 0)} |")
    L.append(f"| Documentos pareados (*n\\**) | {_num(r['n_valido'], 0)} |")
    L.append(f"| Documentos descartados | {_num(r['n_descartado'], 0)} |")
    L.append(f"| Nível de significância (α) | {_num(LIMIAR_ALFA, 2)} |")
    L.append("")
    L.append(f"> **Variável primária:** nota Likert **mediana** {rot['artigo']} "
             f"{len(r['avaliadores'])} {rot['avaliadores']}, por documento e fonte. Com 3 "
             f"{rot['avaliadores']} em escala de 4 pontos, a mediana coincide com a moda "
             "sempre que esta existe e é sempre um valor legítimo da escala.\n")

    L.append("### Síntese\n")
    for item in _sintese_grupo(r):
        L.append(f"- {item}")
    L.append("")

    # --- falhas e descarte ---------------------------------------------------
    L.append("---\n\n## Falhas e descarte global pareado\n")
    L.append(f"Critério de {rot['falha']}: erro registrado, JSON não interpretável ou ausência "
             "do campo `nota`. Documentos com falha em **qualquer** célula fonte × "
             f"{rot['avaliador'].lower()} são removidos de **todas** as fontes, preservando o "
             "pareamento estrito exigido por Friedman e Wilcoxon.\n")
    L.append(_md(r["falhas"]))
    L.append("")
    taxa = 100.0 * r["n_descartado"] / max(r["n_bruto"], 1)
    L.append(f"**n\\* = {r['n_valido']} documentos** completos em {len(r['fontes'])} × "
             f"{len(r['avaliadores'])} = {len(r['fontes']) * len(r['avaliadores'])} células "
             f"({r['n_descartado']} descartados, {_num(taxa, 2)}%).\n")

    # --- concordância interna ------------------------------------------------
    L.append(f"---\n\n## {rot['concordancia']}\n")
    if interna:
        L.append(f"Aferida de forma **global**: todas as combinações documento × fonte "
                 f"empilhadas numa única matriz de itens × {rot['avaliadores']}.\n")
        L.append("| Métrica | Valor |")
        L.append("|---|---|")
        L.append(f"| Itens (documento × fonte) | {_num(interna['n itens'], 0)} |")
        L.append(f"| {rot['plural']} | {len(r['avaliadores'])} |")
        L.append(f"| Concordância observada P_o | {_num(interna['P_o'])} |")
        L.append(f"| Concordância esperada por acaso P_e | {_num(interna['P_e'])} |")
        L.append(f"| **Fleiss κw** | **{_num(interna['kappa'])}** "
                 f"[IC 95%: {_num(interna['ic_inf'])}; {_num(interna['ic_sup'])}] |")
        L.append(f"| Interpretação | nível *{interna['interpretacao']}* |")
        L.append(f"| Concordância exata | {_pct(interna['exata'])} |")
        L.append(f"| Itens com amplitude ≤ 1 ponto | {_pct(interna['amplitude_1'])} |")
        L.append("")
        if interna["aprovado"]:
            L.append(f"✅ **Aprovado** — κw = {_num(interna['kappa'])} ≥ "
                     f"{_num(LIMIAR_KAPPA, 2)}. A variável primária deste grupo pode ser usada "
                     "sem ressalva de confiabilidade.\n")
        else:
            L.append(f"⚠️ **Reprovado** — κw = {_num(interna['kappa'])} < "
                     f"{_num(LIMIAR_KAPPA, 2)}. Os resultados deste grupo devem ser reportados "
                     "com ressalva. Se `P_o` estiver alto, a queda decorre de efeito de teto "
                     "da escala, não de instabilidade real.\n")

        L.append(f"### Concordância par a par entre {rot['avaliadores']} (Cohen κw)\n")
        L.append(_md(r["pares"], indice=False))
        L.append("")

    L.append("### Instâncias ambíguas (amplitude > 2 pontos)\n")
    if r["ambiguos"].empty:
        L.append(f"✅ Nenhum item com discordância superior a 2 pontos entre "
                 f"{rot['avaliadores']}.\n")
    else:
        L.append(f"⚠️ {len(r['ambiguos'])} item(ns) com discordância superior a 2 pontos — "
                 "instâncias ambíguas a reportar:\n")
        L.append(_md(r["ambiguos"].head(30), indice=False))
        L.append("")
        if len(r["ambiguos"]) > 30:
            L.append(f"_Exibindo 30 de {len(r['ambiguos'])}; lista completa em "
                     "`itens_ambiguos.csv`._\n")

    severidade = r["severidade"]
    L.append(f"### {rot['deriva'].capitalize()}\n")
    if severidade:
        L.append(f"Friedman com {rot['avaliadores']} como tratamentos e itens como blocos. "
                 "O κw mede dispersão, mas não detecta severidade sistemática; este teste "
                 "cobre essa lacuna.\n")
        medias = ", ".join(f"{rot['sigla']}{a} = {_num(m, 3)}"
                           for a, m in severidade["medias"].items())
        L.append(f"Médias: {medias} (amplitude {_num(severidade['amplitude'], 3)}). "
                 f"χ²_F = {_num(severidade['chi2'], 3)}, gl = {severidade.get('gl', '—')}, "
                 f"p = {_p(severidade['p'])}.\n")
        L.append(("⚠️ Há severidade sistemática entre "
                  f"{rot['avaliadores']}.\n") if severidade["significativo"]
                 else "✅ Sem evidência de severidade sistemática.\n")
    else:
        L.append(f"_Não computável (exige ao menos 3 {rot['avaliadores']})._\n")

    # --- descritivas e comparação entre fontes -------------------------------
    L.append("---\n\n## Descritivas da variável primária\n")
    L.append(_md(r["descritivas"]))
    L.append("")

    comparacao = r["comparacao"]
    L.append("---\n\n## Comparação entre as fontes avaliadas\n")
    L.append("**H₀:** as distribuições da nota mediana são iguais para todas as fontes. A "
             "opção não paramétrica decorre da natureza ordinal da escala Likert; o "
             "Shapiro-Wilk abaixo formaliza a verificação de normalidade das diferenças "
             "pareadas.\n")

    L.append("### Normalidade das diferenças pareadas (Shapiro-Wilk)\n")
    normalidade = comparacao["normalidade"].copy()
    if not normalidade.empty:
        normalidade["p"] = normalidade["p"].map(_p)
        L.append(_md(normalidade, indice=False))
        L.append("")
    else:
        L.append("_Não computável._\n")

    friedman = comparacao["friedman"]
    L.append("### Omnibus de Friedman\n")
    if friedman:
        L.append("| Estatística | Valor |")
        L.append("|---|---|")
        L.append(f"| χ²_F | {_num(friedman['chi2'])} |")
        L.append(f"| Graus de liberdade | {friedman['gl']} |")
        L.append(f"| p-valor | {_p(friedman['p'])} |")
        L.append(f"| W de Kendall | {_num(friedman['kendall_w'])} |")
        L.append(f"| n (documentos) | {comparacao['n']} |")
        L.append("")
        L.append(("**H₀ rejeitada** — há diferença entre fontes; os contrastes post-hoc estão "
                  "autorizados.\n") if friedman["rejeita_h0"] else
                 ("**H₀ não rejeitada** — sem evidência de diferença entre fontes; os "
                  "contrastes a seguir são exploratórios.\n"))
    else:
        L.append("_Não computado: exige k ≥ 3 fontes e n ≥ 2 documentos._\n")

    L.append("### Ranks médios (1 = melhor)\n")
    L.append(_md(comparacao["ranks_medios"].to_frame("Rank médio")))
    L.append("")

    L.append("### Contrastes post-hoc (Wilcoxon + Holm)\n")
    contrastes = comparacao["contrastes"]
    if not contrastes.empty:
        tabela = pd.DataFrame({
            "Contraste": contrastes["contraste"],
            "Mediana A": contrastes["mediana_1"], "Mediana B": contrastes["mediana_2"],
            "Média A": contrastes["media_1"].round(4), "Média B": contrastes["media_2"].round(4),
            "n′": contrastes["n_efetivo"], "z (abs.)": contrastes["z"].abs().round(4),
            "p bruto": contrastes["p"].map(_p), "p Holm": contrastes["p_holm"].map(_p),
            "r": contrastes["r"].round(4), "Efeito": contrastes["efeito"],
            "Signif.": contrastes["significativo"].map({True: "sim", False: "não"}),
        })
        L.append(_md(tabela, indice=False))
        L.append("")
        L.append(f"Contrastes significativos após Holm: "
                 f"**{int(contrastes['significativo'].sum())}** de {len(contrastes)}.\n")
    else:
        L.append("_Sem contrastes computáveis._\n")

    L.append("---\n\n## Viabilidade de produção (descritivo)\n")
    L.append(f"Proporção de documentos com nota mediana ≥ {PISO_ADEQUACAO}, com IC 95% Wilson. "
             f"Não é teste formal: proporções cujo **limite inferior ≥ "
             f"{_num(LIMIAR_VIABILIDADE, 2)}** são lidas como evidência de viabilidade prática.\n")
    viabilidade = r["descritivas"][[f"P(mediana ≥ {PISO_ADEQUACAO})",
                                    "IC95 inf", "IC95 sup"]].copy()
    viabilidade["Viável"] = np.where(viabilidade["IC95 inf"] >= LIMIAR_VIABILIDADE, "sim", "não")
    L.append(_md(viabilidade))
    L.append("")

    # --- comparação bayesiana ------------------------------------------------
    L.extend(_bloco_bayes_grupo(r))

    # --- descritivos complementares -----------------------------------------
    if not r["auditoria"].empty or not r["problemas"].empty or not r["custos"].empty:
        L.append("---\n\n## Análises complementares\n")
    if not r["auditoria"].empty:
        L.append("### Aderência à taxonomia da rubrica\n")
        L.append(_md(r["auditoria"], indice=False))
        L.append("")
        fora = r["auditoria"][r["auditoria"]["Na rubrica"] == "não"]
        if not fora.empty:
            L.append(f"⚠️ {int(fora['Ocorrências'].sum())} ocorrência(s) fora da taxonomia "
                     "prescrita — o *prompt* (ou a instrução aos avaliadores) não restringiu "
                     "efetivamente a saída à lista fechada.\n")
    if not r["problemas"].empty:
        L.append("### Categorias de problema apontadas\n")
        L.append(_md(r["problemas"]))
        L.append("")
        L.append(_md(r["taxa_problemas"]))
        L.append("")
    if not r["custos"].empty:
        L.append("### Custo e desempenho\n")
        L.append(_md(r["custos"], indice=False))
        L.append("")

    L.append("---\n\n## Figuras\n")
    for arquivo in r["figuras"]:
        L.append(f"- `{arquivo}`")
    L.append("")
    L.append(_bloco_reprodutibilidade(r.get("bayes") or None))

    texto = "\n".join(L)
    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write(texto)
    return texto


#: plurais dos rótulos de entidade usados nos relatórios
_PLURAIS = {"juiz": "juízes", "avaliador": "avaliadores", "fonte": "fontes",
            "protocolo": "protocolos", "grupo": "grupos"}


def _plural(rotulo: str) -> str:
    """Plural do rótulo, sem inventar regra morfológica genérica."""
    return _PLURAIS.get(rotulo, f"{rotulo}s")


def _bloco_como_ler_bayes(cfg, pares_info: list, rotulo_entidade: str = "fonte",
                          referencia: str = None) -> list:
    """Subseção 'Como ler esta análise' — inserida antes das tabelas bayesianas.

    Explica a ROPE e a leitura do heatmap, e fecha com dois exemplos gerados a
    partir dos extremos: o par mais próximo da equivalência e o mais distante.
    Dois, e não um, porque cada extremo exercita um caminho de leitura que o
    outro não cobre.
    """
    plural = _plural(rotulo_entidade)
    L = ["### Como ler esta análise\n"]

    L.append(
        f"A comparação é **pareada por documento** e roda no `baycomp`: para cada par, "
        f"conta-se em quantos documentos o primeiro {rotulo_entidade} recebeu nota maior, "
        f"igual ou menor, e daí sai a probabilidade posterior de cada relação.\n")

    # A ROPE aqui não é um parâmetro livre: decorre da escala ser inteira.
    L.append(
        f"**ROPE = {_num(cfg.rope, 2)}.** A escala Likert é inteira, então uma diferença "
        f"de até meio ponto significa exatamente **notas iguais**. Não é uma margem "
        f"arbitrada nem calibrada: é a tradução direta da escala. Por isso não há análise "
        f"de sensibilidade a ela nesta etapa — mudá-la deixaria de representar a escala.\n")

    L.append(
        f"**Método:** `{cfg.metodo}`. Na escala ordinal, a distância entre as notas 2 e 3 "
        f"não é comparável à distância entre 3 e 4; por isso o teste conta **direções** e "
        f"descarta magnitude.\n")

    L.append(
        f"**Como ler o heatmap:** cada célula mostra a relação da **linha** em relação à "
        f"**coluna** — verde = superior, vermelho = inferior, azul = equivalente, cinza = "
        f"incerto. O número é a **probabilidade posterior** da categoria colorida, e a "
        f"intensidade da cor a acompanha. Cinza significa que nenhuma das três alcançou o "
        f"limiar de {_num(cfg.limiar, 2)} — desfecho legítimo, não ausência de resultado.\n")

    favoravel, desafiador = _pares_didaticos(pares_info)
    for rotulo, exemplo in (("mais favorável", favoravel),
                            ("mais desafiador", desafiador)):
        if exemplo is not None:
            L.append(_texto_exemplo(exemplo, cfg, plural, referencia, rotulo))
    return L


def _pares_didaticos(pares_info: list) -> tuple:
    """Os dois extremos: (maior P(equivalência), menor P(equivalência)).

    Devolve ``(par, None)`` quando há um só — repetir o mesmo par sob dois
    rótulos confundiria em vez de esclarecer.
    """
    validos = [p for p in pares_info or []
               if p.get("p_equiv") == p.get("p_equiv")]      # descarta NaN
    if not validos:
        return None, None
    favoravel = max(validos, key=lambda p: p["p_equiv"])
    desafiador = min(validos, key=lambda p: p["p_equiv"])
    return (favoravel, None) if favoravel is desafiador else (favoravel, desafiador)


def _texto_exemplo(exemplo: dict, cfg, plural: str, referencia: str,
                   rotulo: str) -> str:
    """Um parágrafo de leitura guiada para um par concreto, com números da execução."""
    nome_a = exemplo.get("juiz") or exemplo.get("nome_a", "A")
    nome_b = referencia or exemplo.get("nome_b", "B")
    p_eq = exemplo.get("p_equiv", float("nan"))
    acima = exemplo.get("acima", "?")
    empate = exemplo.get("empate", "?")
    abaixo = exemplo.get("abaixo", "?")
    n_obs = exemplo.get("n", "?")

    atinge = p_eq >= cfg.limiar
    posicao = f"{'acima' if atinge else 'abaixo'} do limiar de {_num(cfg.limiar, 2)}"
    veredito = (
        "a **equivalência prática está estabelecida**: a diferença quase certamente cabe "
        "dentro da ROPE." if atinge else
        f"a equivalência **não é estabelecida** — a evidência não basta para afirmar que "
        f"os dois {plural} são praticamente indistinguíveis.")

    return (
        f"**Exemplo de leitura — cenário {rotulo}:** `{nome_a}` em relação a `{nome_b}`. "
        f"Em {n_obs} itens pareados, `{nome_a}` ficou acima em {acima}, empatou em {empate} "
        f"e ficou abaixo em {abaixo}. A probabilidade posterior de as notas serem "
        f"praticamente equivalentes é **{_num(p_eq)}** ({posicao}); {veredito}\n")


def _pares_info_de_tabela(tabela) -> list:
    """Converte a tabela do grupo no formato consumido por `_bloco_como_ler_bayes`."""
    info = []
    for _, row in tabela.iterrows():
        rotulo = next((c for c in tabela.columns if c.endswith("(A × B)")), None)
        nomes = str(row[rotulo]).split(" × ") if rotulo else ["A", "B"]
        info.append({
            "nome_a": nomes[0], "nome_b": nomes[-1],
            "n": int(row.get("n", 0)),
            "acima": int(row.get("A melhor", 0)),
            "empate": int(row.get("Empate", 0)),
            "abaixo": int(row.get("B melhor", 0)),
            # sem a coluna, NaN: `_pares_didaticos` descarta em vez de eleger
            "p_equiv": float(row.get("P(equiv.)", float("nan"))),
        })
    return info


def _bloco_bayes_grupo(r: dict) -> list:
    """Seção bayesiana do `estatisticas.md` — vazia quando a etapa não rodou."""
    b = r.get("bayes")
    if not b:
        return []
    cfg = b["config"]
    L = ["---\n\n## Comparação bayesiana entre as fontes\n"]
    L.append("Complementa — não substitui — o Friedman e os contrastes de Wilcoxon acima. "
             "A pergunta muda: em vez de *há diferença detectável?*, responde *qual a "
             "probabilidade de esta fonte superar aquela, e qual a de serem praticamente "
             "equivalentes?*. Equivalência aqui é **achado**, não falha em rejeitar H₀ — que "
             "é justamente o que o teste de hipótese nula não consegue afirmar.\n")
    L.append("| Parâmetro | Valor |")
    L.append("|---|---|")
    L.append(f"| Biblioteca | `baycomp` |")
    L.append(f"| Teste | `{cfg.metodo}` ({_NOME_TESTE.get(cfg.metodo, cfg.metodo)}) |")
    L.append(f"| Variável | nota Likert mediana por documento |")
    L.append(f"| ROPE | {_num(cfg.rope, 2)} (notas iguais, escala inteira) |")
    L.append(f"| Limiar de classificação | {_num(cfg.limiar, 2)} |")
    L.append(f"| Amostras / semente | {_num(cfg.amostras, 0)} / {cfg.semente} |")
    L.append("")
    L.extend(_bloco_como_ler_bayes(cfg, _pares_info_de_tabela(b["tabela"]),
                                   rotulo_entidade="fonte"))
    L.append("### Relações par a par\n")
    L.append(_md(b["tabela"], indice=False))
    L.append("")
    L.append("### Síntese por fonte\n")
    L.append(_md(b["resumo"]))
    L.append("")
    L.append("A síntese conta relações, **não ordena as fontes**: relações podem ser "
             "intransitivas, e transformar contagem de vitórias em ranking criaria uma ordem "
             "que os dados não sustentam.\n")
    return L


def _sintese_grupo(r: dict) -> list:
    """Sumário executivo da análise interna."""
    rot = r["grupo"].rotulos
    itens = []
    interna = r["interna"]
    if interna:
        itens.append(f"**{rot['concordancia']}:** "
                     f"{'aprovada' if interna['aprovado'] else 'reprovada'} — κw = "
                     f"{_num(interna['kappa'])} "
                     f"[{_num(interna['ic_inf'])}; {_num(interna['ic_sup'])}], "
                     f"P_o = {_num(interna['P_o'])} (critério ≥ {_num(LIMIAR_KAPPA, 2)}).")
    friedman = r["comparacao"].get("friedman")
    if friedman and friedman["chi2"] == friedman["chi2"]:
        itens.append(f"**Entre fontes:** H₀ "
                     f"{'rejeitada' if friedman['rejeita_h0'] else 'não rejeitada'} "
                     f"(χ²_F = {_num(friedman['chi2'], 2)}, p = {_p(friedman['p'])}); "
                     f"melhor rank médio: `{r['comparacao']['melhor']}`.")
    viaveis = r["descritivas"][r["descritivas"]["IC95 inf"] >= LIMIAR_VIABILIDADE]
    itens.append(f"**Viabilidade:** {', '.join(viaveis.index) if len(viaveis) else 'nenhuma fonte'} "
                 f"atinge o critério (LI do IC 95% ≥ {_num(LIMIAR_VIABILIDADE, 2)}).")
    itens.append(f"**Dados:** {r['n_valido']} documentos pareados "
                 f"({r['n_descartado']} descartados de {r['n_bruto']}).")
    return itens


# =============================================================================
# 9. Relatório — validação dos juízes
# =============================================================================

def escrever_relatorio_validacao(v: dict, caminho: str) -> str:
    """Escreve o `validacao.md`: veredito, evidência e nota lateral."""
    referencia = v["referencia"]
    humana = v["referencia_humana"]
    L = []

    titulo = ("Validação do juiz LLM por concordância com a avaliação humana" if humana
              else "Concordância entre juízes LLM (ensaio metodológico)")
    L.append(f"# {titulo}\n")
    L.append(f"Gerado em {datetime.now():%d/%m/%Y %H:%M} por `realizar_avaliacoes.py`.\n")
    if not humana:
        L.append("> ⚠️ **O grupo de referência é do tipo LLM, não humano.** Este relatório é "
                 "um ensaio metodológico do pipeline: mede convergência entre juízes LLM, o "
                 "que **não constitui validação** — dois modelos podem convergir por "
                 "compartilharem os mesmos vieses. A validação exige grupo de referência com "
                 "especialistas humanos.\n")
    if len(v["grupos_humanos"]) > 1:
        L.append(f"> ⚠️ **{len(v['grupos_humanos'])} grupos do tipo humano** "
                 f"({', '.join(v['grupos_humanos'])}); `{referencia}` foi usado como "
                 "referência. Se os especialistas foram exportados como grupos separados, "
                 "consolide-os num único grupo (um avaliador por pasta).\n")

    rotulo_ok = "VALIDADO" if humana else "CONVERGENTE"
    rotulo_ressalva = f"{rotulo_ok} COM RESSALVA"
    rotulo_nao = f"NÃO {rotulo_ok}"

    def _rotular(status):
        return {"VALIDADO": rotulo_ok, "VALIDADO COM RESSALVA": rotulo_ressalva,
                "NÃO VALIDADO": rotulo_nao}[status]

    # --- veredito ------------------------------------------------------------
    L.append("## Veredito\n")
    gates_ordenados = sorted(v["gates"], key=lambda g: -(g["kappa"] if g["kappa"] == g["kappa"] else -9))
    veredito = pd.DataFrame([{
        "Juiz": g["juiz"],
        "κw": round(g["kappa"], 4),
        "IC 95%": f"[{_num(g['ic_inf'], 3)}; {_num(g['ic_sup'], 3)}]",
        "Wilcoxon p": _p(g["p_wilcoxon"]),
        "Dif. média": round(g["media_dif"], 3),
        "McNemar p": _p(g["p_mcnemar"]),
        "Acurácia": _pct(g["acuracia"]),
        "Critérios": f"{sum(g['criterios'].values())} de 3",
        "Resultado": _rotular(g["status"]),
    } for g in gates_ordenados])
    L.append(_md(veredito, indice=False))
    L.append("")
    for gate in gates_ordenados:
        L.append(_paragrafo_veredito(gate, referencia, _rotular(gate["status"])))
    L.append("")

    # --- critérios -----------------------------------------------------------
    L.append("---\n\n## Critérios de validação\n")
    L.append(f"Todos aferidos no **agregado** — as {len(v['fontes'])} fontes empilhadas numa "
             f"única série de {len(v['documentos']) * len(v['fontes'])} itens pareados, sem "
             "estratificação. O juiz precisa passar nos três.\n")
    L.append("| # | Critério | Estatística | Aprova se |")
    L.append("|---|---|---|---|")
    L.append(f"| 1 | Concordância ordinal com o humano | κw de Cohen ponderado (quadrático) | "
             f"κw ≥ {_num(LIMIAR_KAPPA, 2)} |")
    L.append(f"| 2 | Ausência de viés sistemático | Wilcoxon bilateral pareado | "
             f"p > {_num(LIMIAR_ALFA, 2)} |")
    L.append(f"| 3 | Equivalência na decisão prática | McNemar sobre nota ≥ {PISO_ADEQUACAO} | "
             f"p > {_num(LIMIAR_ALFA, 2)} |")
    L.append("")
    L.append(f"O critério 1 é aferido no **valor pontual** do κw. O IC 95% (bootstrap de "
             f"{BOOTSTRAP_REPLICAS} réplicas de documentos) é reportado como medida de "
             "precisão, não como critério: exigir que o limite inferior alcançasse o limiar "
             "reprovaria o juiz por tamanho de amostra, não por falta de concordância.\n")
    L.append("Nos critérios 2 e 3, **não rejeitar H₀ é o resultado desejado**. Isso indica "
             "ausência de viés detectável, não equivalência demonstrada.\n")
    L.append(f"### {rotulo_ressalva.capitalize()}\n")
    L.append(f"Status intermediário previsto no plano contingencial do método: critérios 1 e "
             "3 atendidos e viés estatisticamente significativo (critério 2), porém com "
             "magnitude média abaixo da **margem de relevância prática** — 0,5 desvio-padrão "
             "das notas da referência (regra da meia-DP; Norman, Sloan & Wyrwich, 2003). "
             f"Nesta execução: DP da referência = {_num(v['dp_referencia'], 4)}, margem = "
             f"**{_num(v['margem'], 4)} ponto**. Com n grande, diferenças praticamente "
             "irrelevantes tornam-se significativas; a margem separa significância "
             "estatística de relevância prática. É classificação descritiva, não teste de "
             "equivalência (que exigiria margem pré-registrada e procedimento TOST).\n")

    # --- cobertura -----------------------------------------------------------
    L.append("---\n\n## Cobertura e pareamento\n")
    L.append("Só entram documentos **e** fontes presentes em todos os grupos. O que fica de "
             "fora permanece na análise interna do respectivo grupo.\n")
    L.append("| Parâmetro | Valor |")
    L.append("|---|---|")
    L.append(f"| Grupos | {', '.join(f'{n} ({g.tipo})' for n, g in v['grupos'].items())} |")
    L.append(f"| Referência | `{referencia}` |")
    L.append(f"| Documentos na interseção | {_num(len(v['documentos']), 0)} |")
    L.append(f"| Fontes na interseção | {len(v['fontes'])} — {', '.join(v['fontes'])} |")
    L.append(f"| Itens pareados | {_num(len(v['documentos']) * len(v['fontes']), 0)} |")
    L.append("")
    L.append(_md(v["cobertura"]))
    L.append("")
    cobertura = v["cobertura"]
    total_docs = int(cobertura["documentos"].max())
    if total_docs:
        gargalos = cobertura[cobertura["docs fora da interseção"] / total_docs > 0.10]
        for nome, linha in gargalos.iterrows():
            perda = 100 * linha["docs fora da interseção"] / total_docs
            L.append(f"⚠️ O grupo `{nome}` reduziu a interseção em "
                     f"{int(linha['docs fora da interseção'])} documentos "
                     f"({_num(perda, 1)}% do maior grupo). Verifique se a perda decorre de "
                     "falhas de avaliação ou de cobertura incompleta do Gold Set.\n")

    # --- evidência -----------------------------------------------------------
    L.append("---\n\n## Evidência detalhada\n")
    L.append("### Concordância ordinal e viés\n")
    L.append(_md(_tabela_concordancia(gates_ordenados), indice=False))
    L.append("")
    L.append("`P_o` é a concordância observada ponderada (dá crédito parcial a diferenças de "
             "1 ponto; a coluna `Exata` é a leitura sem ponderação). `n′` é o número de pares "
             "com diferença não nula — os empates são descartados pelo Wilcoxon, de modo que "
             "**`r = |z|/√n′` mede a consistência de direção entre as discordâncias, não a "
             "magnitude do desacordo**: um `r` grande com `Média dif.` pequena indica viés "
             "consistente porém de pequena magnitude. A magnitude é lida em `Média dif.`, "
             "confrontada com a margem de relevância prática.\n")
    L.append(f"### Decisão binária (adequado = nota ≥ {PISO_ADEQUACAO})\n")
    L.append(_md(_tabela_decisao(gates_ordenados), indice=False))
    L.append("")
    L.append(f"`{referencia}` é o padrão de referência. **FP**: itens aprovados pelo juiz e "
             "reprovados pela referência (o erro que importa na triagem — material inadequado "
             "passando); **FN**: o sentido oposto. Sensibilidade = proporção dos adequados da "
             "referência que o juiz também aprova; especificidade = proporção dos inadequados "
             "que o juiz também reprova.\n")

    # --- diferença entre avaliadores -----------------------------------------
    entre = v["entre_grupos"]
    L.append("---\n\n## Diferença de severidade entre os avaliadores\n")
    L.append("Friedman com os grupos como tratamentos e os itens (documento × fonte) como "
             "blocos, seguido de contrastes Wilcoxon com correção de Holm. Responde se algum "
             "avaliador é sistematicamente mais severo e, no post-hoc, quais diferem entre si. "
             "Rank 1 = notas mais altas (avaliador mais leniente).\n")
    L.append(_md(entre["ranks"]))
    L.append("")
    friedman = entre["friedman"]
    if friedman:
        L.append(f"**Friedman:** χ²_F = {_num(friedman['chi2'], 3)}, gl = {friedman['gl']}, "
                 f"p = {_p(friedman['p'])}, n = {_num(entre['n'], 0)} itens — "
                 f"{'há' if friedman['rejeita_h0'] else 'não há'} evidência de diferença "
                 "sistemática entre os avaliadores.\n")
    else:
        L.append("_Omnibus de Friedman não computado: exige 3 ou mais grupos._\n")

    contrastes = entre["contrastes"]
    if not contrastes.empty:
        L.append("### Contrastes post-hoc (Wilcoxon + Holm)\n")
        tabela = pd.DataFrame({
            "Contraste": contrastes["contraste"],
            "Média dif.": contrastes["media_dif"].round(4),
            "Mediana dif.": contrastes["mediana_dif"],
            "n′": contrastes["n_efetivo"], "z (abs.)": contrastes["z"].abs().round(3),
            "p bruto": contrastes["p"].map(_p), "p Holm": contrastes["p_holm"].map(_p),
            "r": contrastes["r"].round(3), "Efeito": contrastes["efeito"],
            "Signif.": contrastes["significativo"].map({True: "sim", False: "não"}),
        })
        L.append(_md(tabela, indice=False))
        L.append("")
        L.append("O **critério 2 do gate usa o `p bruto`**, não o corrigido. A correção de Holm "
                 "protege contra falsos positivos; como aqui rejeitar H₀ é o resultado "
                 "desfavorável ao juiz, adotar o valor corrigido tornaria o gate mais permissivo. "
                 "O `p Holm` consta para leitura conjunta dos contrastes.\n")

    # --- leitura bayesiana ---------------------------------------------------
    L.extend(_bloco_bayes_validacao(v))

    # --- nota lateral --------------------------------------------------------
    if v["laterais"]:
        L.append("---\n\n## Nota lateral: concordância entre os juízes LLM\n")
        L.append("Não faz parte do gate — a validação é contra o humano. Convergência entre "
                 "juízes LLM não é evidência de validade: modelos podem convergir por "
                 "compartilharem os mesmos vieses de pré-treinamento.\n")
        L.append(_md(_tabela_concordancia(v["laterais"], lateral=True), indice=False))
        L.append("")
        L.append(_md(_tabela_decisao(v["laterais"], lateral=True), indice=False))
        L.append("")

    # --- ranks ---------------------------------------------------------------
    L.append("---\n\n## Ordenação das fontes (descritivo)\n")
    L.append("Rank médio de cada fonte segundo cada grupo (1 = melhor). Descritivo, sem teste: "
             "com poucas fontes na interseção, qualquer correlação entre ordenações teria n "
             "insuficiente. Mostra se o juiz ordena os protocolos como o humano ordenaria.\n")
    L.append(_md(v["ranks"]))
    L.append("")

    # --- limitações ----------------------------------------------------------
    L.append("---\n\n## Limitações\n")
    if not humana:
        L.append("- **A referência é um juiz LLM.** Nada aqui constitui validação: "
                 "convergência entre modelos pode refletir vieses compartilhados de "
                 "pré-treinamento, não correção do julgamento.\n")
    L.append(f"- A validação é **global**, para o conjunto das {len(v['fontes'])} fontes. Não "
             "sustenta afirmar que o juiz é igualmente confiável em cada fonte isoladamente.\n"
             f"- Com {len(v['documentos'])} documentos na interseção, o κw pontual é frágil; o "
             "IC 95% é a informação relevante sobre precisão.\n"
             "- Não rejeitar H₀ nos critérios 2 e 3 indica ausência de viés detectável, não "
             "equivalência demonstrada — testes de equivalência formal exigiriam definir uma "
             "margem e um n maior.\n"
             "- No agregado, cada documento entra tantas vezes quantas forem as fontes. É "
             "coerente com a unidade de julgamento (a extração avaliada, não o documento), e o "
             "bootstrap reamostra documentos inteiros para preservar essa dependência.\n")
    if v.get("bayes"):
        cfg = v["bayes"]["config"]
        L.append(f"- A conclusão bayesiana depende do limiar de "
                 f"{_num(cfg.limiar_veredito, 2)}, fixado antes da análise. A ROPE de "
                 f"{_num(cfg.rope, 2)} não é parâmetro livre: decorre de a escala Likert ser "
                 "inteira, e alterá-la deixaria de representar 'notas iguais'.\n"
                 "- O teste de sinais descarta a **magnitude** de cada divergência e conta só "
                 "direções. Na escala ordinal isso é adequado — a distância entre as notas 2 e "
                 "3 não é comparável à distância entre 3 e 4 —, mas significa que a análise não "
                 "distingue divergências de um ponto das de dois.\n"
                 "- A análise é **independente por conjunto de dados**: não modela a "
                 "variabilidade compartilhada entre datasets nem atualiza sequencialmente o "
                 "conhecimento. O `baycomp.HierarchicalTest` faria isso, mas exige `pystan`, "
                 "pressupõe validação cruzada dentro de cada conjunto e, com poucos conjuntos, "
                 "a variância entre grupos não é identificável.\n")

    L.append("---\n\n## Figuras\n")
    for arquivo in v["figuras"]:
        L.append(f"- `{arquivo}`")
    L.append("")
    L.append(_bloco_reprodutibilidade(v.get("bayes") or None))

    texto = "\n".join(L)
    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write(texto)
    return texto


def _bloco_bayes_validacao(v: dict) -> list:
    """Seção bayesiana do `validacao.md` — vazia quando a etapa não rodou."""
    b = v.get("bayes")
    if not b:
        return []
    cfg = b["config"]
    referencia = v["referencia"]
    L = ["---\n\n## Leitura bayesiana (complementar ao gate)\n"]
    L.append(f"O gate acima continua sendo o que decide: κw, Wilcoxon e McNemar. Esta seção "
             f"acrescenta o que aqueles testes não conseguem afirmar — a probabilidade "
             f"posterior de **equivalência prática** entre o juiz e `{referencia}`. Não "
             "rejeitar H₀ não prova equivalência; uma posterior concentrada dentro da ROPE, "
             "sim, é evidência a favor dela.\n")
    L.append("| Parâmetro | Valor |")
    L.append("|---|---|")
    L.append("| Biblioteca | `baycomp` |")
    L.append(f"| Teste | `{cfg.metodo}` ({_NOME_TESTE.get(cfg.metodo, cfg.metodo)}) |")
    L.append(f"| ROPE | {_num(cfg.rope, 2)} (notas iguais, escala inteira) |")
    L.append(f"| Limiar do veredito | {_num(cfg.limiar_veredito, 2)} |")
    L.append(f"| Limiar de classificação do heatmap | {_num(cfg.limiar, 2)} |")
    L.append(f"| Amostras / semente | {_num(cfg.amostras, 0)} / {cfg.semente} |")
    L.append("")
    L.extend(_bloco_como_ler_bayes(cfg, b.get("pares", []),
                                   rotulo_entidade="avaliador", referencia=referencia))
    L.append(f"### Juiz × `{referencia}`\n")
    L.append(_md(b["tabela"], indice=False))
    L.append("")
    L.append("`P(equiv. notas)` usa as notas medianas; `P(equiv. decisão)` usa a binarização "
             f"em nota ≥ {PISO_ADEQUACAO}, preservando o pareamento — é o análogo bayesiano "
             "do McNemar. A coluna `Leitura` combina as duas:\n")
    L.append(f"- **SEM VIÉS RELEVANTE** — as duas probabilidades de equivalência atingem "
             f"{_num(cfg.limiar_veredito, 2)};")
    L.append("- **VIÉS RELEVANTE** — é quase certo que a divergência **excede** a ROPE;")
    L.append("- **INCONCLUSIVO** — os dados não sustentam nenhuma das duas leituras. É "
             "desfecho legítimo, e a resposta honesta é ampliar a amostra.\n")
    L.append("O veredito julga **magnitude**, nunca direção. `P(juiz > ref.)` próximo de 1 com "
             "`P(equiv. notas)` também alto não é contradição: significa que o juiz é "
             "confiavelmente mais leniente, mas por uma margem sem relevância prática — "
             "situação que o teste de hipótese nula não consegue expressar.\n")
    if b.get("tabela_matriz") is not None:
        L.append("### Relações entre todos os avaliadores\n")
        L.append(_md(b["tabela_matriz"], indice=False))
        L.append("")
        L.append(_md(b["resumo"]))
        L.append("")
        L.append("Inclui os pares juiz × juiz, que permanecem **fora do gate**: convergência "
                 "entre modelos não é evidência de validade.\n")
        L.append(f"⚠️ Esta matriz classifica ao limiar {_num(cfg.limiar, 2)}, enquanto o "
                 f"veredito acima exige {_num(cfg.limiar_veredito, 2)}. Um mesmo par pode "
                 "aparecer como `equivalente` aqui e `INCONCLUSIVO` na tabela do juiz — não é "
                 "inconsistência: a matriz descreve o panorama, o veredito decide.\n")
    return L


def _paragrafo_veredito(g: dict, referencia: str, rotulo: str) -> str:
    """Frase de veredito de um juiz, com os critérios que falharam, se houver."""
    criterios = g["criterios"]
    if g["status"] == "VALIDADO":
        return (f"✅ **`{g['juiz']}`: {rotulo}** — κw = {_num(g['kappa'])} "
                f"(concordância no nível *{g['interpretacao']}*), sem viés sistemático "
                f"detectável (Wilcoxon p = {_p(g['p_wilcoxon'])}) e sem viés direcional na "
                f"decisão adequado/inadequado (McNemar p = {_p(g['p_mcnemar'])}), com "
                f"acurácia de {_pct(g['acuracia'])} contra `{referencia}`.")

    if g["status"] == "VALIDADO COM RESSALVA":
        return (f"🟡 **`{g['juiz']}`: {rotulo}** — κw = {_num(g['kappa'])} "
                f"(*{g['interpretacao']}*) e decisão binária sem viés direcional (McNemar "
                f"p = {_p(g['p_mcnemar'])}). Há viés sistemático estatisticamente "
                f"significativo (Wilcoxon p = {_p(g['p_wilcoxon'])}, {g['direcao']}), porém "
                f"de magnitude média {_num(abs(g['media_dif']), 3)} ponto — abaixo da margem "
                f"de relevância prática de {_num(g['margem'], 3)} ponto (meia-DP da "
                "referência). O uso em massa deve ser reportado com esta ressalva.")

    falhas = []
    if not criterios["concordancia"]:
        falhas.append(f"κw = {_num(g['kappa'])} < {_num(LIMIAR_KAPPA, 2)} "
                      f"(concordância no nível *{g['interpretacao']}*, "
                      f"P_o = {_num(g['p_o'])})")
    if not criterios["sem_vies"]:
        detalhe = f"Wilcoxon p = {_p(g['p_wilcoxon'])}, dif. média = "                   f"{_num(g['media_dif'], 3)}, {g['direcao']}"
        if g["margem"] == g["margem"] and not g["vies_toleravel"]:
            detalhe += f"; acima da margem de {_num(g['margem'], 3)} ponto"
        falhas.append(f"viés sistemático ({detalhe})")
    if not criterios["decisao"]:
        falhas.append(f"viés direcional na decisão binária (McNemar p = "
                      f"{_p(g['p_mcnemar'])}: {g['b01']} itens aprovados pelo juiz e "
                      f"reprovados pela referência, contra {g['b10']} no sentido oposto)")
    return (f"❌ **`{g['juiz']}`: {rotulo}.** Critério(s) não atendido(s): "
            f"{'; '.join(falhas)}.")


def _tabela_concordancia(gates: list, lateral: bool = False) -> pd.DataFrame:
    """Bloco de concordância ordinal e viés: uma linha por par."""
    rotulo = "Par" if lateral else "Juiz"
    return pd.DataFrame([{
        rotulo: f"{g['juiz']} × {g['referencia']}" if lateral else g["juiz"],
        "n": g["n"],
        "κw": round(g["kappa"], 4),
        "IC95 inf": round(g["ic_inf"], 4), "IC95 sup": round(g["ic_sup"], 4),
        "P_o": round(g["p_o"], 4),
        "Exata": _pct(g["exata"]), "Dif. ≤ 1": _pct(g["amplitude_1"]),
        "Média dif.": round(g["media_dif"], 4),
        "n′": g["n_efetivo"], "Wilcoxon p": _p(g["p_wilcoxon"]),
        "r": round(g["r"], 3), "Direção": g["direcao"],
    } for g in gates])


def _tabela_decisao(gates: list, lateral: bool = False) -> pd.DataFrame:
    """Bloco da decisão binária (nota ≥ piso): uma linha por par."""
    rotulo = "Par" if lateral else "Juiz"
    linhas = []
    for g in gates:
        registro = {
            rotulo: f"{g['juiz']} × {g['referencia']}" if lateral else g["juiz"],
            "κ binário": round(g["kappa_bin"], 4),
            "FP (juiz aprova)": g["b01"], "FN (juiz reprova)": g["b10"],
            "McNemar p": _p(g["p_mcnemar"]), "Método": g["metodo_mcnemar"],
            "Acurácia": _pct(g["acuracia"]),
            "Sensib.": _pct(g["sensibilidade"]),
            "Especif.": _pct(g["especificidade"]),
        }
        if not lateral:
            registro["Status"] = g["status"]
        linhas.append(registro)
    return pd.DataFrame(linhas)


def _bloco_reprodutibilidade(bayes_resultado: dict = None) -> str:
    """Rodapé com versões e convenções usadas."""
    import sklearn
    import statsmodels
    extra = []
    if bayes_resultado:
        cfg = bayes_resultado["config"]
        extra = [f"| Posterior bayesiana | baycomp `{cfg.metodo}`, "
                 f"ROPE {_num(cfg.rope, 2)}, {_num(cfg.amostras, 0)} amostras, "
                 f"semente {cfg.semente} |"]
    return "\n".join([
        "---\n", "## Reprodutibilidade\n",
        "| Item | Valor |", "|---|---|",
        f"| Python | {sys.version.split()[0]} |",
        f"| pandas / numpy / scipy | {pd.__version__} / {np.__version__} "
        f"/ {__import__('scipy').__version__} |",
        f"| scikit-learn / statsmodels | {sklearn.__version__} / "
        f"{statsmodels.__version__} |",
        "| Pesos do Kappa | quadráticos |",
        "| Faixas de interpretação | McHugh (2012) |",
        "| Correção múltipla | Holm-Bonferroni |",
        "| Margem de relevância prática | 0,5 DP da referência (meia-DP) |",
        f"| Bootstrap | {BOOTSTRAP_REPLICAS} réplicas de documentos, semente {SEMENTE} |",
    ] + extra + [
        "",
        "### Origem das estatísticas\n",
        "| Estatística | Implementação |",
        "|---|---|",
    ] + [f"| {nome} | {origem} |"
         for nome, origem in dependencias(bool(bayes_resultado)).items()] + [
        "",
        "As definições operacionais (fórmulas internas) constam em "
        "`realizar_avaliacoes_teste.py`, que verifica que coincidem com os pacotes.",
        "",
    ])


# =============================================================================
# 10. Orquestração e CLI
# =============================================================================

def parse_grupos(especificacoes: Sequence[str], base: str) -> list:
    """Converte ``nome:tipo`` em objetos ``Grupo``, validando tipo e existência.

    Raises:
        ValueError: se algum grupo vier sem tipo ou com tipo desconhecido.
    """
    sem_tipo = [e for e in especificacoes if ":" not in e]
    if sem_tipo:
        raise ValueError(
            f"Tipo não informado para: {', '.join(sem_tipo)}.\n"
            f"Use nome:tipo, com tipo em {TIPOS_VALIDOS}. Ex.: gpt5:llm humanos:humano")

    grupos = []
    for especificacao in especificacoes:
        nome, _, tipo = especificacao.partition(":")
        tipo = tipo.strip().lower()
        if tipo not in TIPOS_VALIDOS:
            raise ValueError(f"Tipo inválido em '{especificacao}': "
                             f"'{tipo}' não está em {TIPOS_VALIDOS}.")
        grupos.append(Grupo(nome=nome, tipo=tipo, pastas=expandir_grupo(base, nome)))

    nomes = [g.nome for g in grupos]
    if len(set(nomes)) != len(nomes):
        raise ValueError(f"Nomes de grupo repetidos: {nomes}")
    return grupos


def parse_aliases(especificacoes: Sequence[str]) -> dict:
    """Converte ``origem=destino`` em dicionário de normalização de nomes de fonte."""
    aliases = {}
    for especificacao in especificacoes or []:
        origem, _, destino = especificacao.partition("=")
        if not origem or not destino:
            raise ValueError(f"Alias inválido: '{especificacao}'. Use origem=destino.")
        aliases[origem.strip()] = destino.strip()
    return aliases


def montar_config_bayes(args, erro: Callable[[str], None]) -> ConfigBayes:
    """Converte as flags `--bayes*` em ``ConfigBayes``, explicando o que faltou.

    Sem ``--bayes`` a etapa não existe. Ajustes informados isoladamente são
    erro, e não silêncio — quem escreveu `--bayes-metodo t` esperava a etapa
    rodar, e deixá-la desligada devolveria um relatório sem a seção pedida.
    """
    ajustes = {"--bayes-rope": args.bayes_rope != BAYES_ROPE_LIKERT,
               "--bayes-metodo": args.bayes_metodo != BAYES_METODO_PADRAO,
               "--bayes-limiar": args.bayes_limiar != BAYES_LIMIAR_PADRAO,
               "--bayes-limiar-veredito": args.bayes_limiar_veredito != BAYES_VEREDITO_PADRAO,
               "--bayes-amostras": args.bayes_amostras != BAYES_AMOSTRAS_PADRAO,
               "--bayes-semente": args.bayes_semente != SEMENTE}
    informados = [flag for flag, mudou in ajustes.items() if mudou]

    if not args.bayes:
        if informados:
            erro(f"{', '.join(informados)} só tem efeito com --bayes. "
                 "Acrescente --bayes para gerar a comparação bayesiana, ou remova "
                 "os ajustes para rodar apenas a análise frequentista.")
        return ConfigBayes(ativo=False)

    if not BAYES_DISPONIVEL:
        erro("--bayes exige o módulo `util_est_bayesiana.py` (e o pacote `baycomp`) "
             "na mesma pasta ou no PYTHONPATH. Sem --bayes o pipeline roda "
             "normalmente, apenas sem a seção bayesiana.")

    # o baycomp devolve só (p_esquerda, p_direita) com rope = 0: a tripla exige rope > 0
    if args.bayes_rope <= 0:
        erro(f"--bayes-rope = {args.bayes_rope} inválido: com ROPE zero o baycomp "
             "não devolve a probabilidade de equivalência. Na escala Likert inteira "
             f"o valor correto é {BAYES_ROPE_LIKERT} ('notas iguais').")
    for flag, valor in (("--bayes-limiar", args.bayes_limiar),
                        ("--bayes-limiar-veredito", args.bayes_limiar_veredito)):
        if not 0.5 <= valor <= 1.0:
            erro(f"{flag} = {valor} fora da faixa admissível [0,5; 1,0].")
    if args.bayes_amostras < 1000:
        erro(f"--bayes-amostras = {args.bayes_amostras} é baixo demais: a variação "
             f"entre sementes dominaria o resultado (padrão: {BAYES_AMOSTRAS_PADRAO}).")
    if args.bayes_metodo == "t":
        print("ℹ `--bayes-metodo t` usa baycomp.CorrelatedTTest, que é analítico: "
              "amostras e semente são ignoradas. Ele foi pensado para escores "
              "contínuos; nesta etapa a escala é ordinal.")

    return ConfigBayes(
        ativo=True, rope=args.bayes_rope, metodo=args.bayes_metodo,
        limiar=args.bayes_limiar, limiar_veredito=args.bayes_limiar_veredito,
        amostras=args.bayes_amostras, semente=args.bayes_semente)


def executar(base: str, grupos: list, saida: str, escala: tuple = ESCALA_PADRAO,
             aliases: dict = None, referencia: str = None,
             leitor: Callable[[str], pd.DataFrame] = pd.read_parquet,
             config_bayes: ConfigBayes = None) -> dict:
    """Executa a análise interna de cada grupo e, com 2+ grupos, a validação."""
    os.makedirs(saida, exist_ok=True)
    resultados = {}
    for grupo in grupos:
        print(f"\n{'=' * 70}")
        print(f"Grupo '{grupo.nome}' ({grupo.tipo}): {len(grupo.pastas)} pastas "
              f"({', '.join(grupo.pastas)})")
        print("=" * 70)
        destino = os.path.join(saida, grupo.nome) if len(grupos) > 1 else saida
        resultados[grupo.nome] = analisar_grupo(
            base, grupo, destino, escala, aliases=aliases, leitor=leitor,
            config_bayes=config_bayes)

    if len(grupos) < 2:
        print(f"\n✅ Concluído (1 grupo, sem validação). "
              f"Resultados em: {os.path.abspath(saida)}")
        return {"grupos": resultados}

    print(f"\n{'=' * 70}")
    print("Validação por concordância")
    print("=" * 70)
    validacao = validar_juizes(resultados, saida, escala, referencia=referencia,
                               config_bayes=config_bayes)
    print(f"\n✅ Concluído. Resultados em: {os.path.abspath(saida)}")
    return {"grupos": resultados, "validacao": validacao}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Análise de avaliações Likert por grupos de avaliadores e validação "
                    "do juiz LLM por concordância com a avaliação humana.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exemplos:\n"
               "  %(prog)s --grupos gpt5:llm\n"
               "  %(prog)s --grupos gpt5:llm sabia4:llm humanos:humano --alias qwen7b=a\n"
               "  %(prog)s --pastas saida_01 saida_02 saida_03\n"
               "  %(prog)s --grupos gpt5:llm humanos:humano --bayes\n"
               "\n"
               "Etapa bayesiana (baycomp):\n"
               "  Sem `--bayes` a etapa não roda e nada muda no restante do pipeline.\n"
               "  A escala Likert é inteira, então a ROPE padrão de 0,5 significa\n"
               "  exatamente 'notas iguais' — não é margem arbitrada.\n")
    parser.add_argument("--base", default=".",
                        help="diretório que contém as pastas dos avaliadores (padrão: .)")
    parser.add_argument("--grupos", nargs="+", metavar="NOME:TIPO",
                        help=f"grupos a analisar; tipo em {TIPOS_VALIDOS}")
    parser.add_argument("--pastas", nargs="+", metavar="PASTA",
                        help="modo direto: pastas de um único grupo LLM "
                             f"(padrão quando nada é informado: {' '.join(PASTAS_PADRAO)})")
    parser.add_argument("--saida", default=PASTA_SAIDA_PADRAO,
                        help=f"pasta de saída (padrão: {PASTA_SAIDA_PADRAO})")
    parser.add_argument("--escala", nargs=2, type=int, default=list(ESCALA_PADRAO),
                        metavar=("MIN", "MAX"), help="limites da escala Likert (padrão: 1 4)")
    parser.add_argument("--alias", nargs="+", default=[], metavar="ORIGEM=DESTINO",
                        help="normaliza nomes de fonte entre grupos (ex.: qwen7b=a)")
    parser.add_argument("--referencia", default=None, metavar="GRUPO",
                        help="grupo padrão de referência (padrão: primeiro grupo do tipo humano)")

    bayesiano = parser.add_argument_group(
        "análise bayesiana (opcional)",
        "Camada complementar ao gate frequentista, via `baycomp`: probabilidade "
        "posterior de superioridade e de equivalência prática, mais o heatmap de "
        "comparação. Nada aqui roda sem `--bayes`.")
    bayesiano.add_argument("--bayes", action="store_true",
                           help="ativa a etapa bayesiana; sem esta flag ela é ignorada por completo")
    bayesiano.add_argument("--bayes-rope", type=float, default=BAYES_ROPE_LIKERT,
                           metavar="R",
                           help="largura da região de equivalência prática sobre as notas "
                                f"(padrão: {BAYES_ROPE_LIKERT}; na escala Likert inteira "
                                "equivale a 'notas iguais'). Deve ser > 0")
    bayesiano.add_argument("--bayes-metodo", choices=("sinais", "postos", "t"),
                           default=BAYES_METODO_PADRAO,
                           help="teste do baycomp: sinais=SignTest (ordinal, padrão), "
                                "postos=SignedRankTest, t=CorrelatedTTest (contínuo)")
    bayesiano.add_argument("--bayes-limiar", type=float, default=BAYES_LIMIAR_PADRAO,
                           metavar="P",
                           help="probabilidade mínima para classificar uma célula do heatmap; "
                                f"abaixo dela a relação fica `incerta` (padrão: {BAYES_LIMIAR_PADRAO})")
    bayesiano.add_argument("--bayes-limiar-veredito", type=float,
                           default=BAYES_VEREDITO_PADRAO, metavar="P",
                           help="probabilidade mínima para o veredito de equivalência juiz × "
                                f"referência (padrão: {BAYES_VEREDITO_PADRAO})")
    bayesiano.add_argument("--bayes-amostras", type=int, default=BAYES_AMOSTRAS_PADRAO,
                           metavar="N",
                           help=f"amostras da posterior (padrão: {BAYES_AMOSTRAS_PADRAO}); "
                                "ignorado por `--bayes-metodo t`, que é analítico")
    bayesiano.add_argument("--bayes-semente", type=int, default=SEMENTE, metavar="S",
                           help=f"semente da amostragem (padrão: {SEMENTE})")
    args = parser.parse_args(argv)

    if args.grupos and args.pastas:
        parser.error("use --grupos ou --pastas, não os dois.")

    config_bayes = montar_config_bayes(args, parser.error)

    aliases = parse_aliases(args.alias)
    try:
        if args.grupos:
            grupos = parse_grupos(args.grupos, args.base)
        else:
            pastas = args.pastas or PASTAS_PADRAO
            faltando = [p for p in pastas
                        if not os.path.exists(os.path.join(args.base, p, ARQUIVO_PARQUET))]
            if faltando:
                parser.error(f"pastas não encontradas: {', '.join(faltando)}")
            grupos = [Grupo(nome="grupo", tipo="llm", pastas=list(pastas))]
    except ValueError as exc:
        parser.error(str(exc))

    if args.referencia and args.referencia not in [g.nome for g in grupos]:
        parser.error(f"--referencia '{args.referencia}' não está entre os grupos informados.")

    executar(base=args.base, grupos=grupos, saida=args.saida,
             escala=tuple(args.escala), aliases=aliases, referencia=args.referencia,
             config_bayes=config_bayes)


if __name__ == "__main__":
    main()