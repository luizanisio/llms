#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

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
======  ======================  ================================  ==========
nº      critério                instrumento                       aprova se
======  ======================  ================================  ==========
1       concordância ordinal    κw de Cohen ponderado             κw >= 0,60
2       ausência de viés        P(equivalência) na diferença      >= 0,95
                                média das notas
3       decisão prática         P(equivalência) na taxa de        >= 0,95
                                adequação (nota >= 3)
======  ======================  ================================  ==========

Os critérios 2 e 3 são bayesianos: ``baycomp.CorrelatedTTest`` (Benavoli et
al., 2017, JMLR 18:1-36), via ``util_est_bayesiana`` — o teste proposto para
comparar dois algoritmos em UM conjunto de dados com observações pareadas.
O cálculo é analítico (Student acumulada): determinístico, sem amostras nem
semente. **Limiar único de 0,95** em todo o trabalho (gate, heatmap e forest
plot). O IC 95% do κw é reportado como medida de precisão, não como critério.

Desfechos: **Validado** (atende aos três) · **Com ressalva** (κw atende,
equivalência inconclusiva) · **Não validado** (κw abaixo do corte ou viés
relevante — a diferença quase certamente excede a ROPE).

ROPE calibrada e controle negativo
----------------------------------
A ROPE das notas é **calibrada pela divergência média entre os pares de
especialistas humanos** — medida entre eles, nunca a partir do par julgado.
A da taxa de adequação segue o mesmo procedimento sobre a binarização.
O **controle negativo** compara os próprios especialistas entre si com a ROPE
calibrada e verifica que saem equivalentes; se não saírem, a margem está
apertada demais — e é melhor descobrir antes de julgar o juiz.
O valor calibrado é reportado no ``validacao.md`` e deve ser **transcrito à
mão** para o YAML da Etapa 2 (a transcrição força o pré-registro consciente).

Limitação assumida: a diferença média de notas Likert É a diferença das
médias — o pareamento não contorna a objeção ordinal. Defesa: robustez
(Norman, 2010; Carifio & Perla, 2008); ROPE ancorada na divergência entre
especialistas, na mesma unidade; e as **contagens ordinais puras** (d>0, d=0,
d<0) reportadas ao lado de toda probabilidade.

Camada bayesiana complementar (opcional, `--bayes`)
---------------------------------------------------
Acrescenta as figuras de panorama — heatmap e "Medindo as diferenças" (forest
plot) — entre fontes e entre avaliadores. Sem a flag, apenas o gate usa a
camada bayesiana.

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
        --bayes
    python realizar_avaliacoes.py --grupos gpt5:llm humanos:humano \\
        --bayes-rope 0.20 --bayes-limiar 0.95
    python realizar_avaliacoes.py --pastas saida_01 saida_02 saida_03

Requisitos: pandas, numpy, scipy, pyarrow e ``util_est_bayesiana`` (baycomp) —
este último é obrigatório na validação, porque os critérios 2 e 3 do gate são
bayesianos. scikit-learn e statsmodels são usados quando presentes (com
fallback para as fórmulas internas, verificadas em
``realizar_avaliacoes_teste.py``); matplotlib e ``util_graficos`` só para as
figuras (a estatística roda mesmo sem eles).

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
# Quando o pacote não está instalado, o cálculo cai na fórmula interna — as
# definições operacionais estão em `realizar_avaliacoes_teste.py`, que verifica
# que coincidem com os pacotes.

try:
    from sklearn.metrics import cohen_kappa_score
    SKLEARN_DISPONIVEL = True
except ImportError:  # fallback: (P_o − P_e)/(1 − P_e) com pesos quadráticos
    cohen_kappa_score = None
    SKLEARN_DISPONIVEL = False

try:
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.proportion import proportion_confint
    from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
    STATSMODELS_DISPONIVEL = True
except ImportError:  # fallback: Holm step-down e Wilson em forma fechada
    multipletests = proportion_confint = aggregate_raters = fleiss_kappa = None
    STATSMODELS_DISPONIVEL = False


def dependencias(bayes_ativo: bool = False) -> dict:
    """Origem efetiva de cada estatística nesta execução (vai ao relatório)."""
    sk = (f"scikit-learn {__import__('sklearn').__version__}"
          if SKLEARN_DISPONIVEL else "fórmula interna (sklearn ausente)")
    sm = (f"statsmodels {__import__('statsmodels').__version__}"
          if STATSMODELS_DISPONIVEL else "fórmula interna (statsmodels ausente)")
    extra = {}
    if BAYES_DISPONIVEL:
        extra["Critérios 2 e 3 do gate / comparação bayesiana"] = (
            "baycomp.CorrelatedTTest (Benavoli et al., 2017), "
            "via util_est_bayesiana — analítico, sem amostras nem semente")
    return {
        "Kappa de Cohen ponderado": sk,
        "Kappa de Light (κw médio par a par)": f"média de κw de Cohen — {sk}",
        "Kappa de Fleiss (sem pesos)": sm,
        "Correção de Holm": sm,
        "IC de Wilson": sm,
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


#: padrões da camada bayesiana — um único teste (`baycomp.CorrelatedTTest`,
#: analítico) e um único limiar de decisão em todo o trabalho.
#:
#: A ROPE das notas é CALIBRADA pela divergência média entre os pares de
#: especialistas humanos — medida entre eles, nunca a partir do par julgado.
#: Os valores abaixo são o fallback quando a calibração não é possível (sem
#: grupo humano com 2+ avaliadores) e não foi dada a flag correspondente.
BAYES_ROPE_PADRAO = 0.5           # fallback das notas: escala inteira, "notas iguais"
BAYES_ROPE_DECISAO_PADRAO = 0.10  # fallback da taxa de adequação (10 p.p.)
#: pisos numéricos da calibração: o baycomp exige ROPE > 0, e especialistas em
#: acordo quase perfeito degenerariam a margem a zero
BAYES_ROPE_MINIMO = 0.05
BAYES_ROPE_DECISAO_MINIMO = 0.01
#: limiar ÚNICO de decisão — gate, heatmap e "Medindo as diferenças"
#: (Benavoli et al., 2017, §3.2)
BAYES_LIMIAR = 0.95

#: nome do único teste, para relatórios
_NOME_TESTE_BAYES = "baycomp.CorrelatedTTest"


@dataclass
class ConfigBayes:
    """Parâmetros da camada bayesiana (``util_est_bayesiana``, camada fina do baycomp).

    Um único teste em todo o pipeline: ``baycomp.CorrelatedTTest`` — analítico
    (Student acumulada), determinístico, sem amostras nem semente. A ROPE
    incide sobre a **diferença média**.

    * ``rope`` / ``rope_decisao`` — ``None`` significa **calibrar** pela
      divergência média entre os pares de especialistas do grupo humano de
      referência; um valor explícito (flag) sobrescreve a calibração e fica
      registrado no relatório como pré-registro manual.
    * ``limiar`` — limiar único de decisão (padrão 0,95).
    * ``ativo`` — liga a camada **complementar** (heatmaps e forest plots
      entre fontes e entre avaliadores). Os critérios 2 e 3 do gate são
      bayesianos e rodam sempre, com ou sem ``--bayes``.
    """
    ativo: bool = False
    rope: float = None
    rope_decisao: float = None
    limiar: float = BAYES_LIMIAR

    def kw(self, rope: float) -> dict:
        """Argumentos das chamadas ao módulo bayesiano, para uma ROPE resolvida."""
        return {"rope": rope, "limiar": self.limiar}


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


def _fleiss_interno(notas: np.ndarray, categorias: list,
                    pesos: np.ndarray = None) -> dict:
    """Fórmula do κ de Fleiss, usada como reserva quando falta `statsmodels`.

    Verificada contra `statsmodels.stats.inter_rater.fleiss_kappa` em
    `realizar_avaliacoes_teste.py`. `pesos` só é usado internamente para
    exibir P_o e P_e; o pipeline chama sempre com pesos identidade.
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

    if pesos is None:
        pesos = np.eye(q)
    p_o = float((np.einsum("ia,ab,ib->i", contagens, pesos, contagens) - m).sum()
                / (n * m * (m - 1)))
    prop = contagens.sum(axis=0) / (n * m)
    p_e = float(prop @ pesos @ prop)
    kappa = np.nan if np.isclose(p_e, 1.0) else (p_o - p_e) / (1.0 - p_e)
    return {"p_o": p_o, "p_e": p_e, "kappa": float(kappa), "n": n}


def fleiss_nao_ponderado(notas: np.ndarray, categorias: list) -> dict:
    """Kappa de Fleiss **sem pesos** para N avaliadores (Fleiss, 1971).

    Vem de ``statsmodels.stats.inter_rater.fleiss_kappa``. Entra no relatório
    como **piso conservador**: por não ponderar, trata uma divergência de 1
    ponto como equivalente a uma de 3, o que subestima a confiabilidade numa
    escala ordinal curta. É o número clássico, reportado para registro.

    Returns:
        dict com ``p_o``, ``p_e``, ``kappa`` e ``n``.
    """
    notas = np.asarray(notas)
    interno = _fleiss_interno(notas, categorias)
    if not STATSMODELS_DISPONIVEL or interno["n"] == 0:
        return interno
    tabela, _ = aggregate_raters(notas)
    interno["kappa"] = float(fleiss_kappa(tabela, method="fleiss"))
    return interno


def kappa_light(matriz: pd.DataFrame, avaliadores: list, categorias: list) -> dict:
    """Kappa de Light (1971): média dos κ de Cohen **ponderados** par a par.

    Coeficiente principal de confiabilidade interna dos grupos. Escolhido
    sobre o Fleiss por dois motivos:

    1. **Preserva a ponderação quadrática.** Numa Likert de 4 pontos a maior
       parte das divergências é de 1 ponto; o Fleiss clássico as pune como se
       fossem de 3, e o coeficiente despenca sem que a confiabilidade real
       tenha mudado.
    2. **Sai inteiro de biblioteca.** Cada κ par a par vem do
       ``sklearn.metrics.cohen_kappa_score``; não há Fleiss ponderado em
       pacote consolidado, e implementá-lo à mão seria a única estatística do
       pipeline sem lastro em software estabelecido.

    Light e Fleiss não coincidem numericamente: Light usa o acaso esperado de
    cada par, o Fleiss um baseline comum a todos (Conger, 1980). A diferença
    costuma ser pequena, e o relatório traz os dois lado a lado.

    Returns:
        dict com ``p_o`` e ``p_e`` (médias dos componentes par a par),
        ``kappa``, ``n`` (itens) e ``pares`` (n.º de pares).
    """
    vazio = {"p_o": np.nan, "p_e": np.nan, "kappa": np.nan, "n": 0, "pares": 0}
    if matriz.empty or len(avaliadores) < 2:
        return vazio
    comps = [cohen_ponderado(matriz[a1], matriz[a2], categorias)
             for a1, a2 in combinations(avaliadores, 2)]
    comps = [c for c in comps if c["kappa"] == c["kappa"]]
    if not comps:
        return vazio
    return {"p_o": float(np.mean([c["p_o"] for c in comps])),
            "p_e": float(np.mean([c["p_e"] for c in comps])),
            "kappa": float(np.mean([c["kappa"] for c in comps])),
            "n": len(matriz), "pares": len(comps)}


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
        if SKLEARN_DISPONIVEL:
            try:
                kappa = float(cohen_kappa_score(a, b, labels=list(categorias),
                                                weights="quadratic"))
            except (ValueError, ZeroDivisionError):
                kappa = np.nan
        else:
            # fórmula interna (idêntica ao sklearn com pesos quadráticos),
            # verificada em realizar_avaliacoes_teste.py
            kappa = np.nan if np.isclose(p_e, 1.0) else float((p_o - p_e) / (1.0 - p_e))
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
    if STATSMODELS_DISPONIVEL:
        inferior, superior = proportion_confint(sucessos, total,
                                                alpha=1 - confianca, method="wilson")
        return (p, float(inferior), float(superior))
    # forma fechada de Wilson (verificada em realizar_avaliacoes_teste.py)
    z = stats.norm.ppf(1 - (1 - confianca) / 2)
    centro = (p + z ** 2 / (2 * total)) / (1 + z ** 2 / total)
    margem = (z / (1 + z ** 2 / total)) * np.sqrt(
        p * (1 - p) / total + z ** 2 / (4 * total ** 2))
    return (p, float(centro - margem), float(centro + margem))


def correcao_holm(p_valores: Sequence[float]) -> list:
    """Correção de Holm-Bonferroni (step-down), preservando a monotonicidade.

    Calculada por ``statsmodels.stats.multitest.multipletests``.
    p-valores ausentes são tratados como 1,0 antes da correção.
    """
    m = len(p_valores)
    if m == 0:
        return []
    limpos = [1.0 if (v is None or v != v) else float(v) for v in p_valores]
    if STATSMODELS_DISPONIVEL:
        return [float(v) for v in multipletests(limpos, method="holm")[1]]
    # step-down de Holm com monotonicidade (verificado em realizar_avaliacoes_teste.py)
    ordem = np.argsort(limpos)
    ajustados = np.empty(m)
    maximo = 0.0
    for posicao, indice in enumerate(ordem):
        maximo = max(maximo, min(1.0, (m - posicao) * limpos[indice]))
        ajustados[indice] = maximo
    return [float(v) for v in ajustados]


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
    # coeficiente principal: κ de Light (κw de Cohen médio par a par, sklearn)
    comp = kappa_light(matriz, avaliadores, categorias)
    # registro: κ de Fleiss clássico, sem pesos (statsmodels) — piso conservador
    fleiss = fleiss_nao_ponderado(valores, categorias)
    ic_inf, ic_sup = ic_bootstrap(
        matriz, lambda d: kappa_light(d, avaliadores, categorias)["kappa"])

    global_ = {
        "n itens": len(matriz),
        "P_o": round(comp["p_o"], 4),
        "P_e": round(comp["p_e"], 4),
        "kappa": round(comp["kappa"], 4),
        "pares": comp["pares"],
        "kappa_fleiss": round(fleiss["kappa"], 4),
        "P_o_fleiss": round(fleiss["p_o"], 4),
        "ic_inf": round(ic_inf, 4), "ic_sup": round(ic_sup, 4),
        "interpretacao": interpretar_kappa(comp["kappa"]),
        "interpretacao_fleiss": interpretar_kappa(fleiss["kappa"]),
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
# 6.5 Camada bayesiana — calibração da ROPE, gate e panorama complementar
# =============================================================================
# Um único teste em todo o pipeline: `baycomp.CorrelatedTTest`, via
# `util_est_bayesiana`. Os critérios 2 e 3 do gate são bayesianos e rodam
# sempre; o que a flag `--bayes` liga é apenas o panorama COMPLEMENTAR
# (heatmap e forest plot entre fontes e entre avaliadores).
#
# Toda a estatística vem do baycomp. Aqui só se organizam as chamadas e se
# formatam os resultados.


def calibrar_rope(matriz_avaliadores: pd.DataFrame, avaliadores: list,
                  piso: int = PISO_ADEQUACAO) -> dict:
    """Calibra as ROPEs pela divergência média entre os pares de especialistas.

    A margem de "praticamente equivalente" deixa de ser arbitrada: passa a ser
    a divergência média observada ENTRE os próprios especialistas humanos —
    medida entre eles, **nunca a partir do par julgado**. A interpretação fica
    relativa ("o juiz diverge da referência tanto quanto os especialistas
    divergem entre si"), o que é parte da defesa contra a objeção ordinal.

    Duas margens, cada uma na sua unidade, ambas ancoradas na **divergência
    absoluta média por item** — média de |nota_i − nota_j| sobre os itens,
    depois sobre os pares. É a magnitude típica do desacordo entre
    especialistas, e não o viés médio entre eles (|média(nota_i − nota_j)|),
    que se anula quando as divergências são simétricas e degeneraria a margem;
    o viés é reportado ao lado, para leitura:

    * ``rope_notas`` — média dos pares de média(|nota_i − nota_j|);
    * ``rope_decisao`` — o mesmo sobre a binarização em ``nota >= piso``
      (taxa de decisões discordantes entre os especialistas).

    Pisos numéricos (``BAYES_ROPE_MINIMO`` e ``BAYES_ROPE_DECISAO_MINIMO``)
    evitam a degeneração a zero quando os especialistas quase não divergem —
    o baycomp exige ROPE > 0. Quando o piso é aplicado, isso fica registrado.

    Args:
        matriz_avaliadores: itens × avaliadores (saída de
            ``matriz_itens_avaliadores`` do grupo humano de referência).
        avaliadores: colunas a considerar (2+).
        piso: nota mínima considerada adequada.

    Returns:
        dict com ``rope_notas``, ``rope_decisao``, ``tabela`` (uma linha por
        par de especialistas), ``n_itens``, ``piso_aplicado`` (bool por margem).
    """
    pares = []
    for a1, a2 in combinations(avaliadores, 2):
        x = matriz_avaliadores[a1].astype(float).to_numpy()
        y = matriz_avaliadores[a2].astype(float).to_numpy()
        bx, by = (x >= piso).astype(float), (y >= piso).astype(float)
        pares.append({
            "Par": f"{a1} × {a2}",
            "n": len(x),
            "Divergência abs. (notas)": float(np.mean(np.abs(x - y))),
            "Viés médio (notas)": float(np.mean(x - y)),
            "Decisões discordantes": float(np.mean(np.abs(bx - by))),
            "Viés médio (taxa)": float(np.mean(bx - by)),
        })
    tabela = pd.DataFrame(pares)
    bruto_notas = (float(tabela["Divergência abs. (notas)"].mean())
                   if len(tabela) else np.nan)
    bruto_taxa = (float(tabela["Decisões discordantes"].mean())
                  if len(tabela) else np.nan)
    rope_notas = max(bruto_notas, BAYES_ROPE_MINIMO)
    rope_decisao = max(bruto_taxa, BAYES_ROPE_DECISAO_MINIMO)
    return {
        "rope_notas": rope_notas, "rope_decisao": rope_decisao,
        "bruto_notas": bruto_notas, "bruto_taxa": bruto_taxa,
        "piso_aplicado": {"notas": bruto_notas < BAYES_ROPE_MINIMO,
                          "decisao": bruto_taxa < BAYES_ROPE_DECISAO_MINIMO},
        "tabela": tabela, "n_itens": int(len(matriz_avaliadores)),
        "n_pares": len(pares),
    }


def controle_negativo(matriz_avaliadores: pd.DataFrame, avaliadores: list,
                      rope: float, rope_decisao: float,
                      limiar: float = BAYES_LIMIAR,
                      piso: int = PISO_ADEQUACAO) -> pd.DataFrame:
    """Controle negativo da Likert: especialista × especialista com a ROPE calibrada.

    Compara os especialistas humanos entre si com a mesma ROPE e o mesmo
    limiar do gate, e verifica que saem **equivalentes**. Se não saírem, a
    margem está apertada demais — e é melhor descobrir antes de julgar o juiz.
    Fecha a pergunta sobre a margem ter sido escolhida convenientemente, com
    dados já disponíveis.
    """
    linhas = []
    for a1, a2 in combinations(avaliadores, 2):
        x = matriz_avaliadores[a1].astype(float).to_numpy()
        y = matriz_avaliadores[a2].astype(float).to_numpy()
        notas = bayes.Comparacao(x, y, rope=rope, limiar=limiar)
        decisao = bayes.Comparacao((x >= piso).astype(float),
                                   (y >= piso).astype(float),
                                   rope=rope_decisao, limiar=limiar)
        contagens = notas.contagens
        linhas.append({
            "Par": f"{a1} × {a2}", "n": len(x),
            "A>": contagens["x_melhor"], "=": contagens["empate"],
            "B>": contagens["y_melhor"],
            "Dif. média": round(notas.diferenca_media, 4),
            "P(equiv. notas)": round(notas.probabilidades["p_rope"], 4),
            "P(equiv. decisão)": round(decisao.probabilidades["p_rope"], 4),
            "Passa": "sim" if (notas.probabilidades["p_rope"] >= limiar
                               and decisao.probabilidades["p_rope"] >= limiar)
                     else "NÃO",
        })
    return pd.DataFrame(linhas)


def matriz_bayesiana(dados: pd.DataFrame, colunas: list, rope: float,
                     limiar: float = BAYES_LIMIAR) -> pd.DataFrame:
    """Compara todos os pares das colunas indicadas (`CorrelatedTTest`)."""
    return bayes.matriz_pares(dados, nomes=list(colunas), rope=rope, limiar=limiar)


def tabela_matriz_bayesiana(matriz: pd.DataFrame, rotulo: str = "Par") -> pd.DataFrame:
    """Formata a matriz para leitura: um par NÃO ordenado por linha.

    As colunas `A melhor`/`Empate`/`B melhor` são contagens ORDINAIS PURAS
    (d>0, d=0, d<0) — a âncora contra a objeção da escala ordinal, reportadas
    ao lado de toda probabilidade.
    """
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
            "IC 95%": f"[{linha['ic_inf']:+.4f}; {linha['ic_sup']:+.4f}]".replace(".", ","),
            "P(A > B)": round(float(linha["p_esquerda"]), 4),
            "P(equiv.)": round(float(linha["p_rope"]), 4),
            "P(A < B)": round(float(linha["p_direita"]), 4),
            "Relação de A": linha["classificacao"],
        })
    return pd.DataFrame(linhas)


def analise_bayesiana_grupo(r: dict, cfg: ConfigBayes) -> dict:
    """Panorama bayesiano entre as fontes, dentro de um grupo (complementar).

    A unidade é a nota mediana por documento — a mesma variável primária do
    Friedman/Wilcoxon. O que muda é a pergunta: em vez de "há diferença
    detectável?", responde "qual a probabilidade de esta fonte superar aquela,
    e qual a de serem praticamente equivalentes?".

    A ROPE segue a mesma regra do gate: valor explícito da flag; senão,
    calibrada pela divergência entre os avaliadores do PRÓPRIO grupo (2+);
    senão, o padrão de escala.
    """
    if len(r["fontes"]) < 2:
        return {}
    rope = cfg.rope
    origem_rope = "flag --bayes-rope (pré-registro manual)"
    if rope is None:
        if len(r["avaliadores"]) >= 2:
            calibracao = calibrar_rope(
                matriz_itens_avaliadores(r["df"], r["avaliadores"]),
                r["avaliadores"])
            rope = calibracao["rope_notas"]
            origem_rope = (f"calibrada pela divergência média entre os "
                           f"{len(r['avaliadores'])} {r['grupo'].rotulos['avaliadores']} do grupo")
        else:
            rope = BAYES_ROPE_PADRAO
            origem_rope = "padrão de escala (grupo com um só avaliador)"
    matriz = matriz_bayesiana(r["pivo"], r["fontes"], rope, cfg.limiar)
    figuras = viz.grafico_bayes(
        matriz, os.path.join(r["saida"], "10_bayes_fontes.png"),
        titulo="Comparação bayesiana entre as fontes (nota mediana)",
        rotulo_entidade="fonte")
    figuras += viz.grafico_diferencas_bayes(
        matriz, os.path.join(r["saida"], "11_bayes_fontes_diferencas.png"),
        titulo="Fontes — Medindo as diferenças (forest plot)")
    matriz.to_csv(os.path.join(r["saida"], "bayes_fontes.csv"),
                  index=False, encoding="utf-8")
    return {"config": cfg, "matriz": matriz, "rope": rope,
            "origem_rope": origem_rope,
            "sintese": bayes.sintese(matriz),
            "tabela": tabela_matriz_bayesiana(matriz, "Fonte"),
            "figuras": figuras}


def analise_bayesiana_validacao(v: dict, cfg: ConfigBayes) -> dict:
    """Panorama bayesiano da validação: matriz entre avaliadores (complementar).

    O gate (critérios 2 e 3) já é bayesiano e vive em `avaliar_par`; esta
    seção acrescenta o panorama entre TODOS os avaliadores — inclusive os
    pares juiz × juiz, que ficam fora do gate — na mesma ROPE e no mesmo
    limiar 0,95.
    """
    longo, nomes = v["longo"], list(v["pivos"].keys())
    matriz = None
    figuras = []
    if len(nomes) >= 2:
        matriz = matriz_bayesiana(longo[nomes], nomes, v["rope"], cfg.limiar)
        figuras = viz.grafico_bayes(
            matriz, os.path.join(v["saida"], "04_bayes_grupos.png"),
            titulo="Comparação bayesiana entre os avaliadores (nota mediana)",
            rotulo_entidade="avaliador")
        figuras += viz.grafico_diferencas_bayes(
            matriz, os.path.join(v["saida"], "05_bayes_grupos_diferencas.png"),
            titulo="Avaliadores — Medindo as diferenças (forest plot)")
        matriz.to_csv(os.path.join(v["saida"], "bayes_grupos.csv"),
                      index=False, encoding="utf-8")

    return {"config": cfg, "matriz": matriz,
            "sintese": bayes.sintese(matriz) if matriz is not None else None,
            "tabela_matriz": (tabela_matriz_bayesiana(matriz, "Avaliador")
                              if matriz is not None else None),
            "figuras": figuras}



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
                categorias: list, rope: float = BAYES_ROPE_PADRAO,
                rope_decisao: float = BAYES_ROPE_DECISAO_PADRAO,
                limiar: float = BAYES_LIMIAR) -> dict:
    """Aplica os três critérios do gate a um par (juiz, referência).

    Tudo no agregado — todas as combinações documento × fonte empilhadas numa
    única série. Sem estratificação por fonte.

    Critérios (limiar único de 0,95):

    1. **concordância ordinal** — κw de Cohen ponderado ≥ 0,60 (frequentista);
    2. **ausência de viés** — P(equivalência) da diferença média das notas,
       ao ``rope`` calibrado, ≥ ``limiar`` (``baycomp.CorrelatedTTest``);
    3. **decisão prática** — P(equivalência) da taxa de adequação
       (nota ≥ piso), ao ``rope_decisao``, ≥ ``limiar``.

    Resultados possíveis:

    * **VALIDADO** — os três critérios atendidos.
    * **VALIDADO COM RESSALVA** — κw atende e a equivalência (2 e/ou 3) ficou
      **inconclusiva** — os dados não bastam para afirmá-la nem para afirmar
      viés relevante.
    * **NÃO VALIDADO** — κw abaixo do corte, ou **viés relevante**: a
      diferença quase certamente excede a ROPE em alguma direção
      (max(P(juiz>ref), P(juiz<ref)) ≥ limiar nas notas ou na decisão).
    """
    sub = longo[["documento", juiz, referencia]].rename(
        columns={juiz: "a", referencia: "b"})
    a = sub["a"].astype(int).to_numpy()
    b = sub["b"].astype(int).to_numpy()

    # critério 1 — concordância ordinal
    comp = cohen_ponderado(a, b, categorias)
    ic_inf, ic_sup = ic_bootstrap(
        sub, lambda d: cohen_kappa(d["a"].astype(int), d["b"].astype(int), categorias))

    # critérios 2 e 3 — bayesianos (P(equivalência) contra a ROPE calibrada)
    if not BAYES_DISPONIVEL:
        raise RuntimeError(
            "Os critérios 2 e 3 do gate são bayesianos e exigem o módulo "
            "`util_est_bayesiana.py` (e o pacote `baycomp`) na mesma pasta "
            "ou no PYTHONPATH.")
    notas = bayes.Comparacao(a.astype(float), b.astype(float),
                             rope=rope, limiar=limiar)
    bin_a, bin_b = a >= PISO_ADEQUACAO, b >= PISO_ADEQUACAO
    decisao = bayes.Comparacao(bin_a.astype(float), bin_b.astype(float),
                               rope=rope_decisao, limiar=limiar)
    met = metricas_binarias(referencia=bin_b, teste=bin_a)

    p_eq_notas = notas.probabilidades["p_rope"]
    p_eq_decisao = decisao.probabilidades["p_rope"]
    # viés relevante = quase certeza de que a diferença EXCEDE a ROPE
    vies_relevante = bool(
        max(notas.probabilidades["p_esquerda"], notas.probabilidades["p_direita"]) >= limiar
        or max(decisao.probabilidades["p_esquerda"],
               decisao.probabilidades["p_direita"]) >= limiar)

    criterios = {
        "concordancia": bool(comp["kappa"] == comp["kappa"] and comp["kappa"] >= LIMIAR_KAPPA),
        "sem_vies": bool(p_eq_notas >= limiar),
        "decisao": bool(p_eq_decisao >= limiar),
    }
    if all(criterios.values()):
        status = "VALIDADO"
    elif not criterios["concordancia"] or vies_relevante:
        status = "NÃO VALIDADO"
    else:
        status = "VALIDADO COM RESSALVA"

    media_dif = notas.diferenca_media
    if abs(media_dif) < 1e-9:
        direcao = "—"
    else:
        direcao = (f"{juiz} mais leniente" if media_dif > 0
                   else f"{referencia} mais leniente")
    contagens = notas.contagens        # ordinais puras: d>0, d=0, d<0

    return {
        "juiz": juiz, "referencia": referencia, "n": len(a),
        "kappa": comp["kappa"], "ic_inf": ic_inf, "ic_sup": ic_sup,
        "p_o": comp["p_o"], "interpretacao": interpretar_kappa(comp["kappa"]),
        "exata": float(np.mean(a == b)), "amplitude_1": float(np.mean(np.abs(a - b) <= 1)),
        "media_dif": media_dif, "mediana_dif": float(np.median(a - b)),
        "ic_notas": notas.ic95, "direcao": direcao,
        "acima": contagens["x_melhor"], "empate": contagens["empate"],
        "abaixo": contagens["y_melhor"],
        "rope": rope, "p_equiv_notas": p_eq_notas,
        "p_juiz_maior": notas.probabilidades["p_esquerda"],   # P(dif > +ROPE)
        "p_juiz_menor": notas.probabilidades["p_direita"],    # P(dif < -ROPE)
        "kappa_bin": cohen_kappa(bin_a.astype(int), bin_b.astype(int), [0, 1]),
        "b01": int(np.sum(bin_a & ~bin_b)), "b10": int(np.sum(~bin_a & bin_b)),
        "rope_decisao": rope_decisao, "p_equiv_decisao": p_eq_decisao,
        "dif_taxa": decisao.diferenca_media, "ic_decisao": decisao.ic95,
        "acuracia": met["acuracia"], "sensibilidade": met["sensibilidade"],
        "especificidade": met["especificidade"],
        "criterios": criterios, "status": status, "limiar": limiar,
        "validado": status == "VALIDADO", "vies_relevante": vies_relevante,
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

    if not BAYES_DISPONIVEL:
        raise RuntimeError(
            "A validação exige `util_est_bayesiana.py` (e o pacote `baycomp`): "
            "os critérios 2 e 3 do gate são bayesianos "
            "(P(equivalência) via baycomp.CorrelatedTTest).")
    cfg = config_bayes if config_bayes is not None else ConfigBayes()

    # ── calibração da ROPE pela divergência entre os especialistas ─────────
    # medida ENTRE os especialistas do grupo humano de referência — nunca a
    # partir do par julgado. Um valor explícito de flag sobrescreve (e fica
    # registrado como pré-registro manual).
    calibracao, controle = None, None
    ref_result = resultados[referencia]
    pode_calibrar = referencia_humana and len(ref_result.get("avaliadores", [])) >= 2
    if pode_calibrar:
        matriz_ref = matriz_itens_avaliadores(ref_result["df"],
                                              ref_result["avaliadores"])
        calibracao = calibrar_rope(matriz_ref, ref_result["avaliadores"])

    rope = cfg.rope if cfg.rope is not None else (
        calibracao["rope_notas"] if calibracao else BAYES_ROPE_PADRAO)
    rope_decisao = cfg.rope_decisao if cfg.rope_decisao is not None else (
        calibracao["rope_decisao"] if calibracao else BAYES_ROPE_DECISAO_PADRAO)
    origem_rope = ("flag (pré-registro manual)" if cfg.rope is not None else
                   ("calibrada pelos especialistas" if calibracao else
                    "padrão de escala (sem 2+ especialistas na referência)"))
    print(f"  ROPE (notas) = {rope:.4f} · ROPE (taxa) = {rope_decisao:.4f} "
          f"[{origem_rope}] · limiar único = {cfg.limiar:.2f}")

    # ── controle negativo: especialista × especialista, mesma ROPE ─────────
    if calibracao is not None and cfg.rope is None:
        controle = controle_negativo(matriz_ref, ref_result["avaliadores"],
                                     rope, rope_decisao, cfg.limiar)
        reprovados = int((controle["Passa"] == "NÃO").sum())
        if reprovados:
            print(f"  ⚠ controle negativo: {reprovados} de {len(controle)} "
                  "par(es) de especialistas NÃO saem equivalentes com a ROPE "
                  "calibrada — a margem pode estar apertada demais.")
        else:
            print(f"  ✅ controle negativo: os {len(controle)} par(es) de "
                  "especialistas saem equivalentes com a ROPE calibrada.")

    gates = [avaliar_par(longo, juiz, referencia, categorias,
                         rope=rope, rope_decisao=rope_decisao, limiar=cfg.limiar)
             for juiz in juizes]
    laterais = [avaliar_par(longo, a, b, categorias,
                            rope=rope, rope_decisao=rope_decisao, limiar=cfg.limiar)
                for a, b in combinations(juizes, 2)]

    resultado = {
        "grupos": {n: r["grupo"] for n, r in resultados.items()},
        "pivos": pivos, "longo": longo, "cobertura": cobertura,
        "documentos": documentos, "fontes": fontes, "categorias": categorias,
        "referencia": referencia, "referencia_humana": referencia_humana,
        "grupos_humanos": humanos, "juizes": juizes,
        "rope": rope, "rope_decisao": rope_decisao, "origem_rope": origem_rope,
        "limiar": cfg.limiar, "calibracao": calibracao, "controle": controle,
        "gates": gates, "laterais": laterais,
        "entre_grupos": comparar_grupos(longo, list(pivos.keys())),
        "ranks": tabela_ranks(pivos, fontes), "saida": saida,
        "config_bayes": cfg,
    }

    if cfg.ativo:
        print("  panorama bayesiano entre os avaliadores...")
        resultado["bayes"] = analise_bayesiana_validacao(resultado, cfg)

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
    if calibracao is not None:
        calibracao["tabela"].to_csv(
            os.path.join(saida, "tabela_calibracao_rope.csv"),
            index=False, encoding="utf-8")
    if controle is not None:
        controle.to_csv(os.path.join(saida, "tabela_controle_negativo.csv"),
                        index=False, encoding="utf-8")
    escrever_relatorio_validacao(resultado, os.path.join(saida, "validacao.md"))

    icones = {"VALIDADO": "✅", "VALIDADO COM RESSALVA": "🟡", "NÃO VALIDADO": "❌"}
    for gate in gates:
        print(f"  {icones[gate['status']]} {gate['status']}: {gate['juiz']} "
              f"(κw = {gate['kappa']:.3f}, "
              f"P(equiv. notas) = {gate['p_equiv_notas']:.4f}, "
              f"P(equiv. decisão) = {gate['p_equiv_decisao']:.4f}, "
              f"dif. média = {gate['media_dif']:+.3f})")
    return resultado


# =============================================================================
# 8. Relatório — análise interna do grupo
# =============================================================================

def _bloco_light_vs_fleiss(interna: dict) -> list:
    """Parágrafo que justifica o coeficiente principal e lê a distância entre os dois."""
    light, fleiss = interna["kappa"], interna["kappa_fleiss"]
    L = [f"O coeficiente principal é o **κ de Light** (Light, 1971): a média dos κ de "
         f"Cohen **ponderados** (pesos quadráticos) dos {interna['pares']} pares de "
         "avaliadores, cada um calculado por `sklearn.metrics.cohen_kappa_score`. A "
         "ponderação importa porque a escala é ordinal e curta: divergir por 1 ponto "
         "não é o mesmo que divergir por 3. O **κ de Fleiss** clássico "
         "(`statsmodels`) vai ao lado, para registro — por não ponderar, funciona "
         "como **piso conservador** da confiabilidade.\n"]

    if light == light and fleiss == fleiss:
        distancia = light - fleiss
        if distancia >= 0.10:
            L.append(f"A distância entre os dois ({_num(distancia, 3)}) é a medida direta "
                     "do peso das divergências de 1 ponto neste grupo: quase todo o "
                     "desacordo é de vizinhança na escala, e some quando a proximidade "
                     "recebe crédito parcial.\n")
        elif distancia <= -0.05:
            L.append(f"O Fleiss ficou **acima** do Light ({_num(-distancia, 3)} de "
                     "diferença), o que indica desacordo concentrado em poucos pares ou "
                     "distribuições marginais bem distintas entre avaliadores — vale "
                     "olhar a tabela par a par antes de ler o valor global.\n")
        else:
            L.append(f"Os dois coeficientes ficam próximos ({_num(distancia, 3)} de "
                     "diferença), sinal de que a ponderação não está carregando a "
                     "conclusão: o resultado se sustenta com ou sem crédito parcial.\n")

    L.append("Não se usa o **Fleiss ponderado** por uma razão de método, não de "
             "estatística: ele não existe em pacote consolidado de Python, e "
             "implementá-lo à mão seria a única medida do pipeline sem lastro em "
             "software estabelecido. O κ de Light entrega a mesma propriedade "
             "(ponderação ordinal para N avaliadores) inteiramente a partir de "
             "biblioteca. Os dois não coincidem numericamente — Light usa o acaso "
             "esperado de cada par; Fleiss, um baseline comum (Conger, 1980).\n")
    return L


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
        L.append(f"| {rot['plural']} | {len(r['avaliadores'])} "
                 f"({interna['pares']} par(es)) |")
        L.append(f"| Concordância observada P_o (ponderada) | {_num(interna['P_o'])} |")
        L.append(f"| Concordância esperada por acaso P_e | {_num(interna['P_e'])} |")
        L.append(f"| **κ de Light (κw de Cohen médio par a par)** | "
                 f"**{_num(interna['kappa'])}** "
                 f"[IC 95%: {_num(interna['ic_inf'])}; {_num(interna['ic_sup'])}] |")
        L.append(f"| Interpretação | nível *{interna['interpretacao']}* |")
        L.append(f"| κ de Fleiss (clássico, sem pesos) | {_num(interna['kappa_fleiss'])} "
                 f"— nível *{interna['interpretacao_fleiss']}* |")
        L.append(f"| Concordância exata | {_pct(interna['exata'])} |")
        L.append(f"| Itens com amplitude ≤ 1 ponto | {_pct(interna['amplitude_1'])} |")
        L.append("")
        L.extend(_bloco_light_vs_fleiss(interna))
        if interna["aprovado"]:
            L.append(f"✅ **Aprovado** — κ de Light = {_num(interna['kappa'])} ≥ "
                     f"{_num(LIMIAR_KAPPA, 2)}. A variável primária deste grupo pode ser usada "
                     "sem ressalva de confiabilidade.\n")
        else:
            L.append(f"⚠️ **Reprovado** — κ de Light = {_num(interna['kappa'])} < "
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


def _bloco_como_ler_bayes(contexto: dict, pares_info: list,
                          rotulo_entidade: str = "fonte",
                          referencia: str = None) -> list:
    """Subseção 'Como ler esta análise' — inserida antes das tabelas bayesianas.

    Explica a ROPE e a leitura do heatmap, e fecha com dois exemplos gerados a
    partir dos extremos: o par mais próximo da equivalência e o mais distante.
    Dois, e não um, porque cada extremo exercita um caminho de leitura que o
    outro não cobre.
    """
    plural = _plural(rotulo_entidade)
    rope = contexto.get("rope")
    limiar = contexto.get("limiar", BAYES_LIMIAR)
    L = ["### Como ler esta análise\n"]

    L.append(
        f"A comparação é **pareada por documento** e roda no `{_NOME_TESTE_BAYES}`: "
        f"a posterior de Student da **diferença média** entre os dois "
        f"{plural} é confrontada com a ROPE, e daí saem as três probabilidades "
        "(superior, equivalente, inferior). O cálculo é analítico — "
        "determinístico, sem amostras nem semente.\n")

    L.append(
        f"**ROPE = {_num(rope, 4)}** ({contexto.get('origem_rope', 'ver calibração')}). "
        "A margem é ancorada na divergência média entre os especialistas humanos, na "
        "mesma unidade das notas — a leitura é **relativa**, não absoluta.\n")

    L.append(
        "**Contagens ordinais ao lado de toda probabilidade.** A diferença média de "
        "notas Likert é a diferença das médias — o pareamento não contorna a objeção "
        "ordinal (robustez: Norman, 2010; Carifio & Perla, 2008). Por isso as colunas "
        "`A melhor`/`Empate`/`B melhor` são contagens **puras** (d > 0, d = 0, d < 0), "
        "independentes da ROPE: um número que não assume intervalos.\n")

    L.append(
        f"**Como ler o heatmap:** cada célula mostra a relação da **linha** em relação à "
        f"**coluna** — verde = superior, vermelho = inferior, azul = equivalente, cinza = "
        f"incerto. O número é a **probabilidade posterior** da categoria colorida, e a "
        f"intensidade da cor a acompanha. Cinza significa que nenhuma das três alcançou o "
        f"limiar de {_num(limiar, 2)} — desfecho legítimo, não ausência de resultado. "
        "\"Medindo as diferenças\" (forest plot) mostra o mesmo conteúdo em magnitude: "
        "ponto = diferença média, barra = IC 95%, faixa = ROPE.\n")

    favoravel, desafiador = _pares_didaticos(pares_info)
    for rotulo, exemplo in (("mais favorável", favoravel),
                            ("mais desafiador", desafiador)):
        if exemplo is not None:
            L.append(_texto_exemplo(exemplo, limiar, plural, referencia, rotulo))
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


def _texto_exemplo(exemplo: dict, limiar: float, plural: str, referencia: str,
                   rotulo: str) -> str:
    """Um parágrafo de leitura guiada para um par concreto, com números da execução."""
    nome_a = exemplo.get("juiz") or exemplo.get("nome_a", "A")
    nome_b = referencia or exemplo.get("nome_b", "B")
    p_eq = exemplo.get("p_equiv", float("nan"))
    acima = exemplo.get("acima", "?")
    empate = exemplo.get("empate", "?")
    abaixo = exemplo.get("abaixo", "?")
    n_obs = exemplo.get("n", "?")

    atinge = p_eq >= limiar
    posicao = f"{'acima' if atinge else 'abaixo'} do limiar de {_num(limiar, 2)}"
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
    L.append(f"| Teste | `{_NOME_TESTE_BAYES}` (analítico — sem amostras nem semente) |")
    L.append("| Variável | nota Likert mediana por documento |")
    L.append(f"| ROPE | {_num(b['rope'], 4)} ({b['origem_rope']}) |")
    L.append(f"| Limiar único | {_num(cfg.limiar, 2)} |")
    L.append("")
    L.extend(_bloco_como_ler_bayes(
        {"rope": b["rope"], "limiar": cfg.limiar, "origem_rope": b["origem_rope"]},
        _pares_info_de_tabela(b["tabela"]), rotulo_entidade="fonte"))
    L.append("### Relações par a par\n")
    L.append(_md(b["tabela"], indice=False))
    L.append("")
    L.append("### Contando as relações\n")
    L.append(_md(b["sintese"]))
    L.append("")
    ciclos = b["sintese"].attrs.get("ciclos") or []
    if ciclos:
        L.append(f"⚠️ **Transitividade violada** — ciclo(s): "
                 f"{'; '.join(' > '.join(map(str, c)) for c in ciclos)}. A leitura "
                 "ordenada da tabela é inválida.\n")
    else:
        L.append("✅ Verificação de transitividade: as relações direcionais não formam "
                 "ciclo.\n")
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
                     f"{'aprovada' if interna['aprovado'] else 'reprovada'} — "
                     f"κ de Light = {_num(interna['kappa'])} "
                     f"[{_num(interna['ic_inf'])}; {_num(interna['ic_sup'])}], "
                     f"κ de Fleiss = {_num(interna['kappa_fleiss'])}, "
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

def _bloco_calibracao_rope(v: dict) -> list:
    """Subseção do `validacao.md`: origem das ROPEs e controle negativo."""
    L = ["### Calibração da ROPE\n"]
    L.append(f"**ROPE (notas) = {_num(v['rope'], 4)} ponto · ROPE (taxa de adequação) = "
             f"{_num(v['rope_decisao'], 4)}** — origem: {v['origem_rope']}.\n")
    calibracao = v.get("calibracao")
    if calibracao is not None:
        L.append("A margem não é arbitrada: é a **divergência absoluta média por item "
                 "entre os pares de especialistas humanos** da referência — medida "
                 "entre eles, nunca a partir do par julgado. A leitura fica relativa: "
                 "o juiz é praticamente equivalente à referência quando o seu "
                 "desvio médio cabe na magnitude típica do desacordo entre os próprios "
                 "especialistas. O viés médio de cada par (que se anula quando as "
                 "divergências são simétricas, e por isso não serve de margem) é "
                 "reportado ao lado.\n")
        L.append(_md(calibracao["tabela"], indice=False))
        L.append("")
        L.append(f"Divergência absoluta média: notas = "
                 f"{_num(calibracao['bruto_notas'], 4)} ponto; decisões discordantes = "
                 f"{_num(calibracao['bruto_taxa'], 4)} "
                 f"({calibracao['n_pares']} par(es), {_num(calibracao['n_itens'], 0)} itens).\n")
        for chave, minimo, rotulo in (("notas", BAYES_ROPE_MINIMO, "notas"),
                                      ("decisao", BAYES_ROPE_DECISAO_MINIMO,
                                       "taxa de adequação")):
            if calibracao["piso_aplicado"][chave]:
                L.append(f"⚠️ A divergência calibrada da {rotulo} ficou abaixo do piso "
                         f"numérico de {_num(minimo, 2)} e o piso foi aplicado — o "
                         "baycomp exige ROPE > 0, e especialistas em acordo quase "
                         "perfeito degenerariam a margem a zero.\n")
        L.append("**Transcreva o valor calibrado à mão para o YAML da Etapa 2** "
                 "(`estatistica.rope_likert`). O fluxo é manual de "
                 "propósito: transcrever força o pré-registro consciente.\n")
    else:
        L.append("Sem grupo humano de referência com 2 ou mais especialistas, a "
                 "calibração não é possível; o valor usado vem da flag ou do padrão "
                 "de escala, e deve ser lido como margem **não calibrada**.\n")

    controle = v.get("controle")
    if controle is not None:
        L.append("### Controle negativo da Likert\n")
        L.append("Os próprios especialistas, comparados entre si com a ROPE calibrada e "
                 f"o limiar de {_num(v['limiar'], 2)}, devem sair **equivalentes**. Se não "
                 "saírem, a margem está apertada demais — e é melhor descobrir antes de "
                 "julgar o juiz. É o análogo do controle negativo do F1 e fecha a "
                 "pergunta sobre a margem ter sido escolhida convenientemente.\n")
        L.append(_md(controle, indice=False))
        L.append("")
        reprovados = int((controle["Passa"] == "NÃO").sum())
        if reprovados:
            L.append(f"⚠️ **{reprovados} de {len(controle)} par(es) não passou.** "
                     "Interprete os vereditos do gate com cautela: com esta margem, "
                     "nem os especialistas seriam considerados equivalentes entre si.\n")
        else:
            L.append("✅ Todos os pares de especialistas saem equivalentes — a margem "
                     "calibrada não é apertada demais.\n")
    return L


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
        "Dif. média": round(g["media_dif"], 3),
        "P(equiv. notas)": _num(g["p_equiv_notas"]),
        "P(equiv. decisão)": _num(g["p_equiv_decisao"]),
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
    L.append("| # | Critério | Instrumento | Aprova se |")
    L.append("|---|---|---|---|")
    L.append(f"| 1 | Concordância ordinal com o humano | κw de Cohen ponderado (quadrático) | "
             f"κw ≥ {_num(LIMIAR_KAPPA, 2)} |")
    L.append(f"| 2 | Ausência de viés | P(equivalência) na diferença média das notas, "
             f"ROPE = {_num(v['rope'], 4)} | ≥ {_num(v['limiar'], 2)} |")
    L.append(f"| 3 | Decisão prática | P(equivalência) na taxa de adequação "
             f"(nota ≥ {PISO_ADEQUACAO}), ROPE = {_num(v['rope_decisao'], 4)} | "
             f"≥ {_num(v['limiar'], 2)} |")
    L.append("")
    L.append(f"O critério 1 é aferido no **valor pontual** do κw. O IC 95% (bootstrap de "
             f"{BOOTSTRAP_REPLICAS} réplicas de documentos) é reportado como medida de "
             "precisão, não como critério: exigir que o limite inferior alcançasse o limiar "
             "reprovaria o juiz por tamanho de amostra, não por falta de concordância.\n")
    L.append(f"Os critérios 2 e 3 são **bayesianos** (`{_NOME_TESTE_BAYES}`, Benavoli et "
             "al., 2017): a probabilidade posterior de a diferença **média** caber na ROPE. "
             "Diferentemente de não rejeitar H₀, uma posterior concentrada dentro da ROPE é "
             "evidência **a favor** da equivalência. O veredito julga **magnitude**, nunca "
             "direção: um juiz pode ser confiavelmente mais leniente por uma margem sem "
             f"relevância prática. O cálculo é analítico (Student acumulada) — "
             "determinístico, sem amostras nem semente. O limiar de "
             f"{_num(v['limiar'], 2)} é **único** em todo o trabalho (gate, heatmap e "
             "forest plot; Benavoli et al., 2017, §3.2).\n")
    L.append(f"### {rotulo_ressalva.capitalize()}\n")
    L.append("Status intermediário: o κw atende ao corte e a equivalência (critério 2 e/ou "
             "3) ficou **inconclusiva** — os dados não bastam para afirmá-la nem para "
             "afirmar viés relevante. É desfecho legítimo; a resposta honesta é ampliar a "
             f"amostra. O `{rotulo_nao}` exige κw abaixo do corte **ou viés relevante** "
             "(a diferença quase certamente excede a ROPE em alguma direção).\n")
    L.extend(_bloco_calibracao_rope(v))

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
             "1 ponto; a coluna `Exata` é a leitura sem ponderação). As colunas "
             "`Acima`/`Empate`/`Abaixo` são **contagens ordinais puras** (d > 0, d = 0, "
             "d < 0) — independentes da ROPE, o número que não assume nada sobre os "
             "intervalos da escala. A magnitude é lida em `Média dif.` com o `IC 95%` da "
             "posterior, confrontada com a ROPE; `P(equiv. notas)` é o critério 2 do gate.\n")
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
        L.append("Seção **contextual**: não integra o gate. O critério 2 é bayesiano "
                 "(P(equivalência) da diferença média contra a ROPE calibrada) e vive na "
                 "tabela de veredito; o Friedman e os contrastes acima descrevem a "
                 "severidade relativa entre todos os avaliadores.\n")

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
             f"- A conclusão bayesiana depende do limiar único de {_num(v['limiar'], 2)} e "
             f"da ROPE de {_num(v['rope'], 4)} ponto — ambos fixados **antes** da análise "
             f"({v['origem_rope']}). O controle negativo existe para mostrar que a margem "
             "não foi escolhida convenientemente.\n"
             "- **A diferença média de notas Likert é a diferença das médias** — o "
             "pareamento não contorna a objeção ordinal. A defesa tem três partes: a "
             "literatura de robustez (Norman, 2010; Carifio & Perla, 2008); a ROPE "
             "ancorada na divergência entre especialistas, na mesma unidade, o que torna "
             "a interpretação relativa; e as contagens ordinais puras reportadas ao lado "
             "de toda probabilidade.\n"
             "- A análise é **independente por conjunto de dados**: não modela a "
             "variabilidade compartilhada entre datasets nem atualiza sequencialmente o "
             "conhecimento. O `baycomp.HierarchicalTest` faria isso, mas exige `pystan`, "
             "pressupõe validação cruzada dentro de cada conjunto e, com poucos conjuntos, "
             "a variância entre grupos não é identificável.\n"
             "- No agregado, cada documento entra tantas vezes quantas forem as fontes. É "
             "coerente com a unidade de julgamento (a extração avaliada, não o documento), e o "
             "bootstrap reamostra documentos inteiros para preservar essa dependência.\n")

    L.append("---\n\n## Figuras\n")
    for arquivo in v["figuras"]:
        L.append(f"- `{arquivo}`")
    L.append("")
    L.append(_bloco_reprodutibilidade(v))

    texto = "\n".join(L)
    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write(texto)
    return texto


def _bloco_bayes_validacao(v: dict) -> list:
    """Seção do panorama bayesiano do `validacao.md` — vazia sem `--bayes`."""
    b = v.get("bayes")
    if not b:
        return []
    cfg = b["config"]
    L = ["---\n\n## Panorama bayesiano entre os avaliadores (complementar)\n"]
    L.append("Os critérios 2 e 3 do gate já são bayesianos e vivem na tabela de "
             "veredito. Esta seção acrescenta o **panorama** entre todos os "
             "avaliadores — inclusive os pares juiz × juiz, que ficam fora do gate — "
             "na mesma ROPE e no MESMO limiar único.\n")
    L.append("| Parâmetro | Valor |")
    L.append("|---|---|")
    L.append(f"| Teste | `{_NOME_TESTE_BAYES}` (analítico — sem amostras nem semente) |")
    L.append("| Variável | nota Likert mediana por documento |")
    L.append(f"| ROPE | {_num(v['rope'], 4)} ({v['origem_rope']}) |")
    L.append(f"| Limiar único | {_num(cfg.limiar, 2)} |")
    L.append("")
    L.extend(_bloco_como_ler_bayes(v, _pares_info_de_tabela(b["tabela_matriz"])
                                   if b.get("tabela_matriz") is not None else [],
                                   rotulo_entidade="avaliador"))
    if b.get("tabela_matriz") is not None:
        L.append("### Relações entre todos os avaliadores\n")
        L.append(_md(b["tabela_matriz"], indice=False))
        L.append("")
        L.append("### Contando as relações\n")
        L.append(_md(b["sintese"]))
        L.append("")
        ciclos = b["sintese"].attrs.get("ciclos") or []
        if ciclos:
            L.append(f"⚠️ **Transitividade violada** — ciclo(s) nas relações "
                     f"direcionais: {'; '.join(' > '.join(map(str, c)) for c in ciclos)}. "
                     "A leitura ordenada desta tabela é inválida.\n")
        else:
            L.append("✅ Verificação de transitividade: as relações direcionais não "
                     "formam ciclo.\n")
        L.append("A síntese conta relações, **não ordena**. Os pares juiz × juiz "
                 "permanecem **fora do gate**: convergência entre modelos não é "
                 "evidência de validade.\n")
    return L


def _paragrafo_veredito(g: dict, referencia: str, rotulo: str) -> str:
    """Frase de veredito de um juiz, com os critérios que falharam, se houver."""
    criterios = g["criterios"]
    if g["status"] == "VALIDADO":
        return (f"✅ **`{g['juiz']}`: {rotulo}** — κw = {_num(g['kappa'])} "
                f"(concordância no nível *{g['interpretacao']}*); equivalência prática "
                f"estabelecida nas notas (P = {_num(g['p_equiv_notas'])}, "
                f"ROPE = {_num(g['rope'], 4)}) e na decisão adequado/inadequado "
                f"(P = {_num(g['p_equiv_decisao'])}), ao limiar de "
                f"{_num(g['limiar'], 2)}; acurácia de {_pct(g['acuracia'])} contra "
                f"`{referencia}`. Contagens ordinais: {g['acima']} acima, "
                f"{g['empate']} empates, {g['abaixo']} abaixo.")

    if g["status"] == "VALIDADO COM RESSALVA":
        pendentes = []
        if not criterios["sem_vies"]:
            pendentes.append(f"notas (P(equiv.) = {_num(g['p_equiv_notas'])})")
        if not criterios["decisao"]:
            pendentes.append(f"decisão (P(equiv.) = {_num(g['p_equiv_decisao'])})")
        return (f"🟡 **`{g['juiz']}`: {rotulo}** — κw = {_num(g['kappa'])} "
                f"(*{g['interpretacao']}*) atende ao corte, mas a equivalência ficou "
                f"**inconclusiva** em {', '.join(pendentes)} — abaixo do limiar de "
                f"{_num(g['limiar'], 2)} sem que o viés relevante esteja estabelecido "
                f"(dif. média = {_num(g['media_dif'], 3)}, IC 95% "
                f"[{_num(g['ic_notas'][0], 3)}; {_num(g['ic_notas'][1], 3)}]). "
                "Desfecho legítimo; a resposta honesta é ampliar a amostra.")

    falhas = []
    if not criterios["concordancia"]:
        falhas.append(f"κw = {_num(g['kappa'])} < {_num(LIMIAR_KAPPA, 2)} "
                      f"(concordância no nível *{g['interpretacao']}*, "
                      f"P_o = {_num(g['p_o'])})")
    if g["vies_relevante"]:
        falhas.append(
            f"viés relevante — a diferença quase certamente excede a ROPE "
            f"(dif. média = {_num(g['media_dif'], 3)}, {g['direcao']}; "
            f"P(acima da ROPE) = {_num(g['p_juiz_maior'])}, "
            f"P(abaixo da ROPE) = {_num(g['p_juiz_menor'])}; "
            f"{g['b01']} itens aprovados só pelo juiz contra {g['b10']} só pela "
            "referência)")
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
        "Acima": g["acima"], "Empate": g["empate"], "Abaixo": g["abaixo"],
        "Média dif.": round(g["media_dif"], 4),
        "IC 95%": f"[{g['ic_notas'][0]:+.4f}; {g['ic_notas'][1]:+.4f}]".replace(".", ","),
        "P(equiv. notas)": round(g["p_equiv_notas"], 4),
        "Direção": g["direcao"],
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
            "Dif. taxa": round(g["dif_taxa"], 4),
            "P(equiv. decisão)": round(g["p_equiv_decisao"], 4),
            "Acurácia": _pct(g["acuracia"]),
            "Sensib.": _pct(g["sensibilidade"]),
            "Especif.": _pct(g["especificidade"]),
        }
        if not lateral:
            registro["Status"] = g["status"]
        linhas.append(registro)
    return pd.DataFrame(linhas)


def _bloco_reprodutibilidade(contexto: dict = None) -> str:
    """Rodapé com versões e convenções usadas.

    Args:
        contexto: dict com `rope`, `rope_decisao`, `limiar` e `origem_rope`
            quando a camada bayesiana foi parametrizada (validação); None na
            análise interna sem esses dados.
    """
    sk = (__import__("sklearn").__version__ if SKLEARN_DISPONIVEL
          else "— (fórmula interna)")
    sm = (__import__("statsmodels").__version__ if STATSMODELS_DISPONIVEL
          else "— (fórmula interna)")
    extra = []
    if contexto and contexto.get("rope") is not None:
        limiar = contexto.get("limiar")
        if limiar is None and contexto.get("config") is not None:
            limiar = contexto["config"].limiar
        rope_decisao = (f", ROPE decisão {_num(contexto['rope_decisao'], 4)}"
                        if contexto.get("rope_decisao") is not None else "")
        extra = [f"| Posterior bayesiana | `{_NOME_TESTE_BAYES}` (analítica), "
                 f"ROPE notas {_num(contexto['rope'], 4)}{rope_decisao}, "
                 f"limiar único {_num(limiar if limiar is not None else BAYES_LIMIAR, 2)} "
                 f"({contexto.get('origem_rope', '—')}) |"]
    return "\n".join([
        "---\n", "## Reprodutibilidade\n",
        "| Item | Valor |", "|---|---|",
        f"| Python | {sys.version.split()[0]} |",
        f"| pandas / numpy / scipy | {pd.__version__} / {np.__version__} "
        f"/ {__import__('scipy').__version__} |",
        f"| scikit-learn / statsmodels | {sk} / {sm} |",
        "| Pesos do Kappa | quadráticos |",
        "| Faixas de interpretação | McHugh (2012) |",
        "| Correção múltipla | Holm-Bonferroni |",
        f"| Bootstrap | {BOOTSTRAP_REPLICAS} réplicas de documentos, semente {SEMENTE} |",
    ] + extra + [
        "",
        "### Origem das estatísticas\n",
        "| Estatística | Implementação |",
        "|---|---|",
    ] + [f"| {nome} | {origem} |"
         for nome, origem in dependencias(bool(contexto)).items()] + [
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

    Os critérios 2 e 3 do gate são bayesianos e rodam **sempre** na validação;
    por isso ``--bayes-rope``, ``--bayes-rope-decisao`` e ``--bayes-limiar``
    têm efeito mesmo sem ``--bayes``. A flag ``--bayes`` liga apenas a camada
    complementar (heatmaps e forest plots entre fontes e entre avaliadores).
    """
    if args.bayes and not BAYES_DISPONIVEL:
        erro("--bayes exige o módulo `util_est_bayesiana.py` (e o pacote `baycomp`) "
             "na mesma pasta ou no PYTHONPATH — assim como o gate da validação, "
             "cujos critérios 2 e 3 são bayesianos.")

    # o baycomp devolve só (p_esquerda, p_direita) com rope = 0: a tripla exige rope > 0
    for flag, valor in (("--bayes-rope", args.bayes_rope),
                        ("--bayes-rope-decisao", args.bayes_rope_decisao)):
        if valor is not None and valor <= 0:
            erro(f"{flag} = {valor} inválido: com ROPE zero o baycomp não devolve "
                 "a probabilidade de equivalência. Omita a flag para usar a ROPE "
                 "calibrada pela divergência entre os especialistas.")
    if not 0.5 <= args.bayes_limiar <= 1.0:
        erro(f"--bayes-limiar = {args.bayes_limiar} fora da faixa admissível "
             "[0,5; 1,0].")
    if args.bayes_limiar != BAYES_LIMIAR:
        print(f"ℹ --bayes-limiar = {args.bayes_limiar}: o limiar deixa de ser o "
              f"único de {BAYES_LIMIAR} usado no restante do trabalho — registre a "
              "escolha e o motivo.")
    if args.bayes_rope is not None:
        print(f"ℹ --bayes-rope = {args.bayes_rope}: a ROPE informada sobrescreve a "
              "calibração pela divergência entre especialistas e fica registrada "
              "como pré-registro manual.")

    return ConfigBayes(
        ativo=bool(args.bayes), rope=args.bayes_rope,
        rope_decisao=args.bayes_rope_decisao, limiar=args.bayes_limiar)


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
               "Camada bayesiana (baycomp.CorrelatedTTest):\n"
               "  Os critérios 2 e 3 do gate são bayesianos e rodam sempre na\n"
               "  validação, com a ROPE calibrada pela divergência média entre os\n"
               "  especialistas humanos e o limiar único de 0,95. `--bayes` liga o\n"
               "  panorama complementar (heatmaps e forest plots).\n")
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
        "camada bayesiana (baycomp.CorrelatedTTest — analítico, determinístico)",
        "Os critérios 2 e 3 do gate são bayesianos e rodam SEMPRE na validação, "
        "com a ROPE calibrada pela divergência entre os especialistas humanos. "
        "As flags de ROPE e limiar afetam o gate; `--bayes` liga apenas o "
        "panorama complementar (heatmaps e forest plots).")
    bayesiano.add_argument("--bayes", action="store_true",
                           help="gera o panorama complementar entre fontes e entre "
                                "avaliadores (heatmap + 'Medindo as diferenças')")
    bayesiano.add_argument("--bayes-rope", type=float, default=None, metavar="R",
                           help="sobrescreve a ROPE das notas (padrão: calibrada pela "
                                "divergência média entre os especialistas da referência; "
                                f"fallback {BAYES_ROPE_PADRAO}). Deve ser > 0; fica "
                                "registrada como pré-registro manual")
    bayesiano.add_argument("--bayes-rope-decisao", type=float, default=None,
                           metavar="R",
                           help="sobrescreve a ROPE da taxa de adequação (padrão: "
                                "calibrada pelos especialistas; fallback "
                                f"{BAYES_ROPE_DECISAO_PADRAO}). Deve ser > 0")
    bayesiano.add_argument("--bayes-limiar", type=float, default=BAYES_LIMIAR,
                           metavar="P",
                           help="limiar ÚNICO de decisão — gate, heatmap e forest plot "
                                f"(padrão: {BAYES_LIMIAR}, Benavoli et al. 2017 §3.2)")
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