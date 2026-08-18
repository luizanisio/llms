#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realizar_avaliacoes_teste.py
============================

Verifica que as **definições operacionais** (fórmulas internas) das
estatísticas descritas no capítulo de método coincidem com as implementações
dos pacotes consolidados (``scikit-learn``, ``statsmodels``, ``scipy``) usados
pelo pipeline ``realizar_avaliacoes.py``.

O pipeline usa **sempre** as bibliotecas; as fórmulas abaixo existem como
referência documentável e verificada, sustentando no texto da dissertação
que os procedimentos são replicáveis e que os cálculos estão corretos.

Estatísticas cobertas:

===============================  ===================================================
estatística                      referência de comparação
===============================  ===================================================
Kappa de Cohen ponderado         ``sklearn.metrics.cohen_kappa_score``
Kappa de Fleiss (não ponderado)  ``statsmodels.stats.inter_rater.fleiss_kappa``
Kappa de Fleiss ponderado        casos analíticos (sem equivalente em pacote)
Correção de Holm                 ``statsmodels.stats.multitest.multipletests``
Teste de McNemar                 ``statsmodels.stats.contingency_tables.mcnemar``
IC de Wilson                     ``statsmodels.stats.proportion.proportion_confint``
Wilcoxon, Friedman, Shapiro      ``scipy.stats`` (usados diretamente)
Bootstrap de documentos          propriedades estruturais e reprodutibilidade
===============================  ===================================================

Uso::

    python realizar_avaliacoes_teste.py           # resumo legível
    python -m unittest realizar_avaliacoes_teste  # saída padrão do unittest

Autor: Luiz Anísio
"""

from __future__ import annotations

import unittest
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import realizar_avaliacoes as ra

TOLERANCIA = 1e-9
CATEGORIAS = [1, 2, 3, 4]

from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


# =============================================================================
# Implementações internas de referência (as fórmulas do arcabouço)
# =============================================================================
# Reproduzidas aqui de forma isolada e legível. São a definição operacional das
# estatísticas descritas no capítulo de método; os testes verificam que os
# pacotes usados em produção calculam exatamente isto.

def pesos_quadraticos(k: int) -> np.ndarray:
    """w[i,j] = 1 − ((i − j) / (k − 1))²  — 1 na diagonal, decrescente fora dela."""
    if k == 1:
        return np.ones((1, 1))
    i, j = np.indices((k, k))
    return 1.0 - ((i - j) / (k - 1)) ** 2


def cohen_formula(a, b, categorias) -> float:
    """κw = (P_o − P_e) / (1 − P_e), sobre a matriz de confusão normalizada."""
    k = len(categorias)
    indice = {c: i for i, c in enumerate(categorias)}
    observado = np.zeros((k, k))
    for x, y in zip(a, b):
        observado[indice[x], indice[y]] += 1
    observado /= observado.sum()
    esperado = np.outer(observado.sum(axis=1), observado.sum(axis=0))
    w = pesos_quadraticos(k)
    p_o, p_e = float((w * observado).sum()), float((w * esperado).sum())
    return np.nan if np.isclose(p_e, 1.0) else (p_o - p_e) / (1.0 - p_e)


def fleiss_formula(notas, categorias, pesos=None) -> float:
    """κw de Fleiss para m avaliadores, com pesos arbitrários (padrão: quadráticos)."""
    notas = np.asarray(notas)
    n, m = notas.shape
    q = len(categorias)
    indice = {c: i for i, c in enumerate(categorias)}
    contagens = np.zeros((n, q))
    for i in range(n):
        for valor in notas[i]:
            contagens[i, indice[valor]] += 1
    w = pesos_quadraticos(q) if pesos is None else pesos
    p_o = float((np.einsum("ia,ab,ib->i", contagens, w, contagens) - m).sum()
                / (n * m * (m - 1)))
    prop = contagens.sum(axis=0) / (n * m)
    p_e = float(prop @ w @ prop)
    return np.nan if np.isclose(p_e, 1.0) else (p_o - p_e) / (1.0 - p_e)


def holm_formula(p_valores) -> list:
    """Holm-Bonferroni step-down: p_(i) ajustado = max acumulado de (m − i) · p_(i)."""
    m = len(p_valores)
    ordem = np.argsort(p_valores)
    ajustados = np.empty(m)
    corrente = 0.0
    for posicao, indice in enumerate(ordem):
        corrente = max(corrente, (m - posicao) * p_valores[indice])
        ajustados[indice] = min(1.0, corrente)
    return ajustados.tolist()


def mcnemar_formula(b01: int, b10: int) -> tuple:
    """Binomial exato com < 25 discordantes; qui-quadrado com correção acima."""
    discordantes = b01 + b10
    if discordantes == 0:
        return 1.0, "sem discordâncias"
    if discordantes < 25:
        return float(stats.binomtest(b01, discordantes, 0.5).pvalue), "binomial exato"
    estat = (abs(b01 - b10) - 1) ** 2 / discordantes
    return float(stats.chi2.sf(estat, 1)), "qui-quadrado com correção"


def wilson_formula(sucessos: int, total: int, confianca: float = 0.95) -> tuple:
    """IC de Wilson: centro e margem deslocados por z²/2n, sem aproximação normal."""
    z = stats.norm.ppf(1 - (1 - confianca) / 2)
    p = sucessos / total
    denom = 1 + z ** 2 / total
    centro = (p + z ** 2 / (2 * total)) / denom
    margem = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return (max(0.0, centro - margem), min(1.0, centro + margem))


# =============================================================================
# Dados sintéticos
# =============================================================================

def gerar_pareado(n: int = 200, semente: int = 11, ruido: float = 0.8) -> tuple:
    """Duas séries Likert pareadas, com correlação moderada e viés leve."""
    rng = np.random.default_rng(semente)
    base = rng.normal(2.8, 0.9, n)
    a = np.clip(np.round(base + rng.normal(0.15, ruido, n)), 1, 4).astype(int)
    b = np.clip(np.round(base + rng.normal(0.0, ruido, n)), 1, 4).astype(int)
    return a, b


def gerar_matriz(n: int = 150, m: int = 3, semente: int = 23) -> np.ndarray:
    """Matriz n × m de notas Likert com concordância realista entre avaliadores."""
    rng = np.random.default_rng(semente)
    base = rng.normal(2.9, 0.85, n)
    return np.clip(np.round(base[:, None] + rng.normal(0, 0.7, (n, m))),
                   1, 4).astype(int)


# =============================================================================
# Testes
# =============================================================================

class TestePesos(unittest.TestCase):
    """A matriz de pesos define o que 'concordância parcial' significa."""

    def test_diagonal_unitaria(self):
        w = pesos_quadraticos(4)
        np.testing.assert_allclose(np.diag(w), 1.0)

    def test_extremos_sem_credito(self):
        """Notas 1 e 4 (extremos opostos) não recebem crédito parcial."""
        w = pesos_quadraticos(4)
        self.assertAlmostEqual(w[0, 3], 0.0)

    def test_penalizacao_progressiva(self):
        """1 ponto de distância penaliza menos que 2, que penaliza menos que 3."""
        w = pesos_quadraticos(4)
        self.assertGreater(w[0, 1], w[0, 2])
        self.assertGreater(w[0, 2], w[0, 3])

    def test_simetrica(self):
        w = pesos_quadraticos(4)
        np.testing.assert_allclose(w, w.T)

    def test_identica_a_implementacao_do_pipeline(self):
        np.testing.assert_allclose(pesos_quadraticos(4), ra._matriz_pesos(4))


class TesteCohen(unittest.TestCase):
    """Kappa de Cohen ponderado — critério 1 do gate de validação."""

    def setUp(self):
        self.a, self.b = gerar_pareado()

    def test_concordancia_perfeita(self):
        self.assertAlmostEqual(
            ra.cohen_kappa(self.a, self.a, CATEGORIAS), 1.0, places=9)

    def test_pipeline_igual_a_formula(self):
        self.assertAlmostEqual(ra.cohen_kappa(self.a, self.b, CATEGORIAS),
                               cohen_formula(self.a, self.b, CATEGORIAS), places=9)

    def test_formula_igual_ao_sklearn(self):
        self.assertAlmostEqual(
            cohen_formula(self.a, self.b, CATEGORIAS),
            cohen_kappa_score(self.a, self.b, labels=CATEGORIAS, weights="quadratic"),
            places=9)

    def test_binario_igual_ao_sklearn(self):
        """Caso usado na Camada 4: binarização adequado/inadequado."""
        bin_a = (self.a >= ra.PISO_ADEQUACAO).astype(int)
        bin_b = (self.b >= ra.PISO_ADEQUACAO).astype(int)
        self.assertAlmostEqual(
            ra.cohen_kappa(bin_a, bin_b, [0, 1]),
            cohen_kappa_score(bin_a, bin_b, labels=[0, 1], weights="quadratic"),
            places=9)

    def test_multiplas_sementes(self):
        for semente in range(5):
            a, b = gerar_pareado(semente=semente)
            with self.subTest(semente=semente):
                self.assertAlmostEqual(
                    ra.cohen_kappa(a, b, CATEGORIAS),
                    cohen_kappa_score(a, b, labels=CATEGORIAS, weights="quadratic"),
                    places=9)

    def test_ponderado_maior_que_nao_ponderado(self):
        """Com dados ordinais, dar crédito parcial não pode reduzir o κ."""
        ponderado = ra.cohen_kappa(self.a, self.b, CATEGORIAS)
        k = len(CATEGORIAS)
        indice = {c: i for i, c in enumerate(CATEGORIAS)}
        obs = np.zeros((k, k))
        for x, y in zip(self.a, self.b):
            obs[indice[x], indice[y]] += 1
        obs /= obs.sum()
        esp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
        p_o, p_e = np.trace(obs), np.trace(esp)
        simples = (p_o - p_e) / (1 - p_e)
        self.assertGreater(ponderado, simples)

    def test_p_o_e_p_e_no_intervalo(self):
        resultado = ra.cohen_ponderado(self.a, self.b, CATEGORIAS)
        for chave in ("p_o", "p_e"):
            self.assertGreaterEqual(resultado[chave], 0.0)
            self.assertLessEqual(resultado[chave], 1.0)


class TesteFleiss(unittest.TestCase):
    """Kappa de Fleiss ponderado — confiabilidade interna dos grupos."""

    def setUp(self):
        self.matriz = gerar_matriz()

    def test_pipeline_igual_a_formula(self):
        self.assertAlmostEqual(
            ra.fleiss_ponderado(self.matriz, CATEGORIAS)["kappa"],
            fleiss_formula(self.matriz, CATEGORIAS), places=9)

    def test_concordancia_perfeita(self):
        matriz = np.repeat(self.matriz[:, [0]], 3, axis=1)
        self.assertAlmostEqual(
            ra.fleiss_ponderado(matriz, CATEGORIAS)["kappa"], 1.0, places=9)

    def test_sem_pesos_igual_ao_statsmodels(self):
        """Com pesos identidade, o κw deve reduzir-se ao Fleiss clássico.

        É a única comparação possível com pacote: não há implementação
        consolidada do Fleiss **ponderado** em Python.
        """
        identidade = np.eye(len(CATEGORIAS))
        nosso = fleiss_formula(self.matriz, CATEGORIAS, pesos=identidade)
        tabela, _ = aggregate_raters(self.matriz)
        self.assertAlmostEqual(nosso, fleiss_kappa(tabela, method="fleiss"), places=9)

    def test_dois_avaliadores_igual_a_cohen(self):
        """Com m = 2, Fleiss e Cohen medem a mesma coisa e ficam próximos."""
        a, b = self.matriz[:, 0], self.matriz[:, 1]
        fleiss = ra.fleiss_ponderado(np.column_stack([a, b]), CATEGORIAS)["kappa"]
        cohen = ra.cohen_kappa(a, b, CATEGORIAS)
        self.assertAlmostEqual(fleiss, cohen, delta=0.05)

    def test_componentes_coerentes(self):
        r = ra.fleiss_ponderado(self.matriz, CATEGORIAS)
        self.assertAlmostEqual(r["kappa"], (r["p_o"] - r["p_e"]) / (1 - r["p_e"]),
                               places=9)


class TesteHolm(unittest.TestCase):
    """Correção de Holm-Bonferroni nos contrastes post-hoc."""

    def setUp(self):
        self.p = [0.001, 0.013, 0.021, 0.045, 0.6, 0.98]

    def test_pipeline_igual_a_formula(self):
        np.testing.assert_allclose(ra.correcao_holm(self.p), holm_formula(self.p),
                                   atol=TOLERANCIA)

    def test_formula_igual_ao_statsmodels(self):
        np.testing.assert_allclose(holm_formula(self.p),
                                   multipletests(self.p, method="holm")[1],
                                   atol=TOLERANCIA)

    def test_menor_p_multiplicado_por_m(self):
        """O menor p-valor é multiplicado pelo número de testes."""
        self.assertAlmostEqual(ra.correcao_holm(self.p)[0], 0.001 * len(self.p))

    def test_monotonicidade(self):
        """A correção step-down não pode inverter a ordem dos p-valores."""
        ajustados = np.array(ra.correcao_holm(self.p))[np.argsort(self.p)]
        self.assertTrue(np.all(np.diff(ajustados) >= -TOLERANCIA))

    def test_nunca_reduz_p(self):
        self.assertTrue(all(aj >= bruto - TOLERANCIA
                            for aj, bruto in zip(ra.correcao_holm(self.p), self.p)))

    def test_nan_tratado_como_um(self):
        self.assertEqual(ra.correcao_holm([np.nan])[0], 1.0)

    def test_lista_vazia(self):
        self.assertEqual(ra.correcao_holm([]), [])


class TesteMcNemar(unittest.TestCase):
    """McNemar — critério 3 do gate (viés direcional na decisão binária)."""

    def test_pipeline_igual_a_formula_exato(self):
        a = np.array([True] * 30 + [False] * 20)
        b = np.array([True] * 22 + [False] * 8 + [True] * 3 + [False] * 17)
        resultado = ra.mcnemar(a, b)
        p, metodo = mcnemar_formula(resultado["b01"], resultado["b10"])
        self.assertAlmostEqual(resultado["p"], p, places=9)
        self.assertEqual(resultado["metodo"], metodo)

    def test_formula_igual_ao_statsmodels(self):
        for b01, b10 in [(8, 2), (12, 5), (30, 14), (46, 1), (23, 5)]:
            with self.subTest(b01=b01, b10=b10):
                p_nosso, _ = mcnemar_formula(b01, b10)
                exato = (b01 + b10) < 25
                tabela = [[40, b01], [b10, 40]]
                p_lib = sm_mcnemar(tabela, exact=exato, correction=not exato).pvalue
                self.assertAlmostEqual(p_nosso, float(p_lib), places=9)

    def test_sem_discordancia(self):
        a = np.array([True, False, True, False])
        self.assertEqual(ra.mcnemar(a, a)["p"], 1.0)

    def test_troca_de_metodo_em_25(self):
        self.assertEqual(mcnemar_formula(12, 12)[1], "binomial exato")
        self.assertEqual(mcnemar_formula(13, 12)[1], "qui-quadrado com correção")

    def test_simetria_maxima_nao_rejeita(self):
        """Discordâncias equilibradas: p = 1 (nenhuma direção preferencial)."""
        self.assertAlmostEqual(mcnemar_formula(10, 10)[0], 1.0, places=9)

    def test_assimetria_extrema_rejeita(self):
        self.assertLess(mcnemar_formula(46, 1)[0], 0.001)

    def test_contagem_direcional(self):
        """b01 = juiz aprova e referência reprova; b10 = o inverso."""
        juiz = np.array([True, True, True, False])
        referencia = np.array([False, False, True, True])
        resultado = ra.mcnemar(juiz, referencia)
        self.assertEqual(resultado["b01"], 2)
        self.assertEqual(resultado["b10"], 1)


class TesteWilson(unittest.TestCase):
    """IC de Wilson — usado na análise descritiva de viabilidade."""

    def test_pipeline_igual_a_formula(self):
        for sucessos, total in [(45, 50), (150, 210), (1, 20), (19, 20)]:
            with self.subTest(sucessos=sucessos, total=total):
                _, li, ls = ra.ic_wilson(sucessos, total)
                esperado = wilson_formula(sucessos, total)
                self.assertAlmostEqual(li, esperado[0], places=9)
                self.assertAlmostEqual(ls, esperado[1], places=9)

    def test_formula_igual_ao_statsmodels(self):
        for sucessos, total in [(45, 50), (150, 210), (1, 20), (0, 30), (30, 30)]:
            with self.subTest(sucessos=sucessos, total=total):
                nosso = wilson_formula(sucessos, total)
                lib = proportion_confint(sucessos, total, alpha=0.05, method="wilson")
                np.testing.assert_allclose(nosso, lib, atol=TOLERANCIA)

    def test_dentro_de_zero_um(self):
        """Vantagem do Wilson sobre a aproximação normal: não extrapola [0, 1]."""
        for sucessos, total in [(0, 10), (10, 10), (1, 3)]:
            _, li, ls = ra.ic_wilson(sucessos, total)
            self.assertGreaterEqual(li, 0.0)
            self.assertLessEqual(ls, 1.0)

    def test_contem_a_proporcao(self):
        p, li, ls = ra.ic_wilson(150, 210)
        self.assertLessEqual(li, p)
        self.assertGreaterEqual(ls, p)

    def test_encolhe_com_n(self):
        _, li_p, ls_p = ra.ic_wilson(40, 50)
        _, li_g, ls_g = ra.ic_wilson(400, 500)
        self.assertLess(ls_g - li_g, ls_p - li_p)


class TesteWilcoxon(unittest.TestCase):
    """Wilcoxon pareado — critério 2 do gate."""

    def setUp(self):
        self.a, self.b = gerar_pareado()

    def test_igual_ao_scipy(self):
        resultado = ra.wilcoxon_pareado(self.a, self.b)
        esperado = stats.wilcoxon(self.a, self.b, alternative="two-sided",
                                  zero_method="wilcox", method="approx",
                                  correction=False)
        self.assertAlmostEqual(resultado["p"], float(esperado.pvalue), places=9)

    def test_n_efetivo_exclui_empates(self):
        resultado = ra.wilcoxon_pareado(self.a, self.b)
        self.assertEqual(resultado["n_efetivo"], int(np.sum(self.a != self.b)))

    def test_efeito_sobre_n_efetivo(self):
        """r = |z| / √n′ — o denominador é o número de discordâncias, não n."""
        resultado = ra.wilcoxon_pareado(self.a, self.b)
        self.assertAlmostEqual(
            resultado["r"], abs(resultado["z"]) / np.sqrt(resultado["n_efetivo"]),
            places=9)

    def test_series_identicas(self):
        resultado = ra.wilcoxon_pareado(self.a, self.a)
        self.assertEqual(resultado["p"], 1.0)
        self.assertEqual(resultado["n_efetivo"], 0)

    def test_faixas_de_efeito(self):
        self.assertEqual(ra.interpretar_efeito(0.05), "desprezível")
        self.assertEqual(ra.interpretar_efeito(0.20), "pequeno")
        self.assertEqual(ra.interpretar_efeito(0.40), "médio")
        self.assertEqual(ra.interpretar_efeito(0.70), "grande")


class TesteBootstrap(unittest.TestCase):
    """IC do κw por reamostragem de documentos (não de itens)."""

    def setUp(self):
        import pandas as pd
        rng = np.random.default_rng(5)
        n_docs, fontes = 60, ["f1", "f2", "f3"]
        linhas = []
        for doc in range(n_docs):
            base = rng.normal(2.9, 0.9)
            for fonte in fontes:
                a = int(np.clip(round(base + rng.normal(0.2, 0.6)), 1, 4))
                b = int(np.clip(round(base + rng.normal(0.0, 0.6)), 1, 4))
                linhas.append({"documento": f"doc{doc:03d}", "fonte": fonte,
                               "a": a, "b": b})
        self.dados = pd.DataFrame(linhas)
        self.estat = lambda d: ra.cohen_kappa(d["a"], d["b"], CATEGORIAS)

    def test_contem_a_estimativa_pontual(self):
        pontual = self.estat(self.dados)
        inf, sup = ra.ic_bootstrap(self.dados, self.estat, replicas=400)
        self.assertLessEqual(inf, pontual)
        self.assertGreaterEqual(sup, pontual)

    def test_reprodutivel(self):
        """Semente fixa: duas execuções devem dar o mesmo IC."""
        primeiro = ra.ic_bootstrap(self.dados, self.estat, replicas=300)
        segundo = ra.ic_bootstrap(self.dados, self.estat, replicas=300)
        np.testing.assert_allclose(primeiro, segundo, atol=TOLERANCIA)

    def test_encolhe_com_mais_documentos(self):
        import pandas as pd
        grande = pd.concat([self.dados.assign(
            documento=self.dados["documento"] + f"_{i}") for i in range(4)],
            ignore_index=True)
        largura_p = np.diff(ra.ic_bootstrap(self.dados, self.estat, replicas=400))[0]
        largura_g = np.diff(ra.ic_bootstrap(grande, self.estat, replicas=400))[0]
        self.assertLess(largura_g, largura_p)

    def test_poucos_documentos_devolve_nan(self):
        poucos = self.dados[self.dados["documento"] < "doc002"]
        self.assertTrue(np.isnan(ra.ic_bootstrap(poucos, self.estat, replicas=50)[0]))


class TesteInterpretacao(unittest.TestCase):
    """Faixas de McHugh (2012) — atenção: não são as de Landis & Koch."""

    def test_faixas(self):
        casos = [(0.10, "nenhum"), (0.30, "mínimo"), (0.50, "fraco"),
                 (0.70, "moderado"), (0.85, "forte"), (0.95, "quase perfeito")]
        for valor, esperado in casos:
            with self.subTest(valor=valor):
                self.assertEqual(ra.interpretar_kappa(valor), esperado)

    def test_limiar_do_criterio(self):
        """0,60 é o corte do gate e o início da faixa 'moderado'."""
        self.assertEqual(ra.interpretar_kappa(0.599), "fraco")
        self.assertEqual(ra.interpretar_kappa(ra.LIMIAR_KAPPA), "moderado")

    def test_nan(self):
        self.assertEqual(ra.interpretar_kappa(np.nan), "indefinido")


class TesteTaxonomia(unittest.TestCase):
    """Normalização dos rótulos de problema do template do Label Studio."""

    def test_categorias_do_template(self):
        esperadas = {"alucinacao", "omissao", "erro_factual",
                     "atribuicao_errada", "nao_consta_indev"}
        self.assertEqual(set(ra.CATEGORIAS_PROBLEMA), esperadas)

    def test_variantes_normalizam(self):
        casos = {
            "Não consta indev": "nao_consta_indev",
            "nao_consta": "nao_consta_indev",
            "NAO_CONSTA_INDEVIDO": "nao_consta_indev",
            "alucinação": "alucinacao",
            "atribucao_errada": "atribuicao_errada",
            "Erro Factual": "erro_factual",
            "omissoes": "omissao",
        }
        for bruto, esperado in casos.items():
            with self.subTest(bruto=bruto):
                self.assertEqual(ra.normalizar_problema(bruto), esperado)

    def test_rotulo_fora_da_rubrica_preservado(self):
        """Rótulos não previstos ficam visíveis na auditoria, não são forçados."""
        self.assertNotIn(ra.normalizar_problema("inventou_coisa"),
                         ra.CATEGORIAS_PROBLEMA)


class TesteMargem(unittest.TestCase):
    """Margem de relevância prática = 0,5 DP (Norman, Sloan & Wyrwich, 2003)."""

    def test_meia_dp(self):
        notas = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype=float)
        self.assertAlmostEqual(0.5 * notas.std(ddof=1),
                               0.5 * np.std(notas, ddof=1), places=9)

    def test_status_intermediario(self):
        """Viés significativo abaixo da margem → validado com ressalva."""
        import pandas as pd
        rng = np.random.default_rng(3)
        n = 300
        base = rng.normal(2.9, 0.9, n)
        b = np.clip(np.round(base), 1, 4).astype(int)
        a = b.copy()
        # viés pequeno, consistente e **dentro** da faixa adequada (3 -> 4):
        # a decisão binária não muda, então o McNemar não acusa direção
        candidatos = np.flatnonzero(b == 3)
        a[rng.choice(candidatos, min(45, len(candidatos)), replace=False)] = 4
        longo = pd.DataFrame({"documento": [f"d{i}" for i in range(n)],
                              "juiz": a, "ref": b})
        margem = 0.5 * float(longo["ref"].std(ddof=1))
        resultado = ra.avaliar_par(longo, "juiz", "ref", CATEGORIAS, margem=margem)
        self.assertLess(resultado["p_wilcoxon"], ra.LIMIAR_ALFA)
        self.assertLess(abs(resultado["media_dif"]), margem)
        self.assertEqual(resultado["status"], "VALIDADO COM RESSALVA")


class TesteMetricasBinarias(unittest.TestCase):
    """Acurácia, sensibilidade e especificidade com a referência como padrão."""

    def test_valores_conhecidos(self):
        referencia = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
        teste = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
        m = ra.metricas_binarias(referencia, teste)
        self.assertEqual((m["VP"], m["FN"], m["FP"], m["VN"]), (3, 1, 1, 3))
        self.assertAlmostEqual(m["acuracia"], 0.75)
        self.assertAlmostEqual(m["sensibilidade"], 0.75)
        self.assertAlmostEqual(m["especificidade"], 0.75)

    def test_juiz_leniente(self):
        """Aprovar tudo: sensibilidade 100%, especificidade 0%."""
        referencia = np.array([1, 1, 0, 0], dtype=bool)
        m = ra.metricas_binarias(referencia, np.ones(4, dtype=bool))
        self.assertAlmostEqual(m["sensibilidade"], 1.0)
        self.assertAlmostEqual(m["especificidade"], 0.0)


class TesteBayesiana(unittest.TestCase):
    """Camada bayesiana: a integração com o `baycomp` via `util_est_bayesiana`.

    A etapa é opcional; sem o módulo (ou sem o `baycomp`) os testes são pulados,
    do mesmo modo que o pipeline pula a etapa sem `--bayes`.
    """

    @classmethod
    def setUpClass(cls):
        if not ra.BAYES_DISPONIVEL:
            raise unittest.SkipTest("util_est_bayesiana/baycomp não disponíveis")
        cls.ub = ra.bayes
        rng = np.random.default_rng(17)
        base = rng.normal(0, 1, 200)
        # notas Likert 1–4 com efeitos conhecidos
        cls.dados = pd.DataFrame({
            "A": np.clip(np.round(2.6 + base + rng.normal(0, 0.4, 200)), 1, 4),
            "B": np.clip(np.round(2.6 + base + 0.02 + rng.normal(0, 0.4, 200)), 1, 4),
            "C": np.clip(np.round(2.6 + base + 1.20 + rng.normal(0, 0.4, 200)), 1, 4),
        })
        cls.kw = {"rope": 0.5, "nsamples": 20_000, "seed": 42}

    # ------------------------------------------------------------ Comparacao
    def test_tres_probabilidades_somam_um(self):
        c = self.ub.Comparacao(self.dados["A"], self.dados["C"], **self.kw)
        self.assertAlmostEqual(sum(c.probabilidades.values()), 1.0, places=6)

    def test_rope_zero_e_recusada(self):
        """Com rope=0 o baycomp devolve só duas probabilidades: erro explícito."""
        with self.assertRaises(ValueError) as ctx:
            self.ub.Comparacao(self.dados["A"], self.dados["B"], rope=0.0)
        self.assertIn("rope", str(ctx.exception).lower())

    def test_metodo_desconhecido(self):
        with self.assertRaises(ValueError):
            self.ub.Comparacao(self.dados["A"], self.dados["B"],
                               rope=0.5, metodo="qualquer")

    def test_direcao_correta(self):
        """C domina A: a leitura de A em relação a C é `inferior`."""
        c = self.ub.Comparacao(self.dados["A"], self.dados["C"], **self.kw)
        self.assertEqual(c.classificacao, "inferior")
        self.assertGreater(c.probabilidades["p_direita"], 0.95)

    def test_serie_identica_e_equivalente(self):
        """Comparar uma série consigo mesma não pode produzir direção."""
        c = self.ub.Comparacao(self.dados["A"], self.dados["A"], **self.kw)
        self.assertEqual(c.classificacao, "equivalente")
        self.assertAlmostEqual(c.probabilidades["p_rope"], 1.0, places=6)

    def test_nao_altera_o_baycomp(self):
        """O pacote é camada fina: o resultado tem de ser o do baycomp."""
        import baycomp
        c = self.ub.Comparacao(self.dados["A"], self.dados["C"], **self.kw)
        direto = baycomp.SignTest.probs(
            self.dados["A"].to_numpy(float), self.dados["C"].to_numpy(float),
            rope=0.5, nsamples=20_000, random_state=42)
        self.assertAlmostEqual(c.probabilidades["p_esquerda"], direto[0], places=9)
        self.assertAlmostEqual(c.probabilidades["p_rope"], direto[1], places=9)
        self.assertAlmostEqual(c.probabilidades["p_direita"], direto[2], places=9)

    def test_contagens_conferem_com_a_rope(self):
        c = self.ub.Comparacao(self.dados["A"], self.dados["C"], **self.kw)
        self.assertEqual(sum(c.contagens.values()), len(self.dados))

    def test_metodo_t_e_deterministico(self):
        """CorrelatedTTest é analítico: sementes diferentes, mesmo resultado."""
        a = self.ub.Comparacao(self.dados["A"], self.dados["C"], rope=0.5,
                               metodo="t", seed=1)
        b = self.ub.Comparacao(self.dados["A"], self.dados["C"], rope=0.5,
                               metodo="t", seed=999)
        self.assertEqual(a.probabilidades, b.probabilidades)

    # ---------------------------------------------------------- matriz_pares
    def test_matriz_cobre_todos_os_pares_ordenados(self):
        m = self.ub.matriz_pares(self.dados, **self.kw)
        k = self.dados.shape[1]
        self.assertEqual(len(m), k * (k - 1))
        self.assertTrue((m["linha"] != m["coluna"]).all())

    def test_matriz_e_simetrica(self):
        """A célula espelhada é derivada, não reestimada: igualdade exata."""
        m = self.ub.matriz_pares(self.dados, **self.kw)
        for _, linha in m.iterrows():
            esp = m[(m["linha"] == linha["coluna"])
                    & (m["coluna"] == linha["linha"])].iloc[0]
            self.assertEqual(linha["p_esquerda"], esp["p_direita"])
            self.assertEqual(linha["p_rope"], esp["p_rope"])
            self.assertAlmostEqual(linha["diferenca_media"],
                                   -esp["diferenca_media"], places=9)

    def test_espelho_inverte_a_classificacao(self):
        m = self.ub.matriz_pares(self.dados, **self.kw)
        oposto = {"superior": "inferior", "inferior": "superior"}
        for _, linha in m.iterrows():
            esp = m[(m["linha"] == linha["coluna"])
                    & (m["coluna"] == linha["linha"])].iloc[0]
            self.assertEqual(esp["classificacao"],
                             oposto.get(linha["classificacao"], linha["classificacao"]))

    def test_matriz_reprodutivel(self):
        a = self.ub.matriz_pares(self.dados, **self.kw)
        b = self.ub.matriz_pares(self.dados, **self.kw)
        np.testing.assert_array_equal(a["p_esquerda"].to_numpy(),
                                      b["p_esquerda"].to_numpy())

    def test_ordem_dos_nomes_e_respeitada(self):
        """Travar a ordem é o que permite comparar duas figuras lado a lado."""
        m = self.ub.matriz_pares(self.dados, nomes=["C", "A", "B"], **self.kw)
        self.assertEqual(m.attrs["nomes"], ["C", "A", "B"])

    def test_matriz_exige_dois_protocolos(self):
        with self.assertRaises(ValueError):
            self.ub.matriz_pares(self.dados[["A"]], **self.kw)

    def test_limiar_mais_exigente_gera_mais_incertos(self):
        conta = lambda limiar: (self.ub.matriz_pares(
            self.dados, limiar=limiar, **self.kw)["classificacao"] == "incerto").sum()
        self.assertGreaterEqual(conta(0.999), conta(0.60))

    def test_resumo_conta_todas_as_relacoes(self):
        m = self.ub.matriz_pares(self.dados, **self.kw)
        r = self.ub.resumo(m)
        self.assertEqual(list(r.index), m.attrs["nomes"])
        self.assertEqual(int(r.to_numpy().sum()), len(m))

    # -------------------------------------------------- integração ao gate
    def test_veredito_julga_magnitude_e_nao_direcao(self):
        """Direção certa com margem trivial não pode ser lida como viés."""
        rng = np.random.default_rng(9)
        n = 400
        ref = np.clip(np.round(rng.normal(2.9, 0.9, n)), 1, 4).astype(int)
        juiz = ref.copy()
        juiz[np.flatnonzero(ref == 3)[:12]] = 4        # viés minúsculo, um só sentido
        longo = pd.DataFrame({"juiz": juiz.astype(float), "ref": ref.astype(float)})
        cfg = ra.ConfigBayes(ativo=True, amostras=20_000, semente=42)
        r = ra.bayes_par(longo, "juiz", "ref", cfg)
        self.assertGreater(r["p_equiv"], 0.95)
        self.assertEqual(r["status"], "SEM VIÉS RELEVANTE")

    def test_bayes_par_devolve_as_duas_comparacoes(self):
        rng = np.random.default_rng(4)
        n = 300
        ref = np.clip(np.round(rng.normal(2.8, 0.9, n)), 1, 4).astype(float)
        longo = pd.DataFrame({"juiz": ref, "ref": ref})
        cfg = ra.ConfigBayes(ativo=True, amostras=20_000, semente=42)
        r = ra.bayes_par(longo, "juiz", "ref", cfg)
        self.assertEqual(r["acima"] + r["empate"] + r["abaixo"], n)
        self.assertEqual(r["decisao_concordante"], n)   # idênticos: nenhuma discordância
        self.assertEqual(r["status"], "SEM VIÉS RELEVANTE")


class TesteFlagsBayes(unittest.TestCase):
    """A etapa só existe quando pedida, e a CLI explica quando não roda."""

    @staticmethod
    def _args(**mudancas):
        padrao = {"bayes": False,
                  "bayes_rope": ra.BAYES_ROPE_LIKERT,
                  "bayes_metodo": ra.BAYES_METODO_PADRAO,
                  "bayes_limiar": ra.BAYES_LIMIAR_PADRAO,
                  "bayes_limiar_veredito": ra.BAYES_VEREDITO_PADRAO,
                  "bayes_amostras": ra.BAYES_AMOSTRAS_PADRAO,
                  "bayes_semente": ra.SEMENTE}
        return type("Args", (), padrao | mudancas)()

    @staticmethod
    def _erro(mensagem):
        raise ValueError(mensagem)

    def test_sem_flag_a_etapa_nao_roda(self):
        self.assertFalse(ra.montar_config_bayes(self._args(), self._erro).ativo)

    def test_ajuste_sem_bayes_e_erro_explicito(self):
        """Silenciar aqui devolveria um relatório sem a seção pedida, sem aviso."""
        with self.assertRaises(ValueError) as ctx:
            ra.montar_config_bayes(self._args(bayes_metodo="t"), self._erro)
        self.assertIn("--bayes", str(ctx.exception))

    def test_rope_zero_e_recusada(self):
        with self.assertRaises(ValueError) as ctx:
            ra.montar_config_bayes(self._args(bayes=True, bayes_rope=0.0), self._erro)
        self.assertIn("ROPE", str(ctx.exception))

    def test_limiar_fora_da_faixa(self):
        with self.assertRaises(ValueError):
            ra.montar_config_bayes(self._args(bayes=True, bayes_limiar=1.5), self._erro)

    def test_amostras_insuficientes(self):
        with self.assertRaises(ValueError):
            ra.montar_config_bayes(self._args(bayes=True, bayes_amostras=10), self._erro)

    def test_configuracao_completa(self):
        cfg = ra.montar_config_bayes(
            self._args(bayes=True, bayes_rope=0.5, bayes_metodo="postos",
                       bayes_limiar=0.85, bayes_amostras=50_000), self._erro)
        self.assertTrue(cfg.ativo)
        self.assertEqual((cfg.rope, cfg.metodo, cfg.limiar), (0.5, "postos", 0.85))
        self.assertEqual(cfg.kw["nsamples"], 50_000)


# =============================================================================
# Execução
# =============================================================================

def _resumo_ambiente() -> None:
    print("=" * 72)
    print("Verificação das estatísticas de realizar_avaliacoes.py")
    print("=" * 72)
    print("  scikit-learn : disponível")
    print("  statsmodels  : disponível")
    print()
    print("Origem das estatísticas no pipeline:")
    for nome, origem in ra.dependencias().items():
        print(f"  • {nome}: {origem}")
    print("=" * 72)


if __name__ == "__main__":
    _resumo_ambiente()
    unittest.main(verbosity=2, exit=False)
    print("\nUse este resultado para sustentar, no capítulo de método, que as "
          "definições operacionais das estatísticas coincidem com os "
          "pacotes usados pelo pipeline.")
