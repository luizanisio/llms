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
    """Camada bayesiana: partição da posterior, simetria e classificação.

    A etapa é opcional; sem `util_est_bayesiana` disponível os testes são
    pulados, do mesmo modo que o pipeline pula a etapa sem `--bayes`.
    """

    @classmethod
    def setUpClass(cls):
        if not ra.BAYES_DISPONIVEL:
            raise unittest.SkipTest("util_est_bayesiana não disponível")
        cls.ub = ra.bayes
        rng = np.random.default_rng(17)
        base = rng.normal(0, 1, 300)
        cls.dados = {
            "A": base + rng.normal(0, 0.30, 300),
            "B": base + 0.01 + rng.normal(0, 0.30, 300),   # ~igual a A
            "C": base + 0.60 + rng.normal(0, 0.30, 300),   # claramente melhor
        }
        cls.kw = {"nsamples": 20_000, "seed": 42}

    # ---------------------------------------------------------------- partição
    def test_tres_probabilidades_somam_um(self):
        """São uma partição do espaço amostral, não três números normalizados."""
        for eps in (0.0, 0.05, 0.20):
            cmp = self.ub.ComparacaoPareada(self.dados["A"], self.dados["C"], **self.kw)
            p = cmp.probabilidades_relacao(eps)
            self.assertAlmostEqual(
                p["p_inferior"] + p["p_equivalente"] + p["p_superior"], 1.0, places=9)

    def test_equivalencia_coincide_com_p_equiv(self):
        """A faixa central é o mesmo ε aplicado sobre a posterior."""
        cmp = self.ub.ComparacaoPareada(self.dados["A"], self.dados["B"], **self.kw)
        self.assertAlmostEqual(cmp.probabilidades_relacao(0.10)["p_equivalente"],
                               cmp.p_equiv(0.10), places=3)

    def test_eps_zero_nao_deixa_massa_no_centro(self):
        """Sem margem não existe equivalência: a decomposição vira P(δ>0)/P(δ<0)."""
        cmp = self.ub.ComparacaoPareada(self.dados["A"], self.dados["C"], **self.kw)
        p = cmp.probabilidades_relacao(0.0)
        self.assertAlmostEqual(p["p_equivalente"], 0.0, places=6)
        self.assertAlmostEqual(p["p_superior"], cmp.p_dom, places=6)

    def test_direcao_correta(self):
        """C domina A: a leitura de A em relação a C é `inferior`."""
        cmp = self.ub.ComparacaoPareada(self.dados["A"], self.dados["C"], **self.kw)
        p = cmp.probabilidades_relacao(0.05)
        self.assertGreater(p["p_inferior"], 0.99)

    def test_serie_identica_e_equivalente(self):
        """Comparar uma série consigo mesma não pode produzir direção."""
        cmp = self.ub.ComparacaoPareada(self.dados["A"], self.dados["A"], **self.kw)
        p = cmp.probabilidades_relacao(0.05)
        self.assertAlmostEqual(p["p_equivalente"], 1.0, places=6)

    # ----------------------------------------------------------------- simetria
    def test_matriz_e_simetrica(self):
        """A célula espelhada é derivada, não reestimada: igualdade exata."""
        matriz = self.ub.matriz_relacoes(self.dados, eps=0.05, limiar=0.80, **self.kw)
        for _, linha in matriz.iterrows():
            espelho = matriz[(matriz["linha"] == linha["coluna"])
                             & (matriz["coluna"] == linha["linha"])].iloc[0]
            self.assertEqual(linha["p_superior"], espelho["p_inferior"])
            self.assertEqual(linha["p_inferior"], espelho["p_superior"])
            self.assertEqual(linha["p_equivalente"], espelho["p_equivalente"])
            self.assertAlmostEqual(linha["delta"], -espelho["delta"], places=9)

    def test_matriz_cobre_todos_os_pares_ordenados(self):
        matriz = self.ub.matriz_relacoes(self.dados, eps=0.05, **self.kw)
        k = len(self.dados)
        self.assertEqual(len(matriz), k * (k - 1))
        self.assertTrue((matriz["linha"] != matriz["coluna"]).all())

    def test_matriz_reprodutivel(self):
        """Mesma semente, mesmos números — exigência do protocolo de análise."""
        a = self.ub.matriz_relacoes(self.dados, eps=0.05, **self.kw)
        b = self.ub.matriz_relacoes(self.dados, eps=0.05, **self.kw)
        np.testing.assert_array_equal(a["p_superior"].to_numpy(),
                                      b["p_superior"].to_numpy())

    # ------------------------------------------------------------ classificação
    def test_classificacao_respeita_o_limiar(self):
        classificar = self.ub.classificar_relacao
        self.assertEqual(classificar(0.02, 0.05, 0.93, 0.80)[0], "superior")
        self.assertEqual(classificar(0.95, 0.03, 0.02, 0.80)[0], "inferior")
        self.assertEqual(classificar(0.06, 0.88, 0.06, 0.80)[0], "equivalente")
        self.assertEqual(classificar(0.20, 0.74, 0.06, 0.80)[0], "incerto")

    def test_incerto_reporta_a_dominante(self):
        """No estado incerto o número mostra quão perto do limiar a evidência chegou."""
        classe, valor = self.ub.classificar_relacao(0.20, 0.74, 0.06, 0.80)
        self.assertEqual(classe, "incerto")
        self.assertAlmostEqual(valor, 0.74)

    def test_empate_exato_resolve_para_incerto(self):
        """Situação limítrofe recebe leitura neutra, não uma relação inventada."""
        self.assertEqual(self.ub.classificar_relacao(0.5, 0.0, 0.5, 0.40)[0], "incerto")

    def test_limiar_mais_exigente_produz_mais_incertos(self):
        conta = lambda limiar: sum(
            self.ub.classificar_relacao(l["p_inferior"], l["p_equivalente"],
                                        l["p_superior"], limiar)[0] == "incerto"
            for _, l in self.ub.matriz_relacoes(self.dados, eps=0.05, **self.kw).iterrows())
        self.assertGreaterEqual(conta(0.99), conta(0.80))

    # -------------------------------------------------------- integração ao gate
    def test_veredito_julga_magnitude_e_nao_direcao(self):
        """Direção certa com margem trivial não pode ser lida como viés."""
        import pandas as pd
        rng = np.random.default_rng(9)
        n = 600
        ref = np.clip(np.round(rng.normal(2.9, 0.9, n)), 1, 4).astype(int)
        juiz = ref.copy()
        # viés minúsculo, sempre no mesmo sentido: P(dominância) satura em 1,
        # mas a magnitude cabe folgadamente dentro de ε
        alvos = np.flatnonzero(ref == 3)[:12]
        juiz[alvos] = 4
        longo = pd.DataFrame({"documento": [f"d{i}" for i in range(n)],
                              "juiz": juiz, "ref": ref})
        cfg = ra.ConfigBayes(ativo=True, eps=0.20, amostras=20_000, semente=42)
        resultado = ra.bayes_par(longo, "juiz", "ref", cfg)
        self.assertGreater(resultado["p_dom"], 0.95)          # direção certa
        self.assertGreater(resultado["p_equiv"], 0.95)        # magnitude trivial
        self.assertEqual(resultado["status"], "SEM VIÉS RELEVANTE")

    def test_calibracao_do_eps_arredonda_para_cima(self):
        """O ε empírico nunca fica abaixo da divergência observada."""
        import pandas as pd
        rng = np.random.default_rng(4)
        n = 200
        notas = np.clip(np.round(rng.normal(2.8, 0.9, n)), 1, 4).astype(int)
        registros = []
        for avaliador, deslocamento in ((1, 0), (2, 1)):
            valores = np.clip(notas + (deslocamento if avaliador == 2 else 0), 1, 4)
            registros += [{"documento": f"d{i}", "fonte": "a",
                           "avaliador": avaliador, "nota": int(v)}
                          for i, v in enumerate(valores)]
        cfg = ra.ConfigBayes(ativo=True, amostras=20_000, semente=42)
        eps, origem = ra.calibrar_eps(pd.DataFrame(registros), [1, 2], cfg)
        self.assertGreater(eps, 0.0)
        self.assertAlmostEqual(eps, np.ceil(eps * 100) / 100, places=9)
        self.assertIn("calibrado", origem)

    def test_grupo_sem_pares_nao_calibra(self):
        """Com um avaliador só, o ε precisa vir por flag — e o pipeline avisa."""
        import pandas as pd
        registros = [{"documento": f"d{i}", "fonte": "a", "avaliador": 1, "nota": 3}
                     for i in range(10)]
        cfg = ra.ConfigBayes(ativo=True, amostras=1_000, semente=42)
        eps, _ = ra.calibrar_eps(pd.DataFrame(registros), [1], cfg)
        self.assertTrue(np.isnan(eps))
        self.assertFalse(ra.resolver_eps(cfg, pd.DataFrame(registros), [1], "x").ativo)


class TesteFlagsBayes(unittest.TestCase):
    """A etapa só existe quando pedida, e a CLI explica quando não roda."""

    @staticmethod
    def _args(**mudancas):
        padrao = {"bayes": False, "bayes_eps": None,
                  "bayes_rope": ra.BAYES_ROPE_PADRAO,
                  "bayes_limiar": ra.BAYES_LIMIAR_PADRAO,
                  "bayes_limiar_veredito": ra.BAYES_VEREDITO_PADRAO,
                  "bayes_amostras": ra.BAYES_AMOSTRAS_PADRAO,
                  "bayes_semente": ra.SEMENTE}
        return type("Args", (), padrao | mudancas)()

    @staticmethod
    def _erro(mensagem):
        raise ValueError(mensagem)

    def test_sem_flag_a_etapa_nao_roda(self):
        cfg = ra.montar_config_bayes(self._args(), self._erro)
        self.assertFalse(cfg.ativo)
        self.assertFalse(cfg.resolvido)

    def test_ajuste_sem_bayes_e_erro_explicito(self):
        """Silenciar aqui devolveria um relatório sem a seção pedida, sem aviso."""
        with self.assertRaises(ValueError) as contexto:
            ra.montar_config_bayes(self._args(bayes_eps=0.08), self._erro)
        self.assertIn("--bayes", str(contexto.exception))

    def test_limiar_fora_da_faixa(self):
        with self.assertRaises(ValueError):
            ra.montar_config_bayes(self._args(bayes=True, bayes_limiar=1.5), self._erro)

    def test_amostras_insuficientes(self):
        with self.assertRaises(ValueError):
            ra.montar_config_bayes(self._args(bayes=True, bayes_amostras=10), self._erro)

    def test_configuracao_completa(self):
        cfg = ra.montar_config_bayes(
            self._args(bayes=True, bayes_eps=0.08, bayes_rope=0.01,
                       bayes_limiar=0.85, bayes_amostras=50_000), self._erro)
        self.assertTrue(cfg.resolvido)
        self.assertEqual((cfg.eps, cfg.rope, cfg.limiar), (0.08, 0.01, 0.85))
        self.assertEqual(cfg.kw_posterior, {"nsamples": 50_000, "seed": ra.SEMENTE})

    def test_ordem_de_calibracao_prioriza_a_referencia(self):
        """O grupo que ancora o ε é analisado primeiro, para valer em toda a execução."""
        grupos = [ra.Grupo("gpt5", "llm"), ra.Grupo("humanos", "humano")]
        cfg = ra.ConfigBayes(ativo=True)
        self.assertEqual([g.nome for g in ra.ordenar_para_calibracao(grupos, cfg)],
                         ["humanos", "gpt5"])
        fixo = ra.ConfigBayes(ativo=True, eps=0.08)
        self.assertEqual([g.nome for g in ra.ordenar_para_calibracao(grupos, fixo)],
                         ["gpt5", "humanos"])
        self.assertEqual([g.nome for g in ra.ordenar_para_calibracao(grupos, None)],
                         ["gpt5", "humanos"])


class TesteModosBayesianos(unittest.TestCase):
    """Os dois caminhos de extração das probabilidades — e a distinção entre eles.

    O ponto central: `proporcao` e `baycomp` produzem rótulos iguais para
    afirmações diferentes. Os testes fixam essa diferença para que nenhuma
    refatoração futura os trate como intercambiáveis.
    """

    @classmethod
    def setUpClass(cls):
        if not ra.BAYES_DISPONIVEL:
            raise unittest.SkipTest("util_est_bayesiana não disponível")
        cls.ub = ra.bayes
        rng = np.random.default_rng(23)
        base = rng.normal(0, 1, 400)
        cls.continuo = {
            "A": base + rng.normal(0, 0.20, 400),
            "B": base + 0.02 + rng.normal(0, 0.20, 400),
            "C": base + 0.50 + rng.normal(0, 0.20, 400),
        }
        cls.kw = {"nsamples": 20_000, "seed": 42}

    def test_baycomp_exige_rope(self):
        """Com rope = 0 o triplet degenera em (0,1,0): 'equivalente' sem significar nada."""
        cmp = self.ub.ComparacaoPareada(self.continuo["A"], self.continuo["C"],
                                        rope=0.0, **self.kw)
        with self.assertRaises(ValueError) as contexto:
            cmp.probabilidades_relacao(modo="baycomp")
        self.assertIn("rope", str(contexto.exception).lower())

    def test_baycomp_ignora_eps_com_aviso(self):
        """A única margem do modo baycomp é a ROPE; o ε não participa."""
        cmp = self.ub.ComparacaoPareada(self.continuo["A"], self.continuo["C"],
                                        rope=0.05, **self.kw)
        with self.assertWarns(UserWarning):
            com_eps = cmp.probabilidades_relacao(eps=0.30, modo="baycomp")
        sem_eps = cmp.probabilidades_relacao(modo="baycomp")
        self.assertEqual(com_eps, sem_eps)

    def test_proporcao_avisa_com_eps_zero(self):
        """Faixa de equivalência de medida nula: nenhuma célula poderá ser azul."""
        with self.assertWarns(UserWarning):
            self.ub.matriz_relacoes(self.continuo, eps=0.0, **self.kw)

    def test_modo_desconhecido(self):
        cmp = self.ub.ComparacaoPareada(self.continuo["A"], self.continuo["B"], **self.kw)
        with self.assertRaises(ValueError):
            cmp.probabilidades_relacao(modo="qualquer")

    def test_ambos_os_modos_somam_um(self):
        cmp = self.ub.ComparacaoPareada(self.continuo["A"], self.continuo["C"],
                                        rope=0.05, **self.kw)
        for modo, eps in (("proporcao", 0.10), ("baycomp", None)):
            p = cmp.probabilidades_relacao(eps, modo=modo)
            self.assertAlmostEqual(
                p["p_inferior"] + p["p_equivalente"] + p["p_superior"], 1.0,
                places=6, msg=f"modo {modo}")

    def test_modos_afirmam_coisas_diferentes(self):
        """Zona central maioritária ≠ magnitude dentro da margem.

        Construído para o caso descrito na documentação: massa relevante na
        ROPE, porém com dominância clara de um lado.
        """
        rng = np.random.default_rng(11)
        n = 800
        x = rng.normal(0, 1, n)
        y = x - rng.choice([0.0, 0.30], size=n, p=[0.65, 0.35])
        cmp = self.ub.ComparacaoPareada(x, y, rope=0.10, **self.kw)
        proporcao = cmp.probabilidades_relacao(0.05, modo="proporcao")
        baycomp = cmp.probabilidades_relacao(modo="baycomp")
        # as duas leituras não coincidem: é isso que a legenda precisa distinguir
        self.assertNotAlmostEqual(proporcao["p_equivalente"],
                                  baycomp["p_equivalente"], places=2)

    def test_simetria_preservada_no_modo_baycomp(self):
        matriz = self.ub.matriz_relacoes(self.continuo, rope=0.05, modo="baycomp",
                                         **self.kw)
        for _, linha in matriz.iterrows():
            espelho = matriz[(matriz["linha"] == linha["coluna"])
                             & (matriz["coluna"] == linha["linha"])].iloc[0]
            self.assertEqual(linha["p_superior"], espelho["p_inferior"])
            self.assertEqual(linha["p_equivalente"], espelho["p_equivalente"])

    def test_atributos_identificam_a_figura(self):
        """Modo, métrica e papel viajam na matriz: é o que impede confundir figuras."""
        matriz = self.ub.matriz_relacoes(self.continuo, rope=0.05, modo="baycomp",
                                         metrica="BERTScore F1", papel="complementar",
                                         **self.kw)
        self.assertEqual(matriz.attrs["modo"], "baycomp")
        self.assertEqual(matriz.attrs["metrica"], "BERTScore F1")
        self.assertEqual(matriz.attrs["papel"], "complementar")
        self.assertIsNone(matriz.attrs["eps"])

    def test_rotulo_central_muda_com_o_modo(self):
        proporcao = self.ub._rotulos_legenda("proporcao", 0.80)
        baycomp = self.ub._rotulos_legenda("baycomp", 0.80)
        self.assertEqual(proporcao["equivalente"], "equivalente")
        self.assertEqual(baycomp["equivalente"], "ROPE maioritária")

    def test_massa_de_empate_limita_delta(self):
        """δ ∈ ±(1 − massa de empate): é o que torna o ε interpretável."""
        rng = np.random.default_rng(3)
        notas = rng.integers(1, 5, 500).astype(float)
        cmp = self.ub.ComparacaoPareada(notas, notas.copy(), rope=0.0, **self.kw)
        self.assertAlmostEqual(cmp.massa_empate, 1.0)
        self.assertLessEqual(abs(float(cmp.delta.mean())), 1e-9)


class TesteRopeCalibrada(unittest.TestCase):
    """Calibração da ROPE pela variação entre execuções do mesmo protocolo."""

    @classmethod
    def setUpClass(cls):
        if not ra.BAYES_DISPONIVEL:
            raise unittest.SkipTest("util_est_bayesiana não disponível")
        cls.ub = ra.bayes

    def test_percentil_e_nao_amplitude(self):
        """Um outlier isolado não pode ditar a margem."""
        base = np.zeros(1000)
        outra = base.copy()
        outra[0] = 10.0                       # caso patológico
        outra[1:] = np.linspace(0, 0.01, 999)
        rope, detalhe = self.ub.calibrar_rope({"e1": base, "e2": outra}, percentil=90)
        self.assertLess(rope, 0.02)           # a amplitude daria 10,0
        self.assertEqual(detalhe["percentil"], 90)

    def test_usa_o_maior_entre_os_pares(self):
        """A ROPE precisa cobrir o pior caso observado entre as execuções."""
        rng = np.random.default_rng(8)
        n = 500
        base = rng.normal(0, 1, n)
        execucoes = {"e1": base,
                     "e2": base + rng.normal(0, 0.01, n),
                     "e3": base + rng.normal(0, 0.05, n)}   # a mais divergente
        rope, detalhe = self.ub.calibrar_rope(execucoes)
        self.assertAlmostEqual(rope, max(detalhe["por_par"].values()))
        self.assertEqual(len(detalhe["por_par"]), 3)

    def test_execucoes_identicas_avisam(self):
        """Decodificação gulosa mede determinismo do decoder, não incerteza."""
        serie = np.linspace(0, 1, 100)
        with self.assertWarns(UserWarning):
            rope, _ = self.ub.calibrar_rope({"e1": serie, "e2": serie.copy()})
        self.assertEqual(rope, 0.0)

    def test_exige_duas_execucoes(self):
        with self.assertRaises(ValueError):
            self.ub.calibrar_rope({"e1": np.zeros(10)})

    def test_exige_pareamento(self):
        with self.assertRaises(ValueError):
            self.ub.calibrar_rope({"e1": np.zeros(10), "e2": np.zeros(11)})

    def test_controle_negativo_aprova_rope_adequada(self):
        """O modelo comparado consigo mesmo tem de sair equivalente."""
        rng = np.random.default_rng(15)
        n = 600
        base = rng.normal(0, 1, n)
        a, b = base + rng.normal(0, 0.01, n), base + rng.normal(0, 0.01, n)
        rope, _ = self.ub.calibrar_rope({"e1": a, "e2": b}, percentil=90)
        resultado = self.ub.controle_negativo(a, b, rope=rope, nsamples=20_000, seed=42)
        self.assertTrue(resultado["aprovado"])
        self.assertEqual(resultado["classificacao"], "equivalente")

    def test_controle_negativo_reprova_rope_pequena(self):
        """Uma ROPE apertada demais é detectada ANTES de olhar os protocolos."""
        rng = np.random.default_rng(15)
        n = 600
        base = rng.normal(0, 1, n)
        a, b = base + rng.normal(0, 0.05, n), base + rng.normal(0, 0.05, n)
        resultado = self.ub.controle_negativo(a, b, rope=1e-6, nsamples=20_000, seed=42)
        self.assertFalse(resultado["aprovado"])
        self.assertIn("pequena demais", resultado["diagnostico"])


class TesteLeituraConjunta(unittest.TestCase):
    """Convergência entre métricas e as duas análises de sensibilidade."""

    @classmethod
    def setUpClass(cls):
        if not ra.BAYES_DISPONIVEL:
            raise unittest.SkipTest("util_est_bayesiana não disponível")
        cls.ub = ra.bayes
        rng = np.random.default_rng(31)
        base = rng.normal(0, 1, 400)
        efeitos = {"P1": 0.0, "P2": 0.02, "P3": 0.60}
        cls.likert = {p: cls.ub.discretizar(base + v + rng.normal(0, 0.35, 400))
                      for p, v in efeitos.items()}
        cls.f1 = {p: np.clip(0.9 + 0.05 * (base + v) + rng.normal(0, 0.02, 400), 0, 1)
                  for p, v in efeitos.items()}
        cls.kw = {"nsamples": 20_000, "seed": 42}
        cls.m_likert = cls.ub.matriz_relacoes(
            cls.likert, eps=0.10, modo="proporcao", metrica="Likert",
            papel="principal", **cls.kw)
        cls.m_f1 = cls.ub.matriz_relacoes(
            cls.f1, rope=0.02, modo="baycomp", metrica="F1",
            papel="complementar", nomes=list(efeitos), **cls.kw)

    def test_convergencia_uma_linha_por_par_nao_ordenado(self):
        tabela = self.ub.tabela_convergencia(self.m_likert, self.m_f1)
        self.assertEqual(len(tabela), 3)          # 3 protocolos = 3 pares

    def test_situacoes_possiveis(self):
        tabela = self.ub.tabela_convergencia(self.m_likert, self.m_f1)
        self.assertTrue(set(tabela["Situação"]) <=
                        {"convergente", "divergente", "sem decisão"})

    def test_incerto_vira_sem_decisao(self):
        """Sem classificação em uma das métricas, não há o que convergir."""
        # ε apertado: o par de efeito quase nulo não alcança o limiar
        m_incerto = self.ub.matriz_relacoes(self.likert, eps=0.02, limiar=0.80,
                                            modo="proporcao", **self.kw)
        tabela = self.ub.tabela_convergencia(m_incerto, self.m_f1)
        self.assertIn("sem decisão", set(tabela["Situação"]))

    def test_convergencia_consigo_mesma_e_total(self):
        tabela = self.ub.tabela_convergencia(self.m_likert, self.m_likert)
        self.assertTrue((tabela["Situação"] == "convergente").all())

    def test_sensibilidade_limiar_conta_mudancas(self):
        tabela = self.ub.sensibilidade_limiar(self.m_likert, limiares=(0.70, 0.90),
                                              referencia=0.80)
        self.assertIn(0.80, list(tabela["Limiar"]))
        # a linha de referência não muda em relação a si mesma
        referencia = tabela[tabela["Referência"] == "sim"].iloc[0]
        self.assertEqual(referencia["Muda vs. referência"], 0)

    def test_sensibilidade_limiar_mais_exigente_gera_mais_incertos(self):
        tabela = self.ub.sensibilidade_limiar(self.m_likert, limiares=(0.60, 0.999))
        incertos = dict(zip(tabela["Limiar"], tabela["incerto"]))
        self.assertGreaterEqual(incertos[0.999], incertos[0.60])

    def test_sensibilidade_margem_inclui_a_referencia(self):
        tabela = self.ub.sensibilidade_margem(
            self.likert, valores=(0.05, 0.20), modo="proporcao",
            referencia=0.10, **self.kw)
        self.assertEqual(list(tabela["Referência"]).count("sim"), 1)
        self.assertIn(0.10, list(tabela["ε"]))

    def test_sensibilidade_margem_rope_reamostra(self):
        """No modo baycomp a coluna é ROPE, e margens maiores engolem as relações."""
        tabela = self.ub.sensibilidade_margem(
            self.f1, valores=(0.01, 0.50), modo="baycomp", referencia=0.01,
            nomes=["P1", "P2", "P3"], **self.kw)
        self.assertIn("ROPE", tabela.columns)
        equivalentes = dict(zip(tabela["ROPE"], tabela["equivalente"]))
        self.assertGreater(equivalentes[0.50], equivalentes[0.01])

    def test_sensibilidade_margem_proporcao_reaproveita_posterior(self):
        """Reaproveitar as amostras precisa dar o MESMO resultado de recalcular."""
        tabela = self.ub.sensibilidade_margem(
            self.likert, valores=(0.10,), modo="proporcao", referencia=0.10,
            limiar=0.80, **self.kw)
        direto = self.ub.matriz_relacoes(self.likert, eps=0.10, limiar=0.80,
                                         modo="proporcao", **self.kw)
        classes = [l["classificacao"] for l in self.ub._pares_unicos(direto)]
        linha = tabela.iloc[0]
        self.assertEqual(int(linha["superior"]), classes.count("superior"))
        self.assertEqual(int(linha["equivalente"]), classes.count("equivalente"))
        self.assertEqual(int(linha["inferior"]), classes.count("inferior"))


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
