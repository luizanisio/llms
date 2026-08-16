#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Nota:
    A amostragem da distribuição a posteriori foi reimplementada porque a versão do baycomp utilizada 
    não expõe o prior nem a semente aleatória, impedindo a análise de sensibilidade à alocação do prior 
    e a reprodutibilidade exigidas pelo protocolo de análise. A equivalência numérica com o pacote foi 
    verificada nos mesmos dados, com divergência máxima de 10⁻⁵, compatível com ruído de Monte Carlo.    

Ferramentas da análise bayesiana pareada do experimento SUMMA.

Método: teste de sinais bayesiano (Benavoli et al., 2017, JMLR 18:1-36).
Implementação de referência da posterior: baycomp (github.com/janezd/baycomp).  <<<<<<<<<<<<<<<< FONTE

Uso:
    from util_est_bayesiana import ComparacaoPareada, ProporcaoBeta, kappa_w
    cmp = ComparacaoPareada(notas_C, notas_D1, rope=0.0)
    print(cmp.resumo(eps=0.05, nomes=("C", "D1")))

    # matriz de comparação entre protocolos + heatmap (PNG)
    from util_est_bayesiana import heatmap_comparacao
    matriz, arquivo = heatmap_comparacao(escores_f1, eps=0.05, rope=0.01,
                                         arquivo_saida="bayes_protocolos.png")

"""



import warnings

import numpy as np

__all__ = ["ComparacaoPareada", "ProporcaoBeta", "kappa_w", "discretizar",
           "ic_bootstrap", "tabela_contraste", "configurar",
           "MODOS", "RELACOES", "CORES_RELACAO", "LIMIAR_CLASSIFICACAO_PADRAO",
           "classificar_relacao", "matriz_relacoes", "resumo_relacoes",
           "heatmap_relacoes", "heatmap_comparacao",
           "tabela_convergencia", "sensibilidade_limiar", "sensibilidade_margem",
           "grafico_curva_sensibilidade_eps",
           "calibrar_rope", "controle_negativo"]

NSAMPLES_PADRAO = 200_000
SEED_PADRAO = 42

#: limiar de decisão para classificar uma relação (probabilidade posterior)
LIMIAR_CLASSIFICACAO_PADRAO = 0.80

#: Como as três probabilidades são extraídas da posterior. NÃO são a mesma
#: afirmação — ver `ComparacaoPareada.probabilidades_relacao`. A escolha é
#: sempre explícita: não há padrão "seguro" que sirva às duas métricas.
#:
#:   "proporcao" — P(|δ| ≤ ε) sobre a diferença de proporções de documentos.
#:                 Usado na escala Likert (ordinal, rope = 0).
#:   "baycomp"   — P(cada zona ser a MAIORITÁRIA), o cálculo padrão do pacote.
#:                 Usado em escores contínuos (F1), onde a ROPE tem sentido.
MODOS = ("proporcao", "baycomp")


def configurar(nsamples=None, seed=None):
    """Ajusta os padrões do módulo.

    Necessário porque argumentos default são fixados na DEFINIÇÃO da função:
    reatribuir `util_est_bayesiana.NSAMPLES_PADRAO` depois do import não teria efeito.
    Por isso as assinaturas usam `None` e resolvem o valor em tempo de chamada.

        import util_est_bayesiana
        util_est_bayesiana.configurar(nsamples=50_000, seed=42)
    """
    global NSAMPLES_PADRAO, SEED_PADRAO
    if nsamples is not None:
        NSAMPLES_PADRAO = int(nsamples)
    if seed is not None:
        SEED_PADRAO = int(seed)
    return {"nsamples": NSAMPLES_PADRAO, "seed": SEED_PADRAO}


# ══════════════════════════════════════════════════════ utilitários de escala

def discretizar(z, cortes=(-1.10, -0.10, 0.85)):
    """Qualidade latente contínua -> nota Likert inteira (1..k)."""
    return (np.searchsorted(np.asarray(cortes), z) + 1).astype(int)


def kappa_w(a, b, k=4):
    """Cohen kappa com pesos quadráticos entre dois avaliadores."""
    a, b = np.asarray(a), np.asarray(b)
    O = np.zeros((k, k))
    for i in range(1, k + 1):
        for j in range(1, k + 1):
            O[i - 1, j - 1] = ((a == i) & (b == j)).sum()
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float) / (k - 1) ** 2
    return float(1 - (W * O).sum() / (W * E).sum())


def ic_bootstrap(fn, *arrays, n_rep=2000, seed=None, q=(2.5, 97.5)):
    """IC percentílico por reamostragem pareada de unidades (documentos).

    Reamostra os MESMOS índices em todos os arrays, preservando o pareamento.
    """
    seed = SEED_PADRAO if seed is None else seed
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    vals = [fn(*[a[i] for a in arrays])
            for i in (rng.integers(0, n, n) for _ in range(n_rep))]
    return tuple(np.percentile(vals, q))


# ══════════════════════════════════════════════════════ proporção (Beta)

class ProporcaoBeta:
    """Posterior de uma proporção — P(nota >= piso), taxa de descarte, etc.

    Prior de Jeffreys Beta(0,5; 0,5) por padrão: fraco e invariante a
    reparametrizações. Substitui o IC de Wilson por uma posterior direta.
    """

    def __init__(self, k, n, prior=0.5, nsamples=None, seed=None):
        nsamples = NSAMPLES_PADRAO if nsamples is None else nsamples
        seed = SEED_PADRAO if seed is None else seed
        self.k, self.n, self.seed = int(k), int(n), seed
        self.amostras = np.random.default_rng(seed).beta(
            self.k + prior, self.n - self.k + prior, size=nsamples)

    @classmethod
    def de_binarias(cls, mascara, **kw):
        m = np.asarray(mascara, bool)
        return cls(int(m.sum()), len(m), **kw)

    @property
    def media(self):
        return float(self.amostras.mean())

    def ic(self, q=(2.5, 97.5)):
        return tuple(np.percentile(self.amostras, q))

    def __sub__(self, outra):
        """Diferença entre duas proporções INDEPENDENTES.

        Duas armadilhas, ambas verificadas aqui:

        1. Sementes iguais fazem as duas posteriores compartilharem o mesmo
           fluxo aleatório. A diferença cancela a variância e o intervalo sai
           artificialmente estreito — erro silencioso e difícil de notar.
        2. Para grupos PAREADOS (mesmas unidades avaliadas duas vezes), use
           ComparacaoPareada: descartar o pareamento infla a incerteza.
        """
        if self.seed == outra.seed:
            raise ValueError(
                "as duas ProporcaoBeta usam a mesma semente: as amostras ficam "
                "correlacionadas e o IC da diferença sai estreito demais. "
                "Passe sementes distintas, ex.: ProporcaoBeta(..., seed=11) e "
                "ProporcaoBeta(..., seed=12).")
        return self.amostras - outra.amostras

    def __repr__(self):
        lo, hi = self.ic()
        return f"{self.k}/{self.n} = {self.media:.3f} [{lo:.3f}; {hi:.3f}]"


# ══════════════════════════════════════════════════════ comparação pareada

class ComparacaoPareada:
    """Comparação bayesiana pareada de duas séries de escores.

    NÃO reimplementa o método: a posterior é a de Benavoli et al. (2017),
    obtida do `baycomp` quando a API expuser as amostras. Esta classe EXPÕE
    as amostras e deriva delas as quantidades que o pacote não oferece.

    ┌────────────────────────────────┬──────────────────────────────────────┐
    │ o baycomp já entrega           │ esta classe acrescenta               │
    ├────────────────────────────────┼──────────────────────────────────────┤
    │ posterior de Dirichlet         │ amostras acessíveis (theta, delta)   │
    │ triplet P(zona maioritária)    │ p_dom = P(θ_esq > θ_dir)             │
    │ gráfico do simplex             │ IC 95% de θ e de δ                   │
    │                                │ p_equiv(ε) em proporção de unidades  │
    │                                │ curva P(equiv) × ε sem re-amostrar   │
    │                                │ eps_critico: tamanho de efeito       │
    └────────────────────────────────┴──────────────────────────────────────┘

    Composição, não herança: o que muda não é COMO a posterior é calculada,
    e sim QUAIS quantidades se derivam dela. Herdar de `SignTest` exigiria
    redefinir `probs()` com outro contrato, quebrando a substituibilidade.
    """

    # rótulos das três zonas — usados em todo o reporte, definidos num só lugar
    ZONAS = ("esquerda", "empate", "direita")

    def __init__(self, x, y, rope=0.0, prior=1.0, prior_place="rope",
                 nsamples=None, seed=None, motor="auto"):
        nsamples = NSAMPLES_PADRAO if nsamples is None else nsamples
        seed = SEED_PADRAO if seed is None else seed
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        if self.x.shape != self.y.shape:
            raise ValueError("séries pareadas devem ter o mesmo tamanho")
        if np.isnan(self.x).any() or np.isnan(self.y).any():
            raise ValueError("NaN presente: aplique o descarte global pareado")

        self.rope, self.prior, self.prior_place = rope, prior, prior_place
        self.nsamples, self.seed, self.motor = nsamples, seed, motor

        d = self.x - self.y
        self.contagens = np.array([(d > rope).sum(),
                                   (np.abs(d) <= rope).sum(),
                                   (d < -rope).sum()])
        self.theta, self.origem = self._amostrar()
        self.delta = self.theta[:, 0] - self.theta[:, 2]

    # ------------------------------------------------------------- construtor
    @classmethod
    def de_binarias(cls, a_bin, b_bin, **kw):
        """A partir de duas máscaras booleanas pareadas (ex.: nota >= 3).

        As três zonas passam a ser: só A adequado / concordantes / só B
        adequado. É o análogo bayesiano do teste de McNemar, e preserva o
        pareamento — ao contrário de comparar duas ProporcaoBeta independentes.
        """
        return cls(np.asarray(a_bin, bool).astype(int),
                   np.asarray(b_bin, bool).astype(int), **kw)

    # ---------------------------------------------------------------- interno
    def _amostrar_proprio(self):
        alpha = np.zeros(3)
        alpha[{"left": 0, "rope": 1, "right": 2}[self.prior_place]] = self.prior
        rng = np.random.default_rng(self.seed)
        return rng.dirichlet(alpha + self.contagens, size=self.nsamples), "própria (Dirichlet)"

    def _amostrar(self):
        """Obtém as amostras da posterior.

        motor="auto"     usa o baycomp SE ele aceitar `prior`/`prior_place`;
                         caso contrário cai no gerador próprio, que os honra.
        motor="baycomp"  força o pacote, ainda que ignore esses parâmetros.
        motor="proprio"  força o gerador próprio (necessário para diagnósticos).

        Por que não usar sempre o baycomp: em versões que não expõem `prior`,
        `prior_place` e a semente, `sensibilidade_prior()` e `estabilidade_mc()`
        passam a medir apenas ruído Monte Carlo — devolvem números que variam,
        parecem um diagnóstico aprovado, e não testam nada.
        """
        if self.motor == "proprio":
            return self._amostrar_proprio()
        try:
            import baycomp
            from baycomp import SignTest
            lugar = {"left": getattr(baycomp, "LEFT", 0),
                     "rope": getattr(baycomp, "ROPE", 1),
                     "right": getattr(baycomp, "RIGHT", 2)}[self.prior_place]
            np.random.seed(self.seed)          # baycomp usa o RNG global legado
            try:
                s = np.asarray(SignTest.sample(
                    self.x, self.y, rope=self.rope, prior=self.prior,
                    prior_place=lugar, nsamples=self.nsamples))
                if s.ndim == 2 and s.shape[1] == 3:
                    return s, "baycomp.SignTest.sample (parâmetros honrados)"
            except TypeError:
                if self.motor == "auto":       # o pacote ignoraria prior/semente
                    return self._amostrar_proprio()
                np.random.seed(self.seed)
                s = np.asarray(SignTest.sample(self.x, self.y, rope=self.rope,
                                               nsamples=self.nsamples))
                if s.ndim == 2 and s.shape[1] == 3:
                    return s, "baycomp.SignTest.sample (prior/semente NÃO honrados)"
        except Exception:
            pass
        return self._amostrar_proprio()

    # ---------------------------------------------------------------- direção
    @property
    def n(self):
        return len(self.x)

    @property
    def p_dom(self):
        """P(θ_esquerda > θ_direita). Não degenera com rope = 0."""
        return float((self.theta[:, 0] > self.theta[:, 2]).mean())

    @property
    def triplet(self):
        """P(cada zona ser a MAIORITÁRIA) — quantidade devolvida pelo baycomp.

        Com rope = 0 e escala ordinal grosseira isto degenera em (0, 1, 0):
        p_rope mede apenas a frequência de empates, não equivalência.
        """
        return np.bincount(self.theta.argmax(axis=1), minlength=3) / self.nsamples

    # ----------------------------------------------------------- equivalência
    def p_equiv(self, eps):
        """P(|θ_esq − θ_dir| < ε). O ε é aplicado SOBRE a posterior."""
        return float((np.abs(self.delta) < eps).mean())

    def probabilidades_relacao(self, eps=None, modo="proporcao"):
        """Três probabilidades que somam 1, por um de dois caminhos distintos.

        A relação é sempre a da série da ESQUERDA em relação à da DIREITA, e a
        convenção é *maior é melhor*: se o escore for um erro, inverta os
        argumentos ao construir o objeto.

        ⚠ **As duas "equivalências" não são a mesma afirmação.** Os rótulos
        coincidem, o significado não — por isso o modo é gravado na matriz e
        impresso na figura, e por isso não existe padrão implícito seguro.

        ``modo="proporcao"`` — partição de δ = θ_sup − θ_inf pela margem ε:

        ┌──────────────┬────────────────┬────────────────────────────────────┐
        │ p_superior   │ P(δ > +ε)      │ a esquerda supera a direita        │
        │ p_equivalente│ P(|δ| ≤ ε)     │ a vantagem, MEDIDA EM PROPORÇÃO DE │
        │              │                │ DOCUMENTOS, não passa de ε         │
        │ p_inferior   │ P(δ < −ε)      │ a esquerda é superada pela direita │
        └──────────────┴────────────────┴────────────────────────────────────┘

        Afirmação sobre **magnitude do efeito**. É o modo da escala Likert, em
        que as diferenças são inteiras e ``rope = 0`` é o valor útil.

        ``modo="baycomp"`` — o triplet do pacote: P(cada zona ser a
        MAIORITÁRIA). Afirmação sobre **qual região concentra mais documentos**,
        não sobre magnitude. É o modo dos escores contínuos (F1), em que a ROPE
        carrega toda a margem — o ε não participa.

        A diferença é observável. Com 40% acima, 35% na ROPE e 25% abaixo, o
        modo baycomp devolve `superior` com alta probabilidade ainda que um
        terço dos documentos esteja dentro da margem de irrelevância; o modo
        proporção olharia δ = 0,15 e, com ε = 0,20, diria `equivalente`. Nenhum
        está errado: respondem perguntas diferentes.

        Raises:
            ValueError: modo desconhecido, ou ``modo="baycomp"`` com rope = 0 —
                nesse caso o triplet degenera em (0, 1, 0) e devolveria "tudo
                equivalente" com aparência de resultado.
        """
        if modo not in MODOS:
            raise ValueError(f"modo desconhecido: {modo!r}; use um de {MODOS}")

        if modo == "baycomp":
            if self.rope == 0:
                raise ValueError(
                    "modo='baycomp' exige rope > 0: com rope = 0 a zona central "
                    "recolhe apenas os empates exatos e o triplet degenera em "
                    "(0, 1, 0) — 'equivalente' em todas as células, sem que isso "
                    "signifique equivalência. Para escala ordinal use "
                    "modo='proporcao'; para escores contínuos, defina a ROPE.")
            if eps is not None:
                warnings.warn(
                    "modo='baycomp' não usa ε: a única margem é a ROPE, aplicada "
                    "sobre os escores brutos. O ε informado foi ignorado.",
                    stacklevel=2)
            t = self.triplet          # (esquerda, empate, direita) = (sup, rope, inf)
            return {"p_inferior": float(t[2]), "p_equivalente": float(t[1]),
                    "p_superior": float(t[0])}

        eps = 0.0 if eps is None else float(eps)
        d = self.delta
        return {"p_inferior": float((d < -eps).mean()),
                "p_equivalente": float((np.abs(d) <= eps).mean()),
                "p_superior": float((d > eps).mean())}

    @property
    def massa_empate(self):
        """Fração de unidades na zona de empate — contexto obrigatório para ler ε.

        δ fica limitado a ±(1 − massa_empate). Com 70% de empates, δ ∈ ±0,30 e
        um ε de 0,08 já consome mais de um quarto da faixa disponível. Sem este
        número, o valor de ε é ininterpretável: não dá para saber se é margem
        apertada ou frouxa.
        """
        return float(self.contagens[1] / self.n)

    def curva_eps(self, grade=None):
        """Curva P(equivalência) × ε. Mesma posterior, sem re-amostrar."""
        grade = np.linspace(0, 0.30, 61) if grade is None else np.asarray(grade)
        return grade, np.array([self.p_equiv(e) for e in grade])

    def eps_critico(self, limiar=0.95):
        """Menor ε que estabelece equivalência — medida de tamanho de efeito."""
        g, p = self.curva_eps()
        acima = np.flatnonzero(p >= limiar)
        return float(g[acima[0]]) if acima.size else float("nan")

    # --------------------------------------------------------------- reporte
    def ic_delta(self, q=(2.5, 97.5)):
        return tuple(np.percentile(self.delta, q))

    def ic_theta(self, zona=0, q=(2.5, 97.5)):
        return tuple(np.percentile(self.theta[:, zona], q))

    def veredito(self, eps, limiar=0.95):
        if self.p_equiv(eps) >= limiar:
            return "equivalência"
        if self.p_dom >= limiar or self.p_dom <= 1 - limiar:
            return "direção"
        return "inconclusivo"

    def como_dict(self, eps, nomes=("A", "B")):
        """Linha pronta para compor um DataFrame de contrastes."""
        a, b = nomes
        lo, hi = self.ic_delta()
        return {"contraste": f"{a} × {b}",
                f"{self.ZONAS[0]}": int(self.contagens[0]),
                f"{self.ZONAS[1]}": int(self.contagens[1]),
                f"{self.ZONAS[2]}": int(self.contagens[2]),
                "Δ dom": self.delta.mean(), "IC 95%": f"[{lo:+.3f}; {hi:+.3f}]",
                "P(dom)": self.p_dom, "P(equiv)": self.p_equiv(eps),
                "ε crít.": self.eps_critico(), "veredito": self.veredito(eps)}

    def resumo(self, eps, nomes=("A", "B")):
        a, b = nomes
        c, n = self.contagens, self.n
        lo, hi = self.ic_delta()
        rot = (f"{a}>{b}", "empate", f"{b}>{a}")
        linha = " | ".join(f"{r}: {v} ({100*v/n:.1f}%)" for r, v in zip(rot, c))
        return (f"{a} × {b}  (n={n}, posterior: {self.origem})\n"
                f"  {linha}\n"
                f"  Δ dominância = {self.delta.mean():+.4f} [{lo:+.4f}; {hi:+.4f}]\n"
                f"  P(dominância) = {self.p_dom:.4f}   "
                f"P(equiv | ε={eps}) = {self.p_equiv(eps):.4f}   "
                f"ε crítico = {self.eps_critico():.3f}\n"
                f"  veredito: {self.veredito(eps).upper()}")

    # --------------------------------------------------------- diagnósticos
    def _clone(self, **mudancas):
        kw = dict(rope=self.rope, prior=self.prior, prior_place=self.prior_place,
                  nsamples=self.nsamples, seed=self.seed, motor=self.motor)
        kw.update(mudancas)
        return ComparacaoPareada(self.x, self.y, **kw)

    def _clone_diag(self, **mudancas):
        """Clone para diagnósticos: sempre no gerador próprio, que honra
        `prior_place` e a semente. Sem isso o diagnóstico mede só ruído."""
        return self._clone(motor="proprio", **mudancas)

    def sensibilidade_prior(self, eps=None):
        """A conclusão não pode depender de onde o prior é alocado."""
        out = {}
        for p in ("left", "rope", "right"):
            c = self._clone_diag(prior_place=p)
            out[p] = {"P(dom)": c.p_dom} | ({} if eps is None else {"P(equiv)": c.p_equiv(eps)})
        return out

    def estabilidade_mc(self, sementes=(1, 7, 42, 2026)):
        """Variação entre sementes deve ficar na 3ª casa decimal."""
        v = [self._clone_diag(seed=s).p_dom for s in sementes]
        return dict(zip(sementes, v)) | {"amplitude": max(v) - min(v)}

    def autoteste(self):
        """Verifica se os diagnósticos são reais ou apenas ruído.

        O ponto decisivo é o CONTROLE NEGATIVO: dois clones com parâmetros
        IDÊNTICOS. Se ainda assim diferirem, o motor de amostragem não honra a
        semente — e então `prior_place` e `seed` "terem efeito" não significa
        nada, porque tudo varia a cada chamada.
        """
        igual = self._clone(motor=self.motor)                       # controle negativo
        reprodutivel = np.allclose(self.theta[:200], igual.theta[:200], atol=1e-12)

        muda_prior = not np.allclose(
            self.theta.mean(axis=0),
            self._clone(prior_place="left", prior=self.n / 4).theta.mean(axis=0), atol=1e-6)
        muda_seed = not np.allclose(
            self.theta[:50], self._clone(seed=self.seed + 1).theta[:50], atol=1e-12)

        return {"posterior": self.origem,
                "reprodutível (mesma semente → mesmas amostras)": reprodutivel,
                "prior_place tem efeito": muda_prior and reprodutivel,
                "semente tem efeito": muda_seed and reprodutivel,
                "contraste saturado (teste não conclui)": self.p_dom in (0.0, 1.0),
                "diagnósticos confiáveis": bool(reprodutivel and muda_prior and muda_seed),
                "obs": ("OK" if reprodutivel else
                        "motor não reprodutível: sensibilidade_prior() e estabilidade_mc() "
                        "usam motor='proprio' e permanecem válidas; só o resultado "
                        "principal vem do baycomp")}

    def comparar_motores(self, eps=0.05):
        """Valida o gerador próprio contra o baycomp nos MESMOS dados.

        Melhor que comparar triplets: funciona com rope = 0 (onde o
        `SignTest.probs` devolve nan) e confronta exatamente as quantidades
        que a análise reporta. Divergências devem ficar no ruído Monte Carlo.
        """
        try:
            import baycomp  # noqa: F401
        except ImportError:
            return {"erro": "baycomp não instalado (pip install baycomp)"}
        b = self._clone(motor="baycomp")
        pr = self._clone(motor="proprio")
        linhas = {}
        for nome, fn in [("p_dom", lambda c: c.p_dom),
                         ("Δ dominância", lambda c: c.delta.mean()),
                         (f"P(equiv|ε={eps})", lambda c: c.p_equiv(eps)),
                         ("θ_esq médio", lambda c: c.theta[:, 0].mean())]:
            vb, vp = float(fn(b)), float(fn(pr))     # float puro: evita np.float64 no repr
            linhas[nome] = {"baycomp": round(vb, 5), "próprio": round(vp, 5),
                            "|dif|": round(abs(vb - vp), 5)}
        dif_max = max(v["|dif|"] for v in linhas.values())
        linhas["veredito"] = ("equivalentes (ruído Monte Carlo)" if dif_max < 0.01
                              else "DIVERGEM — investigar")
        linhas["origem baycomp"] = b.origem
        return linhas

    def validar_contra_baycomp(self):
        """Confere o triplet contra o pacote. Exige rope > 0."""
        try:
            from baycomp import SignTest
        except ImportError:
            return "baycomp não instalado (pip install baycomp)"
        if self.rope == 0:
            return ("validação exige rope > 0: com rope = 0 o baycomp calcula "
                    "pl/(pl+pr) = 0/0 e devolve nan")
        ref = np.asarray(SignTest.probs(self.x, self.y, rope=self.rope,
                                        nsamples=self.nsamples), float)
        meu = self.triplet
        if ref[0] > ref[2] and meu[0] < meu[2]:
            ref = ref[::-1]                       # convenções de lado invertidas
        return (f"baycomp    : ({ref[0]:.4f}, {ref[1]:.4f}, {ref[2]:.4f})\n"
                f"esta classe: ({meu[0]:.4f}, {meu[1]:.4f}, {meu[2]:.4f})\n"
                f"divergência máxima: {np.abs(ref - meu).max():.4f}")

    # ---------------------------------------------------------------- gráfico
    def plotar_curva_eps(self, ax=None, eps_ref=None, rotulo=None, limiar=0.95):
        """Curva P(equivalência) × ε. Aceita `ax` para sobrepor contrastes."""
        import matplotlib.pyplot as plt
        criar = ax is None
        if criar:
            _, ax = plt.subplots(figsize=(9, 4.5))
        g, p = self.curva_eps()
        ax.plot(g, p, lw=2, label=rotulo)
        if criar:
            ax.axhline(limiar, color="0.35", ls="--", lw=1)
            if eps_ref is not None:
                ax.axvline(eps_ref, color="0.35", ls=":", lw=1)
                ax.annotate(f"ε pré-registrado = {eps_ref}", xy=(eps_ref, 1.0),
                            xytext=(eps_ref + 0.012, 0.72), fontsize=9, color="0.30",
                            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.8))
            ax.set_xlabel("ε (proporção de unidades)")
            ax.set_ylabel("P(equivalência prática)")
            ax.set_ylim(-0.02, 1.05)
            ax.grid(alpha=0.25)
        return ax

    def __repr__(self):
        c = self.contagens
        return (f"<ComparacaoPareada n={self.n} rope={self.rope} "
                f"zonas={dict(zip(self.ZONAS, c.tolist()))} p_dom={self.p_dom:.3f}>")


# ══════════════════════════════════════════════════════ matriz de relações

#: as três relações possíveis entre dois protocolos, na ordem do reporte
RELACOES = ("inferior", "equivalente", "superior")

#: cor de cada estado visual do heatmap
CORES_RELACAO = {
    "superior":    "#2e7d4f",   # verde
    "equivalente": "#2c6e91",   # azul
    "inferior":    "#b23a48",   # vermelho
    "incerto":     "#8a8a8a",   # cinza
}


def classificar_relacao(p_inferior, p_equivalente, p_superior,
                        limiar=None) -> tuple:
    """Classifica uma célula a partir das três probabilidades posteriores.

    Devolve ``(classificacao, probabilidade)``, onde a probabilidade é sempre a
    **dominante** — inclusive no estado `incerto`, em que ela informa quão perto
    do limiar a evidência chegou.

    `incerto` é categoria explícita, não uma quarta relação: significa que
    nenhuma das três alcançou o limiar, e não que exista alguma relação
    intermediária entre elas. Empates exatos resolvem para `incerto`, que é a
    leitura neutra — nunca se infere uma classificação que as probabilidades não
    sustentam.
    """
    limiar = LIMIAR_CLASSIFICACAO_PADRAO if limiar is None else float(limiar)
    valores = {"inferior": float(p_inferior),
               "equivalente": float(p_equivalente),
               "superior": float(p_superior)}
    dominante = max(valores, key=valores.get)
    maximo = valores[dominante]
    empatadas = [k for k, v in valores.items() if abs(v - maximo) < 1e-12]
    if maximo >= limiar and len(empatadas) == 1:
        return dominante, maximo
    return "incerto", maximo


def matriz_relacoes(dados, nomes=None, eps=0.0, rope=0.0, limiar=None,
                    modo="proporcao", metrica=None, papel=None, **kw):
    """Comparação bayesiana pareada de todos os pares, em formato longo.

    Args:
        dados: DataFrame com uma coluna por protocolo, ou dicionário
            ``{nome: sequência de escores}``. As séries são **pareadas**: a
            linha *i* de todas elas é o mesmo caso de teste.
        nomes: subconjunto e ordem dos protocolos (padrão: todos, na ordem em
            que aparecem). **Trave esta ordem** quando for gerar mais de uma
            matriz para comparar visualmente: heatmaps com linhas em ordens
            diferentes induzem leitura errada.
        eps: margem sobre a posterior, em proporção de unidades. Só tem efeito
            no modo `proporcao`.
        rope: margem sobre os **escores brutos**. Obrigatória (> 0) no modo
            `baycomp`; na escala Likert use 0.
        limiar: corte de decisão da classificação (padrão 0,80).
        modo: `proporcao` ou `baycomp` — ver ``probabilidades_relacao``.
        metrica: rótulo da métrica ("Likert", "BERTScore F1"), impresso na
            figura. Não afeta o cálculo, evita confundir figuras fora de
            contexto.
        papel: `principal` ou `complementar`, também só para o reporte.
        **kw: repassados a ``ComparacaoPareada`` (``prior``, ``prior_place``,
            ``nsamples``, ``seed``, ``motor``).

    Returns:
        DataFrame com uma linha por par ORDENADO (i, j), i ≠ j. As colunas
        `p_*` são lidas como "a relação de `linha` em relação a `coluna`".
        Os parâmetros da execução ficam em ``matriz.attrs`` e são lidos pelo
        heatmap — preserve-os ao filtrar ou serializar a matriz.

    Cada par não ordenado é amostrado **uma única vez**: a célula espelhada é
    derivada trocando `p_inferior` com `p_superior`. Isso garante a simetria
    exigida — P(Pi > Pj) é o mesmo número que P(Pj < Pi), não uma segunda
    estimativa Monte Carlo aproximada — e reduz o custo à metade.
    """
    import pandas as pd

    if modo not in MODOS:
        raise ValueError(f"modo desconhecido: {modo!r}; use um de {MODOS}")
    if modo == "proporcao" and float(eps) == 0.0:
        warnings.warn(
            "modo='proporcao' com ε = 0: a faixa de equivalência tem medida nula "
            "e nenhuma célula poderá ser classificada como equivalente. Informe "
            "um ε calibrado.", stacklevel=2)

    if hasattr(dados, "columns"):
        disponiveis = list(dados.columns)
    else:
        disponiveis = list(dados.keys())
    serie = lambda nome: np.asarray(dados[nome], float)
    nomes = list(disponiveis if nomes is None else nomes)
    faltando = [n for n in nomes if n not in disponiveis]
    if faltando:
        raise KeyError(f"protocolos ausentes nos dados: {faltando}")
    if len(nomes) < 2:
        raise ValueError("a matriz de relações exige ao menos dois protocolos")

    linhas = []
    for i, a in enumerate(nomes):
        for b in nomes[i + 1:]:
            cmp = ComparacaoPareada(serie(a), serie(b), rope=rope, **kw)
            p = cmp.probabilidades_relacao(eps if modo == "proporcao" else None,
                                           modo=modo)
            lo, hi = cmp.ic_delta()
            comum = {"n": cmp.n, "rope": cmp.rope,
                     "eps": float(eps) if modo == "proporcao" else float("nan"),
                     "modo": modo, "massa_empate": cmp.massa_empate,
                     "delta": float(cmp.delta.mean()), "ic_inf": float(lo),
                     "ic_sup": float(hi), "eps_critico": cmp.eps_critico(),
                     "origem": cmp.origem}
            # a célula espelhada é DERIVADA, não reamostrada: garante simetria exata
            for esquerda, direita, virar in ((a, b, False), (b, a, True)):
                p_inf = p["p_superior"] if virar else p["p_inferior"]
                p_sup = p["p_inferior"] if virar else p["p_superior"]
                classe, dominante = classificar_relacao(
                    p_inf, p["p_equivalente"], p_sup, limiar)
                linhas.append({
                    "linha": esquerda, "coluna": direita, **comum,
                    "delta": -comum["delta"] if virar else comum["delta"],
                    "ic_inf": -comum["ic_sup"] if virar else comum["ic_inf"],
                    "ic_sup": -comum["ic_inf"] if virar else comum["ic_sup"],
                    "p_dom": 1 - cmp.p_dom if virar else cmp.p_dom,
                    "p_inferior": p_inf, "p_equivalente": p["p_equivalente"],
                    "p_superior": p_sup,
                    "classificacao": classe, "probabilidade": dominante,
                    "contagem_superior": int(cmp.contagens[2] if virar else cmp.contagens[0]),
                    "contagem_empate": int(cmp.contagens[1]),
                    "contagem_inferior": int(cmp.contagens[0] if virar else cmp.contagens[2]),
                })

    matriz = pd.DataFrame(linhas)
    # attrs alimenta o heatmap; pandas descarta attrs em algumas operações,
    # então quem filtrar a matriz precisa recopiá-los
    matriz.attrs.update(
        nomes=nomes, eps=float(eps) if modo == "proporcao" else None,
        rope=float(rope), modo=modo, metrica=metrica, papel=papel,
        n=int(matriz["n"].iloc[0]) if len(matriz) else 0,
        massa_empate=float(matriz["massa_empate"].mean()) if len(matriz) else float("nan"),
        limiar=LIMIAR_CLASSIFICACAO_PADRAO if limiar is None else float(limiar))
    return matriz


def resumo_relacoes(matriz, rotulo_entidade="protocolo"):
    """Quantas vezes cada entidade é superior, equivalente, inferior ou incerta.

    Não produz ranking: com relações possivelmente intransitivas, ordenar por
    contagem de vitórias criaria uma ordem que os dados não sustentam. A tabela
    serve para localizar rapidamente quem é superior a vários, quem é
    predominantemente equivalente e onde a evidência não fecha.

    Args:
        rotulo_entidade: nome da entidade comparada — "protocolo", "avaliador",
            "fonte" etc. Usado como nome do índice da tabela retornada.
    """
    import pandas as pd
    nomes = matriz.attrs.get("nomes") or sorted(matriz["linha"].unique())
    ordem = ["superior", "equivalente", "inferior", "incerto"]
    contagens = (matriz.groupby(["linha", "classificacao"]).size()
                 .unstack(fill_value=0)
                 .reindex(index=nomes, columns=ordem, fill_value=0))
    contagens.index.name = rotulo_entidade
    contagens.columns.name = None
    return contagens.rename(columns={"superior": "superior a",
                                     "equivalente": "equivalente a",
                                     "inferior": "inferior a"})


# ══════════════════════════════════════════════════════ heatmap


def _cor_celula(classificacao, probabilidade):
    """Cor da célula: categoria define o matiz, probabilidade define a saturação.

    A escala de saturação começa em 1/3 (probabilidade de uma relação sorteada
    ao acaso entre as três) e não em 0, para que a diferença visual entre 0,74 e
    0,95 seja perceptível — que é a faixa onde a leitura acontece.
    """
    from matplotlib.colors import to_rgb
    cor = np.array(to_rgb(CORES_RELACAO[classificacao]))
    fracao = float(np.clip((float(probabilidade) - 1 / 3) / (1 - 1 / 3), 0.0, 1.0))
    # o cinza do estado `incerto` fica deliberadamente lavado: incerteza não deve
    # competir visualmente com as relações que a evidência sustenta
    mistura = (0.10 + 0.35 * fracao) if classificacao == "incerto" else (0.16 + 0.84 * fracao)
    return tuple(1.0 - mistura * (1.0 - cor)), mistura


#: Rótulo da faixa central em cada modo. São afirmações DIFERENTES e os nomes
#: precisam refletir isso na figura — ver `probabilidades_relacao`.
ROTULO_CENTRAL = {"proporcao": "equivalente", "baycomp": "ROPE maioritária"}


def _rotulos_legenda(modo, limiar):
    """Texto de cada estado visual, adaptado ao modo."""
    return {"superior": "superior",
            "equivalente": ROTULO_CENTRAL.get(modo, "equivalente"),
            "inferior": "inferior",
            "incerto": f"incerto (< {limiar:.2f})".replace(".", ",")}


def _subtitulo_padrao(atributos, limiar):
    """Contexto sem o qual os números da figura não se interpretam.

    A margem e a massa de empates entram aqui de propósito: ε = 0,08 é apertado
    ou frouxo dependendo de quanta massa sobra fora da zona de empate, e sem o
    modo declarado as duas 'equivalências' se confundem.
    """
    modo = atributos.get("modo", "proporcao")
    if modo == "baycomp":
        explicacao = "cor = zona posterior maioritária (cálculo padrão do baycomp)"
        margem = (f"ROPE = {atributos['rope']:.4g}".replace(".", ",")
                  if atributos.get("rope") is not None else None)
    else:
        explicacao = "cor = relação da linha em relação à coluna"
        margem = (f"ε = {atributos['eps']:.4g} (proporção de documentos)".replace(".", ",")
                  if atributos.get("eps") is not None else None)

    parametros = [margem]
    if atributos.get("n"):
        parametros.append(f"n = {atributos['n']:,}".replace(",", "."))
    massa = atributos.get("massa_empate")
    if massa is not None and massa == massa:
        parametros.append(f"empates = {100 * massa:.0f}%")
    parametros.append(f"limiar = {limiar:.2f}".replace(".", ","))
    return (f"{explicacao} · número = probabilidade posterior\n"
            + " · ".join(p for p in parametros if p))


def heatmap_relacoes(matriz, arquivo_saida=None, titulo=None, subtitulo=None,
                     nomes=None, limiar=None, eixo=None, casas=1,
                     legenda=True, figsize=None, dpi=150, rotacao_x=0,
                     referencia=None, rotulo_entidade="protocolo"):
    """Heatmap categórico + quantitativo das relações entre protocolos.

    A cor comunica a **categoria**, a saturação comunica a **magnitude** da
    evidência e o número impresso na célula traz a probabilidade posterior
    dominante. A diagonal é neutra: `(Pi, Pi)` não é uma comparação.

    Os rótulos se adaptam ao modo gravado em ``matriz.attrs``: no modo
    `baycomp` a faixa central vira "ROPE maioritária", porque ali a afirmação é
    sobre qual zona concentra mais documentos, e não sobre magnitude.

    Args:
        matriz: saída de ``matriz_relacoes`` ou dicionário
            ``{(linha, coluna): (p_inferior, p_equivalente, p_superior)}``.
        arquivo_saida: caminho do PNG; quando ``None``, apenas devolve o eixo.
        limiar: corte de decisão; padrão vem de ``matriz.attrs`` ou 0,80.
        eixo: eixo matplotlib existente, para compor figuras maiores.
        casas: casas decimais do percentual impresso na célula.
        referencia: nome do grupo de referência (aparece em negrito nos eixos e
            em anotação no subtítulo). ``None`` desativa essa marcação.
        rotulo_entidade: nome da entidade comparada, usado no ylabel e no título
            automático. Ex.: ``"protocolo"``, ``"avaliador"``, ``"fonte"``.

    Returns:
        ``(eixo, caminho)`` — `caminho` é ``None`` se nada foi gravado.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    celulas, atributos = _normalizar_entrada_heatmap(matriz)
    nomes = list(nomes or atributos.get("nomes") or
                 dict.fromkeys([n for par in celulas for n in par]))
    limiar = (limiar if limiar is not None
              else atributos.get("limiar", LIMIAR_CLASSIFICACAO_PADRAO))
    modo = atributos.get("modo", "proporcao")
    k = len(nomes)

    # max_chars é usado tanto no cálculo de altura quanto no posicionamento da
    # legenda — precisa estar disponível mesmo quando o eixo é passado externamente
    max_chars = max((len(str(n)) for n in nomes), default=0)

    criar = eixo is None
    if criar:
        largura = max(6.0, 1.35 * k + 3.0)
        # reserva espaço extra no eixo inferior para rótulos longos e legenda;
        # max_chars estima a altura projetada dos rótulos rotacionados
        extra_rotulo = max(0.0, (max_chars - 6) * 0.08) if rotacao_x else 0.0
        extra_legenda = 1.0 if legenda else 0.3
        altura = max(4.4, 1.05 * k + 1.9 + extra_rotulo + extra_legenda)
        figura, eixo = plt.subplots(figsize=figsize or (largura, altura))
    else:
        figura = eixo.figure

    eixo.set_xlim(-0.5, k - 0.5)
    eixo.set_ylim(k - 0.5, -0.5)
    eixo.set_xticks(range(k), nomes, fontsize=9, rotation=rotacao_x,
                    ha="center" if not rotacao_x else "right")
    eixo.set_yticks(range(k), nomes, fontsize=9)
    # destaque em negrito para o grupo de referência nos dois eixos
    if referencia is not None:
        for tick in eixo.get_xticklabels():
            if tick.get_text() == referencia:
                tick.set_fontweight("bold")
        for tick in eixo.get_yticklabels():
            if tick.get_text() == referencia:
                tick.set_fontweight("bold")
    eixo.set_aspect("equal")
    eixo.grid(visible=False)
    for lado in ("top", "right", "bottom", "left"):
        eixo.spines[lado].set_visible(False)
    eixo.tick_params(length=0)

    for i, a in enumerate(nomes):
        for j, b in enumerate(nomes):
            cor_texto = "#22333b"   # padrão escuro
            if i == j:
                cor, mistura, rotulo = np.array([0.93, 0.93, 0.93]), 0.0, "—"
            else:
                dados = celulas.get((a, b))
                if dados is None:
                    cor, mistura, rotulo = np.array([1.0, 1.0, 1.0]), 0.0, "n/d"
                else:
                    classe, probabilidade = dados["classificacao"], dados["probabilidade"]
                    cor, mistura = _cor_celula(classe, probabilidade)
                    rotulo = f"{100 * probabilidade:.{casas}f}%".replace(".", ",")
                    # em células inconclusivas, o texto indica a relação dominante
                    if classe == "incerto":
                        p_inf = dados.get("p_inferior", 0.0)
                        p_eq  = dados.get("p_equivalente", 0.0)
                        p_sup = dados.get("p_superior", 0.0)
                        dominante_valor = max(p_inf, p_eq, p_sup)
                        # empate: diferença < 0,01 entre os dois maiores
                        valores_ord = sorted([p_inf, p_eq, p_sup], reverse=True)
                        empate = (valores_ord[0] - valores_ord[1]) < 0.01
                        if not empate:
                            if dominante_valor == p_sup:
                                cor_texto = "#2e7d32"   # verde escuro — superior
                            elif dominante_valor == p_inf:
                                cor_texto = "#b71c1c"   # vermelho escuro — inferior
                            else:
                                cor_texto = "#1565c0"   # azul escuro — equivalente
                        # empate verdadeiro: mantém cor padrão (#22333b)
            eixo.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cor,
                                     edgecolor="white", linewidth=1.6))
            # células coloridas (não-inconclusivas): branco se escuro, escuro se claro
            if mistura > 0.62 and cor_texto == "#22333b":
                cor_texto = "white"
            eixo.text(j, i, rotulo, ha="center", va="center", fontsize=9.5,
                      color=cor_texto)

    eixo.set_xlabel("comparado com", fontsize=10, labelpad=8)
    eixo.set_ylabel(rotulo_entidade, fontsize=10, labelpad=8)
    # anotação textual da referência — aparece antes do subtítulo técnico
    # (negrito nos tick labels já foi aplicado acima; aqui só o texto)
    _ref_nota = (f"referência: {referencia}" if referencia is not None else None)
    detalhe = subtitulo if subtitulo is not None else _subtitulo_padrao(atributos, limiar)

    # métrica e papel identificam a figura fora de contexto: sem isso, o heatmap
    # da métrica complementar pode ser lido como se fosse o da principal
    metrica, papel = atributos.get("metrica"), atributos.get("papel")
    if titulo is None and metrica:
        titulo = f"Comparação bayesiana entre {rotulo_entidade}s — {metrica}"
    if metrica or papel:
        identificacao = " · ".join(filter(None, [
            f"métrica: {metrica}" if metrica else None,
            f"análise {papel}" if papel else None]))
        detalhe = f"{identificacao}\n{detalhe}" if detalhe else identificacao

    # referência entra no subtítulo logo após a identificação da métrica
    if _ref_nota:
        detalhe = f"{_ref_nota}\n{detalhe}" if detalhe else _ref_nota

    if detalhe:
        eixo.annotate(detalhe, xy=(0.5, 1.0), xycoords="axes fraction",
                      xytext=(0, 10), textcoords="offset points",
                      ha="center", va="bottom", fontsize=8.5, color="0.35")
    if titulo:
        eixo.set_title(titulo, fontsize=12,
                       pad=14 + (12 * (detalhe.count("\n") + 1) if detalhe else 0))

    if legenda:
        rotulos = _rotulos_legenda(modo, limiar)
        # empurra a legenda para baixo quando os rótulos são rotacionados, para
        # não sobrepor o label "comparado com"; -0.10 cobre o caso sem rotação
        legenda_y = -0.10 - (0.008 * max_chars if rotacao_x else 0)
        eixo.legend(handles=[Patch(facecolor=CORES_RELACAO[c], label=r)
                             for c, r in rotulos.items()],
                    loc="upper center", bbox_to_anchor=(0.5, legenda_y),
                    ncol=4, frameon=False, fontsize=9, handlelength=1.2)

    caminho = None
    if arquivo_saida:
        figura.tight_layout()
        figura.savefig(arquivo_saida, dpi=dpi, bbox_inches="tight", facecolor="white")
        caminho = arquivo_saida
        if criar:
            plt.close(figura)
    return eixo, caminho


def _normalizar_entrada_heatmap(matriz):
    """Aceita o DataFrame de ``matriz_relacoes`` ou um dicionário de triplas."""
    if hasattr(matriz, "columns"):
        celulas = {(linha["linha"], linha["coluna"]): dict(linha)
                   for _, linha in matriz.iterrows()}
        return celulas, dict(matriz.attrs)

    celulas = {}
    for (a, b), valores in matriz.items():
        if isinstance(valores, dict):
            p_inf = valores["p_inferior"]
            p_eq = valores["p_equivalente"]
            p_sup = valores["p_superior"]
        else:
            p_inf, p_eq, p_sup = valores
        classe, dominante = classificar_relacao(p_inf, p_eq, p_sup)
        celulas[(a, b)] = {"p_inferior": p_inf, "p_equivalente": p_eq,
                           "p_superior": p_sup, "classificacao": classe,
                           "probabilidade": dominante}
    return celulas, {}


def heatmap_comparacao(dados, nomes=None, eps=0.0, rope=0.0, limiar=None,
                       modo="proporcao", metrica=None, papel=None,
                       arquivo_saida=None, titulo=None, subtitulo=None,
                       casas=1, legenda=True, figsize=None, dpi=150,
                       rotacao_x=0, referencia=None, rotulo_entidade="protocolo",
                       **kw):
    """Atalho de uma chamada: calcula a matriz e desenha o heatmap.

    É o ponto de entrada pensado para ser alimentado por configuração — as
    chaves de um YAML (comparação entre protocolos) ou as flags da linha de
    comando (avaliação LLM × humanos) mapeiam diretamente nos argumentos
    ``modo``, ``eps``, ``rope``, ``limiar``, ``nsamples`` e ``seed``.

    Returns:
        ``(matriz, caminho)``.
    """
    matriz = matriz_relacoes(dados, nomes=nomes, eps=eps, rope=rope,
                             limiar=limiar, modo=modo, metrica=metrica,
                             papel=papel, **kw)
    _, caminho = heatmap_relacoes(matriz, arquivo_saida=arquivo_saida,
                                  titulo=titulo, subtitulo=subtitulo,
                                  limiar=limiar, casas=casas, legenda=legenda,
                                  figsize=figsize, dpi=dpi, rotacao_x=rotacao_x,
                                  referencia=referencia,
                                  rotulo_entidade=rotulo_entidade)
    return matriz, caminho


# ══════════════════════════════════════════════════════ leitura conjunta

def _pares_unicos(matriz):
    """Itera uma linha por par NÃO ordenado, preservando a ordem de `nomes`."""
    vistos = set()
    for _, linha in matriz.iterrows():
        chave = frozenset((linha["linha"], linha["coluna"]))
        if chave not in vistos:
            vistos.add(chave)
            yield linha


def tabela_convergencia(matriz_principal, matriz_complementar,
                        rotulos=("principal", "complementar")):
    """Confronta as duas métricas par a par, sem presumir equivalência entre elas.

    A Likert mede qualidade percebida; o F1 mede fidelidade de extração. As duas
    podem capturar propriedades distintas do desempenho, e divergir é **achado**,
    não inconsistência — mas isso só se sustenta se a divergência for reportada
    explicitamente, e não deixada para o leitor extrair comparando duas figuras
    a olho.

    A coluna `situação` distingue três casos:

    * `convergente` — as duas métricas classificam o par da mesma forma;
    * `divergente` — classificações conflitantes (uma superior, outra inferior
      ou equivalente); é o caso que merece discussão no texto;
    * `sem decisão` — ao menos uma das métricas ficou `incerto`, e portanto não
      há o que convergir ou divergir.

    Args:
        matriz_principal: saída de ``matriz_relacoes`` da métrica primária.
        matriz_complementar: idem, da métrica complementar. Os pares precisam
            coincidir; pares presentes em apenas uma matriz são ignorados e
            contados no atributo ``ignorados`` do resultado.
        rotulos: nomes das colunas, para o relatório.
    """
    import pandas as pd

    a_rot, b_rot = rotulos
    complementar = {frozenset((l["linha"], l["coluna"])): l
                    for l in _pares_unicos(matriz_complementar)}

    linhas, ignorados = [], 0
    for principal in _pares_unicos(matriz_principal):
        chave = frozenset((principal["linha"], principal["coluna"]))
        outro = complementar.get(chave)
        if outro is None:
            ignorados += 1
            continue
        # a matriz complementar pode ter o par na ordem espelhada; realinha
        if outro["linha"] != principal["linha"]:
            outro = matriz_complementar[
                (matriz_complementar["linha"] == principal["linha"])
                & (matriz_complementar["coluna"] == principal["coluna"])].iloc[0]

        classe_a, classe_b = principal["classificacao"], outro["classificacao"]
        if "incerto" in (classe_a, classe_b):
            situacao = "sem decisão"
        elif classe_a == classe_b:
            situacao = "convergente"
        else:
            situacao = "divergente"

        linhas.append({
            "Par (A × B)": f"{principal['linha']} × {principal['coluna']}",
            f"Relação ({a_rot})": classe_a,
            f"P ({a_rot})": round(float(principal["probabilidade"]), 4),
            f"Relação ({b_rot})": classe_b,
            f"P ({b_rot})": round(float(outro["probabilidade"]), 4),
            "Situação": situacao,
        })

    tabela = pd.DataFrame(linhas)
    tabela.attrs.update(
        ignorados=ignorados,
        modo_principal=matriz_principal.attrs.get("modo"),
        modo_complementar=matriz_complementar.attrs.get("modo"),
        metrica_principal=matriz_principal.attrs.get("metrica"),
        metrica_complementar=matriz_complementar.attrs.get("metrica"))
    return tabela


# ══════════════════════════════════════════════════════ análises de sensibilidade

def sensibilidade_limiar(matriz, limiares=(0.70, 0.80, 0.90), referencia=None):
    """Quantas células mudam de categoria ao variar o limiar de decisão.

    Custo desprezível: a posterior **já está amostrada** e o limiar só reclassifica
    números prontos. Não confundir com ``sensibilidade_margem``, que mexe na
    margem e pode exigir reamostragem.

    Args:
        referencia: limiar operacional contra o qual as mudanças são contadas
            (padrão: o gravado em ``matriz.attrs``).
    """
    import pandas as pd

    referencia = referencia if referencia is not None else matriz.attrs.get(
        "limiar", LIMIAR_CLASSIFICACAO_PADRAO)
    pares = list(_pares_unicos(matriz))

    def classificar(limiar):
        return [classificar_relacao(l["p_inferior"], l["p_equivalente"],
                                    l["p_superior"], limiar)[0] for l in pares]

    base = classificar(referencia)
    linhas = []
    for limiar in sorted({*limiares, referencia}):
        classes = classificar(limiar)
        linhas.append({
            "Limiar": limiar,
            "superior": classes.count("superior"),
            "equivalente": classes.count("equivalente"),
            "inferior": classes.count("inferior"),
            "incerto": classes.count("incerto"),
            "Muda vs. referência": sum(x != y for x, y in zip(classes, base)),
            "Referência": "sim" if limiar == referencia else "não",
        })
    tabela = pd.DataFrame(linhas)
    tabela.attrs.update(referencia=referencia, pares=len(pares),
                        modo=matriz.attrs.get("modo"))
    return tabela


def grafico_curva_sensibilidade_eps(
        matriz, arquivo_saida=None, eps_ref=None, limiar_linha=0.95,
        titulo=None, figsize=(9, 4.5), dpi=150, nsamples=None, seed=None):
    """Curvas P(equivalência) × ε para todos os pares únicos da matriz.

    Reconstrói a posterior Dirichlet de cada par a partir das contagens já
    gravadas na matriz (``contagem_superior``, ``contagem_empate``,
    ``contagem_inferior``), sem re-amostrar do zero. Custo desprezível.

    O gráfico responde: *a qual valor mínimo de ε cada par precisa para ser
    classificado como equivalente?* Pares com curva que nunca ultrapassa o
    limiar não têm equivalência possível em nenhuma margem.

    Args:
        matriz: saída de ``matriz_relacoes`` (modo ``proporcao``).
        arquivo_saida: caminho do PNG. ``None`` retorna o eixo sem salvar.
        eps_ref: ε operacional usado na análise (linha vertical pontilhada).
            Padrão: ``matriz.attrs['eps']``.
        limiar_linha: limiar de equivalência que define a linha horizontal
            (padrão 0,95, o limiar do veredito bayesiano).
        titulo: título da figura; padrão gerado automaticamente.
        nsamples / seed: controle da reamostragem Dirichlet (padrões do módulo).

    Returns:
        ``(eixo, caminho)`` — `caminho` é ``None`` se nada foi gravado.
    """
    import matplotlib.pyplot as plt

    nsamples = NSAMPLES_PADRAO if nsamples is None else nsamples
    seed = SEED_PADRAO if seed is None else seed
    eps_ref = eps_ref if eps_ref is not None else matriz.attrs.get("eps")
    modo = matriz.attrs.get("modo", "proporcao")

    figura, eixo = plt.subplots(figsize=figsize)

    for linha in _pares_unicos(matriz):
        nome_a, nome_b = linha["linha"], linha["coluna"]
        # reconstrói a posterior Dirichlet diretamente das contagens gravadas
        contagens = np.array([
            linha["contagem_superior"],
            linha["contagem_empate"],
            linha["contagem_inferior"],
        ], dtype=float)
        rng = np.random.default_rng(seed)
        theta = rng.dirichlet(contagens + 1.0, size=nsamples)  # prior unitário
        delta = theta[:, 0] - theta[:, 2]

        # curva P(equiv) × ε
        grade = np.linspace(0, 0.30, 121)
        p_curva = np.array([(np.abs(delta) < e).mean() for e in grade])

        # ε crítico (menor ε que atinge o limiar)
        acima = np.flatnonzero(p_curva >= limiar_linha)
        eps_crit = float(grade[acima[0]]) if acima.size else float("nan")
        eps_crit_str = f"{eps_crit:.3f}".replace(".", ",") if not np.isnan(eps_crit) else "n/a"

        rotulo = f"{nome_a} × {nome_b}  (ε crít. = {eps_crit_str})"
        eixo.plot(grade, p_curva, label=rotulo, linewidth=1.8)

    # linha horizontal: limiar de equivalência
    eixo.axhline(limiar_linha, color="0.35", ls="--", lw=1)
    eixo.text(0.305, limiar_linha,
              f" limiar {str(limiar_linha).replace('.', ',')}",
              va="center", fontsize=8.5, color="0.35")

    # linha vertical: ε operacional
    if eps_ref is not None and not np.isnan(float(eps_ref)):
        eixo.axvline(eps_ref, color="0.45", ls=":", lw=1)
        eixo.annotate(
            f"ε = {str(round(float(eps_ref), 4)).replace('.', ',')}",
            xy=(eps_ref, limiar_linha * 0.55),
            xytext=(eps_ref + 0.012, limiar_linha * 0.55 - 0.12),
            fontsize=8.5, color="0.30",
            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.8))

    # rótulos e grade
    eixo.set_xlabel("ε (proporção de documentos)", fontsize=10)
    eixo.set_ylabel("P(equivalência prática)", fontsize=10)
    eixo.set_xlim(-0.005, 0.305)
    eixo.set_ylim(-0.02, 1.05)
    eixo.grid(alpha=0.25)
    eixo.legend(fontsize=8.5, loc="lower right")

    # subtítulo com metadados
    metrica = matriz.attrs.get("metrica")
    papel = matriz.attrs.get("papel")
    if titulo is None:
        titulo = "Sensibilidade da equivalência ao ε"
        if metrica:
            titulo += f" — {metrica}"
    eixo.set_title(titulo, fontsize=11)
    if metrica or papel:
        detalhe = " · ".join(filter(None, [
            f"métrica: {metrica}" if metrica else None,
            f"análise {papel}" if papel else None]))
        eixo.annotate(detalhe, xy=(0.5, 1.0), xycoords="axes fraction",
                      xytext=(0, 6), textcoords="offset points",
                      ha="center", va="bottom", fontsize=8, color="0.4")
        eixo.set_title(titulo, fontsize=11, pad=22)

    caminho = None
    if arquivo_saida:
        figura.tight_layout()
        figura.savefig(arquivo_saida, dpi=dpi, bbox_inches="tight", facecolor="white")
        caminho = arquivo_saida
        plt.close(figura)
    return eixo, caminho


def sensibilidade_margem(dados, valores, nomes=None, modo="proporcao",
                         rope=0.0, eps=0.0, limiar=None, referencia=None, **kw):
    """Varre a margem que define a categoria: ε no modo proporção, ROPE no baycomp.

    O custo é **assimétrico**, e a diferença importa no planejamento:

    * ε atua sobre a posterior já amostrada — a varredura inteira reaproveita
      as mesmas amostras e sai praticamente de graça;
    * a ROPE atua sobre os escores brutos e muda as contagens — cada valor
      exige uma nova amostragem.

    No F1 essa varredura não é um complemento metodológico: com escores
    comprimidos, a ROPE é o que determina o resultado (pequena demais esvazia a
    zona central, grande demais engole tudo), e a transição entre os dois
    extremos é rápida. Reportar a matriz em três ROPEs é o mínimo defensável.

    Args:
        valores: margens a testar.
        referencia: margem operacional; as mudanças são contadas contra ela
            (padrão: `eps` ou `rope`, conforme o modo).
    """
    import pandas as pd

    if modo not in MODOS:
        raise ValueError(f"modo desconhecido: {modo!r}; use um de {MODOS}")
    referencia = referencia if referencia is not None else (
        eps if modo == "proporcao" else rope)
    valores = sorted({*[float(v) for v in valores], float(referencia)})

    if hasattr(dados, "columns"):
        disponiveis = list(dados.columns)
    else:
        disponiveis = list(dados.keys())
    nomes = list(disponiveis if nomes is None else nomes)
    pares = [(a, b) for i, a in enumerate(nomes) for b in nomes[i + 1:]]

    # modo proporção: amostra UMA vez e reaproveita para todos os ε
    reutilizaveis = None
    if modo == "proporcao":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reutilizaveis = [ComparacaoPareada(np.asarray(dados[a], float),
                                               np.asarray(dados[b], float),
                                               rope=rope, **kw)
                             for a, b in pares]

    def classificar(margem):
        classes = []
        for indice, (a, b) in enumerate(pares):
            if modo == "proporcao":
                p = reutilizaveis[indice].probabilidades_relacao(margem, modo=modo)
            else:
                # ROPE muda as contagens: não há como evitar reamostrar
                cmp = ComparacaoPareada(np.asarray(dados[a], float),
                                        np.asarray(dados[b], float),
                                        rope=margem, **kw)
                p = cmp.probabilidades_relacao(None, modo=modo)
            classes.append(classificar_relacao(
                p["p_inferior"], p["p_equivalente"], p["p_superior"], limiar)[0])
        return classes

    base = classificar(referencia)
    linhas = []
    for margem in valores:
        classes = classificar(margem)
        linhas.append({
            "ε" if modo == "proporcao" else "ROPE": margem,
            "superior": classes.count("superior"),
            "equivalente": classes.count("equivalente"),
            "inferior": classes.count("inferior"),
            "incerto": classes.count("incerto"),
            "Muda vs. referência": sum(x != y for x, y in zip(classes, base)),
            "Referência": "sim" if margem == referencia else "não",
        })
    tabela = pd.DataFrame(linhas)
    tabela.attrs.update(modo=modo, referencia=referencia, pares=len(pares))
    return tabela


# ══════════════════════════════════════════════════════ calibração da ROPE

def calibrar_rope(execucoes, percentil=90):
    """ROPE ancorada na variação entre execuções do MESMO protocolo.

    Mesmo argumento do ε calibrado pela divergência entre avaliadores humanos:
    se duas medições que *deveriam* ser iguais diferem em X, diferenças dessa
    ordem não são sinal. A frase que isso autoriza no texto: *a ROPE absorve
    90% da variação observada entre duas execuções do mesmo modelo*.

    Três cuidados que a escolha da estatística embute:

    1. **Percentil, não amplitude.** Máximo menos mínimo é dominado por
       outliers e cresce com n; com milhares de documentos capturaria um caso
       patológico.
    2. **Diferença PAREADA por documento**, não diferença de médias — a ROPE
       incide sobre |x − y| de cada caso, e a diferença de médias é ordens de
       grandeza menor, produzindo uma ROPE inútil.
    3. **Use um protocolo treinado**, não o modelo base: a instabilidade do
       zero-shot (falhas de formato, saídas fora do esquema) não representa a
       variação dos artefatos que serão comparados.

    ⚠ Com decodificação gulosa (temperatura 0) as execuções saem quase
    idênticas e a ROPE calibrada assim tende a zero — mediria determinismo do
    decoder, não incerteza relevante. Verifique a configuração de amostragem
    antes de usar este caminho, e confirme o resultado com
    ``controle_negativo``.

    Args:
        execucoes: dicionário ``{rótulo: escores}`` ou lista de sequências, com
            ao menos duas execuções pareadas nos mesmos documentos.
        percentil: percentil de |diferença| absorvido pela ROPE.

    Returns:
        ``(rope, detalhe)`` — `detalhe` traz o valor por par e o texto pronto
        para o relatório.
    """
    if not isinstance(execucoes, dict):
        execucoes = {f"execução {i + 1}": v for i, v in enumerate(execucoes)}
    rotulos = list(execucoes)
    if len(rotulos) < 2:
        raise ValueError("a calibração da ROPE exige ao menos duas execuções pareadas")

    series = {r: np.asarray(execucoes[r], float) for r in rotulos}
    tamanhos = {len(v) for v in series.values()}
    if len(tamanhos) > 1:
        raise ValueError("as execuções devem estar pareadas nos mesmos documentos")

    por_par = {}
    for i, a in enumerate(rotulos):
        for b in rotulos[i + 1:]:
            diferencas = np.abs(series[a] - series[b])
            por_par[f"{a} × {b}"] = float(np.percentile(diferencas, percentil))
    # o maior entre os pares: a ROPE precisa cobrir o pior caso observado
    rope = max(por_par.values())
    detalhe = {
        "por_par": por_par, "percentil": percentil, "n": len(next(iter(series.values()))),
        "texto": (f"calibrada pelo percentil {percentil} das diferenças pareadas "
                  f"entre {len(rotulos)} execuções do mesmo protocolo ("
                  + ", ".join(f"{par}: {v:.5f}" for par, v in por_par.items())
                  + f"); adotado o maior: {rope:.5f}"),
    }
    if rope == 0:
        warnings.warn(
            "ROPE calibrada em zero: as execuções são idênticas. Provável "
            "decodificação gulosa — a calibração está medindo determinismo do "
            "decoder, não incerteza. Use outra âncora (sementes de treino, ou a "
            "distribuição de |ΔF1| entre itens julgados equivalentes na Likert).",
            stacklevel=2)
    return rope, detalhe


def controle_negativo(x, y, rope, modo="baycomp", limiar=None, **kw):
    """Verifica a ROPE comparando um protocolo CONSIGO MESMO.

    Duas execuções do mesmo modelo precisam sair equivalentes. Se não saírem, a
    ROPE está pequena demais — e é melhor descobrir isso antes de olhar os
    protocolos, não depois. É o teste que transforma a calibração de plausível
    em demonstrada, e custa uma comparação.

    Returns:
        Dicionário com a classificação obtida, as três probabilidades e
        ``aprovado``.
    """
    cmp = ComparacaoPareada(np.asarray(x, float), np.asarray(y, float),
                            rope=rope, **kw)
    p = cmp.probabilidades_relacao(None if modo == "baycomp" else 0.0, modo=modo)
    classe, dominante = classificar_relacao(
        p["p_inferior"], p["p_equivalente"], p["p_superior"], limiar)
    return {
        "classificacao": classe, "probabilidade": dominante, **p,
        "rope": float(rope), "modo": modo, "n": cmp.n,
        "massa_empate": cmp.massa_empate,
        "aprovado": classe == "equivalente",
        "diagnostico": ("ROPE compatível com o ruído do próprio modelo"
                        if classe == "equivalente" else
                        f"ROPE pequena demais: o modelo comparado consigo mesmo "
                        f"saiu '{classe}' ({dominante:.1%}). Aumente a margem ou "
                        "reveja a âncora da calibração."),
    }


# ══════════════════════════════════════════════════════ tabela de contrastes

def tabela_contraste(pares, dados, eps, **kw):
    """DataFrame de vários contrastes de uma vez.

    `pares`: lista de (rótulo_da_questão, nome_A, nome_B)
    `dados`: objeto indexável por nome (DataFrame ou dict de arrays)
    """
    import pandas as pd
    linhas = []
    for questao, a, b in pares:
        cmp = ComparacaoPareada(dados[a], dados[b], **kw)
        linhas.append({"questão": questao, **cmp.como_dict(eps, nomes=(a, b))})
    return pd.DataFrame(linhas)


# ══════════════════════════════════════════════════════ demonstração

if __name__ == "__main__":
    rg = np.random.default_rng(21)
    q = rg.normal(0, 1, 4000)
    C = discretizar(q + rg.normal(0, 0.06, 4000))
    D1 = discretizar(q + 0.15 + rg.normal(0, 0.06, 4000))

    cmp = ComparacaoPareada(C, D1, rope=0.0, nsamples=50_000)
    print(cmp.resumo(eps=0.05, nomes=("C", "D1")))
    print("\nrepr :", cmp)
    print("triplet (degenera com rope=0):", np.round(cmp.triplet, 3))
    # autoteste em contraste NÃO saturado (efeito pequeno)
    D2 = discretizar(q + 0.000 + rg.normal(0, 0.06, 4000))   # efeito nulo -> p_dom ~ 0,5
    fraco = ComparacaoPareada(C, D2, rope=0.0, nsamples=50_000)
    print("\nautoteste (contraste fraco):")
    for k, v in fraco.autoteste().items():
        print(f"  {k}: {v}")
    print("sensibilidade ao prior:", fraco.sensibilidade_prior())
    print("estabilidade MC:", fraco.estabilidade_mc())
    print("\ncomparação entre motores (mesmos dados):")
    for k, v in cmp.comparar_motores(eps=0.05).items():
        print(f"  {k}: {v}")

    # binárias pareadas (critério 3 da Fase A)
    b = ComparacaoPareada.de_binarias(C >= 3, D1 >= 3, nsamples=50_000)
    print("\nbinarizado (nota >= 3):", b.resumo(eps=0.10, nomes=("C", "D1")))
    print("viabilidade C:", ProporcaoBeta.de_binarias(C >= 3, nsamples=50_000))

    # matriz de relações + heatmap (o que alimenta a comparação entre protocolos)
    # ── os dois modos, lado a lado, sobre os mesmos documentos ───────────────
    rp = np.random.default_rng(7)
    n_p, base = 300, rp.normal(0, 1, 300)
    latente = {"P1": 0.00, "P2": 0.01, "P3": 0.35, "P4": -0.07}

    # qualidade (Likert): escala ordinal, rope = 0, ε sobre a posterior
    likert = {p: discretizar(base + v + rp.normal(0, 0.30, n_p))
              for p, v in latente.items()}
    matriz_likert, arquivo = heatmap_comparacao(
        likert, eps=0.08, rope=0.0, nsamples=50_000,
        modo="proporcao", metrica="Likert", papel="principal",
        arquivo_saida="demo_heatmap_likert.png")

    # eficiência (F1): escala contínua, cálculo padrão do baycomp, ROPE > 0
    f1 = {p: np.clip(0.90 + 0.05 * (base + v) / 4 + rp.normal(0, 0.01, n_p), 0, 1)
          for p, v in latente.items()}
    rope, detalhe = calibrar_rope({"execução 1": f1["P1"],
                                   "execução 2": f1["P1"] + rp.normal(0, 0.004, n_p)},
                                  percentil=90)
    print(f"\nROPE calibrada: {rope:.5f}\n  {detalhe['texto']}")
    print("controle negativo:", controle_negativo(
        f1["P1"], f1["P1"] + rp.normal(0, 0.004, n_p), rope=rope,
        nsamples=50_000)["diagnostico"])

    matriz_f1, _ = heatmap_comparacao(
        f1, rope=rope, nsamples=50_000,
        modo="baycomp", metrica="BERTScore F1", papel="complementar",
        nomes=list(latente),                     # MESMA ordem do heatmap Likert
        arquivo_saida="demo_heatmap_f1.png")

    print("\nconvergência entre as métricas:")
    print(tabela_convergencia(matriz_likert, matriz_f1,
                              rotulos=("Likert", "F1")).to_string(index=False))
    print("\nsensibilidade ao limiar (Likert):")
    print(sensibilidade_limiar(matriz_likert).to_string(index=False))
    print("\nsensibilidade à ROPE (F1) — reamostra a cada valor:")
    print(sensibilidade_margem(f1, valores=(rope / 2, rope, rope * 2),
                               nomes=list(latente), modo="baycomp",
                               referencia=rope, nsamples=20_000).to_string(index=False))
    print("\nsíntese por protocolo (Likert):")
    print(resumo_relacoes(matriz_likert).to_string())
    print(f"\nheatmaps gravados: {arquivo}, demo_heatmap_f1.png")
