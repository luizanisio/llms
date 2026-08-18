# -*- coding: utf-8 -*-

"""
Camada bayesiana da comparação entre protocolos (Fase B).

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Descrição:
-----------
Acrescenta ao pipeline `comparar_extracoes.py` a comparação pareada bayesiana
entre protocolos — heatmap de relações, curva de sensibilidade ao ε, varredura
da ROPE e um relatório Markdown com as tabelas e a leitura descritiva.

O cálculo inteiro vive em `util_est_bayesiana` (teste de sinais bayesiano de
Benavoli et al., 2017; `baycomp` como implementação de referência). Este módulo
é a **ponte**: lê a configuração do YAML, monta os DataFrames pareados a partir
dos dados que o pipeline já carregou, nomeia os arquivos e escreve o relatório.
Nenhuma conta estatística é feita aqui — mesma regra que separa
`realizar_avaliacoes.py` de `realizar_avaliacoes_graficos.py`.

Camada **complementar**, não substitutiva: Friedman, Wilcoxon, Nemenyi e os
tamanhos de efeito de `util_analise_estatistica` continuam sendo a análise
principal. A leitura bayesiana acrescenta o que o teste de hipótese nula não
consegue expressar — a probabilidade posterior de equivalência prática — e
trata "equivalente" como achado, não como falha em rejeitar H₀.

Dois alvos, cada um com o teste do baycomp adequado à sua escala:

  * **Likert do juiz LLM** (principal) — escala ordinal → `baycomp.SignTest`,
    com `rope = 0,5`. A escala é inteira, então meio ponto significa exatamente
    "notas iguais": não é margem arbitrada. Mede qualidade percebida.
  * **Métricas automáticas** (complementar) — escores contínuos →
    `baycomp.CorrelatedTTest`, com a ROPE calibrada. Ele usa a **magnitude** das
    diferenças, e não apenas as direções. Medem similaridade com o modelo base,
    ou seja **fidelidade de destilação, não qualidade** — um protocolo que
    reproduz fielmente um erro do professor é premiado. Por isso entram como
    triangulação, nunca como veredito.

O teste usado aparece no título de cada figura e no cabeçalho de cada seção.

Configuração (YAML) — ver `configurar_bayesiana` para a lista completa:

    estatistica_bayesiana:
      ativo: true
      metricas_automaticas:
        rope: 0.01
        campos: ["(global)"]
        metricas: [bertscore]

Uso:
    from comparar_extracoes_baycomp import executar_analise_bayesiana
    executar_analise_bayesiana(analisador, dados_analise, config, pasta_saida)
"""

import os
import re
import glob
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

# A camada bayesiana é opcional: sem `util_est_bayesiana` (ou sem matplotlib) o
# pipeline segue normalmente, apenas sem esta análise.
try:
    import util_est_bayesiana as bayes
    BAYES_DISPONIVEL = True
except ImportError:  # pragma: no cover - depende do ambiente
    bayes = None
    BAYES_DISPONIVEL = False

from util_analise_estatistica import (MAPA_METRICA_SUFIXO, MAPA_METRICA_DISPLAY,
                                      montar_mapa_aliases, montar_dataframes_llm)


# ============================================================================
# 1. Configuração
# ============================================================================

#: Padrões da etapa. Todos sobrescritíveis pela chave do YAML.
LIMIAR_PADRAO = 0.80        # classificação das células do heatmap
#: A escala Likert é inteira: meio ponto separa "notas iguais" de "notas
#: diferentes". Não é margem calibrada — é a tradução direta da escala.
ROPE_LIKERT_PADRAO = 0.5
AMOSTRAS_PADRAO = 200_000
SEMENTE_PADRAO = 42

#: nome da classe do baycomp por trás de cada método, para o relatório
NOME_TESTE = {"sinais": "baycomp.SignTest",
              "postos": "baycomp.SignedRankTest",
              "t": "baycomp.CorrelatedTTest"}

#: Acima deste número de protocolos o heatmap deixa de ser legível e a matriz
#: reintroduz as comparações todos-contra-todos que um desenho pré-registrado
#: evita. Não bloqueia a execução — apenas avisa, porque o recorte é decisão de
#: quem escreveu o YAML.
PROTOCOLOS_ALERTA = 8

#: Nome da subpasta de saída, no mesmo padrão de `estatisticas/` e `graficos/`.
PASTA_SAIDA = 'bayesiana'


def _slug(texto) -> str:
    """Normaliza um nome de recorte para uso em nome de arquivo."""
    import unicodedata
    sem_acento = unicodedata.normalize('NFKD', str(texto)).encode('ascii', 'ignore').decode()
    return re.sub(r'_+', '_', re.sub(r'[^0-9A-Za-z]+', '_', sem_acento)).strip('_').lower()


@dataclass
class Recorte:
    """Um subconjunto nomeado de protocolos — rende um heatmap por métrica.

    Existe para resolver o problema de escala do panorama completo: 16
    protocolos geram 120 pares e uma figura ilegível. Declarando um recorte por
    questão de pesquisa, a mesma comparação já processada produz várias figuras
    focadas, cada uma com os protocolos que aquela questão de fato contrasta.

    `nome` vira prefixo dos arquivos e título da seção do relatório. Quando é
    `None`, há um único recorte anônimo e os nomes de arquivo ficam como eram.
    """
    nome: str = None
    protocolos: list = field(default_factory=list)

    @property
    def prefixo(self) -> str:
        """Prefixo dos arquivos gerados: vazio no recorte anônimo."""
        return f'{_slug(self.nome)}_' if self.nome else ''

    @property
    def rotulo(self) -> str:
        """Nome legível para console e relatório."""
        return str(self.nome) if self.nome else 'recorte único'


def _ler_recortes(valor) -> list:
    """Normaliza a chave `protocolos` do YAML nas três formas aceitas.

    * ausente/vazia → um recorte anônimo com todos os protocolos;
    * lista → um recorte anônimo com os protocolos listados;
    * dicionário `{nome: [protocolos]}` → um recorte nomeado por entrada, na
      ordem declarada (o YAML preserva a ordem do documento).
    """
    if not valor:
        return [Recorte()]
    if isinstance(valor, dict):
        return [Recorte(nome=str(nome), protocolos=list(lista or []))
                for nome, lista in valor.items()]
    return [Recorte(protocolos=list(valor))]


@dataclass
class ConfigBayes:
    """Parâmetros da comparação bayesiana pareada entre protocolos.

    Toda a estatística vem do **baycomp**, via `util_est_bayesiana`. Cada escala
    usa o teste adequado à sua natureza:

    * **Likert** (ordinal) → `SignTest`, com ``rope_likert = 0,5``. A escala é
      inteira, então meio ponto significa exatamente "notas iguais": não é
      margem arbitrada, é a tradução direta da escala.
    * **Métricas automáticas** (contínuas) → `CorrelatedTTest`, com a ``rope``
      calibrada. Ele usa a magnitude das diferenças, e não apenas as direções.

    A ROPE precisa ser **> 0** nos dois casos: com ROPE zero o baycomp devolve
    apenas `(p_esquerda, p_direita)`, sem a probabilidade de equivalência.
    """
    ativo: bool = False
    limiar: float = LIMIAR_PADRAO
    amostras: int = AMOSTRAS_PADRAO
    semente: int = SEMENTE_PADRAO
    incluir_base: bool = False
    recortes: list = field(default_factory=lambda: [Recorte()])
    # --- Likert (principal) ---
    rope_likert: float = ROPE_LIKERT_PADRAO
    metodo_likert: str = "sinais"
    # --- métricas automáticas (complementares) ---
    rope: float = 0.0
    metodo_metricas: str = "t"
    rope_sensibilidade: list = field(default_factory=list)
    campos: list = field(default_factory=list)
    metricas: list = field(default_factory=list)

    def kw(self, metodo: str, rope: float) -> dict:
        """Argumentos repassados ao `util_est_bayesiana`."""
        return {"rope": rope, "metodo": metodo, "limiar": self.limiar,
                "nsamples": self.amostras, "seed": self.semente}

    @property
    def kw_likert(self) -> dict:
        return self.kw(self.metodo_likert, self.rope_likert)

    @property
    def kw_metricas(self) -> dict:
        return self.kw(self.metodo_metricas, self.rope)

    @property
    def tem_likert(self) -> bool:
        """A escala Likert é inteira: a ROPE decorre dela, sempre disponível."""
        return self.rope_likert > 0

    @property
    def tem_automaticas(self) -> bool:
        """A seção complementar exige alvos declarados e ROPE > 0."""
        return bool(self.campos and self.metricas and self.rope > 0)

    @property
    def grade_rope(self) -> list:
        """Valores da varredura de sensibilidade à ROPE.

        Padrão (rope/2, rope, rope*2) porque com escores comprimidos é a ROPE —
        e não os dados — que determina o resultado: pequena demais esvazia a zona
        central e o heatmap satura em verde/vermelho; grande demais engole tudo e
        ele fica azul. A transição entre os extremos é rápida, e reportar a
        matriz em três ROPEs é o mínimo defensável.
        """
        if self.rope_sensibilidade:
            return sorted({float(v) for v in self.rope_sensibilidade if float(v) > 0})
        return sorted({self.rope / 2, self.rope, self.rope * 2})


def configurar_bayesiana(config: dict) -> ConfigBayes:
    """Lê a configuração da etapa bayesiana do YAML.

    A chave canônica é `estatistica_bayesiana` no nível raiz; também é aceita
    dentro de `configuracao_comparacao`, ao lado de `campos_estatisticas` (mesma
    tolerância que o YAML já pratica em `execucao.divisao`/`execucao-divisao`).

    Chaves reconhecidas::

        estatistica_bayesiana:
          ativo: true              # sem isto, nada roda
          limiar: 0.80             # classificação das células do heatmap
          rope_likert: 0.5         # (opcional) "notas iguais"; decorre da escala inteira
          metodo_likert: sinais    # (opcional) sinais | postos | t
          amostras: 200000         # amostras da posterior
          semente: 42              # reprodutibilidade
          incluir_base: false      # inclui o modelo base na matriz da Likert
          protocolos:              # recorte e ORDEM; aceita alias ou rótulo.
            Q1_ajuste_fino: [A, B, C]        # dicionário = um heatmap por questão,
            Q3_escalonamento: [D1, D2, D3]   #   com o nome virando prefixo dos arquivos
          # protocolos: [A, B, D1] # lista simples = recorte único, sem prefixo
          metricas_automaticas:
            rope: 0.01             # OBRIGATÓRIA (> 0) para a seção complementar
            metodo: t              # (opcional) t = CorrelatedTTest, adequado a contínuo
            rope_sensibilidade: [0.005, 0.01, 0.02]   # padrão: rope/2, rope, 2·rope
            campos: ["(global)"]   # campos do YAML (ex.: "(global)", "Resumo")
            metricas: [bertscore, sbert_medio]

    Returns:
        ConfigBayes — com `ativo=False` quando a chave está ausente ou desligada.
    """
    bloco = config.get('estatistica_bayesiana')
    if not isinstance(bloco, dict):
        bloco = config.get('configuracao_comparacao', {}).get('estatistica_bayesiana')
    if not isinstance(bloco, dict):
        return ConfigBayes(ativo=False)

    if not bloco.get('ativo', True):
        return ConfigBayes(ativo=False)

    automaticas = bloco.get('metricas_automaticas') or {}
    if not isinstance(automaticas, dict):
        automaticas = {}

    return ConfigBayes(
        ativo=True,
        limiar=float(bloco.get('limiar', LIMIAR_PADRAO)),
        amostras=int(bloco.get('amostras', AMOSTRAS_PADRAO)),
        semente=int(bloco.get('semente', SEMENTE_PADRAO)),
        incluir_base=bool(bloco.get('incluir_base', False)),
        recortes=_ler_recortes(bloco.get('protocolos')),
        rope_likert=float(bloco.get('rope_likert', ROPE_LIKERT_PADRAO)),
        metodo_likert=str(bloco.get('metodo_likert', 'sinais')),
        rope=float(automaticas.get('rope', 0.0) or 0.0),
        metodo_metricas=str(automaticas.get('metodo', 't')),
        rope_sensibilidade=list(automaticas.get('rope_sensibilidade') or []),
        campos=list(automaticas.get('campos') or []),
        metricas=list(automaticas.get('metricas') or []),
    )


# ============================================================================
# 2. Formatação
# ============================================================================

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


def _tabela_md(df: pd.DataFrame, indice: bool = False) -> list:
    """Converte um DataFrame em linhas de tabela Markdown."""
    if df is None or len(df) == 0:
        return ["_(sem dados)_", ""]
    colunas = ([df.index.name or ""] if indice else []) + [str(c) for c in df.columns]
    linhas = [f"| {' | '.join(colunas)} |", f"|{'|'.join(['---'] * len(colunas))}|"]
    for chave, registro in df.iterrows():
        celulas = ([str(chave)] if indice else []) + [
            _num(v) if isinstance(v, (float, np.floating)) else str(v) for v in registro
        ]
        linhas.append(f"| {' | '.join(celulas)} |")
    linhas.append("")
    return linhas


def _pares_unicos(matriz: pd.DataFrame):
    """Itera uma linha por par NÃO ordenado, preservando a ordem de `nomes`."""
    vistos = set()
    for _, linha in matriz.iterrows():
        chave = frozenset((linha["linha"], linha["coluna"]))
        if chave not in vistos:
            vistos.add(chave)
            yield linha


def tabela_pares(matriz: pd.DataFrame) -> pd.DataFrame:
    """Formata a matriz longa para leitura: uma linha por par NÃO ordenado.

    Mesmo formato de `realizar_avaliacoes.tabela_matriz_bayesiana`, para que as
    tabelas da Fase A e da Fase B sejam lidas da mesma maneira.
    """
    linhas = []
    for linha in _pares_unicos(matriz):
        linhas.append({
            "Par (A × B)": f"{linha['linha']} × {linha['coluna']}",
            "n": int(linha["n"]),
            "A melhor": int(linha["x_melhor"]),
            "Empate": int(linha["empate"]),
            "B melhor": int(linha["y_melhor"]),
            "Dif. média": _num(linha["diferenca_media"], 4),
            "P(A > B)": _num(linha["p_esquerda"], 4),
            "P(equiv.)": _num(linha["p_rope"], 4),
            "P(A < B)": _num(linha["p_direita"], 4),
            "Relação de A": linha["classificacao"],
        })
    return pd.DataFrame(linhas)


def achados(matriz: pd.DataFrame, rotulo_entidade: str = "protocolo") -> list:
    """Leitura descritiva automática da matriz, em frases prontas para o relatório.

    Descreve o panorama sem produzir ranking: com relações possivelmente
    intransitivas, ordenar por contagem de vitórias criaria uma ordem que os
    dados não sustentam.
    """
    if matriz is None or len(matriz) == 0:
        return []

    pares = list(_pares_unicos(matriz))
    total = len(pares)
    contagem = pd.Series([p["classificacao"] for p in pares]).value_counts()
    decididos = total - int(contagem.get("incerto", 0))

    frases = [
        f"São {total} par(es) comparado(s). "
        f"{decididos} alcançaram o limiar de {_num(matriz.attrs.get('limiar'), 2)} "
        f"({int(contagem.get('superior', 0)) + int(contagem.get('inferior', 0))} com direção, "
        f"{int(contagem.get('equivalente', 0))} com equivalência) e "
        f"{int(contagem.get('incerto', 0))} permaneceram inconclusivos."
    ]

    resumo = bayes.resumo(matriz)
    superiores = resumo["superior"]
    if superiores.max() > 0:
        lideres = list(superiores[superiores == superiores.max()].index)
        frases.append(
            f"Maior número de superioridades: {', '.join(map(str, lideres))} "
            f"({int(superiores.max())} de {len(resumo) - 1} comparações).")
    equivalentes = resumo["equivalente"]
    if equivalentes.max() > 0:
        estaveis = list(equivalentes[equivalentes == equivalentes.max()].index)
        frases.append(
            f"Mais equivalências: {', '.join(map(str, estaveis))} "
            f"({int(equivalentes.max())}) — equivalência é achado, não falha do teste.")

    # a direção satura com n grande: o conteúdo migra para as contagens
    saturados = [p for p in pares if max(p["p_esquerda"], p["p_direita"]) > 0.999]
    if saturados:
        frases.append(
            f"{len(saturados)} par(es) com P(direção) > 0,999: com n = "
            f"{_num(matriz.attrs.get('n'), 0)} a probabilidade de dominância satura. "
            "O conteúdo informativo desses pares está nas contagens e na diferença "
            "média, não na probabilidade.")

    # Limitação do SignTest: com muitos empates a zona central vence por ser a
    # maior, mesmo havendo desequilíbrio claro entre as laterais.
    if matriz.attrs.get("metodo") == "sinais":
        dominadas = [p for p in pares
                     if p["classificacao"] == "equivalente"
                     and max(p["x_melhor"], p["y_melhor"]) >= 2 * max(min(p["x_melhor"],
                                                                          p["y_melhor"]), 1)]
        if dominadas:
            nomes = ", ".join(f"{p['linha']} × {p['coluna']}" for p in dominadas[:5])
            frases.append(
                f"{len(dominadas)} par(es) classificados como equivalentes apesar de um "
                f"lado vencer ao menos o dobro de documentos ({nomes}"
                f"{'…' if len(dominadas) > 5 else ''}). O `SignTest` responde *qual zona "
                "é a maior*, e com muitos empates a central vence mesmo havendo "
                "desequilíbrio — leia as contagens ao lado da probabilidade.")

    return frases


# ============================================================================
# 3. Montagem dos DataFrames pareados
# ============================================================================

def selecionar_protocolos(selecao, rotulos, mapa_aliases) -> tuple:
    """Resolve a lista `protocolos` do YAML em rótulos, na ORDEM declarada.

    Duas funções numa só, e a segunda é a que costuma passar despercebida:

    1. **recorte** — comparar todos contra todos com muitos protocolos gera uma
       figura ilegível e reintroduz as comparações que um desenho pré-registrado
       evita;
    2. **ordem** — linhas e colunas saem na sequência declarada. Isso importa
       porque os heatmaps da Likert e das métricas automáticas são lidos lado a
       lado: com as linhas em ordens diferentes, a comparação visual induz
       leitura errada.

    Aceita tanto o **alias** (o que aparece nas figuras e tabelas, e portanto o
    nome que quem lê o relatório tem em mãos) quanto o **rótulo** do YAML. Em
    caso de colisão entre o alias de um modelo e o rótulo de outro, o rótulo
    vence — é o identificador canônico.

    Args:
        selecao: lista do YAML; vazia devolve `rotulos` inalterada.
        rotulos: rótulos disponíveis, na ordem natural do YAML.
        mapa_aliases: {rotulo: alias}.

    Returns:
        tuple(escolhidos, ausentes) — `ausentes` traz os nomes que não casaram
        com nada, para virarem aviso explícito. Um nome errado precisa aparecer:
        silenciosamente encolher o heatmap é pior do que não filtrar.
    """
    if not selecao:
        return list(rotulos), []

    # DUAS passadas, e a ordem entre elas é o que garante a precedência: todos os
    # rótulos primeiro, aliases depois. Registrando rótulo+alias na mesma
    # iteração, o alias de um modelo listado antes venceria o rótulo de um
    # modelo listado depois — o oposto do documentado.
    indice = {}
    for rotulo in rotulos:
        indice.setdefault(str(rotulo).strip().lower(), rotulo)
    for rotulo in rotulos:
        indice.setdefault(str(mapa_aliases.get(rotulo, rotulo)).strip().lower(), rotulo)

    escolhidos, ausentes, vistos = [], [], set()
    for nome in selecao:
        rotulo = indice.get(str(nome).strip().lower())
        if rotulo is None:
            ausentes.append(str(nome))
        elif rotulo not in vistos:
            vistos.add(rotulo)
            escolhidos.append(rotulo)
    return escolhidos, ausentes


def _dados_likert(dados_analise, rotulos_alvo, mapa_aliases) -> pd.DataFrame:
    """Notas Likert do juiz LLM, uma coluna por protocolo, pareadas por documento.

    Reaproveita `montar_dataframes_llm` — a mesma fonte usada pela análise
    frequentista — para que as duas leituras descrevam exatamente os mesmos
    escores.
    """
    _, df_nota = montar_dataframes_llm(dados_analise, rotulos_alvo, mapa_aliases)
    if df_nota is None or df_nota.empty:
        return pd.DataFrame()
    # descarte pareado: um documento sem nota em qualquer protocolo sai de todos,
    # senão as colunas deixam de ser comparáveis linha a linha
    return df_nota.dropna()


def _dados_metrica(df_resultados, protocolos, campo, sufixo, mapa_aliases) -> pd.DataFrame:
    """Escores de uma métrica automática, uma coluna por protocolo.

    Mesmo padrão de coluna usado por `executar_analise_estatistica`:
    `{protocolo}_{campo}_{sufixo}_F1`.
    """
    df_largo = pd.DataFrame()
    for proto in protocolos:
        coluna = f'{proto}_{campo}_{sufixo}_F1'
        if coluna in df_resultados.columns:
            df_largo[mapa_aliases.get(proto, proto)] = df_resultados[coluna]
    if df_largo.empty:
        return df_largo
    return df_largo.dropna()


# ============================================================================
# 4. Execução de um alvo (Likert ou métrica automática)
# ============================================================================

def _analisar_alvo(dados, cfg: ConfigBayes, metodo: str, rope: float,
                   metrica: str, papel: str, nome_base: str, pasta: str) -> dict:
    """Calcula a matriz, grava CSV/PNG e devolve tudo o que o relatório precisa.

    Args:
        dados: DataFrame largo (documentos × protocolos), já pareado.
        metodo: `sinais` (Likert, ordinal) ou `t` (métricas contínuas).
        rope: margem sobre os escores brutos; obrigatória e > 0 nos dois casos.
        nome_base: prefixo dos arquivos gerados na pasta de saída.

    Returns:
        dict com `matriz`, `tabela`, `resumo`, `achados` e `figuras` — ou vazio
        se não houver ao menos dois protocolos.
    """
    if dados is None or len(dados.columns) < 2 or len(dados) < 2:
        return {}

    nomes = list(dados.columns)
    matriz = bayes.matriz_pares(dados, nomes=nomes, **cfg.kw(metodo, rope))

    figuras = []
    _, arquivo_heatmap = bayes.heatmap(
        matriz, arquivo_saida=os.path.join(pasta, f'{nome_base}_heatmap.png'),
        titulo=f'{metrica} — {NOME_TESTE.get(metodo, metodo)}',
        rotulo="protocolo")
    if arquivo_heatmap:
        figuras.append(os.path.basename(arquivo_heatmap))

    matriz.to_csv(os.path.join(pasta, f'{nome_base}.csv'), index=False, encoding='utf-8')

    resultado = {
        "matriz": matriz,
        "tabela": tabela_pares(matriz),
        "resumo": bayes.resumo(matriz),
        "achados": achados(matriz),
        "figuras": figuras,
        "n": int(matriz.attrs.get("n") or 0),
        "protocolos": nomes,
        "metrica": metrica,
        "papel": papel,
        "metodo": metodo,
        "rope": rope,
    }

    # A varredura só se aplica às métricas contínuas: na Likert a ROPE decorre
    # da escala ser inteira e alterá-la deixaria de representar "notas iguais".
    if papel == "complementar":
        sensibilidade = _sensibilidade_rope(dados, nomes, cfg)
        sensibilidade.to_csv(os.path.join(pasta, f'{nome_base}_sensibilidade_rope.csv'),
                             index=False, encoding='utf-8')
        resultado["sensibilidade"] = sensibilidade

    return resultado


def _sensibilidade_rope(dados, nomes: list, cfg: ConfigBayes) -> pd.DataFrame:
    """Quantas células mudam de categoria ao variar a ROPE.

    A ROPE age sobre os escores brutos e muda as contagens, então **cada valor
    exige uma nova comparação** — ao contrário do limiar, que só reclassifica
    números prontos. Com escores comprimidos esta varredura não é complemento
    metodológico: é a ROPE que determina o resultado.
    """
    referencia = None
    linhas = []
    for valor in cfg.grade_rope:
        m = bayes.matriz_pares(dados, nomes=nomes, **cfg.kw(cfg.metodo_metricas, valor))
        classes = [p["classificacao"] for p in _pares_unicos(m)]
        if referencia is None or valor == cfg.rope:
            referencia = classes
        linhas.append({"ROPE": valor,
                       **{c: classes.count(c) for c in
                          ("superior", "equivalente", "inferior", "incerto")},
                       "Classes": classes})
    for linha in linhas:
        linha["Muda vs. referência"] = sum(
            a != b for a, b in zip(linha.pop("Classes"), referencia))
        linha["Referência"] = "sim" if linha["ROPE"] == cfg.rope else "não"
    return pd.DataFrame(linhas)


# ============================================================================
# 5. Relatório Markdown
# ============================================================================

_LEITURA = """\
| elemento da figura | o que comunica |
|---|---|
| cor | a relação da linha em relação à coluna |
| intensidade | a magnitude da probabilidade posterior |
| número | a mesma probabilidade, explícita |
| cinza | `incerto` — nenhuma das três alcançou o limiar |

`incerto` é categoria própria, não uma quarta relação: significa ausência de
evidência suficiente, e não a existência de algo intermediário entre as três.
A diagonal é neutra porque `(Pi, Pi)` não é comparação.

Cada par não ordenado é amostrado uma única vez e a célula espelhada é derivada
trocando inferior por superior — `P(Pi > Pj)` e `P(Pj < Pi)` são o mesmo número
por construção, não duas estimativas Monte Carlo que por acaso coincidem.

**As três probabilidades somam 1** e vêm da mesma posterior: `P(A > B)`,
`P(equiv.)` e `P(A < B)`. A margem que as separa é sempre a **ROPE**, aplicada
sobre os escores brutos antes da posterior. Reporte as contagens ao lado delas:
é o que permite ver o desequilíbrio entre as laterais quando a zona central
vence apenas por concentrar mais documentos.
"""

_TESTES_TABELA = """\
| teste | o que faz com os documentos | equivalência significa | usado em |
|---|---|---|---|
| `SignTest` | **conta** direções em três caixas | P(a caixa da ROPE ser a **maior**) | Likert (ordinal) |
| `CorrelatedTTest` | **mede** a diferença média | P(a diferença média cair **dentro** da ROPE) | escores contínuos |

O primeiro descarta magnitude — adequado numa escala em que a distância entre as
notas 2 e 3 não é comparável à distância entre 3 e 4. O segundo usa a magnitude,
que é o que interessa num escore contínuo.

**Limitação do `SignTest`:** ele responde *qual zona é a maior*, não *quem vence
e por quanto*. Com muitos empates a zona central vence mesmo havendo
desequilíbrio claro entre as laterais — por isso as contagens vão ao lado das
probabilidades em toda tabela.
"""


def _secao_alvo(resultado: dict, cfg: ConfigBayes, numero: str, nivel: int = 2) -> list:
    """Bloco Markdown de um alvo (uma métrica).

    `nivel` é a profundidade do cabeçalho do alvo: 2 quando não há recortes
    nomeados, 3 quando os alvos ficam aninhados sob a seção do recorte. Os
    blocos internos acompanham, para o sumário do Markdown não sair furado.
    """
    h_alvo, h_sub = '#' * nivel, '#' * (nivel + 1)
    L = [f'{h_alvo} {numero}. {resultado["metrica"]} — análise {resultado["papel"]}', '']

    L.append(
        f'> `{NOME_TESTE.get(resultado["metodo"], resultado["metodo"])}` · '
        f'ROPE = {_num(resultado["rope"], 5)} (sobre os escores brutos) · '
        f'n = {_num(resultado["n"], 0)} documentos pareados · '
        f'{len(resultado["protocolos"])} protocolos · '
        f'limiar de classificação = {_num(cfg.limiar, 2)}'
    )
    L.append('')

    if resultado["papel"] == "complementar":
        L.append(
            '> ⚠ Esta métrica mede similaridade com o modelo base — portanto '
            '**fidelidade de destilação, não qualidade**. Um protocolo que '
            'reproduz fielmente um erro do modelo base é premiado por ela. '
            'Serve como triangulação da Likert, nunca como veredito.'
        )
        L.append('')

    for figura in resultado["figuras"]:
        L.append(f'![{figura}]({figura})')
        L.append('')

    L.append(f'{h_sub} Comparações par a par')
    L.append('')
    L += _tabela_md(resultado["tabela"])

    L.append(f'{h_sub} Síntese por protocolo')
    L.append('')
    L.append('Contagem de relações — **não é ranking**: com relações possivelmente')
    L.append('intransitivas, ordenar por vitórias criaria uma ordem que os dados não')
    L.append('sustentam.')
    L.append('')
    L += _tabela_md(resultado["resumo"], indice=True)

    if resultado.get("sensibilidade") is not None:
        L.append(f'{h_sub} Sensibilidade à ROPE')
        L.append('')
        L.append('Quantas células mudam de categoria ao variar a **ROPE**. Aqui a varredura')
        L.append('não é um complemento metodológico: com escores comprimidos, é a ROPE que')
        L.append('determina o resultado, e a transição entre os extremos é rápida.')
        L.append('')
        L += _tabela_md(resultado["sensibilidade"])

    if resultado["achados"]:
        L.append(f'{h_sub} Leitura descritiva')
        L.append('')
        for frase in resultado["achados"]:
            L.append(f'- {frase}')
        L.append('')

    return L


def gerar_markdown(resultados: list, cfg: ConfigBayes,
                   arquivo: str, avisos: list = None) -> str:
    """Escreve o relatório consolidado com todas as seções geradas."""
    L = ['# Comparação bayesiana entre protocolos', '']

    origem = getattr(bayes, "__name__", "util_est_bayesiana")
    L.append(
        f'> Gerado em {datetime.now().strftime("%Y-%m-%d %H:%M")} · '
        f'módulo `{origem}` · amostras da posterior = {_num(cfg.amostras, 0)} · '
        f'semente = {cfg.semente} · numpy {np.__version__} · pandas {pd.__version__}'
    )
    L.append('')
    L.append(
        '> **Método:** comparação bayesiana pareada com o pacote `baycomp` '
        '(Benavoli et al., 2017, JMLR 18:1-36). `util_est_bayesiana` é uma camada '
        'fina: organiza as chamadas, monta a matriz de todos os pares e desenha o '
        'heatmap — a estatística é integralmente do pacote.'
    )
    L.append('')
    L.append(
        '> **Camada complementar.** Friedman, Wilcoxon + Holm, Nemenyi e os '
        'tamanhos de efeito em `estatisticas/` continuam sendo a análise '
        'principal. O que esta seção acrescenta é a probabilidade posterior de '
        '**equivalência prática** — que o teste de hipótese nula não consegue '
        'expressar — tratando "equivalente" como achado, e não como falha em '
        'rejeitar H₀. "Inconclusivo" também é desfecho legítimo.'
    )
    L.append('')

    # os recortes precisam ficar registrados: um heatmap com 4 dos 16 protocolos
    # e outro com os 16 são figuras diferentes, e quem lê o relatório meses
    # depois não tem como saber qual foi sem isso
    declarados = [r for r in cfg.recortes if r.protocolos]
    if declarados:
        L.append(
            '> **Recortes de protocolos** — declarados em '
            '`estatistica_bayesiana.protocolos`. A ordem dentro de cada recorte é '
            'a declarada, para que os heatmaps das diferentes métricas possam ser '
            'lidos lado a lado:'
        )
        L.append('>')
        for recorte in declarados:
            nome = f'`{recorte.nome}`' if recorte.nome else 'recorte único'
            L.append(f'> - {nome}: {", ".join(map(str, recorte.protocolos))}')
        L.append('')

    for aviso in (avisos or []):
        L.append(f'> ⚠ {aviso}')
        L.append('')

    # --- parâmetros ---
    L.append('## 1. Parâmetros da análise')
    L.append('')
    parametros = pd.DataFrame([
        {'Parâmetro': 'teste (Likert)', 'Valor': NOME_TESTE.get(cfg.metodo_likert, '—'),
         'Aplica-se a': 'seção principal — escala ordinal'},
        {'Parâmetro': 'ROPE (Likert)', 'Valor': _num(cfg.rope_likert, 2),
         'Aplica-se a': 'notas iguais (escala inteira)'},
        {'Parâmetro': 'teste (métricas)', 'Valor': NOME_TESTE.get(cfg.metodo_metricas, '—'),
         'Aplica-se a': 'seção complementar — escala contínua'},
        {'Parâmetro': 'ROPE (métricas)', 'Valor': _num(cfg.rope, 5),
         'Aplica-se a': 'margem sobre os escores brutos'},
        {'Parâmetro': 'limiar de classificação', 'Valor': _num(cfg.limiar, 2),
         'Aplica-se a': 'células do heatmap'},
        {'Parâmetro': 'amostras da posterior', 'Valor': _num(cfg.amostras, 0),
         'Aplica-se a': 'testes amostrais (CorrelatedTTest é analítico)'},
        {'Parâmetro': 'semente', 'Valor': str(cfg.semente),
         'Aplica-se a': 'testes amostrais'},
    ])
    L += _tabela_md(parametros)
    L.append(
        'A **ROPE da Likert não é parâmetro livre**: a escala é inteira, então meio '
        'ponto significa exatamente "notas iguais". Já a ROPE das métricas contínuas '
        'é calibrada e, com escores comprimidos, determina o resultado — por isso a '
        'varredura de sensibilidade acompanha cada seção complementar.'
    )
    L.append('')

    # --- como ler ---
    L.append('## 2. Como ler as figuras e as tabelas')
    L.append('')
    L.append(_LEITURA)
    L.append('')
    L.append('### Os dois testes afirmam coisas diferentes')
    L.append('')
    L.append(_TESTES_TABELA)
    L.append('')

    # --- seções por alvo, agrupadas por recorte ---
    # Com recortes nomeados, o relatório cresce rápido (recortes × campos ×
    # métricas). Agrupar sob um `##` por recorte e rebaixar os alvos para `###`
    # mantém o sumário navegável; sem nomes, a estrutura antiga é preservada.
    nomeados = [r for r in cfg.recortes if r.nome]
    secao = 3
    if nomeados:
        for recorte in cfg.recortes:
            do_recorte = [r for r in resultados if r.get("recorte") is recorte]
            if not do_recorte:
                continue
            L.append(f'## {secao}. Recorte: {recorte.rotulo}')
            L.append('')
            L.append(f'> Protocolos, na ordem declarada: '
                     f'**{", ".join(map(str, do_recorte[0]["protocolos"]))}** · '
                     f'{len(do_recorte)} métrica(s) analisada(s) · '
                     f'arquivos com o prefixo `bayes_{recorte.prefixo}`')
            L.append('')
            for sub, resultado in enumerate(do_recorte, start=1):
                L += _secao_alvo(resultado, cfg, f'{secao}.{sub}', nivel=3)
            secao += 1
    else:
        for resultado in resultados:
            L += _secao_alvo(resultado, cfg, str(secao), nivel=2)
            secao += 1

    # --- limitações ---
    L.append(f'## {secao}. Limitações registradas')
    L.append('')
    L.append(
        '- **A ROPE das métricas contínuas é pré-registrada.** A varredura de '
        'sensibilidade existe para demonstrar que a conclusão **não** depende de um '
        'número escolhido a dedo. Ler a varredura e então adotar o valor que produz '
        'o resultado desejado é escolher a conclusão e inventar o critério depois — '
        'a versão bayesiana do *p-hacking*, e detectável. Na Likert a questão não se '
        'coloca: a ROPE decorre de a escala ser inteira.'
    )
    L.append(
        '- **O `SignTest` responde qual zona é a maior**, não quem vence e por '
        'quanto. Com muitos empates — comum na Likert de 4 pontos — a zona central '
        'vence mesmo havendo desequilíbrio claro entre as laterais. Reporte sempre '
        'as contagens ao lado das probabilidades.'
    )
    L.append(
        '- **P(dominância) satura com n grande.** Com milhares de documentos a '
        'probabilidade de direção vai a 1 mesmo para diferenças triviais. O '
        'conteúdo científico está nas contagens e no Δ com IC de credibilidade.'
    )
    L.append(
        '- **As métricas automáticas medem fidelidade ao modelo base**, não '
        'qualidade. Divergir da Likert é achado, não inconsistência: as duas '
        'podem capturar propriedades distintas do desempenho.'
    )
    L.append(
        '- **A comparação é pareada por documento.** Documentos sem escore em '
        'algum protocolo saem de todos, para que as colunas permaneçam '
        'comparáveis linha a linha.'
    )
    L.append('')

    with open(arquivo, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    return arquivo


# ============================================================================
# 6. Ponto de entrada do pipeline
# ============================================================================

def _processar_recorte(recorte: Recorte, cfg: ConfigBayes, analisador, dados_analise,
                       pasta: str, rotulo_base: str, protocolos: list,
                       mapa_aliases: dict, sem_juiz_global: bool = False) -> tuple:
    """Executa todos os alvos de um recorte e devolve `(resultados, avisos)`.

    Cada recorte é independente: seus arquivos levam o prefixo do nome e os
    avisos citam o recorte, para que um problema em `Q3` não pareça afetar `Q1`.
    """
    resultados, avisos = [], []
    prefixo, rotulo = recorte.prefixo, recorte.rotulo
    # nos avisos, identifica o recorte apenas quando ele tem nome
    marca = f"Recorte `{recorte.nome}` — " if recorte.nome else ""

    # Validação única da seleção, contra base + protocolos: um alias inexistente
    # precisa virar aviso, não um heatmap silenciosamente menor.
    if recorte.protocolos:
        _, ausentes = selecionar_protocolos(recorte.protocolos,
                                            [rotulo_base] + protocolos, mapa_aliases)
        if ausentes:
            avisos.append(
                f"{marca}nomes em `estatistica_bayesiana.protocolos` que não "
                f"casaram com nenhum modelo do YAML e foram ignorados: "
                f"{', '.join(ausentes)}. Use o alias ou o rótulo de "
                f"`modelo_base`/`modelos_comparacao`."
            )
            print(f"   ⚠️  {avisos[-1]}")

    # A base só entra na Likert quando pedida — explicitamente na lista ou por
    # `incluir_base`. Nas métricas automáticas ela nunca entra: as colunas medem
    # similaridade COM a base, e base contra si mesma é 1,0 por construção.
    candidatos_likert = (([rotulo_base] + protocolos)
                         if (recorte.protocolos or cfg.incluir_base) else list(protocolos))
    rotulos_likert, _ = selecionar_protocolos(recorte.protocolos, candidatos_likert, mapa_aliases)
    rotulos_metricas, _ = selecionar_protocolos(recorte.protocolos, protocolos, mapa_aliases)

    if recorte.protocolos:
        selecionados = [mapa_aliases.get(r, r) for r in rotulos_likert]
        # recorte que não resolveu nada: um único aviso e segue para o próximo,
        # em vez de repetir a falha em cada alvo
        if len(selecionados) < 2:
            avisos.append(
                f"{marca}o recorte resolveu {len(selecionados)} protocolo(s) — "
                "menos que os dois necessários para uma comparação pareada. "
                "Nenhuma seção foi gerada para ele."
            )
            print(f"\n   ⚠️  {avisos[-1]}")
            return resultados, avisos

        print(f"\n   🔍 {rotulo}: {', '.join(map(str, selecionados))} "
              f"({len(selecionados)} de {len(protocolos) + 1} disponíveis, na ordem declarada)")
        if len(rotulos_metricas) < 2:
            avisos.append(
                f"{marca}o recorte deixou menos de dois protocolos para as "
                "métricas automáticas (o modelo base não participa dessa seção). "
                "As seções complementares foram omitidas."
            )
            print(f"   ⚠️  {avisos[-1]}")

    maior = max(len(rotulos_likert), len(rotulos_metricas))
    if maior > PROTOCOLOS_ALERTA:
        print(f"   ⚠️  {maior} protocolos → {maior * (maior - 1) // 2} pares. "
              "O heatmap pode ficar difícil de ler; use `estatistica_bayesiana."
              "protocolos` para declarar um recorte por questão de pesquisa.")

    # ── alvo principal: Likert do juiz LLM ──────────────────────────────────
    # Ordinal, rope = 0, ε sobre a posterior. É a métrica de qualidade percebida.
    dados_likert = _dados_likert(dados_analise, rotulos_likert, mapa_aliases)

    if sem_juiz_global:
        # a causa é única e já foi reportada uma vez, antes do laço de recortes:
        # repeti-la por recorte encheria o relatório com o mesmo parágrafo
        pass
    elif dados_likert.empty or len(dados_likert.columns) < 2:
        # com recorte ativo há duas causas possíveis, e culpar a errada faz
        # perder tempo procurando arquivos que existem
        com_nota = list(dados_likert.columns)
        if recorte.protocolos and com_nota:
            detalhe = (
                f"Dos protocolos do recorte, apenas {', '.join(map(str, com_nota))} "
                f"tem nota do juiz LLM. Revise `estatistica_bayesiana.protocolos` "
                f"ou gere as avaliações dos demais."
            )
        else:
            detalhe = (
                "Preencha `configuracao_comparacao.campos_dataset.avaliacao` com a "
                "coluna de avaliação do parquet (ou disponibilize os arquivos "
                "`{id}.avaliacao.json` na pasta de cada modelo)."
            )
        avisos.append(
            f"{marca}sem notas do juiz LLM (`" + "{protocolo}_nota`) para ao menos "
            f"dois protocolos: a seção principal foi omitida. {detalhe}"
        )
        print(f"   ⚠️  {avisos[-1]}")
    elif not cfg.tem_likert:
        avisos.append(
            f"{marca}notas do juiz LLM disponíveis, mas `rope_likert` foi definida como "
            "zero ou negativa: a seção principal foi omitida. Com ROPE zero o baycomp "
            f"não devolve a probabilidade de equivalência; na escala inteira o valor "
            f"correto é {ROPE_LIKERT_PADRAO}."
        )
        print(f"   ⚠️  {avisos[-1]}")
    else:
        print(f"   → Likert do juiz LLM ({len(dados_likert)} docs, "
              f"{len(dados_likert.columns)} protocolos)")
        resultado = _analisar_alvo(
            dados_likert, cfg, metodo=cfg.metodo_likert, rope=cfg.rope_likert,
            metrica="Likert (juiz LLM)", papel="principal",
            nome_base=f'bayes_{prefixo}likert_juiz', pasta=pasta)
        if resultado:
            resultado["recorte"] = recorte
            resultados.append(resultado)

    # ── alvos complementares: métricas automáticas ──────────────────────────
    # Contínuas, ROPE > 0, cálculo padrão do baycomp. Medem fidelidade ao base.
    if cfg.tem_automaticas and len(rotulos_metricas) >= 2:
        df_resultados = analisador._resultados
        for campo in cfg.campos:
            for metrica in cfg.metricas:
                sufixo = MAPA_METRICA_SUFIXO.get(metrica, metrica)
                display = MAPA_METRICA_DISPLAY.get(metrica, metrica)

                dados = _dados_metrica(df_resultados, rotulos_metricas, campo,
                                       sufixo, mapa_aliases)
                if dados.empty or len(dados.columns) < 2:
                    continue

                nome_base = (f'bayes_{prefixo}{campo}_{metrica}'
                             .replace('(', '').replace(')', ''))
                print(f"   → {campo} × {display} ({len(dados)} docs, "
                      f"{len(dados.columns)} protocolos)")
                resultado = _analisar_alvo(
                    dados, cfg, metodo=cfg.metodo_metricas, rope=cfg.rope,
                    metrica=f'{display} — {campo}', papel="complementar",
                    nome_base=nome_base, pasta=pasta)
                if resultado:
                    resultado["recorte"] = recorte
                    resultados.append(resultado)

    return resultados, avisos


def executar_analise_bayesiana(analisador, dados_analise, config, pasta_saida) -> dict:
    """Função principal chamada por `comparar_extracoes.py`.

    Monta os DataFrames pareados a partir do que o pipeline já carregou, executa
    a comparação bayesiana de cada alvo configurado e escreve
    `bayesiana/analise_bayesiana.md` com as tabelas, figuras e a leitura
    descritiva.

    A etapa é silenciosamente ignorada quando a chave `estatistica_bayesiana`
    não existe ou está com `ativo: false` — o pipeline se comporta exatamente
    como antes.

    Args:
        analisador: instância de JsonAnaliseDataFrame com `_resultados` populado.
        dados_analise: instância de JsonAnaliseDados (fonte da avaliação LLM).
        config: dict do YAML completo.
        pasta_saida: pasta raiz de saída da comparação.

    Returns:
        dict com `resultados` (um por alvo), `avisos` e `arquivo_md`.
    """
    cfg = configurar_bayesiana(config)
    if not cfg.ativo:
        return {}

    if not BAYES_DISPONIVEL:
        print("   ⚠️  Módulo util_est_bayesiana não encontrado. Pulando análise bayesiana.")
        return {}

    if analisador is None or getattr(analisador, '_resultados', None) is None:
        print("   ⚠️  Analisador sem resultados. Pulando análise bayesiana.")
        return {}

    print("\n🎲 Análise Bayesiana — comparação pareada entre protocolos")

    pasta = os.path.join(pasta_saida, PASTA_SAIDA)
    os.makedirs(pasta, exist_ok=True)

    # Limpeza da execução anterior (mesmo padrão de estatisticas/ e graficos/):
    # sem isso, um alvo removido do YAML deixa figuras órfãs que o relatório não
    # cita mais, mas que continuam na pasta parecendo atuais.
    antigos = (glob.glob(os.path.join(pasta, '*.md')) +
               glob.glob(os.path.join(pasta, '*.png')) +
               glob.glob(os.path.join(pasta, '*.csv')))
    removidos = 0
    for arq in antigos:
        try:
            os.remove(arq)
            removidos += 1
        except OSError:
            pass
    if removidos:
        print(f"   🧹 {removidos} arquivos antigos removidos da pasta {PASTA_SAIDA}/")

    mapa_aliases = montar_mapa_aliases(config)
    rotulo_base = analisador.rotulos[1] if len(analisador.rotulos) > 1 else ''
    protocolos = list(analisador.rotulos[2:]) if len(analisador.rotulos) > 2 else []

    if len(protocolos) < 2:
        print("   ⚠️  Menos de dois protocolos. Sem análise bayesiana.")
        return {}

    resultados, avisos = [], []

    # Causa global de ausência da seção principal: sem avaliação do juiz em
    # protocolo nenhum. Detectada uma vez para não repetir o mesmo parágrafo em
    # cada recorte — com 6 recortes seriam 6 avisos idênticos no relatório.
    sem_juiz_global = _dados_likert(dados_analise, protocolos, mapa_aliases).shape[1] < 2
    if sem_juiz_global:
        avisos.append(
            "Sem notas do juiz LLM (`" + "{protocolo}_nota`) para ao menos dois "
            "protocolos: a seção principal (Likert) foi omitida em todos os "
            "recortes. Preencha `configuracao_comparacao.campos_parquet.avaliacao` "
            "com a coluna de avaliação do parquet (ou disponibilize os arquivos "
            "`{id}.avaliacao.json` na pasta de cada modelo). As métricas "
            "automáticas, complementares, seguem normalmente."
        )
        print(f"   ⚠️  {avisos[-1]}")

    # A ROPE ausente invalida TODAS as seções complementares, em qualquer
    # recorte — o aviso sai uma única vez, antes do laço.
    if cfg.campos and cfg.metricas and cfg.rope <= 0:
        avisos.append(
            "Métricas automáticas declaradas sem `metricas_automaticas.rope` > 0: "
            "as seções complementares foram omitidas. No modo `baycomp` a ROPE é "
            "obrigatória — sem margem, todo par vira `superior` ou `inferior`."
        )
        print(f"   ⚠️  {avisos[-1]}")

    # ── um bloco de análises por recorte ────────────────────────────────────
    # Recortes nomeados existem para o panorama completo: 16 protocolos geram
    # 120 pares e uma figura ilegível. Declarando um recorte por questão de
    # pesquisa, a MESMA comparação já processada rende várias figuras focadas —
    # e ajustá-las custa apenas `--bayesiana`, sem refazer a análise pesada.
    for recorte in cfg.recortes:
        parciais, avisos_recorte = _processar_recorte(
            recorte, cfg, analisador, dados_analise, pasta,
            rotulo_base, protocolos, mapa_aliases, sem_juiz_global)
        resultados += parciais
        avisos += avisos_recorte

    if not resultados:
        print("   ⚠️  Nenhum alvo bayesiano pôde ser calculado.")
        return {"resultados": [], "avisos": avisos, "arquivo_md": None}

    arquivo_md = gerar_markdown(resultados, cfg,
                                os.path.join(pasta, 'analise_bayesiana.md'),
                                avisos=avisos)
    print(f"   ✅ {len(resultados)} alvo(s) analisado(s) → {arquivo_md}")

    return {"resultados": resultados, "avisos": avisos, "arquivo_md": arquivo_md}
