"""
util_semclinbr.py — ponte entre o formato nativo do SemClinBr e o esquema JSON
usado no pipeline de extração (CL+PT).

Esquema da LLM (espelha os atributos do XML nativo):

    {"entities":  [{"id": 1, "text": "...", "tag": "A|B", "abbr": ""}],
     "relations": [{"annotation1": 2, "annotation2": 1, "reltype": "associated_with"}]}

Os offsets (start/end) NÃO são pedidos à LLM: são resolvidos por alinhamento
determinístico contra o texto original (alinhar_entidades).

Funções principais
------------------
parse_semclinbr_xml(path)              -> Documento (com offsets originais)
xml_to_target_json(doc)                -> dict  (gabarito, ids 1..n, sem offsets)
carregar_predicao(saida_bruta)         -> dict | None  (parse robusto da saída)
alinhar_entidades(texto, entidades)    -> list[Entidade] com start/end
json_para_xml(texto, pred)             -> str   (XML no formato nativo)
avaliar(gold, pred, modo)              -> dict  (P/R/F1)
avaliar_spans(gold, pred, parcial)     -> dict  (F1 ignorando rótulo — proxy S_i)
avaliar_por_sty(gold, pred, modo)      -> dict
avaliar_relacoes(gold_doc, ents, rels) -> dict
auditar_direcao_relacoes(docs)         -> dict  (checa a convenção de negation_of)
montar_instancia(doc, prompt_base)     -> dict  (linha do dataset de SFT)

Dependências: apenas biblioteca padrão.
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

# ---------------------------------------------------------------------------
# Estruturas
# ---------------------------------------------------------------------------


@dataclass
class Entidade:
    id: int
    text: str
    tags: list[str]          # interno: lista; serializa para "A|B" no JSON/XML
    abbr: str = ""
    start: int | None = None
    end: int | None = None
    alinhada: bool = True

    @property
    def span(self) -> tuple[int | None, int | None]:
        return (self.start, self.end)

    @property
    def tag_str(self) -> str:
        return "|".join(self.tags)


@dataclass
class Relacao:
    annotation1: int
    annotation2: int
    reltype: str


@dataclass
class Documento:
    doc_id: str
    texto: str
    entidades: list[Entidade] = field(default_factory=list)
    relacoes: list[Relacao] = field(default_factory=list)


def _split_tags(valor) -> list[str]:
    """Aceita 'A|B', ['A','B'] ou ''."""
    if valor is None:
        return []
    if isinstance(valor, (list, tuple)):
        return [str(t).strip() for t in valor if str(t).strip()]
    return [t.strip() for t in str(valor).split("|") if t.strip()]


# ---------------------------------------------------------------------------
# STY -> SGR (McCray et al., 2001). Parcial: cobre os STYs frequentes do corpus.
# Para os 100 STYs completos, carregue SemGroups.txt do UMLS via carregar_semgroups().
# ---------------------------------------------------------------------------

STY2SGR: dict[str, str] = {
    "Body Location or Region": "Anatomy",
    "Body Part, Organ, or Organ Component": "Anatomy",
    "Body Space or Junction": "Anatomy",
    "Body System": "Anatomy",
    "Tissue": "Anatomy",
    "Organic Chemical": "Chemicals & Drugs",
    "Pharmacologic Substance": "Chemicals & Drugs",
    "Antibiotic": "Chemicals & Drugs",
    "Element, Ion, or Isotope": "Chemicals & Drugs",
    "Quantitative Concept": "Concepts & Ideas",
    "Qualitative Concept": "Concepts & Ideas",
    "Temporal Concept": "Concepts & Ideas",
    "Spatial Concept": "Concepts & Ideas",
    "Functional Concept": "Concepts & Ideas",
    "Idea or Concept": "Concepts & Ideas",
    "Intellectual Product": "Concepts & Ideas",
    "Medical Device": "Devices",
    "Drug Delivery Device": "Devices",
    "Disease or Syndrome": "Disorders",
    "Finding": "Disorders",
    "Injury or Poisoning": "Disorders",
    "Sign or Symptom": "Disorders",
    "Pathologic Function": "Disorders",
    "Mental or Behavioral Dysfunction": "Disorders",
    "Neoplastic Process": "Disorders",
    "Anatomical Abnormality": "Disorders",
    "Patient or Disabled Group": "Living Beings",
    "Professional or Occupational Group": "Living Beings",
    "Population Group": "Living Beings",
    "Age Group": "Living Beings",
    "Family Group": "Living Beings",
    "Health Care Related Organization": "Organizations",
    "Laboratory or Test Result": "Phenomena",
    "Phenomenon or Process": "Phenomena",
    "Clinical Attribute": "Physiology",
    "Organism Function": "Physiology",
    "Physiologic Function": "Physiology",
    "Diagnostic Procedure": "Procedures",
    "Health Care Activity": "Procedures",
    "Therapeutic or Preventive Procedure": "Procedures",
    "Laboratory Procedure": "Procedures",
    "Educational Activity": "Procedures",
    # STYs extras do SemClinBr (não pertencem ao UMLS)
    "Abbreviation": "Abbreviation",
    "Negation": "Negation",
}


def carregar_semgroups(path: str | Path) -> None:
    """Carrega o SemGroups.txt oficial do UMLS (formato ABBR|GROUP|TUI|STY)."""
    for linha in Path(path).read_text(encoding="utf-8").splitlines():
        partes = linha.strip().split("|")
        if len(partes) == 4:
            STY2SGR[partes[3]] = partes[1]


def para_sgr(tags: Sequence[str]) -> frozenset[str]:
    return frozenset(STY2SGR.get(t, t) for t in tags)


# ---------------------------------------------------------------------------
# Leitura do XML nativo
# ---------------------------------------------------------------------------


def _mapa_offsets_crlf(texto: str) -> dict[int, int]:
    """Mapeia offset no espaço CRLF original -> offset no texto normalizado.

    Os offsets gravados no XML do SemClinBr foram calculados sobre o texto
    original, em que cada quebra de linha ocupa DOIS caracteres (`\\r\\n`).
    A normalização de fim de linha é obrigatória para qualquer parser XML
    (XML 1.0 §2.11), então `texto` chega aqui com `\\n` — um caractere a menos
    por quebra. Sem esta conversão, todo span depois da primeira quebra fica
    deslocado, e o deslocamento cresce com o número de quebras anteriores:
    o F1 por span vira ruído em vez de medir reconhecimento.
    """
    mapa: dict[int, int] = {}
    desloc = 0
    for i, ch in enumerate(texto):
        mapa[i + desloc] = i
        if ch == "\n":
            # a posição do `\r` que existia no original aponta para o mesmo `\n`
            desloc += 1
            mapa[i + desloc] = i
    mapa[len(texto) + desloc] = len(texto)
    return mapa


def parse_semclinbr_xml(path: str | Path) -> Documento:
    """Lê um arquivo .xml do SemClinBr, convertendo os offsets para o texto lido.

    Ver `_mapa_offsets_crlf`: os offsets do XML estão no espaço CRLF original e
    precisam ser reancorados no texto normalizado que o modelo de fato recebe.
    """
    path = Path(path)
    raiz = ET.fromstring(path.read_text(encoding="utf-8"))
    texto = raiz.findtext("TEXT") or ""
    mapa = _mapa_offsets_crlf(texto)
    fim = len(texto)

    entidades = [
        Entidade(
            id=int(ann.get("id")),
            text=ann.get("text", ""),
            tags=_split_tags(ann.get("tag", "")),
            abbr=ann.get("abbr", "") or "",
            start=mapa.get(int(ann.get("start")), fim),
            end=mapa.get(int(ann.get("end")), fim),
        )
        for ann in raiz.findall("./TAGS/annotation")
    ]

    relacoes = [
        Relacao(
            annotation1=int(rel.get("annotation1")),
            annotation2=int(rel.get("annotation2")),
            reltype=rel.get("reltype", ""),
        )
        for rel in raiz.findall("./RELATIONS/rel")
    ]

    return Documento(doc_id=path.stem, texto=texto, entidades=entidades, relacoes=relacoes)


def xml_to_target_json(doc: Documento) -> dict:
    """Converte o documento no gabarito que a LLM deve produzir.

    Reordena por (start, end) — ordem de aparição — e reindexa os ids em 1..n,
    a convenção exigida pelo prompt. Os ids originais (1259, 1260…) são
    arbitrários e não avaliáveis. Offsets são descartados: voltam depois via
    alinhar_entidades().

    O `text` é reescrito a partir do offset, não copiado do atributo do XML: o
    atributo vem tokenizado pela ferramenta de anotação ("35 , 7ºC", "MÉDIA
    QUANTIDADE .", espaços duplos colapsados) e não ocorre literalmente na nota.
    Treinar contra ele contradiria a instrução do prompt ("spans exactly as they
    appear") e quebraria o round-trip do alinhamento em ~4,5% das entidades.
    O span é a autoridade — ver README §2, "Canonicalização".
    """
    ordenadas = sorted(doc.entidades, key=lambda e: (e.start, e.end))
    remap = {ent.id: novo for novo, ent in enumerate(ordenadas, start=1)}

    return {
        "entities": [
            {"id": remap[e.id], "text": doc.texto[e.start:e.end] or e.text,
             "tag": e.tag_str, "abbr": e.abbr}
            for e in ordenadas
        ],
        "relations": [
            {
                "annotation1": remap[r.annotation1],
                "annotation2": remap[r.annotation2],
                "reltype": r.reltype,
            }
            for r in doc.relacoes
            if r.annotation1 in remap and r.annotation2 in remap
        ],
    }


# ---------------------------------------------------------------------------
# Parse robusto da saída do modelo
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def carregar_predicao(saida_bruta: str) -> dict | None:
    """Extrai o objeto JSON da saída do modelo.

    Retorna None quando não há JSON parseável — o chamador deve contabilizar
    isso na taxa de falha de parsing (métrica de robustez do protocolo).
    """
    if not saida_bruta:
        return None
    txt = _FENCE.sub("", saida_bruta).strip()
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        ini, fim = txt.find("{"), txt.rfind("}")
        if ini == -1 or fim <= ini:
            return None
        try:
            obj = json.loads(txt[ini : fim + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    obj.setdefault("entities", [])
    obj.setdefault("relations", [])
    return obj


def _para_entidade(bruta: dict, fallback_id: int) -> Entidade:
    try:
        eid = int(bruta.get("id", fallback_id))
    except (TypeError, ValueError):
        eid = fallback_id
    return Entidade(
        id=eid,
        text=str(bruta.get("text", "") or ""),
        tags=_split_tags(bruta.get("tag", bruta.get("tags"))),
        abbr=str(bruta.get("abbr", "") or ""),
    )


def para_relacoes(brutas: Iterable[dict]) -> list[Relacao]:
    saida = []
    for r in brutas or []:
        try:
            saida.append(
                Relacao(
                    annotation1=int(r["annotation1"]),
                    annotation2=int(r["annotation2"]),
                    reltype=str(r.get("reltype", "") or ""),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return saida


# ---------------------------------------------------------------------------
# Alinhamento de offsets (pós-processamento determinístico)
# ---------------------------------------------------------------------------

_ESPACOS = re.compile(r"\s+")


def _normaliza(s: str) -> str:
    """Casefold + remoção de acentos + colapso de espaços (só para fallback)."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return _ESPACOS.sub(" ", s).strip().casefold()


def _busca_flexivel(texto: str, alvo: str, inicio: int) -> tuple[int, int] | None:
    """Busca tolerante a variação de espaços em branco."""
    if not alvo.strip():
        return None
    padrao = r"\s+".join(re.escape(tok) for tok in alvo.split())
    m = re.compile(padrao, re.IGNORECASE).search(texto, inicio)
    return (m.start(), m.end()) if m else None


def _busca_fuzzy(texto: str, alvo: str, inicio: int, limiar: float) -> tuple[int, int] | None:
    """Última tentativa: janela deslizante com similaridade de sequência.

    Cobre o caso em que a LLM reescreve levemente o trecho (acento perdido,
    caixa alterada). Custo O(n) por entidade — aceitável para notas curtas
    (~148 tokens de média no SemClinBr).
    """
    alvo_norm = _normaliza(alvo)
    if not alvo_norm:
        return None
    largura = len(alvo)
    melhor, melhor_score = None, 0.0
    passo = max(1, largura // 4)
    for i in range(inicio, max(inicio, len(texto) - largura + 1) + 1, passo):
        for delta in (-2, 0, 2):
            j = i + largura + delta
            if j <= i or j > len(texto):
                continue
            score = difflib.SequenceMatcher(None, alvo_norm, _normaliza(texto[i:j])).ratio()
            if score > melhor_score:
                melhor, melhor_score = (i, j), score
    return melhor if melhor_score >= limiar else None


def alinhar_entidades(
    texto: str,
    entidades: Iterable[dict | Entidade],
    limiar_fuzzy: float = 0.90,
) -> list[Entidade]:
    """Resolve start/end para entidades sem offsets.

    Estratégia de cursor: a busca de cada entidade parte do `start` da entidade
    anterior (não do `end`), o que permite spans aninhados e sobrepostos
    ("CURATIVO" dentro de "CURATIVO COM CARVÃO ATIVADO") sem quebrar a ordem.
    Cascata: exata -> flexível a espaços -> fuzzy -> global -> falha.

    Entidades não alinhadas recebem alinhada=False e contam como falso-positivo
    na avaliação (taxa de não-alinhamento reportada à parte).
    """
    resultado: list[Entidade] = []
    cursor = 0

    for bruta in entidades:
        ent = bruta if isinstance(bruta, Entidade) else _para_entidade(bruta, len(resultado) + 1)

        alvo = ent.text
        pos = texto.find(alvo, cursor) if alvo else -1
        achado = (pos, pos + len(alvo)) if pos != -1 else None

        if achado is None:
            achado = _busca_flexivel(texto, alvo, cursor)
        if achado is None:
            achado = _busca_fuzzy(texto, alvo, cursor, limiar_fuzzy)
        if achado is None:  # recomeça do zero: a LLM pode ter quebrado a ordem
            pos = texto.find(alvo) if alvo else -1
            achado = (pos, pos + len(alvo)) if pos != -1 else _busca_flexivel(texto, alvo, 0)

        if achado is None:
            ent.start = ent.end = None
            ent.alinhada = False
        else:
            ent.start, ent.end = achado
            ent.alinhada = True
            # canonicaliza: o offset é a autoridade, não a cópia do modelo.
            # Corrige espaço duplicado, caixa e acento alterados pela LLM e
            # garante que o XML gerado seja internamente consistente.
            ent.text = texto[ent.start : ent.end]
            cursor = ent.start  # permite aninhamento

        resultado.append(ent)

    return resultado


# ---------------------------------------------------------------------------
# JSON -> XML nativo
# ---------------------------------------------------------------------------


def json_para_xml(texto: str, pred: dict, offset_inicial: int = 1) -> str:
    """Reconstrói o XML no formato SemClinBr a partir da predição alinhada."""
    entidades = alinhar_entidades(texto, pred.get("entities", []))
    validas = {e.id for e in entidades if e.alinhada}
    esc = {'"': "&quot;"}

    linhas = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        "<ANNOTATIONS>",
        f"<TEXT>{escape(texto)}</TEXT>",
        "<TAGS>",
    ]
    for e in entidades:
        if not e.alinhada:
            continue
        linhas.append(
            '<annotation id="{i}" tag="{tag}" start="{s}" end="{t}" text="{txt}" abbr="{ab}" />'.format(
                i=e.id + offset_inicial - 1,
                tag=escape(e.tag_str, esc),
                s=e.start,
                t=e.end,
                txt=escape(e.text, esc),
                ab=escape(e.abbr, esc),
            )
        )
    linhas += ["</TAGS>", "<RELATIONS>"]
    for r in para_relacoes(pred.get("relations", [])):
        if r.annotation1 in validas and r.annotation2 in validas:
            linhas.append(
                '<rel annotation1="{a}" annotation2="{b}" reltype="{t}" />'.format(
                    a=r.annotation1 + offset_inicial - 1,
                    b=r.annotation2 + offset_inicial - 1,
                    t=escape(r.reltype, esc),
                )
            )
    linhas += ["</RELATIONS>", "</ANNOTATIONS>"]
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Avaliação — os quatro critérios do artigo, como P/R/F1
# ---------------------------------------------------------------------------

MODOS = ("strict", "lenient", "flexible", "relaxed")


def _rotulo(ent: Entidade, modo: str) -> frozenset[str]:
    return para_sgr(ent.tags) if modo in ("flexible", "relaxed") else frozenset(ent.tags)


def _spans_iguais(a: Entidade, b: Entidade) -> bool:
    return a.start == b.start and a.end == b.end


def _spans_sobrepostos(a: Entidade, b: Entidade) -> bool:
    return a.start < b.end and b.start < a.end


def _prf(acertos: float, n_gold: int, n_pred: int) -> dict:
    p = acertos / n_pred if n_pred else 0.0
    r = acertos / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"acertos": acertos, "n_gold": n_gold, "n_pred": n_pred,
            "precisao": p, "revocacao": r, "f1": f1}


def avaliar(
    gold: Sequence[Entidade],
    pred: Sequence[Entidade],
    modo: str = "strict",
    rotulo_exato: bool = True,
) -> dict:
    """Precisão / revocação / F1 entre gabarito e predição.

    modo: strict (span exato + STY), lenient (span parcial + STY),
          flexible (span exato + SGR), relaxed (span parcial + SGR).

    Espelha o artigo: em lenient/relaxed a sobreposição parcial vale meio-acerto.
    rotulo_exato=True exige conjuntos de rótulos idênticos; False aceita
    interseção não vazia (diagnóstico de erro de multi-rótulo).
    """
    if modo not in MODOS:
        raise ValueError(f"modo deve ser um de {MODOS}")

    parcial_ok = modo in ("lenient", "relaxed")
    g = [e for e in gold if e.start is not None]
    p = [e for e in pred if e.alinhada and e.start is not None]
    nao_alinhadas = sum(1 for e in pred if not e.alinhada)

    usados: set[int] = set()
    acertos = 0.0

    for gi in g:
        for j, pj in enumerate(p):  # exatos têm prioridade sobre parciais
            if j in usados or not _spans_iguais(gi, pj):
                continue
            rg, rp = _rotulo(gi, modo), _rotulo(pj, modo)
            if (rg == rp) if rotulo_exato else bool(rg & rp):
                usados.add(j)
                acertos += 1.0
                break
        else:
            if not parcial_ok:
                continue
            for j, pj in enumerate(p):
                if j in usados or not _spans_sobrepostos(gi, pj):
                    continue
                rg, rp = _rotulo(gi, modo), _rotulo(pj, modo)
                if (rg == rp) if rotulo_exato else bool(rg & rp):
                    usados.add(j)
                    acertos += 0.5
                    break

    saida = _prf(acertos, len(g), len(p) + nao_alinhadas)
    saida.update({"modo": modo, "nao_alinhadas": nao_alinhadas})
    return saida


def avaliar_spans(
    gold: Sequence[Entidade], pred: Sequence[Entidade], parcial: bool = False
) -> dict:
    """F1 de detecção de span, IGNORANDO o rótulo.

    Esta é a métrica a usar no proxy de dificuldade S_i quando o prompt não
    fornece inventário de rótulos (condição principal, espelhando o PubMed):
    o modelo base zero-shot inventa nomes de tag, então as quatro métricas
    rotuladas colapsam perto de zero e perdem poder de estratificação, mas a
    detecção de span continua variando entre instâncias.
    """
    g = [e for e in gold if e.start is not None]
    p = [e for e in pred if e.alinhada and e.start is not None]
    nao_alinhadas = sum(1 for e in pred if not e.alinhada)

    usados: set[int] = set()
    acertos = 0.0
    for gi in g:
        for j, pj in enumerate(p):
            if j not in usados and _spans_iguais(gi, pj):
                usados.add(j)
                acertos += 1.0
                break
        else:
            if parcial:
                for j, pj in enumerate(p):
                    if j not in usados and _spans_sobrepostos(gi, pj):
                        usados.add(j)
                        acertos += 0.5
                        break

    saida = _prf(acertos, len(g), len(p) + nao_alinhadas)
    saida.update({"modo": "span_parcial" if parcial else "span_exato",
                  "nao_alinhadas": nao_alinhadas})
    return saida


def avaliar_por_sty(
    gold: Sequence[Entidade], pred: Sequence[Entidade], modo: str = "strict"
) -> dict[str, dict]:
    """F1 por rótulo — comparável a Souza et al. e à Fig. 4 do artigo."""
    rotulos = {t for e in gold for t in e.tags} | {t for e in pred for t in e.tags}
    saida = {}
    for r in sorted(rotulos):
        g = [e for e in gold if r in e.tags]
        p = [e for e in pred if r in e.tags]
        if g or p:
            saida[r] = avaliar(g, p, modo=modo, rotulo_exato=False)
    return saida


def avaliar_por_sgr(
    gold: Sequence[Entidade], pred: Sequence[Entidade], modo: str = "flexible"
) -> dict[str, dict]:
    """F1 por grupo semântico — comparável aos SGRs de Souza et al.

    Souza et al. reportam F1 0,76 para "Disorder" e 0,70 para "Procedure".
    Note que os nomes dos SGRs do UMLS são plurais ("Disorders", "Procedures");
    a correspondência é direta.
    """
    grupos = {g for e in gold for g in para_sgr(e.tags)} | {
        g for e in pred for g in para_sgr(e.tags)
    }
    saida = {}
    for gr in sorted(grupos):
        g = [e for e in gold if gr in para_sgr(e.tags)]
        p = [e for e in pred if gr in para_sgr(e.tags)]
        if g or p:
            saida[gr] = avaliar(g, p, modo=modo, rotulo_exato=False)
    return saida


def avaliar_relacoes(
    gold_doc: Documento, pred_ents: Sequence[Entidade], pred_rels: Sequence[Relacao]
) -> dict:
    """P/R/F1 de relações, ancoradas em spans (não em ids).

    Uma relação acerta quando os spans de origem e destino coincidem com os do
    gabarito e o reltype é igual. Ancorar em span evita punir a predição por
    numeração diferente.
    """

    def chave(ents: Sequence[Entidade], rels: Sequence[Relacao]) -> set:
        idx = {e.id: e for e in ents if getattr(e, "alinhada", True) and e.start is not None}
        return {
            (idx[r.annotation1].span, idx[r.annotation2].span, r.reltype)
            for r in rels
            if r.annotation1 in idx and r.annotation2 in idx
        }

    g = chave(gold_doc.entidades, gold_doc.relacoes)
    p = chave(pred_ents, pred_rels)
    return _prf(len(g & p), len(g), len(p))


# ---------------------------------------------------------------------------
# Linha de métricas por documento (unidade da análise pareada)
# ---------------------------------------------------------------------------


def avaliar_documento(
    doc: Documento, saida_bruta: str, protocolo: str = "", seed: int | None = None
) -> dict:
    """Produz UMA linha de métricas para um par (documento, protocolo).

    Esta é a unidade da análise estatística pareada: os mesmos documentos de
    teste passam por todos os protocolos, e Friedman/Wilcoxon operam sobre a
    coluna 'f1_strict' (variável primária), com as demais como complementares.

    Falha de parsing e não-alinhamento entram como colunas próprias, para que a
    taxa de falha seja reportada em vez de silenciosamente virar F1 zero.
    """
    linha = {
        "id_arquivo": doc.doc_id,
        "protocolo": protocolo,
        "seed": seed,
        "n_entidades_gold": len(doc.entidades),
        "n_relacoes_gold": len(doc.relacoes),
        "falha_parsing": False,
    }

    pred = carregar_predicao(saida_bruta)
    if pred is None:
        linha["falha_parsing"] = True
        for m in MODOS:
            linha[f"f1_{m}"] = 0.0
            linha[f"precisao_{m}"] = 0.0
            linha[f"revocacao_{m}"] = 0.0
        linha.update({"f1_span_exato": 0.0, "f1_span_parcial": 0.0,
                      "f1_relacoes": 0.0, "n_entidades_pred": 0,
                      "nao_alinhadas": 0, "taxa_nao_alinhamento": 0.0})
        return linha

    ents = alinhar_entidades(doc.texto, pred.get("entities", []))
    rels = para_relacoes(pred.get("relations", []))

    for m in MODOS:
        r = avaliar(doc.entidades, ents, modo=m)
        linha[f"f1_{m}"] = r["f1"]
        linha[f"precisao_{m}"] = r["precisao"]
        linha[f"revocacao_{m}"] = r["revocacao"]

    linha["f1_span_exato"] = avaliar_spans(doc.entidades, ents)["f1"]
    linha["f1_span_parcial"] = avaliar_spans(doc.entidades, ents, parcial=True)["f1"]
    linha["f1_relacoes"] = avaliar_relacoes(doc, ents, rels)["f1"]

    nao_alinhadas = sum(1 for e in ents if not e.alinhada)
    linha["n_entidades_pred"] = len(ents)
    linha["nao_alinhadas"] = nao_alinhadas
    linha["taxa_nao_alinhamento"] = nao_alinhadas / len(ents) if ents else 0.0
    return linha


# ---------------------------------------------------------------------------
# Auditoria da convenção de direção das relações
# ---------------------------------------------------------------------------


def auditar_direcao_relacoes(docs: Iterable[Documento]) -> dict:
    """Verifica empiricamente a direção de negation_of no corpus.

    Rode assim que baixar os XMLs. Se 'a1_eh_negacao' dominar, a pista de
    negação é annotation1; caso contrário, inverta na descrição do esquema.
    """
    contagem = {"a1_eh_negacao": 0, "a2_eh_negacao": 0, "indefinido": 0}
    for doc in docs:
        idx = {e.id: e for e in doc.entidades}
        for r in doc.relacoes:
            if r.reltype != "negation_of":
                continue
            a, b = idx.get(r.annotation1), idx.get(r.annotation2)
            if a is None or b is None:
                continue
            a_neg, b_neg = "Negation" in a.tags, "Negation" in b.tags
            if a_neg and not b_neg:
                contagem["a1_eh_negacao"] += 1
            elif b_neg and not a_neg:
                contagem["a2_eh_negacao"] += 1
            else:
                contagem["indefinido"] += 1
    return contagem


# ---------------------------------------------------------------------------
# Construção do dataset de treino
# ---------------------------------------------------------------------------


def montar_instancia(doc: Documento, prompt_base: str,
                     marcador: str = "<<--TEXTO-->>") -> dict:
    """Monta uma instância de SFT: prompt + gabarito JSON serializado.

    As colunas extras alimentam o componente estrutural do proxy de dificuldade.
    """
    gabarito = xml_to_target_json(doc)
    tags = [_split_tags(e["tag"]) for e in gabarito["entities"]]
    return {
        "id_arquivo": doc.doc_id,
        "prompt": prompt_base.replace(marcador, doc.texto),
        "gabarito": json.dumps(gabarito, ensure_ascii=False),
        "n_entidades": len(gabarito["entities"]),
        "n_relacoes": len(gabarito["relations"]),
        "n_rotulos_distintos": len({t for ts in tags for t in ts}),
        "n_multirotulo": sum(1 for ts in tags if len(ts) > 1),
        "n_chars": len(doc.texto),
    }


def carregar_corpus(diretorio: str | Path) -> list[Documento]:
    return [parse_semclinbr_xml(p) for p in sorted(Path(diretorio).glob("*.xml"))]


# ---------------------------------------------------------------------------
# Corpus: inventário de rótulos, splits e exportação do dataset de treino
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """Extract the structured clinical annotations from the clinical note below in tag <NOTE></NOTE>.

Use only the following semantic types as tags:
<<--TAGS-->>

Return a valid JSON object matching the following schema exactly:
{
  "entities": [
    {
      "id": "integer — sequential, starting at 1, following the order of appearance in the text",
      "text": "string — the span exactly as it appears in the text",
      "tag": "string — semantic type of the span; multiple types separated by |",
      "abbr": "string — expanded form when the span is an abbreviation"
    }
  ],
  "relations": [
    {
      "annotation1": "integer — id of the source entity",
      "annotation2": "integer — id of the target entity",
      "reltype": "string — associated_with or negation_of"
    }
  ]
}

If no entity or relation is present, return an empty list. If a field does not apply, return an empty string. Do not hallucinate information.
Make sure to extract the spans exactly as they appear in the original text, preserving spelling, accentuation and casing, and to list them in the order in which they appear in the text.

<NOTE>
<<--TEXTO-->>
</NOTE>
"""


class CorpusSemClinBr:
    """Carrega o corpus, define splits, deriva o inventário de rótulos e exporta
    o dataset de treinamento.

    Ordem obrigatória das operações:

        1. definir_splits()      -- particiona 70/20/10 de forma determinística
        2. inventario_tags()     -- deriva os rótulos APENAS do split de treino
        3. exportar()            -- grava dataset + divisão + prompt + inventário

    O inventário sai só do treino de propósito: derivá-lo do corpus inteiro
    deixaria vazar para o prompt a existência de STYs que só ocorrem no teste.
    É vazamento fraco (metadado, não rótulo por instância), mas evitá-lo é
    gratuito. Use apenas_treino=False apenas se quiser declarar o contrário.

    Uso:
        corpus = CorpusSemClinBr("dados/semclinbr_xml", seed=42)
        corpus.definir_splits()
        corpus.inventario_tags(cobertura=0.95)
        corpus.exportar("dados/")
    """

    SPLITS = ("treino", "teste", "validacao")

    def __init__(self, diretorio: str | Path, seed: int = 42):
        self.diretorio = Path(diretorio)
        self.seed = seed
        self.documentos: list[Documento] = carregar_corpus(self.diretorio)
        if not self.documentos:
            raise ValueError(f"Nenhum .xml encontrado em {self.diretorio}")
        self.split: dict[str, str] = {}
        self.tags: list[str] = []
        self.freq_tags: dict[str, int] = {}
        self.cobertura_inventario: float | None = None

    # -- splits -------------------------------------------------------------

    def _chave_estavel(self, doc_id: str) -> str:
        """Hash determinístico e independente da ordem dos arquivos no disco."""
        import hashlib

        return hashlib.md5(f"{self.seed}:{doc_id}".encode()).hexdigest()

    def _estrato(self, doc: Documento, n_faixas: int = 4) -> int:
        """Faixa de complexidade por quantidade de entidades (quartis)."""
        contagens = sorted(len(d.entidades) for d in self.documentos)
        if not contagens:
            return 0
        n = len(doc.entidades)
        for i in range(1, n_faixas):
            corte = contagens[int(len(contagens) * i / n_faixas)]
            if n < corte:
                return i - 1
        return n_faixas - 1

    def definir_splits(
        self,
        proporcoes: tuple[float, float, float] = (0.70, 0.20, 0.10),
        estratificar: bool = True,
    ) -> dict[str, str]:
        """Particiona em treino / teste / validação de forma determinística.

        A partição é estável sob mudança do conjunto de arquivos (usa hash do
        id, não a posição na lista) e estratificada por quartil de quantidade
        de entidades, para que as três partições tenham distribuição de
        complexidade comparável — pré-condição para o proxy S_i fazer sentido.
        """
        if abs(sum(proporcoes) - 1.0) > 1e-9:
            raise ValueError("proporcoes deve somar 1.0")

        grupos: dict[int, list[Documento]] = {}
        for doc in self.documentos:
            grupos.setdefault(self._estrato(doc) if estratificar else 0, []).append(doc)

        self.split = {}
        for docs in grupos.values():
            docs = sorted(docs, key=lambda d: self._chave_estavel(d.doc_id))
            n = len(docs)
            # maior resto: evita que o arredondamento por estrato acumule e
            # desloque as proporções globais (85/24/11 em vez de 84/24/12).
            exatos = [n * p for p in proporcoes]
            cotas = [int(x) for x in exatos]
            sobra = n - sum(cotas)
            ordem = sorted(range(3), key=lambda i: -(exatos[i] - cotas[i]))
            for i in ordem[:sobra]:
                cotas[i] += 1
            n_tr, n_te = cotas[0], cotas[1]
            for i, doc in enumerate(docs):
                if i < n_tr:
                    self.split[doc.doc_id] = "treino"
                elif i < n_tr + n_te:
                    self.split[doc.doc_id] = "teste"
                else:
                    self.split[doc.doc_id] = "validacao"
        return self.split

    def docs_do_split(self, nome: str) -> list[Documento]:
        return [d for d in self.documentos if self.split.get(d.doc_id) == nome]

    # -- inventário de rótulos ---------------------------------------------

    def inventario_tags(
        self,
        apenas_treino: bool = True,
        cobertura: float | None = None,
        minimo: int = 1,
    ) -> list[str]:
        """Deriva a lista de rótulos que vai no prompt, a partir do corpus real.

        cobertura: se informado (ex.: 0.95), trunca a lista nos rótulos mais
                   frequentes que cobrem essa fração das anotações; o resto vira
                   cauda longa fora do prompt.
        minimo:    frequência mínima para entrar na lista.

        Retorna a lista ordenada por frequência decrescente e registra
        self.cobertura_inventario — a fração das anotações do corpus INTEIRO
        cujos rótulos aparecem na lista. Esse número é característica declarada
        do experimento: é o teto imposto pelo prompt.
        """
        if apenas_treino and not self.split:
            raise RuntimeError("Chame definir_splits() antes de inventario_tags()")

        base = self.docs_do_split("treino") if apenas_treino else self.documentos
        freq: dict[str, int] = {}
        for doc in base:
            for ent in doc.entidades:
                for t in ent.tags:
                    freq[t] = freq.get(t, 0) + 1

        ordenadas = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        ordenadas = [(t, c) for t, c in ordenadas if c >= minimo]

        if cobertura is not None:
            total = sum(c for _, c in ordenadas)
            acumulado, corte = 0, len(ordenadas)
            for i, (_, c) in enumerate(ordenadas, start=1):
                acumulado += c
                if total and acumulado / total >= cobertura:
                    corte = i
                    break
            ordenadas = ordenadas[:corte]

        self.tags = [t for t, _ in ordenadas]
        self.freq_tags = dict(ordenadas)

        # cobertura medida sobre o corpus inteiro, não só o treino
        no_prompt = set(self.tags)
        dentro = fora = 0
        for doc in self.documentos:
            for ent in doc.entidades:
                for t in ent.tags:
                    if t in no_prompt:
                        dentro += 1
                    else:
                        fora += 1
        self.cobertura_inventario = dentro / (dentro + fora) if (dentro + fora) else 0.0
        return self.tags

    def montar_prompt(self, template: str = PROMPT_TEMPLATE,
                      por_linha: int = 3) -> str:
        """Injeta o inventário no template, deixando <<--TEXTO-->> para depois."""
        if not self.tags:
            raise RuntimeError("Chame inventario_tags() antes de montar_prompt()")
        linhas, atual = [], []
        for t in self.tags:
            atual.append(t)
            if len(atual) == por_linha:
                linhas.append("; ".join(atual) + ";")
                atual = []
        if atual:
            linhas.append("; ".join(atual) + ";")
        if linhas:
            linhas[-1] = linhas[-1][:-1] + "."
        return template.replace("<<--TAGS-->>", "\n".join(linhas))

    # -- exportação ---------------------------------------------------------

    def linhas_dataset(self, incluir_prompt: bool = False,
                       colunas_extras: bool = True) -> list[dict]:
        """Gera as linhas do dataset: id, texto, split, resposta (+ extras)."""
        if not self.split:
            raise RuntimeError("Chame definir_splits() antes de exportar")
        prompt = self.montar_prompt() if incluir_prompt else None

        linhas = []
        for doc in self.documentos:
            gabarito = xml_to_target_json(doc)
            linha = {
                "id": doc.doc_id,
                "texto": doc.texto,
                "split": self.split[doc.doc_id],
                "resposta": json.dumps(gabarito, ensure_ascii=False),
            }
            if incluir_prompt:
                linha["prompt"] = prompt.replace("<<--TEXTO-->>", doc.texto)
            if colunas_extras:
                tags = [_split_tags(e["tag"]) for e in gabarito["entities"]]
                fora = sum(1 for ts in tags for t in ts if t not in set(self.tags)) if self.tags else 0
                linha.update({
                    "n_entidades": len(gabarito["entities"]),
                    "n_relacoes": len(gabarito["relations"]),
                    "n_rotulos_distintos": len({t for ts in tags for t in ts}),
                    "n_multirotulo": sum(1 for ts in tags if len(ts) > 1),
                    "n_chars": len(doc.texto),
                    "n_tags_fora_do_prompt": fora,
                })
            linhas.append(linha)
        return linhas

    def exportar(
        self,
        destino: str | Path,
        nome: str = "semclinbr",
        incluir_prompt: bool = False,
        formato: str = "auto",
    ) -> dict[str, Path]:
        """Grava dataset, divisão, prompt e inventário.

        Arquivos gerados em `destino`:
          {nome}.parquet | {nome}.csv  -- id, texto, split, resposta (+ extras)
          divisao_{nome}.csv           -- id_arquivo, alvo (formato do framework)
          prompt_{nome}.txt            -- prompt com o inventário já injetado
          inventario_{nome}.csv        -- rotulo, frequencia (no treino)

        O prompt é gravado junto porque o inventário é derivado dos dados:
        sem esse arquivo, o experimento não é reprodutível.
        """
        destino = Path(destino)
        destino.mkdir(parents=True, exist_ok=True)
        linhas = self.linhas_dataset(incluir_prompt=incluir_prompt)
        gerados: dict[str, Path] = {}

        usar_parquet = formato == "parquet" or formato == "auto"
        if usar_parquet:
            try:
                import pandas as pd

                caminho = destino / f"{nome}.parquet"
                pd.DataFrame(linhas).to_parquet(caminho, index=False)
                gerados["dataset"] = caminho
            except Exception:
                usar_parquet = False
        if not usar_parquet:
            import csv as _csv

            caminho = destino / f"{nome}.csv"
            with caminho.open("w", encoding="utf-8", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=list(linhas[0].keys()))
                w.writeheader()
                w.writerows(linhas)
            gerados["dataset"] = caminho

        import csv as _csv

        caminho = destino / f"divisao_{nome}.csv"
        with caminho.open("w", encoding="utf-8", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["id_arquivo", "alvo"])
            for ln in linhas:
                w.writerow([ln["id"], ln["split"]])
        gerados["divisao"] = caminho

        if self.tags:
            caminho = destino / f"prompt_{nome}.txt"
            caminho.write_text(self.montar_prompt(), encoding="utf-8")
            gerados["prompt"] = caminho

            caminho = destino / f"inventario_{nome}.csv"
            with caminho.open("w", encoding="utf-8", newline="") as fh:
                w = _csv.writer(fh)
                w.writerow(["rotulo", "frequencia_treino"])
                for t in self.tags:
                    w.writerow([t, self.freq_tags[t]])
            gerados["inventario"] = caminho

        return gerados

    # -- diagnóstico --------------------------------------------------------

    def estatisticas(self) -> dict:
        """Resumo para conferência antes de treinar."""
        por_split = {s: len(self.docs_do_split(s)) for s in self.SPLITS} if self.split else {}
        n = len(self.documentos)
        ents = [len(d.entidades) for d in self.documentos]
        return {
            "n_documentos": n,
            "n_entidades": sum(ents),
            "n_relacoes": sum(len(d.relacoes) for d in self.documentos),
            "entidades_por_doc_min_mediana_max": (
                min(ents), sorted(ents)[len(ents) // 2], max(ents)
            ) if ents else None,
            "docs_por_split": por_split,
            "proporcao_por_split": {s: round(v / n, 4) for s, v in por_split.items()},
            "n_rotulos_no_corpus": len(
                {t for d in self.documentos for e in d.entidades for t in e.tags}
            ),
            "n_rotulos_no_prompt": len(self.tags),
            "cobertura_inventario": self.cobertura_inventario,
        }


if __name__ == '__main__':
    from pprint import pprint

    diretorio_base = Path(__file__).parent / "dados"
    diretorio_xml = diretorio_base / "SemClinBr-xml-public-v1"
    
    print(f"Carregando corpus de {diretorio_xml}...")
    corpus = CorpusSemClinBr(diretorio_xml, seed=42)
    
    print("Definindo splits (70/20/10)...")
    corpus.definir_splits()
    
    print("Derivando inventário de rótulos do conjunto de treinamento...")
    # Apenas do treino para não vazar tags exclusivas de teste no prompt
    corpus.inventario_tags(apenas_treino=True)
    
    print(f"Exportando arquivos para {diretorio_base}...")
    arquivos_gerados = corpus.exportar(
        destino=diretorio_base, 
        nome="semclinbr", 
        incluir_prompt=True
    )
    
    # Gabarito no formato que o framework de comparação e de treino espera
    # (colunas `chave` e `resposta`), equivalente ao saida_pubmed_prof.parquet
    # do experimento PubMed. É o `modelo_base` de 03/06 e o gold dataset do 04.
    arquivo_gabarito = Path(__file__).parent / "saidas" / "saida_semclinbr_gold.parquet"
    print(f"Gravando gabarito em {arquivo_gabarito}...")
    import pandas as pd

    linhas = corpus.linhas_dataset(colunas_extras=False)
    df_gabarito = pd.DataFrame({
        "chave": [ln["id"] for ln in linhas],
        "resposta": [ln["resposta"] for ln in linhas],
        "erro": "",
    })
    arquivo_gabarito.parent.mkdir(parents=True, exist_ok=True)
    df_gabarito.to_parquet(arquivo_gabarito, index=False)
    arquivos_gerados["gabarito"] = arquivo_gabarito

    print("\nEstatísticas do Corpus:")
    pprint(corpus.estatisticas())

    print("\nArquivos gerados:")
    for tipo, caminho in arquivos_gerados.items():
        print(f"  {tipo}: {caminho}")