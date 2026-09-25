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
    stats_parse: dict = field(default_factory=dict)


def _esc_md(texto: str) -> str:
    """Escapa um valor para caber numa célula de tabela Markdown."""
    return (texto.replace("\\", "\\\\").replace("|", "\\|")
                 .replace("\r", "\\r").replace("\n", "\\n"))


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
    # Anatomy
    "Body Location or Region": "Anatomy",
    "Body Part, Organ, or Organ Component": "Anatomy",
    "Body Space or Junction": "Anatomy",
    "Body Substance": "Anatomy",
    "Body System": "Anatomy",
    "Cell": "Anatomy",
    "Tissue": "Anatomy",
    # Chemicals & Drugs
    "Amino Acid, Peptide, or Protein": "Chemicals & Drugs",
    "Antibiotic": "Chemicals & Drugs",
    "Biologically Active Substance": "Chemicals & Drugs",
    "Clinical Drug": "Chemicals & Drugs",
    "Element, Ion, or Isotope": "Chemicals & Drugs",
    "Enzyme": "Chemicals & Drugs",
    "Hazardous or Poisonous Substance": "Chemicals & Drugs",
    "Hormone": "Chemicals & Drugs",
    "Immunologic Factor": "Chemicals & Drugs",
    "Inorganic Chemical": "Chemicals & Drugs",
    "Nucleic Acid, Nucleoside, or Nucleotide": "Chemicals & Drugs",
    "Organic Chemical": "Chemicals & Drugs",
    "Pharmacologic Substance": "Chemicals & Drugs",
    "Substance": "Chemicals & Drugs",
    "Vitamin": "Chemicals & Drugs",
    # Concepts & Ideas
    "Classification": "Concepts & Ideas",
    "Functional Concept": "Concepts & Ideas",
    "Idea or Concept": "Concepts & Ideas",
    "Intellectual Product": "Concepts & Ideas",
    "Qualitative Concept": "Concepts & Ideas",
    "Quantitative Concept": "Concepts & Ideas",
    "Regulation or Law": "Concepts & Ideas",
    "Spatial Concept": "Concepts & Ideas",
    "Temporal Concept": "Concepts & Ideas",
    # Devices
    "Drug Delivery Device": "Devices",
    "Medical Device": "Devices",
    # Disorders
    "Acquired Abnormality": "Disorders",
    "Anatomical Abnormality": "Disorders",
    "Cell or Molecular Dysfunction": "Disorders",
    "Congenital Abnormality": "Disorders",
    "Disease or Syndrome": "Disorders",
    "Finding": "Disorders",
    "Injury or Poisoning": "Disorders",
    "Mental or Behavioral Dysfunction": "Disorders",
    "Neoplastic Process": "Disorders",
    "Pathologic Function": "Disorders",
    "Sign or Symptom": "Disorders",
    # Genes & Molecular Sequences
    "Amino Acid Sequence": "Genes & Molecular Sequences",
    # Living Beings
    "Age Group": "Living Beings",
    "Bacterium": "Living Beings",
    "Family Group": "Living Beings",
    "Fish": "Living Beings",
    "Fungus": "Living Beings",
    "Group": "Living Beings",
    "Patient or Disabled Group": "Living Beings",
    "Plant": "Living Beings",
    "Population Group": "Living Beings",
    "Professional or Occupational Group": "Living Beings",
    "Virus": "Living Beings",
    # Objects / Manufactured Objects
    "Biomedical or Dental Material": "Objects",
    "Food": "Objects",
    "Manufactured Object": "Objects",
    "Physical Object": "Objects",
    # Occupations
    "Biomedical Occupation or Discipline": "Occupations",
    # Organizations
    "Health Care Related Organization": "Organizations",
    "Organization": "Organizations",
    # Phenomena
    "Event": "Phenomena",
    "Laboratory or Test Result": "Phenomena",
    "Natural Phenomenon or Process": "Phenomena",
    "Phenomenon or Process": "Phenomena",
    # Physiology
    "Clinical Attribute": "Physiology",
    "Mental Process": "Physiology",
    "Molecular Function": "Physiology",
    "Organ or Tissue Function": "Physiology",
    "Organism Attribute": "Physiology",
    "Organism Function": "Physiology",
    "Physiologic Function": "Physiology",
    # Procedures
    "Diagnostic Procedure": "Procedures",
    "Educational Activity": "Procedures",
    "Health Care Activity": "Procedures",
    "Laboratory Procedure": "Procedures",
    "Therapeutic or Preventive Procedure": "Procedures",
    # Activities & Behaviors
    "Activity": "Activities & Behaviors",
    "Behavior": "Activities & Behaviors",
    "Daily or Recreational Activity": "Activities & Behaviors",
    "Individual Behavior": "Activities & Behaviors",
    "Machine Activity": "Activities & Behaviors",
    "Occupational Activity": "Activities & Behaviors",
    "Research Activity": "Activities & Behaviors",
    "Social Behavior": "Activities & Behaviors",
    # Corpus specific
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


def _mapa_crlf_para_lf(texto_crlf: str) -> dict[int, int]:
    """Mapeia offset no espaço CRLF -> offset no texto normalizado (LF).

    Para cada posição no texto CRLF, calcula a posição correspondente no
    texto onde ``\\r\\n`` foi substituído por ``\\n`` (normalização XML 1.0 §2.11).
    """
    mapa: dict[int, int] = {}
    pos_lf = 0
    for pos_crlf, ch in enumerate(texto_crlf):
        mapa[pos_crlf] = pos_lf
        if ch != "\r":          # \r é absorvido; \n avança normalmente
            pos_lf += 1
    mapa[len(texto_crlf)] = pos_lf
    return mapa


def _spans_compativeis(span: str, attr: str) -> bool:
    """Verifica se o span (do texto) corresponde ao atributo text do XML.

    A ferramenta de anotação tokeniza o atributo text (insere espaços antes
    de pontuação, colapsa espaços duplos, escapa entidades XML). A comparação
    ignora todas as diferenças de whitespace e de entidades XML.
    """
    def normalizar(t: str) -> str:
        t = t.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
        return re.sub(r"\s+", "", t)
    return normalizar(span) == normalizar(attr)


def _trim_span(texto: str, start: int, end: int) -> tuple[int, int]:
    """Remove whitespace das bordas do span (espaço, tab, ``\\n``, ``\\r``)."""
    while start < end and texto[start] in " \t\n\r":
        start += 1
    while end > start and texto[end - 1] in " \t\n\r":
        end -= 1
    return start, end


def _candidato_valido(
    texto: str, start: int, end: int, attr: str
) -> tuple[int, int] | None:
    """Devolve o span (já trimado) se ele corresponder ao atributo text; senão None."""
    if start < 0 or end > len(texto) or start >= end:
        return None
    s, e = _trim_span(texto, start, end)
    if s >= e:
        return None
    return (s, e) if _spans_compativeis(texto[s:e], attr) else None


def _deltas_por_raio(raio: int) -> list[tuple[int, int]]:
    """Deslocamentos (Δstart, Δend) ordenados da menor para a maior correção.

    A ordem garante que a entidade seja fixada pelo ajuste mínimo que a torna
    compatível com o atributo ``text``: primeiro os que mexem em uma borda só,
    depois os que mexem nas duas. ``(0, 0)`` fica de fora (já foi testado).
    """
    deltas = [
        (ds, de)
        for ds in range(-raio, raio + 1)
        for de in range(-raio, raio + 1)
        if (ds, de) != (0, 0)
    ]
    deltas.sort(key=lambda d: (abs(d[0]) + abs(d[1]), abs(d[0]), abs(d[1]), d[0], d[1]))
    return deltas


def resolver_span(
    texto: str, start: int, end: int, attr: str, raio: int = 1
) -> tuple[int, int, str, tuple[int, int]]:
    """Passada única que fixa a posição final de uma entidade.

    Cascata, da correção nula à maior, parando na primeira que valida contra o
    atributo ``text`` do XML (via :func:`_spans_compativeis`):

    1. ``exata``       — o span já corresponde;
    2. ``trim``        — corresponde depois de remover whitespace das bordas;
    3. ``shift``       — corresponde deslocando as bordas em até ``raio``
       caracteres para a esquerda e/ou direita (o corpus tem ~400 spans com a
       borda esquerda um caractere adiantada: ``"ORADA"`` para ``"CORADA"``);
    4. ``descartada``  — nenhum candidato valida; a entidade é removida.

    Retorna ``(start, end, tipo, (Δstart, Δend))``. Em ``descartada`` os
    offsets voltam inalterados, para o relatório poder mostrar o span original.
    """
    alvo = _candidato_valido(texto, start, end, attr)
    if alvo is not None:
        tipo = "exata" if alvo == (start, end) else "trim"
        return alvo[0], alvo[1], tipo, (0, 0)

    for ds, de in _deltas_por_raio(raio):
        alvo = _candidato_valido(texto, start + ds, end + de, attr)
        if alvo is not None:
            return alvo[0], alvo[1], "shift", (ds, de)

    return start, end, "descartada", (0, 0)


def _pontuar_espaco(texto_lf: str, pares: list[tuple[int, int]], attrs: list[str]) -> int:
    """Quantas anotações validam contra o atributo text sob um espaço de offset."""
    return sum(
        1
        for (s, e), attr in zip(pares, attrs)
        if _candidato_valido(texto_lf, s, e, attr) is not None
    )


def _escolher_espaco_offsets(
    texto_crlf: str, texto_lf: str, offsets: list[tuple[int, int]], attrs: list[str]
) -> tuple[list[tuple[int, int]], str]:
    """Decide se os offsets do documento estão no espaço CRLF ou LF.

    O corpus SemClinBr é inconsistente: a maioria dos documentos com quebra de
    linha tem offsets calculados sobre o texto CRLF original, mas parte já está
    no espaço LF (pós-normalização XML 1.0 §2.11).

    A decisão pontua **todas** as anotações do documento nos dois espaços, com a
    comparação tolerante de :func:`_spans_compativeis`, e fica com o que validar
    mais. As duas exigências são necessárias: anotações que caem antes da
    primeira quebra de linha validam nos dois espaços e não discriminam, e o
    atributo ``text`` cru não casa nos documentos com entidades XML duplamente
    escapadas (``&amp;gt;``). Amostrar poucas anotações, ou comparar sem
    tolerância, escolhe LF indevidamente em 26 documentos (932 anotações).

    Empate mantém LF (nenhum mapeamento aplicado), que é o caso conservador.
    """
    fim = len(texto_lf)
    pares_lf = [(min(s, fim), min(e, fim)) for s, e in offsets]
    if "\r\n" not in texto_crlf or not offsets:
        return pares_lf, "LF"

    mapa = _mapa_crlf_para_lf(texto_crlf)
    pares_crlf = [(mapa.get(s, fim), mapa.get(e, fim)) for s, e in offsets]

    n_lf = _pontuar_espaco(texto_lf, pares_lf, attrs)
    n_crlf = _pontuar_espaco(texto_lf, pares_crlf, attrs)
    return (pares_crlf, "CRLF") if n_crlf > n_lf else (pares_lf, "LF")


def parse_semclinbr_xml(path: str | Path, raio_ajuste: int = 2) -> Documento:
    """Lê um arquivo .xml do SemClinBr, convertendo os offsets para o texto lido.

    O corpus é inconsistente na convenção de offsets: a maioria dos documentos
    grava offsets no espaço CRLF (texto com ``\\r\\n``), mas um subconjunto já
    usa o espaço LF (pós-normalização XML). ``_escolher_espaco_offsets``
    auto-detecta qual convenção cada documento segue, pontuando as duas.

    Resolvido o espaço, **uma única passada** (:func:`resolver_span`) fixa a
    posição de cada entidade, sempre validando contra o atributo ``text`` do
    XML: span exato → trim de whitespace nas bordas → deslocamento de até
    ``raio_ajuste`` caracteres em cada borda → descarte.

    ``raio_ajuste=1`` (padrão) cobre o erro de borda de um caractere, que é o
    padrão dominante no corpus. ``raio_ajuste=2`` recupera ~16 entidades a mais;
    ``0`` desliga o ajuste por deslocamento.

    As estatísticas de correção ficam em ``doc.stats_parse``.
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    raiz = ET.fromstring(raw_bytes.decode("utf-8"))
    texto_lf = raiz.findtext("TEXT") or ""      # parser XML normaliza CRLF→LF

    annotations = raiz.findall("./TAGS/annotation")

    # --- extrair o TEXT com \r\n preservado (antes da normalização XML) ---
    marcador_ini = b"<TEXT>"
    marcador_fim = b"</TEXT>"
    idx_ini = raw_bytes.find(marcador_ini)
    idx_fim = raw_bytes.find(marcador_fim)
    if idx_ini >= 0 and idx_fim > idx_ini:
        texto_crlf = raw_bytes[idx_ini + len(marcador_ini):idx_fim].decode("utf-8")
    else:
        texto_crlf = texto_lf               # fallback conservador

    # --- decidir o espaço de offsets e mapear ---
    offsets = [(int(a.get("start")), int(a.get("end"))) for a in annotations]
    attrs = [a.get("text", "") for a in annotations]
    pares, espaco = _escolher_espaco_offsets(texto_crlf, texto_lf, offsets, attrs)

    # --- passada única de ajuste de posição + validação ---
    entidades: list[Entidade] = []
    contagem = {"exata": 0, "trim": 0, "shift": 0, "descartada": 0}
    descartadas: list[dict] = []
    ajustadas: list[dict] = []

    n_trim_aplicado = 0

    for ann, attr, (start, end) in zip(annotations, attrs, pares):
        s, e, tipo, delta = resolver_span(texto_lf, start, end, attr, raio=raio_ajuste)
        contagem[tipo] += 1
        # o trim pode atuar sozinho ou depois do deslocamento ("IRC " -> "IRC")
        trim_atuou = tipo != "descartada" and (s, e) != (start + delta[0], end + delta[1])
        n_trim_aplicado += int(trim_atuou)

        if tipo == "descartada":
            descartadas.append({
                "doc_id": path.stem,
                "id": int(ann.get("id")),
                "attr_text": attr,
                "span": texto_lf[start:end] if 0 <= start < end <= len(texto_lf)
                        else "(fora do texto)",
                "start": start,
                "end": end,
            })
            continue

        if tipo != "exata":
            ajustadas.append({
                "doc_id": path.stem,
                "id": int(ann.get("id")),
                "tipo": tipo,
                "attr_text": attr,
                "span_antes": texto_lf[start:end] if 0 <= start < end <= len(texto_lf)
                              else "(fora do texto)",
                "span_depois": texto_lf[s:e],
                "delta": delta,
                "trim": trim_atuou,
                "start": s,
                "end": e,
            })

        entidades.append(Entidade(
            id=int(ann.get("id")),
            text=attr,
            tags=_split_tags(ann.get("tag", "")),
            abbr=ann.get("abbr", "") or "",
            start=s,
            end=e,
        ))

    # Relações: manter apenas as que referenciam entidades preservadas
    ids_preservados = {ent.id for ent in entidades}
    relacoes = [
        Relacao(
            annotation1=int(rel.get("annotation1")),
            annotation2=int(rel.get("annotation2")),
            reltype=rel.get("reltype", ""),
        )
        for rel in raiz.findall("./RELATIONS/rel")
        if int(rel.get("annotation1")) in ids_preservados
        and int(rel.get("annotation2")) in ids_preservados
    ]

    stats = {
        "n_exatas": contagem["exata"],
        "n_corrigidas_trim": contagem["trim"],
        "n_corrigidas_shift": contagem["shift"],
        "n_descartadas": contagem["descartada"],
        "n_trim_aplicado": n_trim_aplicado,
        "raio_ajuste": raio_ajuste,
        "n_total_xml": len(annotations),
        "espaco_offsets": espaco,
        "descartadas": descartadas,
        "ajustadas": ajustadas,
    }

    return Documento(
        doc_id=path.stem, texto=texto_lf,
        entidades=entidades, relacoes=relacoes, stats_parse=stats,
    )


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
    Spans já ocupados são rastreados para que palavras repetidas avancem para a
    próxima ocorrência em vez de colapsarem na primeira.
    Cascata: exata -> flexível a espaços -> fuzzy -> global -> falha.

    Entidades não alinhadas recebem alinhada=False e contam como falso-positivo
    na avaliação (taxa de não-alinhamento reportada à parte).
    """
    resultado: list[Entidade] = []
    cursor = 0
    spans_ocupados: set[tuple[int, int]] = set()

    for bruta in entidades:
        ent = bruta if isinstance(bruta, Entidade) else _para_entidade(bruta, len(resultado) + 1)

        alvo = ent.text
        if not alvo:
            ent.start = ent.end = None
            ent.alinhada = False
            resultado.append(ent)
            continue

        # 1. Busca exata a partir do cursor, pulando spans já ocupados
        achado = None
        pos = cursor
        while pos < len(texto):
            p = texto.find(alvo, pos)
            if p == -1:
                break
            cand = (p, p + len(alvo))
            if cand not in spans_ocupados:
                achado = cand
                break
            pos = p + 1

        # 2. Busca flexível a partir do cursor
        if achado is None:
            cand = _busca_flexivel(texto, alvo, cursor)
            if cand and cand not in spans_ocupados:
                achado = cand

        # 3. Busca fuzzy a partir do cursor
        if achado is None:
            cand = _busca_fuzzy(texto, alvo, cursor, limiar_fuzzy)
            if cand and cand not in spans_ocupados:
                achado = cand

        # 4. Recomeça do zero: a LLM pode ter quebrado a ordem do texto
        if achado is None:
            pos = 0
            while pos < len(texto):
                p = texto.find(alvo, pos)
                if p == -1:
                    break
                cand = (p, p + len(alvo))
                if cand not in spans_ocupados:
                    achado = cand
                    break
                pos = p + 1
            if achado is None:
                cand = _busca_flexivel(texto, alvo, 0)
                if cand and cand not in spans_ocupados:
                    achado = cand

        if achado is None:
            ent.start = ent.end = None
            ent.alinhada = False
        else:
            ent.start, ent.end = achado
            ent.alinhada = True
            # canonicaliza: o offset é a autoridade, não a cópia do modelo.
            ent.text = texto[ent.start : ent.end]
            spans_ocupados.add(achado)
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
    numeração diferente. A relação 'associated_with' é simétrica (não-direcionada).
    """

    def chave(ents: Sequence[Entidade], rels: Sequence[Relacao]) -> set:
        idx = {e.id: e for e in ents if getattr(e, "alinhada", True) and e.start is not None}
        chaves = set()
        for r in rels:
            if r.annotation1 in idx and r.annotation2 in idx:
                s1, s2 = idx[r.annotation1].span, idx[r.annotation2].span
                if r.reltype == "associated_with":
                    # associacao é nao-direcionada: canonicaliza por ordem de span
                    chaves.add((min(s1, s2), max(s1, s2), r.reltype))
                else:
                    chaves.add((s1, s2, r.reltype))
        return chaves

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
        linha["acertos_strict"] = 0.0
        linha["f1_strict_overlap"] = 0.0
        linha["precisao_strict_overlap"] = 0.0
        linha["revocacao_strict_overlap"] = 0.0
        linha["acertos_strict_overlap"] = 0.0
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
        if m == "strict":
            linha["acertos_strict"] = r["acertos"]

    # F1 estrito tolerante a multi-rótulo (interseção não-vazia de rótulos)
    r_ov = avaliar(doc.entidades, ents, modo="strict", rotulo_exato=False)
    linha["f1_strict_overlap"] = r_ov["f1"]
    linha["precisao_strict_overlap"] = r_ov["precisao"]
    linha["revocacao_strict_overlap"] = r_ov["revocacao"]
    linha["acertos_strict_overlap"] = r_ov["acertos"]

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
    """Carrega o corpus, deriva o inventário de rótulos e exporta o dataset.

    **Esta classe não divide o corpus em treino/teste/validação.** Quem define
    os alvos é o passo 03 (`03_compara_gold_full.yaml`), que calcula a
    dificuldade de cada documento e grava `dados/divisao_Gold_Qwen7B.csv` com
    as colunas `id`, `alvo` e `dificuldade`. Todos os passos seguintes (04
    treino, 05 extração, 07 NER, 08 baseline) se apoiam nesse arquivo, e só nos
    ids que constam nele. Um segundo sorteio aqui criaria uma divisão paralela
    que não coincide com a operativa — caminho direto para data leakage.

    Ordem das operações:

        1. inventario_tags()  -- rótulos do prompt, derivados do corpus
        2. exportar()         -- dataset + prompt + inventário

    Uso:
        corpus = CorpusSemClinBr("dados/SemClinBr-xml-public-v1")
        corpus.inventario_tags()
        corpus.exportar("dados/")
    """

    def __init__(self, diretorio: str | Path):
        self.diretorio = Path(diretorio)
        self.documentos: list[Documento] = carregar_corpus(self.diretorio)
        if not self.documentos:
            raise ValueError(f"Nenhum .xml encontrado em {self.diretorio}")
        self.tags: list[str] = []
        self.freq_tags: dict[str, int] = {}
        self.cobertura_inventario: float | None = None

    # -- inventário de rótulos ---------------------------------------------

    def inventario_tags(
        self,
        cobertura: float | None = None,
        minimo: int = 1,
    ) -> list[str]:
        """Deriva a lista de rótulos que vai no prompt, a partir do corpus.

        cobertura: se informado (ex.: 0.95), trunca a lista nos rótulos mais
                   frequentes que cobrem essa fração das anotações; o resto vira
                   cauda longa fora do prompt.
        minimo:    frequência mínima para entrar na lista.

        O inventário sai do **corpus inteiro**, não de um split. Isso torna o
        prompt uma propriedade do corpus, fixada antes de existir qualquer
        divisão: os quatro protocolos comparados (A zero-shot, B, C, D*) recebem
        exatamente o mesmo estímulo, que é a condição do desenho pareado. Derivar
        do treino exigiria a saída do modelo base, que por sua vez depende do
        prompt — circularidade — e faria o protocolo A rodar com um prompt
        diferente dos demais.

        O custo é declarado no README §3: quatro STYs do corpus só ocorrem no
        split de teste (5 de 8 699 anotações, 0,06%). É vazamento de metadado,
        não de rótulo por instância — o prompt não diz qual documento tem qual
        rótulo, e os modelos ajustados sequer conseguem emitir esses tipos,
        porque eles nunca aparecem nos alvos de treino.

        Retorna a lista ordenada por frequência decrescente e registra
        self.cobertura_inventario — a fração das anotações do corpus cujos
        rótulos aparecem na lista. Esse número é característica declarada do
        experimento: é o teto imposto pelo prompt.
        """
        freq: dict[str, int] = {}
        for doc in self.documentos:
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
        """Gera as linhas do dataset: id, texto, resposta (+ extras).

        **Sem coluna `split`**: a divisão treino/teste/validação é atribuição do
        passo 03, que a grava em `divisao_Gold_Qwen7B.csv`. Quem precisa de um
        subconjunto filtra por aquele arquivo (nos YAMLs, via `filtro_externo`
        ou `arquivo_referencia`), nunca por uma coluna deste parquet.
        """
        prompt = self.montar_prompt() if incluir_prompt else None

        linhas = []
        for doc in self.documentos:
            gabarito = xml_to_target_json(doc)
            linha = {
                "id": doc.doc_id,
                "texto": doc.texto,
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
        """Grava dataset, prompt e inventário.

        Arquivos gerados em `destino`:
          {nome}.parquet | {nome}.csv  -- id, texto, resposta (+ extras)
          prompt_{nome}.txt            -- prompt com o inventário já injetado
          inventario_{nome}.csv        -- rotulo, frequencia
          {nome}.md                    -- relatório de qualidade das anotações

        **Nenhum arquivo de divisão é gerado aqui.** A divisão operativa é
        `divisao_Gold_Qwen7B.csv`, produzida pelo passo 03 a partir dos
        critérios de dificuldade.

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

        if self.tags:
            caminho = destino / f"prompt_{nome}.txt"
            caminho.write_text(self.montar_prompt(), encoding="utf-8")
            gerados["prompt"] = caminho

            caminho = destino / f"inventario_{nome}.csv"
            with caminho.open("w", encoding="utf-8", newline="") as fh:
                w = _csv.writer(fh)
                w.writerow(["rotulo", "frequencia_corpus"])
                for t in self.tags:
                    w.writerow([t, self.freq_tags[t]])
            gerados["inventario"] = caminho

        # Relatório de qualidade das anotações (mesmo nome base do dataset)
        caminho_md = destino / f"{nome}.md"
        self.gerar_relatorio_qualidade_md(caminho_md)
        gerados["relatorio_qualidade"] = caminho_md

        return gerados

    def gerar_relatorio_qualidade_md(self, caminho_md: Path) -> None:
        """Gera um relatório .md com as estatísticas de qualidade das anotações.

        O relatório reflete a passada única de ajuste de posição
        (:func:`resolver_span`), separando cada desfecho:

        - ``exata``      — o offset do XML já casava com o atributo ``text``;
        - ``trim``       — casou após remover whitespace das bordas;
        - ``shift``      — casou após deslocar as bordas em até ``raio_ajuste``
          caracteres;
        - ``descartada`` — nenhum candidato validou; entidade removida.

        Lista individualmente as entidades ajustadas e as descartadas.
        """
        def soma(chave: str) -> int:
            return sum(d.stats_parse.get(chave, 0) for d in self.documentos)

        n_exatas = soma("n_exatas")
        n_trim = soma("n_corrigidas_trim")
        n_shift = soma("n_corrigidas_shift")
        n_desc = soma("n_descartadas")
        n_trim_aplicado = soma("n_trim_aplicado")
        n_xml = soma("n_total_xml")
        raios = {d.stats_parse.get("raio_ajuste", 1) for d in self.documentos}
        raio = max(raios) if raios else 1
        n_preservadas = n_exatas + n_trim + n_shift
        pct = lambda v: f"{v / n_xml * 100:.2f}%" if n_xml else "—"

        docs_crlf = sum(
            1 for d in self.documentos if d.stats_parse.get("espaco_offsets") == "CRLF"
        )
        docs_com_desc = sum(
            1 for d in self.documentos if d.stats_parse.get("n_descartadas", 0) > 0
        )

        todas_desc: list[dict] = []
        todas_aj: list[dict] = []
        for d in self.documentos:
            todas_desc.extend(d.stats_parse.get("descartadas", []))
            todas_aj.extend(d.stats_parse.get("ajustadas", []))

        linhas_md = [
            "# Relatório de qualidade das anotações\n",
            "Resultado da passada única de ajuste de posição aplicada na leitura",
            "dos XMLs (`resolver_span`): cada entidade é validada contra o atributo",
            "`text` do XML e, quando necessário, reposicionada pela menor correção",
            "que a torna compatível.\n",
            "## Resumo\n",
            "| Métrica | Valor |",
            "|---|---|",
            f"| Total no XML original | {n_xml} |",
            f"| Exatas (offset já correto) | {n_exatas} ({pct(n_exatas)}) |",
            f"| Corrigidas só por trim de whitespace | {n_trim} ({pct(n_trim)}) |",
            f"| Corrigidas por deslocamento de borda (±{raio} caractere{'s' if raio > 1 else ''}) | {n_shift} ({pct(n_shift)}) |",
            f"| **Preservadas (exatas + trim + shift)** | **{n_preservadas}** ({pct(n_preservadas)}) |",
            f"| Descartadas (offset irrecuperável) | {n_desc} ({pct(n_desc)}) |",
            f"| Documentos com offsets no espaço CRLF | {docs_crlf} de {len(self.documentos)} |",
            f"| Documentos afetados por descarte | {docs_com_desc} |\n",
            f"O trim de whitespace nunca resgata uma entidade sozinho ({n_trim}"
            " casos): a comparação com o atributo `text` já ignora whitespace."
            f" Ele é aplicado em {n_trim_aplicado} spans, sempre depois do"
            " deslocamento, para encostar a borda no token (`\"IRC \"` →"
            " `\"IRC\"`).\n",
        ]

        if todas_aj:
            contagem: dict[tuple[int, int], int] = {}
            for a in todas_aj:
                if a["tipo"] == "shift":
                    d = tuple(a["delta"])
                    contagem[d] = contagem.get(d, 0) + 1
            if contagem:
                linhas_md.append("### Deslocamentos aplicados\n")
                linhas_md.append("| Δinício | Δfim | Entidades |")
                linhas_md.append("|---|---|---|")
                for (ds, de), n in sorted(contagem.items(), key=lambda kv: -kv[1]):
                    linhas_md.append(f"| {ds:+d} | {de:+d} | {n} |")
                linhas_md.append("")

            linhas_md.append("## Anotações ajustadas\n")
            linhas_md.append(
                "| doc_id | id | Ajuste | Atributo text | Span antes | Span depois | start | end |"
            )
            linhas_md.append("|---|---|---|---|---|---|---|---|")
            for a in sorted(todas_aj, key=lambda x: (x["doc_id"], x["id"])):
                ds, de = a["delta"]
                rotulo = a["tipo"] if a["tipo"] != "shift" else f"shift ({ds:+d},{de:+d})"
                linhas_md.append(
                    f"| {a['doc_id']} | {a['id']} | {rotulo} | {_esc_md(a['attr_text'])} "
                    f"| {_esc_md(a['span_antes'])} | {_esc_md(a['span_depois'])} "
                    f"| {a['start']} | {a['end']} |"
                )
            linhas_md.append("")

        if todas_desc:
            linhas_md.append("## Anotações descartadas\n")
            linhas_md.append("| doc_id | id | Atributo text | Span encontrado | start | end |")
            linhas_md.append("|---|---|---|---|---|---|")
            for d in sorted(todas_desc, key=lambda x: (x["doc_id"], x["id"])):
                linhas_md.append(
                    f"| {d['doc_id']} | {d['id']} | {_esc_md(d['attr_text'])} "
                    f"| {_esc_md(d['span'])} | {d['start']} | {d['end']} |"
                )

        caminho_md.write_text("\n".join(linhas_md) + "\n", encoding="utf-8")

    # -- diagnóstico --------------------------------------------------------

    def estatisticas(self) -> dict:
        """Resumo para conferência antes de treinar.

        Sem contagem por split: a divisão é do passo 03 (`divisao_Gold_Qwen7B.csv`).
        """
        n = len(self.documentos)
        ents = [len(d.entidades) for d in self.documentos]
        return {
            "n_documentos": n,
            "n_entidades": sum(ents),
            "n_relacoes": sum(len(d.relacoes) for d in self.documentos),
            "entidades_por_doc_min_mediana_max": (
                min(ents), sorted(ents)[len(ents) // 2], max(ents)
            ) if ents else None,
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
    corpus = CorpusSemClinBr(diretorio_xml)

    print("Derivando inventário de rótulos do corpus...")
    corpus.inventario_tags()

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

    # Relatório de qualidade das anotações
    caminho_md = arquivo_gabarito.with_suffix(".md")
    corpus.gerar_relatorio_qualidade_md(caminho_md)
    print(f"Relatório de qualidade: {caminho_md}")

    print("\nEstatísticas do Corpus:")
    pprint(corpus.estatisticas())

    print("\nArquivos gerados:")
    for tipo, caminho in arquivos_gerados.items():
        print(f"  {tipo}: {caminho}")