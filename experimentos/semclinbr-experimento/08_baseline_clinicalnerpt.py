#!/usr/bin/env python3
"""Baseline externo — ClinicalNERpt (BioBERTpt, HAILab-PUCPR).

Roda os modelos `pucpr/clinicalnerpt-*` do HuggingFace sobre o split de
teste do SemClinBr e compara o desempenho contra todos os nossos protocolos
em uma tabela mestra invertida (linhas = protocolos, colunas = categorias).

Executar com:
    python 08_baseline_clinicalnerpt.py --config 08_baseline_clinicalnerpt.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_BASE.parent.parent / "src"))

from util_semclinbr import (  # noqa: E402
    Entidade,
    alinhar_entidades,
    avaliar_por_sgr,
    avaliar_por_sty,
    carregar_predicao,
    carregar_semgroups,
    parse_semclinbr_xml,
)


# ---------------------------------------------------------------------------
# Configuração e caminhos
# ---------------------------------------------------------------------------


def resolver_pasta_base(config: dict) -> Path:
    for p in config.get("misc", {}).get("pastas_base", []):
        if os.path.isdir(p):
            return Path(p)
    return _BASE


def caminho(pasta_base: Path, relativo: str) -> Path:
    p = Path(relativo)
    return p if p.is_absolute() else pasta_base / p


# ---------------------------------------------------------------------------
# Carga do corpus de teste
# ---------------------------------------------------------------------------


def carregar_docs_teste(pasta_base: Path, cfg_corpus: dict) -> dict:
    """Carrega XMLs do split de teste."""
    divisao = pd.read_csv(caminho(pasta_base, cfg_corpus["divisao"]))
    col_id = "id_arquivo" if "id_arquivo" in divisao.columns else "id"
    divisao = divisao.rename(columns={col_id: "id_arquivo"})
    divisao["id_arquivo"] = divisao["id_arquivo"].astype(str).str.strip()
    ids_teste = set(
        divisao.loc[divisao["alvo"] == cfg_corpus["split"], "id_arquivo"]
    )

    diretorio = caminho(pasta_base, cfg_corpus["xml"])
    docs = {}
    for arq in sorted(diretorio.glob("*.xml")):
        if arq.stem in ids_teste:
            docs[arq.stem] = parse_semclinbr_xml(arq)
    print(f"📄 {len(docs)} documentos do split '{cfg_corpus['split']}'")
    return docs


# ---------------------------------------------------------------------------
# Fusão de subwords e Inferência com ClinicalNERpt
# ---------------------------------------------------------------------------


def merge_contiguous_tokens(preds: list[dict], texto: str) -> list[dict]:
    """Funde subwords (WordPiece ##) e fragmentos contíguos do mesmo tipo.

    Modelos treinados com WordPiece rotulam subwords individuais com B- ou I-.
    Esta função agrega os fragmentos adjacentes no span completo da palavra/termo.
    """
    if not preds:
        return []
    merged = []
    curr = None
    for p in preds:
        s, e = p["start"], p["end"]
        if curr is not None and s <= curr["end"] + 1:
            curr["end"] = max(curr["end"], e)
            curr["score"] = max(curr.get("score", 0), p.get("score", 0))
        else:
            if curr is not None:
                curr["word"] = texto[curr["start"] : curr["end"]]
                merged.append(curr)
            curr = {"start": s, "end": e, "score": p.get("score", 0)}
    if curr is not None:
        curr["word"] = texto[curr["start"] : curr["end"]]
        merged.append(curr)
    return merged


def rodar_modelo_ner(
    modelo_hf: str,
    target_tag: str,
    docs: dict,
    batch_size: int = 8,
    max_length: int = 512,
) -> dict[str, list[Entidade]]:
    """Roda um modelo clinicalnerpt-* do HuggingFace sobre os documentos."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    ner_pipe = pipeline(
        "token-classification",
        model=modelo_hf,
        batch_size=batch_size,
        device=device,
    )

    textos = []
    doc_ids = []
    for doc_id, doc in docs.items():
        # Truncar para limite BERT de contexto
        texto = doc.texto[: max_length * 4]
        textos.append(texto)
        doc_ids.append(doc_id)

    print(f"   🔄 Rodando {modelo_hf} em {len(textos)} docs...")
    saidas_raw = ner_pipe(textos)

    resultados = {}
    total_ents = 0
    for doc_id, raw_preds in zip(doc_ids, saidas_raw):
        doc_texto = docs[doc_id].texto
        merged = merge_contiguous_tokens(raw_preds, doc_texto)
        ents = [
            Entidade(
                id=i + 1,
                text=m["word"],
                tags=[target_tag],
                start=m["start"],
                end=m["end"],
                alinhada=True,
            )
            for i, m in enumerate(merged)
        ]
        resultados[doc_id] = ents
        total_ents += len(ents)

    print(f"   📊 {modelo_hf}: {total_ents} entidades extraídas (com fusão de subwords)")
    return resultados


# ---------------------------------------------------------------------------
# Carga das predições dos nossos protocolos
# ---------------------------------------------------------------------------


def carregar_nossos_protocolos(
    pasta_base: Path, protocolos: list[dict], campos: dict, docs: dict
) -> dict[str, dict[str, list[Entidade]]]:
    """Carrega e alinha as predições dos nossos protocolos (A, B, C, D*...)."""
    resultado = {}
    col_id, col_resp = campos["id"], campos["resposta"]

    for proto in protocolos:
        alias = proto["alias"]
        arquivo = caminho(pasta_base, proto["arquivo"])
        if not arquivo.is_file():
            print(f"   ⏭️  {alias}: {arquivo.name} ainda não existe")
            continue

        df = pd.read_parquet(arquivo)
        df[col_id] = df[col_id].astype(str).str.strip()
        saidas_map = dict(zip(df[col_id], df[col_resp].fillna("")))

        ents_por_doc = {}
        for doc_id, doc in docs.items():
            bruta = saidas_map.get(doc_id, "")
            pred = carregar_predicao(bruta)
            if pred is None:
                ents_por_doc[doc_id] = []
                continue
            ents = alinhar_entidades(doc.texto, pred.get("entities", []))
            ents_por_doc[doc_id] = ents

        resultado[alias] = ents_por_doc
        print(f"   ✅ Protocolo {alias:6s}: {len(ents_por_doc)} docs alinhados")

    return resultado


# ---------------------------------------------------------------------------
# Avaliação de um alvo específico (STY ou SGR)
# ---------------------------------------------------------------------------


def avaliar_alvo(
    docs: dict,
    ents_por_doc: dict[str, list[Entidade]],
    escopo: str,
    alvo: str,
) -> dict:
    """Calcula Micro F1 (P/R) e Mediana por documento para um alvo (STY ou SGR)."""
    f1_docs = []
    tot_tp = 0.0
    tot_gold = 0
    tot_pred = 0

    for doc_id, doc in docs.items():
        ents = ents_por_doc.get(doc_id, [])
        if escopo == "SGR":
            res = avaliar_por_sgr(doc.entidades, ents, modo="flexible").get(alvo)
        else:
            res = avaliar_por_sty(doc.entidades, ents, modo="strict").get(alvo)

        if res and res["n_gold"] > 0:
            f1_docs.append(res["f1"])
            tot_tp += res["acertos"]
            tot_gold += res["n_gold"]
            tot_pred += res["n_pred"]

    p_mic = tot_tp / tot_pred if tot_pred else 0.0
    r_mic = tot_tp / tot_gold if tot_gold else 0.0
    f1_mic = 2 * p_mic * r_mic / (p_mic + r_mic) if (p_mic + r_mic) else 0.0
    f1_med = float(np.median(f1_docs)) if f1_docs else 0.0

    return {
        "micro_f1": f1_mic,
        "precisao_micro": p_mic,
        "revocacao_micro": r_mic,
        "mediana_f1": f1_med,
        "n_gold": tot_gold,
        "n_pred": tot_pred,
        "n_docs": len(f1_docs),
    }


# ---------------------------------------------------------------------------
# Geração de Relatórios (Markdown, XLSX e CSV)
# ---------------------------------------------------------------------------


def gerar_relatorios(
    linhas_tabela: list[dict],
    linhas_num_micro: list[dict],
    linhas_num_med: list[dict],
    colunas_ordem: list[str],
    totais_gold: dict[str, int],
    caminho_md: Path,
    caminho_xlsx: Path,
    caminho_csv: Path,
) -> None:
    """Gera o relatório em Markdown, planilha Excel (.xlsx) e CSV."""
    df_formatado = pd.DataFrame(linhas_tabela)
    df_micro = pd.DataFrame(linhas_num_micro)
    df_med = pd.DataFrame(linhas_num_med)

    # 1. Markdown
    linhas_md = [
        "# Comparação com Baseline Externo — ClinicalNERpt (BioBERTpt)\n",
        "Tabela mestra consolidada invertida: **linhas representam os protocolos** e ",
        "**colunas representam as categorias clínicas** cobertas pelos modelos especializados ",
        "do [HAILab-PUCPR](https://huggingface.co/pucpr) (BioBERTpt).\n",
        "> **Convenção das células:** `Micro F1 (Mediana F1)`.",
        "> - **Micro F1**: Métrica clássica da literatura de NER (soma de TP, FP, FN no split de teste inteiro).",
        "> - **Mediana F1**: Mediana do F1 por documento clínico (métrica primária da análise pareada).\n",
    ]

    # Linha de total de anotações gold por coluna
    linha_gold_header = "| **Anotações Gold** | " + " | ".join(
        str(totais_gold.get(c, "—")) for c in colunas_ordem if c != "Protocolo"
    ) + " |"

    cols = ["Protocolo"] + [c for c in colunas_ordem if c != "Protocolo"]
    linhas_md.append("| " + " | ".join(cols) + " |")
    linhas_md.append("| " + " | ".join(["---"] * len(cols)) + " |")
    linhas_md.append(linha_gold_header)

    for linha in linhas_tabela:
        valores = [str(linha.get(c, "—")) for c in cols]
        linhas_md.append("| " + " | ".join(valores) + " |")

    linhas_md.append("\n## Legenda das Categorias:\n")
    linhas_md.append("- **Farmaco (STY)**: `Pharmacologic Substance` — substâncias farmacológicas específicas (ex.: propofol, fentanil).")
    linhas_md.append("- **Quimicos (SGR)**: `Chemicals & Drugs` — grupo semântico completo do UMLS (fármacos, enzimas, hormônios).")
    linhas_md.append("- **Doenca (STY)**: `Disease or Syndrome` — patologias e síndromes formais.")
    linhas_md.append("- **Desordens (SGR)**: `Disorders` — grupo semântico completo de desordens (sintomas, achados, lesões).")
    linhas_md.append("- **Diag. (STY)**: `Diagnostic Procedure` — procedimentos diagnósticos (ex.: exames, biópsias, tomografias).")
    linhas_md.append("- **Dispositivo (STY)**: `Medical Device` — dispositivos e equipamentos médicos (ex.: cateter, dreno, prótese).")
    linhas_md.append("- **Media Macro**: Média aritmética simples do desempenho nas 6 categorias avaliadas.")

    caminho_md.write_text("\n".join(linhas_md) + "\n", encoding="utf-8")

    # 2. CSV
    df_formatado.to_csv(caminho_csv, index=False)

    # 3. Excel (.xlsx) com múltiplas abas: Formatada, Micro F1 Puro e Mediana Pura
    with pd.ExcelWriter(caminho_xlsx, engine="openpyxl") as writer:
        df_formatado.to_excel(writer, sheet_name="Comparativo Completo", index=False)
        df_micro.to_excel(writer, sheet_name="Micro F1 (Puro)", index=False)
        df_med.to_excel(writer, sheet_name="Mediana F1 (Puro)", index=False)

    print(f"   📄 Markdown salvo em: {caminho_md}")
    print(f"   📊 Planilha XLSX salva em: {caminho_xlsx}")
    print(f"   📑 Planilha CSV salva em: {caminho_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="08_baseline_clinicalnerpt.yaml")
    args = ap.parse_args()

    caminho_config = Path(args.config)
    if not caminho_config.is_absolute():
        caminho_config = _BASE / caminho_config
    config = yaml.safe_load(caminho_config.read_text(encoding="utf-8"))

    pasta_base = resolver_pasta_base(config)
    cfg_corpus = config["corpus"]
    cfg_exec = config["execucao"]
    print(f"📂 Pasta base: {pasta_base}")

    if cfg_corpus.get("semgroups"):
        arquivo_sg = caminho(pasta_base, cfg_corpus["semgroups"])
        if arquivo_sg.is_file():
            carregar_semgroups(arquivo_sg)
            print(f"🗺️  SemGroups: {arquivo_sg.name}")

    # 1. Carregar documentos do split de teste
    docs = carregar_docs_teste(pasta_base, cfg_corpus)

    # 2. Rodar modelos baseline do HuggingFace
    predicoes_baseline = {}
    for cfg_m in config["modelos_baseline"]:
        nome = cfg_m["nome"]
        alvo = cfg_m["alvo"]
        try:
            ents_b = rodar_modelo_ner(
                nome,
                alvo,
                docs,
                batch_size=cfg_exec.get("batch_size", 8),
                max_length=cfg_exec.get("max_length", 512),
            )
            predicoes_baseline[nome] = ents_b
        except Exception as e:
            print(f"   ⚠️  Erro em {nome}: {e}")
            predicoes_baseline[nome] = {cid: [] for cid in docs}

    # 3. Carregar predições de todos os nossos protocolos
    nossos_protocolos = carregar_nossos_protocolos(
        pasta_base, config["protocolos"], cfg_exec["campos_parquet"], docs
    )

    # 4. Montar a tabela mestre invertida
    # Lista de alvos a avaliar
    alvos_config = config["modelos_baseline"]
    nomes_colunas = [m["coluna"] for m in alvos_config] + ["Media Macro"]

    # Calcular contagem gold por alvo
    totais_gold = {}
    for cfg_m in alvos_config:
        col = cfg_m["coluna"]
        esc = cfg_m["escopo"]
        alv = cfg_m["alvo"]
        # Gold é obtido avaliando os próprios docs
        gold_eval = avaliar_alvo(docs, {cid: docs[cid].entidades for cid in docs}, esc, alv)
        totais_gold[col] = gold_eval["n_gold"]

    linhas_tabela = []
    linhas_num_micro = []
    linhas_num_med = []

    # A) Linha do ClinicalNERpt (BioBERTpt)
    linha_bl = {"Protocolo": "ClinicalNERpt (BioBERTpt)"}
    linha_bl_micro = {"Protocolo": "ClinicalNERpt (BioBERTpt)"}
    linha_bl_med = {"Protocolo": "ClinicalNERpt (BioBERTpt)"}

    micros_bl = []
    meds_bl = []
    for cfg_m in alvos_config:
        nome_mod = cfg_m["nome"]
        col = cfg_m["coluna"]
        esc = cfg_m["escopo"]
        alv = cfg_m["alvo"]
        ents_m = predicoes_baseline.get(nome_mod, {})
        res = avaliar_alvo(docs, ents_m, esc, alv)

        micros_bl.append(res["micro_f1"])
        meds_bl.append(res["mediana_f1"])
        linha_bl[col] = f"{res['micro_f1']:.3f} ({res['mediana_f1']:.3f})"
        linha_bl_micro[col] = round(res["micro_f1"], 4)
        linha_bl_med[col] = round(res["mediana_f1"], 4)

    media_mic_bl = float(np.mean(micros_bl)) if micros_bl else 0.0
    media_med_bl = float(np.mean(meds_bl)) if meds_bl else 0.0
    linha_bl["Media Macro"] = f"{media_mic_bl:.3f} ({media_med_bl:.3f})"
    linha_bl_micro["Media Macro"] = round(media_mic_bl, 4)
    linha_bl_med["Media Macro"] = round(media_med_bl, 4)

    linhas_tabela.append(linha_bl)
    linhas_num_micro.append(linha_bl_micro)
    linhas_num_med.append(linha_bl_med)

    # B) Linhas de cada protocolo nosso (A, B, C, D1...D25, Gold)
    for alias, ents_p in nossos_protocolos.items():
        linha_p = {"Protocolo": alias}
        linha_p_micro = {"Protocolo": alias}
        linha_p_med = {"Protocolo": alias}

        micros_p = []
        meds_p = []
        for cfg_m in alvos_config:
            col = cfg_m["coluna"]
            esc = cfg_m["escopo"]
            alv = cfg_m["alvo"]
            res = avaliar_alvo(docs, ents_p, esc, alv)

            micros_p.append(res["micro_f1"])
            meds_p.append(res["mediana_f1"])
            linha_p[col] = f"{res['micro_f1']:.3f} ({res['mediana_f1']:.3f})"
            linha_p_micro[col] = round(res["micro_f1"], 4)
            linha_p_med[col] = round(res["mediana_f1"], 4)

        media_mic_p = float(np.mean(micros_p)) if micros_p else 0.0
        media_med_p = float(np.mean(meds_p)) if meds_p else 0.0
        linha_p["Media Macro"] = f"{media_mic_p:.3f} ({media_med_p:.3f})"
        linha_p_micro["Media Macro"] = round(media_mic_p, 4)
        linha_p_med["Media Macro"] = round(media_med_p, 4)

        linhas_tabela.append(linha_p)
        linhas_num_micro.append(linha_p_micro)
        linhas_num_med.append(linha_p_med)

    # 5. Salvar arquivos
    pasta_saida = caminho(pasta_base, config["saida"]["pasta"])
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_md = pasta_saida / config["saida"]["arquivo_comparacao"]
    caminho_xlsx = pasta_saida / config["saida"].get("arquivo_xlsx", "comparacao_baseline.xlsx")
    caminho_csv = pasta_saida / config["saida"].get("arquivo_csv", "comparacao_baseline.csv")

    print("\n🏁 Gerando relatórios consolidados...")
    gerar_relatorios(
        linhas_tabela,
        linhas_num_micro,
        linhas_num_med,
        nomes_colunas,
        totais_gold,
        caminho_md,
        caminho_xlsx,
        caminho_csv,
    )


if __name__ == "__main__":
    main()
