#!/usr/bin/env python3
"""Baseline externo — ClinicalNERpt (BioBERTpt, HAILab-PUCPR).

Roda os modelos `pucpr/clinicalnerpt-*` do HuggingFace sobre o split de
teste do SemClinBr e compara o F1 por STY coberto com os nossos protocolos.

Cada clinicalnerpt-* é um sequence labeler (BertForTokenClassification)
especializado em UM tipo de entidade.  Este script:
1. Carrega os docs do split de teste via `parse_semclinbr_xml`
2. Roda cada modelo e extrai entidades via token-classification pipeline
3. Converte spans IOB2 para Entidade com tags do SemClinBr
4. Calcula F1 por STY usando `avaliar_por_sty`
5. Carrega as predições dos nossos protocolos e calcula o mesmo F1 filtrado
6. Gera `comparacao_baseline.md` agrupado por modelo baseline

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
    avaliar,
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
# Carga do corpus
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
# Inference com clinicalnerpt
# ---------------------------------------------------------------------------


def rodar_modelo_ner(
    modelo_hf: str,
    docs: dict,
    stys: list[str],
    batch_size: int = 8,
    max_length: int = 512,
) -> dict[str, list[Entidade]]:
    """Roda um modelo clinicalnerpt-* sobre os documentos.

    Retorna {doc_id: [Entidade, ...]} com as tags mapeadas para os STYs.
    """
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    ner_pipe = pipeline(
        "token-classification",
        model=modelo_hf,
        aggregation_strategy="simple",
        batch_size=batch_size,
        device=device,
    )

    # Tag padrão: usa o primeiro STY para entidades detectadas
    tag_padrao = stys[0] if stys else "Unknown"
    resultados = {}

    textos = []
    doc_ids = []
    for doc_id, doc in docs.items():
        # Truncar para max_length tokens (BERT limit)
        texto = doc.texto[:max_length * 4]  # heurística conservadora
        textos.append(texto)
        doc_ids.append(doc_id)

    print(f"   🔄 Rodando {modelo_hf} em {len(textos)} docs...")
    saidas = ner_pipe(textos)

    for doc_id, preds in zip(doc_ids, saidas):
        ents = []
        for i, pred in enumerate(preds):
            ents.append(Entidade(
                id=i + 1,
                text=pred["word"],
                tags=[tag_padrao],
                start=pred["start"],
                end=pred["end"],
                alinhada=True,
            ))
        resultados[doc_id] = ents

    return resultados


# ---------------------------------------------------------------------------
# Carga das predições dos nossos protocolos
# ---------------------------------------------------------------------------


def carregar_nossos_protocolos(
    pasta_base: Path, protocolos: list[dict], campos: dict, docs: dict
) -> dict[str, dict[str, list[Entidade]]]:
    """Carrega e alinha as predições dos nossos protocolos.

    Retorna {alias: {doc_id: [Entidade, ...]}}
    """
    resultado = {}
    for proto in protocolos:
        alias = proto["alias"]
        arquivo = caminho(pasta_base, proto["arquivo"])
        if not arquivo.is_file():
            print(f"   ⏭️  {alias}: {arquivo} não existe")
            continue

        df = pd.read_parquet(arquivo)
        col_id, col_resp = campos["id"], campos["resposta"]
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
        print(f"   ✅ {alias}: {len(ents_por_doc)} docs")

    return resultado


# ---------------------------------------------------------------------------
# Comparação e geração do relatório
# ---------------------------------------------------------------------------


def calcular_f1_por_sty_filtrado(
    docs: dict,
    ents_pred: dict[str, list[Entidade]],
    stys_filtro: set[str],
    modo: str = "strict",
) -> dict[str, dict]:
    """Calcula F1 por STY, filtrando apenas os STYs do filtro."""
    f1_por_sty: dict[str, list[float]] = {}
    n_gold_por_sty: dict[str, int] = {}

    for doc_id, doc in docs.items():
        pred = ents_pred.get(doc_id, [])
        gold_filtrado = [e for e in doc.entidades if set(e.tags) & stys_filtro]
        pred_filtrado = [e for e in pred if set(e.tags) & stys_filtro]

        if not gold_filtrado:
            continue

        resultados = avaliar_por_sty(gold_filtrado, pred_filtrado, modo=modo)
        for sty, r in resultados.items():
            if sty in stys_filtro:
                if sty not in f1_por_sty:
                    f1_por_sty[sty] = []
                    n_gold_por_sty[sty] = 0
                f1_por_sty[sty].append(r["f1"])
                n_gold_por_sty[sty] += r["n_gold"]

    resumo = {}
    for sty in sorted(f1_por_sty):
        valores = f1_por_sty[sty]
        resumo[sty] = {
            "f1_mediano": float(np.median(valores)) if valores else 0.0,
            "f1_medio": float(np.mean(valores)) if valores else 0.0,
            "n_gold": n_gold_por_sty.get(sty, 0),
            "n_docs": len(valores),
        }
    return resumo


def gerar_relatorio_md(
    modelos_resultados: list[dict],
    nossos_resultados: dict[str, dict[str, dict]],
    caminho_md: Path,
) -> None:
    """Gera o relatório .md agrupado por modelo baseline."""
    linhas = [
        "# Comparação com baseline externo — ClinicalNERpt\n",
        "Modelos do [HAILab-PUCPR](https://huggingface.co/pucpr) (BioBERTpt),",
        "cada um especializado em um tipo de entidade, avaliados sobre o",
        "split de teste do SemClinBr.\n",
        "> **Nota:** Cada modelo `clinicalnerpt-*` cobre apenas 1 tipo de",
        "> entidade. A comparação filtra apenas os STYs que o modelo baseline",
        "> possui — garantindo uma comparação justa.\n",
    ]

    for mr in modelos_resultados:
        nome_curto = mr["nome"].split("/")[-1]
        linhas.append(f"## {nome_curto}\n")
        linhas.append(f"**Modelo:** `{mr['nome']}`\n")
        linhas.append(f"**STYs cobertos:** {', '.join(mr['stys'])}\n")

        # Cabeçalho da tabela
        aliases_nossos = sorted(nossos_resultados.keys())
        cols = ["STY", "n_gold", nome_curto] + aliases_nossos
        linhas.append("| " + " | ".join(cols) + " |")
        linhas.append("| " + " | ".join(["---"] * len(cols)) + " |")

        stys_set = set(mr["stys"])
        baseline_resumo = mr["resumo"]

        for sty in sorted(stys_set):
            bl = baseline_resumo.get(sty, {})
            f1_bl = f"{bl.get('f1_mediano', 0):.3f}" if bl else "—"
            n_gold = str(bl.get("n_gold", 0)) if bl else "—"

            valores = [sty, n_gold, f1_bl]
            for alias in aliases_nossos:
                nosso = nossos_resultados.get(alias, {}).get(sty, {})
                f1_nosso = f"{nosso.get('f1_mediano', 0):.3f}" if nosso else "—"
                valores.append(f1_nosso)

            linhas.append("| " + " | ".join(valores) + " |")

        linhas.append("")

    caminho_md.write_text("\n".join(linhas) + "\n", encoding="utf-8")


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

    # 1. Carregar docs de teste
    docs = carregar_docs_teste(pasta_base, cfg_corpus)

    # 2. Rodar cada modelo baseline
    modelos_resultados = []
    for cfg_modelo in config["modelos_baseline"]:
        nome = cfg_modelo["nome"]
        stys = cfg_modelo["stys"]
        stys_set = set(stys)

        try:
            ents_baseline = rodar_modelo_ner(
                nome, docs, stys,
                batch_size=cfg_exec.get("batch_size", 8),
                max_length=cfg_exec.get("max_length", 512),
            )

            # Calcular F1 por STY para o baseline
            resumo = calcular_f1_por_sty_filtrado(
                docs, ents_baseline, stys_set, modo="strict"
            )

            n_ents = sum(len(e) for e in ents_baseline.values())
            print(f"   📊 {nome}: {n_ents} entidades extraídas")
            for sty, r in resumo.items():
                print(f"      {sty}: F1 mediano={r['f1_mediano']:.3f} "
                      f"({r['n_docs']} docs, {r['n_gold']} gold)")

            modelos_resultados.append({
                "nome": nome,
                "stys": stys,
                "resumo": resumo,
            })

        except Exception as e:
            print(f"   ⚠️  {nome}: erro — {e}")
            modelos_resultados.append({
                "nome": nome,
                "stys": stys,
                "resumo": {},
            })

    # 3. Carregar nossos protocolos
    nossos_protocolos = carregar_nossos_protocolos(
        pasta_base, config["protocolos"],
        cfg_exec["campos_parquet"], docs,
    )

    # 4. Calcular F1 dos nossos protocolos filtrado por STYs de cada baseline
    nossos_resultados: dict[str, dict[str, dict]] = {}
    for alias, ents_por_doc in nossos_protocolos.items():
        todos_stys = set()
        for mr in modelos_resultados:
            todos_stys.update(mr["stys"])

        resumo_nosso = calcular_f1_por_sty_filtrado(
            docs, ents_por_doc, todos_stys, modo="strict"
        )
        nossos_resultados[alias] = resumo_nosso

    # 5. Gerar relatório
    pasta_saida = caminho(pasta_base, config["saida"]["pasta"])
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_md = pasta_saida / config["saida"]["arquivo_comparacao"]

    gerar_relatorio_md(modelos_resultados, nossos_resultados, caminho_md)
    print(f"\n🏁 Relatório: {caminho_md}")


if __name__ == "__main__":
    main()
