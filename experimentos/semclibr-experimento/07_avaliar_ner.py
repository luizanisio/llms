#!/usr/bin/env python3
"""Trilha NER-F1 do experimento SemClinBr — a variável primária do README §6.

O `comparar_extracoes.py` (passos 03 e 06) mede similaridade textual sobre o
JSON de saída. É o que ordena o currículo e alimenta gráficos e análise
bayesiana, mas não é o F1 de reconhecimento de entidades que o corpus exige e
que os sistemas publicados sobre ele reportam.

Este script roda sobre os MESMOS parquets de saída dos passos 02 e 05 e produz,
por documento de teste e por protocolo, a linha de métricas de
`util_semclinbr.avaliar_documento()`. A estatística (Friedman + Wilcoxon com
Holm + tamanho de efeito) sai de `util_analise_estatistica.AnaliseEstatistica`,
o mesmo módulo usado pelo `comparar_extracoes.py` — nenhum teste novo.

Executar com:
    python 07_avaliar_ner.py --config 07_avaliar_ner.yaml
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
    MODOS,
    alinhar_entidades,
    avaliar_documento,
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
    """Primeira `pastas_base` que existir; senão, a pasta do próprio script."""
    for p in config.get("misc", {}).get("pastas_base", []):
        if os.path.isdir(p):
            return Path(p)
    return _BASE


def caminho(pasta_base: Path, relativo: str) -> Path:
    p = Path(relativo)
    return p if p.is_absolute() else pasta_base / p


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


def carregar_divisao(pasta_base: Path, cfg_corpus: dict) -> pd.DataFrame:
    """Divisão do passo 03 (com `dificuldade`) ou a original, como fallback."""
    principal = caminho(pasta_base, cfg_corpus["divisao"])
    if principal.is_file():
        df = pd.read_csv(principal)
        print(f"📑 Divisão: {principal.name}")
    else:
        alternativa = caminho(pasta_base, cfg_corpus["divisao_fallback"])
        df = pd.read_csv(alternativa)
        print(f"⚠️  {principal.name} não encontrado — usando {alternativa.name} "
              f"(sem recorte por dificuldade; rode o passo 03 antes)")
    coluna_id = "id_arquivo" if "id_arquivo" in df.columns else "id"
    df = df.rename(columns={coluna_id: "id_arquivo"})
    df["id_arquivo"] = df["id_arquivo"].astype(str).str.strip()
    return df


def carregar_documentos(pasta_base: Path, cfg_corpus: dict,
                        ids: set[str]) -> dict:
    """Lê só os XMLs do split avaliado — o gabarito com offsets."""
    diretorio = caminho(pasta_base, cfg_corpus["xml"])
    docs = {}
    for arquivo in sorted(diretorio.glob("*.xml")):
        if arquivo.stem in ids:
            docs[arquivo.stem] = parse_semclinbr_xml(arquivo)
    faltando = ids - set(docs)
    if faltando:
        print(f"⚠️  {len(faltando)} documento(s) do split sem XML correspondente")
    return docs


def carregar_saida(pasta_base: Path, arquivo: str, campos: dict) -> dict | None:
    """Mapa {chave: resposta bruta} de um parquet de saída dos passos 02/05."""
    caminho_abs = caminho(pasta_base, arquivo)
    if not caminho_abs.is_file():
        return None
    df = pd.read_parquet(caminho_abs)
    col_id, col_resp = campos["id"], campos["resposta"]
    if col_id not in df.columns or col_resp not in df.columns:
        print(f"⚠️  {caminho_abs.name}: colunas '{col_id}'/'{col_resp}' ausentes")
        return None
    df[col_id] = df[col_id].astype(str).str.strip()
    return dict(zip(df[col_id], df[col_resp].fillna("")))


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------


def avaliar_protocolo(docs: dict, saidas: dict, alias: str) -> tuple[list, list]:
    """Uma linha de métricas por documento + o detalhamento por rótulo.

    `avaliar_documento` é a autoridade sobre a linha principal. O detalhamento
    por STY/SGR precisa das entidades alinhadas, que ela não devolve, então o
    alinhamento é refeito aqui — determinístico, mesmo resultado.
    """
    linhas, por_rotulo = [], []
    for doc_id, doc in docs.items():
        bruta = saidas.get(doc_id, "")
        linhas.append(avaliar_documento(doc, bruta, protocolo=alias))

        pred = carregar_predicao(bruta)
        if pred is None:
            continue
        ents = alinhar_entidades(doc.texto, pred.get("entities", []))
        for escopo, resultados in (
            ("STY", avaliar_por_sty(doc.entidades, ents, modo="strict")),
            ("SGR", avaliar_por_sgr(doc.entidades, ents, modo="flexible")),
        ):
            for rotulo, r in resultados.items():
                # só rótulos presentes no gabarito: F1 de um rótulo que o
                # documento não tem mede alucinação, não reconhecimento
                if r["n_gold"]:
                    por_rotulo.append({
                        "id_arquivo": doc_id, "protocolo": alias,
                        "escopo": escopo, "rotulo": rotulo,
                        "f1": r["f1"], "precisao": r["precisao"],
                        "revocacao": r["revocacao"], "n_gold": r["n_gold"],
                    })
    return linhas, por_rotulo


# ---------------------------------------------------------------------------
# Agregações (README §7.1 a §7.5)
# ---------------------------------------------------------------------------


def _mediana_iqr(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return ""
    q1, q3 = np.percentile(s, [25, 75])
    return f"{s.median():.3f} [{q1:.3f}–{q3:.3f}]"


def tabela_ancora(df: pd.DataFrame, metricas: list[str], piso: float,
                  ordem: list[str]) -> pd.DataFrame:
    """§7.1 — mediana [IQR] por protocolo. Medianas, não médias: as
    distribuições de F1 por documento são assimétricas e têm massa em zero."""
    linhas = []
    for alias in ordem:
        g = df[df["protocolo"] == alias]
        if g.empty:
            continue
        linha = {"protocolo": alias, "n_documentos": len(g)}
        for m in metricas:
            linha[m] = _mediana_iqr(g[m])
        linha["precisao_strict"] = _mediana_iqr(g["precisao_strict"])
        linha["revocacao_strict"] = _mediana_iqr(g["revocacao_strict"])
        linha["taxa_falha_parsing"] = f"{g['falha_parsing'].mean():.1%}"
        linha["taxa_nao_alinhamento"] = _mediana_iqr(g["taxa_nao_alinhamento"])
        # §7.5 — viabilidade com IC 95% de Wilson
        k, n = int((g["f1_strict"] >= piso).sum()), len(g)
        p = k / n
        z = 1.96
        den = 1 + z * z / n
        centro = (p + z * z / (2 * n)) / den
        margem = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        linha[f"viabilidade_f1_strict>={piso:.2f}"] = (
            f"{p:.1%} [{max(0, centro - margem):.1%}–{min(1, centro + margem):.1%}]"
        )
        linhas.append(linha)
    return pd.DataFrame(linhas)


def tabela_por_rotulo(df_rot: pd.DataFrame, ordem: list[str]) -> pd.DataFrame:
    """§7.2 — mediana de F1 por STY e por SGR, por protocolo."""
    if df_rot.empty:
        return pd.DataFrame()
    agg = (df_rot.groupby(["escopo", "rotulo", "protocolo"])
           .agg(f1_mediana=("f1", "median"),
                n_documentos=("f1", "size"),
                n_gold_total=("n_gold", "sum"))
           .reset_index())
    pivot = agg.pivot_table(index=["escopo", "rotulo", "n_gold_total"],
                            columns="protocolo", values="f1_mediana")
    colunas = [c for c in ordem if c in pivot.columns]
    return (pivot[colunas].reset_index()
            .sort_values(["escopo", "n_gold_total"], ascending=[True, False]))


def tabela_por_dificuldade(df: pd.DataFrame, divisao: pd.DataFrame,
                           ordem: list[str]) -> pd.DataFrame:
    """§7.3 — a tabela-âncora recortada por faixa do proxy S_i. É onde o CL
    deveria aparecer: ganho concentrado nas faixas difíceis."""
    if "dificuldade" not in divisao.columns:
        return pd.DataFrame()
    df = df.merge(divisao[["id_arquivo", "dificuldade"]], on="id_arquivo", how="left")
    linhas = []
    for faixa in ["facil", "medio", "dificil"]:
        sub = df[df["dificuldade"] == faixa]
        for alias in ordem:
            g = sub[sub["protocolo"] == alias]
            if g.empty:
                continue
            linhas.append({
                "dificuldade": faixa, "protocolo": alias, "n_documentos": len(g),
                "f1_strict": _mediana_iqr(g["f1_strict"]),
                "f1_relaxed": _mediana_iqr(g["f1_relaxed"]),
                "f1_relacoes": _mediana_iqr(g["f1_relacoes"]),
            })
    return pd.DataFrame(linhas)


def tabela_decomposicao(df: pd.DataFrame, ordem: list[str]) -> pd.DataFrame:
    """§7.4 — onde o erro se concentra: fronteira de span vs. escolha de rótulo."""
    linhas = []
    for alias in ordem:
        g = df[df["protocolo"] == alias]
        if g.empty:
            continue
        linhas.append({
            "protocolo": alias,
            "lenient_menos_strict": (g["f1_lenient"] - g["f1_strict"]).median(),
            "flexible_menos_strict": (g["f1_flexible"] - g["f1_strict"]).median(),
            "span_exato_menos_strict": (g["f1_span_exato"] - g["f1_strict"]).median(),
            "relaxed_menos_strict": (g["f1_relaxed"] - g["f1_strict"]).median(),
        })
    return pd.DataFrame(linhas).round(4)


def rodar_estatisticas(df: pd.DataFrame, metricas: list[str], pasta: Path,
                       lang: str) -> None:
    """Friedman + Wilcoxon com Holm + tamanho de efeito, uma análise por métrica."""
    from util_analise_estatistica import AnaliseEstatistica

    pasta.mkdir(parents=True, exist_ok=True)
    for metrica in metricas:
        largo = df.pivot_table(index="id_arquivo", columns="protocolo",
                               values=metrica)
        if largo.shape[1] < 2:
            print(f"   ⏭️  {metrica}: menos de 2 protocolos, sem contraste")
            continue
        analise = AnaliseEstatistica(largo, {
            "metrica_nome": metrica,
            "campo": "NER",
            "tecnica": metrica.replace("f1_", "F1 ").title(),
            "arquivo_md": str(pasta / f"estatistica_{metrica}.md"),
            "arquivo_cd_png": str(pasta / f"cd_{metrica}.png"),
            "lang": lang,
        })
        resumo = analise.processar()
        analise.salvar()
        p = resumo.get("friedman_p")
        print(f"   📊 {metrica}: K={resumo.get('K')} N={resumo.get('N')}"
              + (f" Friedman p={p:.2e}" if p is not None else ""))


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="07_avaliar_ner.yaml")
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
            print(f"🗺️  SemGroups do UMLS: {arquivo_sg.name}")

    divisao = carregar_divisao(pasta_base, cfg_corpus)
    ids_split = set(divisao.loc[divisao["alvo"] == cfg_corpus["split"], "id_arquivo"])
    docs = carregar_documentos(pasta_base, cfg_corpus, ids_split)
    print(f"📄 {len(docs)} documentos do split '{cfg_corpus['split']}'")

    campos = cfg_exec["campos_parquet"]
    linhas, por_rotulo, ordem = [], [], []
    for proto in config["protocolos"]:
        alias = proto["alias"]
        saidas = carregar_saida(pasta_base, proto["arquivo"], campos)
        if saidas is None:
            print(f"   ⏭️  {alias}: {proto['arquivo']} ainda não existe")
            continue
        l, r = avaliar_protocolo(docs, saidas, alias)
        linhas.extend(l)
        por_rotulo.extend(r)
        ordem.append(alias)
        f1 = np.median([x["f1_strict"] for x in l]) if l else float("nan")
        falhas = sum(x["falha_parsing"] for x in l)
        print(f"   ✅ {alias}: {len(l)} docs, f1_strict mediano={f1:.3f}, "
              f"falhas de parsing={falhas}")

    if not linhas:
        sys.exit("❌ Nenhum protocolo avaliado — rode os passos 02 e 05 antes.")

    df = pd.DataFrame(linhas)
    df_rot = pd.DataFrame(por_rotulo)

    pasta_saida = caminho(pasta_base, config["saida"]["pasta"])
    pasta_saida.mkdir(parents=True, exist_ok=True)

    metricas = [cfg_exec["metrica_primaria"]] + cfg_exec["metricas_complementares"]
    df.to_parquet(pasta_saida / "metricas_ner.parquet", index=False)
    tabela_ancora(df, metricas, cfg_exec["piso_viabilidade"], ordem).to_csv(
        pasta_saida / "tabela_ancora.csv", index=False)
    tabela_por_rotulo(df_rot, ordem).to_csv(
        pasta_saida / "por_rotulo.csv", index=False)
    tabela_por_dificuldade(df, divisao, ordem).to_csv(
        pasta_saida / "por_dificuldade.csv", index=False)
    tabela_decomposicao(df, ordem).to_csv(
        pasta_saida / "decomposicao_erro.csv", index=False)

    print(f"\n📊 Análise estatística (primária: {cfg_exec['metrica_primaria']})")
    rodar_estatisticas(df, metricas, pasta_saida / "estatisticas",
                       cfg_exec.get("lang", "en"))

    print(f"\n🏁 Saída em: {pasta_saida}")


if __name__ == "__main__":
    main()
