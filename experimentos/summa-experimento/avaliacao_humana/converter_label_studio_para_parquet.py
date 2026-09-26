#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
converter_label_studio_para_parquet.py
======================================
Converte os arquivos de exportação do Label Studio (JSON) dos avaliadores humanos
para o formato Parquet esperado pelo pipeline de análise e validação (`realizar_avaliacoes.py`).

Para cada avaliador, gera:
  - `<destino>/saida_av_XX/saida_juiz_llm.parquet` (estrutura padrão de pastas)
  - `<destino>/saida_av_XX.parquet` (cópia direta opcional na raiz)

Esquema de saída gerado:
  - chave:     "<doc_id>_<fonte>" (ex: "214629796_gpt5")
  - resposta:  JSON string '{"nota": 1..4, "problemas": [...], "justificativa": "..."}'
  - resumo:    JSON string '{"tempo": <lead_time_segundos>, "avaliador": "<nome_ou_id>", "task_id": "..."}'
  - erro:      "" (ou motivo se houver falha de anotação)
  - rodada:    índice do avaliador (1, 2, 3...)

Uso:
  # Processar um ou mais arquivos de exportação:
  python converter_label_studio_para_parquet.py avaliador1.json avaliador2.json avaliador3.json
  
  # Especificar pasta de saída (padrão: ../avaliacao_llm_humana):
  python converter_label_studio_para_parquet.py --saida ../avaliacao_llm_humana avaliador1.json avaliador2.json avaliador3.json

  # Validar arquivo de teste/treino:
  python converter_label_studio_para_parquet.py tarefas_treino_preenchido.json --prefixo saida_teste --valida-apenas

  python converter_label_studio_para_parquet.py --saida ../avaliacao_llm_humana tarefas_treino_preenchido.json --prefixo saida_teste
"""

import os
import sys
import json
import re
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional

DE_PARA_FONTES = {
    "FA": "qwen7b",
    "FB": "qwen235b",
    "FC": "gpt5",
}

def normalizar_fonte(fonte: Any) -> str:
    """Aplica o de-para dos modelos (FA, FB, FC -> qwen7b, qwen235b, gpt5)."""
    nome = str(fonte).strip() if fonte is not None else ""
    return DE_PARA_FONTES.get(nome, DE_PARA_FONTES.get(nome.upper(), nome))

def parse_nota(valor: Any) -> Optional[int]:
    """Extrai nota inteira (1 a 4) a partir de string, alias ou lista de choices do Label Studio."""
    if valor is None:
        return None
    if isinstance(valor, list) and len(valor) > 0:
        valor = valor[0]
    
    texto = str(valor).strip()
    match = re.search(r"^[1-4]", texto)
    if match:
        return int(match.group(0))
    return None

def processar_export_label_studio(
    caminho_json: str,
    id_avaliador: int = 1,
    nome_avaliador: Optional[str] = None
) -> pd.DataFrame:
    """Lê exportação do Label Studio e transforma em registros planos para o parquet."""
    if not os.path.exists(caminho_json):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_json}")

    with open(caminho_json, "r", encoding="utf-8") as f:
        tarefas = json.load(f)

    if not isinstance(tarefas, list):
        raise ValueError(f"O arquivo {caminho_json} não contém uma lista JSON de tarefas.")

    registros = []
    total_tarefas = len(tarefas)
    tarefas_com_anotacao = 0
    anotacoes_canceladas = 0

    for idx_t, tarefa in enumerate(tarefas, start=1):
        data = tarefa.get("data", {})
        doc_id = str(data.get("doc_id", "")).strip()
        task_id = data.get("task_id", f"item_{idx_t:04d}")

        if not doc_id:
            # Fallback para task id do label studio se não houver doc_id
            doc_id = str(tarefa.get("id", idx_t))

        annotations = tarefa.get("annotations", [])
        
        # Filtra anotações não canceladas
        valid_annotations = [a for a in annotations if not a.get("was_cancelled", False)]
        if not valid_annotations:
            if any(a.get("was_cancelled", False) for a in annotations):
                anotacoes_canceladas += 1
            # Registra como falha para cada uma das 3 colunas
            for col_idx in (1, 2, 3):
                fonte_bruta = str(data.get(f"fonte_real_col{col_idx}", f"col{col_idx}")).strip()
                fonte = normalizar_fonte(fonte_bruta)
                chave = f"{doc_id}_{fonte}"
                resumo_falha = {"task_id": task_id, "doc_id": doc_id}
                if fonte != fonte_bruta:
                    resumo_falha["fonte_bruta"] = fonte_bruta
                registros.append({
                    "chave": chave,
                    "resumo": json.dumps(resumo_falha, ensure_ascii=False),
                    "resposta": json.dumps({"nota": None, "problemas": []}, ensure_ascii=False),
                    "erro": "tarefa_nao_anotada_ou_cancelada",
                    "rodada": id_avaliador
                })
            continue

        tarefas_com_anotacao += 1
        # Pega a última anotação submetida
        ultima_anotacao = valid_annotations[-1]
        lead_time = ultima_anotacao.get("lead_time")
        completed_by = ultima_anotacao.get("completed_by")
        
        # Agrupa os resultados da anotação por from_name
        resultados_map = {}
        for r in ultima_anotacao.get("result", []):
            fn = r.get("from_name")
            if fn:
                resultados_map[fn] = r.get("value", {})

        # Processa cada uma das 3 colunas (fontes desblindadas)
        for col_idx in (1, 2, 3):
            fonte_bruta = str(data.get(f"fonte_real_col{col_idx}", f"col{col_idx}")).strip()
            fonte = normalizar_fonte(fonte_bruta)
            chave = f"{doc_id}_{fonte}"

            # Extração da nota
            nota_info = resultados_map.get(f"nota_col{col_idx}", {})
            raw_nota_choice = nota_info.get("choices", [None])[0] if isinstance(nota_info, dict) else None
            nota = parse_nota(raw_nota_choice)

            # Extração dos problemas
            prob_info = resultados_map.get(f"problemas_col{col_idx}", {})
            problemas = prob_info.get("choices", []) if isinstance(prob_info, dict) else []
            if not isinstance(problemas, list):
                problemas = [str(problemas)]

            # Extração de justificativa (se houver)
            justif_info = resultados_map.get(f"justificativa_col{col_idx}", {})
            justificativas = justif_info.get("text", []) if isinstance(justif_info, dict) else []
            justificativa = justificativas[0] if (isinstance(justificativas, list) and justificativas) else ""

            # Conclusão (checkbox de passagem)
            conc_info = resultados_map.get(f"conclusao_col{col_idx}", {})
            concluida = bool(conc_info.get("choices"))

            resp_obj: Dict[str, Any] = {
                "nota": nota,
                "problemas": problemas
            }
            if justificativa:
                resp_obj["justificativa"] = justificativa

            resumo_obj: Dict[str, Any] = {
                "task_id": task_id,
                "doc_id": doc_id,
                "coluna_label_studio": f"col{col_idx}",
                "concluida": concluida
            }
            if fonte != fonte_bruta:
                resumo_obj["fonte_bruta"] = fonte_bruta
            if lead_time is not None:
                resumo_obj["tempo"] = round(float(lead_time), 2)
            if nome_avaliador:
                resumo_obj["avaliador"] = nome_avaliador
            elif completed_by is not None:
                resumo_obj["avaliador_id"] = completed_by

            erro = ""
            if nota is None:
                erro = "nota_ausente"

            registros.append({
                "chave": chave,
                "resumo": json.dumps(resumo_obj, ensure_ascii=False),
                "resposta": json.dumps(resp_obj, ensure_ascii=False),
                "erro": erro,
                "rodada": id_avaliador
            })

    df = pd.DataFrame(registros)
    return df

def validar_consistencia_dataframe(df: pd.DataFrame, nome: str) -> bool:
    """Verifica regras essenciais de integridade e consistência para o pipeline."""
    print(f"\n--- Relatório de Consistência: {nome} ---")
    print(f"Total de registros (linhas): {len(df)}")
    
    # 1. Chave única
    duplicatas = df[df.duplicated(subset=["chave"], keep=False)]
    if not duplicatas.empty:
        print(f"❌ ATENÇÃO: Encontradas {len(duplicatas)} chaves duplicadas!")
    else:
        print("✔ Chaves únicas: 100% OK")

    # 2. Decomposição de chave
    erros_chave = 0
    fontes = set()
    documentos = set()
    for ch in df["chave"]:
        partes = str(ch).split("_", 1)
        if len(partes) != 2 or not partes[0] or not partes[1]:
            erros_chave += 1
        else:
            documentos.add(partes[0])
            fontes.add(partes[1])
            
    if erros_chave > 0:
        print(f"❌ Formato de chave inválido em {erros_chave} itens.")
    else:
        print(f"✔ Padrão de chave '<doc>_<fonte>' OK ({len(documentos)} docs, fontes: {sorted(list(fontes))})")

    # 3. Notas válidas (1 a 4)
    notas = []
    notas_nulas = 0
    for r in df["resposta"]:
        try:
            d = json.loads(r)
            n = d.get("nota")
            if n is not None:
                notas.append(n)
            else:
                notas_nulas += 1
        except Exception:
            notas_nulas += 1

    print(f"✔ Distribuição de notas válidas: {dict(pd.Series(notas).value_counts().sort_index())}")
    if notas_nulas > 0:
        print(f"⚠ Aviso: {notas_nulas} itens sem nota preenchida.")

    # 4. Problemas
    problemas_contagem: Dict[str, int] = {}
    for r in df["resposta"]:
        try:
            d = json.loads(r)
            for p in d.get("problemas", []):
                problemas_contagem[p] = problemas_contagem.get(p, 0) + 1
        except Exception:
            pass
    if problemas_contagem:
        print(f"✔ Problemas apontados: {problemas_contagem}")
    else:
        print("ℹ Nenhum problema registrado nas anotações.")

    return erros_chave == 0 and duplicatas.empty

def main():
    parser = argparse.ArgumentParser(description="Converte exportação do Label Studio para Parquet de avaliação.")
    parser.add_argument("arquivos", nargs="+", help="Arquivos JSON exportados do Label Studio")
    parser.add_argument("--saida", default="../avaliacao_llm_humana", help="Diretório base de saída para as pastas saida_av_XX")
    parser.add_argument("--prefixo", default="saida_av", help="Prefixo dos grupos/pastas (padrão: saida_av)")
    parser.add_argument("--valida-apenas", action="store_true", help="Apenas valida os dados sem salvar os arquivos finais")
    args = parser.parse_args()

    for idx, caminho in enumerate(args.arquivos, start=1):
        nome_base = os.path.splitext(os.path.basename(caminho))[0]
        # Extrai nome do avaliador se o nome do arquivo seguir o padrão (ex: tarefas_avaliacao_label_studio_Rafael.json)
        match_nome = re.search(r"studio_([A-Za-z0-9]+)", nome_base, re.IGNORECASE)
        nome_av = match_nome.group(1) if match_nome else f"Avaliador_{idx:02d}"

        print(f"\n========================================================")
        print(f"Processando [{idx}/{len(args.arquivos)}]: {caminho} ({nome_av})")
        print(f"========================================================")

        df = processar_export_label_studio(caminho, id_avaliador=idx, nome_avaliador=nome_av)
        valido = validar_consistencia_dataframe(df, nome_av)

        if not args.valida_apenas:
            pasta_grupo = os.path.join(args.saida, f"{args.prefixo}_{idx:02d}")
            os.makedirs(pasta_grupo, exist_ok=True)
            
            caminho_parquet_interno = os.path.join(pasta_grupo, "saida_juiz_llm.parquet")
            df.to_parquet(caminho_parquet_interno, index=False)
            print(f"💾 Salvo para o pipeline: {caminho_parquet_interno}")

            # Salva também cópia direta na raiz de saída para conveniência
            caminho_parquet_raiz = os.path.join(args.saida, f"{args.prefixo}_{idx:02d}.parquet")
            df.to_parquet(caminho_parquet_raiz, index=False)
            print(f"💾 Salvo arquivo único: {caminho_parquet_raiz}")

if __name__ == "__main__":
    main()
