#!/usr/bin/env python3
"""
Gera o parquet de entrada para o juiz LLM (util_vllm_batch.py).

Para cada um dos 70 documentos de avaliação e para cada um dos 3 modelos
(GPT-5, Qwen 235b, Qwen 7b), monta o prompt completo do juiz substituindo
os placeholders <<--TEXTO_ACORDAO-->> e <<--TEXTO_EXTRACAO-->> no template
prompt_juiz_llm.txt.

Resultado: parquet com 210 linhas (70 × 3), colunas:
  chave, texto, seq_documento_acordao, modelo, dificuldade, alvo, fold

Uso:
    python gerar_entrada_juiz_llm.py
"""

import os
import sys
import json
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Caminhos (relativos a este script)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ARQUIVO_AVALIACAO = os.path.join(SCRIPT_DIR, "arquivo_avaliacao.csv")
PROMPT_TEMPLATE   = os.path.join(SCRIPT_DIR, "prompt_juiz_llm.txt")

ARQUIVO_INTEGRAS = os.path.join(SCRIPT_DIR, "..", "dados", "integras_experimento_summa_novos.parquet")

MODELOS_ARQUIVOS = {
    "gpt5":    os.path.join(SCRIPT_DIR, "..", "saida", "saida_oa_gpt5.parquet"),
    "qwen235b": os.path.join(SCRIPT_DIR, "..", "saida", "saida_or_235b.parquet"),
    "qwen7b":  os.path.join(SCRIPT_DIR, "..", "saida", "saida_qwen7b.parquet"),
}

SAIDA_DIR = os.path.join(SCRIPT_DIR, "dados")
ARQUIVO_SAIDA = os.path.join(SAIDA_DIR, "entrada_juiz_llm.parquet")

# Placeholders no template do prompt do juiz
PH_ACORDAO  = "<<--TEXTO_ACORDAO-->>"
PH_EXTRACAO = "<<--TEXTO_EXTRACAO-->>"


# ---------------------------------------------------------------------------
# Formatação: JSON da extração → texto plano estruturado
# ---------------------------------------------------------------------------

def _formatar_lista(itens, rotulo_vazio="Não consta"):
    """Formata uma lista como itens de texto plano (um por linha com '- ')."""
    if not itens:
        return rotulo_vazio
    if isinstance(itens, str):
        return itens.strip() if itens.strip() else rotulo_vazio
    return "\n".join(f"- {item}" for item in itens)


def formatar_extracao_texto_plano(json_str: str, chave_debug: str) -> str:
    """Converte o JSON de extração SUMMA em texto plano estruturado.

    Campos formatados (na ordem do prompt do juiz):
        MATÉRIA, TEMAS (PONTO, ARGUMENTOS, DOUTRINA, CONCEITOS), RESUMO.

    Args:
        json_str: string JSON da coluna 'resposta' do parquet de saída.
        chave_debug: identificador para mensagens de erro.

    Returns:
        Texto plano formatado.

    Raises:
        ValueError: se o JSON não puder ser parseado ou faltar campos obrigatórios.
    """
    try:
        dados = json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"[{chave_debug}] JSON da extração inválido: {exc}\n"
            f"  Conteúdo (primeiros 300 chars): {str(json_str)[:300]}"
        ) from exc

    if not isinstance(dados, dict):
        raise ValueError(
            f"[{chave_debug}] Extração não é um dicionário JSON. "
            f"Tipo recebido: {type(dados).__name__}"
        )

    # --- Matéria ---
    # Campo pode estar ausente em extrações de modelos menores (ex: qwen7b).
    # Isso é uma omissão legítima que o juiz LLM deve avaliar — não aborta.
    materia = dados.get("Materia") or dados.get("Matéria", "")
    if not materia or not str(materia).strip():
        logging.warning(
            f"[{chave_debug}] Campo 'Materia' ausente ou vazio na extração. "
            f"Será apresentado como 'Não consta' para avaliação do juiz."
        )
        materia = "Não consta"

    # --- Temas ---
    temas = dados.get("Temas", [])
    if not isinstance(temas, list):
        raise ValueError(
            f"[{chave_debug}] Campo 'Temas' não é uma lista. "
            f"Tipo recebido: {type(temas).__name__}"
        )
    if len(temas) == 0:
        raise ValueError(
            f"[{chave_debug}] Campo 'Temas' é uma lista vazia — extração sem temas."
        )

    # --- Resumo ---
    # Campo pode estar ausente em modelos menores — warning, não aborta.
    resumo = dados.get("Resumo", "")
    if not resumo or not str(resumo).strip():
        # Fallback: 'Dispositivo' usado em alguns formatos
        resumo = dados.get("Dispositivo", "")
    if not resumo or not str(resumo).strip():
        logging.warning(
            f"[{chave_debug}] Campo 'Resumo' (e 'Dispositivo') ausente ou vazio na extração. "
            f"Será apresentado como 'Não consta' para avaliação do juiz."
        )
        resumo = "Não consta"

    # --- Montar texto plano ---
    partes = []
    partes.append(f"MATÉRIA:\n{materia}")

    for i, tema in enumerate(temas, 1):
        if not isinstance(tema, dict):
            raise ValueError(
                f"[{chave_debug}] Tema {i} não é um dicionário. "
                f"Tipo recebido: {type(tema).__name__}"
            )

        ponto = tema.get("Ponto", "Não consta")
        argumentos = tema.get("Argumentos", [])
        doutrina = tema.get("Doutrina", [])
        conceitos = tema.get("Conceitos", [])

        partes.append(f"\n--- Tema {i} ---")
        partes.append(f"PONTO: {ponto}")
        partes.append(f"ARGUMENTOS:\n{_formatar_lista(argumentos)}")
        partes.append(f"DOUTRINA:\n{_formatar_lista(doutrina)}")
        partes.append(f"CONCEITOS:\n{_formatar_lista(conceitos)}")

    partes.append(f"\nRESUMO:\n{resumo}")

    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Validações
# ---------------------------------------------------------------------------

def validar_arquivo_existe(caminho: str, descricao: str):
    """Valida que um arquivo existe, abortando com mensagem clara."""
    if not os.path.isfile(caminho):
        logging.error(f"Arquivo não encontrado ({descricao}): {caminho}")
        sys.exit(1)


def validar_coluna_existe(df: pd.DataFrame, coluna: str, descricao: str):
    """Valida que uma coluna existe no DataFrame."""
    if coluna not in df.columns:
        logging.error(
            f"Coluna '{coluna}' não encontrada em {descricao}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )
        sys.exit(1)


def validar_texto_nao_vazio(texto: str, chave_debug: str, descricao: str):
    """Valida que o texto não é nulo/vazio."""
    if not texto or not str(texto).strip():
        raise ValueError(
            f"[{chave_debug}] {descricao} está vazio ou nulo."
        )


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main():
    # --- 1. Validar arquivos de entrada ---
    logging.info("Validando arquivos de entrada...")
    validar_arquivo_existe(ARQUIVO_AVALIACAO, "arquivo de avaliação")
    validar_arquivo_existe(PROMPT_TEMPLATE, "template do prompt do juiz")
    validar_arquivo_existe(ARQUIVO_INTEGRAS, "parquet de íntegras")
    for nome, caminho in MODELOS_ARQUIVOS.items():
        validar_arquivo_existe(caminho, f"saída do modelo {nome}")

    # --- 2. Carregar template do prompt ---
    with open(PROMPT_TEMPLATE, "r", encoding="utf-8") as f:
        template = f.read()

    if PH_ACORDAO not in template:
        logging.error(f"Placeholder '{PH_ACORDAO}' não encontrado no template.")
        sys.exit(1)
    if PH_EXTRACAO not in template:
        logging.error(f"Placeholder '{PH_EXTRACAO}' não encontrado no template.")
        sys.exit(1)
    logging.info("Template do prompt carregado com sucesso.")

    # --- 3. Carregar IDs de avaliação ---
    df_avaliacao = pd.read_csv(ARQUIVO_AVALIACAO)
    validar_coluna_existe(df_avaliacao, "seq_documento_acordao", "arquivo de avaliação")
    ids_avaliacao = df_avaliacao["seq_documento_acordao"].astype(str).tolist()
    logging.info(f"IDs de avaliação: {len(ids_avaliacao)}")

    if len(ids_avaliacao) == 0:
        logging.error("Nenhum ID encontrado no arquivo de avaliação.")
        sys.exit(1)

    # Mapear metadados por ID (dificuldade, alvo, fold)
    meta_por_id = {}
    for _, row in df_avaliacao.iterrows():
        sid = str(row["seq_documento_acordao"])
        meta_por_id[sid] = {
            "dificuldade": row.get("dificuldade", ""),
            "alvo": row.get("alvo", ""),
            "fold": row.get("fold", ""),
        }

    # --- 4. Carregar íntegras ---
    logging.info("Carregando parquet de íntegras...")
    df_integras = pd.read_parquet(ARQUIVO_INTEGRAS)
    validar_coluna_existe(df_integras, "seq_documento_acordao", "parquet de íntegras")
    validar_coluna_existe(df_integras, "integra", "parquet de íntegras")

    # Indexar por seq_documento_acordao
    integras_por_id = {}
    for _, row in df_integras.iterrows():
        sid = str(row["seq_documento_acordao"])
        if sid in meta_por_id:
            integras_por_id[sid] = str(row["integra"]) if pd.notna(row["integra"]) else ""

    # Validar cobertura
    ids_faltando_integras = set(ids_avaliacao) - set(integras_por_id.keys())
    if ids_faltando_integras:
        logging.error(
            f"IDs do arquivo de avaliação não encontrados no parquet de íntegras: "
            f"{ids_faltando_integras}"
        )
        sys.exit(1)
    logging.info(f"Íntegras carregadas: {len(integras_por_id)} documentos.")

    # --- 5. Carregar extrações dos modelos ---
    extracoes = {}  # {modelo: {seq_doc_str: json_str}}
    for nome_modelo, caminho in MODELOS_ARQUIVOS.items():
        logging.info(f"Carregando extrações do modelo '{nome_modelo}'...")
        df_modelo = pd.read_parquet(caminho)
        validar_coluna_existe(df_modelo, "chave", f"saída do modelo {nome_modelo}")
        validar_coluna_existe(df_modelo, "resposta", f"saída do modelo {nome_modelo}")

        extracoes_modelo = {}
        for _, row in df_modelo.iterrows():
            chave = str(row["chave"])
            if chave in meta_por_id:
                resposta = str(row["resposta"]) if pd.notna(row["resposta"]) else ""
                extracoes_modelo[chave] = resposta

        # Validar cobertura
        ids_faltando = set(ids_avaliacao) - set(extracoes_modelo.keys())
        if ids_faltando:
            logging.error(
                f"IDs sem extração no modelo '{nome_modelo}': {ids_faltando}"
            )
            sys.exit(1)

        # Validar erros na coluna erro
        if "erro" in df_modelo.columns:
            df_70 = df_modelo[df_modelo["chave"].astype(str).isin(ids_avaliacao)]
            erros = df_70[df_70["erro"].notna() & (df_70["erro"] != "")]
            if len(erros) > 0:
                for _, row_err in erros.iterrows():
                    logging.error(
                        f"Extração com erro no modelo '{nome_modelo}', "
                        f"chave={row_err['chave']}: {row_err['erro']}"
                    )
                sys.exit(1)

        extracoes[nome_modelo] = extracoes_modelo
        logging.info(f"  → {len(extracoes_modelo)} extrações carregadas (sem erros).")

    # --- 6. Montar as 210 linhas ---
    logging.info("Montando prompts do juiz LLM...")
    linhas = []
    erros_encontrados = []

    for seq_doc_str in ids_avaliacao:
        texto_acordao = integras_por_id[seq_doc_str]

        # Validar que a íntegra não está vazia
        try:
            validar_texto_nao_vazio(texto_acordao, seq_doc_str, "Íntegra do acórdão")
        except ValueError as exc:
            erros_encontrados.append(str(exc))
            continue

        meta = meta_por_id[seq_doc_str]

        for nome_modelo in MODELOS_ARQUIVOS:
            chave_composta = f"{seq_doc_str}_{nome_modelo}"
            json_str = extracoes[nome_modelo][seq_doc_str]

            # Validar que a resposta não está vazia
            try:
                validar_texto_nao_vazio(json_str, chave_composta, "Resposta do modelo")
            except ValueError as exc:
                erros_encontrados.append(str(exc))
                continue

            # Converter JSON → texto plano
            try:
                texto_extracao = formatar_extracao_texto_plano(json_str, chave_composta)
            except ValueError as exc:
                erros_encontrados.append(str(exc))
                continue

            # Montar prompt completo
            prompt_completo = template.replace(PH_ACORDAO, texto_acordao)
            prompt_completo = prompt_completo.replace(PH_EXTRACAO, texto_extracao)

            # Validação final: placeholders não devem restar
            if PH_ACORDAO in prompt_completo:
                erros_encontrados.append(
                    f"[{chave_composta}] Placeholder '{PH_ACORDAO}' não foi substituído."
                )
                continue
            if PH_EXTRACAO in prompt_completo:
                erros_encontrados.append(
                    f"[{chave_composta}] Placeholder '{PH_EXTRACAO}' não foi substituído."
                )
                continue

            linhas.append({
                "chave": chave_composta,
                "texto": prompt_completo,
                "seq_documento_acordao": seq_doc_str,
                "modelo": nome_modelo,
                "dificuldade": meta["dificuldade"],
                "alvo": meta["alvo"],
                "fold": meta["fold"],
            })

    # --- 7. Verificar erros ---
    if erros_encontrados:
        logging.error(f"\n{'='*60}")
        logging.error(f"ABORTAR: {len(erros_encontrados)} erro(s) de validação encontrado(s):")
        logging.error(f"{'='*60}")
        for i, erro in enumerate(erros_encontrados, 1):
            logging.error(f"  {i}. {erro}")
        sys.exit(1)

    # --- 8. Validar contagem esperada ---
    esperado = len(ids_avaliacao) * len(MODELOS_ARQUIVOS)
    if len(linhas) != esperado:
        logging.error(
            f"Quantidade de linhas geradas ({len(linhas)}) difere do esperado ({esperado}). "
            f"Verifique os dados de entrada."
        )
        sys.exit(1)

    # --- 9. Salvar parquet ---
    os.makedirs(SAIDA_DIR, exist_ok=True)
    df_saida = pd.DataFrame(linhas)
    df_saida.to_parquet(ARQUIVO_SAIDA, index=False)

    logging.info(f"{'='*60}")
    logging.info(f"Parquet gerado com sucesso: {ARQUIVO_SAIDA}")
    logging.info(f"  Linhas: {len(df_saida)}")
    logging.info(f"  Colunas: {list(df_saida.columns)}")
    logging.info(f"  Modelos: {list(MODELOS_ARQUIVOS.keys())}")
    logging.info(f"  Documentos: {len(ids_avaliacao)}")
    logging.info(f"{'='*60}")

    # Resumo por modelo
    for nome_modelo in MODELOS_ARQUIVOS:
        n = len(df_saida[df_saida["modelo"] == nome_modelo])
        logging.info(f"  {nome_modelo}: {n} avaliações")

    # Resumo por alvo/dificuldade
    logging.info("\nDistribuição por alvo × dificuldade:")
    for alvo in sorted(df_saida["alvo"].unique()):
        for dif in sorted(df_saida["dificuldade"].unique()):
            n = len(df_saida[(df_saida["alvo"] == alvo) & (df_saida["dificuldade"] == dif)])
            # Dividir por 3 modelos para ter a contagem de documentos
            logging.info(f"  {alvo:>10} / {dif:<8}: {n//3} docs ({n} avaliações)")


if __name__ == "__main__":
    main()
