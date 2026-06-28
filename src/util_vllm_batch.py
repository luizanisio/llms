#!/usr/bin/env python3

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Inferência em lote usando vLLM com configuração via YAML.

Uso:
    python util_vllm_batch.py --config config_batch.yaml

Se o arquivo YAML não existir, o programa pergunta se deseja criá-lo
com um exemplo comentado e explicativo.

Entrada: arquivo parquet ou csv (com colunas chave + texto) ou pasta com arquivos .txt
Saída: arquivo parquet ou csv (com colunas chave, resumo, resposta, erro) ou pasta com .txt/.json

Retomada automática: ao reiniciar, ignora itens já processados com sucesso
e reprocessa itens com erro ou ausentes na saída.

Exemplo de YAML:
    Veja a função _YAML_EXEMPLO no final do código ou execute:
    python util_vllm_batch.py --config meu_config.yaml
    (com um arquivo que não existe para gerar o exemplo)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from datetime import timedelta
from typing import Dict, List, Any, Optional, Tuple

# Garante que a pasta src está no sys.path
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    import yaml
except ImportError:
    print("❌ O pacote 'pyyaml' não está instalado.")
    print("   Instale com: pip install pyyaml")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("❌ O pacote 'pandas' não está instalado.")
    print("   Instale com: pip install pandas")
    sys.exit(1)

# Helpers de detecção de API remota (util_openai.py)
from util_openai import eh_modelo_api_remota, extrair_nome_modelo_api, validar_modelo_api
from util import Util

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MAX_PROMPTS_LOG = 5
_COLUNAS_SAIDA_PARQUET = ["chave", "resumo", "resposta", "erro"]

_YAML_EXEMPLO = """\
# ==========================================================================
# Configuração para inferência em lote com vLLM
# Executar com: python util_vllm_batch.py --config {nome_arquivo}
# ==========================================================================

# --- Miscelânea ---
# pasta_base: diretório base (opcional) que será concatenado com os caminhos relativos de arquivos e pastas.
# pastas_modelos: lista de pastas alternativas (opcional) onde o modelo base e o LoRA serão buscados de forma independente.
#   A ordem de busca é: 1) Caminho exato/fornecido, 2) Relativo ao diretório deste YAML, 3) Pastas nesta lista, 4) pasta_base.
misc:
  pasta_base: ""
  pastas_modelos:
    # - "/caminho/alternativo/modelos_base"
    # - "/caminho/alternativo/loras_treinados"

# --- Modelo ---
# caminho: caminho para modelo HuggingFace (local ou hub)
#   Prefixos para APIs remotas (não usa vLLM local):
#     or:modelo  — OpenRouter (ex: "or:qwen/qwen3.5-35b-a3b")
#     tg:modelo  — Together.ai (ex: "tg:meta-llama/Llama-3-70b-chat-hf")
#     vl:modelo  — vLLM Server remoto (ex: "vl:meu-modelo")
#     oa:modelo  — OpenAI (ex: "oa:gpt-5-nano")
#   Sem prefixo: usa vLLM local com modelo HuggingFace
# lora: caminho para adaptador LoRA treinado (opcional, deixe vazio se não usar)
modelo:
  caminho: "/caminho/para/modelo"
  lora: ""

# --- Configuração do vLLM (ignorada para modelos de API remota) ---
# gpu_memory_utilization: fração da VRAM a usar (0.0 a 1.0). Padrão: 0.90
# max_model_len: tamanho máximo do contexto em tokens. Ajuste conforme o modelo.
# tensor_parallel_size: número de GPUs para paralelismo de tensor (1 = single GPU)
# dtype: tipo de dado ("auto", "float16", "bfloat16"). "auto" detecta automaticamente.
# enforce_eager: True desabilita CUDA graphs (mais lento, menos memória). Recomendado para 7B.
# enable_prefix_caching: True para reusar cache de prompts com prefixos idênticos. Padrão: false.
# max_num_seqs: limite máximo de sequências (prompts) concorrentes por batch. Padrão: 256. Reduza (ex: 16) para economizar VRAM.
# quantization: método de quantização do modelo (ex: "bitsandbytes", "awq", "gptq").
# load_format: formato de carga. Para "bitsandbytes" on-the-fly, também use "bitsandbytes" aqui.
# Obs: Quaisquer outros parâmetros adicionados aqui serão passados diretamente para o init do vLLM
vllm:
  gpu_memory_utilization: 0.90
  max_model_len: 32768
  tensor_parallel_size: 1
  dtype: "auto"
  enforce_eager: true
  # enable_prefix_caching: true
  # max_num_seqs: 16
  # quantization: "bitsandbytes"
  # load_format: "bitsandbytes"

# --- Parâmetros de Geração ---
# max_tokens: máximo de tokens na resposta gerada
# temperature: criatividade (0.0 = determinístico, 1.0 = criativo). Para extração use ~0.01.
# top_k: top-k sampling (2 = quase determinístico)
# top_p: nucleus sampling
# batch_size: quantos prompts enviar por vez (controla salvamento parcial). Opcional.
#   Para APIs remotas: define o número de chamadas concorrentes (threads).
# max_itens: limite máximo de itens a processar (útil para testes). Opcional.
# think: controle de reasoning para APIs remotas (opcional).
#   Valores: "low", "medium", "high", "minimal" ou combinação "high:medium" (reasoning:verbosity)
#   Deixe vazio ou omita para desabilitar reasoning.
geracao:
  max_tokens: 2048
  temperature: 0.01
  top_k: 2
  top_p: 0.9
  # batch_size: 64
  # max_itens: 10
  # think: "low"

# --- Entrada ---
# arquivo: caminho para arquivo .parquet/.csv OU pasta com arquivos .txt
#   Se parquet/csv: usa campo_chave como ID e campo_texto como conteúdo
#   Se pasta: cada arquivo .txt é um item (nome sem extensão = chave)
# campo_chave: nome da coluna com o ID (apenas para parquet/csv). Padrão: "id"
# campo_texto: nome da coluna com o texto (apenas para parquet/csv). Padrão: "texto"
# prompt_template: arquivo .txt com template do prompt (opcional).
#   Se informado, o texto do campo_texto substitui o placeholder variavel_texto.
#   Se vazio, o texto é usado diretamente como conteúdo do prompt.
# variavel_texto: placeholder no template a ser substituído pelo texto. Padrão: "<--TEXTO-->"
# system_prompt: system prompt para modelos instruct (opcional).
#   Se informado, o chat template do modelo é aplicado automaticamente.
#   Se vazio, o texto montado é enviado diretamente ao modelo (sem chat template).
# filtro: configuração opcional para processar apenas um subconjunto de IDs
#   arquivo: caminho para arquivo CSV com os IDs a serem processados
#   campo_id: nome da coluna com os IDs no arquivo CSV
entrada:
  arquivo: "./entrada.parquet"
  campo_chave: "id"
  campo_texto: "texto"
  prompt_template: ""
  variavel_texto: "<--TEXTO-->"
  system_prompt: ""
  # filtro:
  #   arquivo: "./filtro_ids.csv"
  #   campo_id: "id_documento"

# --- Saída ---
# arquivo: caminho para arquivo .parquet OU pasta para arquivos .txt/.json
#   Se termina com .parquet: gera parquet com colunas [chave, resumo, resposta, erro]
#   Caso contrário: trata como pasta e gera {chave}.txt + {chave}.json por item
# tipo_saida: tipo da resposta gerada. Valores possíveis:
#   "str"  — resposta é texto livre (padrão)
#   "json" — resposta é JSON. O texto gerado será parseado com UtilJson.mensagem_to_json
#            (extrai e corrige JSON de respostas de LLM). Se não for possível parsear,
#            a resposta original é mantida e o campo erro é preenchido.
# Retomada: ao reiniciar, itens já processados com sucesso são ignorados.
#   Itens com erro ou ausentes são reprocessados automaticamente.
saida:
  arquivo: "./saida.parquet"
  tipo_saida: "str"
"""


# ---------------------------------------------------------------------------
# Funções de Configuração
# ---------------------------------------------------------------------------

def _resolver_caminho(caminho: str, base_dir: str, pasta_base: str = "", obrigatorio: bool = True) -> str:
    """Resolve caminho relativo em relação ao diretório base ou pasta_base."""
    return Util.resolver_caminho(caminho, base_dir, pasta_base, obrigatorio=obrigatorio)


def carregar_config(yaml_path: str) -> Dict[str, Any]:
    """Carrega e valida o YAML de configuração.

    Args:
        yaml_path: caminho para o arquivo YAML

    Returns:
        Dicionário com a configuração processada (caminhos resolvidos)

    Raises:
        FileNotFoundError: se o arquivo não existir
        ValueError: se faltar configuração obrigatória
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Arquivo YAML não encontrado: '{yaml_path}'")

    with open(yaml_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}

    if not isinstance(config, dict):
        raise ValueError(f"YAML deve conter um dicionário, recebido: {type(config)}")

    base_dir = os.path.dirname(os.path.abspath(yaml_path))

    # --- Miscelânea ---
    misc = config.get("misc", {}) or {}
    
    pastas_base = misc.get("pastas_base", [])
    if isinstance(pastas_base, str):
        pastas_base = [pastas_base]
        
    pasta_base_ativa = ""
    for pb in pastas_base:
        pb_abs = pb if os.path.isabs(pb) else os.path.normpath(os.path.join(base_dir, pb))
        if os.path.isdir(pb_abs):
            pasta_base_ativa = pb_abs
            break
            
    if not pasta_base_ativa and pastas_base:
        pb = pastas_base[0]
        pasta_base_ativa = pb if os.path.isabs(pb) else os.path.normpath(os.path.join(base_dir, pb))
        
    pastas_modelos = misc.get("pastas_modelos", misc.get("pasta_modelos", []))
    if isinstance(pastas_modelos, str):
        pastas_modelos = [pastas_modelos] if pastas_modelos else []
        
    # --- Modelo (obrigatório) ---
    modelo = config.get("modelo", {}) or {}
    caminho_modelo = modelo.get("caminho", "")
    if not caminho_modelo:
        raise ValueError("modelo.caminho é obrigatório no YAML")
        
    # Para modelos de API remota (or:, tg:, vl:, oa:), preserva o prefixo
    if eh_modelo_api_remota(caminho_modelo):
        modelo["caminho"] = caminho_modelo.strip()
    else:
        # Busca dinâmica para o caminho do modelo
        modelo["caminho"] = Util.buscar_caminho_modelo(caminho_modelo, base_dir, pastas_modelos, pasta_base_ativa)
    
    lora = modelo.get("lora", "")
    if lora:
        # Busca dinâmica para o caminho do LoRA
        modelo["lora"] = Util.buscar_caminho_modelo(lora, base_dir, pastas_modelos, pasta_base_ativa)
        
    config["modelo"] = modelo

    # --- vLLM (defaults) ---
    vllm = config.get("vllm", {}) or {}
    vllm.setdefault("gpu_memory_utilization", 0.90)
    vllm.setdefault("max_model_len", 32768)
    vllm.setdefault("tensor_parallel_size", 1)
    vllm.setdefault("dtype", "auto")
    vllm.setdefault("enforce_eager", True)
    config["vllm"] = vllm

    # --- Geração (defaults) ---
    geracao = config.get("geracao", {}) or {}
    geracao.setdefault("max_tokens", 2048)
    geracao.setdefault("temperature", 0.01)
    geracao.setdefault("top_k", 2)
    geracao.setdefault("top_p", 0.9)
    geracao.setdefault("batch_size", 64)
    geracao.setdefault("max_itens", 0)
    # think: controle de reasoning para APIs remotas (None = sem reasoning)
    if "think" not in geracao or geracao["think"] is None:
        geracao["think"] = None
    else:
        geracao["think"] = str(geracao["think"]).strip() or None
    config["geracao"] = geracao

    # --- Entrada (obrigatório) ---
    entrada = config.get("entrada", {}) or {}
    arquivo_entrada = entrada.get("arquivo", "")
    if not arquivo_entrada:
        raise ValueError("entrada.arquivo é obrigatório no YAML")
    entrada["arquivo"] = _resolver_caminho(arquivo_entrada, base_dir, pasta_base_ativa)
    entrada.setdefault("campo_chave", "id")
    entrada.setdefault("campo_texto", "texto")
    prompt_tpl = entrada.get("prompt_template", "")
    if prompt_tpl:
        entrada["prompt_template"] = _resolver_caminho(prompt_tpl, base_dir, pasta_base_ativa)
    entrada.setdefault("variavel_texto", "<--TEXTO-->")
    entrada.setdefault("system_prompt", "")

    filtro = entrada.get("filtro", {})
    if filtro and isinstance(filtro, dict):
        arquivo_filtro = filtro.get("arquivo", "")
        if arquivo_filtro:
            filtro["arquivo"] = _resolver_caminho(arquivo_filtro, base_dir, pasta_base_ativa)
        entrada["filtro"] = filtro

    config["entrada"] = entrada

    # --- Saída (obrigatório) ---
    saida = config.get("saida", {}) or {}
    caminho_saida = saida.get("arquivo", "")
    if not caminho_saida:
        raise ValueError("saida.arquivo é obrigatório no YAML")
    saida["arquivo"] = _resolver_caminho(caminho_saida, base_dir, pasta_base_ativa, obrigatorio=False)
    # tipo_saida: "str" (padrão) ou "json"/"dict"
    tipo_saida = str(saida.get("tipo_saida", "str")).strip().lower()
    if tipo_saida in ("json", "dict"):
        tipo_saida = "json"
    else:
        tipo_saida = "str"
    saida["tipo_saida"] = tipo_saida
    config["saida"] = saida

    return config


def validar_config(config: Dict[str, Any]) -> List[str]:
    """Valida a configuração antes de iniciar o processamento.

    Verifica existência de arquivos/diretórios críticos e coerência
    dos parâmetros para evitar que o vLLM seja inicializado (processo
    caro) apenas para falhar depois por falta de um arquivo.

    Args:
        config: configuração já carregada via carregar_config

    Returns:
        Lista de mensagens de erro. Lista vazia = tudo ok.
    """
    erros: List[str] = []

    # --- Modelo ---
    caminho_modelo = config["modelo"]["caminho"]
    if eh_modelo_api_remota(caminho_modelo):
        # Valida modelo de API remota via util_openai
        ok, msg = validar_modelo_api(caminho_modelo)
        if not ok:
            erros.append(
                f"Modelo de API remota inválido: {msg}\n"
                f"   Verifique modelo.caminho no YAML."
            )
    else:
        if not os.path.exists(caminho_modelo):
            erros.append(
                f"Modelo não encontrado: '{caminho_modelo}'\n"
                f"   Verifique modelo.caminho no YAML."
            )

    lora = config["modelo"].get("lora", "")
    if lora and not os.path.exists(lora):
        erros.append(
            f"Adaptador LoRA não encontrado: '{lora}'\n"
            f"   Verifique modelo.lora no YAML."
        )

    # --- Entrada ---
    cfg_entrada = config["entrada"]
    arquivo_entrada = cfg_entrada["arquivo"]
    if arquivo_entrada.lower().endswith((".parquet", ".csv")):
        if not os.path.isfile(arquivo_entrada):
            erros.append(
                f"Arquivo de entrada não encontrado: '{arquivo_entrada}'\n"
                f"   Verifique entrada.arquivo no YAML."
            )
    else:
        if not os.path.isdir(arquivo_entrada):
            erros.append(
                f"Pasta de entrada não encontrada: '{arquivo_entrada}'\n"
                f"   Verifique entrada.arquivo no YAML."
            )

    # --- Prompt template ---
    prompt_tpl = cfg_entrada.get("prompt_template", "")
    if prompt_tpl:
        if not os.path.isfile(prompt_tpl):
            erros.append(
                f"Arquivo de prompt template não encontrado: '{prompt_tpl}'\n"
                f"   Verifique entrada.prompt_template no YAML."
            )
        else:
            # Verifica se a variável de substituição existe no template
            variavel = cfg_entrada.get("variavel_texto", "<--TEXTO-->")
            try:
                with open(prompt_tpl, "r", encoding="utf-8") as f:
                    conteudo_tpl = f.read()
                if variavel not in conteudo_tpl:
                    erros.append(
                        f"Variável '{variavel}' não encontrada no template "
                        f"'{os.path.basename(prompt_tpl)}'.\n"
                        f"   Verifique entrada.variavel_texto no YAML ou o conteúdo do template."
                    )
            except Exception as e:
                erros.append(
                    f"Erro ao ler template '{prompt_tpl}': {e}"
                )

    # --- Filtro ---
    filtro = cfg_entrada.get("filtro", {})
    if filtro and isinstance(filtro, dict):
        arquivo_filtro = filtro.get("arquivo", "")
        if arquivo_filtro and not os.path.isfile(arquivo_filtro):
            erros.append(
                f"Arquivo de filtro não encontrado: '{arquivo_filtro}'\n"
                f"   Verifique entrada.filtro.arquivo no YAML."
            )

    # --- Parâmetros de geração ---
    cfg_geracao = config["geracao"]
    max_tokens = cfg_geracao.get("max_tokens", 2048)
    if not isinstance(max_tokens, (int, float)) or max_tokens <= 0:
        erros.append(
            f"geracao.max_tokens deve ser um inteiro positivo, recebido: {max_tokens}"
        )

    temperature = cfg_geracao.get("temperature", 0.01)
    if not isinstance(temperature, (int, float)) or temperature < 0:
        erros.append(
            f"geracao.temperature deve ser >= 0, recebido: {temperature}"
        )

    # Validações de vLLM só fazem sentido para modelo local
    if not eh_modelo_api_remota(caminho_modelo):
        gpu_mem = config["vllm"].get("gpu_memory_utilization", 0.9)
        if not isinstance(gpu_mem, (int, float)) or not (0.0 < gpu_mem <= 1.0):
            erros.append(
                f"vllm.gpu_memory_utilization deve estar entre 0.0 e 1.0 (exclusive/inclusive), "
                f"recebido: {gpu_mem}"
            )

    return erros


# ---------------------------------------------------------------------------
# Funções de Entrada
# ---------------------------------------------------------------------------

def _eh_pasta(caminho: str) -> bool:
    """Detecta se o caminho é pasta (existente ou não-parquet)."""
    if os.path.isdir(caminho):
        return True
    # Se não existe ainda e não termina com .parquet, trata como pasta
    return not caminho.lower().endswith(".parquet")


def carregar_entrada(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Carrega itens de entrada (parquet ou pasta de .txt).

    Returns:
        Lista de dicts ``{chave: str, texto: str}``
    """
    cfg_entrada = config["entrada"]
    arquivo = cfg_entrada["arquivo"]

    # Lê filtro de IDs se configurado
    ids_filtro = None
    filtro_config = cfg_entrada.get("filtro", {})
    if filtro_config and isinstance(filtro_config, dict):
        arquivo_filtro = filtro_config.get("arquivo")
        campo_id_filtro = filtro_config.get("campo_id")
        
        if arquivo_filtro and campo_id_filtro:
            if os.path.exists(arquivo_filtro):
                try:
                    df_filtro = pd.read_csv(arquivo_filtro)
                    if campo_id_filtro in df_filtro.columns:
                        ids_filtro = set(df_filtro[campo_id_filtro].astype(str).str.strip())
                        print(f"🔍 Filtro carregado: {len(ids_filtro)} IDs de '{arquivo_filtro}' (campo '{campo_id_filtro}')")
                    else:
                        print(f"⚠️  Aviso: Coluna '{campo_id_filtro}' não encontrada no arquivo de filtro '{arquivo_filtro}'.")
                except Exception as e:
                    print(f"⚠️  Aviso: Erro ao ler arquivo de filtro '{arquivo_filtro}': {e}")
            else:
                print(f"⚠️  Aviso: Arquivo de filtro não encontrado: {arquivo_filtro}")

    if arquivo.lower().endswith((".parquet", ".csv")):
        return _carregar_entrada_parquet(arquivo, cfg_entrada, ids_filtro)
    elif os.path.isdir(arquivo):
        return _carregar_entrada_pasta(arquivo, ids_filtro)
    else:
        raise FileNotFoundError(
            f"Entrada não encontrada: '{arquivo}' "
            f"(esperado arquivo .parquet/.csv ou pasta com .txt)"
        )


def _carregar_entrada_parquet(
    arquivo: str, cfg_entrada: Dict[str, Any], ids_filtro: Optional[set] = None
) -> List[Dict[str, str]]:
    """Carrega entrada de arquivo parquet ou csv."""
    if not os.path.isfile(arquivo):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: '{arquivo}'")

    from util_pandas import ler_dataset
    df = ler_dataset(arquivo)
    
    filtro_config = cfg_entrada.get("filtro", {})
    dataset_filtro = filtro_config.get("dataset_filtro")
    if not dataset_filtro:
        dataset_filtro = cfg_entrada.get("dataset_filtro")
        
    if dataset_filtro and isinstance(dataset_filtro, dict):
        from util_pandas import aplicar_filtro_dataset
        df = aplicar_filtro_dataset(df, dataset_filtro)

    campo_chave = cfg_entrada["campo_chave"]
    campo_texto = cfg_entrada["campo_texto"]

    if campo_chave not in df.columns:
        raise ValueError(
            f"Coluna '{campo_chave}' não encontrada no parquet. "
            f"Colunas disponíveis: {list(df.columns)}"
        )
    if campo_texto not in df.columns:
        raise ValueError(
            f"Coluna '{campo_texto}' não encontrada no parquet. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    itens = []
    for _, row in df.iterrows():
        chave = str(row[campo_chave])
        if ids_filtro is not None and str(chave).strip() not in ids_filtro:
            continue
        texto = str(row[campo_texto]) if pd.notna(row[campo_texto]) else ""
        if texto.strip():
            itens.append({"chave": chave, "texto": texto})

    return itens


def _carregar_entrada_pasta(pasta: str, ids_filtro: Optional[set] = None) -> List[Dict[str, str]]:
    """Carrega entrada de pasta com arquivos .txt."""
    if not os.path.isdir(pasta):
        raise FileNotFoundError(f"Pasta de entrada não encontrada: '{pasta}'")

    itens = []
    for nome in sorted(os.listdir(pasta)):
        if not nome.lower().endswith(".txt"):
            continue
        caminho = os.path.join(pasta, nome)
        if not os.path.isfile(caminho):
            continue
        chave = os.path.splitext(nome)[0]
        if ids_filtro is not None and str(chave).strip() not in ids_filtro:
            continue
        with open(caminho, "r", encoding="utf-8") as f:
            texto = f.read()
        if texto.strip():
            itens.append({"chave": chave, "texto": texto})

    return itens


# ---------------------------------------------------------------------------
# Funções de Prompt
# ---------------------------------------------------------------------------

def _carregar_template(caminho: str) -> str:
    """Carrega template de prompt de arquivo .txt."""
    if not os.path.isfile(caminho):
        raise FileNotFoundError(f"Arquivo de template não encontrado: '{caminho}'")
    with open(caminho, "r", encoding="utf-8") as f:
        return f.read()


def montar_prompt(texto: str, config: Dict[str, Any]) -> str:
    """Monta o conteúdo do prompt (sem chat template).

    Se prompt_template estiver configurado, substitui variavel_texto pelo texto.
    Caso contrário, retorna o texto diretamente.

    Args:
        texto: conteúdo do campo_texto (entrada)
        config: configuração completa

    Returns:
        String com o conteúdo do prompt montado
    """
    texto = texto.replace('<br>','\n')
    cfg_entrada = config["entrada"]
    template_path = cfg_entrada.get("prompt_template", "")

    if template_path:
        template = _carregar_template(template_path)
        variavel = cfg_entrada.get("variavel_texto", "<--TEXTO-->")
        if variavel not in template:
            print(
                f"⚠️  Variável '{variavel}' não encontrada no template. "
                f"O texto será concatenado ao final."
            )
            return template + "\n" + texto
        return template.replace(variavel, texto)

    return texto


def _processar_resposta(texto_resposta: str, config: Dict[str, Any]) -> Tuple[str, str]:
    """Processa e valida a resposta do modelo conforme tipo_saida configurado.

    Validações aplicadas:
    - tipo_saida == "str": verifica se a resposta tem conteúdo (não vazia).
    - tipo_saida == "json": tenta extrair JSON da resposta usando
      UtilJson.mensagem_to_json e valida o JSON resultante com json.loads.
      Se a extração ou validação falhar, retorna a resposta original com erro.

    Itens com erro serão reprocessados automaticamente na próxima execução.

    Args:
        texto_resposta: texto bruto da resposta do modelo
        config: configuração completa

    Returns:
        Tupla (resposta_processada, erro):
        - resposta_processada: JSON serializado (tipo_saida=json) ou texto original
        - erro: string vazia se ok, mensagem de erro se falhou
    """
    tipo_saida = config["saida"].get("tipo_saida", "str")

    # --- Validação para saída texto (str) ---
    if tipo_saida != "json":
        if not texto_resposta or not texto_resposta.strip():
            return texto_resposta, "Resposta vazia gerada pelo modelo"
        return texto_resposta, ""

    # --- Validação para saída JSON ---
    if not texto_resposta or not texto_resposta.strip():
        return texto_resposta, "Resposta vazia — não foi possível extrair JSON"

    def _validar_json_serializado(json_str: str) -> Tuple[str, str]:
        """Valida se a string JSON serializada é um JSON válido."""
        try:
            json.loads(json_str)
            return json_str, ""
        except (json.JSONDecodeError, ValueError) as e:
            return json_str, f"JSON extraído é inválido: {e}"

    try:
        from util_openai import UtilJson
        resultado_json = UtilJson.mensagem_to_json(texto_resposta, padrao=None)
        if resultado_json is None:
            return texto_resposta, "Não foi possível extrair JSON da resposta"
        json_str = json.dumps(resultado_json, ensure_ascii=False, indent=2)
        return _validar_json_serializado(json_str)
    except ImportError:
        # Fallback: tenta com Util.mensagem_to_json de util.py
        try:
            from util import Util
            resultado_json = Util.mensagem_to_json(texto_resposta, padrao=None)
            if resultado_json is None:
                return texto_resposta, "Não foi possível extrair JSON da resposta"
            json_str = json.dumps(resultado_json, ensure_ascii=False, indent=2)
            return _validar_json_serializado(json_str)
        except Exception as e:
            return texto_resposta, f"Erro ao extrair JSON: {type(e).__name__}: {e}"
    except Exception as e:
        return texto_resposta, f"Erro ao extrair JSON: {type(e).__name__}: {e}"


def formatar_prompt_final(
    conteudo: str, config: Dict[str, Any], tokenizer
) -> str:
    """Formata prompt com chat template do tokenizer.

    Para modelos Instruct, o chat template é essencial para que o modelo
    saiba quando parar de gerar. Sem ele, o modelo gera até max_tokens.

    - Se system_prompt estiver configurado: aplica chat template com
      mensagens [system, user] + generation prompt.
    - Caso contrário: aplica chat template apenas com [user] + generation prompt.
    - Fallback: se o tokenizer não suporta chat template, retorna o conteúdo cru.

    Args:
        conteudo: prompt montado (via montar_prompt)
        config: configuração completa
        tokenizer: tokenizer do vLLM

    Returns:
        String formatada pronta para o modelo
    """
    system_prompt = config["entrada"].get("system_prompt", "")

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conteudo},
        ]
    else:
        messages = [
            {"role": "user", "content": conteudo},
        ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return conteudo


# ---------------------------------------------------------------------------
# Funções de Saída Existente (Retomada)
# ---------------------------------------------------------------------------

def _saida_eh_parquet(config: Dict[str, Any]) -> bool:
    """Verifica se a saída configurada é parquet."""
    return config["saida"]["arquivo"].lower().endswith(".parquet")


def carregar_saida_existente(config: Dict[str, Any]) -> set:
    """Identifica chaves já processadas com sucesso na saída.

    Retomada:
    - Parquet: itens cujo campo 'erro' está vazio/NaN são considerados ok.
    - Pasta: itens cujo arquivo {chave}.txt existe.

    Returns:
        Conjunto de chaves (strings) já processadas com sucesso
    """
    arquivo_saida = config["saida"]["arquivo"]

    if _saida_eh_parquet(config):
        return _carregar_saida_parquet(arquivo_saida)
    else:
        return _carregar_saida_pasta(arquivo_saida)


def _carregar_saida_parquet(arquivo: str) -> set:
    """Identifica chaves ok no parquet de saída."""
    arquivo_bak = arquivo + ".bak"

    # Se o arquivo principal não existe mas o backup existe, restaura
    if not os.path.isfile(arquivo) and os.path.isfile(arquivo_bak):
        print(f"⚠️  Arquivo '{arquivo}' não encontrado. Restaurando do backup '{arquivo_bak}'...")
        import shutil
        try:
            shutil.copy2(arquivo_bak, arquivo)
        except Exception as e:
            print(f"⚠️  Erro ao restaurar backup: {e}")

    if not os.path.isfile(arquivo):
        return set()

    try:
        df = pd.read_parquet(arquivo)
    except Exception as e:
        print(f"⚠️  Erro ao ler parquet de saída existente: {e}")
        if os.path.isfile(arquivo_bak):
            print(f"⚠️  Tentando restaurar do backup '{arquivo_bak}'...")
            import shutil
            try:
                shutil.copy2(arquivo_bak, arquivo)
                df = pd.read_parquet(arquivo)
            except Exception as e2:
                print(f"⚠️  Erro ao ler parquet do backup: {e2}")
                return set()
        else:
            return set()

    if "chave" not in df.columns:
        return set()

    # Itens com sucesso: erro está vazio/NaN E resposta tem conteúdo
    mask_sem_erro = pd.Series(True, index=df.index)
    if "erro" in df.columns:
        mask_sem_erro = df["erro"].isna() | (df["erro"].astype(str).str.strip() == "")

    mask_com_resposta = pd.Series(True, index=df.index)
    if "resposta" in df.columns:
        mask_com_resposta = df["resposta"].notna() & (df["resposta"].astype(str).str.strip() != "")

    ok = df[mask_sem_erro & mask_com_resposta]

    # Conta itens descartados por resposta vazia (para log informativo)
    sem_erro_sem_resposta = df[mask_sem_erro & ~mask_com_resposta]
    if len(sem_erro_sem_resposta) > 0:
        print(
            f"   ⚠️  {len(sem_erro_sem_resposta)} item(ns) com resposta vazia "
            f"serão reprocessado(s)"
        )

    return set(ok["chave"].astype(str))


def _carregar_saida_pasta(pasta: str) -> set:
    """Identifica chaves ok na pasta de saída."""
    if not os.path.isdir(pasta):
        return set()

    chaves = set()
    vazios = 0
    for nome in os.listdir(pasta):
        if nome.lower().endswith(".txt"):
            caminho = os.path.join(pasta, nome)
            if not os.path.isfile(caminho):
                continue
            # Verifica se o arquivo tem conteúdo
            try:
                tamanho = os.path.getsize(caminho)
                if tamanho == 0:
                    vazios += 1
                    continue
                with open(caminho, "r", encoding="utf-8") as f:
                    conteudo = f.read(512)  # lê só início para eficiência
                if not conteudo.strip():
                    vazios += 1
                    continue
            except Exception:
                vazios += 1
                continue
            chave = os.path.splitext(nome)[0]
            chaves.add(chave)

    if vazios > 0:
        print(
            f"   ⚠️  {vazios} arquivo(s) .txt com resposta vazia "
            f"serão reprocessado(s)"
        )

    return chaves


# ---------------------------------------------------------------------------
# Funções de Salvamento
# ---------------------------------------------------------------------------

def salvar_resultados_parquet(
    arquivo_saida: str,
    resultados: List[Dict[str, Any]],
    append: bool = True,
) -> None:
    """Salva ou atualiza parquet de saída.

    Se append=True e o arquivo já existe, combina resultados existentes
    com novos, sobrescrevendo itens pela chave (últimos vencem).

    Args:
        arquivo_saida: caminho do parquet
        resultados: lista de dicts com {chave, resumo, resposta, erro}
        append: se True, mescla com dados existentes
    """
    df_novo = pd.DataFrame(resultados, columns=_COLUNAS_SAIDA_PARQUET)

    if append and os.path.isfile(arquivo_saida):
        try:
            # Cria backup do arquivo antes de abri-lo/modificá-lo
            import shutil
            arquivo_bak = arquivo_saida + ".bak"
            shutil.copy2(arquivo_saida, arquivo_bak)

            df_existente = pd.read_parquet(arquivo_saida)
            # Remove itens que serão atualizados
            chaves_novas = set(df_novo["chave"].astype(str))
            df_existente = df_existente[
                ~df_existente["chave"].astype(str).isin(chaves_novas)
            ]
            df_final = pd.concat([df_existente, df_novo], ignore_index=True)
        except Exception as e:
            print(f"⚠️  Erro ao mesclar com parquet existente: {e}")
            df_final = df_novo
    else:
        df_final = df_novo

    # Garante diretório existe
    pasta = os.path.dirname(arquivo_saida)
    if pasta:
        os.makedirs(pasta, exist_ok=True)

    # Escrita atômica: salva num arquivo temporário e renomeia
    arquivo_tmp = arquivo_saida + ".tmp"
    df_final.to_parquet(arquivo_tmp, index=False)
    os.replace(arquivo_tmp, arquivo_saida)


def salvar_resultado_pasta(
    pasta_saida: str,
    chave: str,
    resposta: str,
    resumo: Dict[str, Any],
    erro: str = "",
) -> None:
    """Salva resultado individual em pasta (.txt + .json).

    Args:
        pasta_saida: pasta de destino
        chave: identificador do item
        resposta: texto da resposta
        resumo: dicionário com usage/model/tempo
        erro: mensagem de erro (vazio se sucesso)
    """
    os.makedirs(pasta_saida, exist_ok=True)

    # .txt com a resposta
    txt_path = os.path.join(pasta_saida, f"{chave}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(resposta)

    # .json com resumo + erro
    json_data = dict(resumo)
    json_data["chave"] = chave
    if erro:
        json_data["erro"] = erro

    json_path = os.path.join(pasta_saida, f"{chave}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Formatação de ETA
# ---------------------------------------------------------------------------

def _formatar_eta(inicio: float, feitos: int, total: int) -> str:
    """Retorna string com ETA estimado."""
    if feitos <= 0:
        return ""
    elapsed = time.time() - inicio
    restantes = total - feitos
    eta_s = (elapsed / feitos) * restantes
    if eta_s < 60:
        return f"⏱️ ~{int(eta_s)}s restantes"
    elif eta_s < 3600:
        return f"⏱️ ~{int(eta_s // 60)}min restantes"
    else:
        h = int(eta_s // 3600)
        m = int((eta_s % 3600) // 60)
        return f"⏱️ ~{h}h{m:02d}min restantes"


def _formatar_tempo(segundos: float) -> str:
    """Formata duração em formato legível."""
    if segundos < 60:
        return f"{segundos:.1f}s"
    elif segundos < 3600:
        m = int(segundos // 60)
        s = int(segundos % 60)
        return f"{m}min{s:02d}s"
    else:
        h = int(segundos // 3600)
        m = int((segundos % 3600) // 60)
        return f"{h}h{m:02d}min"


# ---------------------------------------------------------------------------
# Funções de Saída Existente — Chaves com Erro
# ---------------------------------------------------------------------------

def _carregar_chaves_com_erro_parquet(arquivo: str) -> set:
    """Identifica chaves com erro no parquet de saída."""
    if not os.path.isfile(arquivo):
        return set()
    try:
        df = pd.read_parquet(arquivo)
    except Exception:
        return set()
    if "chave" not in df.columns or "erro" not in df.columns:
        return set()
    mask_com_erro = df["erro"].notna() & (df["erro"].astype(str).str.strip() != "")
    return set(df[mask_com_erro]["chave"].astype(str))


def _carregar_chaves_com_erro_pasta(pasta: str) -> set:
    """Identifica chaves com erro na pasta de saída (arquivos .txt vazios ou ausentes com .json)."""
    if not os.path.isdir(pasta):
        return set()
    chaves_erro = set()
    for nome in os.listdir(pasta):
        if not nome.lower().endswith(".json"):
            continue
        chave = os.path.splitext(nome)[0]
        caminho_json = os.path.join(pasta, nome)
        caminho_txt = os.path.join(pasta, f"{chave}.txt")
        try:
            with open(caminho_json, "r", encoding="utf-8") as f:
                dados = json.load(f)
            if dados.get("erro"):
                chaves_erro.add(chave)
                continue
        except Exception:
            pass
        # txt vazio ou ausente = erro
        if not os.path.isfile(caminho_txt):
            chaves_erro.add(chave)
        else:
            try:
                if os.path.getsize(caminho_txt) == 0:
                    chaves_erro.add(chave)
            except Exception:
                chaves_erro.add(chave)
    return chaves_erro


def carregar_chaves_com_erro(config: Dict[str, Any]) -> set:
    """Retorna conjunto de chaves que possuem erro na saída existente."""
    arquivo_saida = config["saida"]["arquivo"]
    if _saida_eh_parquet(config):
        return _carregar_chaves_com_erro_parquet(arquivo_saida)
    else:
        return _carregar_chaves_com_erro_pasta(arquivo_saida)


# ---------------------------------------------------------------------------
# Log de Processamento (incremental)
# ---------------------------------------------------------------------------

class BatchLog:
    """Gerencia o log incremental de processamento em lote.

    O log é gravado em arquivo texto junto à saída, com blocos estruturados
    para facilitar a leitura e o registro de experimentos.
    """

    def __init__(self, config: Dict[str, Any]):
        arquivo_saida = config["saida"]["arquivo"]
        eh_parquet = _saida_eh_parquet(config)
        if eh_parquet:
            pasta = os.path.dirname(os.path.abspath(arquivo_saida))
            prefixo = os.path.splitext(os.path.basename(arquivo_saida))[0] + "_"
            self._log_path = os.path.join(pasta, f"{prefixo}processamento.log")
            self._prompt_log_path = os.path.join(pasta, f"{prefixo}prompts.log")
        else:
            pasta = os.path.abspath(arquivo_saida)
            os.makedirs(pasta, exist_ok=True)
            self._log_path = os.path.join(pasta, "processamento.log")
            self._prompt_log_path = os.path.join(pasta, "prompts.log")

        self._prompts_registrados = 0

        # Limpa arquivo de log de prompts se a saída ainda não existe (nova execução)
        existe_saida = os.path.isfile(arquivo_saida) if eh_parquet else os.path.isdir(arquivo_saida)
        if not existe_saida:
            if os.path.isfile(self._prompt_log_path):
                try:
                    os.remove(self._prompt_log_path)
                except Exception:
                    pass

    # -- helpers --
    @staticmethod
    def _agora() -> str:
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    def _escrever(self, texto: str) -> None:
        pasta = os.path.dirname(self._log_path)
        if pasta:
            os.makedirs(pasta, exist_ok=True)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(texto)

    # -- blocos --
    def registrar_prompt(self, chave: str, prompt: str) -> None:
        """Registra exemplos de prompts gerados, até um limite de 5 por execução."""
        if self._prompts_registrados >= MAX_PROMPTS_LOG:
            return
        
        self._prompts_registrados += 1
        pasta = os.path.dirname(self._prompt_log_path)
        if pasta:
            os.makedirs(pasta, exist_ok=True)
            
        bloco = (
            "=================================================================\n"
            f"Data/Hora: {self._agora()}\n"
            f"Chave: {chave}\n"
            f"Ordem nesta execução: {self._prompts_registrados}/5\n"
            "=================================================================\n"
            f"{prompt}\n\n"
        )
        with open(self._prompt_log_path, "a", encoding="utf-8") as f:
            f.write(bloco)

    def registrar_inicio(
        self, itens_novos: int, itens_erro: int, itens_concluidos: int
    ) -> None:
        bloco = (
            "\n==========================\n"
            f"Processamento iniciado em: {self._agora()}\n"
            f"Itens novos: {itens_novos}\n"
            f"Itens com erro: {itens_erro}\n"
            f"Itens concluídos: {itens_concluidos}\n"
        )
        self._escrever(bloco)

    def registrar_batch(
        self,
        batch_num: int,
        itens_batch: int,
        itens_corrigidos: int,
        itens_concluidos: int,
        itens_erro: int,
        inicio_batch: float,
        velocidade: float,
        previsao_termino: str,
        tokens_gerados_minuto: float = 0,
        tokens_processados_minuto: float = 0,
    ) -> None:
        bloco = (
            "---------------------------------------------\n"
            f"Batch {batch_num} iniciado em: {datetime.fromtimestamp(inicio_batch).strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"Itens do batch: {itens_batch}\n"
            f"Itens corrigidos: {itens_corrigidos}\n"
            f"Itens concluídos: {itens_concluidos}\n"
            f"Itens com erro: {itens_erro}\n"
            f"Batch finalizado em: {self._agora()}\n"
            f"Velocidade: {velocidade:.1f} itens/minuto\n"
            f"Tokens gerados/minuto: {tokens_gerados_minuto:.1f}\n"
            f"Tokens processados/minuto: {tokens_processados_minuto:.1f}\n"
            f"Previsão de término: {previsao_termino}\n"
        )
        self._escrever(bloco)

    def registrar_final(
        self,
        itens_processados: int,
        itens_corrigidos: int,
        itens_concluidos: int,
        itens_erro: int,
        velocidade: float,
        tokens_gerados_minuto: float = 0,
        tokens_processados_minuto: float = 0,
    ) -> None:
        bloco = (
            "---------------------------------------------\n"
            f"Processamento finalizado em: {self._agora()}\n"
            f"Itens processados: {itens_processados}\n"
            f"Itens corrigidos: {itens_corrigidos}\n"
            f"Itens concluídos: {itens_concluidos}\n"
            f"Itens com erro: {itens_erro}\n"
            f"Velocidade: {velocidade:.1f} itens/minuto\n"
            f"Tokens gerados/minuto: {tokens_gerados_minuto:.1f}\n"
            f"Tokens processados/minuto: {tokens_processados_minuto:.1f}\n"
            "==========================\n"
        )
        self._escrever(bloco)


# ---------------------------------------------------------------------------
# Processamento em Batch
# ---------------------------------------------------------------------------

def processar_batch(
    engine,
    tokenizer,
    itens_pendentes: List[Dict[str, str]],
    config: Dict[str, Any],
    batch_log: Optional["BatchLog"] = None,
    chaves_erro_anteriores: Optional[set] = None,
) -> Dict[str, Any]:
    """Processa itens pendentes em batches, salvando incrementalmente.

    Args:
        engine: VLLMInferenceEngine inicializado
        tokenizer: tokenizer do modelo
        itens_pendentes: lista de {chave, texto}
        config: configuração completa
        batch_log: instância de BatchLog para registrar o log incremental
        chaves_erro_anteriores: chaves que tinham erro antes desta execução

    Returns:
        Dicionário com estatísticas do processamento
    """
    cfg_geracao = config["geracao"]
    max_tokens = cfg_geracao["max_tokens"]
    temperature = cfg_geracao["temperature"]
    top_k = cfg_geracao["top_k"]
    top_p = cfg_geracao["top_p"]
    batch_size = cfg_geracao.get("batch_size", 64)

    eh_parquet = _saida_eh_parquet(config)
    arquivo_saida = config["saida"]["arquivo"]

    if chaves_erro_anteriores is None:
        chaves_erro_anteriores = set()

    total = len(itens_pendentes)
    stats = {
        "processados_ok": 0,
        "processados_erro": 0,
        "tokens_entrada": 0,
        "tokens_saida": 0,
        "corrigidos": 0,
        "erros_mantidos": 0,
        "erros_novos": 0,
    }

    print(f"\n🔄 Processando {total} item(ns) em batches de {batch_size}...")
    inicio_total = time.time()

    # Cache do template (carrega uma vez)
    _template_cache = {}

    for batch_start in range(0, total, batch_size):
        batch_itens = itens_pendentes[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        # Monta prompts para o batch
        prompts = []
        for item in batch_itens:
            conteudo = montar_prompt(item["texto"], config)
            prompt_final = formatar_prompt_final(conteudo, config, tokenizer)
            prompts.append(prompt_final)
            if batch_log:
                batch_log.registrar_prompt(item["chave"], prompt_final)

        # Info do batch
        eta = _formatar_eta(inicio_total, batch_start, total)
        print(
            f"\n📦 Batch {batch_num}/{total_batches} "
            f"({len(batch_itens)} itens) {eta}"
        )

        # Geração via vLLM
        inicio_batch = time.time()
        try:
            resultados_vllm = engine.generate_batch(
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                n=1,
            )
        except Exception as e:
            # Erro no batch inteiro: marca todos com erro
            print(f"❌ Erro no batch {batch_num}: {e}")
            tempo_batch = time.time() - inicio_batch
            resultados_batch = []
            for item in batch_itens:
                resultados_batch.append({
                    "chave": item["chave"],
                    "resumo": json.dumps({
                        "model": config["modelo"]["caminho"],
                        "tempo": round(tempo_batch / len(batch_itens), 3),
                    }, ensure_ascii=False),
                    "resposta": "",
                    "erro": f"{type(e).__name__}: {e}",
                })
                stats["processados_erro"] += 1
                # Classifica erro para o log
                if item["chave"] in chaves_erro_anteriores:
                    stats["erros_mantidos"] += 1
                else:
                    stats["erros_novos"] += 1

            if eh_parquet:
                salvar_resultados_parquet(arquivo_saida, resultados_batch)
            else:
                for r in resultados_batch:
                    resumo = json.loads(r["resumo"])
                    salvar_resultado_pasta(
                        arquivo_saida, r["chave"], r["resposta"],
                        resumo, r["erro"]
                    )

            # Log do batch com erro total
            if batch_log:
                batch_corrigidos = 0
                batch_ok = 0
                batch_erros = len(batch_itens)
                velocidade = len(batch_itens) / (tempo_batch / 60) if tempo_batch > 0 else 0
                feitos = min(batch_start + len(batch_itens), total)
                restantes = total - feitos
                if velocidade > 0:
                    minutos_restantes = restantes / velocidade
                    eta_dt = datetime.now() + timedelta(minutes=minutos_restantes)
                    previsao = eta_dt.strftime("%d/%m/%Y %H:%M:%S")
                else:
                    previsao = "—"
                batch_log.registrar_batch(
                    batch_num=batch_num,
                    itens_batch=len(batch_itens),
                    itens_corrigidos=batch_corrigidos,
                    itens_concluidos=batch_ok,
                    itens_erro=batch_erros,
                    inicio_batch=inicio_batch,
                    velocidade=velocidade,
                    previsao_termino=previsao,
                    tokens_gerados_minuto=0,
                    tokens_processados_minuto=0,
                )
            continue

        tempo_batch = time.time() - inicio_batch
        tempo_por_item = tempo_batch / len(batch_itens) if batch_itens else 0

        # Contadores do batch para o log
        batch_corrigidos = 0
        batch_ok = 0
        batch_erros = 0

        # Processa resultados do batch
        resultados_batch = []
        for i, item in enumerate(batch_itens):
            if i < len(resultados_vllm):
                res = resultados_vllm[i]
                prompt_tokens = res.get("prompt_tokens", 0) or len(tokenizer.encode(prompts[i]))
                completion_tokens = res.get("tokens", 0)
                finish_reason = res.get("finish_reason", "")

                resumo = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "finish_reason": finish_reason,
                    "model": config["modelo"]["caminho"],
                    "tempo": round(tempo_por_item, 3),
                }

                # Processa resposta conforme tipo_saida (str ou json)
                resposta_raw = res.get("output", "")
                resposta_final, erro_json = _processar_resposta(
                    resposta_raw, config
                )

                resultados_batch.append({
                    "chave": item["chave"],
                    "resumo": json.dumps(resumo, ensure_ascii=False),
                    "resposta": resposta_final,
                    "erro": erro_json,
                })

                tem_erro = bool(erro_json)

                if not tem_erro:
                    stats["processados_ok"] += 1
                    batch_ok += 1
                    # Era erro antes? Então foi corrigido
                    if item["chave"] in chaves_erro_anteriores:
                        stats["corrigidos"] += 1
                        batch_corrigidos += 1
                else:
                    stats["processados_erro"] += 1
                    batch_erros += 1
                    if item["chave"] in chaves_erro_anteriores:
                        stats["erros_mantidos"] += 1
                    else:
                        stats["erros_novos"] += 1
                stats["tokens_entrada"] += prompt_tokens
                stats["tokens_saida"] += completion_tokens
            else:
                # Resultado ausente (não deveria acontecer)
                resultados_batch.append({
                    "chave": item["chave"],
                    "resumo": json.dumps({
                        "model": config["modelo"]["caminho"],
                        "tempo": round(tempo_por_item, 3),
                    }, ensure_ascii=False),
                    "resposta": "",
                    "erro": "Resultado ausente do vLLM",
                })
                stats["processados_erro"] += 1
                batch_erros += 1
                if item["chave"] in chaves_erro_anteriores:
                    stats["erros_mantidos"] += 1
                else:
                    stats["erros_novos"] += 1

        # Salvamento incremental
        if eh_parquet:
            salvar_resultados_parquet(arquivo_saida, resultados_batch)
        else:
            for r in resultados_batch:
                resumo = json.loads(r["resumo"])
                salvar_resultado_pasta(
                    arquivo_saida, r["chave"], r["resposta"],
                    resumo, r["erro"]
                )

        tokens_batch = sum(
            json.loads(r["resumo"]).get("completion_tokens", 0)
            for r in resultados_batch
            if not r["erro"]
        )
        feitos = min(batch_start + len(batch_itens), total)
        print(
            f"   ✅ Batch {batch_num} concluído em {_formatar_tempo(tempo_batch)} "
            f"({tokens_batch} tokens gerados) — "
            f"progresso: {feitos}/{total} ({100 * feitos // total}%)"
        )

        # Log do batch
        if batch_log:
            elapsed_total = time.time() - inicio_total
            velocidade = feitos / (elapsed_total / 60) if elapsed_total > 0 else 0
            restantes = total - feitos
            if velocidade > 0:
                minutos_restantes = restantes / velocidade
                eta_dt = datetime.now() + timedelta(minutes=minutos_restantes)
                previsao = eta_dt.strftime("%d/%m/%Y %H:%M:%S")
            else:
                previsao = "—"
            # Calcula métricas de tokens do batch
            batch_tokens_saida = sum(
                json.loads(r["resumo"]).get("completion_tokens", 0)
                for r in resultados_batch
            )
            batch_tokens_entrada = sum(
                json.loads(r["resumo"]).get("prompt_tokens", 0)
                for r in resultados_batch
            )
            minutos_batch = tempo_batch / 60 if tempo_batch > 0 else 1
            tg_min = batch_tokens_saida / minutos_batch
            tp_min = (batch_tokens_entrada + batch_tokens_saida) / minutos_batch
            batch_log.registrar_batch(
                batch_num=batch_num,
                itens_batch=len(batch_itens),
                itens_corrigidos=batch_corrigidos,
                itens_concluidos=batch_ok,
                itens_erro=batch_erros,
                inicio_batch=inicio_batch,
                velocidade=velocidade,
                previsao_termino=previsao,
                tokens_gerados_minuto=tg_min,
                tokens_processados_minuto=tp_min,
            )

    stats["tempo_total"] = time.time() - inicio_total

    # Log final
    if batch_log:
        tempo_total = stats["tempo_total"]
        velocidade_final = total / (tempo_total / 60) if tempo_total > 0 else 0
        minutos_total = tempo_total / 60 if tempo_total > 0 else 1
        tg_min_final = stats["tokens_saida"] / minutos_total
        tp_min_final = (stats["tokens_entrada"] + stats["tokens_saida"]) / minutos_total
        batch_log.registrar_final(
            itens_processados=total,
            itens_corrigidos=stats["corrigidos"],
            itens_concluidos=stats["processados_ok"],
            itens_erro=stats["processados_erro"],
            velocidade=velocidade_final,
            tokens_gerados_minuto=tg_min_final,
            tokens_processados_minuto=tp_min_final,
        )

    return stats

# ---------------------------------------------------------------------------
# Processamento em Batch via API remota (or:, tg:, vl:, oa:)
# ---------------------------------------------------------------------------

def _chamar_api_item(
    item: Dict[str, str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Chama get_resposta para um item individual e retorna resultado padronizado.

    Args:
        item: dict com {chave, texto}
        config: configuração completa

    Returns:
        Dict com {chave, resumo (json string), resposta, erro}
    """
    from util_openai import get_resposta

    cfg_geracao = config["geracao"]
    caminho_modelo = config["modelo"]["caminho"]
    system_prompt = config["entrada"].get("system_prompt", "")
    tipo_saida = config["saida"].get("tipo_saida", "str")

    # Monta o conteúdo do prompt (template + texto)
    conteudo = montar_prompt(item["texto"], config)

    # Monta messages no formato da API
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": conteudo})

    # Parâmetros da chamada
    max_tokens = cfg_geracao.get("max_tokens", 2048)
    temperature = cfg_geracao.get("temperature", 0.01)
    think = cfg_geracao.get("think", None)
    as_json = (tipo_saida == "json")

    inicio_item = time.time()
    try:
        resultado_api = get_resposta(
            prompt=messages,
            modelo=caminho_modelo,
            think=think,
            as_json=as_json,
            temperature=temperature,
            max_tokens=max_tokens,
            silencioso=True,
        )
    except Exception as e:
        tempo_item = time.time() - inicio_item
        return {
            "chave": item["chave"],
            "resumo": json.dumps({
                "model": caminho_modelo,
                "tempo": round(tempo_item, 3),
            }, ensure_ascii=False),
            "resposta": "",
            "erro": f"{type(e).__name__}: {e}",
        }

    tempo_item = time.time() - inicio_item

    # Extrai informações do retorno padronizado de get_resposta
    erro_api = resultado_api.get("erro", "")
    usage = resultado_api.get("usage", {})
    model_usado = resultado_api.get("model", caminho_modelo)

    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0
    finish_reason = usage.get("finished_reason", "")

    resumo = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "finish_reason": finish_reason,
        "model": model_usado,
        "tempo": round(tempo_item, 3),
    }

    if erro_api:
        # Erro da API — retorna sem resposta
        return {
            "chave": item["chave"],
            "resumo": json.dumps(resumo, ensure_ascii=False),
            "resposta": str(resultado_api.get("resposta", "")),
            "erro": erro_api,
        }

    # Processa resposta
    resposta_raw = resultado_api.get("resposta", "")
    if as_json:
        # Quando as_json=True, get_resposta já fez o parse — serializa de volta
        if isinstance(resposta_raw, (dict, list)):
            resposta_final = json.dumps(resposta_raw, ensure_ascii=False, indent=2)
            erro_json = ""
        else:
            # Fallback: a API retornou texto — aplica pós-processamento
            resposta_final, erro_json = _processar_resposta(str(resposta_raw), config)
    else:
        resposta_final, erro_json = _processar_resposta(str(resposta_raw), config)

    return {
        "chave": item["chave"],
        "resumo": json.dumps(resumo, ensure_ascii=False),
        "resposta": resposta_final,
        "erro": erro_json,
    }


def processar_batch_api(
    itens_pendentes: List[Dict[str, str]],
    config: Dict[str, Any],
    batch_log: Optional["BatchLog"] = None,
    chaves_erro_anteriores: Optional[set] = None,
) -> Dict[str, Any]:
    """Processa itens pendentes via API remota (or:, tg:, vl:, oa:).

    Usa ThreadPoolExecutor com batch_size como limite de concorrência.
    Compatível com o mesmo sistema de salvamento, log e retomada do modo vLLM.

    Args:
        itens_pendentes: lista de {chave, texto}
        config: configuração completa
        batch_log: instância de BatchLog para registrar o log incremental
        chaves_erro_anteriores: chaves que tinham erro antes desta execução

    Returns:
        Dicionário com estatísticas do processamento
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cfg_geracao = config["geracao"]
    batch_size = cfg_geracao.get("batch_size", 64)

    eh_parquet = _saida_eh_parquet(config)
    arquivo_saida = config["saida"]["arquivo"]

    if chaves_erro_anteriores is None:
        chaves_erro_anteriores = set()

    total = len(itens_pendentes)
    stats = {
        "processados_ok": 0,
        "processados_erro": 0,
        "tokens_entrada": 0,
        "tokens_saida": 0,
        "corrigidos": 0,
        "erros_mantidos": 0,
        "erros_novos": 0,
    }

    nome_modelo = extrair_nome_modelo_api(config["modelo"]["caminho"])
    print(f"\n🌐 Processando {total} item(ns) via API remota ({nome_modelo})")
    print(f"   Concorrência: {batch_size} threads")
    inicio_total = time.time()

    for batch_start in range(0, total, batch_size):
        batch_itens = itens_pendentes[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        # Info do batch
        eta = _formatar_eta(inicio_total, batch_start, total)
        print(
            f"\n📦 Batch {batch_num}/{total_batches} "
            f"({len(batch_itens)} itens) {eta}"
        )

        inicio_batch = time.time()

        # Contadores do batch para o log
        batch_corrigidos = 0
        batch_ok = 0
        batch_erros = 0

        # Loga os prompts antes de enviar para as threads
        if batch_log and getattr(batch_log, "_prompts_registrados", 0) < 5:
            system_prompt = config["entrada"].get("system_prompt", "")
            for item in batch_itens:
                if batch_log._prompts_registrados >= 5:
                    break
                conteudo = montar_prompt(item["texto"], config)
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": conteudo})
                prompt_str = json.dumps(messages, ensure_ascii=False, indent=2)
                batch_log.registrar_prompt(item["chave"], prompt_str)

        # Executa chamadas em paralelo usando ThreadPoolExecutor
        resultados_batch = [None] * len(batch_itens)
        with ThreadPoolExecutor(max_workers=min(batch_size, len(batch_itens))) as executor:
            future_to_idx = {
                executor.submit(_chamar_api_item, item, config): i
                for i, item in enumerate(batch_itens)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    resultados_batch[idx] = future.result()
                except Exception as e:
                    # Erro inesperado no future — não deveria acontecer
                    item = batch_itens[idx]
                    resultados_batch[idx] = {
                        "chave": item["chave"],
                        "resumo": json.dumps({
                            "model": config["modelo"]["caminho"],
                            "tempo": 0,
                        }, ensure_ascii=False),
                        "resposta": "",
                        "erro": f"Erro inesperado: {type(e).__name__}: {e}",
                    }

        tempo_batch = time.time() - inicio_batch

        # Classifica resultados e atualiza stats
        for i, resultado in enumerate(resultados_batch):
            item = batch_itens[i]
            tem_erro = bool(resultado["erro"])

            # Extrai tokens do resumo para stats
            try:
                resumo_dict = json.loads(resultado["resumo"])
                prompt_tokens = resumo_dict.get("prompt_tokens", 0) or 0
                completion_tokens = resumo_dict.get("completion_tokens", 0) or 0
            except Exception:
                prompt_tokens = 0
                completion_tokens = 0

            if not tem_erro:
                stats["processados_ok"] += 1
                batch_ok += 1
                if item["chave"] in chaves_erro_anteriores:
                    stats["corrigidos"] += 1
                    batch_corrigidos += 1
            else:
                stats["processados_erro"] += 1
                batch_erros += 1
                if item["chave"] in chaves_erro_anteriores:
                    stats["erros_mantidos"] += 1
                else:
                    stats["erros_novos"] += 1

            stats["tokens_entrada"] += prompt_tokens
            stats["tokens_saida"] += completion_tokens

        # Salvamento incremental
        if eh_parquet:
            salvar_resultados_parquet(arquivo_saida, resultados_batch)
        else:
            for r in resultados_batch:
                resumo = json.loads(r["resumo"])
                salvar_resultado_pasta(
                    arquivo_saida, r["chave"], r["resposta"],
                    resumo, r["erro"]
                )

        tokens_batch = sum(
            json.loads(r["resumo"]).get("completion_tokens", 0)
            for r in resultados_batch
            if not r["erro"]
        )
        feitos = min(batch_start + len(batch_itens), total)
        print(
            f"   ✅ Batch {batch_num} concluído em {_formatar_tempo(tempo_batch)} "
            f"({tokens_batch} tokens gerados) — "
            f"progresso: {feitos}/{total} ({100 * feitos // total}%)"
        )

        # Log do batch
        if batch_log:
            elapsed_total = time.time() - inicio_total
            velocidade = feitos / (elapsed_total / 60) if elapsed_total > 0 else 0
            restantes = total - feitos
            if velocidade > 0:
                minutos_restantes = restantes / velocidade
                eta_dt = datetime.now() + timedelta(minutes=minutos_restantes)
                previsao = eta_dt.strftime("%d/%m/%Y %H:%M:%S")
            else:
                previsao = "—"
            batch_tokens_saida = sum(
                json.loads(r["resumo"]).get("completion_tokens", 0)
                for r in resultados_batch
            )
            batch_tokens_entrada = sum(
                json.loads(r["resumo"]).get("prompt_tokens", 0)
                for r in resultados_batch
            )
            minutos_batch = tempo_batch / 60 if tempo_batch > 0 else 1
            tg_min = batch_tokens_saida / minutos_batch
            tp_min = (batch_tokens_entrada + batch_tokens_saida) / minutos_batch
            batch_log.registrar_batch(
                batch_num=batch_num,
                itens_batch=len(batch_itens),
                itens_corrigidos=batch_corrigidos,
                itens_concluidos=batch_ok,
                itens_erro=batch_erros,
                inicio_batch=inicio_batch,
                velocidade=velocidade,
                previsao_termino=previsao,
                tokens_gerados_minuto=tg_min,
                tokens_processados_minuto=tp_min,
            )

    stats["tempo_total"] = time.time() - inicio_total

    # Log final
    if batch_log:
        tempo_total = stats["tempo_total"]
        velocidade_final = total / (tempo_total / 60) if tempo_total > 0 else 0
        minutos_total = tempo_total / 60 if tempo_total > 0 else 1
        tg_min_final = stats["tokens_saida"] / minutos_total
        tp_min_final = (stats["tokens_entrada"] + stats["tokens_saida"]) / minutos_total
        batch_log.registrar_final(
            itens_processados=total,
            itens_corrigidos=stats["corrigidos"],
            itens_concluidos=stats["processados_ok"],
            itens_erro=stats["processados_erro"],
            velocidade=velocidade_final,
            tokens_gerados_minuto=tg_min_final,
            tokens_processados_minuto=tp_min_final,
        )

    return stats


# ---------------------------------------------------------------------------
# Inicialização do Engine vLLM
# ---------------------------------------------------------------------------

def inicializar_engine(config: Dict[str, Any]):
    """Inicializa VLLMInferenceEngine e retorna (engine, tokenizer).

    Args:
        config: configuração completa

    Returns:
        Tupla (engine, tokenizer)
    """
    try:
        from treinar_vllm_inference import (
            VLLMInferenceEngine,
            VLLMConfig,
            VLLM_AVAILABLE,
        )
    except ImportError:
        print("❌ Módulo treinar_vllm_inference não encontrado!")
        print("   Certifique-se de que treinar_vllm_inference.py está na pasta src/")
        sys.exit(1)

    if not VLLM_AVAILABLE:
        print("❌ vLLM não está instalado! Instale com: pip install vllm")
        sys.exit(1)

    cfg_vllm = config["vllm"].copy()
    cfg_modelo = config["modelo"]

    # Extrai os parâmetros já mapeados
    gpu_mem = float(cfg_vllm.pop("gpu_memory_utilization", 0.9))
    tp_size = int(cfg_vllm.pop("tensor_parallel_size", 1))
    dtype = str(cfg_vllm.pop("dtype", "auto"))
    max_len = cfg_vllm.pop("max_model_len", None)
    if max_len is not None:
        max_len = int(max_len)
    eager = bool(cfg_vllm.pop("enforce_eager", False))
    quantization = cfg_vllm.pop("quantization", None)
    load_format = cfg_vllm.pop("load_format", "auto")

    vllm_config = VLLMConfig(
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=tp_size,
        dtype=dtype,
        max_model_len=max_len,
        enforce_eager=eager,
        quantization=quantization,
        load_format=load_format,
        extra_kwargs=cfg_vllm
    )

    lora_path = cfg_modelo.get("lora", "") or None

    print("\n🚀 Inicializando vLLM...")
    print(f"   Modelo: {cfg_modelo['caminho']}")
    if lora_path:
        print(f"   LoRA: {lora_path}")
    print(f"   Max Model Len: {max_len}")
    print(f"   GPU Memory: {gpu_mem * 100:.0f}%")
    print(f"   Tensor Parallel: {tp_size} GPU(s)")
    if quantization:
        print(f"   Quantização: {quantization}")
    if cfg_vllm:
        print(f"   vLLM Extra Kwargs: {list(cfg_vllm.keys())}")

    engine = VLLMInferenceEngine(
        model_path=cfg_modelo["caminho"],
        config=vllm_config,
        lora_path=lora_path,
    )

    tokenizer = engine.llm.get_tokenizer()
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    return engine, tokenizer


# ---------------------------------------------------------------------------
# Shutdown do Engine
# ---------------------------------------------------------------------------

def finalizar_engine(engine) -> None:
    """Faz shutdown limpo do engine vLLM.

    Segue o padrão do UtilPredicaoVLLM para evitar warnings do NCCL.
    """
    import gc

    if engine is None:
        return

    try:
        llm = getattr(engine, "llm", None)
        if llm is not None:
            llm_engine = getattr(llm, "llm_engine", None)
            engine_core = getattr(llm_engine, "engine_core", None)
            if engine_core is not None and hasattr(engine_core, "shutdown"):
                try:
                    engine_core.shutdown(timeout=15)
                except TypeError as err:
                    if "timeout" in str(err):
                        engine_core.shutdown()
                    else:
                        raise
                print("✅ vLLM EngineCore shutdown concluído.")
            elif hasattr(llm, "shutdown"):
                llm.shutdown()
            elif hasattr(llm, "close"):
                llm.close()
    except Exception as e:
        print(f"⚠️  Erro no shutdown do vLLM: {e}")

    del engine
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    # Destrói process group se existir
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    print("🧹 Engine vLLM liberado e memória GPU limpa.")


# ---------------------------------------------------------------------------
# Criação de YAML de Exemplo
# ---------------------------------------------------------------------------

def _criar_yaml_exemplo(caminho: str) -> None:
    """Cria arquivo YAML de exemplo com comentários explicativos."""
    nome = os.path.basename(caminho)
    conteudo = _YAML_EXEMPLO.replace("{nome_arquivo}", nome)

    pasta = os.path.dirname(caminho)
    if pasta:
        os.makedirs(pasta, exist_ok=True)

    with open(caminho, "w", encoding="utf-8") as f:
        f.write(conteudo)

    print(f"✅ Arquivo de configuração criado: {caminho}")
    print("   Edite os parâmetros e execute novamente.")


# ---------------------------------------------------------------------------
# Gráficos e Resumo
# ---------------------------------------------------------------------------

def _gerar_resumo_e_graficos(config: Dict[str, Any], stats: Dict[str, Any]) -> None:
    """Gera resumo final e gráficos das predições (semelhante ao treinar_unsloth.py)."""
    arquivo_saida = config["saida"]["arquivo"]
    eh_parquet = _saida_eh_parquet(config)
    
    if eh_parquet:
        pasta_saida = os.path.dirname(os.path.abspath(arquivo_saida))
        prefixo = os.path.splitext(os.path.basename(arquivo_saida))[0] + "_"
    else:
        pasta_saida = os.path.abspath(arquivo_saida)
        prefixo = ""
        
    if not os.path.isdir(pasta_saida):
        os.makedirs(pasta_saida, exist_ok=True)
        
    registros = []
    
    if eh_parquet:
        try:
            import pandas as pd
            if os.path.exists(arquivo_saida):
                df = pd.read_parquet(arquivo_saida)
                for _, row in df.iterrows():
                    try:
                        r_json = json.loads(row["resumo"])
                        registros.append({
                            'id': str(row["chave"]),
                            'input_tokens': int(r_json.get("prompt_tokens", 0)),
                            'output_tokens': int(r_json.get("completion_tokens", 0)),
                            'time_s': float(r_json.get("tempo", 0)),
                        })
                    except Exception:
                        pass
        except Exception as e:
            print(f"   ⚠️  Erro ao ler parquet para gerar estatísticas: {e}")
    else:
        for nome in sorted(os.listdir(pasta_saida)):
            if not nome.endswith('.json') or nome in ('resumo.json', 'resumo_geral.json'):
                continue
            try:
                caminho = os.path.join(pasta_saida, nome)
                with open(caminho, 'r', encoding='utf-8') as f:
                    dados = json.load(f)
                chave_str = str(dados.get('chave', nome.replace('.json', '')))
                registros.append({
                    'id': chave_str,
                    'input_tokens': int(dados.get('prompt_tokens', dados.get('input_tokens', 0))),
                    'output_tokens': int(dados.get('completion_tokens', dados.get('output_tokens', 0))),
                    'time_s': float(dados.get('tempo', dados.get('time_s', 0))),
                })
            except Exception:
                continue

    if not registros:
        print("   ⚠️  Nenhum registro válido encontrado para gerar gráficos e CSV.")
        return

    print("\n   --- Gerando artefatos de resumo ---")

    # --- CSV ---
    import csv
    csv_path = os.path.join(pasta_saida, f'{prefixo}predicoes.csv')
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'input_tokens', 'output_tokens', 'time_s'])
            writer.writeheader()
            writer.writerows(registros)
        print(f"   📄 CSV gerado: {csv_path}")
    except Exception as e:
        print(f"   ⚠️  Erro ao gerar CSV: {e}")

    # --- Resumo JSON ---
    resumo_geral = {
        'data_geracao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'modelo_usado': config["modelo"]["caminho"],
        'formato_saida': config["saida"].get("tipo_saida", "str"),
        'total_registros': len(registros),
        'processados_ok': stats.get('processados_ok', 0),
        'processados_erro': stats.get('processados_erro', 0),
        'input_tokens_total': sum(r['input_tokens'] for r in registros),
        'output_tokens_total': sum(r['output_tokens'] for r in registros),
        'tempo_total_s': round(stats.get('tempo_total', 0), 2),
    }
    resumo_file = os.path.join(pasta_saida, f'{prefixo}resumo.json')
    try:
        with open(resumo_file, 'w', encoding='utf-8') as f:
            json.dump(resumo_geral, f, ensure_ascii=False, indent=2)
        print(f"   📄 JSON gerado: {resumo_file}")
    except Exception as e:
        print(f"   ⚠️  Erro ao gerar resumo JSON: {e}")

    # --- Gráficos (boxplots) ---
    try:
        from util_graficos import UtilGraficos
        inputs  = [r['input_tokens'] for r in registros]
        outputs = [r['output_tokens'] for r in registros]
        totais  = [r['input_tokens'] + r['output_tokens'] for r in registros]
        tempos  = [r['time_s'] for r in registros]

        nota_maximos = (f"max(entrada)={max(inputs):,}  "
                        f"max(saída)={max(outputs):,}  "
                        f"max(total)={max(totais):,}").replace(',', '.')
        tokens_png = os.path.join(pasta_saida, f'{prefixo}predicoes_tokens.png')
        UtilGraficos.gerar_boxplot(
            dados={'Entrada': inputs, 'Saída': outputs, 'Total': totais},
            titulo='Distribuição de Tokens — Predições vLLM Batch',
            ylabel='Tokens',
            arquivo_saida=tokens_png,
            nota=nota_maximos,
        )
        print(f"   📊 Boxplot tokens: {tokens_png}")

        t_sorted = sorted(tempos)
        q1_tempo = t_sorted[len(t_sorted) // 4] if len(t_sorted) >= 4 else t_sorted[0]
        nota_tempo = (f"min={min(tempos):.1f}s  "
                      f"Q1={q1_tempo:.1f}s (75% acima)  "
                      f"max={max(tempos):.1f}s")
        tempo_png = os.path.join(pasta_saida, f'{prefixo}predicoes_tempo.png')
        UtilGraficos.gerar_boxplot(
            dados={'Tempo (s)': tempos},
            titulo='Distribuição de Tempo de Geração — Predições vLLM Batch',
            ylabel='Segundos',
            arquivo_saida=tempo_png,
            nota=nota_tempo,
        )
        print(f"   ⏱️  Boxplot tempo: {tempo_png}")
    except ImportError as e:
        print(f"   ⚠️  util_graficos não encontrado, gráficos não gerados. Erro original: {e}")
    except Exception as e:
        print(f"   ⚠️  Não foi possível gerar gráficos: {e}")


# ---------------------------------------------------------------------------
# CLI Principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada da CLI."""
    parser = argparse.ArgumentParser(
        description="Inferência em lote com vLLM configurada por YAML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Exemplos:
  %(prog)s --config config_batch.yaml       # Executa inferência em lote
  %(prog)s --config novo.yaml               # Cria YAML de exemplo (se não existir)

Entrada: arquivo .parquet ou pasta com .txt
Saída: arquivo .parquet ou pasta com .txt/.json
Retomada: itens já processados com sucesso são ignorados automaticamente.
""",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Arquivo YAML de configuração (se não existir, oferece criar exemplo)",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Apenas exibe o prompt formatado do primeiro item pendente e sai",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Visualiza os resultados já processados na saída e sai",
    )

    args = parser.parse_args()
    config_path = args.config

    # --- YAML não existe: oferece criar exemplo ---
    if not os.path.isfile(config_path):
        print(f"\n⚠️  Arquivo não encontrado: {config_path}")
        resposta = input("   Deseja criar um arquivo de exemplo? (s/n): ").strip().lower()
        if resposta in ("s", "sim", "y", "yes"):
            _criar_yaml_exemplo(config_path)
        else:
            print("   Operação cancelada.")
        sys.exit(0)

    # --- Carrega configuração ---
    print(f"\n📖 Carregando configuração: {config_path}")
    try:
        config = carregar_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Erro na configuração: {e}")
        sys.exit(1)

    # --- Validação pré-execução ---
    erros_validacao = validar_config(config)
    if erros_validacao:
        print(f"\n❌ {len(erros_validacao)} erro(s) de validação encontrado(s):")
        for i, erro in enumerate(erros_validacao, 1):
            print(f"   {i}. {erro}")
        print("\n   Corrija os erros acima e execute novamente.")
        sys.exit(1)
    print("✅ Configuração validada com sucesso.")

    if args.view:
        try:
            from util_vllm_batch_view import visualizar_saida_config
            print("\n" + "="*80)
            print("👁️  VISUALIZAÇÃO DE RESULTADOS")
            print("="*80)
            visualizar_saida_config(config)
            sys.exit(0)
        except ImportError as e:
            print(f"\n❌ Erro ao importar util_vllm_batch_view: {e}")
            sys.exit(1)

    # --- Lock de Execução ---
    lock_file_path = os.path.abspath(config_path) + ".lock"
    _lock_fd = None
    try:
        import fcntl
        _lock_fd = open(lock_file_path, 'w')
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            print(f"\n❌ Já existe uma instância em execução para '{config_path}'.")
            print(f"   (Lock file: {lock_file_path} está em uso)")
            sys.exit(1)
    except ImportError:
        # Fallback para Windows
        try:
            import msvcrt
            _lock_fd = open(lock_file_path, 'w')
            try:
                msvcrt.locking(_lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                print(f"\n❌ Já existe uma instância em execução para '{config_path}'.")
                print(f"   (Lock file: {lock_file_path} está em uso)")
                sys.exit(1)
        except ImportError:
            pass # Sem suporte a lock

    # --- Carrega entrada ---
    print(f"\n📂 Carregando entrada: {config['entrada']['arquivo']}")
    try:
        itens = carregar_entrada(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Erro ao carregar entrada: {e}")
        sys.exit(1)

    if not itens:
        print("⚠️  Nenhum item encontrado na entrada.")
        sys.exit(0)

    print(f"   📊 {len(itens)} item(ns) carregado(s)")

    # Aplica limite max_itens (se configurado) logo após a carga
    max_itens = config["geracao"].get("max_itens", 0)
    if max_itens > 0 and len(itens) > max_itens:
        print(f"   ⚠️  Limitando dados de entrada aos {max_itens} primeiros itens (configuração max_itens)")
        itens = itens[:max_itens]

    # --- Verifica saída existente (retomada) ---
    print(f"\n🔍 Verificando saída existente: {config['saida']['arquivo']}")
    chaves_ok = carregar_saida_existente(config)
    chaves_erro = carregar_chaves_com_erro(config)

    # Filtra pendentes
    itens_pendentes = [item for item in itens if item["chave"] not in chaves_ok]

    if not itens_pendentes:
        print(f"✅ Todos os {len(itens)} itens já foram processados com sucesso!")
        print("   Gerando gráficos e resumos para os dados existentes...")
        stats_vazio = {
            'processados_ok': 0,
            'processados_erro': 0,
            'tokens_entrada': 0,
            'tokens_saida': 0,
            'tempo_total': 0,
        }
        _gerar_resumo_e_graficos(config, stats_vazio)
        sys.exit(0)

    skip_count = len(itens) - len(itens_pendentes)
    if skip_count > 0:
        print(f"   ⏭️  {skip_count} item(ns) já processado(s) — ignorando")
    print(f"   📝 {len(itens_pendentes)} item(ns) pendente(s)")

    if args.prompt:
        print("\n" + "="*80)
        print("🛠️  TESTE DE PROMPT (PRIMEIRO ITEM PENDENTE)")
        print("="*80)
        item = itens_pendentes[0]
        print(f"Chave: {item['chave']}")
        conteudo = montar_prompt(item["texto"], config)
        print("\n--- Conteúdo preenchido (truncado se muito grande) ---")
        if len(conteudo) >1000:
            _tamanho = min(int(len(conteudo) /2), 500)
            print(conteudo[:_tamanho], '[..]', conteudo[-_tamanho:])
        else:
            print(conteudo)

        caminho_modelo = config["modelo"]["caminho"]
        if eh_modelo_api_remota(caminho_modelo):
            # Modo API remota: exibe messages no formato JSON
            print("\n--- Messages (formato API remota) ---")
            system_prompt = config["entrada"].get("system_prompt", "")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": conteudo})
            print(f"Modelo: {caminho_modelo}")
            think = config["geracao"].get("think")
            if think:
                print(f"Think: {think}")
            tipo_saida = config["saida"].get("tipo_saida", "str")
            print(f"as_json: {tipo_saida == 'json'}")
            print(f"\n--- Messages JSON ({len(messages)} mensagem(ns)) ---")
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if len(content) > 2000:
                    _tamanho = min(int(len(content) / 2), 1000)
                    content = content[:_tamanho] + ' [..]' + content[-_tamanho:]
                print(f"  [{role}]: {content[:200]}{'...' if len(content)>200 else ''}")
        else:
            # Modo vLLM local: aplica chat template
            print("\n--- Aplicando Chat Template ---")
            system_prompt = config["entrada"].get("system_prompt", "")
            if system_prompt:
                print(f"System prompt: {system_prompt[:200]}...")
            else:
                print("(sem system prompt — chat template aplicado apenas com role=user)")
            print("Carregando tokenizer do HuggingFace...")
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config["modelo"]["caminho"])
                prompt_final = formatar_prompt_final(conteudo, config, tokenizer)
                print(f"\n--- Prompt Final ({len(prompt_final)} chars, enviado ao modelo) ---")
                if len(prompt_final) > 2000:
                    _tamanho = min(int(len(prompt_final) / 2), 1000)
                    print(prompt_final[:_tamanho], '[..]', prompt_final[-_tamanho:])
                else:
                    print(prompt_final)
            except ImportError:
                print("❌ Biblioteca 'transformers' não encontrada para o teste.")
            except Exception as e:
                print(f"❌ Erro ao carregar tokenizer: {e}")
        print("="*80)
        sys.exit(0)

    # --- Exibe configuração ---
    caminho_modelo = config["modelo"]["caminho"]
    modo_api = eh_modelo_api_remota(caminho_modelo)
    cfg_geracao = config["geracao"]
    print(f"\n⚙️  Configuração de geração:")
    if modo_api:
        _, desc_api = validar_modelo_api(caminho_modelo)
        print(f"   🌐 Modo: API remota — {desc_api}")
    else:
        print(f"   🖥️  Modo: vLLM local")
    print(f"   Max tokens: {cfg_geracao['max_tokens']}")
    print(f"   Temperature: {cfg_geracao['temperature']}")
    if not modo_api:
        print(f"   Top-k: {cfg_geracao['top_k']}")
        print(f"   Top-p: {cfg_geracao['top_p']}")
    print(f"   Batch size: {cfg_geracao.get('batch_size', 64)}")
    if cfg_geracao.get('max_itens', 0) > 0:
        print(f"   Max itens: {cfg_geracao['max_itens']}")
    if modo_api and cfg_geracao.get('think'):
        print(f"   Think: {cfg_geracao['think']}")

    saida_tipo = "parquet" if _saida_eh_parquet(config) else "pasta"
    print(f"   Saída: {config['saida']['arquivo']} ({saida_tipo})")

    # --- Inicializa log de processamento ---
    batch_log = BatchLog(config)
    # Calcula itens novos: pendentes que não tinham erro antes (são realmente novos)
    itens_novos = len([it for it in itens_pendentes if it["chave"] not in chaves_erro])
    batch_log.registrar_inicio(
        itens_novos=itens_novos,
        itens_erro=len(chaves_erro),
        itens_concluidos=len(chaves_ok),
    )

    # --- Processamento: API remota ou vLLM local ---
    if modo_api:
        # Modo API remota — não precisa de engine vLLM
        try:
            stats = processar_batch_api(
                itens_pendentes, config,
                batch_log=batch_log,
                chaves_erro_anteriores=chaves_erro,
            )

            # --- Resumo final ---
            print("\n" + "=" * 70)
            print("📊 RESUMO FINAL")
            print("=" * 70)
            print(f"   ✅ Processados com sucesso: {stats['processados_ok']}")
            if stats.get('corrigidos', 0) > 0:
                print(f"   🔧 Erros corrigidos: {stats['corrigidos']}")
            if stats["processados_erro"] > 0:
                print(f"   ❌ Processados com erro: {stats['processados_erro']}")
            if stats.get('erros_mantidos', 0) > 0:
                print(f"   🔁 Erros mantidos: {stats['erros_mantidos']}")
            if stats.get('erros_novos', 0) > 0:
                print(f"   🆕 Erros novos: {stats['erros_novos']}")
            print(f"   🔢 Tokens entrada: {stats['tokens_entrada']:,}")
            print(f"   🔢 Tokens saída: {stats['tokens_saida']:,}")
            print(f"   ⏱️  Tempo total: {_formatar_tempo(stats['tempo_total'])}")
            if stats["processados_ok"] > 0 and stats["tempo_total"] > 0:
                throughput = stats["tokens_saida"] / stats["tempo_total"]
                print(f"   🚀 Throughput: {throughput:.0f} tokens/s")
            print(f"   💾 Saída: {config['saida']['arquivo']}")

            # Gera resumo final e gráficos
            _gerar_resumo_e_graficos(config, stats)

            print("=" * 70)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrompido pelo usuário.")
            print("   Os resultados parciais foram salvos.")
            print(f"   Execute novamente para continuar de onde parou.")
        except Exception as e:
            print(f"\n❌ Erro: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Modo vLLM local
        engine = None
        try:
            engine, tokenizer = inicializar_engine(config)

            # --- Processa ---
            stats = processar_batch(
                engine, tokenizer, itens_pendentes, config,
                batch_log=batch_log,
                chaves_erro_anteriores=chaves_erro,
            )

            # --- Resumo final ---
            print("\n" + "=" * 70)
            print("📊 RESUMO FINAL")
            print("=" * 70)
            print(f"   ✅ Processados com sucesso: {stats['processados_ok']}")
            if stats.get('corrigidos', 0) > 0:
                print(f"   🔧 Erros corrigidos: {stats['corrigidos']}")
            if stats["processados_erro"] > 0:
                print(f"   ❌ Processados com erro: {stats['processados_erro']}")
            if stats.get('erros_mantidos', 0) > 0:
                print(f"   🔁 Erros mantidos: {stats['erros_mantidos']}")
            if stats.get('erros_novos', 0) > 0:
                print(f"   🆕 Erros novos: {stats['erros_novos']}")
            print(f"   🔢 Tokens entrada: {stats['tokens_entrada']:,}")
            print(f"   🔢 Tokens saída: {stats['tokens_saida']:,}")
            print(f"   ⏱️  Tempo total: {_formatar_tempo(stats['tempo_total'])}")
            if stats["processados_ok"] > 0:
                throughput = stats["tokens_saida"] / stats["tempo_total"]
                print(f"   🚀 Throughput: {throughput:.0f} tokens/s")
            print(f"   💾 Saída: {config['saida']['arquivo']}")

            # Gera resumo final e gráficos
            _gerar_resumo_e_graficos(config, stats)

            print("=" * 70)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrompido pelo usuário.")
            print("   Os resultados parciais foram salvos.")
            print(f"   Execute novamente para continuar de onde parou.")
        except Exception as e:
            print(f"\n❌ Erro: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if engine is not None:
                print("\n🛑 Finalizando engine vLLM...")
                finalizar_engine(engine)


if __name__ == "__main__":
    # Carrega .env se disponível
    try:
        from util import UtilEnv
        UtilEnv.carregar_env(pastas=["./", "../", "../src/"])
    except Exception:
        pass

    main()