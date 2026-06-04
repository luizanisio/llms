#!/usr/bin/env python3

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Inferência em lote usando vLLM com configuração via YAML.

Uso:
    python util_vllm_batch.py --config config_batch.yaml

Se o arquivo YAML não existir, o programa pergunta se deseja criá-lo
com um exemplo comentado e explicativo.

Entrada: arquivo parquet (com colunas chave + texto) ou pasta com arquivos .txt
Saída: arquivo parquet (com colunas chave, resumo, resposta, erro) ou pasta com .txt/.json

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


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_COLUNAS_SAIDA_PARQUET = ["chave", "resumo", "resposta", "erro"]

_YAML_EXEMPLO = """\
# ==========================================================================
# Configuração para inferência em lote com vLLM
# Executar com: python util_vllm_batch.py --config {nome_arquivo}
# ==========================================================================

# --- Modelo ---
# caminho: caminho para modelo HuggingFace (local ou hub)
# lora: caminho para adaptador LoRA treinado (opcional, deixe vazio se não usar)
modelo:
  caminho: "/caminho/para/modelo"
  lora: ""

# --- Configuração do vLLM ---
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
# max_itens: limite máximo de itens a processar (útil para testes). Opcional.
geracao:
  max_tokens: 2048
  temperature: 0.01
  top_k: 2
  top_p: 0.9
  # batch_size: 64
  # max_itens: 10

# --- Entrada ---
# arquivo: caminho para arquivo .parquet OU pasta com arquivos .txt
#   Se parquet: usa campo_chave como ID e campo_texto como conteúdo
#   Se pasta: cada arquivo .txt é um item (nome sem extensão = chave)
# campo_chave: nome da coluna com o ID (apenas para parquet). Padrão: "id"
# campo_texto: nome da coluna com o texto (apenas para parquet). Padrão: "texto"
# prompt_template: arquivo .txt com template do prompt (opcional).
#   Se informado, o texto do campo_texto substitui o placeholder variavel_texto.
#   Se vazio, o texto é usado diretamente como conteúdo do prompt.
# variavel_texto: placeholder no template a ser substituído pelo texto. Padrão: "<--TEXTO-->"
# system_prompt: system prompt para modelos instruct (opcional).
#   Se informado, o chat template do modelo é aplicado automaticamente.
#   Se vazio, o texto montado é enviado diretamente ao modelo (sem chat template).
entrada:
  arquivo: "./entrada.parquet"
  campo_chave: "id"
  campo_texto: "texto"
  prompt_template: ""
  variavel_texto: "<--TEXTO-->"
  system_prompt: ""

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

def _resolver_caminho(caminho: str, base_dir: str) -> str:
    """Resolve caminho relativo em relação ao diretório base."""
    if not caminho:
        return caminho
    if os.path.isabs(caminho):
        return caminho
    return os.path.normpath(os.path.join(base_dir, caminho))


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

    # --- Modelo (obrigatório) ---
    modelo = config.get("modelo", {}) or {}
    caminho_modelo = modelo.get("caminho", "")
    if not caminho_modelo:
        raise ValueError("modelo.caminho é obrigatório no YAML")
    modelo["caminho"] = _resolver_caminho(caminho_modelo, base_dir)
    lora = modelo.get("lora", "")
    if lora:
        modelo["lora"] = _resolver_caminho(lora, base_dir)
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
    config["geracao"] = geracao

    # --- Entrada (obrigatório) ---
    entrada = config.get("entrada", {}) or {}
    arquivo_entrada = entrada.get("arquivo", "")
    if not arquivo_entrada:
        raise ValueError("entrada.arquivo é obrigatório no YAML")
    entrada["arquivo"] = _resolver_caminho(arquivo_entrada, base_dir)
    entrada.setdefault("campo_chave", "id")
    entrada.setdefault("campo_texto", "texto")
    prompt_tpl = entrada.get("prompt_template", "")
    if prompt_tpl:
        entrada["prompt_template"] = _resolver_caminho(prompt_tpl, base_dir)
    entrada.setdefault("variavel_texto", "<--TEXTO-->")
    entrada.setdefault("system_prompt", "")
    config["entrada"] = entrada

    # --- Saída (obrigatório) ---
    saida = config.get("saida", {}) or {}
    arquivo_saida = saida.get("arquivo", "")
    if not arquivo_saida:
        raise ValueError("saida.arquivo é obrigatório no YAML")
    saida["arquivo"] = _resolver_caminho(arquivo_saida, base_dir)
    # tipo_saida: "str" (padrão) ou "json"/"dict"
    tipo_saida = str(saida.get("tipo_saida", "str")).strip().lower()
    if tipo_saida in ("json", "dict"):
        tipo_saida = "json"
    else:
        tipo_saida = "str"
    saida["tipo_saida"] = tipo_saida
    config["saida"] = saida

    return config


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

    if arquivo.lower().endswith(".parquet"):
        return _carregar_entrada_parquet(arquivo, cfg_entrada)
    elif os.path.isdir(arquivo):
        return _carregar_entrada_pasta(arquivo)
    else:
        raise FileNotFoundError(
            f"Entrada não encontrada: '{arquivo}' "
            f"(esperado arquivo .parquet ou pasta com .txt)"
        )


def _carregar_entrada_parquet(
    arquivo: str, cfg_entrada: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Carrega entrada de arquivo parquet."""
    if not os.path.isfile(arquivo):
        raise FileNotFoundError(f"Arquivo parquet de entrada não encontrado: '{arquivo}'")

    df = pd.read_parquet(arquivo)
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
        texto = str(row[campo_texto]) if pd.notna(row[campo_texto]) else ""
        if texto.strip():
            itens.append({"chave": chave, "texto": texto})

    return itens


def _carregar_entrada_pasta(pasta: str) -> List[Dict[str, str]]:
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
    """Processa a resposta do modelo conforme tipo_saida configurado.

    Se tipo_saida == "json", tenta extrair JSON da resposta usando
    UtilJson.mensagem_to_json (mesmo padronizador de util_openai.py).
    Se a extração falhar, retorna a resposta original com mensagem de erro.

    Args:
        texto_resposta: texto bruto da resposta do modelo
        config: configuração completa

    Returns:
        Tupla (resposta_processada, erro):
        - resposta_processada: JSON serializado (tipo_saida=json) ou texto original
        - erro: string vazia se ok, mensagem de erro se falhou
    """
    tipo_saida = config["saida"].get("tipo_saida", "str")

    if tipo_saida != "json":
        return texto_resposta, ""

    if not texto_resposta or not texto_resposta.strip():
        return texto_resposta, "Resposta vazia — não foi possível extrair JSON"

    try:
        from util_openai import UtilJson
        resultado_json = UtilJson.mensagem_to_json(texto_resposta, padrao=None)
        if resultado_json is None:
            return texto_resposta, "Não foi possível extrair JSON da resposta"
        # Serializa o dict para string JSON formatada
        return json.dumps(resultado_json, ensure_ascii=False, indent=2), ""
    except ImportError:
        # Fallback: tenta com Util.mensagem_to_json de util.py
        try:
            from util import Util
            resultado_json = Util.mensagem_to_json(texto_resposta, padrao=None)
            if resultado_json is None:
                return texto_resposta, "Não foi possível extrair JSON da resposta"
            return json.dumps(resultado_json, ensure_ascii=False, indent=2), ""
        except Exception as e:
            return texto_resposta, f"Erro ao extrair JSON: {type(e).__name__}: {e}"
    except Exception as e:
        return texto_resposta, f"Erro ao extrair JSON: {type(e).__name__}: {e}"


def formatar_prompt_final(
    conteudo: str, config: Dict[str, Any], tokenizer
) -> str:
    """Formata prompt com chat template do tokenizer (se system_prompt configurado).

    Se system_prompt estiver configurado, aplica chat template do modelo
    com mensagens [system, user] + generation prompt.
    Caso contrário, retorna o conteúdo diretamente.

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
    if not os.path.isfile(arquivo):
        return set()

    try:
        df = pd.read_parquet(arquivo)
    except Exception as e:
        print(f"⚠️  Erro ao ler parquet de saída existente: {e}")
        return set()

    if "chave" not in df.columns:
        return set()

    # Itens com sucesso: erro está vazio/NaN
    if "erro" in df.columns:
        ok = df[df["erro"].isna() | (df["erro"].astype(str).str.strip() == "")]
    else:
        ok = df  # sem coluna erro, todos são considerados ok

    return set(ok["chave"].astype(str))


def _carregar_saida_pasta(pasta: str) -> set:
    """Identifica chaves ok na pasta de saída."""
    if not os.path.isdir(pasta):
        return set()

    chaves = set()
    for nome in os.listdir(pasta):
        if nome.lower().endswith(".txt"):
            chave = os.path.splitext(nome)[0]
            chaves.add(chave)

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
            df_existente = pd.read_parquet(arquivo_saida)
            # Remove itens que serão atualizados
            chaves_novas = set(df_novo["chave"].astype(str))
            df_existente = df_existente[
                ~df_existente["chave"].astype(str).isin(chaves_novas)
            ]
            df_final = pd.concat([df_existente, df_novo], ignore_index=True)
        except Exception:
            df_final = df_novo
    else:
        df_final = df_novo

    # Garante diretório existe
    pasta = os.path.dirname(arquivo_saida)
    if pasta:
        os.makedirs(pasta, exist_ok=True)

    df_final.to_parquet(arquivo_saida, index=False)


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
# Processamento em Batch
# ---------------------------------------------------------------------------

def processar_batch(
    engine,
    tokenizer,
    itens_pendentes: List[Dict[str, str]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Processa itens pendentes em batches, salvando incrementalmente.

    Args:
        engine: VLLMInferenceEngine inicializado
        tokenizer: tokenizer do modelo
        itens_pendentes: lista de {chave, texto}
        config: configuração completa

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

    total = len(itens_pendentes)
    stats = {
        "processados_ok": 0,
        "processados_erro": 0,
        "tokens_entrada": 0,
        "tokens_saida": 0,
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

            if eh_parquet:
                salvar_resultados_parquet(arquivo_saida, resultados_batch)
            else:
                for r in resultados_batch:
                    resumo = json.loads(r["resumo"])
                    salvar_resultado_pasta(
                        arquivo_saida, r["chave"], r["resposta"],
                        resumo, r["erro"]
                    )
            continue

        tempo_batch = time.time() - inicio_batch
        tempo_por_item = tempo_batch / len(batch_itens) if batch_itens else 0

        # Processa resultados do batch
        resultados_batch = []
        for i, item in enumerate(batch_itens):
            if i < len(resultados_vllm):
                res = resultados_vllm[i]
                prompt_tokens = len(tokenizer.encode(prompts[i]))
                completion_tokens = res.get("tokens", 0)

                resumo = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
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

                if not erro_json:
                    stats["processados_ok"] += 1
                else:
                    stats["processados_erro"] += 1
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
            json.loads(r["resumo"]).get("usage", {}).get("completion_tokens", 0)
            for r in resultados_batch
            if not r["erro"]
        )
        feitos = min(batch_start + len(batch_itens), total)
        print(
            f"   ✅ Batch {batch_num} concluído em {_formatar_tempo(tempo_batch)} "
            f"({tokens_batch} tokens gerados) — "
            f"progresso: {feitos}/{total} ({100 * feitos // total}%)"
        )

    stats["tempo_total"] = time.time() - inicio_total
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
                engine_core.shutdown(timeout=15)
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
    else:
        pasta_saida = os.path.abspath(arquivo_saida)
        
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
                            'id': row["chave"],
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
                registros.append({
                    'id': dados.get('chave', nome.replace('.json', '')),
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
    csv_path = os.path.join(pasta_saida, 'predicoes.csv')
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
    resumo_file = os.path.join(pasta_saida, "resumo.json")
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
        tokens_png = os.path.join(pasta_saida, 'predicoes_tokens.png')
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
        tempo_png = os.path.join(pasta_saida, 'predicoes_tempo.png')
        UtilGraficos.gerar_boxplot(
            dados={'Tempo (s)': tempos},
            titulo='Distribuição de Tempo de Geração — Predições vLLM Batch',
            ylabel='Segundos',
            arquivo_saida=tempo_png,
            nota=nota_tempo,
        )
        print(f"   ⏱️  Boxplot tempo: {tempo_png}")
    except ImportError:
        print("   ⚠️  util_graficos não encontrado, gráficos não gerados.")
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

    # --- Verifica saída existente (retomada) ---
    print(f"\n🔍 Verificando saída existente: {config['saida']['arquivo']}")
    chaves_ok = carregar_saida_existente(config)

    # Filtra pendentes
    itens_pendentes = [item for item in itens if item["chave"] not in chaves_ok]

    if not itens_pendentes:
        print(f"✅ Todos os {len(itens)} itens já foram processados com sucesso!")
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
        
        system_prompt = config["entrada"].get("system_prompt", "")
        if system_prompt:
            print("\n--- Aplicando Chat Template ---")
            print("Carregando tokenizer do HuggingFace...")
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config["modelo"]["caminho"])
                prompt_final = formatar_prompt_final(conteudo, config, tokenizer)
                print("\n--- Prompt Final (enviado ao modelo) ---")
                print(prompt_final)
            except ImportError:
                print("❌ Biblioteca 'transformers' não encontrada para o teste.")
            except Exception as e:
                print(f"❌ Erro ao carregar tokenizer: {e}")
        else:
            print("\n--- Prompt Final (sem system prompt/chat template) ---")
            print(conteudo)
        print("="*80)
        sys.exit(0)

    # Aplica limite max_itens (se configurado)
    max_itens = config["geracao"].get("max_itens", 0)
    if max_itens > 0 and len(itens_pendentes) > max_itens:
        print(f"   ⚠️  Limitando processamento a {max_itens} itens (configuração max_itens)")
        itens_pendentes = itens_pendentes[:max_itens]

    # --- Exibe configuração ---
    # TODO: incluir na impressão um resumo das configurações do vLLM
    cfg_geracao = config["geracao"]
    print(f"\n⚙️  Configuração de geração:")
    print(f"   Max tokens: {cfg_geracao['max_tokens']}")
    print(f"   Temperature: {cfg_geracao['temperature']}")
    print(f"   Top-k: {cfg_geracao['top_k']}")
    print(f"   Top-p: {cfg_geracao['top_p']}")
    print(f"   Batch size: {cfg_geracao.get('batch_size', 64)}")
    if cfg_geracao.get('max_itens', 0) > 0:
        print(f"   Max itens: {cfg_geracao['max_itens']}")

    saida_tipo = "parquet" if _saida_eh_parquet(config) else "pasta"
    print(f"   Saída: {config['saida']['arquivo']} ({saida_tipo})")

    # --- Inicializa vLLM ---
    engine = None
    try:
        engine, tokenizer = inicializar_engine(config)

        # --- Processa ---
        stats = processar_batch(engine, tokenizer, itens_pendentes, config)

        # --- Resumo final ---
        print("\n" + "=" * 70)
        print("📊 RESUMO FINAL")
        print("=" * 70)
        print(f"   ✅ Processados com sucesso: {stats['processados_ok']}")
        if stats["processados_erro"] > 0:
            print(f"   ❌ Processados com erro: {stats['processados_erro']}")
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