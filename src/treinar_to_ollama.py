#!/usr/bin/env python3
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Geração de Modelfile para importação de modelos no Ollama.
Chamado automaticamente pelo pipeline de merge (executar_merge).

O merge gera HF safetensors. Para usar no Ollama, converta para GGUF
usando llama.cpp. Veja treinar_ollama_readme.md para instruções completas.
"""

import os
from treinar_unsloth_logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Templates Ollama por família de modelo (Go template syntax)
# Ref: https://github.com/ollama/ollama/blob/main/docs/modelfile.md
# ---------------------------------------------------------------------------

OLLAMA_TEMPLATES = {
    'gemma': (
        '<start_of_turn>user\n'
        '{{ if .System }}{{ .System }}\n\n{{ end }}'
        '{{ .Prompt }}<end_of_turn>\n'
        '<start_of_turn>model\n'
        '{{ .Response }}<end_of_turn>\n'
    ),
    'llama3': (
        '<|begin_of_text|>'
        '{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n'
        '{{ .System }}<|eot_id|>{{ end }}'
        '<|start_header_id|>user<|end_header_id|>\n\n'
        '{{ .Prompt }}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
        '{{ .Response }}<|eot_id|>'
    ),
    'qwen2.5': (
        '{{ if .System }}<|im_start|>system\n'
        '{{ .System }}<|im_end|>\n{{ end }}'
        '<|im_start|>user\n'
        '{{ .Prompt }}<|im_end|>\n'
        '<|im_start|>assistant\n'
        '{{ .Response }}<|im_end|>\n'
    ),
    'chatml': (
        '{{ if .System }}<|im_start|>system\n'
        '{{ .System }}<|im_end|>\n{{ end }}'
        '<|im_start|>user\n'
        '{{ .Prompt }}<|im_end|>\n'
        '<|im_start|>assistant\n'
        '{{ .Response }}<|im_end|>\n'
    ),
    'phi3': (
        '{{ if .System }}<|system|>\n{{ .System }}<|end|>\n{{ end }}'
        '<|user|>\n{{ .Prompt }}<|end|>\n'
        '<|assistant|>\n{{ .Response }}<|end|>\n'
    ),
    'mistral': (
        '[INST] '
        '{{ if .System }}{{ .System }}\n\n{{ end }}'
        '{{ .Prompt }} [/INST]'
        '{{ .Response }}'
    ),
    'llama2': (
        '[INST] '
        '<<SYS>>\n{{ .System }}\n<</SYS>>\n\n'
        '{{ .Prompt }} [/INST]\n'
        '{{ .Response }}'
    ),
    'deepseek': (
        '{{ if .System }}System: {{ .System }}\n\n{{ end }}'
        'User: {{ .Prompt }}\n'
        'Assistant: {{ .Response }}'
    ),
}


def detectar_template_ollama(modelo_base: str) -> tuple:
    """Detecta o template Ollama adequado a partir do nome do modelo base.

    Returns:
        (template_name, template_str) ou (None, None) se não reconhecido.
    """
    from treinar_chat_templates import MODEL_TEMPLATE_MAP
    model_lower = modelo_base.replace('-', '').lower()
    for model_key, template_name in MODEL_TEMPLATE_MAP.items():
        if model_key in model_lower:
            tmpl = OLLAMA_TEMPLATES.get(template_name)
            if tmpl:
                return template_name, tmpl
    return None, None


def _gerar_modelfile(modelo_base: str, nome_modelo: str, pasta: str,
                     quantizacao: str, max_seq: int,
                     template_str: str | None) -> str:
    """Gera o conteúdo do Modelfile para Ollama.
    
    O FROM usa um placeholder <./MeuModelo.gguf> que deve ser substituído
    pelo caminho real do arquivo GGUF após conversão com llama.cpp.
    """
    # Bloco TEMPLATE
    if template_str:
        template_block = f'TEMPLATE """\n{template_str}\n"""'
    else:
        template_block = f'''\
# Template não reconhecido para '{modelo_base}'.
# Adicione o TEMPLATE adequado manualmente.
# Consulte: https://github.com/ollama/ollama/blob/main/docs/modelfile.md'''

    return f'''\
# Modelfile gerado automaticamente pelo pipeline de treinamento
# Modelo base: {modelo_base}
# Formato merge: {quantizacao}
# Pasta: {os.path.basename(pasta)}
#
# === INSTRUÇÕES ===
# 1. Converta o modelo para GGUF usando llama.cpp (veja treinar_ollama_readme.md)
#    python convert_hf_to_gguf.py /<caminho_safetensors> --outfile /<caminho_safetensors>/<./MeuModelo.gguf> --outtype f16
# 2. Substitua <./MeuModelo.gguf> abaixo pelo nome real do arquivo gerado
# 3. Execute (dentro da pasta com o ModelFile e o .gguf):
#      ollama create {nome_modelo} -f Modelfile
#      ollama run {nome_modelo}

FROM <./MeuModelo.gguf>

{template_block}

# num_ctx: contexto usado no treinamento ({max_seq}).
# O modelo merged preserva o contexto nativo do modelo base — que pode ser maior.
# Ajuste conforme necessidade e VRAM disponível.
PARAMETER num_ctx {max_seq}
PARAMETER temperature 0.01
PARAMETER top_p 0.9
PARAMETER top_k 20
'''


def gerar_modelfile_ollama(dirname: str, yaml_config, quantizacao: str) -> None:
    """Gera Modelfile na pasta de merge para futura importação no Ollama.

    O Modelfile é gerado com placeholder FROM <./MeuModelo.gguf>.
    O usuário deve converter para GGUF com llama.cpp e ajustar o FROM.

    Se modelo.ollama estiver configurado no YAML, usa esse nome no Modelfile.
    """
    modelo_base = yaml_config.modelo.base
    max_seq = yaml_config.treinamento.max_seq_length

    # Usa modelo.ollama se configurado, senão gera nome a partir do diretório
    if hasattr(yaml_config.modelo, 'ollama') and yaml_config.modelo.ollama:
        nome_modelo = yaml_config.modelo.ollama
        logger.info(f"<cinza>   📋 Usando nome do modelo.ollama: {nome_modelo}</cinza>")
    else:
        nome_modelo = os.path.basename(dirname).replace('(', '_').replace(')', '')
        logger.info(f"<cinza>   📋 Nome do modelo gerado automaticamente: {nome_modelo}</cinza>")

    template_name, template_str = detectar_template_ollama(modelo_base)

    # Modelfile
    conteudo_modelfile = _gerar_modelfile(
        modelo_base, nome_modelo, dirname, quantizacao, max_seq, template_str
    )
    modelfile_path = os.path.join(dirname, "Modelfile")
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(conteudo_modelfile)
    logger.info("<verde>   ✅ Modelfile criado (ajuste o FROM após converter para GGUF)</verde>")

