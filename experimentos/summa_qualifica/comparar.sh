#!/bin/bash
# Comparação de extrações de documentos via YAML
# Uso: ./comparar.sh [config.yaml]
# Sem argumentos: exibe menu interativo de seleção de YAML na pasta atual
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ../../src/comparar_extracoes.py "$@"
