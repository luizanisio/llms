#!/bin/bash

# Compacta todas as subpastas da pasta atual em um único arquivo zip,
# ignorando pastas 'chkpt' e arquivos pesados de modelo (tensores, pesos),
# para transferência rápida dos dados de métricas e comparações.

NOME_ZIP="treinos_metricas.zip"

echo "Iniciando compactação de métricas no diretório: $(pwd)"

# Remove o arquivo zip se ele já existir para não adicionar aos arquivos antigos
if [ -f "$NOME_ZIP" ]; then
    echo "Removendo arquivo $NOME_ZIP antigo..."
    rm "$NOME_ZIP"
fi

# Lista de pastas no diretório atual
pastas=(*/)

# Verifica se encontrou alguma pasta
if [ "${#pastas[@]}" -eq 0 ] || [ "${pastas[0]}" = "*/" ]; then
    echo "Nenhuma pasta encontrada no diretório atual."
    exit 0
fi

echo "Compactando pastas em $NOME_ZIP (ignorando chkpt e arquivos pesados de modelo)..."

# Compacta tudo exceto:
#   - pastas chkpt (checkpoints intermediários)
#   - *.safetensors (pesos em formato safetensors)
#   - *.bin (pesos em formato PyTorch antigo)
#   - *.gguf (pesos em formato GGUF/llama.cpp)
#   - *.pt (checkpoints PyTorch)
#   - *.onnx (pesos em formato ONNX)
zip -r "$NOME_ZIP" "${pastas[@]}" \
    -x "*/chkpt/*" -x "*/chkpt" \
    -x "*.safetensors" \
    -x "*.bin" \
    -x "*.gguf" \
    -x "*.pt" \
    -x "*.onnx"

echo "Compactação concluída! Arquivo: $NOME_ZIP"
