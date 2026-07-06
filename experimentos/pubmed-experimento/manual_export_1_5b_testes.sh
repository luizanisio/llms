#!/bin/bash

# Interrompe o script inteiramente caso o usuário aperte Ctrl+C,
# evitando que o loop inicie o próximo job.
trap "echo 'Script interrompido pelo usuário (Ctrl+C)!'; exit 130" INT

# === Correção para PyTorch + CUDA 13 via pip ===
# Exporta o caminho dos headers do CUDA (como curand.h) para o compilador do flashinfer encontrar.
export CPATH="$(python -c 'import sys, glob; print(":".join(glob.glob(f"{sys.prefix}/lib/python*/site-packages/nvidia/*/include")))'):$CPATH"
# ===============================================

echo "=== Iniciando job: $(date) ==="
echo "Host     : $(hostname)"
echo "Pasta    : $SCRIPT_DIR"
echo "Python   : $(which python)"
echo "GPU info :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi indisponível"
echo "==============================="


#python baixar-qwen1.5b.py
PROTOCOLS=("b" "c" "d1" "d2" "d3" "d4" "d5" "d6" "d7" "d8")

for PROTOCOL in "${PROTOCOLS[@]}"; do
    CONFIG_FILE="05_extracao_${PROTOCOL}_teste.yaml"
    echo "=== Iniciando extração do protocolo: $PROTOCOL ==="
    
    # Roda a extração 10 vezes (útil para repescagem de erros)
    for i in {1..10}; do
        echo "Rodada $i/10 para o protocolo $PROTOCOL..."
        python ../../src/util_vllm_batch.py --config "$CONFIG_FILE"
    done
done


echo "=== Job finalizado: $(date) ==="

# Se necessário
# Derruba o script
#   pkill -f manual_export_1_5b_testes.sh
# Derruba os scripts python
#   pkill -f util_vllm_batch.py
# Derruba os processos do vLLM atrelados a eles
#   pkill -f VLLM::EngineCore
#
# zip -r saidas_vnnnnn ./saidas