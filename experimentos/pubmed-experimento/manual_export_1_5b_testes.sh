#!/bin/bash


echo "=== Iniciando job: $(date) ==="
echo "Host     : $(hostname)"
echo "Pasta    : $SCRIPT_DIR"
echo "Python   : $(which python)"
echo "GPU info :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi indisponível"
echo "==============================="


#python baixar-qwen1.5b.py
PROTOCOLS=("b" "c" "d1" "d2" "d3" "d4" "d5" "d6")

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