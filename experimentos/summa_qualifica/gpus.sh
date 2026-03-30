#!/bin/bash

# deve ser executado com source para exportar a variável de ambiente para o processo atual
# source gpus.sh 0,1,2,3
if [ -n "$1" ]; then
    export CUDA_VISIBLE_DEVICES="$1"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "Use 'source gpus.sh ...' para que a configuração de GPUs seja aplicada ao ambiente atual."
fi

