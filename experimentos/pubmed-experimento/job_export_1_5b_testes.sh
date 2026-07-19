#!/bin/bash
# =============================================================================
# PARÂMETROS DO JOB (linhas #SBATCH são lidas pelo Slurm; demais são comentários)
# =============================================================================

# Nome do job — aparece no squeue e no nome dos arquivos de log (%x)
#SBATCH --job-name=pubmed-extracao-testes

# Partição de execução:
#   gpu    — GPU exclusiva, VRAM completa (80 GB), sem limite de tempo padrão (produção)
#   shared — GPU compartilhada via MPS, limite de 4 h, VRAM não reservada (testes)
#SBATCH --partition=gpu

# Recurso de GPU:
#   gpu:1  — 1 GPU exclusiva (partição gpu)
#   mps:50 — 50 % de compute compartilhado (partição shared — NÃO usar aqui)
#SBATCH --gres=gpu:1

# CPUs disponíveis para o processo Python (data loading, tokenização, I/O)
#SBATCH --cpus-per-task=8

# RAM do sistema (CPU). vLLM com 20 k prompts e contexto de 32 k precisa de folga
#SBATCH --mem=64G

# Tempo máximo de execução (HH:MM:SS). Job é cancelado ao atingir o limite.
# 8 protocolos × 20 rodadas × ~2-4 h por protocolo ≈ estimativa conservadora
#SBATCH --time=72:00:00

# Arquivo de saída padrão: <job-name>_<job-id>.out
#SBATCH --output=jobs_logs/%x_%j.out

# Arquivo de saída de erros: <job-name>_<job-id>.err
#SBATCH --error=jobs_logs/%x_%j.err

# Notificações por e-mail: END = ao terminar, FAIL = se falhar
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=luizanisio@gmail.com

# =============================================================================

# pasta do próprio script (funciona independente de onde o sbatch for chamado)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

# Constante de diretório base para facilitar portabilidade
BASE_DIR="/students/luiz.abatitucci/llms/experimentos/pubmed-experimento"
SRC_DIR="$(dirname $(dirname "$BASE_DIR"))/src"

source /opt/conda/etc/profile.d/conda.sh
conda activate luizbat02

# echo "Configurando variáveis de ambiente..."
# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CUDA_HOME/bin:$PATH

echo "=== Iniciando job: $(date) ==="
echo "Host     : $(hostname)"
echo "Pasta    : $SCRIPT_DIR"
echo "Python   : $(which python)"
echo "GPU info :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi indisponível"
echo "==============================="


PROTOCOLS=("b" "c" "d1" "d2" "d3" "d4" "d5" "d6" "d7" "d8" "d9" "d10" "d11" "d12")

for PROTOCOL in "${PROTOCOLS[@]}"; do
    CONFIG_FILE="05_extracao_${PROTOCOL}_teste.yaml"
    ARQUIVO_SAIDA="$BASE_DIR/saidas/saida_pubmed_1_5b(${PROTOCOL})_teste.parquet"
                          
    if [ -f "$ARQUIVO_SAIDA" ]; then
        echo "=== Arquivo $ARQUIVO_SAIDA já existe. Pulando extração do protocolo $PROTOCOL. ==="
        continue
    fi

    echo ""
    echo "============================================================"
    echo "=== Iniciando extração do protocolo: $PROTOCOL ==="
    echo "=== Config: $CONFIG_FILE ==="
    echo "=== Hora: $(date) ==="
    echo "============================================================"

    # Roda a extração 20 vezes (útil para repescagem de erros)
    for i in $(seq 1 20); do
        echo "--- Rodada $i/20 para o protocolo $PROTOCOL --- $(date)"
        python $SRC_DIR/util_vllm_batch.py --config $BASE_DIR/$CONFIG_FILE
    done

    echo "=== Protocolo $PROTOCOL finalizado: $(date) ==="
done


echo "=== Job finalizado: $(date) ==="
