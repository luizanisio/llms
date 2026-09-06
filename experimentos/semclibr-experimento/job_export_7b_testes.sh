#!/bin/bash
# =============================================================================
# PARÂMETROS DO JOB (linhas #SBATCH são lidas pelo Slurm; demais são comentários)
# =============================================================================

# Nome do job — aparece no squeue e no nome dos arquivos de log (%x)
#SBATCH --job-name=semclinbr-extracao-testes

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
BASE_DIR="/students/luiz.abatitucci/llms/experimentos/semclibr-experimento"
SRC_DIR="$(dirname $(dirname "$BASE_DIR"))/src"

source /opt/conda/etc/profile.d/conda.sh
conda activate luizbat02

# Resolver erro OSError: [Errno 5] Input/output error no Triton cache (pasta da rede)
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID:-$$}"

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


PROTOCOLS=("b" "b16" "b16r8" "c" "d1" "d2" "d3" "d4" "d5" "d6" "d7" "d8" "d9" "d10" "d11" "d12" "d13" "d14" "d15" "d16" "d17" "d18" "d19" "d20" "d21" "d22" "d23" "d24" "d25")

for PROTOCOL in "${PROTOCOLS[@]}"; do
    CONFIG_FILE="05_extracao_${PROTOCOL}_teste.yaml"
    ARQUIVO_SAIDA="$BASE_DIR/saidas/saida_semclinbr_7b(${PROTOCOL})_teste.parquet"
                          
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

    # A repescagem de erros é nativa: geracao.tentativas: 20 no YAML
    python $SRC_DIR/util_vllm_batch.py --config $BASE_DIR/$CONFIG_FILE

    echo "=== Protocolo $PROTOCOL finalizado: $(date) ==="
done


echo "=== Job finalizado: $(date) ==="
