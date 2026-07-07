#!/bin/bash
# =============================================================================
# PARÂMETROS DO JOB (linhas #SBATCH são lidas pelo Slurm; demais são comentários)
# =============================================================================

# Nome do job — aparece no squeue e no nome dos arquivos de log (%x)
#SBATCH --job-name=pubmed-treinar-bc

# Partição de execução:
#SBATCH --partition=gpu

# Recurso de GPU:
#SBATCH --gres=gpu:1

# CPUs disponíveis para o processo Python (data loading, tokenização, I/O)
#SBATCH --cpus-per-task=32

# RAM do sistema (CPU)
#SBATCH --mem=64G

# Tempo máximo de execução (HH:MM:SS)
#SBATCH --time=48:00:00

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

#python $SRC_DIR/treinar_unsloth.py --treinar $BASE_DIR/04_treinar_d_mini_ff_lora.yaml

python $SRC_DIR/treinar_unsloth.py --treinar $BASE_DIR/04_treinar_b.yaml

python $SRC_DIR/treinar_unsloth.py --treinar $BASE_DIR/04_treinar_c.yaml

echo "=== Job finalizado: $(date) ==="
