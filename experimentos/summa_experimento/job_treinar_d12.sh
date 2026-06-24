#!/bin/bash
# =============================================================================
# PARÂMETROS DO JOB (linhas #SBATCH são lidas pelo Slurm; demais são comentários)
# =============================================================================

# Nome do job — aparece no squeue e no nome dos arquivos de log (%x)
#SBATCH --job-name=treinar_d12_summa

# Partição de execução:
#SBATCH --partition=gpu

# Recurso de GPU:
#SBATCH --gres=gpu:1

# CPUs disponíveis para o processo Python (data loading, tokenização, I/O)
#SBATCH --cpus-per-task=32

# RAM do sistema (CPU)
#SBATCH --mem=128G

# Tempo máximo de execução (HH:MM:SS)
#SBATCH --time=99:00:00

# Arquivo de saída padrão: <job-name>_<job-id>.out
#SBATCH --output=/students/luiz.abatitucci/llms/experimentos/summa_experimento/jobs_logs/%x_%j.out

# Arquivo de saída de erros: <job-name>_<job-id>.err
#SBATCH --error=/students/luiz.abatitucci/llms/experimentos/summa_experimento/jobs_logs/%x_%j.err

# Notificações por e-mail: END = ao terminar, FAIL = se falhar
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=luizanisio@gmail.com

# =============================================================================

# pasta do próprio script (funciona independente de onde o sbatch for chamado)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate luizbat02

echo "Configurando variáveis de ambiente..."
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
# Permite que o alocador CUDA expanda segmentos existentes em vez de exigir blocos
# contíguos — elimina OOM por fragmentação (reservado-mas-não-alocado ~10 GB)
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Iniciando job: $(date) ==="
echo "Host     : $(hostname)"
echo "Pasta    : $SCRIPT_DIR"
echo "Python   : $(which python)"
echo "GPU info :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi indisponível"
echo "==============================="

python /students/luiz.abatitucci/llms/src/treinar_unsloth.py --treinar /students/luiz.abatitucci/llms/experimentos/summa_experimento/04_treinar_d1.yaml

python /students/luiz.abatitucci/llms/src/treinar_unsloth.py --treinar /students/luiz.abatitucci/llms/experimentos/summa_experimento/04_treinar_d2.yaml

echo "=== Job finalizado: $(date) ==="
