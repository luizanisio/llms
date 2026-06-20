#!/bin/bash
# =============================================================================
# PARÂMETROS DO JOB (linhas #SBATCH são lidas pelo Slurm; demais são comentários)
# =============================================================================

# Nome do job — aparece no squeue e no nome dos arquivos de log (%x)
#SBATCH --job-name=treinar_pytorch

# Partição de execução:
#SBATCH --partition=gpu

# Recurso de GPU:
#SBATCH --gres=gpu:1

# CPUs disponíveis para o processo Python (data loading, tokenização, I/O)
#SBATCH --cpus-per-task=16

# RAM do sistema (CPU)
#SBATCH --mem=64G

# Tempo máximo de execução (HH:MM:SS)
#SBATCH --time=48:00:00

# Arquivo de saída padrão: <job-name>_<job-id>.out
#SBATCH --output=/students/luiz.abatitucci/jobs_logs/%x_%j.out

# Arquivo de saída de erros: <job-name>_<job-id>.err
#SBATCH --error=/students/luiz.abatitucci/jobs_logs/%x_%j.err

# Notificações por e-mail: END = ao terminar, FAIL = se falhar
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=luizanisio@gmail.com

# =============================================================================

# pasta do próprio script (funciona independente de onde o sbatch for chamado)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate luizbat02

echo "=== Iniciando job: $(date) ==="
echo "Host     : $(hostname)"
echo "Pasta    : $SCRIPT_DIR"
echo "Python   : $(which python)"
echo "GPU info :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi indisponível"
echo "==============================="


echo "1. Instalando pytorch"
pip install "torch>=2.12.0" torchvision --index-url https://download.pytorch.org/whl/cu130

echo "1. Instalando requirements"
pip install -r /students/luiz.abatitucci/llms/src/requirements.txt

python /students/luiz.abatitucci/llms/src/teste_ambiente.py

echo "=== Job de Instalação Finalizado: $(date) ==="
