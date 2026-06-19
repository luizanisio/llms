#!/bin/bash
# =============================================================================
# PARÂMETROS DO JOB (linhas #SBATCH são lidas pelo Slurm; demais são comentários)
# =============================================================================

# Nome do job — aparece no squeue e no nome dos arquivos de log (%x)
#SBATCH --job-name=treinar_cuda_flash

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
conda activate luizbat01

echo "=== Iniciando job: $(date) ==="
echo "Host     : $(hostname)"
echo "Pasta    : $SCRIPT_DIR"
echo "Python   : $(which python)"
echo "GPU info :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi indisponível"
echo "==============================="

echo "Verificando ambiente PyTorch atual..."
python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"

echo "Instalando compilador nvcc (13.0) via conda..."
conda install -c nvidia cuda-nvcc=13.0 -y

echo "Configurando variáveis de ambiente..."
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

echo "Verificando compilador nvcc local do conda..."
nvcc --version

echo "Limpando cache do pip..."
pip cache purge

echo "Instalando flash-attn a partir do código fonte (pode demorar ~20 minutos)..."
# Usa MAX_JOBS proporcional ao cpus-per-task (16) para acelerar o build
TMPDIR=/var/tmp MAX_JOBS=16 pip install flash-attn --no-build-isolation --no-deps --force-reinstall

echo "=== Testando instalação ==="
python -c "from flash_attn import flash_attn_func; print('✅ flash-attn instalado com sucesso _o/')"
python -c "from flash_attn.ops.triton.rotary import apply_rotary; print('✅ rotary (Triton) OK _o/')"

echo "=== Job de Instalação Finalizado: $(date) ==="
