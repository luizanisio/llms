#!/bin/bash
# =============================================================================
# PARÂMETROS DO JOB (linhas #SBATCH são lidas pelo Slurm; demais são comentários)
# =============================================================================

# Nome do job — aparece no squeue e no nome dos arquivos de log (%x)
#SBATCH --job-name=treinar_d_mini_pubmed

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
#SBATCH --output=/students/luiz.abatitucci/llms/experimentos/pubmed-experimento/jobs_logs/%x_%j.out

# Arquivo de saída de erros: <job-name>_<job-id>.err
#SBATCH --error=/students/luiz.abatitucci/llms/experimentos/pubmed-experimento/jobs_logs/%x_%j.err

# Notificações por e-mail: END = ao terminar, FAIL = se falhar
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=luizanisio@gmail.com

# =============================================================================

# pasta do próprio script (funciona independente de onde o sbatch for chamado)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate luizbat02

echo "=== Diagnóstico Detalhado PyTorch vs GPU ==="
python -c "
import torch
print(f'1. PyTorch version: {torch.__version__}')
print(f'2. PyTorch CUDA version: {torch.version.cuda}')
print(f'3. torch.cuda.is_available(): {torch.cuda.is_available()}')
print(f'4. Device count: {torch.cuda.device_count()}')

# Tenta forçar a inicialização para capturar o erro real
try:
    torch._C._cuda_init()
    print('5. Inicialização interna do CUDA: Sucesso')
except Exception as e:
    print(f'5. ERRO na inicialização interna do CUDA: {e}')
"
echo "---"
echo "=== NVIDIA-SMI COMPLETO (Compute Node) ==="
nvidia-smi
echo "==========================================="
echo "=== Iniciando job: $(date) ==="
echo "Host     : $(hostname)"
echo "Pasta    : $SCRIPT_DIR"
echo "Python   : $(which python)"
echo "GPU info :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi indisponível"
echo "==============================="

python /students/luiz.abatitucci/llms/src/treinar_unsloth.py --treinar /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/04_treinar_b_mini_local.yaml


python /students/luiz.abatitucci/llms/src/treinar_unsloth.py --treinar /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/04_treinar_d_mini_ff_lora.yaml

echo "=== Job finalizado: $(date) ==="
