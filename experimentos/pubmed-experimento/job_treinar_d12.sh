#!/bin/bash
# =============================================================================
# PARÂMETROS DO JOB (linhas #SBATCH são lidas pelo Slurm; demais são comentários)
# =============================================================================

# Nome do job — aparece no squeue e no nome dos arquivos de log (%x)
#SBATCH --job-name=treinar_d12_pubmed

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

echo "Configurando variáveis de ambiente..."
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

echo "=== Iniciando job: $(date) ==="
echo "Host     : $(hostname)"
echo "Pasta    : $SCRIPT_DIR"
echo "Python   : $(which python)"
echo "GPU info :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi indisponível"
echo "==============================="

CONFIGS=(
  "04_treinar_d1.yaml"
  "04_treinar_d2.yaml"
)

OUT_BASE="/students/luiz.abatitucci/llms/experimentos/pubmed-experimento/treinos"

for CONFIG in "${CONFIGS[@]}"; do
  SUFFIX=$(echo "$CONFIG" | sed 's/04_treinar_//' | sed 's/\.yaml//')
  MODEL_DIR="${OUT_BASE}/Qwen2.5-1.5B-Instruct(${SUFFIX})"
  LOSS_FILE="${MODEL_DIR}/treinamento/treinamento_loss.png"
  
  echo "========================================="
  echo "Processando configuração: $CONFIG"
  
  if [ -f "$LOSS_FILE" ]; then
    echo "=> Já treinado. Arquivo $LOSS_FILE encontrado. Pulando."
  else
    echo "=> Arquivo $LOSS_FILE não encontrado. Iniciando treinamento..."
    python /students/luiz.abatitucci/llms/src/treinar_unsloth.py --treinar "/students/luiz.abatitucci/llms/experimentos/pubmed-experimento/$CONFIG"
  fi
done

echo "=== Job finalizado: $(date) ==="
