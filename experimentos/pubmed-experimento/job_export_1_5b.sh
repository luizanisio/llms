#!/bin/bash
# =============================================================================
# PARÂMETROS DO JOB (linhas #SBATCH são lidas pelo Slurm; demais são comentários)
# =============================================================================

# Nome do job — aparece no squeue e no nome dos arquivos de log (%x)
#SBATCH --job-name=pubmed-qwen1.5b-extracao

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
# 20 k prompts × ~32 k tokens @ ~750 tok/s estimado ≈ 30-40 h no caso médio.
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
conda activate luizbat01

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


#python baixar-qwen1.5b.py
python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

# Mais rodadas para repescagem do que deu erro
python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml

python /students/luiz.abatitucci/llms/src/util_vllm_batch.py --config /students/luiz.abatitucci/llms/experimentos/pubmed-experimento/02_pubmed_rct_1_5b.yaml


echo "=== Job finalizado: $(date) ==="