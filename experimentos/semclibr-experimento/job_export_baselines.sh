#!/bin/bash
# =============================================================================
# PARÂMETROS DO JOB (linhas #SBATCH são lidas pelo Slurm; demais são comentários)
# =============================================================================

# Nome do job — aparece no squeue e no nome dos arquivos de log (%x)
#SBATCH --job-name=semclinbr-extracao-baselines

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
#SBATCH --time=24:00:00

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

echo "=== Iniciando job: $(date) ==="
echo "Host     : $(hostname)"
echo "Pasta    : $SCRIPT_DIR"
echo "Python   : $(which python)"
echo "GPU info :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi indisponível"
echo "==============================="


CONFIGS=("02_semclinbr_1_5b.yaml" "02_semclinbr_7b.yaml")

for CONFIG_FILE in "${CONFIGS[@]}"; do
    echo ""
    echo "============================================================"
    echo "=== Iniciando extração com config: $CONFIG_FILE ==="
    echo "=== Hora: $(date) ==="
    echo "============================================================"

    python $SRC_DIR/util_vllm_batch.py --config $BASE_DIR/$CONFIG_FILE

    echo "=== Extração com $CONFIG_FILE finalizada: $(date) ==="
done

# =============================================================================
# COMPARAÇÃO — 03_compara_gold_full.yaml
# =============================================================================
echo ""
echo "============================================================"
echo "=== Iniciando comparação: 03_compara_gold_full.yaml ==="
echo "=== Hora: $(date) ==="
echo "============================================================"

CONFIG_COMPARA="$BASE_DIR/03_compara_gold_full.yaml"
SAIDA_PASTA=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_COMPARA'))['saida']['pasta'])" 2>/dev/null)

if [ -n "$SAIDA_PASTA" ] && [ -d "$BASE_DIR/$SAIDA_PASTA" ]; then
    echo "Pasta de saída ($SAIDA_PASTA) já existe. Ignorando a comparação."
else
    python "$SRC_DIR/comparar_extracoes.py" --config "$CONFIG_COMPARA"
fi

echo "=== Comparação finalizada: $(date) ==="

echo "=== Job finalizado: $(date) ==="
