#!/bin/bash
# =============================================================================
# PARÂMETROS DO JOB (linhas #SBATCH são lidas pelo Slurm; demais são comentários)
# =============================================================================

# Nome do job — aparece no squeue e no nome dos arquivos de log (%x)
#SBATCH --job-name=summa-compara-testes

# Partição de execução:
#   gpu    — GPU exclusiva, VRAM completa (80 GB), sem limite de tempo padrão (produção)
#   shared — GPU compartilhada via MPS, limite de 4 h, VRAM não reservada (testes)
#SBATCH --partition=gpu

# Recurso de GPU (BERTScore e SBERT rodam em GPU no pré-cálculo das métricas)
#SBATCH --gres=gpu:1

# CPUs disponíveis para o processo Python (ROUGE/Levenshtein em max_workers=40)
#SBATCH --cpus-per-task=20

# RAM do sistema (CPU)
#SBATCH --mem=64G

# Tempo máximo de execução (HH:MM:SS). Job é cancelado ao atingir o limite.
#SBATCH --time=99:00:00

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
BASE_DIR="/students/luiz.abatitucci/llms/experimentos/summa-experimento"
SRC_DIR="$(dirname $(dirname "$BASE_DIR"))/src"

source /opt/conda/etc/profile.d/conda.sh
conda activate luizbat02

echo "=== Iniciando job: $(date) ==="
echo "Host     : $(hostname)"
echo "Pasta    : $SCRIPT_DIR"
echo "Python   : $(which python)"
echo "GPU info :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi indisponível"
echo "==============================="

# não roda o 03_compara_q235_full.yaml pois já fez a divisão de dificuldade e
# incorporou a divisão de treino/teste/validação (ver job_compara_full.sh)
#echo "1/8 - Executando 03_compara_q235_full.yaml..."
#python "$SRC_DIR/comparar_extracoes.py" --config "$BASE_DIR/03_compara_q235_full.yaml"

echo "2/8 - Executando 06_compara_experimentais.yaml..."
python "$SRC_DIR/comparar_extracoes.py" --config "$BASE_DIR/06_compara_experimentais.yaml"

echo "3/8 - Executando 06_compara_ablacoes.yaml..."
python "$SRC_DIR/comparar_extracoes.py" --config "$BASE_DIR/06_compara_ablacoes.yaml"

echo "4/8 - Executando 06_compara_ordem_cl.yaml..."
python "$SRC_DIR/comparar_extracoes.py" --config "$BASE_DIR/06_compara_ordem_cl.yaml"

echo "5/8 - Executando 06_compara_ordem_pt.yaml..."
python "$SRC_DIR/comparar_extracoes.py" --config "$BASE_DIR/06_compara_ordem_pt.yaml"

echo "6/8 - Executando 06_compara_fronteiras.yaml..."
python "$SRC_DIR/comparar_extracoes.py" --config "$BASE_DIR/06_compara_fronteiras.yaml"

echo "7/8 - Executando 06_compara_capacidade.yaml..."
python "$SRC_DIR/comparar_extracoes.py" --config "$BASE_DIR/06_compara_capacidade.yaml"

echo "8/8 - Executando 06_compara_todos.yaml..."
python "$SRC_DIR/comparar_extracoes.py" --config "$BASE_DIR/06_compara_todos.yaml"

echo "=== Job finalizado: $(date) ==="
