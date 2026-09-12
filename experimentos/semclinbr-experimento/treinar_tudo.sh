#!/bin/bash
# =============================================================================
# _treinar_tudo.sh — Treinamento completo + Extração de todos os protocolos
# =============================================================================
# Script para servidor sem Slurm.
# - Treina todos os protocolos sequencialmente, verificando se já existe
#   o arquivo treinamento_loss.png para pular protocolos já treinados.
# - Ao final, se TODOS os treinamentos foram concluídos com sucesso,
#   inicia a extração de todos os protocolos, verificando se já existe
#   o arquivo de saída .parquet para pular extrações já feitas.
# - Pode ser reiniciado a qualquer momento: protocolos já processados
#   serão automaticamente ignorados.
# =============================================================================

set -euo pipefail

# pasta do próprio script (funciona independente de onde for chamado)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

BASE_DIR="$SCRIPT_DIR"
SRC_DIR="$(dirname "$(dirname "$BASE_DIR")")/src"

# Resolver possível erro de I/O no Triton cache
export TRITON_CACHE_DIR="/tmp/triton_cache_$$"

echo "=== Iniciando _treinar_tudo.sh: $(date) ==="
echo "Host     : $(hostname)"
echo "Pasta    : $SCRIPT_DIR"
echo "Base     : $BASE_DIR"
echo "SRC      : $SRC_DIR"
echo "Python   : $(which python)"
echo "GPU info :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi indisponível"
echo "==========================================="

# =============================================================================
# LISTA DE TODOS OS PROTOCOLOS (mesma ordem dos jobs de treinamento e extração)
# =============================================================================
PROTOCOLS=(
  "b"
  "b16"
  "b16r8"
  "c"
  "d1"
  "d1a"
  "d1b"
  "d2"
  "d3"
  "d4"
  "d5"
  "d6"
  "d7"
  "d8"
  "d9"
  "d10"
  "d11"
  "d12"
  "d13"
  "d14"
  "d15"
  "d16"
  "d17"
  "d18"
  "d19"
  "d20"
  "d21"
  "d22"
  "d23"
  "d24"
  "d25"
)

OUT_BASE="$BASE_DIR/treinos"

# =============================================================================
# FASE 1: TREINAMENTO
# =============================================================================
echo ""
echo "###########################################################################"
echo "### FASE 1: TREINAMENTO DE TODOS OS PROTOCOLOS"
echo "###########################################################################"
echo ""

TREINO_OK=0
TREINO_SKIP=0
TREINO_FAIL=0
FALHAS_TREINO=()

for PROTOCOL in "${PROTOCOLS[@]}"; do
  CONFIG="04_treinar_${PROTOCOL}.yaml"
  MODEL_DIR="${OUT_BASE}/Qwen2.5-7B-Instruct(${PROTOCOL})"
  LOSS_FILE="${MODEL_DIR}/treinamento/treinamento_loss.png"

  echo "========================================="
  echo "Processando treinamento: protocolo=${PROTOCOL}  config=${CONFIG}"

  # Verifica se o arquivo de configuração existe
  if [ ! -f "$BASE_DIR/$CONFIG" ]; then
    echo "=> AVISO: Arquivo de configuração $CONFIG não encontrado. Pulando."
    TREINO_SKIP=$((TREINO_SKIP + 1))
    continue
  fi

  if [ -f "$LOSS_FILE" ]; then
    echo "=> Já treinado. Arquivo $LOSS_FILE encontrado. Pulando."
    TREINO_OK=$((TREINO_OK + 1))
  else
    echo "=> Arquivo $LOSS_FILE não encontrado. Iniciando treinamento..."
    echo "=> Início: $(date)"

    if python "$SRC_DIR/treinar_unsloth.py" --treinar "$BASE_DIR/$CONFIG"; then
      echo "=> Treinamento concluído com sucesso: $(date)"
      TREINO_OK=$((TREINO_OK + 1))
    else
      echo "=> ERRO no treinamento do protocolo ${PROTOCOL}: $(date)"
      TREINO_FAIL=$((TREINO_FAIL + 1))
      FALHAS_TREINO+=("$PROTOCOL")
    fi
  fi
done

echo ""
echo "========================================="
echo "RESUMO DO TREINAMENTO:"
echo "  Concluídos/Já existentes : $TREINO_OK"
echo "  Ignorados (sem config)   : $TREINO_SKIP"
echo "  Falhas                   : $TREINO_FAIL"
if [ ${#FALHAS_TREINO[@]} -gt 0 ]; then
  echo "  Protocolos com falha     : ${FALHAS_TREINO[*]}"
fi
echo "========================================="

# =============================================================================
# VERIFICAÇÃO: todos os treinamentos devem estar concluídos para a extração
# =============================================================================
echo ""
echo "Verificando se todos os treinamentos estão concluídos antes da extração..."

TODOS_TREINADOS=true
PENDENTES=()

for PROTOCOL in "${PROTOCOLS[@]}"; do
  CONFIG="04_treinar_${PROTOCOL}.yaml"
  # Só valida protocolos que possuem configuração de treinamento
  if [ ! -f "$BASE_DIR/$CONFIG" ]; then
    continue
  fi

  MODEL_DIR="${OUT_BASE}/Qwen2.5-7B-Instruct(${PROTOCOL})"
  LOSS_FILE="${MODEL_DIR}/treinamento/treinamento_loss.png"

  if [ ! -f "$LOSS_FILE" ]; then
    TODOS_TREINADOS=false
    PENDENTES+=("$PROTOCOL")
  fi
done

if [ "$TODOS_TREINADOS" = false ]; then
  echo ""
  echo "###########################################################################"
  echo "### EXTRAÇÃO NÃO INICIADA: há treinamentos pendentes"
  echo "### Protocolos sem treinamento concluído: ${PENDENTES[*]}"
  echo "###########################################################################"
  echo ""
  echo "=== Script finalizado (sem extração): $(date) ==="
  exit 1
fi

echo "=> Todos os treinamentos concluídos. Prosseguindo para extração."

# =============================================================================
# FASE 2: EXTRAÇÃO
# =============================================================================
echo ""
echo "###########################################################################"
echo "### FASE 2: EXTRAÇÃO DE TODOS OS PROTOCOLOS"
echo "###########################################################################"
echo ""

EXTRACAO_OK=0
EXTRACAO_SKIP=0
EXTRACAO_FAIL=0
FALHAS_EXTRACAO=()

for PROTOCOL in "${PROTOCOLS[@]}"; do
  CONFIG_FILE="05_extracao_${PROTOCOL}_teste.yaml"
  ARQUIVO_SAIDA="$BASE_DIR/saidas/saida_semclinbr_7b(${PROTOCOL})_teste.parquet"

  echo "========================================="
  echo "Processando extração: protocolo=${PROTOCOL}  config=${CONFIG_FILE}"

  # Verifica se o arquivo de configuração de extração existe
  if [ ! -f "$BASE_DIR/$CONFIG_FILE" ]; then
    echo "=> AVISO: Arquivo de configuração $CONFIG_FILE não encontrado. Pulando."
    EXTRACAO_SKIP=$((EXTRACAO_SKIP + 1))
    continue
  fi

  if [ -f "$ARQUIVO_SAIDA" ]; then
    echo "=> Arquivo $ARQUIVO_SAIDA já existe. Pulando extração."
    EXTRACAO_OK=$((EXTRACAO_OK + 1))
  else
    echo "=> Arquivo de saída não encontrado. Iniciando extração..."
    echo "=> Início: $(date)"

    if python "$SRC_DIR/util_vllm_batch.py" --config "$BASE_DIR/$CONFIG_FILE"; then
      echo "=> Extração concluída com sucesso: $(date)"
      EXTRACAO_OK=$((EXTRACAO_OK + 1))
    else
      echo "=> ERRO na extração do protocolo ${PROTOCOL}: $(date)"
      EXTRACAO_FAIL=$((EXTRACAO_FAIL + 1))
      FALHAS_EXTRACAO+=("$PROTOCOL")
    fi
  fi
done

echo ""
echo "========================================="
echo "RESUMO DA EXTRAÇÃO:"
echo "  Concluídas/Já existentes : $EXTRACAO_OK"
echo "  Ignoradas (sem config)   : $EXTRACAO_SKIP"
echo "  Falhas                    : $EXTRACAO_FAIL"
if [ ${#FALHAS_EXTRACAO[@]} -gt 0 ]; then
  echo "  Protocolos com falha      : ${FALHAS_EXTRACAO[*]}"
fi
echo "========================================="

# =============================================================================
# RESUMO FINAL
# =============================================================================
echo ""
echo "###########################################################################"
echo "### RESUMO FINAL"
echo "###########################################################################"
echo ""
echo "TREINAMENTO: ${TREINO_OK} ok | ${TREINO_SKIP} ignorados | ${TREINO_FAIL} falhas"
echo "EXTRAÇÃO   : ${EXTRACAO_OK} ok | ${EXTRACAO_SKIP} ignoradas | ${EXTRACAO_FAIL} falhas"
echo ""

if [ $TREINO_FAIL -gt 0 ] || [ $EXTRACAO_FAIL -gt 0 ]; then
  echo "*** ATENÇÃO: Houve falhas durante a execução. ***"
  [ ${#FALHAS_TREINO[@]} -gt 0 ] && echo "  Falhas treino   : ${FALHAS_TREINO[*]}"
  [ ${#FALHAS_EXTRACAO[@]} -gt 0 ] && echo "  Falhas extração : ${FALHAS_EXTRACAO[*]}"
  echo ""
  echo "=== Script finalizado com erros: $(date) ==="
  exit 1
else
  echo "*** Tudo concluído com sucesso! ***"
  echo ""
  echo "=== Script finalizado: $(date) ==="
  exit 0
fi
