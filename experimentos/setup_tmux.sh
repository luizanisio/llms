#!/bin/bash
TMUX_CONF="/opt/app-root/src/.tmux.conf"
MINICONDA="/opt/app-root/src/miniconda"

# Inicializa conda
source "$MINICONDA/etc/profile.d/conda.sh"
conda activate base

# Instala tmux se não estiver disponível
if ! command -v tmux &>/dev/null; then
  echo "[setup_tmux] Instalando tmux..."
  conda install -y -c conda-forge tmux -q
  echo "[setup_tmux] tmux instalado: $(tmux -V 2>/dev/null)"
else
  echo "[setup_tmux] tmux já disponível: $(tmux -V 2>/dev/null)"
fi

# Cria .tmux.conf se não existir
if [[ ! -f "$TMUX_CONF" ]]; then
  echo "[setup_tmux] Criando $TMUX_CONF..."
  cat > "$TMUX_CONF" << TMUXCONF
set -g default-terminal "xterm-256color"
set -g default-shell /bin/bash
set -g default-command "source /opt/app-root/src/miniconda/etc/profile.d/conda.sh && conda activate base && exec bash"
TMUXCONF
  echo "[setup_tmux] .tmux.conf criado."
else
  echo "[setup_tmux] .tmux.conf já existe."
fi

# Função para abrir tmux com TERM correto e .tmux.conf do NFS
tm() {
  TERM=xterm-256color TMUX_TMPDIR=/tmp tmux -f "$TMUX_CONF" "$@"
}
export -f tm

# Detecta se foi executado com source ou sh/bash diretamente
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo ""
  echo "AVISO: execute com source para carregar as funções no terminal atual:"
  echo "  source /opt/app-root/src/setup_tmux.sh"
  echo ""
else
  echo "[setup_tmux] source aplicado corretamente — funções carregadas no terminal atual."
fi

echo ""
echo "=== setup_tmux pronto ==="
echo "  conda:         $(conda --version)"
echo "  tmux:          $(tmux -V 2>/dev/null)"
echo ""
echo "  tm new -s nome       iniciar sessão protegida"
echo "  tm attach -t nome    reconectar após queda"
echo "  tm ls                listar sessões ativas"
echo "  Ctrl+B D             sair sem encerrar a sessão"
echo ""
