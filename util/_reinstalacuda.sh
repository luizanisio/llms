# ══════════════════════════════════════════════════════════════════════
# ETAPA 1 — Remover o env treina
# ══════════════════════════════════════════════════════════════════════
conda deactivate
conda env remove -n treina -y

# ══════════════════════════════════════════════════════════════════════
# ETAPA 2 — Remover CUDA 13.2 do sistema e instalar CUDA 12.8 via runfile
# ══════════════════════════════════════════════════════════════════════
sudo apt remove --purge -y cuda-toolkit cuda-toolkit-13-2
sudo apt autoremove -y

# Baixar o runfile do CUDA 12.8 (só o toolkit, sem driver)
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sudo sh cuda_12.8.1_570.124.06_linux.run --toolkit --silent --override
rm cuda_12.8.1_570.124.06_linux.run

# Configurar variáveis de ambiente
sed -i '/CUDA_HOME/d' ~/.bashrc
sed -i '/\/usr\/local\/cuda/d' ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
source ~/.bashrc
nvcc --version

# ══════════════════════════════════════════════════════════════════════
# ETAPA 3 — Recriar o env e instalar tudo do zero
# ══════════════════════════════════════════════════════════════════════
conda create -n treina python=3.13.9 -y
conda activate treina
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
TMPDIR=/var/tmp pip install -r src/requirements.txt
TMPDIR=/var/tmp MAX_JOBS=4 pip install flash-attn --no-build-isolation --force-reinstall

# ══════════════════════════════════════════════════════════════════════
# ETAPA 4 — Validar
# ══════════════════════════════════════════════════════════════════════
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
python -c "from flash_attn import flash_attn_func; print('flash-attn OK')"
nvcc --version