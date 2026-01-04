# Configuração em ambiente legado GPU NVIDIA MX130
Configuração necessária para o funcionamento do bert em ambiente legado GPU NVIDIA MX130

## Ambiente conda
conda create -n p310 python=3.10
conda activate p310
python -m pip install --upgrade pip

### Instalação do PyTorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

### instalar requirements_p310.txt
pip install -r requirements_p310.txt

