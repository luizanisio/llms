# 🧪 Dicas e Configurações de Ambiente para Experimentos

> 📖 **Para detalhes sobre o design conceitual, os objetivos de pesquisa (Summa, PubMed, Puil) e as ablações (d1 a d8), consulte o [README_protocolos.md](./README_protocolos.md).**

Este guia centraliza as práticas recomendadas para execução de experimentos, configuração do ambiente e resolução de problemas comuns.

---

## ⚠️ Instruções para Recriar o Ambiente (Recomendado)

Para evitar conflitos crônicos entre versões de bibliotecas em C++ (CUDA, FlashInfer, vLLM e PyTorch), é altamente recomendado recriar o ambiente conda do zero.

**Atenção:** O PyTorch com suporte a CUDA **não** está no PyPI padrão. Ele deve ser instalado manualmente (Passo 2). Instalar sem seguir os passos abaixo pode resultar em versão CPU do torch, suporte a GPU ou flash attention nativo.

### Passo 0: Preparar o CUDA Toolkit (se `nvcc` não for encontrado)
Certifique-se de que o compilador CUDA está acessível rodando `which nvcc`.

**H100 / Linux Nativo (CUDA 12.8):**
Se necessário, instale no próprio ambiente conda:
```bash
conda install -c conda-forge "gcc<14.0" "gxx<14.0"
conda install -c nvidia cuda-nvcc=12.8 -y
export CUDA_HOME=$CONDA_PREFIX
```

**RTX 3060 / WSL2 Debian 13:**
```bash
sudo apt update && sudo apt install -y wget gnupg ca-certificates
wget https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit-12-8

# Alternativa via Conda:
# conda install -c nvidia/label/cuda-12.8.0 cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
```

### Passo 1: Criar o ambiente Conda
```bash
conda remove -n luizbat01 --all   # Se precisar remover o antigo
conda create -n luizbat01 python=3.12.11 -y
conda activate luizbat01
```

### Passo 2: Instalar PyTorch (fixado para CUDA 12.8)
É essencial que seja instalado a partir do index da NVIDIA/PyTorch:
```bash
pip install "torch==2.10.0" torchvision "setuptools==80.10.2" "fsspec<=2026.4.0" --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```
*(Nota: Para H100 ou RTX 3060 as versões testadas foram as mesmas de dependências, garantindo `torch==2.11.0+cu128` ou `torch==2.10.0+cu128`).*

### Passo 3: Instalar o vLLM (Evita quebra no CUDA 12.8 e FlashInfer)
```bash
pip install vllm==0.18.1 triton==3.6.0
```

### Passo 4: Instalar os pacotes adicionais
Instale o restante das dependências listadas nos arquivos `requirements`. Em caso de erro de espaço no dispositivo, mude o diretório temporário:
```bash
TMPDIR=/var/tmp pip install -r ../src/requirements_sem_versao.txt
# OU
TMPDIR=/var/tmp pip install -r ../src/requirements.txt
```
> **Alinhamento do PIP:** Se houver conflitos com numpy ou cuda-python, ajuste executando: `pip install "numpy<2.4.0" "cuda-python<13" "cuda-bindings<13" "cuda-pathfinder>=1.4.2"`

### Passo 5: Compilar o Flash-Attn (Opcional)
Desde o PyTorch 2.0, o SDPA nativo já cobre economia de memória para atenção na maioria dos casos (só ativar `flash_attention_2: true` no YAML de treinamento). Use este passo apenas se precisar de features avançadas como sliding window, paged attention ou FA-3 na H100.

**H100 ou H200/ Servidor:**
```bash
TMPDIR=/var/tmp TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-deps --force-reinstall
```

**RTX 3060 em ambiente WSL2:** já que o instalador pode não identificar corretamente a GPU
```bash
TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=1 pip install flash-attn --no-build-isolation --no-deps --force-reinstall
```

**Outras GPUs**
```bash
TMPDIR=/var/tmp MAX_JOBS=8 pip install flash-attn --no-build-isolation --no-deps --force-reinstall
```

### Passo 6: Validar o Ambiente
Valide a instalação executando:
```bash
python ../src/teste_ambiente.py
```
Esse script verifica pacotes, versões, CUDA, GPU e backends SDPA, alertando sobre qualquer configuração pendente.

---

## ⚡ Redução de Uso de VRAM (Otimizações)

O treinamento de modelos foi migrado nativamente para **HuggingFace Transformers + PEFT** para maior compatibilidade Multi-GPU. Para otimização de memória, usamos **Flash Attention** e **Liger-Kernel**, que atuam de formas complementares:

### Flash Attention (SDPA / flash-attn)
Atua no bloco de atenção `O(n) VRAM`.
* **PyTorch SDPA:** Nativo. Sem instalação extra. Recomendado para RTX 3060 e fine-tuning local.
* **flash-attn:** Pacote pip otimizado. Recomendado para contextos longos, H100 (FA-3) e multi-GPU avançado.

### Liger-Kernel
Triton kernels focados em partes que *não* são atenção (RMSNorm, RoPE, SwiGLU, FusedLinearCrossEntropy). Economiza VRAM significativamente.
* ⚠️ **Não suporta `device_map="auto"`.** Em configurações multi-GPU ou sharded FSDP com PEFT, pode ser necessário desabilitar certas flags (`fused_linear_cross_entropy=False`).

---

## 🆘 Solução de Problemas Comuns

- **"CUDA_HOME environment variable is not set" ao compilar flash-attn:**
  Verifique `which nvcc` e exporte as variáveis: `export CUDA_HOME=...; export PATH=$CUDA_HOME/bin:$PATH`.
- **"undefined symbol: _ZN3c104cuda..." ao importar flash-attn:**
  Seu pacote `flash-attn` compilou contra um CUDA diferente do PyTorch. Garanta que o `CUDA_HOME` reflete o mesmo CUDA do PyTorch e recompile (Passo 5).
- **"No space left on device":**
  Sempre passe a flag de sistema `TMPDIR=/var/tmp` antes do comando pip.
- **VLLM ou FlashInfer sem achar `curand.h` no CUDA 12.8:**
  As versões instaladas via pip perdem o sufixo `-cu12`. Ajuste os scripts para repassar o diretório para o compilador:
  `export CPATH="$(python -c 'import sys, glob; print(":".join(glob.glob(f"{sys.prefix}/lib/python*/site-packages/nvidia/*/include")))'):$CPATH"`

---

## 🖥️ Dicas de Execução em Background

### Uso do `tmux` (Recomendado)
O `tmux` é leve e permite deixar sessões rodando sem que a queda da conexão derrube o script.

> No terminal do Jupyter Notebook ocorrem erros com as configurações de cor e PATH padrão do tmux, por isso existe o utilitário `setup_tmux.sh`, que encapsula a solução usando o atalho `tm`.

```bash
# Inicializa e configura o atalho (rode isso uma vez no shell atual)
source setup_tmux.sh

tm new -s treino         # cria nova sessão chamada 'treino'
tm attach -t treino      # reconecta a sessão caso ela já exista
tm ls                    # lista sessões
```
*Dica:* Para sair de uma sessão sem interrompê-la (detach), pressione `Ctrl+B`, solte, e depois pressione `D`.

### Uso do `nohup` (Alternativa)
Mais simples para apenas enviar um comando para background e fechar o terminal.

```bash
nohup python util_vllm_batch.py --config 05_extracao_b_teste.yaml &

# Visualizar as últimas 50 linhas
tail -n 50 nohup.out

# Acompanhar o progresso em tempo real
tail -f nohup.out
```
