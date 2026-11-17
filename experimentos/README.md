# 🧪 Experimentos CL+PT — Ponto de Partida

Esta pasta reúne os experimentos do framework de Curriculum Learning com escalonamento de
capacidade (CL+PT). Este arquivo indica onde está cada informação e descreve como preparar o
ambiente de execução.

## 📖 Mapa da Documentação

| Para… | Leia |
| :-- | :-- |
| Executar o pipeline, etapa por etapa (comandos) | [README_PASSO_A_PASSO.md](./README_PASSO_A_PASSO.md) |
| Entender o que cada protocolo testa e qual recorte responde a cada pergunta de pesquisa | [README_protocolos.md](./README_protocolos.md) |
| Configurar ou depurar filtros de dados (extração, treino, comparação) | [README_filtros.md](./README_filtros.md) |
| Consultar números, decisões de implementação e o estado de cada experimento | [INFORMACOES_TREINAMENTO.md](./INFORMACOES_TREINAMENTO.md) |
| Ver os hiperparâmetros efetivos de cada protocolo e as conferências YAML × treino | [RESUMO_EXPERIMENTOS.md](./RESUMO_EXPERIMENTOS.md) (gerado por `tabela_experimentos.py`) |
| Detalhes de domínio (dataset, prompt, gabarito, métricas específicas) | README de cada pasta (tabela abaixo) |

## 📂 Diretórios de Experimento

| Diretório | Descrição | README |
| :-- | :-- | :-- |
| [`summa-experimento/`](./summa-experimento/) | Experimento principal — extração de metadados jurídicos (Qwen 7B, destilação do Qwen3-235B) | [README_SUMMA_EXPERIMENTO.md](./summa-experimento/README_SUMMA_EXPERIMENTO.md) |
| [`pubmed-experimento/`](./pubmed-experimento/) | Generalização — seções de abstracts médicos PubMed 20k RCT (Qwen 1.5B) | [README_PUBMED_EXPERIMENTO.md](./pubmed-experimento/README_PUBMED_EXPERIMENTO.md) |
| [`semclinbr-experimento/`](./semclinbr-experimento/) | Generalização — NER + RE clínico em notas pt-br do SemClinBr (Qwen 7B) | [README_SEMCLIMBR_EXPERIMENTO.md](./semclinbr-experimento/README_SEMCLIMBR_EXPERIMENTO.md) |
| [`puil-mini-experimento/`](./puil-mini-experimento/) | Validação local rápida do pipeline (RTX 3060/2060) | [README_MINI_EXPERIMENTO.md](./puil-mini-experimento/README_MINI_EXPERIMENTO.md) |
| [`summa_qualifica/`](./summa_qualifica/) | Versão preliminar do Summa usada na qualificação (referência histórica) | — |

O código do framework está em [`../src/`](../src/) (ver `src/README.md`).

---

## ⚙️ Ambiente

Para evitar conflitos entre versões de bibliotecas em C++ (CUDA, FlashInfer, vLLM e PyTorch),
recrie o ambiente conda do zero.

**Atenção:** o PyTorch com suporte a CUDA **não** está no PyPI padrão. Ele deve ser instalado
manualmente (Passo 2); sem isso, o pip pode instalar a versão CPU do torch, sem GPU nem flash
attention nativo.

### Versões registradas

Os ambientes usados nos experimentos estão congelados em `freeze_*.txt` (saída de `pip freeze`):

| Arquivo | Máquina | torch | vLLM | triton | transformers | peft | trl | flash-attn | liger-kernel |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| `freeze_hpc_h100_pucpr.txt` | H100 80GB, cluster Slurm (13/07/2026) | 2.10.0+cu128 | 0.18.1 | 3.6.0 | 4.57.6 | 0.19.1 | 1.7.0 | 2.8.3.post1 | 0.8.0 |
| `freeze_rhos_h200.txt` | H200 NVL (27/06/2026) | 2.10.0 | 0.18.1 | 3.6.0 | 4.57.6 | 0.19.1 | 1.7.0 | — (SDPA) | 0.8.0 |
| `freeze_rtx3060.txt` | RTX 3060, WSL2 (04/07/2026) | 2.10.0 | 0.18.1 | 3.6.0 | 4.57.6 | 0.18.1 | 1.0.0 | — (SDPA) | 0.7.0 |

Todos usam `bitsandbytes==0.49.2` e runtime CUDA 12.8. A GPU de cada treino está registrada no
`RESUMO_TREINO.md` da pasta do treino.

### Passo 0: Preparar o CUDA Toolkit (se `nvcc` não for encontrado)

Verifique com `which nvcc`.

**H100 / Linux nativo (CUDA 12.8):** instale no próprio ambiente conda:
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
conda remove -n luizbat01 --all   # se precisar remover o antigo
conda create -n luizbat01 python=3.12.11 -y
conda activate luizbat01
```

### Passo 2: Instalar o PyTorch (CUDA 12.8)
Instale a partir do index do PyTorch:
```bash
pip install "torch==2.10.0" torchvision "setuptools==80.10.2" "fsspec<=2026.4.0" --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

### Passo 3: Instalar o vLLM (evita quebra no CUDA 12.8 e FlashInfer)
```bash
pip install vllm==0.18.1 triton==3.6.0
```

### Passo 4: Instalar os pacotes adicionais
Em caso de erro de espaço no dispositivo, mude o diretório temporário:
```bash
TMPDIR=/var/tmp pip install -r ../src/requirements.txt
# ou, sem versões fixas:
TMPDIR=/var/tmp pip install -r ../src/requirements_sem_versao.txt
```
> **Alinhamento do pip:** se houver conflitos com numpy ou cuda-python, ajuste com
> `pip install "numpy<2.4.0" "cuda-python<13" "cuda-bindings<13" "cuda-pathfinder>=1.4.2"`.

### Passo 5: Compilar o Flash-Attn (opcional)
Desde o PyTorch 2.0, o SDPA nativo já cobre a economia de memória da atenção na maioria dos casos
(basta `flash_attention_2: true` no YAML de treino, com `full_com_sdpa: true` como fallback). Use
este passo para contextos longos, FA-3 na H100 ou features como sliding window.

**H100 / H200:**
```bash
TMPDIR=/var/tmp TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-deps --force-reinstall
```

**RTX 3060 em WSL2** (o instalador pode não identificar a GPU):
```bash
TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=1 pip install flash-attn --no-build-isolation --no-deps --force-reinstall
```

**Outras GPUs:**
```bash
TMPDIR=/var/tmp MAX_JOBS=8 pip install flash-attn --no-build-isolation --no-deps --force-reinstall
```

### Passo 6: Validar o ambiente
```bash
python ../src2/teste_ambiente.py
```
Verifica pacotes, versões, CUDA, GPU e backends SDPA, alertando sobre qualquer configuração
pendente.

---

## ⚡ Redução de Uso de VRAM

O treinamento usa **HuggingFace Transformers + PEFT**. Para memória, **Flash Attention** e
**Liger-Kernel** atuam de forma complementar:

### Flash Attention (SDPA / flash-attn)
Atua no bloco de atenção (`O(n)` de VRAM).
* **PyTorch SDPA:** nativo, sem instalação extra. Recomendado para RTX 3060 e fine-tuning local.
* **flash-attn:** pacote pip otimizado. Recomendado para contextos longos, H100 (FA-3) e multi-GPU.

### Liger-Kernel
Kernels Triton para as partes que *não* são atenção (RMSNorm, RoPE, SwiGLU,
FusedLinearCrossEntropy). Economiza VRAM significativamente.
* Sem flash-attn (SDPA) ou com 2+ GPUs (`device_map="auto"`), o framework desliga
  automaticamente a *Fused Cross Entropy* (evita `NaN` no `eval_loss` em 16 bits e *device
  mismatch*), mantendo os demais kernels.
* flash-attn + Liger juntos permitem o Full FT do Qwen 7B com contexto 32k em uma H200.

---

## 🆘 Solução de Problemas Comuns

- **"CUDA_HOME environment variable is not set" ao compilar flash-attn:**
  verifique `which nvcc` e exporte as variáveis: `export CUDA_HOME=...; export PATH=$CUDA_HOME/bin:$PATH`.
- **"undefined symbol: _ZN3c104cuda..." ao importar flash-attn:**
  o `flash-attn` foi compilado contra um CUDA diferente do PyTorch. Garanta que o `CUDA_HOME`
  reflete o mesmo CUDA do PyTorch e recompile (Passo 5).
- **"No space left on device":**
  passe `TMPDIR=/var/tmp` antes do comando pip.
- **vLLM ou FlashInfer sem achar `curand.h` no CUDA 12.8:**
  as versões instaladas via pip perdem o sufixo `-cu12`. Repasse o diretório para o compilador:
  `export CPATH="$(python -c 'import sys, glob; print(":".join(glob.glob(f"{sys.prefix}/lib/python*/site-packages/nvidia/*/include")))'):$CPATH"`
- **Loss `NaN`/`Inf` em Full FT:** a LR de Full FT precisa ser bem menor que a de LoRA (os
  experimentos usam `5e-06` em etapas `full` e `2e-05` em `lora`); `max_grad_norm: 0.3` ajuda em
  sequências longas.

---

## Execução em background (tmux / nohup)

### `tmux` (recomendado)
Leve, e a queda da conexão não derruba o script. No terminal do Jupyter há erros com as
configurações de cor e PATH padrão do tmux; o utilitário `setup_tmux.sh` encapsula a solução no
atalho `tm`.

```bash
source setup_tmux.sh     # configura o atalho no shell atual (uma vez)

tm new -s treino         # cria a sessão 'treino'
tm attach -t treino      # reconecta a uma sessão existente
tm ls                    # lista sessões
```
Para sair sem interromper (detach): `Ctrl+B`, solte, e depois `D`.

### `nohup` (alternativa)
```bash
nohup python ../../src/util_vllm_batch.py --config 05_extracao_b_teste.yaml &

tail -n 50 nohup.out     # últimas 50 linhas
tail -f nohup.out        # acompanhar em tempo real
```

No cluster, os jobs Slurm de cada pasta (`job_treinar_*.sh`, `job_export_*_testes.sh`,
`job_compara_testes.sh`) cumprem esse papel: `sbatch <job>.sh`.
