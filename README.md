# LLMs 2025

Este repositório contém pacotes e utilitários para treinamento, avaliação, uso e comparação de Large Language Models (LLMs), com foco inicial na família Gemma, Qwen, Deepseek e Llama.

## 📁 Estrutura do Projeto (`src/`)

Os scripts principais de execução estão organizados na pasta `src/`:

- **Treinamento (`treinar_unsloth.py`)**: Realiza fine-tuning de modelos usando Unsloth e arquivos YAML de configuração. Suporta Curriculum Learning (múltiplas etapas com metas customizadas de Loss ou Épocas), LoRA e checkpoints automáticos. 👉 **[Ler Documentação Completa](./src/treinar_unsloth.md)**.
- **Avaliação (`treinar_unsloth_avaliar.py`)**: Gera estatísticas de tokens, valida arquivos divisões e orquestra execuções de validação pontuais e em lote de inferência. 👉 **[Ler Documentação Completa](./src/treinar_unsloth.md)**.
- **Predição (`treinar_realizar_predicoes.py`)**: Motor unificado de inferência com CLI desacoplado suportando otimizações avançadas em HuggingFace, vLLM, API Local do Unsloth e API do Ollama.
- **Comparação de Extrações (`comparar_extracoes.py`)**: Ferramenta para comparar e avaliar extrações de diferentes modelos contra um gabarito humano, suportando ROUGE, SBERT e Levenshtein, a fim de gerar divisões de treino baseadas em dificuldade calculada. 👉 **[Ler Documentação Completa](./src/comparar_extracoes.md)**.

## 🚀 Guias e Planejamentos (Para Pesquisadores e Desenvolvedores)

- 📝 **[Guia de Contribuição e Futuras Funcionalidades](./src/treinar_TODO_PLANEJAMENTO.md)**: Documentação central sobre como estender o código, limitações e comportamento anormais conhecidos dos motores integrados (e.g. exceções em Tensor Parallelism Multi-GPU com vLLM) e os próximos desenvolvimentos pretendidos.

---

## 🛠️ Instalação Básica e Guias de Referência em Notebooks (Colab)

-  Notebook com exemplo para predições: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luizanisio/llms/blob/main//Git_LuizAnisio_LLMs_GEMMA_exemplos_2025.ipynb)

-  Notebook com exemplo para treinamento: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luizanisio/llms/blob/main//ntb_treinamento/Passo_a_Passo_LLM_Fine_Tuning_2025.ipynb)
> 💡 Escolha o modelo de acordo com sua GPU no colab. A A100 é sempre a melhor escolha. A L4 dá para treinar modelos pequenos.<br>

- Notebook com exemplo de treinamento com **Unsloth** - amplia o leque de treinamento em GPUs com menos memória: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luizanisio/llms/blob/main//ntb_treinamento/Passo_a_Passo_Unsloth_LLM_Fine_Tuning_2025.ipynb)


> 💡 **Instalação manual do unsloth.** Consultar: [Unsloth Notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks)
```python
#@title Instalando Unsloth (Colab)
import os, re
from IPython.display import clear_output
try:
  import unsloth
  print("✅ Unsloth e vllm OK _o/")
  import transformers
  print("✅ Transformers OK _o/")
except ImportError as e:
    clear_output()
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    import torch; v = re.match(r"[0-9\.]{3,}", str(torch.__version__)).group(0)
    xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")
    !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth
    !pip install transformers==4.55.4
    !pip install --no-deps trl==0.22.2
    clear_output()
    import unsloth
    print('✅ Instalação do Unsloth e Transformers concluídas _o/')
```

## Para importar utilitários nos Colabs:
Alguns utilitários que estão sendo desenvolvidos podem ser aproveitados nos colabs de estudo de forma simples, mantendo os códigos do repositório em sincronia remota.
```python
#@title Importando classes do git
!curl https://raw.githubusercontent.com/luizanisio/llms/refs/heads/main/util/get_git.py -o ./get_git.py
import get_git
get_git.sync() # copia a pasta src do git para o content do colab
get_git.deps() # para verificar e instalar dependências no colab
```

Exemplo básico usando a infraestrutura do repositório instanciada:
```python
from src.util_prompt import Prompt
pr = Prompt('4b', usar_unsloth=False) # carrega modelo Gemma local/Colab
pr.prompt('Qual o próximo número da sequência 1, 1, 2, 3, 5, 8 ...?')
```
