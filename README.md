# LLMs
Pacotes em desenvolvimento para estudos com LLMs

## Em construção 
- Desenvolvimento de pacotes para predição, treinamento e avaliação de LLMs

## Treino do GEMMA 3
- src/treinar_gemma3.py \<arquivo yaml\> \[--gpu n\]
- exemplo de arquivo yaml para o treino
```yaml
dataset_train_path: ../dataset/df_treino.parquet
train_prompt_col: messages
base_model_name: google/gemma-3-1b-pt
output_dir: ../modelos/gemma3_1b
batch_size: 3
grad_batch_size: 5
num_train_epochs: 1
max_seq_length: 4096
lora_r: 8
dataset_eval_path: ../dataset/df_teste.parquet
eval_prompt_col: messages
eval_steps: 10%
save_checkpoints: true
```

## Notebook
- Alguns utilitários que estão sendo desenvolvidos e podem ser aproveitados nos colabs de estudo de forma simples, mantendo os códigos centralizados.
- Para carregar as classes no colab ou jupyter:
  - copiar a pasta util com o arquivo get_git.py para a pasta do notebook do jupyter ou colab

Para importar a classe Util:
```python
#@title Importando classes do git
from util.get_git import sync_git_util
Util = sync_git_util()
```

Caso esteja em uma subpasta, ajustar o path do util:
```python
#@title Importando classes do git em subpasta
import sys, os
sys.path.append(os.path.abspath(".."))  # sobe um nível na árvore

from util.get_git import sync_git_util
Util = sync_git_util(dest_root='../')
```

Para testar a classe Util:
```python
lista = [1,2,3,[4,5,6],7,[8,9,[10,11]]]
Util.flatten_listas(lista)
```
