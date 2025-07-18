# LLMs 2025
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
O dataframe tem que ter uma coluna com as mensagens no formato:
```json
[{"role": "user", "content": "prompt qualquer"},{"role": "assistant","content":"resposta qualquer"}]
```
- o nome da coluna é informado no yaml no parâmetro: train_prompt_col
- ao final do treino, será gerado um arquivo metrics_summary.json (train/eval loss final do treino) e metrics_summary.json (train/eval loss por steps)
- se configurado save_checkpoints = true, os checkpoints serão gravados na pasta chkpt do modelo
 
## Notebook
- Alguns utilitários que estão sendo desenvolvidos e podem ser aproveitados nos colabs de estudo de forma simples, mantendo os códigos centralizados.
- Para carregar as classes no colab ou jupyter:
```
!curl https://raw.githubusercontent.com/luizanisio/llms/refs/heads/main/util/get_git.py -o ./get_git.py
```
 
Para importar a classe Util:
```python
#@title Importando classes do git
import get_git
Util = get_git.sync_git_util()
JsonAnalise = get_git.sync_git_json_analise()
```
 
Caso esteja em uma subpasta ou pasta específica, defina o path de destino:
```python
#@title Importando classes do git em pasta definida
import get_git
Util = get_git.sync_git_util('/teste')
JsonAnalise = get_git.sync_git_json_analise('/teste')
```
 
Para testar a classe Util:
```python
lista = [1,2,3,[4,5,6],7,[8,9,[10,11]]]
Util.flatten_listas(lista)
```
 
Para testar a classe JsonAnalise:
```python
JsonAnalise.teste_compara(exemplo=3)
```
