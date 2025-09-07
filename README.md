# LLMs 2025

-  Notebook com exemplo para predições: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luizanisio/llms/blob/main//Git_LuizAnisio_LLMs_GEMMA_exemplos_2025.ipynb)

-  Notebook com exemplo de treinamento: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luizanisio/llms/blob/main//ntb_treinamento/Passo_a_Passo_LLM_Fine_Tuning_2025.ipynb)
> 💡 copie o arquivo `puil_treinamento.txt` para o seu colab antes de iniciar o treinamento. Escolha o modelo de acordo com sua GPU no colab. A A100 é sempre a melhor escolha. A L4 dá para treinar modelos pequenos.

Pacotes em desenvolvimento para estudos com LLMs

- Desenvolvimento de pacotes para predição, treinamento e avaliação de LLMs (📦 [Diagramas](./diagramas.md) )
- Inicalmente o foco é o Gemma 3



## Treino do GEMMA 3
- esse código é mais completo, por linha de comando, para treinos mais longos
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
 
## Para importar as classes Util e UtilAnalise:
```python
#@title Importando classes do git
import get_git
# copia a pasta src do git para o content do colab
get_git.sync()
# para verificar e instalar dependências no colab
get_git.deps() 
```
 
Para testar a classe Util:
```python
from src.util import Util
lista = [1,2,3,[4,5,6],7,[8,9,[10,11]]]
Util.flatten_listas(lista)
```
 
Para testar a classe JsonAnalise:
```python
from src.util_json import JsonAnalise
JsonAnalise.teste_compara(exemplo=3)
```

## Classe Prompt com Gemma3
Para importar a classe Prompt:
```python
#@title Importando classes do git
# para verificar e instalar dependências no colab
# utilize unsloth = True caso deseje usar unsloth 
get_git.deps(unsloth=False) 

from src.util_prompt import Prompt
# carrega o modelo 1b, 4b, 12b ou 27b
# o padrão é não usar Unsloth
pr = Prompt('4b', usar_unsloth=False) 
pr.prompt('Qual o próximo número da sequência 1, 1, 2, 3, 5, 8 ...?')
```
```
O próximo número da sequência é 13.
Essa é a famosa sequência de Fibonacci, onde cada número é a soma dos dois números anteriores.
*   1 + 1 = 2
*   1 + 2 = 3
*   2 + 3 = 5
*   3 + 5 = 8
*   5 + 8 = 13
```

```python
_prompt_teste = '''Retorne um json válido com a estrutrua {"mensagem": com a mensagem do usuário, "itens": com uma lista de itens quando ele enumerar algo }
                   Mensagem do usuário: Eu preciso comprar abacaxi, pera e 2L de leite.'''
r = pr.prompt_to_json(_prompt_teste)
print(r)
```
```
{'mensagem': 'Eu preciso comprar abacaxi, pera e 2L de leite.', 
 'itens': ['abacaxi', 'pera', '2L de leite'], 
 'usage': {'input_tokens': 64, 'output_tokens': 60, 'time': 5.186101675033569}}
```

* Use GPU uma T4 suporta o 4B, para o 12B é melhor uma L4 e para o 27B use a A100
