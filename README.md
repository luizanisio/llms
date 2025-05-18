# LLMs
Pacotes em desenvolvimento para estudos com LLMs

## Em construção 
- Desenvolvimento de pacotes para predição, treinamento e avaliação de LLMs

## Notebook
- Para carregar as classes no colab ou jupyter
- copiar a pasta util com o arquivo get_git.py para a pasta do notebook do jupyter ou colab
-  copiar apara uma célula e rodar:
```python
#@title Importando classes do git
from util.get_git import sync_git_util
Util = sync_git_util()
```

Para testar:
```python
lista = [1,2,3,[4,5,6],7,[8,9,[10,11]]]
Util.flatten_listas(lista)
```
