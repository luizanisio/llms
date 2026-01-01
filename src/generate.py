# -*- coding: utf-8 -*-

"""
Script para geração interativa de respostas com LLMs.

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Descrição:
-----------
Script de linha de comando para interação com modelos de linguagem locais.
Permite fazer perguntas e obter respostas de forma interativa.

Uso:
----
    python generate.py <nome_do_modelo>

Exemplo:
--------
    python generate.py gemma-3-4b-it
"""

import sys
from util_prompt import Prompt, UtilLLM
from util import UtilEnv
import os

# Carrega as variáveis do arquivo .env se existirem

token = None
if UtilEnv.carregar_env(pastas = ['./', '../']):
    token = UtilEnv.get_str('HF_TOKEN')
if token:
    print('* Token do hugging face CARREGADO!')
else:
    print('* Token do hugging face NÃO CARREGADO!')

def imprimir_modelos_disponiveis():
  """Imprime a lista de modelos disponíveis."""
  print("Você deve especificar um modelo na linha de comando.")
  print("Exemplo de uso: python seu_script.py <nome_do_modelo>")
  
  UtilLLM.print_atalhos()
    
def main():
  """
  Função principal que gerencia o fluxo do programa.
  """
  # sys.argv é uma lista que contém os argumentos da linha de comando.
  # sys.argv[0] é sempre o nome do script.
  # sys.argv[1] seria o primeiro argumento, e assim por diante.
  
  if len(sys.argv) < 2:
    imprimir_modelos_disponiveis()
    # Encerra o programa se nenhum modelo for fornecido
    return
  
  # avalia os parâmetros fornecidos, se o termo unsloth for encontrado, ativa o modo unsloth
  modelo = sys.argv[1]
  modo_unsloth = 'unsloth' in sys.argv[2:] if len(sys.argv) > 2 else False
  
  p =  Prompt(modelo=modelo, max_seq_length = 4096, usar_unsloth=modo_unsloth)

  while True:
    print('===========================================================')
    pergunta_usuario = input("Faça uma pergunta (ou pressione Enter para sair): ")
    
    if not pergunta_usuario:
      print("Saindo do programa. Até mais!")
      break
    
    resposta = p.prompt(pergunta_usuario)
    print('RESPOSTA:', resposta)

if __name__ == "__main__":
  main()