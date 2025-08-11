import sys
from util_prompt import UtilLMM, Prompt
from dotenv import load_dotenv
import os

# Carrega as variáveis do arquivo .env se existirem
load_dotenv('./env')
token = os.getenv('hf')
if token:
    print('* Token do hugging face carregado!')

def imprimir_modelos_disponiveis():
  """Imprime a lista de modelos disponíveis."""
  print("Você deve especificar um modelo na linha de comando.")
  print("Exemplo de uso: python seu_script.py <nome_do_modelo>")
  
  UtilLMM.print_atalhos()
    
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
  
  modelo = sys.argv[1]
  
  p =  Prompt(modelo=modelo, max_seq_length = 4096, token = token)

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