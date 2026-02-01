import sys
import os
sys.path.append("../../src")
import pandas as pd
# Import da nova classe de configuração YAML
from treinar_unsloth_util import YamlTreinamento, TIPO_ENTRADA_PASTAS
from util import UtilEnv, UtilCriptografia

# carrega o env
util_env = UtilEnv()
util_env.carregar_env(pastas = ['./','../../src'])

# carrega o yaml
yaml_treinamento = YamlTreinamento("treinar_qwen15.yaml")

# identifica o arquivo de entrada
arquivo_parquet = ""
coluna_texto = ""

if yaml_treinamento.tipo_entrada == TIPO_ENTRADA_PASTAS:
    if yaml_treinamento.pastas and yaml_treinamento.pastas.entrada.dataframe:
        arquivo_parquet = yaml_treinamento.pastas.entrada.dataframe
        coluna_texto = yaml_treinamento.pastas.entrada.dataframe_col
else:
    # modo dataset
    arquivo_parquet = yaml_treinamento.dataset.train_file
    coluna_texto = yaml_treinamento.dataset.train_prompt_col

if not arquivo_parquet:
    print("Arquivo parquet não identificado na configuração.")
    sys.exit(1)

print(f"Lendo: {arquivo_parquet}")
dataset = pd.read_parquet(arquivo_parquet)

# Verifica criptografia
if yaml_treinamento.formatos.env_chave_criptografia:
    chave = os.getenv(yaml_treinamento.formatos.env_chave_criptografia)
    if chave:
        print(f"Descriptografando coluna '{coluna_texto}'...")
        # Configura a chave para o UtilCriptografia (que usa CHAVE_CRIPT por padrão)
        os.environ['CHAVE_CRIPT'] = chave 
        cripto = UtilCriptografia()
        
        def _decrypt(x):
            try:
                return cripto.decriptografar(str(x))
            except Exception as e:
                return f"[Erro Decript: {e}] {str(x)[:20]}..."

        if coluna_texto in dataset.columns:
            dataset[coluna_texto] = dataset[coluna_texto].apply(_decrypt)
    else:
        print(f"Aviso: Variável {yaml_treinamento.formatos.env_chave_criptografia} não encontrada.")

# imprime os primeiros dados
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 50)
print(dataset.head())

if coluna_texto in dataset.columns and len(dataset) > 0:
    print("\n--- Exemplo de Texto (primeiros 500 caracteres) ---")
    print(dataset.iloc[0][coluna_texto][:500])
