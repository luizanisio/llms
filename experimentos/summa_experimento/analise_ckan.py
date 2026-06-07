'''
 Confere a saída gerada acionando manualmente o UtilCkan e a saída gerada pelo CLI após refatoração do projeto util_ckan.py.
'''

import pandas as pd
from tqdm import tqdm

# vamos considerar a chave seq_documento_acordao como chave principal dos registros 

# Vamos carregar o arquivo criado manualmente com o CKAN
arq_manual = './dados/integras_experimento_summa_novos.parquet'
df_manual = pd.read_parquet(arq_manual)

# vamos carregar o arquivo criado com o CLI do CKAN com os mesmos parâmetros
arq_cli = './dados/teste_integras_experimento_summa.parquet'
df_cli = pd.read_parquet(arq_cli)

# vamos identificar quantidade de seq_documento_acordao que tem integra vazia no arquivo manual 
print(f"Quantidade de seq_documento_acordao que tem integra vazia no arquivo manual: {len(df_manual[df_manual['integra'] == ''])}")

# vamos ignorar arquivos com integras vazias
df_manual = df_manual[df_manual['integra'] != ''].copy()
df_cli = df_cli[df_cli['integra'] != ''].copy()

# vamos listar quantidade de seq_documento_acordao que estão no manual e não estão no cli (ignorando os que não possuem íntegra)
lista_manual = set(df_manual['seq_documento_acordao'].unique())
lista_cli = set(df_cli['seq_documento_acordao'].unique())

lista_diff_manual = lista_manual - lista_cli
print(f"Quantidade de seq_documento_acordao que estão no manual e não estão no cli: {len(lista_diff_manual)}")


# vamos listar quantidade de seq_documento_acordao que estão no cli e não estão no manual (ignorando os que não possuem íntegra)
lista_diff_cli = lista_cli - lista_manual
print(f"Quantidade de seq_documento_acordao que estão no cli e não estão no manual: {len(lista_diff_cli)}")


# vamos verificar se há registros diferentes entre os dois arquivos mas com o mesmo seq_documento_acordao (seriam inconsistentes). Podemos ignorar a chave integra que é texto.
colunas = ['num_registro', 'sg_classe', 'ano', 'dt_publicacao']

# Fazemos um merge (inner join) pela chave principal para comparar os campos
df_comuns = pd.merge(
    df_manual[['seq_documento_acordao'] + colunas],
    df_cli[['seq_documento_acordao'] + colunas],
    on='seq_documento_acordao',
    suffixes=('_manual', '_cli')
)

# Identificamos onde há diferenças em qualquer uma das colunas comparadas
diff_mask = False
for col in colunas:
    diff_mask = diff_mask | (df_comuns[f'{col}_manual'] != df_comuns[f'{col}_cli'])

df_diferentes = df_comuns[diff_mask]

qtd_diferentes = len(df_diferentes)
qtd_iguais = len(df_comuns) - qtd_diferentes

print(f"Quantidade de registros iguais (interseção): {qtd_iguais}")
print(f"Quantidade de registros diferentes (interseção): {qtd_diferentes}")

##exit()
if qtd_diferentes > 0:
    print(f"\nRegistros com divergências nos metadados (amostra até 50/{qtd_diferentes}):")
    # Imprime todas as colunas para facilitar a comparação visual
    print(df_diferentes.head(50).to_string(index=False))

# Opcionalmente, mostramos alguns registros que não tiveram correspondência (apenas no manual)
if len(lista_diff_manual) > 0:
    print(f"\nRegistros apenas no manual (amostra até 50/{len(lista_diff_manual)}):")
    amostra_manual = df_manual[df_manual['seq_documento_acordao'].isin(lista_diff_manual)][['seq_documento_acordao'] + colunas]
    print(amostra_manual.head(50).to_string(index=False))

