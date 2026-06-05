import pandas as pd
arq = './saida/saida.parquet'
dados = pd.read_parquet(arq)
print(dados.head())
q = len(dados)

print('#'*80)
print(dados['resposta'][q-1])
print('-='*40)
print(dados['resumo'][q-1])