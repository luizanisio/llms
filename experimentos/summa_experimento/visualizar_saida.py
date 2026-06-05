import pandas as pd
arq = './saida/saida.parquet'
arq_origem = './dados/integras_experimento_summa.parquet'
dados = pd.read_parquet(arq)
dados_origem = pd.read_parquet(arq_origem)
print(dados.head())
q = len(dados)

print('#'*80)
print(f'Total de registros: {q}')
print('#'*80)
print('Documento:', dados['chave'][q-1])
print('-='*40)
print('RESPOSTA:')
print(dados['resposta'][q-1])
print('-='*40)
print('RESUMO:')
print(dados['resumo'][q-1])
print('-='*40)
# busca o registro com seq_documento_acordao igual à chave do documento na origem
chave = dados['chave'][q-1]
itens = dados_origem[dados_origem['seq_documento_acordao'] == int(chave)]
if len(itens) >  0:
    print('INTEGRA:')
    print(itens['integra'].values[0].replace('<br>','\n'))
else:    
    print('INTEGRA NÃO ENCONTRADA')

print('-='*40)
