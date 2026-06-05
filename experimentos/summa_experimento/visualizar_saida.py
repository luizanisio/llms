import pandas as pd
import json
import numpy as np
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

# Imprime resumo de total de erros ou json não válido, total com json válido, métricas e token de entrada e saída
# TODO: finalizar implementação da impressão do resumo
resumo = {'total':q, 'json_validos':0, 'json_invalidos':0, 'tokens_entrada':0, 'tokens_saida':0, 'tokens_entrada_max':0, 'tokens_saida_max':0, 
          'tokens_entrada_min': np.inf, 'tokens_saida_min':np.inf, 'metricas':{}}

for item in dados.itertuples():
    if item.erro:
        resumo['json_invalidos'] += 1
        ok = False
    else:
        try:
            j = json.loads(item.resposta)
            resumo['json_validos'] += 1
            ok = True
        except Exception as e:
            resumo['json_invalidos'] += 1
            ok = False
    if ok:
        usage = json.loads(item.resumo)
        resumo['tokens_entrada'] += usage['prompt_tokens']
        resumo['tokens_saida'] += usage['completion_tokens']
        resumo['tokens_entrada_max'] = max(usage['prompt_tokens'],resumo['tokens_entrada_max'])
        resumo['tokens_saida_max'] = max(usage['completion_tokens'],resumo['tokens_saida_max'])
        resumo['tokens_entrada_min'] = min(usage['prompt_tokens'],resumo['tokens_entrada_min'])
        resumo['tokens_saida_min'] = min(usage['completion_tokens'],resumo['tokens_saida_min'])


print(f'Total de registros: {resumo['total']}')
print(f'JSON válidos: {resumo['json_validos']}')
print(f'JSON inválidos: {resumo['json_invalidos']}')
print(f'Tokens de entrada: {resumo['tokens_entrada']}')
print(f'Tokens de saída: {resumo['tokens_saida']}')
print(f'Tokens de entrada máximo: {resumo['tokens_entrada_max']}')
print(f'Tokens de saída máximo: {resumo['tokens_saida_max']}')
print(f'Tokens de entrada mínimo: {resumo['tokens_entrada_min']}')
print(f'Tokens de saída mínimo: {resumo['tokens_saida_min']}')
print(f'Métricas: {resumo['metricas']}')