import pandas as pd
import json
import numpy as np
import sys
# TODO: pode receber o nome do arquivo de saída via args
if len(sys.argv) > 1:
    arq = sys.argv[1]
else:
    arq = './saida/saida_qwen7b.parquet'
arq_origem = './dados/integras_experimento_summa_novos.parquet'
dados = pd.read_parquet(arq)
dados_origem = pd.read_parquet(arq_origem)
print(dados.head())
q = len(dados)

# Buscar um registro com erro e um sem erro
reg_com_erro = None
reg_sem_erro = None

for _, row in dados.iterrows():
    # Verifica se há conteúdo de erro
    tem_erro = bool(row['erro'] and str(row['erro']).strip() not in ('None', 'nan', ''))
    if tem_erro and reg_com_erro is None:
        reg_com_erro = row
    elif not tem_erro and reg_sem_erro is None:
        reg_sem_erro = row
    
    if reg_com_erro is not None and reg_sem_erro is not None:
        break

def print_json(valor:str):
    """Imprime um valor string formatado como json"""
    try:
        _valor = valor.strip(' \t\n') if isinstance(valor,str) else valor
        print('JSON ✅', json.dumps(json.loads(valor), indent=4, ensure_ascii=False))
    except Exception as e:
        if len(valor) > 2000:
            print('JSON ❌', str(valor)[:2000], f'[..] {len(str(valor))} caracteres')
        else:
            print('JSON ❌', valor)

def exibir_registro(titulo_bloco, row):
    if row is None:
        print(f"\n{'='*80}\n--- NENHUM REGISTRO {titulo_bloco} ENCONTRADO ---\n{'='*80}")
        return
        
    print(f"\n{'='*80}")
    print(f"--- EXEMPLO DE REGISTRO {titulo_bloco} ---")
    print(f"{'='*80}")
    
    chave = row['chave']
    
    # Busca na origem
    itens = dados_origem[dados_origem['seq_documento_acordao'] == int(chave)]
    if len(itens) > 0:
        dados_item = itens.iloc[0]
        print(f"🎯 Documento: {chave} | Registro: {dados_item.get('num_registro', '')} | Classe: {dados_item.get('sg_classe', '')} | Publicação: {dados_item.get('dt_publicacao', '')}")
        print('-'*80)
        print("📄 ÍNTEGRA DA ORIGEM:")
        print(str(dados_item['integra']).replace('<br>', '\n'))
    else:
        print(f"🎯 Documento: {chave}")
        print('-'*80)
        print("📄 ÍNTEGRA DA ORIGEM: NÃO ENCONTRADA")
        
    print('-'*80)
    print("🤖 RESPOSTA DA EXTRAÇÃO:")
    print_json(row['resposta'])
    
    if row['erro'] and str(row['erro']).strip() not in ('None', 'nan', ''):
        print('-'*80)
        print("❌ ERRO DA EXTRAÇÃO:")
        print(str(row['erro']).strip(' \t\n')[:200])
        
    print('-'*80)
    print("📊 RESUMO (Tempos/Tokens):")
    print_json(row['resumo'])
    print(f"{'='*80}\n")

print('#'*80)
print(f'Total de registros lidos na saída: {q}')
print('#'*80)

exibir_registro("COM SUCESSO (SEM ERRO)", reg_sem_erro)
exibir_registro("COM FALHA (COM ERRO)", reg_com_erro)

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


print(f"Total de registros: {resumo['total']}")
print(f"JSON válidos: {resumo['json_validos']}")
print(f"JSON inválidos: {resumo['json_invalidos']}")
print(f"Tokens de entrada: {resumo['tokens_entrada']}")
print(f"Tokens de saída: {resumo['tokens_saida']}")
print(f"Tokens de entrada máximo: {resumo['tokens_entrada_max']}")
print(f"Tokens de saída máximo: {resumo['tokens_saida_max']}")
print(f"Tokens de entrada mínimo: {resumo['tokens_entrada_min']}")
print(f"Tokens de saída mínimo: {resumo['tokens_saida_min']}")
print(f"Métricas: {resumo['metricas']}")