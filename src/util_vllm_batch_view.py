import os
import json
import numpy as np

def print_json(valor: str):
    """Imprime um valor string formatado como json"""
    try:
        _valor = valor.strip(' \t\n') if isinstance(valor, str) else valor
        print('JSON ✅', json.dumps(json.loads(_valor), indent=4, ensure_ascii=False))
    except Exception:
        if len(str(valor)) > 2000:
            print('JSON ❌', str(valor)[:2000], f'[..] {len(str(valor))} caracteres')
        else:
            print('JSON ❌', valor)

def _buscar_origem(chave: str, itens_origem: list) -> str:
    for item in itens_origem:
        if str(item.get('chave', '')) == str(chave):
            return str(item.get('texto', ''))
    return "NÃO ENCONTRADA"

def exibir_registro(titulo_bloco: str, row: dict, itens_origem: list):
    if row is None:
        print(f"\n{'='*80}\n--- NENHUM REGISTRO {titulo_bloco} ENCONTRADO ---\n{'='*80}")
        return
        
    print(f"\n{'='*80}")
    print(f"--- EXEMPLO DE REGISTRO {titulo_bloco} ---")
    print(f"{'='*80}")
    
    chave = row.get('chave', '')
    
    # Busca na origem
    print(f"🎯 Documento: {chave}")
    print('-'*80)
    print("📄 ÍNTEGRA DA ORIGEM:")
    texto_origem = _buscar_origem(chave, itens_origem)
    print(texto_origem.replace('<br>', '\n'))
        
    print('-'*80)
    print("🤖 RESPOSTA DA EXTRAÇÃO:")
    print_json(row.get('resposta', ''))
    
    if row.get('erro') and str(row['erro']).strip() not in ('None', 'nan', ''):
        print('-'*80)
        print("❌ ERRO DA EXTRAÇÃO:")
        print(str(row['erro']).strip(' \t\n')[:200])
        
    print('-'*80)
    print("📊 RESUMO (Tempos/Tokens):")
    resumo_str = row.get('resumo', '{}')
    if isinstance(resumo_str, dict):
        resumo_str = json.dumps(resumo_str)
    print_json(resumo_str)
    print(f"{'='*80}\n")

def visualizar_saida_config(config: dict):
    """Lê os resultados processados e os dados de origem para exibir exemplos"""
    # Importação local para evitar dependência circular se necessário
    from util_vllm_batch import _saida_eh_parquet, carregar_entrada
    
    print(f"📂 Carregando entrada: {config['entrada']['arquivo']}")
    try:
        itens_origem = carregar_entrada(config)
        print(f"   📊 {len(itens_origem)} item(ns) de entrada carregado(s)")
    except Exception as e:
        print(f"⚠️ Erro ao carregar entrada para visualização: {e}")
        itens_origem = []

    arquivo_saida = config["saida"]["arquivo"]
    eh_parquet = _saida_eh_parquet(config)
    
    registros = []
    print(f"\n📂 Lendo saída: {arquivo_saida}")
    
    if eh_parquet:
        try:
            import pandas as pd
            if os.path.exists(arquivo_saida):
                df = pd.read_parquet(arquivo_saida)
                for _, row in df.iterrows():
                    registros.append({
                        'chave': str(row.get('chave', '')),
                        'resposta': str(row.get('resposta', '')),
                        'erro': str(row.get('erro', '')),
                        'resumo': str(row.get('resumo', '{}'))
                    })
        except Exception as e:
            print(f"⚠️ Erro ao ler parquet de saída: {e}")
    else:
        if os.path.isdir(arquivo_saida):
            for nome in os.listdir(arquivo_saida):
                if nome.endswith('.txt'):
                    chave = nome.replace('.txt', '')
                    txt_path = os.path.join(arquivo_saida, nome)
                    json_path = os.path.join(arquivo_saida, f"{chave}.json")
                    
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            resposta = f.read()
                    except Exception:
                        resposta = ""
                        
                    erro = ""
                    resumo = "{}"
                    try:
                        if os.path.exists(json_path):
                            with open(json_path, 'r', encoding='utf-8') as f:
                                dados_json = json.load(f)
                                erro = str(dados_json.get('erro', ''))
                                # remove as chaves que não são de resumo
                                resumo_dict = {k:v for k,v in dados_json.items() if k not in ['chave', 'erro']}
                                resumo = json.dumps(resumo_dict)
                    except Exception:
                        pass
                        
                    registros.append({
                        'chave': chave,
                        'resposta': resposta,
                        'erro': erro,
                        'resumo': resumo
                    })
                    
    if not registros:
        print(f"⚠️ Nenhum registro encontrado na saída.")
        return
        
    q = len(registros)
    
    reg_com_erro = None
    reg_sem_erro = None
    
    for row in registros:
        tem_erro = bool(row.get('erro') and str(row['erro']).strip() not in ('None', 'nan', ''))
        if tem_erro and reg_com_erro is None:
            reg_com_erro = row
        elif not tem_erro and reg_sem_erro is None:
            reg_sem_erro = row
            
        if reg_com_erro is not None and reg_sem_erro is not None:
            break
            
    print('#'*80)
    print(f'Total de registros lidos na saída: {q}')
    print('#'*80)

    exibir_registro("COM SUCESSO (SEM ERRO)", reg_sem_erro, itens_origem)
    exibir_registro("COM FALHA (COM ERRO)", reg_com_erro, itens_origem)

    resumo = {
        'total': q, 
        'json_validos': 0, 
        'json_invalidos': 0, 
        'tokens_entrada': 0, 
        'tokens_saida': 0, 
        'tokens_entrada_max': 0, 
        'tokens_saida_max': 0, 
        'tokens_entrada_min': np.inf, 
        'tokens_saida_min': np.inf
    }

    # Verifica se a saída deveria ser json
    tipo_saida = config.get("saida", {}).get("tipo_saida", "str")
    
    for item in registros:
        erro_item = item.get('erro', '')
        tem_erro = bool(erro_item and str(erro_item).strip() not in ('None', 'nan', ''))
        
        if tem_erro:
            resumo['json_invalidos'] += 1
        else:
            if tipo_saida == "json":
                try:
                    json.loads(item['resposta'])
                    resumo['json_validos'] += 1
                except Exception:
                    resumo['json_invalidos'] += 1
            else:
                # Se não é json exigido e não tem erro, consideramos "válido" no contexto de sucesso
                resumo['json_validos'] += 1
                
        # Tokens
        try:
            r_json = json.loads(item['resumo'])
            pt = int(r_json.get('prompt_tokens', r_json.get('input_tokens', 0)))
            ct = int(r_json.get('completion_tokens', r_json.get('output_tokens', 0)))
            
            if pt > 0 or ct > 0:
                resumo['tokens_entrada'] += pt
                resumo['tokens_saida'] += ct
                resumo['tokens_entrada_max'] = max(pt, resumo['tokens_entrada_max'])
                resumo['tokens_saida_max'] = max(ct, resumo['tokens_saida_max'])
                resumo['tokens_entrada_min'] = min(pt, resumo['tokens_entrada_min'])
                resumo['tokens_saida_min'] = min(ct, resumo['tokens_saida_min'])
        except Exception:
            pass

    if resumo['tokens_entrada_min'] == np.inf:
        resumo['tokens_entrada_min'] = 0
    if resumo['tokens_saida_min'] == np.inf:
        resumo['tokens_saida_min'] = 0

    print(f"Total de registros: {resumo['total']}")
    if tipo_saida == "json":
        print(f"JSON válidos (sucesso): {resumo['json_validos']}")
        print(f"JSON inválidos (erro): {resumo['json_invalidos']}")
    else:
        print(f"Processados com sucesso: {resumo['json_validos']}")
        print(f"Processados com erro: {resumo['json_invalidos']}")
        
    print(f"Tokens de entrada: {resumo['tokens_entrada']}")
    print(f"Tokens de saída: {resumo['tokens_saida']}")
    print(f"Tokens de entrada máximo: {resumo['tokens_entrada_max']}")
    print(f"Tokens de saída máximo: {resumo['tokens_saida_max']}")
    print(f"Tokens de entrada mínimo: {resumo['tokens_entrada_min']}")
    print(f"Tokens de saída mínimo: {resumo['tokens_saida_min']}")

if __name__ == '__main__':
    print('Módulo para visualização de saídas exportadas.')
    print('Utilize: python util_vllm_batch.py --config config_batch.yaml --view')
