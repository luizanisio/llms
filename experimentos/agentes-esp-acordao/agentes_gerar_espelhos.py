
# -*- coding: utf-8 -*-
"""
Script para geração paralela de espelhos de acórdãos usando agentes especializados.

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/experimentos/agentes-esp-acordao
Data: 14/11/2025

Descrição:
-----------
Processa em paralelo um conjunto de acórdãos jurídicos (peças processuais) e gera
espelhos estruturados através do sistema de agentes orquestrados. Suporta diferentes
modelos LLM (GPT-5, Gemma-3) e mantém sessão de controle com estatísticas de execução.
"""

import pandas as pd
import os, sys, json

sys.path.extend(['./utils','./src','../../utils','../../src'])
from util import UtilEnv, UtilCriptografia, UtilArquivos
from agentes_orquestrador import AgenteOrquestradorEspelho
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
if not UtilEnv.carregar_env('.env', pastas=['../']):
    raise EnvironmentError('Não foi possível carregar o arquivo .env')

CRIPT = UtilCriptografia()
MODELO_ESPELHO = None
MODELO_ESPELHO_THINK = None
CALLABLE_RESPOSTA = None

sessao = {}

def print_sessao():
    global sessao
    _resumo = {f' - {k}: {v}' for k, v in sessao.items()}
    _resumo = ' | '.join(sorted(_resumo))
    _linha = '=' * 80
    print(f'\n{_linha}\nRESUMO SESSÃO: \n{_resumo}\n{_linha}')

def avaliar_parada_por_erro(erro):
    if isinstance(erro, str):
       _erro = erro.lower()
    elif isinstance(erro, dict) and len(erro) > 0:
         _erro = erro.get('erro','').lower() or erro.get('erros','').lower()
    if not erro:
        return
    if ('modelo' in _erro and 'não encontrado' in _erro) or \
        ('authorize' in _erro):
        msg = '|' * 60 + '\n'
        _erro = json.dumps(erro, ensure_ascii=False, indent=2) if isinstance(erro, dict) else str(erro)
        msg += f'| ERRO CRÍTICO: {_erro}\n'
        msg += '|' * 60 + '\n'
        print(msg)
        exit(1)

def gerar_respostas(row, pasta_extracao):
    global sessao
    id_peca = row['id_peca']
    #print(f'\n{"="*80}\nProcessando peça id_peca={id_peca}\n{"="*80}\n')
    try:
        texto = CRIPT.decriptografar(row['texto'])
        if not texto or len(texto) < 100:
            print(f'Texto da peça {id_peca} muito curto ou inexistente.')
            sessao['sem_texto'] = sessao.get('sem_texto', 0) + 1
            return
        #print('TEXTO PEÇA ""', texto[:500], '...')
        
        # Instancia o orquestrador com o texto, id_peca e pasta de observabilidade
        orq = AgenteOrquestradorEspelho(
            id_peca=id_peca, 
            texto_peca=texto,
            pasta_extracao=pasta_extracao,
            observabilidade=True,
            modelo_espelho=MODELO_ESPELHO,
            modelo_think=MODELO_ESPELHO_THINK,
            callable_modelo=CALLABLE_RESPOSTA
        )
        
        # Executa a orquestração
        espelho = orq.executar()
        erros = orq.get_mensagens_erro(espelho)
        #print('ESPELHO: ', json.dumps(espelho, ensure_ascii=False, indent=2))
        #print('ERROS: ', json.dumps(erros, ensure_ascii=False, indent=2))
        if espelho:
            if erros:
                sessao['com_erro'] = sessao.get('com_erro', 0) + 1
                avaliar_parada_por_erro(erros)
            else:
                campos_com_valor = {k: v for k, v in espelho.items() if v not in [None, '', [], {}]}
                campos_sem_valor = {k: v for k, v in espelho.items() if v in [None, '', [], {}]}
                sessao['cp_preenchidos'] = sessao.get('cp_preenchidos', 0) + len(campos_com_valor)
                sessao['cp_vazios'] = sessao.get('cp_vazios', 0) + len(campos_sem_valor)
            if espelho.get('carregado'):
               sessao['existentes'] = sessao.get('existentes', 0) + 1
            else:
               sessao['concluidos'] = sessao.get('concluidos', 0) + 1
        
        if (sessao.get('concluidos',0)+sessao.get('com_erro',0))  % 10 == 0:
            print_sessao()
    except Exception as e:
        print(f'Erro ao processar peça id_peca={id_peca}: {traceback.format_exc()}')


if __name__ == '__main__':
    #id_peca = ['202200038900.29.', '202200205729.40.']
    #id_peca = '202200205729.40.'
    id_peca = None
    
    # Define pasta de saída
    
    modelo_azure = True
    if modelo_azure:
        MODELO_ESPELHO = 'gpt5'
        MODELO_ESPELHO_THINK = 'l'
        from util_get_resposta import get_resposta
        CALLABLE_RESPOSTA = get_resposta
        PASTA_RAIZ = './saidas/'
        PASTA_EXTRACAO = os.path.join(PASTA_RAIZ, 'espelhos_agentes_gpt5/')
        DATAFRAME_ESPELHOS = os.path.join(PASTA_RAIZ, 'espelhos_acordaos_consolidado_textos.parquet')
    else:
        TAMANHO = '27b'
        MODELO_ESPELHO = f'or:google/gemma-3-{TAMANHO}-it:floor'
        MODELO_ESPELHO_THINK = 'low:low'
        from util_openai import get_resposta
        def def_resposta_router(*args, **kwargs):
            kwargs['silencioso'] = True
            return get_resposta(*args, **kwargs)
        CALLABLE_RESPOSTA = def_resposta_router
        PASTA_RAIZ = '/content/drive/MyDrive/TCC 2025 - Compartilhado no Drive/dados/'
        PASTA_EXTRACAO = os.path.join(PASTA_RAIZ, f'espelhos_agentes_gemma3_{TAMANHO}/')
        DATAFRAME_ESPELHOS = os.path.join(PASTA_RAIZ, 'espelhos_acordaos_consolidado_textos.parquet')
    os.makedirs(PASTA_EXTRACAO, exist_ok=True)

    df = pd.read_parquet(DATAFRAME_ESPELHOS)
    df = df[df['nomeOrgaoJulgador'].isin(['TERCEIRA SEÇÃO','TERCEIRA SEÃ‡ÃƒO','QUINTA TURMA','SEXTA TURMA'])]
    print('DataFrame carregado com ', len(df), 'peças para processamento.')
    if isinstance(id_peca, str) and id_peca.strip():
        df = df[df['id_peca'] == id_peca]
        print(f' - filtrado para id_peca={id_peca}, total de {len(df)} peças.')
    if isinstance(id_peca, list) and len(id_peca) > 0:
        df = df[df['id_peca'].isin(id_peca)]
        print(f' - filtrado para lista de id_peca, total de {len(df)} peças.')
        
    # Executar extrações em paralelo com threads
    NUM_THREADS = 20  # Ajuste se precisar
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(gerar_respostas, row, PASTA_EXTRACAO): row['id_peca'] for _, row in df.iterrows()}
        
        for future in tqdm(as_completed(futures), desc='Extraindo espelhos', ncols=60, total=len(df)):
            try:
                future.result()
            except Exception as e:
                id_peca = futures[future]
                raise Exception(f'Erro ao processar peça id_peca={id_peca}: {str(e)}\n{traceback.format_exc()}')
    print_sessao()   
    # for idx, row in df.iterrows():
    #     gerar_respostas(row, PASTA_EXTRACAO)
    #     print_sessao()

    # Carrega dados da peça
    lst = UtilArquivos.listar_arquivos(PASTA_EXTRACAO, mascara='*.json')
    lst = [arq for arq in lst if '.resumo.' not in arq]
    # Informa onde os arquivos foram salvos
    print('\n' + '='*80)
    print(f'ARQUIVOS GERADOS: {PASTA_EXTRACAO}')
    print(f' - Total de arquivos: {len(lst)}')
    print('='*80)
