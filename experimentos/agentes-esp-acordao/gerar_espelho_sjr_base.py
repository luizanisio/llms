import sys
# -*- coding: utf-8 -*-
"""
Geração de espelhos usando abordagem base (prompt único).

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/experimentos/agentes-esp-acordao
Data: 14/11/2025

Descrição:
-----------
Gera espelhos de acórdãos usando abordagem tradicional com prompt único e extenso,
sem divisão em agentes especializados. Serve como baseline para comparação com
abordagem multi-agentes.
"""

sys.path.extend(['../','./','./src'])
from util import UtilEnv, Util, UtilArquivos, UtilTextos, UtilDataHora, UtilCriptografia
# caso não consiga usar o STJOpenAIA, usa util_openai.py
try:
    from stjiautilbase.stj_openaia import STJOpenAIA
except ImportError:
    from util_openai import get_resposta
    class STJOpenAIA:
        def prompt(self, **kwargs):
            if 'sg_modelo' in kwargs:
               kwargs['modelo'] = kwargs.pop('sg_modelo','')
            if 'prompt_retorna_json' in kwargs:
              kwargs['as_json'] = kwargs.pop('prompt_retorna_json')
            kwargs = {c:v for c,v in kwargs.items() if c not in {'retorno_resumido','controle_aia','sem_erro'}}
            kwargs['silencioso'] = True
            res = get_resposta(**kwargs)
            res['tratada'] = True
            return res
import pandas as pd
import os, sys

from tqdm import tqdm
import json
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
import traceback

from prompt_espelho_base import PROMPT_BASE_SJR_S3_JSON, PROMPT_USER

UtilEnv.carregar_env('.env', pastas=['../'])

LOCK_SESSAO = Lock()
CRIPT = UtilCriptografia()
PASTA_SAIDAS_EXTRACAO=None
PASTA_RAIZ=None
DATAFRAME_ESPELHOS =None
ARQ_DF =None
ARQ_LOGS =None

def configurar_pasta(pasta_raiz,pasta_extracao):
  global PASTA_SAIDAS_EXTRACAO, DATAFRAME_ESPELHOS, ARQ_DF, ARQ_LOGS, PASTA_RAIZ
  PASTA_RAIZ=pasta_raiz
  PASTA_SAIDAS_EXTRACAO=os.path.join(pasta_raiz, pasta_extracao)
  DATAFRAME_ESPELHOS = os.path.join(pasta_raiz,'espelhos_acordaos_consolidado_textos.parquet')
  ARQ_DF = os.path.join(PASTA_SAIDAS_EXTRACAO,'dados_extracao_base.csv')
  ARQ_LOGS = os.path.join(PASTA_SAIDAS_EXTRACAO,'log_extracao_base.txt')
  os.makedirs(PASTA_RAIZ, exist_ok=True)
  os.makedirs(PASTA_SAIDAS_EXTRACAO, exist_ok=True)


OA = STJOpenAIA()
SESSAO = {}
LOG = []
ERROS_CONSECUTIVOS = 0

MODELO_ESPELHO = 'GPT5' 
MODELO_ESPELHO_THINK = 'l'


print( '====================================================================')
print(f'CONEXÕES PREPARADAS, analisando pasta {PASTA_SAIDAS_EXTRACAO} ....')
print( '====================================================================')

def log(txt:str):
    global LOG
    msg_log = UtilDataHora.data_hora_str() + '\t' + str(txt)
    LOG.append(msg_log)
    with open(ARQ_LOGS, 'a') as f:
        f.write(f'{msg_log}\n')
    

def soma_sessao(tipo:str):
    global SESSAO
    with LOCK_SESSAO:
         if tipo not in SESSAO:
            SESSAO[tipo] = 1
         else:
            SESSAO[tipo] += 1
    return SESSAO[tipo]

def print_resumo():
    print('----------------------------------------------------------------------')
    print('RESUMO:', ' || '.join([f'{c}={v}' for c,v in SESSAO.items()]), f' || LOGS: {len(LOG)}')
    print('----------------------------------------------------------------------')

def get_extracao(row, somente_verificar = False):
    global SESSAO, ERROS_CONSECUTIVOS
    texto = CRIPT.decriptografar(row['texto'])
    arquivo_saida = os.path.join(PASTA_SAIDAS_EXTRACAO, f'{row["id_peca"]}.json')
    arquivo_resumo = os.path.join(PASTA_SAIDAS_EXTRACAO, f'{row["id_peca"]}_resumo.json')
    if UtilArquivos.tamanho_arquivo(arquivo_saida) > 0:
        if not somente_verificar:
           soma_sessao('existem')
        return True
    if somente_verificar:
        return False
    prompt_user = PROMPT_USER.replace('<<--texto-->>', texto)
    ini = time()
    messages = [
        {"role": "system", "content": PROMPT_BASE_SJR_S3_JSON},
        {"role": "user", "content": prompt_user}]
    try:
        espelho_res = OA.prompt(prompt = messages, 
                            sg_modelo=MODELO_ESPELHO, papel='', 
                            think = MODELO_ESPELHO_THINK,
                            sem_erro=True, 
                            prompt_retorna_json=True,
                            temperature=0, 
                            retorno_resumido=True,
                            controle_aia={'projeto': 'analise_espelho', '~nolog~': True})
        ERROS_CONSECUTIVOS = 0
    except Exception as e:
        ERROS_CONSECUTIVOS += 1
        print(f'ERRO: {e}\n{traceback.format_exc()}')
        Util.pausa(5)
        soma_sessao('erro-api')
        log(f'Erro gerando espelho id={row["id_peca"]} >> {traceback.format_exc()}')
        return False
    if 'erro' in espelho_res:
        soma_sessao('erro-espelho')
        log(f'Erro gerando espelho id={row["id_peca"]} >> {espelho_res}')
        return False
    try:
        tratada = espelho_res.pop('tratada',False)
        if 'erro' in espelho_res:
           espelho = {'erro': espelho_res['erro']}
        elif not tratada:
           resposta = espelho_res.get('response','')
           espelho = UtilTextos.mensagem_to_json(resposta, padrao=resposta)
        else:
           espelho = espelho_res.get('resposta')
        
    except Exception as e:
        soma_sessao('erro-json')
        log(f'Erro convertendo espelho em json id={row["id_peca"]} >> {traceback.format_exc()}')
        return False
    usage = espelho_res.get('usage',{})
    resumo = {  "input_tokens": usage.get('prompt_tokens'),
                "output_tokens": usage.get('completion_tokens'),
                "reasoning_tokens": usage.get('completion_tokens_details',{}).get('reasoning_tokens'),
                "cached_tokens": usage.get('prompt_tokens_details',{}).get('cached_tokens'),
                "finish_reason": espelho_res.get('finish_reason'),
                "model": MODELO_ESPELHO,
                "think": MODELO_ESPELHO_THINK,
                "time": time()-ini,
                "id_peca": row['id_peca'],
                "model_id": espelho_res.get('model')
                }
        
    with open(arquivo_saida, 'w') as f:
         if isinstance(espelho, str):
            f.write(espelho)
         else:
            f.write(json.dumps(espelho, indent=2, ensure_ascii=False))                            
    with open(arquivo_resumo, 'w') as f:
         f.write(json.dumps(resumo, indent=2, ensure_ascii=False))
    soma_sessao('RESUMO CRIADO')
    return True
    
CAMPOS_RESUMO_DF = {'id_peca','id_espelho','sg_ramo_direito','sg_classe','num_ministro','ano','nomeOrgaoJulgador',
                    'input_tokens','output_tokens', 'reasoning_tokens','cached_tokens',
                    'finish_reason','model','think','time','erro'}
def get_resumo_espelho(row):
    ''' O resumo é obrigatoriamente gerado junto com a extração e no formato JSON.
        Dados pode não ser json se houver erro na geração do espelho.
    '''
    arquivo_dados = os.path.join(PASTA_SAIDAS_EXTRACAO, f'{row["id_peca"]}.json')
    arquivo_resumo = os.path.join(PASTA_SAIDAS_EXTRACAO, f'{row["id_peca"]}_resumo.json')
    if not os.path.isfile(arquivo_resumo):
        return {}
    try:
        resumo = UtilArquivos.carregar_json(arquivo_resumo)
    except Exception as e:
        raise Exception(f'Erro lendo resumo {arquivo_resumo}: {e}')
    try:
        dados = UtilArquivos.carregar_json(arquivo_dados)
    except Exception as e:
        dados = {'erro': f'Erro {e}'}
    # campos para o resumo
    res = {c:v for c,v in dados.items() if c in CAMPOS_RESUMO_DF}
    resumo = {c:v for c,v in resumo.items() if c in CAMPOS_RESUMO_DF}
    linha = {c:v for c,v in dict(row).items() if c in CAMPOS_RESUMO_DF}
    res.update(resumo)
    res.update(linha)
    resumo['time'] = round(resumo.get('time',0),2)
    # Retorna também uma estatística simples do conteúdo extraído
    res['qtd_teses'] = len(dados.get('teseJuridica',[])) if isinstance(dados.get('teseJuridica',[]), list) else 0
    res['qtd_jurisprudencias'] = len(dados.get('jurisprudenciaCitada',[])) if isinstance(dados.get('jurisprudenciaCitada',[]), list) else 0
    res['qtd_referencias'] = len(dados.get('referenciasLegislativas',[])) if isinstance(dados.get('referenciasLegislativas',[]), list) else 0
    res['qtd_notas'] = len(dados.get('notas',[])) if isinstance(dados.get('notas',[]), list) else 0
    res['qtd_info_complementares'] = len(dados.get('informacoesComplementares',[])) if isinstance(dados.get('informacoesComplementares',[]), list) else 0
    res['qtd_termos_auxiliares'] = len(dados.get('termosAuxiliares',[])) if isinstance(dados.get('termosAuxiliares',[]), list) else 0
    res['qtd_temas'] = len(dados.get('tema',[])) if isinstance(dados.get('tema',[]), list) else 0
    return res

if __name__ == '__main__':
    
    #MODELO_ESPELHO = 'or:openai/gpt-oss-20b' 
    MODELO_ESPELHO = 'or:google/gemma-3-27b-it'
    MODELO_ESPELHO_THINK = 'l'

    configurar_pasta(pasta_raiz ='/content/drive/MyDrive/TCC 2025 - Compartilhado no Drive/dados',
                     pasta_extracao='espelho_base_gemma3_32' )

    assert os.path.isdir(PASTA_SAIDAS_EXTRACAO), 'Pasta de saída não existe!'
    #ao = STJOpenAIA()
    #r = ao.prompt(prompt = 'Responda aleatoriamente sim ou não!',modelo=MODELO_ESPELHO, think=MODELO_ESPELHO_THINK)
    #print(r)
    #exit()
    print(f'Carregando dataframe: {DATAFRAME_ESPELHOS}')
    df = pd.read_parquet(DATAFRAME_ESPELHOS)
    assert os.path.isfile(DATAFRAME_ESPELHOS), 'Dataframe de espelhos não existe!'
    # filtra para a Terceira Seção / Quinta e Sexta Turmas
    df = df[df['nomeOrgaoJulgador'].isin(['TERCEIRA SEÇÃO','TERCEIRA SEÃ‡ÃƒO','QUINTA TURMA','SEXTA TURMA'])]
    #df = df[:10]
    tempo_resumo = time()

    existem = []
    for i, row in tqdm(df.iterrows(), desc = 'Verificando peças com extração', ncols = 60, total=len(df)):
        if get_extracao(row, somente_verificar=True):    
           existem.append(row['id_peca'])
    total = len(df)
    df_extrair = df[~df['id_peca'].isin(existem)]
    print(f'Ignorando {len(existem)}/{total} registros com extração encontrada')

    # Executar extrações em paralelo com threads
    NUM_THREADS = 3  # Ajuste conforme necessário
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(get_extracao, row): row['id_peca'] for _, row in df_extrair.iterrows()}
        
        for future in tqdm(as_completed(futures), desc='Extraindo espelhos', ncols=60, total=len(df_extrair)):
            try:
                future.result()
            except Exception as e:
                id_peca = futures[future]
                log(f'Erro na thread para id={id_peca}: {traceback.format_exc()}')

    df_consolidado = []
    for i, row in tqdm(df.iterrows(), desc = 'Consolidando resumos', ncols = 60, total=len(df)):
        resumo = get_resumo_espelho(row)
        df_consolidado.append(resumo)
        print('|' * 60)
        print(resumo)
    df_consolidado = pd.DataFrame(df_consolidado)
    df_consolidado.to_csv(os.path.join(PASTA_SAIDAS_EXTRACAO,'espelhos_acordaos_resumo_extracao.csv'), index=False, encoding='utf-8-sig')
    
    print_resumo()
    
    print('\n####################\nFIM')
    if any(LOG):
       print(f'LOGS: {len(LOG)} Erros e/ou Avisos gravados em "{ARQ_LOGS}"')
