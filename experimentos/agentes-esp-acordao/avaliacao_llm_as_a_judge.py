# -*- coding: utf-8 -*-
"""
Avaliação de extrações usando LLM-as-a-Judge com métricas de precisão e recall.

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/experimentos/agentes-esp-acordao
Data: 14/11/2025

Descrição:
-----------
Avalia qualidade das extrações de espelhos comparando com texto original do acórdão.
Utiliza GPT-5 como juiz para calcular precision, recall e F1-score das extrações
feitas por diferentes modelos (base, agentes, raw).
"""

import pandas as pd
import os, sys, json

sys.path.extend(['./utils','./src'])
from util import UtilEnv, UtilCriptografia, UtilArquivos
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from stjiautilbase.stj_openaia import STJOpenAIA
from stjiautilbase.stj_utilitarios import UtilArquivos, UtilTextos
from tqdm import tqdm
from prompt_espelho_agentes import PAPEL_LLM_AS_A_JUDGE, PROMPT_LLM_AS_A_JUDGE
if not UtilEnv.carregar_env('.env', pastas=['../']):
    raise EnvironmentError('Não foi possível carregar o arquivo .env')

CRIPT = UtilCriptografia()
MODELO_JUIZ = 'gpt5'
MODELO_JUIZ_THINK = 'm:l'
OA = STJOpenAIA()

sessao = {}

def print_sessao(linhas=True):
    global sessao
    _resumo = {f' - {k}: {v}' for k, v in sessao.items()}
    _resumo = ' | '.join(sorted(_resumo))
    _linha = '=' * 80
    if linhas:
        print(f'\n{_linha}\nRESUMO SESSÃO: \n{_resumo}\n{_linha}')
    else:
        print(f'RESUMO SESSÃO: {_resumo}')

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

def resposta_ok(arquivo):
    if not os.path.isfile(arquivo):
        return False
    r = UtilArquivos.carregar_arquivo(arquivo, juntar_linhas=True)
    r = UtilTextos.mensagem_to_json(r, padrao={})
    if len(r) >1 and 'precision' in r and 'recall' in r:
        return True
    return False

def gerar_respostas(row, pasta_extracao):
    global sessao
    id_peca = row['id_peca']
    #print(f'\n{"="*80}\nProcessando peça id_peca={id_peca}\n{"="*80}\n')
    try:
        arquivo_extracao = os.path.join(pasta_extracao, f'{id_peca}.json')
        arquivo_resposta = os.path.join(pasta_extracao, f'{id_peca}.avaliacao.json')
        arquivo_resposta_log = os.path.join(pasta_extracao, f'{id_peca}.avaliacao.log')
        if resposta_ok(arquivo_resposta):
            #print(f'Peça {id_peca} já possui avaliação, pulando ...')
            sessao['ja_avaliado'] = sessao.get('ja_avaliado', 0) + 1
            return
        texto = CRIPT.decriptografar(row['texto'])
        _linha = '=' * 60
        prompt= '<vazio>'
        if not texto or len(texto) < 100:
            print(f'Texto da peça {id_peca} muito curto ou inexistente.')
            sessao['sem_texto'] = sessao.get('sem_texto', 0) + 1
            avaliacao = {'erro': 'sem_texto', 'nota': 0, 'explicacao': 'Texto da peça muito curto ou inexistente.'}
        elif not os.path.isfile(arquivo_extracao):
            print(f'Arquivo de extração não encontrado para peça {id_peca}.')
            sessao['sem_extracao'] = sessao.get('sem_extracao', 0) + 1
            avaliacao = {'erro': 'sem_arquivo', 'nota': 0, 'explicacao': 'Arquivo de extração não encontrado'}
        else:
            extracao = UtilArquivos.carregar_arquivo(arquivo_extracao, juntar_linhas=True)
            extracao = UtilTextos.mensagem_to_json(extracao, padrao={})
            if len(extracao) == 0:
                print(f'Arquivo de extração vazio para peça {id_peca}.')
                return {'erro': 'sem_extracao', 'nota': 0, 'explicacao': 'Arquivo de extração vazio'}
                
            extracao = json.dumps(extracao, ensure_ascii=False, indent=2)    
            prompt = PROMPT_LLM_AS_A_JUDGE.replace('<--texto-->', texto).replace('<--extracao-->', extracao)
            #print('Prompt gerado para peça ', id_peca)
            #print('Prompt:', prompt[:500],'\n [...] \n', prompt[-500:])
            resposta = OA.prompt(prompt=prompt, papel = PAPEL_LLM_AS_A_JUDGE,
                                sg_modelo = MODELO_JUIZ, 
                                think=MODELO_JUIZ_THINK,
                                sem_erro=True, prompt_retorna_json=True,
                                retorno_resumido=True,
                                controle_aia={'~nolog~': True, 'projeto': 'llm as a judge espelho'},
                                temperature=0.0)
            avaliacao_txt = resposta.get('response', {})
            avaliacao = UtilTextos.mensagem_to_json(avaliacao_txt, padrao={})
            if 'erro' in resposta:
                print(f'{_linha}\nErro na resposta do LLM para peça {id_peca}: {resposta["erro"]}\n{_linha}\n')
                sessao['com_erro'] = sessao.get('com_erro', 0) + 1
                return
        '''
            print('AVALIAÇÃO resposta:', json.dumps(resposta, ensure_ascii=False, indent=2))
            print('AVALIAÇÃO TEXTO:', avaliacao_txt)
        print('AVALIAÇÃO JSON:', json.dumps(avaliacao, ensure_ascii=False, indent=2))
        exit()
        '''
        if len(avaliacao) == 0:
            print(f'\n{_linha}\nArquivo de avaliação inválido para peça {id_peca}.\n{avaliacao_txt}\n{_linha}\n')
            sessao['sem_avaliacao'] = sessao.get('sem_avaliacao', 0) + 1
            return {'erro': 'sem_avaliacao', 'nota': 0, 'explicacao': 'Arquivo de avaliação inválido'}
        # grava a resposta na mesma pasta
       
        _avaliacao_txt_dump = json.dumps(avaliacao, ensure_ascii=False, indent=2)
        with open(arquivo_resposta_log, 'w') as f:
            _log = f'PROMPT LLM AS A JUDGE:\n{prompt}\n{_linha}\n\nRESPOSTA LLM AS A JUDGE:\n{_avaliacao_txt_dump}'
            f.write(_log)
        with open(arquivo_resposta, 'w') as f:
            f.write(_avaliacao_txt_dump)
        sessao['com_sucesso'] = sessao.get('com_sucesso', 0) + 1
        return avaliacao
    except Exception as e:
        print(f'Erro ao processar peça id_peca={id_peca}: {traceback.format_exc()}')


if __name__ == '__main__':
    #id_peca = ['202200038900.29.', '202200205729.40.']
    #id_peca = '202200205729.40.'
    id_peca = None
    
    # Define pasta de saída
    QTD_LLM_AS_A_JUDGE = 300
    print(f'Quantidade de LLM as a Judge por peça: {QTD_LLM_AS_A_JUDGE}')
    PASTA_RAIZ = './saidas/'
    PASTAS_EXTRACAO = [
        'espelhos_agentes_gpt5/',
        'espelhos_agentes_gemma3_12b/',
        'espelhos_agentes_gemma3_27b/',
        'espelhos_base_gpt5/',
        'espelhos_base_gemma3_12b/',
        'espelhos_base_gemma3_27b/',
        'espelhos_raw/',
    ]
    PASTAS_EXTRACAO = [os.path.join(PASTA_RAIZ, p) for p in PASTAS_EXTRACAO]
    
    DATAFRAME_ESPELHOS = os.path.join(PASTA_RAIZ, 'espelhos_acordaos_consolidado_textos.parquet')
    assert os.path.isfile(DATAFRAME_ESPELHOS), f'Arquivo do DataFrame não encontrado: {DATAFRAME_ESPELHOS}'
    for p in PASTAS_EXTRACAO:
        assert os.path.isdir(p), f'Pasta de extração não encontrada: {p}'
    
    df = pd.read_parquet(DATAFRAME_ESPELHOS)
    df = df[df['nomeOrgaoJulgador'].isin(['TERCEIRA SEÇÃO','TERCEIRA SEÃ‡ÃƒO','QUINTA TURMA','SEXTA TURMA'])]
    
    if isinstance(id_peca, str) and id_peca.strip():
        df = df[df['id_peca'] == id_peca]
        print(f' - filtrado para id_peca={id_peca}, total de {len(df)} peças.')
    elif isinstance(id_peca, list) and len(id_peca) > 0:
        df = df[df['id_peca'].isin(id_peca)]
        print(f' - filtrado para lista de id_peca, total de {len(df)} peças.')
    else:
        df = df.sample(n=QTD_LLM_AS_A_JUDGE, random_state=42).reset_index(drop=True)
    #df=df[:1]
    print('DataFrame carregado com ', len(df), 'peças para processamento e avaliação LLM-AS-A-JUDGE.')
      
    for PASTA_EXTRACAO in PASTAS_EXTRACAO:
        print(f'\n{"#"*80}\nIniciando avaliações LLM as a Judge para extrações em: {PASTA_EXTRACAO}\n{"#"*80}\n')  
        # Executar extrações em paralelo com threads
        NUM_THREADS = 10  # Ajuste conforme necessário
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
        lst = [arq for arq in lst if '.avaliacao.' in arq]
        # Informa onde os arquivos foram salvos
        print('='*80)
        print_sessao(linhas = False)   
        print('.'*80)
        print(f'ARQUIVOS GERADOS: {PASTA_EXTRACAO}')
        print(f'IDS ANALISADOS: {len(df)}')
        print(f' - Total de arquivos: {len(lst)}')
        print('='*80)
        sessao = {}
