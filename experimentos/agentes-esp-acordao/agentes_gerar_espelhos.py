
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
from threading import Lock
if not UtilEnv.carregar_env('.env', pastas=['../']):
    raise EnvironmentError('Não foi possível carregar o arquivo .env')

CRIPT = UtilCriptografia()
MODELO_ESPELHO = None
MODELO_ESPELHO_THINK = None
CALLABLE_RESPOSTA = None
ARQUIVO_LOG = None
LOCK_LOG = Lock()

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

def registrar_log_inconsistencia(id_peca, tipo_inconsistencia, detalhes=''):
    """Registra inconsistências no arquivo de log de forma thread-safe."""
    global ARQUIVO_LOG, LOCK_LOG
    if not ARQUIVO_LOG:
        return
    # inicia o arquivo caso id_peca seja None
    if id_peca is None:
        with LOCK_LOG:
            if not os.path.exists(ARQUIVO_LOG):
                with open(ARQUIVO_LOG, 'w', encoding='utf-8') as f:
                    f.write(f'Log de Inconsistências - Iniciado em {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        return
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    mensagem = f"[{timestamp}] {id_peca} | {tipo_inconsistencia}"
    if detalhes:
        mensagem += f" | {detalhes}"
    mensagem += "\n"
    
    with LOCK_LOG:
        with open(ARQUIVO_LOG, 'a', encoding='utf-8') as f:
            f.write(mensagem)

def gerar_respostas(row, pasta_extracao):
    global sessao
    id_peca = row['id_peca']
    #print(f'\n{"="*80}\nProcessando peça id_peca={id_peca}\n{"="*80}\n')
    try:
        texto = CRIPT.decriptografar(row['texto'])
        if not texto or len(texto) < 100:
            print(f'Texto da peça {id_peca} muito curto ou inexistente.')
            sessao['sem_texto'] = sessao.get('sem_texto', 0) + 1
            registrar_log_inconsistencia(id_peca, 'SEM_TEXTO', f'Tamanho: {len(texto) if texto else 0} caracteres')
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
            # Verifica se AgenteCampos não retornou nenhum campo (campos_identificados vazio)
            metadados = espelho.get('metadados', {})
            campos_identificados = metadados.get('campos_identificados', [])
            if not campos_identificados or len(campos_identificados) == 0:
                sessao['sem_campos'] = sessao.get('sem_campos', 0) + 1
                registrar_log_inconsistencia(id_peca, 'NENHUM_CAMPO_IDENTIFICADO', 'AgenteCampos não identificou campos para extração')
            elif erros:
                sessao['com_erro'] = sessao.get('com_erro', 0) + 1
                registrar_log_inconsistencia(id_peca, 'COM_ERRO', json.dumps(erros, ensure_ascii=False))
                avaliar_parada_por_erro(erros)
            else:
                campos_com_valor = {k: v for k, v in espelho.items() if v not in [None, '', [], {}]}
                campos_sem_valor = {k: v for k, v in espelho.items() if v in [None, '', [], {}]}
                sessao['cp_preenchidos'] = sessao.get('cp_preenchidos', 0) + len(campos_com_valor)
                sessao['cp_vazios'] = sessao.get('cp_vazios', 0) + len(campos_sem_valor)
                
                # Registra se todos os campos estão vazios
                if len(campos_com_valor) == 0:
                    registrar_log_inconsistencia(id_peca, 'TODOS_CAMPOS_VAZIOS', f'Total de campos: {len(campos_sem_valor)}')
            
            if espelho.get('carregado'):
               sessao['existentes'] = sessao.get('existentes', 0) + 1
            else:
               sessao['concluidos'] = sessao.get('concluidos', 0) + 1
        else:
            sessao['sem_espelho'] = sessao.get('sem_espelho', 0) + 1
            registrar_log_inconsistencia(id_peca, 'SEM_ESPELHO', 'Orquestrador não retornou espelho')
        
        if (sessao.get('excecoes',0)+sessao.get('concluidos',0)+sessao.get('existentes',0)+sessao.get('com_erro',0))  % 10 == 0:
            print_sessao()
    except Exception as e:
        sessao['excecoes'] = sessao.get('excecoes', 0) + 1
        erro_msg = str(e)
        registrar_log_inconsistencia(id_peca, 'EXCECAO', erro_msg[:200])
        print(f'Erro ao processar peça id_peca={id_peca}: {traceback.format_exc()}')

def teste_open_router():
    from util_openai import get_resposta
    prompt = "Responda 2 + 2 = ? no formato json: {'resposta': valor}"
    # modelos gratuitos nem sempre estão disponíveis
    modelo = 'or:google/gemma-3-3b-it:floor'
    modelo = 'or:google/gemma-3-27b-it:free'
    modelo = 'or:google/gemini-2.0-flash-exp:free'
    resposta = get_resposta(prompt, papel = 'responser rápido',
                            modelo=modelo, 
                            max_tokens=150,
                            as_json=True, silencioso=False)
    if not isinstance(resposta, dict):
        print('Resposta do modelo não está em formato JSON:', resposta)
        exit(1)
    if ('error' in resposta):
        print('Erro retornado pelo modelo:', resposta)
        exit(1)
    if not resposta.get('resposta'):
        print('Resposta do modelo não contém o campo "resposta":', resposta)
        exit(1)
    print('Resposta do modelo:', resposta.get('resposta'))
    exit(0)

if __name__ == '__main__':
    #teste_open_router()
    #id_peca = ['202200038900.29.', '202200205729.40.']
    #id_peca = '202200205729.40.'
    id_peca = None

    # Executar extrações em paralelo com threads
    NUM_THREADS = 10  # Ajuste se precisar

    # Define pasta de saída
    
    modelo_azure = False
    if modelo_azure:
        MODELO_ESPELHO = 'gpt5'
        MODELO_ESPELHO_THINK = 'l'
        from util_get_resposta import get_resposta
        CALLABLE_RESPOSTA = get_resposta
        PASTA_RAIZ = './saidas/'
        PASTA_EXTRACAO = os.path.join(PASTA_RAIZ, 'espelhos_agentes_gpt5/')
        DATAFRAME_ESPELHOS = os.path.join(PASTA_RAIZ, 'espelhos_acordaos_consolidado_textos.parquet')
    else:
        NUM_THREADS = 3
        TAMANHO = '12b' # 12b ou 27b
        MODELO_ESPELHO = f'or:google/gemma-3-{TAMANHO}-it:floor'#:floor free nitro'
        MODELO_ESPELHO_THINK = 'low:low'
        from util_openai import get_resposta
        def def_resposta_router(*args, **kwargs):
            kwargs['silencioso'] = True
            return get_resposta(*args, **kwargs)
        CALLABLE_RESPOSTA = def_resposta_router
        #PASTA_RAIZ = '/content/drive/MyDrive/TCC 2025 - Compartilhado no Drive/dados/'
        PASTA_RAIZ = './saidas/'
        PASTA_EXTRACAO = os.path.join(PASTA_RAIZ, f'espelhos_agentes_gemma3_{TAMANHO}/')
        DATAFRAME_ESPELHOS = os.path.join(PASTA_RAIZ, 'espelhos_acordaos_consolidado_textos.parquet')
    os.makedirs(PASTA_EXTRACAO, exist_ok=True)

    # arquivo de log
    ARQUIVO_LOG = os.path.join(PASTA_EXTRACAO, 'log_inconsistencias.txt')
    registrar_log_inconsistencia(None, None)  # inicia o arquivo de log

    # Carrega DataFrame com peças a processar
    df = pd.read_parquet(DATAFRAME_ESPELHOS)
    df = df[df['nomeOrgaoJulgador'].isin(['TERCEIRA SEÇÃO','TERCEIRA SEÃ‡ÃƒO','QUINTA TURMA','SEXTA TURMA'])]
    print('DataFrame carregado com ', len(df), 'peças para processamento.')
    if isinstance(id_peca, str) and id_peca.strip():
        df = df[df['id_peca'] == id_peca]
        print(f' - filtrado para id_peca={id_peca}, total de {len(df)} peças.')
    if isinstance(id_peca, list) and len(id_peca) > 0:
        df = df[df['id_peca'].isin(id_peca)]
        print(f' - filtrado para lista de id_peca, total de {len(df)} peças.')
        
    # identifica peças únicas para processamento
    print(f'Identificando peças únicas para processamento de {len(df)} peças...')
    pecas_unicas = list(set(df['id_peca']))
    pecas = []
    for p in pecas_unicas:
        primeira_ocorrencia = df[df['id_peca'] == p].iloc[0]
        pecas.append(primeira_ocorrencia)
    df_unicas = pd.DataFrame(pecas)
    print(f'Total de peças únicas para processamento: {len(df_unicas)}')

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        
        futures = {executor.submit(gerar_respostas, row, PASTA_EXTRACAO): row['id_peca'] for _, row in df_unicas.iterrows()}
        
        for future in tqdm(as_completed(futures), desc='Extraindo espelhos', ncols=60, total=len(df_unicas)):
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
    print(f' - Total de peças analisadas: {len(df)}')
    print(f' - Total de peças únicas analisadas: {len(df_unicas)}')
    print('='*80)
