from stjiautilbase.stj_utilitarios import UtilEnv, Util, UtilArquivos, UtilTextos, UtilDataHora, UtilTipos
from stjiautilbase.stj_openaia import STJOpenAIA

import pandas as pd
import os, sys

sys.path.append('../')
from util_cript import UtilCriptografia
from tqdm import tqdm
import json
from threading import Lock
from time import time

LOCK_SESSAO = Lock()
PASTA_SAIDAS_EXTRACAO='./saidas/espelhos_base/'
PASTA_SAIDAS_RAW = './saidas/espelhos_raw/'
PASTA_ESPELHOS = './saidas/downloads_esp_stj/'
DATAFRAME_ESPELHOS = './saidas/espelhos_acordaos_consolidado_textos.parquet'
DATAFRAME_RESUMIDO = './saidas/espelhos_acordaos_consolidado_resumido.csv'
DATAFRAME_RESUMIDO_CAMPOS = './saidas/espelhos_acordaos_consolidado_resumido_campos.csv'
CRIPT = UtilCriptografia()
ARQ_DF = os.path.join(PASTA_SAIDAS_EXTRACAO,'dados_extracao_base.csv')
ARQ_LOGS = os.path.join(PASTA_SAIDAS_EXTRACAO,'log_extracao_base.txt')

DF_TEXTOS = None
CRIPT = UtilCriptografia()

def get_df_textos():
    global DF_TEXTOS
    if DF_TEXTOS is None:
        if os.path.exists(DATAFRAME_ESPELHOS):
            DF_TEXTOS = pd.read_parquet(DATAFRAME_ESPELHOS)
        else:
            raise FileNotFoundError(f'Arquivo {DATAFRAME_ESPELHOS} não encontrado.')
    return DF_TEXTOS

def get_texto_peca(id_peca: str):
    df = get_df_textos()
    print('Buscando texto para id_peca:', id_peca, 'colunas disponíveis:', df.columns.tolist(), 'linhas:', df.shape[0])
    row = df[df['id_peca'] == id_peca]
    if row.empty:
        return None
    texto = row.iloc[0]['texto']
    return CRIPT.decriptografar(texto)

def carregar_espelhos(ids: set):
    # lista os arquivos json da pasta PASTA_ESPELHOS
    espelhos = dict()
    for arquivo in tqdm(os.listdir(PASTA_ESPELHOS), desc='Carregando espelhos'):
        if not arquivo.endswith('.json'):
            continue
        caminho_arquivo = os.path.join(PASTA_ESPELHOS, arquivo)
        try:
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                conteudo = json.load(f)
            # mantém apenas os espelhos com id_espelho na lista ids
            for espelho in conteudo:
                id_espelho = UtilTipos.to_int(espelho.get('id'), default=0)
                if id_espelho in ids:
                    espelhos[id_espelho] = espelho
        except Exception as e:
            print(f'Erro ao carregar arquivo {caminho_arquivo}: {e}')
            continue
    return espelhos

def carregar_espelho_gerado(id_peca: int):
    caminho_arquivo = os.path.join(PASTA_SAIDAS_EXTRACAO, f'{id_peca}.json')
    if not os.path.exists(caminho_arquivo):
        return None
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            conteudo = json.load(f)
        return conteudo
    except Exception as e:
        print(f'Erro ao carregar arquivo {caminho_arquivo}: {e}')
        return None

def carregar_texto(id_peca: int):
    caminho_arquivo = os.path.join(PASTA_SAIDAS_EXTRACAO, f'texto_{id_peca}.txt')
    if not os.path.exists(caminho_arquivo):
        return None
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            conteudo = f.read()
        return conteudo
    except Exception as e:
        print(f'Erro ao carregar arquivo {caminho_arquivo}: {e}')
        return None

