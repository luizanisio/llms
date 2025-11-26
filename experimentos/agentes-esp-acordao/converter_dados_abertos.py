# -*- coding: utf-8 -*-
"""
Conversor de dados abertos de espelhos para formato padronizado.

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/experimentos/agentes-esp-acordao
Data: 14/11/2025

Descrição:
-----------
Converte espelhos extraídos de dados abertos do STJ para formato JSON padronizado
compatível com as extrações feitas pelos agentes, facilitando comparações.
"""

from codecs import escape_encode
from stjiautilbase.stj_utilitarios import UtilEnv, Util, UtilArquivos, UtilTextos, UtilDataHora, UtilTipos
from stjiautilbase.stj_openaia import STJOpenAIA

import pandas as pd
import os, sys

sys.path.extend(['./utils','./src'])
from tqdm import tqdm
import json
from prompt_espelho_base import PROMPT_BASE_SJR_S3_JSON, PROMPT_USER

'''
Converte os dados abertos extraídos em espelhos no formato similar ao gerado pelos agentes.
Salva os espelhos convertidos na pasta espelhos_raw
'''

UtilEnv.carregar_env('.env', pastas=['../'])
from util_espelhos import DATAFRAME_RESUMIDO, PASTA_SAIDAS_RAW, carregar_espelhos

if __name__ == '__main__':
    df = pd.read_csv(DATAFRAME_RESUMIDO)
    # terceira seção e quintas e sextas turmas
    df = df[df['nomeOrgaoJulgador'].isin(['TERCEIRA SEÇÃO','TERCEIRA SEÃ‡ÃƒO','QUINTA TURMA','SEXTA TURMA'])]
    ids = df['id_espelho'].unique().tolist()
    espelhos = carregar_espelhos(set(ids))
    print(f'Foram carregados {len(espelhos)} espelhos.')
    campos = ['jurisprudenciaCitada','notas','informacoesComplementares', 'termosAuxiliares', 'teseJuridica', 'tema', 'referenciasLegislativas']
    os.makedirs(PASTA_SAIDAS_RAW, exist_ok=True)
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Convertendo espelhos'):
        id_peca = row['id_peca']
        arquivo = os.path.join(PASTA_SAIDAS_RAW, f'{id_peca}.json')    
        id_espelho = UtilTipos.to_int(row['id_espelho'])
        espelho = espelhos.get(id_espelho)
        espelho = {c:v for c,v in espelho.items() if c in campos} if espelho else {}
        with open(arquivo, 'w', encoding='utf-8') as f:
            json.dump(espelho, f, ensure_ascii=False, indent=2)
