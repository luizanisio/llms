import sys

from numpy import isin
sys.path.append('../')

import os
import requests
import pandas as pd
import json
import pandas as pd
import regex as re
import traceback
from stjiautilbase.stj_utilitarios import UtilEnv, UtilArquivos
from stjiautildb.stj_oracle import Oracle_Conexao
from stjiautildb.stj_singlestore import SingleStore_Conexao
from tqdm import tqdm
from util_cript import UtilCriptografia

if not UtilEnv.carregar_env():
   print('Não foi possível carregar o arquivo .env')
   exit()
DB = Oracle_Conexao()
DBSS = SingleStore_Conexao()
CRIPT = UtilCriptografia()
# Configurações
CKAN_BASE_URL = 'https://dadosabertos.web.stj.jus.br'
DATASET_IDs = [('S3','espelhos-de-acordaos-terceira-secao'),
               ('S2','espelhos-de-acordaos-segunda-secao'),
               ('S1','espelhos-de-acordaos-primeira-secao'),
               ('T1','espelhos-de-acordaos-primeira-turma'),
               ('T2','espelhos-de-acordaos-segunda-turma'),
               ('T3','espelhos-de-acordaos-terceira-turma'),
               ('T4','espelhos-de-acordaos-quarta-turma'),
               ('T5','espelhos-de-acordaos-quinta-turma'),
               ('T6','espelhos-de-acordaos-sexta-turma'),
               ('CE','espelhos-de-acordaos-corte-especial')]
DOWNLOAD_DIR = 'downloads_esp_stj'
TOTAL_JSON = 0
print('==========================================================')
print('Carregando lista de peças ...')
ARQ_SUMMA = os.path.join(DOWNLOAD_DIR,'pecas_exportadas.csv')
if not os.path.isfile(ARQ_SUMMA):
   ARQ_SUMMA = './saidas/pecas_exportadas.csv'
if not os.path.isfile(ARQ_SUMMA):
    print(f'ARQUIVO SUMMA "pecas_exportadas.csv" não encontrado em "{DOWNLOAD_DIR}" ou "./saidas')   
    exit()
print('==========================================================')
print(f'CARREGANDO dados de {ARQ_SUMMA}')    
DF_SUMMA = pd.read_csv(ARQ_SUMMA)
print(f' - lista de peças carregada com {len(DF_SUMMA)} peças')
DOCUMENTOS = set(DF_SUMMA['seq_documento_acordao'])
PROCESSOS = set(DF_SUMMA['num_registro'])
print('==========================================================')
# Certifique-se de que o diretório de download existe
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Função para obter os metadados do dataset
def obter_metadados_dataset(dataset_id):
    url = f'{CKAN_BASE_URL}/api/3/action/package_show'
    params = {'id': dataset_id}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()['result']

# Função para baixar um recurso
def baixar_recurso(orgao_julgador, resource):
    global TOTAL_JSON
    nome_arquivo = f"{orgao_julgador}_{resource['name']}"
    url_recurso = resource['url']
    caminho_arquivo = os.path.join(DOWNLOAD_DIR, nome_arquivo)
    
    # ignora json
    if not str(nome_arquivo).lower().endswith('.json'):
        return
    
    print(f'Baixando {nome_arquivo}...')
    resposta = requests.get(url_recurso, stream=True)
    resposta.raise_for_status()
    if not os.path.isfile(caminho_arquivo):
        TOTAL_JSON += 1
        with open(caminho_arquivo, 'wb') as f:
            for chunk in resposta.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f'{nome_arquivo} salvo em {caminho_arquivo}')
    
    print(f'Analisando {nome_arquivo}')

# Execução principal
def main():
    for orgao_julgador, dataset_id in DATASET_IDs:
        print(f'Obtendo metadados de {orgao_julgador} do dataset "{dataset_id}"...')
        metadados = obter_metadados_dataset(dataset_id)
        recursos = metadados['resources']

        if not recursos:
            print('Nenhum recurso encontrado no dataset.')
            return
        
        for recurso in recursos:
            try:
                if recurso.get('format','').lower() == 'json':
                   baixar_recurso(orgao_julgador, recurso)
            except Exception as e:
                print(f'Erro ao baixar {recurso["name"]}: {e}')        

def get_texto_acordao(id_peca, pasta, criptografado=False):
    ''' Retorna o texto do acórdão a partir do id_peca e da pasta onde está o JSON '''
    _pasta = pasta.replace('<<tipo>>','texto_publico')
    arquivo = os.path.join(_pasta, f'{id_peca}.txt')
    if not os.path.isfile(arquivo):
        print(f'Arquivo {arquivo} não encontrado.')
        return ''
    txt = open(arquivo, 'r', encoding='utf-8').read()
    if not txt:
        txt = open(arquivo, 'r', encoding='latin-1').read()
    txt=txt.replace('<br>','\n').strip(' \n\r\t')
    if not criptografado:
        return txt
    txt = CRIPT.criptografar(txt)
    return txt

def get_seq_documento_acordao(num_registro, data_publicacao):
    ''' Extrai o seq_documento_acordao do texto do espelho '''
    global DB
    with DB.bloco_autocommit() as db:
        sql = '''
        SELECT seq_documento
          FROM db2sa.documento_stj
         WHERE num_rg_doc_stj = :num_registro AND cod_tipo_doc = 5
           AND dt_publicacao = TO_DATE(:data_publicacao, 'YYYY-MM-DD')
        '''
        resultado = db.get_valor(sql, {'num_registro': num_registro, 'data_publicacao': data_publicacao}, default=None)
        if resultado is None:
            return 
    return resultado

def get_principal_sucessivo(seq_documento_acordao):
    ''' Retorna o seq_documento_acordao principal se for um acórdão sucessivo '''
    global DBSS
    with DBSS.bloco_autocommit() as db:
        sql = '''select TIPO from ejuris.documentos d where seq_documento = %(seq_documento_acordao)s'''
        resultado = db.get_valor(sql, {'seq_documento_acordao': seq_documento_acordao}, default=None)
    return resultado 

COLUNAS_IMPORTAR = {"siglaClasse", "descricaoClasse", "nomeOrgaoJulgador", "ministroRelator", 
                    "tipoDeDecisao", "dataDecisao", "jurisprudenciaCitada", "notas", 
                    "informacoesComplementares", "termosAuxiliares", "teseJuridica", 
                    "tema", "referenciasLegislativas", "acordaosSimilares"}
COLUNAS_RESUMIDAS = {"siglaClasse", "descricaoClasse", "nomeOrgaoJulgador", "ministroRelator", 
                    "tipoDeDecisao", "dataDecisao"}
def consolidar_saidas():
    ''' Consolida os arquivos JSON baixados em um único arquivo CSV
        contendo as colunas importadas e identificando o id_peca do dataset Summa com base no 
        num_registro e data_publicacao
    '''
    print('Consolidando arquivos JSON baixados em um único CSV ...')
    arquivos_json = [f for f in os.listdir(DOWNLOAD_DIR) if f.lower().endswith('.json')]
    dados_consolidados = []
    DF_SUMMA['dt_publicacao'] = [str(d).strip() for d in DF_SUMMA['dt_publicacao']]
    DF_SUMMA['num_registro'] = [str(n).strip() for n in DF_SUMMA['num_registro']]
    id_peca_acordao = {s['id_peca']: s['seq_documento_acordao'] for s in DF_SUMMA.to_dict(orient='records')}
    registros = set(DF_SUMMA['num_registro'])
    dados_registros_espelho = []
    for arquivo in tqdm(arquivos_json):
        caminho_arquivo = os.path.join(DOWNLOAD_DIR, arquivo)
        try:
            dados = UtilArquivos.carregar_json(caminho_arquivo)
        except Exception as e:
            print(f'Erro ao ler {caminho_arquivo}: {e}')
            conteudo_texto = open(caminho_arquivo, 'r', encoding='utf-8').read()
            if 'sem lançamentos para o mês' in conteudo_texto.lower():
                print('Arquivo contém mensagem de sem lançamentos para o mês. Ignorando erro.')
                continue
            raise
            
        def _formatar_valor_(valor):
            if isinstance(valor,str):
                return valor.replace('\n',' ').replace('\r',' ').strip()
            elif isinstance(valor,float):
                return round(valor, 3)
            return valor
        # percorre os dados e extrai as informações necessárias
        for item in tqdm(dados, desc=f'Processando {arquivo}'):
            num_registro = str(item.get('numeroRegistro')).strip() if item.get('numeroRegistro') else None
            if not num_registro:
                continue
            if num_registro not in registros:
                # não está na lista de esperados pelo SUMMA
                continue
            data_publicacao = item.get('dataPublicacao')
            # procura a data no texto de dataPublicacao
            dts = re.findall(r'(\d{2}/\d{2}/\d{4})', str(data_publicacao))
            if not any(dts):
                continue
            data_publicacao = dts[0]
            # converte para AAAA-MM-DD
            dt_formatada = re.sub(r'(\d{2})/(\d{2})/(\d{4})', r'\3-\2-\1', data_publicacao) if data_publicacao else None
            #print('Procurando id_peca para', num_registro, dt_formatada)
            #if any(_ for _ in DF_SUMMA['num_registro'] if str(_).strip() == num_registro):
            #    print(f'Número de registro {num_registro} encontrado na lista de peças do SUMMA.')
            #    print(f'Data procurada: {dt_formatada} >>  Datas no SUMMA: ', list(DF_SUMMA[DF_SUMMA['num_registro'] == num_registro]['dt_publicacao'].unique()))
            #    exit()
            # print('Datas no SUMMA: ', list(DF_SUMMA['dt_publicacao'].unique())[:10])
            # exit()
            seq_documento_acordao = None
            
            if dt_formatada:
                seq_documento_acordao = get_seq_documento_acordao(num_registro, dt_formatada)

            # dados para espelhos de registros no SUMMA mas não encontrados no SUMMA pela data de publicação
            _dados_espelho = {c:_formatar_valor_(v) for c,v in item.items() if c in COLUNAS_RESUMIDAS}
            _dados_espelho['id_espelho'] = item.get('id')
            _dados_espelho['seq_documento_acordao'] = seq_documento_acordao
            _dados_espelho['dt_formatada'] = dt_formatada
            
            if seq_documento_acordao is None:
                # registra os não encontrados
                dados_registros_espelho.append(_dados_espelho)
                continue # só tenta pelo seq_documento_acordao

            # filtra o dataframe do SUMMA para encontrar o id_peca
            if seq_documento_acordao:
                df_filtrado = DF_SUMMA[DF_SUMMA['seq_documento_acordao'] == seq_documento_acordao]
            else:
                df_filtrado = DF_SUMMA[(DF_SUMMA['num_registro'] == num_registro) & 
                                    (DF_SUMMA['dt_publicacao'] == dt_formatada)]
            if not df_filtrado.empty:
                summa = df_filtrado.to_dict(orient='records')[0]
                summa.update({c:v for c,v in item.items() if c in COLUNAS_IMPORTAR})  # adiciona os dados do espelho
                summa['id_espelho'] = item.get('id') # id para id_espelho
                summa['texto'] = get_texto_acordao(summa['id_peca'], summa['pasta'], criptografado=True)
                if not summa['texto']:
                    print(f'ATENÇÃO: Texto do acórdão não encontrado para id_peca {summa["id_peca"]} em {summa["pasta"]}')
                    exit()
                dados_consolidados.append(summa)
    if len(dados_consolidados) == 0:
        print('\n\nATENÇÃO: Nenhum dado consolidado encontrado.\n')
        return
    df_consolidado = pd.DataFrame(dados_consolidados)
    arquivo_saida = os.path.join('saidas', 'espelhos_acordaos_consolidado_textos.parquet')
    os.makedirs('saidas', exist_ok=True)
    df_consolidado.to_parquet(arquivo_saida, index=False)
    print(f'Arquivo consolidado salvo em {arquivo_saida} com {len(df_consolidado)} registros.')
    print('-' *60)
    print('Gravando colunas resumidas em CSV ...')
    arquivo_saida_csv = os.path.join('saidas', 'espelhos_acordaos_consolidado_resumido.csv')
    colunas_resumidas = list(DF_SUMMA.columns) + list(COLUNAS_RESUMIDAS) + ['id_espelho']
    df_resumido = df_consolidado[[c for c in colunas_resumidas if c in df_consolidado.columns]]
    df_resumido.to_csv(arquivo_saida_csv, index=False, encoding='utf-8-sig')
    print(f'Arquivo resumido salvo em {arquivo_saida_csv} com {len(df_resumido)} registros.')
    print('-' *60)
    # verifica os ids_peca que não foram encontrados
    ids_peca_encontrados = set(df_consolidado['id_peca'].dropna().unique())
    ids_peca_totais = set(DF_SUMMA['id_peca'].dropna().unique())
    ids_peca_nao_encontrados = ids_peca_totais - ids_peca_encontrados
    print(f'Total de id_peca no SUMMA: {len(ids_peca_totais)}')
    print(f'Total de id_peca encontrados nos espelhos: {len(ids_peca_encontrados)}')
    print(f'Total de id_peca NÃO encontrados nos espelhos: {len(ids_peca_nao_encontrados)}')
    print(f'Total de espelhos não encontrados nos SUMMA: {len(dados_registros_espelho)}')

    print('Verificando ids_peca não encontrados ...')
    # espelhos com registros não encontrados no summa
    if dados_registros_espelho:
        arquivo_registros_espelho = os.path.join('saidas', 'espelhos_acordaos_espelho_sem_summa.csv')
        df_registros_espelho = pd.DataFrame(dados_registros_espelho)
        df_registros_espelho.to_csv(arquivo_registros_espelho, index=False, encoding='utf-8-sig')
        print(f'Arquivo com registros dos espelhos não encontrados no SUMMA salvo em {arquivo_registros_espelho} com {len(df_registros_espelho)} registros.')
    # grava um csv com os ids não encontrados nos espelhos
    if ids_peca_nao_encontrados:
        arquivo_ids_nao_encontrados = os.path.join('saidas', 'espelhos_acordaos_summa_sem_espelho.csv')
        dados = []
        for id_peca in tqdm(ids_peca_nao_encontrados, desc='Processando ids não encontrados'):
            seq_documento_acordao = id_peca_acordao.get(id_peca)
            dados.append({'id_peca': id_peca, 'tipo': get_principal_sucessivo(seq_documento_acordao), 'seq_documento_acordao': seq_documento_acordao})
        nao_encontradas = pd.DataFrame(dados)
        nao_encontradas.to_csv(arquivo_ids_nao_encontrados, index=False, encoding='utf-8-sig')
        print(f'Arquivo com IDs não encontrados salvo em {arquivo_ids_nao_encontrados}')

def testar_saida_parquet():
    ''' Testa a leitura do arquivo parquet gerado '''
    arquivo = os.path.join('saidas', 'espelhos_acordaos_consolidado_textos.parquet')
    if not os.path.isfile(arquivo):
        print(f'Arquivo {arquivo} não encontrado.')
        return
    print(f'Lendo arquivo {arquivo} ...')
    df = pd.read_parquet(arquivo)
    print(f'Arquivo lido com {len(df)} registros e colunas: {list(df.columns)}')
    df['texto'] = df['texto'].apply(lambda x: CRIPT.decriptografar(x) if isinstance(x,str) else '')
    print(df.head())
    # imprime os 10 primeiros textos completos
    # filtra terceira seção
    #df = df[df['nomeOrgaoJulgador'].isin(['TERCEIRA SEÇÃO','TERCEIRA SEÃ‡ÃƒO','QUINTA TURMA','SEXTA TURMA'])]
    df = df[df['nomeOrgaoJulgador'].isin(['TERCEIRA SEÇÃO','TERCEIRA SEÃ‡ÃƒO'])]
    for i, row in df.head(10).iterrows():
        print(f'\n\n--- TEXTO COMPLETO DO ACÓRDÃO id_peca {row["id_peca"]} ---\n')
        print(row['texto'][:5000])  # imprime os primeiros 5000 caracteres
        dados = dict(row)
        dados.pop('texto', None)
        print('DADOS:','|' * 70)
        print(dados['nomeOrgaoJulgador'], '-', dados['siglaClasse'], dados['descricaoClasse'])
        print(dados)
        print('\n\n--- FIM DO TEXTO ---\n')

if __name__ == '__main__':
    '''
    main()
    print('==========================================================')
    print(f'Foram baixados {TOTAL_JSON} arquivos JSON')
    print('==========================================================')
    '''
       
    consolidar_saidas()
    #testar_saida_parquet()
