
# -*- coding: utf-8 -*-

'''

Autor: Luiz Anísio
Data: Março 2026
Fonte: https://github.com/luizanisio/llms/tree/main/src

Descrição:
-----------
Divisão dos modelos avaliados de acordo com os resultados obtidos.
 
Utiliza os resultados de F1 score das análises globais feitas como base para a divisão.
Cria uma tabela (em csv) com os seguintes campos:
- id do documento
- nome_modelo
- família do modelo (se disponível, ou nulo)
- métricas dinâmicas de F1 ((global)_*_F1 e (estrutura)_*_F1)
- dificuldade (fácil, médio, difícil)
- dificuldade_int (valor de 1 a 10 calculado com 1- normalização da soma das pontuações)
- alvo (treino, teste, validação)

Para cada rótulo de dificuldade, separa os documentos em treino, teste e validação com proporção de 70%, 20% e 10% respectivamente.
Grava a tabela csv na pasta do modelo, ordenada pela pontuação mais baixa (fácil) para a mais alta (difícil).
Considera a pasta jsons criada pela classe JsonAnaliseDataFrame.
Cria uma pasta "divisoes" dentro da pasta de análises
Dentro da pasta "divisoes" cria um arquivo csv para cada modelo.
'''

import os
import sys
import json
import glob
import re
import pandas as pd
import shutil
import argparse
from tqdm import tqdm

from util_pandas import UtilPandasExcel
from xlsxwriter.utility import xl_col_to_name

class UtilJsonDivisoes:
    def __init__(self, pasta_analises: str, divisao_grupos: tuple = (0.7, 0.2, 0.1)):
        self.pasta_analises = pasta_analises
        self.divisao_grupos = divisao_grupos
        self.pasta_jsons = os.path.join(self.pasta_analises, 'jsons')
        self.pasta_saida = os.path.join(self.pasta_analises, 'divisoes')
        
        # Regex para identificar as colunas de métricas que entram no cálculo
        # "(global)_*_F1" ou "(estrutura)_*_F1"
        self.regex_metricas = re.compile(r'^(\(global\)|\(estrutura\))_.*_F1$')

    def limpar_saida(self):
        """Remove apenas os arquivos de divisões antigos (divisao_*.csv)."""
        os.makedirs(self.pasta_saida, exist_ok=True)
        arquivos_antigos = glob.glob(os.path.join(self.pasta_saida, 'divisao_*.csv')) + glob.glob(os.path.join(self.pasta_saida, 'divisao_*.xlsx'))
        for f in arquivos_antigos:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Erro ao remover arquivo antigo {f}: {e}")

    def processar(self):
        """Método principal que lê os dados de JSON e gera os arquivos CSV por modelo."""
        if not os.path.exists(self.pasta_jsons):
            print(f"Erro: Pasta {self.pasta_jsons} não encontrada.")
            return

        arquivos_json = glob.glob(os.path.join(self.pasta_jsons, 'analise_*.json'))
        if not arquivos_json:
            print(f"Nenhum arquivo JSON encontrado em {self.pasta_jsons}.")
            return

        self.limpar_saida()

        # Dicionário de resultados agrupados por modelo: 
        # modelos_dados = { 'nome_modelo': [ {'id': val, 'nome_modelo': ..., ...}, ... ] }
        modelos_dados = {}

        print(f"📊 Processando {len(arquivos_json)} arquivos JSON de análise...")

        for arquivo in tqdm(arquivos_json, desc="Processando arquivos JSON"):
            with open(arquivo, 'r', encoding='utf-8') as f:
                try:
                    conteudo = json.load(f)
                except Exception as e:
                    print(f"Erro ao ler o arquivo {arquivo}: {e}")
                    continue

            # 1. Busca ID do Documento (busca a chave raiz que comece com 'id', case-insensitive)
            id_doc = None
            for key in conteudo.keys():
                if key.lower().startswith('id'):
                    id_doc = conteudo[key]
                    break
            
            # Fallback se a chave id não for encontrada, extrai do nome do arquivo
            if id_doc is None:
                match_id = re.search(r'analise_(.+?)\.json', os.path.basename(arquivo))
                id_doc = match_id.group(1) if match_id else "desconhecido"

            # 2. Itera pelas chaves que representam os modelos avaliados
            # Sabendo que as avaliações são chaves que representam os dicionários internos
            for chave, valor in conteudo.items():
                if isinstance(valor, dict):
                    nome_modelo = chave
                    
                    if nome_modelo not in modelos_dados:
                        modelos_dados[nome_modelo] = []
                    
                    linha_modelo = {
                        'id': id_doc, 
                        'nome_modelo': nome_modelo,
                        'familia_modelo': None # Familia não está presente nos JSONs da análise
                    }
                    
                    # Extrai as métricas de interesse (terminadas em _F1) para salvar e calcular
                    for metrica, val_metrica in valor.items():
                        if metrica.endswith('_F1'):
                             linha_modelo[metrica] = val_metrica
                             
                    modelos_dados[nome_modelo].append(linha_modelo)

        # Trata e salva as informações por modelo
        for nome_modelo, dados_lista in modelos_dados.items():
            self._processar_modelo(nome_modelo, dados_lista)

    def _processar_modelo(self, nome_modelo: str, dados_lista: list):
        df = pd.DataFrame(dados_lista)
        if df.empty:
            return

        # Identifica colunas-alvo que casam com a expressão regular
        colunas_alvo = [col for col in df.columns if self.regex_metricas.match(col)]
        
        if not colunas_alvo:
            print(f"⚠️ Aviso: Modelo {nome_modelo} não possui as métricas alvo F1.")
            
        # 3. Tratamento de valores nulos nas métricas alvo com a média do próprio modelo
        for col in colunas_alvo:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            media_coluna = df[col].mean()
            if pd.isna(media_coluna):
                # Se não houver nenhum valor válido na série toda
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(media_coluna)

        # Para distribuir bem as notas de dificuldade, usamos posições dos itens (ranking)
        # em vez dos valores absolutos (min-max que concentra nos extremos).
        # Garantimos as proporções: 30% dificil, 40% medio e 30% facil
        
        # 4. Cálculo do nível de dificuldade global do documento
        if colunas_alvo:
            # Soma das colunas alvo
            df['soma_pontuacoes'] = df[colunas_alvo].sum(axis=1)
        else:
            df['soma_pontuacoes'] = 0.0

        # Ordena pelo desempenho (ascendente: menor nota = pior = mais difícil)
        df = df.sort_values(by=['soma_pontuacoes']).copy()
        
        n_total = len(df)
        
        # Define as categorias usando fatiamento no DataFrame ordenado
        # 30% dificil
        n_dificil = int(n_total * 0.3)
        # 30% facil
        n_facil = int(n_total * 0.3)
        # 40% medio
        n_medio = n_total - n_dificil - n_facil
        
        categorias = ['dificil'] * n_dificil + ['medio'] * n_medio + ['facil'] * n_facil
        df['dificuldade'] = categorias

        def distribuir_notas(sub_df, notas_possiveis):
            if len(sub_df) == 0:
                return pd.Series(dtype=int)
            if len(sub_df) <= len(notas_possiveis):
                return pd.Series([notas_possiveis[i % len(notas_possiveis)] for i in range(len(sub_df))], index=sub_df.index)
            
            # O qcut distribui baseando na quantidade em caixas quase iguais
            return pd.qcut(range(len(sub_df)), q=len(notas_possiveis), labels=notas_possiveis)

        # Geração do valor de 1 a 10 considerando as partições de desempenho:
        # Categoria dificil ganha as notas [10, 9, 8]
        # Categoria medio ganha [7, 6, 5, 4]
        # Categoria facil ganha [3, 2, 1]
        # A ordem de distribuição importa (já que df está na ordem do pior desempenho ao melhor)
        df['dificuldade_int'] = 0
        df.loc[df['dificuldade'] == 'dificil', 'dificuldade_int'] = distribuir_notas(df[df['dificuldade'] == 'dificil'], [10, 9, 8])
        df.loc[df['dificuldade'] == 'medio', 'dificuldade_int'] = distribuir_notas(df[df['dificuldade'] == 'medio'], [7, 6, 5, 4])
        df.loc[df['dificuldade'] == 'facil', 'dificuldade_int'] = distribuir_notas(df[df['dificuldade'] == 'facil'], [3, 2, 1])
        
        df['dificuldade_int'] = df['dificuldade_int'].astype(int)

        # 5. Organização e Divisão Baseada nos Grupos (Treinamento, Teste e Validação)
        dfs_finais = []
        seed_fixa = 42 # Semente fixa para amostragem reprodutível
        
        for dif in ['facil', 'medio', 'dificil']:
            df_sub = df[df['dificuldade'] == dif].copy()
            if df_sub.empty:
                continue
            
            # Embaralha os dados pertencentes ao mesmo nível de dificuldade
            df_sub = df_sub.sample(frac=1.0, random_state=seed_fixa)
            n_total = len(df_sub)
            
            p_treino, p_teste, _ = self.divisao_grupos
            
            n_treino = int(n_total * p_treino)
            n_teste = int(n_total * p_teste)
            n_val = n_total - n_treino - n_teste
            
            # Associa grupos às partições embaralhadas
            grupos = ['treino'] * n_treino + ['teste'] * n_teste + ['validacao'] * n_val
            df_sub['alvo'] = grupos
            
            dfs_finais.append(df_sub)

        if not dfs_finais:
            return

        df_final = pd.concat(dfs_finais)
        
        # O cabeçalho determina ordenar "da pontuação mais baixa (facil) para a mais alta (dificil)"
        # Note que a dificuldade aumenta de "1" (facil) para "10" (dificil). Ordenamos então por esse indicador.
        df_final = df_final.sort_values(by=['dificuldade_int', 'id'], ascending=[True, True])
        
        # Limpeza da coluna temporária
        df_final = df_final.drop(columns=['soma_pontuacoes'])
        
        # 6. Salvar na formatação correta do CSV
        nome_arquivo_seguro = re.sub(r'[^a-zA-Z0-9_\-()]', '_', nome_modelo)
        caminho_csv = os.path.join(self.pasta_saida, f'divisao_{nome_arquivo_seguro}.csv')
        
        df_final.to_csv(caminho_csv, index=False, encoding='utf-8')
        print(f"   ✓ Divisões para [{nome_modelo}]: {len(df_final)} registros -> {caminho_csv}")

        # 7. Salvar na formatação correta do Excel com mapa de calor (util_pandas)
        caminho_xlsx = os.path.join(self.pasta_saida, f'divisao_{nome_arquivo_seguro}.xlsx')
        try:
            upd = UtilPandasExcel(caminho_xlsx)
            upd.write_df(df_final, 'Divisões', auto_width_colums_list=True)
            
            # Aplica mapa de calor nas métricas F1 (colunas_alvo)
            qtd_linhas = len(df_final)
            colunas_df = df_final.columns.tolist()
            
            for col in colunas_alvo:
                if col in colunas_df:
                    idx_col = colunas_df.index(col)
                    letra_col = xl_col_to_name(idx_col)
                    range_celulas = f"{letra_col}2:{letra_col}{qtd_linhas + 1}"
                    # Usa o conditional_color da UtilPandasExcel
                    upd.conditional_color(sheet_name='Divisões', cells=range_celulas)
                    
            upd.save()
        except Exception as e:
            print(f"   ⚠️ Erro ao gerar excel para [{nome_modelo}]: {e}")


if __name__ == '__main__':
    ''' Recebe como parâmetro da linha de comando o nome da pasta de análise criada pela classe JsonAnaliseDataFrame
        Exemplo: python util_json_divisoes.py nome_da_pasta
        Remove os arquivos de divisões criados: divisao_<nome_modelo>.csv  
        Cria novamente as divisões
    '''
    parser = argparse.ArgumentParser(description="Gerador de Divisões em Treino/Teste/Validação via Curriculum Learning Múltiplas Métricas")
    parser.add_argument('pasta_analise', type=str, help="Caminho da pasta de análises contendo a pasta 'jsons'")
    args = parser.parse_args()
    
    pasta_alvo = os.path.abspath(args.pasta_analise)
    
    if not os.path.isdir(pasta_alvo):
        print(f"A pasta '{pasta_alvo}' não existe.")
        sys.exit(1)
        
    print(f"\n🚀 Iniciando as divisões de dados na pasta: {pasta_alvo}")
    util = UtilJsonDivisoes(pasta_analises=pasta_alvo)
    util.processar()
    print("\n✅ Processo concluído.")