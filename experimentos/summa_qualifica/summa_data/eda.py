# código para extrair as análises descritivas dos dados usados no experimento, principalmente os campos usados na estratificação do dataset

import pandas as pd
import sys, os
sys.path.append("../../../src")
import util  # garante que a pasta src está no sys.path

arquivo = "./pecas_exportadas_textos.parquet"
distribuicao_treino_teste_validacao = "../treino_simples/_divisao_fold11/divisao_unica.csv"
pasta_saida = './eda_output'
os.makedirs(pasta_saida, exist_ok=True)

df = pd.read_parquet(arquivo)
df_distribuicao = pd.read_csv(distribuicao_treino_teste_validacao)

print(df.columns)

# cria um dataframe com as colunas de interesse
df_eda = df[["id_peca", "num_ministro", "sg_ramo_direito", "sg_classe", "fold"]].copy()
df_eda["num_ministro"] = df_eda["num_ministro"].astype(str)


def print_dados():
    ''' imprime as estatísticas descritivas dos dados, como a distribuição de ministros, ramos do direito e classes, além de verificar a correspondência entre o dataframe de EDA e o dataframe de distribuição (treino/teste/validação) para garantir que as análises sejam feitas apenas nas instâncias mapeadas no CSV de distribuição (fold=11).
    '''
    
    # calcula as estatísticas descritivas
    estatisticas = df_eda.describe(include="all")

    # salva o dataframe de estatísticas em um arquivo CSV
    #estatisticas.to_csv("estatisticas_eda.csv")
    print(estatisticas)
    print('-'*100)
    print(df_distribuicao.columns)
    print(df_distribuicao.head())

def gerar_graficos_eda():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    plt.rcParams.update({
        'font.size': 24,
        'axes.titlesize': 20,
        'axes.labelsize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 18,
        'legend.fontsize': 20,
        'legend.title_fontsize': 20,
    })

    # Paletas usadas em util_graficos.py:
    # - PuBuGn: sequencial padrão (barras/boxplot)
    # - tab10:  qualitativa padrão para múltiplas categorias com hue
    PALETA_PRINCIPAL = 'PuBuGn'
    PALETA_QUALITATIVA = 'PuBuGn' #'tab10'

    # ---------------------------------------------------------
    # 0. MANTER GRÁFICOS ORIGINAIS (Apenas instâncias mapeadas no CSV de distribuição, ou seja, fold=11)
    # ---------------------------------------------------------
    df_inner = pd.merge(df_eda, df_distribuicao[['id', 'alvo']], left_on='id_peca', right_on='id', how='inner')
    
    # Gráfico original por Ministro
    df_ministro = df_inner.groupby(['num_ministro', 'alvo']).size().reset_index(name='contagem')
    df_ministro.to_csv(os.path.join(pasta_saida,'distribuicao_por_ministro.csv'), index=False)
    
    n_alvos = df_ministro['alvo'].nunique()
    pal_alvo = sns.color_palette(PALETA_PRINCIPAL, n_alvos)
    plt.figure(figsize=(12, 10))
    sns.barplot(data=df_ministro, x='num_ministro', y='contagem', hue='alvo', palette=pal_alvo)
    plt.title('Distribuição Absoluta de Documentos (Treino/Teste/Validação) por Ministro')
    plt.xlabel('Número do Ministro')
    plt.ylabel('Quantidade de Documentos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida,'distribuicao_por_ministro.png'), dpi=150)
    plt.close()
    
    # Gráfico original por Classe
    df_classe = df_inner.groupby(['sg_classe', 'alvo']).size().reset_index(name='contagem')
    df_classe.to_csv(os.path.join(pasta_saida,'distribuicao_por_classe.csv'), index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_classe, x='sg_classe', y='contagem', hue='alvo', palette=pal_alvo)
    plt.title('Distribuição Absoluta de Documentos (Treino/Teste/Validação) por Classe')
    plt.xlabel('Sigla da Classe')
    plt.ylabel('Quantidade de Documentos')
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida,'distribuicao_por_classe.png'), dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # 1. GRÁFICOS PERCENTUAIS (COMPARATIVO ESTRATIFICADO)
    # ---------------------------------------------------------
    # Fazemos um left join para incluir as instâncias do fold != 11.
    df_all = pd.merge(df_eda, df_distribuicao[['id', 'alvo']], left_on='id_peca', right_on='id', how='left')
    
    # Cria a coluna "grupo_estratificacao" combinando o fold e o alvo.
    df_all['grupo_estratificacao'] = np.where(
        df_all['fold'] != 11, 
        'Fold != 11 (Reservado)', 
        'Fold 11 (' + df_all['alvo'].str.capitalize() + ')'
    )
    
    # Paleta qualitativa tab10 — a mesma usada nos gráficos de boxplot/quantidade do util_graficos.py
    grupos_ordenados = sorted(df_all['grupo_estratificacao'].unique())
    n_grupos = len(grupos_ordenados)
    pal_grupos = dict(zip(grupos_ordenados, sns.color_palette(PALETA_QUALITATIVA, n_grupos)))

    colunas_analise = ['num_ministro', 'sg_ramo_direito', 'sg_classe']
    nomes_x = {
        'num_ministro': 'Código do Ministro',
        'sg_ramo_direito': 'Ramo do Direito',
        'sg_classe': 'Sigla da Classe'
    }
    
    for col in colunas_analise:
        # Conta documentos por (grupo_estratificacao, col)
        counts = df_all.groupby(['grupo_estratificacao', col]).size().reset_index(name='contagem')
        # Pega o total de documentos por grupo_estratificacao
        totals = df_all.groupby('grupo_estratificacao').size().reset_index(name='total_grupo')
        
        merged = pd.merge(counts, totals, on='grupo_estratificacao')
        merged['percentual'] = (merged['contagem'] / merged['total_grupo']) * 100
        
        nome_base = f'comparativo_estratificacao_{col}'
        merged.to_csv(os.path.join(pasta_saida, f'{nome_base}.csv'), index=False)
        
        if col == 'num_ministro':
            plt.figure(figsize=(16, 8))
        else:
            plt.figure(figsize=(12, 8))
        sns.barplot(data=merged, x=col, y='percentual', hue='grupo_estratificacao',
                    palette=pal_grupos, hue_order=grupos_ordenados)
        plt.title(f'Estratificação Comparativa: Proporção de {col} nos Folds e Alvos')
        plt.xlabel(nomes_x[col])
        plt.ylabel('Proporção dentro do Grupo (%)')
        if col == 'num_ministro':
            plt.xticks(rotation=45)
        plt.legend(title='Grupo Estratificação', loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, f'{nome_base}.png'), dpi=150)
        plt.close()

if __name__ == "__main__":
    print_dados()
    gerar_graficos_eda()    