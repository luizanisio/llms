# -*- coding: utf-8 -*-

"""
Utilitários para geração de gráficos com Matplotlib e Seaborn.

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Descrição:
-----------
Classe utilitária para criação de gráficos estatísticos incluindo boxplots,
gráficos de quantidade, soma e média com suporte a múltiplas colunas e paletas de cores.
"""

import math
try:
    import pandas as pd
    from pandas.api.types import is_integer_dtype
except ImportError as e:
    raise ImportError("Pandas não está instalado. Instale com 'pip install pandas'") from e
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
#import locale
#locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

class Cores(Enum):
    PuBuGn     = 'PuBuGn'          # Sequencial
    Blues      = 'Blues'           # Sequencial
    Tab10      = 'tab10'           # Qualitativa
    Colorblind = 'colorblind'      # Seaborn: 6 cores CVD-friendly
    Viridis    = 'viridis'         # Matplotlib: perceptualmente uniforme
    Plasma     = 'plasma'          # Matplotlib
    Cividis    = 'cividis'         # Matplotlib: otimizada para daltonismo
    Cubehelix  = 'cubehelix'       # Matplotlib: bom em tons de cinza
    Set2       = 'Set2'            # ColorBrewer qualitativa, 8 cores
    Dark2      = 'Dark2'           # ColorBrewer qualitativa, 8 cores
    Accent     = 'Accent'          # ColorBrewer qualitativa, 8 cores
    RdYlGn     = 'RdYlGn'          # Divergente (Vermelho-Amarelo-Verde)

class UtilGraficos:
    """
    Utilitários para geração de gráficos com Matplotlib e Seaborn.
    
    Esta classe fornece métodos para criar gráficos estatísticos sofisticados com
    formatação automática, suporte a múltiplas paletas de cores e organização em grid.
    
    Métodos principais:
    -------------------
    grafico_multi_colunas : Cria múltiplos gráficos organizados em grid
        - Suporta quantidade, soma, média e boxplot
        - Configuração flexível por gráfico
        - Paletas de cores customizáveis
        - Opções de ordenação e filtragem
    
    Exemplo:
    --------
    >>> config = {
    ...     'Gráfico 1': {
    ...         'df': df,
    ...         'colunas': ['col1', 'col2'],
    ...         'x': 'Valores',
    ...         'y': 'Contagem',
    ...         'agregacao': 'quantidade',
    ...         'paleta': Cores.PuBuGn
    ...     }
    ... }
    >>> UtilGraficos.grafico_multi_colunas(config, plots_por_linha=2)
    """

    @classmethod
    def grafico_multi_colunas(cls, configuracao,
                              plots_por_linha: int = 2,
                              paleta_cores=Cores.PuBuGn,
                              arquivo_saida: str = None):
        """
        Gera vários gráficos dispostos em um grid com `plots_por_linha` colunas.

        Cada gráfico pode ser:
        - 'quantidade' (contagem por valor de cada coluna),
        - 'soma' (soma dos valores por coluna),
        - 'media' ou 'média' (média dos valores por coluna),
        - 'boxplot' (boxplot dos valores de cada coluna),
        - 'histograma' (histograma da distribuição dos valores de cada coluna).

        Params:
        - configuracao: dict de
            'Título': {
                'df': DataFrame,
                'colunas': [...],
                'alias': [...],         # opcional - nome apresentado para as colunas
                'x': 'rótulo eixo X',
                'y': 'rótulo eixo Y',
                'agregacao': 'quantidade'|'soma'|'media'|'média'|'boxplot'|'histograma',
                'paleta': Cores ou lista # opcional, por gráfico
                'ylim': (min, max)       # opcional, limites do eixo Y (ex: (0, 1))
                'x_crescente': bool      # opcional, ordena eixo x em ordem crescente
                'xcat_crescente': bool   # opcional, ordena categorias de x em ordem crescente
                'dropnan': bool          # opcional, remove linhas com NaN (padrão: False)
                'drop_zero': bool        # opcional, remove valores <= 0 (padrão: False)
                'mostrar_legenda': bool  # opcional, exibe legenda (padrão: True para 'quantidade')
                'rotacao_labels': int    # opcional, ângulo de rotação dos labels do eixo X
            }
        - plots_por_linha: quantos subplots/gráficos por linha no layout (padrão: 2)
        - paleta_cores: paleta padrão Cores.PuBuGn (pode ser sobrescrita em cfg['paleta'])
        - arquivo_saida: caminho do arquivo para salvar (None = exibir na tela)
        """
        total = len(configuracao)
        ncols = plots_por_linha
        nrows = math.ceil(total / ncols)

        fig, axes = plt.subplots(nrows=nrows,
                                ncols=ncols,
                                figsize=(ncols * 10, nrows * 6),
                                squeeze=False)
        axes_flat = axes.flatten()
        pal_padrao = paleta_cores.value if isinstance(paleta_cores, Cores) else paleta_cores

        for ax, (titulo, cfg) in zip(axes_flat, configuracao.items()):
            df      = cfg['df']
            cols    = cfg['colunas']
            aliases = cfg.get('alias', cols)
            xlabel  = cfg.get('x', '')
            ylabel  = cfg.get('y', '')
            aggr    = cfg.get('agregacao', 'quantidade').lower()
            pal     = cfg.get('paleta', pal_padrao)
            pal     = pal.value if isinstance(pal, Cores) else pal
            drop_nan = cfg.get('dropnan', False)
            drop_zero = cfg.get('drop_zero', False)
            rotacao = cfg.get('rotacao_labels', 0)
            
            ax.clear()
            ax.grid(False)
            
            # Configura bordas em cinza claro para não atrapalhar visualização
            # quando dados tocam 0 ou 1
            for spine in ax.spines.values():
                spine.set_edgecolor('#CCCCCC')
                spine.set_linewidth(1.0)

            # Validação de DataFrame vazio
            if df.empty or not cols:
                ax.text(0.5, 0.5, 'Sem dados disponíveis', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(titulo)
                continue

            if aggr == 'quantidade':
                df_plot = df[cols].copy()
                # Aplicar filtros antes de processar
                for col in cols:
                    if drop_nan:
                        df_plot = df_plot[df_plot[col].notna()]
                    if drop_zero:
                        df_plot = df_plot[df_plot[col] > 0]
                
                # counts por valor de cada coluna
                vc = {cls._corrige_rotulos(col): df_plot[col].value_counts()
                    for col in cols}
                plot_df = pd.DataFrame(vc).transpose().fillna(0)

                ordem = plot_df.max(axis=1).sort_values(
                    ascending=not cfg.get('x_crescente', False)
                ).index
                plot_df = plot_df.loc[ordem]

                if 'xcat_crescente' in cfg:
                    cols_ord = plot_df.sum(axis=0).sort_values(
                        ascending=cfg['xcat_crescente']
                    ).index
                    plot_df = plot_df[cols_ord]

                paleta = sns.color_palette(pal, plot_df.shape[1])
                plot_df.plot(kind='bar',
                            ax=ax,
                            color=paleta,
                            stacked=False,
                            legend=cfg.get('mostrar_legenda', True))

                for patch in ax.patches:
                    h = patch.get_height()
                    if h > 0:
                        ax.annotate(str(int(h)),
                                    xy=(patch.get_x() + patch.get_width()/2, h),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom',
                                    fontsize=8)
                
                # Rotação de labels
                if rotacao == 0 and len(plot_df) > 5:
                    rotacao = 45
                ax.tick_params(axis='x', rotation=rotacao)

            elif aggr in ('soma', 'media', 'média'):
                df_plot = df[cols].copy()
                # Aplicar filtros antes de agregar
                for col in cols:
                    if drop_nan:
                        df_plot = df_plot[df_plot[col].notna()]
                    if drop_zero:
                        df_plot = df_plot[df_plot[col] > 0]
                
                # determina o formato de exibição (int ou float)
                if aggr == 'soma' and all(is_integer_dtype(df_plot[c]) for c in cols):
                    fmt = 'int'
                else:
                    fmt = 'float'

                valores = [df_plot[col].sum() if aggr == 'soma' else df_plot[col].mean()
                        for col in cols]

                if 'x_crescente' in cfg:
                    asc = cfg['x_crescente']
                    pares = list(zip(aliases, valores))
                    pares.sort(key=lambda x: x[1], reverse=not asc)
                    aliases_plot, valores_plot = zip(*pares) if len(pares) > 0 else ([], [])
                else:
                    aliases_plot, valores_plot = aliases, valores

                paleta = sns.color_palette(pal, len(aliases_plot))
                barras = ax.bar(aliases_plot, valores_plot, color=paleta)

                for bar, v in zip(barras, valores_plot):
                    ax.annotate(cls._formata_numero(v, fmt),
                                xy=(bar.get_x() + bar.get_width()/2, v),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)
                
                # Rotação de labels
                if rotacao == 0 and len(aliases_plot) > 5:
                    rotacao = 45
                ax.tick_params(axis='x', rotation=rotacao)

            elif aggr in ('boxplot'):
                # boxplot dos valores de cada coluna
                df_plot = df[cols].copy()
                # Aplicar filtros
                if drop_nan:
                    df_plot = df_plot.dropna()
                if drop_zero:
                    df_plot = df_plot[(df_plot > 0).all(axis=1)]
                
                # renomeia colunas para usar aliases no eixo x
                df_plot = df_plot.rename(columns=dict(zip(cols, aliases)))
                
                if len(df_plot) > 0 and len(df_plot.columns) > 0:
                    paleta = sns.color_palette(pal, df_plot.shape[1])
                    sns.boxplot(data=df_plot,
                                ax=ax,
                                palette=paleta,
                                boxprops=dict(alpha=.8))
                else:
                    ax.text(0.5, 0.5, 'Sem dados após filtros', 
                           ha='center', va='center', transform=ax.transAxes)
                
                # Rotação de labels
                if rotacao == 0 and len(aliases) > 5:
                    rotacao = 45
                ax.tick_params(axis='x', rotation=rotacao)

            elif aggr in ('histograma'):
                # histograma dos valores de cada coluna
                df_plot = df[cols].copy()
                # Aplicar filtros
                if drop_nan:
                    df_plot = df_plot.dropna()
                if drop_zero:
                    df_plot = df_plot[(df_plot > 0).all(axis=1)]
                
                if len(df_plot) > 0 and len(df_plot.columns) > 0:
                    paleta = sns.color_palette(pal, len(cols))
                    for i, col in enumerate(cols):
                        alias = aliases[i] if i < len(aliases) else col
                        ax.hist(df_plot[col].dropna(), 
                               bins=30, 
                               alpha=0.7, 
                               color=paleta[i],
                               label=alias,
                               edgecolor='black',
                               linewidth=0.5)
                    
                    if len(cols) > 1:
                        ax.legend(loc='best', frameon=True, framealpha=0.9)
                else:
                    ax.text(0.5, 0.5, 'Sem dados após filtros', 
                           ha='center', va='center', transform=ax.transAxes)

            else:
                raise ValueError(
                    f"Agregação '{aggr}' não suportada. "
                    "Use 'quantidade', 'soma', 'media', 'boxplot' ou 'histograma'.")

            ax.set_title(titulo, fontsize=12, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            
            # Configurar limites do eixo Y se especificado
            if 'ylim' in cfg:
                ylim = cfg['ylim']
                if isinstance(ylim, (tuple, list)) and len(ylim) == 2:
                    ax.set_ylim(ylim)
            
            # Adiciona margem no eixo Y para evitar que dados em 0 ou 1 toquem as bordas
            # Usa 3% do range como margem (mais generoso que o padrão matplotlib de 5% que não funciona bem)
            ymin, ymax = ax.get_ylim()
            y_range = ymax - ymin
            if y_range > 0:
                margin = y_range * 0.03  # 3% de margem
                ax.set_ylim(ymin - margin, ymax + margin)
            
            # Ajustar legenda se presente
            if ax.get_legend():
                ax.legend(loc='best', frameon=True, framealpha=0.9)

        # remove subplots extras
        for ax in axes_flat[total:]:
            ax.axis('off')

        plt.tight_layout()
        if not arquivo_saida:
            plt.show()
        else:
            plt.savefig(arquivo_saida, dpi=300, bbox_inches='tight')
        if not arquivo_saida:
            plt.show()
        else:
            plt.savefig(arquivo_saida, dpi=300, bbox_inches='tight')
            plt.close()

    @classmethod
    def gerar_grafico_barras(cls, df: pd.DataFrame, titulo: str,
                                ylabel: str = '', xlabel: str = '',
                                arquivo_saida: str = None,
                                paleta_cores=Cores.PuBuGn,
                                mostrar_valores: bool = True,
                                rotacao_labels: int = 0,
                                stacked: bool = False,
                                ylim: tuple = None):
        """
        Gera um gráfico de barras (agrupadas ou empilhadas) a partir de um DataFrame.
        
        Args:
            df: DataFrame com índice (eixo X) e colunas (grupos/barras)
            titulo: Título do gráfico
            ylabel: Rótulo do eixo Y
            xlabel: Rótulo do eixo X
            arquivo_saida: Caminho para salvar o arquivo (opcional)
            paleta_cores: Paleta de cores a utilizar
            mostrar_valores: Se deve mostrar valores
            rotacao_labels: Rotação dos rótulos do eixo X
            stacked: Se True, empilha as barras. Se False, agrupa.
            ylim: Tupla (min, max) para limites do eixo Y
        """
        if df.empty:
            print(f"⚠️  Aviso: DataFrame vazio para gráfico de barras: {titulo}")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        
        num_cols = len(df.columns)
        paleta = sns.color_palette(paleta_cores.value if isinstance(paleta_cores, Cores) else Cores.Cividis.value, num_cols)
        
        df.plot(kind='bar', stacked=stacked, ax=ax, color=paleta, width=0.8)
        
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        
        if ylim:
            ax.set_ylim(ylim)
        
        # Adiciona valores
        if mostrar_valores:
            for c in ax.containers:
                # Se for stacked, mostra valores no centro. Se agrupado, acima da barra.
                # Para agrupado, bar_label funciona bem.
                # Filtrando 0 para limpar viz
                labels = [cls._formata_numero(v, 'float') if v > 0.01 else '' for v in c.datavalues]
                ax.bar_label(c, labels=labels, label_type='center' if stacked else 'edge', 
                             fontsize=8, color='black' if not stacked else 'white', 
                             fontweight='bold', padding=3)

        # Rotação de labels
        if rotacao_labels == 0 and len(df) > 5:
            rotacao_labels = 45
        ax.tick_params(axis='x', rotation=rotacao_labels)
        
        # Grid apenas no fundo
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Legenda
        # Move legenda para fora se tiver muitas colunas
        if num_cols > 10:
             ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
        else:
             ax.legend(loc='best', frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        
        if not arquivo_saida:
            plt.show()
        else:
            plt.savefig(arquivo_saida, dpi=300, bbox_inches='tight')
            plt.close()

    @classmethod
    def gerar_grafico_empilhado(cls, df: pd.DataFrame, titulo: str,
                                ylabel: str = '', xlabel: str = '',

                                arquivo_saida: str = None,
                                paleta_cores=Cores.RdYlGn,
                                mostrar_valores: bool = True,
                                rotacao_labels: int = 0):
        """
        Gera um gráfico de barras empilhadas.
        
        Args:
            df: DataFrame com índice (eixo X) e colunas (segmentos da pilha)
            titulo: Título do gráfico
            ylabel: Rótulo do eixo Y
            xlabel: Rótulo do eixo X
            arquivo_saida: Caminho para salvar o arquivo (opcional)
            paleta_cores: Paleta de cores a utilizar
            mostrar_valores: Se deve mostrar valores dentro das barras
            rotacao_labels: Rotação dos rótulos do eixo X
        """
        if df.empty:
            print("⚠️  Aviso: DataFrame vazio para gráfico empilhado")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        
        paleta = sns.color_palette(paleta_cores.value, len(df.columns))
        
        df.plot(kind='bar', stacked=True, ax=ax, color=paleta, width=0.8)
        
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        
        # Adiciona valores
        if mostrar_valores:
            for c in ax.containers:
                # Otimização: só mostra se houver espaço/valor > 0
                labels = [str(int(v)) if v > 0 else '' for v in c.datavalues]
                ax.bar_label(c, labels=labels, label_type='center', fontsize=9, color='black', fontweight='bold')
        
        # Rotação de labels
        if rotacao_labels == 0 and len(df) > 5:
            rotacao_labels = 45
        ax.tick_params(axis='x', rotation=rotacao_labels)
        
        # Grid apenas no fundo
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Legenda
        ax.legend(loc='best', frameon=True, framealpha=0.9, title='Status')
        
        plt.tight_layout()
        
        if not arquivo_saida:
            plt.show()
        else:
            plt.savefig(arquivo_saida, dpi=300, bbox_inches='tight')
            plt.close()

    @classmethod
    def gerar_boxplot(cls, dados: dict, titulo: str, 
                      ylabel: str = '', xlabel: str = '', 
                      arquivo_saida: str = None, 
                      paleta_cores=Cores.PuBuGn,
                      mostrar_valores: bool = True,
                      rotacao_labels: int = 45):
        """
        Gera um gráfico de boxplot simplificado a partir de um dicionário de listas.
        
        Args:
            dados: dict {categoria: [lista_de_valores]}
            titulo: Título do gráfico
            ylabel: Rótulo do eixo Y
            xlabel: Rótulo do eixo X
            arquivo_saida: Caminho para salvar o arquivo (opcional)
            paleta_cores: Paleta de cores a utilizar
            mostrar_valores: Se deve mostrar valores (não aplicável a boxplot padrão, mantido para compatibilidade)
            rotacao_labels: Rotação dos rótulos do eixo X
        """
        if not dados:
            print("⚠️  Aviso: Sem dados para gerar boxplot")
            return

        # Converte dict de listas para DataFrame
        # Como as listas podem ter tamanhos diferentes, cria um dict de Series
        import pandas as pd
        dados_series = {k: pd.Series(v) for k, v in dados.items()}
        df = pd.DataFrame(dados_series)
        
        colunas = list(dados.keys())
        
        config = {
            titulo: {
                'df': df,
                'colunas': colunas,
                'x': xlabel,
                'y': ylabel,
                'agregacao': 'boxplot',
                'paleta': paleta_cores,
                'rotacao_labels': rotacao_labels
            }
        }
        
        cls.grafico_multi_colunas(
            configuracao=config,
            plots_por_linha=1,
            paleta_cores=paleta_cores,
            arquivo_saida=arquivo_saida
        )

    @classmethod
    def gerar_grafico_linhas(cls, series: dict, titulo: str,
                              ylabel: str = '', xlabel: str = '',
                              arquivo_saida: str = None,
                              marcadores_verticais: list = None,
                              marcadores_epoca: list = None,
                              preencher_area: bool = False,
                              figsize: tuple = (12, 6),
                              dpi: int = 150,
                              info_text: str = None):
        """
        Gera um gráfico de linhas para séries temporais ou evolução de métricas.
        
        Args:
            series: dict {nome_serie: {'x': [valores_x], 'y': [valores_y], 'cor': 'blue', 'estilo': '-'}}
                   Exemplo: {'Train Loss': {'x': [1,2,3], 'y': [0.5, 0.3, 0.2], 'cor': 'blue'}}
            titulo: Título do gráfico
            ylabel: Rótulo do eixo Y
            xlabel: Rótulo do eixo X
            arquivo_saida: Caminho para salvar o arquivo (opcional)
            marcadores_verticais: Lista de {'x': valor, 'cor': 'gray', 'estilo': ':', 'alpha': 0.4}
            marcadores_epoca: Lista de {'x': valor, 'label': 'Época 1', 'cor': 'green'}
            preencher_area: Se True, preenche área abaixo das linhas
            figsize: Tamanho da figura (largura, altura)
            dpi: DPI para salvar imagem
            info_text: Texto informativo para exibir no canto superior esquerdo
        
        Returns:
            str: Caminho do arquivo salvo, ou None se erro/exibição
        
        Exemplo:
            >>> series = {
            ...     'Train Loss': {'x': steps, 'y': train_loss, 'cor': 'blue', 'marcador': 'o'},
            ...     'Eval Loss': {'x': steps, 'y': eval_loss, 'cor': 'red', 'marcador': 's'}
            ... }
            >>> UtilGraficos.gerar_grafico_linhas(series, 'Evolução do Loss', 'Loss', 'Step')
        """
        if not series:
            print("⚠️  Aviso: Sem dados para gerar gráfico de linhas")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plotar séries
        for nome, config in series.items():
            x = config.get('x', [])
            y = config.get('y', [])
            cor = config.get('cor', 'blue')
            estilo = config.get('estilo', '-')
            marcador = config.get('marcador', None)
            tamanho_marcador = config.get('tamanho_marcador', 4)
            largura = config.get('largura', 2)
            alpha = config.get('alpha', 0.8)
            
            if x and y:
                ax.plot(x, y, color=cor, linestyle=estilo, linewidth=largura,
                       label=nome, marker=marcador, markersize=tamanho_marcador, alpha=alpha)
                
                if preencher_area:
                    ax.fill_between(x, y, alpha=0.2, color=cor)
        
        # Marcadores de época (linhas verticais destacadas com label)
        if marcadores_epoca:
            for m in marcadores_epoca:
                x_pos = m.get('x', 0)
                label = m.get('label', '')
                cor = m.get('cor', 'green')
                alpha = m.get('alpha', 0.7)
                
                ax.axvline(x=x_pos, color=cor, linestyle='--', alpha=alpha, linewidth=1.5)
                if label:
                    ax.text(x_pos, ax.get_ylim()[1] * 0.95, label,
                           rotation=90, va='top', ha='right', fontsize=9, color=cor)
        
        # Marcadores verticais genéricos (checkpoints, eval, etc)
        # O primeiro marcador com label será adicionado à legenda
        if marcadores_verticais:
            label_usado = False
            for m in marcadores_verticais:
                x_pos = m.get('x', 0)
                cor = m.get('cor', 'gray')
                estilo = m.get('estilo', ':')
                alpha = m.get('alpha', 0.4)
                largura = m.get('largura', 1)
                label = m.get('label', '') if not label_usado else ''
                texto = m.get('texto', '')
                cor_texto = m.get('cor_texto', cor)
                
                ax.axvline(x=x_pos, color=cor, linestyle=estilo, alpha=alpha, 
                          linewidth=largura, label=label if label else None)
                
                if texto:
                    ax.text(x_pos, ax.get_ylim()[1] * 0.95, texto,
                           rotation=90, va='top', ha='right', fontsize=9, color=cor_texto)
                
                if label:
                    label_usado = True
        
        # Configurações do gráfico
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Texto informativo
        if info_text:
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if not arquivo_saida:
            plt.show()
            return None
        else:
            plt.savefig(arquivo_saida, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            return arquivo_saida

    @classmethod
    def _corrige_rotulos(cls, r):
        if r.startswith('ext_'):
           r = r.replace('ext_', '')
        if len(r) > 3 and '_' in r:
           r = r.split('_')[0]
        return r

    @classmethod
    def _formata_numero(cls, v, fmt:str):
        if fmt == 'int':
            s = f"{int(v):,}"
            return s.replace(",", ".")
        else:
            s = f"{v:,.2f}"
            return s.replace(",", "|").replace(".", ",").replace("|", ".")
            
if __name__ == '__main__':
    # Exemplo de uso
    import numpy as np

    # Criar DataFrame de exemplo
    np.random.seed(42)
    df_exemplo = pd.DataFrame({
        'A': np.random.randint(0, 100, 20),
        'B': np.random.randint(0, 50, 20),
        'C': np.random.randint(20, 80, 20),
        'D': np.random.randint(10, 90, 20),
    })

    config = {
        'Gráfico de Quantidade': {
            'df': df_exemplo,
            'colunas': ['A', 'B'],
            'x': 'Valores',
            'y': 'Contagem',
            'agregacao': 'quantidade',
            'x_crescente': True,
            'rotacao_labels': 45
        },
        'Gráfico de Soma': {
            'df': df_exemplo,
            'colunas': ['C', 'D'],
            'x': 'Colunas',
            'y': 'Soma dos Valores',
            'agregacao': 'soma'
        },
        'Gráfico de Boxplot': {
            'df': df_exemplo,
            'colunas': ['A', 'B', 'C', 'D'],
            'x': 'Colunas',
            'y': 'Distribuição dos Valores',
            'agregacao': 'boxplot',
            'drop_zero': True
        },
        'Gráfico de Histograma': {
            'df': df_exemplo,
            'colunas': ['A', 'B'],
            'x': 'Valores',
            'y': 'Frequência',
            'agregacao': 'histograma'
        }
    }

    UtilGraficos.grafico_multi_colunas(config, plots_por_linha=2)
