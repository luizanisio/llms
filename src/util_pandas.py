# -*- coding: utf-8 -*-

"""
Utilitários para exportação de DataFrames para Excel com formatação.

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Descrição:
-----------
Classe para facilitar a criação de planilhas Excel com formatação automática,
ajuste de largura de colunas, formatação condicional e destaque de células.
"""

import pandas as pd
import os
import re
try:
    from xlsxwriter.utility import xl_col_to_name
except ImportError:
    raise ImportError('Módulo xlsxwriter não encontrado. Instale com "pip install xlsxwriter"')    

# exemplo de colorir por quantidade
# q = len(dados_finais)
# tp.conditional_color(sheet_name=sn, cells=f'C1:C{q+1}')
# 
# escala invertida (verde=menor, vermelho=maior - útil para tokens/erros):
# tp.conditional_color(sheet_name=sn, cells=f'C1:C{q+1}', min_value=1, max_value=0)

class UtilPandasExcel:
    """
    Utilitário para exportação de DataFrames para Excel com formatação avançada.
    
    Fornece recursos para criar planilhas Excel profissionais com:
    - Ajuste automático de largura de colunas
    - Formatação de cabeçalhos
    - Formatação condicional com gradiente de cores
    - Destaque de células
    - Congelamento de painéis
    - Proteção contra fórmulas e URLs
    
    Atributos:
    ----------
    FORMAT_HEADER : dict
        Formatação padrão para cabeçalhos
    FORMAT_DEFAULT : dict
        Formatação padrão para células
    FORMAT_HIGHLIGHT : dict
        Formatação para células destacadas
    
    Exemplo:
    --------
    >>> upd = UtilPandasExcel('./planilha.xlsx')
    >>> upd.write_df(df, 'Dados')
    >>> upd.conditional_color(sheet_name='Dados', cells='C2:C100')
    >>> upd.save()
    """
    FORMAT_HEADER = {'bold':  True, 'align': 'left', 'valign': 'top', 'text_wrap': True, 'bg_color': '#CCCCCC'}
    FORMAT_DEFAULT = {'bold':  False, 'align': 'left', 'valign': 'top', 'text_wrap': True}
    FORMAT_HIGHLIGHT = {'bold':  True, 'align': 'left', 'valign': 'top', 'text_wrap': True, 'bg_color': '#FFFD9A'}
    
    MAX_WIDTH = 100
    RE_MAIUSCULAS = re.compile('[A-Z]')

    def __str_size__(self,texto):
        sz = 0.0
        for lt in texto:
            sz = sz + (1.33 if self.RE_MAIUSCULAS.search(lt) else 1)
        return round(sz + 0.5)

    def __init__(self, nome_arquivo:str, columns_auto_width = True, header_formatting = True):
        if (nome_arquivo.lower()[-5:] != '.xlsx') and (nome_arquivo.lower()[-4:] != '.xls'):
            nome_arquivo = f'{nome_arquivo}.xlsx'
        self.nome_arquivo = nome_arquivo
        self.writer = pd.ExcelWriter(self.nome_arquivo, engine='xlsxwriter')
        self.columns_auto_width = columns_auto_width
        self.header_formatting = header_formatting
        self.WB_HEADER_FORMAT = self.writer.book.add_format(self.FORMAT_HEADER)
        self.WB_DEFAULT_FORMAT = self.writer.book.add_format(self.FORMAT_DEFAULT)
        self.WB_HIGHLIGHT_FORMAT = self.writer.book.add_format(self.FORMAT_HIGHLIGHT)

    def write_df(self,df,sheet_name : str, auto_width_colums_list = True, columns_titles = None):
        df.to_excel(self.writer, sheet_name=f'{sheet_name}', index = False)
        if not (columns_titles is None):
            worksheet = self.writer.sheets[f'{sheet_name}']
            for n, value in enumerate(columns_titles):
                worksheet.write(0, n, value)
        # formata os tamanhos das colunas
        self.__auto_width_colums__(df = df, sheet_name = sheet_name,columns_list = auto_width_colums_list, columns_titles=columns_titles)
        # formata o cabeçalho
        self.__format_header__(df = df, sheet_name=sheet_name)

    def write_dfs(self,dataframes: dict,auto_width_colums_list = True):
        for n,d in dataframes.items():
            self.write_df(df = d, sheet_name = n, auto_width_colums_list = auto_width_colums_list)

    # recebe o endereço da célula e o valor
    def write_cell(self, sheet_name : str, cell:str, value, is_header = False):
        worksheet = self.writer.sheets[f'{sheet_name}']
        fm = self.WB_HEADER_FORMAT if is_header else None 
        #worksheet.write(0, colx, value, self.WB_HEADER_FORMAT)        
        worksheet.write(f'{cell}', value, fm)

    # recebe a posição inicial e final e grava a lista na linha
    # exmeplos:
    # upd.write_cell(sheet_name='Resumo de Entidades', cell='E1', value = 'TREINO', is_header = True)
    # upd.write_cells(sheet_name='Resumo de Entidades', col=5,line=0, values = ['TIPO', 'INICIO','FIM','PAI','MEDIA'], is_header= True)
    # upd.write_cells(sheet_name='Resumo de Entidades', col=5,line=2, values = ['TIPO', 'INICIO','FIM','PAI','MEDIA'], is_header= False)
    def write_cells(self, sheet_name : str, col:int, line:int, values = [], is_header = False):
        worksheet = self.writer.sheets[f'{sheet_name}']
        fm = self.WB_HEADER_FORMAT if is_header else None 
        #worksheet.write(0, colx, value, self.WB_HEADER_FORMAT)        
        for n, value in enumerate(values):
            worksheet.write(line, col + n, value, fm)

    # recebe um json e grava uma tabela com cabeçalho
    # exmeplos:
    # upd.write_cells(sheet_name='Resumo de Entidades', col=5,line=4, values = [{'INICIO':1,'FIM':2},{'INICIO':3,'FIM':4}], is_header= False, col_order=['INICIO','FIM'])
    def write_table(self, sheet_name : str, col:int, line:int, values = [], is_header = False, col_order = None, columns_titles = None):
        worksheet = self.writer.sheets[f'{sheet_name}']
        fm = self.WB_HEADER_FORMAT if is_header else None 
        _col_order = list(col_order) if not col_order is None else None
        _col_title = _col_order if columns_titles is None else list(columns_titles)
        # se não receber a ordem, cria a lista com todas as colunas
        if _col_order is None:
            _col_order = []
            for value in values:
                for c in value.keys():
                    if not c in _col_order:
                        _col_order.append(c)
            _col_title = _col_order
        # grava a coluna de cabeçalhos
        self.write_cells(sheet_name=sheet_name, col=col, line=line, values = _col_title, is_header = True)
        _col_size = [self.__str_size__(c) + 2 for c in _col_title]
        # grava os dados
        for n, value in enumerate(values):
            for colx, k in enumerate(_col_order):
                vl = value.get(k)
                if not (vl is None):
                    worksheet.write(line + n + 1, col + colx, vl, fm)
                    _col_size[colx] = max(self.__str_size__(f'{vl}') , _col_size[colx])
        # ajusta a largura das células
        for n, s in enumerate(_col_size):
            _sz = min(self.MAX_WIDTH, s)
            worksheet.set_column(col + n, col + n, _sz)  # set column width   

    def __auto_width_colums__(self,df:pd.DataFrame, sheet_name : str, columns_list = [], columns_titles = None):
        if (columns_list == False ):
            return
        # inspirado em https://stackoverflow.com/questions/17326973/is-there-a-way-to-auto-adjust-excel-column-widths-with-pandas-excelwriter
        worksheet = self.writer.sheets[sheet_name]  # pull worksheet object
        for idx, col in enumerate(df):  # loop through all columns
            try:
                series = df[col]
                
                # CORREÇÃO: Valida se realmente é uma Series (não DataFrame)
                if not isinstance(series, pd.Series):
                    print(f"⚠️  Aviso: coluna '{col}' retornou {type(series)} ao invés de Series. Pulando ajuste de largura.")
                    continue
                
                if (columns_list == True) or (len(columns_list) == 0) or (col in columns_list):
                    # Obtém nome da coluna de forma segura
                    col_name = str(series.name) if hasattr(series, 'name') else str(col)
                    
                    max_len = max((
                        series.astype(str).map(self.__str_size__).max(),  # len of largest item
                        self.__str_size__(col_name)  # len of column name/header
                        )) + 2  # adding a little extra space
                    if (columns_titles is not None) and (len(columns_titles)>=idx-1):
                        max_len = max((max_len,self.__str_size__(columns_titles[idx])))
                    max_len = max_len if max_len <= self.MAX_WIDTH else self.MAX_WIDTH # verifica o maior tamanho de uma coluna
                    worksheet.set_column(idx, idx, max_len)  # set column width
            except Exception as e:
                # CRÍTICO: Erros de código devem ser propagados, não silenciados
                import traceback
                print(f"❌ ERRO ao ajustar largura da coluna '{col}' na aba '{sheet_name}':")
                print(f"   Tipo da coluna: {type(col)}")
                print(f"   Tipo do DataFrame: {type(df)}")
                print(f"   Colunas do DF: {df.columns.tolist()}")
                traceback.print_exc()
                raise  # Re-lança exceção para não ocultar erros de código        

    def __format_header__(self, df : pd.DataFrame , sheet_name : str):
        if not self.header_formatting:
            return
        # inspirado em https://stackoverflow.com/questions/39919548/xlsxwriter-trouble-formatting-pandas-dataframe-cells-using-xlsxwriter
        # Write the header manually
        worksheet = self.writer.sheets[f'{sheet_name}']
        for colx, value in enumerate(df.columns.values):
            worksheet.write(0, colx, value, self.WB_HEADER_FORMAT)        
        #worksheet.set(0,0,cell_format =self.WB_HEADER_FORMAT)

    def conditional_color(self, sheet_name, cells, min_value = 0, mid_value = 0.75, max_value = 1):
        """
        Aplica formatação condicional com escala de 3 cores.
        
        Args:
            sheet_name: nome da aba
            cells: range de células (ex: 'A1:A10')
            min_value: valor mínimo da escala
            mid_value: valor médio da escala
            max_value: valor máximo da escala
        
        Comportamento:
            - Escala normal (min_value < max_value):
              vermelho = valores baixos, amarelo = médios, verde = valores altos
            
            - Escala invertida (min_value > max_value):
              verde = valores baixos (melhor), amarelo = médios, vermelho = valores altos (pior)
              Útil para métricas onde menor é melhor (tokens, erros, latência)
        
        Exemplos:
            # Escala normal (0.0 = vermelho, 1.0 = verde)
            excel.conditional_color('Métricas', 'B2:B10', min_value=0, max_value=1)
            
            # Escala invertida para tokens (menos tokens = verde)
            excel.conditional_color('Tokens', 'C2:C10', min_value=1, max_value=0)
        """
        # inspirado em https://xlsxwriter.readthedocs.io/working_with_conditional_formats.html
        worksheet = self.writer.sheets[f'{sheet_name}']
        #fm = FORMAT_CONDITIONAL_3_COLOR = {'type': '3_color_scale', min_value, mid_value, max_value}
        if min_value > max_value:
            # escala invertida: cores também invertidas (verde para menor, vermelho para maior)
            worksheet.conditional_format(f'{cells}', {
                'type': '3_color_scale',
                'min_color': '#63BE7B',  # verde (valor mínimo/melhor)
                'min_type': 'num',
                'min_value': max_value,
                'mid_color': '#FFEB84',  # amarelo
                'mid_type': 'num',
                'mid_value': mid_value,
                'max_color': '#F8696B',  # vermelho (valor máximo/pior)
                'max_type': 'num',
                'max_value': min_value
            })
        else:
            # escala normal: vermelho para menor, verde para maior
            worksheet.conditional_format(f'{cells}', {
                'type': '3_color_scale',
                'min_color': '#F8696B',  # vermelho (valor mínimo)
                'min_type': 'num',
                'min_value': min_value,
                'mid_color': '#FFEB84',  # amarelo
                'mid_type': 'num',
                'mid_value': mid_value,
                'max_color': '#63BE7B',  # verde (valor máximo)
                'max_type': 'num',
                'max_value': max_value
            })

    def _calculate_color(self, value, min_value = 0.0, mid_value = 0.5, max_value = 1.0, 
                        min_color = '#F8696B', mid_color = '#FFEB84', max_color = '#63BE7B'):
        """
        Calcula a cor RGB para um valor baseado em uma escala de 3 cores.
        
        Args:
            value: valor numérico
            min_value: valor mínimo da escala
            mid_value: valor médio da escala
            max_value: valor máximo da escala
            min_color: cor para o valor mínimo (hex, ex: '#F8696B' = vermelho)
            mid_color: cor para o valor médio (hex, ex: '#FFEB84' = amarelo)
            max_color: cor para o valor máximo (hex, ex: '#63BE7B' = verde)
        
        Returns:
            string com cor hex (ex: '#FF0000')
        """
        if value is None or not isinstance(value, (int, float)):
            return None
        
        # Converte hex para RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Converte RGB para hex, garantindo valores válidos
        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(
                max(0, min(255, int(rgb[0]))),
                max(0, min(255, int(rgb[1]))),
                max(0, min(255, int(rgb[2])))
            )
        
        # Interpola entre duas cores
        def interpolate_color(color1, color2, ratio):
            # Garante que ratio esteja entre 0 e 1
            ratio = max(0.0, min(1.0, ratio))
            return tuple(c1 + (c2 - c1) * ratio for c1, c2 in zip(color1, color2))
        
        min_rgb = hex_to_rgb(min_color)
        mid_rgb = hex_to_rgb(mid_color)
        max_rgb = hex_to_rgb(max_color)
        
        # Detecta se é escala invertida (min_value > max_value, ex: LOSS)
        is_inverted = min_value > max_value
        
        # Clamp value aos limites
        actual_min = min(min_value, max_value)
        actual_max = max(min_value, max_value)
        value = max(actual_min, min(actual_max, value))
        
        # Determina em qual metade da escala o valor está
        # Para escala invertida, precisamos ajustar a lógica
        if is_inverted:
            # Escala invertida (ex: 1.0 -> vermelho, 0.5 -> amarelo, 0.0 -> verde)
            # min_value=1.0, mid_value=0.5, max_value=0.0
            if value >= mid_value:
                # Entre min (1.0) e mid (0.5)
                if min_value == mid_value:
                    ratio = 0.0
                else:
                    ratio = (min_value - value) / (min_value - mid_value)
                result_rgb = interpolate_color(min_rgb, mid_rgb, ratio)
            else:
                # Entre mid (0.5) e max (0.0)
                if mid_value == max_value:
                    ratio = 1.0
                else:
                    ratio = (mid_value - value) / (mid_value - max_value)
                result_rgb = interpolate_color(mid_rgb, max_rgb, ratio)
        else:
            # Escala normal (ex: 0.0 -> vermelho, 0.5 -> amarelo, 1.0 -> verde)
            if value <= mid_value:
                # Entre min e mid
                if mid_value == min_value:
                    ratio = 0.0
                else:
                    ratio = (value - min_value) / (mid_value - min_value)
                result_rgb = interpolate_color(min_rgb, mid_rgb, ratio)
            else:
                # Entre mid e max
                if max_value == mid_value:
                    ratio = 1.0
                else:
                    ratio = (value - mid_value) / (max_value - mid_value)
                result_rgb = interpolate_color(mid_rgb, max_rgb, ratio)
        
        return rgb_to_hex(result_rgb)
    
    def write_cell_with_color(self, sheet_name, row, col, value, 
                             min_value = 0.0, mid_value = 0.5, max_value = 1.0,
                             min_color = '#F8696B', mid_color = '#FFEB84', max_color = '#63BE7B'):
        """
        Escreve uma célula com cor de fundo calculada baseada no valor.
        
        Args:
            sheet_name: nome da aba
            row: linha (0-based)
            col: coluna (0-based)
            value: valor numérico a escrever
            min_value, mid_value, max_value: escala de valores
            min_color, mid_color, max_color: cores correspondentes (hex)
        """
        worksheet = self.writer.sheets[f'{sheet_name}']
        
        # Calcula a cor
        bg_color = self._calculate_color(value, min_value, mid_value, max_value,
                                        min_color, mid_color, max_color)
        
        if bg_color:
            # Cria formato com a cor de fundo
            # Nota: xlsxwriter precisa de 'pattern' para aplicar bg_color
            cell_format = self.writer.book.add_format({
                'bg_color': bg_color,
                'pattern': 1,  # CRÍTICO: Necessário para xlsxwriter aplicar bg_color
                'align': 'left',
                'valign': 'top',
                'num_format': '0' if isinstance(value, int) else '0.000'  # Formato numérico
            })
            worksheet.write(row, col, value, cell_format)
        else:
            # Sem cor, usa formato padrão
            worksheet.write(row, col, value, self.WB_DEFAULT_FORMAT)

    def highlight_bgcolor(self, sheet_name, cells,  min_value = 0, max_value = 1):
        # inspirado em https://xlsxwriter.readthedocs.io/working_with_conditional_formats.html
        worksheet = self.writer.sheets[f'{sheet_name}']
        worksheet.conditional_format(f'{cells}', {'type': 'cell', 'criteria': 'between','minimum':  min_value,'maximum':  max_value, 'format':   self.WB_HIGHLIGHT_FORMAT})

    # cria um range para a posição das colunas informadas
    def range_cols(self,first_col , last_col , first_row = None, last_row = None):
        _fr = '' if first_row is None else f'{first_row}'
        _lr = '' if last_row is None else f'{last_row}'
        return f'{xl_col_to_name(first_col)}{_fr}:{xl_col_to_name(last_col)}{_lr}'

    def congelar_painel(self, sheet_name, first_row = 1, first_col = 0):
        worksheet = self.writer.sheets[f'{sheet_name}']
        worksheet.freeze_panes(first_row, first_col)

    RE_URL = re.compile(r'https?://')
    RE_FORMULA = re.compile(r'^(\s*\=)*')
    # substitui formulas e urls
    @staticmethod
    def clear_string(value):
        return UtilPandasExcel.RE_FORMULA.sub('', UtilPandasExcel.RE_URL.sub('((url))',value))

    def save(self):
        # Versões mais recentes do pandas usam close() ao invés de save()
        if hasattr(self.writer, 'close'):
            self.writer.close()
        elif hasattr(self.writer, 'save'):
            self.writer.save()
        elif hasattr(self.writer, '_save'):
            self.writer._save()
        else:
            raise AttributeError("Writer não possui método save(), close() ou _save()")


if __name__ == '__main__':
    teste = [{"ano":"2000", "quantidade":10, 'linha_grande' : "aka slkjsdlfjasldjf lasdjflsjdflaksjdlkf"},
             {"ano":"2001", "quantidade":15, "linha_grande" : "dlsafalskjdf lflak sjdflasldfasldkfjsaldkfjalsdfjlaskjdflkas"}]
    df = pd.DataFrame(teste)

    tp = UtilPandasExcel('./teste')
    tp.write_df(df,'teste',True)
    tp.header_formatting = False
    tp.write_df(df,'teste_ano',['ano'])
    tp.write_dfs({'teste2': df, 'teste3':df})

    tp.save()
    print(df)

