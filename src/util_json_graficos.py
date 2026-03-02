# -*- coding: utf-8 -*-
"""
Módulo de geração de gráficos para análise de JSONs.

Extraído de util_json.py para melhor organização do código.
Contém a classe JsonAnaliseGraficos com todos os métodos de geração de gráficos.

Autor: Luiz Anísio
Data: 01/03/2026
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from typing import List, Optional
from util_graficos import UtilGraficos, Cores

# ═════════════════════════════════════════════════════════════════════════
# TRADUÇÕES PARA GRÁFICOS (pt = Português, en = English)
# ═════════════════════════════════════════════════════════════════════════
_TRADUCOES = {
    # Gráficos comparativos globais e por campo
    'global_score_titulo':       {'pt': 'Performance Global (Score Médio) por Técnica',       'en': 'Global Performance (Average Score) by Technique'},
    'global_score_ylabel':       {'pt': 'Score Médio (F1 ou Sim)',                            'en': 'Average Score (F1 or Sim)'},
    'campo_score_titulo':        {'pt': 'Performance por Campo - {tecnica} ({tipo_score} Médio)', 'en': 'Performance by Field - {tecnica} (Average {tipo_score})'},
    'campo_score_ylabel':        {'pt': '{tipo_score} Médio',                                 'en': 'Average {tipo_score}'},
    'campo_xlabel':              {'pt': 'Campo',                                              'en': 'Field'},
    'modelo_xlabel':             {'pt': 'Modelo',                                             'en': 'Model'},
    'similaridade':              {'pt': 'Similaridade',                                       'en': 'Similarity'},
    'f1_score':                  {'pt': 'F1 Score',                                           'en': 'F1 Score'},
    # Avaliação LLM
    'llm_campo_titulo':          {'pt': 'Performance por Campo - Avaliação LLM ({tipo_score} Médio)', 'en': 'Performance by Field - LLM Evaluation (Average {tipo_score})'},
    'llm_P':                     {'pt': 'Avaliação LLM - Precision',                         'en': 'LLM Evaluation - Precision'},
    'llm_R':                     {'pt': 'Avaliação LLM - Recall',                            'en': 'LLM Evaluation - Recall'},
    'llm_F1':                    {'pt': 'Avaliação LLM - F1-Score',                           'en': 'LLM Evaluation - F1-Score'},
    'llm_nota':                  {'pt': 'Avaliação LLM - Nota Geral',                        'en': 'LLM Evaluation - Overall Score'},
    'llm_global_P':              {'pt': 'Avaliação LLM - Precision (Global)',                 'en': 'LLM Evaluation - Precision (Global)'},
    'llm_global_R':              {'pt': 'Avaliação LLM - Recall (Global)',                    'en': 'LLM Evaluation - Recall (Global)'},
    'llm_global_F1':             {'pt': 'Avaliação LLM - F1-Score (Global)',                  'en': 'LLM Evaluation - F1-Score (Global)'},
    'llm_global_nota':           {'pt': 'Avaliação LLM - Nota Geral',                        'en': 'LLM Evaluation - Overall Score'},
    'llm_campo_P':               {'pt': 'Avaliação LLM - {campo} - Precision',               'en': 'LLM Evaluation - {campo} - Precision'},
    'llm_campo_R':               {'pt': 'Avaliação LLM - {campo} - Recall',                  'en': 'LLM Evaluation - {campo} - Recall'},
    'llm_campo_F1':              {'pt': 'Avaliação LLM - {campo} - F1-Score',                 'en': 'LLM Evaluation - {campo} - F1-Score'},
    # Tokens
    'tokens_input':              {'pt': 'Consumo de Tokens de Entrada',                       'en': 'Input Token Consumption'},
    'tokens_output':             {'pt': 'Consumo de Tokens de Saída',                         'en': 'Output Token Consumption'},
    'tokens_total':              {'pt': 'Consumo Total de Tokens',                            'en': 'Total Token Consumption'},
    'tokens_cache':              {'pt': 'Tokens em Cache',                                    'en': 'Cached Tokens'},
    'tokens_reason':             {'pt': 'Tokens de Raciocínio',                               'en': 'Reasoning Tokens'},
    'tokens_ylabel':             {'pt': 'Quantidade de Tokens',                               'en': 'Token Count'},
    # Observabilidade
    'obs_SEG':                   {'pt': 'Tempo de Execução (segundos)',                       'en': 'Execution Time (seconds)'},
    'obs_REV':                   {'pt': 'Revisões/Loops',                                     'en': 'Revisions/Loops'},
    'obs_IT':                    {'pt': 'Iterações',                                          'en': 'Iterations'},
    'obs_AGT':                   {'pt': 'Agentes Executados',                                 'en': 'Agents Executed'},
    'obs_QTD':                   {'pt': 'Campos Preenchidos',                                 'en': 'Fields Filled'},
    'obs_BYTES':                 {'pt': 'Tamanho dos Campos (bytes)',                         'en': 'Field Size (bytes)'},
    'obs_OK':                    {'pt': 'Taxa de Sucesso',                                    'en': 'Success Rate'},
    'obs_ylabel_SEG':            {'pt': 'Segundos',                                           'en': 'Seconds'},
    'obs_ylabel_REV':            {'pt': 'Quantidade',                                         'en': 'Count'},
    'obs_ylabel_IT':             {'pt': 'Quantidade',                                         'en': 'Count'},
    'obs_ylabel_AGT':            {'pt': 'Quantidade',                                         'en': 'Count'},
    'obs_ylabel_QTD':            {'pt': 'Quantidade',                                         'en': 'Count'},
    'obs_ylabel_BYTES':          {'pt': 'Bytes',                                              'en': 'Bytes'},
    'obs_ylabel_OK':             {'pt': 'Proporção',                                          'en': 'Proportion'},
    'obs_titulo':                {'pt': 'Observabilidade: {titulo}',                          'en': 'Observability: {titulo}'},
    'obs_metrica_xlabel':        {'pt': 'Métrica',                                            'en': 'Metric'},
    # Boxplots de métricas (instância)
    'modelos_xlabel':            {'pt': 'Modelos',                                            'en': 'Models'},
    'campos_xlabel':             {'pt': 'Campos',                                             'en': 'Fields'},
    'comp_global_f1':            {'pt': 'Comparação Global F1 entre Modelos',                 'en': 'Global F1 Comparison Between Models'},
    'comp_global_sim':           {'pt': 'Comparação Global Similaridade entre Modelos',       'en': 'Global Similarity Comparison Between Models'},
    'dist_f1_campo':             {'pt': 'Distribuição de F1 por Campo',                       'en': 'F1 Distribution by Field'},
    'sim_ylabel':                {'pt': 'Similaridade (SIM)',                                  'en': 'Similarity (SIM)'},
    # Status
    'status_titulo':             {'pt': 'Status das Extrações por Modelo',                    'en': 'Extraction Status by Model'},
    'status_ylabel':             {'pt': 'Quantidade de Documentos',                           'en': 'Number of Documents'},
    'status_sucesso':            {'pt': 'Sucesso',                                            'en': 'Success'},
    'status_erro':               {'pt': 'Erro',                                               'en': 'Error'},
    'status_inexistente':        {'pt': 'Inexistente',                                        'en': 'Missing'},
}

def traduzir_rotulos(key: str, lang: str = 'pt', **kwargs) -> str:
    """Retorna tradução para a chave na linguagem especificada. Suporta .format(**kwargs)."""
    entry = _TRADUCOES.get(key, {})
    text = entry.get(lang, entry.get('pt', key))
    if kwargs:
        text = text.format(**kwargs)
    return text


class JsonAnaliseGraficos:
    """
    Classe responsável pela geração de gráficos de análise.
    
    Recebe por agregação os objetos necessários do JsonAnaliseDataFrame:
    - dados_analise: container JsonAnaliseDados com dados e configurações
    - rotulos: lista de rótulos [id, origem, modelo1, modelo2, ...]
    - pasta_analises: pasta base para saída de arquivos
    - relatorio: instância de JsonAnaliseRelatorio (opcional)
    - lang: idioma dos gráficos ('pt' ou 'en')
    - to_df_callable: callable que retorna o DataFrame consolidado
    
    Uso:
        graficos = JsonAnaliseGraficos(
            dados_analise=obj.dados_analise,
            rotulos=obj.rotulos,
            pasta_analises=obj.pasta_analises,
            relatorio=obj.relatorio,
            lang='en',
            to_df_callable=obj.to_df
        )
        graficos.gerar_graficos_metricas(arquivo_excel='resultado.xlsx')
    """
    
    def __init__(self, 
                 dados_analise=None,
                 rotulos: list = None,
                 pasta_analises: str = None,
                 relatorio=None,
                 gerar_relatorio: bool = True,
                 lang: str = 'pt',
                 to_df_callable=None):
        self._dados_analise = dados_analise
        self._rotulos = rotulos or []
        self._pasta_analises = pasta_analises
        self._relatorio = relatorio
        self._gerar_relatorio = gerar_relatorio and relatorio is not None
        self._lang = lang if lang in ('pt', 'en') else 'pt'
        self._to_df_callable = to_df_callable

    # ═════════════════════════════════════════════════════════════════════════
    # Propriedades de compatibilidade (mapeiam para os atributos internos)
    # ═════════════════════════════════════════════════════════════════════════
    
    @property 
    def dados_analise(self):
        return self._dados_analise
    
    @property
    def rotulos(self):
        return self._rotulos
    
    @property
    def pasta_analises(self):
        return self._pasta_analises
    
    @property
    def relatorio(self):
        return self._relatorio
    
    @property
    def gerar_relatorio(self):
        return self._gerar_relatorio
    
    def to_df(self):
        """Delega para o callable fornecido."""
        if self._to_df_callable:
            return self._to_df_callable()
        return None

    def _consolidar_graficos_relatorio(self, pasta_saida: str):
        """
        Consolida todos os gráficos PNG da pasta e atualiza o relatório.
        
        Args:
            pasta_saida: pasta onde estão os gráficos
        """
        if not self.gerar_relatorio or self.relatorio is None:
            return
        
        import glob
        
        # Busca todos os PNGs
        graficos_png = glob.glob(os.path.join(pasta_saida, '*.png'))
        
        if not graficos_png:
            return
        
        # Lista para consolidar todos os gráficos
        graficos_consolidados = []
        
        for arquivo_completo in sorted(graficos_png):
            nome_arq = os.path.basename(arquivo_completo)
            
            # Categoriza e descreve o gráfico
            if 'tokens' in nome_arq:
                tipo_token = nome_arq.replace('grafico_tokens_', '').replace('.png', '')
                categoria = 'Tokens (Consumo)'
                descricao = f"Consumo de tokens - {tipo_token.upper()}"
            
            elif 'comparativo_campos_llm' in nome_arq:
                # Gráfico comparativo LLM por campo (comparativo_campos_llm_f1_score.png)
                metrica = nome_arq.replace('comparativo_campos_llm_', '').replace('_score.png', '').upper()
                categoria = 'Avaliação LLM'
                descricao = f"Comparativo por Campo - {metrica}"
            
            elif 'avaliacaollm' in nome_arq:
                # Diferencia global vs por campo
                if any(c in nome_arq for c in ['_tema_', '_notas_', '_referencias_', '_informacoes_', '_jurisprudencia_', '_tese_', '_termos_']):
                    partes = nome_arq.replace('grafico_bp_avaliacaollm_', '').replace('.png', '').split('_')
                    campo = '_'.join(partes[:-1]) if len(partes) > 1 else partes[0]
                    metrica = partes[-1].upper() if len(partes) > 1 else 'F1'
                    categoria = 'Avaliação LLM'
                    descricao = f"Campo: {campo} - {metrica}"
                else:
                    metrica = nome_arq.split('_')[-1].replace('.png', '').upper()
                    categoria = 'Avaliação LLM'
                    descricao = f"Global - {metrica}"
            
            elif 'observabilidade' in nome_arq:
                metrica = nome_arq.replace('grafico_bp_observabilidade_', '').replace('grafico_observabilidade_', '').replace('.png', '').upper()
                categoria = 'Observabilidade'
                descricao = f"{metrica.replace('_', ' ')}"
            
            else:
                # Gráficos de métricas de comparação
                categoria = 'Métricas de Comparação'
                # Extrai campo e métrica do nome
                nome_base = nome_arq.replace('grafico_bp_', '').replace('grafico_comp_', '').replace('grafico_dist_', '').replace('.png', '')
                descricao = nome_base.replace('_', ' ').title()
            
            graficos_consolidados.append({
                'arquivo': nome_arq,
                'descricao': descricao,
                'categoria': categoria
            })
        
        # Atualiza relatório com lista consolidada
        self.relatorio.set_graficos_completo(graficos_consolidados)


    def gerar_graficos_metricas(self, arquivo_excel: str = None, pasta_saida: str = None, 
                                paleta: str = 'Cividis') -> List[str]:
        """
        Gera gráficos boxplot para cada campo e métrica.
        
        Cria boxplots comparando modelos para cada combinação de:
        - Campo (ex: (global), (estrutura), resumo, fatos, etc)
        - Métrica (F1, P, R, LS)
        
        Padrão de nomes: grafico_bp_<campo>_<metrica>.png
        
        Args:
            arquivo_excel: caminho do arquivo Excel (se None, usa DataFrame em memória)
            pasta_saida: pasta para salvar gráficos (se None, usa pasta_analises)
            paleta: paleta de cores para os gráficos (padrão: Cividis - otimizada para daltonismo)
        
        Returns:
            Lista com caminhos dos arquivos gerados
        
        Note:
            A limpeza de gráficos antigos deve ser feita antes de chamar este método.
        """
        import pandas as pd
        import glob
        
        # Define pasta de saída
        if pasta_saida is None:
            pasta_saida = self.pasta_analises or '.'
        if os.path.basename(os.path.normpath(pasta_saida)) != 'graficos':
            pasta_saida = os.path.join(pasta_saida, 'graficos')
        os.makedirs(pasta_saida, exist_ok=True)
        
        # Carrega DataFrame
        if arquivo_excel:
            df = self._carregar_dataframe_de_excel(arquivo_excel)
        else:
            df = self.to_df()
        
        if df is None or df.empty:
            print("⚠️  Aviso: DataFrame vazio, nenhum gráfico gerado")
            return []
        
        # Extrai estrutura de métricas
        estrutura = self._extrair_estrutura_metricas(df)
        
        # Gera gráficos boxplot
        arquivos_gerados = self._gerar_boxplots_por_campo_metrica(
            df, estrutura, pasta_saida, paleta
        )
        
        # Gera gráficos adicionais de interesse
        arquivos_adicionais = self._gerar_graficos_adicionais(
            df, estrutura, pasta_saida, paleta
        )
        
        arquivos_gerados.extend(arquivos_adicionais)
        
        print(f"✅ {len(arquivos_gerados)} gráficos gerados em: {pasta_saida}")
        return arquivos_gerados
    

    @classmethod
    def gerar_graficos_de_excel(cls, arquivo_excel: str, pasta_saida: str = None, 
                                paleta: str = 'Cividis', limpar_graficos_antigos: bool = True) -> List[str]:
        """
        Método estático para gerar gráficos diretamente de um arquivo Excel.
        
        Consolida TODAS as abas de resultados em um único DataFrame para gerar
        gráficos com TODOS OS MODELOS comparados lado a lado.
        
        Args:
            arquivo_excel: caminho do arquivo Excel com resultados
            pasta_saida: pasta para salvar gráficos (se None, usa diretório do Excel)
            paleta: paleta de cores (padrão: Cividis)
            limpar_graficos_antigos: se True, remove gráficos antigos
        
        Returns:
            Lista com caminhos dos arquivos gerados
        
        Exemplo:
            >>> JsonAnaliseDataFrame.gerar_graficos_de_excel('analise_resultados.xlsx')
        """
        import pandas as pd
        import glob
        
        if not os.path.isfile(arquivo_excel):
            raise FileNotFoundError(f"Arquivo não encontrado: {arquivo_excel}")
        
        # Define pasta de saída
        if pasta_saida is None:
            pasta_saida = os.path.dirname(arquivo_excel) or '.'
        if os.path.basename(os.path.normpath(pasta_saida)) != 'graficos':
            pasta_saida = os.path.join(pasta_saida, 'graficos')
        os.makedirs(pasta_saida, exist_ok=True)
        
        # Limpa gráficos antigos
        if limpar_graficos_antigos:
            padroes = ['grafico_*.png', 'boxplot_*.png', 'tokens_*.png', 'avaliacao_*.png', 'observabilidade_*.png', 'comparativo_*.png']
            graficos_antigos = []
            for p in padroes:
                graficos_antigos.extend(glob.glob(os.path.join(pasta_saida, p)))
            
            for arquivo in graficos_antigos:
                try:
                    os.remove(arquivo)
                except Exception as e:
                    print(f"⚠️  Aviso: Não foi possível remover {arquivo}: {e}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # LÊ CONFIGURAÇÕES DA ABA CONFIG (SE EXISTIR)
        # ═══════════════════════════════════════════════════════════════════════
        
        try:
            xl_file = pd.ExcelFile(arquivo_excel)
            
            # Tenta ler aba Config para obter configurações
            nome_campo_id = None
            rotulos_modelos_ordenados = []
            lang = 'pt'  # Idioma padrão dos gráficos
            if 'Config' in xl_file.sheet_names:
                df_config = pd.read_excel(arquivo_excel, sheet_name='Config')
                # Busca o valor de nome_campo_id
                row_nome_campo_id = df_config[df_config['parametro'] == 'nome_campo_id']
                if not row_nome_campo_id.empty:
                    nome_campo_id = row_nome_campo_id.iloc[0]['valor']
                    print(f"   📋 Configuração carregada: nome_campo_id='{nome_campo_id}'")
                
                # Busca a ordem dos modelos para manter o padrao YAML
                row_rotulos_modelos = df_config[df_config['parametro'] == 'rotulos_modelos']
                if not row_rotulos_modelos.empty:
                    rotulos_csv = row_rotulos_modelos.iloc[0]['valor']
                    if pd.notna(rotulos_csv) and str(rotulos_csv).strip():
                        rotulos_modelos_ordenados = [r.strip() for r in str(rotulos_csv).split(',')]
                        print(f"   📋 Ordem dos modelos carregada: {rotulos_modelos_ordenados}")
                
                # Busca linguagem dos gráficos (pt ou en)
                lang = 'pt'
                row_lang = df_config[df_config['parametro'] == 'linguagem_graficos']
                if not row_lang.empty:
                    lang_val = str(row_lang.iloc[0]['valor']).strip().lower()
                    if lang_val in ('pt', 'en'):
                        lang = lang_val
                        print(f"   📋 Idioma dos gráficos: {lang}")
            
            # ═══════════════════════════════════════════════════════════════════════
            # CONSOLIDA TODAS AS ABAS EM UM ÚNICO DATAFRAME
            # ═══════════════════════════════════════════════════════════════════════
            
            abas_resultados = [aba for aba in xl_file.sheet_names if aba.startswith('Resultados')]
            
            if not abas_resultados:
                # Fallback: tenta aba padrão
                abas_resultados = ['Resultados']
            
            # Carrega todas as abas e consolida em um único DataFrame
            df_consolidado = None
            col_id_nome = None
            
            for aba in abas_resultados:
                df_aba = pd.read_excel(arquivo_excel, sheet_name=aba)
                
                # Identifica coluna ID (primeira coluna)
                if col_id_nome is None:
                    col_id_nome = df_aba.columns[0]
                
                # Extrai técnica do nome da aba (ex: Resultados_BERTScore -> bertscore)
                if '_' in aba:
                    # Converte hífen para underscore para manter consistência
                    # Ex: SBERT-Pequeno -> sbert_pequeno
                    tecnica_nome = aba.split('_', 1)[1].lower().replace('-', '_')
                else:
                    tecnica_nome = 'geral'
                
                # ADICIONA TÉCNICA AO NOME DAS COLUNAS (exceto coluna ID)
                # Isso evita conflitos no merge e mantém a informação da técnica
                colunas_renomeadas = {col_id_nome: col_id_nome}  # ID não muda
                for col in df_aba.columns:
                    if col != col_id_nome:
                        # Insere técnica antes da métrica
                        # Ex: agentes_gpt5_(global)_F1 -> agentes_gpt5_(global)_bertscore_F1
                        partes = col.split('_')
                        metrica = partes[-1]
                        if metrica in ['F1', 'P', 'R', 'LS', 'SIM']:
                            # Reconstrói: modelo_campo_tecnica_metrica
                            novo_nome = '_'.join(partes[:-1]) + f'_{tecnica_nome}_{metrica}'
                            colunas_renomeadas[col] = novo_nome
                        else:
                            colunas_renomeadas[col] = col
                
                df_aba_renomeada = df_aba.rename(columns=colunas_renomeadas)
                
                if df_consolidado is None:
                    # Primeira aba: usa como base
                    df_consolidado = df_aba_renomeada
                else:
                    # Demais abas: merge nas colunas ID
                    df_consolidado = pd.merge(
                        df_consolidado,
                        df_aba_renomeada,
                        on=col_id_nome,
                        how='outer'
                    )
            
            if df_consolidado is None or df_consolidado.empty:
                print("⚠️  Aviso: Nenhum dado encontrado nas abas de resultados")
                return []
            
            # Extrai estrutura do DataFrame consolidado
            estrutura = cls._extrair_estrutura_metricas_estatico(df_consolidado, tecnica_aba='', rotulos_modelos_ordenados=rotulos_modelos_ordenados)
            
            # Gera gráficos com TODOS os modelos
            arquivos_gerados = cls._gerar_boxplots_por_campo_metrica_estatico(
                df_consolidado, estrutura, pasta_saida, paleta, tecnica_aba='', rotulos_modelos_ordenados=rotulos_modelos_ordenados, lang=lang
            )
            
            # ═══════════════════════════════════════════════════════════════════════
            # GERA GRÁFICOS DE TOKENS SE A ABA EXISTIR
            # ═══════════════════════════════════════════════════════════════════════
            if 'Resumo_Tokens' in xl_file.sheet_names:
                try:
                    
                    df_tokens = pd.read_excel(arquivo_excel, sheet_name='Resumo_Tokens')
                    
                    if not df_tokens.empty:
                        # Identifica coluna ID (primeira coluna)
                        col_id_tokens = df_tokens.columns[0]
                        
                        # Identifica tipos de tokens disponíveis
                        tipos_tokens = set()
                        for col in df_tokens.columns:
                            if col != col_id_tokens and '_' in col:
                                tipo = col.split('_')[-1]
                                if tipo not in ['finish']:
                                    tipos_tokens.add(tipo)
                        
                        # Mapeia paleta
                        try:
                            paleta_enum = Cores[paleta]
                        except KeyError:
                            paleta_enum = Cores.Cividis
                        
                        # Gera gráfico para cada tipo de token
                        for tipo in sorted(tipos_tokens):
                            colunas_tipo = [col for col in df_tokens.columns 
                                          if col.endswith(f'_{tipo}') and 
                                          df_tokens[col].dtype in [np.int64, np.float64]]
                            
                            if len(colunas_tipo) == 0:
                                continue
                            
                            # Prepara dados mantendo a ordem YAML
                            dados_grafico = {}
                            modelos_col = {col.rsplit('_', 1)[0]: col for col in colunas_tipo}
                            
                            # Usa a ordem preferencial
                            if rotulos_modelos_ordenados:
                                for modelo in rotulos_modelos_ordenados:
                                    if modelo in modelos_col:
                                        col = modelos_col[modelo]
                                        valores = df_tokens[col].dropna().tolist()
                                        if len(valores) > 0:
                                            dados_grafico[modelo] = valores
                                        del modelos_col[modelo]
                            
                            # Fallback para os restantes na ordem em que aparecem no df
                            for modelo, col in modelos_col.items():
                                valores = df_tokens[col].dropna().tolist()
                                if len(valores) > 0:
                                    dados_grafico[modelo] = valores
                            
                            if len(dados_grafico) == 0:
                                continue
                            
                            # Arquivo e título
                            arquivo_grafico = os.path.join(pasta_saida, f'tokens_{tipo}.png')
                            titulo = traduzir_rotulos(f'tokens_{tipo}', lang) if f'tokens_{tipo}' in _TRADUCOES else f'Tokens ({tipo})'
                            
                            # Gera gráfico
                            try:
                                UtilGraficos.gerar_boxplot(
                                    dados=dados_grafico,
                                    titulo=titulo,
                                    ylabel=traduzir_rotulos('tokens_ylabel', lang),
                                    xlabel=traduzir_rotulos('modelo_xlabel', lang),
                                    arquivo_saida=arquivo_grafico,
                                    paleta_cores=paleta_enum,
                                    mostrar_valores=True,
                                    rotacao_labels=45
                                )
                                arquivos_gerados.append(arquivo_grafico)
                                print(f"   ✓ Gráfico de tokens gerado: {os.path.basename(arquivo_grafico)}")
                            except Exception as e:
                                print(f"⚠️  Erro ao gerar gráfico de tokens ({tipo}): {e}")
                
                except Exception as e:
                    print(f"⚠️  Aviso: Não foi possível gerar gráficos de tokens: {e}")
            
            # ═══════════════════════════════════════════════════════════════════════
            # GERA GRÁFICOS DE AVALIAÇÃO LLM SE A ABA EXISTIR
            # ═══════════════════════════════════════════════════════════════════════
            if 'Avaliação LLM' in xl_file.sheet_names:
                try:
                    
                    df_avaliacao = pd.read_excel(arquivo_excel, sheet_name='Avaliação LLM')
                    
                    if not df_avaliacao.empty:
                        # Identifica coluna ID (primeira coluna)
                        col_id_avaliacao = df_avaliacao.columns[0]
                        
                        # Identifica métricas disponíveis (P, R, F1, nota)
                        metricas_disponiveis = set()
                        for col in df_avaliacao.columns:
                            if col == col_id_avaliacao:
                                continue
                            if '_' in col:
                                metrica = col.rsplit('_', 1)[-1]
                                if metrica in ['P', 'R', 'F1', 'nota'] and df_avaliacao[col].dtype in [np.int64, np.float64]:
                                    metricas_disponiveis.add(metrica)
                        
                        # Mapeia paleta
                        try:
                            paleta_enum = Cores[paleta]
                        except KeyError:
                            paleta_enum = Cores.Cividis
                        
                        # Gera gráfico para cada métrica
                        for metrica in sorted(metricas_disponiveis):
                            colunas_metrica = [col for col in df_avaliacao.columns 
                                             if col.endswith(f'_{metrica}') and 
                                             df_avaliacao[col].dtype in [np.int64, np.float64]]
                            
                            if len(colunas_metrica) == 0:
                                continue
                            
                            # Prepara dados mantendo a ordem YAML
                            dados_grafico = {}
                            modelos_col = {col.rsplit('_', 1)[0]: col for col in colunas_metrica}
                            
                            # Usa a ordem preferencial
                            if rotulos_modelos_ordenados:
                                for modelo in rotulos_modelos_ordenados:
                                    if modelo in modelos_col:
                                        col = modelos_col[modelo]
                                        valores = df_avaliacao[col].dropna().tolist()
                                        if len(valores) > 0:
                                            dados_grafico[modelo] = valores
                                        del modelos_col[modelo]
                            
                            # Fallback para os restantes
                            for modelo, col in modelos_col.items():
                                valores = df_avaliacao[col].dropna().tolist()
                                if len(valores) > 0:
                                    dados_grafico[modelo] = valores
                            
                            if len(dados_grafico) == 0:
                                continue
                            
                            # Arquivo e título
                            arquivo_grafico = os.path.join(pasta_saida, f'avaliacao_llm_{metrica.lower()}.png')
                            titulo_key = f'llm_{metrica}'
                            titulo = traduzir_rotulos(titulo_key, lang) if titulo_key in _TRADUCOES else f'LLM Evaluation - {metrica}'
                            
                            # Gera gráfico
                            try:
                                UtilGraficos.gerar_boxplot(
                                    dados=dados_grafico,
                                    titulo=titulo,
                                    ylabel=metrica,
                                    xlabel=traduzir_rotulos('modelo_xlabel', lang),
                                    arquivo_saida=arquivo_grafico,
                                    paleta_cores=paleta_enum,
                                    mostrar_valores=True,
                                    rotacao_labels=45
                                )
                                arquivos_gerados.append(arquivo_grafico)
                                print(f"   ✓ Gráfico de avaliação LLM gerado: {os.path.basename(arquivo_grafico)}")
                            except Exception as e:
                                print(f"⚠️  Erro ao gerar gráfico de avaliação LLM ({metrica}): {e}")
                
                except Exception as e:
                    print(f"⚠️  Aviso: Não foi possível gerar gráficos de avaliação LLM: {e}")
            
            # ═══════════════════════════════════════════════════════════════════════
            # GRÁFICOS DE OBSERVABILIDADE
            # ═══════════════════════════════════════════════════════════════════════
            if 'Observabilidade' in xl_file.sheet_names:
                try:
                    df_obs = pd.read_excel(arquivo_excel, sheet_name='Observabilidade')
                    
                    # Sufixos de interesse
                    sufixos_info = {
                        'SEG': {'titulo_key': 'obs_SEG', 'ylabel_key': 'obs_ylabel_SEG'},
                        'REV': {'titulo_key': 'obs_REV', 'ylabel_key': 'obs_ylabel_REV'},
                        'IT':  {'titulo_key': 'obs_IT',  'ylabel_key': 'obs_ylabel_IT'},
                        'AGT': {'titulo_key': 'obs_AGT', 'ylabel_key': 'obs_ylabel_AGT'},
                        'QTD': {'titulo_key': 'obs_QTD', 'ylabel_key': 'obs_ylabel_QTD'},
                        'BYTES': {'titulo_key': 'obs_BYTES', 'ylabel_key': 'obs_ylabel_BYTES'},
                        'OK': {'titulo_key': 'obs_OK', 'ylabel_key': 'obs_ylabel_OK'}
                    }
                    
                    for sufixo, info in sufixos_info.items():
                        # Identifica colunas com este sufixo
                        colunas_sufixo = [col for col in df_obs.columns 
                                        if col != col_id_nome and f'_{sufixo}' in col and 
                                        sufixo in col.split('_')]
                        
                        if not colunas_sufixo:
                            continue
                        
                        # Prepara DataFrame com conversões necessárias
                        df_plot = df_obs.copy()
                        
                        # Para sufixo OK, converte 'sim'/'não' para 1/0
                        if sufixo == 'OK':
                            for col in colunas_sufixo:
                                df_plot[col] = df_plot[col].map({
                                    'sim': 1, 'não': 0, 
                                    'Sim': 1, 'Não': 0,
                                    True: 1, False: 0,
                                    1: 1, 0: 0
                                })
                        
                        # Para BYTES, garante tipo numérico
                        elif sufixo == 'BYTES':
                            for col in colunas_sufixo:
                                df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
                        
                        # Extrai aliases preservando ordem no df
                        aliases = []
                        colunas_sufixo_ordenadas = []
                        
                        # Usa a ordem preferencial
                        if rotulos_modelos_ordenados:
                           for rotulo in rotulos_modelos_ordenados:
                               for col in colunas_sufixo:
                                   if col.startswith(rotulo + '_'):
                                       colunas_sufixo_ordenadas.append(col)
                                       aliases.append(col.rsplit('_', 1)[0])
                                       break
                        
                        # Fallback
                        for col in colunas_sufixo:
                           if col not in colunas_sufixo_ordenadas:
                               colunas_sufixo_ordenadas.append(col)
                               aliases.append(col.rsplit('_', 1)[0])
                               
                        colunas_sufixo = colunas_sufixo_ordenadas
                        
                        # Gera gráfico
                        arquivo_grafico = os.path.join(pasta_saida, f'observabilidade_{sufixo}.png')
                        titulo = traduzir_rotulos('obs_titulo', lang, titulo=traduzir_rotulos(info['titulo_key'], lang))
                        ylabel = traduzir_rotulos(info['ylabel_key'], lang)
                        
                        # Configura gráfico boxplot usando grafico_multi_colunas
                        configuracao = {
                            titulo: {
                                'df': df_plot,
                                'colunas': colunas_sufixo,
                                'alias': aliases,
                                'x': traduzir_rotulos('obs_metrica_xlabel', lang),
                                'y': ylabel,
                                'agregacao': 'boxplot',
                                'paleta': paleta_enum,
                                'dropnan': True,
                                'rotacao_labels': 45 if len(colunas_sufixo) > 5 else 0
                            }
                        }
                        
                        try:
                            UtilGraficos.grafico_multi_colunas(
                                configuracao=configuracao,
                                plots_por_linha=1,
                                paleta_cores=paleta_enum,
                                arquivo_saida=arquivo_grafico
                            )
                            arquivos_gerados.append(arquivo_grafico)
                            print(f"   ✓ Gráfico de observabilidade gerado: {os.path.basename(arquivo_grafico)}")
                        except Exception as e:
                            print(f"⚠️  Erro ao gerar gráfico de observabilidade ({sufixo}): {e}")
                
                except Exception as e:
                    print(f"⚠️  Aviso: Não foi possível gerar gráficos de observabilidade: {e}")
            
            # ═══════════════════════════════════════════════════════════════════════
            # GRÁFICOS COMPARATIVOS DE SCORE (F1 ou SIM) (GLOBAL E POR CAMPO)
            # ═══════════════════════════════════════════════════════════════════════
            if df_consolidado is not None and not df_consolidado.empty:
                try:
                    # Resolve paleta
                    try:
                        paleta_enum_score = Cores[paleta]
                    except KeyError:
                        paleta_enum_score = Cores.Cividis

                    # Identifica modelos a partir das colunas globais e preserva ordem
                    # Coluna Global: <modelo>_(global)_<tecnica>_F1 ou _SIM
                    known_models = list(rotulos_modelos_ordenados)
                    # Suporta F1 (para BERT/ROUGE) e SIM (para Levenshtein)
                    sufixos_validos = ['_F1', '_SIM']
                    
                    colunas_global_score = [
                        c for c in df_consolidado.columns 
                        if '_(global)_' in c and any(c.endswith(s) for s in sufixos_validos)
                    ]
                    
                    dados_global = []
                    tecnicas_globais = set()
                    
                    for col in colunas_global_score:
                        partes = col.split('_(global)_')
                        if len(partes) == 2:
                            modelo = partes[0]
                            if modelo not in known_models:
                                known_models.append(modelo)
                            
                            resto = partes[1] # tecnica_F1 ou tecnica_SIM
                            
                            sufixo_encontrado = next((s for s in sufixos_validos if resto.endswith(s)), None)
                            if not sufixo_encontrado:
                                continue
                                
                            tecnica = resto[:-len(sufixo_encontrado)]
                            # Normaliza nome da técnica
                            tecnica = tecnica.replace('_', ' ').strip()
                            tecnicas_globais.add(tecnica)
                            
                            media_score = df_consolidado[col].mean()
                            dados_global.append({
                                'Modelo': modelo,
                                'Técnica': tecnica,
                                'Score': media_score
                            })
                    
                    # 1. GRÁFICO GLOBAL SCORE
                    if dados_global:
                        df_global = pd.DataFrame(dados_global)
                        # Pivot: Index=Modelo, Columns=Técnica
                        df_pivot = df_global.pivot(index='Modelo', columns='Técnica', values='Score')
                        df_pivot = df_pivot.reindex(index=[m for m in known_models if m in df_pivot.index])
                        
                        arquivo_global = os.path.join(pasta_saida, 'comparativo_global_score.png')
                        
                        # Usa gerar_grafico_barras (barras agrupadas)
                        UtilGraficos.gerar_grafico_barras(
                            df=df_pivot,
                            titulo=traduzir_rotulos('global_score_titulo', lang),
                            xlabel=traduzir_rotulos('modelo_xlabel', lang),
                            ylabel=traduzir_rotulos('global_score_ylabel', lang),
                            arquivo_saida=arquivo_global,
                            paleta_cores=paleta_enum_score,
                            stacked=False,
                            ylim=(0, 1.05),
                            rotacao_labels=0,
                            lang=lang
                        )
                        arquivos_gerados.append(arquivo_global)
                        print(f"   ✓ Gráfico comparativo global gerado: {os.path.basename(arquivo_global)}")

                    # 2. GRÁFICOS POR CAMPO (UM ARQUIVO POR TÉCNICA)
                    tecnicas_map = {} # tecnica -> lista de colunas
                    
                    colunas_score_campos = [
                        c for c in df_consolidado.columns 
                        if any(c.endswith(s) for s in sufixos_validos) and '_(global)_' not in c and c != col_id_nome
                    ]
                    
                    for col in colunas_score_campos:
                        # Técnicas conhecidas (ordem importa: sbert_* antes de sbert para match correto)
                        for t in ['bertscore', 'rouge', 'rouge1', 'rouge2', 'levenshtein', 
                                  'sbert_grande', 'sbert_medio', 'sbert_pequeno', 'sbert']:
                            # Verifica se coluna termina com _{tecnica}_{sufixo}
                            # Ex: _bertscore_F1 ou _levenshtein_SIM
                            match = False
                            for s in sufixos_validos:
                                if col.endswith(f'_{t}{s}'):
                                    match = True
                                    break
                            
                            if match:
                                if t not in tecnicas_map: tecnicas_map[t] = []
                                tecnicas_map[t].append(col)
                                break
                    
                    for tecnica, colunas in tecnicas_map.items():
                        dados_tecnica = []
                        modelos_no_grafico = set()
                        
                        for col in colunas:
                            # Identifica o sufixo exato usado na coluna
                            sufixo_usado = next((s for s in sufixos_validos if col.endswith(s)), '_F1')
                            
                            sufixo_completo = f'_{tecnica}{sufixo_usado}'
                            base = col[:-len(sufixo_completo)] # modelo_campo
                            
                            # Tenta casar com known_models
                            modelo_match = None
                            for m in sorted(known_models, key=len, reverse=True):
                                if base.startswith(m + '_'):
                                    modelo_match = m
                                    break
                            
                            if modelo_match:
                                campo = base[len(modelo_match)+1:] # remove modelo_
                                media = df_consolidado[col].mean()
                                dados_tecnica.append({
                                    'Campo': campo,
                                    'Modelo': modelo_match,
                                    'Score': media
                                })
                                modelos_no_grafico.add(modelo_match)
                        
                        if dados_tecnica:
                            df_tec = pd.DataFrame(dados_tecnica)
                            # Pivot: Index=Campo, Columns=Modelo, Values=Score
                            df_pivot = df_tec.pivot(index='Campo', columns='Modelo', values='Score')
                            
                            # Mantém a ordem dos modelos e campos
                            df_pivot = df_pivot.reindex(columns=[m for m in known_models if m in df_pivot.columns])
                            
                            campos_ordem = []
                            for item in dados_tecnica:
                                if item['Campo'] not in campos_ordem:
                                    campos_ordem.append(item['Campo'])
                            df_pivot = df_pivot.reindex(index=campos_ordem)
                            
                            arquivo_tec = os.path.join(pasta_saida, f'comparativo_campos_{tecnica.lower()}_score.png')
                            
                            tipo_score = traduzir_rotulos('similaridade', lang) if tecnica == 'levenshtein' else traduzir_rotulos('f1_score', lang)
                            
                            # Usa gerar_grafico_barras (agrupado por modelo para cada campo)
                            # Index (Campo) será o eixo X. Colunas (Modelos) serão as barras.
                            UtilGraficos.gerar_grafico_barras(
                                df=df_pivot,
                                titulo=traduzir_rotulos('campo_score_titulo', lang, tecnica=tecnica.upper(), tipo_score=tipo_score),
                                xlabel=traduzir_rotulos('campo_xlabel', lang),
                                ylabel=traduzir_rotulos('campo_score_ylabel', lang, tipo_score=tipo_score),
                                arquivo_saida=arquivo_tec,
                                paleta_cores=paleta_enum_score,
                                stacked=False,
                                ylim=(0, 1.05),
                                rotacao_labels=45,
                                lang=lang
                            )
                            arquivos_gerados.append(arquivo_tec)
                            print(f"   ✓ Gráfico comparativo de campos ({tecnica}) gerado: {os.path.basename(arquivo_tec)}")

                except Exception as e:
                    print(f"⚠️  Erro ao gerar gráficos comparativos de Score: {e}")
            
            # ═══════════════════════════════════════════════════════════════════════
            # GRÁFICOS COMPARATIVOS LLM POR CAMPO (F1, P, R médios por campo/modelo)
            # ═══════════════════════════════════════════════════════════════════════
            if 'Avaliação LLM Campos' in xl_file.sheet_names:
                try:
                    df_llm_campos = pd.read_excel(arquivo_excel, sheet_name='Avaliação LLM Campos')
                    
                    if not df_llm_campos.empty:
                        # Resolve paleta
                        try:
                            paleta_enum_llm = Cores[paleta]
                        except KeyError:
                            paleta_enum_llm = Cores.Cividis
                        
                        # Identifica coluna ID (primeira coluna)
                        col_id_llm = df_llm_campos.columns[0]
                        
                        # Identifica modelos conhecidos
                        known_models_llm = list(rotulos_modelos_ordenados) if rotulos_modelos_ordenados else []
                        
                        # Métricas de interesse para gráficos comparativos
                        metricas_llm_alvo = ['F1', 'P', 'R']
                        
                        for metrica_alvo in metricas_llm_alvo:
                            # Coleta dados: para cada coluna modelo_campo_metrica, extrai modelo, campo e média
                            dados_llm_campos = []
                            
                            for col in df_llm_campos.columns:
                                if col == col_id_llm:
                                    continue
                                # Verifica se termina com a métrica alvo (ex: _F1, _P, _R)
                                if not col.endswith(f'_{metrica_alvo}'):
                                    continue
                                # Verifica se é numérica
                                if df_llm_campos[col].dtype not in [np.int64, np.float64]:
                                    continue
                                
                                # Identifica modelo (maior match primeiro para evitar ambiguidade)
                                modelo_match = None
                                for m in sorted(known_models_llm, key=len, reverse=True):
                                    if col.startswith(m + '_'):
                                        modelo_match = m
                                        break
                                
                                # Fallback: descobre modelo a partir de colunas se não foi passado
                                if modelo_match is None and not known_models_llm:
                                    # Tenta extrair modelo: tudo antes do penúltimo _ (campo_metrica)
                                    partes = col.rsplit('_', 2)
                                    if len(partes) >= 3:
                                        modelo_match = '_'.join(partes[:-2])
                                        if modelo_match not in known_models_llm:
                                            known_models_llm.append(modelo_match)
                                
                                if modelo_match is None:
                                    continue
                                
                                # Extrai campo: entre modelo_ e _metrica
                                sufixo_metrica = f'_{metrica_alvo}'
                                prefixo_modelo = f'{modelo_match}_'
                                campo = col[len(prefixo_modelo):-len(sufixo_metrica)]
                                
                                if not campo:
                                    continue
                                
                                media = df_llm_campos[col].mean()
                                dados_llm_campos.append({
                                    'Campo': campo,
                                    'Modelo': modelo_match,
                                    'Score': media
                                })
                            
                            if dados_llm_campos:
                                df_llm_tec = pd.DataFrame(dados_llm_campos)
                                # Pivot: Index=Campo, Columns=Modelo, Values=Score
                                df_pivot_llm = df_llm_tec.pivot(index='Campo', columns='Modelo', values='Score')
                                
                                # Mantém a ordem dos modelos
                                if known_models_llm:
                                    df_pivot_llm = df_pivot_llm.reindex(
                                        columns=[m for m in known_models_llm if m in df_pivot_llm.columns]
                                    )
                                
                                # Mantém a ordem dos campos conforme aparecem nos dados
                                campos_ordem_llm = []
                                for item in dados_llm_campos:
                                    if item['Campo'] not in campos_ordem_llm:
                                        campos_ordem_llm.append(item['Campo'])
                                df_pivot_llm = df_pivot_llm.reindex(index=campos_ordem_llm)
                                
                                titulos_metrica_llm = {
                                    'F1': 'F1 Score',
                                    'P': 'Precision',
                                    'R': 'Recall'
                                }
                                tipo_score_llm = titulos_metrica_llm.get(metrica_alvo, metrica_alvo)
                                
                                arquivo_llm = os.path.join(
                                    pasta_saida, f'comparativo_campos_llm_{metrica_alvo.lower()}_score.png'
                                )
                                
                                UtilGraficos.gerar_grafico_barras(
                                    df=df_pivot_llm,
                                    titulo=traduzir_rotulos('llm_campo_titulo', lang, tipo_score=tipo_score_llm),
                                    xlabel=traduzir_rotulos('campo_xlabel', lang),
                                    ylabel=traduzir_rotulos('campo_score_ylabel', lang, tipo_score=tipo_score_llm),
                                    arquivo_saida=arquivo_llm,
                                    paleta_cores=paleta_enum_llm,
                                    stacked=False,
                                    ylim=(0, 1.05),
                                    rotacao_labels=45,
                                    lang=lang
                                )
                                arquivos_gerados.append(arquivo_llm)
                                print(f"   ✓ Gráfico comparativo LLM por campo ({metrica_alvo}) gerado: {os.path.basename(arquivo_llm)}")
                
                except Exception as e:
                    print(f"⚠️  Erro ao gerar gráficos comparativos LLM por campo: {e}")
            
            # ═══════════════════════════════════════════════════════════════════════
        
        except Exception as e:
            raise RuntimeError(f"Erro ao processar Excel: {e}")

        
        print(f"✅ {len(arquivos_gerados)} gráficos gerados em: {pasta_saida}")
        return arquivos_gerados
    

    def gerar_graficos_tokens(self, arquivo_excel: str = None, pasta_saida: str = None,
                              paleta: str = 'Cividis') -> List[str]:
        """
        Gera gráficos de barras/boxplot para consumo de tokens por modelo.
        
        Cria gráficos comparando modelos para cada tipo de token:
        - input: tokens de entrada
        - output: tokens de saída
        - total: total de tokens
        - cache: tokens em cache
        - reason: tokens de raciocínio
        
        Padrão de nomes: grafico_tokens_<tipo>.png
        
        Args:
            arquivo_excel: caminho do arquivo Excel (se None, usa DataFrame em memória)
            pasta_saida: pasta para salvar gráficos (se None, usa pasta_analises)
            paleta: paleta de cores para os gráficos (padrão: Cividis)
        
        Returns:
            Lista com caminhos dos arquivos gerados
        
        Note:
            A limpeza de gráficos antigos deve ser feita antes de chamar este método.
        """
        
        # Define pasta de saída
        if pasta_saida is None:
            pasta_saida = self.pasta_analises or '.'
        if os.path.basename(os.path.normpath(pasta_saida)) != 'graficos':
            pasta_saida = os.path.join(pasta_saida, 'graficos')
        os.makedirs(pasta_saida, exist_ok=True)
        
        # Carrega DataFrame de tokens
        if arquivo_excel:
            try:
                df_tokens = pd.read_excel(arquivo_excel, sheet_name='Resumo_Tokens')
            except Exception as e:
                print(f"⚠️  Aviso: Não foi possível carregar aba 'Resumo_Tokens': {e}")
                return []
        else:
            df_tokens = self._criar_dataframe_tokens()
        
        if df_tokens is None or df_tokens.empty:
            print("⚠️  Aviso: Nenhum dado de tokens disponível para gráficos")
            return []
        
        # Mapeia string de paleta para enum
        try:
            paleta_enum = Cores[paleta]
        except KeyError:
            print(f"⚠️  Paleta '{paleta}' desconhecida, usando Cividis")
            paleta_enum = Cores.Cividis
        
        arquivos_gerados = []
        
        # Usa nome do campo ID configurado
        nome_campo_id = self.dados_analise.config.nome_campo_id
        
        # Identifica tipos de tokens disponíveis (input, output, total, cache, reason)
        tipos_tokens = set()
        for col in df_tokens.columns:
            if col != nome_campo_id and '_' in col:
                tipo = col.split('_')[-1]
                if tipo not in ['finish']:  # Exclui finish_reason
                    tipos_tokens.add(tipo)
        
        # Para cada tipo de token, gera um gráfico comparativo usando grafico_multi_colunas
        for tipo in sorted(tipos_tokens):
            # Filtra colunas desse tipo
            colunas_tipo = [col for col in df_tokens.columns 
                           if col.endswith(f'_{tipo}') and df_tokens[col].dtype in [np.int64, np.float64]]
            
            if len(colunas_tipo) == 0:
                continue
            
            # Define nome do arquivo
            arquivo_grafico = os.path.join(pasta_saida, f'grafico_tokens_{tipo}.png')
            
            # Título e labels
            titulo = traduzir_rotulos(f'tokens_{tipo}', self._lang) if f'tokens_{tipo}' in _TRADUCOES else f'Tokens ({tipo})'
            
            # Extrai aliases (nomes dos modelos sem sufixo) presevando ordem no df
            aliases = []
            colunas_tipo_ordenadas = []
            
            # Pega lista de modelos inteira incluindo origem, mas só usa os que tem colunas
            rotulos = None
            if hasattr(self.dados_analise, 'rotulos_modelos'):
                rotulos = [self.dados_analise.rotulo_true] + self.dados_analise.rotulos_modelos
                
            if rotulos:
                for rotulo in rotulos:
                    for col in colunas_tipo:
                        if col.startswith(rotulo + '_'):
                            colunas_tipo_ordenadas.append(col)
                            aliases.append(col.rsplit('_', 1)[0])
                            break
            
            # Fallback para colunas restantes/todas caso rotulos falhe
            for col in colunas_tipo:
                if col not in colunas_tipo_ordenadas:
                    colunas_tipo_ordenadas.append(col)
                    aliases.append(col.rsplit('_', 1)[0])

            colunas_tipo = colunas_tipo_ordenadas
            # Configura gráfico boxplot usando grafico_multi_colunas
            configuracao = {
                titulo: {
                    'df': df_tokens,
                    'colunas': colunas_tipo,
                    'alias': aliases,
                    'x': traduzir_rotulos('modelo_xlabel', self._lang),
                    'y': traduzir_rotulos('tokens_ylabel', self._lang),
                    'agregacao': 'boxplot',
                    'paleta': paleta_enum,
                    'dropnan': True,
                    'rotacao_labels': 45
                }
            }
            
            # Gera gráfico
            try:
                UtilGraficos.grafico_multi_colunas(
                    configuracao=configuracao,
                    plots_por_linha=1,
                    paleta_cores=paleta_enum,
                    arquivo_saida=arquivo_grafico
                )
                arquivos_gerados.append(arquivo_grafico)
                print(f"   ✓ Gráfico gerado: {os.path.basename(arquivo_grafico)}")
            except Exception as e:
                print(f"⚠️  Erro ao gerar gráfico de tokens ({tipo}): {e}")
        
        if len(arquivos_gerados) > 0:
            print(f"✅ {len(arquivos_gerados)} gráficos de tokens gerados em: {pasta_saida}")
        
        return arquivos_gerados
    

    def gerar_graficos_avaliacao_llm(self, arquivo_excel: str = None, pasta_saida: str = None,
                                      paleta: str = 'Cividis') -> List[str]:
        """
        Gera gráficos boxplot para métricas de avaliação LLM por modelo.
        
        Cria gráficos separados para:
        - Métricas globais: P, R, F1, nota (padrão: grafico_bp_avaliacaollm_<metrica>.png)
        - Métricas por campo: modelo_campo_P/R/F1 (padrão: grafico_bp_avaliacaollm_<campo>_<metrica>.png)
        
        Args:
            arquivo_excel: caminho do arquivo Excel (se None, usa DataFrame em memória)
            pasta_saida: pasta para salvar gráficos (se None, usa pasta_analises)
            paleta: paleta de cores para os gráficos (padrão: Cividis)
        
        Returns:
            Lista com caminhos dos arquivos gerados
        
        Note:
            A limpeza de gráficos antigos deve ser feita antes de chamar este método.
        """
        import pandas as pd
        from util_graficos import UtilGraficos, Cores
        
        # Define pasta de saída
        if pasta_saida is None:
            pasta_saida = self.pasta_analises or '.'
        if os.path.basename(os.path.normpath(pasta_saida)) != 'graficos':
            pasta_saida = os.path.join(pasta_saida, 'graficos')
        os.makedirs(pasta_saida, exist_ok=True)
        
        # Carrega DataFrames de avaliação LLM (agora são dois: global e campos)
        if arquivo_excel:
            try:
                df_global = pd.read_excel(arquivo_excel, sheet_name='Avaliação LLM')
            except Exception as e:
                print(f"⚠️  Aviso: Não foi possível carregar aba 'Avaliação LLM': {e}")
                df_global = None
            
            try:
                df_campos = pd.read_excel(arquivo_excel, sheet_name='Avaliação LLM Campos')
            except Exception as e:
                # Normal não ter esta aba se não houver métricas por campo
                df_campos = None
        else:
            df_global, df_campos = self._criar_dataframe_avaliacao_llm()
        
        if df_global is None and df_campos is None:
            print("⚠️  Aviso: Nenhum dado de avaliação LLM disponível para gráficos")
            return []
        
        print(f"   📊 Gerando gráficos de avaliação LLM...")
        if df_global is not None:
            print(f"      - Métricas globais: {len(df_global)} linhas × {len(df_global.columns)} colunas")
        if df_campos is not None:
            print(f"      - Métricas por campo: {len(df_campos)} linhas × {len(df_campos.columns)} colunas")
        
        # Mapeia string de paleta para enum
        try:
            paleta_enum = Cores[paleta]
        except KeyError:
            print(f"⚠️  Paleta '{paleta}' desconhecida, usando Cividis")
            paleta_enum = Cores.Cividis
        
        arquivos_gerados = []
        
        # Usa nome do campo ID configurado
        nome_campo_id = self.dados_analise.config.nome_campo_id
        
        # Obtém lista de rótulos dos modelos (exclui apenas 'id', inclui origem e destinos)
        # rotulos[0] = 'id', rotulos[1] = origem, rotulos[2:] = destinos
        rotulos_modelos = self.rotulos[1:] if len(self.rotulos) > 1 else []
        
        # ═════════════════════════════════════════════════════════════════════
        # EXPORTA CSV CONSOLIDADO COM MÉTRICAS LLM (GLOBAL + POR CAMPO)
        # ═════════════════════════════════════════════════════════════════════
        try:
            self._exportar_csv_avaliacao_llm(
                df_global, df_campos, rotulos_modelos, nome_campo_id, pasta_saida
            )
        except Exception as e:
            print(f"⚠️  Erro ao exportar CSV de avaliação LLM: {e}")
        
        # ═════════════════════════════════════════════════════════════════════
        # GERA GRÁFICOS PARA MÉTRICAS GLOBAIS
        # ═════════════════════════════════════════════════════════════════════
        if df_global is not None:
            lang = self._lang
            
            # Para cada métrica numérica global
            for col in df_global.columns:
                if col == nome_campo_id or col.endswith('_explicacao'):
                    continue
                
                # Verifica se é coluna numérica
                if df_global[col].dtype not in [np.int64, np.float64]:
                    continue
                
                # Identifica modelo e métrica usando rótulos conhecidos
                modelo_identificado = None
                for rotulo in rotulos_modelos:
                    if col.startswith(rotulo + '_'):
                        modelo_identificado = rotulo
                        break
                
                if modelo_identificado is None:
                    continue
                
                # Extrai métrica (tudo depois do modelo_)
                metrica = col[len(modelo_identificado) + 1:]  # +1 para remover o underscore
                
                # Verifica se já gerou gráfico para esta métrica
                arquivo_grafico = os.path.join(pasta_saida, f'grafico_bp_avaliacaollm_{metrica.lower()}.png')
                if arquivo_grafico in arquivos_gerados:
                    continue
                
                # Coleta todas as colunas desta métrica
                colunas_por_modelo = []
                aliases_ordenados = []
                
                for rotulo in rotulos_modelos:
                    col_modelo = f'{rotulo}_{metrica}'
                    if col_modelo in df_global.columns:
                        colunas_por_modelo.append(col_modelo)
                        aliases_ordenados.append(rotulo)
                
                if not colunas_por_modelo:
                    continue
                
                # Título
                titulo_key = f'llm_global_{metrica}'
                titulo = traduzir_rotulos(titulo_key, lang) if titulo_key in _TRADUCOES else f'LLM Evaluation - {metrica} (Global)'
                
                # Configura gráfico boxplot
                configuracao = {
                    titulo: {
                        'df': df_global,
                        'colunas': colunas_por_modelo,
                        'alias': aliases_ordenados,
                        'x': traduzir_rotulos('modelo_xlabel', lang),
                        'y': metrica,
                        'agregacao': 'boxplot',
                        'paleta': paleta_enum,
                        'ylim': (0, 1) if metrica in ['P', 'R', 'F1'] else None,
                        'dropnan': True,
                        'rotacao_labels': 0 if len(aliases_ordenados) <= 5 else 45
                    }
                }
                
                # Gera gráfico
                try:
                    UtilGraficos.grafico_multi_colunas(
                        configuracao=configuracao,
                        plots_por_linha=1,
                        paleta_cores=paleta_enum,
                        arquivo_saida=arquivo_grafico
                    )
                    arquivos_gerados.append(arquivo_grafico)
                    print(f"   ✓ Gráfico gerado: {os.path.basename(arquivo_grafico)}")
                except Exception as e:
                    print(f"⚠️  Erro ao gerar gráfico de avaliação LLM global ({metrica}): {e}")
        
        # ═════════════════════════════════════════════════════════════════════
        # GERA GRÁFICOS PARA MÉTRICAS POR CAMPO
        # ═════════════════════════════════════════════════════════════════════
        if df_campos is not None:
            print(f"   📊 Processando gráficos por campo ({len(df_campos.columns)} colunas)...")
            # Agrupa colunas por campo e métrica
            # Estrutura: {campo: {metrica: [colunas]}}
            estrutura_campos = {}
            
            for col in df_campos.columns:
                if col == nome_campo_id:
                    continue
                
                # Verifica se é coluna numérica
                if df_campos[col].dtype not in [np.int64, np.float64]:
                    continue
                
                # Formato esperado: rotulo_campo_metrica (ex: agentes_tema_P, base_p_tema_F1)
                # Tenta identificar o modelo usando self.rotulos
                campo_metrica = None
                for rotulo in rotulos_modelos:
                    if col.startswith(rotulo + '_'):
                        # Remove o prefixo do modelo
                        campo_metrica = col[len(rotulo) + 1:]
                        break
                
                if not campo_metrica:
                    continue
                
                # Extrai campo e métrica do sufixo campo_metrica
                partes = campo_metrica.split('_')
                if len(partes) < 2:
                    continue
                
                # Métrica é sempre a última parte
                metrica = partes[-1]
                # Campo é tudo antes da métrica
                campo = '_'.join(partes[:-1])
                
                if campo not in estrutura_campos:
                    estrutura_campos[campo] = {}
                if metrica not in estrutura_campos[campo]:
                    estrutura_campos[campo][metrica] = []
                
                estrutura_campos[campo][metrica].append(col)
            
            print(f"   📊 Campos identificados: {len(estrutura_campos)} ({', '.join(sorted(estrutura_campos.keys())[:5])}...)")
            
            # Gera gráficos por campo e métrica
            for campo in sorted(estrutura_campos.keys()):
                for metrica in sorted(estrutura_campos[campo].keys()):
                    colunas_campo = estrutura_campos[campo][metrica]
                    
                    if not colunas_campo:
                        continue
                    
                    # Define nome do arquivo
                    arquivo_grafico = os.path.join(pasta_saida, 
                        f'grafico_bp_avaliacaollm_{campo.lower()}_{metrica.lower()}.png')
                    
                    # Título
                    titulo_key = f'llm_campo_{metrica}'
                    titulo = traduzir_rotulos(titulo_key, self._lang, campo=campo) if titulo_key in _TRADUCOES else f'LLM Evaluation - {campo} - {metrica}'
                    
                    # Agrupa colunas por modelo
                    colunas_por_modelo = []
                    aliases_ordenados = []
                    
                    for rotulo in rotulos_modelos:
                        col_modelo = f'{rotulo}_{campo}_{metrica}'
                        if col_modelo in colunas_campo:
                            colunas_por_modelo.append(col_modelo)
                            aliases_ordenados.append(rotulo)
                    
                    if not colunas_por_modelo:
                        continue
                    
                    # Configura gráfico boxplot
                    configuracao = {
                        titulo: {
                            'df': df_campos,
                            'colunas': colunas_por_modelo,
                            'alias': aliases_ordenados,
                            'x': traduzir_rotulos('modelo_xlabel', self._lang),
                            'y': metrica,
                            'agregacao': 'boxplot',
                            'paleta': paleta_enum,
                            'ylim': (0, 1) if metrica in ['P', 'R', 'F1'] else None,
                            'dropnan': True,
                            'rotacao_labels': 0 if len(aliases_ordenados) <= 5 else 45
                        }
                    }
                    
                    # Gera gráfico
                    try:
                        UtilGraficos.grafico_multi_colunas(
                            configuracao=configuracao,
                            plots_por_linha=1,
                            paleta_cores=paleta_enum,
                            arquivo_saida=arquivo_grafico
                        )
                        arquivos_gerados.append(arquivo_grafico)
                        print(f"   ✓ Gráfico gerado: {os.path.basename(arquivo_grafico)}")
                    except Exception as e:
                        print(f"⚠️  Erro ao gerar gráfico de avaliação LLM por campo ({campo}/{metrica}): {e}")
        
        if len(arquivos_gerados) > 0:
            print(f"✅ {len(arquivos_gerados)} gráficos de avaliação LLM gerados em: {pasta_saida}")
        
        return arquivos_gerados

    def _exportar_csv_avaliacao_llm(self, df_global, df_campos, rotulos_modelos, 
                                     nome_campo_id, pasta_saida):
        """
        Exporta CSV consolidado com métricas P, R, F1 da avaliação LLM.
        
        Formato:
            Modelo, Campo, Precision, Recall, F1
            GPT-5, Global, 0.78, 0.82, 0.87
            GPT-5, tema, 0.77, 0.81, 0.87
            ...
        
        Args:
            df_global: DataFrame com métricas globais (pode ser None)
            df_campos: DataFrame com métricas por campo (pode ser None)
            rotulos_modelos: lista de rótulos dos modelos
            nome_campo_id: nome da coluna ID
            pasta_saida: pasta para salvar o CSV
        """
        linhas_csv = []
        lang = self._lang
        
        # Rótulos de coluna segundo o idioma
        col_modelo = traduzir_rotulos('modelo_xlabel', lang)
        col_campo = traduzir_rotulos('campo_xlabel', lang)
        campo_global = 'Global'
        
        # ── Métricas globais ────────────────────────────────────────────────
        if df_global is not None:
            for modelo in rotulos_modelos:
                p_col = f'{modelo}_P'
                r_col = f'{modelo}_R'
                f1_col = f'{modelo}_F1'
                
                p_val = df_global[p_col].mean() if p_col in df_global.columns else None
                r_val = df_global[r_col].mean() if r_col in df_global.columns else None
                f1_val = df_global[f1_col].mean() if f1_col in df_global.columns else None
                
                if any(v is not None for v in [p_val, r_val, f1_val]):
                    linhas_csv.append({
                        col_modelo: modelo,
                        col_campo: campo_global,
                        'Precision': round(p_val, 4) if p_val is not None else '',
                        'Recall': round(r_val, 4) if r_val is not None else '',
                        'F1': round(f1_val, 4) if f1_val is not None else ''
                    })
        
        # ── Métricas por campo ──────────────────────────────────────────────
        if df_campos is not None:
            # Descobre campos disponíveis a partir das colunas
            campos_encontrados = set()
            for col in df_campos.columns:
                if col == nome_campo_id:
                    continue
                for modelo in rotulos_modelos:
                    if col.startswith(modelo + '_'):
                        resto = col[len(modelo) + 1:]  # campo_METRICA
                        partes = resto.split('_')
                        if len(partes) >= 2 and partes[-1] in ('P', 'R', 'F1'):
                            campo = '_'.join(partes[:-1])
                            campos_encontrados.add(campo)
                        break
            
            # Para cada modelo+campo, extrai médias de P, R, F1
            for modelo in rotulos_modelos:
                for campo in sorted(campos_encontrados):
                    p_col = f'{modelo}_{campo}_P'
                    r_col = f'{modelo}_{campo}_R'
                    f1_col = f'{modelo}_{campo}_F1'
                    
                    p_val = df_campos[p_col].mean() if p_col in df_campos.columns else None
                    r_val = df_campos[r_col].mean() if r_col in df_campos.columns else None
                    f1_val = df_campos[f1_col].mean() if f1_col in df_campos.columns else None
                    
                    if any(v is not None for v in [p_val, r_val, f1_val]):
                        linhas_csv.append({
                            col_modelo: modelo,
                            col_campo: campo,
                            'Precision': round(p_val, 4) if p_val is not None else '',
                            'Recall': round(r_val, 4) if r_val is not None else '',
                            'F1': round(f1_val, 4) if f1_val is not None else ''
                        })
        
        if not linhas_csv:
            print("⚠️  Nenhum dado de avaliação LLM para exportar como CSV")
            return
        
        # Monta DataFrame e salva
        df_csv = pd.DataFrame(linhas_csv, columns=[col_modelo, col_campo, 'Precision', 'Recall', 'F1'])
        
        # Separador decimal conforme idioma (vírgula para pt, ponto para en)
        sep_decimal = ',' if lang == 'pt' else '.'
        sep_csv = ';' if lang == 'pt' else ','
        
        # Salva na pasta de análise caso a pasta de saída seja filha dela 
        _pasta = self.pasta_analises if self.pasta_analises and self.pasta_analises in pasta_saida else pasta_saida
        arquivo_csv = os.path.join(_pasta, 'avaliacao_llm.csv')
        
        df_csv.to_csv(arquivo_csv, index=False, sep=sep_csv, decimal=sep_decimal, encoding='utf-8-sig')
        print(f"   ✓ CSV de avaliação LLM exportado: {os.path.basename(arquivo_csv)} ({len(linhas_csv)} linhas)")

    def gerar_graficos_observabilidade(self, arquivo_excel: str = None, pasta_saida: str = None,
                                       paleta: str = 'Cividis') -> List[str]:
        """
        Gera gráficos de observabilidade agrupados por sufixo (SEG, REV, IT, BYTES, QTD, AGT, OK).
        Cada arquivo PNG contém apenas boxplots com dados do mesmo sufixo para comparação entre modelos.
        
        Args:
            arquivo_excel: Caminho do arquivo Excel (se None, usa self._resultados)
            pasta_saida: Pasta para salvar gráficos (se None, usa self.pasta_analises)
            paleta: Paleta de cores (default: 'Cividis')
        
        Returns:
            Lista com caminhos dos arquivos gerados
        """
        import pandas as pd
        import os
        
        # Define pasta de saída
        if pasta_saida is None:
            pasta_saida = self.pasta_analises
            if pasta_saida is None:
                pasta_saida = '.'
        if os.path.basename(os.path.normpath(pasta_saida)) != 'graficos':
            pasta_saida = os.path.join(pasta_saida, 'graficos')
        os.makedirs(pasta_saida, exist_ok=True)
        
        # Mapeia string de paleta para enum
        try:
            paleta_enum = Cores[paleta]
        except KeyError:
            print(f"⚠️  Paleta '{paleta}' desconhecida, usando Cividis")
            paleta_enum = Cores.Cividis
        
        # Carrega dados de observabilidade
        if arquivo_excel:
            try:
                df_obs = pd.read_excel(arquivo_excel, sheet_name='Observabilidade')
            except Exception as e:
                print(f"⚠️  Erro ao carregar aba 'Observabilidade': {e}")
                return []
        else:
            df_obs = self._criar_dataframe_observabilidade()
        
        if df_obs is None or len(df_obs) == 0:
            print("⚠️  Nenhum dado de observabilidade disponível para gerar gráficos")
            return []
        
        # Identifica nome do campo ID
        nome_campo_id = self.dados_analise.config.nome_campo_id
        
        # Define sufixos de interesse e suas descrições
        sufixos_info = {
            'SEG': {
                'titulo': 'Tempo de Execução (segundos)',
                'ylabel': 'Segundos',
                'descricao': 'Tempo total de execução por modelo'
            },
            'REV': {
                'titulo': 'Revisões/Loops',
                'ylabel': 'Quantidade',
                'descricao': 'Número de revisões/loops por modelo'
            },
            'IT': {
                'titulo': 'Iterações',
                'ylabel': 'Quantidade',
                'descricao': 'Máximo de iterações por modelo'
            },
            'AGT': {
                'titulo': 'Agentes Executados',
                'ylabel': 'Quantidade',
                'descricao': 'Número de agentes executados por modelo'
            },
            'QTD': {
                'titulo': 'Campos Preenchidos',
                'ylabel': 'Quantidade',
                'descricao': 'Número de campos com valores (origem)'
            },
            'BYTES': {
                'titulo': 'Tamanho dos Campos (bytes)',
                'ylabel': 'Bytes',
                'descricao': 'Tamanho em bytes dos campos (origem)'
            },
            'OK': {
                'titulo': 'Taxa de Sucesso',
                'ylabel': 'Proporção',
                'descricao': 'Proporção de execuções bem-sucedidas'
            }
        }
        
        arquivos_gerados = []
        
        print(f"\n📊 Gerando gráficos de observabilidade por sufixo...")
        
        # Para cada sufixo, agrupa as colunas correspondentes
        for sufixo, info in sufixos_info.items():
            # Identifica colunas com este sufixo
            colunas_sufixo = []
            for col in df_obs.columns:
                if col == nome_campo_id:
                    continue
                # Verifica se termina com _SUFIXO ou contém _SUFIXO (sem ser parte do nome)
                if f'_{sufixo}' in col:
                    # Valida que é realmente o sufixo (não parte do nome)
                    partes = col.split('_')
                    if sufixo in partes:
                        colunas_sufixo.append(col)
            
            if not colunas_sufixo:
                continue
            
            # Prepara DataFrame com conversões necessárias
            df_plot = df_obs.copy()
            
            # Para sufixo OK, converte 'sim'/'não' para 1/0
            if sufixo == 'OK':
                for col in colunas_sufixo:
                    df_plot[col] = df_plot[col].map({
                        'sim': 1, 'não': 0, 
                        'Sim': 1, 'Não': 0,
                        True: 1, False: 0,
                        1: 1, 0: 0
                    })
            
            # Para BYTES, garante tipo numérico
            elif sufixo == 'BYTES':
                for col in colunas_sufixo:
                    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
            
            # Extrai aliases preservando ordem no df
            aliases = []
            colunas_sufixo_ordenadas = []
            
            # Pega lista de modelos do _rotulos_modelos e usa como base de ordenacao
            rotulos = None
            if hasattr(self.dados_analise, 'rotulos_modelos'):
                rotulos = [self.dados_analise.rotulo_true] + self.dados_analise.rotulos_modelos
                
            if rotulos:
               for rotulo in rotulos:
                   for col in colunas_sufixo:
                       if col.startswith(rotulo + '_'):
                           colunas_sufixo_ordenadas.append(col)
                           aliases.append(col.rsplit('_', 1)[0])
                           break
            
            # Fallback
            for col in colunas_sufixo:
               if col not in colunas_sufixo_ordenadas:
                   colunas_sufixo_ordenadas.append(col)
                   aliases.append(col.rsplit('_', 1)[0])
                   
            colunas_sufixo = colunas_sufixo_ordenadas
            
            # Gera boxplot usando UtilGraficos
            titulo = f"Observabilidade: {info['titulo']}"
            ylabel = info['ylabel']
            arquivo_grafico = os.path.join(pasta_saida, f'grafico_bp_observabilidade_{sufixo}.png')
            
            # Configura gráfico boxplot usando grafico_multi_colunas
            configuracao = {
                titulo: {
                    'df': df_plot,
                    'colunas': colunas_sufixo,
                    'alias': aliases,
                    'x': traduzir_rotulos('obs_metrica_xlabel', self._lang),
                    'y': ylabel,
                    'agregacao': 'boxplot',
                    'paleta': paleta_enum,
                    'dropnan': True,
                    'rotacao_labels': 90 if len(colunas_sufixo)>10 else 45 if len(colunas_sufixo) > 5 else 0
                }
            }
            
            try:
                UtilGraficos.grafico_multi_colunas(
                    configuracao=configuracao,
                    plots_por_linha=1,
                    paleta_cores=paleta_enum,
                    arquivo_saida=arquivo_grafico
                )
                arquivos_gerados.append(arquivo_grafico)
                print(f"   ✓ {sufixo}: {len(colunas_sufixo)} métricas → {os.path.basename(arquivo_grafico)}")
            except Exception as e:
                print(f"   ⚠️  Erro ao gerar gráfico {sufixo}: {e}")
        
        if len(arquivos_gerados) > 0:
            print(f"\n✅ {len(arquivos_gerados)} gráficos de observabilidade gerados em: {pasta_saida}")
        else:
            print(f"\n⚠️  Nenhum gráfico de observabilidade foi gerado")
        
        return arquivos_gerados
    

    def _carregar_dataframe_de_excel(self, arquivo_excel: str):
        """Carrega DataFrame do arquivo Excel (usa primeira aba de Resultados)"""
        import pandas as pd
        
        try:
            xl_file = pd.ExcelFile(arquivo_excel)
            
            # Procura por aba que começa com 'Resultados'
            for aba in xl_file.sheet_names:
                if aba.startswith('Resultados'):
                    return pd.read_excel(arquivo_excel, sheet_name=aba)
            
            # Fallback: tenta 'Resultados' padrão
            return pd.read_excel(arquivo_excel, sheet_name='Resultados')
        except Exception as e:
            print(f"⚠️  Erro ao carregar Excel: {e}")
            return None
    

    def _extrair_estrutura_metricas(self, df) -> dict:
        """
        Extrai estrutura de métricas do DataFrame.
        
        Returns:
            dict com estrutura:
            {
                'campo': {
                    'tecnica': {
                        'metrica': ['Modelo1_campo_tecnica_metrica', 'Modelo2_campo_tecnica_metrica']
                    }
                }
            }
        """
        import re
        
        estrutura = {}
        col_id = df.columns[0]  # Primeira coluna é ID
        
        # Extrai modelos diretamente das colunas do DataFrame
        # Formato das colunas: Modelo_campo_[tecnica_]metrica
        # Precisamos identificar todos os modelos únicos
        modelos_unicos = []
        # Técnicas conhecidas (ordem importa: sbert_grande antes de sbert para match correto)
        tecnicas_conhecidas = ['bertscore', 'rouge2', 'rouge1', 'rouge', 'levenshtein', 
                               'sbert_grande', 'sbert_medio', 'sbert_pequeno', 'sbert']
        metricas_validas = ['F1', 'P', 'R', 'LS', 'SIM']
        
        # Primeira passagem: identifica todos os modelos únicos
        for col in df.columns[1:]:  # Pula coluna ID
            # Tenta identificar o modelo (tudo antes do primeiro padrão de campo conhecido)
            # Campos conhecidos: (global), (estrutura), ou qualquer texto antes de técnica ou métrica
            
            # Busca por padrões de campo e técnica
            match_global = re.search(r'_\(global\)_', col)
            match_estrutura = re.search(r'_\(estrutura\)_', col)
            
            if match_global:
                modelo = col[:match_global.start()]
            elif match_estrutura:
                modelo = col[:match_estrutura.start()]
            else:
                # Tenta encontrar a técnica ou métrica e extrair o modelo antes
                partes = col.split('_')
                # Procura pela primeira técnica ou métrica conhecida
                idx_tec_met = -1
                for i, parte in enumerate(partes):
                    if parte in tecnicas_conhecidas or parte in metricas_validas:
                        idx_tec_met = i
                        break
                
                if idx_tec_met > 0:
                    # Modelo é tudo antes da primeira técnica/métrica, menos o campo (última parte antes)
                    modelo = '_'.join(partes[:idx_tec_met-1]) if idx_tec_met > 1 else partes[0]
                else:
                    # Fallback: pega apenas a primeira parte
                    modelo = partes[0]
            
            if modelo and modelo not in modelos_unicos:
                modelos_unicos.append(modelo)
        
        # Mantém a mesma ordem em que os modelos apareceram
        rotulos_modelos = modelos_unicos
        
        # Padrão: Modelo_campo_tecnica_metrica (ex: GPT4_(global)_rouge2_F1)
        
        for col in df.columns[1:]:  # Pula coluna ID
            # Identifica qual modelo corresponde a esta coluna
            modelo_encontrado = None
            resto_col = None
            
            for rotulo in rotulos_modelos:
                if col.startswith(f'{rotulo}_'):
                    modelo_encontrado = rotulo
                    resto_col = col[len(rotulo) + 1:]  # Remove "modelo_"
                    break
            
            if modelo_encontrado is None:
                continue  # Não conseguiu identificar o modelo
            
            # Agora processa o resto da coluna: campo_[tecnica_]metrica
            partes_resto = resto_col.split('_')
            
            if len(partes_resto) < 2:
                continue  # Formato inválido
            
            # Identifica se tem técnica no nome
            tecnica_encontrada = None
            metrica_pos = -1
            
            # Procura pela técnica
            for i, parte in enumerate(partes_resto):
                if parte in tecnicas_conhecidas:
                    tecnica_encontrada = parte
                    metrica_pos = i + 1
                    break
            
            # Extrai campo, técnica e métrica
            if tecnica_encontrada is None:
                # Formato sem técnica: campo_metrica
                metrica = partes_resto[-1]
                if metrica not in ['F1', 'P', 'R', 'LS', 'SIM']:
                    continue
                
                # Campo é tudo exceto a métrica
                campo = '_'.join(partes_resto[:-1])
                tecnica = 'geral'
            else:
                # Formato com técnica: campo_tecnica_metrica
                metrica = partes_resto[metrica_pos] if metrica_pos < len(partes_resto) else None
                if metrica not in ['F1', 'P', 'R', 'LS', 'SIM']:
                    continue
                
                # Campo é tudo até a técnica
                idx_tecnica = partes_resto.index(tecnica_encontrada)
                campo = '_'.join(partes_resto[:idx_tecnica])
                tecnica = tecnica_encontrada
            
            # Adiciona à estrutura
            if campo not in estrutura:
                estrutura[campo] = {}
            if tecnica not in estrutura[campo]:
                estrutura[campo][tecnica] = {}
            if metrica not in estrutura[campo][tecnica]:
                estrutura[campo][tecnica][metrica] = []
            
            estrutura[campo][tecnica][metrica].append(col)
        
        return estrutura
    

    @staticmethod
    def _extrair_estrutura_metricas_estatico(df, tecnica_aba: str = '', rotulos_modelos_ordenados: list = None) -> dict:
        """
        Versão estática de _extrair_estrutura_metricas.
        Extrai estrutura mesmo quando há ou não técnica nos nomes das colunas.
        
        Formatos suportados:
        - Modelo_campo_tecnica_metrica (ex: agentes_gpt5_(global)_bertscore_F1)
        - Modelo_campo_metrica (ex: agentes_gpt5_(global)_F1 - quando vem de aba específica)
        
        IMPORTANTE: Modelo pode conter underscores (ex: agentes_gpt5, base_gemma3)
        """
        estrutura = {}
        col_id = df.columns[0]
        
        # Técnicas e métricas conhecidas (ordem importa: sbert_* antes de sbert)
        tecnicas_conhecidas = ['bertscore', 'rouge2', 'rouge1', 'rouge', 'rougel', 'levenshtein',
                               'sbert_grande', 'sbert_medio', 'sbert_pequeno', 'sbert']
        metricas_validas = ['F1', 'P', 'R', 'LS', 'SIM']
        
        for col in df.columns[1:]:
            partes = col.split('_')
            
            if len(partes) < 3:
                continue
            
            # Última parte DEVE ser métrica
            metrica = partes[-1]
            if metrica not in metricas_validas:
                continue
            
            # Detecta se há técnica (penúltima ou antes da métrica)
            tecnica_encontrada = None
            idx_tecnica = -1
            
            for i, parte in enumerate(partes[:-1]):  # Não inclui métrica
                if parte in tecnicas_conhecidas:
                    tecnica_encontrada = parte
                    idx_tecnica = i
                    break
            
            if tecnica_encontrada:
                # Formato: Modelo_..._campo_tecnica_metrica
                # Campo está entre início e técnica
                # Modelo é a primeira parte que não é campo nem técnica
                # Simplificação: campo é a parte imediatamente antes da técnica
                campo = partes[idx_tecnica - 1] if idx_tecnica > 0 else ''
                # Modelo é tudo antes do campo
                modelo = '_'.join(partes[:idx_tecnica-1]) if idx_tecnica > 1 else partes[0]
                tecnica = tecnica_encontrada
            else:
                # Formato: Modelo_..._campo_metrica (sem técnica)
                # Campo é a penúltima parte
                # Modelo é tudo antes do campo
                campo = partes[-2] if len(partes) >= 2 else ''
                modelo = '_'.join(partes[:-2]) if len(partes) > 2 else partes[0]
                # Converte hífen para underscore para manter consistência
                tecnica = tecnica_aba.lower().replace('-', '_') if tecnica_aba else 'geral'
            
            # Adiciona à estrutura
            if campo not in estrutura:
                estrutura[campo] = {}
            if tecnica not in estrutura[campo]:
                estrutura[campo][tecnica] = {}
            if metrica not in estrutura[campo][tecnica]:
                estrutura[campo][tecnica][metrica] = []
            
            estrutura[campo][tecnica][metrica].append(col)
        
        return estrutura
    

    def _gerar_boxplots_por_campo_metrica(self, df, estrutura: dict, 
                                         pasta_saida: str, paleta: str) -> List[str]:
        """
        Gera boxplots agrupando todos os modelos para cada combinação campo+metrica+tecnica.
        
        Um gráfico por campo: compara TODOS os modelos lado a lado naquele campo específico.
        
        Padrão de nomes: grafico_bp_<tecnica>_<campo>_<metrica>.png
        Exemplo: grafico_bp_BERTScore_notas_F1.png (todos os modelos comparados para o campo 'notas')
        """
        from util_graficos import UtilGraficos, Cores
        
        arquivos_gerados = []
        
        # Mapeia string de paleta para enum
        try:
            paleta_enum = Cores[paleta] if isinstance(paleta, str) else paleta
        except KeyError:
            print(f"⚠️  Paleta '{paleta}' não encontrada, usando 'Cividis'")
            paleta_enum = Cores.Cividis
        
        # Obtém lista de rótulos dos modelos (exclui 'id' e 'True')
        # self.rotulos = ['id', 'True', 'modelo1', 'modelo2', ...]
        rotulos_modelos = self.rotulos[2:] if len(self.rotulos) > 2 else []
        
        if not rotulos_modelos:
            print("⚠️  Aviso: Nenhum rótulo de modelo encontrado em self.rotulos")
            return []
        
        # Para cada campo+técnica+métrica, agrupa colunas por modelo
        for campo, tecnicas in estrutura.items():
            for tecnica, metricas in tecnicas.items():
                for metrica, colunas in metricas.items():
                    if not colunas:
                        continue
                    
                    # Nome da técnica para arquivo (normalizado)
                    tecnica_nome = {
                        'bertscore': 'BERTScore',
                        'rouge2': 'ROUGE2',
                        'rouge1': 'ROUGE1',
                        'rouge': 'ROUGEL',
                        'levenshtein': 'Levenshtein',
                        'geral': 'Geral'
                    }.get(tecnica, tecnica.upper())
                    
                    # Nome do campo para arquivo (normaliza)
                    campo_safe = campo.replace('(', '').replace(')', '').replace('.', '_')
                    
                    # Nome do arquivo: grafico_bp_<tecnica>_<campo>_<metrica>.png
                    nome_arquivo = f'grafico_bp_{tecnica_nome}_{campo_safe}_{metrica}.png'
                    caminho_completo = os.path.join(pasta_saida, nome_arquivo)
                    
                    # Identifica qual coluna pertence a qual modelo usando rótulos
                    # Formato esperado: <rotulo>_<campo>_<tecnica>_<metrica>
                    colunas_por_modelo = []  # Lista de colunas na ordem dos rótulos
                    aliases_ordenados = []   # Nomes dos modelos para legenda
                    
                    for rotulo in rotulos_modelos:
                        # Busca coluna que começa com este rótulo
                        col_modelo = None
                        for col in colunas:
                            if col.startswith(f'{rotulo}_'):
                                col_modelo = col
                                break
                        
                        # Se encontrou, adiciona à lista ordenada
                        if col_modelo is not None:
                            colunas_por_modelo.append(col_modelo)
                            aliases_ordenados.append(rotulo)
                    
                    # Se não encontrou nenhuma coluna, pula este gráfico
                    if not colunas_por_modelo:
                        print(f"⚠️  Aviso: Nenhuma coluna encontrada para {campo}/{tecnica}/{metrica}")
                        continue
                    
                    # Título do gráfico
                    titulo = f'{campo} - {metrica} ({tecnica_nome})'
                    
                    # Configuração: um boxplot por modelo, comparando todos lado a lado
                    config = {
                        titulo: {
                            'df': df,
                            'colunas': colunas_por_modelo,  # Colunas ordenadas por modelo
                            'alias': aliases_ordenados,      # Nomes dos modelos (rótulos)
                            'x': traduzir_rotulos('modelos_xlabel', self._lang),
                            'y': metrica,
                            'agregacao': 'boxplot',
                            'paleta': paleta_enum,
                            'ylim': (0, 1),                  # Fixar eixo Y em [0, 1] para métricas
                            'dropnan': True,
                            'drop_zero': False,
                            'rotacao_labels': 0 if len(aliases_ordenados) <= 5 else 45
                        }
                    }
                    
                    try:
                        UtilGraficos.grafico_multi_colunas(
                            config,
                            plots_por_linha=1,
                            arquivo_saida=caminho_completo
                        )
                        arquivos_gerados.append(caminho_completo)
                    except Exception as e:
                        print(f"⚠️  Erro ao gerar gráfico {nome_arquivo}: {e}")
        
        return arquivos_gerados
    

    @staticmethod
    def _gerar_boxplots_por_campo_metrica_estatico(df, estrutura: dict, 
                                                   pasta_saida: str, paleta: str,
                                                   tecnica_aba: str = '', rotulos_modelos_ordenados: list = None,
                                                   lang: str = 'pt') -> List[str]:
        """
        Versão estática - Gera boxplots agrupando todos os modelos para cada campo.
        
        Um gráfico por campo: compara TODOS os modelos lado a lado naquele campo específico.
        
        Padrão: grafico_bp_<tecnica>_<campo>_<metrica>.png
        """
        from util_graficos import UtilGraficos, Cores
        
        arquivos_gerados = []
        
        try:
            paleta_enum = Cores[paleta] if isinstance(paleta, str) else paleta
        except KeyError:
            paleta_enum = Cores.Cividis
        
        # Extrai lista de modelos únicos de todas as colunas do DataFrame
        # Formato das colunas: <modelo>_<campo>_<tecnica>_<metrica>
        modelos_unicos = []
        col_id = df.columns[0]  # Primeira coluna é ID
        
        for col in df.columns[1:]:  # Pula coluna ID
            # Extrai modelo: tudo antes do primeiro padrão conhecido de campo
            # Padrões: _(<campo>)_ ou _(global)_ ou _(estrutura)_
            
            # Busca por padrões de campo especiais
            match_global = re.search(r'_\(global\)_', col)
            match_estrutura = re.search(r'_\(estrutura\)_', col)
            match_campo_parenteses = re.search(r'_\([^)]+\)_', col)
            
            if match_global:
                modelo = col[:match_global.start()]
            elif match_estrutura:
                modelo = col[:match_estrutura.start()]
            elif match_campo_parenteses:
                modelo = col[:match_campo_parenteses.start()]
            else:
                # Para campos sem parênteses, tenta encontrar o padrão _campo_tecnica_metrica
                # Remove última parte (métrica: F1, P, R, LS)
                partes = col.rsplit('_', 1)
                if len(partes) >= 2 and partes[1] in ['F1', 'P', 'R', 'LS', 'SIM']:
                    resto = partes[0]
                    # Remove penúltima parte se for técnica conhecida
                    partes2 = resto.rsplit('_', 1)
                    tecnicas_conhecidas = ['bertscore', 'rouge', 'rouge1', 'rouge2', 'rougel', 'levenshtein',
                                           'sbert_grande', 'sbert_medio', 'sbert_pequeno', 'sbert']
                    if len(partes2) >= 2 and partes2[1] in tecnicas_conhecidas:
                        resto = partes2[0]
                    # Remove campo (antepenúltima parte)
                    partes3 = resto.rsplit('_', 1)
                    if len(partes3) >= 2:
                        modelo = partes3[0]
                    else:
                        modelo = resto
                else:
                    # Fallback: pega primeira parte
                    modelo = col.split('_')[0]
            
            if modelo and modelo not in modelos_unicos:
                modelos_unicos.append(modelo)
        
        # Mantém a ordem original do DataFrame ou do YAML se fornecida
        if rotulos_modelos_ordenados:
            modelos_ordenados = [m for m in rotulos_modelos_ordenados if m in modelos_unicos]
            for m in modelos_unicos:
                if m not in modelos_ordenados:
                    modelos_ordenados.append(m)
        else:
            modelos_ordenados = modelos_unicos
        
        for campo, tecnicas in estrutura.items():
            for tecnica, metricas in tecnicas.items():
                for metrica, colunas in metricas.items():
                    if not colunas:
                        continue
                    
                    # Nome da técnica para arquivo
                    tecnica_nome = {
                        'bertscore': 'BERTScore',
                        'rouge2': 'ROUGE2',
                        'rouge1': 'ROUGE1',
                        'rouge': 'ROUGEL',
                        'levenshtein': 'Levenshtein',
                        'geral': 'Geral'
                    }.get(tecnica, tecnica_aba if tecnica_aba else tecnica.upper())
                    
                    campo_safe = campo.replace('(', '').replace(')', '').replace('.', '_')
                    
                    # Nome: boxplot_<tecnica>_<campo>_<metrica>.png
                    nome_arquivo = f'boxplot_{tecnica_nome}_{campo_safe}_{metrica}.png'
                    caminho_completo = os.path.join(pasta_saida, nome_arquivo)
                    
                    # Identifica qual coluna pertence a qual modelo (ordenado)
                    colunas_por_modelo = []
                    aliases_ordenados = []
                    
                    for modelo in modelos_ordenados:
                        # Busca coluna que começa com este modelo
                        col_modelo = None
                        for col in colunas:
                            if col.startswith(f'{modelo}_'):
                                col_modelo = col
                                break
                        
                        # Se encontrou, adiciona à lista ordenada
                        if col_modelo is not None:
                            colunas_por_modelo.append(col_modelo)
                            aliases_ordenados.append(modelo)
                    
                    # Se não encontrou nenhuma coluna, pula
                    if not colunas_por_modelo:
                        print(f"⚠️  Aviso: Nenhuma coluna encontrada para {campo}/{tecnica}/{metrica}")
                        continue
                    
                    # Título
                    titulo = f'{campo} - {metrica} ({tecnica_nome})'
                    
                    # Configuração: um boxplot por modelo, comparando todos lado a lado
                    config = {
                        titulo: {
                            'df': df,
                            'colunas': colunas_por_modelo,  # Colunas ordenadas por modelo
                            'alias': aliases_ordenados,      # Nomes dos modelos
                            'x': traduzir_rotulos('modelos_xlabel', lang),
                            'y': metrica,
                            'agregacao': 'boxplot',
                            'paleta': paleta_enum,
                            'ylim': (0, 1),                  # Fixar eixo Y em [0, 1] para métricas
                            'dropnan': True,
                            'drop_zero': False,
                            'rotacao_labels': 0 if len(aliases_ordenados) <= 5 else 45
                        }
                    }
                    
                    try:
                        UtilGraficos.grafico_multi_colunas(
                            config,
                            plots_por_linha=1,
                            arquivo_saida=caminho_completo
                        )
                        arquivos_gerados.append(caminho_completo)
                    except Exception as e:
                        print(f"⚠️  Erro ao gerar gráfico {nome_arquivo}: {e}")
        
        return arquivos_gerados
    

    def _gerar_graficos_adicionais(self, df, estrutura: dict, 
                                   pasta_saida: str, paleta: str) -> List[str]:
        """
        Gera gráficos adicionais de interesse:
        1. Comparação global F1 entre modelos (todos os campos)
        2. Distribuição de métricas por técnica
        3. Comparação Loss (LS) entre modelos
        
        Padrão de nomes: grafico_comp_*.png, grafico_dist_*.png
        """
        from util_graficos import UtilGraficos, Cores
        
        arquivos_gerados = []
        
        try:
            paleta_enum = Cores[paleta] if isinstance(paleta, str) else paleta
        except KeyError:
            paleta_enum = Cores.Cividis
        
        # 1. Gráfico de comparação F1 global entre todos os modelos
        if '(global)' in estrutura:
            for tecnica, metricas in estrutura['(global)'].items():
                if 'F1' in metricas and metricas['F1']:
                    colunas_f1 = metricas['F1']
                    
                    # Extrai aliases corretamente
                    aliases = []
                    for col in colunas_f1:
                        partes = col.split('_')
                        # Remove métrica
                        partes = partes[:-1]
                        # Remove técnica se existir (ordem importa: sbert_* antes de sbert)
                        tecnicas_conhecidas = ['bertscore', 'rouge2', 'rouge1', 'rouge', 'rougel', 'levenshtein',
                                               'sbert_grande', 'sbert_medio', 'sbert_pequeno', 'sbert']
                        if partes and partes[-1] in tecnicas_conhecidas:
                            partes = partes[:-1]
                        # Remove campo
                        if len(partes) > 1:
                            partes = partes[:-1]
                        # Modelo é o que sobrou
                        modelo = '_'.join(partes) if partes else col.split('_')[0]
                        aliases.append(modelo)
                    
                    tecnica_label = tecnica.upper() if tecnica != 'geral' else ''
                    titulo = traduzir_rotulos('comp_global_f1', self._lang)
                    if tecnica_label:
                        titulo += f' ({tecnica_label})'
                    
                    nome_arquivo = f'grafico_comp_global_f1_{tecnica}.png'
                    caminho_completo = os.path.join(pasta_saida, nome_arquivo)
                    
                    config = {
                        titulo: {
                            'df': df,
                            'colunas': colunas_f1,
                            'alias': aliases,
                            'x': traduzir_rotulos('modelos_xlabel', self._lang),
                            'y': 'F1 Score',
                            'agregacao': 'boxplot',
                            'paleta': paleta_enum,
                            'ylim': (0, 1),                  # Fixar eixo Y em [0, 1] para métricas
                            'dropnan': True,
                            'rotacao_labels': 0 if len(aliases) <= 5 else 45
                        }
                    }
                    
                    try:
                        UtilGraficos.grafico_multi_colunas(
                            config,
                            plots_por_linha=1,
                            arquivo_saida=caminho_completo
                        )
                        arquivos_gerados.append(caminho_completo)
                    except Exception as e:
                        print(f"⚠️  Erro ao gerar gráfico {nome_arquivo}: {e}")
        
        # 2. Gráfico de média F1 por campo (compara desempenho entre campos)
        # Coleta todas as colunas F1 de todos os campos e técnicas
        todas_f1_por_campo = {}
        for campo, tecnicas in estrutura.items():
            for tecnica, metricas in tecnicas.items():
                if 'F1' in metricas and metricas['F1']:
                    chave = f'{campo}_{tecnica}' if tecnica != 'geral' else campo
                    todas_f1_por_campo[chave] = metricas['F1']
        
        if len(todas_f1_por_campo) > 1:
            # Pega apenas o primeiro modelo para comparar entre campos
            primeira_col_por_campo = {}
            for campo_tec, colunas in todas_f1_por_campo.items():
                if colunas:
                    primeira_col_por_campo[campo_tec] = colunas[0]
            
            if len(primeira_col_por_campo) > 1:
                nome_arquivo = 'grafico_dist_f1_por_campo.png'
                caminho_completo = os.path.join(pasta_saida, nome_arquivo)
                
                config = {
                    traduzir_rotulos('dist_f1_campo', self._lang): {
                        'df': df,
                        'colunas': list(primeira_col_por_campo.values()),
                        'alias': [k.replace('_', ' ') for k in primeira_col_por_campo.keys()],
                        'x': traduzir_rotulos('campos_xlabel', self._lang),
                        'y': 'F1 Score',
                        'agregacao': 'boxplot',
                        'paleta': paleta_enum,
                        'ylim': (0, 1),                  # Fixar eixo Y em [0, 1] para métricas
                        'dropnan': True,
                        'rotacao_labels': 90
                    }
                }
                
                try:
                    UtilGraficos.grafico_multi_colunas(
                        config,
                        plots_por_linha=1,
                        arquivo_saida=caminho_completo
                    )
                    arquivos_gerados.append(caminho_completo)
                except Exception as e:
                    print(f"⚠️  Erro ao gerar gráfico {nome_arquivo}: {e}")
        
        # 3. Gráfico de comparação SIM (Levenshtein) global entre modelos
        if '(global)' in estrutura:
            for tecnica, metricas in estrutura['(global)'].items():
                if 'SIM' in metricas and metricas['SIM']:
                    colunas_sim = metricas['SIM']
                    
                    # Extrai aliases corretamente
                    aliases = []
                    for col in colunas_sim:
                        partes = col.split('_')
                        # Remove métrica
                        partes = partes[:-1]
                        # Remove técnica se existir (ordem importa: sbert_* antes de sbert)
                        tecnicas_conhecidas = ['bertscore', 'rouge2', 'rouge1', 'rouge', 'rougel', 'levenshtein',
                                               'sbert_grande', 'sbert_medio', 'sbert_pequeno', 'sbert']
                        if partes and partes[-1] in tecnicas_conhecidas:
                            partes = partes[:-1]
                        # Remove campo
                        if len(partes) > 1:
                            partes = partes[:-1]
                        # Modelo é o que sobrou
                        modelo = '_'.join(partes) if partes else col.split('_')[0]
                        aliases.append(modelo)
                    
                    tecnica_label = tecnica.upper() if tecnica != 'geral' else 'Levenshtein'
                    titulo = traduzir_rotulos('comp_global_sim', self._lang)
                    if tecnica_label:
                        titulo += f' ({tecnica_label})'
                    
                    nome_arquivo = f'grafico_comp_global_sim_{tecnica}.png'
                    caminho_completo = os.path.join(pasta_saida, nome_arquivo)
                    
                    config = {
                        titulo: {
                            'df': df,
                            'colunas': colunas_sim,
                            'alias': aliases,
                            'x': traduzir_rotulos('modelos_xlabel', self._lang),
                            'y': traduzir_rotulos('sim_ylabel', self._lang),
                            'agregacao': 'boxplot',
                            'paleta': paleta_enum,
                            'ylim': (0, 1),                  # Fixar eixo Y em [0, 1] para métricas
                            'dropnan': True,
                            'rotacao_labels': 0 if len(aliases) <= 5 else 45
                        }
                    }
                    
                    try:
                        UtilGraficos.grafico_multi_colunas(
                            config,
                            plots_por_linha=1,
                            arquivo_saida=caminho_completo
                        )
                        arquivos_gerados.append(caminho_completo)
                    except Exception as e:
                        print(f"⚠️  Erro ao gerar gráfico {nome_arquivo}: {e}")
        
        return arquivos_gerados


