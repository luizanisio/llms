# -*- coding: utf-8 -*-
"""
Classe para análise estatística de resultados de avaliação LLM-as-a-Judge.
Realiza testes de normalidade (Shapiro-Wilk), testes de hipótese (Wilcoxon)
e análises de eficiência de custo.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

class AnaliseEstatistica:
    def __init__(self, df_dados, config=None):
        """
        Inicializa a classe de análise estatística.
        
        Args:
            df_dados (pd.DataFrame): DataFrame com os dados para análise.
                Colunas esperadas: valor1, valor2, custo1, custo2, familia
            config (dict, optional): Configurações adicionais.
                Chaves esperadas:
                - rotulo1: Nome do rótulo 1 (ex: "Base")
                - rotulo2: Nome do rótulo 2 (ex: "Agentes")
                - arquivo_saida: Caminho para salvar o relatório markdown
        """
        self.df = df_dados.copy()
        self.config = config or {}
        self.rotulo1 = self.config.get('rotulo1', 'Rótulo 1')
        self.rotulo2 = self.config.get('rotulo2', 'Rótulo 2')
        self.arquivo_saida = self.config.get('arquivo_saida', 'relatorio_estatistico.md')
        
        # Validação básica
        colunas_necessarias = ['valor1', 'valor2', 'custo1', 'custo2', 'familia']
        for col in colunas_necessarias:
            if col not in self.df.columns:
                raise ValueError(f"Coluna obrigatória não encontrada: {col}")
                
        # Calcula deltas
        self.df['delta_valor'] = self.df['valor2'] - self.df['valor1']
        self.df['delta_custo'] = self.df['custo2'] - self.df['custo1']
        
        self.resultados_shapiro = []
        self.resultados_wilcoxon = []
        self.resultados_eficiencia = []
        self.resultados_geral = []

    def processar_analise(self):
        """Executa todas as análises e gera o relatório."""
        self._calcular_shapiro()
        self._calcular_wilcoxon()
        self._calcular_eficiencia()
        self._calcular_desempenho_geral()
        self._gerar_relatorio_markdown()

    def _calcular_desempenho_geral(self):
        """
        Calcula o desempenho geral consolidado (ranking).
        Colunas: Posição, Rótulo_Família, Tipo_Rótulo, Média, Desvio, Mediana, Suporte
        """
        dados_geral = []
        
        # Processar Rótulo 1 (ex: Base)
        # Agrupa por família
        familias = self.df['familia'].unique()
        for fam in familias:
            df_fam = self.df[self.df['familia'] == fam]
            
            # Rótulo 1
            v1 = df_fam['valor1'].dropna()
            if len(v1) > 0:
                dados_geral.append({
                    'familia': fam,
                    'rotulo_composto': f"{self.rotulo1} {fam}",
                    'tipo_rotulo': self.rotulo1,
                    'media': v1.mean(),
                    'std': v1.std(),
                    'mediana': v1.median(),
                    'suporte': len(v1)
                })
            
            # Rótulo 2
            v2 = df_fam['valor2'].dropna()
            if len(v2) > 0:
                dados_geral.append({
                    'familia': fam,
                    'rotulo_composto': f"{self.rotulo2} {fam}",
                    'tipo_rotulo': self.rotulo2,
                    'media': v2.mean(),
                    'std': v2.std(),
                    'mediana': v2.median(),
                    'suporte': len(v2)
                })
        
        df_geral = pd.DataFrame(dados_geral)
        
        # Ordenar por média decrescente
        if not df_geral.empty:
            df_geral = df_geral.sort_values(by='media', ascending=False).reset_index(drop=True)
            df_geral['posicao'] = df_geral.index + 1
        
        self.resultados_geral = df_geral

    def _calcular_shapiro(self):
        """
        Realiza o teste de Shapiro-Wilk para verificar normalidade dos deltas.
        Analisa globalmente e por família.
        """
        resultados = []
        
        # Global
        deltas_global = self.df['delta_valor'].dropna()
        stat, p = stats.shapiro(deltas_global)
        resultados.append({
            'familia': 'Global',
            'n': len(deltas_global),
            'shapiro_w': stat,
            'shapiro_p': p,
            'normal': p > 0.05
        })
        
        # Por família
        familias = self.df['familia'].unique()
        for familia in familias:
            df_fam = self.df[self.df['familia'] == familia]
            deltas = df_fam['delta_valor'].dropna()
            if len(deltas) >= 3: # Shapiro requer pelo menos 3 amostras
                stat, p = stats.shapiro(deltas)
                resultados.append({
                    'familia': familia,
                    'n': len(deltas),
                    'shapiro_w': stat,
                    'shapiro_p': p,
                    'normal': p > 0.05
                })
        
        self.resultados_shapiro = pd.DataFrame(resultados)

    def _calcular_wilcoxon(self):
        """
        Realiza o teste de postos pareados de Wilcoxon.
        Calcula também o tamanho do efeito (Cohen's d aproximado ou r).
        """
        resultados = []
        
        grupos = [{'nome': 'Global', 'df': self.df}]
        for familia in self.df['familia'].unique():
            grupos.append({'nome': familia, 'df': self.df[self.df['familia'] == familia]})
            
        for grupo in grupos:
            nome = grupo['nome']
            df_g = grupo['df']
            
            v1 = df_g['valor1'].dropna()
            v2 = df_g['valor2'].dropna()
            
            # Garante pareamento
            idx_comum = v1.index.intersection(v2.index)
            v1 = v1.loc[idx_comum]
            v2 = v2.loc[idx_comum]
            
            if len(v1) == 0:
                continue

            media1 = v1.mean()
            media2 = v2.mean()
            diferenca = media2 - media1
            
            # Wilcoxon
            # alternative='two-sided' é o padrão, mas podemos querer saber se melhorou, etc.
            # Aqui vamos usar two-sided para ver se há diferença
            try:
                stat, p_value = stats.wilcoxon(v1, v2)
            except ValueError:
                # Pode acontecer se todos os valores forem idênticos (diferença zero)
                p_value = 1.0
            
            # Tamanho do efeito (básico: Cohen's d para pareado)
            # d = mean(diff) / std(diff)
            diff = v2 - v1
            std_diff = diff.std()
            
            if np.isclose(std_diff, 0):
                cohen_d = 0.0 # Diferença constante, sem variabilidade no delta
            else:
                cohen_d = diff.mean() / std_diff
            
            # Classificação do Tamanho do Efeito (Cohen's d) - Cohen (1988)
            abs_d = abs(cohen_d)
            if abs_d < 0.10:
                tamanho_efeito = "Insignificante"
            elif abs_d < 0.30:
                tamanho_efeito = "Pequeno"
            elif abs_d < 0.50:
                tamanho_efeito = "Médio"
            else:
                tamanho_efeito = "Grande"

            resultados.append({
                'contexto': nome,
                'media_rotulo1': media1,
                'media_rotulo2': media2,
                'diferenca': diferenca,
                'p_value': p_value,
                'significante': p_value < 0.05,
                'cohen_d': cohen_d,
                'tamanho_efeito': tamanho_efeito
            })
            
        self.resultados_wilcoxon = pd.DataFrame(resultados)

    def _calcular_eficiencia(self):
        """Calcula métricas de eficiência relacionadas ao custo."""
        resultados = []
        
        grupos = [{'nome': 'Global', 'df': self.df}]
        for familia in self.df['familia'].unique():
            grupos.append({'nome': familia, 'df': self.df[self.df['familia'] == familia]})
            
        for grupo in grupos:
            nome = grupo['nome']
            df_g = grupo['df']
            
            # Médias
            custo_med1 = df_g['custo1'].mean()
            custo_med2 = df_g['custo2'].mean()
            
            valor_med1 = df_g['valor1'].mean()
            valor_med2 = df_g['valor2'].mean()
            
            # Deltas percentuais
            delta_custo_pct = ((custo_med2 - custo_med1) / custo_med1 * 100) if custo_med1 > 0 else 0
            delta_valor_pct = ((valor_med2 - valor_med1) / valor_med1 * 100) if valor_med1 > 0 else 0
            
            # Eficiência (Valor / Custo)
            # Evitar divisão por zero e custos muito baixos
            efi1 = (valor_med1 / custo_med1) if custo_med1 > 0 else 0
            efi2 = (valor_med2 / custo_med2) if custo_med2 > 0 else 0
            
            delta_efi = ((efi2 - efi1) / efi1 * 100) if efi1 > 0 else 0
            
            resultados.append({
                'familia': nome,
                'custo_medio1': custo_med1,
                'custo_medio2': custo_med2,
                'delta_custo_pct': delta_custo_pct,
                'delta_valor_pct': delta_valor_pct,
                'eficiencia1': efi1,
                'eficiencia2': efi2,
                'delta_eficiencia_pct': delta_efi
            })
            
        self.resultados_eficiencia = pd.DataFrame(resultados)

    def _gerar_relatorio_markdown(self):
        """Gera o conteúdo do relatório em Markdown."""
        lines = []
        lines.append(f"# Relatório de Análise Estatística: {self.rotulo1} vs {self.rotulo2}")
        lines.append("")
        
        # 0. Desempenho Geral (Ranking)
        lines.append("## 1. Desempenho Geral (Ranking)")
        lines.append("Ranking ordenado pelo valor médio de desempenho.")
        lines.append("")
        lines.append(f"| Posição | Rótulo / Família | Tipo | Média | Desvio Padrão | Mediana | Suporte |")
        lines.append(f"|---|---|---|---|---|---|---|")
        
        if not self.resultados_geral.empty:
            for _, row in self.resultados_geral.iterrows():
                lines.append(f"| {row['posicao']} | {row['rotulo_composto']} | {row['tipo_rotulo']} | {row['media']:.4f} | {row['std']:.4f} | {row['mediana']:.4f} | {row['suporte']} |")
        else:
            lines.append(f"| - | - | - | - | - | - | - |")
        lines.append("")

        # 1. Shapiro-Wilk
        lines.append("## 2. Teste de Normalidade (Shapiro-Wilk)")
        lines.append("Verifica se a distribuição das diferenças (deltas) entre os pares segue uma distribuição normal.")
        lines.append("")
        lines.append("| Família | Amostras (n) | Shapiro W | Shapiro p | Deltas Normais |")
        lines.append("|---|---|---|---|---|")
        
        for _, row in self.resultados_shapiro.iterrows():
            normal_str = "Sim" if row['normal'] else "Não"
            lines.append(f"| {row['familia']} | {row['n']} | {row['shapiro_w']:.4f} | {row['shapiro_p']:.4f} | {normal_str} |")
        lines.append("")
        
        # 2. Wilcoxon
        lines.append("## 3. Análise de Diferenças (Wilcoxon Signed-Rank)")
        lines.append("Teste não-paramétrico para comparar duas amostras pareadas.")
        lines.append("")
        header_wilcoxon = f"| Contexto | Média {self.rotulo1} | Média {self.rotulo2} | Diferença | P-value | Sig. (p<0.05) | Cohen's d | Tamanho do Efeito |"
        lines.append(header_wilcoxon)
        lines.append("|---|---|---|---|---|---|---|---|")
        
        for _, row in self.resultados_wilcoxon.iterrows():
            sig_str = "Sim" if row['significante'] else "Não"
            # Formatando p-value para notação científica se for muito pequeno
            p_val_str = f"{row['p_value']:.4f}" if row['p_value'] > 0.0001 else f"{row['p_value']:.2e}"
            
            lines.append(f"| {row['contexto']} | {row['media_rotulo1']:.4f} | {row['media_rotulo2']:.4f} | {row['diferenca']:.4f} | {p_val_str} | {sig_str} | {row['cohen_d']:.4f} | {row['tamanho_efeito']} |")
        lines.append("")
        
        # 3. Eficiência
        lines.append("## 4. Análise de Eficiência e Custos")
        lines.append("Comparação de custo-benefício (Valor / Custo).")
        lines.append("")
        header_efi = f"| Família | Custo Médio {self.rotulo1} | Custo Médio {self.rotulo2} | Δ Custo (%) | Δ Valor (%) | Efic. {self.rotulo1} | Efic. {self.rotulo2} | Δ Efic. (%) |"
        lines.append(header_efi)
        lines.append("|---|---|---|---|---|---|---|---|")
        
        for _, row in self.resultados_eficiencia.iterrows():
            lines.append(f"| {row['familia']} | {row['custo_medio1']:.4f} | {row['custo_medio2']:.4f} | {row['delta_custo_pct']:.4f}% | {row['delta_valor_pct']:.4f}% | {row['eficiencia1']:.4f} | {row['eficiencia2']:.4f} | {row['delta_eficiencia_pct']:.4f}% |")
            
        self.markdown_content = "\n".join(lines)
        
    def salvar_relatorio(self):
        """Salva o relatório Markdown no arquivo especificado."""
        if not hasattr(self, 'markdown_content'):
            self.processar_analise()
            
        with open(self.arquivo_saida, 'w', encoding='utf-8') as f:
            f.write(self.markdown_content)
        print(f"Relatório salvo em: {self.arquivo_saida}")

if __name__ == "__main__":
    # Teste rápido se executado diretamente
    print("Executando teste básico da classe...")
    
    # Dados dummy
    data = {
        'valor1': np.random.normal(0.5, 0.1, 100),
        'valor2': np.random.normal(0.6, 0.1, 100), # Ligeiramente melhor
        'custo1': [0.001] * 100,
        'custo2': [0.002] * 100, # Dobro do custo
        'familia': ['ModeloA'] * 50 + ['ModeloB'] * 50
    }
    df = pd.DataFrame(data)
    
    analise = AnaliseEstatistica(df, config={
        'rotulo1': 'Base',
        'rotulo2': 'Agente',
        'arquivo_saida': 'teste_analise_estatistica.md'
    })
    
    analise.processar_analise()
    analise.salvar_relatorio()
