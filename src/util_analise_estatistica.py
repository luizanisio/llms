# -*- coding: utf-8 -*-

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Análise estatística para comparação de K protocolos sobre N documentos pareados.
Testes: Friedman (omnibus), Nemenyi (post-hoc / CD Diagram), 
        Wilcoxon signed-rank + Holm-Bonferroni (par a par), Shapiro-Wilk (normalidade).
Gera relatórios Markdown e Critical Difference Diagrams (PNG).
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

# Lazy imports para scikit_posthocs e matplotlib (evita overhead se não usado)
_sp = None
_plt = None

def _get_sp():
    global _sp
    if _sp is None:
        import scikit_posthocs as sp
        _sp = sp
    return _sp

def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


# ============================================================================
# Textos i18n
# ============================================================================

_TEXTOS = {
    'pt': {
        'titulo_principal': 'Análise Estatística',
        'sec_ranking': 'Ranking de Desempenho',
        'sec_friedman': 'Teste Omnibus (Friedman)',
        'sec_shapiro': 'Normalidade dos Deltas (Shapiro-Wilk)',
        'sec_wilcoxon': 'Comparações Par a Par (Wilcoxon + Holm-Bonferroni)',
        'sec_cd': 'Diagrama de Diferença Crítica',
        'sec_grupos': 'Grupos de Equivalência',
        # Colunas
        'col_pos': 'Pos', 'col_protocolo': 'Protocolo', 'col_media': 'Média',
        'col_desvio': 'σ', 'col_mediana': 'Mediana', 'col_n': 'n',
        'col_grupo': 'Grupo', 'col_sig': 'Sig.', 'col_efeito': 'Efeito',
        'col_par': 'Par', 'col_normal': 'Normal',
        'col_prot1': 'Protocolo 1', 'col_prot2': 'Protocolo 2',
        'col_p_bruto': 'p (bruto)', 'col_p_corrigido': 'p (corrigido)',
        # Legendas
        'leg_friedman': '**K**: protocolos comparados · **N**: documentos pareados · **χ²**: estatística qui-quadrado de Friedman · **df**: graus de liberdade (K−1)',
        'leg_shapiro': '**n**: amostras pareadas · **W**: estatística Shapiro-Wilk (mais próximo de 1 = mais normal) · **Normal**: p > 0,05',
        'leg_wilcoxon': '**Δ**: diferença média (Protocolo 2 − Protocolo 1) · **p (corrigido)**: ajustado por Holm-Bonferroni · **Cohen\'s d**: magnitude do efeito — Insignificante (<0,10), Pequeno (0,10–0,30), Médio (0,30–0,50), Grande (≥0,50)',
        # Mensagens
        'msg_friedman_sig': 'O teste de Friedman indica diferença estatisticamente significativa entre os protocolos (p < 0,05). Procedendo com testes post-hoc.',
        'msg_friedman_ns': 'O teste de Friedman **não** encontrou diferença significativa entre os protocolos (p ≥ 0,05). Os testes post-hoc são apresentados a título informativo.',
        'msg_grupos': 'Protocolos no mesmo grupo não diferem significativamente entre si (Nemenyi, α=0,05).',
        'msg_n_insuficiente': 'Análise não realizada: número de amostras pareadas ({n}) inferior ao mínimo requerido ({min_n}).',
        'msg_k_insuficiente': 'Análise Friedman não aplicável: requer pelo menos 3 protocolos (K={k}).',
        # Efeitos Cohen
        'efeito_insignificante': 'Insignificante', 'efeito_pequeno': 'Pequeno',
        'efeito_medio': 'Médio', 'efeito_grande': 'Grande',
        'sim': 'Sim', 'nao': 'Não',
    },
    'en': {
        'titulo_principal': 'Statistical Analysis',
        'sec_ranking': 'Performance Ranking',
        'sec_friedman': 'Omnibus Test (Friedman)',
        'sec_shapiro': 'Normality of Deltas (Shapiro-Wilk)',
        'sec_wilcoxon': 'Pairwise Comparisons (Wilcoxon + Holm-Bonferroni)',
        'sec_cd': 'Critical Difference Diagram',
        'sec_grupos': 'Equivalence Groups',
        # Columns
        'col_pos': 'Rank', 'col_protocolo': 'Protocol', 'col_media': 'Mean',
        'col_desvio': 'σ', 'col_mediana': 'Median', 'col_n': 'n',
        'col_grupo': 'Group', 'col_sig': 'Sig.', 'col_efeito': 'Effect',
        'col_par': 'Pair', 'col_normal': 'Normal',
        'col_prot1': 'Protocol 1', 'col_prot2': 'Protocol 2',
        'col_p_bruto': 'p (raw)', 'col_p_corrigido': 'p (corrected)',
        # Legends
        'leg_friedman': '**K**: number of protocols compared · **N**: number of paired documents · **χ²**: Friedman chi-squared statistic · **df**: degrees of freedom (K−1)',
        'leg_shapiro': '**n**: paired samples · **W**: Shapiro-Wilk statistic (closer to 1 = more normal) · **Normal**: p > 0.05',
        'leg_wilcoxon': '**Δ**: mean difference (Protocol 2 − Protocol 1) · **p (corrected)**: Holm-Bonferroni adjusted · **Cohen\'s d**: effect magnitude — Negligible (<0.10), Small (0.10–0.30), Medium (0.30–0.50), Large (≥0.50)',
        # Messages
        'msg_friedman_sig': 'The Friedman test indicates a statistically significant difference among protocols (p < 0.05). Proceeding with post-hoc tests.',
        'msg_friedman_ns': 'The Friedman test found **no** significant difference among protocols (p ≥ 0.05). Post-hoc tests are presented for informational purposes.',
        'msg_grupos': 'Protocols in the same group do not differ significantly from each other (Nemenyi, α=0.05).',
        'msg_n_insuficiente': 'Analysis not performed: number of paired samples ({n}) below the required minimum ({min_n}).',
        'msg_k_insuficiente': 'Friedman analysis not applicable: requires at least 3 protocols (K={k}).',
        # Cohen effects
        'efeito_insignificante': 'Negligible', 'efeito_pequeno': 'Small',
        'efeito_medio': 'Medium', 'efeito_grande': 'Large',
        'sim': 'Yes', 'nao': 'No',
    }
}


class AnaliseEstatistica:
    """Análise estatística para comparação de K protocolos sobre N documentos pareados.
    
    Aceita DataFrame largo N×K onde cada coluna é um protocolo e cada linha é um documento.
    Executa: Ranking, Friedman, Shapiro-Wilk, Nemenyi, Wilcoxon corrigido, Cohen's d.
    Gera: relatório Markdown (.md) e Critical Difference Diagram (.png).
    """

    def __init__(self, df_scores, config=None):
        """
        Args:
            df_scores: DataFrame largo N×K onde:
                - Cada coluna é um protocolo (nome = rótulo ou alias)
                - Cada linha é um documento
                - Valores = scores (float F1 ou int Likert)
                - NaN = documento sem score para aquele protocolo
            config: dict com:
                - 'metrica_nome': str — nome completo da métrica (ex: '(global)_bertscore_F1')
                - 'campo': str — nome do campo (ex: '(global)')
                - 'tecnica': str — nome da técnica (ex: 'BERTScore')
                - 'arquivo_md': str — caminho do relatório .md
                - 'arquivo_cd_png': str — caminho do CD diagram .png
                - 'lang': str — idioma ('pt' ou 'en', default 'en')
                - 'min_amostras': int — mínimo de amostras pareadas (default 20)
                - 'formato_grupo': str — formato do grupo (default 'G-{:02d}')
                - 'alpha': float — nível de significância (default 0.05)
        """
        self.config = config or {}
        self.lang = self.config.get('lang', 'en')
        self.t = _TEXTOS.get(self.lang, _TEXTOS['en'])
        self.alpha = self.config.get('alpha', 0.05)
        self.min_amostras = self.config.get('min_amostras', 20)
        self.formato_grupo = self.config.get('formato_grupo', 'G-{:02d}')
        
        self.metrica_nome = self.config.get('metrica_nome', '')
        self.campo = self.config.get('campo', '')
        self.tecnica = self.config.get('tecnica', '')
        self.arquivo_md = self.config.get('arquivo_md', 'relatorio_estatistico.md')
        self.arquivo_cd_png = self.config.get('arquivo_cd_png', '')
        
        # Remove linhas com qualquer NaN (precisamos de dados pareados completos)
        self.df = df_scores.dropna().copy()
        self.protocolos = list(self.df.columns)
        self.K = len(self.protocolos)
        self.N = len(self.df)
        
        # Resultados (populados por processar())
        self.ranking = pd.DataFrame()
        self.friedman_resultado = {}
        self.shapiro_resultados = []
        self.wilcoxon_resultados = []
        self.nemenyi_pvalores = pd.DataFrame()
        self.grupos = {}  # {protocolo: 'G-01'}
        self.resumo = {}
        self.markdown_content = ''
        self._analise_realizada = False
    
    def processar(self):
        """Executa todas as análises e retorna resumo."""
        if self.N < self.min_amostras:
            self._gerar_relatorio_insuficiente()
            return self.resumo
        
        self._calcular_ranking()
        
        if self.K >= 3:
            self._calcular_friedman()
            self._calcular_nemenyi()
            self._calcular_grupos()
        elif self.K == 2:
            # Com apenas 2 protocolos, Friedman não se aplica
            self.friedman_resultado = {'aplicavel': False, 'motivo': 'K=2'}
        
        self._calcular_shapiro()
        self._calcular_wilcoxon_corrigido()
        self._calcular_effect_sizes()
        
        self._analise_realizada = True
        self._gerar_relatorio_md()
        self._gerar_cd_diagram()
        
        self.resumo = {
            'metrica': self.metrica_nome,
            'K': self.K,
            'N': self.N,
            'friedman_p': self.friedman_resultado.get('p_valor'),
            'friedman_sig': self.friedman_resultado.get('significante', False),
            'n_grupos': len(set(self.grupos.values())) if self.grupos else 0,
        }
        return self.resumo
    
    def salvar(self):
        """Salva o relatório .md e o CD diagram .png."""
        if self.arquivo_md and self.markdown_content:
            os.makedirs(os.path.dirname(self.arquivo_md) or '.', exist_ok=True)
            with open(self.arquivo_md, 'w', encoding='utf-8') as f:
                f.write(self.markdown_content)
            print(f"   📄 Relatório: {os.path.basename(self.arquivo_md)}")
    
    # ========================================================================
    # Análises
    # ========================================================================
    
    def _calcular_ranking(self):
        """Ranking dos protocolos por média decrescente."""
        dados = []
        for proto in self.protocolos:
            vals = self.df[proto]
            dados.append({
                'protocolo': proto,
                'media': vals.mean(),
                'std': vals.std(),
                'mediana': vals.median(),
                'n': len(vals),
            })
        df = pd.DataFrame(dados).sort_values('media', ascending=False).reset_index(drop=True)
        df['posicao'] = df.index + 1
        self.ranking = df
    
    def _calcular_friedman(self):
        """Teste omnibus de Friedman para K>=3 protocolos pareados."""
        if self.K < 3:
            self.friedman_resultado = {'aplicavel': False, 'motivo': f'K={self.K}'}
            return
        
        arrays = [self.df[p].values for p in self.protocolos]
        try:
            chi2, p_valor = stats.friedmanchisquare(*arrays)
            self.friedman_resultado = {
                'aplicavel': True,
                'K': self.K,
                'N': self.N,
                'chi2': chi2,
                'df': self.K - 1,
                'p_valor': p_valor,
                'significante': p_valor < self.alpha,
            }
        except Exception as e:
            self.friedman_resultado = {'aplicavel': False, 'motivo': str(e)}
    
    def _calcular_nemenyi(self):
        """Post-hoc Nemenyi-Friedman: matriz de p-valores K×K."""
        if self.K < 3:
            return
        sp = _get_sp()
        try:
            self.nemenyi_pvalores = sp.posthoc_nemenyi_friedman(self.df)
        except Exception as e:
            print(f"   ⚠️  Erro no Nemenyi: {e}")
            self.nemenyi_pvalores = pd.DataFrame()
    
    def _calcular_grupos(self):
        """Atribui grupos G-01, G-02, ... baseado em Nemenyi usando CLD."""
        if self.nemenyi_pvalores.empty:
            return
        sp = _get_sp()
        try:
            cld = sp.compact_letter_display(self.nemenyi_pvalores, alpha=self.alpha)
            # cld retorna Series com letras padded (ex: 'a  ', 'ab ', ' b ')
            # Espaços = não pertence ao grupo; letras = pertence
            # Extrai letras únicas (sem espaço)
            letras_unicas = sorted(set(
                c for v in cld.values for c in str(v) if c.strip()
            ))
            mapa_letras = {letra: self.formato_grupo.format(i + 1) 
                          for i, letra in enumerate(letras_unicas)}
            
            self.grupos = {}
            for proto, letras_raw in cld.items():
                letras = [c for c in str(letras_raw) if c.strip()]
                grupos_proto = ' '.join(mapa_letras[l] for l in sorted(letras))
                self.grupos[proto] = grupos_proto
        except Exception as e:
            print(f"   ⚠️  Erro ao calcular grupos (CLD): {e}")
            self.grupos = {}
    
    def _calcular_shapiro(self):
        """Shapiro-Wilk nos deltas de cada par (triângulo inferior)."""
        self.shapiro_resultados = []
        for i, (p1, p2) in enumerate(combinations(self.protocolos, 2)):
            deltas = (self.df[p2] - self.df[p1]).dropna()
            if len(deltas) >= 3:
                try:
                    w, p = stats.shapiro(deltas)
                    self.shapiro_resultados.append({
                        'proto1': p1, 'proto2': p2,
                        'n': len(deltas), 'W': w, 'p_valor': p,
                        'normal': p > self.alpha,
                    })
                except Exception:
                    pass
    
    def _calcular_wilcoxon_corrigido(self):
        """Wilcoxon signed-rank par a par com correção Holm-Bonferroni."""
        pares = list(combinations(self.protocolos, 2))
        resultados_brutos = []
        
        for p1, p2 in pares:
            v1, v2 = self.df[p1], self.df[p2]
            diff = v2 - v1
            
            try:
                _, p_bruto = stats.wilcoxon(v1, v2)
            except ValueError:
                p_bruto = 1.0
            
            resultados_brutos.append({
                'proto1': p1, 'proto2': p2,
                'n': len(v1),
                'media_p1': v1.mean(), 'media_p2': v2.mean(),
                'diferenca': diff.mean(),
                'p_bruto': p_bruto,
            })
        
        if not resultados_brutos:
            return
        
        # Correção Holm-Bonferroni
        p_brutos = [r['p_bruto'] for r in resultados_brutos]
        p_corrigidos = self._holm_bonferroni(p_brutos)
        
        for r, p_corr in zip(resultados_brutos, p_corrigidos):
            r['p_corrigido'] = p_corr
            r['significante'] = p_corr < self.alpha
        
        self.wilcoxon_resultados = resultados_brutos
    
    def _calcular_effect_sizes(self):
        """Cohen's d para cada par nos resultados de Wilcoxon."""
        for r in self.wilcoxon_resultados:
            p1, p2 = r['proto1'], r['proto2']
            diff = self.df[p2] - self.df[p1]
            std_diff = diff.std()
            
            if np.isclose(std_diff, 0):
                cohen_d = 0.0
            else:
                cohen_d = diff.mean() / std_diff
            
            abs_d = abs(cohen_d)
            if abs_d < 0.10:
                tamanho = self.t['efeito_insignificante']
            elif abs_d < 0.30:
                tamanho = self.t['efeito_pequeno']
            elif abs_d < 0.50:
                tamanho = self.t['efeito_medio']
            else:
                tamanho = self.t['efeito_grande']
            
            r['cohen_d'] = cohen_d
            r['tamanho_efeito'] = tamanho
    
    @staticmethod
    def _holm_bonferroni(p_valores):
        """Correção Holm-Bonferroni para múltiplas comparações."""
        m = len(p_valores)
        if m == 0:
            return []
        
        # Ordena por p-valor
        indices = list(range(m))
        indices.sort(key=lambda i: p_valores[i])
        
        p_corrigidos = [0.0] * m
        p_acumulado = 0.0
        
        for rank, idx in enumerate(indices):
            fator = m - rank  # Holm: multiplica por (m - rank)
            p_ajustado = p_valores[idx] * fator
            p_acumulado = max(p_acumulado, p_ajustado)  # Garante monotonicidade
            p_corrigidos[idx] = min(p_acumulado, 1.0)
        
        return p_corrigidos
    
    # ========================================================================
    # Geração de relatório Markdown
    # ========================================================================
    
    @staticmethod
    def _fmt_p(p):
        """Formata p-valor para exibição."""
        if p is None:
            return '—'
        if p < 0.0001:
            return f'{p:.2e}'
        return f'{p:.4f}'
    
    def _gerar_relatorio_insuficiente(self):
        """Gera relatório indicando amostras insuficientes."""
        L = []
        titulo = f'{self.t["titulo_principal"]}: {self.campo} — {self.tecnica}'
        L.append(f'# {titulo}')
        L.append('')
        msg = self.t['msg_n_insuficiente'].format(n=self.N, min_n=self.min_amostras)
        L.append(f'> [!WARNING]\n> {msg}')
        self.markdown_content = '\n'.join(L)
    
    def _gerar_relatorio_md(self):
        """Gera o relatório Markdown completo."""
        L = []
        t = self.t
        
        titulo = f'{t["titulo_principal"]}: {self.campo} — {self.tecnica}'
        L.append(f'# {titulo}')
        L.append('')
        
        # --- 1. Ranking ---
        L.append(f'## 1. {t["sec_ranking"]}')
        L.append('')
        L.append(f'| {t["col_pos"]} | {t["col_protocolo"]} | {t["col_media"]} | {t["col_desvio"]} | {t["col_mediana"]} | {t["col_n"]} | {t["col_grupo"]} |')
        L.append('|---|---|---|---|---|---|---|')
        for _, r in self.ranking.iterrows():
            grupo = self.grupos.get(r['protocolo'], '—')
            L.append(
                f'| {int(r["posicao"])} | {r["protocolo"]} '
                f'| {r["media"]:.4f} | {r["std"]:.4f} | {r["mediana"]:.4f} '
                f'| {int(r["n"])} | {grupo} |'
            )
        L.append('')
        
        # --- 2. Friedman ---
        L.append(f'## 2. {t["sec_friedman"]}')
        L.append('')
        
        fr = self.friedman_resultado
        if fr.get('aplicavel'):
            L.append('| K | N | χ² | df | p-value | Sig. |')
            L.append('|---|---|---|---|---|---|')
            L.append(
                f'| {fr["K"]} | {fr["N"]} | {fr["chi2"]:.2f} '
                f'| {fr["df"]} | {self._fmt_p(fr["p_valor"])} '
                f'| {t["sim"] if fr["significante"] else t["nao"]} |'
            )
            L.append('')
            L.append(f'> {t["leg_friedman"]}')
            L.append('')
            if fr['significante']:
                L.append(f'{t["msg_friedman_sig"]}')
            else:
                L.append(f'{t["msg_friedman_ns"]}')
        else:
            motivo = fr.get('motivo', '?')
            if 'K=' in str(motivo):
                L.append(t['msg_k_insuficiente'].format(k=self.K))
            else:
                L.append(f'> Friedman: {motivo}')
        L.append('')
        
        # --- 3. Shapiro-Wilk ---
        if self.shapiro_resultados:
            L.append(f'## 3. {t["sec_shapiro"]}')
            L.append('')
            L.append(f'| {t["col_par"]} | {t["col_n"]} | W | p-value | {t["col_normal"]} |')
            L.append('|---|---|---|---|---|')
            for r in self.shapiro_resultados:
                par = f'{r["proto1"]}↔{r["proto2"]}'
                L.append(
                    f'| {par} | {r["n"]} | {r["W"]:.4f} '
                    f'| {self._fmt_p(r["p_valor"])} '
                    f'| {t["sim"] if r["normal"] else t["nao"]} |'
                )
            L.append('')
            L.append(f'> {t["leg_shapiro"]}')
            L.append('')
        
        # --- 4. Wilcoxon ---
        if self.wilcoxon_resultados:
            L.append(f'## 4. {t["sec_wilcoxon"]}')
            L.append('')
            L.append(f'| {t["col_prot1"]} | {t["col_prot2"]} | Δ | {t["col_p_bruto"]} | {t["col_p_corrigido"]} | {t["col_sig"]} | Cohen\'s d | {t["col_efeito"]} |')
            L.append('|---|---|---|---|---|---|---|---|')
            
            # Ordena por p-valor corrigido
            resultados_ord = sorted(self.wilcoxon_resultados, key=lambda x: x.get('p_corrigido', 1.0))
            for r in resultados_ord:
                sinal = '+' if r['diferenca'] >= 0 else ''
                L.append(
                    f'| {r["proto1"]} | {r["proto2"]} '
                    f'| {sinal}{r["diferenca"]:.4f} '
                    f'| {self._fmt_p(r["p_bruto"])} '
                    f'| {self._fmt_p(r["p_corrigido"])} '
                    f'| {t["sim"] if r["significante"] else t["nao"]} '
                    f'| {r["cohen_d"]:.4f} | {r["tamanho_efeito"]} |'
                )
            L.append('')
            L.append(f'> {t["leg_wilcoxon"]}')
            L.append('')
        
        # --- 5. CD Diagram ---
        if self.arquivo_cd_png and os.path.exists(self.arquivo_cd_png):
            L.append(f'## 5. {t["sec_cd"]}')
            L.append('')
            L.append(f'![CD Diagram]({os.path.basename(self.arquivo_cd_png)})')
            L.append('')
        
        # --- 6. Grupos ---
        if self.grupos:
            L.append(f'## 6. {t["sec_grupos"]}')
            L.append('')
            L.append(t['msg_grupos'])
            L.append('')
            # Tabela resumo dos grupos
            L.append(f'| {t["col_protocolo"]} | {t["col_grupo"]} |')
            L.append('|---|---|')
            for _, r in self.ranking.iterrows():
                proto = r['protocolo']
                grupo = self.grupos.get(proto, '—')
                L.append(f'| {proto} | {grupo} |')
            L.append('')
        
        self.markdown_content = '\n'.join(L)
    
    # ========================================================================
    # CD Diagram
    # ========================================================================
    
    def _gerar_cd_diagram(self):
        """Gera Critical Difference Diagram como PNG."""
        if not self.arquivo_cd_png or self.nemenyi_pvalores.empty or self.K < 3:
            return
        
        sp = _get_sp()
        plt = _get_plt()
        
        try:
            # Calcula ranks médios
            ranks = self.df.rank(axis=1, ascending=False).mean()
            
            fig, ax = plt.subplots(figsize=(max(10, self.K * 0.8), 4))
            sp.critical_difference_diagram(
                ranks, self.nemenyi_pvalores,
                ax=ax, alpha=self.alpha
            )
            
            os.makedirs(os.path.dirname(self.arquivo_cd_png) or '.', exist_ok=True)
            fig.savefig(self.arquivo_cd_png, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            print(f"   📊 CD Diagram: {os.path.basename(self.arquivo_cd_png)}")
        except Exception as e:
            print(f"   ⚠️  Erro ao gerar CD Diagram: {e}")
            plt.close('all')


# ============================================================================
# Função de conveniência para o pipeline comparar_extracoes.py
# ============================================================================

def executar_analise_estatistica(analisador, dados_analise, config, pasta_saida, lang='en'):
    """
    Função principal chamada por comparar_extracoes.py.
    Descobre alvos (campo×métrica) e executa análise estatística para cada um.
    
    Args:
        analisador: instância de JsonAnaliseDataFrame com _resultados populado
        dados_analise: instância de JsonAnaliseDados com avaliação LLM
        config: dict do YAML completo
        pasta_saida: pasta raiz de saída
        lang: idioma ('pt' ou 'en')
    
    Returns:
        list[dict]: resumos de cada análise executada
    """
    import glob
    
    if analisador is None or not hasattr(analisador, '_resultados') or analisador._resultados is None:
        print("   ⚠️  Analisador sem resultados. Pulando análise estatística.")
        return []
    
    df_resultados = analisador._resultados
    
    # Pasta de estatísticas
    pasta_estat = os.path.join(pasta_saida, 'estatisticas')
    os.makedirs(pasta_estat, exist_ok=True)
    
    # Limpeza (mesmo padrão da pasta graficos/)
    antigos = glob.glob(os.path.join(pasta_estat, '*.md')) + \
              glob.glob(os.path.join(pasta_estat, '*.png'))
    if antigos:
        erros = 0
        for arq in antigos:
            try:
                os.remove(arq)
            except Exception:
                erros += 1
        total = len(antigos) - erros
        if total > 0:
            print(f"   🧹 {total} arquivos antigos removidos da pasta estatisticas/")
    
    # Descobre alvos de campos_estatisticas
    conf_comp = config.get('configuracao_comparacao', {})
    campos_est = conf_comp.get('campos_estatisticas', {})
    campos_virtuais = config.get('campos_virtuais', {})
    
    if campos_est:
        alvos_campos = campos_est.get('campos', [])
        alvos_metricas = campos_est.get('metricas', [])
    else:
        # Default: (global) para métricas que possuem (global)
        alvos_campos = ['(global)']
        # Detecta quais métricas têm (global) configurado
        campos_config = conf_comp.get('campos', {})
        alvos_metricas = []
        for metrica, campos_lista in campos_config.items():
            if isinstance(campos_lista, list) and '(global)' in campos_lista:
                alvos_metricas.append(metrica)
        if not alvos_metricas:
            print("   ⚠️  Nenhuma métrica com (global) encontrada. Sem análise estatística.")
            return []
    
    # Mapa de aliases: {rotulo: alias}
    mapa_aliases = {}
    modelo_base = config.get('modelo_base', {})
    alias_base = modelo_base.get('alias', modelo_base.get('rotulo', ''))
    if alias_base:
        mapa_aliases[modelo_base.get('rotulo', '')] = alias_base
    
    for m in config.get('modelos_comparacao', []):
        if not m.get('ativo', True):
            continue
        rotulo = m.get('rotulo', '')
        alias = m.get('alias', rotulo)
        mapa_aliases[rotulo] = alias
    
    # Mapa de nomes de métrica para sufixo nas colunas do DataFrame
    mapa_metrica_sufixo = {
        'bertscore': 'bertscore',
        'rouge_l': 'rouge',  # ROUGE-L usa 'rouge' no sufixo
        'rouge_1': 'rouge1',
        'rouge_2': 'rouge2',
        'levenshtein': 'levenshtein',
        'sbert': 'sbert',
        'sbert_pequeno': 'sbert_pequeno',
        'sbert_medio': 'sbert_medio',
        'sbert_grande': 'sbert_grande',
    }
    
    mapa_metrica_display = {
        'bertscore': 'BERTScore',
        'rouge_l': 'ROUGE-L',
        'rouge_1': 'ROUGE-1',
        'rouge_2': 'ROUGE-2',
        'levenshtein': 'Levenshtein',
        'sbert': 'SBERT',
        'sbert_pequeno': 'SBERT-Small',
        'sbert_medio': 'SBERT-Medium',
        'sbert_grande': 'SBERT-Large',
    }
    
    # Identifica protocolos (modelos) disponíveis
    rotulo_true = analisador.rotulos[1] if len(analisador.rotulos) > 1 else ''
    protocolos = list(analisador.rotulos[2:]) if len(analisador.rotulos) > 2 else []
    
    if not protocolos:
        print("   ⚠️  Nenhum protocolo encontrado para análise estatística.")
        return []
    
    resumos = []
    
    print(f"\n📊 Análise Estatística — {len(alvos_campos)} campo(s) × {len(alvos_metricas)} métrica(s)")
    
    for campo in alvos_campos:
        for metrica in alvos_metricas:
            sufixo = mapa_metrica_sufixo.get(metrica, metrica)
            display = mapa_metrica_display.get(metrica, metrica)
            
            # Padrão de coluna: {protocolo}_{campo}_{sufixo}_F1
            # Tenta encontrar as colunas no DataFrame
            df_largo = pd.DataFrame()
            
            for proto in protocolos:
                # Tenta o padrão completo: {proto}_{campo}_{sufixo}_F1
                col_candidatas = [
                    f'{proto}_{campo}_{sufixo}_F1',
                ]
                
                col_encontrada = None
                for col in col_candidatas:
                    if col in df_resultados.columns:
                        col_encontrada = col
                        break
                
                if col_encontrada:
                    alias = mapa_aliases.get(proto, proto)
                    df_largo[alias] = df_resultados[col_encontrada]
            
            if df_largo.empty or len(df_largo.columns) < 2:
                continue
            
            # Remove linhas com NaN
            df_largo = df_largo.dropna()
            
            metrica_nome = f'{campo}_{sufixo}_F1'
            nome_base = f'estat_{campo}_{metrica}'.replace('(', '').replace(')', '')
            arquivo_md = os.path.join(pasta_estat, f'{nome_base}.md')
            arquivo_png = os.path.join(pasta_estat, f'{nome_base}_cd.png')
            
            print(f"   → {campo} × {display} ({len(df_largo)} docs, {len(df_largo.columns)} protocolos)")
            
            analise = AnaliseEstatistica(df_largo, config={
                'metrica_nome': metrica_nome,
                'campo': campo,
                'tecnica': display,
                'arquivo_md': arquivo_md,
                'arquivo_cd_png': arquivo_png,
                'lang': lang,
                'min_amostras': 20,
            })
            
            resumo = analise.processar()
            analise.salvar()
            resumos.append(resumo)
    
    # --- LLM-as-a-Judge (seção separada, se disponível) ---
    if dados_analise and hasattr(dados_analise, 'avaliacao_llm') and dados_analise.avaliacao_llm:
        _processar_llm_estatisticas(dados_analise, config, protocolos, mapa_aliases, pasta_estat, lang, resumos)
    
    if resumos:
        sig_count = sum(1 for r in resumos if r.get('friedman_sig'))
        print(f"\n   ✅ {len(resumos)} análise(s) concluída(s) ({sig_count} com Friedman significativo)")
    else:
        print("   ⚠️  Nenhuma análise estatística gerada (combinações campo×métrica não encontradas nos dados).")
    
    return resumos


def _processar_llm_estatisticas(dados_analise, config, protocolos, mapa_aliases, pasta_estat, lang, resumos):
    """Processa análise estatística para LLM-as-a-Judge (separada das métricas de similaridade)."""
    pk = dados_analise.config.nome_campo_id
    
    # Tenta extrair F1 e nota por protocolo
    df_f1 = pd.DataFrame()
    df_nota = pd.DataFrame()
    
    for proto in protocolos:
        f1_vals = {}
        nota_vals = {}
        
        for item in dados_analise.dados_completos:
            id_doc = item.get(pk)
            if not id_doc:
                continue
            evals = dados_analise.get_avaliacao(str(id_doc))
            if not evals:
                continue
            
            f1 = evals.get(f'{proto}_F1')
            nota = evals.get(f'{proto}_nota')
            
            if f1 is not None:
                f1_vals[id_doc] = f1
            if nota is not None:
                nota_vals[id_doc] = nota
        
        alias = mapa_aliases.get(proto, proto)
        if f1_vals:
            df_f1[alias] = pd.Series(f1_vals)
        if nota_vals:
            df_nota[alias] = pd.Series(nota_vals)
    
    # Processa F1 LLM
    if len(df_f1.columns) >= 2:
        df_f1 = df_f1.dropna()
        if len(df_f1) >= 20:
            print(f"   → LLM-as-a-Judge F1 ({len(df_f1)} docs, {len(df_f1.columns)} protocolos)")
            analise = AnaliseEstatistica(df_f1, config={
                'metrica_nome': 'llm_F1',
                'campo': 'LLM-as-a-Judge',
                'tecnica': 'F1',
                'arquivo_md': os.path.join(pasta_estat, 'estat_llm_f1.md'),
                'arquivo_cd_png': os.path.join(pasta_estat, 'estat_llm_f1_cd.png'),
                'lang': lang,
                'min_amostras': 20,
            })
            resumos.append(analise.processar())
            analise.salvar()
    
    # Processa Nota/Likert LLM
    if len(df_nota.columns) >= 2:
        df_nota = df_nota.dropna()
        if len(df_nota) >= 20:
            print(f"   → LLM-as-a-Judge Nota ({len(df_nota)} docs, {len(df_nota.columns)} protocolos)")
            analise = AnaliseEstatistica(df_nota, config={
                'metrica_nome': 'llm_nota',
                'campo': 'LLM-as-a-Judge',
                'tecnica': 'Nota (Likert)',
                'arquivo_md': os.path.join(pasta_estat, 'estat_llm_nota.md'),
                'arquivo_cd_png': os.path.join(pasta_estat, 'estat_llm_nota_cd.png'),
                'lang': lang,
                'min_amostras': 20,
            })
            resumos.append(analise.processar())
            analise.salvar()


# ============================================================================
# Execução standalone para teste
# ============================================================================

if __name__ == "__main__":
    print("Executando teste com dados sintéticos...")
    np.random.seed(42)
    n = 100
    
    # Simula 5 protocolos com diferenças conhecidas
    df_test = pd.DataFrame({
        'A': np.random.normal(0.60, 0.12, n).clip(0, 1),
        'B': np.random.normal(0.65, 0.10, n).clip(0, 1),
        'D1': np.random.normal(0.80, 0.08, n).clip(0, 1),
        'D2': np.random.normal(0.82, 0.07, n).clip(0, 1),
        'D3': np.random.normal(0.85, 0.06, n).clip(0, 1),
    })
    
    pasta_teste = '/tmp/teste_estatistica'
    os.makedirs(pasta_teste, exist_ok=True)
    
    analise = AnaliseEstatistica(df_test, config={
        'metrica_nome': '(global)_bertscore_F1',
        'campo': '(global)',
        'tecnica': 'BERTScore',
        'arquivo_md': os.path.join(pasta_teste, 'estat_global_bertscore.md'),
        'arquivo_cd_png': os.path.join(pasta_teste, 'estat_global_bertscore_cd.png'),
        'lang': 'en',
    })
    
    resumo = analise.processar()
    analise.salvar()
    
    print(f"\nResumo: {resumo}")
    print(f"\nGrupos: {analise.grupos}")
    print(f"\nFriedman: χ²={analise.friedman_resultado.get('chi2', '?'):.2f}, "
          f"p={analise.friedman_resultado.get('p_valor', '?')}")
    
    if os.path.exists(os.path.join(pasta_teste, 'estat_global_bertscore.md')):
        with open(os.path.join(pasta_teste, 'estat_global_bertscore.md'), 'r') as f:
            print(f"\n{'='*60}")
            print(f.read())
    
    print("\n✅ Teste concluído!")
