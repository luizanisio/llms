# -*- coding: utf-8 -*-

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Análise estatística para comparação de K protocolos sobre N documentos pareados.
Testes: Friedman (omnibus), Nemenyi (post-hoc / CD Diagram), 
        Wilcoxon signed-rank + Holm-Bonferroni (par a par), Shapiro-Wilk (normalidade).
Gera relatórios Markdown e Critical Difference Diagrams (PNG).

Requisitos: scipy>=1.9 (wilcoxon zstatistic; fallback via norm.isf se ausente),
            scikit-posthocs>=0.9 (compact_letter_display, critical_difference_diagram).
"""

import os
import pandas as pd
import numpy as np
from scipy import stats, optimize
from itertools import combinations
from util_est_bayesiana import Comparacao

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
# Mapas de métrica (compartilhados com comparar_extracoes_baycomp.py)
# ============================================================================
# Traduzem o nome da métrica no YAML (chave de configuracao_comparacao.campos)
# para o sufixo usado nas colunas do DataFrame de resultados e para o rótulo
# exibido em relatórios/figuras. Ficam no nível do módulo — e não dentro de
# executar_analise_estatistica — porque a análise bayesiana precisa da MESMA
# tradução; duplicar os dicionários faria os dois relatórios divergirem ao
# incluir uma métrica nova.

#: nome no YAML → sufixo da coluna `{protocolo}_{campo}_{sufixo}_F1`
MAPA_METRICA_SUFIXO = {
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

#: nome no YAML → rótulo legível em relatórios e títulos de figura
MAPA_METRICA_DISPLAY = {
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
        'sec_nemenyi': 'Post-hoc de Nemenyi / Distância Crítica',
        'sec_cd': 'Diagrama de Diferença Crítica',
        'sec_grupos': 'Grupos de Equivalência',
        # Colunas — ranking
        'col_pos': 'Pos', 'col_protocolo': 'Protocolo', 'col_media': 'Média',
        'col_desvio': 'σ', 'col_mediana': 'Mediana', 'col_n': 'n',
        'col_grupo': 'Grupo', 'col_sig': 'Sig.', 'col_efeito': 'Efeito',
        'col_par': 'Par', 'col_normal': 'Normal',
        'col_q1': 'Q1', 'col_q3': 'Q3',
        'col_rank_medio': 'Rank Médio', 'col_pct_perfeito': '% Perfeito',
        'col_skewness': 'Assimetria',
        # Colunas — wilcoxon
        'col_prot1': 'Protocolo 1', 'col_prot2': 'Protocolo 2',
        'col_p_bruto': 'p (bruto)', 'col_p_corrigido': 'p (corrigido)',
        'col_z': '\\|z\\|', 'col_n_prime': "n'", 'col_r': 'r',
        'col_efeito_r': 'Efeito (r)', 'col_efeito_d': 'Efeito (d)',
        'col_delta_med': 'Δ med.', 'col_pct_empates': '% emp.',
        'col_mesmo_grupo': 'Mesmo Grupo?',
        # Legendas
        'leg_friedman': '**K**: protocolos comparados · **N**: documentos pareados · **χ²**: estatística qui-quadrado de Friedman · **df**: graus de liberdade (K−1) · **W**: W de Kendall (concordância entre blocos)',
        'leg_shapiro': '**n**: amostras pareadas · **W**: estatística Shapiro-Wilk (mais próximo de 1 = mais normal) · **Normal**: p > 0,05',
        'leg_wilcoxon': "**Δ**: diferença média (Protocolo 2 − Protocolo 1) · **Δ med.**: mediana das diferenças pareadas · **|z|**: magnitude da estatística z do Wilcoxon (o teste bilateral não define direção — a direção é dada por Δ) · **n'**: pares com diferença não-nula · **r**: tamanho de efeito r = |z|/√n' (Tomczak & Tomczak, 2014) — Insignificante (<0,10), Pequeno (0,10–0,30), Médio (0,30–0,50), Grande (≥0,50) · **% emp.**: percentual de empates (pares com Δ=0) · **Cohen's d**: tamanho de efeito paramétrico (secundário) — Insignificante (<0,20), Pequeno (0,20–0,50), Médio (0,50–0,80), Grande (≥0,80) (Cohen, 1988) · **p (corrigido)**: ajustado por Holm-Bonferroni",
        'leg_ranking': '**Rank Médio**: rank médio do Friedman (1 = melhor) · **Q1/Q3**: percentis 25 e 75 · **% Perfeito**: proporção de scores = 1,000 · **Assimetria**: skewness (< 0 = cauda à esquerda)',
        # Mensagens
        'msg_friedman_sig': 'O teste de Friedman indica diferença estatisticamente significativa entre os protocolos (p < 0,05). Procedendo com testes post-hoc.',
        'msg_friedman_ns': 'O teste de Friedman **não** encontrou diferença significativa entre os protocolos (p ≥ 0,05). Os testes post-hoc são apresentados a título informativo.',
        'msg_grupos': 'Protocolos no mesmo grupo não diferem significativamente entre si (Nemenyi, α=0,05).',
        'msg_grupos_nota': 'Grupos definidos pelo critério de **Nemenyi** (α = 0,05); contrastes de Wilcoxon podem detectar diferenças dentro de um mesmo grupo — nesses casos, interpretar pelo tamanho de efeito.',
        'msg_holm_nota': 'Correção de Holm aplicada sobre m = {m} comparações.',
        'msg_shapiro_n_grande': 'Com n = {n}, o Shapiro-Wilk detecta desvios mínimos de normalidade; a decisão não-paramétrica apoia-se também na forma das distribuições (efeito de teto, assimetria à esquerda).',
        'msg_n_insuficiente': 'Análise não realizada: número de amostras pareadas ({n}) inferior ao mínimo requerido ({min_n}).',
        'msg_k_insuficiente': 'Análise Friedman não aplicável: requer pelo menos 3 protocolos (K={k}).',
        'msg_nemenyi_cd': 'Distância Crítica (CD) = {cd:.4f} (q_α = {q_alpha:.4f}, k = {k}, N = {n}). Pares com diferença de ranks ≤ CD são considerados equivalentes.',
        'msg_metadados': 'Gerado em {data} · scipy {scipy_v} · scikit-posthocs {sp_v} · numpy {np_v} · pandas {pd_v} · N = {n} documentos · K = {k} protocolos',
        'msg_pareamento': 'Pareamento: {n_bruto} documentos brutos → {n_descartado} descartados (descarte global pareado: falha em ≥1 protocolo) → **N = {n} pareados**. Faltantes por protocolo: {detalhe}.',
        'msg_pareamento_sem_descarte': 'Pareamento: {n_bruto} documentos, nenhum descartado (todos os protocolos com scores completos).',
        # Efeitos
        'efeito_insignificante': 'Insignificante', 'efeito_pequeno': 'Pequeno',
        'efeito_medio': 'Médio', 'efeito_grande': 'Grande',
        'sim': 'Sim', 'nao': 'Não',
        'nota_rope_calibracao': '📌 **Nota de calibração da ROPE:** o par mais exigente desta tabela é {par}, com |Δ| = {delta} e desvio-padrão da posterior = {sd}; a ROPE sugerida é **{rope}** — a menor margem sob a qual TODOS os pares desta tabela seriam declarados `equivalente` ao limiar de {limiar}. O maior |Δ| isolado ({max_delta}) **não** serve como ROPE: fixá-la no Δ observado centra a posterior na borda do intervalo e trava a probabilidade de equivalência em ~0,50. Detalhes e bloco YAML em `00_rope_sugerido.md`.',
    },
    'en': {
        'titulo_principal': 'Statistical Analysis',
        'sec_ranking': 'Performance Ranking',
        'sec_friedman': 'Omnibus Test (Friedman)',
        'sec_shapiro': 'Normality of Deltas (Shapiro-Wilk)',
        'sec_wilcoxon': 'Pairwise Comparisons (Wilcoxon + Holm-Bonferroni)',
        'sec_nemenyi': 'Nemenyi Post-hoc / Critical Difference',
        'sec_cd': 'Critical Difference Diagram',
        'sec_grupos': 'Equivalence Groups',
        # Columns — ranking
        'col_pos': 'Rank', 'col_protocolo': 'Protocol', 'col_media': 'Mean',
        'col_desvio': 'σ', 'col_mediana': 'Median', 'col_n': 'n',
        'col_grupo': 'Group', 'col_sig': 'Sig.', 'col_efeito': 'Effect',
        'col_par': 'Pair', 'col_normal': 'Normal',
        'col_q1': 'Q1', 'col_q3': 'Q3',
        'col_rank_medio': 'Mean Rank', 'col_pct_perfeito': '% Perfect',
        'col_skewness': 'Skewness',
        # Columns — wilcoxon
        'col_prot1': 'Protocol 1', 'col_prot2': 'Protocol 2',
        'col_p_bruto': 'p (raw)', 'col_p_corrigido': 'p (corrected)',
        'col_z': '\\|z\\|', 'col_n_prime': "n'", 'col_r': 'r',
        'col_efeito_r': 'Effect (r)', 'col_efeito_d': 'Effect (d)',
        'col_delta_med': 'Δ med.', 'col_pct_empates': '% ties',
        'col_mesmo_grupo': 'Same Group?',
        # Legends
        'leg_friedman': '**K**: number of protocols compared · **N**: number of paired documents · **χ²**: Friedman chi-squared statistic · **df**: degrees of freedom (K−1) · **W**: Kendall\'s W (block concordance)',
        'leg_shapiro': '**n**: paired samples · **W**: Shapiro-Wilk statistic (closer to 1 = more normal) · **Normal**: p > 0.05',
        'leg_wilcoxon': "**Δ**: mean difference (Protocol 2 − Protocol 1) · **Δ med.**: median of paired differences · **|z|**: magnitude of the Wilcoxon z-statistic (the two-sided test carries no direction — direction is given by Δ) · **n'**: pairs with non-zero difference · **r**: effect size r = |z|/√n' (Tomczak & Tomczak, 2014) — Negligible (<0.10), Small (0.10–0.30), Medium (0.30–0.50), Large (≥0.50) · **% ties**: percentage of tied pairs (Δ=0) · **Cohen's d**: parametric effect size (secondary) — Negligible (<0.20), Small (0.20–0.50), Medium (0.50–0.80), Large (≥0.80) (Cohen, 1988) · **p (corrected)**: Holm-Bonferroni adjusted",
        'leg_ranking': '**Mean Rank**: Friedman mean rank (1 = best) · **Q1/Q3**: 25th and 75th percentiles · **% Perfect**: proportion of scores = 1.000 · **Skewness**: skewness (< 0 = left tail)',
        # Messages
        'msg_friedman_sig': 'The Friedman test indicates a statistically significant difference among protocols (p < 0.05). Proceeding with post-hoc tests.',
        'msg_friedman_ns': 'The Friedman test found **no** significant difference among protocols (p ≥ 0.05). Post-hoc tests are presented for informational purposes.',
        'msg_grupos': 'Protocols in the same group do not differ significantly from each other (Nemenyi, α=0.05).',
        'msg_grupos_nota': 'Groups defined by the **Nemenyi** criterion (α = 0.05); Wilcoxon contrasts may detect differences within the same group — in such cases, interpret by effect size.',
        'msg_holm_nota': 'Holm correction applied over m = {m} comparisons.',
        'msg_shapiro_n_grande': 'With n = {n}, the Shapiro-Wilk test detects minimal departures from normality; the non-parametric decision also relies on the shape of the distributions (ceiling effect, left skewness).',
        'msg_n_insuficiente': 'Analysis not performed: number of paired samples ({n}) below the required minimum ({min_n}).',
        'msg_k_insuficiente': 'Friedman analysis not applicable: requires at least 3 protocols (K={k}).',
        'msg_nemenyi_cd': 'Critical Difference (CD) = {cd:.4f} (q_α = {q_alpha:.4f}, k = {k}, N = {n}). Pairs with rank difference ≤ CD are considered equivalent.',
        'msg_metadados': 'Generated on {data} · scipy {scipy_v} · scikit-posthocs {sp_v} · numpy {np_v} · pandas {pd_v} · N = {n} docs · K = {k} protocols',
        'msg_pareamento': 'Pairing: {n_bruto} raw documents → {n_descartado} discarded (global paired discard: failure in ≥1 protocol) → **N = {n} paired**. Missing per protocol: {detalhe}.',
        'msg_pareamento_sem_descarte': 'Pairing: {n_bruto} documents, none discarded (all protocols with complete scores).',
        # Cohen effects
        'efeito_insignificante': 'Negligible', 'efeito_pequeno': 'Small',
        'efeito_medio': 'Medium', 'efeito_grande': 'Large',
        'sim': 'Yes', 'nao': 'No',
        'nota_rope_calibracao': '📌 **ROPE calibration note:** the most demanding pair in this table is {par}, with |Δ| = {delta} and posterior standard deviation = {sd}; the suggested ROPE is **{rope}** — the smallest margin under which EVERY pair in this table would be classified `equivalent` at the {limiar} threshold. The largest |Δ| alone ({max_delta}) is **not** a usable ROPE: pinning it to the observed Δ centres the posterior on the interval boundary and locks the equivalence probability at ~0.50. Details and YAML block in `00_rope_sugerido.md`.',
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
        # Limiar ÚNICO de decisão da camada bayesiana (`estatistica.limiar`).
        # Entra aqui porque a ROPE sugerida é derivada dele — ver
        # `_rope_para_equivalencia`.
        self.limiar_bayes = float(self.config.get('limiar_bayes', 0.95) or 0.95)
        # Bateria de calibração (réplicas do MESMO protocolo)? Só nesse caso a
        # ROPE sugerida tem sentido, e só nesse caso a nota é emitida.
        self.calibracao_rope = bool(self.config.get('calibracao_rope', False))
        self.min_amostras = self.config.get('min_amostras', 20)
        self.formato_grupo = self.config.get('formato_grupo', 'G-{:02d}')
        
        self.metrica_nome = self.config.get('metrica_nome', '')
        self.campo = self.config.get('campo', '')
        self.tecnica = self.config.get('tecnica', '')
        self.arquivo_md = self.config.get('arquivo_md', 'relatorio_estatistico.md')
        self.arquivo_cd_png = self.config.get('arquivo_cd_png', '')
        
        # Rastreia descarte ANTES do dropna (descarte global pareado):
        # documentos brutos, NaN por protocolo e N final pareado.
        self.n_bruto = len(df_scores)
        self.descartes_por_protocolo = {
            col: int(df_scores[col].isna().sum()) for col in df_scores.columns
        }
        
        # Remove linhas com qualquer NaN (precisamos de dados pareados completos)
        self.df = df_scores.dropna().copy()
        self.protocolos = list(self.df.columns)
        self.K = len(self.protocolos)
        self.N = len(self.df)
        self.n_descartado = self.n_bruto - self.N
        
        # Resultados (populados por processar())
        self.ranking = pd.DataFrame()
        self.friedman_resultado = {}
        self.shapiro_resultados = []
        self.wilcoxon_resultados = []
        self.nemenyi_pvalores = pd.DataFrame()
        self.grupos = {}  # {protocolo: 'G-01' ou 'G-01 G-02' se sobreposto}
        self.grupos_sets = {}  # {protocolo: {'G-01', ...}}
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
        
        self.max_delta = 0.0
        self.rope_calibrado = 0.0
        self.rope_detalhe = {}
        if self.wilcoxon_resultados:
            self.max_delta = max(abs(r['diferenca']) for r in self.wilcoxon_resultados)
            # A ROPE sugerida é a do par MAIS EXIGENTE: a margem que basta para
            # o par pior colocado já basta para todos os outros.
            pior = max(self.wilcoxon_resultados,
                       key=lambda r: self._rope_para_equivalencia(r, arredondar=False))
            self.rope_calibrado = self._rope_para_equivalencia(pior)
            self.rope_detalhe = {
                'rope': self.rope_calibrado,
                'delta': abs(float(pior['diferenca'])),
                'sd': float(np.sqrt(max(float(pior.get('var_posterior', 0.0)), 0.0))),
                'par': f"{pior['proto1']} × {pior['proto2']}",
                'n': int(pior.get('n', self.N)),
                'limiar': self.limiar_bayes,
            }

        self._analise_realizada = True
        self._gerar_relatorio_md()
        self._gerar_cd_diagram()
        
        self.resumo = {
            'metrica': self.metrica_nome,
            'campo': self.campo,
            'is_llm': 'llm_' in self.metrica_nome,
            'K': self.K,
            'N': self.N,
            'friedman_p': self.friedman_resultado.get('p_valor'),
            'friedman_sig': self.friedman_resultado.get('significante', False),
            # Conta rótulos distintos de grupo (overlap-safe): um protocolo em
            # 'G-01 G-02' contribui com 2 rótulos, não com 1 grupo novo.
            'n_grupos': len(set(
                g for v in self.grupos.values() for g in str(v).split()
            )) if self.grupos else 0,
            'max_delta': self.max_delta,
            'rope_calibrado': self.rope_calibrado,
            'rope_detalhe': self.rope_detalhe,
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
        """Ranking dos protocolos por média decrescente com descritivas não-paramétricas."""
        # Calcula ranks médios do Friedman (mesmos valores usados no CD diagram)
        ranks_friedman = self.df.rank(axis=1, ascending=False).mean()
        
        dados = []
        for proto in self.protocolos:
            vals = self.df[proto]
            dados.append({
                'protocolo': proto,
                'media': vals.mean(),
                'std': vals.std(),
                'mediana': vals.median(),
                'q1': vals.quantile(0.25),
                'q3': vals.quantile(0.75),
                'n': len(vals),
                'rank_medio': ranks_friedman[proto],
                'pct_perfeito': float((vals == 1.0).mean()),
                # Guard: coluna constante pode gerar NaN em versões antigas do pandas
                'skewness': float(vals.skew()) if np.isfinite(vals.skew()) else 0.0,
            })
        df = pd.DataFrame(dados).sort_values('media', ascending=False).reset_index(drop=True)
        df['posicao'] = df.index + 1
        self.ranking = df
    
    def _calcular_friedman(self):
        """Teste omnibus de Friedman para K>=3 protocolos pareados + Kendall's W."""
        if self.K < 3:
            self.friedman_resultado = {'aplicavel': False, 'motivo': f'K={self.K}'}
            return
        
        arrays = [self.df[p].values for p in self.protocolos]
        try:
            chi2, p_valor = stats.friedmanchisquare(*arrays)
            # Kendall's W: tamanho de efeito natural do Friedman
            # W = χ² / (N × (k - 1)), varia de 0 (sem concordância) a 1 (concordância total)
            kendall_w = chi2 / (self.N * (self.K - 1)) if self.N > 0 and self.K > 1 else 0.0
            self.friedman_resultado = {
                'aplicavel': True,
                'K': self.K,
                'N': self.N,
                'chi2': chi2,
                'df': self.K - 1,
                'p_valor': p_valor,
                'significante': p_valor < self.alpha,
                'kendall_w': kendall_w,
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
            # Espaços = não pertence ao grupo; letras = pertence.
            # Um protocolo pode pertencer a MAIS DE UM grupo (CLD sobreposto).
            letras_por_proto = {
                proto: [c for c in str(letras_raw) if c.strip()]
                for proto, letras_raw in cld.items()
            }
            letras_unicas = sorted(set(
                l for letras in letras_por_proto.values() for l in letras
            ))
            
            # Renumera grupos por desempenho: G-01 = grupo cujo melhor membro
            # tem o menor rank médio do Friedman (1 = melhor). Sem isso, a
            # numeração segue a ordem alfabética das letras do CLD (arbitrária).
            ranks_friedman = self.df.rank(axis=1, ascending=False).mean()
            rank_por_letra = {
                letra: min(
                    ranks_friedman[p] for p, ls in letras_por_proto.items() if letra in ls
                )
                for letra in letras_unicas
            }
            letras_ordenadas = sorted(letras_unicas, key=lambda l: rank_por_letra[l])
            mapa_letras = {letra: self.formato_grupo.format(i + 1)
                          for i, letra in enumerate(letras_ordenadas)}
            
            self.grupos = {}
            self.grupos_sets = {}  # {protocolo: set de rótulos de grupo} — para lógica de sobreposição
            for proto, letras in letras_por_proto.items():
                rotulos = sorted(mapa_letras[l] for l in letras)
                self.grupos[proto] = ' '.join(rotulos)
                self.grupos_sets[proto] = set(rotulos)
        except Exception as e:
            print(f"   ⚠️  Erro ao calcular grupos (CLD): {e}")
            self.grupos = {}
            self.grupos_sets = {}
    
    @staticmethod
    def _compartilham_grupo(grupos1, grupos2):
        """True se dois protocolos compartilham pelo menos um grupo de equivalência.
        
        Aceita strings ('G-01 G-02') ou sets. Necessário porque o CLD permite
        pertencimento múltiplo — comparação de strings inteiras falharia em
        casos de sobreposição (ex: 'G-02 G-03' vs 'G-03').
        """
        if not grupos1 or not grupos2:
            return False
        s1 = set(grupos1.split()) if isinstance(grupos1, str) else set(grupos1)
        s2 = set(grupos2.split()) if isinstance(grupos2, str) else set(grupos2)
        return bool(s1 & s2)
    
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
        """Wilcoxon signed-rank par a par com z-statistic, r=|z|/√n', Δ mediano e correção Holm-Bonferroni."""
        pares = list(combinations(self.protocolos, 2))
        resultados_brutos = []
        
        for p1, p2 in pares:
            v1, v2 = self.df[p1], self.df[p2]
            diff = v2 - v1
            n_total = len(v1)
            n_prime = int(np.sum(diff != 0))  # pares efetivos (sem empate)

            # Posterior do CorrelatedTTest com a correção de Nadeau-Bengio
            # (`var * (1/n + 1/(n-1))`, runs=1) — exatamente a variância que a
            # camada bayesiana usa. É recalculada aqui para que a ROPE sugerida
            # seja auto-consistente com o teste que a consome, sem depender de
            # a bayesiana ter rodado.
            var_post = (float(np.var(diff, ddof=1)) * (1.0 / n_total + 1.0 / (n_total - 1))
                        if n_total > 1 else 0.0)
            
            # NOTA: no teste bilateral com method="approx", o zstatistic do scipy
            # NÃO carrega direção (wilcoxon(v1,v2) e wilcoxon(v2,v1) retornam o
            # mesmo z). Reportamos |z|; a direção é dada por Δ e Δ mediano.
            z_stat = 0.0
            try:
                res = stats.wilcoxon(v1, v2, zero_method="wilcox", method="approx")
                p_bruto = res.pvalue
                z_raw = getattr(res, 'zstatistic', None)
                if z_raw is None or not np.isfinite(z_raw):
                    # Fallback (scipy antigo sem zstatistic): deriva |z| do p bilateral.
                    p_seguro = max(min(p_bruto, 1.0), 1e-300)
                    z_raw = stats.norm.isf(p_seguro / 2.0)
                z_stat = abs(float(z_raw))
            except (ValueError, AttributeError):
                p_bruto = 1.0
                z_stat = 0.0
            
            # Tamanho de efeito r = |z|/√n' (Tomczak & Tomczak, 2014)
            r_efeito = abs(z_stat) / np.sqrt(n_prime) if n_prime > 0 else 0.0
            pct_empates = 1.0 - (n_prime / n_total) if n_total > 0 else 0.0
            
            # Classificação do r (mesmas faixas de Cohen)
            abs_r = abs(r_efeito)
            if abs_r < 0.10:
                tamanho_r = self.t['efeito_insignificante']
            elif abs_r < 0.30:
                tamanho_r = self.t['efeito_pequeno']
            elif abs_r < 0.50:
                tamanho_r = self.t['efeito_medio']
            else:
                tamanho_r = self.t['efeito_grande']
            
            resultados_brutos.append({
                'proto1': p1, 'proto2': p2,
                'n': n_total,
                'n_prime': n_prime,
                'media_p1': v1.mean(), 'media_p2': v2.mean(),
                'diferenca': diff.mean(),
                'delta_mediano': float(np.median(diff)),
                'var_posterior': var_post,
                'gl_posterior': n_total - 1,
                'z_stat': z_stat,
                'r_efeito': r_efeito,
                'tamanho_efeito_r': tamanho_r,
                'pct_empates': pct_empates,
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
    
    def _rope_para_equivalencia(self, resultado, arredondar=True):
        """Menor ROPE que classifica este par como `equivalente` ao limiar.

        A camada bayesiana não compara o Δ pontual com a ROPE: ela integra a
        posterior da diferença média sobre ``[-R, +R]`` e exige que essa massa
        alcance o limiar. Fixar ``R = |Δ|`` — o que a versão anterior deste
        módulo sugeria — centra a posterior exatamente na borda do intervalo,
        deixando metade da massa de fora e travando P(equivalência) em ~0,50.
        É um teto estrutural: não melhora com mais dados, porque a posterior
        apenas estreita em torno da borda.

        O valor útil é o menor ``R`` que satisfaz

            P(-R < δ < R) ≥ limiar,   δ ~ t(Δ, var_posterior, n-1)

        Com ``arredondar=True`` o valor sai arredondado **para cima** na 4ª casa
        decimal — a mesma precisão com que é transcrito no YAML. Arredondar para
        o mais próximo quebraria a garantia: uma ROPE exigida de 0,005731 viraria
        0,0057 e devolveria `incerto` justamente no par que a definiu.
        """
        # Recupera as estatísticas já calculadas do par (para não refazer o teste)
        media = float(resultado.get('diferenca', 0.0))
        var = float(resultado.get('var_posterior', 0.0))
        gl = int(resultado.get('gl_posterior', 0))
        
        # Cria uma comparação "fantasma" apenas para usar o método analítico
        # Injetando diretamente as propriedades na instância para pular o __init__
        comp = Comparacao.__new__(Comparacao)
        comp.diferenca_media = media
        comp.variancia = var
        comp.gl = gl
        comp.limiar = min(max(float(self.limiar_bayes), 1e-6), 1.0 - 1e-9)
        
        return comp.rope_minima_equivalencia(arredondar=arredondar)

    def _calcular_effect_sizes(self):
        """Cohen's d para cada par nos resultados de Wilcoxon (métrica secundária)."""
        for r in self.wilcoxon_resultados:
            p1, p2 = r['proto1'], r['proto2']
            diff = self.df[p2] - self.df[p1]
            std_diff = diff.std()
            
            if np.isclose(std_diff, 0):
                cohen_d = 0.0
            else:
                cohen_d = diff.mean() / std_diff
            
            # Faixas convencionais de Cohen (1988) para d: 0.2 / 0.5 / 0.8.
            # (As faixas 0.1/0.3/0.5 são as do r — usá-las no d superestima os rótulos.)
            abs_d = abs(cohen_d)
            if abs_d < 0.20:
                tamanho = self.t['efeito_insignificante']
            elif abs_d < 0.50:
                tamanho = self.t['efeito_pequeno']
            elif abs_d < 0.80:
                tamanho = self.t['efeito_medio']
            else:
                tamanho = self.t['efeito_grande']
            
            r['cohen_d'] = cohen_d
            r['tamanho_efeito_d'] = tamanho
    
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
        """Formata p-valor para exibição. Trata underflow float64 (p=0.0)."""
        if p is None:
            return '—'
        if p == 0.0:
            return '< 1e-300'
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
        """Gera o relatório Markdown completo com todas as métricas enriquecidas."""
        L = []
        t = self.t
        
        titulo = f'{t["titulo_principal"]}: {self.campo} — {self.tecnica}'
        L.append(f'# {titulo}')
        L.append('')
        
        # --- Metadados de reprodutibilidade ---
        try:
            import scipy
            scipy_v = scipy.__version__
        except Exception:
            scipy_v = '?'
        try:
            import scikit_posthocs
            sp_v = scikit_posthocs.__version__
        except Exception:
            sp_v = '?'
        from datetime import datetime
        metadados = t['msg_metadados'].format(
            data=datetime.now().strftime('%Y-%m-%d %H:%M'),
            scipy_v=scipy_v, sp_v=sp_v,
            np_v=np.__version__, pd_v=pd.__version__,
            n=self.N, k=self.K
        )
        L.append(f'> {metadados}')
        L.append('')
        
        # --- Pareamento / descarte global ---
        if self.n_descartado > 0:
            detalhe = ' · '.join(
                f'{p}: {v}' for p, v in self.descartes_por_protocolo.items() if v > 0
            ) or '—'
            L.append('> ' + t['msg_pareamento'].format(
                n_bruto=self.n_bruto, n_descartado=self.n_descartado,
                n=self.N, detalhe=detalhe))
        else:
            L.append('> ' + t['msg_pareamento_sem_descarte'].format(n_bruto=self.n_bruto))
        L.append('')
        
        # --- 1. Ranking ---
        L.append(f'## 1. {t["sec_ranking"]}')
        L.append('')
        L.append(
            f'| {t["col_pos"]} | {t["col_protocolo"]} | {t["col_media"]} | {t["col_desvio"]} '
            f'| {t["col_mediana"]} | {t["col_q1"]} | {t["col_q3"]} '
            f'| {t["col_rank_medio"]} | {t["col_pct_perfeito"]} | {t["col_skewness"]} '
            f'| {t["col_n"]} | {t["col_grupo"]} |'
        )
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in self.ranking.iterrows():
            grupo = self.grupos.get(r['protocolo'], '—')
            L.append(
                f'| {int(r["posicao"])} | {r["protocolo"]} '
                f'| {r["media"]:.4f} | {r["std"]:.4f} | {r["mediana"]:.4f} '
                f'| {r["q1"]:.4f} | {r["q3"]:.4f} '
                f'| {r["rank_medio"]:.2f} | {r["pct_perfeito"]:.1%} | {r["skewness"]:.2f} '
                f'| {int(r["n"])} | {grupo} |'
            )
        L.append('')
        L.append(f'> {t["leg_ranking"]}')
        L.append('')
        
        # --- 2. Friedman ---
        L.append(f'## 2. {t["sec_friedman"]}')
        L.append('')
        
        fr = self.friedman_resultado
        if fr.get('aplicavel'):
            L.append('| K | N | χ² | df | p-value | W | Sig. |')
            L.append('|---|---|---|---|---|---|---|')
            L.append(
                f'| {fr["K"]} | {fr["N"]} | {fr["chi2"]:.2f} '
                f'| {fr["df"]} | {self._fmt_p(fr["p_valor"])} '
                f'| {fr["kendall_w"]:.4f} '
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
            # Nota sobre Shapiro-Wilk em n grande
            if self.N >= 500:
                L.append(f'> *{t["msg_shapiro_n_grande"].format(n=self.N)}*')
                L.append('')
        
        # --- 4. Wilcoxon ---
        if self.wilcoxon_resultados:
            L.append(f'## 4. {t["sec_wilcoxon"]}')
            L.append('')
            # Tabela completa com todas as colunas enriquecidas
            L.append(
                f'| {t["col_prot1"]} | {t["col_prot2"]} '
                f'| Δ | {t["col_delta_med"]} '
                f'| {t["col_z"]} | {t["col_n_prime"]} | {t["col_pct_empates"]} '
                f'| {t["col_r"]} | {t["col_efeito_r"]} '
                f'| {t["col_p_bruto"]} | {t["col_p_corrigido"]} | {t["col_sig"]} '
                f'| Cohen\'s d | {t["col_efeito_d"]} '
                f'| {t["col_mesmo_grupo"]} |'
            )
            L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
            
            # Ordena por p-valor corrigido
            resultados_ord = sorted(self.wilcoxon_resultados, key=lambda x: x.get('p_corrigido', 1.0))
            for r in resultados_ord:
                sinal = '+' if r['diferenca'] >= 0 else ''
                sinal_med = '+' if r['delta_mediano'] >= 0 else ''
                # Concordância Wilcoxon × Nemenyi (interseção de grupos — overlap-safe)
                g1 = self.grupos.get(r['proto1'], '')
                g2 = self.grupos.get(r['proto2'], '')
                mesmo_grupo = t['sim'] if self._compartilham_grupo(g1, g2) else t['nao']
                
                L.append(
                    f'| {r["proto1"]} | {r["proto2"]} '
                    f'| {sinal}{r["diferenca"]:.4f} | {sinal_med}{r["delta_mediano"]:.4f} '
                    f'| {r["z_stat"]:.2f} | {r["n_prime"]} | {r["pct_empates"]:.1%} '
                    f'| {r["r_efeito"]:.4f} | {r["tamanho_efeito_r"]} '
                    f'| {self._fmt_p(r["p_bruto"])} '
                    f'| {self._fmt_p(r["p_corrigido"])} '
                    f'| {t["sim"] if r["significante"] else t["nao"]} '
                    f'| {r["cohen_d"]:.4f} | {r["tamanho_efeito_d"]} '
                    f'| {mesmo_grupo} |'
                )
            L.append('')
            L.append(f'> {t["leg_wilcoxon"]}')
            L.append('')
            # Nota sobre m da correção de Holm
            m = len(self.wilcoxon_resultados)
            L.append(f'> *{t["msg_holm_nota"].format(m=m)}*')
            L.append('')
            # Nota de calibração da ROPE — só faz sentido em bateria de réplicas
            detalhe = getattr(self, 'rope_detalhe', {}) or {}
            if self.calibracao_rope and detalhe.get('rope', 0.0) > 0:
                L.append('> ' + t['nota_rope_calibracao'].format(
                    par=detalhe['par'],
                    delta=f"{detalhe['delta']:.4f}",
                    sd=f"{detalhe['sd']:.4f}",
                    rope=f"{detalhe['rope']:.4f}",
                    limiar=f"{detalhe.get('limiar', 0.95):.2f}",
                    max_delta=f"{self.max_delta:.4f}",
                ))
                L.append('')
        
        # --- 5. Nemenyi Post-hoc (seção materializada) ---
        if not self.nemenyi_pvalores.empty and self.K >= 3:
            L.append(f'## 5. {t["sec_nemenyi"]}')
            L.append('')
            
            # Calcula CD: CD = q_α × √(k(k+1)/(6N))
            # q_α da tabela de Nemenyi (Demšar, 2006) = quantil da Studentized Range / √2.
            # O scikit-posthocs embute o mesmo fator √2 na matriz de p-valores do Nemenyi,
            # portanto esta CD é consistente com os agrupamentos do CD diagram.
            # Ex.: k=7, α=0.05 → ppf=4.1696 → q_α = 2.9484 (valor tabelado em Demšar).
            try:
                from scipy.stats import studentized_range
                q_alpha = studentized_range.ppf(1 - self.alpha, self.K, np.inf) / np.sqrt(2.0)
                if not np.isfinite(q_alpha):
                    q_alpha = 0.0
            except Exception:
                q_alpha = 0.0
            
            cd = q_alpha * np.sqrt(self.K * (self.K + 1) / (6.0 * self.N)) if q_alpha > 0 else 0.0
            
            if cd > 0:
                L.append(t['msg_nemenyi_cd'].format(cd=cd, q_alpha=q_alpha, k=self.K, n=self.N))
                L.append('')
            
            # Tabela de ranks médios
            ranks_friedman = self.df.rank(axis=1, ascending=False).mean()
            ranks_sorted = ranks_friedman.sort_values()
            
            L.append(f'| {t["col_protocolo"]} | {t["col_rank_medio"]} | {t["col_grupo"]} |')
            L.append('|---|---|---|')
            for proto, rank_val in ranks_sorted.items():
                grupo = self.grupos.get(proto, '—')
                L.append(f'| {proto} | {rank_val:.2f} | {grupo} |')
            L.append('')
            
            # CD Diagram (PNG embeddado)
            if self.arquivo_cd_png and os.path.exists(self.arquivo_cd_png):
                L.append(f'![CD Diagram]({os.path.basename(self.arquivo_cd_png)})')
                L.append('')
        elif self.arquivo_cd_png and os.path.exists(self.arquivo_cd_png):
            # Fallback: K < 3 mas ainda mostra o diagram se existir
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
            L.append(f'> *{t["msg_grupos_nota"]}*')
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

def montar_mapa_aliases(config):
    """Mapa {rotulo: alias} do modelo base e dos modelos de comparação ativos.

    Compartilhado com comparar_extracoes_baycomp.py para que as duas análises
    rotulem os mesmos protocolos de forma idêntica em tabelas e figuras.
    """
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

    return mapa_aliases


def _ler_config_frequentista(config):
    """Lê a configuração da análise frequentista do YAML.

    Formato novo (unificado)::

        estatistica:
          frequentista: true
          protocolos:
            Q1_ajuste_fino: [A, B, C]
            Panorama_Geral: "TODOS"

    Formato legado (sem chave `estatistica`): a frequentista roda sempre que
    a flag ``--estatisticas`` é passada ou na execução completa.

    Returns:
        tuple(ativo, recortes_dict):
          ativo: bool — se a frequentista está habilitada
          recortes_dict: dict {nome: lista_protocolos} ou None (todos contra todos)
    """
    bloco = config.get('estatistica')
    if isinstance(bloco, dict):
        # Formato novo: a frequentista é controlada pela sub-chave `frequentista`
        ativo = bloco.get('frequentista', True)
        protocolos_raw = bloco.get('protocolos')
        return ativo, protocolos_raw
    # Formato legado: sem chave `estatistica` → a frequentista roda sempre
    return True, None


def _slug(texto):
    """Normaliza um nome de recorte para uso em nome de arquivo."""
    import unicodedata
    sem_acento = unicodedata.normalize('NFKD', str(texto)).encode('ascii', 'ignore').decode()
    import re as _re
    return _re.sub(r'_+', '_', _re.sub(r'[^0-9A-Za-z]+', '_', sem_acento)).strip('_').lower()


def _gerar_relatorio_rope_global(pasta_estat, dados_por_campo, metricas_usadas, limiar, lang):
    """Gera `00_rope_sugerido.md` com a ROPE calibrada na posterior, por campo.

    ``dados_por_campo`` é ``{campo: {rope, delta, sd, par, n}}`` — o par mais
    exigente de cada campo e a ROPE que o torna equivalente ao ``limiar``.
    """
    arquivo_md = os.path.join(pasta_estat, '00_rope_sugerido.md')
    pt = (lang == 'pt')
    rope_global = max(d['rope'] for d in dados_por_campo.values())
    campos_ord = sorted(dados_por_campo.keys())
    n_docs = max(d.get('n', 0) for d in dados_por_campo.values())
    n_fmt = f'{n_docs:,}'.replace(',', '.')   # separador de milhar pt-BR

    L = ['# ' + ('Sugestão de Calibração da ROPE' if pt else 'ROPE Calibration Suggestion'), '']

    if pt:
        L.append(f'Bateria marcada como **calibração** (`estatistica.calibracao_rope: true`): os '
                 f'protocolos comparados aqui são réplicas do mesmo regime de treinamento e dos '
                 f'mesmos dados, de modo que qualquer diferença entre eles é ruído — não efeito.')
        L.append('')
        L.append(f'**ROPE sugerida (global): `{rope_global:.4f}`** — o maior valor entre os campos, '
                 f'usado como *fallback* para campos não listados em `rope_por_campo`.')
    else:
        L.append('Batch marked as **calibration** (`estatistica.calibracao_rope: true`): the protocols '
                 'compared here are replicas of the same training regime and the same data, so any '
                 'difference between them is noise — not effect.')
        L.append('')
        L.append(f'**Suggested ROPE (global): `{rope_global:.4f}`** — the largest value across fields, '
                 f'used as a fallback for fields not listed in `rope_por_campo`.')
    L.append('')

    # ----------------------------------------------------------- tabela
    L.append('## ' + ('Valor sugerido por campo' if pt else 'Suggested value per field'))
    L.append('')
    if pt:
        L.append('| Campo | Par mais exigente | \\|Δ\\| | sd da posterior | ROPE sugerida |')
    else:
        L.append('| Field | Most demanding pair | \\|Δ\\| | posterior sd | Suggested ROPE |')
    L.append('|---|---|---|---|---|')
    for campo in campos_ord:
        d = dados_por_campo[campo]
        L.append(f"| {campo} | {d['par']} | {d['delta']:.4f} | {d['sd']:.4f} | `{d['rope']:.4f}` |")
    L.append('')

    # ----------------------------------------------------------- explicacao
    L.append('## ' + ('Como o valor é calculado' if pt else 'How the value is computed'))
    L.append('')
    if pt:
        L.extend([
            'A camada bayesiana **não** compara o Δ observado com a ROPE. Para cada par de '
            'protocolos ela constrói a posterior da *diferença média* pelo '
            '`baycomp.CorrelatedTTest` (Benavoli et al., 2017):',
            '',
            '```',
            'δ ~ t(Δ, s², n-1),   s² = var(d) · (1/n + 1/(n-1))',
            '```',
            '',
            'onde `d` são as diferenças pareadas documento a documento, `Δ = média(d)` e o fator '
            '`(1/n + 1/(n-1))` é a correção de Nadeau-Bengio aplicada pelo pacote com `runs=1`. '
            'A classificação `equivalente` exige que a **massa** dessa posterior dentro da ROPE '
            'alcance o limiar:',
            '',
            '```',
            'P(-R < δ < R) ≥ limiar',
            '```',
            '',
            '### Por que o maior |Δ| observado não serve como ROPE',
            '',
            'Fixar `R = |Δ|máx` — a sugestão da versão anterior deste relatório — centra a '
            'posterior do par mais discrepante **exatamente na borda** do intervalo. Metade da '
            'massa cai para fora e `P(equivalência) ≈ 0,50`, abaixo de qualquer limiar útil. '
            'Pior: isso **não melhora com mais dados**, porque a posterior apenas estreita em '
            'torno da borda. A bateria de calibração, rodada com a própria ROPE que ela sugeriu, '
            'reprovava por construção — todos os pares saíam `incerto`.',
            '',
            '### A regra usada',
            '',
            'Para cada campo, a ROPE sugerida é o **menor `R` que basta para todos os pares de '
            'réplicas daquele campo**:',
            '',
            '```',
            'ROPE(campo) = min { R : P(-R < δ < R) ≥ limiar para TODO par de réplicas }',
            '```',
            '',
            'O valor é arredondado **para cima** na 4ª casa decimal — a mesma precisão da '
            'transcrição no YAML —, para que o número copiado preserve a garantia.',
            '',
            f'Como a condição é monotônica em `R`, o valor é determinado pelo par mais exigente e '
            f'resolvido numericamente (Brent) sobre a t de Student. Em forma aproximada, '
            f'`R ≈ |Δ| + t_{{{limiar:.2f}}} · s` desse par: o Δ observado entre as réplicas **mais** '
            f'a margem de incerteza sobre esse próprio Δ. O limiar usado é '
            f'`{limiar:.2f}` (`estatistica.limiar`).',
            '',
            'A consequência prática é a auto-consistência: rodar esta mesma bateria com a ROPE '
            'sugerida produz, por construção, equivalência em 100% dos pares de réplicas. Esse é '
            'o *sanity check* da calibração — se ele falhar, a calibração não é válida.',
            '',
            '### Por que a ROPE é por campo',
            '',
            'O termo dominante da fórmula é `t · s`, e `s` (o ruído de medida) é **estável dentro '
            'de um campo** entre os pares de réplicas, mas genuinamente diferente entre campos: '
            'campos com escores comprimidos perto de 1,0 têm distribuição por documento mais '
            'dispersa e `s` maior. A coluna `sd da posterior` na tabela acima torna isso '
            'auditável. Uma ROPE única forçaria todos os campos ao valor do pior deles, '
            'desperdiçando poder nos demais — por isso `rope_por_campo`.',
        ])
    else:
        L.extend([
            'The Bayesian layer does **not** compare the observed Δ against the ROPE. For each pair '
            'of protocols it builds the posterior of the *mean difference* via '
            '`baycomp.CorrelatedTTest` (Benavoli et al., 2017):',
            '',
            '```',
            'δ ~ t(Δ, s², n-1),   s² = var(d) · (1/n + 1/(n-1))',
            '```',
            '',
            'where `d` are the per-document paired differences, `Δ = mean(d)`, and the factor '
            '`(1/n + 1/(n-1))` is the Nadeau-Bengio correction the package applies with `runs=1`. '
            'The `equivalent` classification requires the posterior **mass** inside the ROPE to '
            'reach the threshold:',
            '',
            '```',
            'P(-R < δ < R) >= threshold',
            '```',
            '',
            '### Why the largest observed |Δ| is not a usable ROPE',
            '',
            'Pinning `R = max|Δ|` — what the previous version of this report suggested — centres '
            'the posterior of the most discrepant pair **exactly on the boundary** of the '
            'interval. Half of the mass falls outside and `P(equivalence) ≈ 0.50`, below any '
            'useful threshold. Worse, this **does not improve with more data**: the posterior '
            'merely narrows around the boundary. The calibration batch, run with the very ROPE it '
            'suggested, failed by construction — every pair came out `uncertain`.',
            '',
            '### The rule in use',
            '',
            'For each field, the suggested ROPE is the **smallest `R` that suffices for every '
            'replica pair of that field**:',
            '',
            '```',
            'ROPE(field) = min { R : P(-R < δ < R) >= threshold for EVERY replica pair }',
            '```',
            '',
            'The value is rounded **up** at the 4th decimal — the same precision used in the YAML '
            'transcription — so that the copied number preserves the guarantee.',
            '',
            f'Since the condition is monotonic in `R`, the value is set by the most demanding pair '
            f'and solved numerically (Brent) over the Student t. In approximate form, '
            f'`R ≈ |Δ| + t_{{{limiar:.2f}}} · s` for that pair: the Δ observed between replicas '
            f'**plus** the uncertainty margin on that very Δ. The threshold in use is '
            f'`{limiar:.2f}` (`estatistica.limiar`).',
            '',
            'The practical consequence is self-consistency: re-running this same batch with the '
            'suggested ROPE yields, by construction, equivalence for 100% of the replica pairs. '
            'That is the calibration sanity check — if it fails, the calibration is not valid.',
            '',
            '### Why the ROPE is per field',
            '',
            'The dominant term is `t · s`, and `s` (the measurement noise) is **stable within a '
            'field** across replica pairs, yet genuinely different across fields: fields with '
            'scores compressed near 1.0 have a more dispersed per-document distribution and a '
            'larger `s`. The `posterior sd` column above makes this auditable. A single ROPE '
            'would force every field to the worst field\'s value, wasting power on the others — '
            'hence `rope_por_campo`.',
        ])
    L.append('')

    # ----------------------------------------------------------- snippet YAML
    L.append('## ' + ('Configuração sugerida para o YAML' if pt else 'Suggested YAML configuration'))
    L.append('')
    L.append('Copie e cole o bloco abaixo na seção `estatistica` do YAML de comparação entre '
             'protocolos distintos:' if pt else
             'Copy and paste the block below into the `estatistica` section of the YAML used to '
             'compare distinct protocols:')
    L.append('')
    L.append('```yaml')
    L.append('metricas_automaticas:')
    L.append(f'    rope: {rope_global:.4f}')
    valores_unicos = set(f"{d['rope']:.4f}" for d in dados_por_campo.values())
    if len(dados_por_campo) > 1 and len(valores_unicos) > 1:
        L.append('    rope_por_campo:')
        for campo in campos_ord:
            L.append(f"      {campo}: {dados_por_campo[campo]['rope']:.4f}")
    L.append('    campos:')
    for campo in campos_ord:
        L.append(f'      - "{campo}"')
    if metricas_usadas:
        L.append('    metricas:')
        for m in metricas_usadas:
            L.append(f'      - {m}')
    L.append('```')
    L.append('')

    # ----------------------------------------------------------- ressalvas
    L.append('## ' + ('Ressalvas' if pt else 'Caveats'))
    L.append('')
    if pt:
        L.extend([
            f'- **Não use esta ROPE nesta mesma bateria como resultado.** Calibrar e testar no '
            f'mesmo dado é circular; aqui a bateria só serve de *sanity check*. O valor é para '
            f'transcrever nos YAMLs que comparam protocolos **distintos**.',
            f'- **O termo `|Δ|` é ruidoso com poucas réplicas.** Com 3 réplicas há apenas 3 pares, '
            f'e o maior deles é o máximo de três observações incertas. Mais réplicas encolhem '
            f'esse termo e estreitam a ROPE.',
            f'- **A ROPE herda o ruído da avaliação.** `s` cai com 1/√n: ampliar o conjunto de '
            f'teste (aqui n = {n_fmt} documentos) reduz a ROPE sem tocar no treinamento.',
            f'- **A ROPE é da métrica automática.** A ROPE da Likert continua pré-registrada à '
            f'mão em `rope_likert`, calibrada pela divergência entre especialistas.',
        ])
    else:
        L.extend([
            '- **Do not report this ROPE as a result of this same batch.** Calibrating and testing '
            'on the same data is circular; here the batch only serves as a sanity check. The value '
            'is meant to be transcribed into the YAMLs that compare **distinct** protocols.',
            '- **The `|Δ|` term is noisy with few replicas.** With 3 replicas there are only 3 '
            'pairs, and the largest is the maximum of three uncertain observations. More replicas '
            'shrink this term and tighten the ROPE.',
            f'- **The ROPE inherits the evaluation noise.** `s` falls with 1/sqrt(n): enlarging the '
            f'test set (here n = {n_docs:,} documents) lowers the ROPE without touching training.',
            '- **This is the automatic-metric ROPE.** The Likert ROPE remains hand-registered in '
            '`rope_likert`, calibrated from the divergence between human specialists.',
        ])
    L.append('')

    with open(arquivo_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f"   📄 Relatório ROPE sugerido: {os.path.basename(arquivo_md)}")


def executar_analise_estatistica(analisador, dados_analise, config, pasta_saida, lang='en'):
    """
    Função principal chamada por comparar_extracoes.py.
    Descobre alvos (campo×métrica) e executa análise estatística para cada um.

    Quando ``estatistica.protocolos`` está definido no YAML, a análise é
    segmentada por cenário (recorte), gerando arquivos com o prefixo do cenário
    (ex: ``Q1_ajuste_fino_estat_...``). Um valor ``"TODOS"`` inclui todos os
    modelos ativos automaticamente. Sem a chave ``protocolos``, faz a análise
    "todos contra todos" (comportamento original).
    
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

    # Verifica se a frequentista está habilitada no novo formato
    freq_ativo, protocolos_config = _ler_config_frequentista(config)
    if not freq_ativo:
        print("   ⚠️  Análise frequentista desativada (estatistica.frequentista: false).")
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
    
    # Descobre alvos de campos e métricas para estatística descritiva e testes
    conf_comp = config.get('configuracao_comparacao', {})
    bloco_estat = config.get('estatistica')
    bloco_estat = bloco_estat if isinstance(bloco_estat, dict) else {}
    
    metricas_auto = bloco_estat.get('metricas_automaticas', {})
    campos_virtuais = config.get('campos_virtuais', {})
    
    if metricas_auto and ('campos' in metricas_auto or 'metricas' in metricas_auto):
        alvos_campos = metricas_auto.get('campos', [])
        alvos_metricas = metricas_auto.get('metricas', [])
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
    
    # Parâmetros compartilhados com a camada bayesiana: o limiar único de
    # decisão (que define a ROPE sugerida) e a marcação de bateria de
    # calibração — só em réplicas do mesmo protocolo a ROPE sugerida faz
    # sentido, porque só aí a diferença observada é integralmente ruído.
    bloco_estat = config.get('estatistica')
    bloco_estat = bloco_estat if isinstance(bloco_estat, dict) else {}
    limiar_bayes = float(bloco_estat.get('limiar', 0.95) or 0.95)
    calibracao_rope = bool(bloco_estat.get('calibracao_rope', False))

    mapa_aliases = montar_mapa_aliases(config)

    # Mapas de métrica → sufixo de coluna / rótulo (constantes do módulo,
    # compartilhadas com comparar_extracoes_baycomp.py)
    mapa_metrica_sufixo = MAPA_METRICA_SUFIXO
    mapa_metrica_display = MAPA_METRICA_DISPLAY

    # Identifica protocolos (modelos) disponíveis
    rotulo_true = analisador.rotulos[1] if len(analisador.rotulos) > 1 else ''
    protocolos = list(analisador.rotulos[2:]) if len(analisador.rotulos) > 2 else []
    
    if not protocolos:
        print("   ⚠️  Nenhum protocolo encontrado para análise estatística.")
        return []

    # Monta os recortes (cenários): cada um gera um bloco de análises com
    # prefixo próprio nos nomes de arquivo, como na bayesiana
    recortes = _montar_recortes_frequentista(protocolos_config, protocolos, mapa_aliases)
    
    resumos = []
    
    n_recortes = len(recortes)
    info_recortes = f" × {n_recortes} recorte(s)" if n_recortes > 1 else ""
    print(f"\n📊 Análise Estatística — {len(alvos_campos)} campo(s) × {len(alvos_metricas)} métrica(s){info_recortes}")

    for nome_recorte, aliases_recorte in recortes:
        prefixo = f'{_slug(nome_recorte)}_' if nome_recorte else ''
        
        if nome_recorte:
            qt_protos = len(aliases_recorte) if aliases_recorte is not None else len(protocolos)
            print(f"\n   🔍 Recorte: {nome_recorte} ({qt_protos} protocolos)")

        for campo in alvos_campos:
            for metrica in alvos_metricas:
                sufixo = mapa_metrica_sufixo.get(metrica, metrica)
                display = mapa_metrica_display.get(metrica, metrica)
                
                # Padrão de coluna: {protocolo}_{campo}_{sufixo}_F1
                # Tenta encontrar as colunas no DataFrame
                df_largo = pd.DataFrame()
                
                for proto in protocolos:
                    alias = mapa_aliases.get(proto, proto)
                    # Filtra pelo recorte: se aliases_recorte está definido,
                    # só inclui protocolos que pertençam ao recorte
                    if aliases_recorte is not None and alias not in aliases_recorte:
                        continue

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
                        df_largo[alias] = df_resultados[col_encontrada]
                
                if df_largo.empty or len(df_largo.columns) < 2:
                    continue
                
                # Reordena colunas pela ordem declarada no recorte
                if aliases_recorte is not None:
                    colunas_ordenadas = [a for a in aliases_recorte if a in df_largo.columns]
                    if colunas_ordenadas:
                        df_largo = df_largo[colunas_ordenadas]

                # Remove linhas com NaN
                df_largo = df_largo.dropna()
                
                metrica_nome = f'{campo}_{sufixo}_F1'
                nome_base = f'{prefixo}estat_{campo}_{metrica}'.replace('(', '').replace(')', '')
                arquivo_md = os.path.join(pasta_estat, f'{nome_base}.md')
                arquivo_png = os.path.join(pasta_estat, f'{nome_base}_cd.png')
                
                print(f"   → {campo} × {display} ({len(df_largo)} docs, {len(df_largo.columns)} protocolos)")
                
                analise = AnaliseEstatistica(df_largo, config={
                    'metrica_nome': metrica_nome,
                    'campo': campo,
                    'limiar_bayes': limiar_bayes,
                    'calibracao_rope': calibracao_rope,
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
        
        # ROPE sugerida: só em bateria de calibração (réplicas do mesmo
        # protocolo). Fora disso o número somaria efeitos reais ao ruído e
        # inflaria a margem de indiferença de quem o copiasse.
        if calibracao_rope:
            # Agrega por campo a ROPE do par mais exigente (excluindo métricas LLM)
            resumos_auto = [r for r in resumos
                            if not r.get('is_llm', False) and r.get('rope_calibrado', 0.0) > 0]
            dados_por_campo = {}
            metricas_usadas_set = set()
            for r in resumos_auto:
                campo = r.get('campo', '(global)')
                detalhe = r.get('rope_detalhe') or {}
                if detalhe.get('rope', 0.0) > dados_por_campo.get(campo, {}).get('rope', 0.0):
                    dados_por_campo[campo] = detalhe
                # Extrai nome da métrica do metrica_nome (ex: "(global)_rouge_F1" → "rouge_l")
                metrica_nome = r.get('metrica', '')
                for nome_yaml, sufixo in MAPA_METRICA_SUFIXO.items():
                    if f'_{sufixo}_' in metrica_nome:
                        metricas_usadas_set.add(nome_yaml)
                        break

            if dados_por_campo:
                rope_global = max(d['rope'] for d in dados_por_campo.values())
                print(f"   📌 ROPE sugerida (global): {rope_global:.4f} "
                      f"(margem que torna as réplicas equivalentes ao limiar de {limiar_bayes:.2f})")
                if len(dados_por_campo) > 1:
                    for campo, d in sorted(dados_por_campo.items()):
                        print(f"       └─ {campo}: {d['rope']:.4f} "
                              f"(|Δ| {d['delta']:.4f} + margem, par {d['par']})")
                _gerar_relatorio_rope_global(pasta_estat, dados_por_campo,
                                             sorted(metricas_usadas_set), limiar_bayes, lang)
            
    else:
        print("   ⚠️  Nenhuma análise estatística gerada (combinações campo×métrica não encontradas nos dados).")
    
    return resumos


def _montar_recortes_frequentista(protocolos_config, protocolos_disponiveis, mapa_aliases):
    """Monta a lista de recortes para a análise frequentista.

    Args:
        protocolos_config: valor bruto de ``estatistica.protocolos`` — pode ser
            ``None`` (todos contra todos), ``dict`` (cenários nomeados) ou
            ``list`` (recorte único anônimo).
        protocolos_disponiveis: rótulos dos modelos disponíveis no analisador.
        mapa_aliases: {rotulo: alias}.

    Returns:
        list[tuple(nome, aliases)]: cada tupla contém o nome do recorte (ou
        ``None`` para o anônimo) e a lista de aliases a incluir (ou ``None``
        para "todos"). A ordem dos aliases respeita a declaração no YAML.
    """
    if protocolos_config is None:
        # Sem configuração: análise global todos-contra-todos (comportamento original)
        return [(None, None)]

    if isinstance(protocolos_config, list):
        # Lista simples: recorte único anônimo com os protocolos listados
        aliases = _resolver_aliases(protocolos_config, protocolos_disponiveis, mapa_aliases)
        return [(None, aliases)]

    if isinstance(protocolos_config, dict):
        recortes = []
        for nome, lista in protocolos_config.items():
            if isinstance(lista, str) and lista.strip().upper() == 'TODOS':
                # "TODOS" → todos os modelos, nomeado
                recortes.append((str(nome), None))
            else:
                aliases = _resolver_aliases(list(lista or []), protocolos_disponiveis, mapa_aliases)
                if len(aliases) >= 2:
                    recortes.append((str(nome), aliases))
                else:
                    print(f"   ⚠️  Recorte '{nome}' resolveu {len(aliases)} protocolo(s) — "
                          "menos que os dois necessários. Ignorado.")
        return recortes if recortes else [(None, None)]

    return [(None, None)]


def _resolver_aliases(selecao, protocolos_disponiveis, mapa_aliases):
    """Resolve nomes do YAML (alias ou rótulo) em aliases, na ordem declarada.

    Mesma lógica de ``selecionar_protocolos`` do módulo bayesiano, sem depender
    dele (evita import circular).
    """
    # Monta índice: rótulo primeiro (precedência), alias depois
    indice = {}
    for rotulo in protocolos_disponiveis:
        indice.setdefault(str(rotulo).strip().lower(), rotulo)
    for rotulo in protocolos_disponiveis:
        indice.setdefault(str(mapa_aliases.get(rotulo, rotulo)).strip().lower(), rotulo)

    aliases, vistos = [], set()
    for nome in selecao:
        rotulo = indice.get(str(nome).strip().lower())
        if rotulo is not None and rotulo not in vistos:
            vistos.add(rotulo)
            aliases.append(mapa_aliases.get(rotulo, rotulo))
    return aliases


def montar_dataframes_llm(dados_analise, protocolos, mapa_aliases):
    """Monta os DataFrames largos (documentos × protocolos) da avaliação LLM-as-a-Judge.

    Fonte única dos escores do juiz para TODAS as análises (frequentista e
    bayesiana): os arquivos `{id}.avaliacao.json` carregados por
    `CargaDadosComparacao` e indexados em `dados_analise.get_avaliacao()`. Esses
    arquivos vêm da coluna `configuracao_comparacao.campos_dataset.avaliacao`
    (entrada parquet/csv) ou já existem na pasta de JSONs — os dois caminhos
    convergem no mesmo loader, então não há caminho de código separado.

    Args:
        dados_analise: instância de JsonAnaliseDados.
        protocolos: rótulos dos modelos a extrair (na ordem desejada).
        mapa_aliases: {rotulo: alias} — as colunas saem já com o alias.

    Returns:
        tuple(df_f1, df_nota): DataFrames indexados pelo id do documento. Vazios
        quando o juiz não produziu a métrica correspondente.
    """
    pk = dados_analise.config.nome_campo_id

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

    return df_f1, df_nota


def _processar_llm_estatisticas(dados_analise, config, protocolos, mapa_aliases, pasta_estat, lang, resumos):
    """Processa análise estatística para LLM-as-a-Judge (separada das métricas de similaridade)."""
    df_f1, df_nota = montar_dataframes_llm(dados_analise, protocolos, mapa_aliases)

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