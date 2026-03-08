# -*- coding: utf-8 -*-

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Análise estatística intra-família para comparação de modelos LLM-as-a-Judge.
Gera todos os pares C(n,2) dentro de cada família e executa testes par a par.
Inclui: desempenho por modelo, Shapiro-Wilk, Wilcoxon signed-rank e eficiência.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations


class AnaliseEstatistica:
    """Análise estatística par a par intra-família."""

    def __init__(self, df_dados, config=None):
        """
        Args:
            df_dados: DataFrame com colunas: id_doc, valor1, valor2, custo1, custo2, familia, rotulo_modelo.
                      valor1/custo1 = modelo base; valor2/custo2 = modelo comparado.
            config: dict com 'rotulo_base', 'familia_base', 'arquivo_saida'.
        """
        self.df = df_dados.copy()
        self.config = config or {}
        self.rotulo_base = self.config.get('rotulo_base', 'Base')
        self.familia_base = self.config.get('familia_base', None)
        self.arquivo_saida = self.config.get('arquivo_saida', 'relatorio_estatistico.md')

        for col in ['id_doc', 'valor1', 'valor2', 'custo1', 'custo2', 'familia', 'rotulo_modelo']:
            if col not in self.df.columns:
                raise ValueError(f"Coluna obrigatória não encontrada: {col}")

        self.resultados_desempenho = pd.DataFrame()
        self.resultados_shapiro = pd.DataFrame()
        self.resultados_wilcoxon = pd.DataFrame()
        self.resultados_eficiencia = pd.DataFrame()

    def processar_analise(self):
        """Executa todas as análises intra-família e gera o relatório."""
        self._calcular_desempenho()
        self._calcular_shapiro()
        self._calcular_wilcoxon()
        self._calcular_eficiencia()
        self._gerar_relatorio_markdown()

    # ========================================================================
    # Helpers
    # ========================================================================

    def _valores_modelo(self, modelo):
        """Retorna (valores_F1, custos) indexados por id_doc."""
        if modelo == self.rotulo_base:
            # Base: valor1/custo1 — deduplica por doc (são iguais em todas as linhas)
            df_base = self.df.drop_duplicates(subset='id_doc').set_index('id_doc')
            return df_base['valor1'], df_base['custo1']
        # Modelo de comparação: valor2/custo2 filtrado por rotulo_modelo
        df_mod = self.df[self.df['rotulo_modelo'] == modelo].set_index('id_doc')
        return df_mod['valor2'], df_mod['custo2']

    def _par_valores(self, m1, m2):
        """Retorna (v1, v2, c1, c2) pareados por id_doc para dois modelos."""
        vals1, costs1 = self._valores_modelo(m1)
        vals2, costs2 = self._valores_modelo(m2)
        idx = vals1.dropna().index.intersection(vals2.dropna().index)
        return vals1.loc[idx], vals2.loc[idx], costs1.loc[idx], costs2.loc[idx]

    def _pares_por_familia(self):
        """Gera todos os pares C(n,2) de modelos dentro de cada família."""
        familias = {}
        for modelo in self.df['rotulo_modelo'].unique():
            fam = self.df[self.df['rotulo_modelo'] == modelo]['familia'].iloc[0]
            familias.setdefault(fam, []).append(modelo)

        # Inclui o modelo base na sua família
        if self.familia_base:
            familias.setdefault(self.familia_base, [])
            if self.rotulo_base not in familias[self.familia_base]:
                familias[self.familia_base].insert(0, self.rotulo_base)

        pares = []
        for fam in sorted(familias):
            for m1, m2 in combinations(familias[fam], 2):
                pares.append((fam, m1, m2))
        return pares

    def _todos_modelos_por_familia(self):
        """Retorna dict {familia: [modelos]} incluindo base."""
        familias = {}
        for modelo in self.df['rotulo_modelo'].unique():
            fam = self.df[self.df['rotulo_modelo'] == modelo]['familia'].iloc[0]
            familias.setdefault(fam, []).append(modelo)
        if self.familia_base:
            familias.setdefault(self.familia_base, [])
            if self.rotulo_base not in familias[self.familia_base]:
                familias[self.familia_base].insert(0, self.rotulo_base)
        return familias

    # ========================================================================
    # Análises
    # ========================================================================

    def _calcular_desempenho(self):
        """Ranking de todos os modelos ordenado por média F1 decrescente."""
        dados = []
        processados = set()
        for fam, modelos in sorted(self._todos_modelos_por_familia().items()):
            for modelo in modelos:
                if modelo in processados:
                    continue
                processados.add(modelo)
                vals, costs = self._valores_modelo(modelo)
                vals = vals.dropna()
                if len(vals) == 0:
                    continue
                dados.append({
                    'familia': fam, 'modelo': modelo,
                    'media': vals.mean(), 'std': vals.std(),
                    'mediana': vals.median(), 'suporte': len(vals),
                    'custo_medio': costs.mean()
                })
        df = pd.DataFrame(dados)
        if not df.empty:
            df['posicao'] = df.index + 1
        self.resultados_desempenho = df

    def _calcular_shapiro(self):
        """Shapiro-Wilk: normalidade dos deltas para cada par intra-família."""
        resultados = []
        for fam, m1, m2 in self._pares_por_familia():
            v1, v2, _, _ = self._par_valores(m1, m2)
            deltas = (v2 - v1).dropna()
            if len(deltas) >= 3:
                stat, p = stats.shapiro(deltas)
                resultados.append({
                    'familia': fam, 'modelo1': m1, 'modelo2': m2,
                    'n': len(deltas), 'shapiro_w': stat,
                    'shapiro_p': p, 'normal': p > 0.05
                })
        self.resultados_shapiro = pd.DataFrame(resultados)

    def _calcular_wilcoxon(self):
        """Wilcoxon signed-rank para cada par intra-família com Cohen's d."""
        resultados = []
        for fam, m1, m2 in self._pares_por_familia():
            v1, v2, _, _ = self._par_valores(m1, m2)
            if len(v1) == 0:
                continue

            diff = v2 - v1
            try:
                _, p_value = stats.wilcoxon(v1, v2)
            except ValueError:
                p_value = 1.0

            std_diff = diff.std()
            cohen_d = 0.0 if np.isclose(std_diff, 0) else diff.mean() / std_diff

            # Cohen (1988)
            abs_d = abs(cohen_d)
            if abs_d < 0.10:   tamanho = "Insignificante"
            elif abs_d < 0.30: tamanho = "Pequeno"
            elif abs_d < 0.50: tamanho = "Médio"
            else:              tamanho = "Grande"

            resultados.append({
                'familia': fam, 'modelo1': m1, 'modelo2': m2,
                'n': len(v1), 'media_m1': v1.mean(), 'media_m2': v2.mean(),
                'diferenca': diff.mean(), 'p_value': p_value,
                'significante': p_value < 0.05,
                'cohen_d': cohen_d, 'tamanho_efeito': tamanho
            })
        self.resultados_wilcoxon = pd.DataFrame(resultados)

    def _calcular_eficiencia(self):
        """Eficiência (valor/custo) para cada par intra-família."""
        resultados = []
        for fam, m1, m2 in self._pares_por_familia():
            v1, v2, c1, c2 = self._par_valores(m1, m2)
            if len(v1) == 0:
                continue

            cm1, cm2 = c1.mean(), c2.mean()
            vm1, vm2 = v1.mean(), v2.mean()

            delta_custo_pct = ((cm2 - cm1) / cm1 * 100) if cm1 > 0 else 0
            delta_valor_pct = ((vm2 - vm1) / vm1 * 100) if vm1 > 0 else 0

            efi1 = (vm1 / cm1) if cm1 > 0 else 0
            efi2 = (vm2 / cm2) if cm2 > 0 else 0
            delta_efi = ((efi2 - efi1) / efi1 * 100) if efi1 > 0 else 0

            resultados.append({
                'familia': fam, 'modelo1': m1, 'modelo2': m2,
                'custo_m1': cm1, 'custo_m2': cm2,
                'delta_custo_pct': delta_custo_pct, 'delta_valor_pct': delta_valor_pct,
                'eficiencia_m1': efi1, 'eficiencia_m2': efi2,
                'delta_eficiencia_pct': delta_efi
            })
        self.resultados_eficiencia = pd.DataFrame(resultados)

    # ========================================================================
    # Relatório Markdown
    # ========================================================================

    def _fmt_p(self, p):
        return f"{p:.4f}" if p > 0.0001 else f"{p:.2e}"

    def _gerar_relatorio_markdown(self):
        L = []
        L.append("# Relatório de Análise Estatística — Comparação Intra-Família")
        L.append("")

        # --- 1. Desempenho (Ranking) ---
        L.append("## 1. Desempenho (Ranking)")
        L.append("")
        L.append("Ranking dos modelos ordenado pela média de desempenho F1 da avaliação LLM-as-a-Judge (decrescente).")
        L.append("")
        L.append(
            "A média e o desvio padrão (σ) indicam a tendência central e a dispersão dos scores F1. "
            "A mediana complementa a análise por ser robusta a outliers. "
            "O suporte indica o número de documentos avaliados para cada modelo."
        )
        L.append("")
        L.append("| Pos | Família | Modelo | Média | σ | Mediana | Suporte | Custo Médio |")
        L.append("|---|---|---|---|---|---|---|---|")
        for _, r in self.resultados_desempenho.iterrows():
            L.append(
                f"| {int(r['posicao'])} | {r['familia']} | {r['modelo']} "
                f"| {r['media']:.4f} | {r['std']:.4f} | {r['mediana']:.4f} "
                f"| {int(r['suporte'])} | {r['custo_medio']:.4f} |"
            )
        L.append("")

        # --- 2. Shapiro-Wilk ---
        L.append("## 2. Normalidade dos Deltas (Shapiro-Wilk)")
        L.append("")
        L.append("Verifica se as diferenças (Δ = Modelo 2 − Modelo 1) de cada par intra-família seguem distribuição normal.")
        L.append("")
        L.append(
            "Se p > 0,05, não se rejeita a normalidade e testes paramétricos (ex: t de Student) seriam aplicáveis. "
            "Se p ≤ 0,05, a distribuição dos deltas não é normal e testes não-paramétricos como Wilcoxon são mais adequados. "
            "Com amostras grandes (n > 30), o Wilcoxon é robusto e geralmente preferido por não assumir normalidade."
        )
        L.append("")
        L.append("| Família | Modelo 1 | Modelo 2 | n | Shapiro W | p-valor | Normal (p>0.05) |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in self.resultados_shapiro.iterrows():
            L.append(
                f"| {r['familia']} | {r['modelo1']} | {r['modelo2']} "
                f"| {r['n']} | {r['shapiro_w']:.4f} | {self._fmt_p(r['shapiro_p'])} "
                f"| {'Sim' if r['normal'] else 'Não'} |"
            )
        L.append("")

        # --- 3. Wilcoxon ---
        L.append("## 3. Teste de Wilcoxon (Signed-Rank)")
        L.append("")
        L.append("Teste não-paramétrico para amostras pareadas, aplicado a cada par de modelos dentro da mesma família.")
        L.append("")
        L.append(
            "A diferença é calculada como Δ = Modelo 2 − Modelo 1: valores positivos indicam que o Modelo 2 superou o Modelo 1. "
            "Um p-valor < 0,05 indica rejeição da hipótese nula de igualdade (diferença estatisticamente significativa). "
            "O Cohen's d mede a magnitude prática do efeito: "
            "Insignificante (|d| < 0,10), Pequeno (0,10–0,30), Médio (0,30–0,50), Grande (≥ 0,50)."
        )
        L.append("")
        L.append("| Família | Modelo 1 | Modelo 2 | n | Média M1 | Média M2 | Δ | p-valor | Sig. | Cohen's d | Efeito |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in self.resultados_wilcoxon.iterrows():
            L.append(
                f"| {r['familia']} | {r['modelo1']} | {r['modelo2']} "
                f"| {r['n']} | {r['media_m1']:.4f} | {r['media_m2']:.4f} "
                f"| {r['diferenca']:.4f} | {self._fmt_p(r['p_value'])} "
                f"| {'Sim' if r['significante'] else 'Não'} "
                f"| {r['cohen_d']:.4f} | {r['tamanho_efeito']} |"
            )
        L.append("")

        # --- 4. Eficiência ---
        L.append("## 4. Eficiência e Custos")
        L.append("")
        L.append("Compara o custo-benefício (eficiência = qualidade / custo) entre cada par de modelos da mesma família.")
        L.append("")
        L.append(
            "Δ Eficiência positivo indica que o Modelo 2 é mais eficiente que o Modelo 1. "
            "Δ Custo positivo significa que o Modelo 2 é mais caro. "
            "O cenário ideal é Δ Valor positivo com Δ Custo negativo (melhor qualidade, menor custo)."
        )
        L.append("")
        L.append("| Família | Modelo 1 | Modelo 2 | Custo M1 | Custo M2 | Δ Custo (%) | Δ Valor (%) | Efic. M1 | Efic. M2 | Δ Efic. (%) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in self.resultados_eficiencia.iterrows():
            L.append(
                f"| {r['familia']} | {r['modelo1']} | {r['modelo2']} "
                f"| {r['custo_m1']:.4f} | {r['custo_m2']:.4f} "
                f"| {r['delta_custo_pct']:.2f}% | {r['delta_valor_pct']:.2f}% "
                f"| {r['eficiencia_m1']:.4f} | {r['eficiencia_m2']:.4f} | {r['delta_eficiencia_pct']:.2f}% |"
            )

        self.markdown_content = "\n".join(L)

    def salvar_relatorio(self):
        """Salva o relatório Markdown no arquivo especificado."""
        if not hasattr(self, 'markdown_content'):
            self.processar_analise()
        with open(self.arquivo_saida, 'w', encoding='utf-8') as f:
            f.write(self.markdown_content)
        print(f"Relatório salvo em: {self.arquivo_saida}")


if __name__ == "__main__":
    print("Executando teste...")
    np.random.seed(42)
    n = 80

    # Simula: base_gpt5 (família GPT-5) comparado contra 4 modelos em 3 famílias
    modelos_cfg = [
        ('agentes_gpt5', 'GPT-5', 0.68, 0.10, 1.8),
        ('base_gemma3(12)', 'Gemma 3 12b', 0.62, 0.12, 0.8),
        ('agentes_gemma3(12)', 'Gemma 3 12b', 0.64, 0.11, 1.2),
        ('base_gemma3(27)', 'Gemma 3 27b', 0.66, 0.09, 1.5),
    ]
    rows = []
    for i in range(n):
        base_f1 = np.random.normal(0.70, 0.10)
        for modelo, fam, mean, std, custo in modelos_cfg:
            rows.append({
                'id_doc': f'doc_{i}', 'valor1': base_f1,
                'valor2': np.random.normal(mean, std),
                'custo1': 1.0, 'custo2': custo + np.random.uniform(-0.1, 0.1),
                'familia': fam, 'rotulo_modelo': modelo
            })

    df = pd.DataFrame(rows)
    analise = AnaliseEstatistica(df, config={
        'rotulo_base': 'base_gpt5', 'familia_base': 'GPT-5',
        'arquivo_saida': '/tmp/teste_analise_estatistica.md'
    })
    analise.processar_analise()
    analise.salvar_relatorio()

    # Mostra pares gerados
    print(f"\nPares intra-família:")
    for fam, m1, m2 in analise._pares_por_familia():
        print(f"  {fam}: {m1} vs {m2}")
    print("OK!")
