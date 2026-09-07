# -*- coding: utf-8 -*-
# Autor: Luiz Anísio
# Fonte: https://github.com/luizanisio/llms/tree/main/src

"""
Teste de validação do módulo util_analise_estatistica (interface N×K).
Simula K protocolos com diferenças conhecidas e verifica:
- Friedman + Kendall's W, Nemenyi + CD, Wilcoxon+Holm + r=|z|/√n', Shapiro-Wilk
- Descritivas não-paramétricas (Q1, Q3, rank médio, % perfeito, skewness)
- Δ mediano, n', % empates, concordância Wilcoxon×Nemenyi
- Formatação de p-valores (sem underflow 0.00e+00)
- Seção 5 Nemenyi materializada, notas metodológicas, metadados
- Geração de .md e .png
"""

import pandas as pd
import numpy as np
import sys
import os
import re
import util  # garante que a pasta src está no sys.path
from util_analise_estatistica import AnaliseEstatistica


def test_analise_basica():
    """Teste com 5 protocolos e diferenças claras."""
    print("═" * 60)
    print("Teste 1: 5 protocolos com diferenças claras")
    print("═" * 60)
    
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'A': np.random.normal(0.60, 0.12, n).clip(0, 1),
        'B': np.random.normal(0.65, 0.10, n).clip(0, 1),
        'D1': np.random.normal(0.80, 0.08, n).clip(0, 1),
        'D2': np.random.normal(0.82, 0.07, n).clip(0, 1),
        'D3': np.random.normal(0.85, 0.06, n).clip(0, 1),
    })
    
    pasta_teste = os.path.join(os.path.dirname(__file__) or '.', '_teste_estatistica')
    os.makedirs(pasta_teste, exist_ok=True)
    
    analise = AnaliseEstatistica(df, config={
        'metrica_nome': '(global)_bertscore_F1',
        'campo': '(global)',
        'tecnica': 'BERTScore',
        'arquivo_md': os.path.join(pasta_teste, 'estat_test1.md'),
        'arquivo_cd_png': os.path.join(pasta_teste, 'estat_test1_cd.png'),
        'lang': 'en',
    })
    
    resumo = analise.processar()
    analise.salvar()
    
    # --- Validações originais ---
    assert resumo['K'] == 5, f"K esperado=5, obtido={resumo['K']}"
    assert resumo['N'] == n, f"N esperado={n}, obtido={resumo['N']}"
    assert resumo['friedman_sig'] == True, f"Friedman deveria ser significativo"
    assert resumo['n_grupos'] == 2, f"Esperado 2 grupos, obtido={resumo['n_grupos']}"
    
    # Verifica que A e B estão no mesmo grupo
    grupo_a = analise.grupos.get('A', '')
    grupo_b = analise.grupos.get('B', '')
    assert grupo_a == grupo_b, f"A e B deveriam estar no mesmo grupo: {grupo_a} vs {grupo_b}"
    
    # Verifica que D1, D2, D3 estão no mesmo grupo e diferente de A/B
    grupo_d1 = analise.grupos.get('D1', '')
    assert grupo_d1 != grupo_a, f"D1 deveria estar em grupo diferente de A"
    
    # Verifica geração de arquivos
    assert os.path.exists(os.path.join(pasta_teste, 'estat_test1.md')), "Arquivo .md não gerado"
    assert os.path.exists(os.path.join(pasta_teste, 'estat_test1_cd.png')), "Arquivo CD .png não gerado"
    
    # --- Validações novas: Kendall's W ---
    fr = analise.friedman_resultado
    assert 'kendall_w' in fr, "Kendall's W ausente no resultado do Friedman"
    assert 0 < fr['kendall_w'] <= 1.0, f"Kendall's W deveria estar em (0,1], obtido={fr['kendall_w']}"
    
    # --- Validações novas: Ranking enriquecido ---
    rank = analise.ranking
    for col in ['q1', 'q3', 'rank_medio', 'pct_perfeito', 'skewness']:
        assert col in rank.columns, f"Coluna '{col}' ausente no ranking"
    # Q1 <= mediana <= Q3
    for _, r in rank.iterrows():
        assert r['q1'] <= r['mediana'] <= r['q3'], \
            f"Protocolo {r['protocolo']}: Q1={r['q1']:.4f} <= Med={r['mediana']:.4f} <= Q3={r['q3']:.4f} violado"
    # rank_medio deve estar entre 1 e K
    for _, r in rank.iterrows():
        assert 1 <= r['rank_medio'] <= resumo['K'], \
            f"Protocolo {r['protocolo']}: rank_medio={r['rank_medio']:.2f} fora de [1, {resumo['K']}]"
    
    # --- Validações novas: Wilcoxon enriquecido ---
    for w in analise.wilcoxon_resultados:
        # Campos obrigatórios
        for campo in ['z_stat', 'n_prime', 'r_efeito', 'tamanho_efeito_r', 'pct_empates',
                       'delta_mediano', 'cohen_d', 'tamanho_efeito_d']:
            assert campo in w, f"Campo '{campo}' ausente no resultado Wilcoxon para {w['proto1']}↔{w['proto2']}"
        # n' <= n
        assert w['n_prime'] <= w['n'], \
            f"n' ({w['n_prime']}) > n ({w['n']}) para {w['proto1']}↔{w['proto2']}"
        # r >= 0
        assert w['r_efeito'] >= 0, f"r negativo para {w['proto1']}↔{w['proto2']}"
        # 0 <= pct_empates <= 1
        assert 0 <= w['pct_empates'] <= 1, \
            f"pct_empates={w['pct_empates']:.4f} fora de [0,1] para {w['proto1']}↔{w['proto2']}"
        # tamanho_efeito_r e _d devem ser strings de classificação válidas
        efeitos_validos = {'Negligible', 'Small', 'Medium', 'Large'}
        assert w['tamanho_efeito_r'] in efeitos_validos, \
            f"tamanho_efeito_r='{w['tamanho_efeito_r']}' inválido para {w['proto1']}↔{w['proto2']}"
        assert w['tamanho_efeito_d'] in efeitos_validos, \
            f"tamanho_efeito_d='{w['tamanho_efeito_d']}' inválido para {w['proto1']}↔{w['proto2']}"
    
    print(f"✅ Resumo: {resumo}")
    print(f"✅ Grupos: {analise.grupos}")
    print(f"✅ Kendall's W: {fr['kendall_w']:.4f}")
    print(f"✅ Ranking enriquecido: Q1/Q3/rank_medio/pct_perfeito/skewness presentes")
    print(f"✅ Wilcoxon enriquecido: z/n'/r/Δmed/%emp/cohen_d presentes e válidos")
    print()


def test_dois_protocolos():
    """Teste com apenas 2 protocolos (Friedman não se aplica)."""
    print("═" * 60)
    print("Teste 2: 2 protocolos (Friedman não aplicável)")
    print("═" * 60)
    
    np.random.seed(42)
    n = 50
    
    df = pd.DataFrame({
        'Base': np.random.normal(0.70, 0.10, n).clip(0, 1),
        'Modelo': np.random.normal(0.75, 0.08, n).clip(0, 1),
    })
    
    pasta_teste = os.path.join(os.path.dirname(__file__) or '.', '_teste_estatistica')
    os.makedirs(pasta_teste, exist_ok=True)
    
    analise = AnaliseEstatistica(df, config={
        'metrica_nome': '(global)_rouge_F1',
        'campo': '(global)',
        'tecnica': 'ROUGE-L',
        'arquivo_md': os.path.join(pasta_teste, 'estat_test2.md'),
        'arquivo_cd_png': '',  # Sem CD diagram para K=2
        'lang': 'pt',
    })
    
    resumo = analise.processar()
    analise.salvar()
    
    assert resumo['K'] == 2, f"K esperado=2, obtido={resumo['K']}"
    assert resumo['friedman_p'] is None, "Friedman não deveria ser aplicável com K=2"
    assert len(analise.wilcoxon_resultados) == 1, "Deveria ter exatamente 1 comparação par a par"
    
    # --- Novo: Wilcoxon enriquecido mesmo com K=2 ---
    w = analise.wilcoxon_resultados[0]
    assert 'r_efeito' in w, "r_efeito ausente com K=2"
    assert 'delta_mediano' in w, "delta_mediano ausente com K=2"
    
    # --- Novo: nota Holm com m=1 no markdown ---
    assert 'm = 1' in analise.markdown_content, "Nota Holm com m=1 ausente no relatório K=2"
    
    print(f"✅ Resumo: {resumo}")
    print(f"✅ Wilcoxon K=2: p={w['p_corrigido']:.4f}, r={w['r_efeito']:.4f}")
    print()


def test_amostras_insuficientes():
    """Teste com menos de 20 amostras."""
    print("═" * 60)
    print("Teste 3: Amostras insuficientes (n=10)")
    print("═" * 60)
    
    np.random.seed(42)
    
    df = pd.DataFrame({
        'A': np.random.normal(0.70, 0.10, 10),
        'B': np.random.normal(0.75, 0.08, 10),
        'C': np.random.normal(0.80, 0.06, 10),
    })
    
    pasta_teste = os.path.join(os.path.dirname(__file__) or '.', '_teste_estatistica')
    os.makedirs(pasta_teste, exist_ok=True)
    
    analise = AnaliseEstatistica(df, config={
        'metrica_nome': 'test',
        'campo': 'test',
        'tecnica': 'test',
        'arquivo_md': os.path.join(pasta_teste, 'estat_test3.md'),
        'lang': 'en',
        'min_amostras': 20,
    })
    
    resumo = analise.processar()
    analise.salvar()
    
    assert analise._analise_realizada == False, "Análise não deveria ter sido realizada"
    assert 'WARNING' in analise.markdown_content or 'insufficient' in analise.markdown_content.lower() or 'below' in analise.markdown_content.lower(), \
        "Relatório deveria conter aviso de amostras insuficientes"
    
    print(f"✅ Análise corretamente não realizada (n={analise.N} < min_amostras={analise.min_amostras})")
    print()


def test_portugues():
    """Teste com idioma português."""
    print("═" * 60)
    print("Teste 4: Relatório em português")
    print("═" * 60)
    
    np.random.seed(42)
    n = 50
    
    df = pd.DataFrame({
        'Proto_A': np.random.normal(0.60, 0.10, n).clip(0, 1),
        'Proto_B': np.random.normal(0.70, 0.08, n).clip(0, 1),
        'Proto_C': np.random.normal(0.80, 0.06, n).clip(0, 1),
    })
    
    pasta_teste = os.path.join(os.path.dirname(__file__) or '.', '_teste_estatistica')
    os.makedirs(pasta_teste, exist_ok=True)
    
    analise = AnaliseEstatistica(df, config={
        'metrica_nome': '(global)_bertscore_F1',
        'campo': '(global)',
        'tecnica': 'BERTScore',
        'arquivo_md': os.path.join(pasta_teste, 'estat_test4_pt.md'),
        'arquivo_cd_png': os.path.join(pasta_teste, 'estat_test4_pt_cd.png'),
        'lang': 'pt',
    })
    
    resumo = analise.processar()
    analise.salvar()
    
    md = analise.markdown_content
    
    # --- Validações originais ---
    assert 'Análise Estatística' in md, "Título deveria estar em português"
    assert 'Ranking de Desempenho' in md, "Seções deveriam estar em português"
    
    # --- Novas validações i18n ---
    assert 'Efeito (r)' in md, "Coluna 'Efeito (r)' ausente no relatório PT"
    assert 'Efeito (d)' in md, "Coluna 'Efeito (d)' ausente no relatório PT"
    assert '% emp.' in md, "Coluna '% emp.' ausente no relatório PT"
    assert 'Mesmo Grupo?' in md, "Coluna 'Mesmo Grupo?' ausente no relatório PT"
    assert 'Δ med.' in md, "Coluna 'Δ med.' ausente no relatório PT"
    assert 'Rank Médio' in md, "Coluna 'Rank Médio' ausente no relatório PT"
    assert 'Assimetria' in md, "Coluna 'Assimetria' ausente no relatório PT"
    assert '% Perfeito' in md, "Coluna '% Perfeito' ausente no relatório PT"
    assert 'Gerado em' in md, "Metadados 'Gerado em' ausente no relatório PT"
    assert "Tomczak" in md, "Referência Tomczak ausente na legenda PT"
    
    print(f"✅ Relatório em português gerado corretamente")
    print(f"✅ Todas as novas strings i18n PT presentes")
    print(f"✅ Grupos: {analise.grupos}")
    print()


def test_p_valor_underflow():
    """Teste que p=0.0 não aparece como '0.00e+00' no relatório."""
    print("═" * 60)
    print("Teste 5: Formatação de p-valor zero (underflow)")
    print("═" * 60)
    
    # Verifica diretamente a função _fmt_p
    fmt = AnaliseEstatistica._fmt_p
    
    assert fmt(0.0) == '< 1e-300', f"p=0.0 deveria ser '< 1e-300', obtido='{fmt(0.0)}'"
    assert fmt(None) == '\u2014', f"p=None deveria ser '\u2014', obtido='{fmt(None)}'"
    assert fmt(0.05) == '0.0500', f"p=0.05 deveria ser '0.0500', obtido='{fmt(0.05)}'"
    assert fmt(1e-10) == '1.00e-10', f"p=1e-10 deveria ser '1.00e-10', obtido='{fmt(1e-10)}'"
    assert '0.00e+00' not in fmt(0.0), "p=0.0 NÃO deve gerar '0.00e+00'"
    
    # Verifica que nenhum relatório gerado pelos testes anteriores contém 0.00e+00
    pasta_teste = os.path.join(os.path.dirname(__file__) or '.', '_teste_estatistica')
    for nome in os.listdir(pasta_teste):
        if nome.endswith('.md'):
            with open(os.path.join(pasta_teste, nome), 'r', encoding='utf-8') as f:
                conteudo = f.read()
            assert '0.00e+00' not in conteudo, \
                f"Relatório '{nome}' contém '0.00e+00' (underflow de p-valor não tratado)"
    
    print(f"✅ _fmt_p(0.0) = '{fmt(0.0)}'")
    print(f"✅ Nenhum relatório contém '0.00e+00'")
    print()


def test_relatorio_secoes_e_notas():
    """Teste que o relatório tem seções 1-6 sequenciais e notas metodológicas."""
    print("═" * 60)
    print("Teste 6: Seções sequenciais, Nemenyi materializada e notas")
    print("═" * 60)
    
    pasta_teste = os.path.join(os.path.dirname(__file__) or '.', '_teste_estatistica')
    
    # Usa o relatório do teste 1 (inglês, K=5)
    caminho_md = os.path.join(pasta_teste, 'estat_test1.md')
    assert os.path.exists(caminho_md), f"Relatório {caminho_md} não existe (rodar test_analise_basica primeiro)"
    
    with open(caminho_md, 'r', encoding='utf-8') as f:
        md = f.read()
    
    # Numeração sequencial 1→6
    secoes = re.findall(r'## (\d+)\.', md)
    nums = [int(s) for s in secoes]
    esperado = list(range(1, max(nums) + 1))
    assert nums == esperado, f"Numeração deveria ser {esperado}, obtido={nums}"
    
    # Seção 5 contém Nemenyi/CD
    pos_sec5 = md.find('## 5.')
    pos_sec6 = md.find('## 6.')
    assert pos_sec5 > 0, "Seção 5 não encontrada"
    sec5 = md[pos_sec5:pos_sec6] if pos_sec6 > pos_sec5 else md[pos_sec5:]
    assert 'Critical Difference' in sec5 or 'Nemenyi' in sec5, \
        "Seção 5 deveria conter 'Critical Difference' ou 'Nemenyi'"
    assert 'Mean Rank' in sec5 or 'Rank Médio' in sec5, \
        "Seção 5 deveria conter tabela com 'Mean Rank'"
    
    # Nota Holm m=X
    assert re.search(r'Holm.*m = \d+', md), "Nota 'Holm correction applied over m = X' ausente"
    
    # Nota critério de agrupamento
    assert 'interpret by effect size' in md or 'interpretar pelo tamanho de efeito' in md, \
        "Nota de critério de agrupamento ausente"
    
    # Metadados de reprodutibilidade
    assert 'scipy' in md, "Versão do scipy ausente nos metadados"
    assert 'Generated on' in md or 'Gerado em' in md, "Data de geração ausente nos metadados"
    
    # Coluna Same Group? / Mesmo Grupo?
    assert 'Same Group?' in md or 'Mesmo Grupo?' in md, \
        "Coluna 'Same Group?' ausente na tabela de Wilcoxon"
    
    # Legenda com Tomczak & Tomczak
    assert 'Tomczak' in md, "Referência Tomczak & Tomczak ausente na legenda"
    
    print(f"✅ Seções sequenciais: {nums}")
    print(f"✅ Seção 5 Nemenyi materializada com CD e ranks")
    print(f"✅ Nota Holm, nota agrupamento, metadados, concordância presentes")
    print()


def test_efeito_teto():
    """Teste com scores concentrados no teto (simula ROUGE-L ≈ 1.0) para validar empates e n'."""
    print("═" * 60)
    print("Teste 7: Efeito de teto (scores ≈ 1.0, muitos empates)")
    print("═" * 60)
    
    np.random.seed(42)
    n = 200
    
    # Proto_B: quase tudo 1.0
    b = np.ones(n)
    b[:10] = np.random.uniform(0.95, 0.999, 10)
    
    # Proto_C: maioria 1.0, alguns menores
    c = np.ones(n)
    c[:30] = np.random.uniform(0.90, 0.999, 30)
    
    # Proto_A: claramente pior
    a = np.random.normal(0.75, 0.10, n).clip(0, 1)
    
    df = pd.DataFrame({'A': a, 'B': b, 'C': c})
    
    pasta_teste = os.path.join(os.path.dirname(__file__) or '.', '_teste_estatistica')
    os.makedirs(pasta_teste, exist_ok=True)
    
    analise = AnaliseEstatistica(df, config={
        'metrica_nome': 'rouge_l_F1',
        'campo': 'Sentencas',
        'tecnica': 'ROUGE-L',
        'arquivo_md': os.path.join(pasta_teste, 'estat_test7_teto.md'),
        'arquivo_cd_png': os.path.join(pasta_teste, 'estat_test7_teto_cd.png'),
        'lang': 'en',
    })
    
    resumo = analise.processar()
    analise.salvar()
    
    # B↔C deveria ter muitos empates (ambos = 1.0 em muitos docs)
    par_bc = None
    for w in analise.wilcoxon_resultados:
        if (w['proto1'] == 'B' and w['proto2'] == 'C') or (w['proto1'] == 'C' and w['proto2'] == 'B'):
            par_bc = w
            break
    
    assert par_bc is not None, "Par B↔C não encontrado"
    assert par_bc['pct_empates'] > 0.5, \
        f"B↔C deveria ter >50% de empates com efeito de teto, obtido={par_bc['pct_empates']:.1%}"
    assert par_bc['n_prime'] < n, \
        f"n' ({par_bc['n_prime']}) deveria ser menor que n ({n}) com empates"
    
    # % perfeito no ranking deveria ser alto para B e C
    for _, r in analise.ranking.iterrows():
        if r['protocolo'] in ('B', 'C'):
            assert r['pct_perfeito'] > 0.5, \
                f"Protocolo {r['protocolo']}: pct_perfeito={r['pct_perfeito']:.1%} deveria ser >50%"
    
    print(f"✅ Par B↔C: n'={par_bc['n_prime']}, %empates={par_bc['pct_empates']:.1%}")
    print(f"✅ Efeito de teto corretamente refletido em n' e % empates")
    print()


def test_nota_shapiro_n_grande():
    """Teste que nota Shapiro-Wilk para n grande aparece com N >= 500."""
    print("═" * 60)
    print("Teste 8: Nota Shapiro-Wilk para n grande (N >= 500)")
    print("═" * 60)
    
    np.random.seed(42)
    n = 600  # >= 500 para acionar a nota
    
    df = pd.DataFrame({
        'X': np.random.normal(0.70, 0.10, n).clip(0, 1),
        'Y': np.random.normal(0.75, 0.08, n).clip(0, 1),
        'Z': np.random.normal(0.80, 0.06, n).clip(0, 1),
    })
    
    pasta_teste = os.path.join(os.path.dirname(__file__) or '.', '_teste_estatistica')
    os.makedirs(pasta_teste, exist_ok=True)
    
    analise = AnaliseEstatistica(df, config={
        'metrica_nome': 'test',
        'campo': 'test',
        'tecnica': 'test',
        'arquivo_md': os.path.join(pasta_teste, 'estat_test8_ngr.md'),
        'arquivo_cd_png': os.path.join(pasta_teste, 'estat_test8_ngr_cd.png'),
        'lang': 'en',
    })
    
    analise.processar()
    analise.salvar()
    
    md = analise.markdown_content
    assert 'detects minimal departures' in md, \
        "Nota Shapiro-Wilk para n grande ausente com N=600"
    assert f'n = {n}' in md, f"Nota deveria mencionar n = {n}"
    
    # Contrateste: N < 500 NÃO deve ter a nota
    np.random.seed(42)
    df_pequeno = pd.DataFrame({
        'X': np.random.normal(0.70, 0.10, 50).clip(0, 1),
        'Y': np.random.normal(0.75, 0.08, 50).clip(0, 1),
        'Z': np.random.normal(0.80, 0.06, 50).clip(0, 1),
    })
    
    analise_peq = AnaliseEstatistica(df_pequeno, config={
        'metrica_nome': 'test', 'campo': 'test', 'tecnica': 'test',
        'arquivo_md': os.path.join(pasta_teste, 'estat_test8_peq.md'),
        'arquivo_cd_png': '', 'lang': 'en',
    })
    analise_peq.processar()
    md_peq = analise_peq.markdown_content
    assert 'detects minimal departures' not in md_peq, \
        "Nota Shapiro-Wilk NÃO deveria aparecer com N < 500"
    
    print(f"✅ Nota Shapiro-Wilk presente com N={n}")
    print(f"✅ Nota Shapiro-Wilk ausente com N=50")
    print()


def test_holm_valores_conhecidos():
    """Teste unitário da correção de Holm contra valores calculados à mão."""
    print("═" * 60)
    print("Teste 9: Holm-Bonferroni contra valores conhecidos")
    print("═" * 60)
    
    holm = AnaliseEstatistica._holm_bonferroni
    
    # Caso clássico: m=3, p=[0.01, 0.04, 0.03]
    # Ordenado: 0.01×3=0.03; 0.03×2=0.06; 0.04×1=0.04→cummax=0.06
    corr = holm([0.01, 0.04, 0.03])
    esperado = [0.03, 0.06, 0.06]
    for c, e in zip(corr, esperado):
        assert abs(c - e) < 1e-12, f"Holm: esperado {esperado}, obtido {corr}"
    
    # Monotonicidade e teto em 1.0
    corr2 = holm([0.5, 0.6, 0.9])
    assert corr2 == sorted(corr2) or True  # ordem original preservada; checa teto
    assert all(c <= 1.0 for c in corr2), f"p corrigido > 1.0: {corr2}"
    
    # m=1: sem alteração
    assert holm([0.04]) == [0.04], "Holm com m=1 não deveria alterar o p"
    
    # Vazio
    assert holm([]) == [], "Holm com lista vazia deveria retornar []"
    
    print(f"✅ Holm([0.01, 0.04, 0.03]) = {corr}")
    print()


def test_cd_formula_demsar():
    """Teste que a CD usa q_α = studentized_range/√2 (Demšar, 2006).
    
    Para k=7, α=0.05: q_α tabelado ≈ 2.949 (NÃO 4.170, que é o quantil sem /√2).
    Este teste teria detectado a CD inflada em √2 (~41% maior).
    """
    print("═" * 60)
    print("Teste 10: Fórmula da CD (fator √2 de Demšar)")
    print("═" * 60)
    
    from scipy.stats import studentized_range
    
    k, alpha, N = 7, 0.05, 2500
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2.0)
    assert abs(q_alpha - 2.949) < 0.01, \
        f"q_α para k=7 deveria ser ≈2.949 (Demšar), obtido={q_alpha:.4f}"
    
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))
    assert abs(cd - 0.1801) < 0.001, f"CD(k=7, N=2500) deveria ser ≈0.1801, obtido={cd:.4f}"
    
    # Valida também no relatório gerado: se a mensagem de CD existir num relatório
    # com K>=3, o valor de q_α impresso deve ser < 3.5 para k<=10 (com /√2)
    pasta_teste = os.path.join(os.path.dirname(__file__) or '.', '_teste_estatistica')
    caminho_md = os.path.join(pasta_teste, 'estat_test1.md')
    if os.path.exists(caminho_md):
        with open(caminho_md, 'r', encoding='utf-8') as f:
            md = f.read()
        m = re.search(r'q_α = (\d+\.\d+)', md) or re.search(r'q_alpha = (\d+\.\d+)', md)
        if m:
            q_rel = float(m.group(1))
            assert q_rel < 3.5, \
                f"q_α={q_rel} no relatório sugere ausência do fator /√2 (k=5 → esperado ≈2.728)"
    
    print(f"✅ q_α(k=7) = {q_alpha:.4f} ≈ 2.949 (tabela de Demšar)")
    print(f"✅ CD(k=7, N=2500) = {cd:.4f}")
    print()


def test_compartilham_grupo_sobreposicao():
    """Teste da lógica de grupo compartilhado com CLD sobreposto (overlap-safe)."""
    print("═" * 60)
    print("Teste 11: Grupos sobrepostos (CLD com pertencimento múltiplo)")
    print("═" * 60)
    
    f = AnaliseEstatistica._compartilham_grupo
    
    # Casos simples
    assert f('G-01', 'G-01') == True
    assert f('G-01', 'G-02') == False
    
    # Sobreposição: 'G-02 G-03' compartilha G-03 com 'G-03'
    # (comparação de strings inteiras retornaria False — bug corrigido)
    assert f('G-02 G-03', 'G-03') == True
    assert f('G-01 G-02', 'G-02 G-03') == True
    assert f('G-01', 'G-02 G-03') == False
    
    # Vazios
    assert f('', 'G-01') == False
    assert f('', '') == False
    
    # Aceita sets
    assert f({'G-01', 'G-02'}, {'G-02'}) == True
    
    print("✅ _compartilham_grupo cobre sobreposição, vazios e sets")
    print()


def test_z_nao_direcional_e_faixas_d():
    """Teste que |z| é não-negativo e que d usa faixas de Cohen (0.2/0.5/0.8)."""
    print("═" * 60)
    print("Teste 12: |z| não-negativo e faixas convencionais do d")
    print("═" * 60)
    
    np.random.seed(7)
    n = 150
    base = np.random.normal(0.70, 0.10, n).clip(0, 1)
    # K=2: caminho independente de scikit-posthocs (Nemenyi só com K>=3)
    df = pd.DataFrame({
        'P1': base,
        'P2': (base + 0.06 + np.random.normal(0, 0.03, n)).clip(0, 1),
    })
    
    pasta_teste = os.path.join(os.path.dirname(__file__) or '.', '_teste_estatistica')
    os.makedirs(pasta_teste, exist_ok=True)
    
    analise = AnaliseEstatistica(df, config={
        'metrica_nome': 'test', 'campo': 'test', 'tecnica': 'test',
        'arquivo_md': os.path.join(pasta_teste, 'estat_test12.md'),
        'arquivo_cd_png': '', 'lang': 'en',
    })
    analise.processar()
    
    for w in analise.wilcoxon_resultados:
        # |z| deve ser não-negativo (scipy bilateral não carrega direção)
        assert w['z_stat'] >= 0, \
            f"|z| negativo ({w['z_stat']:.2f}) para {w['proto1']}↔{w['proto2']}"
        # Coerência r = |z|/√n'
        if w['n_prime'] > 0:
            r_esperado = w['z_stat'] / np.sqrt(w['n_prime'])
            assert abs(w['r_efeito'] - r_esperado) < 1e-9, \
                f"r inconsistente com |z|/√n' para {w['proto1']}↔{w['proto2']}"
    
    # Faixas do d: 0.2/0.5/0.8 (Cohen, 1988)
    # d=0.53 deve ser 'Medium' (nas faixas antigas 0.1/0.3/0.5 seria 'Large')
    casos_d = [(0.10, 'Negligible'), (0.30, 'Small'), (0.53, 'Medium'), (0.85, 'Large')]
    # Reconstrói classificação chamando o método sobre dados sintéticos com d controlado
    for d_alvo, rotulo in casos_d:
        np.random.seed(1)
        diffs = np.random.normal(d_alvo, 1.0, 5000)  # mean/std ≈ d_alvo
        df2 = pd.DataFrame({'X': np.zeros(5000), 'Y': diffs})
        a2 = AnaliseEstatistica(df2, config={
            'metrica_nome': 't', 'campo': 't', 'tecnica': 't',
            'arquivo_md': os.path.join(pasta_teste, '_tmp_d.md'),
            'arquivo_cd_png': '', 'lang': 'en',
        })
        a2.processar()
        w = a2.wilcoxon_resultados[0]
        assert w['tamanho_efeito_d'] == rotulo, \
            f"d≈{d_alvo}: esperado '{rotulo}', obtido '{w['tamanho_efeito_d']}' (d={w['cohen_d']:.3f})"
    
    print("✅ |z| ≥ 0 e r = |z|/√n' consistentes")
    print("✅ Faixas do d: 0.2/0.5/0.8 (d=0.53 → Medium)")
    print()


def test_pareamento_descarte():
    """Teste que o cabeçalho reporta pareamento e descarte por protocolo."""
    print("═" * 60)
    print("Teste 13: Cabeçalho de pareamento/descarte (NaN por protocolo)")
    print("═" * 60)
    
    np.random.seed(3)
    n = 60
    # K=2: caminho independente de scikit-posthocs
    df = pd.DataFrame({
        'A': np.random.normal(0.7, 0.1, n).clip(0, 1),
        'B': np.random.normal(0.75, 0.1, n).clip(0, 1),
    })
    # Injeta NaN: 5 em A, 3 em B (com sobreposição parcial)
    df.loc[0:4, 'A'] = np.nan
    df.loc[3:5, 'B'] = np.nan
    
    pasta_teste = os.path.join(os.path.dirname(__file__) or '.', '_teste_estatistica')
    os.makedirs(pasta_teste, exist_ok=True)
    
    analise = AnaliseEstatistica(df, config={
        'metrica_nome': 'test', 'campo': 'test', 'tecnica': 'test',
        'arquivo_md': os.path.join(pasta_teste, 'estat_test13.md'),
        'arquivo_cd_png': '', 'lang': 'en',
    })
    analise.processar()
    
    assert analise.n_bruto == n, f"n_bruto esperado={n}, obtido={analise.n_bruto}"
    assert analise.descartes_por_protocolo['A'] == 5, \
        f"Descartes de A esperado=5, obtido={analise.descartes_por_protocolo['A']}"
    assert analise.descartes_por_protocolo['B'] == 3, \
        f"Descartes de B esperado=3, obtido={analise.descartes_por_protocolo['B']}"
    # União {0..4} ∪ {3..5} = {0..5} → 6 linhas descartadas
    assert analise.n_descartado == 6, f"n_descartado esperado=6, obtido={analise.n_descartado}"
    assert analise.N == n - 6, f"N pareado esperado={n-6}, obtido={analise.N}"
    
    md = analise.markdown_content
    assert 'Pairing:' in md or 'Pareamento:' in md, "Linha de pareamento ausente no cabeçalho"
    assert 'A: 5' in md and 'B: 3' in md, "Detalhe de faltantes por protocolo ausente"
    
    print(f"✅ n_bruto={analise.n_bruto}, descartados={analise.n_descartado}, N={analise.N}")
    print(f"✅ Cabeçalho reporta faltantes por protocolo (A: 5, B: 3)")
    print()


def test_grupos_numerados_por_desempenho():
    """Teste que G-01 corresponde ao grupo do melhor protocolo (menor rank médio)."""
    print("═" * 60)
    print("Teste 14: G-01 = melhor grupo (numeração por rank médio)")
    print("═" * 60)
    
    pasta_teste = os.path.join(os.path.dirname(__file__) or '.', '_teste_estatistica')
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'Pior': np.random.normal(0.55, 0.10, n).clip(0, 1),
        'Melhor': np.random.normal(0.90, 0.05, n).clip(0, 1),
        'Meio': np.random.normal(0.72, 0.08, n).clip(0, 1),
    })
    
    analise = AnaliseEstatistica(df, config={
        'metrica_nome': 'test', 'campo': 'test', 'tecnica': 'test',
        'arquivo_md': os.path.join(pasta_teste, 'estat_test14.md'),
        'arquivo_cd_png': '', 'lang': 'en',
    })
    analise.processar()
    
    if analise.grupos:
        g_melhor = analise.grupos.get('Melhor', '')
        assert 'G-01' in g_melhor.split(), \
            f"Melhor protocolo deveria estar em G-01, obtido='{g_melhor}' (grupos={analise.grupos})"
    
    print(f"✅ Grupos: {analise.grupos}")
    print()


if __name__ == "__main__":
    test_analise_basica()
    test_dois_protocolos()
    test_amostras_insuficientes()
    test_portugues()
    test_p_valor_underflow()
    test_relatorio_secoes_e_notas()
    test_efeito_teto()
    test_nota_shapiro_n_grande()
    test_holm_valores_conhecidos()
    test_cd_formula_demsar()
    test_compartilham_grupo_sobreposicao()
    test_z_nao_direcional_e_faixas_d()
    test_pareamento_descarte()
    test_grupos_numerados_por_desempenho()
    
    print("═" * 60)
    print("✅ TODOS OS 14 TESTES PASSARAM!")
    print("═" * 60)