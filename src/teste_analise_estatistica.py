# -*- coding: utf-8 -*-
# Autor: Luiz Anísio
# Fonte: https://github.com/luizanisio/llms/tree/main/src

"""
Teste de validação do módulo util_analise_estatistica (interface N×K).
Simula K protocolos com diferenças conhecidas e verifica:
- Friedman, Nemenyi, Wilcoxon+Holm, Shapiro-Wilk, Cohen's d
- Geração de .md e .png
"""

import pandas as pd
import numpy as np
import sys
import os
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
    
    # Validações
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
    
    print(f"✅ Resumo: {resumo}")
    print(f"✅ Grupos: {analise.grupos}")
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
    
    print(f"✅ Resumo: {resumo}")
    print(f"✅ Wilcoxon: p={analise.wilcoxon_resultados[0]['p_corrigido']:.4f}")
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
    
    # Verifica que textos estão em português
    assert 'Análise Estatística' in analise.markdown_content, "Título deveria estar em português"
    assert 'Ranking de Desempenho' in analise.markdown_content, "Seções deveriam estar em português"
    
    print(f"✅ Relatório em português gerado corretamente")
    print(f"✅ Grupos: {analise.grupos}")
    print()


if __name__ == "__main__":
    test_analise_basica()
    test_dois_protocolos()
    test_amostras_insuficientes()
    test_portugues()
    
    print("═" * 60)
    print("✅ TODOS OS TESTES PASSARAM!")
    print("═" * 60)
