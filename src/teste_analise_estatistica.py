# Autor: Luiz Anísio
# Fonte: https://github.com/luizanisio/llms/tree/main/src

import pandas as pd
import numpy as np
import sys
import os
# Ensure we can import from the same directory whether run from root or src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from util_analise_estatistica import AnaliseEstatistica

def test_analise():
    print("Iniciando teste de validação da AnaliseEstatistica...")
    
    # Criando dados sintéticos
    # Criando dados sintéticos baseados nas imagens (Tabela III)
    np.random.seed(42)  # Seed para reprodutibilidade
    
    # Configuração dos tamanhos de amostra (Tabela II: 300 pares por família)
    n_gpt = 300
    n_gemma12 = 300
    n_gemma27 = 300
    
    # Família 1: GPT5
    # Base: Média 0.8325, Std 0.0721
    # Agente: Média 0.9271, Std 0.0394
    # Shapiro (GPT5): W=0.9183 (Não normal) -> Adicionamos leve skew se possível, mas mantendo normal para média/std corretos
    gpt_base = np.random.normal(0.8325, 0.0721, n_gpt)
    gpt_agente = np.random.normal(0.9271, 0.0394, n_gpt)
    gpt_custo_base = np.full(n_gpt, 38835) # Custo aproximado do relatório anterior
    gpt_custo_agente = np.full(n_gpt, 99247) 

    # Família 2: Gemma3(12B)
    # Base: Média 0.6247, Std 0.0966
    # Agente: Média 0.6494, Std 0.1008
    gemma12_base = np.random.normal(0.6247, 0.0966, n_gemma12)
    gemma12_agente = np.random.normal(0.6494, 0.1008, n_gemma12)
    gemma12_custo_base = np.full(n_gemma12, 38835)
    gemma12_custo_agente = np.full(n_gemma12, 153069)

    # Família 3: Gemma3(27B)
    # Base: Média 0.7084, Std 0.1043
    # Agente: Média 0.7244, Std 0.1041
    gemma27_base = np.random.normal(0.7084, 0.1043, n_gemma27)
    gemma27_agente = np.random.normal(0.7244, 0.1041, n_gemma27)
    gemma27_custo_base = np.full(n_gemma27, 38835)
    gemma27_custo_agente = np.full(n_gemma27, 105985)

    # Concatenando
    df = pd.DataFrame({
        'valor1': np.concatenate([gpt_base, gemma12_base, gemma27_base]),
        'valor2': np.concatenate([gpt_agente, gemma12_agente, gemma27_agente]),
        'custo1': np.concatenate([gpt_custo_base, gemma12_custo_base, gemma27_custo_base]),
        'custo2': np.concatenate([gpt_custo_agente, gemma12_custo_agente, gemma27_custo_agente]),
        'familia': ['agentes_gpt5'] * n_gpt + ['agentes_gemma3(12B)'] * n_gemma12 + ['agentes_gemma3(27B)'] * n_gemma27
    })
    
    # Clip para garantir valores entre 0 e 1 (analise de F1)
    df['valor1'] = df['valor1'].clip(0, 1)
    df['valor2'] = df['valor2'].clip(0, 1)
    
    output_file = 'relatorio_teste_validacao.md'
    
    analise = AnaliseEstatistica(df, config={
        'rotulo1': 'Base',
        'rotulo2': 'Agentes',
        'arquivo_saida': output_file
    })
    
    try:
        analise.processar_analise()
        analise.salvar_relatorio()
        
        if os.path.exists(output_file):
            print(f"✅ Relatório gerado com sucesso: {output_file}")
            with open(output_file, 'r') as f:
                content = f.read()
                print("\n--- Conteúdo do Relatório ---")
                print(content)
                print("-----------------------------")
                
                # Verificação simples se a tabela nova existe
                if "Desempenho Geral (Ranking)" in content:
                    print("✅ Tabela 'Desempenho Geral' encontrada.")
                else:
                    print("❌ ERRO: Tabela 'Desempenho Geral' NÃO encontrada.")
        else:
            print("❌ Erro: Arquivo de relatório não foi criado.")
            
    except Exception as e:
        print(f"❌ Erro durante a execução: {str(e)}")
        raise e

if __name__ == "__main__":
    test_analise()
