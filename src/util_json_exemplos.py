# -*- coding: utf-8 -*-

'''
Exemplos para testes da classe JsonAnalise.

Autor: Luiz Anísio
Data: Janeiro 2025
Fonte: https://github.com/luizanisio/llms/tree/main/src

Descrição:
-----------
Exemplos atualizados para a NOVA estrutura JsonAnalise (sem alinhamento)
 
 Estrutura: metrica_global, nivel_campos, campos_bertscore, campos_rouge, etc.
'''

class JsonAnaliseExemplos:
    """
    Exemplos atualizados para nova estrutura do JsonAnalise.
    
    Cada exemplo retorna: (true_json, pred_json, config, esperado)
    
    Estrutura do dicionário 'esperado':
    {
        # Métricas globais (valor exato ou [min, max])
        '(global)_F1': valor_ou_intervalo,
        '(global)_P': valor_ou_intervalo,
        '(global)_R': valor_ou_intervalo,
        '(global)_LS': valor_ou_intervalo,
        
        # Métricas estruturais
        '(estrutura)_F1': valor_ou_intervalo,
        '(estrutura)_P': valor_ou_intervalo,
        '(estrutura)_R': valor_ou_intervalo,
        '(estrutura)_LS': valor_ou_intervalo,
        
        # Métricas por campo (opcional)
        'campos': {
            'nome_campo': {
                'F1': valor_ou_intervalo,
                'P': valor_ou_intervalo,
                'R': valor_ou_intervalo,
                'LS': valor_ou_intervalo
            }
        }
    }
    """
    
    @staticmethod
    def exemplo_1_identicos():
        """JSONs idênticos - todas as métricas devem ser perfeitas"""
        json_data = {
            "nome": "João Silva",
            "idade": 30,
            "cidade": "São Paulo",
            "ativo": True
        }
        
        config = {
            'campos_rouge2': ['(global)'],
            'nivel_campos': 1
        }
        
        esperado = {
            '(global)_rouge2_F1': 1.0,
            '(global)_rouge2_P': 1.0,
            '(global)_rouge2_R': 1.0,
            '(global)_rouge2_LS': 0.0,
            '(estrutura)_rouge2_F1': 1.0,
            '(estrutura)_rouge2_P': 1.0,
            '(estrutura)_rouge2_R': 1.0,
            '(estrutura)_rouge2_LS': 0.0,
            'campos': {
                'nome': {'rouge2': {'F1': 1.0, 'P': 1.0, 'R': 1.0, 'LS': 0.0}},
                'idade': {'rouge2': {'F1': 1.0, 'P': 1.0, 'R': 1.0, 'LS': 0.0}},
                'cidade': {'rouge2': {'F1': 1.0, 'P': 1.0, 'R': 1.0, 'LS': 0.0}},
                'ativo': {'rouge2': {'F1': 1.0, 'P': 1.0, 'R': 1.0, 'LS': 0.0}}
            }
        }
        
        return json_data, json_data.copy(), config, esperado
    
    @staticmethod
    def exemplo_2_estrutura_identica_valores_diferentes():
        """Estrutura idêntica mas valores diferentes"""
        true_json = {
            "nome": "João Silva",
            "idade": 30,
            "cidade": "São Paulo"
        }
        pred_json = {
            "nome": "Maria Santos",
            "idade": 25,
            "cidade": "Rio de Janeiro"
        }
        
        config = {
            'campos_levenshtein': ['(global)'],
            'nivel_campos': 1
        }
        
        esperado = {
            # Global: Levenshtein retorna apenas SIM
            # Similaridade moderada (valores diferentes mas estrutura ajuda)
            '(global)_SIM': [0.6, 0.95],
            
            # Estrutura: perfeita (mesmos campos)
            '(estrutura)_F1': 1.0,
            '(estrutura)_P': 1.0,
            '(estrutura)_R': 1.0,
            
            # Campos não analisados em Levenshtein
            'campos': {}
        }
        
        return true_json, pred_json, config, esperado
    
    @staticmethod
    def exemplo_3_estrutura_parcial():
        """Estrutura parcialmente diferente"""
        true_json = {
            "nome": "João",
            "idade": 30,
            "cidade": "SP"
        }
        pred_json = {
            "nome": "João",
            "telefone": "11999999999"
        }
        
        config = {
            'campos_levenshtein': ['(global)'],
            'nivel_campos': 1
        }
        
        esperado = {
            # Global: Levenshtein retorna apenas SIM (parcial - campo 'nome' igual)
            '(global)_SIM': [0.3, 0.8],
            
            # Estrutura: 1 comum de 4 total (nome, idade, cidade, telefone)
            '(estrutura)_F1': [0.3, 0.7],
            '(estrutura)_P': [0.4, 0.6],  # 1/2 pred
            '(estrutura)_R': [0.2, 0.4],  # 1/3 true
            
            # Campo comum não analisado com Levenshtein
            'campos': {}
        }
        
        return true_json, pred_json, config, esperado
    
    @staticmethod
    def exemplo_4_bertscore_semantico():
        """Similaridade semântica com BERTScore"""
        true_json = {
            "descricao": "O réu foi condenado por tráfico de drogas"
        }
        pred_json = {
            "descricao": "O acusado foi sentenciado por narcotráfico"
        }
        
        config = {
            'campos_bertscore': ['(global)', 'descricao'],
            'nivel_campos': 1
        }
        
        esperado = {
            # BERTScore detecta similaridade semântica (sem LS)
            '(global)_F1': [0.6, 0.9],
            '(global)_P': [0.6, 0.9],
            '(global)_R': [0.6, 0.9],
            
            # Estrutura: perfeita (mesmo campo)
            '(estrutura)_F1': 1.0,
            '(estrutura)_P': 1.0,
            '(estrutura)_R': 1.0,
            
            'campos': {
                'descricao': {
                    'F1': [0.6, 0.9]
                }
            }
        }
        
        return true_json, pred_json, config, esperado
    
    @staticmethod
    def exemplo_5_rouge_texto_longo():
        """ROUGE para textos longos"""
        true_json = {
            "fatos": "O autor ajuizou ação de cobrança contra o réu alegando dívida não paga no valor de R$ 10.000,00"
        }
        pred_json = {
            "fatos": "Foi ajuizada ação de cobrança com alegação de dívida não paga de R$ 10.000,00"
        }
        
        config = {
            'campos_rouge': ['(global)', 'fatos'],
            'nivel_campos': 1
        }
        
        esperado = {
            # ROUGE detecta overlap de palavras
            '(global)_F1': [0.6, 0.92],  # Ajustado para 0.92
            '(global)_P': [0.6, 0.92],
            '(global)_R': [0.6, 0.92],
            
            # Estrutura: perfeita
            '(estrutura)_F1': 1.0,
            
            'campos': {
                'fatos': {
                    'F1': [0.6, 0.9]
                }
            }
        }
        
        return true_json, pred_json, config, esperado
    
    @staticmethod
    def exemplo_6_nivel_2_aninhado():
        """Campos aninhados com nível 2"""
        true_json = {
            "pessoa": {
                "nome": "João Silva",
                "endereco": {
                    "rua": "Av. Paulista",
                    "numero": 1000
                }
            },
            "valor": 5000
        }
        pred_json = {
            "pessoa": {
                "nome": "João Silva",
                "endereco": {
                    "rua": "Av. Paulista",
                    "numero": 1000
                }
            },
            "valor": 5000
        }
        
        config = {
            'campos_levenshtein': ['(global)'],
            'nivel_campos': 2  # Extrai até 2 níveis
        }
        
        esperado = {
            '(global)_SIM': 1.0,  # Levenshtein retorna apenas SIM
            '(estrutura)_F1': 1.0,
            
            'campos': {}  # Campos não analisados com Levenshtein
        }
        
        return true_json, pred_json, config, esperado
    
    @staticmethod
    def exemplo_7_mix_metricas():
        """Mix de métricas diferentes por campo"""
        true_json = {
            "resumo": "Ação de cobrança de dívida bancária",
            "legislacao": "LEG:FED LEI:008078 ANO:1990 CDC-90",
            "valor": 15000.50
        }
        pred_json = {
            "resumo": "Processo de execução de título bancário",
            "legislacao": "LEG:FED LEI:8078 ANO:1990 CDC-90",
            "valor": 15000.50
        }
        
        config = {
            'campos_levenshtein': ['(global)'],
            'campos_bertscore': ['resumo'],
            'campos_rouge': ['legislacao'],
            'nivel_campos': 1
        }
        
        esperado = {
            # Global: Levenshtein retorna apenas SIM
            '(global)_SIM': [0.5, 0.95],
            
            # Estrutura: perfeita
            '(estrutura)_F1': 1.0,
            '(estrutura)_P': 1.0,
            '(estrutura)_R': 1.0,
            
            'campos': {
                'resumo': {  # BERTScore: boa similaridade semântica
                    'F1': [0.5, 0.85]
                },
                'legislacao': {  # ROUGE: alto overlap de palavras
                    'F1': [0.7, 1.0]
                }
                # valor não é analisado (sem configuração específica)
            }
        }
        
        return true_json, pred_json, config, esperado
    
    @staticmethod
    def exemplo_8_campos_faltantes():
        """Campos ausentes em pred ou true"""
        true_json = {
            "nome": "João",
            "idade": 30,
            "cidade": "SP",
            "telefone": "11999999999"
        }
        pred_json = {
            "nome": "João",
            "idade": 30,
            "email": "joao@example.com"
        }
        
        config = {
            'campos_levenshtein': ['(global)'],
            'nivel_campos': 1
        }
        
        esperado = {
            # Levenshtein retorna apenas SIM
            # 2 comuns (nome, idade) de 5 total (nome, idade, cidade, telefone, email)
            '(global)_SIM': [0.5, 0.85],
            
            # Estrutura: 2 comuns de 5 total
            '(estrutura)_F1': [0.5, 0.7],
            '(estrutura)_P': [0.6, 0.7],  # 2/3 pred
            '(estrutura)_R': [0.4, 0.6],  # 2/4 true
            
            'campos': {}  # Campos não analisados com Levenshtein
        }
        
        return true_json, pred_json, config, esperado
    
    @staticmethod
    def exemplo_9_listas_simples():
        """Listas simples (não aninhadas)"""
        true_json = {
            "tags": ["crime", "droga", "condenação"]
        }
        pred_json = {
            "tags": ["crime", "droga", "sentença"]
        }
        
        config = {
            'campos_levenshtein': ['(global)'],
            'nivel_campos': 1
        }
        
        esperado = {
            # Levenshtein retorna apenas SIM (2 de 3 iguais)
            '(global)_SIM': [0.6, 0.97],
            
            # Estrutura: perfeita (mesmo campo)
            '(estrutura)_F1': 1.0,
            
            'campos': {}  # Campos não analisados com Levenshtein
        }
        
        return true_json, pred_json, config, esperado
    
    @staticmethod
    def exemplo_10_valores_nulos():
        """Valores None/null"""
        true_json = {
            "nome": "João",
            "observacao": None
        }
        pred_json = {
            "nome": "João",
            "observacao": None
        }
        
        config = {
            'campos_levenshtein': ['(global)'],
            'nivel_campos': 1
        }
        
        esperado = {
            '(global)_SIM': 1.0,  # Levenshtein retorna apenas SIM
            
            '(estrutura)_F1': 1.0,
            
            'campos': {}  # Campos não analisados com Levenshtein
        }
        
        return true_json, pred_json, config, esperado
    
    @staticmethod
    def lista_exemplos():
        """Retorna lista de todos os exemplos disponíveis"""
        return [
            ('exemplo_1_identicos', JsonAnaliseExemplos.exemplo_1_identicos),
            ('exemplo_2_estrutura_identica_valores_diferentes', JsonAnaliseExemplos.exemplo_2_estrutura_identica_valores_diferentes),
            ('exemplo_3_estrutura_parcial', JsonAnaliseExemplos.exemplo_3_estrutura_parcial),
            ('exemplo_4_bertscore_semantico', JsonAnaliseExemplos.exemplo_4_bertscore_semantico),
            ('exemplo_5_rouge_texto_longo', JsonAnaliseExemplos.exemplo_5_rouge_texto_longo),
            ('exemplo_6_nivel_2_aninhado', JsonAnaliseExemplos.exemplo_6_nivel_2_aninhado),
            ('exemplo_7_mix_metricas', JsonAnaliseExemplos.exemplo_7_mix_metricas),
            ('exemplo_8_campos_faltantes', JsonAnaliseExemplos.exemplo_8_campos_faltantes),
            ('exemplo_9_listas_simples', JsonAnaliseExemplos.exemplo_9_listas_simples),
            ('exemplo_10_valores_nulos', JsonAnaliseExemplos.exemplo_10_valores_nulos),
        ]
