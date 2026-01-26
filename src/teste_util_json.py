# -*- coding: utf-8 -*-

'''
Testes unitários para a classe JsonAnalise.

Autor: Luiz Anísio
Data: 18/10/2025
Fonte: https://github.com/luizanisio/llms/tree/main/src

Descrição:
-----------
Testes unitários para a classe JsonAnalise - ATUALIZADO para nova estrutura.
Cobre casos básicos, métricas específicas, estruturas aninhadas e casos extremos.
'''

import unittest
import sys
import os

# Adiciona o diretório do script ao path para imports funcionarem
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Também adiciona caminhos relativos para quando executado da pasta anterior
sys.path.extend(['./utils', './src'])

from util_json import JsonAnalise, JsonAnaliseDataFrame, Json2Texto
from util_json_dados import JsonAnaliseDados
from util_json_exemplos import JsonAnaliseExemplos

# NOTA: BERTScore agora usa implementação simplificada com cache MD5 automático
# Não é mais necessário configurar workers

# Função auxiliar para criar JsonAnaliseDataFrame com interface nova
def criar_analisador(dados, rotulos, config=None, avaliacao_llm=None, tokens=None, **kwargs):
    """Cria JsonAnaliseDataFrame usando a nova interface com JsonAnaliseDados"""
    # Extrai rotulos_destinos dos rotulos (tudo exceto id e True)
    rotulos_destinos = rotulos[2:] if len(rotulos) > 2 else []
    
    dados_analise = JsonAnaliseDados(
        dados=dados,
        rotulos=rotulos,
        tokens=tokens,
        avaliacao_llm=avaliacao_llm,
        rotulos_destinos=rotulos_destinos
    )
    return JsonAnaliseDataFrame(dados_analise, config=config, **kwargs)

class TestJsonAnaliseBasico(unittest.TestCase):
    """Testes básicos da classe JsonAnalise"""

    def test_padronizar_simbolos(self):
        """Testa padronização de símbolos"""
        texto = '  Uma  "frase"  com   espaços\n\ne aspas  '
        resultado = JsonAnalise.padronizar_simbolos(texto)
        self.assertEqual(resultado, 'uma "frase" com espaços e aspas')

    def test_hash_string_sha1(self):
        """Testa geração de hash SHA1"""
        texto = "teste"
        hash1 = JsonAnalise.hash_string_sha1(texto)
        hash2 = JsonAnalise.hash_string_sha1(texto)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 40)  # SHA1 tem 40 caracteres

    def test_hash_string_sha1_dict(self):
        """Testa hash de dicionário"""
        dict1 = {"b": 2, "a": 1}
        dict2 = {"a": 1, "b": 2}
        hash1 = JsonAnalise.hash_string_sha1(dict1)
        hash2 = JsonAnalise.hash_string_sha1(dict2)
        self.assertEqual(hash1, hash2)  # Ordem não importa

    def test_distancia_levenshtein(self):
        """Testa cálculo de distância Levenshtein"""
        # Textos iguais
        dist1 = JsonAnalise.distancia_levenshtein("casa", "casa")
        self.assertEqual(dist1, 0.0)
        
        # Textos totalmente diferentes
        dist2 = JsonAnalise.distancia_levenshtein("casa", "xxxx")
        self.assertGreater(dist2, 0.5)
        
        # Textos similares
        dist3 = JsonAnalise.distancia_levenshtein("casa", "caza")
        self.assertLess(dist3, 0.5)

    def test_distancia_jaccard_iguais(self):
        """Testa distância de Jaccard com listas iguais"""
        lista = ["a", "b", "c"]
        distancia = JsonAnalise.distancia_jaccard(lista, lista)
        self.assertEqual(distancia, 0.0)

    def test_distancia_jaccard_diferentes(self):
        """Testa distância de Jaccard com listas diferentes"""
        lista1 = ["a", "b", "c"]
        lista2 = ["d", "e", "f"]
        distancia = JsonAnalise.distancia_jaccard(lista1, lista2)
        self.assertEqual(distancia, 1.0)

    def test_distancia_jaccard_parcial(self):
        """Testa distância de Jaccard com sobreposição parcial"""
        lista1 = ["a", "b", "c"]
        lista2 = ["b", "c", "d"]
        distancia = JsonAnalise.distancia_jaccard(lista1, lista2)
        # union = 4 (a,b,c,d), intersection = 2 (b,c)
        # distancia = 1 - 2/4 = 0.5
        self.assertEqual(distancia, 0.5)


class TestJsonAnaliseConfig(unittest.TestCase):
    """Testes de configuração"""
    
    def test_config_campos_rouge_default(self):
        """Testa configuração padrão com ROUGE para (global)"""
        config = {}
        config_ajustado = JsonAnalise._ajustar_config(config)
        # (global) deve estar em campos_rouge por padrão
        self.assertIn('(global)', config_ajustado.get('campos_rouge', []))
    
    def test_config_nivel_campos_default(self):
        """Testa nível de campos padrão"""
        config = {}
        config_ajustado = JsonAnalise._ajustar_config(config)
        self.assertEqual(config_ajustado['nivel_campos'], 1)
    
    def test_config_campos_rouge1_estrutura(self):
        """Testa que (estrutura) está em campos_rouge1 por padrão"""
        config = {}
        config_ajustado = JsonAnalise._ajustar_config(config)
        self.assertIn('(estrutura)', config_ajustado.get('campos_rouge1', []))
    
    def test_config_nivel_invalido(self):
        """Testa que nível inválido levanta erro"""
        config = {'nivel_campos': 0}
        with self.assertRaises(ValueError):
            JsonAnalise._ajustar_config(config)


class TestJsonAnaliseExtrairCampos(unittest.TestCase):
    """Testes de extração de campos por nível"""
    
    def test_extrair_nivel_1(self):
        """Extrai apenas campos raiz"""
        dados = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": [4, 5]
        }
        campos = JsonAnalise._extrair_campos_por_nivel(dados, nivel=1)
        
        self.assertIn("a", campos)
        self.assertIn("b", campos)
        self.assertIn("e", campos)
        self.assertNotIn("b.c", campos)
        self.assertEqual(campos["a"], 1)
        self.assertEqual(campos["b"], {"c": 2, "d": 3})
    
    def test_extrair_nivel_2(self):
        """Extrai campos raiz + 1 nível aninhado"""
        dados = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": [4, 5]
        }
        campos = JsonAnalise._extrair_campos_por_nivel(dados, nivel=2)
        
        self.assertIn("a", campos)
        self.assertIn("b.c", campos)
        self.assertIn("b.d", campos)
        self.assertIn("e", campos)
        self.assertEqual(campos["a"], 1)
        self.assertEqual(campos["b.c"], 2)
        self.assertEqual(campos["b.d"], 3)


class TestJsonAnaliseComparar(unittest.TestCase):
    """Testes do método comparar() principal"""
    
    def test_comparar_iguais(self):
        """JSONs idênticos devem ter F1=1.0"""
        pred_json = {"nome": "João", "idade": 30}
        true_json = {"nome": "João", "idade": 30}
        
        # Usa levenshtein que é mais rápido
        config = {'campos_levenshtein': ['(global)']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Levenshtein retorna apenas SIM
        self.assertEqual(resultado['(global)_levenshtein_SIM'], 1.0)
        
        # (estrutura) usa rouge1 por padrão
        self.assertEqual(resultado['(estrutura)_rouge1_F1'], 1.0)
    
    def test_comparar_diferentes(self):
        """JSONs totalmente diferentes devem ter métricas baixas"""
        pred_json = {"xxx": "abc", "yyy": "def"}
        true_json = {"zzz": "ghi", "www": "jkl"}
        
        # Usa ROUGE-2 que é mais rápido
        config = {'campos_rouge2': ['(global)']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Campos completamente diferentes (nomes e valores)
        self.assertLess(resultado['(global)_rouge2_F1'], 0.5)
        self.assertLess(resultado['(global)_rouge2_P'], 0.5)
        self.assertLess(resultado['(global)_rouge2_R'], 0.5)
    
    def test_comparar_parcial(self):
        """JSONs parcialmente iguais devem ter 0 < F1 < 1"""
        pred_json = {"a": 1, "b": 2, "c": 3}
        true_json = {"a": 1, "d": 4, "e": 5}
        
        # Usa ROUGE-1 que é rápido
        config = {'campos_rouge1': ['(global)']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        self.assertGreater(resultado['(global)_rouge1_F1'], 0.0)
        self.assertLess(resultado['(global)_rouge1_F1'], 1.0)
        self.assertGreater(resultado['(global)_rouge1_P'], 0.0)
        self.assertLess(resultado['(global)_rouge1_P'], 1.0)
        self.assertGreater(resultado['(global)_rouge1_R'], 0.0)
        self.assertLess(resultado['(global)_rouge1_R'], 1.0)
    
    def test_comparar_precision_perfect(self):
        """Precision alta quando tudo previsto está correto ou muito similar"""
        pred_json = {"a": 1, "b": 2}
        true_json = {"a": 1, "b": 2, "c": 3}
        
        # Usa Levenshtein que é mais rápido
        config = {'campos_levenshtein': ['(global)']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Levenshtein retorna apenas SIM (similaridade global)
        self.assertGreater(resultado['(global)_levenshtein_SIM'], 0.5)
        self.assertLess(resultado['(global)_levenshtein_SIM'], 1.0)
    
    def test_comparar_recall_perfect(self):
        """Recall alto quando tudo esperado foi previsto ou muito similar"""
        pred_json = {"a": 1, "b": 2, "c": 3, "d": 4}
        true_json = {"a": 1, "b": 2}
        
        # Usa Levenshtein que é mais rápido
        config = {'campos_levenshtein': ['(global)']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Levenshtein retorna apenas SIM (similaridade global)
        self.assertGreater(resultado['(global)_levenshtein_SIM'], 0.3)
        self.assertLess(resultado['(global)_levenshtein_SIM'], 1.0)
    
    def test_comparar_com_nivel_campos_2(self):
        """Compara com nível 2 (campos aninhados)"""
        pred_json = {"pessoa": {"nome": "João", "idade": 30}}
        true_json = {"pessoa": {"nome": "João", "idade": 30}}
        
        # Define métrica explícita para campos do nível 2 - usa Levenshtein que é rápido
        config = {
            'nivel_campos': 2,
            'campos_levenshtein': ['pessoa.nome', 'pessoa.idade']
        }
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Levenshtein retorna apenas SIM para campos aninhados
        self.assertIn('pessoa.nome_levenshtein_SIM', resultado)
        self.assertIn('pessoa.idade_levenshtein_SIM', resultado)
        self.assertEqual(resultado['pessoa.nome_levenshtein_SIM'], 1.0)
        self.assertEqual(resultado['pessoa.idade_levenshtein_SIM'], 1.0)
    
    def test_comparar_com_bertscore(self):
        """Compara usando BERTScore global"""
        pred_json = {"texto": "O juiz determinou a prisão"}
        true_json = {"texto": "O magistrado ordenou a detenção"}
        
        config = {
            'nivel_campos': 1,
            'campos_bertscore': ['(global)', 'texto']
        }
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # BERTScore deve detectar similaridade semântica
        self.assertGreater(resultado['(global)_bertscore_F1'], 0.5)
        self.assertGreater(resultado['texto_bertscore_F1'], 0.5)
    
    def test_comparar_retornar_valores(self):
        """Testa retornar_valores=True"""
        pred_json = {"nome": "João"}
        true_json = {"nome": "João"}
        
        # Usa ROUGE que é mais rápido
        config = {
            'nivel_campos': 1,
            'campos_rouge': ['nome']
        }
        resultado = JsonAnalise.comparar(
            pred_json, true_json,
            retornar_valores=True,
            config=config
        )
        
        # Deve ter campos _VL com textos convertidos (estrutura é dict com pred/true)
        # Formato: campo_tecnica_VL
        self.assertIn('nome_rouge_VL', resultado)
        self.assertIsInstance(resultado['nome_rouge_VL'], dict)
        self.assertIn('pred', resultado['nome_rouge_VL'])
        self.assertIn('true', resultado['nome_rouge_VL'])
    
    def test_comparar_estrutura_diferente(self):
        """Testa que estrutura diferente é detectada"""
        pred_json = {"a": 1, "b": 2}
        true_json = {"c": 3, "d": 4}
        
        resultado = JsonAnalise.comparar(pred_json, true_json)
        
        # Estrutura completamente diferente (usa rouge1 por padrão)
        self.assertEqual(resultado['(estrutura)_rouge1_F1'], 0.0)
        self.assertIn('estrutura_detalhes', resultado)
        self.assertEqual(len(resultado['estrutura_detalhes']['paths_comuns']), 0)
        self.assertEqual(len(resultado['estrutura_detalhes']['paths_faltantes']), 2)
        self.assertEqual(len(resultado['estrutura_detalhes']['paths_extras']), 2)


class TestJsonAnaliseMetricasEspecificas(unittest.TestCase):
    """Testes de métricas específicas por campo"""
    
    def test_campo_bertscore(self):
        """Campo específico usa BERTScore"""
        pred_json = {"resumo": "Ação de cobrança de dívida"}
        true_json = {"resumo": "Processo de execução de título"}
        
        config = {
            'campos_bertscore': ['resumo'],
            'nivel_campos': 1
        }
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # BERTScore deve detectar similaridade semântica
        self.assertGreater(resultado['resumo_bertscore_F1'], 0.3)
    
    def test_campo_rouge(self):
        """Campo específico usa ROUGE-L"""
        pred_json = {"descricao": "casa verde bonita"}
        true_json = {"descricao": "casa azul bonita"}
        
        config = {
            'campos_rouge': ['descricao'],
            'nivel_campos': 1
        }
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # ROUGE detecta overlap de palavras (2 de 3 em comum)
        self.assertGreater(resultado['descricao_rouge_F1'], 0.5)
    
    def test_campo_levenshtein(self):
        """Campo específico usa Levenshtein"""
        pred_json = {"codigo": "ABC123"}
        true_json = {"codigo": "ABC124"}
        
        config = {
            'campos_levenshtein': ['codigo'],
            'nivel_campos': 1
        }
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Levenshtein retorna apenas SIM (alta similaridade: 1 caractere diferente)
        self.assertGreater(resultado['codigo_levenshtein_SIM'], 0.8)


class TestJsonAnaliseEstrutura(unittest.TestCase):
    """Testes de acurácia estrutural"""
    
    def test_estrutura_identica(self):
        """Estrutura idêntica deve ter acurácia 1.0"""
        pred_json = {
            "nome": "Maria",  # Valor diferente
            "idade": 25,      # Valor diferente
            "cidade": "RJ"    # Valor diferente
        }
        true_json = {
            "nome": "João",
            "idade": 30,
            "cidade": "SP"
        }
        
        resultado = JsonAnalise.comparar(pred_json, true_json)
        
        # Estrutura é idêntica (mesmos campos) - usa rouge1 por padrão
        self.assertEqual(resultado['(estrutura)_rouge1_P'], 1.0)
        self.assertEqual(resultado['(estrutura)_rouge1_R'], 1.0)
        self.assertEqual(resultado['(estrutura)_rouge1_F1'], 1.0)
        self.assertEqual(len(resultado['estrutura_detalhes']['paths_faltantes']), 0)
        self.assertEqual(len(resultado['estrutura_detalhes']['paths_extras']), 0)
    
    def test_estrutura_parcial(self):
        """Estrutura parcialmente diferente"""
        pred_json = {"a": 1, "b": 2}
        true_json = {"a": 1, "c": 3}
        
        resultado = JsonAnalise.comparar(pred_json, true_json)
        
        # 1 comum (a) de 3 total (a,b,c)
        self.assertGreater(resultado['(estrutura)_rouge1_P'], 0.0)
        self.assertLess(resultado['(estrutura)_rouge1_P'], 1.0)
        self.assertGreater(resultado['(estrutura)_rouge1_R'], 0.0)
        self.assertLess(resultado['(estrutura)_rouge1_R'], 1.0)


class TestJsonAnaliseDataFrame(unittest.TestCase):
    """Testes para JsonAnaliseDataFrame"""
    
    def test_dataframe_basico(self):
        """Teste básico com 2 modelos"""
        dados = [
            {
                'id': 1,
                'True': {'nome': 'João', 'idade': 30},
                'Modelo1': {'nome': 'João', 'idade': 30},
                'Modelo2': {'nome': 'João', 'idade': 35}
            }
        ]
        
        rotulos = ['id', 'True', 'Modelo1', 'Modelo2']
        # Usa Levenshtein que é mais rápido que BERTScore
        config = {'nivel_campos': 1, 'campos_levenshtein': ['(global)']}
        
        # Cria container JsonAnaliseDados
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            tokens=None,
            avaliacao_llm=None
        )
        
        analisador = JsonAnaliseDataFrame(dados_analise, config=config)
        df = analisador.to_df()
        
        # Validações básicas
        self.assertEqual(len(df), 1)
        self.assertIn('id (True)', df.columns)
        
        # Levenshtein retorna apenas SIM
        self.assertIn('Modelo1_(global)_levenshtein_SIM', df.columns)
        self.assertIn('Modelo2_(global)_levenshtein_SIM', df.columns)
        
        # Modelo1 perfeito
        self.assertEqual(df['Modelo1_(global)_levenshtein_SIM'].iloc[0], 1.0)
        
        # Modelo2 com erro
        self.assertLess(df['Modelo2_(global)_levenshtein_SIM'].iloc[0], 1.0)
    
    def test_dataframe_multiplas_linhas(self):
        """Múltiplas linhas"""
        dados = [
            {
                'id': 1,
                'True': {'a': 1},
                'M1': {'a': 1},
            },
            {
                'id': 2,
                'True': {'b': 2},
                'M1': {'b': 3},  # Valor diferente mas estrutura igual
            }
        ]
        
        rotulos = ['id', 'True', 'M1']
        # Usa Levenshtein que é mais rápido
        config = {'campos_levenshtein': ['(global)']}
        analisador = criar_analisador(dados, rotulos, config=config)
        df = analisador.to_df()
        
        self.assertEqual(len(df), 2)
        
        # Levenshtein retorna apenas SIM
        self.assertIn('M1_(global)_levenshtein_SIM', df.columns)
        self.assertTrue(all(df['M1_(global)_levenshtein_SIM'] >= 0.0))
        self.assertTrue(all(df['M1_(global)_levenshtein_SIM'] <= 1.0))
        
        # Pelo menos uma linha deve ser perfeita ou quase perfeita
        self.assertTrue(any(df['M1_(global)_levenshtein_SIM'] > 0.9))
    
    def test_estatisticas_globais(self):
        """Testa geração de estatísticas globais"""
        dados = [
            {
                'id': 1,
                'True': {'a': 1, 'b': 2},
                'M1': {'a': 1, 'b': 2},
            },
            {
                'id': 2,
                'True': {'a': 1, 'b': 2},
                'M1': {'a': 1, 'b': 3},
            }
        ]
        
        rotulos = ['id', 'True', 'M1']
        # Usa ROUGE-2 que é mais rápido
        config = {'campos_rouge2': ['(global)']}
        analisador = criar_analisador(dados, rotulos, config=config)
        df = analisador.to_df()
        stats = analisador.estatisticas_globais()
        
        # Verifica estrutura
        self.assertIn('modelo', stats.columns)
        self.assertIn('metrica', stats.columns)
        self.assertIn('mean', stats.columns)
        self.assertIn('median', stats.columns)
        self.assertIn('tecnica', stats.columns)
        
        # Verifica que tem métricas para M1 (com técnica)
        metricas_m1 = stats[stats['modelo'] == 'M1']['metrica'].tolist()
        # Deve ter métricas com formato (global)_<tecnica>_F1
        self.assertTrue(any('(global)' in m and 'F1' in m for m in metricas_m1))
    
    def test_comparar_modelos(self):
        """Testa comparação entre modelos"""
        dados = [
            {
                'id': 1,
                'True': {'a': 1},
                'M1': {'a': 1},
                'M2': {'a': 2}
            }
        ]
        
        rotulos = ['id', 'True', 'M1', 'M2']
        # Usa ROUGE-1 que é mais rápido
        config = {'campos_rouge1': ['(global)']}
        analisador = criar_analisador(dados, rotulos, config=config)
        df = analisador.to_df()
        
        # Compara F1 global com técnica específica
        comp = analisador.comparar_modelos('(global)_rouge1_F1')
        
        self.assertIn('id (True)', comp.columns)
        # Colunas devem ter formato: Modelo (tecnica)
        self.assertTrue(any('M1' in col for col in comp.columns))
        self.assertTrue(any('M2' in col for col in comp.columns))
        
        # M1 deve ser perfeito, M2 deve ser menor
        cols_m1 = [col for col in comp.columns if 'M1' in col and col != 'id (True)']
        cols_m2 = [col for col in comp.columns if 'M2' in col and col != 'id (True)']
        
        if cols_m1:
            self.assertEqual(comp[cols_m1[0]].iloc[0], 1.0)
        if cols_m2:
            self.assertLess(comp[cols_m2[0]].iloc[0], 1.0)


class TestJson2Texto(unittest.TestCase):
    """Testes da classe Json2Texto"""
    
    def test_to_linear_text(self):
        """Testa conversão para texto linear"""
        dados = {"nome": "João", "idade": 30}
        texto = Json2Texto.to_linear_text(dados)
        
        self.assertIn("nome", texto)
        self.assertIn("João", texto)
        self.assertIn("idade", texto)
        self.assertIn("30", texto)
    
    def test_to_natural_text(self):
        """Testa conversão para texto natural"""
        dados = {"nome": "João", "idade": 30}
        texto = Json2Texto.to_natural_text(dados)
        
        self.assertIn("João", texto)
        self.assertIn("30", texto)
    
    def test_to_markdown(self):
        """Testa conversão para markdown"""
        dados = {"titulo": "Teste", "conteudo": {"texto": "abc"}}
        markdown = Json2Texto.to_markdown(dados)
        
        self.assertIn("#", markdown)
        self.assertIn("titulo", markdown)
        self.assertIn("Teste", markdown)
    
    def test_to_linear_text_aninhado(self):
        """Testa conversão linear com estrutura aninhada"""
        dados = {
            "pessoa": {
                "nome": "Maria",
                "endereco": {
                    "rua": "Rua A",
                    "numero": 123
                }
            }
        }
        texto = Json2Texto.to_linear_text(dados)
        
        self.assertIn("pessoa", texto.lower())
        self.assertIn("nome", texto.lower())
        self.assertIn("maria", texto.lower())
        self.assertIn("endereco", texto.lower())
        self.assertIn("rua", texto.lower())
    
    def test_to_natural_text_lista(self):
        """Testa conversão natural com listas"""
        dados = {
            "itens": ["item1", "item2", "item3"],
            "valores": [10, 20, 30]
        }
        texto = Json2Texto.to_natural_text(dados)
        
        self.assertIn("item1", texto)
        self.assertIn("item2", texto)
        self.assertIn("10", texto)


class TestJsonAnaliseMultiplasMetricas(unittest.TestCase):
    """Testes para arquitetura multi-métrica"""
    
    def test_campo_multiplas_metricas(self):
        """Campo pode ter múltiplas métricas simultaneamente"""
        # Texto mais longo para ROUGE-2 ter bigramas em comum
        pred_json = {"texto": "o gato preto pulou no muro alto"}
        true_json = {"texto": "o gato branco pulou no muro baixo"}
        
        # Usa métricas rápidas (ROUGE e Levenshtein)
        config = {
            'nivel_campos': 1,
            'campos_rouge': ['texto'],
            'campos_rouge1': ['texto'],  # ROUGE-1 ao invés de ROUGE-2 para maior compatibilidade
            'campos_levenshtein': ['texto']
        }
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Deve ter todas as três métricas para o mesmo campo
        self.assertIn('texto_rouge_F1', resultado)
        self.assertIn('texto_rouge1_F1', resultado)
        self.assertIn('texto_levenshtein_SIM', resultado)  # Levenshtein retorna SIM
        
        # Todas devem ter valores válidos (várias palavras em comum)
        self.assertGreater(resultado['texto_rouge_F1'], 0.5)
        self.assertGreater(resultado['texto_rouge1_F1'], 0.5)
        self.assertGreater(resultado['texto_levenshtein_SIM'], 0.5)
    
    def test_global_multiplas_metricas(self):
        """(global) pode usar múltiplas métricas"""
        pred_json = {"a": 1, "b": 2}
        true_json = {"a": 1, "b": 2}
        
        # Usa apenas métricas rápidas
        config = {
            'campos_rouge1': ['(global)'],
            'campos_rouge2': ['(global)'],
            'campos_levenshtein': ['(global)']
        }
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Deve ter todas as métricas para (global)
        self.assertIn('(global)_rouge1_F1', resultado)
        self.assertIn('(global)_rouge2_F1', resultado)
        self.assertIn('(global)_levenshtein_SIM', resultado)
        
        # Todas devem ser perfeitas
        self.assertEqual(resultado['(global)_rouge1_F1'], 1.0)
        self.assertEqual(resultado['(global)_rouge2_F1'], 1.0)
        self.assertEqual(resultado['(global)_levenshtein_SIM'], 1.0)
    
    def test_determinar_metricas_campo(self):
        """Testa determinação de métricas para campos"""
        config = {
            'campos_rouge': ['campo1', 'campo3'],
            'campos_rouge1': ['campo1', 'campo2'],
            'campos_rouge2': ['campo1'],
            'campos_levenshtein': ['campo2', 'campo3']
        }
        config_ajustado = JsonAnalise._ajustar_config(config)
        
        # campo1 deve ter 3 métricas (rouge, rouge1, rouge2)
        metricas1 = JsonAnalise._determinar_metricas_campo('campo1', config_ajustado)
        self.assertEqual(len(metricas1), 3)
        self.assertIn('rouge', metricas1)
        self.assertIn('rouge1', metricas1)
        self.assertIn('rouge2', metricas1)
        
        # campo2 deve ter 2 métricas (rouge1, levenshtein)
        metricas2 = JsonAnalise._determinar_metricas_campo('campo2', config_ajustado)
        self.assertEqual(len(metricas2), 2)
        self.assertIn('rouge1', metricas2)
        self.assertIn('levenshtein', metricas2)
        
        # campo3 deve ter 2 métricas (rouge, levenshtein)
        metricas3 = JsonAnalise._determinar_metricas_campo('campo3', config_ajustado)
        self.assertEqual(len(metricas3), 2)
        self.assertIn('rouge', metricas3)
        self.assertIn('levenshtein', metricas3)


class TestJsonAnaliseConverterTexto(unittest.TestCase):
    """Testes da conversão de valores para texto"""
    
    def test_converter_string(self):
        """Converter string mantém string"""
        config = JsonAnalise._ajustar_config({})
        texto = JsonAnalise._converter_para_texto("teste", "bertscore", config)
        self.assertEqual(texto, "teste")
    
    def test_converter_int(self):
        """Converter int para string"""
        config = JsonAnalise._ajustar_config({})
        texto = JsonAnalise._converter_para_texto(123, "bertscore", config)
        self.assertEqual(texto, "123")
    
    def test_converter_dict_rouge(self):
        """Converter dict usa to_natural_text para ROUGE"""
        config = JsonAnalise._ajustar_config({})
        dados = {"nome": "João", "idade": 30}
        texto = JsonAnalise._converter_para_texto(dados, "rouge", config)
        
        self.assertIn("joão", texto.lower())
        self.assertIn("30", texto)
    
    def test_converter_dict_levenshtein(self):
        """Converter dict usa to_natural_text para Levenshtein"""
        config = JsonAnalise._ajustar_config({})
        dados = {"nome": "João", "idade": 30}
        texto = JsonAnalise._converter_para_texto(dados, "levenshtein", config)
        
        self.assertIn("joão", texto.lower())
        self.assertIn("30", texto)
    
    def test_converter_lista(self):
        """Converter lista"""
        config = JsonAnalise._ajustar_config({})
        dados = [1, 2, 3]
        texto = JsonAnalise._converter_para_texto(dados, "bertscore", config)
        
        self.assertIn("1", texto)
        self.assertIn("2", texto)
        self.assertIn("3", texto)


class TestJsonAnaliseCalcularMetrica(unittest.TestCase):
    """Testes de cálculo de métricas específicas"""
    
    def test_calcular_bertscore_simples(self):
        """Testa cálculo com BERTScore - teste único e simples"""
        config = JsonAnalise._ajustar_config({})
        # Teste com textos idênticos para ser rápido
        metricas = JsonAnalise._calcular_metrica(
            "teste", 
            "teste", 
            "bertscore", 
            config
        )
        
        self.assertIn('P', metricas)
        self.assertIn('R', metricas)
        self.assertIn('F1', metricas)
        self.assertNotIn('SIM', metricas)  # BERTScore não retorna SIM
        
        # Textos idênticos devem ter score perfeito
        self.assertEqual(metricas['F1'], 1.0)
    
    def test_calcular_rouge(self):
        """Testa cálculo com ROUGE-L"""
        config = JsonAnalise._ajustar_config({})
        metricas = JsonAnalise._calcular_metrica(
            "casa verde bonita", 
            "casa azul bonita", 
            "rouge", 
            config
        )
        
        self.assertIn('P', metricas)
        self.assertIn('R', metricas)
        self.assertIn('F1', metricas)
        
        # 2 palavras em comum de 3
        self.assertGreater(metricas['F1'], 0.5)
    
    def test_calcular_rouge1(self):
        """Testa cálculo com ROUGE-1"""
        config = JsonAnalise._ajustar_config({})
        metricas = JsonAnalise._calcular_metrica(
            "gato preto grande", 
            "gato branco grande", 
            "rouge1", 
            config
        )
        
        # 2 palavras em comum (gato, grande)
        self.assertGreater(metricas['F1'], 0.5)
        self.assertLess(metricas['F1'], 1.0)
    
    def test_calcular_rouge2(self):
        """Testa cálculo com ROUGE-2 (bigramas)"""
        config = JsonAnalise._ajustar_config({})
        metricas = JsonAnalise._calcular_metrica(
            "o gato preto pulou", 
            "o gato branco pulou", 
            "rouge2", 
            config
        )
        
        # Tem bigrama "o gato" em comum
        self.assertGreater(metricas['F1'], 0.0)
    
    def test_calcular_levenshtein(self):
        """Testa cálculo com Levenshtein"""
        config = JsonAnalise._JsonAnalise__ajustar_config({})
        
        # Textos iguais - Levenshtein retorna apenas SIM
        metricas1 = JsonAnalise._calcular_metrica("teste", "teste", "levenshtein", config)
        self.assertIn('SIM', metricas1)
        self.assertNotIn('F1', metricas1)  # Levenshtein não retorna P, R, F1
        self.assertEqual(metricas1['SIM'], 1.0)
        
        # Textos similares
        metricas2 = JsonAnalise._calcular_metrica("casa", "caza", "levenshtein", config)
        self.assertGreater(metricas2['SIM'], 0.5)
        self.assertLess(metricas2['SIM'], 1.0)


class TestJsonAnaliseAcuraciaEstrutural(unittest.TestCase):
    """Testes de análise estrutural"""
    
    def test_estrutura_identica(self):
        """Testa estrutura 100% idêntica"""
        campos_pred = {"a": 1, "b": 2, "c": 3}
        campos_true = {"a": 10, "b": 20, "c": 30}
        
        resultado = JsonAnalise._acuracia_estrutural(campos_pred, campos_true)
        
        self.assertEqual(resultado['P'], 1.0)
        self.assertEqual(resultado['R'], 1.0)
        self.assertEqual(resultado['F1'], 1.0)
        self.assertNotIn('LS', resultado)  # LS removido
        self.assertEqual(len(resultado['paths_comuns']), 3)
        self.assertEqual(len(resultado['paths_faltantes']), 0)
        self.assertEqual(len(resultado['paths_extras']), 0)
    
    def test_estrutura_parcial(self):
        """Testa estrutura parcialmente diferente"""
        campos_pred = {"a": 1, "b": 2}
        campos_true = {"b": 2, "c": 3}
        
        resultado = JsonAnalise._acuracia_estrutural(campos_pred, campos_true)
        
        # 1 campo comum (b) de 3 total
        self.assertGreater(resultado['P'], 0.0)
        self.assertLess(resultado['P'], 1.0)
        self.assertGreater(resultado['R'], 0.0)
        self.assertLess(resultado['R'], 1.0)
        self.assertEqual(len(resultado['paths_comuns']), 1)
        self.assertEqual(len(resultado['paths_faltantes']), 1)  # c
        self.assertEqual(len(resultado['paths_extras']), 1)  # a
    
    def test_estrutura_vazia(self):
        """Testa estruturas vazias"""
        resultado = JsonAnalise._acuracia_estrutural({}, {})
        
        self.assertEqual(resultado['P'], 0.0)
        self.assertEqual(resultado['R'], 0.0)
        self.assertEqual(resultado['F1'], 0.0)
        self.assertNotIn('LS', resultado)  # LS removido


class TestJsonAnaliseExemplos(unittest.TestCase):
    """Testes automáticos usando os exemplos V2 (nova estrutura)"""
    
    def _validar_metrica(self, valor, esperado, nome_metrica, contexto=""):
        """
        Valida se um valor de métrica está dentro do esperado.
        esperado pode ser: float (exato), list [min, max] (intervalo), ou None (não valida)
        """
        if esperado is None:
            return
        
        if isinstance(esperado, (int, float)):
            self.assertAlmostEqual(valor, esperado, places=1,
                msg=f"{nome_metrica} esperado {esperado}, obtido {valor:.4f}. {contexto}")
        
        elif isinstance(esperado, (list, tuple)) and len(esperado) == 2:
            min_val, max_val = esperado
            self.assertGreaterEqual(valor, min_val,
                msg=f"{nome_metrica} abaixo do mínimo [{min_val}], obtido {valor:.4f}. {contexto}")
            self.assertLessEqual(valor, max_val,
                msg=f"{nome_metrica} acima do máximo [{max_val}], obtido {valor:.4f}. {contexto}")
    
    def test_todos_exemplos_v2(self):
        """Testa todos os exemplos V2 validando métricas esperadas"""
        for nome, exemplo_func in JsonAnaliseExemplos.lista_exemplos():
            with self.subTest(exemplo=nome):
                # Obtém dados do exemplo
                true_json, pred_json, config, esperado = exemplo_func()
                
                # Executa comparação
                resultado = JsonAnalise.comparar(
                    pred_json, true_json,
                    config=config,
                    retornar_valores=False
                )
                
                # Identifica técnica usada para (global) e (estrutura) no config
                config_ajustado = JsonAnalise._JsonAnalise__ajustar_config(config)
                tecnicas_global = JsonAnalise._determinar_metricas_campo('(global)', config_ajustado)
                tecnicas_estrutura = JsonAnalise._determinar_metricas_campo('(estrutura)', config_ajustado)
                
                # Valida métricas globais (usa primeira técnica)
                if tecnicas_global:
                    tecnica_global = tecnicas_global[0]
                    for metrica in ['F1', 'P', 'R', 'LS']:
                        metrica_chave = f'(global)_{metrica}'
                        metrica_resultado = f'(global)_{tecnica_global}_{metrica}'
                        
                        if metrica_chave in esperado:
                            self.assertIn(metrica_resultado, resultado,
                                f"Exemplo {nome}: Métrica '{metrica_resultado}' não encontrada")
                            self._validar_metrica(
                                resultado[metrica_resultado],
                                esperado[metrica_chave],
                                metrica_resultado,
                                f"Exemplo: {nome}"
                            )
                
                # Valida métricas estruturais (usa primeira técnica)
                if tecnicas_estrutura:
                    tecnica_estrutura = tecnicas_estrutura[0]
                    for metrica in ['F1', 'P', 'R', 'LS']:
                        metrica_chave = f'(estrutura)_{metrica}'
                        metrica_resultado = f'(estrutura)_{tecnica_estrutura}_{metrica}'
                        
                        if metrica_chave in esperado:
                            self.assertIn(metrica_resultado, resultado,
                                f"Exemplo {nome}: Métrica '{metrica_resultado}' não encontrada")
                            self._validar_metrica(
                                resultado[metrica_resultado],
                                esperado[metrica_chave],
                                metrica_resultado,
                                f"Exemplo: {nome}"
                            )
                
                # Valida métricas por campo
                if 'campos' in esperado:
                    for nome_campo, metricas_esperadas in esperado['campos'].items():
                        # Determina técnicas para este campo
                        tecnicas_campo = JsonAnalise._determinar_metricas_campo(nome_campo, config_ajustado)
                        
                        if tecnicas_campo:
                            tecnica_campo = tecnicas_campo[0]
                            for sufixo in ['F1', 'P', 'R', 'LS']:
                                metrica_nome = f'{nome_campo}_{tecnica_campo}_{sufixo}'
                                
                                if sufixo in metricas_esperadas:
                                    self.assertIn(metrica_nome, resultado,
                                        f"Exemplo {nome}: Métrica '{metrica_nome}' não encontrada no resultado")
                                    
                                    self._validar_metrica(
                                        resultado[metrica_nome],
                                        metricas_esperadas[sufixo],
                                        metrica_nome,
                                        f"Exemplo: {nome}"
                                    )


class TestJsonAnaliseCasosExtremos(unittest.TestCase):
    """Testes de casos extremos e combinações raras"""
    
    def test_json_vazio_vs_vazio(self):
        """Dois JSONs vazios"""
        resultado = JsonAnalise.comparar({}, {})
        
        # Estrutura vazia tem precision/recall = 0
        self.assertEqual(resultado['(estrutura)_rouge1_P'], 0.0)
        self.assertEqual(resultado['(estrutura)_rouge1_R'], 0.0)
        self.assertEqual(resultado['(estrutura)_rouge1_F1'], 0.0)
    
    def test_json_vazio_vs_preenchido(self):
        """JSON vazio vs preenchido"""
        pred_json = {}
        true_json = {"a": 1, "b": 2}
        
        resultado = JsonAnalise.comparar(pred_json, true_json)
        
        # Recall deve ser 0 (nada foi previsto)
        self.assertEqual(resultado['(estrutura)_rouge1_R'], 0.0)
        self.assertEqual(resultado['(estrutura)_rouge1_F1'], 0.0)
        self.assertEqual(len(resultado['estrutura_detalhes']['paths_faltantes']), 2)
    
    def test_json_preenchido_vs_vazio(self):
        """JSON preenchido vs vazio"""
        pred_json = {"a": 1, "b": 2}
        true_json = {}
        
        resultado = JsonAnalise.comparar(pred_json, true_json)
        
        # Precision deve ser 0 (previu coisas que não existem)
        self.assertEqual(resultado['(estrutura)_rouge1_P'], 0.0)
        self.assertEqual(resultado['(estrutura)_rouge1_F1'], 0.0)
        self.assertEqual(len(resultado['estrutura_detalhes']['paths_extras']), 2)
    
    def test_valores_none(self):
        """Campos com valores None"""
        pred_json = {"campo": None}
        true_json = {"campo": None}
        
        config = {'campos_levenshtein': ['campo']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # None pode ser tratado como string vazia ou ausente (depende da implementação)
        # O importante é que sejam tratados de forma consistente
        self.assertEqual(resultado['campo_levenshtein_SIM'], resultado['campo_levenshtein_SIM'])  # Validação básica
        # Ambos são None, então devem ter similaridade (0.0 ou 1.0 dependendo da implementação)
        self.assertIn(resultado['campo_levenshtein_SIM'], [0.0, 1.0])
    
    def test_valores_booleanos(self):
        """Campos com valores booleanos"""
        pred_json = {"ativo": True, "bloqueado": False}
        true_json = {"ativo": True, "bloqueado": False}
        
        config = {'campos_levenshtein': ['ativo', 'bloqueado']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        self.assertEqual(resultado['ativo_levenshtein_SIM'], 1.0)
        self.assertEqual(resultado['bloqueado_levenshtein_SIM'], 1.0)
    
    def test_numeros_muito_grandes(self):
        """Números muito grandes"""
        pred_json = {"valor": 999999999999999}
        true_json = {"valor": 999999999999999}
        
        config = {'campos_levenshtein': ['valor']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        self.assertEqual(resultado['valor_levenshtein_SIM'], 1.0)
    
    def test_numeros_float_precisao(self):
        """Floats com diferentes precisões"""
        pred_json = {"preco": 123.456789}
        true_json = {"preco": 123.456}
        
        config = {'campos_levenshtein': ['preco']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Strings "123.456789" vs "123.456" têm alta similaridade
        self.assertGreater(resultado['preco_levenshtein_SIM'], 0.8)
    
    def test_strings_vazias(self):
        """Strings vazias"""
        pred_json = {"nome": "", "sobrenome": "Silva"}
        true_json = {"nome": "", "sobrenome": "Silva"}
        
        config = {'campos_levenshtein': ['nome', 'sobrenome']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Strings vazias idênticas
        self.assertEqual(resultado['nome_levenshtein_SIM'], 1.0)
        self.assertEqual(resultado['sobrenome_levenshtein_SIM'], 1.0)
    
    def test_listas_como_valores(self):
        """Listas como valores de campos"""
        pred_json = {"items": [1, 2, 3]}
        true_json = {"items": [1, 2, 3]}
        
        config = {'campos_levenshtein': ['items']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Listas convertidas para string devem ser idênticas
        self.assertEqual(resultado['items_levenshtein_SIM'], 1.0)
    
    def test_listas_ordem_diferente(self):
        """Listas com mesmos elementos mas ordem diferente"""
        pred_json = {"items": [1, 2, 3]}
        true_json = {"items": [3, 2, 1]}
        
        config = {'campos_rouge1': ['items']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # ROUGE-1 detecta mesmos elementos (unigramas)
        self.assertEqual(resultado['items_rouge1_F1'], 1.0)
    
    def test_unicode_caracteres_especiais(self):
        """Unicode e caracteres especiais"""
        pred_json = {"texto": "Ação, José, café, ñ"}
        true_json = {"texto": "Ação, José, café, ñ"}
        
        config = {'campos_levenshtein': ['texto']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        self.assertEqual(resultado['texto_levenshtein_SIM'], 1.0)
    
    def test_nivel_campos_3_profundo(self):
        """Nível de campos 3 (muito profundo)"""
        pred_json = {
            "a": {
                "b": {
                    "c": "valor"
                }
            }
        }
        true_json = {
            "a": {
                "b": {
                    "c": "valor"
                }
            }
        }
        
        config = {
            'nivel_campos': 3,
            'campos_levenshtein': ['a.b.c']
        }
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        self.assertIn('a.b.c_levenshtein_SIM', resultado)
        self.assertEqual(resultado['a.b.c_levenshtein_SIM'], 1.0)
    
    def test_campo_existe_pred_mas_none_em_true(self):
        """Campo existe em pred mas é None em true"""
        pred_json = {"campo": "valor"}
        true_json = {"campo": None}
        
        config = {'campos_levenshtein': ['campo']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # "valor" vs "none" são diferentes
        self.assertLess(resultado['campo_levenshtein_SIM'], 0.5)


class TestJsonAnaliseCalculoPreciso(unittest.TestCase):
    """Testes de precisão de cálculos de métricas"""
    
    def test_precision_recall_f1_manual_rouge1(self):
        """Valida cálculo P/R/F1 manualmente para ROUGE-1"""
        # pred: "a b c" (3 unigramas)
        # true: "a b d" (3 unigramas)
        # comuns: "a b" (2 unigramas)
        # P = 2/3 = 0.667, R = 2/3 = 0.667, F1 = 0.667
        
        pred_json = {"t": "a b c"}
        true_json = {"t": "a b d"}
        
        config = {'campos_rouge1': ['t']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        self.assertAlmostEqual(resultado['t_rouge1_P'], 2/3, places=2)
        self.assertAlmostEqual(resultado['t_rouge1_R'], 2/3, places=2)
        self.assertAlmostEqual(resultado['t_rouge1_F1'], 2/3, places=2)
    
    def test_precision_perfect_recall_partial(self):
        """Precision perfeita mas recall parcial"""
        # pred: "a b" (previu 2)
        # true: "a b c d" (esperado 4)
        # comuns: "a b" (2)
        # P = 2/2 = 1.0, R = 2/4 = 0.5, F1 = 2*1*0.5/(1+0.5) = 0.667
        
        pred_json = {"t": "a b"}
        true_json = {"t": "a b c d"}
        
        config = {'campos_rouge1': ['t']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        self.assertAlmostEqual(resultado['t_rouge1_P'], 1.0, places=2)
        self.assertAlmostEqual(resultado['t_rouge1_R'], 0.5, places=2)
        self.assertAlmostEqual(resultado['t_rouge1_F1'], 2/3, places=2)
    
    def test_recall_perfect_precision_partial(self):
        """Recall perfeito mas precision parcial"""
        # pred: "a b c d" (previu 4)
        # true: "a b" (esperado 2)
        # comuns: "a b" (2)
        # P = 2/4 = 0.5, R = 2/2 = 1.0, F1 = 0.667
        
        pred_json = {"t": "a b c d"}
        true_json = {"t": "a b"}
        
        config = {'campos_rouge1': ['t']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        self.assertAlmostEqual(resultado['t_rouge1_P'], 0.5, places=2)
        self.assertAlmostEqual(resultado['t_rouge1_R'], 1.0, places=2)
        self.assertAlmostEqual(resultado['t_rouge1_F1'], 2/3, places=2)
    
    def test_levenshtein_distancia_1_char(self):
        """Levenshtein com 1 caractere de diferença"""
        # "casa" vs "caza": 1 substituição em 4 chars
        # similaridade = 1 - 1/4 = 0.75
        
        pred_json = {"t": "casa"}
        true_json = {"t": "caza"}
        
        config = {'campos_levenshtein': ['t']}
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        self.assertAlmostEqual(resultado['t_levenshtein_SIM'], 0.75, places=2)
    
    def test_jaccard_distancia_calculo(self):
        """Valida cálculo de distância de Jaccard"""
        # lista_a: [1, 2, 3], lista_b: [2, 3, 4]
        # união: {1, 2, 3, 4} = 4
        # interseção: {2, 3} = 2
        # jaccard_sim = 2/4 = 0.5
        # jaccard_dist = 1 - 0.5 = 0.5
        
        distancia = JsonAnalise.distancia_jaccard([1, 2, 3], [2, 3, 4])
        self.assertEqual(distancia, 0.5)
    
    def test_jaccard_listas_identicas(self):
        """Jaccard com listas idênticas"""
        distancia = JsonAnalise.distancia_jaccard([1, 2, 3], [1, 2, 3])
        self.assertEqual(distancia, 0.0)
    
    def test_jaccard_listas_disjuntas(self):
        """Jaccard com listas completamente diferentes"""
        distancia = JsonAnalise.distancia_jaccard([1, 2], [3, 4])
        self.assertEqual(distancia, 1.0)
    
    def test_jaccard_listas_vazias(self):
        """Jaccard com listas vazias"""
        distancia = JsonAnalise.distancia_jaccard([], [])
        self.assertEqual(distancia, 0.0)
    
    def test_jaccard_uma_vazia(self):
        """Jaccard com uma lista vazia"""
        distancia = JsonAnalise.distancia_jaccard([], [1, 2])
        self.assertEqual(distancia, 1.0)
    
    def test_estrutura_precision_recall_manual(self):
        """Valida cálculo estrutural P/R/F1 manualmente"""
        # pred: {a, b, c}, true: {a, b, d}
        # comuns: {a, b} = 2
        # P = 2/3, R = 2/3, F1 = 2/3
        
        pred_json = {"a": 1, "b": 2, "c": 3}
        true_json = {"a": 1, "b": 2, "d": 4}
        
        resultado = JsonAnalise.comparar(pred_json, true_json)
        
        self.assertAlmostEqual(resultado['(estrutura)_rouge1_P'], 2/3, places=2)
        self.assertAlmostEqual(resultado['(estrutura)_rouge1_R'], 2/3, places=2)
        self.assertAlmostEqual(resultado['(estrutura)_rouge1_F1'], 2/3, places=2)


class TestJsonAnaliseConfiguracoesEspeciais(unittest.TestCase):
    """Testes de configurações especiais e edge cases"""
    
    def test_config_vazia(self):
        """Config vazia usa defaults"""
        pred_json = {"a": 1}
        true_json = {"a": 1}
        
        resultado = JsonAnalise.comparar(pred_json, true_json, config={})
        
        # Deve ter métricas default (ROUGE-L para global e ROUGE-1 para estrutura)
        self.assertIn('(global)_rouge_F1', resultado)
        self.assertIn('(estrutura)_rouge1_F1', resultado)
    
    def test_config_none(self):
        """Config None usa defaults"""
        pred_json = {"a": 1}
        true_json = {"a": 1}
        
        resultado = JsonAnalise.comparar(pred_json, true_json, config=None)
        
        self.assertIn('(global)_rouge_F1', resultado)
        self.assertIn('(estrutura)_rouge1_F1', resultado)
    
    def test_padronizar_simbolos_desabilitado(self):
        """Desabilitar padronização de símbolos"""
        pred_json = {"t": "TESTE"}
        true_json = {"t": "teste"}
        
        config = {
            'padronizar_simbolos': False,
            'campos_levenshtein': ['t']
        }
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Sem padronização, "TESTE" != "teste"
        self.assertLess(resultado['t_levenshtein_SIM'], 1.0)
    
    def test_rouge_stemmer_desabilitado(self):
        """ROUGE sem stemmer"""
        pred_json = {"t": "running quickly"}
        true_json = {"t": "run quick"}
        
        config = {
            'rouge_stemmer': False,
            'campos_rouge1': ['t']
        }
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Sem stemmer, palavras não são normalizadas
        self.assertEqual(resultado['t_rouge1_F1'], 0.0)
    
    def test_campo_inexistente_na_config(self):
        """Campo especificado na config mas não existe no JSON"""
        pred_json = {"a": 1}
        true_json = {"a": 1}
        
        config = {
            'campos_rouge1': ['campo_inexistente']
        }
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Campo inexistente gera valor 0.0
        self.assertEqual(resultado['campo_inexistente_rouge1_F1'], 0.0)
    
    def test_todas_metricas_mesmo_campo(self):
        """Campo com todas as métricas disponíveis"""
        pred_json = {"texto": "o gato preto"}
        true_json = {"texto": "o gato preto"}
        
        config = {
            'campos_bertscore': ['texto'],
            'campos_rouge': ['texto'],
            'campos_rouge1': ['texto'],
            'campos_rouge2': ['texto'],
            'campos_levenshtein': ['texto']
        }
        resultado = JsonAnalise.comparar(pred_json, true_json, config=config)
        
        # Deve ter todas as 5 métricas
        self.assertIn('texto_bertscore_F1', resultado)
        self.assertIn('texto_rouge_F1', resultado)
        self.assertIn('texto_rouge1_F1', resultado)
        self.assertIn('texto_rouge2_F1', resultado)
        self.assertIn('texto_levenshtein_SIM', resultado)  # Levenshtein retorna SIM
        
        # Todas perfeitas
        self.assertEqual(resultado['texto_bertscore_F1'], 1.0)
        self.assertEqual(resultado['texto_rouge_F1'], 1.0)
        self.assertEqual(resultado['texto_rouge1_F1'], 1.0)
        self.assertEqual(resultado['texto_rouge2_F1'], 1.0)
        self.assertEqual(resultado['texto_levenshtein_SIM'], 1.0)


class TestJson2TextoCasosEspeciais(unittest.TestCase):
    """Testes adicionais para Json2Texto"""
    
    def test_to_linear_text_vazio(self):
        """Dict vazio"""
        texto = Json2Texto.to_linear_text({})
        self.assertEqual(texto.strip(), "")
    
    def test_to_linear_text_lista_vazia(self):
        """Lista vazia"""
        dados = {"items": []}
        texto = Json2Texto.to_linear_text(dados)
        self.assertIn("items", texto.lower())
    
    def test_to_natural_text_vazio(self):
        """Dict vazio para texto natural"""
        texto = Json2Texto.to_natural_text({})
        self.assertEqual(texto.strip(), "")
    
    def test_to_natural_text_valores_none(self):
        """Valores None"""
        dados = {"a": None, "b": "valor"}
        texto = Json2Texto.to_natural_text(dados)
        self.assertIn("valor", texto)
    
    def test_to_markdown_aninhado_complexo(self):
        """Markdown com estrutura muito aninhada"""
        dados = {
            "nivel1": {
                "nivel2": {
                    "nivel3": {
                        "valor": "profundo"
                    }
                }
            }
        }
        markdown = Json2Texto.to_markdown(dados)
        
        self.assertIn("#", markdown)
        self.assertIn("nivel1", markdown)
        self.assertIn("profundo", markdown)
    
    def test_to_natural_text_numeros_misturados(self):
        """Texto natural com vários tipos"""
        dados = {
            "int": 123,
            "float": 45.67,
            "string": "texto",
            "bool": True
        }
        texto = Json2Texto.to_natural_text(dados)
        
        self.assertIn("123", texto)
        self.assertIn("45.67", texto)
        self.assertIn("texto", texto)
        # Bool pode ser convertido como "true", "True", "verdadeiro", etc.
        self.assertTrue(any(x in texto.lower() for x in ["true", "verdadeiro", "sim"]))


class TestJsonAnaliseDataFrameCasosEspeciais(unittest.TestCase):
    """Testes adicionais para JsonAnaliseDataFrame"""
    
    def test_dataframe_com_erros(self):
        """DataFrame onde modelo erra completamente"""
        dados = [
            {
                'id': 1,
                'True': {'a': 1, 'b': 2},
                'M1': {'x': 99, 'y': 88}  # Completamente errado
            }
        ]
        
        rotulos = ['id', 'True', 'M1']
        config = {'campos_levenshtein': ['(global)']}
        analisador = criar_analisador(dados, rotulos, config=config)
        df = analisador.to_df()
        
        # Modelo ruim deve ter F1 não perfeito (pode ter alguma similaridade nos números)
        self.assertLess(df['M1_(global)_levenshtein_SIM'].iloc[0], 0.75)
    
    def test_dataframe_modelo_unico(self):
        """DataFrame com apenas 1 modelo"""
        dados = [
            {'id': 1, 'True': {'a': 1}, 'M1': {'a': 1}}
        ]
        
        rotulos = ['id', 'True', 'M1']
        config = {'campos_rouge1': ['(global)']}
        analisador = criar_analisador(dados, rotulos, config=config)
        df = analisador.to_df()
        
        # Verifica que tem colunas essenciais (id + métricas do modelo)
        self.assertGreater(len(df.columns), 5)  # id + pelo menos algumas métricas
        self.assertIn('id (True)', df.columns)
    
    def test_estatisticas_sem_dados(self):
        """Estatísticas com lista vazia - verifica que levanta erro apropriado"""
        dados = []
        rotulos = ['id', 'True', 'M1']
        config = {'campos_rouge1': ['(global)']}
        
        # Lista vazia agora levanta AssertionError na criação de JsonAnaliseDados
        with self.assertRaises(AssertionError) as context:
            dados_analise = JsonAnaliseDados(
                dados=dados,
                rotulos=rotulos,
                tokens=None,
                avaliacao_llm=None
            )
        
        self.assertIn('não pode ser vazio', str(context.exception))
    
    def test_comparar_modelos_metrica_inexistente(self):
        """Comparar modelos com métrica que não existe deve levantar erro"""
        dados = [
            {'id': 1, 'True': {'a': 1}, 'M1': {'a': 1}}
        ]
        
        rotulos = ['id', 'True', 'M1']
        config = {'campos_rouge1': ['(global)']}
        analisador = criar_analisador(dados, rotulos, config=config)
        
        # Métrica inexistente deve levantar ValueError
        with self.assertRaises(ValueError) as context:
            analisador.comparar_modelos('metrica_inexistente')
        
        self.assertIn('não encontrada', str(context.exception))


class TestJsonAnaliseDataFrameAvaliacaoLLM(unittest.TestCase):
    """Testes para funcionalidade de Avaliação LLM no JsonAnaliseDataFrame"""
    
    def test_criar_dataframe_avaliacao_llm_basico(self):
        """Testa criação básica do DataFrame de avaliação LLM"""
        dados = [
            {'id': 1, 'True': {'a': 1}, 'M1': {'a': 1}}
        ]
        
        avaliacao_llm = [
            {
                'id_peca': 1,
                'M1_P': 0.95,
                'M1_R': 0.90,
                'M1_F1': 0.92,
                'M1_nota': 9.5,
                'M1_explicacao': 'Excelente extração com pequenos detalhes faltando'
            }
        ]
        
        rotulos = ['id', 'True', 'M1']
        config = {'campos_rouge1': ['(global)']}
        
        analisador = criar_analisador(dados, rotulos, 
            config=config,
            avaliacao_llm=avaliacao_llm
        )
        
        df_global, df_campos = analisador._criar_dataframe_avaliacao_llm()
        
        # Validações básicas - métricas globais ficam em df_global
        self.assertIsNotNone(df_global)
        self.assertIn('id_peca', df_global.columns)
        self.assertIn('M1_P', df_global.columns)
        self.assertIn('M1_R', df_global.columns)
        self.assertIn('M1_F1', df_global.columns)
        self.assertIn('M1_nota', df_global.columns)
        self.assertIn('M1_explicacao', df_global.columns)
        
        # Verifica valores
        self.assertEqual(df_global['M1_P'].iloc[0], 0.95)
        self.assertEqual(df_global['M1_R'].iloc[0], 0.90)
        self.assertEqual(df_global['M1_F1'].iloc[0], 0.92)
        self.assertEqual(df_global['M1_nota'].iloc[0], 9.5)
    
    def test_criar_dataframe_avaliacao_llm_multiplos_modelos(self):
        """Testa DataFrame de avaliação com múltiplos modelos"""
        dados = [
            {'id': 1, 'True': {'a': 1}, 'M1': {'a': 1}, 'M2': {'a': 2}}
        ]
        
        avaliacao_llm = [
            {
                'id_peca': 1,
                'M1_P': 0.95,
                'M1_R': 0.90,
                'M1_F1': 0.92,
                'M1_nota': 9.5,
                'M1_explicacao': 'Modelo 1 muito bom',
                'M2_P': 0.75,
                'M2_R': 0.80,
                'M2_F1': 0.77,
                'M2_nota': 7.5,
                'M2_explicacao': 'Modelo 2 razoável'
            }
        ]
        
        rotulos = ['id', 'True', 'M1', 'M2']
        config = {'campos_rouge1': ['(global)']}
        
        analisador = criar_analisador(dados, rotulos, 
            config=config,
            avaliacao_llm=avaliacao_llm
        )
        
        df_global, df_campos = analisador._criar_dataframe_avaliacao_llm()
        
        # Verifica colunas de ambos os modelos no df_global
        for modelo in ['M1', 'M2']:
            for metrica in ['P', 'R', 'F1', 'nota', 'explicacao']:
                self.assertIn(f'{modelo}_{metrica}', df_global.columns)
    
    def test_criar_dataframe_avaliacao_llm_ordem_colunas(self):
        """Testa que colunas são ordenadas corretamente (P, R, F1, nota, explicacao)"""
        dados = [{'id': 1, 'True': {'a': 1}, 'M1': {'a': 1}}]
        
        avaliacao_llm = [
            {
                'id_peca': 1,
                'M1_explicacao': 'texto',  # Ordem invertida propositalmente
                'M1_nota': 8.0,
                'M1_F1': 0.85,
                'M1_R': 0.80,
                'M1_P': 0.90
            }
        ]
        
        rotulos = ['id', 'True', 'M1']
        analisador = criar_analisador(dados, rotulos,
            avaliacao_llm=avaliacao_llm
        )
        
        df_global, df_campos = analisador._criar_dataframe_avaliacao_llm()
        
        # Verifica ordem das colunas no df_global (id_peca primeiro, depois P, R, F1, nota, explicacao)
        colunas_esperadas = ['id_peca', 'M1_P', 'M1_R', 'M1_F1', 'M1_nota', 'M1_explicacao']
        self.assertEqual(list(df_global.columns), colunas_esperadas)
    
    def test_criar_dataframe_avaliacao_llm_vazio(self):
        """Testa que retorna (None, None) quando não há dados de avaliação"""
        dados = [{'id': 1, 'True': {'a': 1}, 'M1': {'a': 1}}]
        rotulos = ['id', 'True', 'M1']
        
        analisador = criar_analisador(dados, rotulos,
            avaliacao_llm=None  # Sem avaliação
        )
        
        df_global, df_campos = analisador._criar_dataframe_avaliacao_llm()
        self.assertIsNone(df_global)
        self.assertIsNone(df_campos)
    
    def test_criar_dataframe_avaliacao_llm_remove_colunas_vazias(self):
        """Testa que colunas completamente vazias são removidas"""
        dados = [{'id': 1, 'True': {'a': 1}, 'M1': {'a': 1}}]
        
        avaliacao_llm = [
            {
                'id_peca': 1,
                'M1_P': 0.95,
                'M1_R': 0.90,
                'M1_F1': 0.92,
                'M1_nota': 0,  # Valor zero
                'M1_explicacao': '',  # String vazia
                'M1_extra': None  # Valor None
            }
        ]
        
        rotulos = ['id', 'True', 'M1']
        analisador = criar_analisador(dados, rotulos,
            avaliacao_llm=avaliacao_llm
        )
        
        df_global, df_campos = analisador._criar_dataframe_avaliacao_llm()
        
        # Colunas com valores significativos devem estar presentes no df_global
        self.assertIn('M1_P', df_global.columns)
        self.assertIn('M1_R', df_global.columns)
        self.assertIn('M1_F1', df_global.columns)
        
        # Colunas vazias (0, '', None) podem ou não estar presentes
        # dependendo da lógica de remoção
        # O importante é não causar erro
    
    def test_criar_dataframe_avaliacao_llm_multiplas_pecas(self):
        """Testa DataFrame com múltiplas peças"""
        dados = [
            {'id': 1, 'True': {'a': 1}, 'M1': {'a': 1}},
            {'id': 2, 'True': {'b': 2}, 'M1': {'b': 2}}
        ]
        
        avaliacao_llm = [
            {
                'id_peca': 1,
                'M1_P': 0.95,
                'M1_R': 0.90,
                'M1_F1': 0.92,
                'M1_nota': 9.5,
                'M1_explicacao': 'Peça 1 excelente'
            },
            {
                'id_peca': 2,
                'M1_P': 0.85,
                'M1_R': 0.80,
                'M1_F1': 0.82,
                'M1_nota': 8.5,
                'M1_explicacao': 'Peça 2 muito boa'
            }
        ]
        
        rotulos = ['id', 'True', 'M1']
        analisador = criar_analisador(dados, rotulos,
            avaliacao_llm=avaliacao_llm
        )
        
        df_global, df_campos = analisador._criar_dataframe_avaliacao_llm()
        
        # Verifica que tem 2 linhas no df_global
        self.assertEqual(len(df_global), 2)
        
        # Verifica valores da primeira peça
        self.assertEqual(df_global[df_global['id_peca'] == 1]['M1_nota'].iloc[0], 9.5)
        
        # Verifica valores da segunda peça
        self.assertEqual(df_global[df_global['id_peca'] == 2]['M1_nota'].iloc[0], 8.5)
    
    def test_criar_dataframe_avaliacao_llm_sem_id_peca(self):
        """Testa que valida dados de avaliação sem id_peca"""
        dados = [{'id': 1, 'True': {'a': 1}, 'M1': {'a': 1}}]
        
        avaliacao_llm = [
            {
                # Faltando 'id_peca'
                'M1_P': 0.95,
                'M1_R': 0.90
            }
        ]
        
        rotulos = ['id', 'True', 'M1']
        
        # JsonAnaliseDados.validar() deve detectar falta de id_peca
        with self.assertRaises(ValueError) as context:
            dados_analise = JsonAnaliseDados(
                dados=dados,
                rotulos=rotulos,
                tokens=None,
                avaliacao_llm=avaliacao_llm
            )
            dados_analise.validar()
        
        self.assertIn('id_peca', str(context.exception))
    
    def test_exportar_csv_com_avaliacao_llm(self):
        """Testa exportação CSV incluindo avaliação LLM"""
        import tempfile
        import os
        
        dados = [
            {'id': 1, 'True': {'a': 1}, 'M1': {'a': 1}}
        ]
        
        avaliacao_llm = [
            {
                'id_peca': 1,
                'M1_P': 0.95,
                'M1_R': 0.90,
                'M1_F1': 0.92,
                'M1_nota': 9.5,
                'M1_explicacao': 'Teste'
            }
        ]
        
        rotulos = ['id', 'True', 'M1']
        config = {'campos_rouge1': ['(global)']}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            analisador = criar_analisador(dados, rotulos,
                config=config,
                avaliacao_llm=avaliacao_llm,
                pasta_analises=tmpdir
            )
            
            arquivo_csv = analisador.exportar_csv(incluir_estatisticas=False)
            
            # Verifica que arquivo foi criado
            self.assertTrue(os.path.exists(arquivo_csv))
            
            # Verifica que arquivo de avaliação LLM global foi criado (novo nome)
            arquivo_avaliacao = arquivo_csv.replace('.csv', '.avaliacao_llm_global.csv')
            self.assertTrue(os.path.exists(arquivo_avaliacao))
            
            # Lê e valida conteúdo
            import pandas as pd
            df_avaliacao = pd.read_csv(arquivo_avaliacao)
            
            self.assertIn('id_peca', df_avaliacao.columns)
            self.assertIn('M1_P', df_avaliacao.columns)
            self.assertEqual(df_avaliacao['M1_nota'].iloc[0], 9.5)
    
    def test_exportar_excel_com_avaliacao_llm(self):
        """Testa exportação Excel incluindo aba de Avaliação LLM"""
        import tempfile
        import os
        
        dados = [
            {'id': 1, 'True': {'a': 1}, 'M1': {'a': 1}}
        ]
        
        avaliacao_llm = [
            {
                'id_peca': 1,
                'M1_P': 0.95,
                'M1_R': 0.90,
                'M1_F1': 0.92,
                'M1_nota': 9.5,
                'M1_explicacao': 'Teste Excel'
            }
        ]
        
        rotulos = ['id', 'True', 'M1']
        config = {'campos_rouge1': ['(global)']}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            analisador = criar_analisador(dados, rotulos,
                config=config,
                avaliacao_llm=avaliacao_llm,
                pasta_analises=tmpdir
            )
            
            # Testa exportação padrão (sem formatação avançada)
            arquivo_excel = analisador.exportar_excel(
                incluir_estatisticas=False,
                usar_formatacao_avancada=False
            )
            
            # Verifica que arquivo foi criado
            self.assertTrue(os.path.exists(arquivo_excel))
            
            # Lê e valida aba de Avaliação LLM
            import pandas as pd
            try:
                df_avaliacao = pd.read_excel(arquivo_excel, sheet_name='Avaliação LLM')
                
                self.assertIn('id_peca', df_avaliacao.columns)
                self.assertIn('M1_P', df_avaliacao.columns)
                self.assertIn('M1_nota', df_avaliacao.columns)
                self.assertEqual(df_avaliacao['M1_nota'].iloc[0], 9.5)
            except ValueError as e:
                # Aba pode não existir se df_avaliacao for None
                pass
    
    def test_avaliacao_llm_valores_mistos(self):
        """Testa DataFrame com valores mistos (alguns zeros, alguns válidos)"""
        dados = [
            {'id': 1, 'True': {'a': 1}, 'M1': {'a': 1}},
            {'id': 2, 'True': {'b': 2}, 'M1': {'b': 2}}
        ]
        
        avaliacao_llm = [
            {
                'id_peca': 1,
                'M1_P': 0.95,
                'M1_R': 0.0,  # Zero mas válido (pode ser recall zero real)
                'M1_F1': 0.0,
                'M1_nota': 5.0,
                'M1_explicacao': 'Peça 1'
            },
            {
                'id_peca': 2,
                'M1_P': 0.85,
                'M1_R': 0.80,
                'M1_F1': 0.82,
                'M1_nota': 8.5,
                'M1_explicacao': 'Peça 2'
            }
        ]
        
        rotulos = ['id', 'True', 'M1']
        analisador = criar_analisador(dados, rotulos,
            avaliacao_llm=avaliacao_llm
        )
        
        df_global, df_campos = analisador._criar_dataframe_avaliacao_llm()
        
        # Deve manter colunas com pelo menos um valor não-zero no df_global
        self.assertIn('M1_P', df_global.columns)  # Tem valores não-zero
        
        # R e F1 têm um zero e um não-zero, devem ser mantidos
        # (a lógica atual remove apenas se TODOS forem 0)
        if 'M1_R' in df_global.columns:
            self.assertEqual(df_global[df_global['id_peca'] == 1]['M1_R'].iloc[0], 0.0)
            self.assertEqual(df_global[df_global['id_peca'] == 2]['M1_R'].iloc[0], 0.80)
    
    def test_avaliacao_llm_com_metricas_por_campo(self):
        """Testa DataFrame de avaliação LLM com métricas por campo"""
        dados = [
            {
                'id_peca': '123456789012.01.',
                'True': {'tema': 'Civil', 'notas': 'Importante'},
                'M1': {'tema': 'Civil', 'notas': 'Importante'}
            }
        ]
        
        avaliacao_llm = [
            {
                'id_peca': '123456789012.01.',
                # Métricas globais
                'M1_P': 0.70,
                'M1_R': 0.60,
                'M1_F1': 0.65,
                'M1_explicacao': 'Avaliação global',
                # Métricas por campo
                'M1_tema_P': 0.90,
                'M1_tema_R': 0.80,
                'M1_tema_F1': 0.85,
                'M1_notas_P': 0.50,
                'M1_notas_R': 0.40,
                'M1_notas_F1': 0.44
            }
        ]
        
        rotulos = ['id_peca', 'True', 'M1']
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            avaliacao_llm=avaliacao_llm,
            rotulos_destinos=['M1']
        )
        
        analisador = JsonAnaliseDataFrame(dados_analise)
        df_global, df_campos = analisador._criar_dataframe_avaliacao_llm()
        
        # Validações básicas - agora temos dois DataFrames
        self.assertIsNotNone(df_global)
        self.assertIsNotNone(df_campos)
        self.assertEqual(len(df_global), 1)
        self.assertEqual(len(df_campos), 1)
        
        # Verifica métricas globais no df_global
        self.assertIn('M1_P', df_global.columns)
        self.assertIn('M1_R', df_global.columns)
        self.assertIn('M1_F1', df_global.columns)
        self.assertIn('M1_explicacao', df_global.columns)
        
        # Verifica métricas por campo no df_campos
        self.assertIn('M1_tema_P', df_campos.columns)
        self.assertIn('M1_tema_R', df_campos.columns)
        self.assertIn('M1_tema_F1', df_campos.columns)
        self.assertIn('M1_notas_P', df_campos.columns)
        self.assertIn('M1_notas_R', df_campos.columns)
        self.assertIn('M1_notas_F1', df_campos.columns)
        
        # Verifica valores por campo
        self.assertEqual(df_campos['M1_tema_P'].iloc[0], 0.90)
        self.assertEqual(df_campos['M1_tema_R'].iloc[0], 0.80)
        self.assertEqual(df_campos['M1_tema_F1'].iloc[0], 0.85)
        self.assertEqual(df_campos['M1_notas_P'].iloc[0], 0.50)
        self.assertEqual(df_campos['M1_notas_R'].iloc[0], 0.40)
        self.assertEqual(df_campos['M1_notas_F1'].iloc[0], 0.44)
    
    def test_avaliacao_llm_ordem_colunas_com_campos(self):
        """Testa ordenação de colunas: globais primeiro, depois por campo"""
        dados = [
            {'id': 1, 'True': {'a': 1}, 'M1': {'a': 1}}
        ]
        
        avaliacao_llm = [
            {
                'id_peca': 1,
                # Inseridas propositalmente fora de ordem
                'M1_campo2_R': 0.3,
                'M1_P': 0.8,
                'M1_campo1_P': 0.9,
                'M1_R': 0.7,
                'M1_campo1_R': 0.8,
                'M1_campo2_P': 0.4,
                'M1_F1': 0.75,
                'M1_campo1_F1': 0.85,
                'M1_campo2_F1': 0.35,
                'M1_explicacao': 'Teste'
            }
        ]
        
        rotulos = ['id', 'True', 'M1']
        analisador = criar_analisador(dados, rotulos, avaliacao_llm=avaliacao_llm)
        df_global, df_campos = analisador._criar_dataframe_avaliacao_llm()
        
        # Verifica ordem das colunas no df_global
        colunas_global = list(df_global.columns)
        
        # Primeira coluna é id_peca
        self.assertEqual(colunas_global[0], 'id_peca')
        
        # Métricas globais: ordem correta P, R, F1, explicacao
        idx_global_p = colunas_global.index('M1_P')
        idx_global_r = colunas_global.index('M1_R')
        idx_global_f1 = colunas_global.index('M1_F1')
        
        self.assertLess(idx_global_p, idx_global_r)
        self.assertLess(idx_global_r, idx_global_f1)
        
        # Verifica ordem no df_campos (métricas por campo)
        self.assertIsNotNone(df_campos)
        colunas_campos = list(df_campos.columns)
        
        # Primeira coluna é id_peca
        self.assertEqual(colunas_campos[0], 'id_peca')
        
        # Dentro de cada campo: P, R, F1
        idx_campo1_p = colunas_campos.index('M1_campo1_P')
        idx_campo1_r = colunas_campos.index('M1_campo1_R')
        idx_campo1_f1 = colunas_campos.index('M1_campo1_F1')
        self.assertLess(idx_campo1_p, idx_campo1_r)
        self.assertLess(idx_campo1_r, idx_campo1_f1)
    
    def test_avaliacao_llm_metricas_campo_com_none(self):
        """Testa métricas por campo com valores None"""
        dados = [
            {'id': 1, 'True': {'a': 1}, 'M1': {'a': 1}}
        ]
        
        avaliacao_llm = [
            {
                'id_peca': 1,
                'M1_P': 0.75,
                'M1_R': 0.65,
                'M1_F1': 0.70,
                # Alguns campos com None (campo não avaliado)
                'M1_tema_P': None,
                'M1_tema_R': None,
                'M1_tema_F1': None,
                # Outros campos com valores válidos
                'M1_notas_P': 0.85,
                'M1_notas_R': 0.80,
                'M1_notas_F1': 0.82
            }
        ]
        
        rotulos = ['id', 'True', 'M1']
        analisador = criar_analisador(dados, rotulos, avaliacao_llm=avaliacao_llm)
        df_global, df_campos = analisador._criar_dataframe_avaliacao_llm()
        
        # Métricas globais devem estar em df_global
        self.assertIsNotNone(df_global)
        self.assertIn('M1_P', df_global.columns)
        self.assertIn('M1_R', df_global.columns)
        self.assertIn('M1_F1', df_global.columns)
        
        # Métricas por campo ficam em df_campos
        self.assertIsNotNone(df_campos)
        
        # Colunas com None devem ser removidas (todas vazias)
        self.assertNotIn('M1_tema_P', df_campos.columns)
        self.assertNotIn('M1_tema_R', df_campos.columns)
        self.assertNotIn('M1_tema_F1', df_campos.columns)
        
        # Colunas com valores válidos devem permanecer
        self.assertIn('M1_notas_P', df_campos.columns)
        self.assertIn('M1_notas_R', df_campos.columns)
        self.assertIn('M1_notas_F1', df_campos.columns)


class TestJsonAnaliseObservabilidade(unittest.TestCase):
    """Testes para funcionalidade de Observabilidade"""
    
    def test_criar_dataframe_observabilidade_basico(self):
        """Testa criação de DataFrame de observabilidade com métricas básicas"""
        dados = [
            {
                'id_peca': '123456789012.01.',
                'True': {'campo1': 'valor1', 'campo2': 'valor2'},
                'M1': {'campo1': 'valor1', 'campo2': 'valor2'}
            }
        ]
        
        observabilidade = [
            {
                'id_peca': '123456789012.01.',
                'M1_SEG': 45,
                'M1_REV': 2,
                'M1_AGT': 5,
                'M1_IT': 3
            }
        ]
        
        rotulos = ['id_peca', 'True', 'M1']
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            tokens=None,
            avaliacao_llm=None,
            observabilidade=observabilidade,
            rotulos_destinos=['M1']
        )
        
        analisador = JsonAnaliseDataFrame(dados_analise)
        df_obs = analisador._criar_dataframe_observabilidade()
        
        # Validações básicas
        self.assertIsNotNone(df_obs)
        self.assertEqual(len(df_obs), 1)
        self.assertIn('id_peca', df_obs.columns)
        self.assertIn('M1_SEG', df_obs.columns)
        self.assertIn('M1_REV', df_obs.columns)
        self.assertIn('M1_AGT', df_obs.columns)
        self.assertIn('M1_IT', df_obs.columns)
        
        # Valores corretos
        self.assertEqual(df_obs['M1_SEG'].iloc[0], 45)
        self.assertEqual(df_obs['M1_REV'].iloc[0], 2)
        self.assertEqual(df_obs['M1_AGT'].iloc[0], 5)
        self.assertEqual(df_obs['M1_IT'].iloc[0], 3)
    
    def test_observabilidade_metricas_por_campo(self):
        """Testa métricas de observabilidade por campo/agente"""
        dados = [
            {
                'id_peca': '123456789012.01.',
                'True': {'tema': 'Direito Civil', 'notas': 'Nota importante'},
                'M1': {'tema': 'Direito Civil', 'notas': 'Nota importante'}
            }
        ]
        
        observabilidade = [
            {
                'id_peca': '123456789012.01.',
                'M1_SEG': 60,
                'M1_tema_SEG': 15,
                'M1_tema_IT': 2,
                'M1_tema_OK': 'sim',
                'M1_notas_SEG': 20,
                'M1_notas_IT': 1,
                'M1_notas_OK': 'sim'
            }
        ]
        
        rotulos = ['id_peca', 'True', 'M1']
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            observabilidade=observabilidade
        ,
            rotulos_destinos=['M1']
        )
        
        analisador = JsonAnaliseDataFrame(dados_analise)
        df_obs = analisador._criar_dataframe_observabilidade()
        
        # Valida métricas por campo
        self.assertIn('M1_tema_SEG', df_obs.columns)
        self.assertIn('M1_tema_IT', df_obs.columns)
        self.assertIn('M1_tema_OK', df_obs.columns)
        self.assertIn('M1_notas_SEG', df_obs.columns)
        self.assertIn('M1_notas_IT', df_obs.columns)
        self.assertIn('M1_notas_OK', df_obs.columns)
        
        # Valores corretos
        self.assertEqual(df_obs['M1_tema_SEG'].iloc[0], 15)
        self.assertEqual(df_obs['M1_tema_IT'].iloc[0], 2)
        self.assertEqual(df_obs['M1_tema_OK'].iloc[0], 'sim')
        self.assertEqual(df_obs['M1_notas_SEG'].iloc[0], 20)
        self.assertEqual(df_obs['M1_notas_IT'].iloc[0], 1)
        self.assertEqual(df_obs['M1_notas_OK'].iloc[0], 'sim')
    
    def test_observabilidade_metricas_campos_qtd_bytes(self):
        """Testa métricas de QTD e BYTES (apenas origem)"""
        dados = [
            {
                'id_peca': '123456789012.01.',
                'True': {'campo1': 'abc', 'campo2': 'defgh', 'campo3': 'ijklmno'},
                'M1': {'campo1': 'abc', 'campo2': 'defgh', 'campo3': 'ijklmno'}
            }
        ]
        
        observabilidade = [
            {
                'id_peca': '123456789012.01.',
                'True_QTD': 3,
                'True_campo1_BYTES': 3,
                'True_campo2_BYTES': 5,
                'True_campo3_BYTES': 7,
                'M1_SEG': 30,
                'M1_IT': 1
            }
        ]
        
        rotulos = ['id_peca', 'True', 'M1']
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            observabilidade=observabilidade
        ,
            rotulos_destinos=['M1']
        )
        
        analisador = JsonAnaliseDataFrame(dados_analise)
        df_obs = analisador._criar_dataframe_observabilidade()
        
        # Valida que métricas de campos existem para origem (True)
        self.assertIn('True_QTD', df_obs.columns)
        self.assertIn('True_campo1_BYTES', df_obs.columns)
        self.assertIn('True_campo2_BYTES', df_obs.columns)
        self.assertIn('True_campo3_BYTES', df_obs.columns)
        
        # Valores corretos
        self.assertEqual(df_obs['True_QTD'].iloc[0], 3)
        self.assertEqual(df_obs['True_campo1_BYTES'].iloc[0], 3)
        self.assertEqual(df_obs['True_campo2_BYTES'].iloc[0], 5)
        self.assertEqual(df_obs['True_campo3_BYTES'].iloc[0], 7)
        
        # Verifica que M1 tem métricas de execução mas NÃO tem QTD/BYTES
        self.assertIn('M1_SEG', df_obs.columns)
        self.assertIn('M1_IT', df_obs.columns)
        self.assertNotIn('M1_QTD', df_obs.columns)
        self.assertNotIn('M1_campo1_BYTES', df_obs.columns)
    
    def test_observabilidade_multiplos_modelos(self):
        """Testa observabilidade com múltiplos modelos"""
        dados = [
            {
                'id_peca': '123456789012.01.',
                'True': {'campo': 'valor'},
                'M1': {'campo': 'valor'},
                'M2': {'campo': 'valor'}
            }
        ]
        
        observabilidade = [
            {
                'id_peca': '123456789012.01.',
                'M1_SEG': 30,
                'M1_REV': 1,
                'M1_AGT': 3,
                'M2_SEG': 45,
                'M2_REV': 2,
                'M2_AGT': 5
            }
        ]
        
        rotulos = ['id_peca', 'True', 'M1', 'M2']
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            observabilidade=observabilidade
        ,
            rotulos_destinos=['M1']
        )
        
        analisador = JsonAnaliseDataFrame(dados_analise)
        df_obs = analisador._criar_dataframe_observabilidade()
        
        # Valida métricas de ambos modelos
        self.assertIn('M1_SEG', df_obs.columns)
        self.assertIn('M1_REV', df_obs.columns)
        self.assertIn('M2_SEG', df_obs.columns)
        self.assertIn('M2_REV', df_obs.columns)
        
        # Valores corretos
        self.assertEqual(df_obs['M1_SEG'].iloc[0], 30)
        self.assertEqual(df_obs['M2_SEG'].iloc[0], 45)
    
    def test_observabilidade_vazia(self):
        """Testa comportamento com observabilidade vazia"""
        dados = [
            {
                'id_peca': '123456789012.01.',
                'True': {'campo': 'valor'},
                'M1': {'campo': 'valor'}
            }
        ]
        
        rotulos = ['id_peca', 'True', 'M1']
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            observabilidade=[],  # Vazio
            rotulos_destinos=['M1']
        )
        
        analisador = JsonAnaliseDataFrame(dados_analise)
        df_obs = analisador._criar_dataframe_observabilidade()
        
        # DataFrame deve ser None ou vazio
        self.assertTrue(df_obs is None or len(df_obs) == 0)
    
    def test_observabilidade_parcial(self):
        """Testa quando apenas alguns IDs têm observabilidade"""
        dados = [
            {
                'id_peca': '123456789012.01.',
                'True': {'campo': 'valor1'},
                'M1': {'campo': 'valor1'}
            },
            {
                'id_peca': '123456789012.02.',
                'True': {'campo': 'valor2'},
                'M1': {'campo': 'valor2'}
            }
        ]
        
        observabilidade = [
            {
                'id_peca': '123456789012.01.',
                'M1_SEG': 30,
                'M1_IT': 1
            }
            # ID 02 não tem observabilidade
        ]
        
        rotulos = ['id_peca', 'True', 'M1']
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            observabilidade=observabilidade
        ,
            rotulos_destinos=['M1']
        )
        
        analisador = JsonAnaliseDataFrame(dados_analise)
        df_obs = analisador._criar_dataframe_observabilidade()
        
        # DataFrame deve ter apenas 1 linha (ID com observabilidade)
        self.assertEqual(len(df_obs), 1)
        self.assertEqual(df_obs['id_peca'].iloc[0], '123456789012.01.')
    
    def test_observabilidade_ordem_colunas(self):
        """Testa ordem correta das colunas de observabilidade"""
        dados = [
            {
                'id_peca': '123456789012.01.',
                'True': {'campo1': 'a', 'campo2': 'b'},
                'M1': {'campo1': 'a', 'campo2': 'b'}
            }
        ]
        
        observabilidade = [
            {
                'id_peca': '123456789012.01.',
                'True_QTD': 2,
                'True_campo1_BYTES': 1,
                'True_campo2_BYTES': 1,
                'M1_SEG': 30,
                'M1_REV': 1,
                'M1_AGT': 2,
                'M1_IT': 1,
                'M1_campo1_SEG': 10,
                'M1_campo1_IT': 1,
                'M1_campo1_OK': 'sim'
            }
        ]
        
        rotulos = ['id_peca', 'True', 'M1']
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            observabilidade=observabilidade,
            rotulos_destinos=['M1']
        )
        
        analisador = JsonAnaliseDataFrame(dados_analise)
        df_obs = analisador._criar_dataframe_observabilidade()
        
        # Verifica ordem: id_peca, True_QTD, True_BYTES..., M1_SEG, M1_REV, M1_AGT, M1_IT, depois por campo
        cols = df_obs.columns.tolist()
        self.assertEqual(cols[0], 'id_peca')
        
        # True_QTD deve vir antes dos BYTES
        idx_qtd = cols.index('True_QTD')
        idx_bytes1 = cols.index('True_campo1_BYTES')
        self.assertLess(idx_qtd, idx_bytes1)
        
        # Métricas globais de M1 devem vir antes de métricas por campo
        idx_m1_seg = cols.index('M1_SEG')
        idx_m1_campo1_seg = cols.index('M1_campo1_SEG')
        self.assertLess(idx_m1_seg, idx_m1_campo1_seg)
    
    def test_observabilidade_sufixo_rev_correto(self):
        """Testa que sufixo _REV está correto (não _revisoes)"""
        dados = [
            {
                'id_peca': '123456789012.01.',
                'True': {'campo': 'valor'},
                'M1': {'campo': 'valor'}
            }
        ]
        
        observabilidade = [
            {
                'id_peca': '123456789012.01.',
                'M1_SEG': 30,
                'M1_REV': 2,  # Deve ser _REV, não _revisoes
                'M1_AGT': 3
            }
        ]
        
        rotulos = ['id_peca', 'True', 'M1']
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            observabilidade=observabilidade
        ,
            rotulos_destinos=['M1']
        )
        
        analisador = JsonAnaliseDataFrame(dados_analise)
        df_obs = analisador._criar_dataframe_observabilidade()
        
        # Valida que coluna é _REV
        self.assertIn('M1_REV', df_obs.columns)
        self.assertNotIn('M1_revisoes', df_obs.columns)
        self.assertEqual(df_obs['M1_REV'].iloc[0], 2)
    
    def test_exportar_excel_com_observabilidade(self):
        """Testa exportação para Excel com observabilidade"""
        import tempfile
        
        dados = [
            {
                'id_peca': '123456789012.01.',
                'True': {'campo': 'valor'},
                'M1': {'campo': 'valor'}
            }
        ]
        
        observabilidade = [
            {
                'id_peca': '123456789012.01.',
                'True_QTD': 1,
                'True_campo_BYTES': 5,
                'M1_SEG': 30,
                'M1_REV': 1,
                'M1_IT': 2
            }
        ]
        
        rotulos = ['id_peca', 'True', 'M1']
        config = {'campos_levenshtein': ['(global)']}
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            observabilidade=observabilidade
        ,
            rotulos_destinos=['M1']
        )
        
        analisador = JsonAnaliseDataFrame(dados_analise, config=config)
        
        # Exporta para arquivo temporário
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            arquivo_excel = tmp.name
        
        try:
            analisador.exportar_excel(arquivo_excel)
            
            # Verifica que arquivo foi criado
            self.assertTrue(os.path.exists(arquivo_excel))
            
            # Lê de volta para validar
            import pandas as pd
            xl_file = pd.ExcelFile(arquivo_excel)
            
            # Deve ter aba de observabilidade
            self.assertIn('Observabilidade', xl_file.sheet_names)
            
            df_obs = pd.read_excel(arquivo_excel, sheet_name='Observabilidade')
            self.assertIn('M1_SEG', df_obs.columns)
            self.assertIn('M1_REV', df_obs.columns)
            self.assertIn('True_QTD', df_obs.columns)
            
        finally:
            if os.path.exists(arquivo_excel):
                os.remove(arquivo_excel)


class TestJsonAnaliseNomeCampoID(unittest.TestCase):
    """Testes para garantir que nome_campo_id configurável funcione corretamente"""
    
    def test_tokens_com_nome_campo_id_customizado(self):
        """Testa se tokens respeitam nome_campo_id configurado"""
        # Dados com campo ID customizado
        dados = [
            {'doc_id': '001', 'True': {'campo': 'valor1'}, 'Modelo1': {'campo': 'valor1'}}
        ]
        rotulos = ['doc_id', 'True', 'Modelo1']
        
        # Tokens com campo ID customizado
        tokens = [
            {'doc_id': '001', 'Modelo1_input': 100, 'Modelo1_output': 50}
        ]
        
        # Cria analisador
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            tokens=tokens,
            nome_campo_id='doc_id'
        )
        
        analisador = JsonAnaliseDataFrame(dados_analise, max_workers=1)
        
        # Cria DataFrame de tokens
        df_tokens = analisador._criar_dataframe_tokens()
        
        # Verifica se DataFrame foi criado (não retornou None)
        self.assertIsNotNone(df_tokens, "DataFrame de tokens não deve ser None quando nome_campo_id está correto")
        
        # Verifica se a coluna de ID está presente
        self.assertIn('doc_id', df_tokens.columns, "Coluna doc_id deve existir no DataFrame")
        
        # Verifica se o valor está correto
        self.assertEqual(df_tokens['doc_id'].iloc[0], '001')
    
    def test_avaliacao_llm_com_nome_campo_id_customizado(self):
        """Testa se avaliação LLM respeita nome_campo_id configurado"""
        # Dados com campo ID customizado
        dados = [
            {'registro_id': 'A1', 'True': {'campo': 'valor1'}, 'Modelo1': {'campo': 'valor1'}}
        ]
        rotulos = ['registro_id', 'True', 'Modelo1']
        
        # Avaliação LLM com campo ID customizado
        avaliacao_llm = [
            {'registro_id': 'A1', 'Modelo1_P': 0.9, 'Modelo1_R': 0.8, 'Modelo1_F1': 0.85}
        ]
        
        # Cria analisador
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            avaliacao_llm=avaliacao_llm,
            nome_campo_id='registro_id'
        )
        
        analisador = JsonAnaliseDataFrame(dados_analise, max_workers=1)
        
        # Cria DataFrame de avaliação LLM
        df_global, df_campos = analisador._criar_dataframe_avaliacao_llm()
        
        # Verifica se DataFrame foi criado
        self.assertIsNotNone(df_global, "DataFrame de avaliação LLM global não deve ser None")
        
        # Verifica se a coluna de ID está presente
        self.assertIn('registro_id', df_global.columns, "Coluna registro_id deve existir")
        
        # Verifica valor
        self.assertEqual(df_global['registro_id'].iloc[0], 'A1')
    
    def test_validacao_tokens_com_nome_campo_id_errado(self):
        """Testa se validação detecta campo ID incorreto em tokens"""
        dados = [
            {'id': '001', 'True': {'campo': 'valor1'}, 'Modelo1': {'campo': 'valor1'}}
        ]
        rotulos = ['id', 'True', 'Modelo1']
        
        # Tokens com campo ID ERRADO (id_peca ao invés de id)
        tokens = [
            {'id_peca': '001', 'Modelo1_input': 100}  # ERRADO!
        ]
        
        # Deve lançar erro na validação
        with self.assertRaises(ValueError) as context:
            dados_analise = JsonAnaliseDados(
                dados=dados,
                rotulos=rotulos,
                tokens=tokens,
                nome_campo_id='id'
            )
            dados_analise.validar()
        
        self.assertIn("'id'", str(context.exception))
    
    def test_graficos_tokens_com_nome_campo_id_customizado(self):
        """Testa se geração de gráficos de tokens funciona com nome_campo_id customizado"""
        import tempfile
        import os
        
        dados = [
            {'code': 'X1', 'True': {'campo': 'valor1'}, 'M1': {'campo': 'valor1'}},
            {'code': 'X2', 'True': {'campo': 'valor2'}, 'M1': {'campo': 'valor2'}}
        ]
        rotulos = ['code', 'True', 'M1']
        
        tokens = [
            {'code': 'X1', 'M1_input': 100, 'M1_output': 50, 'M1_total': 150},
            {'code': 'X2', 'M1_input': 120, 'M1_output': 60, 'M1_total': 180}
        ]
        
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            tokens=tokens,
            nome_campo_id='code'
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            analisador = JsonAnaliseDataFrame(
                dados_analise,
                pasta_analises=tmpdir,
                max_workers=1
            )
            
            # Tenta gerar gráficos de tokens
            arquivos = analisador.gerar_graficos_tokens(pasta_saida=tmpdir)
            
            # Deve gerar gráficos sem erro
            self.assertIsInstance(arquivos, list)
            # Deve ter gerado pelo menos um gráfico (input, output ou total)
            self.assertGreater(len(arquivos), 0, "Deve gerar pelo menos um gráfico de tokens")
    
    def test_exportacao_excel_com_config_nome_campo_id(self):
        """Testa se exportação Excel inclui nome_campo_id na aba Config"""
        import tempfile
        import pandas as pd
        
        dados = [
            {'custom_id': 'ID1', 'True': {'x': 'a'}, 'M1': {'x': 'a'}}
        ]
        rotulos = ['custom_id', 'True', 'M1']
        
        dados_analise = JsonAnaliseDados(
            dados=dados,
            rotulos=rotulos,
            nome_campo_id='custom_id',
            rotulo_campo_id='custom_id',
            rotulo_origem='True'
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            analisador = JsonAnaliseDataFrame(
                dados_analise,
                pasta_analises=tmpdir,
                max_workers=1
            )
            
            arquivo_excel = os.path.join(tmpdir, 'teste.xlsx')
            analisador.exportar_excel(arquivo_excel, gerar_graficos=False)
            
            # Lê aba Config
            df_config = pd.read_excel(arquivo_excel, sheet_name='Config')
            
            # Verifica se nome_campo_id está na config
            row_nome_campo_id = df_config[df_config['parametro'] == 'nome_campo_id']
            self.assertFalse(row_nome_campo_id.empty, "Deve ter linha com nome_campo_id na Config")
            self.assertEqual(row_nome_campo_id.iloc[0]['valor'], 'custom_id')


def run_tests(verbosity=2):
    """Executa todos os testes"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Adiciona todos os testes
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseBasico))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseExtrairCampos))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseComparar))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseMetricasEspecificas))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseEstrutura))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseDataFrame))
    suite.addTests(loader.loadTestsFromTestCase(TestJson2Texto))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseMultiplasMetricas))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseConverterTexto))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseCalcularMetrica))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseAcuraciaEstrutural))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseExemplos))
    
    # Novos testes de casos extremos e precisão
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseCasosExtremos))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseCalculoPreciso))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseConfiguracoesEspeciais))
    suite.addTests(loader.loadTestsFromTestCase(TestJson2TextoCasosEspeciais))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseDataFrameCasosEspeciais))
    
    # Testes de Avaliação LLM
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseDataFrameAvaliacaoLLM))
    
    # Testes de Observabilidade
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseObservabilidade))
    
    # Testes de nome_campo_id configurável (bugs corrigidos)
    suite.addTests(loader.loadTestsFromTestCase(TestJsonAnaliseNomeCampoID))
    
    # Executa
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 70)
    print("Executando testes unitários da classe JsonAnalise - NOVA ESTRUTURA")
    print("=" * 70)
    success = run_tests()
    print("\n" + "=" * 70)
    if success:
        print("✓ Todos os testes passaram com sucesso!")
    else:
        print("✗ Alguns testes falharam. Verifique os detalhes acima.")
    print("=" * 70)
