# -*- coding: utf-8 -*-

'''
Utilitários para avaliar F1, Precision e Recall de JSONs.

Autor: Luiz Anísio
Data: 17/07/2025
Fonte: https://github.com/luizanisio/llms/tree/main/src

Descrição:
-----------
Utiliza técnicas de Levenshtein, ROUGE e BERTScore para alinhamento e cálculo de similaridade.
Realiza comparação campo a campo de acordo com as configurações fornecidas.

FILOSOFIA DE SELEÇÃO DE MÉTRICAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. BERTScore → Textos longos com nuances semânticas (padrão para (global))
2. ROUGE-L   → Estruturas/sequências ordenadas
3. ROUGE-2   → Frases médias, precisão de bigramas
4. ROUGE-1   → Termos individuais, palavras-chave (padrão para (estrutura))
5. Levenshtein → Textos curtos exatos (nomes, IDs, valores numéricos)
'''

# Só emite um aviso se a dependência não estiver instalada
# o erro será lançado apenas se a função for chamada
importados = set()
def importar(modulo):
    if modulo in importados:
        return
    if modulo == 'Levenshtein':
        try:
            import Levenshtein
            globals()['Levenshtein'] = Levenshtein
            importados.add('Levenshtein')
        except ImportError as e:
            raise ImportError('Módulo python-Levenshtein não instalado. Instale com: pip install python-Levenshtein')
    if modulo == 'cachetools':
        try:
            import cachetools
            globals()['cachetools'] = cachetools
            importados.add('cachetools')
        except ImportError as e:
            # cachetools é opcional, mas recomendado para performance
            print(f'⚠️  Aviso: cachetools não instalado. Instale com: pip install cachetools')
            print(f'           Performance pode ser impactada sem cache de alinhamento de listas.')
            globals()['cachetools'] = None
    if modulo == 'rouge_score':
        try:
            from rouge_score import rouge_scorer
            globals()['rouge_scorer'] = rouge_scorer
            importados.add('rouge_score')
        except ImportError as e:
            raise ImportError('Módulo rouge-score não instalado. Instale com: pip install rouge-score')
    if modulo == 'pandas':
        try:
            import pandas as pd
            globals()['pd'] = pd
            importados.add('pandas')
        except ImportError as e:
            raise ImportError('Módulo pandas não instalado. Instale com: pip install pandas')
    if modulo == 'sklearn':
        try:
            from sklearn.metrics import precision_recall_fscore_support
            globals()['precision_recall_fscore_support'] = precision_recall_fscore_support
            importados.add('sklearn')
        except ImportError as e:
            raise ImportError('Módulo scikit-learn não instalado. Instale com: pip install scikit-learn')

import json
import re
import os, sys
import numpy as np
from typing import Any, Dict, List, Tuple, Union, Iterable
from tqdm import tqdm
import hashlib
from copy import deepcopy
from datetime import datetime
import threading
import pandas as pd
sys.path.extend(['./utils','./src'])
from util import Util
from util_json_exemplos import JsonAnaliseExemplos
from util_pandas import UtilPandasExcel
from util_graficos import UtilGraficos, Cores
from util_bertscore import bscore
from util_json_dados import JsonAnaliseDados
from concurrent.futures import ThreadPoolExecutor, as_completed

# OTIMIZAÇÃO: Variável global para controlar inicialização do BERTScore
BS_MAX_WORKERS = None # definido dinamicamente na classe JsonAnaliseDataFrame
MAX_STRING_MD = 5000 # tamanho máximo da string nos exemplos de markdown

# ═════════════════════════════════════════════════════════════════════════
# SISTEMA DE CACHE PARA OTIMIZAÇÃO DE ALINHAMENTO DE LISTAS
# ═════════════════════════════════════════════════════════════════════════
# PROBLEMA: Quando um campo é analisado com múltiplas métricas (ex: ROUGE-L e ROUGE-2),
#           o alinhamento de listas é executado repetidamente com os mesmos dados.
# SOLUÇÃO: Cache LRU com @cached do cachetools + Lock integrado para thread-safety.
# VANTAGENS: Lock direto na anotação, controle de tamanho, políticas de expiração.
# ═════════════════════════════════════════════════════════════════════════

importar('cachetools')

# Caches LRU com limite de tamanho para controle de memória
_cache_alinhamento = cachetools.LRUCache(maxsize=1000)
_cache_conversao = cachetools.LRUCache(maxsize=2000)

# Locks para thread-safety - usados diretamente no @cached
_lock_alinhamento = threading.Lock()
_lock_conversao = threading.Lock()

def _limpar_caches():
    """Limpa todos os caches (útil para testes ou liberar memória)"""
    with _lock_alinhamento:
        _cache_alinhamento.clear()
    with _lock_conversao:
        _cache_conversao.clear()

# ═════════════════════════════════════════════════════════════════════════
# Funções cached com @cached do cachetools para evitar reprocessamento
# Lock integrado no decorador para thread-safety automática
# O @cached usa os parâmetros como chave de cache automaticamente
# ═════════════════════════════════════════════════════════════════════════

@cachetools.cached(cache=_cache_alinhamento, lock=_lock_alinhamento)
def _alinhar_listas_cached(metrica_alinhamento: str, 
                           lista_pred_tuple: tuple, lista_true_tuple: tuple) -> tuple:
    """
    Versão cached de alinhar_listas usando @cached do cachetools.
    Recebe tuplas ao invés de listas para serem hashable.
    Thread-safe através do lock integrado no decorador @cached.
    
    O decorador @cached usa automaticamente os parâmetros (metrica, tuplas) 
    como chave de cache - não precisa de hash explícito.
    
    Args:
        metrica_alinhamento: métrica para alinhamento
        lista_pred_tuple: tupla com elementos da lista pred
        lista_true_tuple: tupla com elementos da lista true
    
    Returns:
        tuple: (lista_pred_alinhada, lista_true_alinhada)
    """
    # Converte tuplas de volta para listas
    lista_pred = list(lista_pred_tuple)
    lista_true = list(lista_true_tuple)
    
    # Executa lógica original de alinhamento
    return _executar_alinhamento(lista_pred, lista_true, metrica_alinhamento)

def _executar_alinhamento(lista_pred: list, lista_true: list, metrica_alinhamento: str) -> tuple:
    """
    Executa o alinhamento de listas (lógica extraída do método alinhar_listas original).
    """
    # Casos triviais
    if not lista_pred and not lista_true:
        return [], []
    if not lista_pred:
        return ([None] * len(lista_true), lista_true)
    if not lista_true:
        return (lista_pred, [None] * len(lista_pred))
    
    # Converte itens para texto para comparação
    def item_to_text(item):
        if isinstance(item, dict):
            return Json2Texto.to_natural_text(item, normalize_whitespace=True)
        elif isinstance(item, str):
            return item
        else:
            return str(item)
    
    textos_pred = [item_to_text(item) for item in lista_pred]
    textos_true = [item_to_text(item) for item in lista_true]
    
    # Calcula matriz de similaridade (pred x true)
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # Se scipy não está disponível, retorna listas sem alinhamento
        return (lista_pred, lista_true)
    
    n_pred = len(lista_pred)
    n_true = len(lista_true)
    
    # Matriz de custo (usamos 1 - similaridade para minimização)
    custo = np.ones((n_pred, n_true))
    
    # Determina métrica a usar
    metrica_usar = metrica_alinhamento
    
    if metrica_alinhamento == 'auto':
        # Calcula tamanho médio dos textos para decidir métrica
        tamanhos = [len(t) for t in textos_pred + textos_true if t]
        tamanho_medio = sum(tamanhos) / len(tamanhos) if tamanhos else 0
        
        if tamanho_medio < 50:
            metrica_usar = 'levenshtein'
        elif tamanho_medio < 100:
            metrica_usar = 'rouge1'
        else:
            metrica_usar = 'rouge2'
    
    # Calcula similaridade usando a métrica escolhida
    if metrica_usar == 'levenshtein':
        importar('Levenshtein')
        for i, texto_pred in enumerate(textos_pred):
            for j, texto_true in enumerate(textos_true):
                if not texto_pred or not texto_true:
                    continue
                # Levenshtein.ratio retorna similaridade [0,1]
                similaridade = Levenshtein.ratio(texto_true, texto_pred)
                custo[i, j] = 1.0 - similaridade
    
    elif metrica_usar in ('rouge1', 'rouge2', 'rouge'):
        importar('rouge_score')
        # Mapeia para tipo ROUGE correto
        tipo_rouge_map = {
            'rouge1': 'rouge1',
            'rouge2': 'rouge2',
            'rouge': 'rougeL'
        }
        tipo_rouge = tipo_rouge_map.get(metrica_usar, 'rouge1')
        scorer = rouge_scorer.RougeScorer([tipo_rouge], use_stemmer=True, split_summaries=True)
        
        for i, texto_pred in enumerate(textos_pred):
            for j, texto_true in enumerate(textos_true):
                if not texto_pred or not texto_true:
                    continue
                # Calcula F1 como medida de similaridade
                scores = scorer.score(texto_true, texto_pred)
                similaridade = scores[tipo_rouge].fmeasure
                custo[i, j] = 1.0 - similaridade
    
    else:
        raise ValueError(f"Métrica de alinhamento '{metrica_usar}' não suportada. Use 'auto', 'levenshtein', 'rouge1', 'rouge2' ou 'rouge'")
    
    # Resolve problema de atribuição (algoritmo húngaro)
    indices_pred, indices_true = linear_sum_assignment(custo)
    
    # Threshold para aceitar pareamento (similaridade mínima)
    THRESHOLD_SIMILARIDADE = 0.1
    
    # Constrói mapeamento de índices
    pred_map = {}
    true_usados = set()
    
    for i, j in zip(indices_pred, indices_true):
        similaridade = 1.0 - custo[i, j]
        if similaridade >= THRESHOLD_SIMILARIDADE:
            pred_map[i] = j
            true_usados.add(j)
    
    # Tamanho final = max(n_pred, n_true)
    tamanho_final = max(n_pred, n_true)
    pred_alinhada = [None] * tamanho_final
    true_alinhada = [None] * tamanho_final
    
    # Preenche true em suas posições originais
    for j in range(n_true):
        true_alinhada[j] = lista_true[j]
    
    # Preenche pred nos índices mapeados
    pred_usados = set()
    for i, j in pred_map.items():
        pred_alinhada[j] = lista_pred[i]
        pred_usados.add(i)
    
    # Itens de pred não pareados vão para posições livres
    pred_nao_pareados = [lista_pred[i] for i in range(n_pred) if i not in pred_usados]
    
    # Encontra posições livres (None) em pred_alinhada
    idx_inserir = 0
    for item in pred_nao_pareados:
        while idx_inserir < len(pred_alinhada) and pred_alinhada[idx_inserir] is not None:
            idx_inserir += 1
        if idx_inserir < len(pred_alinhada):
            pred_alinhada[idx_inserir] = item
            idx_inserir += 1
    
    # Remove None's trailing se ambos são None
    while len(pred_alinhada) > 0 and pred_alinhada[-1] is None and true_alinhada[-1] is None:
        pred_alinhada.pop()
        true_alinhada.pop()
    
    return (pred_alinhada, true_alinhada)

@cachetools.cached(cache=_cache_conversao, lock=_lock_conversao)
def _converter_para_texto_cached(metrica: str, normalize_ws: bool, valor_json: str) -> str:
    """
    Versão cached de _converter_para_texto usando @cached do cachetools.
    Thread-safe através do lock integrado no decorador @cached.
    
    O decorador @cached usa automaticamente os parâmetros (metrica, normalize_ws, valor_json)
    como chave de cache - não precisa de hash explícito.
    
    Args:
        metrica: tipo de métrica
        normalize_ws: se deve normalizar whitespace
        valor_json: valor serializado como JSON string
    
    Returns:
        str: texto convertido
    """
    # Desserializa o valor
    valor = json.loads(valor_json)
    
    # Garante que é um dict para Json2Texto
    if not isinstance(valor, dict):
        valor_dict = {'_valor': valor}
    else:
        valor_dict = valor
    
    if metrica == 'bertscore':
        texto = Json2Texto.to_linear_text(valor_dict, normalize_whitespace=normalize_ws)
    else:
        # rouge, rouge1, rouge2, levenshtein
        texto = Json2Texto.to_natural_text(valor_dict, normalize_whitespace=normalize_ws)
    
    return texto


class JsonAnalise:
    ''' Configuração para métodos que usam o parâmetro "config" (dict):
        
        PARÂMETROS DE MÉTRICAS MÚLTIPLAS:
        Um campo pode participar de múltiplas listas de métricas, gerando análises independentes.
        
        - campos_bertscore: list - campos analisados com BERTScore (similaridade semântica)
        - campos_rouge: list - campos analisados com ROUGE-L (coerência geral)
        - campos_rouge1: list - campos analisados com ROUGE-1 (overlap de unigramas)
        - campos_rouge2: list - campos analisados com ROUGE-2 (overlap de bigramas)
        - campos_levenshtein: list - campos analisados com Levenshtein (distância de edição)
        
        Campos especiais: 
        - '(global)' - se não estiver em nenhuma lista, é adicionado automaticamente em campos_bertscore
        - '(estrutura)' - se não estiver em nenhuma lista, é adicionado automaticamente em campos_rouge1
        
        PARÂMETROS DE ESTRUTURA:
        - nivel_campos: int (padrão 1) - profundidade de extração de campos
        - padronizar_simbolos: bool (padrão True) - normalização de texto
        - rouge_stemmer: bool (padrão True) - stemming no ROUGE
        
        ESTRUTURA DE SAÍDA:
        Formato: <campo>_<tecnica>_<metrica>
        Exemplos:
            - (global)_rouge2_F1: 0.85       # padrão para (global)
            - (estrutura)_rouge1_F1: 0.90    # padrão para (estrutura)
            - (global)_bertscore_F1: 0.80    # se (global) em campos_bertscore
            - resumo_bertscore_F1: 0.90
            - descricao_rouge_F1: 0.88
        
        Exemplo completo:
        {
            'nivel_campos': 2,
            'campos_bertscore': ['(global)', 'resumo'],
            'campos_rouge': ['fatos'],
            'campos_rouge2': ['texto_longo'],  # (global) será adicionado aqui automaticamente
            'padronizar_simbolos': True
        }
    '''
    RE_UNE_ESPACO = re.compile(r"\s+")
    RE_UNE_ENTER = re.compile(r"\n+")
    METRICAS_VALIDAS = {'bertscore', 'rouge', 'rouge1', 'rouge2', 'levenshtein'}

    @classmethod
    def padronizar_simbolos(cls, texto: Union[str, dict]) -> str:
        """ Padroniza alguns símbolos para comparação mais precisa """
        # Une quebras de linha em espaço
        saida = cls.RE_UNE_ENTER.sub(' ', texto.strip())
        # Une múltiplos espaços em um único
        saida = cls.RE_UNE_ESPACO.sub(' ', saida)
        # Corrige aspas especiais
        saida = saida.replace("“", '"').replace("”", '"').replace("'", '"')
        return saida.lower()

    @classmethod
    def verifica_versao(cls):
        print(f'JsonAnalise carregado corretamente em {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}!')

    @classmethod
    def distancia_levenshtein(
        cls,
        texto1: str,
        texto2: str,
        padronizar_simbolos: bool = True
    ) -> float:
        """
        Retorna a distância de Levenshtein entre texto1 e texto2.
        Se padronizar=True, aplica padronização de símbolos antes do cálculo.
        """
        importar('Levenshtein')
        
        if padronizar_simbolos:
            texto1 = cls.padronizar_simbolos(texto1)
            texto2 = cls.padronizar_simbolos(texto2)
        # usa Levenshtein.distance do pacote python-Levenshtein
        return 1-Levenshtein.ratio(texto1, texto2)

    @classmethod
    def distancia_jaccard(
        cls, 
        lista_a: list, 
        lista_b: list,
        outras_distancias: Union[List[float], None] = None,
        as_dict = False
    ) -> Union[float, dict]:
        """
        Calcula a distância de Jaccard entre duas listas.
        
        A distância de Jaccard mede a dissimilaridade entre dois conjuntos:
        distancia = 1 - (interseção / união)
        
        Args:
            lista_a, lista_b: listas a comparar via Jaccard
            outras_distancias: lista de distâncias adicionais para ponderar no resultado final
                              Útil para combinar distância de Jaccard com similaridades semânticas
            as_dict: se True, retorna dict com detalhes; se False, retorna apenas a distância
        
        Returns:
            float ou dict: distância média ponderada (0 = iguais, 1 = totalmente diferentes)
        """
        log = {'distancia': 0, 'distancias': []}
        
        # Calcula distância de Jaccard
        conjunto_a = set(lista_a)
        conjunto_b = set(lista_b)
        uniao = conjunto_a.union(conjunto_b)
        tamanho_uniao = len(uniao)
        
        if tamanho_uniao == 0:
            # Ambas as listas estão vazias
            # Se há outras distâncias, usa apenas elas; senão retorna 0
            if isinstance(outras_distancias, (tuple, list, set)) and len(outras_distancias) > 0:
                log['distancias'] = list(outras_distancias)
                log['distancia'] = sum(outras_distancias) / len(outras_distancias)
            # Senão, distância = zero (são iguais)
            return log if as_dict else log['distancia']
        
        intersecao = conjunto_a.intersection(conjunto_b)
        tamanho_intersecao = len(intersecao)
        distancia_jaccard = 1.0 - (tamanho_intersecao / tamanho_uniao)
        
        log['distancia'] = distancia_jaccard
        if as_dict:
            # Detalha: 0 para itens na interseção, 1 para itens só na união
            log['distancias'] = [0] * tamanho_intersecao + [1] * (tamanho_uniao - tamanho_intersecao)
        
        # Se não há outras distâncias, retorna apenas Jaccard
        if not isinstance(outras_distancias, (tuple, list, set)) or len(outras_distancias) == 0:
            return log if as_dict else log['distancia']
        
        # Pondera com outras distâncias
        peso_total = tamanho_uniao
        soma_ponderada = distancia_jaccard * tamanho_uniao
        
        for dist in outras_distancias:
            soma_ponderada += dist
            peso_total += 1
        
        log['distancias'].extend(list(outras_distancias))
        log['distancia'] = soma_ponderada / peso_total
        
        return log if as_dict else log['distancia']

    @classmethod
    def alinhar_listas(cls, lista_pred: list, lista_true: list, metrica_alinhamento: str = 'auto') -> tuple:
        """
        Alinha duas listas de objetos usando similaridade textual para encontrar melhor pareamento.
        
        OTIMIZAÇÃO: Usa @cached com lock para thread-safety e evitar realinhamento.
        O decorador @cached usa automaticamente as tuplas como chave de cache.
        
        Problema: Listas com itens similares mas em ordens diferentes causam baixa precision/recall.
        Solução: Usa algoritmo de casamento bipartido (húngaro) baseado em similaridade textual.
        
        Args:
            lista_pred: lista predita (pode ter ordem diferente)
            lista_true: lista verdadeira/esperada
            metrica_alinhamento: métrica para calcular similaridade
                - 'auto' (padrão): seleciona automaticamente baseado no tamanho do texto
                  • < 50 chars: levenshtein (textos muito curtos, IDs, nomes)
                  • 50-200 chars: rouge1 (palavras-chave, frases curtas)
                  • > 200 chars: rouge2 (textos médios/longos, bigramas)
                - 'levenshtein': distância de edição (ideal para textos curtos exatos)
                - 'rouge1': overlap de unigramas (ideal para frases)
                - 'rouge2': overlap de bigramas (ideal para textos longos)
                - 'rouge': ROUGE-L (sequências ordenadas)
        
        Returns:
            tuple: (lista_pred_alinhada, lista_true_alinhada)
                   - Ambas listas terão mesmo tamanho
                   - Itens alinhados maximizam similaridade global
                   - Itens sem par são preservados com None no pareamento
        
        Example:
            pred = [item_A, item_B, item_C]  # 3 itens
            true = [item_1, item_2, item_3, item_4, item_5]  # 5 itens
            
            # Se item_A ≈ item_1, item_B ≈ item_5, item_C ≈ item_4:
            pred_alin = [item_A, None, None, item_C, item_B]
            true_alin = [item_1, item_2, item_3, item_4, item_5]
        """
        # ═════════════════════════════════════════════════════════════════════════
        # OTIMIZAÇÃO: Usa @cached com lock integrado para thread-safety
        # ═════════════════════════════════════════════════════════════════════════
        
        # Converte listas para tuplas (hashable)
        try:
            # Serializa dicts para JSON para serem hashable
            lista_pred_tuple = tuple(
                json.dumps(item, sort_keys=True) if isinstance(item, dict) else item 
                for item in lista_pred
            )
            lista_true_tuple = tuple(
                json.dumps(item, sort_keys=True) if isinstance(item, dict) else item 
                for item in lista_true
            )
            
            # Chama função cached - lock automático via decorador
            # O @cached usa automaticamente (metrica, tuplas) como chave de cache
            resultado = _alinhar_listas_cached(
                metrica_alinhamento,
                lista_pred_tuple, 
                lista_true_tuple
            )
            
            return resultado
            
        except Exception as e:
            # Em caso de erro no cache, executa diretamente sem cache
            return _executar_alinhamento(lista_pred, lista_true, metrica_alinhamento)


    @classmethod
    def print_analise_config(cls, config):
        ''' Print do config ajustado para debug'''
        print(f'CONFIG: {json.dumps(cls.__ajustar_config(config), indent=2, ensure_ascii=False)}')

    @classmethod
    def __ajustar_config(cls, config: dict):
        """Ajusta e valida a configuração, normalizando campos e aplicando valores padrão"""
        if (config is not None) and config.get('~cópia-validada~'):
           return config
        config = {} if config is None else deepcopy(config)
        
        # Nível de campos (padrão: 1)
        nivel_campos = config.get('nivel_campos', 1)
        if not isinstance(nivel_campos, int) or nivel_campos < 1:
            raise ValueError(f"nivel_campos deve ser int >= 1, recebido: {nivel_campos}")
        config['nivel_campos'] = nivel_campos
        
        # Padronização do texto
        config['padronizar_simbolos'] = config.get('padronizar_simbolos', True) if isinstance(config.get('padronizar_simbolos'), bool) else True
        
        # Configurações rouge
        config['rouge_stemmer'] = config.get('rouge_stemmer', True) if isinstance(config.get('rouge_stemmer'), bool) else True
        
       
        # Normaliza nomes de campos com aliases
        aliases = {
            'campos_rouge_1': 'campos_rouge1',
            'campos_rouge_2': 'campos_rouge2',
        }
        for alias, nome_correto in aliases.items():
            if alias in config and nome_correto not in config:
                config[nome_correto] = config.pop(alias)
        
        # Valida campos como lista (agora permite múltiplas métricas por campo)
        campos_lista_type = ['campos_rouge', 'campos_rouge1', 'campos_rouge2', 'campos_bertscore', 'campos_levenshtein']
        for campo in campos_lista_type:
            config[campo] = list(config[campo]) if isinstance(config.get(campo), (set, tuple, list)) else []
        
        # Define métricas padrão para campos especiais se não estiverem em nenhuma lista
        # (global) -> campos_bertscore se não estiver em nenhuma lista
        # (estrutura) -> campos_rouge1 se não estiver em nenhuma lista
        for campo_especial, lista_padrao in [('(global)', 'campos_bertscore'), ('(estrutura)', 'campos_rouge1')]:
            # Verifica se o campo especial está em alguma lista
            esta_em_alguma = any(campo_especial in config.get(lista, []) for lista in campos_lista_type)
            if not esta_em_alguma:
                config[lista_padrao].append(campo_especial)
        
        config['~cópia-validada~'] = True
        return config

    @classmethod
    def _extrair_campos_por_nivel(cls, dados: dict, nivel: int) -> dict:
        """
        Extrai campos de um JSON até o nível especificado.
        
        Args:
            dados: dicionário a ser analisado
            nivel: profundidade de extração (1 = raiz, 2 = raiz + 1 nível, etc)
        
        Returns:
            dict {nome_campo: valor} onde:
                - nivel 1: {'campo': valor}
                - nivel 2: {'campo': valor, 'campo.subcampo': valor}
        
        Exemplo:
            dados = {'a': 1, 'b': {'c': 2, 'd': 3}}
            nivel 1 -> {'a': 1, 'b': {'c': 2, 'd': 3}}
            nivel 2 -> {'a': 1, 'b.c': 2, 'b.d': 3}
        """
        if not isinstance(dados, dict):
            return {}
        
        campos = {}
        
        def _extrair(obj: Any, prefixo: str, nivel_atual: int):
            if nivel_atual > nivel:
                return
            
            if isinstance(obj, dict):
                for chave, valor in obj.items():
                    campo_completo = f"{prefixo}.{chave}" if prefixo else chave
                    
                    if nivel_atual == nivel:
                        # Chegou no nível desejado, adiciona o valor como está
                        campos[campo_completo] = valor
                    else:
                        # Continua descendo se for dict ou lista
                        if isinstance(valor, dict):
                            _extrair(valor, campo_completo, nivel_atual + 1)
                        else:
                            # Valor escalar ou lista: adiciona aqui
                            campos[campo_completo] = valor
            else:
                # Não é dict, adiciona como está
                campos[prefixo] = obj
        
        _extrair(dados, '', 1)
        return campos

    @classmethod
    def _determinar_metricas_campo(cls, campo: str, config: dict) -> list:
        """
        Determina quais métricas usar para um campo específico.
        Um campo pode participar de múltiplas métricas.
        
        Args:
            campo: nome do campo (ex: 'nome', '(global)', '(estrutura)')
            config: configuração ajustada
        
        Returns:
            list: lista de métricas ('bertscore', 'rouge', 'rouge1', 'rouge2', 'levenshtein')
        
        Note:
            __ajustar_config já garante que (global) está em campos_rouge2 e
            (estrutura) está em campos_rouge1 se não estiverem em outras listas.
        """
        metricas = []
        
        # Verifica em quais listas o campo aparece
        if campo in config.get('campos_bertscore', []):
            metricas.append('bertscore')
        if campo in config.get('campos_rouge', []):
            metricas.append('rouge')
        if campo in config.get('campos_rouge1', []):
            metricas.append('rouge1')
        if campo in config.get('campos_rouge2', []):
            metricas.append('rouge2')
        if campo in config.get('campos_levenshtein', []):
            metricas.append('levenshtein')
        
        return metricas

    @classmethod
    def _converter_para_texto(cls, valor: Any, metrica: str, config: dict) -> str:
        """
        Converte um valor para texto usando Json2Texto conforme a métrica.
        
        OTIMIZAÇÃO: Usa @cached com lock para thread-safety e evitar reconversão.
        O decorador @cached usa automaticamente os parâmetros como chave de cache.
        
        Args:
            valor: valor a ser convertido (pode ser dict, list, str, int, etc)
            metrica: tipo de métrica ('bertscore', 'rouge', 'rouge1', 'rouge2', 'levenshtein')
            config: configuração ajustada
        
        Returns:
            str: texto convertido
        """
        # ═════════════════════════════════════════════════════════════════════════
        # CASOS RÁPIDOS: String e escalares (não usam cache - conversão trivial)
        # ═════════════════════════════════════════════════════════════════════════
        
        # Se já é string, apenas padroniza
        if isinstance(valor, str):
            if config.get('padronizar_simbolos', True):
                return cls.padronizar_simbolos(valor)
            return valor
        
        # Se é valor escalar (int, float, bool, None), converte para string
        if isinstance(valor, (int, float, bool)) or valor is None:
            texto = str(valor) if valor is not None else ''
            if config.get('padronizar_simbolos', True):
                return cls.padronizar_simbolos(texto)
            return texto
        
        # ═════════════════════════════════════════════════════════════════════════
        # CASOS COMPLEXOS: Dict/List (usa @cached com lock integrado)
        # ═════════════════════════════════════════════════════════════════════════
        
        normalize_ws = config.get('padronizar_simbolos', True)
        
        try:
            # Serializa valor para JSON (hashable)
            valor_json = json.dumps(valor, sort_keys=True, ensure_ascii=False)
            
            # Chama função cached - lock automático via decorador
            # O @cached usa automaticamente (metrica, normalize_ws, valor_json) como chave
            texto = _converter_para_texto_cached(
                metrica, normalize_ws, valor_json
            )
            
            return texto
            
        except (TypeError, ValueError) as e:
            # Em caso de erro na serialização, executa conversão direta
            if not isinstance(valor, dict):
                valor_dict = {'_valor': valor}
            else:
                valor_dict = valor
            
            if metrica == 'bertscore':
                texto = Json2Texto.to_linear_text(valor_dict, normalize_whitespace=normalize_ws)
            else:
                texto = Json2Texto.to_natural_text(valor_dict, normalize_whitespace=normalize_ws)
            
            return texto

    @classmethod
    def _converter_pares_para_texto(cls, valor_true: Any, valor_pred: Any, 
                                     metrica: str, config: dict, alinhar: bool = True) -> Tuple[str, str]:
        """
        Converte um par de valores (true, pred) para texto, aplicando alinhamento de listas se necessário.
        
        OTIMIZAÇÃO: Centraliza lógica de alinhamento + conversão, evitando duplicação.
        
        Args:
            valor_true: valor verdadeiro/esperado
            valor_pred: valor predito
            metrica: tipo de métrica ('bertscore', 'rouge', 'rouge1', 'rouge2', 'levenshtein')
            config: configuração ajustada
            alinhar: se True, aplica alinhamento de listas quando aplicável
        
        Returns:
            tuple: (texto_true, texto_pred) - textos convertidos e opcionalmente alinhados
        
        Alinhamento de listas:
            - Só aplica se ambos os valores forem listas não-vazias
            - Só aplica se pelo menos uma lista contém objetos complexos (dict/list)
            - Usa algoritmo húngaro com métrica 'auto' para pareamento ótimo
            - Em caso de erro, continua sem alinhamento
        """
        # ═════════════════════════════════════════════════════════════════════════
        # ALINHAMENTO DE LISTAS (antes da conversão para texto)
        # ═════════════════════════════════════════════════════════════════════════
        if alinhar and isinstance(valor_pred, list) and isinstance(valor_true, list):
            # Só alinha se ambas as listas têm conteúdo
            if len(valor_pred) > 0 and len(valor_true) > 0:
                # Verifica se são listas de objetos complexos (não escalares)
                tem_objetos_pred = any(isinstance(item, (dict, list)) for item in valor_pred)
                tem_objetos_true = any(isinstance(item, (dict, list)) for item in valor_true)
                
                # Só alinha se pelo menos uma lista tiver objetos complexos
                if tem_objetos_pred or tem_objetos_true:
                    try:
                        # Usa 'auto' para seleção automática da métrica baseada no tamanho do texto
                        valor_pred, valor_true = cls.alinhar_listas(valor_pred, valor_true, metrica_alinhamento='auto')
                    except Exception as e:
                        # Se falhar o alinhamento, continua sem alinhar
                        pass
        
        # ═════════════════════════════════════════════════════════════════════════
        # CONVERSÃO PARA TEXTO
        # ═════════════════════════════════════════════════════════════════════════
        texto_true = cls._converter_para_texto(valor_true, metrica, config)
        texto_pred = cls._converter_para_texto(valor_pred, metrica, config)
        
        return texto_true, texto_pred

    @classmethod
    def _calcular_metrica(cls, texto_pred: str, texto_true: str, metrica: str, config: dict) -> dict:
        """
        Calcula métricas de similaridade entre dois textos.
        
        Args:
            texto_pred: texto predito
            texto_true: texto verdadeiro/esperado
            metrica: tipo de métrica ('bertscore', 'rouge', 'rouge1', 'rouge2', 'levenshtein')
            config: configuração ajustada
        
        Returns:
            dict com chaves:
                - P: precision (0-1)
                - R: recall (0-1)
                - F1: f1-score (0-1)
                - SIM: similaridade (0-1) - usado apenas por Levenshtein
        """
        if metrica == 'bertscore':
            importar('bert_score')
            # BERTScoreService retorna floats arredondados com decimais=3
            P, R, F1 = bscore([texto_pred], [texto_true], decimais = 3, max_workers=BS_MAX_WORKERS)
            return {
                'P': P[0],
                'R': R[0],
                'F1': F1[0]
            }
        
        elif metrica in ('rouge', 'rouge1', 'rouge2'):
            importar('rouge_score')
            # Mapeia para tipo ROUGE
            tipo_rouge = {'rouge': 'rougeL', 'rouge1': 'rouge1', 'rouge2': 'rouge2'}[metrica]
            
            scorer = rouge_scorer.RougeScorer(
                [tipo_rouge],
                use_stemmer=config.get('rouge_stemmer', True),
                split_summaries=True
            )
            scores = scorer.score(texto_true, texto_pred)
            score_obj = scores[tipo_rouge]
            
            return {
                'P': round(score_obj.precision, 3),
                'R': round(score_obj.recall, 3),
                'F1': round(score_obj.fmeasure, 3)
            }
        
        elif metrica == 'levenshtein':
            importar('Levenshtein')
            # Levenshtein ratio retorna similaridade [0,1]
            similaridade = Levenshtein.ratio(texto_true, texto_pred)
            # Levenshtein retorna apenas SIM (não usa P, R, F1)
            return {
                'SIM': round(similaridade, 3)
            }
        
        else:
            raise ValueError(f"Métrica '{metrica}' não suportada")

    @classmethod
    def _acuracia_estrutural(cls, campos_pred: dict, campos_true: dict) -> dict:
        """
        Calcula acurácia estrutural comparando apenas as chaves dos campos extraídos.
        
        Args:
            campos_pred: campos extraídos do JSON predito {campo: valor}
            campos_true: campos extraídos do JSON verdadeiro {campo: valor}
        
        Returns:
            dict com:
                - P: precision (campos corretos / total pred)
                - R: recall (campos corretos / total true)
                - F1: f1-score estrutural
                - paths_comuns: lista de campos presentes em ambos
                - paths_faltantes: lista de campos em true mas não em pred
                - paths_extras: lista de campos em pred mas não em true
        """
        chaves_pred = set(campos_pred.keys())
        chaves_true = set(campos_true.keys())
        
        comuns = chaves_pred & chaves_true
        faltantes = chaves_true - chaves_pred
        extras = chaves_pred - chaves_true
        
        total_comuns = len(comuns)
        total_pred = len(chaves_pred)
        total_true = len(chaves_true)
        
        precision = total_comuns / total_pred if total_pred > 0 else 0.0
        recall = total_comuns / total_true if total_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'P': round(precision, 3),
            'R': round(recall, 3),
            'F1': round(f1, 3),
            'paths_comuns': sorted(comuns),
            'paths_faltantes': sorted(faltantes),
            'paths_extras': sorted(extras)
        }

    @classmethod
    def rouge_scorer(cls, texto1: str, texto2: str, config: dict) -> float:
        """
        Calcula similaridade ROUGE entre dois textos
        config['rouge_stemmer'] = True/False - padrão True
        Retorna o F1-score da métrica ROUGE selecionada
        """
        importar('rouge_score')
        
        config = cls.__ajustar_config(config)
        
        # Determina qual métrica ROUGE usar baseado nos campos configurados
        if 'campos_rouge1' in config and config['campos_rouge1']:
            tipo = 'rouge1'
        elif 'campos_rouge2' in config and config['campos_rouge2']:
            tipo = 'rouge2'
        else:
            tipo = 'rougeL'  # padrão
        
        scorer = rouge_scorer.RougeScorer(
            [tipo], 
            use_stemmer=config['rouge_stemmer'], 
            split_summaries=True
        )
        scores = scorer.score(texto1, texto2)
        return scores[tipo].fmeasure  # Retorna F1-score


    @classmethod
    def comparar(
        cls, 
        pred_json: dict, 
        true_json: dict, 
        retornar_valores: bool = False, 
        id_origem = None, 
        config: Union[dict, None] = None
    ) -> dict:
        """
        Compara dois JSONs calculando métricas com suporte a múltiplas técnicas por campo.
        
        Arquitetura multi-métrica:
        1. Extração de campos por nível (config['nivel_campos'])
        2. Análise global - pode usar múltiplas métricas se (global) estiver nas listas
        3. Análise estrutural - pode usar múltiplas métricas se (estrutura) estiver nas listas
        4. Análise por campo - cada campo pode ter múltiplas métricas
        
        Args:
            pred_json: JSON previsto/gerado
            true_json: JSON verdadeiro/esperado
            retornar_valores: se True, inclui campo _VL com textos convertidos
            id_origem: identificador opcional para rastreabilidade
            config: dicionário de configuração
        
        Returns:
            dict com chaves no formato: <campo>_<metrica>_<sufixo>
            Exemplos:
                - (global)_bertscore_F1: 0.85
                - (global)_rouge_F1: 0.80
                - resumo_bertscore_F1: 0.90
                - resumo_rouge_F1: 0.88
        """
        config = cls.__ajustar_config(config)
        nivel_campos = config['nivel_campos']
        
        resultado = {}
        
        if id_origem is not None:
            resultado['id_origem'] = id_origem
        
        # 1. EXTRAÇÃO DE CAMPOS
        campos_pred = cls._extrair_campos_por_nivel(pred_json, nivel_campos)
        campos_true = cls._extrair_campos_por_nivel(true_json, nivel_campos)
        
        # 2. ANÁLISE GLOBAL - suporta múltiplas métricas
        metricas_global = cls._determinar_metricas_campo('(global)', config)
        
        for metrica in metricas_global:
            # Usa método otimizado que alinha + converte em uma única chamada
            texto_true, texto_pred = cls._converter_pares_para_texto(
                true_json, pred_json, metrica, config, alinhar=False  # Global não usa alinhamento
            )
            metricas = cls._calcular_metrica(texto_pred, texto_true, metrica, config)
            
            prefixo = f'(global)_{metrica}'
            
            # Levenshtein retorna apenas SIM
            if metrica == 'levenshtein':
                resultado[f'{prefixo}_SIM'] = metricas['SIM']
            else:
                # Outras métricas retornam P, R, F1
                resultado[f'{prefixo}_P'] = metricas['P']
                resultado[f'{prefixo}_R'] = metricas['R']
                resultado[f'{prefixo}_F1'] = metricas['F1']
            
            if retornar_valores:
                resultado[f'{prefixo}_VL'] = {'pred': texto_pred, 'true': texto_true}
        
        # 3. ANÁLISE ESTRUTURAL - suporta múltiplas métricas
        # IMPORTANTE: Levenshtein não é aplicado em análise estrutural (compara apenas chaves)
        metricas_estrutura = cls._determinar_metricas_campo('(estrutura)', config)
        
        for metrica in metricas_estrutura:
            estrutura = cls._acuracia_estrutural(campos_pred, campos_true)
            
            prefixo = f'(estrutura)_{metrica}'
            resultado[f'{prefixo}_P'] = estrutura['P']
            resultado[f'{prefixo}_R'] = estrutura['R']
            resultado[f'{prefixo}_F1'] = estrutura['F1']
            
            if metrica == metricas_estrutura[0]:
                resultado['estrutura_detalhes'] = {
                    'paths_comuns': estrutura['paths_comuns'],
                    'paths_faltantes': estrutura['paths_faltantes'],
                    'paths_extras': estrutura['paths_extras']
                }
                
                if retornar_valores:
                    resultado[f'{prefixo}_VL'] = {
                        'pred': list(campos_pred.keys()),
                        'true': list(campos_true.keys())
                    }
        
        # 4. ANÁLISE POR CAMPO - cada campo pode ter múltiplas métricas
        todos_campos = set(campos_pred.keys()) | set(campos_true.keys())
        
        for campo in sorted(todos_campos):
            valor_pred = campos_pred.get(campo)
            valor_true = campos_true.get(campo)
            
            metricas_campo = cls._determinar_metricas_campo(campo, config)
            
            for metrica in metricas_campo:
                prefixo = f'{campo}_{metrica}'
                
                # Campo ausente: métricas zeradas
                if valor_pred is None or valor_true is None:
                    if metrica == 'levenshtein':
                        # Levenshtein retorna apenas SIM
                        resultado[f'{prefixo}_SIM'] = 0.0
                    else:
                        # Outras métricas retornam P, R, F1
                        resultado[f'{prefixo}_P'] = 0.0
                        resultado[f'{prefixo}_R'] = 0.0
                        resultado[f'{prefixo}_F1'] = 0.0
                    
                    if retornar_valores:
                        resultado[f'{prefixo}_VL'] = {
                            'pred': None if valor_pred is None else cls._converter_para_texto(valor_pred, metrica, config),
                            'true': None if valor_true is None else cls._converter_para_texto(valor_true, metrica, config)
                        }
                    continue
                
                # ═════════════════════════════════════════════════════════════
                # CONVERSÃO COM ALINHAMENTO AUTOMÁTICO
                # Método otimizado: alinha + converte em uma única chamada
                # ═════════════════════════════════════════════════════════════
                texto_true, texto_pred = cls._converter_pares_para_texto(
                    valor_true, valor_pred, metrica, config, alinhar=True
                )
                
                metricas_resultado = cls._calcular_metrica(texto_pred, texto_true, metrica, config)
                
                if metrica == 'levenshtein':
                    # Levenshtein retorna apenas SIM
                    resultado[f'{prefixo}_SIM'] = metricas_resultado['SIM']
                else:
                    # Outras métricas retornam P, R, F1
                    resultado[f'{prefixo}_P'] = metricas_resultado['P']
                    resultado[f'{prefixo}_R'] = metricas_resultado['R']
                    resultado[f'{prefixo}_F1'] = metricas_resultado['F1']
                
                if retornar_valores:
                    resultado[f'{prefixo}_VL'] = {'pred': texto_pred, 'true': texto_true}
        
        return resultado

    @classmethod
    def hash_string_sha1(cls, texto):
        ''' retorna o sha1 do texto ou json recebido '''
        if isinstance(texto, bytes):
          _txt = texto
        elif isinstance(texto, dict):
           _txt = json.dumps(texto, sort_keys = True).encode("utf-8")
        else:
           _txt = '|'.join([str(_) for _ in texto]) if type(texto) is list else str(texto)
           _txt = _txt.encode('utf-8')
        return hashlib.sha1(_txt).hexdigest()

class JsonAnaliseDataFrame():
    """
    Compara JSONs em lote e gera DataFrame com métricas.
    
    Recebe JsonAnaliseDados com todos os dados necessários para análise.
    
    Args:
        dados_analise: JsonAnaliseDados - container com dados, rotulos, tokens, etc
        config: dict - configuração do JsonAnalise (campos_bertscore, campos_rouge, etc)
        pasta_analises: str - pasta para salvar JSONs de análise individual
        max_workers: int - número de threads para processamento paralelo
        filtro_callable: function - filtro opcional filtro(dict, origem:bool) -> dict
        incluir_valores_analise: bool - inclui nos jsons de análise os textos convertidos
        gerar_exemplos_md: bool - gera arquivo Markdown com exemplos
        max_exemplos_md_por_metrica: int - máximo de exemplos por métrica no MD
    
    Example:
        >>> dados_analise = JsonAnaliseDados(dados, rotulos, tokens, avaliacao_llm, ...)
        >>> analisador = JsonAnaliseDataFrame(dados_analise, config={...})
        >>> df = analisador.to_df()
    """
     
    def __init__(self, 
                 dados_analise: 'JsonAnaliseDados',
                 config: dict = None,
                 pasta_analises: str = 'analises_json', 
                 max_workers: int = 4, 
                 filtro_callable = None,
                 incluir_valores_analise: bool = False,
                 gerar_exemplos_md: bool = True,
                 max_exemplos_md_por_metrica: int = 5):
        importar('pandas')
        
        # Valida que recebeu JsonAnaliseDados
        if not isinstance(dados_analise, JsonAnaliseDados):
            raise TypeError(
                f"Esperado JsonAnaliseDados, recebido {type(dados_analise).__name__}. "
                f"Use: JsonAnaliseDados(dados, rotulos, ...) antes de criar JsonAnaliseDataFrame"
            )
        
        # Valida consistência dos dados
        dados_analise.validar()
        
        # Armazena o container de dados - única fonte de verdade
        self.dados_analise = dados_analise
        
        # Configurações
        self.config = config
        self.pasta_analises = pasta_analises
        self.filtro_callable = filtro_callable
        self.max_workers = max_workers
        
        self._resultados = None  # Cache do DataFrame
        self._incluir_valores_analise = incluir_valores_analise

        # define o limite de workers do BERTScoreService
        global BS_MAX_WORKERS
        BS_MAX_WORKERS = self.max_workers
        
        # Controle de exemplos para Markdown
        self.gerar_exemplos_md = gerar_exemplos_md
        self.max_exemplos_md_por_metrica = max_exemplos_md_por_metrica
        self._exemplos_contador = {}  # {(campo, tecnica, metrica): count}
        self._exemplos_lock = threading.Lock()  # Thread-safe para append
        self._arquivo_exemplos_md = None  # Será definido na exportação
        self._lock = threading.Lock()  # Lock global para escrita thread-safe
    
    # ═════════════════════════════════════════════════════════════════════════
    # Properties para acesso aos dados via dados_analise
    # ═════════════════════════════════════════════════════════════════════════
    
    @property
    def dados(self):
        """Acessa dados via dados_analise"""
        return self.dados_analise.dados
    
    @property
    def rotulos(self):
        """Acessa rótulos via dados_analise"""
        return self.dados_analise.rotulos
    
    @property
    def tokens(self):
        """Acessa tokens via dados_analise"""
        return self.dados_analise.tokens
    
    @property
    def avaliacao_llm(self):
        """Acessa avaliações LLM via dados_analise"""
        return self.dados_analise.avaliacao_llm
    
    @property
    def observabilidade(self):
        """Acessa observabilidade via dados_analise"""
        return self.dados_analise.observabilidade
    
    def _criar_dataframe_tokens(self):
        """
        Cria DataFrame consolidado de tokens a partir da lista self.tokens.
        
        Returns:
            DataFrame com colunas: id_peca e colunas de tokens por modelo
            Exemplo: id_peca, GPT5_input, GPT5_output, GPT5_total, GPT5_cache, ...
        """
        if self.tokens is None or len(self.tokens) == 0:
            return None
        
        import pandas as pd
        
        # Consolida todos os dicts em um DataFrame
        df_tokens = pd.DataFrame(self.tokens)
        
        # Usa nome do campo ID configurado
        nome_campo_id = self.dados_analise.config.nome_campo_id
        
        # Reordena colunas: campo ID primeiro, depois agrupadas por modelo
        if nome_campo_id not in df_tokens.columns:
            return None
        
        # Extrai todos os rótulos de modelos (exceto campo ID)
        colunas_tokens = [col for col in df_tokens.columns if col != nome_campo_id]
        
        # Agrupa colunas por modelo e tipo de token
        # Formato esperado: <modelo>_<tipo> (ex: GPT5_input, GPT5_output, etc)
        colunas_ordenadas = [nome_campo_id]
        
        # Identifica todos os modelos únicos
        modelos = set()
        for col in colunas_tokens:
            if '_' in col:
                modelo = col.rsplit('_', 1)[0]
                modelos.add(modelo)
        
        # Para cada modelo, adiciona suas colunas de tokens na ordem: input, output, total, cache, reason, finish
        tipos_ordem = ['input', 'output', 'total', 'cache', 'reason', 'finish']
        for modelo in sorted(modelos):
            for tipo in tipos_ordem:
                col_nome = f'{modelo}_{tipo}'
                if col_nome in df_tokens.columns:
                    colunas_ordenadas.append(col_nome)
        
        # Adiciona colunas restantes que não foram classificadas
        for col in colunas_tokens:
            if col not in colunas_ordenadas:
                colunas_ordenadas.append(col)
        
        return df_tokens[colunas_ordenadas]
    
    def _criar_dataframe_avaliacao_llm(self):
        """
        Cria DataFrame consolidado de avaliações LLM a partir da lista self.avaliacao_llm.
        Remove colunas que estão completamente vazias (todos valores 0, None ou '').
        
        Agora suporta métricas por campo:
        - Métricas globais: modelo_P, modelo_R, modelo_F1, modelo_nota, modelo_explicacao
        - Métricas por campo: modelo_campo_P, modelo_campo_R, modelo_campo_F1
        
        Returns:
            DataFrame com colunas ordenadas: id_peca, métricas globais, métricas por campo
            None se não houver dados
        """
        if self.avaliacao_llm is None or len(self.avaliacao_llm) == 0:
            return None
        
        import pandas as pd
        
        # Consolida todos os dicts em um DataFrame
        df_avaliacao = pd.DataFrame(self.avaliacao_llm)
        
        # Usa nome do campo ID configurado
        nome_campo_id = self.dados_analise.config.nome_campo_id
        
        # Verifica se tem coluna de ID
        if nome_campo_id not in df_avaliacao.columns:
            return None
        
        # Remove colunas completamente vazias (todos None, 0, '' ou NaN)
        colunas_manter = [nome_campo_id]
        for col in df_avaliacao.columns:
            if col == nome_campo_id:
                continue
            
            # Verifica se a coluna tem algum valor significativo
            valores = df_avaliacao[col].dropna()
            if len(valores) == 0:
                continue  # Coluna vazia
            
            # Verifica se todos são 0 ou string vazia
            tem_valor = False
            for val in valores:
                if isinstance(val, str):
                    if val.strip():  # String não-vazia
                        tem_valor = True
                        break
                elif isinstance(val, (int, float)):
                    if val != 0:  # Número diferente de zero
                        tem_valor = True
                        break
                else:
                    tem_valor = True  # Outro tipo
                    break
            
            if tem_valor:
                colunas_manter.append(col)
        
        if len(colunas_manter) <= 1:  # Apenas campo ID
            return None
        
        # Reordena colunas: campo ID primeiro, depois agrupadas por modelo e campo
        df_avaliacao = df_avaliacao[colunas_manter]
        
        # Identifica estrutura: modelos e campos
        # Formato: modelo_P, modelo_R, modelo_F1 (globais)
        #          modelo_campo_P, modelo_campo_R, modelo_campo_F1 (por campo)
        estrutura = {}  # {modelo: {'globais': [...], 'campos': {campo: [...]}}}
        
        for col in colunas_manter[1:]:  # Pula id_peca
            if '_' not in col:
                continue
            
            partes = col.split('_')
            
            # Métricas globais: modelo_metrica (ex: agentes_P, agentes_explicacao)
            if len(partes) == 2:
                modelo, metrica = partes
                if modelo not in estrutura:
                    estrutura[modelo] = {'globais': [], 'campos': {}}
                estrutura[modelo]['globais'].append((metrica, col))
            
            # Métricas por campo: modelo_campo_metrica (ex: agentes_tema_P)
            elif len(partes) >= 3:
                modelo = partes[0]
                metrica = partes[-1]  # Última parte (P, R, F1)
                campo = '_'.join(partes[1:-1])  # Meio (pode ter underscores)
                
                if modelo not in estrutura:
                    estrutura[modelo] = {'globais': [], 'campos': {}}
                if campo not in estrutura[modelo]['campos']:
                    estrutura[modelo]['campos'][campo] = []
                estrutura[modelo]['campos'][campo].append((metrica, col))
        
        # Monta ordem das colunas: id_peca, depois por modelo (globais + campos)
        colunas_ordenadas = [nome_campo_id]
        
        # Ordem preferencial de métricas
        ordem_metricas_globais = ['P', 'R', 'F1', 'nota', 'explicacao']
        ordem_metricas_campos = ['P', 'R', 'F1']
        
        for modelo in sorted(estrutura.keys()):
            info = estrutura[modelo]
            
            # Adiciona métricas globais primeiro
            for metrica in ordem_metricas_globais:
                for m, col in info['globais']:
                    if m == metrica:
                        colunas_ordenadas.append(col)
            
            # Adiciona outras métricas globais não classificadas
            for metrica, col in info['globais']:
                if col not in colunas_ordenadas:
                    colunas_ordenadas.append(col)
            
            # Adiciona métricas por campo (ordenadas alfabeticamente por campo)
            for campo in sorted(info['campos'].keys()):
                for metrica in ordem_metricas_campos:
                    for m, col in info['campos'][campo]:
                        if m == metrica:
                            colunas_ordenadas.append(col)
        
        # Adiciona colunas restantes que não foram classificadas
        for col in colunas_manter:
            if col not in colunas_ordenadas:
                colunas_ordenadas.append(col)
        
        return df_avaliacao[colunas_ordenadas]
    
    def _criar_dataframe_observabilidade(self):
        """
        Cria DataFrame consolidado de observabilidade a partir da lista self.observabilidade.
        Remove colunas que estão completamente vazias (todos valores 0, None ou '').
        
        Returns:
            DataFrame com colunas: id_peca e colunas de observabilidade por modelo
            Exemplo: id_peca, modelo1_SEG, modelo1_REV, modelo1_AGT, modelo1_IT, modelo1_QTD,
                     modelo1_campo_BYTES, modelo1_AgenteCampo_SEG, modelo1_AgenteCampo_IT, ...
            Sufixos: _SEG (tempo), _REV (revisões), _AGT (agentes), _IT (iterações), 
                     _OK (sucesso), _QTD (campos preenchidos), _campo_BYTES (bytes por campo)
            None se não houver dados
        """
        observabilidade = self.dados_analise.observabilidade
        if observabilidade is None or len(observabilidade) == 0:
            return None
        
        import pandas as pd
        
        # Consolida todos os dicts em um DataFrame
        df_obs = pd.DataFrame(observabilidade)
        
        # Usa nome do campo ID configurado
        nome_campo_id = self.dados_analise.config.nome_campo_id
        
        # Verifica se tem coluna de ID
        if nome_campo_id not in df_obs.columns:
            return None
        
        # Remove colunas completamente vazias (todos None, 0, '' ou NaN)
        colunas_manter = [nome_campo_id]
        for col in df_obs.columns:
            if col == nome_campo_id:
                continue
            
            # Verifica se a coluna tem algum valor significativo
            valores = df_obs[col].dropna()
            if len(valores) == 0:
                continue  # Coluna vazia
            
            # Verifica se todos são 0 ou string vazia
            tem_valor = False
            for val in valores:
                if isinstance(val, str):
                    if val.strip():  # String não-vazia
                        tem_valor = True
                        break
                elif isinstance(val, (int, float)):
                    if val != 0:  # Número diferente de zero
                        tem_valor = True
                        break
                else:
                    tem_valor = True  # Outro tipo
                    break
            
            if tem_valor:
                colunas_manter.append(col)
        
        if len(colunas_manter) <= 1:  # Apenas campo ID
            return None
        
        # Reordena colunas: campo ID primeiro, depois agrupadas por modelo
        df_obs = df_obs[colunas_manter]
        
        # Agrupa colunas por modelo e tipo (formato: modelo_metrica ou modelo_agente_metrica)
        colunas_ordenadas = [nome_campo_id]
        
        # Identifica todos os modelos únicos
        modelos = set()
        for col in colunas_manter[1:]:  # Pula id_peca
            if '_' in col:
                # Extrai modelo (primeira parte antes do _)
                modelo = col.split('_')[0]
                modelos.add(modelo)
        
        # Para cada modelo, ordena colunas:
        # 1. Métricas globais: SEG, REV, AGT, IT, revisoes, QTD
        # 2. Métricas de campos base: campo1_BYTES, campo2_BYTES, ...
        # 3. Métricas por agente: AgenteName_SEG, AgenteName_IT, AgenteName_OK
        for modelo in sorted(modelos):
            # Primeiro: métricas globais
            metricas_globais = ['SEG', 'REV', 'AGT', 'IT', 'QTD']
            for metrica in metricas_globais:
                col_nome = f'{modelo}_{metrica}'
                if col_nome in df_obs.columns:
                    colunas_ordenadas.append(col_nome)
            
            # Segundo: métricas de bytes por campo (*_BYTES)
            colunas_bytes = [col for col in colunas_manter 
                           if col.startswith(f'{modelo}_') 
                           and col.endswith('_BYTES')
                           and col not in colunas_ordenadas]
            for col in sorted(colunas_bytes):
                colunas_ordenadas.append(col)
            
            # Terceiro: métricas por agente (identificadas por ter mais de 2 underscores ou nome de agente)
            colunas_agente = [col for col in colunas_manter 
                            if col.startswith(f'{modelo}_') 
                            and col not in colunas_ordenadas
                            and not col.endswith('_BYTES')]
            
            # Agrupa por agente
            agentes = {}
            for col in colunas_agente:
                # Formato: modelo_AgenteNome_metrica
                partes = col.split('_', 2)  # Divide em no máximo 3 partes
                if len(partes) == 3:
                    _, agente, metrica = partes
                    if agente not in agentes:
                        agentes[agente] = []
                    agentes[agente].append((col, metrica))
            
            # Ordena agentes e suas métricas
            for agente in sorted(agentes.keys()):
                # Ordena métricas na ordem: SEG (tempo), IT (iterações), OK (sucesso)
                ordem_metricas = ['SEG', 'IT', 'OK']
                for metrica_tipo in ordem_metricas:
                    for col, metrica in agentes[agente]:
                        if metrica == metrica_tipo:
                            colunas_ordenadas.append(col)
                
                # Adiciona métricas restantes
                for col, metrica in agentes[agente]:
                    if col not in colunas_ordenadas:
                        colunas_ordenadas.append(col)
        
        # Adiciona colunas restantes que não foram classificadas
        for col in colunas_manter:
            if col not in colunas_ordenadas:
                colunas_ordenadas.append(col)
        
        return df_obs[colunas_ordenadas]
    
    def _incluir_exemplo_metrica(self, campo: str, tecnica: str, metrica: str, 
                                  texto_pred: str, texto_true: str, 
                                  valor_metrica: float, id_origem: str = None,
                                  modelo: str = None):
        """
        Avalia se precisa adicionar exemplo para essa métrica+campo e faz append no arquivo MD.
        
        Args:
            campo: nome do campo (ex: '(global)', 'resumo', 'fatos')
            tecnica: técnica usada (ex: 'bertscore', 'rouge', 'rouge1', 'rouge2', 'levenshtein')
            metrica: métrica específica (ex: 'P', 'R', 'F1', 'SIM')
            texto_pred: texto predito
            texto_true: texto verdadeiro
            valor_metrica: valor calculado da métrica
            id_origem: identificador do documento (opcional)
            modelo: nome do modelo que gerou a predição (opcional)
        """
        if not self.gerar_exemplos_md or self._arquivo_exemplos_md is None:
            return
        
        # Cria chave única para controlar contagem
        chave = (campo, tecnica, metrica)
        
        with self._exemplos_lock:
            # Verifica se já tem exemplos suficientes
            count = self._exemplos_contador.get(chave, 0)
            if count >= self.max_exemplos_md_por_metrica:
                return
            
            # Incrementa contador
            self._exemplos_contador[chave] = count + 1
            
            # Prepara conteúdo do exemplo
            linhas = []
            
            # Cabeçalho apenas no primeiro exemplo desta métrica
            if count == 0:
                linhas.append(f"\n## {campo} - {tecnica.upper()} - {metrica}\n")
            
            linhas.append(f"\n### Exemplo {count + 1}")
            if id_origem:
                linhas.append(f"**ID:** `{id_origem}`")
            if modelo:
                linhas.append(f"**Modelo:** `{modelo}`")
            linhas.append(f"**Valor {metrica}:** `{valor_metrica:.4f}`\n")
            
            # Textos comparados (trunca se muito longo)
            texto_pred_exibir = texto_pred[:MAX_STRING_MD] + ('...' if len(texto_pred) > MAX_STRING_MD else '')
            texto_true_exibir = texto_true[:MAX_STRING_MD] + ('...' if len(texto_true) > MAX_STRING_MD else '')
            
            linhas.append("**Texto Predito:**")
            linhas.append("```")
            linhas.append(texto_pred_exibir)
            linhas.append("```\n")
            
            linhas.append("**Texto Esperado:**")
            linhas.append("```")
            linhas.append(texto_true_exibir)
            linhas.append("```\n")
            
            linhas.append("---\n")
            
            # Faz append no arquivo
            try:
                with open(self._arquivo_exemplos_md, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(linhas))
            except Exception as e:
                # Silenciosamente ignora erros de escrita para não impactar o processamento
                pass

    def _filtro_callable(self, json_dict: dict, origem: bool) -> dict:
        """Aplica filtro callable se existir. origem=True para true_json, False para pred"""
        if self.filtro_callable is not None and callable(self.filtro_callable):
            return self.filtro_callable(json_dict, origem=origem)
        return json_dict

    def _comparar_linha(self, linha: dict, rotulos: List[str]) -> dict:
        """
        Compara uma linha (dict) contendo id, True e modelos.
        Retorna dict com id e resultados de comparação para cada modelo.
        """
        rotulo_id = rotulos[0]
        rotulo_true = rotulos[1]
        
        id_origem = linha[rotulo_id]
        true_json_original = linha[rotulo_true]
        
        # Aplica filtro ao true_json
        true_json = self._filtro_callable(true_json_original, origem=True)
        if true_json is None:
            return None  # Pula linha
        
        resultados_linha = {rotulo_id: id_origem}
        pula_linha = True
        
        # Compara com cada predição
        for rotulo_modelo in rotulos[2:]:
            if rotulo_modelo not in linha:
                continue
            
            pred_json_original = linha[rotulo_modelo]
            
            # Aplica filtro ao pred_json
            pred_json = self._filtro_callable(pred_json_original, origem=False)
            if pred_json is None:
                continue  # Pula esta predição
            
            pula_linha = False
            
            # Compara usando JsonAnalise.comparar() - sempre inclui valores para poder extrair exemplos
            incluir_valores = self._incluir_valores_analise or self.gerar_exemplos_md
            resultado = JsonAnalise.comparar(
                pred_json=pred_json,
                true_json=true_json,
                config=self.config,
                retornar_valores=incluir_valores,
                id_origem=id_origem
            )
            
            # OTIMIZAÇÃO: Extrai exemplos de forma thread-safe se MD habilitado
            if self.gerar_exemplos_md and self._arquivo_exemplos_md and self._lock:
                with self._lock:
                    self._extrair_exemplos_resultado(resultado, id_origem, rotulo_modelo)
            
            # Remove valores se não foram solicitados originalmente
            if not self._incluir_valores_analise and incluir_valores:
                resultado = {k: v for k, v in resultado.items() if not k.endswith('_VL')}
            
            # Armazena resultado com chave True_Modelo
            campo_resultado = f'{rotulo_true}_{rotulo_modelo}'
            resultados_linha[campo_resultado] = resultado
        
        return None if pula_linha else resultados_linha

    def _inicializar_arquivo_md(self, arquivo_exemplos: str):
        """Inicializa o arquivo Markdown para exemplos"""
        if not self.gerar_exemplos_md:
            return
        
        # Define nome do arquivo MD baseado no arquivo de exportação
        self._arquivo_exemplos_md = arquivo_exemplos

        # Cria cabeçalho do arquivo
        try:
            with open(self._arquivo_exemplos_md, 'w', encoding='utf-8') as f:
                f.write(f"# Exemplos de Métricas de Comparação\n\n")
                f.write(f"**Data de Geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Documentos Analisados:** {len(self.dados)}\n\n")
                f.write(f"**Máximo de Exemplos por Métrica:** {self.max_exemplos_md_por_metrica}\n\n")
                f.write("---\n\n")
                f.write("Este arquivo contém exemplos de comparações realizadas, organizados por campo, técnica e métrica.\n\n")
            print(f"📝 Arquivo de exemplos criado: {self._arquivo_exemplos_md}")
        except Exception as e:
            print(f"⚠️  Aviso: Não foi possível criar arquivo de exemplos MD: {e}")
            self.gerar_exemplos_md = False
    
    def _extrair_exemplos_resultado(self, resultado: dict, id_origem: str, modelo: str):
        """
        Extrai exemplos do resultado de comparação e adiciona ao arquivo MD.
        
        Args:
            resultado: dict com métricas no formato campo_tecnica_metrica: valor
            id_origem: identificador do documento
            modelo: nome do modelo
        """
        if not self.gerar_exemplos_md or not self._arquivo_exemplos_md:
            return
        
        # Procura por chaves que terminam com _VL (contêm os textos)
        for chave, valor in resultado.items():
            if not chave.endswith('_VL') or not isinstance(valor, dict):
                continue
            
            # Extrai informações da chave: campo_tecnica_VL
            # Exemplo: (global)_bertscore_VL -> campo=(global), tecnica=bertscore
            partes = chave.rsplit('_', 1)[0]  # Remove _VL
            partes_split = partes.rsplit('_', 1)  # Separa campo_tecnica
            
            if len(partes_split) == 2:
                campo, tecnica = partes_split
            else:
                continue  # Formato inesperado
            
            texto_pred = valor.get('pred', '')
            texto_true = valor.get('true', '')
            
            # Ignora se não há textos
            if not texto_pred or not texto_true:
                continue
            
            # Converte listas para string
            if isinstance(texto_pred, list):
                texto_pred = ', '.join(str(x) for x in texto_pred)
            if isinstance(texto_true, list):
                texto_true = ', '.join(str(x) for x in texto_true)
            
            # Busca as métricas correspondentes
            for metrica_nome in ['F1', 'P', 'R', 'SIM']:
                chave_metrica = f'{campo}_{tecnica}_{metrica_nome}'
                if chave_metrica in resultado:
                    valor_metrica = resultado[chave_metrica]
                    self._incluir_exemplo_metrica(
                        campo=campo,
                        tecnica=tecnica,
                        metrica=metrica_nome,
                        texto_pred=str(texto_pred),
                        texto_true=str(texto_true),
                        valor_metrica=valor_metrica,
                        id_origem=id_origem,
                        modelo=modelo
                    )

    def to_df(self):
        """Executa comparações e retorna DataFrame com métricas"""
        import pandas as pd
        
        if self._resultados is not None:
            return self._resultados
        
        # Limpa arquivos antigos antes de processar
        self._limpar_arquivos_analise()
        
        # Inicializa arquivo MD se ainda não foi inicializado
        if self.gerar_exemplos_md and self._arquivo_exemplos_md is None:
            arquivo_padrao = os.path.join(
                self.pasta_analises,
                'analise_exemplos.md'
            )
            self._inicializar_arquivo_md(arquivo_padrao)
        
        # OTIMIZAÇÃO: Para poucos dados (< 10), processa sequencialmente
        # Evita overhead de threads e carregamento múltiplo do BERTScore
        usar_paralelo = len(self.dados) >= 10 and self.max_workers > 1
        
        _resultados = []
        
        if usar_paralelo:
            # Processamento paralelo para grandes volumes
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                progresso = tqdm(total=len(self.dados), desc="Comparando JSONs", unit="linha")
                
                for linha in self.dados:
                    future = executor.submit(self._comparar_linha, linha=linha, rotulos=self.rotulos)
                    futures[future] = linha
                    progresso.update(1)
                progresso.close()
                
                progresso = tqdm(total=len(futures), desc="Consolidando DataFrame", unit="linha")
                for future in as_completed(futures):
                    resultado = future.result()
                    progresso.update(1)
                    if resultado is not None:
                        _resultados.append(resultado)
                progresso.close()
        else:
            # Processamento sequencial para pequenos volumes (mais rápido!)
            progresso = tqdm(self.dados, desc="Processando arquivos", unit="arquivo")
            for linha in progresso:
                resultado = self._comparar_linha(linha=linha, rotulos=self.rotulos)
                if resultado is not None:
                    _resultados.append(resultado)
        
        # OTIMIZAÇÃO: Salva JSONs em lote ao invés de um por um
        if self.pasta_analises and len(_resultados) > 0:
            self._salvar_analises_lote(_resultados)
        
        # Transforma resultados em DataFrame
        _linhas_df = [self._linha_para_df_row(res) for res in _resultados]
        
        self._resultados = pd.DataFrame(_linhas_df)
        return self._resultados
    
    def _salvar_analises_lote(self, resultados: list):
        """Salva análises individuais em lote (mais eficiente que uma por vez)"""
        os.makedirs(self.pasta_analises, exist_ok=True)
        rotulo_id = self.rotulos[0]
        
        for res in resultados:
            id_origem = res.get(rotulo_id)
            arquivo = os.path.join(self.pasta_analises, f'analise_{id_origem}.json')
            try:
                with open(arquivo, 'w', encoding='utf-8') as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
            except Exception:
                pass  # Silenciosamente ignora erros de I/O
    
    def _linha_para_df_row(self, resultado: dict) -> dict:
        """
        Transforma resultado de comparação em linha de DataFrame.
        
        Nova estrutura com múltiplas métricas por campo:
        Entrada:
        {
            'id': 1,
            'True_Modelo1': {
                '(global)_bertscore_P': 0.9,
                '(global)_rouge_F1': 0.85,
                'campo_bertscore_F1': 0.88,
                'campo_rouge_F1': 0.92,
                ...
            }
        }
        
        Saída:
        {
            'id (True)': 1,
            'Modelo1_(global)_bertscore_P': 0.9,
            'Modelo1_(global)_rouge_F1': 0.85,
            'Modelo1_campo_bertscore_F1': 0.88,
            ...
        }
        """
        linha_df = {}
        rotulo_id = self.rotulos[0]
        rotulo_true = self.rotulos[1]
        
        for chave, valor in resultado.items():
            if chave == rotulo_id:
                linha_df[f'{rotulo_id} ({rotulo_true})'] = valor
                continue
            
            if isinstance(valor, dict):
                nome_modelo = chave.replace(f'{rotulo_true}_', '')
                
                for metrica, val_metrica in valor.items():
                    if metrica in ('id_origem', 'estrutura_detalhes') or metrica.endswith('_VL'):
                        continue
                    
                    linha_df[f'{nome_modelo}_{metrica}'] = val_metrica
        
        return linha_df
    
    def estatisticas_globais(self):
        """
        Calcula estatísticas agregadas para cada métrica por modelo e técnica.
        Com suporte a múltiplas métricas, um campo pode aparecer várias vezes.
        
        Returns:
            DataFrame com colunas: modelo, metrica, tecnica, mean, median, std
        """
        import pandas as pd
        
        if self._resultados is None:
            self.to_df()
        
        modelos = [r for r in self.rotulos[2:]]
        
        # Mapeia técnicas dos nomes das métricas
        def extrair_tecnica(nome_metrica):
            """Extrai nome da técnica do formato campo_tecnica_sufixo"""
            if '_bertscore_' in nome_metrica:
                return 'BERTScore'
            elif '_rouge2_' in nome_metrica or 'rouge2' in nome_metrica:
                return 'ROUGE-2'
            elif '_rouge1_' in nome_metrica or 'rouge1' in nome_metrica:
                return 'ROUGE-1'
            elif '_rouge_' in nome_metrica or 'rouge_' in nome_metrica:
                return 'ROUGE-L'
            elif '_levenshtein_' in nome_metrica:
                return 'Levenshtein'
            elif '(estrutura)_' in nome_metrica:
                return 'Estrutural'
            else:
                config = JsonAnalise._JsonAnalise__ajustar_config(self.config)
                return config.get('metrica_global', 'rouge2').upper()
        
        stats = []
        
        for modelo in modelos:
            colunas_modelo = [c for c in self._resultados.columns if c.startswith(f'{modelo}_')]
            
            for col in colunas_modelo:
                metrica = col.replace(f'{modelo}_', '')
                
                valores = self._resultados[col].dropna()
                if len(valores) == 0:
                    continue
                
                tecnica = extrair_tecnica(metrica)
                
                stats.append({
                    'modelo': modelo,
                    'metrica': metrica,
                    'tecnica': tecnica,
                    'mean': round(float(valores.mean()), 3),
                    'median': round(float(valores.median()), 3),
                    'std': round(float(valores.std()), 3)
                })
        
        return pd.DataFrame(stats)
    
    def comparar_modelos(self, metrica: str = '(global)_F1'):
        """
        Compara diferentes modelos para uma métrica específica.
        
        Args:
            metrica: nome da métrica para comparar. Exemplos:
                - '(global)_P', '(global)_R', '(global)_F1', '(global)_LS'
                - '(estrutura)_P', '(estrutura)_R', '(estrutura)_F1', '(estrutura)_LS'
                - 'campo_P', 'campo_R', 'campo_F1', 'campo_LS'
                Aceita abreviações: 'P', 'R', 'F1', 'LS' (busca por (global)_metrica)
        
        Returns:
            DataFrame com ID e colunas da métrica para cada modelo
        """
        import pandas as pd
        
        if self._resultados is None:
            self.to_df()
        
        rotulo_id = self.rotulos[0]
        rotulo_true = self.rotulos[1]
        
        # Normaliza nome da métrica (aceita abreviações simples)
        if metrica in ['P', 'R', 'F1', 'LS']:
            metrica = f'(global)_{metrica}'
        
        # Busca colunas que terminam com a métrica desejada
        # As colunas têm formato: Modelo_metrica
        # Onde metrica pode ser:
        #   - (global)_rouge2_F1
        #   - (global)_bertscore_F1  
        #   - campo_rouge_F1
        # Precisamos buscar colunas que terminem com _metrica
        
        metrica_escaped = re.escape(metrica)
        pattern = f'.*_{metrica_escaped}$'
        
        colunas_metrica = []
        for col in self._resultados.columns:
            # Ignora a coluna de ID
            if col == f'{rotulo_id} ({rotulo_true})':
                continue
            
            # Busca colunas que terminem com _metrica
            if re.match(pattern, col):
                colunas_metrica.append(col)
        
        if not colunas_metrica:
            raise ValueError(
                f"Métrica '{metrica}' não encontrada no DataFrame.\n"
                f"Colunas disponíveis: {', '.join(sorted(set(c.split('_')[-1] for c in self._resultados.columns[1:])))}\n"
                f"Exemplos de métricas completas: '(global)_rouge2_F1', '(estrutura)_rouge1_P', 'campo_bertscore_F1'"
            )
        
        # Cria DataFrame com ID e colunas da métrica
        col_id = f'{rotulo_id} ({rotulo_true})'
        df_comparacao = self._resultados[[col_id] + colunas_metrica].copy()
        
        # Renomeia colunas para facilitar leitura
        # Com múltiplas técnicas: 'Modelo_(campo)_tecnica_metrica' -> 'Modelo (tecnica)'
        # Ex: 'GPT4_(global)_rouge2_F1' -> 'GPT4 (ROUGE-2)'
        rename_map = {}
        nomes_tecnicas = {
            'bertscore': 'BERTScore',
            'rouge2': 'ROUGE-2',
            'rouge1': 'ROUGE-1',
            'rouge': 'ROUGE-L',
            'levenshtein': 'Levenshtein'
        }
        
        # Contador para evitar nomes duplicados
        nomes_usados = {}
        
        for col in colunas_metrica:
            # Extrai modelo e técnica do nome da coluna
            # Formato: Modelo_campo_tecnica_sufixo
            partes = col.split('_')
            
            # Primeira parte é o modelo
            modelo = partes[0]
            
            # Identifica a técnica (busca por técnicas conhecidas)
            tecnica_encontrada = None
            for tec in nomes_tecnicas.keys():
                if f'_{tec}_' in col:
                    tecnica_encontrada = nomes_tecnicas[tec]
                    break
            
            # Nome base: Modelo (tecnica) ou apenas Modelo se não identificou técnica
            if tecnica_encontrada:
                novo_nome = f'{modelo} ({tecnica_encontrada})'
            else:
                novo_nome = modelo
            
            # CORREÇÃO: Evita nomes duplicados adicionando sufixo numérico
            if novo_nome in nomes_usados:
                nomes_usados[novo_nome] += 1
                novo_nome = f'{novo_nome} [{nomes_usados[novo_nome]}]'
            else:
                nomes_usados[novo_nome] = 0
            
            rename_map[col] = novo_nome
        
        df_comparacao.rename(columns=rename_map, inplace=True)
        
        return df_comparacao
    
    def exportar_csv(self, arquivo: str = None, separador: str = ',', incluir_estatisticas: bool = True) -> str:
        """
        Exporta o DataFrame para arquivo CSV.
        
        Args:
            arquivo: nome do arquivo (se None, usa 'analise_resultados.csv')
            separador: separador CSV (padrão: ',')
            incluir_estatisticas: se True, cria arquivo adicional com estatísticas globais
        
        Returns:
            caminho completo do arquivo gerado
        """
        if arquivo is None:
            arquivo = 'analise_resultados.csv'
        
        # Garante extensão .csv
        if not arquivo.endswith('.csv'):
            arquivo += '.csv'
        
        # Adiciona pasta se configurada
        if self.pasta_analises:
            os.makedirs(self.pasta_analises, exist_ok=True)
            arquivo = os.path.join(self.pasta_analises, arquivo)
        
        # Define o nome do arquivo MD baseado no CSV (se ainda não foi definido)
        if self.gerar_exemplos_md and self._arquivo_exemplos_md is None:
            self._arquivo_exemplos_md = os.path.splitext(arquivo)[0] + '_exemplos.md'

        # Gera DataFrame (inicializa MD internamente se necessário)
        if self._resultados is None:
            self.to_df()
        
        # Exporta resultados principais
        self._resultados.to_csv(arquivo, index=False, sep=separador, encoding='utf-8-sig')
        
        # Exporta estatísticas em arquivo separado
        if incluir_estatisticas:
            arquivo_stats = arquivo.replace('.csv', '.estatisticas.csv')
            stats = self.estatisticas_globais()
            stats.to_csv(arquivo_stats, index=False, sep=separador, encoding='utf-8-sig')
        
        # Exporta avaliação LLM se disponível
        df_avaliacao = self._criar_dataframe_avaliacao_llm()
        if df_avaliacao is not None:
            arquivo_avaliacao = arquivo.replace('.csv', '.avaliacao_llm.csv')
            df_avaliacao.to_csv(arquivo_avaliacao, index=False, sep=separador, encoding='utf-8-sig')
            print(f"   ✓ Avaliação LLM CSV: {arquivo_avaliacao}")
        
        return arquivo

    def exportar_excel(self, arquivo: str = None, incluir_estatisticas: bool = True, 
                      usar_formatacao_avancada: bool = True, congelar_paineis: bool = True,
                      gerar_graficos: bool = False) -> str:
        '''
        Exporta o DataFrame para arquivo Excel com múltiplas abas e formatação avançada.
        
        Args:
            arquivo: nome do arquivo (se None, usa 'analise_resultados.xlsx')
            incluir_estatisticas: se True, cria aba adicional com estatísticas
            usar_formatacao_avancada: se True, usa UtilPandasExcel com mapas de calor
            gerar_graficos: se True, gera gráficos boxplot automaticamente
        
        Returns:
            caminho completo do arquivo gerado
        '''
        import pandas as pd
        
        if arquivo is None:
            arquivo = 'analise_resultados.xlsx'
        
        # Garante extensão .xlsx
        if not arquivo.endswith('.xlsx'):
            arquivo += '.xlsx'
        
        # Adiciona pasta se configurada
        if self.pasta_analises:
            os.makedirs(self.pasta_analises, exist_ok=True)
            arquivo = os.path.join(self.pasta_analises, arquivo)
        
        # Define o nome do arquivo MD baseado no Excel (se ainda não foi definido)
        if self.gerar_exemplos_md and self._arquivo_exemplos_md is None:
            self._arquivo_exemplos_md = os.path.splitext(arquivo)[0] + '_exemplos.md'

        # Gera DataFrame (inicializa MD internamente se necessário)
        if self._resultados is None:
            self.to_df()
        
        # Usa formatação avançada se disponível e solicitada
        if usar_formatacao_avancada:
            try:
                return self._exportar_excel_formatado(arquivo, self._resultados, incluir_estatisticas, congelar_paineis, gerar_graficos)
            except ImportError:
                if str(os.getenv('UTILPANDAS','')).lower() in ('0','false','não','n'):
                   print("⚠️  UtilPandasExcel não disponível, usando exportação padrão")
                else:
                   raise ImportError("UtilPandasExcel não está instalado. Para continuar sem ele, configura UTILPANDAS=0 nas variáveis de ambiente!")
        
        # Exportação padrão (fallback)
        with pd.ExcelWriter(arquivo, engine='openpyxl') as writer:
            self._resultados.to_excel(writer, sheet_name='Resultados', index=False)
            
            if incluir_estatisticas:
                stats = self.estatisticas_globais()
                stats.to_excel(writer, sheet_name='Estatísticas', index=False)
                
                # Busca por qualquer métrica (global)_*_F1 disponível
                metricas_f1_global = stats[stats['metrica'].str.contains(r'\(global\)_.*_F1', regex=True)]['metrica'].unique()
                if len(metricas_f1_global) > 0:
                    try:
                        comp_f1 = self.comparar_modelos(metricas_f1_global[0])
                        comp_f1.to_excel(writer, sheet_name='Comparação_F1', index=False)
                    except ValueError:
                        pass
        
        return arquivo

    def _exportar_excel_formatado(self, arquivo: str, df_exportar, incluir_estatisticas: bool, 
                                  congelar_paineis: bool, gerar_graficos: bool = False) -> str:
        '''
        Exporta Excel com formatação avançada usando UtilPandasExcel.
        Cria abas separadas por técnica (ROUGE-2, BERTScore, etc.) e remove nome da técnica das colunas.
        Aplica mapas de calor em colunas de métricas.
        '''
        from util_pandas import UtilPandasExcel
        import re
        
        # Cria o exportador
        excel = UtilPandasExcel(arquivo, columns_auto_width=True, header_formatting=True)
        
        # ═══════════════════════════════════════════════════════════════════════
        # SEPARA DATAFRAME POR TÉCNICA
        # ═══════════════════════════════════════════════════════════════════════
        
        # Identifica todas as técnicas presentes nas colunas
        # Padrão: Modelo_campo_tecnica_metrica (ex: GPT4_(global)_rouge2_F1)
        tecnicas_encontradas = set()
        # A primeira coluna é sempre o ID (formato: "id (rotulo_true)")
        col_id = df_exportar.columns[0]
        
        for col in df_exportar.columns:
            if col == col_id:
                continue  # Pula coluna ID
            # Extrai técnica do nome: busca padrão _tecnica_ entre campo e métrica
            # Ex: GPT4_(global)_rouge2_F1 -> rouge2
            match = re.search(r'_(bertscore|rouge2|rouge1|rouge|levenshtein)_', col)
            if match:
                tecnicas_encontradas.add(match.group(1))
        
        # Mapeamento de técnicas para nomes legíveis
        nomes_tecnicas = {
            'bertscore': 'BERTScore',
            'rouge2': 'ROUGE-2',
            'rouge1': 'ROUGE-1',
            'rouge': 'ROUGE-L',
            'levenshtein': 'Levenshtein'
        }
        
        # Para cada técnica, cria uma aba separada
        for tecnica in sorted(tecnicas_encontradas):
            nome_aba = f'Resultados_{nomes_tecnicas.get(tecnica, tecnica)}'
            
            # Filtra colunas dessa técnica
            colunas_tecnica = [col_id]  # Sempre inclui ID
            for col in df_exportar.columns:
                if col == col_id:
                    continue
                if f'_{tecnica}_' in col:
                    colunas_tecnica.append(col)
            
            if len(colunas_tecnica) <= 1:
                continue  # Pula se só tem ID
            
            # Cria DataFrame filtrado
            df_tecnica = df_exportar[colunas_tecnica].copy()
            
            # Remove nome da técnica das colunas (ex: GPT4_(global)_rouge2_F1 -> GPT4_(global)_F1)
            rename_map = {}
            for col in df_tecnica.columns:
                if col == col_id:
                    continue  # Pula ID
                novo_nome = col.replace(f'_{tecnica}_', '_')
                rename_map[col] = novo_nome
            df_tecnica.rename(columns=rename_map, inplace=True)
            
            # Exporta aba
            excel.write_df(df_tecnica, nome_aba, auto_width_colums_list=True)
            if congelar_paineis:
                excel.congelar_painel(nome_aba, 1, 1)  # Congela primeira linha e coluna
            
            # Aplica cores nas colunas de métricas (P, R, F1, SIM)
            colunas_metricas = [col for col in df_tecnica.columns
                               if col != col_id and any(col.endswith(f'_{m}') for m in ['P', 'R', 'F1', 'SIM'])]
            
            if len(colunas_metricas) > 0 and len(df_tecnica) > 0:
                for col_name in colunas_metricas:
                    col_idx = df_tecnica.columns.get_loc(col_name)
                    
                    for row_idx in range(len(df_tecnica)):
                        valor = df_tecnica.iloc[row_idx, col_idx]
                        if valor is not None and not (isinstance(valor, float) and np.isnan(valor)) and isinstance(valor, (int, float)):
                            # Todas as métricas usam escala 0-1 (verde = melhor)
                            excel.write_cell_with_color(nome_aba, row_idx + 1, col_idx, valor,
                                                       min_value=0.0, mid_value=0.5, max_value=1.0)
        
        # ═══════════════════════════════════════════════════════════════════════
        # ABA DE ESTATÍSTICAS
        # ═══════════════════════════════════════════════════════════════════════
        if incluir_estatisticas:
            stats = self.estatisticas_globais()
            excel.write_df(stats, 'Estatísticas', auto_width_colums_list=True)
            if congelar_paineis:
                excel.congelar_painel('Estatísticas', 1, 1)
            
            # Aplica cores nas colunas 'mean', 'median' E 'std' das estatísticas
            colunas_stats_metricas = [col for col in ['mean', 'median', 'std'] if col in stats.columns]
            
            if len(colunas_stats_metricas) > 0 and len(stats) > 0:
                for col_name in colunas_stats_metricas:
                    col_idx = stats.columns.get_loc(col_name)
                    for idx, row in stats.iterrows():
                        valor = row[col_name]
                        
                        if valor is not None and not (isinstance(valor, float) and np.isnan(valor)) and isinstance(valor, (int, float)):
                            row_num = idx + 1
                            
                            if col_name == 'std':
                                # Std: menor é melhor (vermelho = alta variação)
                                _min, _mid, _max = 2.0, 0.5, 0.0
                            else:
                                # mean/median: maior é melhor (verde = alto)
                                _min, _mid, _max = 0.0, 0.5, 1.0
                            
                            excel.write_cell_with_color('Estatísticas', row_num, col_idx, valor,
                                                           min_value=_min, mid_value=_mid, max_value=_max)
            
            # Exporta comparação F1
            # Tenta encontrar uma métrica F1 global disponível
            try:
                # Busca por qualquer métrica (global)_*_F1 nas estatísticas
                stats = self.estatisticas_globais()
                metricas_f1_global = stats[stats['metrica'].str.contains(r'\(global\)_.*_F1', regex=True)]['metrica'].unique()
                
                if len(metricas_f1_global) > 0:
                    # Usa a primeira métrica F1 global encontrada
                    metrica_f1 = metricas_f1_global[0]
                    comp_f1 = self.comparar_modelos(metrica_f1)
                    
                    # CORREÇÃO: NÃO usar to_excel() - escrever manualmente para aplicar cores
                    worksheet = excel.writer.book.add_worksheet('Comparação_F1')
                    excel.writer.sheets['Comparação_F1'] = worksheet
                    
                    # Escreve cabeçalhos
                    for col_idx, col_name in enumerate(comp_f1.columns):
                        worksheet.write(0, col_idx, col_name, excel.WB_HEADER_FORMAT)
                    
                    if congelar_paineis:
                        excel.congelar_painel('Comparação_F1', 1, 1)

                    # Aplica cores em todas as colunas numéricas (exceto ID) - célula por célula
                    try:
                        df_numericas = comp_f1.select_dtypes(include=[np.number])
                        colunas_numericas = df_numericas.columns.tolist()
                        
                        if len(colunas_numericas) > 0 and len(comp_f1) > 0:
                            # Escreve dados linha por linha aplicando cores
                            for row_idx in range(len(comp_f1)):
                                for col_idx, col_name in enumerate(comp_f1.columns):
                                    valor = comp_f1.iloc[row_idx, col_idx]
                                    
                                    # Se é coluna numérica, aplica cor
                                    if col_name in colunas_numericas and valor is not None and not (isinstance(valor, float) and np.isnan(valor)):
                                        _min, _mid, _max = 0.0, 0.5, 1.0
                                        excel.write_cell_with_color('Comparação_F1', row_idx + 1, col_idx, valor,
                                                                       min_value=_min, mid_value=_mid, max_value=_max)
                                    else:
                                        # Escreve valor sem cor
                                        worksheet.write(row_idx + 1, col_idx, valor, excel.WB_DEFAULT_FORMAT)
                            
                    except Exception as e_colors:
                        print(f"⚠️  Aviso: Erro ao aplicar cores na aba 'Comparação_F1': {e_colors}")
                else:
                    print(f"⚠️  Aviso: Nenhuma métrica F1 global encontrada para criar aba 'Comparação_F1'")
            except ValueError as e:
                # Log do erro mas não impede a exportação
                print(f"⚠️  Aviso: Não foi possível criar aba 'Comparação_F1': {e}")
            except Exception as e:
                # Log de outros erros inesperados
                import traceback
                print(f"⚠️  Erro ao criar aba 'Comparação_F1': {type(e).__name__}: {e}")
                traceback.print_exc()
        
        # ═══════════════════════════════════════════════════════════════════════
        # ABA DE RESUMO DE TOKENS
        # ═══════════════════════════════════════════════════════════════════════
        if self.tokens is not None and len(self.tokens) > 0:
            df_tokens = self._criar_dataframe_tokens()
            if df_tokens is not None:
                # Escreve DataFrame normalmente
                excel.write_df(df_tokens, 'Resumo_Tokens', auto_width_colums_list=True)
                
                if congelar_paineis:
                    excel.congelar_painel('Resumo_Tokens', 1, 1)
                
                # Usa nome do campo ID configurado
                nome_campo_id = self.dados_analise.config.nome_campo_id
                
                # Identifica colunas numéricas para aplicar cores condicionais
                colunas_tokens_numericas = [col for col in df_tokens.columns 
                                            if col != nome_campo_id and 
                                            not col.endswith('_finish') and
                                            df_tokens[col].dtype in [np.int64, np.float64]]
                
                print(f"   🎨 Aplicando formatação condicional em {len(colunas_tokens_numericas)} colunas de tokens...")
                
                # Aplica formatação condicional por tipo de token (escala relativa aos dados)
                if len(colunas_tokens_numericas) > 0 and len(df_tokens) > 0:
                    from xlsxwriter.utility import xl_col_to_name
                    
                    for col_name in colunas_tokens_numericas:
                        col_idx = df_tokens.columns.get_loc(col_name)
                        col_letter = xl_col_to_name(col_idx)
                        
                        # Define range de células (da linha 2 até última linha)
                        cells_range = f'{col_letter}2:{col_letter}{len(df_tokens) + 1}'
                        
                        # Calcula valores reais da coluna para escala adequada
                        col_min = 0 # nenhum token é o mínimo real possível
                        col_max = float(df_tokens[col_name].max()) # máximo real na coluna
                        col_mid = (col_min + col_max) / 2
                        
                        # Aplica escala de 3 cores invertida (verde = menos tokens/melhor, vermelho = mais tokens/pior)
                        # min_value > max_value ativa a inversão de cores no conditional_color
                        excel.conditional_color('Resumo_Tokens', cells_range, 
                                              min_value=col_max,  # valor alto = vermelho (pior)
                                              mid_value=col_mid,  # valor médio = amarelo
                                              max_value=col_min)  # valor baixo = verde (melhor)
                
                print(f"   ✅ Formatação condicional aplicada em Resumo_Tokens")
        
        # ═══════════════════════════════════════════════════════════════════════
        # ABA DE AVALIAÇÃO LLM
        # ═══════════════════════════════════════════════════════════════════════
        df_avaliacao = self._criar_dataframe_avaliacao_llm()
        if df_avaliacao is not None:
            # Escreve DataFrame normalmente
            excel.write_df(df_avaliacao, 'Avaliação LLM', auto_width_colums_list=True)
            
            if congelar_paineis:
                excel.congelar_painel('Avaliação LLM', 1, 1)
            
            # Usa nome do campo ID configurado
            nome_campo_id = self.dados_analise.config.nome_campo_id
            
            # Identifica colunas numéricas para aplicar cores condicionais
            # P, R, F1, nota (escala 0-1 ou 0-10)
            colunas_metricas = []
            for col in df_avaliacao.columns:
                if col == nome_campo_id or col.endswith('_explicacao'):
                    continue
                if df_avaliacao[col].dtype in [np.int64, np.float64]:
                    colunas_metricas.append(col)
            
            print(f"   🎨 Aplicando formatação condicional em {len(colunas_metricas)} colunas de avaliação LLM...")
            
            # Aplica formatação condicional nas métricas (escala 0-1 ou 0-10)
            if len(colunas_metricas) > 0 and len(df_avaliacao) > 0:
                from xlsxwriter.utility import xl_col_to_name
                
                for col_name in colunas_metricas:
                    col_idx = df_avaliacao.columns.get_loc(col_name)
                    col_letter = xl_col_to_name(col_idx)
                    
                    # Define range de células (da linha 2 até última linha)
                    cells_range = f'{col_letter}2:{col_letter}{len(df_avaliacao) + 1}'
                    
                    # Calcula valores reais da coluna para escala adequada
                    col_min = float(df_avaliacao[col_name].min())
                    col_max = float(df_avaliacao[col_name].max())
                    col_mid = (col_min + col_max) / 2
                    
                    # Aplica escala de 3 cores (verde = melhor, vermelho = pior)
                    # min_value < max_value = escala normal
                    excel.conditional_color('Avaliação LLM', cells_range, 
                                          min_value=col_min,   # valor baixo = vermelho (pior)
                                          mid_value=col_mid,   # valor médio = amarelo
                                          max_value=col_max)   # valor alto = verde (melhor)
            
            print(f"   ✅ Formatação condicional aplicada em Avaliação LLM")
        
        # ═══════════════════════════════════════════════════════════════════════
        # ABA DE OBSERVABILIDADE
        # ═══════════════════════════════════════════════════════════════════════
        df_observabilidade = self._criar_dataframe_observabilidade()
        if df_observabilidade is not None:
            # Escreve DataFrame normalmente
            excel.write_df(df_observabilidade, 'Observabilidade', auto_width_colums_list=True)
            
            if congelar_paineis:
                excel.congelar_painel('Observabilidade', 1, 1)
            
            # Usa nome do campo ID configurado
            nome_campo_id = self.dados_analise.config.nome_campo_id
            
            # Identifica colunas numéricas para aplicar cores condicionais
            # Exclui coluna de ID e colunas de sucesso _OK (que são texto "sim"/"não")
            colunas_metricas_obs = []
            for col in df_observabilidade.columns:
                if col == nome_campo_id or col.endswith('_OK'):
                    continue
                # Acessa a coluna usando .loc para evitar problemas com parênteses
                try:
                    col_serie = df_observabilidade.loc[:, col]
                    if col_serie.dtype in [np.int64, np.float64]:
                        colunas_metricas_obs.append(col)
                except:
                    pass  # Ignora colunas problemáticas
            
            print(f"   🎨 Aplicando formatação condicional em {len(colunas_metricas_obs)} colunas de observabilidade...")
            
            # Aplica formatação condicional nas métricas
            if len(colunas_metricas_obs) > 0 and len(df_observabilidade) > 0:
                from xlsxwriter.utility import xl_col_to_name
                
                for col_name in colunas_metricas_obs:
                    col_idx = df_observabilidade.columns.get_loc(col_name)
                    col_letter = xl_col_to_name(col_idx)
                    
                    # Define range de células (da linha 2 até última linha)
                    cells_range = f'{col_letter}2:{col_letter}{len(df_observabilidade) + 1}'
                    
                    # Calcula valores reais da coluna para escala adequada
                    col_serie = df_observabilidade.loc[:, col_name]
                    col_min = float(col_serie.min())
                    col_max = float(col_serie.max())
                    col_mid = (col_min + col_max) / 2
                    
                    # Determina escala baseada no tipo de métrica:
                    # - _SEG (tempo): vermelho (alto) = pior, verde (baixo) = melhor
                    # - _IT (iterações), _REV (loops revisão), revisoes: vermelho (alto) = pior, verde (baixo) = melhor
                    # - _AGT (agentes executados): verde (alto) = melhor, vermelho (baixo) = pior
                    if any(keyword in col_name for keyword in ['_SEG', 'duracao']):
                        # Inverte escala: verde = menos tempo (melhor)
                        excel.conditional_color('Observabilidade', cells_range, 
                                              min_value=col_max,   # valor alto = vermelho (pior)
                                              mid_value=col_mid,
                                              max_value=col_min)   # valor baixo = verde (melhor)
                    elif any(keyword in col_name for keyword in ['_IT', '_REV', 'revisoes']):
                        # Inverte escala: verde = menos iterações/loops (melhor)
                        excel.conditional_color('Observabilidade', cells_range, 
                                              min_value=col_max,   # valor alto = vermelho (pode ser pior)
                                              mid_value=col_mid,
                                              max_value=col_min)   # valor baixo = verde (pode ser melhor)
                    elif '_AGT' in col_name:
                        # Escala normal: verde = mais agentes executados (melhor)
                        excel.conditional_color('Observabilidade', cells_range, 
                                              min_value=col_min,   # valor baixo = vermelho
                                              mid_value=col_mid,
                                              max_value=col_max)   # valor alto = verde (melhor)
                    else:
                        # Escala padrão para outras métricas
                        excel.conditional_color('Observabilidade', cells_range, 
                                              min_value=col_min,
                                              mid_value=col_mid,
                                              max_value=col_max)
            
            print(f"   ✅ Formatação condicional aplicada em Observabilidade")
        
        # ═══════════════════════════════════════════════════════════════════════
        # ABA DE CONFIGURAÇÃO (para regenerar gráficos posteriormente)
        # ═══════════════════════════════════════════════════════════════════════
        config_data = {
            'parametro': ['nome_campo_id', 'rotulo_campo_id', 'rotulo_origem'],
            'valor': [
                self.dados_analise.config.nome_campo_id,
                self.dados_analise.config.rotulo_campo_id,
                self.dados_analise.config.rotulo_origem
            ],
            'descricao': [
                'Nome interno do campo ID (usado em DataFrames)',
                'Rótulo de exibição do campo ID',
                'Rótulo do modelo de referência/ground truth'
            ]
        }
        df_config = pd.DataFrame(config_data)
        excel.write_df(df_config, 'Config', auto_width_colums_list=True)
        if congelar_paineis:
            excel.congelar_painel('Config', 1, 0)
        
        print(f"   📋 Aba Config criada com configurações do sistema")
        
        excel.save()
        
        # Gera gráficos se solicitado
        if gerar_graficos:
            # Limpa gráficos antigos UMA VEZ antes de gerar todos os novos
            import glob
            pasta_saida = self.pasta_analises or '.'
            graficos_antigos = glob.glob(os.path.join(pasta_saida, 'grafico_*.png'))
            if graficos_antigos:
                print(f"   🧹 Removendo {len(graficos_antigos)} gráficos antigos...")
                for arquivo_grafico in graficos_antigos:
                    try:
                        os.remove(arquivo_grafico)
                    except Exception as e:
                        print(f"⚠️  Aviso: Não foi possível remover {arquivo_grafico}: {e}")
            
            # IMPORTANTE: Usa DataFrame em memória, não carrega do Excel!
            # Isso garante que todos os modelos/técnicas sejam consolidados
            self.gerar_graficos_metricas(arquivo_excel=None)
            # Gera gráficos de tokens se houver dados disponíveis
            if self.tokens is not None and len(self.tokens) > 0:
                self.gerar_graficos_tokens(arquivo_excel=None)
            # Gera gráficos de avaliação LLM se houver dados disponíveis
            if self.avaliacao_llm is not None and len(self.avaliacao_llm) > 0:
                self.gerar_graficos_avaliacao_llm(arquivo_excel=None)
            # Gera gráficos de observabilidade se houver dados disponíveis
            if self.observabilidade is not None and len(self.observabilidade) > 0:
                self.gerar_graficos_observabilidade(arquivo_excel=None)
        
        return arquivo

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
        os.makedirs(pasta_saida, exist_ok=True)
        
        # Limpa gráficos antigos
        if limpar_graficos_antigos:
            graficos_antigos = glob.glob(os.path.join(pasta_saida, 'grafico_*.png'))
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
            if 'Config' in xl_file.sheet_names:
                df_config = pd.read_excel(arquivo_excel, sheet_name='Config')
                # Busca o valor de nome_campo_id
                row_nome_campo_id = df_config[df_config['parametro'] == 'nome_campo_id']
                if not row_nome_campo_id.empty:
                    nome_campo_id = row_nome_campo_id.iloc[0]['valor']
                    print(f"   📋 Configuração carregada: nome_campo_id='{nome_campo_id}'")
            
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
                    tecnica_nome = aba.split('_', 1)[1].lower().replace('-', '')
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
                        if metrica in ['F1', 'P', 'R', 'LS']:
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
            estrutura = cls._extrair_estrutura_metricas_estatico(df_consolidado, tecnica_aba='')
            
            # Gera gráficos com TODOS os modelos
            arquivos_gerados = cls._gerar_boxplots_por_campo_metrica_estatico(
                df_consolidado, estrutura, pasta_saida, paleta, tecnica_aba=''
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
                            
                            # Prepara dados
                            dados_grafico = {}
                            for col in colunas_tipo:
                                modelo = col.rsplit('_', 1)[0]
                                valores = df_tokens[col].dropna().tolist()
                                if len(valores) > 0:
                                    dados_grafico[modelo] = valores
                            
                            if len(dados_grafico) == 0:
                                continue
                            
                            # Arquivo e título
                            arquivo_grafico = os.path.join(pasta_saida, f'grafico_tokens_{tipo}.png')
                            titulos_tipos = {
                                'input': 'Consumo de Tokens de Entrada',
                                'output': 'Consumo de Tokens de Saída',
                                'total': 'Consumo Total de Tokens',
                                'cache': 'Tokens em Cache',
                                'reason': 'Tokens de Raciocínio'
                            }
                            titulo = titulos_tipos.get(tipo, f'Tokens ({tipo})')
                            
                            # Gera gráfico
                            try:
                                UtilGraficos.gerar_boxplot(
                                    dados=dados_grafico,
                                    titulo=titulo,
                                    ylabel='Quantidade de Tokens',
                                    xlabel='Modelo',
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
                            
                            # Prepara dados
                            dados_grafico = {}
                            for col in colunas_metrica:
                                modelo = col.rsplit('_', 1)[0]
                                valores = df_avaliacao[col].dropna().tolist()
                                if len(valores) > 0:
                                    dados_grafico[modelo] = valores
                            
                            if len(dados_grafico) == 0:
                                continue
                            
                            # Arquivo e título
                            arquivo_grafico = os.path.join(pasta_saida, f'grafico_bp_avaliacaollm_{metrica.lower()}.png')
                            titulos_metricas = {
                                'P': 'Avaliação LLM - Precision',
                                'R': 'Avaliação LLM - Recall',
                                'F1': 'Avaliação LLM - F1-Score',
                                'nota': 'Avaliação LLM - Nota Geral'
                            }
                            titulo = titulos_metricas.get(metrica, f'Avaliação LLM - {metrica}')
                            
                            # Gera gráfico
                            try:
                                UtilGraficos.gerar_boxplot(
                                    dados=dados_grafico,
                                    titulo=titulo,
                                    ylabel=metrica,
                                    xlabel='Modelo',
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
                        'SEG': {'titulo': 'Tempo de Execução (segundos)', 'ylabel': 'Segundos'},
                        'REV': {'titulo': 'Revisões/Loops', 'ylabel': 'Quantidade'},
                        'IT': {'titulo': 'Iterações', 'ylabel': 'Quantidade'},
                        'AGT': {'titulo': 'Agentes Executados', 'ylabel': 'Quantidade'},
                        'QTD': {'titulo': 'Campos Preenchidos', 'ylabel': 'Quantidade'},
                        'BYTES': {'titulo': 'Tamanho dos Campos (bytes)', 'ylabel': 'Bytes'},
                        'OK': {'titulo': 'Taxa de Sucesso', 'ylabel': 'Proporção'}
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
                        
                        # Extrai aliases (nomes das colunas sem sufixo)
                        aliases = [col.rsplit('_', 1)[0] for col in colunas_sufixo]
                        
                        # Gera gráfico
                        arquivo_grafico = os.path.join(pasta_saida, f'grafico_bp_observabilidade_{sufixo}.png')
                        titulo = f"Observabilidade: {info['titulo']}"
                        ylabel = info['ylabel']
                        
                        # Configura gráfico boxplot usando grafico_multi_colunas
                        configuracao = {
                            titulo: {
                                'df': df_plot,
                                'colunas': colunas_sufixo,
                                'alias': aliases,
                                'x': 'Métrica',
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
            titulos_tipos = {
                'input': 'Consumo de Tokens de Entrada',
                'output': 'Consumo de Tokens de Saída',
                'total': 'Consumo Total de Tokens',
                'cache': 'Tokens em Cache',
                'reason': 'Tokens de Raciocínio'
            }
            titulo = titulos_tipos.get(tipo, f'Tokens ({tipo})')
            
            # Extrai aliases (nomes dos modelos sem sufixo)
            aliases = [col.rsplit('_', 1)[0] for col in colunas_tipo]
            
            # Configura gráfico boxplot usando grafico_multi_colunas
            configuracao = {
                titulo: {
                    'df': df_tokens,
                    'colunas': colunas_tipo,
                    'alias': aliases,
                    'x': 'Modelo',
                    'y': 'Quantidade de Tokens',
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
        os.makedirs(pasta_saida, exist_ok=True)
        
        # Carrega DataFrame de avaliação LLM
        if arquivo_excel:
            try:
                df_avaliacao = pd.read_excel(arquivo_excel, sheet_name='Avaliação LLM')
            except Exception as e:
                print(f"⚠️  Aviso: Não foi possível carregar aba 'Avaliação LLM': {e}")
                return []
        else:
            df_avaliacao = self._criar_dataframe_avaliacao_llm()
        
        if df_avaliacao is None or df_avaliacao.empty:
            print("⚠️  Aviso: Nenhum dado de avaliação LLM disponível para gráficos")
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
        
        # ═════════════════════════════════════════════════════════════════════
        # SEPARA MÉTRICAS GLOBAIS DE MÉTRICAS POR CAMPO
        # ═════════════════════════════════════════════════════════════════════
        # Estrutura: {metrica: {'globais': [...], 'campos': {campo: [...]}}}
        estrutura_metricas = {}
        
        # Obtém lista de rótulos dos modelos (exclui apenas 'id', inclui origem e destinos)
        # rotulos[0] = 'id', rotulos[1] = origem, rotulos[2:] = destinos
        rotulos_modelos = self.rotulos[1:] if len(self.rotulos) > 1 else []
        
        for col in df_avaliacao.columns:
            if col == nome_campo_id or col.endswith('_explicacao'):
                continue
            
            # Verifica se é coluna numérica
            if df_avaliacao[col].dtype not in [np.int64, np.float64]:
                continue
            
            if '_' not in col:
                continue
            
            # Identifica qual modelo é (pode conter parênteses, underscores, etc)
            modelo_identificado = None
            for rotulo in rotulos_modelos:
                if col.startswith(rotulo + '_'):
                    modelo_identificado = rotulo
                    break
            
            if modelo_identificado is None:
                continue  # Não conseguiu identificar o modelo
            
            # Remove o prefixo do modelo para obter resto: campo_metrica ou só metrica
            sufixo = col[len(modelo_identificado) + 1:]  # +1 para remover o underscore
            
            # Verifica se é métrica global (sufixo é P, R, F1, nota, etc)
            metricas_globais = ['P', 'R', 'F1', 'nota']
            if sufixo in metricas_globais:
                # Global: modelo_metrica (ex: agentes_gpt5_P, base_gemma3(12)_nota)
                metrica = sufixo
                if metrica not in estrutura_metricas:
                    estrutura_metricas[metrica] = {'globais': [], 'campos': {}}
                estrutura_metricas[metrica]['globais'].append(col)
            else:
                # Por campo: modelo_campo_metrica (ex: agentes_gpt5_tema_P)
                # Divide sufixo em campo e métrica (última parte após último underscore)
                if '_' in sufixo:
                    partes_sufixo = sufixo.rsplit('_', 1)  # rsplit para pegar último underscore
                    campo = partes_sufixo[0]
                    metrica = partes_sufixo[1]
                    
                    if metrica not in estrutura_metricas:
                        estrutura_metricas[metrica] = {'globais': [], 'campos': {}}
                    if campo not in estrutura_metricas[metrica]['campos']:
                        estrutura_metricas[metrica]['campos'][campo] = []
                    estrutura_metricas[metrica]['campos'][campo].append(col)
        
        # ═════════════════════════════════════════════════════════════════════
        # GERA GRÁFICOS PARA MÉTRICAS GLOBAIS
        # ═════════════════════════════════════════════════════════════════════
        titulos_metricas = {
            'P': 'Avaliação LLM - Precision (Global)',
            'R': 'Avaliação LLM - Recall (Global)',
            'F1': 'Avaliação LLM - F1-Score (Global)',
            'nota': 'Avaliação LLM - Nota Geral'
        }
        
        # Obtém lista de rótulos dos modelos (exclui apenas 'id', inclui origem e destinos)
        # rotulos[0] = 'id', rotulos[1] = origem, rotulos[2:] = destinos
        rotulos_modelos = self.rotulos[1:] if len(self.rotulos) > 1 else []
        
        for metrica in sorted(estrutura_metricas.keys()):
            colunas_globais = estrutura_metricas[metrica]['globais']
            
            if len(colunas_globais) == 0:
                continue
            
            # Define nome do arquivo
            arquivo_grafico = os.path.join(pasta_saida, f'grafico_bp_avaliacaollm_{metrica.lower()}.png')
            
            # Título
            titulo = titulos_metricas.get(metrica, f'Avaliação LLM - {metrica} (Global)')
            
            # Agrupa colunas por modelo usando os rótulos originais
            colunas_por_modelo = []
            aliases_ordenados = []
            
            for rotulo in rotulos_modelos:
                # Busca coluna que começa com este rótulo: rotulo_metrica
                col_modelo = None
                for col in colunas_globais:
                    if col == f'{rotulo}_{metrica}':
                        col_modelo = col
                        break
                
                if col_modelo is not None:
                    colunas_por_modelo.append(col_modelo)
                    aliases_ordenados.append(rotulo)
            
            # Se não encontrou nenhuma coluna, pula
            if not colunas_por_modelo:
                continue
            
            # Configura gráfico boxplot usando grafico_multi_colunas
            configuracao = {
                titulo: {
                    'df': df_avaliacao,
                    'colunas': colunas_por_modelo,
                    'alias': aliases_ordenados,
                    'x': 'Modelo',
                    'y': metrica,
                    'agregacao': 'boxplot',
                    'paleta': paleta_enum,
                    'ylim': (0, 1) if metrica in ['P', 'R', 'F1'] else None,  # Fixar eixo Y para métricas [0, 1]
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
        for metrica in sorted(estrutura_metricas.keys()):
            campos = estrutura_metricas[metrica]['campos']
            
            if len(campos) == 0:
                continue
            
            # Para cada campo, gera um gráfico separado
            for campo in sorted(campos.keys()):
                colunas_campo = campos[campo]
                
                if len(colunas_campo) == 0:
                    continue
                
                # Define nome do arquivo
                arquivo_grafico = os.path.join(pasta_saida, 
                    f'grafico_bp_avaliacaollm_{campo.lower()}_{metrica.lower()}.png')
                
                # Título
                titulos_metricas_campo = {
                    'P': f'Avaliação LLM - {campo} - Precision',
                    'R': f'Avaliação LLM - {campo} - Recall',
                    'F1': f'Avaliação LLM - {campo} - F1-Score'
                }
                titulo = titulos_metricas_campo.get(metrica, f'Avaliação LLM - {campo} - {metrica}')
                
                # Agrupa colunas por modelo usando os rótulos originais
                colunas_por_modelo = []
                aliases_ordenados = []
                
                for rotulo in rotulos_modelos:
                    # Busca coluna que começa com este rótulo: rotulo_campo_metrica
                    col_modelo = None
                    for col in colunas_campo:
                        if col == f'{rotulo}_{campo}_{metrica}':
                            col_modelo = col
                            break
                    
                    if col_modelo is not None:
                        colunas_por_modelo.append(col_modelo)
                        aliases_ordenados.append(rotulo)
                
                # Se não encontrou nenhuma coluna, pula
                if not colunas_por_modelo:
                    continue
                
                # Configura gráfico boxplot
                configuracao = {
                    titulo: {
                        'df': df_avaliacao,
                        'colunas': colunas_por_modelo,
                        'alias': aliases_ordenados,
                        'x': 'Modelo',
                        'y': metrica,
                        'agregacao': 'boxplot',
                        'paleta': paleta_enum,
                        'ylim': (0, 1) if metrica in ['P', 'R', 'F1'] else None,  # Fixar eixo Y para métricas [0, 1]
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
                    print(f"⚠️  Erro ao gerar gráfico de avaliação LLM por campo ({campo}.{metrica}): {e}")
        
        if len(arquivos_gerados) > 0:
            print(f"✅ {len(arquivos_gerados)} gráficos de avaliação LLM gerados em: {pasta_saida}")
        
        return arquivos_gerados
    
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
            
            # Extrai aliases (nomes das colunas sem sufixo)
            aliases = [col.rsplit('_', 1)[0] for col in colunas_sufixo]
            
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
                    'x': 'Métrica',
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
        modelos_unicos = set()
        tecnicas_conhecidas = ['bertscore', 'rouge2', 'rouge1', 'rouge', 'levenshtein']
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
            
            if modelo:
                modelos_unicos.add(modelo)
        
        # Ordena modelos para manter consistência
        rotulos_modelos = sorted(modelos_unicos)
        
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
    def _extrair_estrutura_metricas_estatico(df, tecnica_aba: str = '') -> dict:
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
        
        # Técnicas e métricas conhecidas
        tecnicas_conhecidas = ['bertscore', 'rouge2', 'rouge1', 'rouge', 'rougel', 'levenshtein']
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
                tecnica = tecnica_aba.lower().replace('-', '') if tecnica_aba else 'geral'
            
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
                            'x': 'Modelos',
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
                                                   tecnica_aba: str = '') -> List[str]:
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
        modelos_unicos = set()
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
                    tecnicas_conhecidas = ['bertscore', 'rouge', 'rouge1', 'rouge2', 'rougel', 'levenshtein']
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
            
            modelos_unicos.add(modelo)
        
        # Ordena modelos para manter consistência visual
        modelos_ordenados = sorted(modelos_unicos)
        
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
                    
                    # Nome: grafico_bp_<tecnica>_<campo>_<metrica>.png
                    nome_arquivo = f'grafico_bp_{tecnica_nome}_{campo_safe}_{metrica}.png'
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
                            'x': 'Modelos',
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
                        # Remove técnica se existir
                        tecnicas_conhecidas = ['bertscore', 'rouge2', 'rouge1', 'rouge', 'rougel', 'levenshtein']
                        if partes and partes[-1] in tecnicas_conhecidas:
                            partes = partes[:-1]
                        # Remove campo
                        if len(partes) > 1:
                            partes = partes[:-1]
                        # Modelo é o que sobrou
                        modelo = '_'.join(partes) if partes else col.split('_')[0]
                        aliases.append(modelo)
                    
                    tecnica_label = tecnica.upper() if tecnica != 'geral' else ''
                    titulo = 'Comparação Global F1 entre Modelos'
                    if tecnica_label:
                        titulo += f' ({tecnica_label})'
                    
                    nome_arquivo = f'grafico_comp_global_f1_{tecnica}.png'
                    caminho_completo = os.path.join(pasta_saida, nome_arquivo)
                    
                    config = {
                        titulo: {
                            'df': df,
                            'colunas': colunas_f1,
                            'alias': aliases,
                            'x': 'Modelos',
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
                    'Distribuição de F1 por Campo': {
                        'df': df,
                        'colunas': list(primeira_col_por_campo.values()),
                        'alias': [k.replace('_', ' ') for k in primeira_col_por_campo.keys()],
                        'x': 'Campos',
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
                        # Remove técnica se existir
                        tecnicas_conhecidas = ['bertscore', 'rouge2', 'rouge1', 'rouge', 'rougel', 'levenshtein']
                        if partes and partes[-1] in tecnicas_conhecidas:
                            partes = partes[:-1]
                        # Remove campo
                        if len(partes) > 1:
                            partes = partes[:-1]
                        # Modelo é o que sobrou
                        modelo = '_'.join(partes) if partes else col.split('_')[0]
                        aliases.append(modelo)
                    
                    tecnica_label = tecnica.upper() if tecnica != 'geral' else 'Levenshtein'
                    titulo = 'Comparação Global Similaridade entre Modelos'
                    if tecnica_label:
                        titulo += f' ({tecnica_label})'
                    
                    nome_arquivo = f'grafico_comp_global_sim_{tecnica}.png'
                    caminho_completo = os.path.join(pasta_saida, nome_arquivo)
                    
                    config = {
                        titulo: {
                            'df': df,
                            'colunas': colunas_sim,
                            'alias': aliases,
                            'x': 'Modelos',
                            'y': 'Similaridade (SIM)',
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

    RE_ARQUIVOS_JSON_ANALISE = re.compile(r'^analise_(\d{12})\.\d+\.\d*\.json$')
    def _limpar_arquivos_analise(self):
        '''Removendo análises antigas da pasta (JSONs, CSVs, Excel, MD e PNGs de gráficos)'''
        if os.path.isdir(self.pasta_analises):
            print(f"\n🧹 Limpando análises antigas em: {self.pasta_analises}")
            arquivos_removidos = 0
            for f in os.listdir(self.pasta_analises):
                caminho_f = os.path.join(self.pasta_analises, f)
                try:
                    if os.path.isfile(caminho_f) and \
                        (f.endswith(('.csv', '.xlsx', '.png', 'exemplos.md')) or self.RE_ARQUIVOS_JSON_ANALISE.match(f)):
                        os.remove(caminho_f)
                        arquivos_removidos += 1
                except Exception as e:
                    print(f"   ⚠️  Erro ao remover {caminho_f}: {e}")
            if arquivos_removidos > 0:
                print(f"   ✓ {arquivos_removidos} arquivo(s) removido(s)")
        
class Json2Texto:
    """
    Converte um dicionário Python em Markdown ou texto para análise de similaridade.
    
    Uso direto (sem instanciar):
        Json2Texto.to_markdown(dados)
        Json2Texto.to_linear_text(dados)
        Json2Texto.to_natural_text(dados)
    
    Modos de conversão:
    1. to_markdown(): Markdown estruturado com títulos e hierarquia
    2. to_linear_text(): Texto linearizado chave:valor (ideal para BERTScore)
    3. to_natural_text(): Texto fluente em parágrafos (ideal para ROUGE)
    
    Estrutura de conversão:
    - chaves do nível raiz -> títulos (#)
    - listas -> itens numerados ou bullets
    - dicionários aninhados -> subtópicos com nome da chave
    - valores escalares -> parágrafos ou inline
    
    Args:
        dados: dicionário a ser convertido
        heading_start_level: nível inicial de títulos (1-6, padrão 1)
        sort_keys: se True, ordena chaves alfabeticamente (padrão False)
        normalize_whitespace: se True, normaliza espaços múltiplos (padrão True)
    """
    
    # --------- API pública ----------
    @classmethod
    def to_markdown(cls, dados: Dict[str, Any], *, heading_start_level: int = 1, sort_keys: bool = False, normalize_whitespace: bool = True) -> str:
        """
        Converte JSON em Markdown estruturado com títulos e hierarquia.
        Ideal para documentação ou visualização humana.
        
        Args:
            dados: dicionário a ser convertido
            heading_start_level: nível inicial de títulos (1-6)
            sort_keys: se True, ordena chaves alfabeticamente
            normalize_whitespace: se True, normaliza espaços múltiplos
            
        Returns:
            str: texto em formato Markdown
        """
        if not isinstance(dados, dict):
            raise TypeError("Json2Texto requer um dict como entrada.")
        
        heading_start_level = max(1, min(6, heading_start_level))
        partes = []
        for k, v in cls._iter_kv(dados, sort_keys):
            partes.append(f"{cls._h(heading_start_level)} {cls._fmt_key(k)}")
            partes.append(cls._render(v, heading_start_level, sort_keys, normalize_whitespace))
        texto = cls._tidy("\n".join(partes))
        return cls._normalize_ws(texto) if normalize_whitespace else texto

    @classmethod
    def to_linear_text(cls, dados: Dict[str, Any], *, sort_keys: bool = False, normalize_whitespace: bool = True) -> str:
        """
        Converte JSON em texto linearizado chave:valor.
        Ideal para BERTScore - mantém estrutura semântica clara.
        Remove marcação markdown mantendo conteúdo informativo.
        
        Args:
            dados: dicionário a ser convertido
            sort_keys: se True, ordena chaves alfabeticamente
            normalize_whitespace: se True, normaliza espaços múltiplos
            
        Returns:
            str: texto linearizado
        """
        if not isinstance(dados, dict):
            raise TypeError("Json2Texto requer um dict como entrada.")
        
        trechos = []
        for k, v in cls._iter_kv(dados, sort_keys):
            trechos.append(cls._kv_line(k, v, sort_keys, normalize_whitespace))
        texto = " ".join(trechos).strip()
        return cls._normalize_ws(texto) if normalize_whitespace else texto
    
    @classmethod
    def to_natural_text(cls, dados: Dict[str, Any], *, sort_keys: bool = False, normalize_whitespace: bool = True) -> str:
        """
        Converte JSON em texto fluente e natural, em parágrafos.
        Ideal para ROUGE - texto corrido facilita análise de n-gramas e coerência.
        Remove hierarquia visual mantendo relações semânticas.
        
        Args:
            dados: dicionário a ser convertido
            sort_keys: se True, ordena chaves alfabeticamente
            normalize_whitespace: se True, normaliza espaços múltiplos
            
        Returns:
            str: texto em linguagem natural
        """
        if not isinstance(dados, dict):
            raise TypeError("Json2Texto requer um dict como entrada.")
        
        paragrafos = []
        for k, v in cls._iter_kv(dados, sort_keys):
            para = cls._to_natural_paragraph(k, v, "", sort_keys, normalize_whitespace)
            if para:
                paragrafos.append(para)
        texto = "\n\n".join(paragrafos)
        return cls._normalize_ws(texto) if normalize_whitespace else texto

    # --------- Internos ----------
    @classmethod
    def _render(cls, val: Any, level: int, sort_keys: bool, normalize_whitespace: bool) -> str:
        """Renderiza valor em formato Markdown estruturado"""
        if cls._is_scalar(val):
            return f"\n{cls._scalar_to_text(val, normalize_whitespace)}\n"
        elif isinstance(val, list):
            linhas = []
            for i, item in enumerate(val, 1):
                # Usa numeração mais concisa para listas
                if cls._is_scalar(item):
                    linhas.append(f"{i}. {cls._scalar_to_text(item, normalize_whitespace)}")
                else:
                    linhas.append(f"{cls._h(level+1)} {i}")
                    linhas.append(cls._render(item, level+1, sort_keys, normalize_whitespace))
            return "\n".join(linhas)
        elif isinstance(val, dict):
            linhas = []
            for k, v in cls._iter_kv(val, sort_keys):
                linhas.append(f"{cls._h(level+1)} {cls._fmt_key(k)}")
                linhas.append(cls._render(v, level+1, sort_keys, normalize_whitespace))
            return "\n".join(linhas)
        else:
            # Fallback para tipos não-usuais
            return f"\n{repr(val)}\n"

    @classmethod
    def _kv_line(cls, k: Any, v: Any, sort_keys: bool, normalize_whitespace: bool) -> str:
        """Gera linha chave:valor linearizada (para BERTScore)"""
        if cls._is_scalar(v):
            return f"{cls._fmt_key(k)}: {cls._scalar_to_text(v, normalize_whitespace)}."
        elif isinstance(v, list):
            # Lista: separa itens com ponto-e-vírgula
            itens = []
            for x in v:
                if cls._is_scalar(x):
                    itens.append(cls._scalar_to_text(x, normalize_whitespace))
                else:
                    itens.append(cls._strip_md(cls._render(x, 1, sort_keys, normalize_whitespace)))
            return f"{cls._fmt_key(k)}: {'; '.join(itens)}."
        elif isinstance(v, dict):
            # Dict: separa pares com ponto-e-vírgula
            pares = []
            for kk, vv in cls._iter_kv(v, sort_keys):
                if cls._is_scalar(vv):
                    pares.append(f"{cls._fmt_key(kk)}: {cls._scalar_to_text(vv, normalize_whitespace)}")
                else:
                    pares.append(f"{cls._fmt_key(kk)}: {cls._strip_md(cls._render(vv, 1, sort_keys, normalize_whitespace))}")
            return f"{cls._fmt_key(k)}: {'; '.join(pares)}."
        return f"{cls._fmt_key(k)}: {repr(v)}."
    
    @classmethod
    def _to_natural_paragraph(cls, k: Any, v: Any, prefix: str, sort_keys: bool, normalize_whitespace: bool) -> str:
        """Gera parágrafo em linguagem natural (para ROUGE)"""
        key_text = cls._fmt_key(k)
        full_key = f"{prefix} {key_text}" if prefix else key_text
        
        if cls._is_scalar(v):
            # Valor escalar: frase simples
            return f"{full_key} é {cls._scalar_to_text(v, normalize_whitespace)}."
        elif isinstance(v, list):
            # Lista: enumera itens naturalmente
            if not v:
                return f"{full_key} está vazio."
            
            # Se todos são escalares, lista como "X, Y e Z"
            if all(cls._is_scalar(x) for x in v):
                if len(v) == 1:
                    return f"{full_key} contém {cls._scalar_to_text(v[0], normalize_whitespace)}."
                elif len(v) == 2:
                    return f"{full_key} contém {cls._scalar_to_text(v[0], normalize_whitespace)} e {cls._scalar_to_text(v[1], normalize_whitespace)}."
                else:
                    itens = [cls._scalar_to_text(x, normalize_whitespace) for x in v[:-1]]
                    ultimo = cls._scalar_to_text(v[-1], normalize_whitespace)
                    return f"{full_key} contém {', '.join(itens)} e {ultimo}."
            else:
                # Lista de objetos complexos
                frases = [f"{full_key} contém {len(v)} itens."]
                if len(v) > 0:
                    frases[-1] += '\n'
                for i, item in enumerate(v, 1):
                    if item is None:
                        # None representa ausência (importante para alinhamento de listas)
                        frases.append(f"Item {i} está vazio.")
                    elif isinstance(item, dict):
                        # Descreve cada item do dict
                        for kk, vv in cls._iter_kv(item, sort_keys):
                            frases.append(cls._to_natural_paragraph(kk, vv, f"No item {i}", sort_keys, normalize_whitespace))
                            frases[-1] += '\n'
                    else:
                        frases.append(f"Item {i} é {cls._strip_md(cls._render(item, 1, sort_keys, normalize_whitespace))}.")
                return " ".join(frases)
        elif isinstance(v, dict):
            # Dict: descreve sub-campos
            if not v:
                return f"{full_key} está vazio."
            
            frases = []
            for kk, vv in cls._iter_kv(v, sort_keys):
                sub_para = cls._to_natural_paragraph(kk, vv, full_key, sort_keys, normalize_whitespace)
                if sub_para:
                    frases.append(sub_para)
            return " ".join(frases)
        else:
            return f"{full_key} é {repr(v)}."

    @classmethod
    def _iter_kv(cls, d: dict, sort_keys: bool) -> Iterable:
        """Itera sobre chaves-valores, opcionalmente ordenando"""
        items = d.items()
        if sort_keys:
            items = sorted(items, key=lambda kv: str(kv[0]))
        return items

    @classmethod
    def _h(cls, level: int) -> str:
        """Gera marcação de título markdown"""
        return "#" * max(1, min(6, level))

    @classmethod
    def _fmt_key(cls, k: Any) -> str:
        """Formata chave removendo espaços extras"""
        return str(k).strip()

    @classmethod
    def _scalar_to_text(cls, x: Any, normalize_whitespace: bool) -> str:
        """Converte valor escalar para texto"""
        if x is None:
            return "null"
        if isinstance(x, bool):
            return "verdadeiro" if x else "falso"
        if isinstance(x, (int, float)):
            return str(x)
        # String: remove espaços extras se normalize_whitespace
        texto = str(x)
        return cls._normalize_ws(texto) if normalize_whitespace else texto

    @classmethod
    def _is_scalar(cls, x: Any) -> bool:
        """Verifica se valor é escalar (não estruturado)"""
        return isinstance(x, (str, int, float, bool)) or x is None

    @classmethod
    def _normalize_ws(cls, s: str) -> str:
        """Normaliza espaços em branco: múltiplos espaços -> 1 espaço"""
        if not s:
            return s
        # Substitui múltiplos espaços por um único
        s = re.sub(r' +', ' ', s)
        # Normaliza quebras de linha (máximo 2 consecutivas)
        s = re.sub(r'\n{3,}', '\n\n', s)
        return s.strip()

    @classmethod
    def _tidy(cls, s: str) -> str:
        """Remove linhas em branco duplicadas"""
        while "\n\n\n" in s:
            s = s.replace("\n\n\n", "\n\n")
        return s.strip() + "\n"

    @classmethod
    def _strip_md(cls, s: str) -> str:
        """Remove marcação markdown (cabeçalhos, bullets) para linearização"""
        linhas = []
        for linha in s.splitlines():
            # Remove linhas de cabeçalho
            linha_stripped = linha.lstrip()
            if linha_stripped.startswith("#"):
                # Extrai apenas o texto do cabeçalho
                texto = re.sub(r'^#+\s*', '', linha_stripped)
                if texto:
                    linhas.append(texto.strip())
            elif linha_stripped:
                linhas.append(linha.strip())
        return " ".join(l for l in linhas if l).strip()


if __name__ == '__main__':
    # recebe o número do exemplo via linha de comando
    import sys
    exemplo_num = 1
    if len(sys.argv) > 1:
        try:
            exemplo_num = int(sys.argv[1])
        except ValueError:
            exemplo_num = sys.argv[1]
    JsonAnalise.teste_compara(exemplo=exemplo_num, retornar=False)
