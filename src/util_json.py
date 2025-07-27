# -*- coding: utf-8 -*-

'''
 Autor Luiz Anísio 17/07/2025
 Utilitários para avaliar F1, Pecision e Recall de Jsons
'''

try:
    deps = []
    try:
       import Levenshtein
    except:
       deps.append(' - Levenshtein >> pip install python-Levenshtein')
    try:
       from rouge_score import rouge_scorer
    except:
       deps.append(' - Rouge Score >> pip install rouge-score')
    if any(deps):
       deps = 'ATENÇÃO:\nDependências não resolvidas:\n' + '\n'.join(deps)
       raise ImportError('')
except ImportError:
    raise ImportError(str(deps)) from None

import json
from statistics import harmonic_mean, mean
import re
import numpy as np
from typing import Any, List, Tuple, Union
from enum import Enum
import hashlib
from copy import deepcopy
from datetime import datetime

class JsonAnalise:
    ''' CONFIG para as chamadas que possuem o parâmetro "config" avaliados de acordo com cada função:
        - campos_alinhar = campos que serão comparados com um threshold para igualar valores antes de calcular o F1
        - campos_embedding = campos que possuem uma lista de floats do embedding e serão comparados pela similaridade no alinhamento
                             - só funciona em conjunto com o alinhamento e se não estiver na lista de alinhamento, é incluído com threshold padrão
        - campos_rouge? = campos que serão comparados com ROUGE-L, ROUGE-1 ou ROUGE-2 no alinhamento
                             - só funciona em conjunto com o alinhamento e se não estiver na lista de alinhamento, é incluído com threshold padrão
        - campos_lista = campos que serão tratados como lista, ou seja, seu conteúdo não será analisado recursivamente
                         - listas de listas são flats das listas internas (máximo lista de lista - 2 níveis)
                         - muda a forma como os itens serão criados para cálculo do f1 e do loss pois a lista inteira será comparada como um único item
                         - se o campo for embeddig, já vira um hash depois da análise da similaridade e não é mais considerado lista
        - padronizar_simbolos = True/False - padrão True - símbolos como aspas especiais, simples ou duplas e lowercase
        config = dicionário de configuração da comparação
                {'campos_embedding': [... nomes dos campos que devem ser comparados como embedding - lista de floats/ints ],
                 'campos_alinhar': {'campo1': threshold1, 'campo2': threshold2, ...},
                 'padronizar_simbolos': True/False - padrão True,
                 'campos_rouge': [... nomes dos campos cuja string será comparada com ROUGE-L] - coerência geral
                 'campos_rouge1': [... nomes dos campos cuja string será comparada com ROUGE-1] - termos
                 'campos_rouge2': [... nomes dos campos cuja string será comparada com ROUGE-2] - bigramas
                 'rouge_stemmer': quando calcular o rouge, usar stemmer True/False padrão True
                }
    '''
    RE_UNE_ESPACO = re.compile(r"\s+")
    RE_UNE_ENTER = re.compile(r"\n+")
    THRESHOLD_PADRAO = 0.95

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
    def json_to_flat(
        cls,
        obj: Union[dict, list],
        as_dict = False,
        ignorar_alinhados = False,
        config: dict = None
    ) -> List[str] | dict:
        """
        Recebe um dict ou lista e retorna uma lista de strings "chave:valor",
        achatando todos os níveis de dicionários e listas aninhadas.
        Não leva em consideração a ordem dos itens
        - campos_lista: não faz o flat da lista, mantém os valores como estão.
                        Se as_dict = False, transforma em string após concatenar com o campo.
                        Se for uma lista de listas e cada lista deve ser flat, colocar campo* (cada item de campo será um flat)
        - as_dict: retorna um dicionário {campo:valor, ...}
                        Se for uma lista, inclui a posição do item no path da chave
        - ignorar_alinhados = True - ignora os campos que foram alinhados pois serão analisados de outra forma no loss

        Exemplos de saída:
          >>> JsonAnalise.json_to_flat({"a": 1, "b": [2,3]})
          ["a:1", "b:2", "b:3"]

          >>> JsonAnalise.json_to_flat({"x": {"y": [4, {"z": 5}]}})
          ["x.y:4", "x.y.z:5"]

          >>> JsonAnalise.json_to_flat({"x": {"y": [4, {"z": 5}]}, "a": [1,2,3]}, campos_lista=["a"], as_dict = True/False)
              - as_dict = True  >> {'x.y[0]': 4, 'x.y[1].z': 5, 'a': [1, 2, 3]}
              - as_dict = False >> ['x.y:4', 'x.y.z:5', 'a:[1, 2, 3]']
        """
        config = cls.__ajustar_config(config)
        campos_lista = config['campos_lista']
        padronizar_simbolos = config['padronizar_simbolos']
        campos_alinhar = config['campos_alinhar']

        if as_dict:
           flat: dict = {}
        else:
           flat: List[str] = []

        def _flatten(campo:str, prefix: str, value: Any):
            # Caso seja dict, itera chaves e aprofunda
            if isinstance(value, dict):
                for k, v in value.items():
                    if ignorar_alinhados and k in campos_alinhar:
                        continue
                    novo_prefixo = f"{prefix}.{k}" if prefix else k
                    _flatten(k, novo_prefixo, v)
            # Caso seja lista, itera itens com mesmo prefixo
            elif isinstance(value, list):
                if campo in campos_lista:
                   # cada item é mantido ou transformado em str se não for dict
                   if as_dict:
                      flat[f"{prefix}"] = value
                   else:
                      # flat da lista ou flat dos itens da lista
                      if len(value)>0 and isinstance(value[0], list):
                         for item in value:
                            _flatten(campo, f"{prefix}", item)
                      else:
                         flat.append(f"{prefix}:{value}")
                else:
                   # cada item é analisado recursivamente
                   for _i, item in enumerate(value):
                       if as_dict:
                          _flatten(campo, f'{prefix}[{_i}]', item)
                       else:
                          _flatten(campo, prefix, item)
            # Caso seja valor primitivo, adiciona ao resultado
            else:
                val = value
                if padronizar_simbolos and isinstance(val, str):
                    val = cls.padronizar_simbolos(val)
                if as_dict:
                   flat[f"{prefix}"]=val
                else:
                   flat.append(f"{prefix}:{val}")

        # Dispara a recursão a partir do objeto raiz
        _flatten('', '', obj)
        return flat

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
        if padronizar_simbolos:
            texto1 = cls.padronizar_simbolos(texto1)
            texto2 = cls.padronizar_simbolos(texto2)
        # usa Levenshtein.distance do pacote python-Levenshtein
        return 1-Levenshtein.ratio(texto1, texto2)

    @classmethod
    def distancia_jaccard(cls, lista_a: list, lista_b: list,
        outras_distancias: Union[List[float],None] = None,
        as_dict = False
    ) -> float:
        """
        Calcula a distância de Jaccard entre list1 e list2
          - usa o conjunto outras_distancias para ponderar uma distância de acordo
            com o número de itens da uniao de a com b e o número de itens de outras_listas
        Args:
          lista_a, lista_b: listas a comparar via Jaccard
          outras_distancias: lista de floats calculados com outra técnica, pode
                            ser um conjunto de similaridades semânticas de outros campos.
        Retorna:
          Distância média ponderada sobre todos os campos.
        """
        log = {'distancia': 0, 'distancias': []}
        # 1) Distância de Jaccard
        set1 = set(lista_a)
        set2 = set(lista_b)
        union_size = len(set1.union(set2))
        if union_size == 0:
            # se ambas as listas estão vazias, distância = zero
            return log if as_dict else 0.0
        intersection_size = len(set1.intersection(set2))
        d_jaccard = 1.0 - (intersection_size / union_size)
        log['distancia'] = d_jaccard
        if as_dict:
           log['distancias'] = [0] * intersection_size + [1] * (union_size - intersection_size)

        if (not isinstance(outras_distancias, (tuple, list, set))) or len(outras_distancias) == 0:
           return log if as_dict else log['distancia']
        # 2) inicia soma ponderada
        total_peso = union_size
        soma_ponderada = d_jaccard * union_size
        # 3) acumula outras distâncias
        for dist in outras_distancias:
            soma_ponderada += dist
            total_peso     += 1
        # 4) retorna média ponderada
        log['distancias'].extend([d for d in outras_distancias])
        log['distancia'] =  soma_ponderada / total_peso
        return log if as_dict else log['distancia']


    @classmethod
    def print_analise_config(cls, config):
        ''' Print do config ajustado para debug'''
        print(f'CONFIG: {json.dumps(cls.__ajustar_config(config), indent=2, ensure_ascii=False)}')

    @classmethod
    def __ajustar_config(cls,config:dict):
        if (config is not None) and config.get('~cópia-validada~'):
           # evita validar várias vezes ao passar de uma função para outra
           return config
        config = {} if config is None else deepcopy(config)
        # threshold padrão
        if (not isinstance(config.get('threshold'), float)):
           config['threshold'] = cls.THRESHOLD_PADRAO
        # padronização do texto
        config['padronizar_simbolos'] = config['padronizar_simbolos'] if isinstance(config.get('padronizar_simbolos'), bool) else True
        # configurações rouge
        config['rouge_stemmer'] = config['rouge_stemmer'] if isinstance(config.get('rouge_stemmer'), bool) else True
        config['campos_rouge'] = config['campos_rouge'] if isinstance(config.get('campos_rouge'), (set, tuple, list)) else []
        if 'campos_rouge_1' in config:
            config['campos_rouge1'] = config.pop('campos_rouge_1')
        if 'campos_rouge_2' in config:
            config['campos_rouge2'] = config.pop('campos_rouge_2')
        config['campos_rouge1'] = config['campos_rouge1'] if isinstance(config.get('campos_rouge1'), (set, tuple, list)) else []
        config['campos_rouge2'] = config['campos_rouge2'] if isinstance(config.get('campos_rouge2'), (set, tuple, list)) else []
        # campos de alinhamento, se for lista, usa threshold padrão
        if isinstance(config.get('campos_alinhar'), dict):
           pass
        elif isinstance(config.get('campos_alinhar'), (list, tuple, set)):
             config['campos_alinhar'] = {c: config['threshold'] for c in config['campos_alinhar']}
        else:
           config['campos_alinhar'] = dict()

        # lista ou listas?
        if ('campos_listas' in config) and ('campo_lista' not in config):
           config['campos_lista'] = config.pop('campos_listas')
        config['campos_lista'] = list(config['campos_lista']) if isinstance(config.get('campos_lista'), (set, tuple, list)) else []
        # embedding ou embeddings?
        if ('campos_embeddings' in config) and ('campos_embedding' not in config):
           config['campos_embedding'] = config.pop('campos_embeddings')
        config['campos_embedding'] = list(config['campos_embedding']) if isinstance(config.get('campos_embedding'), (set, tuple, list)) else []
        config['~cópia-validada~'] = True
        # os campos de embedding e rouge devem estar nos campos de alinhamento
        campos_alinhamento_auto = config['campos_embedding'] + config['campos_rouge'] + config['campos_rouge1'] + config['campos_rouge2']
        for campo in campos_alinhamento_auto:
           if campo not in config['campos_alinhar']:
              config['campos_alinhar'][campo] = config['threshold']
        return config

    @classmethod
    def rouge_scorer(cls, texto1:str, texto2:str, config:dict):
        ''' config['rouge_stemmer'] = True/False - padrão True
        '''
        config = cls.__ajustar_config(config)
        tipo = 'rouge1' if 'campos_rouge1' in config else 'rouge2' if 'campos_rouge2' in config else 'rougeL'
        scorer = rouge_scorer.RougeScorer([tipo], use_stemmer=config['rouge_stemmer'], split_summaries=True)
        scores = scorer.score(texto1, texto2)
        return scores[tipo][2] # precision, recall e  >> 2 = F1

    @classmethod
    def comparar(cls, pred_json: dict, true_json: dict, retornar_dados: bool = False, id_origem = None, config: dict|None = None) -> dict:
        ''' retornar_dados = True >> retorna a lista true/pred ("key:value") usada para calcular o F1,
                                     uma chave "alinhamento" para análise dos campos comparados pela similaridade,
                                     dados de comparação entre as chaves e valores.
            id_origem = qualquer valor para armazenar no dicionário - útil em threads para permitir rastreabilidade do dado original
        '''
        config = cls.__ajustar_config(config)
        alinhamento = []
        if 'campos_alinhar' in config and any(config['campos_alinhar']):
           pred_json, true_json = deepcopy(pred_json), deepcopy(true_json)
           alinhamento = cls.alinhar_similares(pred_json, true_json, config= config)

        # 1) Flatten para listas de strings
        # os alinhados serão considerados igruais
        # os não alinhados serão considerados diferentes, ignorando a similaridade
        pred_strs = cls.json_to_flat(pred_json, config= config, ignorar_alinhados = False)
        true_strs = cls.json_to_flat(true_json, config= config, ignorar_alinhados = False)

        # 2) Calcula precision / recall / f1
        pred_set = set(pred_strs)
        true_set = set(true_strs)
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0 else 0.0)
        #
        res = {"precision": precision,
               "recall": recall,
               "f1": f1  }

        # loss - considera sempre a similaridade
        # desconsiderando o alinhamento que tornou os dados iguais para o cálculo do F1, Precision e Recall
        outras = []
        if any(alinhamento):
           # campos com alinhamento, usa as similaridades
           # campos sem alinhamento, ou é igual ou é diferente
           pred_loss = cls.json_to_flat(pred_json, config= config, ignorar_alinhados = True)
           true_loss = cls.json_to_flat(true_json, config= config, ignorar_alinhados = True)
           outras = [1-al['sim'] for al in alinhamento]
           loss = cls.distancia_jaccard(pred_loss, true_loss, outras, as_dict = True)
           res['loss'] = loss['distancia']
           res['loss_args'] = {'pred': pred_loss, 'true': true_loss, 'distancias': loss['distancias']}
        else:
           # somente campos sem alinhamento, ou é igual ou é diferente
           loss = cls.distancia_jaccard(pred_strs, true_strs, as_dict = True)
           res['loss'] = loss['distancia']
           res['loss_args'] = {'pred': pred_strs, 'true': true_strs, 'distancias': loss['distancias']}

        # finalizando
        if not (id_origem is None):
           res['id_origem'] = id_origem
        # para avaliação do dataset
        if retornar_dados:
            res['pred'] = pred_strs
            res['true'] = true_strs
            if any(alinhamento):
               res['alinhamento'] = alinhamento
        return res

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

    @classmethod
    def alinhar_similares(
        cls,
        pred_json: Union[dict, list],
        true_json: Union[dict, list],
        config: dict
    ) -> List[dict]:
        """
        ATENÇÃO: altera os dicionários enviados
        Alinha valores de pred_json aos de true_json quando similaridade ≥ thresholds[chave].
        - config['campos_embedding'] = [campo1, campo2, ...] calcula a similaridade e converte a lista de floats/int para um hash para permitir a comparação ao calcular f1, precision e recall.
        - config['campos_alinhar'] = {'nome_da_chave': float_threshold, ...}
        Retorna um log de todas as comparações feitas e se houve alinhamento.
        """
        config = cls.__ajustar_config(config)
        thresholds = config['campos_alinhar']
        campos_embedding = config['campos_embedding']
        campos_rouge = config['campos_rouge']
        campos_rouge1 = config['campos_rouge1']
        campos_rouge2 = config['campos_rouge2']
        if not any(thresholds):
           return []
        log: List[dict] = []

        def _eh_lista_emb(lista1:list, lista2:list):
            # verifica se a comparação deve ser lista de emb com lista de emb
            # a lista 1 é quem define a verdade
            if not isinstance(lista1, list):
                return False # não é lista - já sai
            return all(isinstance(_, list) for _ in lista1) # é lista de listas

        def _sim_texto(texto1, texto2, campo):
            if campo in campos_rouge:
               return 'Rouge-L', cls.rouge_scorer(texto1, texto2, config)
            if campo in campos_rouge1:
               return 'Rouge-1', cls.rouge_scorer(texto1, texto2, config)
            if campo in campos_rouge2:
               return 'Rouge-1', cls.rouge_scorer(texto1, texto2, config)
            return 'Levenshtein', 1-cls.distancia_levenshtein(texto1, texto2, padronizar_simbolos=config['padronizar_simbolos'])

        def _recurse(pred_node: Any, true_node: Any, path: str):
            if isinstance(true_node, dict):
                for key in true_node:
                    # compara os campos para alinhamento
                    th = thresholds.get(key, None)
                    if th is None:
                       # se for lista, tenta recursivo pois pode conter
                       # um dicionário com uma chave para alinhar
                       if isinstance(true_node[key], (list, tuple)):
                          _pn = pred_node.get(key) if isinstance(pred_node, dict) else dict()
                          _recurse(true_node[key], _pn, f"{path}.{key}")
                       continue
                    _sim_zero = False
                    full_path = f"{path}.{key}" if path else key
                    # corrige chaves faltantes
                    if (not isinstance(pred_node, dict)) or (key not in pred_node):
                        #print('Não existe em pred', key, list(pred_node.keys()))
                        pred_node[key] = f'~Não existe em pred~'
                        _sim_zero = True
                    #print('analisando par', key, type(true_node[key]), type(pred_node[key]))
                    # inicia a análise do par de chaves
                    v_pred = pred_node[key]
                    v_true = true_node[key]

                    if _sim_zero:
                        entry = {'chave':key,
                            'path': full_path, 'pred': pred_node[key],
                            'true': '~não comparado~', 'sim': 0, 'tipo': 'órfão'
                        }
                        # hash do embedding
                        pred_node[key] = 'órfão:'+cls.hash_string_sha1(pred_node[key])
                        true_node[key] = 'órfão:'+cls.hash_string_sha1(true_node[key])
                        log.append(entry)
                    # VETOR ÚNICO DE FLOATS vs FLOATS (embedding único)
                    elif key in campos_embedding and\
                       not _eh_lista_emb(v_true, v_pred):
                          sim = 0.0 # se não forem embeddings
                          erro = None
                          try:
                              a = np.array(v_pred, dtype=float)
                              b = np.array(v_true, dtype=float)
                              denom = np.linalg.norm(a) * np.linalg.norm(b)
                              sim = float(a.dot(b) / denom) if denom > 0 else 0.0
                          except Exception as e:
                                  if isinstance(e,(ValueError, TypeError)):
                                    erro = str(e)
                                    pass # considera diferente
                          entry = {'chave':key,
                              'path': full_path, 'pred': 'vet:'+cls.hash_string_sha1(v_pred),
                              'true': 'vet:'+cls.hash_string_sha1(v_true), 'sim': sim, 'tipo': 'embedding'
                          }
                          if sim >= th:
                              pred_node[key] = v_true.copy()
                              entry['pred'] = entry['true']
                              entry['alinhado'] = True
                          else:
                              entry['alinhado'] = False
                          if erro:
                              entry['erro'] = erro
                          # hash do embedding
                          pred_node[key] = 'vet:'+cls.hash_string_sha1(pred_node[key])
                          true_node[key] = 'vet:'+cls.hash_string_sha1(true_node[key])
                          log.append(entry)

                    # LISTA DE EMBEDDINGS (listas de listas de floats)
                    elif key in campos_embedding:
                        used_true = set()
                        ls_pred = v_pred if isinstance(v_pred, (list, tuple)) else []
                        ls_true = v_true if isinstance(v_true, (list, tuple)) else []
                        for i, emb_pred in enumerate(ls_pred):
                            best_j, best_sim = None, -1.0
                            erro = None
                            try:
                              a = np.array(emb_pred, dtype=float)
                              for j, emb_true in enumerate(ls_true):
                                  if j in used_true:
                                      continue
                                  b = np.array(emb_true, dtype=float)
                                  denom = np.linalg.norm(a) * np.linalg.norm(b)
                                  sim = float(a.dot(b) / denom) if denom > 0 else 0.0
                                  if sim > best_sim:
                                      best_sim, best_j = sim, j
                            except Exception as e:
                                  if isinstance(e,(ValueError, TypeError)):
                                    erro = str(e)
                                    pass # considera diferente

                            if best_j is not None:
                                used_true.add(best_j)
                            true_emb = ls_true[best_j] if best_j is not None else []
                            entry = {'chave':key,
                                'path': f"{full_path}[{i}]", 'pred': 'vet:'+cls.hash_string_sha1(emb_pred),
                                'true': 'vet:'+cls.hash_string_sha1(true_emb), 'sim': max(0, best_sim), 'tipo': 'embedding'
                            }
                            if best_sim >= th:
                                ls_pred[i] = true_emb.copy()
                                entry['pred'] = entry['true']
                                entry['alinhado'] = True
                            else:
                                entry['alinhado'] = False
                            if erro:
                                entry['erro'] = erro
                            log.append(entry)
                        # hash dos embeddings
                        pred_node[key] = ['vet:'+cls.hash_string_sha1(_) for _ in v_pred]
                        true_node[key] = ['vet:'+cls.hash_string_sha1(_) for _ in true_node[key]]
                        #pred_node[key] = v_pred

                    # STRING vs STRING
                    elif isinstance(v_pred, str) and isinstance(v_true, str):
                        tipo, sim = _sim_texto(v_pred, v_true, key)
                        entry = {'chave':key,
                            'path': full_path, 'pred': v_pred,
                            'true': v_true, 'sim': sim, 'tipo': tipo
                        }
                        if sim >= th:
                            pred_node[key] = v_true
                            entry['alinhado'] = True
                        else:
                            entry['alinhado'] = False
                        log.append(entry)

                    # LISTA DE STRINGS
                    elif (isinstance(v_pred, list) and isinstance(v_true, list)
                          and all(isinstance(x, str) for x in v_pred)
                          and all(isinstance(x, str) for x in v_true)):
                        used_true = set()
                        for i, item_pred in enumerate(v_pred):
                            best_j, best_sim = None, -1.0
                            tipo = '-'
                            for j, item_true in enumerate(v_true):
                                if j in used_true:
                                    continue
                                tipo, r = _sim_texto(item_pred, item_true, key)
                                if r > best_sim:
                                    best_sim, best_j = r, j
                            if best_j is not None:
                                used_true.add(best_j)
                            true_val = v_true[best_j] if best_j is not None else ''
                            entry = {'chave':key,
                                'path': f"{full_path}[{i}]", 'pred': item_pred,
                                'true': true_val, 'sim': max(0,best_sim), 'tipo': tipo
                            }
                            if best_sim >= th:
                                v_pred[i] = true_val
                                entry['alinhado'] = True
                            else:
                                entry['alinhado'] = False
                            log.append(entry)
                        pred_node[key] = v_pred
                    # Continua recursão dentro de estruturas aninhadas
                    _recurse(pred_node[key], true_node[key], full_path)

            # Se true for lista de dicts, percorre índice a índice para achar estruturas aninhadas
            elif isinstance(true_node, list) and len(true_node)>0 and isinstance(true_node[0], dict):
                lst_pred = list(pred_node) if isinstance(pred_node, (list, tuple)) else []
                max_len = max(len(lst_pred), len(true_node))
                #print(f'Avaliando lista: {path} {type(true_node)} {type(pred_node)}')
                for i in range(max_len):
                    ip = lst_pred[i] if i < len(lst_pred) else {}
                    it = true_node[i] if i < len(true_node) else {}
                    sub_path = f"{path}[{i}]"
                    #print(f'Avaliando lista: {sub_path} {type(ip)} {type(it)}')
                    ip = ip if isinstance(ip, dict) else {'~vl~': ip}
                    it = it if isinstance(it, dict) else {'~vl~': it}
                    _recurse(ip, it, sub_path)

            # caso contrário (primitivos), nada a fazer
            else:
                return

        # cria chaves em pred se não existirem
        #_corrige_pred(pred_json, true_json)
        # analisa as chaves para alinhamento
        _recurse(pred_json, true_json, "")
        # retorna
        return log

    @classmethod
    def paths(cls, dicionario: dict, finais: list):
        ''' retorna uma lista flat com as chaves e tipos dos valores
            finais é uma lista de chaves que não devem ser avaliadas recursivamente
        '''
        res = []
        _finais = set(finais) if isinstance(finais, (set, tuple, list)) else {}
        def _get_path(valor: Any, path:str, chave_final = False):
            if isinstance(valor, dict) and not chave_final:
               for key in sorted(list(valor)):
                   full_path = f"{path}.{key}" if path else key
                   _get_path(valor[key], full_path, chave_final=key in _finais)
            elif isinstance(valor, (set, list, tuple)) and not chave_final:
                 [_get_path(_v, f'{path}[{_i}]') for _i, _v in enumerate(valor)]
            else:
                 _tp = str(type(valor)).replace("<class '", "").replace("'>", "")
                 res.append(f"{path}:{_tp}")
        _get_path(dicionario, "")
        return res

    @classmethod
    def teste_compara(cls, exemplo = 1, retornar = False):
        #teste
        config = {}
        if exemplo in (1,2):
            true_json = {
              "legislacoes": [
                "LEG:FED LEI:011343 ANO:2006 LDR-06 LEI DE DROGAS ART:00033 PAR:00004 ART:00042",
                "LEG:FED RES:000005 ANO:2012 (SENADO FEDERAL)",
                "LEG:FED DEL:002848 ANO:1940 CP-40 CÓDIGO PENAL ART:00059"
              ],
              "info_complementar": None
            }
            pred_json = {
              "legislacoes": [
                "LEG:FED LEI:0011343 ANO:2006 LDR-06 LEI DE DROGAS ART:00033 PAR:00004 ART:00042",
                "LEG:FED DEL:002848 ANO:1940 CP-40 CÓDIGO PENAL ART:00059"
              ],
              "info_complementar": None
            }
            config['campos_alinhar'] = {'legislacoes':0.91}
            config['campos_rouge'] = ['legislacoes']
            if exemplo == 2:
               del true_json['legislacoes'][1]
        elif exemplo == 3:
            true_json = {"dados": ["C", "B", "A"],"frase": "Uma frase qualquer", "nome": "Luiz Anísio", 'interno': {'outra_chave': [2,4]}, 'vetor': [0.94,0.34,0.853,0.234], 'vetores': [[0.14,0.34,0.853,0.234], [0.94,0.74,0.853,0.7]]}
            pred_json = {"frase": "Uma outra frase qualquer", "nome": "Luiz       Anisio", "dados": ["A", "B", "C"], 'interno': {'outra_chave': [2,3,4]}, 'vetor': [0.94,0.34,0.853,0.54], 'vetores': [[0.97,0.74,0.853,0.7], [0.94,0.36,0.853,0.234]]}
            config['campos_alinhar'] = {'nome':0.91, 'vetor': 0.9, 'vetores': 0.99}
            config['campos_embedding'] = ['vetor','vetores']
            config['campos_lista'] = ['vetor','vetores']
            config['campos_rouge'] = ['frase']

        elif exemplo == 4:
            true_json = {"frase": "Essa é uma frase qualquer", "valor": 1}
            pred_json = {"frase": "Essa é uma outra frase qualquer", "valor": 1}
            config['campos_alinhar'] = {'frase':0.91}

        elif exemplo == 5:
            true_json = {"dados": ["C", "B", "A"], "nome": "Luiz Anísio", 'interno': {'outra_chave': [4,2]}, 'vetor': [0.93,0.34,0.853,0.234], 'vetores': [[0.14,0.34,0.853,0.234], [0.94,0.74,0.853,0.7]]}
            pred_json = {"dados": ["C", "B", "A"], "nome": "Luiz Anisio", 'interno': {'outra_chave': [2,4]}, 'vetor': [0.95,0.34,0.853,0.234], 'vetores': [[0.94,0.74,0.853,0.7], [0.15,0.33,0.853,0.234]]}
            _interno = {'Pontos': [{'Ponto': 'alinhado será', 'lista': [1,2,3]}]}
            true_json.update(_interno)
            pred_json.update(_interno)
            config={'campos_lista': ['lista'],
                    'campos_embedding': ['vetor','vetores'],
                    'campos_alinhar': ['Ponto']}
        elif exemplo == 6:
            true_json = {'Pontos': [{'Ponto': 'alinhado será', 'lista': [1,2,3]}]}
            pred_json = {'Pontos': [{'Ponto': 'alinhado foi', 'lista': [1,2,3]}]}
            config={'campos_lista': ['lista'],
                    'campos_embedding': ['vetor','vetores'],
                    'campos_alinhar': ['Ponto']}
        else:
            # exemplo com chaves faltantes
            true_json = {'vetores': [[0.14,0.34,0.853,0.234], [0.94,0.74,0.853,0.7]], 'numero': 1, 'string': 'a', 'vetor_true': [0.14,0.34,0.853,0.234] }
            pred_json = {'vetor_pred': [0.14,0.34,0.853,0.234], 'float': 0.57, 'dados': [2,5,8]}
            config={'campos_embedding': ['vetor_pred','vetor_true','vetores']}

        if not retornar:
           print(f'ALINHANDO DICIONÁRIOS PELA SIMILARIDADE EXEMPLO {exemplo}:')
        r = cls.comparar(pred_json=pred_json, true_json=true_json, config=config, retornar_dados=True)
        if retornar:
           return r
        print(json.dumps(r, indent=2, ensure_ascii=False))
