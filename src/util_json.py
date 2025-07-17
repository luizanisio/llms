# -*- coding: utf-8 -*-

'''
 Autor Luiz Anísio 17/07/2025
 Utilitários para avaliar F1, Pecision e Recall de Jsons
'''

try:
   import Levenshtein
except ImportError:
   raise ImportError('Considere instalar: pip install python-Levenshtein')
import json
from statistics import harmonic_mean, mean
import re
import numpy as np
from typing import Any, List, Union
from enum import Enum
import hashlib
from copy import deepcopy
from datetime import datetime

class JsonAnalise:
    RE_UNE_ESPACO = re.compile(r"\s+")
    RE_UNE_ENTER = re.compile(r"\n+")

    @classmethod
    def verifica_versao(cls):
        dthr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'Util carregado corretamente em {dthr}!')
     
    @classmethod
    def padronizar_simbolos(cls, texto: Union[str, dict]) -> str:
        """ Padroniza alguns símbolos para comparação mais precisa """
        # Une quebras de linha em espaço
        saida = cls.RE_UNE_ENTER.sub(' ', texto.strip())
        # Une múltiplos espaços em um único
        saida = cls.RE_UNE_ESPACO.sub(' ', saida)
        # Corrige aspas especiais
        saida = saida.replace("“", '"').replace("”", '"')
        return saida

    @classmethod
    def json_to_flat(
        cls,
        obj: Union[dict, list],
        padronizar_simbolos: bool = True
    ) -> List[str]:
        """
        Recebe um dict ou lista e retorna uma lista de strings "chave:valor",
        achatando todos os níveis de dicionários e listas aninhadas.
        Não leva em consideração a ordem dos itens
        
        Exemplos de saída:
        
        >>> AvaliaJson.json_to_flat({"a": 1, "b": [2,3]})
        ["a:1", "b:2", "b:3"]
        
        >>> AvaliaJson.json_to_flat({"x": {"y": [4, {"z": 5}]}})
        ["x.y:4", "x.y.z:5"]
        """
        flat: List[str] = []

        def _flatten(prefix: str, value: Any):
            # Caso seja dict, itera chaves e aprofunda
            if isinstance(value, dict):
                for k, v in value.items():
                    novo_prefixo = f"{prefix}.{k}" if prefix else k
                    _flatten(novo_prefixo, v)
            # Caso seja lista, itera itens com mesmo prefixo
            elif isinstance(value, list):
                for item in value:
                    _flatten(prefix, item)
            # Caso seja valor primitivo, adiciona ao resultado
            else:
                val = value
                if padronizar_simbolos and isinstance(val, str):
                    val = cls.padronizar_simbolos(val)
                flat.append(f"{prefix}:{val}")

        # Dispara a recursão a partir do objeto raiz
        _flatten('', obj)
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
        return Levenshtein.distance(texto1, texto2)

    @classmethod
    def comparar_json(cls, pred_json: dict, true_json: dict, retornar_dados = False, id_original = None) -> dict:
        ''' retornar_dados  retorna as chaves true/pred com os dados originais dos rótulos key:value para análise posterior
        '''
        # 1) Flatten para listas de strings
        pred_strs = cls.json_to_flat(pred_json)
        true_strs = cls.json_to_flat(true_json)

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
        if id_original:
          res['id'] = id_original
        # para avaliação do dataset
        if retornar_dados:
            res['pred'] = pred_strs
            res['true'] = true_strs
        return res

    @classmethod
    def hash_string_sha1(cls, texto):
        ''' retorna o sha1 do texto ou json recebido '''
        if isinstance(texto, dict):
           _txt = json.dumps(texto, sort_keys = True).encode("utf-8")
        else:
           _txt = '|'.join(str(texto)) if type(texto) is list else str(texto)
        return hashlib.sha1(_txt.encode('utf-8')).hexdigest()       

    @classmethod
    def alinhar_similares(
        cls,
        pred_json: Union[dict, list],
        true_json: Union[dict, list],
        thresholds: dict
    ) -> List[dict]:
        """
        Alinha valores de pred_json aos de true_json quando similaridade ≥ thresholds[chave].
        thresholds: {'nome_da_chave': float_threshold, ...}
        embedding: converte a lista de floats para um hash para permitir a comparação e calcular f1, precision e recall.
        Retorna um log de todas as comparações feitas e se houve alinhamento.
        """
        log: List[dict] = []

        def _recurse(pred_node: Any, true_node: Any, path: str):
            if isinstance(pred_node, dict) and isinstance(true_node, dict):
                for key in pred_node:
                    if key not in true_node:
                        continue
                    full_path = f"{path}.{key}" if path else key
                    th = thresholds.get(key, None)

                    if th is not None:
                        v_pred = pred_node[key]
                        v_true = true_node[key]

                        # 1) STRING vs STRING
                        if isinstance(v_pred, str) and isinstance(v_true, str):
                            s1 = cls.padronizar_simbolos(v_pred)
                            s2 = cls.padronizar_simbolos(v_true)
                            sim = Levenshtein.ratio(s1, s2)
                            entry = {
                                'chave': full_path, 'pred': v_pred,
                                'true': v_true, 'sim': sim
                            }
                            if sim >= th:
                                pred_node[key] = v_true
                                entry['alinhado'] = True
                            else:
                                entry['alinhado'] = False
                            log.append(entry)

                        # 2) VETOR ÚNICO DE FLOATS vs FLOATS (embedding único)
                        elif (isinstance(v_pred, list) and isinstance(v_true, list)
                              and all(isinstance(x, (int, float)) for x in v_pred)
                              and all(isinstance(x, (int, float)) for x in v_true)):
                            a = np.array(v_pred, dtype=float)
                            b = np.array(v_true, dtype=float)
                            denom = np.linalg.norm(a) * np.linalg.norm(b)
                            sim = float(a.dot(b) / denom) if denom > 0 else 0.0
                            entry = {
                                'chave': full_path, 'pred': cls.hash_string_sha1(v_pred),
                                'true': cls.hash_string_sha1(v_true), 'sim': sim
                            }
                            if sim >= th:
                                pred_node[key] = v_true.copy()
                                entry['pred'] = entry['true']
                                entry['alinhado'] = True
                            else:
                                entry['alinhado'] = False
                            # hash do embedding
                            pred_node[key] = cls.hash_string_sha1(pred_node[key])
                            true_node[key] = cls.hash_string_sha1(true_node[key])
                            log.append(entry)

                        # 3) LISTA DE STRINGS
                        elif (isinstance(v_pred, list) and isinstance(v_true, list)
                              and all(isinstance(x, str) for x in v_pred)
                              and all(isinstance(x, str) for x in v_true)):
                            used_true = set()
                            for i, item_pred in enumerate(v_pred):
                                s1 = cls.padronizar_simbolos(item_pred)
                                best_j, best_sim = None, -1.0
                                for j, item_true in enumerate(v_true):
                                    if j in used_true:
                                        continue
                                    s2 = cls.padronizar_simbolos(item_true)
                                    r = Levenshtein.ratio(s1, s2)
                                    if r > best_sim:
                                        best_sim, best_j = r, j
                                if best_j is not None:
                                    used_true.add(best_j)
                                true_val = v_true[best_j] if best_j is not None else ''
                                entry = {
                                    'chave': f"{full_path}[{i}]", 'pred': item_pred,
                                    'true': true_val, 'sim': best_sim
                                }
                                if best_sim >= th:
                                    v_pred[i] = true_val
                                    entry['alinhado'] = True
                                else:
                                    entry['alinhado'] = False
                                log.append(entry)
                            pred_node[key] = v_pred

                        # 4) LISTA DE EMBEDDINGS (listas de listas de floats)
                        elif (isinstance(v_pred, list) and isinstance(v_true, list)
                              and all(isinstance(item, list) and all(isinstance(x, (int, float)) for x in item)
                                      for item in v_pred)
                              and all(isinstance(item, list) and all(isinstance(x, (int, float)) for x in item)
                                      for item in v_true)):
                            used_true = set()
                            for i, emb_pred in enumerate(v_pred):
                                a = np.array(emb_pred, dtype=float)
                                best_j, best_sim = None, -1.0
                                for j, emb_true in enumerate(v_true):
                                    if j in used_true:
                                        continue
                                    b = np.array(emb_true, dtype=float)
                                    denom = np.linalg.norm(a) * np.linalg.norm(b)
                                    sim = float(a.dot(b) / denom) if denom > 0 else 0.0
                                    if sim > best_sim:
                                        best_sim, best_j = sim, j
                                if best_j is not None:
                                    used_true.add(best_j)
                                true_emb = v_true[best_j] if best_j is not None else []
                                entry = {
                                    'chave': f"{full_path}[{i}]", 'pred': cls.hash_string_sha1(emb_pred),
                                    'true': cls.hash_string_sha1(true_emb), 'sim': best_sim
                                }
                                if best_sim >= th:
                                    v_pred[i] = true_emb.copy()
                                    entry['pred'] = entry['true']
                                    entry['alinhado'] = True
                                else:
                                    entry['alinhado'] = False
                                log.append(entry)
                            # hash dos embeddings
                            pred_node[key] = [cls.hash_string_sha1(_) for _ in v_pred]
                            true_node[key] = [cls.hash_string_sha1(_) for _ in true_node[key]]
                            #pred_node[key] = v_pred

                    # Continua recursão dentro de estruturas aninhadas
                    _recurse(pred_node[key], true_node[key], full_path)

            # Se ambos forem listas, percorre índice a índice para achar estruturas aninhadas
            elif isinstance(pred_node, list) and isinstance(true_node, list):
                max_len = max(len(pred_node), len(true_node))
                for i in range(max_len):
                    ip = pred_node[i] if i < len(pred_node) else None
                    it = true_node[i] if i < len(true_node) else None
                    sub_path = f"{path}[{i}]"
                    _recurse(ip, it, sub_path)

            # caso contrário (primitivos), nada a fazer
            else:
                return

        _recurse(pred_json, true_json, "")
        return log

    @classmethod
    def alinhar_comparar(cls, pred_json, true_json, thresholds, preservar_jsons:bool = False):
        ''' preservar_jsons = True copia os jsons para não alterá-los
        '''
        if preservar_jsons:
            pred_json = deepcopy(pred_json)
            true_json = deepcopy(true_json)
        log = cls.alinhar_similares(pred_json, true_json, thresholds)
        res = cls.comparar_json(pred_json, true_json)
        res['alinhamento'] = log
        return res

    @classmethod
    def teste_compara(cls, exemplo = 1):
        #teste
        if exemplo == 1:
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
            compara = {'legislacoes':0.91}
        elif exemplo == 2:
            true_json = {"dados": ["C", "B", "A"], "nome": "Luiz Anísio", 'interno': {'outra_chave': [2,4]}, 'vetor': [0.94,0.34,0.853,0.234], 'vetores': [[0.14,0.34,0.853,0.234], [0.94,0.74,0.853,0.7]]}
            pred_json = {"nome": "Luiz       Anisio", "dados": ["A", "B", "C"], 'interno': {'outra_chave': [2,3,4]}, 'vetor': [0.94,0.34,0.853,0.54], 'vetores': [[0.97,0.74,0.853,0.7], [0.94,0.36,0.853,0.234]]}
            compara = {'nome':0.91, 'vetor': 0.9, 'vetores': 0.99}
        else:
            true_json = {"frase": "Essa é uma frase qualquer", "valor": 1}
            pred_json = {"frase": "Essa é uma outra frase qualquer", "valor": 1}
            compara = {'frase':0.91}
            
        print(f'ALINHANDO DICIONÁRIOS PELA SIMILARIDADE EXEMPLO {exemplo}:')
        r = cls.alinhar_comparar(pred_json=pred_json, true_json=true_json, thresholds=compara, preservar_jsons=True)
        print(json.dumps(r, indent=2, ensure_ascii=False))
