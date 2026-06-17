
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
except ImportError:
    raise ImportError('Módulo sentence_transformers não instalado. Instale com: pip install sentence-transformers')

# ═══════════════════════════════════════════════════════════════════════════
# CACHE SBERT EM DISCO (padrão idêntico ao BERTScoreCache)
# ═══════════════════════════════════════════════════════════════════════════

import util
from util import UtilEnv

PASTA_LOCAL_SBERT = UtilEnv.get_hf_home()
TEMPO_PROGRESSO = 30  # segundos entre prints de progresso (para processamento lento)

class SBERTCache:
    """
    Cache de resultados SBERT em disco baseado em MD5 (mesmo padrão do BERTScoreCache).

    Cada par de textos gera um arquivo JSON com P, R, F1.
    A ordem dos textos não importa: (A, B) e (B, A) geram a mesma chave,
    mas P e R são trocados automaticamente quando necessário.

    O diretório de cache é separado por modelo para evitar colisões.

    Exemplo:
        cache = SBERTCache(modelo='pequeno')
        P, R, F1 = cache.processar(['pred1'], ['true1'], sbert_instance)
    """

    def __init__(self, modelo: str = 'pequeno', cache_dir: str = None,
                 usar_cache: bool = True, atualizar_cache: bool = True):
        if cache_dir is None:
            cache_dir = os.environ.get('SBERT_CACHE_PATH')
        if not cache_dir:
            from util import UtilEnv
            cache_dir = UtilEnv.get_hf_home(subpasta='sbert_cache')
            
        real_name = modelo
        if 'BERTScoreLike' in globals():
            real_name = globals()['BERTScoreLike'].resolve_model_name(modelo)
            
        modelo_hash = hashlib.md5(real_name.encode('utf-8')).hexdigest()[:8]
        folder_name = f"{modelo}_{modelo_hash}"
        
        # Subdiretório por modelo
        self.cache_dir = os.path.join(cache_dir, folder_name)
        self.modelo = folder_name
        self.usar_cache = usar_cache
        self.atualizar_cache = atualizar_cache
        self._ensure_dir()
        self._update_hash_map(cache_dir, folder_name, real_name)

    def _update_hash_map(self, base_cache_dir: str, folder_name: str, real_name: str):
        if not self.usar_cache and not self.atualizar_cache:
            return
        try:
            os.makedirs(base_cache_dir, exist_ok=True)
            map_file = os.path.join(base_cache_dir, 'models_map.json')
            
            mapping = {}
            if os.path.exists(map_file):
                try:
                    with open(map_file, 'r', encoding='utf-8') as f:
                        mapping = json.load(f)
                except Exception:
                    pass
            
            if mapping.get(folder_name) != real_name:
                mapping[folder_name] = real_name
                import tempfile
                fd, tmp_path = tempfile.mkstemp(dir=base_cache_dir, prefix='map_', suffix='.json')
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, indent=4, ensure_ascii=False)
                os.replace(tmp_path, map_file)
        except Exception:
            pass

    def _ensure_dir(self):
        if not self.usar_cache and not self.atualizar_cache:
            return
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError:
            pass

    def _get_key_info(self, text1: str, text2: str) -> dict:
        t1_str = str(text1)
        t2_str = str(text2)
        b1 = t1_str.encode('utf-8')
        b2 = t2_str.encode('utf-8')
        h1 = hashlib.md5(b1).hexdigest()
        h2 = hashlib.md5(b2).hexdigest()

        swapped = False
        if h1 > h2:
            swapped = True
            h_first, h_second = h2, h1
            bytes_first, bytes_second = len(b2), len(b1)
        else:
            h_first, h_second = h1, h2
            bytes_first, bytes_second = len(b1), len(b2)

        filename = f"{h_first}-{h_second}.json"
        filepath = os.path.join(self.cache_dir, filename)
        return {
            'filepath': filepath,
            'swapped': swapped,
            'bytes1': bytes_first,
            'bytes2': bytes_second,
        }

    def _validate_cache_data(self, data: dict, info: dict) -> bool:
        required_keys = ['P', 'R', 'F1', 'bytes1', 'bytes2']
        if not all(k in data for k in required_keys):
            return False
        if data['bytes1'] != info['bytes1'] or data['bytes2'] != info['bytes2']:
            return False
        return True

    def get_batch(self, preds: List[str], trues: List[str]) -> Tuple[
            List[Optional[float]], List[Optional[float]], List[Optional[float]],
            List[int], List[str], List[str], List[dict]]:
        """
        Recupera resultados do cache. Retorna listas de P/R/F1 (None onde ausente)
        e listas com os pares não encontrados (misses).
        """
        n = len(preds)
        final_P: List[Optional[float]] = [None] * n
        final_R: List[Optional[float]] = [None] * n
        final_F1: List[Optional[float]] = [None] * n

        missed_indices: List[int] = []
        missed_preds: List[str] = []
        missed_trues: List[str] = []
        missed_meta: List[dict] = []

        if not self.usar_cache:
            missed_indices = list(range(n))
            missed_preds = list(preds)
            missed_trues = list(trues)
            missed_meta = [self._get_key_info(p, t) for p, t in zip(preds, trues)]
            return final_P, final_R, final_F1, missed_indices, missed_preds, missed_trues, missed_meta

        for i, (p, t) in enumerate(zip(preds, trues)):
            info = self._get_key_info(p, t)
            loaded = False
            if os.path.exists(info['filepath']):
                try:
                    with open(info['filepath'], 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if self._validate_cache_data(data, info):
                        p_val = data['P']
                        r_val = data['R']
                        f1_val = data['F1']
                        if info['swapped']:
                            p_val, r_val = r_val, p_val
                        final_P[i] = p_val
                        final_R[i] = r_val
                        final_F1[i] = f1_val
                        loaded = True
                except Exception:
                    pass

            if not loaded:
                missed_indices.append(i)
                missed_preds.append(p)
                missed_trues.append(t)
                missed_meta.append(info)

        return final_P, final_R, final_F1, missed_indices, missed_preds, missed_trues, missed_meta

    def save_batch(self, meta_list: List[dict], P_list: List[float],
                   R_list: List[float], F1_list: List[float]):
        if not self.atualizar_cache:
            return
        for i, meta in enumerate(meta_list):
            p_save, r_save = P_list[i], R_list[i]
            if meta['swapped']:
                p_save, r_save = r_save, p_save
            data = {
                'P': p_save,
                'R': r_save,
                'F1': F1_list[i],
                'bytes1': meta['bytes1'],
                'bytes2': meta['bytes2'],
            }
            try:
                with open(meta['filepath'], 'w', encoding='utf-8') as f:
                    json.dump(data, f)
            except Exception:
                pass

    def carregar_tudo_em_memoria(self, verbose: bool = True, chaves_necessarias: set = None) -> dict:
        """
        Carrega todo o cache JSON deste modelo SBERT do disco para a memória.
        Se chaves_necessarias for fornecido, não fará listdir, carregando apenas
        os arquivos solicitados.
        """
        cache_mem = {}
        if not os.path.exists(self.cache_dir):
            return cache_mem
            
        if chaves_necessarias is not None:
            arquivos = [f"{c}.json" for c in chaves_necessarias]
        else:
            arquivos = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            
        erros = 0
        
        for arquivo in arquivos:
            filepath = os.path.join(self.cache_dir, arquivo)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                chave = arquivo[:-5]  # remove '.json'
                cache_mem[chave] = data
            except Exception:
                erros += 1
        
        if verbose:
            print(f"   📦 [SBERTCache:{self.modelo}] {len(cache_mem)} entradas carregadas em memória"
                  f"{f' ({erros} erros)' if erros else ''}")
        
        return cache_mem

    @staticmethod
    def lookup_em_memoria(text1: str, text2: str, cache_mem: dict) -> tuple:
        """
        Busca resultado SBERT no cache em memória (sem I/O de disco).
        
        Método estático para evitar instanciação da classe (que faz I/O) no hot path.
        Replica a lógica de _get_key_info usando o dict em memória.
        
        Returns:
            (P, R, F1) se encontrado, ou None se cache miss
        """
        # Calcula hashes MD5 (mesma lógica de _get_key_info)
        b1 = str(text1).encode('utf-8')
        b2 = str(text2).encode('utf-8')
        h1 = hashlib.md5(b1).hexdigest()
        h2 = hashlib.md5(b2).hexdigest()
        
        swapped = False
        if h1 > h2:
            swapped = True
            h_first, h_second = h2, h1
            bytes_first, bytes_second = len(b2), len(b1)
        else:
            h_first, h_second = h1, h2
            bytes_first, bytes_second = len(b1), len(b2)
        
        chave = f"{h_first}-{h_second}"
        data = cache_mem.get(chave)
        
        if data is None:
            return None
        
        if data.get('bytes1') != bytes_first or data.get('bytes2') != bytes_second:
            return None
        
        p_val = data['P']
        r_val = data['R']
        f1_val = data['F1']
        
        if swapped:
            p_val, r_val = r_val, p_val
        
        return (p_val, r_val, f1_val)

    def limpar_cache(self, tempo_minutos: int = None, verbose: bool = True) -> int:
        if not os.path.exists(self.cache_dir):
            if verbose:
                print(f"📁 [SBERTCache:{self.modelo}] Diretório não existe: {self.cache_dir}")
            return 0

        import time
        arquivos_removidos = 0
        tempo_limite = time.time() - (tempo_minutos * 60) if tempo_minutos is not None else None

        try:
            for nome in os.listdir(self.cache_dir):
                if not nome.endswith('.json'):
                    continue
                caminho = os.path.join(self.cache_dir, nome)
                if tempo_limite is not None:
                    if os.path.getmtime(caminho) > tempo_limite:
                        continue
                os.remove(caminho)
                arquivos_removidos += 1
            if verbose:
                if arquivos_removidos:
                    print(f"✅ [SBERTCache:{self.modelo}] {arquivos_removidos} arquivo(s) removido(s)")
                else:
                    print(f"ℹ️ [SBERTCache:{self.modelo}] Nenhum arquivo para remover")
        except Exception as e:
            if verbose:
                print(f"⚠️ [SBERTCache:{self.modelo}] Erro ao limpar cache: {e}")

        return arquivos_removidos


def sbert_score(preds: List[str], trues: List[str],
                modelo: str = 'pequeno',
                decimais: int = 3,
                usar_cache: bool = True,
                atualizar_cache: bool = True,
                verbose: bool = False) -> Tuple[List[float], List[float], List[float]]:
    """
    Calcula SBERT (bertscore_like) com cache automático em disco baseado em MD5.

    Padrão idêntico ao bscore() do BERTScore: verifica cache, calcula apenas os
    pares ausentes e salva no cache.

    Args:
        preds: Lista de textos preditos
        trues: Lista de textos de referência
        modelo: Tamanho do modelo SBERT ('pequeno', 'medio', 'grande')
        decimais: Casas decimais para arredondamento
        usar_cache: Se True, lê do cache em disco
        atualizar_cache: Se True, salva novos resultados no cache
        verbose: Se True, exibe progresso
        
    Returns:
        Tupla (P, R, F1) com listas de floats
    """
    try:
        from util_sysinfo import MemoryLogger
        MemoryLogger.set_nome_etapa(f"Análise - Calculando SBERT ({modelo})")
    except ImportError:
        pass

    if not isinstance(preds, (list, tuple)) or not isinstance(trues, (list, tuple)):
        raise TypeError("preds e trues devem ser listas ou tuplas de strings")
    if len(preds) != len(trues):
        raise ValueError(f"preds ({len(preds)}) e trues ({len(trues)}) devem ter o mesmo tamanho")

    cache = SBERTCache(modelo=modelo, usar_cache=usar_cache, atualizar_cache=atualizar_cache)

    # -------------------------------------------------------------------------
    # TRATAMENTO DE PARES COM STRINGS VAZIAS
    # -------------------------------------------------------------------------
    # Sentenças vazias iguais = match perfeito (1.0).
    # Se apenas uma é vazia = mismatch (0.0).
    def _is_text_empty(t):
        return not t or (isinstance(t, str) and t.strip() == "")

    preds_filtrados = []
    trues_filtrados = []
    resultados_vazios = {}  # idx -> (P, R, F1)
    mapa_filtrado_para_original = []

    for i in range(len(preds)):
        p_vazio = _is_text_empty(preds[i])
        t_vazio = _is_text_empty(trues[i])
        if p_vazio and t_vazio:
            resultados_vazios[i] = (1.0, 1.0, 1.0)
        elif p_vazio or t_vazio:
            resultados_vazios[i] = (0.0, 0.0, 0.0)
        else:
            preds_filtrados.append(preds[i])
            trues_filtrados.append(trues[i])
            mapa_filtrado_para_original.append(i)

    # Se todos os pares são vazios, retorna direto
    if not preds_filtrados:
        final_P = [round(resultados_vazios[i][0], decimais) for i in range(len(preds))]
        final_R = [round(resultados_vazios[i][1], decimais) for i in range(len(preds))]
        final_F1 = [round(resultados_vazios[i][2], decimais) for i in range(len(preds))]
        return final_P, final_R, final_F1

    final_P_filt, final_R_filt, final_F1_filt, missed_idx, missed_preds, missed_trues, missed_meta = cache.get_batch(preds_filtrados, trues_filtrados)

    if missed_preds:
        # verbose - quantidade a calcular
        if verbose and len(preds_filtrados) > 0:
            n_cache = len(preds_filtrados) - len(missed_idx)
            print(f"   [SBERTCache:{modelo}] Cache: {n_cache}/{len(preds_filtrados)} pares | Calcular: {len(missed_idx)}")
        sbert = BERTScoreLike.get_instance(modelo)
        new_P = []
        new_R = []
        new_F1 = []
        total_miss = len(missed_preds)
        t0 = time.time()
        _ultimo_print = t0
        for i_miss, (p_text, t_text) in enumerate(zip(missed_preds, missed_trues), 1):
            res = sbert.comparar_textos(
                p_text, t_text,
                metodo='bertscore_like',
                unitizador='sentencas',
                threshold=None,
                detalhes_nivel='nenhum',
            )
            new_P.append(round(res['P'], decimais))
            new_R.append(round(res['R'], decimais))
            new_F1.append(round(res['F1'], decimais))
            # Progresso a cada TEMPO_PROGRESSO segundos ou no último item
            agora = time.time()
            if verbose and (agora - _ultimo_print >= TEMPO_PROGRESSO or i_miss == total_miss):
                decorrido = agora - t0
                vel = i_miss / decorrido if decorrido > 0 else 0
                restante = (total_miss - i_miss) / vel if vel > 0 else 0
                pct = i_miss / total_miss * 100
                print(f"   [SBERT:{modelo}] {i_miss}/{total_miss} ({pct:.0f}%) "
                      f"| {decorrido:.0f}s decorrido | ~{restante:.0f}s restante "
                      f"| {vel:.1f} pares/s")
                _ultimo_print = agora

        # Salva no cache
        cache.save_batch(missed_meta, new_P, new_R, new_F1)

        # Preenche resultados filtrados
        for j, idx in enumerate(missed_idx):
            final_P_filt[idx] = new_P[j]
            final_R_filt[idx] = new_R[j]
            final_F1_filt[idx] = new_F1[j]

    # Reconstrói arrays completos mesclando resultados filtrados com pares vazios
    final_P = [0.0] * len(preds)
    final_R = [0.0] * len(preds)
    final_F1 = [0.0] * len(preds)

    # Preenche pares vazios pré-computados
    for idx, (p, r, f1) in resultados_vazios.items():
        final_P[idx] = p
        final_R[idx] = r
        final_F1[idx] = f1

    # Preenche pares calculados (mapeando de volta para índices originais)
    for j, idx_original in enumerate(mapa_filtrado_para_original):
        final_P[idx_original] = final_P_filt[j] if final_P_filt[j] is not None else 0.0
        final_R[idx_original] = final_R_filt[j] if final_R_filt[j] is not None else 0.0
        final_F1[idx_original] = final_F1_filt[j] if final_F1_filt[j] is not None else 0.0

    # Arredonda resultados finais
    final_P = [round(v, decimais) for v in final_P]
    final_R = [round(v, decimais) for v in final_R]
    final_F1 = [round(v, decimais) for v in final_F1]

    return final_P, final_R, final_F1

class BERTScoreLike:
    """
    Classe utilitária para uso de modelos Sentence-BERT (SBERT).
    Permite comparar textos e objetos JSON semanticamente.

    Implementa um "BERTScore-like" trocando token-level matching por
    matching de unidades textuais (sentenças/linhas/campos) com embeddings SBERT.
    
    Uso thread-safe (recomendado para processamento paralelo):
        sbert = BERTScoreLike.get_instance("pequeno")  # Singleton por modelo
    
    Uso direto (instância independente):
        sbert = BERTScoreLike(modelo="medio")
    """

    MODELOS = {
        "pequeno": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "medio": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "grande": "intfloat/multilingual-e5-large",
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # CACHE GLOBAL THREAD-SAFE (Singleton por modelo)
    # ═══════════════════════════════════════════════════════════════════════════
    _instances: Dict[str, "BERTScoreLike"] = {}
    _lock = threading.Lock()

    @classmethod
    def configurar_modelos(cls, novos_modelos: Dict[str, str]):
        """
        Atualiza o dicionário global de modelos SBERT.
        Ex: BERTScoreLike.configurar_modelos({'grande': 'novo/modelo'})
        """
        if novos_modelos:
            for k, v in novos_modelos.items():
                cls.MODELOS[k.lower()] = v

    @classmethod
    def get_instance(cls, modelo: str = "medio") -> "BERTScoreLike":
        """
        Obtém uma instância singleton do modelo SBERT (thread-safe).
        """
        # Resolve o nome real do modelo para usar como chave de cache
        modelo_key = cls.resolve_model_name(modelo)
        
        # Double-checked locking para performance
        if modelo_key not in cls._instances:
            with cls._lock:
                # Verifica novamente dentro do lock
                if modelo_key not in cls._instances:
                    cls._instances[modelo_key] = cls(modelo=modelo_key)
        
        return cls._instances[modelo_key]

    @classmethod
    def resolve_model_name(cls, modelo: str) -> str:
        """Resolve o nome real do modelo baseado no alias configurado no dicionário."""
        modelo_key = modelo.lower()
        return cls.MODELOS.get(modelo_key, modelo_key)
    
    @classmethod
    def clear_instances(cls):
        """
        Limpa o cache de instâncias (útil para testes ou liberar memória).
        Thread-safe.
        """
        with cls._lock:
            cls._instances.clear()

    def __init__(self, modelo: str = "medio"):
        """
        Inicializa o modelo SBERT.

        Args:
            modelo (str): Nome do modelo ou alias ("pequeno", "medio", "grande").
                          Padrão: "medio".
        """
        import threading
        self._instance_lock = threading.Lock()
        self.nome_modelo = self.MODELOS.get(modelo.lower(), modelo)
        print(f"Carregando modelo SBERT: {self.nome_modelo} ...")
        self.model = SentenceTransformer(self.nome_modelo)
        print("Modelo SBERT carregado.")
        self._emb_cache: Dict[str, np.ndarray] = {}

    # -------------------------
    # Utilitários
    # -------------------------

    @staticmethod
    def _norm_text(s: Any) -> str:
        if s is None:
            return ""
        s = str(s)
        s = re.sub(r"\s+", " ", s.strip())
        return s

    @staticmethod
    def _split_sentencas(texto: str) -> List[str]:
        t = BERTScoreLike._norm_text(texto)
        if not t:
            return []
        partes = re.split(r"(?<=[\.\!\?\;\:])\s+|\n+", t)
        units = [p.strip() for p in partes if p and p.strip()]
        return units if units else [t]

    @staticmethod
    def _split_linhas(texto: str) -> List[str]:
        # mantém unidades “campo a campo” (linhas) – ideal para JSON planificado
        t = texto or ""
        linhas = [BERTScoreLike._norm_text(x) for x in str(t).splitlines()]
        return [x for x in linhas if x]

    def _encode_texts(self, textos: List[str]) -> np.ndarray:
        """
        Encode com normalize_embeddings=True (para cos_sim efetivo).
        Cache por string exata (normalizada).
        """
        if not textos:
            return np.zeros((0, 1), dtype=np.float32)

        with self._instance_lock:
            embs: List[Optional[np.ndarray]] = [None] * len(textos)
            to_encode: List[str] = []
            idx_map: List[int] = []

            for i, t in enumerate(textos):
                t_norm = self._norm_text(t)
                if not t_norm:
                    embs[i] = None
                    continue
                if t_norm in self._emb_cache:
                    embs[i] = self._emb_cache[t_norm]
                else:
                    to_encode.append(t_norm)
                    idx_map.append(i)

            if to_encode:
                enc = self.model.encode(
                    to_encode,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                for j, i in enumerate(idx_map):
                    self._emb_cache[to_encode[j]] = enc[j]
                    embs[i] = enc[j]

            dim = next((e.shape[0] for e in embs if e is not None), 0)
            if dim == 0:
                return np.zeros((len(textos), 1), dtype=np.float32)

            out = np.vstack([
                e if e is not None else np.zeros((dim,), dtype=np.float32)
                for e in embs
            ])
            return out

    @staticmethod
    def _apply_threshold(x: np.ndarray, threshold: Optional[float]) -> np.ndarray:
        if threshold is None:
            return x
        return np.where(x >= threshold, x, 0.0)

    @staticmethod
    def _f1(p: float, r: float) -> float:
        return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def _cosine_texto_inteiro(self, a: str, b: str) -> float:
        a = self._norm_text(a)
        b = self._norm_text(b)
        if not a and not b:
            return 1.0
        emb = self.model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return float(np.dot(emb[0], emb[1]))

    def _bertscore_like_unidades(
        self,
        cand_units: List[str],
        ref_units: List[str],
        threshold: Optional[float] = None,
        detalhes_nivel: str = "simples",  # "nenhum" | "simples" | "alinhamento"
    ) -> Dict[str, Any]:
        """
        Precision: média_i max_j cos(cand_i, ref_j)
        Recall:    média_j max_i cos(cand_i, ref_j)
        F1:        harmônica(P, R)

        threshold: se definido, similaridades < threshold viram 0.
        detalhes_nivel:
          - "nenhum": detalhes={}
          - "simples": estatísticas agregadas
          - "alinhamento": inclui melhor match por unidade (índice e score)
        """
        if not cand_units and not ref_units:
            return {"P": 1.0, "R": 1.0, "F1": 1.0, "detalhes": {}}
        if not cand_units or not ref_units:
            return {"P": 0.0, "R": 0.0, "F1": 0.0, "detalhes": {
                "n_cand_units": len(cand_units),
                "n_ref_units": len(ref_units),
                "motivo": "uma das listas de unidades está vazia",
            }}

        cand_emb = self._encode_texts(cand_units)
        ref_emb = self._encode_texts(ref_units)

        sim_t = sbert_util.cos_sim(cand_emb, ref_emb)  # torch tensor
        sim = sim_t.detach().cpu().numpy()

        # melhores matches
        best_ref_idx_for_cand = sim.argmax(axis=1)
        best_cand_scores = sim.max(axis=1)

        best_cand_idx_for_ref = sim.argmax(axis=0)
        best_ref_scores = sim.max(axis=0)

        # threshold
        best_cand_scores_thr = self._apply_threshold(best_cand_scores, threshold)
        best_ref_scores_thr = self._apply_threshold(best_ref_scores, threshold)

        P = float(best_cand_scores_thr.mean()) if best_cand_scores_thr.size else 0.0
        R = float(best_ref_scores_thr.mean()) if best_ref_scores_thr.size else 0.0
        F1 = self._f1(P, R)

        detalhes: Dict[str, Any] = {}
        if detalhes_nivel != "nenhum":
            detalhes.update({
                "n_cand_units": len(cand_units),
                "n_ref_units": len(ref_units),
                "threshold": threshold,
                "P_mean_raw": float(best_cand_scores.mean()),
                "R_mean_raw": float(best_ref_scores.mean()),
                "P_mean_thr": P,
                "R_mean_thr": R,
            })

        if detalhes_nivel == "alinhamento":
            # inclui o melhor par para cada unidade (pode ser grande; use com parcimônia)
            detalhes["alinhamento_cand_para_ref"] = [
                {"cand_i": i, "ref_j": int(best_ref_idx_for_cand[i]), "score": float(best_cand_scores[i])}
                for i in range(len(cand_units))
            ]
            detalhes["alinhamento_ref_para_cand"] = [
                {"ref_j": j, "cand_i": int(best_cand_idx_for_ref[j]), "score": float(best_ref_scores[j])}
                for j in range(len(ref_units))
            ]

        return {"P": P, "R": R, "F1": F1, "detalhes": detalhes}

    # -------------------------
    # Texto: API pública (unificada)
    # -------------------------

    def comparar_textos(
        self,
        candidato: str,
        referencia: str,
        metodo: str = "bertscore_like",         # "cosine" | "media" | "bertscore_like"
        unitizador: str = "sentencas",          # "sentencas" | "linhas"
        threshold: Optional[float] = None,
        detalhes_nivel: str = "simples",        # "nenhum" | "simples" | "alinhamento"
    ) -> Dict[str, Any]:
        """
        Retorna sempre: {"P","R","F1","detalhes"}.

        Args:
            candidato (str): Texto candidato (predição).
            referencia (str): Texto de referência (gabarito).
        """
        metodo = (metodo or "bertscore_like").lower()
        unitizador = (unitizador or "sentencas").lower()

        t_cand = self._norm_text(candidato)
        t_ref = self._norm_text(referencia)

        if metodo in ("cosine", "media"):
            s = self._cosine_texto_inteiro(t_cand, t_ref)
            return {
                "P": s,
                "R": s,
                "F1": s,
                "detalhes": {
                    "metodo": "cosine_texto_inteiro",
                    "threshold": None,
                } if detalhes_nivel != "nenhum" else {},
            }

        if metodo == "bertscore_like":
            if unitizador == "linhas":
                u_cand = self._split_linhas(candidato)  # preserva \n
                u_ref = self._split_linhas(referencia)
            elif unitizador == "sentencas":
                u_cand = self._split_sentencas(t_cand)
                u_ref = self._split_sentencas(t_ref)
            else:
                raise ValueError(f"unitizador inválido: {unitizador}. Use 'sentencas' ou 'linhas'.")

            out = self._bertscore_like_unidades(u_cand, u_ref, threshold=threshold, detalhes_nivel=detalhes_nivel)
            if detalhes_nivel != "nenhum":
                out["detalhes"]["metodo"] = f"bertscore_like_{unitizador}"
            return out

        raise ValueError(f"metodo inválido: {metodo}. Use 'cosine'/'media' ou 'bertscore_like'.")

    # -------------------------
    # JSON: preparar texto/unidades e reutilizar comparar_textos
    # -------------------------

    @staticmethod
    def _flatten_json(obj: Any, prefix: str = "") -> Dict[str, str]:
        out: Dict[str, str] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                out.update(BERTScoreLike._flatten_json(v, key))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                key = f"{prefix}[{i}]"
                out.update(BERTScoreLike._flatten_json(v, key))
        else:
            out[prefix] = BERTScoreLike._norm_text(obj)
        return out

    @staticmethod
    def _render_fields(fields: Dict[str, str], include_key_ctx: bool = True) -> List[str]:
        linhas: List[str] = []
        for k in sorted(fields.keys()):
            v = fields[k]
            linhas.append(f"{k}: {v}" if include_key_ctx else v)
        return linhas

    def comparar_json(
        self,
        candidato: Any,
        referencia: Any,
        include_key_ctx: bool = True,
        metodo: str = "bertscore_like",       # delega para comparar_textos
        threshold: Optional[float] = None,
        detalhes: str = "simples",            # "nenhum" | "simples" | "chaves"
        detalhes_nivel_texto: str = "simples" # repassa p/ comparar_textos
    ) -> Dict[str, Any]:
        """
        Prepara o JSON como texto planificado e reutiliza comparar_textos().
        
        Args:
            candidato (Any): Objeto candidato (predição).
            referencia (Any): Objeto de referência (gabarito).
        """
        f_cand = self._flatten_json(candidato)
        f_ref = self._flatten_json(referencia)

        linhas_cand = self._render_fields(f_cand, include_key_ctx=include_key_ctx)
        linhas_ref = self._render_fields(f_ref, include_key_ctx=include_key_ctx)

        # Texto canônico (uma linha por campo)
        txt_cand = "\n".join(linhas_cand)
        txt_ref = "\n".join(linhas_ref)

        out = self.comparar_textos(
            txt_cand,
            txt_ref,
            metodo=metodo,
            unitizador="linhas",  # aqui faz sentido 1 unidade = 1 campo
            threshold=threshold,
            detalhes_nivel=detalhes_nivel_texto,
        )

        # Enriquecimento opcional de detalhes específicos do JSON
        det = out.get("detalhes", {}) if isinstance(out, dict) else {}
        if detalhes != "nenhum":
            det.update({
                "tipo": "json_planificado",
                "include_key_ctx": include_key_ctx,
                "n_campos_cand": len(f_cand),
                "n_campos_ref": len(f_ref),
            })

        if detalhes == "chaves":
            keys_cand = set(f_cand.keys())
            keys_ref = set(f_ref.keys())
            common = keys_cand & keys_ref
            det.update({
                "coverage_chaves_cand": float(len(common) / max(1, len(keys_cand))),
                "missing_chaves_cand_em_ref": sorted(keys_cand - keys_ref),
                "extra_chaves_ref": sorted(keys_ref - keys_cand),
            })

        out["detalhes"] = det
        return out

if __name__ == "__main__":
    # Inicializando modelo SBERT para testes (uma única vez)...
    # identifica o tamanho do modelo pelo argumento
    import sys
    from util_print import print_cores
    if len(sys.argv) > 1:
        modelo = sys.argv[1]
    else:
        modelo = "medio"
    if modelo=='stjiris':
        modelo = 'stjiris/bert-large-portuguese-cased-legal-mlm-mkd-nli-sts-v1'
    m = BERTScoreLike(modelo=modelo)

    # Texto: BERTScore-like com SBERT
    print_cores("\n<azul>=== TESTE DE COMPARAÇÃO DE TEXTOS ===</azul>")
    textos = [("A decisão foi reformada.", "O acórdão foi modificado."), 
              ("O recurso especial não foi conhecido no Tribunal por ausência de prequestionamento.", 
               "O STJ não conheceu do recurso especial em razão da falta de prequestionamento."), 
              ("O recurso especial não foi conhecido no Tribunal por ausência de prequestionamento.", 
               "O recurso especial foi conhecido e provido, afastando a alegada ausência de prequestionamento."),
    ]
    for i, texto in enumerate(textos, 1):
        p1, p2 = texto
        resultado = m.comparar_textos(p1, p2, metodo="bertscore_like", threshold=0.50)
        print_cores(f"<amarelo>Par {i}:</amarelo>")
        print_cores(f"  <cinza>Texto 1:</cinza> {p1}")
        print_cores(f"  <cinza>Texto 2:</cinza> {p2}")
        f1, p, r = resultado.get('F1', 0), resultado.get('P', 0), resultado.get('R', 0)
        cor = "verde" if f1 >= 0.7 else "vermelho"
        print_cores(f"  <cinza>Resultado:</cinza> <{cor}>P: {p:.3f} | R: {r:.3f} | F1: {f1:.3f}</{cor}>")
        print_cores("<cinza>" + "-" * 50 + "</cinza>")

    print_cores("\n<azul>=== TESTE DE COMPARAÇÃO DE JSON ===</azul>")
    # JSON: robusto a troca de chave
    gold = {"decisao": "negou provimento", "fundamento": "ausência de prova"}
    pred = {"fundamento": "não havia prova suficiente", "decisao": "provimento negado"}
    resultado = m.comparar_json(pred, gold, include_key_ctx=True, threshold=0.65)
    print_cores("<amarelo>Robusto a troca de chave:</amarelo>")
    print_cores(f"  <cinza>Gold:</cinza> {gold}")
    print_cores(f"  <cinza>Pred:</cinza> {pred}")
    f1, p, r = resultado.get('F1', 0), resultado.get('P', 0), resultado.get('R', 0)
    cor = "verde" if f1 >= 0.7 else "vermelho"
    print_cores(f"  <cinza>Resultado:</cinza> <{cor}>P: {p:.3f} | R: {r:.3f} | F1: {f1:.3f}</{cor}>")
    print_cores("<cinza>" + "-" * 50 + "</cinza>")

    print_cores("<amarelo>Exemplo com Precision alto e Recall baixo:</amarelo>")
    gold = {"decisao": "negou provimento", "fundamento": "ausência de prova"}
    pred = {"decisao": "negou provimento"}
    resultado = m.comparar_json(pred, gold, include_key_ctx=True, threshold=0.65)
    print_cores(f"  <cinza>Gold:</cinza> {gold}")
    print_cores(f"  <cinza>Pred:</cinza> {pred}")
    f1, p, r = resultado.get('F1', 0), resultado.get('P', 0), resultado.get('R', 0)
    cor = "verde" if f1 >= 0.7 else "vermelho"
    print_cores(f"  <cinza>Resultado:</cinza> <{cor}>P: {p:.3f} | R: {r:.3f} | F1: {f1:.3f}</{cor}>")
    print_cores("<cinza>" + "-" * 50 + "</cinza>")

    print_cores("<amarelo>Exemplo com Precision baixo e Recall alto:</amarelo>")
    gold = {"decisao": "negou provimento", "fundamento": "ausência de prova"}
    pred = {"decisao": "negou provimento", "fundamento": "ausência de prova", "outro": "outro"}
    resultado = m.comparar_json(pred, gold, include_key_ctx=True, threshold=0.65)
    print_cores(f"  <cinza>Gold:</cinza> {gold}")
    print_cores(f"  <cinza>Pred:</cinza> {pred}")
    f1, p, r = resultado.get('F1', 0), resultado.get('P', 0), resultado.get('R', 0)
    cor = "verde" if f1 >= 0.7 else "vermelho"
    print_cores(f"  <cinza>Resultado:</cinza> <{cor}>P: {p:.3f} | R: {r:.3f} | F1: {f1:.3f}</{cor}>")
