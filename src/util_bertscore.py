"""
BERTScore - Classe para cálculo de BERTScore com cache em disco baseado em MD5.

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Descrição:
-----------
Implementação simplificada do BERTScore com cache automático de resultados.
O cache é baseado em MD5 dos textos e salvo em arquivos JSON individuais.

A ordem dos textos não importa: (A, B) e (B, A) geram a mesma chave de cache,
mas os valores de P e R são trocados automaticamente quando necessário.

Como usar:
----------
    from util_bertscore import bscore
    
    # Calcular BERTScore para múltiplos pares
    preds = ['texto predito 1', 'texto predito 2']
    trues = ['texto esperado 1', 'texto esperado 2']
    
    P, R, F1 = bscore(preds, trues, lang='pt', verbose=True)
    
    # Chamadas subsequentes com os mesmos textos usam cache automaticamente
    P, R, F1 = bscore(preds, trues)  # Instantâneo! ⚡

Variáveis de ambiente:
----------------------
    BERTSCORE_DEVICE: 'cuda', 'cpu', 'auto' (padrão: 'auto')
    BERTSCORE_CACHE_PATH: Diretório para cache (padrão: ./_bertmodels/bs_cache/)

Limpeza de cache:
-----------------
    from util_bertscore import BERTScoreCache
    
    cache = BERTScoreCache()
    
    # Remove caches com mais de 24 horas (1440 minutos)
    cache.limpar_cache(tempo_minutos=1440)
    
    # Remove todo o cache
    cache.limpar_cache()
"""
try:
    from bert_score import score
except ImportError:
    raise ImportError('Módulo bert_score não instalado. Instale com: pip install bert_score')    
from typing import List, Tuple, Optional
import os
import hashlib
import json

from util import UtilEnv
if UtilEnv.carregar_env('.env', pastas=['../','./']):
   pass
   
_locais_ = [f'{_}_bertmodels/' for _ in ['./','../'] if os.path.isdir(f'{_}_bertmodels/')]
PASTA_LOCAL = _locais_[0] if len(_locais_)>0 else './_bertmodels/'

# Configura cache local se PASTA_LOCAL estiver definida
if PASTA_LOCAL:
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    os.environ['HF_HOME'] = PASTA_LOCAL

# Leitura das variáveis de ambiente
BERTSCORE_DEVICE = os.getenv('BERTSCORE_DEVICE', 'auto').strip()

# Lógica 'auto' para o device
if BERTSCORE_DEVICE.lower() == 'auto':
    try:
        # Teste simples de predição para verificar GPU
        score(['a'], ['a'], lang="pt", verbose=False, device='cuda')
        BERTSCORE_DEVICE = 'cuda'
        print("🚀 [BERTScore] CUDA detectado e ativado (auto).")
    except Exception:
        BERTSCORE_DEVICE = 'cpu'
        print("⚠️ [BERTScore] CUDA não disponível. Usando CPU (auto).")
elif BERTSCORE_DEVICE.lower() == 'gpu':
    BERTSCORE_DEVICE = 'cuda'
else:
    # Se não for auto/gpu, mantém o que veio (ex: cpu, cuda:0) ou fallback para cpu
    BERTSCORE_DEVICE = BERTSCORE_DEVICE or 'cpu'

class BERTScoreCache:
    """
    Gerencia o cache de resultados do BERTScore para evitar recálculos desnecessários.
    
    O cache é baseado no hash MD5 dos textos (hipótese e referência).
    A ordem dos textos não importa para armazenamento: (A, B) é armazenado igual a (B, A),
    mas recuperado com P e R trocados se necessário.
    
    Exemplo:
        cache = BERTScoreCache()
        P, R, F1 = cache.processar(['pred1', 'pred2'], ['true1', 'true2'])
    """
    def __init__(self, cache_dir: str = None, usar_cache: bool = True, atualizar_cache: bool = True):
        """
        Inicializa o gerenciador de cache.
        
        Args:
            cache_dir: Diretório para salvar os arquivos de cache.
                       Se None, usa BERTSCORE_CACHE_PATH ou padrão local.
            usar_cache: Se False, não lê do cache (sempre recalcula).
            atualizar_cache: Se False, não salva novos resultados no cache.
        
        Exemplos de uso:
            - usar_cache=True, atualizar_cache=True: Lê e salva (padrão)
            - usar_cache=False, atualizar_cache=True: Recalcula mas atualiza cache
            - usar_cache=True, atualizar_cache=False: Lê cache mas não salva novos
            - usar_cache=False, atualizar_cache=False: Não usa cache
        """
        if cache_dir is None:
            cache_dir = os.environ.get('BERTSCORE_CACHE_PATH')
        
        if not cache_dir:
            cache_dir = os.path.join(PASTA_LOCAL, 'bs_cache')
            
        self.cache_dir = cache_dir
        self.usar_cache = usar_cache
        self.atualizar_cache = atualizar_cache
        self._ensure_dir()
        
    def _ensure_dir(self):
        """Garante que o diretório de cache existe."""
        if not self.usar_cache and not self.atualizar_cache:
            return
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError:
            pass  # Pode falhar em concorrência, mas se existir ok

    def _get_key_info(self, text1: str, text2: str) -> dict:
        """
        Calcula hashes e informações para chave de cache.
        Normaliza a ordem para garantir que (A,B) e (B,A) gerem a mesma chave.
        """
        # Garante string e encoding
        t1_str = str(text1)
        t2_str = str(text2)
        b1 = t1_str.encode('utf-8')
        b2 = t2_str.encode('utf-8')
        
        h1 = hashlib.md5(b1).hexdigest()
        h2 = hashlib.md5(b2).hexdigest()
        
        # Ordenação determinística
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
            'h1': h1,
            'h2': h2
        }

    def get_batch(self, preds: List[str], trues: List[str]) -> Tuple[
            List[Optional[float]], List[Optional[float]], List[Optional[float]], 
            List[int], List[str], List[str], List[dict]]:
        """
        Recupera resultados do cache para uma lista de pares.
        
        Returns:
            Tuple contendo:
            - Listas de P, R, F1 (preenchidas com None onde não achou)
            - Lista de índices originais dos itens não encontrados
            - Lista de preds não encontrados
            - Lista de trues não encontrados
            - Lista de metadados para salvar os não encontrados depois
        """
        n = len(preds)
        final_P = [None] * n
        final_R = [None] * n
        final_F1 = [None] * n
        
        missed_indices = []
        missed_preds = []
        missed_trues = []
        missed_meta = []

        if not self.usar_cache:
            # Se cache desativado, marca tudo como "não encontrado"
            missed_indices = list(range(n))
            missed_preds = list(preds)
            missed_trues = list(trues)
            missed_meta = [self._get_key_info(p, t) for p, t in zip(preds, trues)]
            return final_P, final_R, final_F1, missed_indices, missed_preds, missed_trues, missed_meta

        for i, (p, t) in enumerate(zip(preds, trues)):
            info = self._get_key_info(p, t)
            filepath = info['filepath']
            swapped = info['swapped']
            
            loaded = False
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if self._validate_cache_data(data, info):
                        p_val = data['P']
                        r_val = data['R']
                        f1_val = data['F1']
                        
                        # Se cache é (A,B) e pedimos (B,A): P_ba = R_ab, R_ba = P_ab
                        if swapped:
                            p_val, r_val = r_val, p_val
                            
                        final_P[i] = p_val
                        final_R[i] = r_val
                        final_F1[i] = f1_val
                        loaded = True
                except Exception:
                    pass  # Erro de leitura/parse = cache miss

            if not loaded:
                missed_indices.append(i)
                missed_preds.append(p)
                missed_trues.append(t)
                missed_meta.append(info)
                
        return final_P, final_R, final_F1, missed_indices, missed_preds, missed_trues, missed_meta

    def _validate_cache_data(self, data: dict, info: dict) -> bool:
        """Valida se os dados do cache correspondem ao esperado."""
        required_keys = ['P', 'R', 'F1', 'bytes1', 'bytes2']
        if not all(k in data for k in required_keys):
            return False
            
        # Verificação simples de colisão/integridade por tamanho
        if data['bytes1'] != info['bytes1'] or data['bytes2'] != info['bytes2']:
            return False
            
        return True

    def save_batch(self, meta_list: List[dict], P_list: List[float], R_list: List[float], F1_list: List[float], verbose: bool = False):
        """Salva novos resultados no cache."""
        if not self.atualizar_cache:
            return
            
        for i, meta in enumerate(meta_list):
            filepath = meta['filepath']
            swapped = meta['swapped']
            
            p_val = P_list[i]
            r_val = R_list[i]
            f1_val = F1_list[i]
            
            # Se swapped, o resultado calculado foi (B,A).
            # Para salvar (A,B), invertemos P e R.
            p_save, r_save = p_val, r_val
            if swapped:
                p_save, r_save = r_val, p_val
                
            data = {
                "P": p_save,
                "R": r_save,
                "F1": f1_val,
                "bytes1": meta['bytes1'],
                "bytes2": meta['bytes2']
            }
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
            except Exception as e:
                if verbose:
                    print(f"⚠️ [BERTScoreCache] Falha ao salvar {filepath}: {e}")
    
    def limpar_cache(self, tempo_minutos: int = None, verbose: bool = True) -> int:
        """
        Remove arquivos de cache antigos.
        
        Args:
            tempo_minutos: Se especificado, remove apenas arquivos mais antigos que este tempo.
                          Se None, remove todos os arquivos do cache.
            verbose: Se True, exibe informações sobre a limpeza
        
        Returns:
            Número de arquivos removidos
        
        Exemplos:
            >>> cache = BERTScoreCache()
            >>> cache.limpar_cache(tempo_minutos=1440)  # Remove caches com mais de 24h
            >>> cache.limpar_cache()  # Remove todo o cache
        """
        if not os.path.exists(self.cache_dir):
            if verbose:
                print(f"📁 [BERTScoreCache] Diretório de cache não existe: {self.cache_dir}")
            return 0
        
        import time
        
        arquivos_removidos = 0
        tempo_limite = None
        
        if tempo_minutos is not None:
            tempo_limite = time.time() - (tempo_minutos * 60)
            if verbose:
                print(f"🧹 [BERTScoreCache] Removendo caches com mais de {tempo_minutos} minutos...")
        else:
            if verbose:
                print(f"🧹 [BERTScoreCache] Removendo todo o cache...")
        
        try:
            for arquivo in os.listdir(self.cache_dir):
                if not arquivo.endswith('.json'):
                    continue
                
                filepath = os.path.join(self.cache_dir, arquivo)
                
                # Se tempo_limite definido, verifica idade do arquivo
                if tempo_limite is not None:
                    try:
                        mtime = os.path.getmtime(filepath)
                        if mtime > tempo_limite:
                            continue  # Arquivo ainda não é antigo o suficiente
                    except OSError:
                        continue
                
                # Remove arquivo
                try:
                    os.remove(filepath)
                    arquivos_removidos += 1
                except OSError as e:
                    if verbose:
                        print(f"⚠️ [BERTScoreCache] Falha ao remover {arquivo}: {e}")
            
            if verbose:
                if arquivos_removidos > 0:
                    print(f"✅ [BERTScoreCache] {arquivos_removidos} arquivo(s) removido(s)")
                else:
                    print(f"ℹ️ [BERTScoreCache] Nenhum arquivo para remover")
        
        except Exception as e:
            if verbose:
                print(f"⚠️ [BERTScoreCache] Erro ao limpar cache: {e}")
        
        return arquivos_removidos


def bscore(preds: List[str], trues: List[str], 
           decimais: int = 3,
           verbose: bool = False,
           lang: str = 'pt',
           device: str = None,
           usar_cache: bool = True,
           atualizar_cache: bool = True,
           apenas_cache: bool = False) -> Tuple[List[float], List[float], List[float]]:
    """
    Calcula BERTScore com cache automático baseado em MD5.
    
    Args:
        preds: Lista de textos preditos/gerados
        trues: Lista de textos verdadeiros/esperados
        decimais: Número de casas decimais para arredondamento (padrão: 3)
        verbose: Se True, exibe progresso do cálculo (padrão: False)
        lang: Idioma do modelo BERT (padrão: 'pt')
        device: Dispositivo para cálculo ('cpu', 'cuda', None=auto)
        usar_cache: Se False, não lê do cache (padrão: True)
        atualizar_cache: Se False, não salva resultados no cache (padrão: True)
        apenas_cache: Se True, retorna um erro para os casos que não existe cache (padrão: False)
    
    Returns:
        Tupla (P, R, F1) com listas de floats para Precision, Recall e F1-score
    
    Raises:
        TypeError: Se preds ou trues não forem listas/tuplas
        ValueError: Se preds e trues não tiverem o mesmo tamanho
        RuntimeError: Se houver erro no processamento BERTScore
    
    Exemplos:
        >>> preds = ['O gato está dormindo', 'Python é ótimo']
        >>> trues = ['O felino está dormindo', 'Python é excelente']
        >>> P, R, F1 = bscore(preds, trues, verbose=True)
        >>> print(f"F1 médio: {sum(F1)/len(F1):.3f}")
        
        # Forçar recálculo mas atualizar cache
        >>> P, R, F1 = bscore(preds, trues, usar_cache=False, atualizar_cache=True)
        
        # Usar cache mas não salvar novos cálculos
        >>> P, R, F1 = bscore(preds, trues, usar_cache=True, atualizar_cache=False)
    """
    # Validação básica
    if not isinstance(preds, (list, tuple)) or not isinstance(trues, (list, tuple)):
        raise TypeError("preds e trues devem ser listas ou tuplas de strings")
    
    if len(preds) != len(trues):
        raise ValueError(f"preds ({len(preds)}) e trues ({len(trues)}) devem ter o mesmo tamanho")
    
    # Se apenas_cache=True, força usar_cache=True (senão seria contraditório)
    if apenas_cache and not usar_cache:
        usar_cache = True
    
    # -------------------------------------------------------------------------
    # FASE 1: USO DO CACHE
    # -------------------------------------------------------------------------
    cache = BERTScoreCache(usar_cache=usar_cache, atualizar_cache=atualizar_cache)
    final_P, final_R, final_F1, missed_indices, missed_preds, missed_trues, missed_meta = cache.get_batch(preds, trues)

    # -------------------------------------------------------------------------
    # FASE 2: PROCESSAMENTO DOS ITENS NÃO ENCONTRADOS NO CACHE
    # -------------------------------------------------------------------------
    if missed_preds:
        _device = device if device is not None else BERTSCORE_DEVICE
        if apenas_cache:
            # Lista os primeiros pares não encontrados para debug
            exemplos = []
            for i in range(min(3, len(missed_preds))):
                idx = missed_indices[i]
                exemplos.append(f"  [{idx}] '{missed_preds[i][:50]}...' vs '{missed_trues[i][:50]}...'")
            msg_exemplos = "\n".join(exemplos)
            raise RuntimeError(
                f"⚠️ [BERTScore] apenas_cache=True mas {len(missed_preds)} par(es) não encontrado(s) no cache.\n"
                f"Exemplos de pares faltantes:\n{msg_exemplos}\n"
                f"Dica: Execute primeiro sem apenas_cache=True para popular o cache."
            )
        
        try:
            # Processa em lote usando bert_score diretamente
            P_tensor, R_tensor, F1_tensor = score(
                missed_preds,
                missed_trues,
                lang=lang,
                verbose=verbose,
                device=_device
            )
            
            # Converte tensors para listas de floats
            mP = [float(p) for p in P_tensor]
            mR = [float(r) for r in R_tensor]
            mF1 = [float(f) for f in F1_tensor]
            
        except Exception as e:
            raise RuntimeError(f"Erro ao calcular BERTScore: {e}") from e

        # Distribui resultados na lista final
        for idx_missed, original_idx in enumerate(missed_indices):
            final_P[original_idx] = mP[idx_missed]
            final_R[original_idx] = mR[idx_missed]
            final_F1[original_idx] = mF1[idx_missed]
            
        # Salva no cache
        cache.save_batch(missed_meta, mP, mR, mF1, verbose=verbose)

    # -------------------------------------------------------------------------
    # FASE 3: ARREDONDAMENTO
    # -------------------------------------------------------------------------
    if isinstance(decimais, int) and decimais > 0:
        decimais = max(1, decimais)
        final_P = [round(x, decimais) for x in final_P]
        final_R = [round(x, decimais) for x in final_R]
        final_F1 = [round(x, decimais) for x in final_F1]

    return final_P, final_R, final_F1


# ============================================================================
# Testes
# ============================================================================

# pares: (hipótese, referência, F1_esperado, P_esperado, R_esperado)
# valores obtidos empiricamente com bert-base-multilingual-cased
PARES_TESTE = [
    ("O gato está no telhado", "O felino está em cima da casa", 0.859, 0.863, 0.855),
    ("Hoje está ensolarado", "O tempo está bom", 0.778, 0.762, 0.794),
    ("Ele comprou um carro novo", "Ele adquiriu um veículo recente", 0.875, 0.879, 0.872),
    ("A casa é azul", "São 4 horas da tarde", 0.721, 0.729, 0.712),  # Par com baixa similaridade semântica
]


def teste_inversao(apenas_cache: bool = False) -> bool:
    '''Testa se o cache lida corretamente com inversão de pares: (A,B) vs (B,A)'''
    print("\n" + "=" * 80)
    print("TESTE DE INVERSÃO DE PARES - Validação do Cache")
    print("=" * 80)
    
    # Prepara listas (A,B) e (B,A) usando PARES_TESTE
    textos_a = [par[0] for par in PARES_TESTE]
    textos_b = [par[1] for par in PARES_TESTE]
    
    # Calcula (A,B) sem cache
    print("\nCalculando (A,B)...")
    P_ab, R_ab, F1_ab = bscore(textos_a, textos_b, usar_cache=False, atualizar_cache=True, verbose=False, apenas_cache=apenas_cache)
    
    # Calcula (B,A) COM cache - deve inverter P/R
    print("Validando (B,A) usando cache...")
    P_ba, R_ba, F1_ba = bscore(textos_b, textos_a, usar_cache=True, atualizar_cache=False, verbose=False, apenas_cache=apenas_cache)
    
    # Valida inversão: P(A,B)=R(B,A), R(A,B)=P(B,A), F1(A,B)=F1(B,A)
    MARGEM = 0.0001
    erros = []
    
    for i in range(len(PARES_TESTE)):
        p_swap_ok = abs(P_ab[i] - R_ba[i]) <= MARGEM
        r_swap_ok = abs(R_ab[i] - P_ba[i]) <= MARGEM
        f1_ok = abs(F1_ab[i] - F1_ba[i]) <= MARGEM
        
        print(f"\nPar {i+1}: {PARES_TESTE[i][0][:30]}... ↔ {PARES_TESTE[i][1][:30]}...")
        print(f"  P(A,B)={P_ab[i]:.4f} vs R(B,A)={R_ba[i]:.4f} {'✓' if p_swap_ok else '✗'}")
        print(f"  R(A,B)={R_ab[i]:.4f} vs P(B,A)={P_ba[i]:.4f} {'✓' if r_swap_ok else '✗'}")
        print(f"  F1(A,B)={F1_ab[i]:.4f} vs F1(B,A)={F1_ba[i]:.4f} {'✓' if f1_ok else '✗'}")
        
        if not p_swap_ok:
            erros.append(f"Par {i+1}: P(A,B)={P_ab[i]:.4f} != R(B,A)={R_ba[i]:.4f}")
        if not r_swap_ok:
            erros.append(f"Par {i+1}: R(A,B)={R_ab[i]:.4f} != P(B,A)={P_ba[i]:.4f}")
        if not f1_ok:
            erros.append(f"Par {i+1}: F1(A,B)={F1_ab[i]:.4f} != F1(B,A)={F1_ba[i]:.4f}")
    
    print("\n" + "=" * 80)
    if erros:
        print("❌ TESTE DE INVERSÃO FALHOU:")
        for erro in erros:
            print(f"  - {erro}")
        return False
    else:
        print("✅ TESTE DE INVERSÃO PASSOU!")
        print("Cache está tratando corretamente a inversão de pares (A,B) ↔ (B,A)")
        return True


def teste():
    import time
    '''Teste com pequenos textos validando valores esperados para F1, P e R com margem de erro de 0.1'''
    print("=" * 80)
    print("TESTE BÁSICO - BERTScore com Cache")
    print("=" * 80)
    
    hipoteses = [par[0] for par in PARES_TESTE]
    referencias = [par[1] for par in PARES_TESTE]
    
    print("\n1ª execução (calculando):")
    inicio = time.time()
    P, R, F1 = bscore(hipoteses, referencias, verbose=True)
    duracao1 = time.time() - inicio
    
    # Validação dos resultados
    MARGEM_ERRO = 0.1
    erros = []
    
    print("\nResultados e Validação:")
    for i, (h, r, f1_esp, p_esp, r_esp) in enumerate(PARES_TESTE):
        print(f"\nPar {i+1}:")
        print(f"  Hipótese:   {h}")
        print(f"  Referência: {r}")
        print(f"  Precision:  {P[i]:.4f} (esperado: {p_esp:.3f})")
        print(f"  Recall:     {R[i]:.4f} (esperado: {r_esp:.3f})")
        print(f"  F1:         {F1[i]:.4f} (esperado: {f1_esp:.3f})")
        
        # Valida cada métrica
        if abs(P[i] - p_esp) > MARGEM_ERRO:
            erros.append(f"Par {i+1}: Precision fora da margem ({P[i]:.4f} vs {p_esp:.3f})")
        if abs(R[i] - r_esp) > MARGEM_ERRO:
            erros.append(f"Par {i+1}: Recall fora da margem ({R[i]:.4f} vs {r_esp:.3f})")
        if abs(F1[i] - f1_esp) > MARGEM_ERRO:
            erros.append(f"Par {i+1}: F1 fora da margem ({F1[i]:.4f} vs {f1_esp:.3f})")
        
        # Indica validação OK
        p_ok = "✓" if abs(P[i] - p_esp) <= MARGEM_ERRO else "✗"
        r_ok = "✓" if abs(R[i] - r_esp) <= MARGEM_ERRO else "✗"
        f1_ok = "✓" if abs(F1[i] - f1_esp) <= MARGEM_ERRO else "✗"
        print(f"  Validação:  P:{p_ok} R:{r_ok} F1:{f1_ok}")
    
    print("\n" + "=" * 80)
    print("2ª execução (usando cache - deve ser instantânea):")
    inicio = time.time()
    P2, R2, F2 = bscore(hipoteses, referencias, verbose=False)
    duracao2 = time.time() - inicio
    
    print(f"Tempo: {duracao2:.4f}s ⚡")
    print(f"Resultados idênticos: {P == P2 and R == R2 and F1 == F2}")
    print(f'Diferença de tempo: {duracao1:.4f}s vs {duracao2:.4f}s = {duracao1 - duracao2:.4f}s')
    print("=" * 80)
    
    # Relatório final
    if erros:
        print("\n❌ ERROS ENCONTRADOS:")
        for erro in erros:
            print(f"  - {erro}")
        print(f"\nMargem de erro aceita: ±{MARGEM_ERRO}")
        return False
    else:
        print("\n✅ TODOS OS TESTES PASSARAM!")
        print(f"Margem de erro aceita: ±{MARGEM_ERRO}")
        
        # Testa inversão de pares (SEM apenas_cache primeiro para garantir que cache existe)
        teste_inversao_ok = teste_inversao(apenas_cache=False)
        
        # Agora testa com apenas_cache=True (deve usar o cache criado acima)
        if teste_inversao_ok:
            print("\n" + "=" * 80)
            print("TESTE COM APENAS_CACHE=TRUE")
            print("=" * 80)
            teste_inversao_ok = teste_inversao(apenas_cache=True)
        
        return teste_inversao_ok
    

if __name__ == "__main__":
    teste()