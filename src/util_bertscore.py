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
    from bert_score import score, BERTScorer
except ImportError:
    raise ImportError('Módulo bert_score não instalado. Instale com: pip install bert_score')

try:
    import torch
except ImportError:
    raise ImportError('Módulo torch não instalado. Instale com: pip install torch')
    
from typing import List, Tuple, Optional
import os
import hashlib
import json
import gc

import util  # garante que a pasta src está no sys.path
from util import UtilEnv
if UtilEnv.carregar_env('.env', pastas=['../','./']):
   pass
   
PASTA_LOCAL = UtilEnv.get_hf_home()

# ══════════════════════════════════════════════════════════════════════════════
# FIX: Desabilita meta tensors no Transformers para evitar erro "Cannot copy out of meta tensor"
# Issue: https://github.com/huggingface/transformers/issues/29651
# ══════════════════════════════════════════════════════════════════════════════
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Evita warnings em multiprocessing
# Força carregamento completo do modelo (sem lazy loading via meta tensors)
if hasattr(torch, '__version__') and torch.__version__ >= '2.0':
    # Para PyTorch 2.0+, desabilita meta device
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Cache global do modelo BERTScorer para reutilização (por cache_key)
_bert_scorer_cache = {}
_bert_scorer_lock = None

# Patch global para o erro do bert_score com transformers >= 4.40
import transformers
if hasattr(transformers, 'BertTokenizer') and not hasattr(transformers.BertTokenizer, 'build_inputs_with_special_tokens'):
    transformers.BertTokenizer.build_inputs_with_special_tokens = lambda self, t0, t1=None: [self.cls_token_id, self.sep_token_id]

# Cache para configurações de modelos (max_tokens e tokenizers)
_model_max_tokens_cache = {}
_model_tokenizer_cache = {}

# Leitura das variáveis de ambiente
BERTSCORE_DEVICE = os.getenv('BERTSCORE_DEVICE', 'auto').strip()

# Fração de overlap para janela deslizante em textos longos (0.0 a 0.9)
# Pode ser configurado via variável de ambiente BERTSCORE_OVERLAP
BERTSCORE_OVERLAP = float(os.getenv('BERTSCORE_OVERLAP', '0.5'))

# Percentual máximo do limite de posição do modelo a ser usado antes de dividir em janelas
# Ex: 0.95 significa que textos com mais de 95% dos tokens do modelo serão divididos em janelas
# Isso deixa uma margem de segurança para tokens especiais ([CLS], [SEP], etc.)
BERTSCORE_MAX_POSITION_PERCENTAGE = float(os.getenv('BERTSCORE_MAX_POSITION_PERCENTAGE', '0.95'))

# Tamanho do mini-batch para processamento de pares pelo scorer.
# Controla quantos pares são enviados ao scorer.score() por vez.
# Valores menores reduzem o consumo de memória, mas aumentam o tempo total.
# Pode ser configurado via variável de ambiente ou via YAML (configuracao_comparacao.modelos.bertscore_batch_size)
BERTSCORE_BATCH_SIZE = int(os.getenv('BERTSCORE_BATCH_SIZE', '1024'))

# Lógica 'auto' para o device
if BERTSCORE_DEVICE.lower() == 'auto':
    try:
        # Teste simples de predição para verificar GPU usando BERTScorer
        test_scorer = BERTScorer(lang="pt", device='cuda', rescale_with_baseline=False)
        # Testa com um par simples
        test_scorer.score(['a'], ['a'])
        del test_scorer
        torch.cuda.empty_cache()
        BERTSCORE_DEVICE = 'cuda'
        print("🚀 [BERTScore] CUDA detectado e ativado (auto).")
    except Exception as e:
        BERTSCORE_DEVICE = 'cpu'
        print(f"⚠️ [BERTScore] CUDA não disponível. Usando CPU (auto). Erro: {str(e)[:100]}")
elif BERTSCORE_DEVICE.lower() == 'gpu':
    BERTSCORE_DEVICE = 'cuda'
else:
    # Se não for auto/gpu, mantém o que veio (ex: cpu, cuda:0) ou fallback para cpu
    BERTSCORE_DEVICE = BERTSCORE_DEVICE or 'cpu'


# ══════════════════════════════════════════════════════════════════════════════
# Funções auxiliares para janela deslizante em textos longos
# ══════════════════════════════════════════════════════════════════════════════

def _get_max_tokens(model_type=None, lang='pt'):
    """
    Obtém o número máximo de tokens suportado pelo modelo BERT.
    
    Lê max_position_embeddings do config do modelo e subtrai 2
    (para tokens especiais [CLS] e [SEP]).
    
    Resultado é cacheado para evitar leituras repetidas.
    
    Args:
        model_type: Nome do modelo HuggingFace (ex: 'stjiris/bert-large-portuguese-cased-legal-mlm-mkd-nli-sts-v1')
        lang: Idioma, usado para inferir o modelo padrão quando model_type é None
    
    Returns:
        int: Número máximo de tokens efetivo (já descontando [CLS] e [SEP])
    """
    cache_key = model_type or lang
    if cache_key in _model_max_tokens_cache:
        return _model_max_tokens_cache[cache_key]
    
    if model_type:
        import transformers
        try:
            config_hf = transformers.AutoConfig.from_pretrained(model_type)
            max_pos = getattr(config_hf, 'max_position_embeddings', 512)
        except Exception:
            max_pos = 512  # Fallback conservador
    else:
        # Modelos padrão do bert_score para maioria dos idiomas: 512
        max_pos = 512
    
    max_tokens = int(max_pos * BERTSCORE_MAX_POSITION_PERCENTAGE)
    _model_max_tokens_cache[cache_key] = max_tokens
    return max_tokens


def _get_tokenizer(model_type=None, lang='pt'):
    """
    Obtém o tokenizer do modelo BERT para tokenização prévia.
    
    Resultado é cacheado para reutilização.
    
    Args:
        model_type: Nome do modelo HuggingFace
        lang: Idioma (padrão: 'pt')
    
    Returns:
        Tokenizer do modelo
    """
    cache_key = model_type or lang
    if cache_key in _model_tokenizer_cache:
        return _model_tokenizer_cache[cache_key]
    
    import transformers
    
    if model_type:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_type)
    else:
        # Modelo padrão usado pelo bert_score para 'pt'
        tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    _model_tokenizer_cache[cache_key] = tokenizer
    return tokenizer


def _criar_janelas_tokens(token_ids, max_tokens, overlap_frac):
    """
    Divide uma lista de IDs de tokens em janelas sobrepostas.
    
    Args:
        token_ids: Lista de IDs de tokens (sem tokens especiais)
        max_tokens: Tamanho máximo de cada janela
        overlap_frac: Fração de sobreposição entre janelas (0.0 a 0.9)
    
    Returns:
        Lista de listas de token IDs (cada uma é uma janela)
    """
    if len(token_ids) <= max_tokens:
        return [token_ids]
    
    step = int(max_tokens * (1 - overlap_frac))
    step = max(1, step)  # Garante progresso mínimo
    
    janelas = []
    for start in range(0, len(token_ids), step):
        end = start + max_tokens
        janela = token_ids[start:end]
        if len(janela) > 0:
            janelas.append(janela)
        if end >= len(token_ids):
            break  # Última janela já cobre o final
    
    return janelas


def _liberar_memoria_gpu(device_str=None):
    """Libera memória da GPU e coleta garbage."""
    gc.collect()
    if device_str and device_str.startswith('cuda'):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _calcular_scores_com_janelas(preds, trues, scorer, model_type=None, lang='pt',
                                  overlap_frac=None, verbose=False, batch_size=32,
                                  mini_batch_size=None):
    """
    Calcula BERTScore para listas de preds/trues, aplicando janela deslizante
    automaticamente para textos que excedem o limite do modelo.
    
    Textos curtos são processados em mini-batches (eficiente e seguro para memória).
    Textos longos são divididos em janelas sobrepostas e processados em mini-batches.
    O score final de textos longos é a média ponderada pelo número de tokens de cada janela.
    
    Args:
        preds: Lista de textos preditos
        trues: Lista de textos verdadeiros
        scorer: BERTScorer já carregado
        model_type: Nome do modelo (para detectar max_tokens)
        lang: Idioma (padrão: 'pt')
        overlap_frac: Fração de overlap (None = usa BERTSCORE_OVERLAP)
        verbose: Se True, exibe informações sobre janelamento
        batch_size: Tamanho do batch interno para scorer.score()
        mini_batch_size: Quantos pares processar por vez (None = usa BERTSCORE_BATCH_SIZE).
                         Controla o consumo de memória: valores menores = menos memória.
    
    Returns:
        Tupla (mP, mR, mF1) — listas de floats na mesma ordem dos inputs
    """
    if overlap_frac is None:
        overlap_frac = BERTSCORE_OVERLAP
    
    if mini_batch_size is None:
        mini_batch_size = BERTSCORE_BATCH_SIZE
    
    max_tokens = _get_max_tokens(model_type=model_type, lang=lang)
    tokenizer = _get_tokenizer(model_type=model_type, lang=lang)
    
    # Detecta device do scorer para limpeza de memória
    _device_str = getattr(scorer, 'device', None)
    if _device_str is not None:
        _device_str = str(_device_str)
    elif BERTSCORE_DEVICE:
        _device_str = BERTSCORE_DEVICE
    
    n = len(preds)
    
    # ── Identifica quais pares precisam de janelamento ──
    # Tokeniza em mini-batches para não acumular todos os tokens na memória
    indices_curtos = []
    indices_longos = []  # (idx, tok_pred, tok_true)
    
    for i in range(n):
        tok_pred = tokenizer.encode(preds[i], add_special_tokens=False)
        tok_true = tokenizer.encode(trues[i], add_special_tokens=False)
        if len(tok_pred) > max_tokens or len(tok_true) > max_tokens:
            indices_longos.append((i, tok_pred, tok_true))
        else:
            indices_curtos.append(i)
    
    # Inicializa resultados
    mP = [0.0] * n
    mR = [0.0] * n
    mF1 = [0.0] * n
    
    # ── Processa pares curtos em mini-batches ──
    if indices_curtos:
        total_curtos = len(indices_curtos)
        num_mini_batches = (total_curtos + mini_batch_size - 1) // mini_batch_size
        
        if verbose and num_mini_batches > 1:
            print(f"📦 [BERTScore] Processando {total_curtos} pares curtos em {num_mini_batches} mini-batch(es) de {mini_batch_size}")
        
        for mb_start in range(0, total_curtos, mini_batch_size):
            mb_indices = indices_curtos[mb_start:mb_start + mini_batch_size]
            preds_mb = [preds[i] for i in mb_indices]
            trues_mb = [trues[i] for i in mb_indices]
            
            P_t, R_t, F1_t = scorer.score(
                preds_mb, trues_mb,
                verbose=False, batch_size=batch_size
            )
            
            for idx_local, idx_original in enumerate(mb_indices):
                mP[idx_original] = float(P_t[idx_local])
                mR[idx_original] = float(R_t[idx_local])
                mF1[idx_original] = float(F1_t[idx_local])
            
            # Libera memória entre mini-batches
            del P_t, R_t, F1_t, preds_mb, trues_mb
            if num_mini_batches > 1:
                _liberar_memoria_gpu(_device_str)
    
    # ── Processa pares longos com janela deslizante em mini-batches ──
    if indices_longos:
        if verbose:
            print(f"📐 [BERTScore] {len(indices_longos)} par(es) com texto longo — "
                  f"usando janela deslizante (max_tokens={max_tokens}, overlap={overlap_frac:.0%})")
        
        # Processa pares longos em mini-batches para controlar memória
        # Cada par longo gera múltiplas janelas, então acumulamos janelas até
        # atingir mini_batch_size e depois processamos
        janelas_preds_acum = []
        janelas_trues_acum = []
        mapa_janelas_acum = []  # [(idx_original, inicio_batch, fim_batch, pesos)]
        total_janelas_processadas = 0
        
        def _processar_janelas_acumuladas():
            """Processa janelas acumuladas e distribui resultados."""
            nonlocal total_janelas_processadas
            if not janelas_preds_acum:
                return
            
            P_t, R_t, F1_t = scorer.score(
                janelas_preds_acum, janelas_trues_acum,
                verbose=False, batch_size=batch_size
            )
            
            for idx_original, inicio, fim, pesos in mapa_janelas_acum:
                p_vals = [float(P_t[k]) for k in range(inicio, fim)]
                r_vals = [float(R_t[k]) for k in range(inicio, fim)]
                f1_vals = [float(F1_t[k]) for k in range(inicio, fim)]
                
                soma_pesos = sum(pesos)
                if soma_pesos > 0:
                    mP[idx_original] = sum(p * w for p, w in zip(p_vals, pesos)) / soma_pesos
                    mR[idx_original] = sum(r * w for r, w in zip(r_vals, pesos)) / soma_pesos
                    mF1[idx_original] = sum(f * w for f, w in zip(f1_vals, pesos)) / soma_pesos
                else:
                    mP[idx_original] = sum(p_vals) / len(p_vals)
                    mR[idx_original] = sum(r_vals) / len(r_vals)
                    mF1[idx_original] = sum(f1_vals) / len(f1_vals)
            
            total_janelas_processadas += len(janelas_preds_acum)
            
            # Libera memória
            del P_t, R_t, F1_t
            _liberar_memoria_gpu(_device_str)
        
        for idx_original, tok_pred, tok_true in indices_longos:
            janelas_pred = _criar_janelas_tokens(tok_pred, max_tokens, overlap_frac)
            janelas_true = _criar_janelas_tokens(tok_true, max_tokens, overlap_frac)
            
            # Alinha: pad com última janela do texto mais curto
            max_janelas = max(len(janelas_pred), len(janelas_true))
            while len(janelas_pred) < max_janelas:
                janelas_pred.append(janelas_pred[-1])
            while len(janelas_true) < max_janelas:
                janelas_true.append(janelas_true[-1])
            
            # Decodifica janelas de volta para texto
            textos_pred = [tokenizer.decode(j, skip_special_tokens=True) for j in janelas_pred]
            textos_true = [tokenizer.decode(j, skip_special_tokens=True) for j in janelas_true]
            
            pesos = [len(janelas_pred[k]) + len(janelas_true[k]) for k in range(max_janelas)]
            
            # Libera tokens originais (já decodificados)
            del janelas_pred, janelas_true, tok_pred, tok_true
            
            inicio = len(janelas_preds_acum)
            janelas_preds_acum.extend(textos_pred)
            janelas_trues_acum.extend(textos_true)
            fim = len(janelas_preds_acum)
            
            mapa_janelas_acum.append((idx_original, inicio, fim, pesos))
            
            # Se acumulou janelas suficientes, processa o mini-batch
            if len(janelas_preds_acum) >= mini_batch_size:
                if verbose:
                    print(f"   📦 Processando mini-batch de {len(janelas_preds_acum)} janelas (longos)...")
                _processar_janelas_acumuladas()
                janelas_preds_acum = []
                janelas_trues_acum = []
                mapa_janelas_acum = []
        
        # Processa janelas restantes
        if janelas_preds_acum:
            if verbose and total_janelas_processadas > 0:
                print(f"   📦 Processando mini-batch final de {len(janelas_preds_acum)} janelas (longos)...")
            _processar_janelas_acumuladas()
        
        if verbose and total_janelas_processadas > 0:
            print(f"   ✅ Total de janelas processadas: {total_janelas_processadas}")
    
    return mP, mR, mF1


def _get_bert_scorer(lang='pt', device=None, model_type=None):
    """
    Obtém ou cria uma instância de BERTScorer com carregamento correto do modelo.
    
    Esta função resolve o problema de 'meta tensors' ao:
    1. Carregar o modelo explicitamente sem lazy loading
    2. Cachear a instância para reutilização
    3. Usar thread lock para segurança em multiprocessing
    
    Args:
        lang: Idioma do modelo (padrão: 'pt')
        device: Device para executar o modelo (None = usa BERTSCORE_DEVICE)
        model_type: Nome do modelo HuggingFace (ex: 'microsoft/deberta-xlarge-mnli').
                    Se especificado, sobrescreve a seleção por `lang`.
    
    Returns:
        BERTScorer configurado e pronto para uso
    """
    global _bert_scorer_cache, _bert_scorer_lock
    
    # Inicializa lock na primeira chamada
    if _bert_scorer_lock is None:
        import threading
        _bert_scorer_lock = threading.Lock()
    
    _device = device if device is not None else BERTSCORE_DEVICE
    cache_key = f"{model_type or lang}_{_device}"
    
    with _bert_scorer_lock:
        # Verifica se já existe no cache
        if cache_key in _bert_scorer_cache:
            return _bert_scorer_cache[cache_key]
        
        # Função auxiliar para evitar OverflowError no Rust backend de Tokenizers fast e outros erros
        def _fix_tokenizer_max_length(s):
            if hasattr(s, '_tokenizer'):
                if hasattr(s._tokenizer, 'model_max_length'):
                    if s._tokenizer.model_max_length > 1_000_000:
                        safe_max = 512
                        try:
                            if hasattr(s, '_model') and hasattr(s._model, 'config'):
                                safe_max = getattr(s._model.config, 'max_position_embeddings', 512)
                        except Exception:
                            pass
                        s._tokenizer.model_max_length = safe_max

        def _make_thread_safe(s):
            import threading
            if not hasattr(s, '_instance_lock'):
                s._instance_lock = threading.Lock()
                original_score = s.score
                def thread_safe_score(*args, **kwargs):
                    with s._instance_lock:
                        return original_score(*args, **kwargs)
                s.score = thread_safe_score

        # Cria novo scorer
        try:
            # SOLUÇÃO: Usa BERTScorer que carrega o modelo corretamente
            # Se model_type foi especificado, usa-o em vez de lang
            if model_type:
                print(f"🔧 [BERTScore] Carregando modelo personalizado: {model_type}")
                
                # Modelos customizados precisam ter 'num_layers' explícito se não 
                # estiverem na lista interna do bert_score
                import transformers
                try:
                    config_hf = transformers.AutoConfig.from_pretrained(model_type)
                    num_layers = getattr(config_hf, 'num_hidden_layers', None)
                except Exception as e:
                    print(f"⚠️ [BERTScore] Erro ao obter config de {model_type}: {e}")
                    num_layers = None
                    
                scorer = BERTScorer(
                    model_type=model_type,
                    num_layers=num_layers,
                    device=_device,
                    rescale_with_baseline=False,
                    batch_size=32
                )
            else:
                scorer = BERTScorer(
                    lang=lang,
                    device=_device,
                    rescale_with_baseline=False,
                    batch_size=32
                )
            
            _fix_tokenizer_max_length(scorer)
            _make_thread_safe(scorer)
            
            # Armazena metadados para verificação
            scorer._cache_key = cache_key
            _bert_scorer_cache[cache_key] = scorer
            return scorer
        except Exception as e:
            # Fallback para CPU em caso de erro
            if _device != 'cpu':
                print(f"⚠️ Erro ao carregar BERTScorer em {_device}, tentando CPU: {str(e)[:100]}")
                cache_key_cpu = f"{model_type or lang}_cpu"
                if model_type:
                    scorer = BERTScorer(
                        model_type=model_type,
                        num_layers=num_layers, # Usa o mesmo num_layers extraído acima
                        device='cpu',
                        rescale_with_baseline=False,
                        batch_size=32
                    )
                else:
                    scorer = BERTScorer(
                        lang=lang,
                        device='cpu',
                        rescale_with_baseline=False,
                        batch_size=32
                    )
                
                _fix_tokenizer_max_length(scorer)
                _make_thread_safe(scorer)
                
                scorer._cache_key = cache_key_cpu
                _bert_scorer_cache[cache_key_cpu] = scorer
                return scorer
            raise

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
    def __init__(self, cache_dir: str = None, usar_cache: bool = True, atualizar_cache: bool = True,
                 model_type: str = None):
        """
        Inicializa o gerenciador de cache.
        
        Args:
            cache_dir: Diretório para salvar os arquivos de cache.
                       Se None, usa BERTSCORE_CACHE_PATH ou padrão local.
            usar_cache: Se False, não lê do cache (sempre recalcula).
            atualizar_cache: Se False, não salva novos resultados no cache.
            model_type: Nome do modelo HuggingFace personalizado. Se especificado,
                        cria subpasta no cache para segregar resultados por modelo.
        
        Exemplos de uso:
            - usar_cache=True, atualizar_cache=True: Lê e salva (padrão)
            - usar_cache=False, atualizar_cache=True: Recalcula mas atualiza cache
            - usar_cache=True, atualizar_cache=False: Lê cache mas não salva novos
            - usar_cache=False, atualizar_cache=False: Não usa cache
        """
        if cache_dir is None:
            cache_dir = os.environ.get('BERTSCORE_CACHE_PATH')
        
        if not cache_dir:
            from util import UtilEnv
            cache_dir = UtilEnv.get_hf_home(subpasta='bs_cache')
        
        # Segregar cache por modelo personalizado (evita colisão entre modelos)
        if model_type:
            # Usa nome seguro para diretório (troca / por _)
            model_dir = model_type.replace('/', '_').replace('\\', '_')
            cache_dir = os.path.join(cache_dir, model_dir)
            
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
    
    def carregar_tudo_em_memoria(self, verbose: bool = True, chaves_necessarias: set = None) -> dict:
        """
        Carrega TODOS os arquivos de cache do disco para um dicionário em memória.
        
        OTIMIZAÇÃO CRÍTICA: Após o pré-cálculo em batch, todos os resultados já existem
        em disco como arquivos JSON individuais. Em vez de fazer ~400k operações de I/O 
        aleatório (open+read+parse por par) durante o processamento de chunks, este método
        carrega tudo de uma vez em um dict para lookup O(1) em memória.
        Se chaves_necessarias for fornecido, evita listar o diretório que pode ser muito grande.
        
        Returns:
            dict mapeando filename_sem_extensão -> {'P': float, 'R': float, 'F1': float, 'bytes1': int, 'bytes2': int}
        """
        cache_mem = {}
        if not os.path.exists(self.cache_dir):
            return cache_mem
        
        # OTIMIZAÇÃO I/O NFS: Sempre lista o diretório primeiro para evitar 
        # milhares de FileNotFoundError ao abrir chaves que não geraram cache (ex: strings vazias).
        try:
            arquivos_existentes = {f for f in os.listdir(self.cache_dir) if f.endswith('.json')}
        except OSError:
            arquivos_existentes = set()
            
        if chaves_necessarias is not None:
            # Filtra apenas os arquivos que realmente existem no disco
            arquivos = [f"{c}.json" for c in chaves_necessarias if f"{c}.json" in arquivos_existentes]
            # O que foi solicitado mas não existe contabiliza como erro (ex: string vazia)
            erros = len(chaves_necessarias) - len(arquivos)
        else:
            arquivos = list(arquivos_existentes)
            erros = 0
            
        for arquivo in arquivos:
            filepath = os.path.join(self.cache_dir, arquivo)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Chave = nome do arquivo sem extensão (ex: "hash1-hash2")
                chave = arquivo[:-5]  # remove '.json'
                cache_mem[chave] = data
            except Exception:
                erros += 1
        
        if verbose:
            print(f"   📦 [BERTScoreCache] {len(cache_mem)} entradas carregadas em memória"
                  f"{f' ({erros} erros)' if erros else ''}")
        
        return cache_mem

    @staticmethod
    def lookup_em_memoria(text1: str, text2: str, cache_mem: dict) -> tuple:
        """
        Busca resultado no cache em memória (sem I/O de disco).
        
        Método estático para evitar instanciação da classe (que faz I/O) no hot path.
        Replica a lógica de _get_key_info + leitura, usando o dict em memória.
        
        Args:
            text1: primeiro texto (pred)
            text2: segundo texto (true)
            cache_mem: dicionário retornado por carregar_tudo_em_memoria()
            
        Returns:
            (P, R, F1) se encontrado, ou None se cache miss
        """
        # Calcula hashes MD5 (mesma lógica de _get_key_info)
        b1 = str(text1).encode('utf-8')
        b2 = str(text2).encode('utf-8')
        h1 = hashlib.md5(b1).hexdigest()
        h2 = hashlib.md5(b2).hexdigest()
        
        # Ordenação determinística (mesma lógica de _get_key_info)
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
        
        # Validação de integridade (tamanho dos textos)
        if data.get('bytes1') != bytes_first or data.get('bytes2') != bytes_second:
            return None
        
        p_val = data['P']
        r_val = data['R']
        f1_val = data['F1']
        
        # Se a ordem foi trocada, P e R se invertem
        if swapped:
            p_val, r_val = r_val, p_val
        
        return (p_val, r_val, f1_val)

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


def liberar_modelos_bert_scorer():
    """
    Remove da memória todos os modelos BERTScore carregados e limpa a VRAM.
    Isso NÃO apaga o cache em disco (arquivos JSON), apenas descarrega os tensores da GPU/CPU.
    """
    global _bert_scorer_cache, _bert_scorer_lock
    
    if _bert_scorer_lock is None:
        import threading
        _bert_scorer_lock = threading.Lock()
        
    with _bert_scorer_lock:
        if _bert_scorer_cache:
            _bert_scorer_cache.clear()
            
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def bscore(preds: List[str], trues: List[str], 
           decimais: int = 3,
           verbose: bool = False,
           lang: str = 'pt',
           device: str = None,
           usar_cache: bool = True,
           atualizar_cache: bool = True,
           mini_batch_size: int = None,
           apenas_cache: bool = False,
           model_type: str = None) -> Tuple[List[float], List[float], List[float]]:
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
        model_type: Nome do modelo HuggingFace personalizado (ex: 'microsoft/deberta-xlarge-mnli').
                    Se especificado, sobrescreve a seleção por `lang`. O cache é segregado
                    automaticamente por modelo para evitar colisão.
        mini_batch_size: Quantos pares processar por vez no scorer (None = usa BERTSCORE_BATCH_SIZE).
                         Valores menores reduzem consumo de memória. Padrão via env: BERTSCORE_BATCH_SIZE=1024.
    
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
        
        # Usar modelo personalizado
        >>> P, R, F1 = bscore(preds, trues, model_type='microsoft/deberta-xlarge-mnli')
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
    # TRATAMENTO DE PARES COM STRINGS VAZIAS
    # -------------------------------------------------------------------------
    # Sentenças vazias iguais = match perfeito (1.0).
    # Se apenas uma é vazia = mismatch (0.0).
    # Remove pares tratados antes de enviar ao scorer/cache.
    def _is_text_empty(t):
        return not t or (isinstance(t, str) and t.strip() == "")
    
    indices_originais = list(range(len(preds)))
    preds_filtrados = []
    trues_filtrados = []
    resultados_vazios = {}  # idx -> (P, R, F1)
    
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
    
    # Se todos os pares são vazios, retorna direto
    if not preds_filtrados:
        final_P = [resultados_vazios[i][0] for i in range(len(preds))]
        final_R = [resultados_vazios[i][1] for i in range(len(preds))]
        final_F1 = [resultados_vazios[i][2] for i in range(len(preds))]
        if isinstance(decimais, int) and decimais > 0:
            final_P = [round(x, decimais) for x in final_P]
            final_R = [round(x, decimais) for x in final_R]
            final_F1 = [round(x, decimais) for x in final_F1]
        return final_P, final_R, final_F1
    
    # Remapeia índices para o subset filtrado
    mapa_filtrado_para_original = []
    for i in range(len(preds)):
        if i not in resultados_vazios:
            mapa_filtrado_para_original.append(i)
    
    # -------------------------------------------------------------------------
    # FASE 1: USO DO CACHE (apenas pares não-vazios)
    # -------------------------------------------------------------------------
    cache = BERTScoreCache(usar_cache=usar_cache, atualizar_cache=atualizar_cache, model_type=model_type)
    final_P_filt, final_R_filt, final_F1_filt, missed_indices, missed_preds, missed_trues, missed_meta = cache.get_batch(preds_filtrados, trues_filtrados)

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
            # SOLUÇÃO: Usa BERTScorer pré-carregado ao invés de score() direto
            # Isso evita o erro "Cannot copy out of meta tensor" que ocorre quando
            # o modelo é carregado com lazy loading (meta tensors)
            scorer = _get_bert_scorer(lang=lang, device=_device, model_type=model_type)
            
            # Calcula scores com suporte a janela deslizante para textos longos
            # Textos curtos são processados normalmente em batch.
            # Textos longos (> max_position_embeddings do modelo) são divididos
            # em janelas sobrepostas e o score final é a média ponderada.
            mP, mR, mF1 = _calcular_scores_com_janelas(
                missed_preds, missed_trues,
                scorer=scorer,
                model_type=model_type,
                lang=lang,
                verbose=verbose,
                batch_size=32,
                mini_batch_size=mini_batch_size
            )
            
            # Libera memória da GPU após processamento
            if _device.startswith('cuda'):
                torch.cuda.empty_cache()
                gc.collect()
            
        except Exception as e:
            raise RuntimeError(f"Erro ao calcular BERTScore: {e}") from e

        # Distribui resultados na lista filtrada
        for idx_missed, original_idx in enumerate(missed_indices):
            final_P_filt[original_idx] = mP[idx_missed]
            final_R_filt[original_idx] = mR[idx_missed]
            final_F1_filt[original_idx] = mF1[idx_missed]
            
        # Salva no cache
        cache.save_batch(missed_meta, mP, mR, mF1, verbose=verbose)

    # -------------------------------------------------------------------------
    # FASE 3: RECONSTRUÇÃO E ARREDONDAMENTO
    # -------------------------------------------------------------------------
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
        final_P[idx_original] = final_P_filt[j]
        final_R[idx_original] = final_R_filt[j]
        final_F1[idx_original] = final_F1_filt[j]
    
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
    ("", "", 1.0, 1.0, 1.0),  # Par de strings vazias
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