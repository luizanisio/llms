"""
BERTScoreService - Serviço singleton para cálculo de BERTScore.

Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Descrição:
-----------
Processa requisições assíncronas de múltiplas threads usando processos isolados.
Resolve problema de predição em ambientes multithread.

Como usar:
----------
    # Obter a instância do serviço (cria apenas uma vez)
    # workers: número de processos > padrão: 0 onde 0 = número de CPUs, -1 = CPUs menos 1
    # max_workers: número máximo de workers automáticos (se workers < 0)
    service = BERTScoreService.get_instance(workers=5, lang="pt")

    # Usar em qualquer thread
    P, R, F1 = service.processar(hipoteses, referencias)
    
    # O serviço é encerrado automaticamente ao finalizar o programa
    
    # Para obter estatísticas do serviço.
    stats = service.get_stats() 

    ENV: BERTSCORE_DEVICE=cuda (para usar gpu) ou auto para detectar automaticamente

"""
try:
    from bert_score import score
except ImportError:
    raise ImportError('Módulo bert_score não instalado. Instale com: pip install bert_score')    
from multiprocessing import Manager, Process, set_start_method, get_start_method
from typing import List, Tuple, Optional
import uuid
import threading
from multiprocessing import cpu_count
import time
import atexit
import os

# Importa BERTScoreCache do módulo simplificado
from util_bertscore import BERTScoreCache

# Força o método 'spawn' para evitar problemas com fork em bibliotecas CUDA/PyTorch
try:
    if get_start_method(allow_none=True) != 'spawn':
        set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Já foi definido

import util  # garante que a pasta src está no sys.path
from util import UtilEnv
if UtilEnv.carregar_env('.env', pastas=['../','./']):
   pass
   
_locais_ = [f'{_}_bertmodels/' for _ in ['./','../'] if os.path.isdir(f'{_}_bertmodels/')]
PASTA_LOCAL = _locais_[0] if len(_locais_)>0 else './_bertmodels/'
VERBOSE_BATCH_SIZE = 5

try:
    BERTSCORE_TIMEOUT = int(os.getenv('BERTSCORE_TIMEOUT', '600'))
except ValueError:
    BERTSCORE_TIMEOUT = 600

# Configura cache local se PASTA_LOCAL estiver definida
if PASTA_LOCAL:
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    # os.environ['TRANSFORMERS_CACHE'] = PASTA_LOCAL # deprecated - removido
    os.environ['HF_HOME'] = PASTA_LOCAL # atual

# ============================================================================
# CONFIGURAÇÃO GLOBAL DE WORKERS E DEVICE
# ============================================================================
# Helpers para variáveis de ambiente
def _get_env_int(key, default=None):
    try:
        val = os.getenv(key)
        return int(val) if val is not None else default
    except ValueError:
        return default

# Leitura das variáveis de ambiente
BERTSCORE_DEVICE = os.getenv('BERTSCORE_DEVICE', 'auto').strip()
_BERTSCORE_WORKERS_ENV = _get_env_int('BERTSCORE_WORKERS')
_BERTSCORE_MAX_WORKERS_ENV = _get_env_int('BERTSCORE_MAX_WORKERS')

# Lógica 'auto' para o device
if BERTSCORE_DEVICE.lower() == 'auto':
    try:
        from bert_score import score
        # Teste simples de predição para verificar GPU
        # Usa textos mínimos 'a' e 'a' como no exemplo de teste
        score(['a'], ['a'], lang="pt", verbose=False, device='cuda')
        BERTSCORE_DEVICE = 'cuda'
        print("🚀 [BERTScoreService] CUDA detectado e ativado (auto).")
    except Exception:
        BERTSCORE_DEVICE = 'cpu'
        print("⚠️ [BERTScoreService] CUDA não disponível ou erro no teste. Usando CPU (auto).")
elif BERTSCORE_DEVICE.lower() == 'gpu':
    BERTSCORE_DEVICE = 'cuda'
else:
    # Se não for auto/gpu, mantém o que veio (ex: cpu, cuda:0) ou fallback para cpu se vazio
    BERTSCORE_DEVICE = BERTSCORE_DEVICE or 'cpu'

# Variável global para pré-configurar número de workers
# Inicializa com valores do ENV se existirem
_BERTSCORE_WORKERS_CONFIG = _BERTSCORE_WORKERS_ENV
_BERTSCORE_MAX_WORKERS_CONFIG = _BERTSCORE_MAX_WORKERS_ENV
_BERTSCORE_DEVICE_CONFIG = BERTSCORE_DEVICE

def configurar_bertscore_workers(workers: int = None, max_workers: int = None, device: str = None):
    """
    Configura o número de workers do BERTScore ANTES da primeira inicialização.
    
    Esta função deve ser chamada ANTES de qualquer uso do bscore() ou BERTScoreService.
    Se o serviço já foi inicializado, a configuração não terá efeito.
    
    Args:
        workers: Número de workers desejado
                 - None: usa padrão automático (número de CPUs)
                 - int positivo: número exato de workers
                 - -1: CPUs - 1 (deixa uma CPU livre)
        max_workers: Limite máximo de workers automáticos
                     - None: sem limite
                     - int positivo: limita workers automáticos a este valor
        device: dispositivo para o BERTScore
    
    Exemplos:
        # Em testes unitários - limitar a 3 workers para economizar recursos
        from util_bertscore import configurar_bertscore_workers
        configurar_bertscore_workers(workers=3)
        # ... depois usar bscore() normalmente
        
        # Em produção - usar máximo de 10 workers mesmo com muitas CPUs
        from util_bertscore import configurar_bertscore_workers
        configurar_bertscore_workers(max_workers=10)
        
        # Em ambiente de desenvolvimento - deixar 1 CPU livre
        from util_bertscore import configurar_bertscore_workers
        configurar_bertscore_workers(workers=-1)
        
        # Em notebook interativo - configurar antes de importar outras classes
        import sys
        sys.path.append('./utils')
        from util_bertscore import configurar_bertscore_workers
        configurar_bertscore_workers(workers=2)
        from util_json import JsonAnalise  # Agora usará 2 workers
    
    Returns:
        bool: True se configuração foi aplicada, False se serviço já estava inicializado
    """
    global _BERTSCORE_WORKERS_CONFIG, _BERTSCORE_MAX_WORKERS_CONFIG, _BERTSCORE_DEVICE_CONFIG
    
    # Verifica se serviço já foi inicializado
    if BERTScoreService._initialized:
        print(("⚠️  [BERTScoreService] Serviço já inicializado. Configuração de workers ignorada."  
        + "\n"  + "Workers: " + str(_BERTSCORE_WORKERS_CONFIG)
        + "\n"  + "Max Workers: " + str(_BERTSCORE_MAX_WORKERS_CONFIG)
        + "\n"  + "Device: " + str(_BERTSCORE_DEVICE_CONFIG)))
        return False
    
    _BERTSCORE_WORKERS_CONFIG = workers
    _BERTSCORE_MAX_WORKERS_CONFIG = max_workers
    _BERTSCORE_DEVICE_CONFIG = device
    return True

# ============================================================================
# Serviço BERTScore global (singleton)
def bscore(preds: List[str] = None, trues: List[str] = None, 
           decimais: int = 3,
           verbose: bool = False,
           lang: str = 'pt', 
           workers: int = -1, 
           max_workers = None,
           usar_cache: bool = True,
           atualizar_cache: bool = True) -> Tuple[List[float], List[float], List[float]]:
    """
    Função wrapper para BERTScoreService com cache e multiprocessing.
    
    Esta versão usa BERTScoreService (multiprocessing) em vez da implementação
    simplificada do util_bertscore.py. Útil quando você quer controle fino sobre
    workers e paralelização.
    
    O serviço é inicializado automaticamente na primeira chamada e reutilizado
    em todas as chamadas subsequentes (padrão singleton).
    
    Args:
        preds: lista de textos preditos/gerados
        trues: lista de textos verdadeiros/esperados
        decimais: número de casas decimais para arredondamento
        verbose: se True, exibe progresso
        lang: idioma do modelo BERT (padrão: 'pt')
        workers: número de workers (usado apenas na primeira inicialização)
        max_workers: limite máximo de workers automáticos
        usar_cache: se False, não lê do cache
        atualizar_cache: se False, não salva no cache
    
    Returns:
        Tupla (P, R, F1) com listas de floats para Precision, Recall e F1-score
        Retorna None se chamado sem parâmetros (apenas para inicialização)
    
    Exemplo:
        # Calcula scores com multiprocessing
        P, R, F1 = bscore(['texto 1', 'texto 2'], ['referência 1', 'referência 2'])
    """
    global _BERTSCORE_WORKERS_CONFIG, _BERTSCORE_MAX_WORKERS_CONFIG

    # Chamada sem parâmetros: inicializa serviço explicitamente e retorna None
    if preds is None or trues is None:
        if _BERTSCORE_WORKERS_CONFIG is not None:
            workers = _BERTSCORE_WORKERS_CONFIG
        if _BERTSCORE_MAX_WORKERS_CONFIG is not None:
            max_workers = _BERTSCORE_MAX_WORKERS_CONFIG
        BERTScoreService.get_instance(workers=workers, max_workers=max_workers, lang=lang)
        return None
    
    # Validação básica
    if not isinstance(preds, (list, tuple)) or not isinstance(trues, (list, tuple)):
        raise TypeError("preds e trues devem ser listas ou tuplas de strings")
    
    if len(preds) != len(trues):
        raise ValueError(f"preds ({len(preds)}) e trues ({len(trues)}) devem ter o mesmo tamanho")
    
    # -------------------------------------------------------------------------
    # FASE 1: USO DO CACHE
    # -------------------------------------------------------------------------
    cache = BERTScoreCache(usar_cache=usar_cache, atualizar_cache=atualizar_cache)
    final_P, final_R, final_F1, missed_indices, missed_preds, missed_trues, missed_meta = cache.get_batch(preds, trues)

    # -------------------------------------------------------------------------
    # FASE 2: PROCESSAMENTO DOS ITENS NÃO ENCONTRADOS NO CACHE
    # -------------------------------------------------------------------------
    if missed_preds:
        # Configuração global (apenas se for inicializar agora)
        if _BERTSCORE_WORKERS_CONFIG is not None:
            workers = _BERTSCORE_WORKERS_CONFIG
        if _BERTSCORE_MAX_WORKERS_CONFIG is not None:
            max_workers = _BERTSCORE_MAX_WORKERS_CONFIG
        
        # Inicializa/Obtém Singleton
        service = BERTScoreService.get_instance(workers=workers, max_workers=max_workers, lang=lang)
        
        try:
            # Processa em lote usando multiprocessing
            mP, mR, mF1 = service.processar(missed_preds, missed_trues, verbose=verbose)
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


def _worker_process(input_queue, lang: str, worker_id: int):
    """
    Processo worker que fica em loop aguardando requisições.
    Cada worker processa uma requisição por vez e retorna o resultado.
    
    :param input_queue: Fila de entrada com requisições (hipoteses, referencias, result_queue, request_id).
    :param lang: Idioma para o modelo BERT.
    :param worker_id: ID do worker para debug.
    """
    while True:
        try:
            item = input_queue.get(timeout=None)
            
            if item is None:  # Sinal de encerramento
                break
            
            hipoteses, referencias, result_queue, request_id = item
            
            # Calcula o BERTScore
            P, R, F1 = score(hipoteses,
                             referencias,
                             lang=lang,
                             verbose=len(hipoteses)>=VERBOSE_BATCH_SIZE,
                             device=BERTSCORE_DEVICE)
            result = ([float(_) for _ in P],
                      [float(_) for _ in R],
                      [float(_) for _ in F1], None, request_id)
            
            result_queue.put(result)
            
        except Exception as e:
            # Em caso de erro, retorna a exceção
            try:
                result = (None, None, None, str(e), request_id)
                result_queue.put(result)
            except:
                pass


class BERTScoreService:
    """
    Serviço singleton persistente para cálculo de BERTScore.
    
    Esta classe implementa um padrão singleton que mantém um pool de workers
    ativo durante toda a execução do programa. Múltiplas threads podem enviar
    requisições simultaneamente e o serviço gerencia a distribuição de trabalho
    e retorno de resultados.
    
    Exemplo de uso:
        # Obter a instância do serviço (cria apenas uma vez)
        service = BERTScoreService.get_instance(workers=5, lang="pt")
        
        # Usar em qualquer thread
        P, R, F1 = service.processar(hipoteses, referencias)
        
        # O serviço é encerrado automaticamente ao finalizar o programa
    """
    
    _instance = None
    _condition = threading.Condition()
    _initialized = False
    
    def __new__(cls):
        """Implementa o padrão Singleton."""
        if cls._instance is None:
            with cls._condition:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._condition.notify_all()
        return cls._instance
    
    def __init__(self):
        """Inicialização do singleton (chamado apenas uma vez)."""
        # Evita reinicialização
        if BERTScoreService._initialized:
            return
        
        self._workers = 5
        self._lang = "pt"
        self._request_counter = 0
        self._request_condition = threading.Condition()
        self._closed = False
        
        # Pool de workers
        self.manager = None
        self.input_queue = None
        self.processes = []
        
        # Registra o encerramento automático
        atexit.register(self._cleanup)
        
        BERTScoreService._initialized = True
    
    @classmethod
    def get_instance(cls, workers= None, max_workers = None, lang: str = "pt") -> 'BERTScoreService':
        """
        Obtém a instância singleton do serviço.
        
        Nota: Os parâmetros workers e lang só são considerados na primeira chamada.
        Chamadas subsequentes retornam a instância existente com os parâmetros originais.
        
        :param workers: Número de processos workers (padrão: 5).
        :param lang: Idioma do modelo BERT (padrão: 'pt').
        :return: Instância do serviço.
        """
        instance = cls()

        if (not isinstance(workers, int)):
            # um processo por CPU
            workers = cpu_count()
        elif workers < 0:
            # deixa uma cpu livre
            workers = cpu_count() -1
        else:
            # quantos processos solicitar
            workers = workers
        if isinstance(max_workers, int) and max_workers >0 and workers > max_workers:
            workers = max_workers

        # Inicializa o pool na primeira vez ou se foi encerrado
        if instance.manager is None or instance._closed:
            with cls._condition:
                if instance.manager is None or instance._closed:
                    instance._workers = workers
                    instance._lang = lang
                    instance._inicializar_pool()
                    _linha = '-' * 30
                    print(f"{_linha}\n 🤖🏳️ [BERTScoreService] Serviço iniciado com {workers} workers (lang={lang}) device ({BERTSCORE_DEVICE})\n{_linha}")
                    cls._condition.notify_all()
        return instance
    
    def _inicializar_pool(self):
        """Inicializa o pool de workers."""
        self._closed = False
        
        # Usa Manager para criar filas que podem ser compartilhadas com spawn
        self.manager = Manager()
        self.input_queue = self.manager.Queue()
        self.processes = []
        
        # Inicia os processos workers
        for i in range(self._workers):
            p = Process(target=_worker_process, args=(self.input_queue, self._lang, i))
            p.daemon = True  # Garante que processos filhos sejam encerrados com o pai
            p.start()
            self.processes.append(p)
    
    def processar(self, hipoteses: List[str], referencias: List[str], 
                  timeout: float = None,
                  verbose = False) -> Tuple[List[float], List[float], List[float]]:
        """
        Processa uma requisição de cálculo de BERTScore.
        
        Esta função é thread-safe e pode ser chamada simultaneamente por múltiplas threads.
        Cada requisição recebe um ID único e é enfileirada para processamento.
        
        :param hipoteses: Lista de strings com as hipóteses.
        :param referencias: Lista de strings com as referências.
        :param timeout: Tempo máximo de espera em segundos (padrão: BERTSCORE_TIMEOUT env var ou 300).
        :return: Tupla com listas de (Precision, Recall, F1).
        :raises ValueError: Se hipóteses e referências não tiverem o mesmo tamanho.
        :raises RuntimeError: Se o serviço não foi inicializado ou houve erro.
        """
        if timeout is None:
            timeout = BERTSCORE_TIMEOUT
        if self._closed or self.manager is None:
            raise RuntimeError(
                "Serviço não inicializado ou foi encerrado. Use BERTScoreService.get_instance() primeiro."
            )
        
        if len(hipoteses) != len(referencias):
            raise ValueError("Hipóteses e referências devem ter o mesmo tamanho")

        # Verifica integridade da fila de entrada
        if self.input_queue is None:
            with self._request_condition:
                if self.input_queue is None:
                    print("⚠️ [BERTScoreService] input_queue inválida. Tentando reinicializar pool...")
                    try:
                        self._cleanup()
                    except Exception as e:
                        print(f"Erro no cleanup durante recuperação: {e}")
                    self._inicializar_pool()
                    self._request_condition.notify_all()
        
        # Incrementa contador thread-safe
        with self._request_condition:
            self._request_counter += 1
            request_num = self._request_counter
            self._request_condition.notify_all()
        
        thread_id = threading.current_thread().name
        if verbose:
            print(f"[BERTScoreService] Requisição #{request_num} de {thread_id} "
                f"({len(hipoteses)} pares)")
        
        start_time = time.time()
        
        try:
            # Cria uma fila única para receber o resultado desta requisição
            result_queue = self.manager.Queue()
            request_id = str(uuid.uuid4())
            
            # Envia a requisição para a fila de entrada
            self.input_queue.put((hipoteses, referencias, result_queue, request_id))
            
            # Aguarda o resultado com timeout
            try:
                P, R, F1, error, returned_id = result_queue.get(timeout=timeout)
            except:
                raise RuntimeError(f"Timeout aguardando resultado do BERTScore")
            
            if error:
                raise RuntimeError(f"Erro ao calcular BERTScore: {error}")
            
            elapsed = time.time() - start_time
            if verbose:
               print(f"[BERTScoreService] Requisição #{request_num} concluída em {elapsed:.2f}s")
            
            return P, R, F1
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f" 🚩 [BERTScoreService] Requisição #{request_num} falhou após {elapsed:.2f}s: {e}")
            raise
    
    def get_stats(self) -> dict:
        """
        Retorna estatísticas do serviço.
        
        :return: Dicionário com informações sobre o serviço.
        """
        return {
            'workers': self._workers,
            'lang': self._lang,
            'total_requests': self._request_counter,
            'active': self.manager is not None and not self._closed,
            'pool_initialized': self.manager is not None
        }
    
    def _cleanup(self):
        """Encerra o pool de workers (chamado automaticamente ao finalizar)."""
        if self.manager is not None and not self._closed:
            print("[BERTScoreService] Encerrando serviço...")
            
            # Envia sinal de encerramento para todos os workers
            if self.input_queue is not None:
                try:
                    for _ in range(self._workers):
                        self.input_queue.put(None)
                except Exception as e:
                    print(f"Erro ao enviar sinal de encerramento (put): {e}")

            # Aguarda todos os processos terminarem
            for p in self.processes:
                try:
                    p.join(timeout=5)
                    if p.is_alive():
                        p.terminate()  # Força o encerramento se necessário
                except Exception as e:
                    print(f"Erro ao encerrar processo worker: {e}")
            
            self._closed = True
            
            # Encerra o manager
            try:
                self.manager.shutdown()
            except Exception as e:
                print(f"Erro ao encerrar manager: {e}")
            
            print("[BERTScoreService] Serviço encerrado")
    
    def shutdown(self):
        """
        Encerra manualmente o serviço.
        
        Útil para testes ou quando se deseja reiniciar o serviço com parâmetros diferentes.
        """
        self._cleanup()
    
    @classmethod
    def reset(cls):
        """
        Reseta o singleton (útil para testes).
        
        AVISO: Use com cuidado. Isso força a criação de uma nova instância.
        """
        with cls._condition:
            if cls._instance is not None and cls._instance.manager is not None:
                cls._instance._cleanup()
            cls._instance = None
            cls._initialized = False
            cls._condition.notify_all()


# ============================================================================
# Testes
# ============================================================================

def teste_basico():
    """Teste básico de funcionalidade."""
    print("=" * 80)
    print("TESTE BÁSICO DE FUNCIONALIDADE")
    print("=" * 80)
    
    pares = [
        ("O gato está no telhado", "O felino está em cima da casa"),
        ("Hoje está ensolarado", "O tempo está bom"),
        ("Ele comprou um carro novo", "Ele adquiriu um veículo recente"),
        ("Vamos almoçar fora?", "Você quer comer em um restaurante?"),
        ("O avião decolou", "O pássaro voou"),
        ("Python é uma linguagem de programação.", "Meu hobby favorito é pedalar aos finais de semana"),
    ]
    
    hipoteses = [par[0] for par in pares]
    referencias = [par[1] for par in pares]
    
    service = BERTScoreService.get_instance(workers=-1, lang="pt")
    P, R, F1 = service.processar(hipoteses, referencias)
    
    print("\nResultados:")
    for i, (h, r) in enumerate(pares):
        print(f"\nPar {i+1}:")
        print(f"  Hipótese:   {h}")
        print(f"  Referência: {r}")
        print(f"  Precision:  {P[i]:.4f}")
        print(f"  Recall:     {R[i]:.4f}")
        print(f"  F1:         {F1[i]:.4f}")


def teste_casos_extremos():
    """Testa casos extremos (strings vazias, iguais, muito diferentes)."""
    print("\n" + "=" * 80)
    print("TESTE DE CASOS EXTREMOS")
    print("=" * 80)
    
    service = BERTScoreService.get_instance()
    
    casos = [
        ("", "", "Strings vazias"),
        ("Teste", "Teste", "Strings idênticas"),
        ("a" * 1000, "b" * 1000, "Strings longas diferentes"),
        ("Olá", "Tchau", "Palavras completamente diferentes"),
    ]
    
    for hip, ref, descricao in casos:
        P, R, F1 = service.processar([hip], [ref])
        print(f"\n{descricao}:")
        print(f"  F1: {F1[0]:.4f}")


def teste_consistencia():
    """Verifica se os resultados são consistentes entre execuções."""
    import numpy as np
    
    print("\n" + "=" * 80)
    print("TESTE DE CONSISTÊNCIA")
    print("=" * 80)
    
    service = BERTScoreService.get_instance()
    
    hipoteses = ["O gato dorme no sofá", "Python é uma linguagem ótima"]
    referencias = ["O felino descansa no sofá", "Python é uma excelente linguagem"]
    
    resultados = []
    for i in range(3):
        P, R, F1 = service.processar(hipoteses, referencias)
        resultados.append(F1)
    
    # Verifica se os resultados são idênticos
    print("\nResultados de 3 execuções:")
    for i, F1 in enumerate(resultados, 1):
        print(f"  Execução {i}: {F1}")
    
    if all(np.allclose(resultados[0], r, rtol=1e-5) for r in resultados[1:]):
        print("\n✓ Resultados consistentes entre execuções")
    else:
        print("\n✗ AVISO: Resultados inconsistentes!")


def teste_multithreading():
    """Testa o serviço com múltiplas threads simultâneas."""
    print("\n" + "=" * 80)
    print("TESTE DE MULTITHREADING")
    print("=" * 80)
    
    service = BERTScoreService.get_instance(workers=-1, lang="pt")
    
    n_threads = 5
    resultados = {}
    erros = []
    results_condition = threading.Condition()
    
    def worker_thread(thread_id):
        """Função executada por cada thread."""
        try:
            hipoteses = [f"Esta é a frase de teste {thread_id} parte {i}" for i in range(3)]
            referencias = [f"Esta é a referência {thread_id} parte {i}" for i in range(3)]
            
            P, R, F1 = service.processar(hipoteses, referencias)
            
            with results_condition:
                resultados[thread_id] = {'F1': F1, 'success': True}
                results_condition.notify_all()
                
        except Exception as e:
            with results_condition:
                erros.append((thread_id, str(e)))
                results_condition.notify_all()
    
    print(f"\nIniciando {n_threads} threads simultâneas...")
    threads = []
    start = time.time()
    
    for i in range(n_threads):
        t = threading.Thread(target=worker_thread, args=(i,), name=f"Worker-{i}")
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    elapsed = time.time() - start
    
    print(f"\nTempo total: {elapsed:.2f}s")
    print(f"Requisições processadas: {len(resultados)}")
    
    if erros:
        print(f"\n✗ {len(erros)} erros encontrados")
    else:
        print("\n✓ Todas as threads completaram com sucesso")
    
    # Mostra estatísticas finais
    stats = service.get_stats()
    print(f"\nEstatísticas finais:")
    print(f"  Total de requisições: {stats['total_requests']}")


def teste_carga_pesada():
    """Testa o serviço com carga pesada."""
    print("\n" + "=" * 80)
    print("TESTE DE CARGA PESADA")
    print("=" * 80)
    
    service = BERTScoreService.get_instance(workers=0, lang="pt")
    
    n_threads = 20
    requests_per_thread = 2
    total_requests = n_threads * requests_per_thread
    
    completed = []
    erros = []
    results_condition = threading.Condition()
    
    def worker_thread(thread_id):
        """Envia múltiplas requisições."""
        for req_id in range(requests_per_thread):
            try:
                hipoteses = [f"Thread {thread_id} requisição {req_id} texto {j}" for j in range(2)]
                referencias = [f"Thread {thread_id} referência {req_id} texto {j}" for j in range(2)]
                
                P, R, F1 = service.processar(hipoteses, referencias)
                
                with results_condition:
                    completed.append((thread_id, req_id))
                    results_condition.notify_all()
                    
            except Exception as e:
                with results_condition:
                    erros.append((thread_id, req_id, str(e)))
                    results_condition.notify_all()
    
    print(f"\nIniciando {n_threads} threads com {requests_per_thread} requisições cada...")
    print(f"Total de requisições: {total_requests}")
    
    threads = []
    start = time.time()
    
    for i in range(n_threads):
        t = threading.Thread(target=worker_thread, args=(i,), name=f"Heavy-{i}")
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    elapsed = time.time() - start
    
    print(f"\nTempo total: {elapsed:.2f}s")
    print(f"Requisições concluídas: {len(completed)}/{total_requests}")
    print(f"Taxa: {len(completed)/elapsed:.2f} requisições/segundo")
    
    if erros:
        print(f"\n✗ {len(erros)} erros encontrados")
        for thread_id, req_id, erro in erros[:3]:
            print(f"  Thread {thread_id}, Req {req_id}: {erro}")
    else:
        print("\n✓ Todas as requisições completaram com sucesso")
    
    stats = service.get_stats()
    print(f"\nEstatísticas finais: {stats['total_requests']} requisições processadas")


def teste_singleton():
    """Testa o comportamento singleton."""
    print("\n" + "=" * 80)
    print("TESTE DO PADRÃO SINGLETON")
    print("=" * 80)
    
    print("\n1. Inicializando serviço...")
    service1 = BERTScoreService.get_instance(workers=-1, lang="pt")
    stats1 = service1.get_stats()
    print(f"   Serviço ativo: {stats1['active']}, Workers: {stats1['workers']}")
    
    print("\n2. Obtendo instância novamente (deve ser a mesma)...")
    service2 = BERTScoreService.get_instance(workers=10, lang="en")  # Parâmetros ignorados
    stats2 = service2.get_stats()
    print(f"   Mesma instância: {service1 is service2}")
    print(f"   Workers mantidos: {stats2['workers']} (configurado inicialmente: {stats1['workers']})")
    
    print("\n3. Testando processamento...")
    hipoteses = ["O gato está no telhado", "Python é ótimo"]
    referencias = ["O felino está na casa", "Python é excelente"]
    
    P, R, F1 = service1.processar(hipoteses, referencias)
    print(f"   F1 scores: {[f'{f:.4f}' for f in F1]}")
    
    print("\n4. Estatísticas do serviço:")
    stats = service1.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    # se o parametro for full, roda todos os testes, caso contrário roda dois testes com bscore()
    import sys
    if len(sys.argv) == 1 or sys.argv[1] != "full":
        print("\n" + "=" * 80)
        print("TESTE RÁPIDO DA FUNÇÃO bscore()")
        print("=" * 80)
        
        # abre 10 threads para o teste com 3 workers de bertscore
        n_teste = 10
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_teste) as executor:
            futures = []
            for i in range(n_teste):
                futures.append(executor.submit(bscore,
                                               ["O gato está no telhado", "Hoje está ensolarado"],
                                               ["O felino está em cima da casa", "O tempo está bom"],
                                               decimais=3, workers=3))
            for i, future in enumerate(futures):
                try:
                    P, R, F1 = future.result()
                    print(f"\nThread {i+1} resultados:")
                    for j in range(len(P)):
                        print(f"  Par {j+1}: Precision={P[j]:.4f}, Recall={R[j]:.4f}, F1={F1[j]:.4f}")
                except Exception as e:
                    print(f"\n✗ Erro na Thread {i+1}: {e}")
        
        sys.exit(0)
    
    print("\n" + "=" * 80)
    print("BATERIA DE TESTES - BERTScoreService (Singleton)")
    print("=" * 80)
    
    try:
        teste_basico()
        BERTScoreService.reset()
        
        teste_casos_extremos()
        BERTScoreService.reset()
        
        teste_consistencia()
        BERTScoreService.reset()
        
        teste_singleton()
        BERTScoreService.reset()
        
        teste_multithreading()
        BERTScoreService.reset()
        
        teste_carga_pesada()
        
        print("\n" + "=" * 80)
        print("TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ ERRO durante os testes: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Garante limpeza
        BERTScoreService.reset()