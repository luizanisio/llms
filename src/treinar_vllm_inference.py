#!/usr/bin/env python3
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Módulo de inferência rápida usando vLLM.

vLLM acelera a inferência em até 24x comparado ao HuggingFace Transformers padrão,
usando técnicas como:
- PagedAttention para gerenciamento eficiente de KV cache
- Continuous batching para maximizar GPU utilization
- Otimizações de kernel CUDA customizadas

Uso como módulo:
    from treinar_vllm_inference import VLLMInferenceEngine

    engine = VLLMInferenceEngine(
        model_path="./modelos/meu_modelo",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,  # 2 GPUs
    )

    results = engine.generate_batch(
        prompts=["prompt1", "prompt2", ...],
        max_tokens=512,
        temperature=0.7,
    )

Requisitos:
    pip install vllm
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    LoRARequest = None


@dataclass
class VLLMConfig:
    """Configuração do motor vLLM."""
    gpu_memory_utilization: float = 0.9  # Usar 90% da VRAM
    tensor_parallel_size: int = 1  # Número de GPUs (paralelismo de tensor)
    dtype: str = "auto"  # "auto", "float16", "bfloat16"
    trust_remote_code: bool = True
    max_model_len: Optional[int] = None  # None = auto-detecta
    enforce_eager: bool = False  # True desabilita CUDA graphs (mais lento, menos memória)


class VLLMInferenceEngine:
    """Motor de inferência usando vLLM para geração de alta performance.

    Funcionalidades:
    - Inferência em batch (múltiplas requisições simultâneas)
    - Paralelismo de tensor (múltiplas GPUs)
    - PagedAttention (gerenciamento eficiente de memória)
    - Até 24x mais rápido que HF Transformers
    - Auto-detecção de memória GPU disponível para evitar OOM
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[VLLMConfig] = None,
        lora_path: Optional[str] = None,
    ):
        """Inicializa motor vLLM.

        Args:
            model_path: Caminho para modelo (HF format ou local)
            config: Configuração do vLLM (usa padrão se None)
            lora_path: Caminho para adaptador LoRA (opcional).
                       Se informado, model_path deve apontar para o modelo BASE
                       e o adapter será aplicado via LoRARequest na geração.
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM não está instalado. Instale com: pip install vllm"
            )

        self.model_path = model_path
        self.config = config or VLLMConfig()
        self.lora_path = lora_path
        self._lora_request = None

        # Auto-detecta gpu_memory_utilization seguro baseado na VRAM livre real.
        # Substitui o valor padrão (0.9) ou qualquer valor configurado que
        # exceda a memória realmente disponível. Isso evita o erro:
        #   "Free memory on device cuda:X (...) is less than desired GPU memory
        #    utilization (...)"
        gpu_util_original = self.config.gpu_memory_utilization
        gpu_util_seguro = self._calcular_gpu_memory_utilization()
        if gpu_util_seguro < gpu_util_original:
            print(f"⚠️  gpu_memory_utilization ajustado: {gpu_util_original:.0%} → {gpu_util_seguro:.0%} "
                  f"(memória livre insuficiente para {gpu_util_original:.0%})")
            self.config.gpu_memory_utilization = gpu_util_seguro

        print(f"🚀 Inicializando vLLM com modelo: {model_path}")
        if lora_path:
            print(f"   LoRA Adapter: {lora_path}")
        print(f"   GPU Memory: {self.config.gpu_memory_utilization*100:.0f}%")
        print(f"   Tensor Parallel: {self.config.tensor_parallel_size} GPU(s)")

        # Inicializa vLLM
        # Permite contextos acima de max_position_embeddings (RoPE scaling)
        # em ambiente controlado de experimento com valores conhecidos
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        try:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                dtype=self.config.dtype,
                trust_remote_code=self.config.trust_remote_code,
                max_model_len=self.config.max_model_len,
                enforce_eager=self.config.enforce_eager,
                enable_lora=bool(lora_path),
            )
            # Cria LoRARequest se adapter informado
            if lora_path:
                self._lora_request = LoRARequest(
                    "lora_adapter",  # nome identificador
                    1,               # ID único do adapter
                    lora_path,       # caminho do adapter
                )
                print(f"✅ vLLM inicializado com LoRA adapter!\n")
            else:
                print("✅ vLLM inicializado com sucesso!\n")
        except Exception as e:
            print(f"❌ Erro ao inicializar vLLM: {e}")
            raise

    # ------------------------------------------------------------------
    # Auto-detecção de memória GPU
    # ------------------------------------------------------------------

    @staticmethod
    def _calcular_gpu_memory_utilization() -> float:
        """Calcula gpu_memory_utilization seguro baseado na memória livre real.

        O vLLM verifica, **dentro do subprocess EngineCore**, que:
            ``gpu_memory_utilization × total_memory  ≤  free_memory``

        Entre a nossa medição (processo pai) e a verificação do EngineCore
        (subprocess), a memória livre cai devido a:
          • Contexto CUDA do subprocess (~1-2 GB)
          • Inicialização NCCL (~0.5-1 GB)
          • Re-aquisição de cache PyTorch pelo processo pai

        IMPORTANTE: NÃO fazemos gc.collect()/empty_cache() antes de medir.
        Isso inflaria artificialmente o valor de free memory (o cache
        PyTorch é devolvido ao CUDA driver, aparece como "free" para
        mem_get_info, mas é re-adquirido antes do subprocess verificar).

        Fórmula:
            ratio = free / total          # fração realmente livre
            utilization = ratio - 0.10    # 10pp de margem p/ subprocess

        Exemplo: GPU 80 GB, 54 GB livres
          → ratio = 54/80 = 0.675
          → utilization = 0.675 − 0.10 = 0.575 → clamp → 0.58
          → vLLM pedirá 0.58 × 80 = 46.4 GB  (< 54 GB ✓)

        Returns:
            Fração segura (entre 0.10 e 0.85) para gpu_memory_utilization.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return 0.50  # fallback sem GPU

            device = torch.cuda.current_device()

            # Mede free memory SEM esvaziar cache — leitura realista
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            except AttributeError:
                # torch < 1.10
                props = torch.cuda.get_device_properties(device)
                total_bytes = props.total_mem
                free_bytes = total_bytes - torch.cuda.memory_reserved(device)

            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)

            # ratio = fração da VRAM total que está realmente livre
            ratio = free_gb / total_gb

            # Subtrai 10 pontos percentuais como margem fixa para o
            # overhead do subprocess EngineCore (contexto CUDA, NCCL, etc.)
            MARGEM_PP = 0.10
            utilization = ratio - MARGEM_PP

            if utilization <= 0:
                print(f"⚠️  GPU {device}: apenas {free_gb:.1f} GB livres "
                      f"(total: {total_gb:.1f} GB). Memória muito baixa!")
                return 0.10

            utilization = max(0.10, min(0.85, utilization))  # clamp

            print(f"🔧 vLLM gpu_memory_utilization auto-detectado: {utilization:.2f} "
                  f"(livre: {free_gb:.1f}/{total_gb:.1f} GB, "
                  f"ratio: {ratio:.2f}, margem: {MARGEM_PP:.0%})")
            return round(utilization, 2)

        except Exception as e:
            print(f"⚠️  Erro ao detectar memória GPU: {e}. Usando fallback 0.50")
            return 0.50

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
        n: int = 1,  # Número de completions por prompt
    ) -> List[Dict[str, Any]]:
        """Gera completions para múltiplos prompts em batch.

        Args:
            prompts: Lista de prompts
            max_tokens: Máximo de tokens por completion
            temperature: Temperatura de sampling (0.0 = determinístico, 1.0 = criativo)
            top_p: Nucleus sampling
            top_k: Top-k sampling
            repetition_penalty: Penalidade para repetição
            stop: Lista de strings de parada (opcional)
            n: Número de completions por prompt

        Returns:
            Lista de dicts com resultados:
            [
                {
                    "prompt": str,
                    "output": str,
                    "tokens": int,
                    "finish_reason": str
                },
                ...
            ]
        """
        if not prompts:
            return []

        # Configuração de sampling
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop,
            n=n,
        )

        print(f"🔄 Gerando {len(prompts)} completion(s)...")

        # Gera em batch (vLLM otimiza automaticamente)
        try:
            gen_kwargs = {}
            if self._lora_request:
                gen_kwargs["lora_request"] = self._lora_request
            outputs = self.llm.generate(prompts, sampling_params, **gen_kwargs)
        except Exception as e:
            print(f"❌ Erro durante geração: {e}")
            raise

        # Processa resultados
        results = []
        for output in outputs:
            for completion in output.outputs:
                results.append({
                    "prompt": output.prompt,
                    "output": completion.text,
                    "tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                })

        print(f"✅ {len(results)} completion(s) gerada(s)!\n")
        return results

    def generate_single(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Gera um único completion (wrapper conveniente).

        Args:
            prompt: Prompt de entrada
            max_tokens: Máximo de tokens
            temperature: Temperatura de sampling
            **kwargs: Argumentos adicionais para generate_batch

        Returns:
            String com o completion gerado
        """
        results = self.generate_batch(
            prompts=[prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            **kwargs
        )

        return results[0]["output"] if results else ""

    def benchmark(
        self,
        num_prompts: int = 100,
        prompt_length: int = 128,
        max_tokens: int = 256,
    ) -> Dict[str, float]:
        """Executa benchmark de performance.

        Args:
            num_prompts: Número de prompts a processar
            prompt_length: Tamanho médio dos prompts (em tokens)
            max_tokens: Máximo de tokens de saída

        Returns:
            Dict com métricas de performance
        """
        import time

        # Gera prompts dummy
        dummy_prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 10)
        prompts = [dummy_prompt] * num_prompts

        print(f"📊 Executando benchmark:")
        print(f"   Prompts: {num_prompts}")
        print(f"   Prompt length: ~{prompt_length} tokens")
        print(f"   Max tokens: {max_tokens}\n")

        start = time.time()
        results = self.generate_batch(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=0.0,  # Determinístico para benchmark
        )
        elapsed = time.time() - start

        # Calcula métricas
        total_tokens = sum(r["tokens"] for r in results)
        throughput = total_tokens / elapsed
        latency = elapsed / num_prompts

        metrics = {
            "total_time": elapsed,
            "total_tokens": total_tokens,
            "throughput_tokens_per_sec": throughput,
            "avg_latency_per_prompt": latency,
        }

        print(f"📈 Resultados:")
        print(f"   Tempo total: {elapsed:.2f}s")
        print(f"   Tokens gerados: {total_tokens:,}")
        print(f"   Throughput: {throughput:.0f} tokens/s")
        print(f"   Latência média: {latency*1000:.0f}ms/prompt\n")

        return metrics


# ---------------------------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------------------------

def check_vllm_available() -> bool:
    """Verifica se vLLM está instalado e disponível."""
    return VLLM_AVAILABLE


def get_recommended_config(num_gpus: int = 1, model_size: str = "7B") -> VLLMConfig:
    """Retorna configuração recomendada baseada no hardware.

    A gpu_memory_utilization é calculada automaticamente a partir da
    memória livre real da GPU via ``_calcular_gpu_memory_utilization``.
    Isso evita OOM quando há outros processos consumindo VRAM.

    Args:
        num_gpus: Número de GPUs disponíveis
        model_size: Tamanho do modelo ("7B", "13B", "70B")

    Returns:
        VLLMConfig otimizado
    """
    # Auto-detecta gpu_memory_utilization — valor final será refinado no
    # __init__ do VLLMInferenceEngine, mas já partimos de um valor seguro.
    gpu_util = VLLMInferenceEngine._calcular_gpu_memory_utilization()

    # 7B: sempre TP=1 — cabe em qualquer GPU moderna (≥16 GB).
    # TP>1 para modelos pequenos adiciona overhead de multiproc (NCCL,
    # shared memory) sem ganho real, e causa RuntimeError: cancelled
    # no multiproc_executor ao tentar alocar KV cache entre GPUs.
    # Para TP>1, enforce_eager=True desabilita CUDA graphs e evita
    # picos de memória durante a captura dos grafos.
    if model_size == "7B":
        return VLLMConfig(
            gpu_memory_utilization=gpu_util,
            tensor_parallel_size=1,
            enforce_eager=True,
        )

    configs = {
        "13B": {
            1: VLLMConfig(gpu_memory_utilization=gpu_util, tensor_parallel_size=1),
            2: VLLMConfig(gpu_memory_utilization=gpu_util, tensor_parallel_size=2, enforce_eager=True),
            4: VLLMConfig(gpu_memory_utilization=gpu_util, tensor_parallel_size=4, enforce_eager=True),
        },
        "70B": {
            2: VLLMConfig(gpu_memory_utilization=gpu_util, tensor_parallel_size=2, enforce_eager=True),
            4: VLLMConfig(gpu_memory_utilization=gpu_util, tensor_parallel_size=4, enforce_eager=True),
            8: VLLMConfig(gpu_memory_utilization=gpu_util, tensor_parallel_size=8, enforce_eager=True),
        },
    }

    config = configs.get(model_size, configs["13B"])
    return config.get(num_gpus, config[min(config.keys())])


# ---------------------------------------------------------------------------
# Exemplo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Teste de inferência vLLM")
    parser.add_argument("model_path", help="Caminho para o modelo")
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms:",
                        help="Prompt de teste")
    parser.add_argument("--gpus", type=int, default=1, help="Número de GPUs")
    parser.add_argument("--benchmark", action="store_true", help="Executar benchmark")

    args = parser.parse_args()

    if not VLLM_AVAILABLE:
        print("❌ vLLM não está instalado!")
        print("   Instale com: pip install vllm")
        sys.exit(1)

    # Configuração recomendada
    config = get_recommended_config(num_gpus=args.gpus)

    # Inicializa engine
    engine = VLLMInferenceEngine(
        model_path=args.model_path,
        config=config,
    )

    if args.benchmark:
        # Benchmark
        engine.benchmark(num_prompts=100, max_tokens=256)
    else:
        # Teste simples
        print(f"📝 Prompt: {args.prompt}\n")
        output = engine.generate_single(
            prompt=args.prompt,
            max_tokens=512,
            temperature=0.7,
        )
        print(f"🤖 Resposta:\n{output}\n")
