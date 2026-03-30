#!/usr/bin/env python3
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Módulo para carregamento de modelos usando HuggingFace Transformers + PEFT.
Substitui a dependência do Unsloth, priorizando simplicidade e compatibilidade
com múltiplas GPUs (DDP, DeepSpeed, FSDP).

Classes:
    - ModelLoader: Carregamento de modelos base e LoRA
    - QuantizationConfig: Configuração de quantização (4-bit, 8-bit)
"""

import os
import torch
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from treinar_unsloth_logging import get_logger
from util_print import print_cores

logger = get_logger(__name__)

# --- Disponibilidade de componentes opcionais de otimização de memória ---
try:
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    _LIGER_DISPONIVEL = True
except ImportError:
    _LIGER_DISPONIVEL = False

try:
    from transformers.utils import is_flash_attn_2_available
    _FLASH_ATTN_DISPONIVEL = is_flash_attn_2_available()
except Exception:
    _FLASH_ATTN_DISPONIVEL = False


# ---------------------------------------------------------------------------
# Configuração de Quantização
# ---------------------------------------------------------------------------

@dataclass
class QuantizationConfig:
    """Configuração de quantização para redução de memória.

    Suporta:
    - 4-bit (NF4 ou FP4) com double quantization
    - 8-bit (int8)
    - Nenhuma (float16/bfloat16)
    """
    nbits: int = 16  # 4, 8 ou 16 (sem quantização)
    compute_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    quant_type: str = "nf4"  # "nf4" ou "fp4" (apenas para 4-bit)
    use_double_quant: bool = True  # Double quantization (economiza ~0.4GB por modelo)

    def to_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        """Converte para BitsAndBytesConfig do transformers."""
        if self.nbits == 16:
            return None

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = dtype_map.get(self.compute_dtype, torch.bfloat16)

        if self.nbits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.use_double_quant,
            )
        elif self.nbits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:
            raise ValueError(f"nbits deve ser 4, 8 ou 16, recebido: {self.nbits}")


# ---------------------------------------------------------------------------
# Carregador de Modelos
# ---------------------------------------------------------------------------

class ModelLoader:
    """Carrega modelos base e aplica adaptadores LoRA usando HF Transformers + PEFT.

    Funcionalidades:
    - Carregamento de modelos base (AutoModelForCausalLM)
    - Quantização 4-bit/8-bit via BitsAndBytes
    - Aplicação de adaptadores LoRA (get_peft_model)
    - Carregamento de modelos LoRA já treinados (PeftModel.from_pretrained)
    - Suporte a device_map="auto" para múltiplas GPUs
    - Merge de adaptadores LoRA para full fine-tuning
    """

    @staticmethod
    def load_base_model(
        model_name: str,
        max_seq_length: int = 4096,
        quant_config: Optional[QuantizationConfig] = None,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        attn_implementation: str = "flash_attention_2",
        use_cache: bool = False,
        use_liger_kernel: bool = False,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Carrega modelo base usando AutoModelForCausalLM ou AutoLigerKernelForCausalLM.

        Args:
            model_name: Nome ou caminho do modelo (HF Hub ou local)
            max_seq_length: Comprimento máximo de sequência
            quant_config: Configuração de quantização (None = sem quantização)
            device_map: Estratégia de distribuição ("auto", "cuda:0", etc.)
            trust_remote_code: Confiar em código remoto (necessário para alguns modelos)
            attn_implementation: Implementação de atenção ("flash_attention_2", "sdpa", "eager")
            use_cache: Ativar cache KV (desabilitar durante treinamento)
            use_liger_kernel: Usar Liger Kernel (fused cross-entropy, RoPE, RMSNorm).
                Reduz pico de VRAM ~40% ao evitar materializar tensor de logits completo.

        Returns:
            Tupla (model, tokenizer)
        """
        print_cores(f"<azul>🔄 Carregando modelo base: {model_name}...</azul>", color_auto=False)

        # Configuração de quantização
        bnb_config = quant_config.to_bnb_config() if quant_config else None

        # Dtype padrão (se não quantizado)
        torch_dtype = torch.bfloat16 if quant_config is None else None

        # IMPORTANTE: NÃO passamos max_position_embeddings ao from_pretrained().
        # Esse parâmetro altera a arquitetura RoPE do modelo, sobrescrevendo o valor
        # nativo (ex: Qwen2.5 = 32768 com rope_theta=1e6 que suporta até 131072).
        # O max_seq_length de treinamento é aplicado apenas na truncagem de dados
        # (via SFTConfig.max_length), não na arquitetura do modelo.

        # Determina a classe de carregamento: Liger Kernel ou AutoModelForCausalLM
        if use_liger_kernel and _LIGER_DISPONIVEL:
            _model_cls = AutoLigerKernelForCausalLM
            print_cores(
                "<verde>🔥 Liger Kernel ativo: fused cross-entropy, RoPE, RMSNorm "
                "(economia ~40% VRAM pico)</verde>", color_auto=False
            )
            # Liger Kernel fused loss requer que hidden_states e lm_head.weight
            # estejam no MESMO device. Com device_map="auto" e múltiplas GPUs,
            # o accelerate distribui camadas entre GPUs (model parallelism),
            # causando RuntimeError: "Expected all tensors to be on the same device".
            # Solução: forçar modelo em uma única GPU quando Liger está ativo.
            if device_map == "auto" and torch.cuda.device_count() > 1:
                n_gpus = torch.cuda.device_count()
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "não definido")
                raise RuntimeError(
                    f"\n{'='*72}\n"
                    f"❌ ERRO: Liger Kernel + múltiplas GPUs ({n_gpus}) detectadas.\n\n"
                    f"O Liger Kernel utiliza fused cross-entropy loss, que exige que\n"
                    f"hidden_states e lm_head.weight estejam no MESMO device.\n"
                    f"Com device_map='auto' e {n_gpus} GPUs visíveis "
                    f"(CUDA_VISIBLE_DEVICES={visible}), o modelo seria\n"
                    f"distribuído entre GPUs, causando erro de device mismatch.\n\n"
                    f"SOLUÇÕES (escolha uma):\n\n"
                    f"  1. Restringir a UMA GPU antes de executar:\n"
                    f"     export CUDA_VISIBLE_DEVICES=0\n"
                    f"     python treinar_unsloth.py config.yaml\n\n"
                    f"  2. Usar DDP com torchrun (cada processo usa 1 GPU):\n"
                    f"     torchrun --nproc_per_node={n_gpus} treinar_unsloth.py config.yaml\n\n"
                    f"  3. Desativar Liger Kernel no YAML (permite model parallelism):\n"
                    f"     treinamento:\n"
                    f"       liger_kernel: false\n"
                    f"{'='*72}"
                )
        else:
            _model_cls = AutoModelForCausalLM
            if use_liger_kernel and not _LIGER_DISPONIVEL:
                # Não deveria chegar aqui (validação é feita antes), mas por segurança
                logger.warning("Liger Kernel solicitado mas não disponível. Usando AutoModelForCausalLM.")

        # Tenta usar flash_attention_2 se disponível, senão fallback para sdpa
        try:
            model = _model_cls.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
                use_cache=use_cache,
            )
        except Exception as e:
            logger.warning(f"Falha ao carregar com attn_implementation={attn_implementation}: {e}")
            logger.info("Tentando com attn_implementation='sdpa' (fallback)...")
            model = _model_cls.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation="sdpa",
                use_cache=use_cache,
            )

        # Tokenizer — NÃO sobrescreve model_max_length.
        # O tokenizer nativo já tem o valor correto (ex: Qwen2.5 = 131072).
        # Forçar model_max_length=max_seq_length causava mensagens falsas de
        # "janela de contexto menor que o contexto dos dados".
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        # Configura pad_token se não existir
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Informações nativas do modelo vs configuração de treinamento
        native_max_pos = getattr(model.config, 'max_position_embeddings', 'N/A')
        native_rope_theta = getattr(model.config, 'rope_theta', 'N/A')
        tokenizer_max = getattr(tokenizer, 'model_max_length', 'N/A')

        print_cores(f"<verde>✅ Modelo base carregado: {type(model).__name__}</verde>", color_auto=False)
        print(f"  - Parâmetros totais: {model.num_parameters():,}")
        print(f"  - Quantização: {quant_config.nbits if quant_config else 16}-bit")
        print(f"  - Device map: {device_map}")
        print(f"  - Atenção: {getattr(model.config, '_attn_implementation', attn_implementation)}")
        print(f"  - Liger Kernel: {'✅ ativo' if (use_liger_kernel and _LIGER_DISPONIVEL) else '❌ desativado'}")
        print(f"  - Contexto nativo: max_position_embeddings={native_max_pos}, rope_theta={native_rope_theta}")
        print(f"  - Tokenizer max length (nativo): {tokenizer_max}")
        print(f"  - Max seq length (treinamento): {max_seq_length}")

        # Alerta se max_seq_length excede o contexto nativo do modelo
        if isinstance(native_max_pos, int) and max_seq_length > native_max_pos:
            logger.warning(
                f"⚠️  max_seq_length ({max_seq_length}) > max_position_embeddings nativo ({native_max_pos}). "
                f"O modelo pode lidar com sequências mais longas via RoPE (rope_theta={native_rope_theta}), "
                f"mas a qualidade pode degradar além do comprimento pré-treinado."
            )

        return model, tokenizer

    @staticmethod
    def apply_lora(
        model: PreTrainedModel,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
        bias: str = "none",
        task_type: TaskType = TaskType.CAUSAL_LM,
    ) -> PeftModel:
        """Aplica adaptadores LoRA a um modelo base.

        Args:
            model: Modelo base (AutoModelForCausalLM)
            r: Rank dos adaptadores LoRA
            lora_alpha: Parâmetro alpha (scaling factor)
            lora_dropout: Dropout nos adaptadores
            target_modules: Módulos alvo (None = auto-detecta q_proj, v_proj, etc.)
            bias: Como tratar bias ("none", "all", "lora_only")
            task_type: Tipo de tarefa (CAUSAL_LM para geração de texto)

        Returns:
            Modelo com adaptadores LoRA aplicados
        """
        print_cores(f"<azul>🔄 Aplicando adaptadores LoRA (r={r}, alpha={lora_alpha})...</azul>", color_auto=False)

        # Prepara modelo para treinamento quantizado (se aplicável)
        model = prepare_model_for_kbit_training(model)

        # Auto-detecta target_modules se não especificado
        if target_modules is None:
            # Módulos comuns para attention (funciona para a maioria dos modelos)
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        # Configuração LoRA
        peft_config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            inference_mode=False,
        )

        # Aplica LoRA
        model = get_peft_model(model, peft_config)

        # Imprime parâmetros treináveis
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / total_params

        print_cores(f"<verde>✅ LoRA aplicado com sucesso!</verde>", color_auto=False)
        print(f"  - Parâmetros treináveis: {trainable_params:,} ({trainable_percent:.2f}%)")
        print(f"  - Parâmetros totais: {total_params:,}")
        print(f"  - Target modules: {target_modules}")

        return model

    @staticmethod
    def load_lora_model(
        base_model_name: str,
        lora_model_path: str,
        max_seq_length: int = 4096,
        quant_config: Optional[QuantizationConfig] = None,
        device_map: str = "auto",
        attn_implementation: str = "flash_attention_2",
        use_liger_kernel: bool = False,
    ) -> Tuple[PeftModel, PreTrainedTokenizer]:
        """Carrega modelo LoRA já treinado (base + adaptadores).

        Args:
            base_model_name: Nome do modelo base (HF Hub)
            lora_model_path: Caminho para os adaptadores LoRA treinados
            max_seq_length: Comprimento máximo de sequência
            quant_config: Configuração de quantização
            device_map: Estratégia de distribuição
            attn_implementation: Implementação de atenção ("flash_attention_2", "sdpa", "eager")
            use_liger_kernel: Usar Liger Kernel para otimização de memória

        Returns:
            Tupla (model, tokenizer)
        """
        print_cores(f"<azul>🔄 Carregando modelo LoRA treinado de: {lora_model_path}...</azul>", color_auto=False)

        # Carrega modelo base
        base_model, tokenizer = ModelLoader.load_base_model(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            quant_config=quant_config,
            device_map=device_map,
            attn_implementation=attn_implementation,
            use_liger_kernel=use_liger_kernel,
        )

        # Carrega adaptadores LoRA
        model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            is_trainable=True,
        )

        print_cores(f"<verde>✅ Modelo LoRA treinado carregado com sucesso!</verde>", color_auto=False)

        # Imprime configuração dos adaptadores
        if hasattr(model, 'peft_config'):
            for adapter_name, config in model.peft_config.items():
                print(f"  - Adapter '{adapter_name}': r={config.r}, alpha={config.lora_alpha}")

        return model, tokenizer

    @staticmethod
    def merge_lora_to_base(model: PeftModel) -> PreTrainedModel:
        """Mescla adaptadores LoRA ao modelo base (para full fine-tuning posterior).

        Args:
            model: Modelo com adaptadores LoRA

        Returns:
            Modelo base com pesos mesclados (não é mais PeftModel)
        """
        print_cores("<azul>🔄 Mesclando adaptadores LoRA ao modelo base...</azul>", color_auto=False)

        # Merge e unload (retorna modelo base com pesos atualizados)
        merged_model = model.merge_and_unload()

        print_cores("<verde>✅ LoRA mesclado ao modelo base!</verde>", color_auto=False)
        print(f"  - Tipo do modelo: {type(merged_model).__name__}")
        print(f"  - Parâmetros totais: {merged_model.num_parameters():,}")

        return merged_model

    @staticmethod
    def check_lora_exists(output_dir: str) -> bool:
        """Verifica se existe modelo LoRA treinado em um diretório.

        Args:
            output_dir: Diretório de saída do treinamento

        Returns:
            True se existem arquivos de adaptadores LoRA válidos
        """
        adapter_config = os.path.join(output_dir, "adapter_config.json")
        adapter_model = os.path.join(output_dir, "adapter_model.safetensors")
        adapter_model_bin = os.path.join(output_dir, "adapter_model.bin")

        return os.path.exists(adapter_config) and (
            os.path.exists(adapter_model) or os.path.exists(adapter_model_bin)
        )

    @staticmethod
    def save_merged_model(
        model: PeftModel,
        output_dir: str,
        tokenizer: PreTrainedTokenizer,
        safe_serialization: bool = True,
    ) -> None:
        """Salva modelo com adaptadores LoRA mesclados ao base model.

        Args:
            model: Modelo PEFT com adaptadores LoRA
            output_dir: Diretório de saída
            tokenizer: Tokenizer a salvar junto
            safe_serialization: Usar safetensors (recomendado)
        """
        print_cores(f"<azul>💾 Mesclando e salvando modelo em: {output_dir}...</azul>", color_auto=False)

        # Mescla adaptadores ao modelo base
        merged_model = model.merge_and_unload()

        # Salva modelo mesclado
        merged_model.save_pretrained(
            output_dir,
            safe_serialization=safe_serialization,
        )

        # Salva tokenizer
        tokenizer.save_pretrained(output_dir)

        print_cores(f"<verde>✅ Modelo mesclado salvo com sucesso!</verde>", color_auto=False)
        print(f"  - Formato: {'safetensors' if safe_serialization else 'pytorch'}")
        print(f"  - Localização: {output_dir}")

    @staticmethod
    def save_model_for_inference(
        model: PreTrainedModel,
        output_dir: str,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        """Salva modelo para inferência (sem necessidade de merge).

        Args:
            model: Modelo (PEFT ou base)
            output_dir: Diretório de saída
            tokenizer: Tokenizer
        """
        print_cores(f"<azul>💾 Salvando modelo para inferência: {output_dir}...</azul>", color_auto=False)

        os.makedirs(output_dir, exist_ok=True)

        # Se for PEFT, salva apenas os adaptadores
        if hasattr(model, 'peft_config'):
            model.save_pretrained(output_dir)
            print_cores("<verde>✅ Adaptadores LoRA salvos!</verde>", color_auto=False)
        else:
            # Modelo completo
            model.save_pretrained(output_dir, safe_serialization=True)
            print_cores("<verde>✅ Modelo completo salvo!</verde>", color_auto=False)

        # Tokenizer
        tokenizer.save_pretrained(output_dir)

    @staticmethod
    def print_model_info(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        """Imprime informações detalhadas sobre o modelo carregado."""
        is_peft = hasattr(model, 'peft_config')

        print_cores("\n<azul>📊 INFORMAÇÕES DO MODELO:</azul>", color_auto=False)
        print(f"  - Tipo: {type(model).__name__}")
        print(f"  - É modelo PEFT: {is_peft}")
        print(f"  - Device: {next(model.parameters()).device}")
        print(f"  - Dtype: {next(model.parameters()).dtype}")

        if is_peft:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  - Parâmetros treináveis: {trainable:,} ({100*trainable/total:.2f}%)")
            print(f"  - Adaptadores:")
            for name, config in model.peft_config.items():
                print(f"    * {name}: r={config.r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
        else:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  - Parâmetros totais: {total:,}")
            print(f"  - Parâmetros treináveis: {trainable:,}")

        # Informações de contexto do modelo
        base_config = model.config if not is_peft else getattr(model, 'config', model.peft_config)
        native_max_pos = getattr(base_config, 'max_position_embeddings', 'N/A')

        print(f"\n  - Tokenizer: {type(tokenizer).__name__}")
        print(f"  - Vocab size: {len(tokenizer)}")
        print(f"  - Tokenizer max length: {tokenizer.model_max_length}")
        print(f"  - Contexto nativo (max_position_embeddings): {native_max_pos}")
        print(f"  - Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
        print()
