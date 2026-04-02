#!/usr/bin/env python3

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Treinar Gemma‑3, Deepseek, Llhama, Qwen usando HuggingFace Transformers + PEFT
        + TRL‑SFTTrainer de forma configurável por yaml.

Uso:
    python treinar_unsloth.py [CONFIG.yaml] [--treinar] [--reset]

* Se CONFIG.yaml for omitido, exibe menu interativo para seleção do arquivo.
* Se nenhuma ação for informada, exibe menu interativo de ações de treinamento.
* Para avaliação, predição e exportação, use: treinar_unsloth_avaliar.py
* O código utilizará automaticamente todas as GPUs disponíveis,
  sendo gerenciado pelo ambiente do sistema operacional.
  Para isolar as GPUs que serão usadas, defina a variável de ambiente para o tensorflow CUDA_VISIBLE_DEVICES = <IDs das GPUs>.
  Exemplo: export CUDA_VISIBLE_DEVICES=0,1,2  (no Linux para utilizar as GPUs 0, 1 e 2)
* O parâmetro **--debug** ativa modo de debug que carrega e apresenta
  a estrutura do dataset e configurações importantes sem executar treino.
* O parâmetro **--modelo N** executa predições em N exemplos do dataset.

**FUNCIONALIDADE DE CHECKPOINTS:**
* O treinamento verifica automaticamente por checkpoints existentes na pasta
  output_dir/chkpt e tenta continuar do último checkpoint válido.
* Se houver erro ao carregar checkpoint (mudança de parâmetros), o treinamento
  reinicia do zero mas preserva o modelo LoRA já treinado.
* Use resume_from_checkpoint: false no YAML para desabilitar checkpoints.

Exemplo de YAML gerado automaticamente:
```yaml
dataset_train_path: "../dataset/data/dados_unificados_sm_treino.parquet"
train_prompt_col: "messages"
base_model_name: "unsloth/gemma-3-12b-it-unsloth-bnb-4bit"
output_dir: "../modelos/gemma-3-12b-refleg20k-v01"
batch_size: 2
grad_batch_size: 5
num_train_epochs: 1
max_seq_length: 4096
lora_r: 8
save_checkpoints: True
resume_from_checkpoint: True
dataset_eval_path: ""     # opcional

# Otimizações de memória GPU (padrão: ativadas)
# flash_attention_2: atenção O(n) VRAM vs O(n²). Requer: pip install flash-attn --no-build-isolation
# liger_kernel: fused cross-entropy ~40% menos VRAM pico. Requer: pip install liger-kernel
# Se ativados e não instalados, o treinamento será interrompido com sugestão de instalação.
flash_attention_2: true
liger_kernel: true
```
"""

import argparse
from cmath import inf
import math
import os, time, json
import sys
import traceback
from typing import Any, Dict
import dataclasses
import yaml
import torch

# Nota: Não precisamos mais desabilitar torch.compile (era necessário apenas para Unsloth)
# Mantido para compatibilidade com sistemas sem compilador C
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
# try:
#     import torch._dynamo
#     torch._dynamo.config.suppress_errors = True
#     torch._dynamo.config.disable = True
# except ImportError:
#     pass

# === Otimizações de memória GPU ===
# Liger Kernel: fused cross-entropy + fused RoPE + fused RMSNorm
#   Reduz pico de VRAM ~40% ao evitar materializar tensor de logits completo
#   (batch × seq_len × vocab_size × 4B). Crítico para sequências longas (>16k tokens).
# Flash Attention 2: atenção O(n) vs O(n²) em VRAM.
# A disponibilidade é verificada pelo treinar_model_loader.py.
# Se ativados no YAML e não instalados, o treinamento será interrompido com instruções.
from treinar_model_loader import _LIGER_DISPONIVEL, _FLASH_ATTN_DISPONIVEL


import pandas as pd

import util  # garante que a pasta src está no sys.path

# Sistema de logging centralizado
from treinar_unsloth_logging import get_logger, configurar_logging, log_separador, log_bloco
from treinar_unsloth_monitor import MonitorRecursos

try:
    from datasets import Dataset
except ImportError:
    print("Erro: O pacote 'datasets' não está instalado.")
    print("Por favor, instale-o executando: pip install datasets")
    sys.exit(1)

try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    print("Erro: O pacote 'trl' não está instalado.")
    print("Por favor, instale-o executando: pip install trl")
    sys.exit(1)

# --- Patch TRL: compatibilidade com Liger Kernel fused cross-entropy ---
# O Liger Kernel fused CE computa a loss diretamente a partir de hidden_states
# SEM materializar o tensor de logits completo (economia ~40% VRAM pico).
# Isso faz outputs.logits = None, mas TRL chama entropy_from_logits(outputs.logits)
# que assume logits != None. Este patch torna a função segura para logits=None.
try:
    import trl.trainer.utils as _trl_utils
    import trl.trainer.sft_trainer as _trl_sft

    _original_entropy_from_logits = _trl_utils.entropy_from_logits

    def _safe_entropy_from_logits(logits):
        """Wrapper que retorna tensor zero quando logits é None (Liger Kernel fused CE)."""
        if logits is None:
            return torch.tensor(0.0)
        return _original_entropy_from_logits(logits)

    # Patch em ambos os módulos (utils define, sft_trainer importa via from...import)
    _trl_utils.entropy_from_logits = _safe_entropy_from_logits
    _trl_sft.entropy_from_logits = _safe_entropy_from_logits
except Exception:
    pass  # Se não conseguir patchar, o erro original do TRL aparecerá

try:
    from transformers import TrainerCallback, GenerationConfig
except ImportError:
    print("Erro: O pacote 'transformers' não está instalado.")
    print("Por favor, instale-o executando: pip install transformers")
    sys.exit(1)

try:
    from peft import PeftModel, LoraConfig, get_peft_model
except ImportError:
    print("Erro: O pacote 'peft' não está instalado.")
    print("Por favor, instale-o executando: pip install peft")
    sys.exit(1)

import numpy as np
from datetime import datetime
from copy import deepcopy

# Novo módulo para carregamento de modelos (substitui Unsloth)
from treinar_model_loader import ModelLoader, QuantizationConfig

# Import da nova classe de configuração YAML e Gerador de Relatório
from treinar_unsloth_util import YamlTreinamento, calcular_rouge_l
from treinar_unsloth_report import GeradorRelatorio
from treinar_chat_templates import TreinarChatTemplate, get_data_collator_for_completion_only
from util import UtilEnv, Util
from util_print import print_cores, exibir_menu_opcoes


# ---------------------------------------------------------------------------
# Logger do módulo
# ---------------------------------------------------------------------------

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------

def _print_mem(tag: str, incluir_ram: bool = True) -> dict:
    """
    Exibe estatísticas de memória (RAM e GPU) para depuração rápida.
    Utiliza Util.dados_hardware() para obter informações centralizadas.
    
    Args:
        tag: Identificador para o print (ex: "ANTES", "DEPOIS")
        incluir_ram: Se True, também exibe informações de RAM
        
    Returns:
        Dict com informações completas de hardware (CPU, RAM, GPU)
    """
    try:
        hardware = Util.dados_hardware(incluir_gpu=True)
    except Exception as e:
        logger.warning(f"[{tag}] Erro ao obter dados de hardware: {e}")
        return {}
    
    # Exibe informações de RAM
    if incluir_ram:
        ram_total = hardware.get('mem_total_gb', 0)
        ram_usada = hardware.get('mem_usada_gb', 0)
        ram_disp = hardware.get('mem_disponivel_gb', 0)
        ram_uso = hardware.get('mem_uso_%', 0)
        logger.info(f"[{tag}] RAM | usada: {ram_usada:.2f} GB / total: {ram_total:.2f} GB ({ram_uso:.1f}%) | disponível: {ram_disp:.2f} GB")
    
    # Exibe informações de GPU
    gpu_info = hardware.get('gpu', {})
    if not gpu_info.get('disponivel', False):
        motivo = gpu_info.get('motivo', 'CUDA não disponível')
        logger.info(f"[{tag}] GPU | {motivo}")
    else:
        gpus = gpu_info.get('gpus', [])
        for gpu in gpus:
            if 'erro' in gpu:
                logger.warning(f"[{tag}] GPU[{gpu['idx']}] Erro: {gpu['erro']}")
            else:
                nome = gpu.get('nome', 'N/A')
                mem_total = gpu.get('mem_total_gb', 0)
                mem_reservada = gpu.get('mem_reservada_gb', 0)
                mem_max_reservada = gpu.get('mem_max_reservada_gb', 0)
                mem_alocada = gpu.get('mem_alocada_gb', 0)
                logger.info(f"[{tag}] GPU[{gpu['idx']}] {nome} | reservada: {mem_max_reservada:.2f} GB / total: {mem_total:.2f} GB | alocada: {mem_alocada:.2f} GB")
    
    return hardware

class JsonLoggerCallback(TrainerCallback):
    """Callback para registrar métricas de treinamento em formato JSONL."""
    
    def __init__(self, path, truncar: bool = True):
        self.path = path
        # zera o arquivo no início apenas se não for retomada
        if truncar:
            open(self.path, "w").close()

    # logs = {'loss': …}  ou  {'eval_loss': …}
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs["step"]  = state.global_step
            logs["epoch"] = state.epoch
            logs["time"]  = time.time()
            with open(self.path, "a") as fp:
                fp.write(json.dumps(logs, ensure_ascii=False) + "\n")

    # garante que também pegamos o dicionário completo emitido
    # pelo método evaluate() externo, se você chamá-lo no fim
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.on_log(args, state, control, metrics)


class MetricsLoggerCallback(TrainerCallback):
    """
    Callback unificado para registrar TODAS as métricas de treinamento.
    
    Registra em um único arquivo JSONL:
    - Treinamento: loss, learning_rate, grad_norm, eval_loss
    - Hardware: CPU (%), RAM (GB), GPU (memória reservada/alocada GB), Disco (%)
    - Curriculum: etapa_alias, etapa_index, step_global, epoch_global
    - Eficiência: instancias_acumuladas, tokens_acumulados
    
    A coleta unificada garante que métricas de hardware e treinamento
    estejam no mesmo registro, com timestamp e contexto consistentes.
    
    Campos-chave em cada registro:
        - event: tipo do evento (train_begin, log, evaluate, train_end)
        - step_global: step contínuo somando todas as etapas do curriculum
        - epoch_global: época contínua somando épocas de todas as etapas
        - etapa_alias/etapa_index: identificação da etapa do curriculum
        - instancias_acumuladas: total de instâncias processadas
        - tokens_acumulados: total de tokens reais processados (baseado no comprimento
          real das instâncias tokenizadas, não em max_seq_length)
        - ram_usada_gb, gpu*_reservada_gb, cpu_uso_%: métricas de hardware
    """
    
    def __init__(self, output_dir: str, etapa_alias: str = "Principal",
                 etapa_index: int = 0, instancias_previas: int = 0,
                 step_offset: int = 0, epoch_offset: float = 0.0,
                 media_tokens_por_instancia: float = 0, tokens_previos: int = 0,
                 retomada: bool = False):
        """
        Args:
            output_dir: Diretório onde salvar o arquivo de métricas
            etapa_alias: Nome da etapa do curriculum (ex: "fácil", "médio")
            etapa_index: Índice da etapa no curriculum (0-based)
            instancias_previas: Total de instâncias treinadas em etapas anteriores
            step_offset: Total de steps acumulados de etapas anteriores (para step_global)
            epoch_offset: Total de épocas acumuladas de etapas anteriores (para epoch_global)
            media_tokens_por_instancia: Média de tokens reais por instância do dataset tokenizado
            tokens_previos: Total de tokens processados em etapas anteriores
            retomada: Se True, preserva métricas anteriores (não trunca arquivo)
        """
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, "treinamento", "training_metrics.jsonl")
        self._train_start_time = None
        self._best_eval_loss = float('inf')
        self._train_losses = []  # Para calcular média móvel
        
        # Informações de curriculum e contagem acumulada
        self._etapa_alias = etapa_alias
        self._etapa_index = etapa_index
        self._instancias_previas = instancias_previas
        self._step_offset = step_offset
        self._epoch_offset = epoch_offset
        self._effective_batch_size = 1  # Calculado em on_train_begin
        self._media_tokens_por_instancia = media_tokens_por_instancia
        self._tokens_previos = tokens_previos
        
        # Cria diretório; trunca arquivo apenas na primeira etapa e se NÃO for retomada
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        if etapa_index == 0 and not retomada:
            open(self.metrics_file, "w").close()
            # Remove gráficos e relatório estatístico anteriores para evitar
            # confusão com dados de um treinamento passado (serão regenerados ao final)
            _dir_treinamento = os.path.dirname(self.metrics_file)
            for _arq_antigo in (
                "treinamento_loss.png",
                "treinamento_tokens.png",
                "hardware_memoria.png",
                "relatorio_estatistico.md",
            ):
                _caminho = os.path.join(_dir_treinamento, _arq_antigo)
                if os.path.isfile(_caminho):
                    os.remove(_caminho)
        
    def _registrar(self, registro: dict):
        """Salva registro no arquivo JSONL.
        
        Adiciona métricas de hardware ao registro e substitui float NaN/Inf
        por None para gerar JSON válido.
        """
        try:
            # Coleta métricas de hardware no mesmo instante
            self._adicionar_hardware(registro)
            
            limpo = {
                k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
                for k, v in registro.items()
            }
            with open(self.metrics_file, "a") as fp:
                fp.write(json.dumps(limpo, ensure_ascii=False) + "\n")
        except Exception:
            pass
    
    def _adicionar_hardware(self, registro: dict) -> None:
        """Adiciona métricas de hardware ao registro (in-place)."""
        try:
            hardware = Util.dados_hardware(incluir_gpu=True)
            
            # CPU
            registro["cpu_uso_%"] = hardware.get("cpu_uso_%", 0)
            registro["cpu_uso_processo_%"] = hardware.get("cpu_uso_processo_%", 0)
            # RAM
            registro["ram_usada_gb"] = hardware.get("mem_usada_gb", 0)
            registro["ram_disponivel_gb"] = hardware.get("mem_disponivel_gb", 0)
            registro["ram_uso_%"] = hardware.get("mem_uso_%", 0)
            # Disco
            registro["disco_uso_%"] = hardware.get("disco_uso_%", 0)
            
            # GPU (pode ter múltiplas)
            gpu_info = hardware.get("gpu", {})
            if gpu_info.get("disponivel", False):
                for gpu in gpu_info.get("gpus", []):
                    idx = gpu.get("idx", 0)
                    registro[f"gpu{idx}_reservada_gb"] = gpu.get("mem_reservada_gb", 0)
                    registro[f"gpu{idx}_alocada_gb"] = gpu.get("mem_alocada_gb", 0)
                    registro[f"gpu{idx}_max_reservada_gb"] = gpu.get("mem_max_reservada_gb", 0)
        except Exception:
            pass
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Marca início do treinamento e calcula batch efetivo para contagem de instâncias."""
        self._train_start_time = time.time()
        # Batch efetivo = per_device * grad_accum * n_gpus (para cálculo de instâncias)
        n_gpus = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1
        self._effective_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * n_gpus
        )
        self._registrar({
            "event": "train_begin",
            "timestamp": self._train_start_time,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_steps": state.max_steps,
            "num_epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
            "grad_accum_steps": args.gradient_accumulation_steps,
            "etapa_alias": self._etapa_alias,
            "etapa_index": self._etapa_index,
            "step_offset": self._step_offset,
            "epoch_offset": self._epoch_offset,
            "instancias_previas": self._instancias_previas,
            "effective_batch_size": self._effective_batch_size,
            "media_tokens_por_instancia": round(self._media_tokens_por_instancia, 1),
            "tokens_previos": self._tokens_previos,
        })
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Registra métricas de treinamento a cada log, incluindo etapa e instâncias acumuladas."""
        if not logs:
            return
        
        # step_global: step contínuo que soma etapas anteriores do curriculum
        step_global = self._step_offset + state.global_step
        # instancias_acumuladas: total de instâncias processadas até este ponto
        instancias_acumuladas = (
            self._instancias_previas + state.global_step * self._effective_batch_size
        )
        
        epoch_local = round(state.epoch, 4) if state.epoch else 0
        epoch_global = round(self._epoch_offset + (state.epoch or 0), 4)
        
        registro = {
            "event": "log",
            "timestamp": time.time(),
            "step": state.global_step,
            "step_global": step_global,
            "epoch": epoch_local,
            "epoch_global": epoch_global,
            "elapsed_seconds": time.time() - self._train_start_time if self._train_start_time else 0,
            "etapa_alias": self._etapa_alias,
            "etapa_index": self._etapa_index,
            "instancias_acumuladas": instancias_acumuladas,
            "tokens_acumulados": round(self._tokens_previos + (instancias_acumuladas - self._instancias_previas) * self._media_tokens_por_instancia) if self._media_tokens_por_instancia > 0 else 0,
        }
        
        # Métricas de treinamento
        if "loss" in logs:
            val = logs["loss"]
            if not (isinstance(val, float) and math.isnan(val)):
                registro["train_loss"] = round(val, 6)
                self._train_losses.append(val)
                # Média móvel das últimas 10 perdas
                if len(self._train_losses) >= 10:
                    registro["train_loss_avg_10"] = round(sum(self._train_losses[-10:]) / 10, 6)
        
        if "learning_rate" in logs:
            registro["learning_rate"] = logs["learning_rate"]
            
        if "grad_norm" in logs:
            val = logs["grad_norm"]
            if not (isinstance(val, float) and math.isnan(val)):
                registro["grad_norm"] = round(val, 6)
        
        # Métricas de avaliação
        if "eval_loss" in logs:
            val = logs["eval_loss"]
            if not (isinstance(val, float) and math.isnan(val)):
                registro["eval_loss"] = round(val, 6)
                if val < self._best_eval_loss:
                    self._best_eval_loss = val
                    registro["is_best_eval"] = True
        
        # Progresso
        if state.max_steps > 0:
            registro["progress_%"] = round(state.global_step / state.max_steps * 100, 2)
        
        self._registrar(registro)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Registra métricas completas de avaliação com contexto de etapa.
        
        Ignora eventos de avaliação global (metric_key_prefix='eval_global')
        pois estes são registrados pelo GlobalEvalCallback como evento separado.
        """
        if not metrics:
            return
        
        # Ignora eventos de avaliação global — são registrados pelo GlobalEvalCallback
        if any(k.startswith("eval_global_") for k in metrics.keys()):
            return
        
        step_global = self._step_offset + state.global_step
        instancias_acumuladas = (
            self._instancias_previas + state.global_step * self._effective_batch_size
        )
        
        epoch_local = round(state.epoch, 4) if state.epoch else 0
        epoch_global = round(self._epoch_offset + (state.epoch or 0), 4)
        
        registro = {
            "event": "evaluate",
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": state.global_step,
            "step_global": step_global,
            "epoch": epoch_local,
            "epoch_global": epoch_global,
            "etapa_alias": self._etapa_alias,
            "etapa_index": self._etapa_index,
            "instancias_acumuladas": instancias_acumuladas,
            "tokens_acumulados": round(self._tokens_previos + (instancias_acumuladas - self._instancias_previas) * self._media_tokens_por_instancia) if self._media_tokens_por_instancia > 0 else 0,
        }
        
        # Copia todas as métricas de avaliação
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                registro[key] = round(value, 6) if isinstance(value, float) else value
        
        self._registrar(registro)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Registra resumo final da etapa com totais acumulados."""
        elapsed = time.time() - self._train_start_time if self._train_start_time else 0
        step_global = self._step_offset + state.global_step
        instancias_acumuladas = (
            self._instancias_previas + state.global_step * self._effective_batch_size
        )
        
        epoch_local = round(state.epoch, 4) if state.epoch else 0
        epoch_global = round(self._epoch_offset + (state.epoch or 0), 4)
        
        registro = {
            "event": "train_end",
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_steps": state.global_step,
            "step_global": step_global,
            "final_epoch": epoch_local,
            "final_epoch_global": epoch_global,
            "total_time_seconds": round(elapsed, 2),
            "total_time_formatted": f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s",
            "best_eval_loss": round(self._best_eval_loss, 6) if self._best_eval_loss != float('inf') else None,
            "final_train_loss_avg": round(sum(self._train_losses[-10:]) / len(self._train_losses[-10:]), 6) if self._train_losses else None,
            "etapa_alias": self._etapa_alias,
            "etapa_index": self._etapa_index,
            "instancias_acumuladas": instancias_acumuladas,
            "tokens_acumulados": round(self._tokens_previos + (instancias_acumuladas - self._instancias_previas) * self._media_tokens_por_instancia) if self._media_tokens_por_instancia > 0 else 0,
        }
        
        self._registrar(registro)

# ---------------------------------------------------------------------------
# Callback para renomear checkpoints com zero-padding
# ---------------------------------------------------------------------------

class CheckpointRenameCallback(TrainerCallback):
    """
    Callback para renomear checkpoints com zero-padding e step global.
    
    Em modo curriculum, global_step reseta em cada etapa. Usa step_offset
    para nomear checkpoints com step global contínuo:
        checkpoint-1 (step_offset=2) -> checkpoint-00003
    """
    
    def __init__(self, checkpoint_base_dir: str, step_offset: int = 0, padding: int = 5):
        self.checkpoint_base_dir = checkpoint_base_dir
        self.step_offset = step_offset
        self.padding = padding
    
    def on_save(self, args, state, control, **kwargs):
        """Renomeia o checkpoint salvo para usar zero-padding com step global."""
        if not state.global_step:
            return
            
        # Caminho original criado pelo Trainer (usa global_step da etapa)
        original_name = f"checkpoint-{state.global_step}"
        # Nome final com step global contínuo + zero-padding
        step_global = self.step_offset + state.global_step
        padded_name = f"checkpoint-{step_global:0{self.padding}d}"
        
        original_path = os.path.join(self.checkpoint_base_dir, original_name)
        padded_path = os.path.join(self.checkpoint_base_dir, padded_name)
        
        if os.path.exists(original_path) and not os.path.exists(padded_path):
            try:
                os.rename(original_path, padded_path)
                logger.debug(f"Checkpoint renomeado: {original_name} -> {padded_name}")
                # Atualiza best_model_checkpoint para que o Trainer consiga
                # encontrar o melhor modelo após a renomeação (load_best_model_at_end)
                if state.best_model_checkpoint and state.best_model_checkpoint.rstrip("/") == original_path.rstrip("/"):
                    state.best_model_checkpoint = padded_path
                    logger.debug(f"best_model_checkpoint atualizado: {padded_path}")
            except Exception as e:
                logger.warning(f"Erro ao renomear checkpoint: {e}")
        elif os.path.exists(padded_path):
            # Esperado em resume_from_checkpoint (checkpoint já foi renomeado)
            logger.debug(f"Checkpoint já existe com zero-padding: {padded_name}")
            # Mesmo sem renomear, garante que best_model_checkpoint aponte para o nome correto
            if state.best_model_checkpoint and state.best_model_checkpoint.rstrip("/") == original_path.rstrip("/"):
                state.best_model_checkpoint = padded_path
                logger.debug(f"best_model_checkpoint corrigido: {padded_path}")


# ---------------------------------------------------------------------------
# Callback para pace_loss (early stopping por loss com mínimo de épocas)
# ---------------------------------------------------------------------------

class PaceLossCallback(TrainerCallback):
    """Callback que interrompe o treinamento quando eval_loss < pace_loss, respeitando pace_epochs mínimo.
    
    Lógica:
        - Ignora qualquer verificação enquanto epoch < pace_epochs (mínimo garantido)
        - Após pace_epochs: se eval_loss < pace_loss → para o treinamento
        - Se pace_epochs_max está configurado, o num_train_epochs já é pace_epochs_max,
          então o treinamento para naturalmente se pace_loss nunca for atingido.
    
    O loss verificado é o eval_loss (validation loss) mais recente, capturado via on_evaluate.
    Usar eval_loss em vez de training loss é o padrão acadêmico: evita decisões
    baseadas em overfitting (training loss baixo mas sem generalização real).
    A verificação é feita no on_epoch_end para decisões em fronteiras de época.
    """
    
    def __init__(self, pace_loss: float, pace_epochs: int, pace_epochs_max: int = 0,
                 etapa_alias: str = "Principal"):
        self.pace_loss = pace_loss
        self.pace_epochs = pace_epochs
        self.pace_epochs_max = pace_epochs_max
        self.etapa_alias = etapa_alias
        self._last_eval_loss = None
        self._last_train_loss = None
        self._stopped = False
        self._stop_epoch = None
        self._stop_loss = None
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Rastreia o último training loss (para log informativo)."""
        if logs and "loss" in logs:
            self._last_train_loss = logs["loss"]
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Rastreia o último eval_loss da etapa atual."""
        if metrics and "eval_loss" in metrics:
            self._last_eval_loss = metrics["eval_loss"]
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Verifica se o pace_loss foi atingido após o mínimo de épocas."""
        if self._stopped:
            return
        
        current_epoch = int(state.epoch) if state.epoch else 0
        
        # Respeita o mínimo de épocas
        if current_epoch < self.pace_epochs:
            return
        
        # Precisa de eval_loss para decidir
        if self._last_eval_loss is None:
            train_info = f", train_loss={self._last_train_loss:.4f}" if self._last_train_loss is not None else ""
            logger.info(
                f"📉 Etapa '{self.etapa_alias}' época {current_epoch}: "
                f"eval_loss indisponível{train_info} (pace_loss requer eval — configure eval_steps)"
            )
            return
        
        # Verifica se eval_loss atingiu o alvo
        if self._last_eval_loss < self.pace_loss:
            self._stopped = True
            self._stop_epoch = current_epoch
            self._stop_loss = self._last_eval_loss
            control.should_training_stop = True
            logger.info(
                f"🎯 pace_loss atingido na etapa '{self.etapa_alias}': "
                f"eval_loss={self._last_eval_loss:.4f} < pace_loss={self.pace_loss} "
                f"(época {current_epoch}/{self.pace_epochs_max or self.pace_epochs})"
            )
        else:
            train_info = f", train_loss={self._last_train_loss:.4f}" if self._last_train_loss is not None else ""
            logger.info(
                f"📉 Etapa '{self.etapa_alias}' época {current_epoch}: "
                f"eval_loss={self._last_eval_loss:.4f}{train_info} (alvo: < {self.pace_loss})"
            )


# ---------------------------------------------------------------------------
# Callback para avaliação global (todas as etapas do curriculum)
# ---------------------------------------------------------------------------

class GlobalEvalCallback(TrainerCallback):
    """Callback que executa avaliação no dataset global (todas as etapas combinadas).
    
    Em treinamentos com curriculum learning multi-etapa, o eval_loss padrão
    reflete apenas a etapa atual. Este callback adiciona eval_loss_global
    avaliando o modelo contra dados de validação de TODAS as etapas,
    permitindo monitorar se o modelo mantém desempenho geral.
    
    O campo eval_loss_global é registrado no training_metrics.jsonl
    junto com o eval_loss da etapa atual para comparação direta.
    """
    
    def __init__(self, global_eval_dataset, metrics_file: str,
                 step_offset: int = 0, epoch_offset: float = 0.0,
                 etapa_alias: str = "Principal", etapa_index: int = 0):
        """
        Args:
            global_eval_dataset: Dataset HF tokenizado com validação de todas as etapas
            metrics_file: Caminho do arquivo training_metrics.jsonl
            step_offset: Steps acumulados de etapas anteriores
            epoch_offset: Épocas acumuladas de etapas anteriores
            etapa_alias: Nome da etapa atual
            etapa_index: Índice da etapa atual
        """
        self._global_ds = global_eval_dataset
        self._metrics_file = metrics_file
        self._step_offset = step_offset
        self._epoch_offset = epoch_offset
        self._etapa_alias = etapa_alias
        self._etapa_index = etapa_index
        self._trainer_ref = None  # Preenchido após criação do trainer
        self._best_eval_loss_global = float('inf')
        self._avaliando_global = False  # Guarda contra recursão
    
    def set_trainer(self, trainer):
        """Armazena referência ao trainer para chamar evaluate()."""
        self._trainer_ref = trainer
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Após cada avaliação padrão, executa avaliação no dataset global."""
        if self._trainer_ref is None or self._global_ds is None:
            return
        if len(self._global_ds) == 0:
            return
        # Evita recursão: on_evaluate é chamado quando o próprio global eval termina
        if self._avaliando_global:
            return
        
        try:
            # Marca que estamos avaliando global (evita recursão)
            self._avaliando_global = True
            
            # Salva dataset de eval original do trainer
            original_eval_ds = self._trainer_ref.eval_dataset
            
            # Desabilita temporariamente load_best_model_at_end para não
            # interferir com a seleção do melhor modelo da etapa atual
            original_load_best = self._trainer_ref.args.load_best_model_at_end
            self._trainer_ref.args.load_best_model_at_end = False
            
            # Executa avaliação no dataset global
            self._trainer_ref.eval_dataset = self._global_ds
            global_metrics = self._trainer_ref.evaluate(
                metric_key_prefix="eval_global"
            )
            
            # Restaura configurações originais
            self._trainer_ref.eval_dataset = original_eval_ds
            self._trainer_ref.args.load_best_model_at_end = original_load_best
            self._avaliando_global = False
            
            eval_loss_global = global_metrics.get("eval_global_loss")
            if eval_loss_global is not None:
                is_best = eval_loss_global < self._best_eval_loss_global
                if is_best:
                    self._best_eval_loss_global = eval_loss_global
                
                step_global = self._step_offset + state.global_step
                epoch_global = round(self._epoch_offset + (state.epoch or 0), 4)
                
                registro = {
                    "event": "evaluate_global",
                    "timestamp": time.time(),
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "step": state.global_step,
                    "step_global": step_global,
                    "epoch_global": epoch_global,
                    "etapa_alias": self._etapa_alias,
                    "etapa_index": self._etapa_index,
                    "eval_loss_global": round(eval_loss_global, 6),
                }
                if is_best:
                    registro["is_best_eval_global"] = True
                
                # Copia métricas adicionais
                for key, value in global_metrics.items():
                    if key != "eval_global_loss" and isinstance(value, (int, float)):
                        registro[key] = round(value, 6) if isinstance(value, float) else value
                
                # Adiciona métricas de hardware
                try:
                    hardware = Util.dados_hardware(incluir_gpu=True)
                    registro["cpu_uso_%"] = hardware.get("cpu_uso_%", 0)
                    registro["ram_usada_gb"] = hardware.get("mem_usada_gb", 0)
                    gpu_info = hardware.get("gpu", {})
                    if gpu_info.get("disponivel", False):
                        for gpu in gpu_info.get("gpus", []):
                            idx = gpu.get("idx", 0)
                            registro[f"gpu{idx}_reservada_gb"] = gpu.get("mem_reservada_gb", 0)
                            registro[f"gpu{idx}_alocada_gb"] = gpu.get("mem_alocada_gb", 0)
                except Exception:
                    pass
                
                # Grava no JSONL
                try:
                    limpo = {
                        k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
                        for k, v in registro.items()
                    }
                    with open(self._metrics_file, "a") as fp:
                        fp.write(json.dumps(limpo, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                
                logger.info(
                    f"📊 eval_loss_global: {eval_loss_global:.4f}"
                    f"{' ⭐ melhor!' if is_best else ''}"
                    f" (etapa '{self._etapa_alias}', step_global={step_global})"
                )
        except Exception as e:
            self._avaliando_global = False
            logger.warning(f"⚠️  Erro na avaliação global: {e}")


# ---------------------------------------------------------------------------
# classe principal
# ---------------------------------------------------------------------------

class LLMsTrainer:
    """Encapsula o fluxo de fine‑tuning de LLMs com LoRA e Unsloth."""

    # Chaves obrigatórias no formato flat (para compatibilidade do método para_config_flat)
    # REQUIRED_KEYS removido pois validação é feita nos dataclasses

    def __init__(self, cfg_path: str, force_base: bool = False):
        # Carrega configuração YAML
        self._yaml_config = YamlTreinamento(cfg_path)
        
        self.force_base = force_base
        
        # Cria a pasta de saída se não existir
        os.makedirs(self._yaml_config.modelo.saida, exist_ok=True)
        
        # Pipeline Universal: etapas e rastreamento unificado
        # Usa apenas etapas treináveis (tipo não-vazio); etapas com tipo vazio
        # são usadas apenas pelo predict para cópia de predições por etapa.
        from treinar_unsloth_pipeline import CurriculumTracker
        self._etapas = self._yaml_config.curriculum_treino
        self._tracker = CurriculumTracker(self._yaml_config.modelo.saida)
        
        # Histórico de treinamento
        from treinar_unsloth_historico import HistoricoTreinamento
        self._historico = HistoricoTreinamento(
            output_dir=self._yaml_config.modelo.saida,
            yaml_path=cfg_path
        )
        
        # Determina se é novo treinamento (sem modelo LoRA existente)
        _arq_adapter = os.path.join(self._yaml_config.modelo.saida, 'adapter_config.json')
        self._is_novo_treinamento = not os.path.exists(_arq_adapter)
        
        # Valida max_seq_length (obrigatório) e exibe info de tokens por etapa
        self._yaml_config.validar_max_seq_length()
        
        # Carrega modelo e tokenizer
        # Nota: O modelo é carregado com seu max_position_embeddings nativo
        # (ex: Qwen2.5 = 32768 com rope_theta=1e6 suportando até 131072).
        # O max_seq_length de treinamento controla apenas a truncagem de dados
        # via SFTConfig.max_length, não a arquitetura do modelo.
        self.model, self.tokenizer = self._load_model()
        
        # Gerenciador de templates de chat
        self.chat_handler = TreinarChatTemplate(self.tokenizer, self._yaml_config.modelo.base)
        self.tokenizer = self.chat_handler.tokenizer
        
        # Carrega datasets a partir do curriculum (pastas/dataframes + divisão)
        self.train_ds = self._load_from_pastas(alvo="treino")
        self.eval_ds = self._load_from_pastas(alvo="validacao")
        
        # Dataset de validação global (todas as etapas) para curriculum multi-etapa
        # Controlado por treinamento.eval_global (padrão: true)
        self.eval_ds_global = None
        if len(self._etapas) > 1 and self._yaml_config.treinamento.eval_global:
            self.eval_ds_global = self._load_global_eval_dataset()
        elif len(self._etapas) > 1 and not self._yaml_config.treinamento.eval_global:
            print_cores('<cinza>   ℹ️  eval_global desativado via YAML (treinamento.eval_global: false)</cinza>', color_auto=False)
        
        # Exibe estatísticas pré-treinamento e armazena para relatório
        ts = self._print_dataset_stats(self.train_ds, "Dataset de Treino")
        es = self._print_dataset_stats(self.eval_ds, "Dataset de Validação") if self.eval_ds and len(self.eval_ds) > 0 else {}
        
        self._dataset_stats = {
            "treino_len": len(self.train_ds),
            "validacao_len": len(self.eval_ds) if self.eval_ds else 0,
            "token_stats": ts
        }

        self.save_checkpoints = self._yaml_config.treinamento.save_checkpoints
        self.trainer = None  # Inicialização lazy no método train()
        self._global_eval_callback = None  # Preenchido em _build_trainer se curriculum multi-etapa
        
        # Inicializa histórico: se é novo treinamento, gera todos os arquivos
        # Se é continuação, registra retomada e verifica YAML
        if self._is_novo_treinamento:
            self._historico.inicializar_novo_treinamento(
                yaml_config=self._yaml_config,
                model=self.model,
                tokenizer=self.tokenizer,
                train_ds=self.train_ds,
                eval_ds=self.eval_ds,
            )
        else:
            self._historico.evento_reinicio(
                motivo="Continuação de treinamento existente"
            )

    # ------------------------- controle no colab ------------------------------
    @classmethod
    def verifica_versao(cls):
        print(f'JsonAnalise carregado corretamente em {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}!')
        
    # ------------------------- configuração ------------------------------
    def _validate_cfg(self) -> None:
        """Validação já realizada pela YamlTreinamento."""
        pass
    
    def _print_dataset_stats(self, dataset: Dataset, nome: str) -> dict:
        """Exibe estatísticas de tokens do dataset e retorna dict com dados."""
        if dataset is None or len(dataset) == 0:
            print(f"📊 {nome}: vazio")
            return {}
        
        lengths = [len(r.get('input_ids', [])) for r in dataset]
        if not lengths or max(lengths) == 0:
            # Dataset ainda não tokenizado, conta mensagens
            print(f"📊 {nome}: {len(dataset)} registros (não tokenizado)")
            return {}
        
        min_l, max_l = min(lengths), max(lengths)
        avg_l = sum(lengths)/len(lengths)
        
        print(f"📊 {nome}:")
        print(f"   Registros: {len(dataset)}")
        print(f"   Tokens: min={min_l}, max={max_l}, média={avg_l:.0f}")
        
        # Alerta se houver sequências que excedem max_seq_length
        max_seq = self._yaml_config.treinamento.max_seq_length
        excedem = sum(1 for l in lengths if l > max_seq)
        if excedem > 0:
            print(f"   ⚠️  {excedem} registros excedem max_seq_length={max_seq}")
        
        return {
            "min": min_l,
            "max": max_l,
            "avg": round(avg_l, 1),
            "exceed_max_seq": excedem
        }
    
    def _load_from_pastas(self, alvo: str) -> Dataset:
        """Carrega dataset a partir de pastas usando YamlTreinamento."""
        print(f"[2/6] Carregando dados de pastas (alvo={alvo})...")
        
        # Carrega mensagens usando dataset_manager via YamlTreinamento
        mensagens = self._yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo=alvo)
        
        if not mensagens:
            print(f"   ⚠️  Nenhum registro encontrado para alvo='{alvo}'")
            return Dataset.from_list([])
        
        print(f"   Encontrados {len(mensagens)} registros para {alvo}")
        
        # Cria LLMsDataset a partir dos dados em memória
        dataset_loader = LLMsDataset(
            data=mensagens,
            tokenizer=self.tokenizer,
            max_seq_length=self._yaml_config.treinamento.max_seq_length
        )
        
        ds = dataset_loader.dataset
        return ds
        
    def _load_global_eval_dataset(self) -> Dataset:
        """Carrega dataset de validação unificado de TODAS as etapas do curriculum.
        
        Utiliza carregar_divisao_completa() para obter os IDs de validação
        de todas as etapas e carregar as mensagens correspondentes.
        Retorna None se houver apenas 1 etapa ou se não houver dados.
        """
        try:
            # Constrói divisão unificada com todas as etapas (incluindo não-treináveis)
            todas_etapas = self._yaml_config.curriculum
            divisao_unificada = self._yaml_config.dataset_manager.carregar_divisao_completa(todas_etapas)
            
            if not divisao_unificada:
                logger.warning("⚠️  Não foi possível construir divisão unificada para eval global")
                return None
            
            # Conta quantos IDs de validação existem na divisão unificada
            ids_val_global = [
                id_arq for id_arq, info in divisao_unificada.items()
                if info["alvo"] == "validacao"
            ]
            
            # Compara com validação da etapa atual
            n_val_etapa = len(self.eval_ds) if self.eval_ds else 0
            n_val_global = len(ids_val_global)
            
            if n_val_global <= n_val_etapa:
                # Validação global não traz dados extras — não vale a pena
                logger.info(f"ℹ️  Eval global: {n_val_global} instância(s) = mesma qtde da etapa atual ({n_val_etapa}). Desativado.")
                return None
            
            # Carrega mensagens de validação usando a divisão unificada
            mensagens = self._yaml_config.dataset_manager.carregar_mensagens_de_pastas(
                alvo="validacao", divisao=divisao_unificada
            )
            
            if not mensagens:
                return None
            
            # Converte para HF Dataset
            dataset_loader = LLMsDataset(
                data=mensagens,
                tokenizer=self.tokenizer,
                max_seq_length=self._yaml_config.treinamento.max_seq_length
            )
            
            ds = dataset_loader.dataset
            if ds and len(ds) > 0:
                print_cores(
                    f"<cinza>   📊 Eval global: {len(ds)} instâncias de validação "
                    f"(todas as etapas combinadas, vs {n_val_etapa} da etapa atual)</cinza>",
                    color_auto=False
                )
                return ds
            
            return None
        except Exception as e:
            logger.warning(f"⚠️  Erro ao carregar eval global: {e}")
            return None

    # ------------------------- modelo ------------------------------------
    def _load_model(self):
        """Carrega modelo usando HuggingFace Transformers + PEFT.

        Lógica:
        1. Valida disponibilidade de flash_attention_2 e liger_kernel
        2. Se --base não foi passado, tenta carregar modelo LoRA já treinado (resume)
        3. Se falhar ou não existir, carrega modelo base
        4. Aplica adaptadores LoRA se configurado (lora.r > 0)
        """
        print_cores("<azul>[1/6] Carregando modelo base…</azul>", color_auto=False)

        # --- Validação de componentes de otimização de memória ---
        use_flash_attn = self._yaml_config.treinamento.flash_attention_2
        use_liger = self._yaml_config.treinamento.liger_kernel

        if use_flash_attn and not _FLASH_ATTN_DISPONIVEL:
            raise RuntimeError(
                "\n" + "=" * 70 + "\n"
                "❌ Flash Attention 2 está ATIVADO no YAML (treinamento.flash_attention_2: true)\n"
                "   mas o pacote 'flash-attn' não está instalado.\n\n"
                "Opções:\n"
                "  1. Instalar: pip install flash-attn --no-build-isolation\n"
                "  2. Desativar no YAML: flash_attention_2: false\n"
                "     (usará SDPA como fallback, mais lento e mais memória)\n"
                + "=" * 70
            )

        if use_liger and not _LIGER_DISPONIVEL:
            raise RuntimeError(
                "\n" + "=" * 70 + "\n"
                "❌ Liger Kernel está ATIVADO no YAML (treinamento.liger_kernel: true)\n"
                "   mas o pacote 'liger-kernel' não está instalado.\n\n"
                "   O Liger Kernel reduz o pico de VRAM em ~40% ao usar fused\n"
                "   cross-entropy (evita materializar o tensor de logits completo).\n\n"
                "Opções:\n"
                "  1. Instalar: pip install liger-kernel\n"
                "  2. Desativar no YAML: liger_kernel: false\n"
                "     (pode causar OOM em modelos grandes ou sequências longas)\n"
                + "=" * 70
            )

        attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"

        # Log das otimizações ativas
        print_cores(f"<cinza>   🛠️  Otimizações de memória GPU:</cinza>", color_auto=False)
        print_cores(f"<cinza>      - Flash Attention 2: {'✅ ativo' if use_flash_attn else '❌ desativado (usando SDPA)'}</cinza>", color_auto=False)
        print_cores(f"<cinza>      - Liger Kernel:      {'✅ ativo (fused CE + RoPE + RMSNorm)' if use_liger else '❌ desativado'}</cinza>", color_auto=False)

        # Configuração de quantização
        nbits = self._yaml_config.treinamento.nbits
        quant_config = QuantizationConfig(
            nbits=nbits,
            compute_dtype="bfloat16",  # Melhor performance em GPUs modernas
            quant_type="nf4",
            use_double_quant=True,
        )

        max_seq_length = self._yaml_config.treinamento.max_seq_length
        lora_model_path = self._yaml_config.modelo.saida
        base_model_name = self._yaml_config.modelo.base

        lora_ok = False
        full_ok = False
        model = None
        tokenizer = None

        # Detecta tipo de modelo na pasta de saída
        from treinar_unsloth_actions import _detectar_tipo_modelo_saida
        tipo_saida = _detectar_tipo_modelo_saida(lora_model_path)

        # Tentativa 1: Carregar modelo LoRA já treinado (para retomada de treinamento)
        if not self.force_base and tipo_saida == 'lora':
            try:
                model, tokenizer = ModelLoader.load_lora_model(
                    base_model_name=base_model_name,
                    lora_model_path=lora_model_path,
                    max_seq_length=max_seq_length,
                    quant_config=quant_config,
                    device_map="auto",  # Agora funciona sem problemas!
                    attn_implementation=attn_impl,
                    use_liger_kernel=use_liger,
                )
                lora_ok = True
            except Exception as e:
                print_cores(f'<vermelho>❌ Erro ao carregar modelo LoRA treinado: {e}</vermelho>', color_auto=False)
                traceback.print_exc()
                print_cores('<amarelo>Tentando carregar modelo base e aplicar LoRA...</amarelo>', color_auto=False)
                time.sleep(2)

        # Tentativa 1b: Carregar modelo FULL fine-tuned (sem LoRA) da saída
        elif not self.force_base and tipo_saida == 'full':
            print_cores(f'<azul>📦 Modelo FULL fine-tuned detectado em: {lora_model_path}</azul>', color_auto=False)
            try:
                model, tokenizer = ModelLoader.load_base_model(
                    model_name=lora_model_path,
                    max_seq_length=max_seq_length,
                    quant_config=quant_config,
                    device_map="auto",
                    attn_implementation=attn_impl,
                    use_liger_kernel=use_liger,
                )
                full_ok = True
                print_cores(f'<verde>✅ Modelo FULL carregado de: {lora_model_path}</verde>', color_auto=False)
            except Exception as e:
                print_cores(f'<vermelho>❌ Erro ao carregar modelo FULL: {e}</vermelho>', color_auto=False)
                traceback.print_exc()
                print_cores('<amarelo>Tentando carregar modelo base...</amarelo>', color_auto=False)
                time.sleep(2)

        elif self.force_base:
            print_cores(f'<cinza>ℹ️  Opção --base ativada: Ignorando busca por modelo LoRA treinado.</cinza>', color_auto=False)

        # Tentativa 2: Carregar modelo base (novo treinamento)
        if not lora_ok and not full_ok:
            model, tokenizer = ModelLoader.load_base_model(
                model_name=base_model_name,
                max_seq_length=max_seq_length,
                quant_config=quant_config,
                device_map="auto",  # Múltiplas GPUs agora suportadas!
                attn_implementation=attn_impl,
                use_liger_kernel=use_liger,
            )

            if self.force_base:
                print_cores(f'<cinza>ℹ️  Opção --base ativada: Não aplicando adaptadores LoRA.</cinza>', color_auto=False)
            else:
                # LoRA será aplicado em _aplicar_etapa_curriculum quando a primeira
                # etapa LoRA iniciar. Isso permite que etapas "full" treinem o modelo
                # base diretamente, sem overhead de adaptadores LoRA.
                tipos_etapas = set(e.tipo for e in self._etapas if e.tipo)
                if "lora" in tipos_etapas and "full" in tipos_etapas:
                    print_cores(
                        f'<cinza>ℹ️  Curriculum misto (full + LoRA): modelo base carregado sem LoRA.</cinza>',
                        color_auto=False
                    )
                    print_cores(
                        f'<cinza>   LoRA será aplicado automaticamente na primeira etapa que o requeira.</cinza>',
                        color_auto=False
                    )
                elif "lora" in tipos_etapas:
                    print_cores(
                        f'<cinza>ℹ️  LoRA será aplicado no início do treinamento.</cinza>',
                        color_auto=False
                    )

        # Rastreia se LoRA já está aplicado ao modelo
        # True apenas se carregou modelo LoRA existente (retomada)
        self._lora_applied = lora_ok

        # Imprime informações do modelo
        ModelLoader.print_model_info(model, tokenizer)

        return model, tokenizer


    # ------------------------- dados -------------------------------------

    @classmethod
    def debug_info(cls, cfg_path: str, yaml_config=None):
        """Exibe informações detalhadas de debug sobre configuração e datasets."""
        print("="*80)
        print(">> MODO INFO / DEBUG - INFORMAÇÕES DE CONFIGURAÇÃO E DATASET")
        print("="*80)
        
        # Carrega configuração usando YamlTreinamento (se não recebeu uma instância pronta)
        if yaml_config is None:
            try:
                yaml_config = YamlTreinamento(cfg_path, validar_caminhos=False)
            except Exception as e:
                print(f"\n❌ Erro ao carregar YAML: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # Mostra informações do YamlTreinamento
        print(f"\n{yaml_config.info()}")
        
        # configuração estruturada
        print("\n📋 CONFIGURAÇÃO ESTRUTURADA:")
        config_dict = {
             "modelo": dataclasses.asdict(yaml_config.modelo),
             "treinamento": dataclasses.asdict(yaml_config.treinamento),
             "lora": dataclasses.asdict(yaml_config.lora),
             "curriculum": dataclasses.asdict(yaml_config.curriculum_config),
        }
        print(json.dumps(config_dict, indent=2, ensure_ascii=False, default=str))

        # carrega o tokenizer para chat_template
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(yaml_config.modelo.base, use_fast=True)
            print(f"\n✅ Tokenizer carregado com sucesso")
        except ImportError:
            print(f"\n❌ Transformers não disponível")
            return
        except Exception as e:
            print(f"\n❌ Erro ao carregar tokenizer: {e}")
            return
        
        # informações do modelo
        template_type = LLMsDataset.template_com_type(tokenizer)
        print(f"\n🤖 MODELO:")
        print(f"  - Nome: {yaml_config.modelo.base}")
        print(f"  - LoRA r: {yaml_config.lora.r}")
        print(f"  - Max seq length: {yaml_config.treinamento.max_seq_length}")
        print(f"  - Template com type: {template_type}")
        
        # Verifica se existe modelo LoRA treinado
        lora_model_path = yaml_config.modelo.saida
        arq_lora = os.path.join(lora_model_path, 'adapter_config.json')
        arq_model = os.path.join(lora_model_path, 'adapter_model.safetensors')
        pytorch_model = os.path.join(lora_model_path, 'pytorch_model.bin')
        
        print(f"\n🔧 VERIFICAÇÃO DE MODELO TREINADO:")
        print(f"  - Pasta do modelo: {lora_model_path}")
        print(f"  - adapter_config.json existe: {os.path.exists(arq_lora)}")
        print(f"  - adapter_model.safetensors existe: {os.path.exists(arq_model)}")
        print(f"  - pytorch_model.bin existe: {os.path.exists(pytorch_model)}")
        
        if os.path.exists(arq_lora):
            try:
                with open(arq_lora, 'r') as f:
                    lora_config = json.load(f)
                print(f"  - Configuração LoRA: r={lora_config.get('r', 'N/A')}, alpha={lora_config.get('lora_alpha', 'N/A')}")
            except:
                print(f"  - Erro ao ler configuração LoRA")
        
        is_trained_lora = (os.path.exists(arq_lora) and 
                          (os.path.exists(arq_model) or os.path.exists(pytorch_model)))
        print(f"  - Modelo LoRA completo detectado: {is_trained_lora}")
        
        if is_trained_lora:
            print(f"  ✅ O modelo será carregado com LoRA treinado")
        elif yaml_config.lora.r not in (0, None, False):
            print(f"  🔄 Será aplicado novo LoRA ao modelo base")
        else:
            print(f"  📄 Será usado modelo base sem LoRA")
        
        # Mostra arquivos pareados do curriculum
        print(f"\n📁 CURRICULUM - ARQUIVOS PAREADOS:")
        try:
            pares = yaml_config.dataset_manager.parear_arquivos()
            print(f"  - Total de pares: {len(pares)}")
            if pares:
                print(f"  - Primeiros 3 pares:")
                for par in pares[:3]:
                    print(f"    * {par.get('id', 'N/A')}")
            
            # Carrega divisão se existir
            divisao = yaml_config.dataset_manager.carregar_ou_criar_divisao()
            if not divisao.empty:
                contagem = divisao['alvo'].value_counts()
                total = len(divisao)
                print(f"\n  📊 Divisão de dados (total = {total}):")
                for alvo, qtd in contagem.items():
                    print(f"    - {alvo}: {qtd}")
            
            # Testa carregamento de mensagens
            print(f"\n  🔄 Carregando amostras de mensagens...")
            
            msgs_treino = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo="treino")
            print(f"    - Mensagens de treino: {len(msgs_treino)}")
            yaml_config.dataset_manager.mostrar_exemplo("Amostra Treino", msgs_treino)

            # Mostra também teste e validação se existirem
            if not yaml_config.dataset_manager.carregar_ou_criar_divisao().empty:
                msgs_teste = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo="teste")
                if msgs_teste:
                    print(f"    - Mensagens de teste: {len(msgs_teste)}")
                    yaml_config.dataset_manager.mostrar_exemplo("Amostra Teste", msgs_teste)
                
                msgs_val = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo="validacao")

                if msgs_val:
                    print(f"    - Mensagens de validação: {len(msgs_val)}")
                    yaml_config.dataset_manager.mostrar_exemplo("Amostra Validação", msgs_val)
                

        except Exception as e:
            print(f"  ❌ Erro ao processar dados: {e}")
            import traceback
            traceback.print_exc()
        
        # informações de checkpoints
        print(f"\n💾 CHECKPOINT INFO:")
        checkpoint_dir = os.path.join(yaml_config.modelo.saida, "chkpt")
        resume_enabled = yaml_config.treinamento.resume_from_checkpoint
        print(f"  - Resume from checkpoint: {resume_enabled}")
        print(f"  - Checkpoint directory: {checkpoint_dir}")
        
        if os.path.exists(checkpoint_dir):
            checkpoints = []
            for item in os.listdir(checkpoint_dir):
                item_path = os.path.join(checkpoint_dir, item)
                if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                    try:
                        step_num = int(item.split("-")[1])
                        checkpoints.append((step_num, item))
                    except (IndexError, ValueError):
                        continue
            
            if checkpoints:
                checkpoints.sort(key=lambda x: x[0])
                print(f"  - Checkpoints encontrados: {len(checkpoints)}")
                for step, name in checkpoints[-3:]:  # mostra os 3 mais recentes
                    print(f"    * {name} (step {step})")
            else:
                print(f"  - Nenhum checkpoint encontrado")
        else:
            print(f"  - Diretório de checkpoints não existe")
        
        # informações de hardware (RAM e GPU)
        print(f"\n🎮 HARDWARE INFO:")
        try:
            hardware = _print_mem("DEBUG")
        except Exception as e:
            print(f"  ❌ Erro ao obter info de hardware: {e}")
            hardware = {}
        
        gerador = GeradorRelatorio(yaml_config)
        gerador.gerar_relatorio(
            dataset_stats=None, 
            train_stats=None,
            hardware_info=hardware,
            print_only=True
        )
        
        print("\n" + "="*80)
        print("✅ INFO / DEBUG COMPLETO - CONFIGURAÇÃO E DATASETS VALIDADOS")
        print("="*80)


    # ------------------------- trainer -----------------------------------
    def _build_trainer(self, etapa_index: int = 0, etapa_alias: str = "Principal",
                       instancias_previas: int = 0, step_offset: int = 0,
                       epoch_offset: float = 0.0, tokens_previos: int = 0,
                       is_retomada: bool = False,
                       etapa=None) -> SFTTrainer:
        """Constrói o SFTTrainer com callbacks de métricas.
        
        Args:
            etapa_index: Índice da etapa do curriculum (0 para treino simples)
            etapa_alias: Nome da etapa do curriculum ("Principal" para treino simples)
            instancias_previas: Instâncias acumuladas de etapas anteriores
            step_offset: Steps acumulados de etapas anteriores (para step_global contínuo)
            epoch_offset: Épocas acumuladas de etapas anteriores (para epoch_global contínuo)
            tokens_previos: Tokens processados em etapas anteriores (para tokens_acumulados contínuo)
            is_retomada: Se True, preserva arquivos de métricas existentes (retomada de treino interrompido)
        """
        print_cores("<azul>[3/6] Configurando trainer…</azul>", color_auto=False)
        
        # === Formatação do Dataset (garante coluna 'text') ===
        # num_proc não deve exceder o número de registros (causa falha silenciosa)
        import os
        n_proc = max(1, (os.cpu_count() or 2) // 2)
        n_proc_train = min(n_proc, len(self.train_ds)) if len(self.train_ds) > 0 else 1

        if "text" not in self.train_ds.column_names:
            self.train_ds = self.chat_handler.formatar_dataset_coluna_text(self.train_ds, num_proc=n_proc_train)
            
        if self.eval_ds and "text" not in self.eval_ds.column_names:
            n_proc_eval = min(n_proc, len(self.eval_ds)) if len(self.eval_ds) > 0 else 1
            self.eval_ds = self.chat_handler.formatar_dataset_coluna_text(self.eval_ds, num_proc=n_proc_eval)
        
        # Validação: interrompe se a formatação falhou (evita StopIteration em SFTTrainer)
        if len(self.train_ds) == 0:
            raise ValueError("Dataset de treino está vazio. Verifique o arquivo de divisão e as pastas de dados.")
        if "text" not in self.train_ds.column_names:
            raise ValueError(
                f"Coluna 'text' ausente após formatação. "
                f"Colunas disponíveis: {self.train_ds.column_names}. "
                f"Verifique se o dataset contém 'messages' ou 'prompt'/'completion'."
            )
            
        # Verifica a integridade da formatação para DEBUG
        self.chat_handler.verificar_dataset_formatado(self.train_ds)
        # =====================================================

        total_examples = len(self.train_ds)
        # cfg = self.cfg (removido)
        
        treino_cfg = self._yaml_config.treinamento
        
        eval_steps = treino_cfg.eval_steps
        n_gpus = max(torch.cuda.device_count(), 1)
        
        # percentual do dataset
        if self.eval_ds and isinstance(eval_steps, str) and eval_steps.endswith('%'):
            try:
                eval_steps_val = int(eval_steps.replace('%', '').strip())
                if eval_steps_val >= 1:
                   _st = treino_cfg.grad_batch_size * treino_cfg.batch_size * n_gpus
                   eval_steps = int((eval_steps_val/100) * (total_examples / _st))
                else:
                   eval_steps = None
            except:
                eval_steps = None
        
        if eval_steps is None or not self.eval_ds:
            eval_steps = 0  
            
        if eval_steps == 0 and self.eval_ds:
             # Fallback cálculo automático se não definido
             _st = treino_cfg.grad_batch_size * treino_cfg.batch_size * n_gpus
             eval_steps = max(1, int((total_examples / 100) / _st))

        if self.eval_ds and eval_steps > 0:
           print_cores(f'<cinza> - avaliando a cada {eval_steps} steps...</cinza>', color_auto=False)
        
        log_steps = eval_steps if isinstance(eval_steps, int) and eval_steps > 0 else 50
        
        if self.save_checkpoints:
            print_cores(f'<cinza> - gravando checkpoints a cada {log_steps} steps</cinza>', color_auto=False)
        
        # Log train_on_responses_only
        if treino_cfg.train_on_responses_only:
            print_cores(f'<cinza> - train_on_responses_only ATIVADO (treina apenas nas respostas do assistant)</cinza>', color_auto=False)

        # Configuração de argumentos de treino
        # Nota: Usamos TrainingArguments padrão ou SFTConfig se disponível no unsloth
        # Para garantir compatibilidade, usamos TrainingArguments que é base
        from trl import SFTConfig
        
        args = SFTConfig(
            per_device_train_batch_size=treino_cfg.batch_size,
            gradient_accumulation_steps=treino_cfg.grad_batch_size,
            warmup_steps=treino_cfg.warmup_steps,
            num_train_epochs=treino_cfg.epochs,
            learning_rate=treino_cfg.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=log_steps,  # Alinhado com eval_steps para sincronizar métricas no gráfico
            optim=treino_cfg.optim,
            weight_decay=treino_cfg.weight_decay,
            lr_scheduler_type=treino_cfg.lr_scheduler_type,
            seed=treino_cfg.seed,
            output_dir=os.path.join(self._yaml_config.modelo.saida, "chkpt"),  # checkpoints em subpasta
            save_strategy="steps" if self.save_checkpoints else "no",
            save_steps=log_steps if self.save_checkpoints else 0,
            eval_strategy="steps" if self.eval_ds and eval_steps > 0 else "no",
            eval_steps=eval_steps if self.eval_ds and eval_steps > 0 else None,
            load_best_model_at_end=True if self.eval_ds and eval_steps > 0 else False,
            report_to="none",
            gradient_checkpointing=True,  # Ativa gradient checkpointing (era "unsloth" antes)
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Evita stream mismatch
            remove_unused_columns=False,
            dataloader_drop_last=False,
            dataset_text_field="text", # usamos a coluna 'text' formatada
            dataset_num_proc=2,
            packing=False,
            per_device_eval_batch_size=1,     # Força batch 1 na validação para economizar VRAM
            eval_accumulation_steps=1,        # Descarrega logits da GPU para CPU a cada passo
            max_length=treino_cfg.max_seq_length,  # max_length substitui max_seq_length no TRL >= 0.12.0
            use_liger_kernel=treino_cfg.liger_kernel and _LIGER_DISPONIVEL,  # Informa ao TRL para NÃO acessar outputs.logits (Liger fused CE retorna logits=None)
        )

        # Monta eval_dataset para o SFTTrainer.
        # Se houver eval_global, passa como dict {"current": eval_ds, "global": eval_ds_global}
        # para que ambos sejam tokenizados por _prepare_dataset na construção do trainer
        # (mesmo mecanismo, mesmo num_proc, sem risco de CUDA fork pós-inicialização).
        #
        _n_eval = len(self.eval_ds) if self.eval_ds is not None else 0
        _n_global = len(self.eval_ds_global) if self.eval_ds_global is not None else 0

        _tem_eval_global = (
            self.eval_ds_global is not None
            and _n_global > 0
        )
        if _tem_eval_global:
            _eval_dataset_arg = {
                "current": self.eval_ds,
                "global": self.eval_ds_global,
            } if self.eval_ds is not None and _n_eval > 0 else {
                "global": self.eval_ds_global,
            }
        else:
            _eval_dataset_arg = self.eval_ds

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,  # processing_class substitui tokenizer no TRL >= 0.12.0
            train_dataset=self.train_ds,
            eval_dataset=_eval_dataset_arg,
            args=args,  # max_length agora está configurado no SFTConfig (TRL >= 0.12.0)
        )
        
        # Aplica train_on_responses_only se configurado
        # O Unsloth SFTTrainer tokeniza o dataset (text → input_ids/attention_mask)
        # e define DataCollatorForLanguageModeling (sem labels).
        # train_on_responses_only mapeia o dataset adicionando labels com -100 nas
        # posições do prompt e tokens reais nas respostas do assistant.
        if treino_cfg.train_on_responses_only:
            dataset_ja_tem_labels = "labels" in (trainer.train_dataset.column_names if trainer.train_dataset else [])
            if dataset_ja_tem_labels:
                # Verifica se os labels existentes já mascaram o prompt corretamente
                amostra = trainer.train_dataset[0]
                labels_amostra = amostra["labels"]
                n_validos = sum(1 for lb in labels_amostra if lb != -100)
                n_total = len(labels_amostra)
                if n_validos > 0 and n_validos < n_total:
                    # Labels já possuem mascaramento parcial (prompt=-100, resposta=válida)
                    print_cores(f'   <verde>✅ train_on_responses_only: dataset já possui labels pré-mascarados</verde>', color_auto=False)
                    print(f'      ({n_validos}/{n_total} tokens com loss, {n_total - n_validos} mascarados)')
                    print(f'      Pulando train_on_responses_only para preservar labels corretos.')
                else:
                    # Labels existem mas sem mascaramento adequado — aplica normalmente
                    trainer = self.chat_handler.aplicar_train_on_responses_only(trainer)
            else:
                trainer = self.chat_handler.aplicar_train_on_responses_only(trainer)
            
            # Verificação pós-aplicação: garante que ao menos alguns labels sejam válidos
            self._verificar_labels_dataset(trainer)
        
        # Remove colunas string residuais dos datasets internos do trainer.
        # O SFTTrainer tokeniza 'text' → input_ids/labels, mas mantém a coluna original.
        # Com remove_unused_columns=False, o collator recebe 'text' (string) e falha.
        _str_cols = {"text", "id", "messages", "prompt", "completion"}
        if trainer.train_dataset is not None:
            for col in _str_cols & set(trainer.train_dataset.column_names):
                trainer.train_dataset = trainer.train_dataset.remove_columns(col)

        # Se eval_dataset é dict (com "global"), extrai e separa antes de limpar.
        _global_ds_tokenized = None
        if isinstance(trainer.eval_dataset, dict):
            _global_ds_tokenized = trainer.eval_dataset.get("global")
            trainer.eval_dataset = trainer.eval_dataset.get("current")  # restaura dataset simples
            if _global_ds_tokenized is not None:
                for col in _str_cols & set(_global_ds_tokenized.column_names):
                    _global_ds_tokenized = _global_ds_tokenized.remove_columns(col)

        if trainer.eval_dataset is not None:
            for col in _str_cols & set(trainer.eval_dataset.column_names):
                trainer.eval_dataset = trainer.eval_dataset.remove_columns(col)
        
        # Configura diretório de saída
        output_dir = self._yaml_config.modelo.saida
        os.makedirs(output_dir, exist_ok=True)
        
        # === CALLBACKS DE MÉTRICAS ===
        
        # 1. JsonLogger (métricas brutas em metrics_stream.jsonl)
        jsonl = os.path.join(output_dir, "metrics_stream.jsonl")
        if not is_retomada and os.path.isfile(jsonl):
            os.remove(jsonl)
        trainer.add_callback(JsonLoggerCallback(jsonl, truncar=not is_retomada))
        
        # 2. MetricsLoggerCallback (métricas unificadas: loss, hardware, curriculum, tokens)
        # Calcula média de tokens reais por instância do dataset tokenizado
        _total_tokens_dataset = sum(len(r.get('input_ids', [])) for r in trainer.train_dataset)
        _num_instancias = len(trainer.train_dataset)
        _media_tokens = _total_tokens_dataset / _num_instancias if _num_instancias > 0 else 0
        self._media_tokens_por_instancia = _media_tokens  # Armazena para uso em train()
        
        trainer.add_callback(MetricsLoggerCallback(
            output_dir,
            etapa_alias=etapa_alias,
            etapa_index=etapa_index,
            instancias_previas=instancias_previas,
            step_offset=step_offset,
            epoch_offset=epoch_offset,
            media_tokens_por_instancia=_media_tokens,
            tokens_previos=tokens_previos,
            retomada=is_retomada,
        ))
        
        # 3. CheckpointRenameCallback (renomeia checkpoints com zero-padding)
        if self.save_checkpoints:
            chkpt_dir = os.path.join(self._yaml_config.modelo.saida, "chkpt")
            os.makedirs(chkpt_dir, exist_ok=True)
            trainer.add_callback(CheckpointRenameCallback(chkpt_dir, step_offset=step_offset))
        
        # 4. GlobalEvalCallback (avaliação em todas as etapas do curriculum)
        # O dataset global já foi tokenizado pelo SFTTrainer junto com eval_ds
        # (via dict eval_dataset na construção) e extraído em _global_ds_tokenized acima.
        self._global_eval_callback = None
        if _global_ds_tokenized is not None and len(_global_ds_tokenized) > 0:
            metrics_file = os.path.join(output_dir, "treinamento", "training_metrics.jsonl")
            self._global_eval_callback = GlobalEvalCallback(
                global_eval_dataset=_global_ds_tokenized,
                metrics_file=metrics_file,
                step_offset=step_offset,
                epoch_offset=epoch_offset,
                etapa_alias=etapa_alias,
                etapa_index=etapa_index,
            )
            trainer.add_callback(self._global_eval_callback)
        
        print_cores(f'<cinza> - callbacks de métricas configurados:</cinza>', color_auto=False)
        print_cores(f'   <cinza>• metrics_stream.jsonl (métricas brutas)</cinza>', color_auto=False)
        print_cores(f'   <cinza>• treinamento/training_metrics.jsonl (loss, hardware, tokens)</cinza>', color_auto=False)
        print_cores(f'   <cinza>• tokens reais: {_total_tokens_dataset:,} total, média {_media_tokens:.0f}/instância ({_num_instancias} instâncias)</cinza>', color_auto=False)
        if self.save_checkpoints:
            print_cores(f'   <cinza>• checkpoint renaming (zero-padding: checkpoint-00001)</cinza>', color_auto=False)
        if self._global_eval_callback is not None:
            print_cores(f'   <cinza>• eval_loss_global (validação de todas as etapas combinadas)</cinza>', color_auto=False)
        
        # 5. PaceLossCallback (early stopping por loss com mínimo de épocas)
        if etapa is not None and etapa.pace_loss > 0:
            pace_cb = PaceLossCallback(
                pace_loss=etapa.pace_loss,
                pace_epochs=etapa.pace_epochs,
                pace_epochs_max=etapa.pace_epochs_max,
                etapa_alias=etapa.alias,
            )
            trainer.add_callback(pace_cb)
            self._pace_loss_callback = pace_cb
            print_cores(f'   <cinza>• pace_loss={etapa.pace_loss} (mín {etapa.pace_epochs} épocas'
                        f'{f", máx {etapa.pace_epochs_max} épocas" if etapa.pace_epochs_max > 0 else ""})</cinza>', color_auto=False)
        else:
            self._pace_loss_callback = None
        
        trainer.model.config.use_cache = False
        
        return trainer

    def _verificar_labels_dataset(self, trainer) -> None:
        """Verifica se o dataset do trainer possui labels válidos após train_on_responses_only.
        
        Confere uma amostra de exemplos para garantir que ao menos alguns tokens
        tenham labels != -100 (necessário para computar loss não-NaN).
        """
        ds = trainer.train_dataset
        if ds is None or "labels" not in ds.column_names:
            print_cores(f'   <cinza>ℹ️  Sem coluna labels no dataset — collator criará labels em tempo de execução.</cinza>', color_auto=False)
            return
        
        n_check = min(10, len(ds))
        total_validos = 0
        total_tokens = 0
        exemplos_vazios = 0
        
        for i in range(n_check):
            labels = ds[i]["labels"]
            if isinstance(labels, list):
                n_val = sum(1 for lb in labels if lb != -100)
                n_tok = len(labels)
            else:
                # tensor
                n_val = int((labels != -100).sum().item())
                n_tok = len(labels)
            total_validos += n_val
            total_tokens += n_tok
            if n_val == 0:
                exemplos_vazios += 1
        
        if total_validos == 0:
            print_cores(f'   <vermelho>❌ ALERTA: Todos os {n_check} exemplos verificados têm labels inteiramente -100!</vermelho>', color_auto=False)
            print_cores(f'      <vermelho>Isso causará loss=NaN e grad_norm=0.0 durante o treinamento.</vermelho>', color_auto=False)
            print(f'      Possíveis causas:')
            print(f'      • Os marcadores de resposta não foram encontrados nos input_ids')
            print(f'      • O chat template não corresponde ao formato esperado pelo modelo')
            # Diagnóstico: mostra o primeiro exemplo para depuração
            if len(ds) > 0 and "input_ids" in ds.column_names:
                ids = ds[0]["input_ids"]
                if hasattr(ids, 'tolist'):
                    ids = ids.tolist()
                print(f'      Diagnóstico — input_ids (primeiros 30 tokens): {ids[:30]}')
                try:
                    texto_decodificado = self.tokenizer.decode(ids[:50], skip_special_tokens=False)
                    print(f'      Diagnóstico — texto decodificado: {texto_decodificado[:200]!r}')
                except Exception:
                    pass
        elif exemplos_vazios > 0:
            pct = exemplos_vazios / n_check * 100
            print_cores(f'   <amarelo>⚠️ {exemplos_vazios}/{n_check} exemplos com labels inteiramente -100 ({pct:.0f}%)</amarelo>', color_auto=False)
            print(f'      Total: {total_validos}/{total_tokens} tokens com loss')
        else:
            pct_resp = total_validos / total_tokens * 100 if total_tokens > 0 else 0
            print_cores(f'   <verde>✅ Labels verificados: {total_validos}/{total_tokens} tokens com loss ({pct_resp:.1f}%)</verde>', color_auto=False)

    # ------------------------- checkpoint management --------------------- 
    def _limpar_metricas_etapa_interrompida(self, etapa_retomada: int) -> None:
        """Remove métricas parciais da etapa interrompida, preservando etapas concluídas.
        
        Ao retomar um treinamento interrompido na etapa N, o arquivo
        training_metrics.jsonl pode conter registros parciais da etapa N
        (antes do crash). Este método remove esses registros para que
        a etapa seja re-registrada de forma limpa, gerando um gráfico
        contínuo e consistente.
        
        Args:
            etapa_retomada: Índice da etapa a retomar (registros com etapa_index >= este valor são removidos)
        """
        metrics_file = os.path.join(self._yaml_config.modelo.saida, "treinamento", "training_metrics.jsonl")
        if not os.path.isfile(metrics_file):
            return
        
        try:
            linhas_mantidas = []
            linhas_removidas = 0
            with open(metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        reg = json.loads(line)
                        idx = reg.get("etapa_index", 0)
                        if idx < etapa_retomada:
                            linhas_mantidas.append(line + "\n")
                        else:
                            linhas_removidas += 1
                    except (json.JSONDecodeError, KeyError):
                        # Mantém linhas não-parseáveis para não perder dados
                        linhas_mantidas.append(line + "\n")
            
            if linhas_removidas > 0:
                with open(metrics_file, "w", encoding="utf-8") as f:
                    f.writelines(linhas_mantidas)
                logger.info(
                    f"<cinza>   🧹 Métricas da etapa interrompida removidas: "
                    f"{linhas_removidas} registros (etapa_index >= {etapa_retomada}), "
                    f"{len(linhas_mantidas)} registros preservados</cinza>"
                )
        except Exception as e:
            logger.warning(f"⚠️  Erro ao limpar métricas da etapa interrompida: {e}")
    def _find_latest_checkpoint(self) -> str:
        """Encontra o checkpoint mais recente na pasta de checkpoints.
        
        Returns:
            str: Caminho para o checkpoint mais recente ou None se não houver
        """
        # verifica se o resume está habilitado na configuração
        if not self._yaml_config.treinamento.resume_from_checkpoint:
            print_cores("<amarelo>⚠️ Checkpoint ignorado por configuração (resume_from_checkpoint=False)</amarelo>", color_auto=False)
            return None
            
        if not self.save_checkpoints:
            return None
            
        checkpoint_dir = os.path.join(self._yaml_config.modelo.saida, "chkpt")
        if not os.path.exists(checkpoint_dir):
            return None
        
        # procura por pastas checkpoint-* 
        checkpoints = []
        for item in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                try:
                    step_num = int(item.split("-")[1])
                    checkpoints.append((step_num, item_path))
                except (IndexError, ValueError):
                    continue
        
        if not checkpoints:
            return None
            
        # retorna o checkpoint com maior número de step
        latest_step, latest_path = max(checkpoints, key=lambda x: x[0])
        
        # verifica se é um checkpoint válido (contém os arquivos necessários)
        required_files = ["training_args.bin", "trainer_state.json"]
        alternative_files = ["model.safetensors"]  # formato alternativo
        
        has_required = all(os.path.exists(os.path.join(latest_path, f)) for f in required_files)
        has_alternative = any(os.path.exists(os.path.join(latest_path, f)) for f in alternative_files)
        
        if has_required or (has_alternative and os.path.exists(os.path.join(latest_path, "trainer_state.json"))):
            print_cores(f"<verde>✅ Checkpoint encontrado: {latest_path} (step {latest_step})</verde>", color_auto=False)
            self._historico.evento_checkpoint_encontrado(latest_path, latest_step)
            return latest_path
        else:
            print(f"⚠️  Checkpoint incompleto encontrado: {latest_path}")
            return None

    # ------------------------- curriculum: preparação por etapa -----------
    def _aplicar_etapa_curriculum(self, step_index: int, etapa) -> None:
        """Configura parâmetros e dados para uma etapa específica do curriculum.
        
        - Aplica pace_epochs, learning_rate e max_seq_length da etapa
        - Alterna entre modo "full" (todos os parâmetros) e "lora" (só adaptadores)
        - Aplica adaptadores LoRA sob demanda na primeira etapa LoRA
        - Para etapas > 0, troca o arquivo de divisão e recarrega os datasets
        - Se max_seq_length mudar entre etapas, recarrega model/tokenizer
        """
        treino = self._yaml_config.treinamento
        msl_anterior = treino.max_seq_length

        # === Alterna modo de treinamento conforme tipo da etapa ===
        if etapa.tipo == "full":
            # Full fine-tuning: desbloqueia todos os parâmetros float (base + LoRA se presente)
            # Parâmetros quantizados (int8/int4 via bitsandbytes) não suportam gradientes
            for param in self.model.parameters():
                if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    param.requires_grad = True
            n_total = sum(p.numel() for p in self.model.parameters())
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if self._lora_applied:
                logger.info(f"🔓 Modo FULL: {n_train:,}/{n_total:,} parâmetros desbloqueados (base + LoRA, quantizados permanecem congelados)")
            else:
                logger.info(f"🔓 Modo FULL: {n_train:,}/{n_total:,} parâmetros desbloqueados para treinamento")
        elif etapa.tipo == "lora":
            # Se LoRA ainda não foi aplicado, aplica agora
            if not self._lora_applied:
                lora_cfg = self._yaml_config.lora
                logger.info(f"🔄 Aplicando adaptadores LoRA (r={lora_cfg.r}, alpha={lora_cfg.alpha}) para etapa '{etapa.alias}'...")
                self.model = ModelLoader.apply_lora(
                    model=self.model,
                    r=lora_cfg.r,
                    lora_alpha=lora_cfg.alpha,
                    lora_dropout=lora_cfg.dropout,
                    target_modules=lora_cfg.target_modules,
                    bias="none",
                )
                self._lora_applied = True
                logger.info(f"✅ LoRA aplicado com sucesso")

            # LoRA fine-tuning: congela base, treina apenas adaptadores LoRA
            for name, param in self.model.named_parameters():
                if "lora_" in name or "modules_to_save" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            n_total = sum(p.numel() for p in self.model.parameters())
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"🔒 Modo LoRA: {n_train:,}/{n_total:,} parâmetros treináveis (apenas adaptadores)")

        # Override de epochs com pace_epochs / pace_epochs_max da etapa
        # Lógica:
        #   - pace_epochs: mínimo de épocas (pace_loss não interrompe antes disso)
        #   - pace_epochs_max: máximo de épocas (força parada se pace_loss não for atingido)
        #   - pace_loss: loss alvo. Após pace_epochs mínimo, se loss < pace_loss → para.
        #
        # Se pace_loss está configurado com pace_epochs_max:
        #   num_train_epochs = pace_epochs_max (treina até o máximo, callback para antes se pace_loss atingido)
        # Se pace_loss está configurado sem pace_epochs_max:
        #   num_train_epochs = pace_epochs (sem teto extra, apenas avalia loss ao final de cada época)
        # Se pace_loss não está configurado:
        #   num_train_epochs = pace_epochs (comportamento original)
        if etapa.pace_epochs > 0:
            if etapa.pace_loss > 0 and etapa.pace_epochs_max > 0:
                # Treina até pace_epochs_max; callback PaceLoss para antes se loss < pace_loss (após pace_epochs mínimo)
                treino.epochs = etapa.pace_epochs_max
                logger.info(
                    f"⏱️  Etapa '{etapa.alias}': epochs={etapa.pace_epochs_max} (max), "
                    f"pace_epochs={etapa.pace_epochs} (mín), pace_loss={etapa.pace_loss}"
                )
            else:
                treino.epochs = etapa.pace_epochs
                if etapa.pace_loss > 0:
                    logger.info(
                        f"⏱️  Etapa '{etapa.alias}': epochs={etapa.pace_epochs}, pace_loss={etapa.pace_loss}"
                    )

        # Override de learning_rate se especificado
        if etapa.learning_rate > 0:
            treino.learning_rate = etapa.learning_rate

        # Override de batch_size se especificado
        if etapa.batch_size > 0:
            treino.batch_size = etapa.batch_size
            logger.info(f"📦 Etapa '{etapa.alias}': batch_size={etapa.batch_size} (override por etapa)")

        # Lê info de tokens da etapa (para auto-estimação e exibição de suficiência)
        info_tokens = self._yaml_config._ler_info_tokens_divisao(etapa.arquivo)

        # Override de max_seq_length: explícito > auto por etapa > global
        if etapa.max_seq_length > 0:
            treino.max_seq_length = etapa.max_seq_length
        elif getattr(treino, 'max_seq_length_auto', False) and info_tokens and info_tokens.get("max", 0) > 0:
            # Global foi auto-estimado → auto-estima por etapa a partir do CSV
            import math
            _est = int(math.ceil(info_tokens["max"] * 1.1 / 128) * 128)
            treino.max_seq_length = _est

        # Log da mudança de max_seq_length (afeta truncagem de dados no SFTConfig)
        # NÃO recarrega o modelo: ele usa max_position_embeddings nativo (ex: 32768)
        # e não precisa ser recarregado quando max_seq_length de treinamento muda.
        msl_atual = treino.max_seq_length
        if msl_atual != msl_anterior:
            _auto_tag = " (auto)" if etapa.max_seq_length == 0 and getattr(treino, 'max_seq_length_auto', False) else ""
            logger.info(f"🔄 max_seq_length mudou: {msl_anterior} → {msl_atual}{_auto_tag} "
                        f"(etapa '{etapa.alias}' - afeta truncagem de dados, não a arquitetura do modelo)")

        # Exibe informações de tokens da divisão da etapa
        if info_tokens:
            suficiente = "✅" if msl_atual >= info_tokens["max"] else "⚠️  INSUFICIENTE 🚩"
            print(f"   📊 Tokens etapa '{etapa.alias}': max={info_tokens['max']}, "
                  f"média={info_tokens['media']:.0f} → max_seq_length={msl_atual} {suficiente}")

        # Para etapas além da primeira, recarrega dados com o arquivo de divisão da etapa
        if step_index > 0 and etapa.arquivo:
            self._yaml_config.curriculum_config.divisao.arquivo = etapa.arquivo
            # Limpa cache de divisão para forçar releitura
            self._yaml_config.dataset_manager._dados_divisao = None

            self.train_ds = self._load_from_pastas(alvo="treino")
            self.eval_ds  = self._load_from_pastas(alvo="validacao")

            # Atualiza estatísticas do dataset
            ts = self._print_dataset_stats(self.train_ds, "Dataset de Treino")
            self._dataset_stats = {
                "treino_len":    len(self.train_ds),
                "validacao_len": len(self.eval_ds) if self.eval_ds else 0,
                "token_stats":   ts,
            }

    # ------------------------- execução ----------------------------------
    def train(self):
        antes = _print_mem("ANTES")
        is_curriculum = len(self._etapas) > 1
        total_etapas = len(self._etapas)

        # ---------------------------------------------------------
        # Controle de Conclusão e Retomada
        # ---------------------------------------------------------
        estado_pipeline = self._tracker.carregar_estado()
        # Usa pace_epochs_max como teto quando configurado (com pace_loss), senão pace_epochs
        def _epochs_efetivos(e):
            if e.pace_loss > 0 and e.pace_epochs_max > 0:
                return e.pace_epochs_max
            return e.pace_epochs if e.pace_epochs > 0 else self._yaml_config.treinamento.epochs
        target_epochs_yaml = sum([_epochs_efetivos(e) for e in self._etapas])
        target_epochs_salvo = estado_pipeline.get("target_epochs", -1.0)
        
        # Se concluído e o número de etapas/épocas no YAML não aumentou, evite recomeçar
        bloqueado_por_etapas = estado_pipeline.get("current_step", 0) >= total_etapas 
        
        # Só libera a continuação se o alvo exigido agora (yaml) for EXPLICITAMENTE maior do que o alvo
        # em que encerrou. Se não há alvo salvo de épocas (-1.0), assume bloqueado por precaução.
        bloqueado_por_epochs = (target_epochs_salvo == -1.0) or (target_epochs_salvo >= target_epochs_yaml)
        
        if self._tracker.is_concluido() and bloqueado_por_etapas and bloqueado_por_epochs:
            print_cores("\n<verde>✅ Treinamento já foi concluído e atingiu seu objetivo final (todas as etapas).</verde>", color_auto=False)
            print_cores("<amarelo>   Evitando continuação indevida de um modelo já finalizado.</amarelo>", color_auto=False)
            print_cores("<cinza>   ↳ Para continuar a partir daqui: adicione uma nova etapa no curriculum ou aumente as epochs.</cinza>", color_auto=False)
            print_cores("<cinza>   ↳ Para reiniciar integralmente: acione o script com a opção --reset.</cinza>\n", color_auto=False)
            return

        # ---------------------------------------------------------
        # Detecção de Retomada (Curriculum)
        # ---------------------------------------------------------
        # Usa curriculum_state.json para identificar etapas já concluídas
        # e restaurar contadores acumulados (steps, épocas, instâncias, tokens).
        # Isso permite pular etapas concluídas e retomar na etapa interrompida
        # com o checkpoint correto e dataset correto.
        etapa_retomada, acumulados_retomada = self._tracker.obter_estado_retomada(total_etapas)
        is_retomada = etapa_retomada > 0

        if is_retomada:
            print_cores(f"<azul>[4/6] Retomando treinamento a partir da etapa {etapa_retomada+1}/{total_etapas} "
                        f"('{self._etapas[etapa_retomada].alias}')…</azul>", color_auto=False)
            # Limpa métricas parciais da etapa interrompida (mantém etapas concluídas)
            self._limpar_metricas_etapa_interrompida(etapa_retomada)
        elif is_curriculum:
            print_cores(f"<azul>[4/6] Iniciando treinamento com {total_etapas} etapas de curriculum…</azul>", color_auto=False)
        else:
            print_cores("<azul>[4/6] Iniciando treinamento…</azul>", color_auto=False)

        # Contadores acumulados para métricas contínuas entre etapas do curriculum
        # Se retomando, restaura contadores das etapas já concluídas
        instancias_acumuladas = acumulados_retomada.get("instancias_acumuladas", 0)
        step_offset_global = acumulados_retomada.get("step_offset_global", 0)
        epoch_offset_global = acumulados_retomada.get("epoch_offset_global", 0.0)
        tokens_acumulados = acumulados_retomada.get("tokens_acumulados", 0)

        for step_index, etapa_atual in enumerate(self._etapas):
            # --- Pula etapas já concluídas na retomada ---
            if step_index < etapa_retomada:
                logger.info(f"<cinza>⏩ Etapa {step_index+1}/{total_etapas}: '{etapa_atual.alias}' — já concluída, pulando</cinza>")
                continue

            # --- Preparação da etapa (full/lora, hiperparâmetros, dados) ---
            # Sempre chamado: configura modo full/lora, aplica LoRA sob demanda,
            # ajusta epochs/lr/max_seq_length e recarrega dados se necessário.
            if is_curriculum:
                logger.info(f"<azul>🔄 Etapa {step_index+1}/{total_etapas}: '{etapa_atual.alias}'</azul>")
            self._aplicar_etapa_curriculum(step_index, etapa_atual)
            if is_curriculum:
                # Conta parâmetros treináveis após alternância full/lora
                n_treinaveis = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                n_totais = sum(p.numel() for p in self.model.parameters())
                self._historico.evento_etapa_curriculum(
                    step_index=step_index,
                    alias=etapa_atual.alias,
                    tipo=etapa_atual.tipo,
                    pace_epochs=etapa_atual.pace_epochs,
                    pace_epochs_max=etapa_atual.pace_epochs_max if etapa_atual.pace_epochs_max > 0 else "N/A",
                    pace_loss=etapa_atual.pace_loss if etapa_atual.pace_loss > 0 else "N/A",
                    max_seq_length=etapa_atual.max_seq_length if etapa_atual.max_seq_length > 0 else self._yaml_config.treinamento.max_seq_length,
                    parametros_treinaveis=f"{n_treinaveis:,}/{n_totais:,} ({n_treinaveis/n_totais*100:.2f}%)" if n_totais > 0 else "N/A",
                )

            # Identifica se esta é a primeira etapa efetiva (para checkpoint e retomada)
            is_primeira_etapa_efetiva = (step_index == etapa_retomada)
            is_retomada_desta_etapa = is_retomada and is_primeira_etapa_efetiva

            # Detecta checkpoint ANTES de construir o trainer para preservar métricas
            checkpoint_path = self._find_latest_checkpoint() if is_primeira_etapa_efetiva else None
            resume_from_checkpoint = checkpoint_path is not None

            # Constrói (ou reconstrói) o trainer com contexto da etapa
            # Se há checkpoint ou retomada de curriculum, preserva arquivos de métricas
            self.trainer = self._build_trainer(
                etapa_index=step_index,
                etapa_alias=etapa_atual.alias,
                instancias_previas=instancias_acumuladas,
                step_offset=step_offset_global,
                epoch_offset=epoch_offset_global,
                tokens_previos=tokens_acumulados,
                is_retomada=is_retomada_desta_etapa or resume_from_checkpoint,
                etapa=etapa_atual,
            )

            # Conecta referência do trainer ao GlobalEvalCallback (necessário para evaluate())
            if self._global_eval_callback is not None:
                self._global_eval_callback.set_trainer(self.trainer)

            # Pipeline Universal: marca início da etapa
            self._tracker.iniciar_etapa(step_index=step_index, etapa=etapa_atual)
            tempo_inicio = time.time()

            # Valida o modelo antes do treinamento
            print("\n🔍 STATUS DO MODELO ANTES DO TREINAMENTO:")
            self.print_modelo_status()

            try:
                if resume_from_checkpoint:
                    print_cores(f"<azul>🔄 Tentando continuar treinamento a partir do checkpoint: {checkpoint_path}</azul>", color_auto=False)
                    try:
                        train_stats = self.trainer.train(resume_from_checkpoint=checkpoint_path)
                        print_cores("<verde>✅ Treinamento continuado com sucesso a partir do checkpoint</verde>", color_auto=False)
                        self._historico.evento_checkpoint_retomado(sucesso=True)
                    except Exception as e:
                        error_msg = str(e)
                        print_cores(f"<vermelho>❌ Erro ao continuar do checkpoint: {error_msg}</vermelho>", color_auto=False)
                        print_cores("<amarelo>🔄 Reiniciando treinamento do início desta etapa...</amarelo>", color_auto=False)
                        self._historico.evento_checkpoint_retomado(sucesso=False, erro=error_msg)

                        train_stats = self.trainer.train()
                else:
                    if step_index == 0 and not is_retomada:
                        print_cores("<azul>🆕 Iniciando novo treinamento</azul>", color_auto=False)
                        self._historico.registrar_evento("TREINO INICIADO", f"Novo treinamento do zero")
                    else:
                        print_cores(f"<azul>▶ Iniciando etapa {step_index+1}/{total_etapas}: '{etapa_atual.alias}'</azul>", color_auto=False)
                    train_stats = self.trainer.train()
            except Exception as e:
                self._tracker.marcar_falha(step_index=step_index, alias=etapa_atual.alias, erro=str(e))
                raise

            depois = _print_mem("DEPOIS")
            tempo_total = time.time() - tempo_inicio
            print_cores("<verde>[5/6] Tempo de execução: {:.2f} s</verde>".format(train_stats.metrics["train_runtime"]), color_auto=False)

            # Valida o modelo após o treinamento
            print("\n🔍 STATUS DO MODELO APÓS O TREINAMENTO:")
            info_modelo = self.print_modelo_status()

            stats = {
                **train_stats.metrics,
                "global_step":       train_stats.global_step,
                "training_loss":     train_stats.training_loss,
                "mem_gpu_before":    antes,
                "mem_gpu_after":     depois,
                "ds_train_len" : len(self.train_ds),
                "ds_eval_len" : len(self.eval_ds) if self.eval_ds else 0,
                "modelo_info": info_modelo,
                "etapa_alias": etapa_atual.alias,
                "etapa_tipo": etapa_atual.tipo,
                "parametros_treinaveis": info_modelo.get("parametros_treinaveis", 0) if isinstance(info_modelo, dict) else 0,
            }

            # Registra informação do pace_loss (se callback ativo)
            if self._pace_loss_callback is not None:
                cb = self._pace_loss_callback
                if cb._stopped:
                    stats["pace_loss_atingido"] = True
                    stats["pace_loss_epoch"] = cb._stop_epoch
                    stats["pace_loss_valor"] = cb._stop_loss
                    logger.info(
                        f"🎯 Etapa '{etapa_atual.alias}' encerrou por pace_loss: "
                        f"eval_loss={cb._stop_loss:.4f} < {cb.pace_loss} na época {cb._stop_epoch}"
                    )
                else:
                    stats["pace_loss_atingido"] = False
                    if etapa_atual.pace_epochs_max > 0:
                        logger.info(
                            f"⏱️  Etapa '{etapa_atual.alias}' encerrou por pace_epochs_max={etapa_atual.pace_epochs_max} "
                            f"(pace_loss={etapa_atual.pace_loss} não atingido pelo eval_loss)"
                        )

            # Gera relatório .md na pasta 'treinamento'
            try:
                 hardware = Util.dados_hardware()
            except:
                 hardware = {}

            gerador = GeradorRelatorio(self._yaml_config)
            gerador.gerar_relatorio(
                dataset_stats=self._dataset_stats,
                train_stats=stats,
                hardware_info=hardware
            )
            self._historico.evento_geracao_estatisticas()

            # Grava o modelo antes do último eval (pode dar erro de memória no eval)
            self._save_model(stats=stats)

            # Garante um eval FINAL mesmo que já tenha havido evals em steps
            eval_loss = None
            eval_loss_global = None
            if self.eval_ds:
                final_eval = self.trainer.evaluate()
                stats.update(final_eval)
                eval_loss = final_eval.get("eval_loss")
            
            # Eval global final (todas as etapas combinadas)
            if self._global_eval_callback is not None and self.eval_ds_global is not None:
                try:
                    original_eval_ds = self.trainer.eval_dataset
                    original_load_best = self.trainer.args.load_best_model_at_end
                    self.trainer.args.load_best_model_at_end = False
                    
                    # Usa o dataset tokenizado do callback
                    self.trainer.eval_dataset = self._global_eval_callback._global_ds
                    final_global_eval = self.trainer.evaluate(metric_key_prefix="eval_global")
                    
                    # Restaura configurações originais
                    self.trainer.eval_dataset = original_eval_ds
                    self.trainer.args.load_best_model_at_end = original_load_best
                    
                    eval_loss_global = final_global_eval.get("eval_global_loss")
                    stats["eval_loss_global"] = eval_loss_global
                    if eval_loss_global is not None:
                        logger.info(f"📊 eval_loss_global FINAL: {eval_loss_global:.4f} (etapa '{etapa_atual.alias}')")
                except Exception as e:
                    logger.warning(f"⚠️  Erro no eval global final: {e}")
            
            self._save_model(stats=stats)
            
            # Registra conclusão no histórico
            self._historico.evento_treinamento_concluido(
                stats, alias=etapa_atual.alias, tipo=etapa_atual.tipo
            )
            self._historico.atualizar_yaml_se_necessario()

            # Pipeline Universal: registra conclusão da etapa com métricas e offsets acumulados
            # PRIMEIRO atualiza contadores (necessário para gravar offsets no state)
            n_gpus = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1
            effective_batch = (
                self._yaml_config.treinamento.batch_size
                * self._yaml_config.treinamento.grad_batch_size
                * n_gpus
            )
            step_offset_global += train_stats.global_step
            instancias_etapa = train_stats.global_step * effective_batch
            instancias_acumuladas += instancias_etapa
            tokens_acumulados += round(instancias_etapa * self._media_tokens_por_instancia)
            # Usa a época real do Trainer (não pace_epochs configurado) para robustez
            # com early stopping ou datasets que não dividem exatamente em batches.
            # ceil() garante que a próxima etapa inicia em fronteira inteira de época.
            import math
            epocas_reais = train_stats.metrics.get(
                "epoch",
                etapa_atual.pace_epochs if etapa_atual.pace_epochs > 0 else self._yaml_config.treinamento.epochs
            )
            epoch_offset_global = math.ceil(epoch_offset_global + epocas_reais)

            # DEPOIS registra conclusão com offsets acumulados (para retomada)
            self._tracker.finalizar_etapa(
                step_index=step_index,
                alias=etapa_atual.alias,
                train_loss=stats.get("training_loss"),
                eval_loss=eval_loss,
                eval_loss_global=eval_loss_global,
                global_step=stats.get("global_step"),
                tempo_segundos=round(tempo_total, 2),
                ds_train_len=stats.get("ds_train_len"),
                ds_eval_len=stats.get("ds_eval_len"),
                step_offset_acumulado=step_offset_global,
                epoch_offset_acumulado=epoch_offset_global,
                instancias_acumuladas=instancias_acumuladas,
                tokens_acumulados=tokens_acumulados,
            )

        if is_curriculum:
            logger.info(f"<verde>✅ TREINAMENTO COMPLETO — {total_etapas} etapas de curriculum finalizadas</verde>")
            self._historico.registrar_evento(
                f"CURRICULUM COMPLETO",
                f"- **Etapas concluídas:** {total_etapas}"
            )
        else:
            logger.info("<verde>✅ TREINAMENTO COMPLETO</verde>")
            
        # Calcula o target total demandado (usa pace_epochs_max quando pace_loss configurado)
        target_epochs_yaml = sum([_epochs_efetivos(e) for e in self._etapas])

        # O treinamento encerrou completamente o pipeline requerido. Registrar trava.
        self._tracker.marcar_conclusao(total_etapas=total_etapas, target_epochs=target_epochs_yaml)
        
    # ------------------------- salvamento --------------------------------
    def _save_model(self, stats = None):
        out_dir = self._yaml_config.modelo.saida
        os.makedirs(out_dir, exist_ok=True)
        print_cores(f"<azul>[6/6] Salvando modelo em {out_dir}…</azul>", color_auto=False)
        
        # Salva o modelo (LoRA ou modelo completo)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        
        # Verifica se o modelo foi salvo corretamente
        adapter_config = os.path.join(out_dir, 'adapter_config.json')
        adapter_model = os.path.join(out_dir, 'adapter_model.safetensors')
        
        if os.path.exists(adapter_config):
            print_cores(f"<verde>✅ Arquivo de configuração LoRA salvo: {adapter_config}</verde>", color_auto=False)
            
        if os.path.exists(adapter_model):
            print_cores(f"<verde>✅ Modelo LoRA salvo: {adapter_model}</verde>", color_auto=False)
        elif os.path.exists(os.path.join(out_dir, 'pytorch_model.bin')):
            print_cores(f"<verde>✅ Modelo PyTorch salvo: pytorch_model.bin</verde>", color_auto=False)
        
        # Log detalhado do que foi salvo
        files_saved = []
        for file in os.listdir(out_dir):
            if file.endswith(('.json', '.safetensors', '.bin')):
                files_saved.append(file)
        
        self._historico.evento_modelo_salvo(out_dir, files_saved)
        
        if stats is not None:
            with open(os.path.join(self._yaml_config.modelo.saida, "metrics_summary.json"), "w") as fp:
                 Util.json_dump(stats, fp, indent=2)
        print_cores(r"<verde>Modelo salvo com sucesso \o/</verde>", color_auto=False)

    def _place_inputs(self, inputs):
        try:
            target = self.model.device
        except AttributeError:
            target = next(self.model.parameters()).device
        return inputs.to(target)
    
    def prompt(self, texto: str, temperatura:int = 0, max_new_tokens: int = 2048, processador = None) -> Dict[str, Any]:
        """Tokeniza um prompt simples para teste rápido."""
        if not texto.strip():
            raise ValueError("Prompt vazio")
        
        # Verifica se o modelo tem LoRA ativo
        #is_peft_model = hasattr(self.model, 'peft_config') or hasattr(self.model, 'base_model')
        #model_type = type(self.model).__name__
        #print(f"🔍 Tipo do modelo: {model_type} | PEFT ativo: {is_peft_model}")
        
        if callable(processador):
           inputs = processador(texto)
        else:
            # cria instância temporária do LLMsDataset para usar o método _process_single_message
            dataset_loader = LLMsDataset(
                path='',
                prompt_col='',
                tokenizer=self.tokenizer,
                max_seq_length=0
            )
            inputs = dataset_loader._process_single_message(messages=texto, 
                                                            inferencia=True, 
                                                            max_length=self._yaml_config.treinamento.max_seq_length)
        _temperatura = temperatura if isinstance(temperatura, (float, int)) else 0.2
        _temperatura = min(max(_temperatura,0.000001),1.0)
        #print(f'########### temperatura: {_temperatura}')
        # configuração da predição
        gen_cfg = GenerationConfig.from_model_config(self.model.config)
        gen_cfg.max_new_tokens = max_new_tokens
        gen_cfg.min_length = 1
        gen_cfg.temperature = float(_temperatura)
        gen_cfg.top_k = 20 if _temperatura > 0.3 else 2
        gen_cfg.do_sample = bool(_temperatura > 0.3)
        # predição
        _inputs = self._place_inputs(inputs['input_ids'])
        _attention_mask = self._place_inputs(inputs.get('attention_mask', torch.ones_like(inputs['input_ids'])))
        input_length = _inputs.shape[1]  # comprimento da sequência de entrada
        
        # Cap max_new_tokens ao espaço restante do contexto do modelo
        # (evita "input length + max_new_tokens exceeds maximum sequence length")
        model_max = getattr(self.model.config, 'max_position_embeddings', None)
        if model_max and (input_length + max_new_tokens) > model_max:
            max_new_tokens = max(256, model_max - input_length)
        gen_cfg.max_new_tokens = max_new_tokens
        
        with torch.inference_mode():
             outputs = self.model.generate(_inputs, 
                                        attention_mask=_attention_mask,
                                        max_new_tokens=max_new_tokens,
                                        generation_config = gen_cfg)   
        # faz o decode só da resposta (remove os tokens de entrada)
        res = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)       
        return {'texto': res, 'prompt_tokens': input_length, 'completion_tokens': len(outputs[0]) - input_length}

    def testar_predicoes(self, n_exemplos: int = 1, temperatura: float = 0.0, max_new_tokens: int = 2048, monitorar_memoria: bool = True) -> Dict[str, Any]:
        """
        Testa o modelo com exemplos do dataset de treino e exibe as predições.
        
        Args:
            n_exemplos: Número de exemplos para testar
            temperatura: Temperatura para geração
            max_new_tokens: Número máximo de tokens a gerar
            monitorar_memoria: Se True, monitora RAM/GPU e gera gráfico
            
        Returns:
            Dict com resultados e métricas de memória (se monitorar_memoria=True)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 TESTANDO MODELO COM {n_exemplos} EXEMPLO(S)")
        logger.info(f"{'='*80}")
        
        # Primeiro valida o status do modelo
        self.print_modelo_status()
        
        # verifica se há dataset disponível
        if not hasattr(self, 'train_ds') or len(self.train_ds) == 0:
            logger.error("❌ Nenhum dataset de treino disponível para teste")
            return {"erro": "Nenhum dataset disponível"}
        
        # limita o número de exemplos ao tamanho do dataset
        n_exemplos = min(n_exemplos, len(self.train_ds))
        
        # Inicia monitoramento de memória se solicitado
        monitor = None
        if monitorar_memoria:
            monitor = MonitorRecursos(
                output_dir=self._yaml_config.modelo.saida,
                intervalo_segundos=0.5,
                nome_arquivo="memoria_predicao"
            )
            monitor.iniciar()
        
        resultados = []
        
        try:
            for i in range(n_exemplos):
                logger.info(f"\n{'-'*60}")
                logger.info(f">> EXEMPLO {i+1}/{n_exemplos}")
                logger.info(f"{'-'*60}")
                
                # pega o registro original do dataset
                # Carrega mensagens do curriculum
                mensagens = self._yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo="treino")
                
                # Cria loader temporário com dados em memória
                dataset_loader = LLMsDataset(
                    data=mensagens,
                    tokenizer=self.tokenizer,
                    max_seq_length=self._yaml_config.treinamento.max_seq_length
                )
                
                sample_row = dataset_loader.dataset[i]
                processador = lambda x: dataset_loader._process_single_message(x, max_length=self._yaml_config.treinamento.max_seq_length, inferencia=True)

                # detecta o formato e extrai prompt/completion esperado
                try:
                    if "messages" in sample_row:
                        messages = sample_row["messages"]
                        if isinstance(messages, list) and len(messages) >= 2:
                            user_msg = messages[0].get("content", "")
                            assistant_msg = messages[1].get("content", "")
                            if isinstance(user_msg, list):
                                user_msg = user_msg[0].get("text", "") if user_msg else ""
                            if isinstance(assistant_msg, list):
                                assistant_msg = assistant_msg[0].get("text", "") if assistant_msg else ""
                            prompt = user_msg
                            resposta_esperada = assistant_msg
                        else:
                            prompt = str(messages)
                            resposta_esperada = "N/A"
                    elif "prompt" in sample_row and "completion" in sample_row:
                        prompt = sample_row["prompt"]
                        resposta_esperada = sample_row["completion"]
                    else:
                        logger.error(f"❌ Formato de dados não reconhecido para exemplo {i+1}")
                        continue
                    
                    logger.info(f">> PROMPT:")
                    if len(prompt) > 500:
                        logger.info(f"   {prompt[:250]} [...] {prompt[-250:]}")
                    else:
                        logger.info(f"   {prompt}")
                    
                    logger.info(f"\n>> RESPOSTA ESPERADA:")
                    if len(resposta_esperada) > 500:
                        logger.info(f"   {resposta_esperada[:250]} [...] {resposta_esperada[-250:]}")
                    else:
                        logger.info(f"   {resposta_esperada[:500]}{'...' if len(resposta_esperada) > 500 else ''}")
                    
                    # gera predição do modelo
                    try:
                        # Coleta métrica de memória antes da predição
                        monitor_snapshot = monitor
                        if not monitor_snapshot:
                            # Cria instância temporária se monitoramento contínuo estiver desativado
                            monitor_snapshot = MonitorRecursos(self._yaml_config.modelo.saida)
                        
                        mem_antes = monitor_snapshot.coletar_metricas()
                        
                        tempo_inicio = time.time()
                        resultado = self.prompt(prompt, 
                                                temperatura=temperatura, 
                                                max_new_tokens=max_new_tokens,
                                                processador = processador)
                        tempo_predicao = time.time() - tempo_inicio
                        
                        # Coleta métrica de memória depois da predição
                        mem_depois = monitor_snapshot.coletar_metricas()
                        
                        resposta_modelo = resultado['texto']
                        
                        logger.info(f"\n>> RESPOSTA DO MODELO:")

                        if len(resposta_modelo) > 500:
                            logger.info(f"   {resposta_modelo[:250]} [...] {resposta_modelo[-250:]}")
                        else:
                            logger.info(f"   {resposta_modelo}")
                        
                        logger.info(f"\n>> ESTATÍSTICAS:")
                        logger.info(f"   - Tokens do prompt: {resultado.get('prompt_tokens', 'N/A')}")
                        logger.info(f"   - Tokens da resposta: {resultado.get('completion_tokens', 'N/A')}")
                        logger.info(f"   - Temperatura: {temperatura}")
                        logger.info(f"   - Tempo de predição: {tempo_predicao:.2f}s")
                        
                        # Cálculo de Rouge-L
                        metricas_rouge, erro_rouge = calcular_rouge_l(resposta_esperada, resposta_modelo)
                        if metricas_rouge:
                             logger.info(f"   - Rouge-L True vs Pred: P={metricas_rouge['P']:.4f} R={metricas_rouge['R']:.4f} F1={metricas_rouge['F1']:.4f}")
                        elif erro_rouge:
                             # Warning simplificado
                             logger.warning(f"   - Rouge-L: {erro_rouge}")
                        
                        # Log de memória
                        ram_diff = mem_depois.ram_usada_gb - mem_antes.ram_usada_gb
                        gpu_diff = mem_depois.gpu_usada_gb - mem_antes.gpu_usada_gb
                        logger.info(f"   - Memória RAM: {mem_antes.ram_usada_gb:.1f}GB -> {mem_depois.ram_usada_gb:.1f}GB (delta: {ram_diff:+.1f}GB)")
                        logger.info(f"   - Memória GPU: {mem_antes.gpu_usada_gb:.1f}GB -> {mem_depois.gpu_usada_gb:.1f}GB (delta: {gpu_diff:+.1f}GB)")
                        
                        resultados.append({
                            "exemplo": i + 1,
                            "prompt_tokens": resultado.get('prompt_tokens'),
                            "completion_tokens": resultado.get('completion_tokens'),
                            "tempo_segundos": round(tempo_predicao, 2),
                        })
                        
                    except Exception as e:
                        logger.error(f"❌ Erro ao gerar predição: {str(e)}\n{traceback.format_exc()}")
                        
                except Exception as e:
                    logger.error(f"❌ Erro ao processar exemplo {i+1}: {str(e)}")
        
        finally:
            # Para monitoramento e gera gráfico
            metricas_memoria = {}
            if monitor:
                metricas_memoria = monitor.parar()
                grafico_path = monitor.gerar_grafico()
                if grafico_path:
                    logger.info(f"📈 Gráfico de uso de memória: {grafico_path}")
        
        logger.info(f"\n{'='*80}")
        logger.info(">> TESTE DE PREDIÇÕES CONCLUÍDO")
        logger.info(f"{'='*80}")
        
        return {
            "resultados": resultados,
            "n_exemplos": len(resultados),
            "metricas_memoria": metricas_memoria,
        }

    def validar_modelo_lora(self) -> dict:
        """Valida o modelo e retorna informações detalhadas, incluindo modo de treinamento."""
        info = {
            'modelo_tipo': type(self.model).__name__,
            'is_peft_model': False,
            'adapters_ativos': [],
            'parametros_treinaveis': 0,
            'parametros_totais': 0,
            'lora_detectado': False,
            'modo_treinamento': 'desconhecido',  # 'full', 'lora' ou 'desconhecido'
        }
        
        # Verifica se é modelo PEFT
        info['is_peft_model'] = hasattr(self.model, 'peft_config') or hasattr(self.model, 'base_model')
        
        if info['is_peft_model']:
            info['lora_detectado'] = True
            
            # Obtém informações dos adaptadores
            if hasattr(self.model, 'peft_config'):
                peft_configs = self.model.peft_config
                for adapter_name, config in peft_configs.items():
                    adapter_info = {
                        'nome': adapter_name,
                        'r': getattr(config, 'r', 'N/A'),
                        'alpha': getattr(config, 'lora_alpha', 'N/A'),
                        'dropout': getattr(config, 'lora_dropout', 'N/A'),
                        'target_modules': getattr(config, 'target_modules', [])
                    }
                    info['adapters_ativos'].append(adapter_info)
        
        # Conta parâmetros treináveis e detecta modo de treinamento
        trainable_params = 0
        total_params = 0
        base_treinaveis = 0
        lora_treinaveis = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if "lora_" in name or "modules_to_save" in name:
                    lora_treinaveis += param.numel()
                else:
                    base_treinaveis += param.numel()
        
        # Conta parâmetros quantizados (int/uint — não suportam gradientes)
        quantizados = 0
        for param in self.model.parameters():
            if param.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                quantizados += param.numel()
        
        info['parametros_treinaveis'] = trainable_params
        info['parametros_totais'] = total_params
        info['parametros_quantizados'] = quantizados
        info['percentual_treinavel'] = (trainable_params / total_params * 100) if total_params > 0 else 0
        info['base_treinaveis'] = base_treinaveis
        info['lora_treinaveis'] = lora_treinaveis
        
        # Determina modo de treinamento efetivo
        if trainable_params == 0:
            info['modo_treinamento'] = 'congelado'
        elif base_treinaveis > 0 and info['lora_detectado']:
            info['modo_treinamento'] = 'full'
        elif lora_treinaveis > 0 and base_treinaveis == 0:
            info['modo_treinamento'] = 'lora'
        elif not info['lora_detectado'] and trainable_params > 0:
            info['modo_treinamento'] = 'full'
        else:
            info['modo_treinamento'] = 'desconhecido'
        
        return info

    def print_modelo_status(self):
        """Imprime o status detalhado do modelo, incluindo modo de treinamento (full/lora)."""
        info = self.validar_modelo_lora()
        nbits = self._yaml_config.treinamento.nbits
        
        modo = info['modo_treinamento']
        quantizados = info['parametros_quantizados']
        tem_quantizados = quantizados > 0
        
        if modo == 'full' and tem_quantizados:
            modo_label = f"🔓 FULL (todos os parâmetros float treináveis — modelo quantizado {nbits}-bit)"
        elif modo == 'full':
            modo_label = "🔓 FULL (todos os parâmetros treináveis)"
        elif modo == 'lora':
            modo_label = "🔒 LoRA (apenas adaptadores treináveis)"
        elif modo == 'congelado':
            modo_label = "❄️  CONGELADO (nenhum parâmetro treinável)"
        else:
            modo_label = "❓ Desconhecido"
        
        print(f"\n{'='*60}")
        print(f"📊 STATUS DETALHADO DO MODELO")
        print(f"{'='*60}")
        print(f"Tipo do modelo:       {info['modelo_tipo']}")
        print(f"Modo de treinamento:  {modo_label}")
        print(f"Parâmetros treináveis: {info['parametros_treinaveis']:,}")
        if info['lora_detectado'] and info['base_treinaveis'] > 0:
            print(f"  ├─ base (float):  {info['base_treinaveis']:,}")
            print(f"  └─ LoRA:          {info['lora_treinaveis']:,}")
        print(f"Parâmetros totais:    {info['parametros_totais']:,}")
        if tem_quantizados:
            print(f"  ├─ quantizados ({nbits}-bit, congelados): {quantizados:,}")
            print(f"  └─ float (treináveis):       {info['parametros_totais'] - quantizados:,}")
        print(f"Percentual treinável: {info['percentual_treinavel']:.4f}%")
        
        if tem_quantizados and modo == 'full':
            print(f"\n⚠️  Parâmetros quantizados ({nbits}-bit) não suportam gradientes.")
            print(f"   Para full fine-tuning real de 100% dos pesos, use nbits: 16.")
        
        if info['adapters_ativos']:
            print(f"\n🔧 ADAPTADORES LoRA:")
            for adapter in info['adapters_ativos']:
                print(f"  - {adapter['nome']}: r={adapter['r']}, alpha={adapter['alpha']}")
                modules = adapter['target_modules']
                if isinstance(modules, str) and len(modules) > 50 and modules.startswith("(?:"):
                     modules_str = "Unsloth Default (Todos os módulos lineares)"
                else:
                     modules_str = str(modules)
                print(f"    Modules: {modules_str}")
        
        print(f"{'='*60}")
        
        return info

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class LLMsDataset:
    """Gerencia datasets para fine‑tuning de LLMs com Unsloth."""

    def __init__(self, path: str = None, prompt_col: str = None, tokenizer=None, 
                 max_seq_length: int = 4096, data: list = None):
        """
        Inicializa o dataset a partir de arquivo ou dados em memória.
        
        Args:
            path: Caminho do arquivo (parquet/json/jsonl/txt)
            prompt_col: Nome da coluna com prompts (para parquet)
            tokenizer: Tokenizer do modelo
            max_seq_length: Tamanho máximo da sequência
            data: Dados já carregados em memória (lista de dicts com 'messages')
        """
        self.path = path
        self.prompt_col = prompt_col
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self._template_com_type = self.template_com_type(tokenizer) if tokenizer else False
        
        # Carrega de dados em memória ou de arquivo
        if data is not None:
            # Dados já carregados (do YamlTreinamento.carregar_mensagens_de_pastas)
            self.dataset = Dataset.from_list(data)
        elif path and os.path.isfile(path):
            self.dataset = self._load_dataset()
        else:
            # Dataset vazio (para preparar apenas para predição)
            self.dataset = Dataset.from_list([])

    @staticmethod
    def template_com_type(tokenizer) -> bool:
        """Identifica se o tokenizer usa format type (lista) ou string no content."""
        msgs_list = [{"role": "user", "content": [{"type":"text","text":"ping"}]}]
        msgs_str  = [{"role": "user", "content": "ping"}]
        try:
            tokenizer.apply_chat_template(msgs_list, tokenize=False)
            return  True # lista de partes funcionou
        except Exception:
            pass
        # se lista falhou, tente string
        tokenizer.apply_chat_template(msgs_str, tokenize=False)  # lança se não suportar
        return False # string funcionou

    def _load_dataset(self):
        """Carrega dataset de arquivo parquet, json, jsonl ou txt."""
        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".parquet":
            df = pd.read_parquet(self.path)
            if self.prompt_col and self.prompt_col not in df.columns:
                raise KeyError(f"Coluna '{self.prompt_col}' não encontrada em {self.path}")
            return Dataset.from_pandas(df)
        elif ext in {".json", ".jsonl", ".txt"}:
            self.prompt_col = None  # json não precisa de coluna
            # tenta utf-8, se falhar tenta latin-1
            try:
                dados = open(self.path, "r", encoding="utf-8").readlines()
            except UnicodeEncodeError:
                dados = open(self.path, "r", encoding="latin-1").readlines()
            registros = [json.loads(linha) for linha in dados if linha.strip()]
            return Dataset.from_list(registros)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {ext}")

    def _detect_dataset_format(self, sample_row) -> str:
        """Detecta o formato do dataset analisando uma linha de exemplo.
        
        Returns:
            'messages': se tem coluna messages com lista de mensagens user/assistant
            'prompt_completion': se tem colunas prompt e completion
        """
        if isinstance(sample_row, dict):
            if "messages" in sample_row:
                return "messages"
            elif "prompt" in sample_row and "completion" in sample_row:
                return "prompt_completion"
        raise ValueError(f"Formato de dataset não reconhecido: {list(sample_row.keys()) if isinstance(sample_row, dict) else type(sample_row)}")

    def _process_single_message(self, messages, max_length: int = None, inferencia: bool = False) -> Dict[str, Any]:
        """Processa uma única mensagem ou par prompt/completion usando chat template.
        
        Args:
            messages: Mensagens no formato dict ou lista de mensagens user/assistant
            max_length: Comprimento máximo da sequência (usa self.max_seq_length se None)
            
        Returns:
            Dict com input_ids, attention_mask e labels prontos para treinamento
        """
        if max_length is None:
            max_length = self.max_seq_length
            
        # se for um dict com coluna prompt e completion, transforma em lista
        if isinstance(messages, str):
            prompt = messages if not self._template_com_type else [{"type": "text", "text": messages}]
            _messages = [{"role": "user", "content": prompt}]
        elif isinstance(messages, dict) and "prompt" in messages and "completion" in messages:
            prompt = messages["prompt"] if not self._template_com_type else [{"type": "text", "text": messages["prompt"]}]
            completion = messages["completion"] if not self._template_com_type else [{"type": "text", "text": messages["completion"]}]
            _messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
        else:
            # confere user assistant e estrutura
            if not (isinstance(messages, (list, np.ndarray, tuple)) and len(messages) == 2):
                raise ValueError(f"Esperado lista de 2 mensagens user/assistant >> {type(messages)}")
            if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
                raise ValueError("Ordem user/assistant inválida")
            _messages = deepcopy(messages)
            for message in _messages:
                if ("content" not in message) or ("role" not in message):
                    raise ValueError("Cada mensagem deve ter uma chave 'content' e 'role'.")
                if self._template_com_type:
                    if isinstance(message["content"], str):
                        message["content"] = [{"type": "text", "text": message["content"]}]
                else:
                    if not isinstance(message["content"], str):
                        raise ValueError("O campo 'content' deve ser uma string.")
        
        # identifica o que é prompt para ajustar attention mask
        prompt_ids = self.tokenizer.apply_chat_template(
                _messages[:1],   # só a primeira mensagem
                tokenize=True,
                add_generation_prompt=True,   # adiciona o cabeçalho do assistant/model
                return_tensors='pt' if inferencia else None,
            )        
        if inferencia:
            # retorna lista de listas que é o esperado pelo generate
            attention_mask = torch.ones_like(prompt_ids)
            return {"input_ids": prompt_ids, "attention_mask": attention_mask}
        # processa tudo para ajustar attention mask
        full_ids = self.tokenizer.apply_chat_template(
                _messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors=None,
            )
        
        #  uma mensagem
        if isinstance(full_ids[0], (list, tuple, np.ndarray)):
            full_ids = full_ids[0]   # extrai a primeira sublista
        if isinstance(prompt_ids[0], (list, tuple, np.ndarray)):
            prompt_ids = prompt_ids[0]  # extrai a primeira sublista
            
        # Converte para lista se for NumPy array para evitar problemas de concatenação
        if isinstance(full_ids, np.ndarray):
            full_ids = full_ids.tolist()
        if isinstance(prompt_ids, np.ndarray):
            prompt_ids = prompt_ids.tolist()
            
        # truncando max_length e aplicando mascara para processar apenas a resposta
        input_ids = full_ids[:max_length]
        attention_mask = [1] * len(input_ids)
        if len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        # 4) Cutoff = início da resposta do assistant (limitado pelo max_length)
        cutoff = min(len(prompt_ids), max_length)

        # 5) Labels = cópia de input_ids; máscara em [0:cutoff) e em padding
        labels = input_ids.copy()
        for j in range(max_length):
            if j < cutoff or attention_mask[j] == 0:
                labels[j] = -100 # labels para serem ignorados no treinamento

        # Garante que todos sejam listas de inteiros (não arrays NumPy)
        return {
            "input_ids": [int(x) for x in input_ids],
            "attention_mask": [int(x) for x in attention_mask],
            "labels": [int(x) for x in labels],
        }

    def preprocess_with_chat_template(self, dataset: Dataset = None, max_length: int = None) -> Dataset:
        """Processa um dataset completo usando chat template e retorna tokens prontos para treinamento.
        
        Args:
            dataset: Dataset a ser processado (usa self.dataset se None)
            max_length: Comprimento máximo da sequência (usa self.max_seq_length se None)
            
        Returns:
            Dataset processado com input_ids, attention_mask e labels
        """
        if dataset is None:
            dataset = self.dataset
            
        if max_length is None:
            max_length = self.max_seq_length
            
        # detecta o formato do dataset analisando o primeiro registro
        sample_row = dataset[0]
        dataset_format = self._detect_dataset_format(sample_row)
        print(f">> Formato detectado: {dataset_format}")
        
        def process_row(row):
            if dataset_format == "messages":
                # dataset com coluna messages
                messages = row[self.prompt_col] if self.prompt_col else row["messages"]
            else:  # prompt_completion
                # dataset com colunas prompt e completion
                messages = {
                    "prompt": row["prompt"],
                    "completion": row["completion"]
                }
            
            return self._process_single_message(messages, max_length)
        
        # aplica processamento a todo o dataset
        processed_dataset = dataset.map(process_row, remove_columns=dataset.column_names)
        print(f">> Dataset processado: {len(processed_dataset)} registros")
        
        return processed_dataset

    def get_processed_dataset(self) -> Dataset:
        """Retorna dataset processado com chat template aplicado."""
        return self.preprocess_with_chat_template()
    
    def print_sample(self, n: int = 1) -> None:
        """Imprime exemplo de registro do dataset processado."""
        print(f'Exemplo de registro do dataset [n={n}]:')
        print(json.dumps(self.get_sample(n), indent=2, ensure_ascii=False))

    def get_sample(self, n: int = 1) -> dict:
        """Retorna registro do dataset com texto decodificado para análise."""
        # detecta formato do primeiro registro
        sample_row = self.dataset[0]
        dataset_format = self._detect_dataset_format(sample_row)
        
        if n == 1:
            row = self.dataset[0]
            if dataset_format == "messages":
                messages = row[self.prompt_col] if self.prompt_col else row["messages"]
            else:  # prompt_completion
                messages = {"prompt": row["prompt"], "completion": row["completion"]}
            
            # processa a mensagem
            result = self._process_single_message(messages)
            # result['input_ids'] já é uma lista de inteiros, não precisa de .tolist()
            texto = self.tokenizer.decode(result['input_ids'], skip_special_tokens=False)
            result['texto_decodificado'] = texto
        else:
            # para múltiplas amostras
            result = {'input_ids': [], 'attention_mask': [], 'labels': [], 'texto_decodificado': []}
            for i in range(min(n, len(self.dataset))):
                row = self.dataset[i]
                if dataset_format == "messages":
                    messages = row[self.prompt_col] if self.prompt_col else row["messages"]
                else:  # prompt_completion
                    messages = {"prompt": row["prompt"], "completion": row["completion"]}
                
                sample_result = self._process_single_message(messages)
                result['input_ids'].append(sample_result['input_ids'])
                result['attention_mask'].append(sample_result['attention_mask'])
                result['labels'].append(sample_result['labels'])
                
                # sample_result['input_ids'] já é uma lista de inteiros
                texto = self.tokenizer.decode(sample_result['input_ids'], skip_special_tokens=False)
                result['texto_decodificado'].append(texto)
        
        return result

    def get_stats(self) -> dict:
        """Retorna estatísticas básicas do dataset."""
        # detecta formato do dataset
        sample_row = self.dataset[0]
        dataset_format = self._detect_dataset_format(sample_row)
        
        return {
            "total_registros": len(self.dataset),
            "colunas": list(self.dataset.column_names),
            "formato_arquivo": os.path.splitext(self.path)[1].lower(),
            "formato_dataset": dataset_format,
            "caminho": self.path,
            "coluna_prompt": self.prompt_col,
            "max_seq_length": self.max_seq_length,
            "template_com_type": self._template_com_type,
        }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _create_default_cfg(path: str) -> None:
    template = {
        "curriculum": {
            "predicao": {
                "pasta": "../saidas/predicoes",
            },
            "saida": {
                "pasta": "../saidas/gold",
                "formato": "texto",
            },
            "entrada": {
                "pasta": "../saidas/textos",
                "prompt_template": "../prompts/template.txt",
                "tag_texto": "<<TEXTO>>",
            },
            "divisao": [
                {
                    "arquivo": "../divisao.csv",
                    "alias": "Principal",
                    "tipo": "lora",
                    "pace_epochs": 1,
                    "proporcao": [
                        {"treino": 0.70},
                        {"validacao": 0.15},
                        {"teste": 0.15}
                    ]
                }
            ],
            "validacao": {
                "exigir_json_valido": True,
                "exigir_ids_pareados": True,
            },
        },
        "misc": {
            "log_level": "INFO",
            "env_chave_criptografia": "",
        },
        "modelo": {
            "base_model_name": "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
            "saida": "../modelos/gemma-3-12b-refleg20k-v01",
        },
        "treinamento": {
            "flash_attention_2": True,
            "liger_kernel": True,
            "eval_steps": "15%",
            "batch_size": 2,
            "grad_batch_size": 5,
            "num_train_epochs": 1,
            "max_seq_length": 4096,
            "learning_rate": 2e-4,
            "save_checkpoints": True,
            "resume_from_checkpoint": True,
            "warmup_steps": 5,
            "nbits": 4,
            "train_on_responses_only": True,
        },
        "lora": {
            "r": 8,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        }
    }
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(template, fp, sort_keys=False, allow_unicode=True)


def _selecionar_yaml_treino():
    """Exibe menu de seleção de YAML na pasta atual com opções de treinamento."""
    from util_menu_opcoes import escolher_yaml
    
    return escolher_yaml(
        pasta='./',
        chave_obrigatoria='modelo',
        titulo='Selecione o arquivo de configuração para treinamento',
        padrao_recente=True,
        opcoes_extras=[
            ("Gerar YAML padrão", "_CRIAR_NOVO"),
            ("Sair", None),
        ]
    )


def _modo_interativo_treinar(yaml_path: str):
    """
    Exibe menu interativo com ações de treinamento.
    
    Returns:
        Nome da ação escolhida ou None se cancelou
    """
    from treinar_unsloth_actions import (
        _exibir_cabecalho_modelo, _verificar_modelo_treinado, _verificar_checkpoints_existem,
        _detectar_tipo_modelo_saida,
    )
    
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=False)
    _exibir_cabecalho_modelo(yaml_config)
    
    # Status atual
    tipo_modelo = _detectar_tipo_modelo_saida(yaml_config.modelo.saida)
    tem_modelo = bool(tipo_modelo)
    tem_checkpoints, qtd_checkpoints = _verificar_checkpoints_existem(yaml_config)
    
    print_cores("\n📊 STATUS ATUAL:", color_auto=False)
    if tipo_modelo == 'lora':
        print_cores("   ✅ Modelo LoRA treinado encontrado", color_auto=False)
    elif tipo_modelo == 'full':
        print_cores("   ✅ Modelo FULL fine-tuned encontrado", color_auto=False)
    else:
        print_cores("   ❌ Nenhum modelo treinado encontrado", color_auto=False)
    
    if tem_checkpoints:
        print_cores(f"   💾 {qtd_checkpoints} checkpoint(s) disponível(is)", color_auto=False)
    else:
        print_cores("   💾 Nenhum checkpoint encontrado", color_auto=False)
    
    # Menu
    itens = [
        ('1', 'treinar',        'Iniciar ou continuar treinamento'),
        ('2', 'reset+treinar',  'Limpar tudo e treinar do zero'),
        ('3', 'reset',          'Limpar treinamento atual'),
        ('---',),
        ('0', 'sair',           'Cancelar e sair'),
    ]
    
    try:
        escolha = exibir_menu_opcoes(
            titulo='<azul>📋 AÇÕES DE TREINAMENTO:</azul>',
            itens=itens,
        )
        
        mapa_acoes = {
            '1': 'treinar', 'treinar': 'treinar', 'train': 'treinar',
            '2': 'reset+treinar',
            '3': 'reset', 'reset': 'reset',
            '0': None, 'sair': None, 'exit': None, 'quit': None,
        }
        
        acao = mapa_acoes.get(escolha)
        
        if escolha not in mapa_acoes:
            logger.warning(f"Opção inválida: '{escolha}'")
            return None
        
        return acao
        
    except (KeyboardInterrupt, EOFError):
        logger.info("\nOperação cancelada pelo usuário.")
        return None


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune LLMs (Gemma, Qwen, Llama, DeepSeek) com configuração YAML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ações de treinamento:
  --treinar         Inicia ou continua o treinamento
  --reset           Limpa o treinamento atual (com confirmação)
  --reset --treinar Limpa e inicia treinamento do zero
  --dicas           Injeta comentários de dicas no YAML de configuração
  
Sem argumentos: modo interativo (seleciona YAML e ação via menu).

Para avaliação, predição e exportação, use: treinar_unsloth_avaliar.py

Exemplos:
  %(prog)s                                 # Modo interativo completo
  %(prog)s config.yaml                     # Seleciona ação via menu
  %(prog)s config.yaml --treinar           # Inicia treinamento
  %(prog)s config.yaml --reset --treinar   # Limpa e treina do zero
"""
    )
    parser.add_argument("config", nargs='?', default=None,
                        help="Arquivo YAML com as configurações (opcional: se omitido, exibe menu)")
    
    # Ações de treinamento
    parser.add_argument("--treinar", action="store_true",
                        help="Inicia ou continua o treinamento")
    parser.add_argument("--reset", action="store_true",
                        help="Limpa treinamento atual (checkpoints e modelo LoRA)")
    
    # Injeção de dicas
    parser.add_argument("--dicas", action="store_true", 
                        help="Injeta comentários de dicas no YAML de configuração")
    
    # Opções
    parser.add_argument("--log-level", type=str, default=None, 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="Nível de log (sobrescreve misc.log_level do YAML)")
    
    args = parser.parse_args()

    # --- Resolve YAML (menu se não fornecido) ---
    cfg_path = args.config
    
    if cfg_path is None:
        resultado = _selecionar_yaml_treino()
        if resultado == "_CRIAR_NOVO":
            nome = "novo_treinamento.yaml"
            _create_default_cfg(nome)
            logger.info(
                f"✅ Arquivo de configuração criado: '{nome}'.\n"
                "   Revise os parâmetros e execute novamente."
            )
            sys.exit(0)
        if resultado is None:
            sys.exit(0)
        cfg_path = resultado
    
    if not os.path.exists(cfg_path):
        _create_default_cfg(cfg_path)
        logger.info(
            f"Arquivo de configuração criado em '{cfg_path}'.\n"
            "Revise os parâmetros e execute novamente para iniciar o treinamento."
        )
        sys.exit(0)
    
    # --- Configura logging ---
    log_level_padrao = "INFO"
    try:
        yaml_config = YamlTreinamento(cfg_path, validar_caminhos=False)
        log_level_padrao = yaml_config.misc.log_level
    except Exception:
        pass

    nivel_log = args.log_level if args.log_level else log_level_padrao
    configurar_logging(nivel=nivel_log)
    
    logger.debug(f"Log level configurado: {nivel_log} (CLI: {args.log_level}, YAML: {log_level_padrao})")

    # --- Injeção de dicas (antes de CUDA) ---
    if args.dicas:
        from treinar_unsloth_actions import executar_injetar_dicas
        executar_injetar_dicas(cfg_path)

    # --- Info CUDA ---
    if torch.cuda.is_available():
        logger.info(f"CUDA disponível — {torch.cuda.device_count()} GPU(s) detectada(s)")
    else:
        logger.warning("CUDA não disponível — treinamento será na CPU (muito mais lento)")

    # --- Importa ações de treinamento ---
    from treinar_unsloth_actions import executar_treinar, executar_reset
    
    # --- Identifica se há ação explícita ---
    tem_acao_explicita = args.treinar or args.reset
    
    if not tem_acao_explicita:
        # Modo interativo: menu de ações de treinamento
        acao = _modo_interativo_treinar(cfg_path)
        if acao == 'treinar':
            executar_treinar(cfg_path, reset=False)
        elif acao == 'reset+treinar':
            executar_treinar(cfg_path, reset=True)
        elif acao == 'reset':
            executar_reset(cfg_path, confirmar=True)
    else:
        # Ação explícita via CLI
        if args.treinar:
            executar_treinar(cfg_path, reset=args.reset)
        elif args.reset:
            executar_reset(cfg_path, confirmar=True)



if __name__ == "__main__":
    # Carrega .env do diretório src (funciona de qualquer pasta)
    UtilEnv.carregar_env(pastas=['./', '../', '../src/'])
    _cli()

