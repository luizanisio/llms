#!/usr/bin/env python3

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Treinar Gemma‑3, Deepseek, Llhama, Qwen usando Unsloth 
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

# Desabilita compilação dinâmica (Dynamo/Inductor) para evitar erros de falta de compilador C
os.environ["TORCH_COMPILE_DISABLE"] = "1"
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except ImportError:
    pass
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
    from unsloth import FastModel
except ImportError:
    print("Erro: O pacote 'unsloth' não está instalado.")
    print("Por favor, instale-o executando: pip install unsloth")
    sys.exit(1)
try:
    from unsloth.chat_templates import get_chat_template, CHAT_TEMPLATES, train_on_responses_only
except ImportError:
    print("Erro: O pacote 'unsloth.chat_templates' não está instalado.")
    print("Por favor, instale-o executando: pip install unsloth")
    sys.exit(1)
try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    print("Erro: O pacote 'trl' não está instalado.")
    print("Por favor, instale-o executando: pip install trl")
    sys.exit(1)
try:
    from transformers import TrainerCallback
except ImportError:
    print("Erro: O pacote 'transformers' não está instalado.")
    print("Por favor, instale-o executando: pip install transformers")
    sys.exit(1)
try:
    from transformers import GenerationConfig
except ImportError:
    print("Erro: O pacote 'transformers' não está instalado.")
    print("Por favor, instale-o executando: pip install transformers")
    sys.exit(1)
try:
    import numpy as np
except ImportError:
    print("Erro: O pacote 'numpy' não está instalado.")
    print("Por favor, instale-o executando: pip install numpy")
    sys.exit(1)
try:
    from datetime import datetime
except ImportError:
    print("Erro: O pacote 'datetime' não está instalado.")
    print("Por favor, instale-o executando: pip install datetime")
    sys.exit(1)
try:
    from copy import deepcopy
except ImportError:
    print("Erro: O pacote 'copy' não está instalado.")
    print("Por favor, instale-o executando: pip install copy")
    sys.exit(1)
from unsloth import FastModel
# from unsloth.chat_templates import get_chat_template, CHAT_TEMPLATES, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from transformers import GenerationConfig
import numpy as np
from datetime import datetime
from copy import deepcopy

# Import da nova classe de configuração YAML e Gerador de Relatório
from treinar_unsloth_util import YamlTreinamento, TIPO_ENTRADA_PASTAS, TIPO_ENTRADA_DATASET, TIPOS_BASEADOS_EM_PASTAS, calcular_rouge_l
from treinar_unsloth_report import GeradorRelatorio
from treinar_unsloth_chat import TreinarChatTemplate
from util import UtilEnv, Util

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
    
    def __init__(self, path):
        self.path = path
        # zera o arquivo no início
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
        - tokens_acumulados: total de tokens processados (instâncias × max_seq_length)
        - ram_usada_gb, gpu*_reservada_gb, cpu_uso_%: métricas de hardware
    """
    
    def __init__(self, output_dir: str, etapa_alias: str = "Principal",
                 etapa_index: int = 0, instancias_previas: int = 0,
                 step_offset: int = 0, epoch_offset: float = 0.0,
                 max_seq_length: int = 0, tokens_previos: int = 0):
        """
        Args:
            output_dir: Diretório onde salvar o arquivo de métricas
            etapa_alias: Nome da etapa do curriculum (ex: "fácil", "médio")
            etapa_index: Índice da etapa no curriculum (0-based)
            instancias_previas: Total de instâncias treinadas em etapas anteriores
            step_offset: Total de steps acumulados de etapas anteriores (para step_global)
            epoch_offset: Total de épocas acumuladas de etapas anteriores (para epoch_global)
            max_seq_length: Comprimento máximo de sequência (para calcular tokens processados)
            tokens_previos: Total de tokens processados em etapas anteriores
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
        self._max_seq_length = max_seq_length
        self._tokens_previos = tokens_previos
        
        # Cria diretório; trunca arquivo apenas na primeira etapa do curriculum
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        if etapa_index == 0:
            open(self.metrics_file, "w").close()
        
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
            "max_seq_length": self._max_seq_length,
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
            "tokens_acumulados": self._tokens_previos + (instancias_acumuladas - self._instancias_previas) * self._max_seq_length if self._max_seq_length > 0 else 0,
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
        """Registra métricas completas de avaliação com contexto de etapa."""
        if not metrics:
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
            "tokens_acumulados": self._tokens_previos + (instancias_acumuladas - self._instancias_previas) * self._max_seq_length if self._max_seq_length > 0 else 0,
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
            "tokens_acumulados": self._tokens_previos + (instancias_acumuladas - self._instancias_previas) * self._max_seq_length if self._max_seq_length > 0 else 0,
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
            except Exception as e:
                logger.warning(f"Erro ao renomear checkpoint: {e}")
        elif os.path.exists(padded_path):
            # Esperado em resume_from_checkpoint (checkpoint já foi renomeado)
            logger.debug(f"Checkpoint já existe com zero-padding: {padded_name}")

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
        from treinar_unsloth_pipeline import CurriculumTracker
        self._etapas = self._yaml_config.curriculum
        self._tracker = CurriculumTracker(self._yaml_config.modelo.saida)
        
        # Valida max_seq_length (obrigatório) e exibe info de tokens por etapa
        self._yaml_config.validar_max_seq_length()
        
        # Carrega modelo e tokenizer
        self.model, self.tokenizer = self._load_model()
        
        # Gerenciador de templates de chat
        self.chat_handler = TreinarChatTemplate(self.tokenizer, self._yaml_config.modelo.base)
        self.tokenizer = self.chat_handler.tokenizer
        
        # Carrega datasets baseado no tipo de entrada
        if self._yaml_config.tipo_entrada in TIPOS_BASEADOS_EM_PASTAS:
            # Modo pastas/curriculum: carrega de arquivos pareados
            self.train_ds = self._load_from_pastas(alvo="treino")
            self.eval_ds = self._load_from_pastas(alvo="validacao")
        else:
            # Modo dataset: carrega de arquivos parquet
            self.train_ds = self._load_split(
                self._yaml_config.dataset.train_file, 
                self._yaml_config.dataset.train_prompt_col, 
                split="treino"
            )
            self.eval_ds = None
            if self._yaml_config.dataset.eval_file:
                self.eval_ds = self._load_split(
                    self._yaml_config.dataset.eval_file,
                    self._yaml_config.dataset.eval_prompt_col or self._yaml_config.dataset.train_prompt_col,
                    split="validação",
                )
        
        # Exibe estatísticas pré-treinamento e armazena para relatório
        ts = self._print_dataset_stats(self.train_ds, "Dataset de Treino")
        es = self._print_dataset_stats(self.eval_ds, "Dataset de Validação") if self.eval_ds and len(self.eval_ds) > 0 else {}
        
        self._dataset_stats = {
            "treino_len": len(self.train_ds),
            "validacao_len": len(self.eval_ds) if self.eval_ds else 0,
            "token_stats": ts
        }

        # Log do primeiro registro
        if len(self.train_ds) > 0:
            self.log_processamento(self.train_ds[0], titulo="Primeiro registro do dataset de treino")
        
        self.save_checkpoints = self._yaml_config.treinamento.save_checkpoints
        self.trainer = None  # Inicialização lazy no método train()

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
        
    # ------------------------- modelo ------------------------------------
    def _load_model(self):
        print("<azul>[1/6] Carregando modelo base…</azul>")
        # Carrega configuração de bits
        nbits = self._yaml_config.treinamento.nbits
        
        lora_ok = False
        
        if not self.force_base:
            # Verifica se existe modelo LoRA já treinado
            lora_model_path = self._yaml_config.modelo.saida
            arq_lora = os.path.join(lora_model_path, 'adapter_config.json')
            arq_model = os.path.join(lora_model_path, 'adapter_model.safetensors')
            
            # Verifica se é um modelo LoRA completo (não apenas um checkpoint)
            is_trained_lora = (os.path.exists(arq_lora) and 
                            (os.path.exists(arq_model) or 
                            os.path.exists(os.path.join(lora_model_path, 'pytorch_model.bin'))))
            
            if is_trained_lora:
                print(f'<azul>🔄 Carregando modelo LoRA já treinado de {lora_model_path}...</azul>')
                try:
                    # Carrega o modelo LoRA já treinado diretamente
                    model, tokenizer = FastModel.from_pretrained(
                        model_name=lora_model_path,  # Carrega da pasta do modelo treinado
                        max_seq_length=self._yaml_config.treinamento.max_seq_length,
                        load_in_4bit=nbits == 4,
                        load_in_8bit=nbits == 8,
                        device_map="auto",
                    )
                    print(f'<verde>✅ Modelo LoRA treinado carregado com sucesso!</verde>')
                    lora_ok = True
                    
                    # Log de informações do modelo carregado
                    self.log_processamento(f"Modelo LoRA carregado de: {lora_model_path}", "LORA_LOADED")
                    
                except Exception as e:
                    print(f'<vermelho>❌ Erro ao carregar modelo LoRA treinado: {e}</vermelho>')
                    traceback.print_exc()
                    print('<amarelo>Tentando carregar modelo base e aplicar LoRA...</amarelo>')
                    time.sleep(2)
        else:
             print(f'<cinza>ℹ️  Opção --base ativada: Ignorando busca por modelo LoRA treinado.</cinza>')
        
        # Se não conseguiu carregar o LoRA ou não existe ou force_base=True, carrega modelo base
        if not lora_ok:
            print(f'<azul>🔄 Carregando modelo base: {self._yaml_config.modelo.base}...</azul>')
            model, tokenizer = FastModel.from_pretrained(
                model_name=self._yaml_config.modelo.base,
                max_seq_length=self._yaml_config.treinamento.max_seq_length,
                load_in_4bit=nbits == 4,
                load_in_8bit=nbits == 8,
                full_finetuning=self._yaml_config.lora.r in (0,None,False) or self.force_base # Se force_base, não prepara para finetuning
            )
            
            # Se usar LoRA, aplica as configurações (Exceto se force_base=True)
            if not self.force_base and self._yaml_config.lora.r not in (0,None,False):
                print(f'<azul>🔄 Aplicando LoRA r={self._yaml_config.lora.r} ao modelo base ...</azul>')
                model = FastModel.get_peft_model(
                    model,
                    finetune_vision_layers=False,
                    finetune_language_layers=True,
                    finetune_attention_modules=True,
                    finetune_mlp_modules=True,
                    r=self._yaml_config.lora.r,
                    lora_alpha=self._yaml_config.lora.alpha,
                    lora_dropout=self._yaml_config.lora.dropout,
                    bias="none",
                    random_state=3407,
                    device_map="auto",
                )
            elif self.force_base:
                print(f'<cinza>ℹ️  Opção --base ativada: Não aplicando adaptadores LoRA.</cinza>')
        # Template agora é aplicado pelo TreinarChatTemplate após o carregamento
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
        
        # Log detalhado do modelo carregado
        model_type = type(model).__name__
        is_peft_model = hasattr(model, 'peft_config') or hasattr(model, 'base_model')
        
        print(f"\n<azul>📊 MODELO CARREGADO:</azul>")
        print(f"  - Tipo: {model_type}")
        print(f"  - É modelo PEFT: {is_peft_model}")
        print(f"  - LoRA carregado: {lora_ok}")
        
        if is_peft_model:
            try:
                if hasattr(model, 'peft_config'):
                    peft_configs = model.peft_config
                    print(f"  - Configurações PEFT: {list(peft_configs.keys())}")
                    for adapter_name, config in peft_configs.items():
                        print(f"    * {adapter_name}: r={getattr(config, 'r', 'N/A')}, alpha={getattr(config, 'lora_alpha', 'N/A')}")
                elif hasattr(model, 'base_model'):
                    print(f"  - Modelo base: {type(model.base_model).__name__}")
            except Exception as e:
                print(f"  - Erro ao obter detalhes PEFT: {e}")
        
        self.log_processamento(self._yaml_config._raw_config, titulo="Configuração do treinamento")
        self.log_processamento(str(model), titulo="Resumo do modelo")
        self.log_processamento(f"Tipo do modelo: {model_type} | PEFT: {is_peft_model} | LoRA OK: {lora_ok}", titulo="Status do modelo")
        self.log_processamento(tokenizer.chat_template, titulo="Template do tokenizer")
        return model, tokenizer


    # ------------------------- dados -------------------------------------
    def _load_split(self, parquet_path: str, prompt_col: str, *, split: str) -> Dataset:
        """Carrega dataset usando classe LLMsDataset para padronização."""
        print(f"[2/6] Lendo {split} de {parquet_path}…")
        
        # cria instância da classe LLMsDataset com detecção automática do template
        dataset_loader = LLMsDataset(
            path=parquet_path,
            prompt_col=prompt_col,
            tokenizer=self.tokenizer,
            max_seq_length=self._yaml_config.treinamento.max_seq_length
        )
        
        # obtém dataset processado
        print(f" - {split} carregado com {len(dataset_loader.dataset)} registros")
        
        return dataset_loader.dataset

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
             "dataset": dataclasses.asdict(yaml_config.dataset) if yaml_config.dataset else None,
             "formatos": dataclasses.asdict(yaml_config.formatos)
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
        
        # Modo pastas/curriculum: mostra arquivos pareados
        if yaml_config.tipo_entrada in TIPOS_BASEADOS_EM_PASTAS:
            print(f"\n📁 MODO PASTAS - ARQUIVOS PAREADOS:")
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
                print(f"  ❌ Erro ao processar pastas: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Modo dataset: carrega de arquivo
            if yaml_config.dataset and yaml_config.dataset.train_file:
                try:
                    train_loader = LLMsDataset(
                        path=yaml_config.dataset.train_file,
                        prompt_col=yaml_config.dataset.train_prompt_col,
                        tokenizer=tokenizer,
                        max_seq_length=yaml_config.treinamento.max_seq_length
                    )
                    train_stats = train_loader.get_stats()
                    print(f"\n📊 DATASET DE TREINO:")
                    print(f"  - Registros: {train_stats['total_registros']}")
                    print(f"  - Colunas: {train_stats['colunas']}")
                    print(f"  - Formato: {train_stats['formato_arquivo']}")
                    print(f"  - Caminho: {train_stats['caminho']}")
                except Exception as e:
                    print(f"\n❌ Erro ao carregar dataset de treino: {e}")
        
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
                       epoch_offset: float = 0.0, tokens_previos: int = 0) -> SFTTrainer:
        """Constrói o SFTTrainer com callbacks de métricas.
        
        Args:
            etapa_index: Índice da etapa do curriculum (0 para treino simples)
            etapa_alias: Nome da etapa do curriculum ("Principal" para treino simples)
            instancias_previas: Instâncias acumuladas de etapas anteriores
            step_offset: Steps acumulados de etapas anteriores (para step_global contínuo)
            epoch_offset: Épocas acumuladas de etapas anteriores (para epoch_global contínuo)
            tokens_previos: Tokens processados em etapas anteriores (para tokens_acumulados contínuo)
        """
        print("<azul>[3/6] Configurando trainer…</azul>")
        
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
           print(f'<cinza> - avaliando a cada {eval_steps} steps...</cinza>')
        
        log_steps = eval_steps if isinstance(eval_steps, int) and eval_steps > 0 else 50
        
        if self.save_checkpoints:
            print(f'<cinza> - gravando checkpoints a cada {log_steps} steps</cinza>')
        
        # Log train_on_responses_only
        if treino_cfg.train_on_responses_only:
            print(f'<cinza> - train_on_responses_only ATIVADO (treina apenas nas respostas do assistant)</cinza>')

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
            logging_steps=1,
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
            gradient_checkpointing="unsloth", 
            remove_unused_columns=False,
            dataloader_drop_last=False,
            dataset_text_field="text", # usamos a coluna 'text' formatada
            max_seq_length=treino_cfg.max_seq_length, 
            dataset_num_proc=2,
            packing=False,
            per_device_eval_batch_size=1,     # Força batch 1 na validação para economizar VRAM
            eval_accumulation_steps=1,        # Descarrega logits da GPU para CPU a cada passo
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            args=args,
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
                    print(f'   <verde>✅ train_on_responses_only: dataset já possui labels pré-mascarados</verde>')
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
        if trainer.eval_dataset is not None:
            for col in _str_cols & set(trainer.eval_dataset.column_names):
                trainer.eval_dataset = trainer.eval_dataset.remove_columns(col)
        
        # Configura diretório de saída
        output_dir = self._yaml_config.modelo.saida
        os.makedirs(output_dir, exist_ok=True)
        
        # === CALLBACKS DE MÉTRICAS ===
        
        # 1. JsonLogger (métricas brutas em metrics_stream.jsonl)
        jsonl = os.path.join(output_dir, "metrics_stream.jsonl")
        if os.path.isfile(jsonl):
            os.remove(jsonl)
        trainer.add_callback(JsonLoggerCallback(jsonl))
        
        # 2. MetricsLoggerCallback (métricas unificadas: loss, hardware, curriculum, tokens)
        trainer.add_callback(MetricsLoggerCallback(
            output_dir,
            etapa_alias=etapa_alias,
            etapa_index=etapa_index,
            instancias_previas=instancias_previas,
            step_offset=step_offset,
            epoch_offset=epoch_offset,
            max_seq_length=self._yaml_config.treinamento.max_seq_length,
            tokens_previos=tokens_previos,
        ))
        
        # 3. CheckpointRenameCallback (renomeia checkpoints com zero-padding)
        if self.save_checkpoints:
            chkpt_dir = os.path.join(self._yaml_config.modelo.saida, "chkpt")
            os.makedirs(chkpt_dir, exist_ok=True)
            trainer.add_callback(CheckpointRenameCallback(chkpt_dir, step_offset=step_offset))
        
        print(f'<cinza> - callbacks de métricas configurados:</cinza>')
        print(f'   <cinza>• metrics_stream.jsonl (métricas brutas)</cinza>')
        print(f'   <cinza>• treinamento/training_metrics.jsonl (loss, hardware, tokens)</cinza>')
        if self.save_checkpoints:
            print(f'   <cinza>• checkpoint renaming (zero-padding: checkpoint-00001)</cinza>')
        
        trainer.model.config.use_cache = False
        
        return trainer

    def _verificar_labels_dataset(self, trainer) -> None:
        """Verifica se o dataset do trainer possui labels válidos após train_on_responses_only.
        
        Confere uma amostra de exemplos para garantir que ao menos alguns tokens
        tenham labels != -100 (necessário para computar loss não-NaN).
        """
        ds = trainer.train_dataset
        if ds is None or "labels" not in ds.column_names:
            print(f'   <cinza>ℹ️  Sem coluna labels no dataset — collator criará labels em tempo de execução.</cinza>')
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
            print(f'   <vermelho>❌ ALERTA: Todos os {n_check} exemplos verificados têm labels inteiramente -100!</vermelho>')
            print(f'      <vermelho>Isso causará loss=NaN e grad_norm=0.0 durante o treinamento.</vermelho>')
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
            print(f'   <amarelo>⚠️ {exemplos_vazios}/{n_check} exemplos com labels inteiramente -100 ({pct:.0f}%)</amarelo>')
            print(f'      Total: {total_validos}/{total_tokens} tokens com loss')
        else:
            pct_resp = total_validos / total_tokens * 100 if total_tokens > 0 else 0
            print(f'   <verde>✅ Labels verificados: {total_validos}/{total_tokens} tokens com loss ({pct_resp:.1f}%)</verde>')

    # ------------------------- checkpoint management --------------------- 
    def _find_latest_checkpoint(self) -> str:
        """Encontra o checkpoint mais recente na pasta de checkpoints.
        
        Returns:
            str: Caminho para o checkpoint mais recente ou None se não houver
        """
        # verifica se o resume está habilitado na configuração
        if not self._yaml_config.treinamento.resume_from_checkpoint:
            print("<amarelo>⚠️ Checkpoint ignorado por configuração (resume_from_checkpoint=False)</amarelo>")
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
            print(f"<verde>✅ Checkpoint encontrado: {latest_path} (step {latest_step})</verde>")
            self.log_processamento(f"Checkpoint encontrado: {latest_path} (step {latest_step})", "CHECKPOINT_FOUND")
            return latest_path
        else:
            print(f"⚠️  Checkpoint incompleto encontrado: {latest_path}")
            self.log_processamento(f"Checkpoint incompleto: {latest_path}", "CHECKPOINT_INCOMPLETE")
            return None

    # ------------------------- curriculum: preparação por etapa -----------
    def _aplicar_etapa_curriculum(self, step_index: int, etapa) -> None:
        """Configura parâmetros e dados para uma etapa específica do curriculum.
        
        - Aplica pace_epochs, learning_rate e max_seq_length da etapa
        - Para etapas > 0, troca o arquivo de divisão e recarrega os datasets
        - Se max_seq_length mudar entre etapas, recarrega model/tokenizer
        """
        treino = self._yaml_config.treinamento
        msl_anterior = treino.max_seq_length

        # Override de epochs com pace_epochs da etapa
        if etapa.pace_epochs > 0:
            treino.epochs = etapa.pace_epochs

        # Override de learning_rate se especificado
        if etapa.learning_rate > 0:
            treino.learning_rate = etapa.learning_rate

        # Override de max_seq_length se especificado
        if etapa.max_seq_length > 0:
            treino.max_seq_length = etapa.max_seq_length

        # Recarga dinâmica: se max_seq_length mudou, recarrega modelo e tokenizer
        msl_atual = treino.max_seq_length
        if msl_atual != msl_anterior and step_index > 0:
            logger.info(f"🔄 max_seq_length mudou: {msl_anterior} → {msl_atual}. Recarregando modelo...")
            print(f"🔄 Recarga dinâmica: max_seq_length {msl_anterior} → {msl_atual}")
            self.model, self.tokenizer = self._load_model()
            self.chat_handler = TreinarChatTemplate(self.tokenizer, self._yaml_config.modelo.base)
            self.tokenizer = self.chat_handler.tokenizer

        # Exibe informações de tokens da divisão da etapa
        info_tokens = self._yaml_config._ler_info_tokens_divisao(etapa.arquivo)
        if info_tokens:
            suficiente = "✅" if msl_atual >= info_tokens["max"] else "⚠️  INSUFICIENTE 🚩"
            print(f"   📊 Tokens etapa '{etapa.alias}': max={info_tokens['max']}, "
                  f"média={info_tokens['media']:.0f} → max_seq_length={msl_atual} {suficiente}")

        # Para etapas além da primeira, recarrega dados com o arquivo de divisão da etapa
        if step_index > 0 and etapa.arquivo:
            self._yaml_config.pastas.divisao.arquivo = etapa.arquivo
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

        if is_curriculum:
            print(f"<azul>[4/6] Iniciando treinamento com {total_etapas} etapas de curriculum…</azul>")
        else:
            print("<azul>[4/6] Iniciando treinamento…</azul>")

        # Contadores acumulados para métricas contínuas entre etapas do curriculum
        instancias_acumuladas = 0
        step_offset_global = 0
        epoch_offset_global = 0.0
        tokens_acumulados = 0

        for step_index, etapa_atual in enumerate(self._etapas):
            # --- Preparação da etapa ---
            if is_curriculum:
                logger.info(f"<azul>🔄 Etapa {step_index+1}/{total_etapas}: '{etapa_atual.alias}'</azul>")
                self._aplicar_etapa_curriculum(step_index, etapa_atual)

            # Constrói (ou reconstrói) o trainer com contexto da etapa
            self.trainer = self._build_trainer(
                etapa_index=step_index,
                etapa_alias=etapa_atual.alias,
                instancias_previas=instancias_acumuladas,
                step_offset=step_offset_global,
                epoch_offset=epoch_offset_global,
                tokens_previos=tokens_acumulados,
            )

            # Pipeline Universal: marca início da etapa
            self._tracker.iniciar_etapa(step_index=step_index, etapa=etapa_atual)
            tempo_inicio = time.time()

            # Valida o modelo antes do treinamento
            print("\n🔍 STATUS DO MODELO ANTES DO TREINAMENTO:")
            self.print_modelo_status()

            # Checkpoint resume apenas na primeira etapa (retomada de treino interrompido)
            checkpoint_path = self._find_latest_checkpoint() if step_index == 0 else None
            resume_from_checkpoint = checkpoint_path is not None

            try:
                if resume_from_checkpoint:
                    print(f"<azul>🔄 Tentando continuar treinamento a partir do checkpoint: {checkpoint_path}</azul>")
                    try:
                        train_stats = self.trainer.train(resume_from_checkpoint=checkpoint_path)
                        print("<verde>✅ Treinamento continuado com sucesso a partir do checkpoint</verde>")
                        self.log_processamento("Treinamento continuado com sucesso do checkpoint", "CHECKPOINT_SUCCESS")
                    except Exception as e:
                        error_msg = str(e)
                        print(f"<vermelho>❌ Erro ao continuar do checkpoint: {error_msg}</vermelho>")
                        print("<amarelo>🔄 Reiniciando treinamento do início...</amarelo>")

                        if any(keyword in error_msg.lower() for keyword in ['config', 'parameter', 'mismatch', 'size']):
                            self.log_processamento(f"Erro de configuração no checkpoint: {error_msg}", "CHECKPOINT_CONFIG_ERROR")
                        else:
                            self.log_processamento(f"Erro geral no checkpoint: {error_msg}", "CHECKPOINT_ERROR")

                        train_stats = self.trainer.train()
                else:
                    if step_index == 0:
                        print("<azul>🆕 Iniciando novo treinamento</azul>")
                    else:
                        print(f"<azul>▶ Iniciando etapa {step_index+1}/{total_etapas}: '{etapa_atual.alias}'</azul>")
                    train_stats = self.trainer.train()
            except Exception as e:
                self._tracker.marcar_falha(step_index=step_index, alias=etapa_atual.alias, erro=str(e))
                raise

            depois = _print_mem("DEPOIS")
            tempo_total = time.time() - tempo_inicio
            print("<verde>[5/6] Tempo de execução: {:.2f} s</verde>".format(train_stats.metrics["train_runtime"]))

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
            }

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

            # Grava o modelo antes do último eval (pode dar erro de memória no eval)
            self._save_model(stats=stats)

            # Garante um eval FINAL mesmo que já tenha havido evals em steps
            eval_loss = None
            if self.eval_ds:
                final_eval = self.trainer.evaluate()
                stats.update(final_eval)
                eval_loss = final_eval.get("eval_loss")
            self._save_model(stats=stats)

            # Pipeline Universal: registra conclusão da etapa com métricas
            self._tracker.finalizar_etapa(
                step_index=step_index,
                alias=etapa_atual.alias,
                train_loss=stats.get("training_loss"),
                eval_loss=eval_loss,
                global_step=stats.get("global_step"),
                tempo_segundos=round(tempo_total, 2),
                ds_train_len=stats.get("ds_train_len"),
                ds_eval_len=stats.get("ds_eval_len"),
            )

            # Atualiza contadores acumulados para a próxima etapa do curriculum
            n_gpus = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1
            effective_batch = (
                self._yaml_config.treinamento.batch_size
                * self._yaml_config.treinamento.grad_batch_size
                * n_gpus
            )
            step_offset_global += train_stats.global_step
            instancias_etapa = train_stats.global_step * effective_batch
            instancias_acumuladas += instancias_etapa
            tokens_acumulados += instancias_etapa * self._yaml_config.treinamento.max_seq_length
            # Usa a época real do Trainer (não pace_epochs configurado) para robustez
            # com early stopping ou datasets que não dividem exatamente em batches.
            # ceil() garante que a próxima etapa inicia em fronteira inteira de época.
            import math
            epocas_reais = train_stats.metrics.get(
                "epoch",
                etapa_atual.pace_epochs if etapa_atual.pace_epochs > 0 else self._yaml_config.treinamento.epochs
            )
            epoch_offset_global = math.ceil(epoch_offset_global + epocas_reais)

        if is_curriculum:
            logger.info(f"<verde>✅ TREINAMENTO COMPLETO — {total_etapas} etapas de curriculum finalizadas</verde>")
        
    # ------------------------- salvamento --------------------------------
    def _save_model(self, stats = None):
        out_dir = self._yaml_config.modelo.saida
        os.makedirs(out_dir, exist_ok=True)
        print(f"<azul>[6/6] Salvando modelo em {out_dir}…</azul>")
        
        # Salva o modelo (LoRA ou modelo completo)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        
        # Verifica se o modelo foi salvo corretamente
        adapter_config = os.path.join(out_dir, 'adapter_config.json')
        adapter_model = os.path.join(out_dir, 'adapter_model.safetensors')
        
        if os.path.exists(adapter_config):
            print(f"<verde>✅ Arquivo de configuração LoRA salvo: {adapter_config}</verde>")
            
        if os.path.exists(adapter_model):
            print(f"<verde>✅ Modelo LoRA salvo: {adapter_model}</verde>")
        elif os.path.exists(os.path.join(out_dir, 'pytorch_model.bin')):
            print(f"<verde>✅ Modelo PyTorch salvo: pytorch_model.bin</verde>")
        
        # Log detalhado do que foi salvo
        files_saved = []
        for file in os.listdir(out_dir):
            if file.endswith(('.json', '.safetensors', '.bin')):
                files_saved.append(file)
        
        self.log_processamento(f"Arquivos salvos em {out_dir}: {files_saved}", "MODEL_SAVED")
        
        if stats is not None:
            with open(os.path.join(self._yaml_config.modelo.saida, "metrics_summary.json"), "w") as fp:
                 json.dump(stats, fp, indent=2)
        print(r"<verde>Modelo salvo com sucesso \o/</verde>")

    def log_processamento(self, msg: str, titulo:str) -> None:
        ''' grava no arquivo de log com o nome _log_processamento_.txt dados importantes
            do processamento do treino como data, hora, parâmetros, dataset, etc
        '''
        arquivo = os.path.join(self._yaml_config.modelo.saida, f"_log_processamento_.txt")
        with open(arquivo, "a") as f:
            _msg = f"{msg}" if isinstance(msg,str) else json.dumps(msg, indent=2, ensure_ascii=False)
            if titulo:
                f.write(f"\n{'='*60}\n[{datetime.now()}] {str(titulo).upper()}\n{'-'*60}\n{_msg}\n{'='*60}\n")
            else:
                f.write(f"[{datetime.now()}] {_msg}\n")

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
                if self._yaml_config.tipo_entrada in TIPOS_BASEADOS_EM_PASTAS:
                    # Modo pastas: recarrega mensagens em memória
                    mensagens = self._yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo="treino")
                    
                    # Cria loader temporário com dados em memória
                    dataset_loader = LLMsDataset(
                        data=mensagens,
                        tokenizer=self.tokenizer,
                        max_seq_length=self._yaml_config.treinamento.max_seq_length
                    )
                else:
                    # Modo dataset: recarrega do arquivo
                    dataset_loader = LLMsDataset(
                        path=self._yaml_config.dataset.train_file,
                        prompt_col=self._yaml_config.dataset.train_prompt_col,
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
        """Valida se o modelo LoRA está carregado corretamente e retorna informações detalhadas."""
        info = {
            'modelo_tipo': type(self.model).__name__,
            'is_peft_model': False,
            'adapters_ativos': [],
            'parametros_treinaveis': 0,
            'parametros_totais': 0,
            'lora_detectado': False
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
        
        # Conta parâmetros treináveis
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        info['parametros_treinaveis'] = trainable_params
        info['parametros_totais'] = total_params
        info['percentual_treinavel'] = (trainable_params / total_params * 100) if total_params > 0 else 0
        
        return info

    def print_modelo_status(self):
        """Imprime o status detalhado do modelo."""
        info = self.validar_modelo_lora()
        
        print(f"\n{'='*60}")
        print(f"📊 STATUS DETALHADO DO MODELO")
        print(f"{'='*60}")
        print(f"Tipo do modelo: {info['modelo_tipo']}")
        print(f"É modelo PEFT: {info['is_peft_model']}")
        print(f"LoRA detectado: {info['lora_detectado']}")
        print(f"Parâmetros treináveis: {info['parametros_treinaveis']:,}")
        print(f"Parâmetros totais: {info['parametros_totais']:,}")
        print(f"Percentual treinável: {info['percentual_treinavel']:.4f}%")
        
        if info['adapters_ativos']:
            print(f"\n🔧 ADAPTADORES ATIVOS:")
            for adapter in info['adapters_ativos']:
                print(f"  - {adapter['nome']}: r={adapter['r']}, alpha={adapter['alpha']}")
                modules = adapter['target_modules']
                if isinstance(modules, str) and len(modules) > 50 and modules.startswith("(?:"):
                     modules_str = "Unsloth Default (Todos os módulos lineares)"
                else:
                     modules_str = str(modules)
                print(f"    Modules: {modules_str}")
        else:
            print(f"\n⚠️  NENHUM ADAPTADOR ATIVO DETECTADO")
        
        print(f"{'='*60}")
        
        # Log no arquivo
        self.log_processamento(info, "STATUS_MODELO_DETALHADO")
        
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
        "formatos": {
            "tipo_entrada": "dataset",
            "formato_saida": "texto",
        },
        "misc": {
            "log_level": "INFO",
            "env_chave_criptografia": "",  # Nome da variável de ambiente com chave Fernet
        },
        "dataset": {
            "train_file": "../dataset/data/dados_unificados_sm_treino.parquet",
            "train_prompt_col": "messages",
            "eval_file": "",
            "eval_prompt_col": "",
        },
        "modelo": {
            "base_model_name": "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
            "saida": "../modelos/gemma-3-12b-refleg20k-v01",
        },
        "treinamento": {
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
        _exibir_cabecalho_modelo, _verificar_modelo_treinado, _verificar_checkpoints_existem
    )
    
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=False)
    _exibir_cabecalho_modelo(yaml_config)
    
    # Status atual
    tem_modelo = _verificar_modelo_treinado(yaml_config)
    tem_checkpoints, qtd_checkpoints = _verificar_checkpoints_existem(yaml_config)
    
    logger.info("\n📊 STATUS ATUAL:")
    if tem_modelo:
        logger.info(f"   ✅ Modelo LoRA treinado encontrado")
    else:
        logger.info(f"   ❌ Nenhum modelo treinado encontrado")
    
    if tem_checkpoints:
        logger.info(f"   💾 {qtd_checkpoints} checkpoint(s) disponível(is)")
    else:
        logger.info(f"   💾 Nenhum checkpoint encontrado")
    
    # Menu
    logger.info("\n📋 AÇÕES DE TREINAMENTO:")
    logger.info("   1. treinar       - Iniciar ou continuar treinamento")
    logger.info("   2. reset+treinar - Limpar tudo e treinar do zero")
    logger.info("   3. reset         - Limpar treinamento atual")
    logger.info("   0. sair          - Cancelar e sair")
    
    try:
        escolha = input("\n❓ Digite o número ou nome da ação: ").strip().lower()
        
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

