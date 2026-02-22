#!/usr/bin/env python3

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Treinar Gemma‑3, Deepseek, Llhama, Qwen usando Unsloth 
        + TRL‑SFTTrainer de forma configurável por yaml.

Uso:
    python treinar_unsloth.py CONFIG.yaml [--debug]

* Se o YAML indicado não existir, um template é criado e o script
  termina — você revisa os valores e executa novamente.
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

# Configuração de path para permitir execução de qualquer diretório
# Detecta o diretório src automaticamente a partir da localização deste arquivo
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

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
from treinar_unsloth_util import YamlTreinamento, TIPO_ENTRADA_PASTAS, TIPO_ENTRADA_DATASET, calcular_rouge_l
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


class HardwareMetricsCallback(TrainerCallback):
    """
    Callback para registrar métricas contínuas de hardware durante o treinamento.
    
    Registra: CPU (uso %), RAM (usada/disponível GB), Disco (uso %), GPU (memória alocada/reservada GB)
    As métricas são salvas em arquivo JSONL separado para análise posterior.
    """
    
    def __init__(self, output_dir: str, intervalo_steps: int = 10):
        """
        Args:
            output_dir: Diretório onde salvar o arquivo de métricas
            intervalo_steps: Intervalo de steps entre registros (evita overhead excessivo)
        """
        self.output_dir = output_dir
        self.intervalo_steps = max(1, intervalo_steps)
        self.metrics_file = os.path.join(output_dir, "treinamento", "hardware_metrics.jsonl")
        self._ultimo_step = -1
        
        # Cria diretório e arquivo
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        open(self.metrics_file, "w").close()  # Limpa arquivo anterior
        
    def _registrar_metricas(self, step: int, epoch: float, fase: str = "train"):
        """Registra métricas de hardware no arquivo JSONL."""
        try:
            hardware = Util.dados_hardware(incluir_gpu=True)
            
            registro = {
                "timestamp": time.time(),
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "step": step,
                "epoch": round(epoch, 4) if epoch else 0,
                "fase": fase,
                # CPU
                "cpu_uso_%": hardware.get("cpu_uso_%", 0),
                "cpu_uso_processo_%": hardware.get("cpu_uso_processo_%", 0),
                # RAM
                "ram_usada_gb": hardware.get("mem_usada_gb", 0),
                "ram_disponivel_gb": hardware.get("mem_disponivel_gb", 0),
                "ram_uso_%": hardware.get("mem_uso_%", 0),
                # Disco
                "disco_uso_%": hardware.get("disco_uso_%", 0),
            }
            
            # GPU (pode ter múltiplas)
            gpu_info = hardware.get("gpu", {})
            if gpu_info.get("disponivel", False):
                gpus = gpu_info.get("gpus", [])
                for gpu in gpus:
                    idx = gpu.get("idx", 0)
                    registro[f"gpu{idx}_reservada_gb"] = gpu.get("mem_reservada_gb", 0)
                    registro[f"gpu{idx}_alocada_gb"] = gpu.get("mem_alocada_gb", 0)
                    registro[f"gpu{idx}_max_reservada_gb"] = gpu.get("mem_max_reservada_gb", 0)
            
            with open(self.metrics_file, "a") as fp:
                fp.write(json.dumps(registro, ensure_ascii=False) + "\n")
                
        except Exception as e:
            # Não interrompe o treinamento por erro de métricas
            pass
    
    def on_step_end(self, args, state, control, **kwargs):
        """Registra métricas a cada N steps."""
        if state.global_step % self.intervalo_steps == 0 and state.global_step != self._ultimo_step:
            self._ultimo_step = state.global_step
            self._registrar_metricas(state.global_step, state.epoch, "train")
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Registra métricas durante avaliação."""
        self._registrar_metricas(state.global_step, state.epoch, "eval")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Registra métricas no início do treinamento."""
        self._registrar_metricas(0, 0, "train_begin")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Registra métricas no final do treinamento."""
        self._registrar_metricas(state.global_step, state.epoch, "train_end")


class MetricsLoggerCallback(TrainerCallback):
    """
    Callback aprimorado para registrar métricas detalhadas de treinamento e validação.
    
    Registra: loss, learning_rate, grad_norm, e métricas de avaliação (eval_loss)
    Salva em arquivo JSONL separado com informações adicionais de contexto.
    """
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Diretório onde salvar o arquivo de métricas
        """
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, "treinamento", "training_metrics.jsonl")
        self._train_start_time = None
        self._best_eval_loss = float('inf')
        self._train_losses = []  # Para calcular média móvel
        
        # Cria diretório e arquivo
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        open(self.metrics_file, "w").close()  # Limpa arquivo anterior
        
    def _registrar(self, registro: dict):
        """Salva registro no arquivo JSONL."""
        try:
            with open(self.metrics_file, "a") as fp:
                fp.write(json.dumps(registro, ensure_ascii=False) + "\n")
        except Exception:
            pass
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Marca início do treinamento."""
        self._train_start_time = time.time()
        self._registrar({
            "event": "train_begin",
            "timestamp": self._train_start_time,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_steps": state.max_steps,
            "num_epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
            "grad_accum_steps": args.gradient_accumulation_steps,
        })
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Registra métricas de treinamento a cada log."""
        if not logs:
            return
            
        registro = {
            "event": "log",
            "timestamp": time.time(),
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else 0,
            "elapsed_seconds": time.time() - self._train_start_time if self._train_start_time else 0,
        }
        
        # Métricas de treinamento
        if "loss" in logs:
            registro["train_loss"] = round(logs["loss"], 6)
            self._train_losses.append(logs["loss"])
            # Média móvel das últimas 10 perdas
            if len(self._train_losses) >= 10:
                registro["train_loss_avg_10"] = round(sum(self._train_losses[-10:]) / 10, 6)
        
        if "learning_rate" in logs:
            registro["learning_rate"] = logs["learning_rate"]
            
        if "grad_norm" in logs:
            registro["grad_norm"] = round(logs["grad_norm"], 6)
        
        # Métricas de avaliação
        if "eval_loss" in logs:
            registro["eval_loss"] = round(logs["eval_loss"], 6)
            if logs["eval_loss"] < self._best_eval_loss:
                self._best_eval_loss = logs["eval_loss"]
                registro["is_best_eval"] = True
        
        # Progresso
        if state.max_steps > 0:
            registro["progress_%"] = round(state.global_step / state.max_steps * 100, 2)
        
        self._registrar(registro)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Registra métricas completas de avaliação."""
        if not metrics:
            return
            
        registro = {
            "event": "evaluate",
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else 0,
        }
        
        # Copia todas as métricas de avaliação
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                registro[key] = round(value, 6) if isinstance(value, float) else value
        
        self._registrar(registro)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Registra resumo final do treinamento."""
        elapsed = time.time() - self._train_start_time if self._train_start_time else 0
        
        registro = {
            "event": "train_end",
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_steps": state.global_step,
            "final_epoch": round(state.epoch, 4) if state.epoch else 0,
            "total_time_seconds": round(elapsed, 2),
            "total_time_formatted": f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s",
            "best_eval_loss": round(self._best_eval_loss, 6) if self._best_eval_loss != float('inf') else None,
            "final_train_loss_avg": round(sum(self._train_losses[-10:]) / len(self._train_losses[-10:]), 6) if self._train_losses else None,
        }
        
        self._registrar(registro)

# ---------------------------------------------------------------------------
# Callback para renomear checkpoints com zero-padding
# ---------------------------------------------------------------------------

class CheckpointRenameCallback(TrainerCallback):
    """
    Callback para renomear checkpoints com zero-padding.
    
    Transforma: checkpoint-8 -> checkpoint-00008
    Isso permite ordenação alfabética correta e evita confusão.
    """
    
    def __init__(self, checkpoint_base_dir: str, padding: int = 5):
        self.checkpoint_base_dir = checkpoint_base_dir
        self.padding = padding
    
    def on_save(self, args, state, control, **kwargs):
        """Renomeia o checkpoint salvo para usar zero-padding."""
        if not state.global_step:
            return
            
        # Caminho esperado do checkpoint original
        original_name = f"checkpoint-{state.global_step}"
        padded_name = f"checkpoint-{state.global_step:0{self.padding}d}"
        
        original_path = os.path.join(self.checkpoint_base_dir, original_name)
        padded_path = os.path.join(self.checkpoint_base_dir, padded_name)
        
        # Se o diretório original existe e o padded não existe, renomeia
        if os.path.exists(original_path) and not os.path.exists(padded_path):
            try:
                os.rename(original_path, padded_path)
                logger.debug(f"Checkpoint renomeado: {original_name} -> {padded_name}")
            except Exception as e:
                logger.warning(f"Erro ao renomear checkpoint: {e}")
        elif os.path.exists(padded_path):
            # Já existe com zero-padding, provavelmente foi renomeado anteriormente
            logger.warning(f"Erro ao renomear checkpoint: já existe com zero-padding {padded_name}")

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
        
        # Carrega modelo e tokenizer
        self.model, self.tokenizer = self._load_model()
        
        # Gerenciador de templates de chat
        self.chat_handler = TreinarChatTemplate(self.tokenizer, self._yaml_config.modelo.base)
        self.tokenizer = self.chat_handler.tokenizer
        
        # Carrega datasets baseado no tipo de entrada
        if self._yaml_config.tipo_entrada == TIPO_ENTRADA_PASTAS:
            # Modo pastas: carrega de arquivos pareados
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
        
        return dataset_loader.dataset
        
    # ------------------------- modelo ------------------------------------
    def _load_model(self):
        print("[1/6] Carregando modelo base…")
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
                print(f'🔄 Carregando modelo LoRA já treinado de {lora_model_path}...')
                try:
                    # Carrega o modelo LoRA já treinado diretamente
                    model, tokenizer = FastModel.from_pretrained(
                        model_name=lora_model_path,  # Carrega da pasta do modelo treinado
                        max_seq_length=self._yaml_config.treinamento.max_seq_length,
                        load_in_4bit=nbits == 4,
                        load_in_8bit=nbits == 8,
                        device_map="auto",
                    )
                    print(f'✅ Modelo LoRA treinado carregado com sucesso!')
                    lora_ok = True
                    
                    # Log de informações do modelo carregado
                    self.log_processamento(f"Modelo LoRA carregado de: {lora_model_path}", "LORA_LOADED")
                    
                except Exception as e:
                    print(f'❌ Erro ao carregar modelo LoRA treinado: {e}')
                    traceback.print_exc()
                    print('Tentando carregar modelo base e aplicar LoRA...')
                    time.sleep(2)
        else:
             print(f'ℹ️  Opção --base ativada: Ignorando busca por modelo LoRA treinado.')
        
        # Se não conseguiu carregar o LoRA ou não existe ou force_base=True, carrega modelo base
        if not lora_ok:
            print(f'🔄 Carregando modelo base: {self._yaml_config.modelo.base}...')
            model, tokenizer = FastModel.from_pretrained(
                model_name=self._yaml_config.modelo.base,
                max_seq_length=self._yaml_config.treinamento.max_seq_length,
                load_in_4bit=nbits == 4,
                load_in_8bit=nbits == 8,
                full_finetuning=self._yaml_config.lora.r in (0,None,False) or self.force_base # Se force_base, não prepara para finetuning
            )
            
            # Se usar LoRA, aplica as configurações (Exceto se force_base=True)
            if not self.force_base and self._yaml_config.lora.r not in (0,None,False):
                print(f'🔄 Aplicando LoRA r={self._yaml_config.lora.r} ao modelo base ...')
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
                print(f'ℹ️  Opção --base ativada: Não aplicando adaptadores LoRA.')
        # Template agora é aplicado pelo TreinarChatTemplate após o carregamento
        model.print_trainable_parameters()
        
        # Log detalhado do modelo carregado
        model_type = type(model).__name__
        is_peft_model = hasattr(model, 'peft_config') or hasattr(model, 'base_model')
        
        print(f"\n📊 MODELO CARREGADO:")
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
    def debug_info(cls, cfg_path: str):
        """Exibe informações detalhadas de debug sobre configuração e datasets."""
        print("="*80)
        print(">> MODO INFO / DEBUG - INFORMAÇÕES DE CONFIGURAÇÃO E DATASET")
        print("="*80)
        
        # Carrega configuração usando YamlTreinamento
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
        
        # Modo pastas: mostra arquivos pareados
        if yaml_config.tipo_entrada == TIPO_ENTRADA_PASTAS:
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
    def _build_trainer(self) -> SFTTrainer:
        print("[3/6] Configurando trainer…")
        
        # === Formatação do Dataset (garante coluna 'text') ===
        # Define num_proc seguro
        import os
        n_proc = max(1, (os.cpu_count() or 2) // 2)

        if "text" not in self.train_ds.column_names:
            self.train_ds = self.chat_handler.formatar_dataset_coluna_text(self.train_ds, num_proc=n_proc)
            
        if self.eval_ds and "text" not in self.eval_ds.column_names:
            self.eval_ds = self.chat_handler.formatar_dataset_coluna_text(self.eval_ds, num_proc=n_proc)
            
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
           print(f' - avaliando a cada {eval_steps} steps...')
        
        log_steps = eval_steps if isinstance(eval_steps, int) and eval_steps > 0 else 50
        
        if self.save_checkpoints:
            print(f' - gravando checkpoints a cada {log_steps} steps')
        
        # Log train_on_responses_only
        if treino_cfg.train_on_responses_only:
            print(f' - train_on_responses_only ATIVADO (treina apenas nas respostas do assistant)')

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
        # Aplica train_on_responses_only se configurado
        if treino_cfg.train_on_responses_only:
            self.trainer = self.chat_handler.aplicar_train_on_responses_only(self.trainer)
        
        # Configura diretório de saída
        output_dir = self._yaml_config.modelo.saida
        os.makedirs(output_dir, exist_ok=True)
        
        # === CALLBACKS DE MÉTRICAS ===
        
        # 1. JsonLogger (métricas brutas em metrics_stream.jsonl)
        jsonl = os.path.join(output_dir, "metrics_stream.jsonl")
        if os.path.isfile(jsonl):
            os.remove(jsonl)
        trainer.add_callback(JsonLoggerCallback(jsonl))
        
        # 2. HardwareMetricsCallback (métricas de RAM, GPU, CPU, Disco)
        # Registra a cada 10 steps para não sobrecarregar
        trainer.add_callback(HardwareMetricsCallback(output_dir, intervalo_steps=10))
        
        # 3. MetricsLoggerCallback (métricas detalhadas de treinamento/validação)
        trainer.add_callback(MetricsLoggerCallback(output_dir))
        
        # 4. CheckpointRenameCallback (renomeia checkpoints com zero-padding)
        if self.save_checkpoints:
            chkpt_dir = os.path.join(self._yaml_config.modelo.saida, "chkpt")
            os.makedirs(chkpt_dir, exist_ok=True)
            trainer.add_callback(CheckpointRenameCallback(chkpt_dir))
        
        print(f' - callbacks de métricas configurados:')
        print(f'   • metrics_stream.jsonl (métricas brutas)')
        print(f'   • treinamento/hardware_metrics.jsonl (RAM, GPU, CPU, Disco)')
        print(f'   • treinamento/training_metrics.jsonl (loss, lr, eval_loss)')
        if self.save_checkpoints:
            print(f'   • checkpoint renaming (zero-padding: checkpoint-00001)')
        
        trainer.model.config.use_cache = False
        
        return trainer

    # ------------------------- checkpoint management --------------------- 
    def _find_latest_checkpoint(self) -> str:
        """Encontra o checkpoint mais recente na pasta de checkpoints.
        
        Returns:
            str: Caminho para o checkpoint mais recente ou None se não houver
        """
        # verifica se o resume está habilitado na configuração
        if not self._yaml_config.treinamento.resume_from_checkpoint:
            print("⚠️ Checkpoint ignorado por configuração (resume_from_checkpoint=False)")
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
            print(f"✅ Checkpoint encontrado: {latest_path} (step {latest_step})")
            self.log_processamento(f"Checkpoint encontrado: {latest_path} (step {latest_step})", "CHECKPOINT_FOUND")
            return latest_path
        else:
            print(f"⚠️  Checkpoint incompleto encontrado: {latest_path}")
            self.log_processamento(f"Checkpoint incompleto: {latest_path}", "CHECKPOINT_INCOMPLETE")
            return None

    # ------------------------- execução ----------------------------------
    def train(self):
        # Inicializa o trainer se necessário (lazy init)
        if self.trainer is None:
            self.trainer = self._build_trainer()
            
        antes = _print_mem("ANTES")
        print("[4/6] Iniciando treinamento…")
        
        # Valida o modelo antes do treinamento
        print("\n🔍 STATUS DO MODELO ANTES DO TREINAMENTO:")
        self.print_modelo_status()
        
        # verifica se existe checkpoint para continuar
        checkpoint_path = self._find_latest_checkpoint()
        resume_from_checkpoint = checkpoint_path is not None
        
        if resume_from_checkpoint:
            print(f"🔄 Tentando continuar treinamento a partir do checkpoint: {checkpoint_path}")
            try:
                train_stats = self.trainer.train(resume_from_checkpoint=checkpoint_path)
                print("✅ Treinamento continuado com sucesso a partir do checkpoint")
                self.log_processamento("Treinamento continuado com sucesso do checkpoint", "CHECKPOINT_SUCCESS")
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Erro ao continuar do checkpoint: {error_msg}")
                print("🔄 Reiniciando treinamento do início...")
                
                # identifica tipos comuns de erro de checkpoint
                if any(keyword in error_msg.lower() for keyword in ['config', 'parameter', 'mismatch', 'size']):
                    self.log_processamento(f"Erro de configuração no checkpoint: {error_msg}", "CHECKPOINT_CONFIG_ERROR")
                else:
                    self.log_processamento(f"Erro geral no checkpoint: {error_msg}", "CHECKPOINT_ERROR")
                
                # reinicia o treinamento do zero
                train_stats = self.trainer.train()
        else:
            print("🆕 Iniciando novo treinamento")
            train_stats = self.trainer.train()
            
        depois = _print_mem("DEPOIS")
        print("[5/6] Tempo de execução: {:.2f} s".format(train_stats.metrics["train_runtime"]))
        
        # Valida o modelo após o treinamento
        print("\n🔍 STATUS DO MODELO APÓS O TREINAMENTO:")
        info_modelo = self.print_modelo_status()
        
        # 2) dicionário de tudo que interessa
        stats = {
            **train_stats.metrics,                  # train_loss, train_runtime, etc.
            "global_step":       train_stats.global_step,
            "training_loss":     train_stats.training_loss,
            "mem_gpu_before":    antes,
            "mem_gpu_after":     depois,
            "ds_train_len" : len(self.train_ds),
            "ds_eval_len" : len(self.eval_ds) if self.eval_ds else 0,
            "modelo_info": info_modelo,  # adiciona informações do modelo
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

        # grava o modelo antes do ultimo eval, pode dar erro de memória no eval    
        self._save_model(stats=stats)
        # 3) garante um eval FINAL mesmo que já tenha havido evals em steps
        if self.eval_ds:
            final_eval = self.trainer.evaluate()     # roda avaliação no eval_dataset
            stats.update(final_eval)                 # adiciona eval_loss, eval_runtime …        self._save_model(stats = stats)
        
    # ------------------------- salvamento --------------------------------
    def _save_model(self, stats = None):
        out_dir = self._yaml_config.modelo.saida
        os.makedirs(out_dir, exist_ok=True)
        print(f"[6/6] Salvando modelo em {out_dir}…")
        
        # Salva o modelo (LoRA ou modelo completo)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        
        # Verifica se o modelo foi salvo corretamente
        adapter_config = os.path.join(out_dir, 'adapter_config.json')
        adapter_model = os.path.join(out_dir, 'adapter_model.safetensors')
        
        if os.path.exists(adapter_config):
            print(f"✅ Arquivo de configuração LoRA salvo: {adapter_config}")
            
        if os.path.exists(adapter_model):
            print(f"✅ Modelo LoRA salvo: {adapter_model}")
        elif os.path.exists(os.path.join(out_dir, 'pytorch_model.bin')):
            print(f"✅ Modelo PyTorch salvo: pytorch_model.bin")
        
        # Log detalhado do que foi salvo
        files_saved = []
        for file in os.listdir(out_dir):
            if file.endswith(('.json', '.safetensors', '.bin')):
                files_saved.append(file)
        
        self.log_processamento(f"Arquivos salvos em {out_dir}: {files_saved}", "MODEL_SAVED")
        
        if stats is not None:
            with open(os.path.join(self._yaml_config.modelo.saida, "metrics_summary.json"), "w") as fp:
                 json.dump(stats, fp, indent=2)
        print(r"Modelo salvo com sucesso \o/")

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
                if self._yaml_config.tipo_entrada == TIPO_ENTRADA_PASTAS:
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


def _verificar_modelo_treinado(yaml_config: YamlTreinamento) -> bool:
    """
    Verifica se existe modelo LoRA treinado na pasta de saída.
    
    Returns:
        True se existir modelo treinado, False caso contrário
    """
    output_dir = yaml_config.modelo.saida
    arq_lora = os.path.join(output_dir, 'adapter_config.json')
    arq_model = os.path.join(output_dir, 'adapter_model.safetensors')
    arq_pytorch = os.path.join(output_dir, 'pytorch_model.bin')
    
    return os.path.exists(arq_lora) and (os.path.exists(arq_model) or os.path.exists(arq_pytorch))


def _perguntar_usar_modelo_base() -> bool:
    """
    Pergunta ao usuário se deseja usar o modelo base para predição.
    
    Returns:
        True para continuar com modelo base, False para cancelar
    """
    logger.warning("\n⚠️  ATENÇÃO: Não foi encontrado modelo LoRA treinado na pasta de saída.")
    logger.info("O modelo base será carregado para predição (sem fine-tuning).\n")
    
    try:
        resposta = input("Deseja continuar com o modelo base? [s/N]: ").strip().lower()
        return resposta in ('s', 'sim', 'y', 'yes')
    except (KeyboardInterrupt, EOFError):
        logger.info("\nOperação cancelada pelo usuário.")
        return False


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune LLMs (Gemma, Qwen, Llama, DeepSeek) com configuração YAML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ações disponíveis:
  --info            Informações gerais do treinamento e modelo
  --stats           Relatório estatístico com tokens de entrada/saída e boxplots
  --treinar         Inicia ou reinicia o treinamento
  --reset           Limpa o treinamento atual (com confirmação)
  --predict         Gera predições para todos os subsets (treino, validacao, teste)
  --predict-treino  Gera predições apenas do subset de treino
  --predict-validacao  Gera predições apenas do subset de validação
  --predict-teste   Gera predições apenas do subset de teste
  
Sem ação: modo interativo que pergunta qual ação executar.

Exemplos:
  %(prog)s config.yaml              # Modo interativo
  %(prog)s config.yaml --info       # Informações detalhadas
  %(prog)s config.yaml --treinar    # Inicia treinamento
  %(prog)s config.yaml --reset --treinar  # Limpa e treina
  %(prog)s config.yaml --predict    # Gera predições de todos os subsets
  %(prog)s config.yaml --modelo 5   # Testa 5 predições interativas
"""
    )
    parser.add_argument("config", help="Arquivo YAML com as configurações.")
    
    # Grupo de ações (mutuamente exclusivas, exceto reset+treinar)
    parser.add_argument("--info", action="store_true", 
                        help="Modo info: exibe estrutura do dataset e configuração sem treinar")
    parser.add_argument("--stats", action="store_true",
                        help="Gera relatório estatístico de tokens com boxplots")
    parser.add_argument("--treinar", action="store_true",
                        help="Inicia ou continua o treinamento")
    parser.add_argument("--reset", action="store_true",
                        help="Limpa treinamento atual (checkpoints e modelo LoRA)")
    
    # Ações de predição
    parser.add_argument("--predict", action="store_true",
                        help="Gera predições para todos os subsets (treino, validacao, teste)")
    parser.add_argument("--predict-treino", action="store_true",
                        help="Gera predições apenas do subset de treino")
    parser.add_argument("--predict-validacao", action="store_true",
                        help="Gera predições apenas do subset de validação")
    parser.add_argument("--predict-teste", action="store_true",
                        help="Gera predições apenas do subset de teste")
    
    # Ação de merge
    parser.add_argument("--merge", action="store_true",
                        help="Exporta e realiza merge do modelo treinado com o base")
    parser.add_argument("--quant", type=str, default=None,
                        help="Método de quantização para merge: 16bit, 4bit, q4_k_m, q8_0, f16 (padrão: interativo ou 16bit)")
    
    # Opções modificadoras
    parser.add_argument("--base", action="store_true",
                        help="Força o uso do modelo base (ignora LoRA treinado) para info e predict")
    
    # Opções extras (mantidas do CLI anterior)
    parser.add_argument("--modelo", type=int, nargs='?', const=1, 
                        help="Modo teste: exibe exemplos de prompt e resposta do modelo treinado (padrão: 1 exemplo)")
    parser.add_argument("--log-level", type=str, default=None, 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="Nível de log (sobrescreve misc.log_level do YAML)")
    

    # Retrocompatibilidade: mantém --debug como alias para --info
    parser.add_argument("--debug", action="store_true", 
                        help=argparse.SUPPRESS)  # Oculto, usa --info

    # Injeção de dicas
    parser.add_argument("--dicas", action="store_true", 
                        help="Injeta comentários de dicas no YAML de configuração")

    
    args = parser.parse_args()

    cfg_path = args.config or "./exemplo.yaml"
    
    # Carrega configuração do YAML para determinar log_level padrão
    log_level_padrao = "INFO"
    if os.path.exists(cfg_path):
        try:
            yaml_config = YamlTreinamento(cfg_path, validar_caminhos=False)
            log_level_padrao = yaml_config.misc.log_level
        except Exception:
            pass  # Usa padrão INFO se falhar
    

    # Log level configuration...
    nivel_log = args.log_level if args.log_level else log_level_padrao
    configurar_logging(nivel=nivel_log)
    
    logger.debug(f"Log level configurado: {nivel_log} (CLI: {args.log_level}, YAML: {log_level_padrao})")


    # Injeção de dicas antes de qualquer validação de CUDA ou criação de template
    if args.dicas:
        from treinar_unsloth_actions import executar_injetar_dicas
        executar_injetar_dicas(cfg_path)

    # informações sobre CUDA

    if torch.cuda.is_available():
        logger.info(f"CUDA disponível — {torch.cuda.device_count()} GPU(s) detectada(s)")
    else:
        logger.warning("CUDA não disponível — treinamento será na CPU (muito mais lento)")

    # Cria YAML template se não existir
    if not os.path.exists(cfg_path):
        _create_default_cfg(cfg_path)
        logger.info(
            f"Arquivo de configuração criado em '{cfg_path}'.\n"
            "Revise os parâmetros e execute novamente para iniciar o treinamento."
        )
        sys.exit(0)

    # Importa módulo de ações
    from treinar_unsloth_actions import (
        executar_info, executar_stats, executar_treinar, 
        executar_reset, executar_predict, executar_merge, modo_interativo, executar_acao
    )
    
    # Retrocompatibilidade: --debug -> --info
    if args.debug:
        args.info = True
        logger.warning("⚠️  O parâmetro --debug está deprecado. Use --info.")



    # Modo --modelo: teste de predições (fluxo separado)
    if args.modelo:
        # Carrega apenas a configuração YAML para verificar modelo
        yaml_config = YamlTreinamento(cfg_path, validar_caminhos=False)
        
        usar_base = getattr(args, 'base', False)
        
        if not usar_base:
            if not _verificar_modelo_treinado(yaml_config):
                if not _perguntar_usar_modelo_base():
                    logger.info("Operação cancelada. Execute um treinamento antes de testar o modelo.")
                    sys.exit(0)
                else:
                    logger.info("Continuando com modelo base (sem fine-tuning)...\n")
                    usar_base = True # Usuário confirmou usar base na pergunta
            else:
                logger.info(f"✅ Modelo LoRA treinado encontrado em: {yaml_config.modelo.saida}")
        else:
            logger.info("ℹ️  Opção --base ativada: Forçando uso do modelo base.")

        trainer = LLMsTrainer(cfg_path, force_base=usar_base)
        n_exemplos = args.modelo if isinstance(args.modelo, int) else 1
        resultado = trainer.testar_predicoes(n_exemplos=n_exemplos, temperatura=0.2, max_new_tokens=512)
        
        # Exibe resumo das métricas de memória
        if resultado.get('metricas_memoria'):
            metricas = resultado['metricas_memoria']
            logger.info("\n📊 RESUMO DE USO DE MEMÓRIA:")
            if 'ram' in metricas:
                logger.info(f"   RAM: máx={metricas['ram'].get('max_gb', 0):.1f} GB, média={metricas['ram'].get('media_gb', 0):.1f} GB")
            if 'gpu' in metricas and metricas['gpu'].get('num_gpus', 0) > 0:
                logger.info(f"   GPU: máx={metricas['gpu'].get('max_gb', 0):.1f} GB, média={metricas['gpu'].get('media_gb', 0):.1f} GB ({metricas['gpu'].get('num_gpus', 0)} GPU(s))")
        
        sys.exit(0)

    # Determina subsets para predict (se houver)
    predict_subsets = None
    has_predict = False
    
    if getattr(args, 'predict_treino', False):
        predict_subsets = predict_subsets or []
        predict_subsets.append('treino')
        has_predict = True
    if getattr(args, 'predict_validacao', False):
        predict_subsets = predict_subsets or []
        predict_subsets.append('validacao')
        has_predict = True
    if getattr(args, 'predict_teste', False):
        predict_subsets = predict_subsets or []
        predict_subsets.append('teste')
        has_predict = True
    if args.predict:
        has_predict = True
        # predict_subsets = None significa todos
    
    # Determina ação a executar
    if args.info:
        executar_info(cfg_path)
    elif args.stats:
        executar_stats(cfg_path)
    elif args.merge:
        executar_merge(cfg_path, quantizacao=args.quant)
    else:
        executed_action = False
        
        # 1. Executa Treinamento (com ou sem Reset)
        if args.treinar:
            executar_treinar(cfg_path, reset=args.reset)
            executed_action = True
        # 2. Se não treinou mas pediu reset
        elif args.reset:
            executar_reset(cfg_path, confirmar=True)
            executed_action = True
            
        # 3. Executa Predição (pode ser executado após treino ou reset)
        if has_predict:
            executar_predict(cfg_path, subsets=predict_subsets, usar_base=args.base)
            executed_action = True
            
        # 4. Se nenhuma ação foi executada via CLI, entra no modo interativo
        if not executed_action:
            acao = modo_interativo(cfg_path)
            if acao:
                executar_acao(acao, cfg_path)



if __name__ == "__main__":
    # Carrega .env do diretório src (funciona de qualquer pasta)
    UtilEnv.carregar_env(pastas=[_SRC_DIR, './', '../', '../src/'])
    _cli()

