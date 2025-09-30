#!/usr/bin/env python3
"""
Autor: Luiz Anisio 05/2025 v005

Treinar Gemma‑3, Deepseek, Llhama, Qwen usando Unsloth 
        + TRL‑SFTTrainer de forma configurável por yaml.

Uso:
    python treinar_unsloth.py CONFIG.yaml [--gpu 1] [--debug]

* Se o YAML indicado não existir, um template é criado e o script
  termina — você revisa os valores e executa novamente.
* O parâmetro opcional **--gpu IDX** permite escolher a GPU CUDA a ser
  utilizada (padrão: 0).  O índice é aplicado via `torch.cuda.set_device()`
  logo no início do programa.
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
import yaml
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, CHAT_TEMPLATES, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from transformers import GenerationConfig
import numpy as np
from datetime import datetime
from copy import deepcopy

# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------

def _print_mem(tag: str, device_idx: int) -> dict:
    """Exibe estatísticas de memória GPU para depuração rápida."""
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA não disponível.")
        return
    torch.cuda.synchronize(device_idx)
    stats = torch.cuda.get_device_properties(device_idx)
    total = round(stats.total_memory / 1024 / 1024 / 1024, 3)
    reserved = round(torch.cuda.max_memory_reserved(device_idx) / 1024 / 1024 / 1024, 3)
    print(f"[{tag}] GPU[{device_idx}] {stats.name} | reservada: {reserved} GB / total: {total} GB")
    return {'mem_total_gb': total, 'mem_reserved_gm': reserved, 'gpu_idx': device_idx, 'name': stats.name}

class JsonLoggerCallback(TrainerCallback):
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

# ---------------------------------------------------------------------------
# classe principal
# ---------------------------------------------------------------------------

class LLMsTrainer:
    """Encapsula o fluxo de fine‑tuning de LLMs com LoRA e Unsloth."""

    REQUIRED_KEYS = {
        "dataset_train_path",
        "base_model_name",
        "output_dir",
        "batch_size",
        "grad_batch_size",
        "num_train_epochs",
        "max_seq_length",
        "lora_r",
    }

    def __init__(self, cfg_path: str, device_idx: int):
        self.device_idx = device_idx
        self.cfg: Dict[str, Any] = self._load_cfg(cfg_path)
        # cria a pasta de saída se não existir
        os.makedirs(self.cfg.get("output_dir", "./saida"), exist_ok=True)
        self._validate_cfg()
        self.model, self.tokenizer = self._load_model()
        self._get_chat_template() # carrega o chat template e identifica se usa type ou str
       # se for json, não precisa de coluna
        self.train_ds = self._load_split(
            self.cfg["dataset_train_path"], self.cfg.get("train_prompt_col"), split="treino"
        )
        # registra o primeiro registro no log
        self.log_processamento(self.train_ds[0], titulo="Primeiro registro do dataset de treino")
        self.eval_ds = None
        if self.cfg.get("dataset_eval_path"):
            self.eval_ds = self._load_split(
                self.cfg["dataset_eval_path"],
                self.cfg.get("eval_prompt_col", self.cfg.get("train_prompt_col")),
                split="teste",
            )
            self.log_processamento(self.eval_ds[0], titulo="Primeiro registro do dataset de teste")
        self.save_checkpoints = self.cfg.get('save_checkpoints','') in {1,'1','True',True,'true','sim','Sim','SIM'}
        self.trainer = self._build_trainer()

    # ------------------------- controle no colab ------------------------------
    @classmethod
    def verifica_versao(cls):
        print(f'JsonAnalise carregado corretamente em {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}!')
        
    # ------------------------- configuração ------------------------------
    @classmethod
    def _load_cfg(cls, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}

    def _validate_cfg(self) -> None:
        missing = self.REQUIRED_KEYS - self.cfg.keys()
        if missing:
            raise ValueError(f"Parâmetros obrigatórios ausentes no YAML: {sorted(missing)}")
        if self.cfg.get("dataset_eval_path") and not self.cfg.get("eval_prompt_col"):
            raise ValueError("Se 'dataset_eval_path' for definido, informe 'eval_prompt_col'.")
        
    # ------------------------- modelo ------------------------------------
    def _load_model(self):
        print("[1/6] Carregando modelo base… (GPU {} )".format(self.device_idx))
        nbits = int(self.cfg.get("nbits", 0))
        
        # Verifica se existe modelo LoRA já treinado
        lora_model_path = self.cfg['output_dir']
        arq_lora = os.path.join(lora_model_path, 'adapter_config.json')
        arq_model = os.path.join(lora_model_path, 'adapter_model.safetensors')
        
        # Verifica se é um modelo LoRA completo (não apenas um checkpoint)
        is_trained_lora = (os.path.exists(arq_lora) and 
                          (os.path.exists(arq_model) or 
                           os.path.exists(os.path.join(lora_model_path, 'pytorch_model.bin'))))
        
        lora_ok = False
        if is_trained_lora:
            print(f'🔄 Carregando modelo LoRA já treinado de {lora_model_path}...')
            try:
                # Carrega o modelo LoRA já treinado diretamente
                model, tokenizer = FastModel.from_pretrained(
                    model_name=lora_model_path,  # Carrega da pasta do modelo treinado
                    max_seq_length=int(self.cfg["max_seq_length"]),
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
        
        # Se não conseguiu carregar o LoRA ou não existe, carrega modelo base
        if not lora_ok:
            print(f'🔄 Carregando modelo base: {self.cfg["base_model_name"]}...')
            model, tokenizer = FastModel.from_pretrained(
                model_name=self.cfg["base_model_name"],
                max_seq_length=int(self.cfg["max_seq_length"]),
                load_in_4bit=nbits == 4,
                load_in_8bit=nbits == 8,
                full_finetuning=self.cfg["lora_r"] in (0,None,False)
            )
            
            # Se usar LoRA, aplica as configurações
            if self.cfg["lora_r"] not in (0,None,False):
                print(f'🔄 Aplicando LoRA r={self.cfg["lora_r"]} ao modelo base ...')
                model = FastModel.get_peft_model(
                    model,
                    finetune_vision_layers=False,
                    finetune_language_layers=True,
                    finetune_attention_modules=True,
                    finetune_mlp_modules=True,
                    r=int(self.cfg["lora_r"]),
                    lora_alpha=int(self.cfg.get("lora_alpha", self.cfg["lora_r"])),
                    lora_dropout=float(self.cfg.get("lora_dropout", 0.0)),
                    bias="none",
                    random_state=3407,
                    device_map="auto",
                )
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
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
        
        self.log_processamento(self.cfg, titulo="Configuração do treinamento")
        self.log_processamento(str(model), titulo="Resumo do modelo")
        self.log_processamento(f"Tipo do modelo: {model_type} | PEFT: {is_peft_model} | LoRA OK: {lora_ok}", titulo="Status do modelo")
        self.log_processamento(tokenizer.chat_template, titulo="Template do tokenizer")
        return model, tokenizer

    def _get_chat_template(self):
        """Configura o chat template adequado para o modelo."""
        if getattr(self.tokenizer, "chat_template", None):
            return  # já tem template definido (modelos "instruct" costumam trazer)
        _nm_teste = self.cfg["base_model_name"].replace('-','').lower()
        if 'gemma' in _nm_teste: key = 'gemma'
        elif 'qwen2' in _nm_teste: key = '"qwen-2.5"'
        elif 'qwen3' in _nm_teste: key = 'chatml'
        elif 'llama33' in _nm_teste:  key = 'llama-3.3'
        elif 'llama32' in _nm_teste:  key = 'llama-3.2'
        elif 'llama31' in _nm_teste:  key = 'llama-3.1'
        elif 'llama3' in _nm_teste:  key = 'llama-3'
        elif 'llama' in _nm_teste:  key = 'llama'
        else: key = 'chatml'
        if key not in CHAT_TEMPLATES:
            key = "chatml"  # último fallback
        self.tokenizer = get_chat_template(self.tokenizer, chat_template=key)

    # ------------------------- dados -------------------------------------
    def _load_split(self, parquet_path: str, prompt_col: str, *, split: str) -> Dataset:
        """Carrega dataset usando classe LLMsDataset para padronização."""
        print(f"[2/6] Lendo {split} de {parquet_path}…")
        
        # cria instância da classe LLMsDataset com detecção automática do template
        dataset_loader = LLMsDataset(
            path=parquet_path,
            prompt_col=prompt_col,
            tokenizer=self.tokenizer,
            max_seq_length=self.cfg["max_seq_length"]
        )
        
        # obtém dataset processado
        processed_ds = dataset_loader.get_processed_dataset()
        print(f" - {split} carregado com {len(processed_ds)} registros")
        
        return processed_ds

    @classmethod
    def debug_info(cls, cfg_path: str):
        """Exibe informações detalhadas de debug sobre configuração e datasets."""
        cfg = cls._load_cfg(cfg_path)
        print("="*80)
        print(">> MODO DEBUG - INFORMAÇÕES DE CONFIGURAÇÃO E DATASET")
        print("="*80)
        
        # configuração
        print("\n📋 CONFIGURAÇÃO:")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))

        # carrega o tokenizer para chat_template
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_name"], use_fast=True)
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
        print(f"  - Nome: {cfg['base_model_name']}")
        print(f"  - LoRA r: {cfg['lora_r']}")
        print(f"  - Max seq length: {cfg['max_seq_length']}")
        print(f"  - Template com type: {template_type}")
        
        # Verifica se existe modelo LoRA treinado
        lora_model_path = cfg.get('output_dir', './saida')
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
        elif cfg['lora_r'] not in (0, None, False):
            print(f"  🔄 Será aplicado novo LoRA ao modelo base")
        else:
            print(f"  📄 Será usado modelo base sem LoRA")
        
        # dataset de treino
        if cfg.get("dataset_train_path"):
            try:
                train_loader = LLMsDataset(
                    path=cfg["dataset_train_path"],
                    prompt_col=cfg.get("train_prompt_col"),
                    tokenizer=tokenizer,
                    max_seq_length=cfg["max_seq_length"]
                )
                train_stats = train_loader.get_stats()
                print(f"\n📊 DATASET DE TREINO:")
                print(f"  - Registros: {train_stats['total_registros']}")
                print(f"  - Colunas: {train_stats['colunas']}")
                print(f"  - Formato: {train_stats['formato_arquivo']}")
                print(f"  - Caminho: {train_stats['caminho']}")
                print(f"  - Formato detectado: {train_stats['formato_dataset']}")

                print(f"\n📝 DADOS ANTES DO PROCESSAMENTO:")
                # Mostra dados originais (primeiro registro)
                raw_sample = train_loader.dataset[0]
                print(f"  - Tipo: {type(raw_sample)}")
                print(f"  - Chaves: {list(raw_sample.keys()) if isinstance(raw_sample, dict) else 'N/A'}")
                if isinstance(raw_sample, dict):
                    for key, value in raw_sample.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  - {key}: {repr(value[:100])}... (truncado)")
                        else:
                            print(f"  - {key}: {repr(value)}")

                print(f"\n🔄 PROCESSANDO DATASET...")
                ds_processado = train_loader.get_processed_dataset()
                
                print(f"\n📝 DADOS APÓS PROCESSAMENTO:")
                try:
                    sample = train_loader.get_sample(1)
                    print(f"  - Input IDs length: {len(sample['input_ids'])}")
                    print(f"  - Input IDs type: {type(sample['input_ids'])}")
                    print(f"  - Primeiros 10 tokens: {sample['input_ids'][:10]}")
                    print(f"  - Attention mask sum: {sum(sample['attention_mask'])}")
                    print(f"  - Labels com -100: {sum(1 for x in sample['labels'] if x == -100)}")
                    print(f"  - Labels válidos: {sum(1 for x in sample['labels'] if x != -100)}")
                    print(f"  - Todos input_ids são int: {all(isinstance(x, int) for x in sample['input_ids'])}")
                    
                    print(f"\n📄 TEXTO DECODIFICADO (primeiros 500 chars):")
                    texto = sample.get('texto_decodificado', '')[:500]
                    print(f"    {repr(texto)}...")
                except Exception as e:
                    print(f"  ❌ Erro ao processar exemplo: {e}")
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"\n❌ Erro ao carregar dataset de treino: {e}")
                import traceback
                traceback.print_exc()
        
        # dataset de avaliação
        if cfg.get("dataset_eval_path"):
            try:
                eval_loader = LLMsDataset(
                    path=cfg["dataset_eval_path"],
                    prompt_col=cfg.get("eval_prompt_col", cfg.get("train_prompt_col")),
                    tokenizer=tokenizer,
                    max_seq_length=cfg["max_seq_length"]
                )
                eval_stats = eval_loader.get_stats()
                print(f"\n📊 DATASET DE AVALIAÇÃO:")
                print(f"  - Registros: {eval_stats['total_registros']}")
                print(f"  - Colunas: {eval_stats['colunas']}")
                print(f"  - Formato: {eval_stats['formato_arquivo']}")
                print(f"  - Caminho: {eval_stats['caminho']}")
            except Exception as e:
                print(f"\n❌ Erro ao carregar dataset de avaliação: {e}")
        
        # informações de checkpoints
        print(f"\n💾 CHECKPOINT INFO:")
        checkpoint_dir = os.path.join(cfg.get("output_dir", "./saida"), "chkpt")
        resume_enabled = cfg.get("resume_from_checkpoint", True)
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
                
                latest_step, latest_name = max(checkpoints, key=lambda x: x[0])
                latest_path = os.path.join(checkpoint_dir, latest_name)
                required_files = ["pytorch_model.bin", "training_args.bin", "trainer_state.json"]
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(latest_path, f))]
                
                if missing_files:
                    print(f"  - Último checkpoint incompleto (faltam: {missing_files})")
                else:
                    print(f"  - Último checkpoint válido: {latest_name}")
            else:
                print(f"  - Nenhum checkpoint encontrado")
        else:
            print(f"  - Diretório de checkpoints não existe")
        
        # informações de GPU
        if torch.cuda.is_available():
            print(f"\n🎮 GPU INFO:")
            try:
                _print_mem("DEBUG", 0)  # usa GPU 0 como padrão para debug
            except Exception as e:
                print(f"  ❌ Erro ao obter info da GPU: {e}")
        else:
            print(f"\n🎮 GPU INFO:")
            print("  - CUDA não disponível")
        
        print("\n" + "="*80)
        print("✅ DEBUG COMPLETO - CONFIGURAÇÃO E DATASETS VALIDADOS")
        print("="*80)

    # ------------------------- trainer -----------------------------------
    def _build_trainer(self) -> SFTTrainer:
        print("[3/6] Configurando trainer…")
        total_examples = len(self.train_ds)
        cfg = self.cfg
        eval_steps = cfg.get("eval_steps")
        n_gpus = max(torch.cuda.device_count(),1)
        # percentual do dataset
        if self.eval_ds and isinstance(eval_steps,str) and eval_steps.endswith('%'):
            try:
                eval_steps = int(eval_steps.replace('%','').strip())
                if eval_steps >= 1:
                   _st =  cfg["grad_batch_size"] * cfg["batch_size"] * n_gpus
                   eval_steps = int((eval_steps/100) * (total_examples / _st))
                else:
                   eval_steps = None
            except:
                eval_steps = None
        if eval_steps is None:
           eval_steps = max(
                1, int((total_examples / 100) / (cfg["grad_batch_size"] * cfg["batch_size"])*n_gpus)
            )
        if self.eval_ds:
           print(f' - avaliando a cada {eval_steps} steps (1 step = {cfg["grad_batch_size"]} * {cfg["batch_size"]} * {n_gpus} = {cfg["grad_batch_size"] * cfg["batch_size"]*n_gpus}) ...')
        if isinstance(eval_steps, int) and eval_steps == 0:
            eval_steps = 1
        log_steps = eval_steps if isinstance(eval_steps, int) else 50
        if self.save_checkpoints:
            print(f' - gravando checkpoints a cada {log_steps} steps')
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,  # reativado para dados já tokenizados
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            args=SFTConfig(
                # não usar dataset_text_field para dados já tokenizados
                per_device_train_batch_size=cfg["batch_size"],  # já validado
                gradient_accumulation_steps=cfg["grad_batch_size"],  # já validado
                warmup_steps=cfg.get("warmup_steps", 5),  # já validado
                num_train_epochs=cfg["num_train_epochs"],  # já validado
                eval_strategy="steps" if self.eval_ds else "no",
                eval_steps=eval_steps if self.eval_ds else None,
                save_strategy="steps" if self.save_checkpoints else 'no',
                save_steps=log_steps,
                output_dir=os.path.join(cfg["output_dir"], "chkpt") if self.save_checkpoints else None,
                save_total_limit=1000,
                learning_rate=float(cfg.get("learning_rate", 2e-4)) ,
                logging_dir=os.path.join(cfg["output_dir"], "tb_logs"),
                logging_strategy="steps",
                logging_first_step=True,
                logging_steps=log_steps,
                report_to=[], #["tensorboard"],
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                gradient_checkpointing="unsloth", # True or "unsloth" for very long context
                remove_unused_columns=False,  # importante: preserva colunas tokenizadas
                dataloader_drop_last=False,   # não descarta último batch incompleto
            ),
        )
        os.makedirs(self.cfg["output_dir"], exist_ok=True)
        jsonl = os.path.join(self.cfg["output_dir"], "metrics_stream.jsonl")
        if os.path.isfile(jsonl):
            os.remove(jsonl)
        trainer.add_callback(JsonLoggerCallback(jsonl))
        trainer.model.config.use_cache = False
        
        return trainer

    # ------------------------- checkpoint management --------------------- 
    def _find_latest_checkpoint(self) -> str:
        """Encontra o checkpoint mais recente na pasta de checkpoints.
        
        Returns:
            str: Caminho para o checkpoint mais recente ou None se não houver
        """
        # verifica se o resume está habilitado na configuração
        if not self.cfg.get("resume_from_checkpoint", True):
            print("📋 Resume from checkpoint desabilitado na configuração")
            return None
            
        if not self.save_checkpoints:
            return None
            
        checkpoint_dir = os.path.join(self.cfg["output_dir"], "chkpt")
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
        antes = _print_mem("ANTES", self.device_idx)
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
            
        depois = _print_mem("DEPOIS", self.device_idx)
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
            "config": dict(self.cfg),
            "ds_train_len" : len(self.train_ds),
            "ds_eval_len" : len(self.eval_ds) if self.eval_ds else 0,
            "modelo_info": info_modelo,  # adiciona informações do modelo
        }
        # grava o modelo antes do ultimo eval, pode dar erro de memória no eval    
        self._save_model(stats=stats)
        # 3) garante um eval FINAL mesmo que já tenha havido evals em steps
        if self.eval_ds:
            final_eval = self.trainer.evaluate()     # roda avaliação no eval_dataset
            stats.update(final_eval)                 # adiciona eval_loss, eval_runtime …        self._save_model(stats = stats)
        
    # ------------------------- salvamento --------------------------------
    def _save_model(self, stats = None):
        out_dir = self.cfg["output_dir"]
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
            with open(os.path.join(self.cfg["output_dir"], "metrics_summary.json"), "w") as fp:
                 json.dump(stats, fp, indent=2)
        print(r"Modelo salvo com sucesso \o/")

    def log_processamento(self, msg: str, titulo:str) -> None:
        ''' grava no arquivo de log com o nome _log_processamento_.txt dados importantes
            do processamento do treino como data, hora, parâmetros, dataset, etc
        '''
        arquivo = os.path.join(self.cfg["output_dir"], f"_log_processamento_.txt")
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
        is_peft_model = hasattr(self.model, 'peft_config') or hasattr(self.model, 'base_model')
        model_type = type(self.model).__name__
        print(f"🔍 Tipo do modelo: {model_type} | PEFT ativo: {is_peft_model}")
        
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
                                                            max_length=self.cfg["max_seq_length"])
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

    def testar_predicoes(self, n_exemplos: int = 1, temperatura: float = 0.2, max_new_tokens: int = 2048) -> None:
        """Testa o modelo com exemplos do dataset de treino e exibe as predições."""
        print(f"\n{'='*80}")
        print(f"🧪 TESTANDO MODELO COM {n_exemplos} EXEMPLO(S)")
        print(f"{'='*80}")
        
        # Primeiro valida o status do modelo
        self.print_modelo_status()
        
        # verifica se há dataset disponível
        if not hasattr(self, 'train_ds') or len(self.train_ds) == 0:
            print("❌ Nenhum dataset de treino disponível para teste")
            return
        
        # limita o número de exemplos ao tamanho do dataset
        n_exemplos = min(n_exemplos, len(self.train_ds))
        
        for i in range(n_exemplos):
            print(f"\n{'-'*60}")
            print(f"📝 EXEMPLO {i+1}/{n_exemplos}")
            print(f"{'-'*60}")
            
            # pega o registro original do dataset
            # precisa acessar o dataset original através do LLMsDataset
            dataset_loader = LLMsDataset(
                path=self.cfg["dataset_train_path"],
                prompt_col=self.cfg.get("train_prompt_col"),
                tokenizer=self.tokenizer,
                max_seq_length=self.cfg["max_seq_length"]
            )
            sample_row = dataset_loader.dataset[i]
            processador = lambda x: dataset_loader._process_single_message(x, max_length=self.cfg["max_seq_length"], inferencia=True)

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
                    print(f"❌ Formato de dados não reconhecido para exemplo {i+1}")
                    continue
                
                print(f">> PROMPT:")
                print(f"   {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
                
                print(f"\n>> RESPOSTA ESPERADA:")
                print(f"   {resposta_esperada[:200]}{'...' if len(resposta_esperada) > 200 else ''}")
                
                # gera predição do modelo
                try:
                    resultado = self.prompt(prompt, 
                                            temperatura=temperatura, 
                                            max_new_tokens=max_new_tokens,
                                            processador = processador)
                    resposta_modelo = resultado['texto']
                    
                    print(f"\n>> RESPOSTA DO MODELO:")
                    print(f"   {resposta_modelo[:500]}{'...' if len(resposta_modelo) > 500 else ''}")
                    
                    print(f"\n>> ESTATÍSTICAS:")
                    print(f"   - Tokens do prompt: {resultado.get('prompt_tokens', 'N/A')}")
                    print(f"   - Tokens da resposta: {resultado.get('completion_tokens', 'N/A')}")
                    print(f"   - Temperatura: {temperatura}")
                    
                except Exception as e:
                    print(f"❌ Erro ao gerar predição: {str(e)}\n{traceback.format_exc()}")
                    
            except Exception as e:
                print(f"❌ Erro ao processar exemplo {i+1}: {str(e)}")
        
        print(f"\n{'='*80}")
        print(">> TESTE DE PREDIÇÕES CONCLUÍDO")
        print(f"{'='*80}")

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
                print(f"    Modules: {adapter['target_modules']}")
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

    def __init__(self, path: str, prompt_col: str, tokenizer, max_seq_length: int):
        self.path = path
        self.prompt_col = prompt_col
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self._template_com_type = self.template_com_type(tokenizer)
        if os.path.isfile(self.path):
            self.dataset = self._load_dataset()
        else:
            # serve para preparar apenas para predição
            self.dataset = Dataset.from_list([])  # dataset vazio

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
        "dataset_train_path": "../dataset/data/dados_unificados_sm_treino.parquet",
        "train_prompt_col": "messages",
        "dataset_eval_path": "",
        "eval_prompt_col": "",
        "eval_steps": "15%",
        "base_model_name": "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "output_dir": "../modelos/gemma-3-12b-refleg20k-v01",
        "batch_size": 2,
        "grad_batch_size": 5,
        "num_train_epochs": 1,
        "max_seq_length": 4096,
        "lora_r": 8,   # 0 ou None para full fine-tuning
        "lora_alpha": 32,   # Opcional: por padrão usa lora_r
        "lora_dropout": 0.05,   # Opcional: dropout para LoRA
        "learning_rate": 2e-4,   # Opcional: taxa de aprendizado
        "save_checkpoints": True,
        "resume_from_checkpoint": True,   # Tenta continuar de checkpoint se existir
        "warmup_steps": 5,
        "nbits": 4,   # 4 ou 8 ou None
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",]
    }
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(template, fp, sort_keys=False, allow_unicode=True)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Fine‑tune Gemma‑3 com opções de GPU, YAML e debug. Se o YAML não existir, um template será criado."
    )
    parser.add_argument("config", help="Arquivo YAML com as configurações.")
    parser.add_argument("--gpu", type=int, default=0, help="Índice da GPU CUDA a usar (default=0)")
    parser.add_argument("--debug", action="store_true", help="Modo debug: exibe estrutura do dataset e configuração sem treinar")
    parser.add_argument("--modelo", type=int, nargs='?', const=1, help="Modo debug: exibe exemplos de prompt e resposta do modelo treinado (padrão: 1 exemplo)")
    args = parser.parse_args()

    # seleciona GPU antes de carregar quaisquer tensors
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"CUDA disponível — usando GPU {args.gpu}")
    else:
        print("CUDA não disponível — treinamento será na CPU (muito mais lento)")

    cfg_path = args.config or "./exemplo.yaml"
    if not os.path.exists(cfg_path):
        _create_default_cfg(cfg_path)
        print(
            f"Arquivo de configuração criado em '{cfg_path}'.\n"
            "Revise os parâmetros e execute novamente para iniciar o treinamento."
        )
        sys.exit(0)

    if args.debug:
        # modo debug: apenas exibe informações sem treinar
        LLMsTrainer.debug_info(cfg_path)
        print("\n>> Modo DEBUG ativado - treinamento não executado")
        sys.exit(0)

    trainer = LLMsTrainer(cfg_path, device_idx=args.gpu)

    if args.modelo:
        # modo teste: executa predições em exemplos do dataset
        n_exemplos = args.modelo if isinstance(args.modelo, int) else 1
        trainer.testar_predicoes(n_exemplos=n_exemplos, temperatura=0.2, max_new_tokens=512)
        sys.exit(0)
    # modo normal: executa treinamento
    trainer.train()



if __name__ == "__main__":
    _cli()
