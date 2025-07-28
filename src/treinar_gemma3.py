#!/usr/bin/env python3
"""
Autor: Luiz Anisio 05/2025 v004

Treinar Gemma‑3 usando Unsloth + TRL‑SFTTrainer de forma configurável.

Uso:
    python treinar_gemma3.py CONFIG.yaml [--gpu 1]

* Se o YAML indicado não existir, um template é criado e o script
  termina — você revisa os valores e executa novamente.
* O parâmetro opcional **--gpu IDX** permite escolher a GPU CUDA a ser
  utilizada (padrão: 0).  O índice é aplicado via `torch.cuda.set_device()`
  logo no início do programa.

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
dataset_eval_path: ""     # opcional
```
"""

import argparse
import os, time, json
import sys
from typing import Any, Dict
import yaml
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import numpy as np
from datetime import datetime

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

class Gemma3Trainer:
    """Encapsula o fluxo de fine‑tuning de Gemma‑3 com LoRA e Unsloth."""

    REQUIRED_KEYS = {
        "dataset_train_path",
        "train_prompt_col",
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
        self._validate_cfg()
        self.model, self.tokenizer = self._load_model()
        self.train_ds = self._load_split(
            self.cfg["dataset_train_path"], self.cfg["train_prompt_col"], split="treino"
        )
        self.eval_ds = None
        if self.cfg.get("dataset_eval_path"):
            self.eval_ds = self._load_split(
                self.cfg["dataset_eval_path"],
                self.cfg.get("eval_prompt_col", self.cfg["train_prompt_col"]),
                split="teste",
            )
        self.save_checkpoints = self.cfg.get('save_checkpoints','') in {1,'1','True',True,'true','sim','Sim','SIM'}
        self.trainer = self._build_trainer()

    # ------------------------- controle no colab ------------------------------
    @classmethod
    def verifica_versao(cls):
        print(f'JsonAnalise carregado corretamente em {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}!')
        
    # ------------------------- configuração ------------------------------
    def _load_cfg(self, path: str) -> Dict[str, Any]:
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
        model, tokenizer = FastModel.from_pretrained(
            model_name=self.cfg["base_model_name"],
            max_seq_length=self.cfg["max_seq_length"],
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=self.cfg["lora_r"],
            lora_alpha=self.cfg["lora_r"],
            lora_dropout=0.0,
            bias="none",
            random_state=3407,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        model.print_trainable_parameters()
        return model, tokenizer

    # ------------------------- dados -------------------------------------
    def _load_split(self, parquet_path: str, prompt_col: str, *, split: str) -> Dataset:
        print(f"[2/6] Lendo {split} de {parquet_path}…")
        df = pd.read_parquet(parquet_path)
        if prompt_col not in df.columns:
            raise KeyError(f"Coluna '{prompt_col}' não encontrada em {parquet_path}")
        print(f" - {split} carregado com {len(df)} registros")

        def build_text(row):
            messages = row[prompt_col]
            if not (isinstance(messages, (list, np.ndarray, np.array, tuple)) and len(messages) == 2):
                raise ValueError(f"Esperado lista de 2 mensagens user/assistant >> {type(messages)}")
            if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
                raise ValueError("Ordem user/assistant inválida")
            return {"text": self.tokenizer.apply_chat_template(list(messages), tokenize=False)}

        return Dataset.from_list(list(df.apply(build_text, axis=1)))

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
            tokenizer=self.tokenizer,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=cfg["batch_size"],
                gradient_accumulation_steps=cfg["grad_batch_size"],
                warmup_steps=cfg.get("warmup_steps", 5),
                num_train_epochs=cfg["num_train_epochs"],
                eval_strategy="steps" if self.eval_ds else "no",
                eval_steps=eval_steps if self.eval_ds else None,
                save_strategy="steps" if self.save_checkpoints else 'no',
                save_steps=log_steps,
                output_dir=os.path.join(cfg["output_dir"], "chkpt") if self.save_checkpoints else None,
                save_total_limit=1000,                
                learning_rate=2e-4,
                logging_dir=os.path.join(cfg["output_dir"], "tb_logs"),
                logging_strategy="steps",
                logging_first_step=True,
                logging_steps=log_steps,
                report_to=["tensorboard"],
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
            ),
        )
        os.makedirs(self.cfg["output_dir"], exist_ok=True)
        jsonl = os.path.join(self.cfg["output_dir"], "metrics_stream.jsonl")
        if os.path.isfile(jsonl):
            os.remove(jsonl)
        trainer.add_callback(JsonLoggerCallback(jsonl))
        trainer.model.config.use_cache = False
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )
        return trainer

    # ------------------------- execução ----------------------------------
    def train(self):
        antes = _print_mem("ANTES", self.device_idx)
        print("[4/6] Iniciando treinamento…")
        train_stats = self.trainer.train()
        depois = _print_mem("DEPOIS", self.device_idx)
        print("[5/6] Tempo de execução: {:.2f} s".format(train_stats.metrics["train_runtime"]))
        
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
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        if stats is not None:
            with open(os.path.join(self.cfg["output_dir"], "metrics_summary.json"), "w") as fp:
                 json.dump(stats, fp, indent=2)
        print("Modelo salvo com sucesso \o/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _create_default_cfg(path: str) -> None:
    template = {
        "dataset_train_path": "../dataset/data/dados_unificados_sm_treino.parquet",
        "train_prompt_col": "messages",
        "base_model_name": "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "output_dir": "../modelos/gemma-3-12b-refleg20k-v01",
        "batch_size": 2,
        "grad_batch_size": 5,
        "num_train_epochs": 1,
        "max_seq_length": 4096,
        "lora_r": 8,
        "dataset_eval_path": "",
        "eval_prompt_col": "",
        "eval_steps": "15%",
        "save_checkpoints": True
    }
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(template, fp, sort_keys=False, allow_unicode=True)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Fine‑tune Gemma‑3 com opções de GPU e YAML. Se o YAML não existir, um template será criado."
    )
    parser.add_argument("config", help="Arquivo YAML com as configurações.")
    parser.add_argument("--gpu", type=int, default=0, help="Índice da GPU CUDA a usar (default=0)")
    args = parser.parse_args()

    # seleciona GPU antes de carregar quaisquer tensors
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"CUDA disponível — usando GPU {args.gpu}")
    else:
        print("CUDA não disponível — treinamento será na CPU (muito mais lento)")

    cfg_path = args.config
    if not os.path.exists(cfg_path):
        _create_default_cfg(cfg_path)
        print(
            f"Arquivo de configuração criado em '{cfg_path}'.\n"
            "Revise os parâmetros e execute novamente para iniciar o treinamento."
        )
        sys.exit(0)

    trainer = Gemma3Trainer(cfg_path, device_idx=args.gpu)
    trainer.train()


if __name__ == "__main__":
    _cli()
