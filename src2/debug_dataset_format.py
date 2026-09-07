
import os
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset

# Simula o carregamento e formatação
model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
print(f"Carregando tokenizer de {model_name}...")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# Simula dados do dataset
messages = [
    [
        {"role": "user", "content": "Olá, qual a sua função?"},
        {"role": "assistant", "content": "Sou um assistente virtual."}
    ]
]

# Cria dataset
dataset = Dataset.from_dict({"messages": messages})

# Aplica template para criar coluna 'text'
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts,}

dataset = dataset.map(formatting_prompts_func, batched=True)

print("Exemplo FORMATADO:")
print(repr(dataset[0]["text"]))

instruction_part = "<|im_start|>user\n"
response_part = "<|im_start|>assistant\n"

text = dataset[0]["text"]
if instruction_part in text:
    print(f"Instruction part '{instruction_part.strip()}' ENCONTRADO.")
else:
    print(f"Instruction part '{instruction_part.strip()}' NÃO ENCONTRADO.")

if response_part in text:
    print(f"Response part '{response_part.strip()}' ENCONTRADO.")
else:
    print(f"Response part '{response_part.strip()}' NÃO ENCONTRADO.")
