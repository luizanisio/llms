from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import os

model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
print(f"Carregando tokenizer de {model_name}...")

# Carrega apenas tokenizer para ser rápido
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except:
    # Se falhar, tenta via FastModel (mas demora carregar modelo)
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name = model_name,
        load_in_4bit = True,
    )

print("Aplicando template qwen-2.5...")
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

messages = [
    {"role": "user", "content": "Olá"},
    {"role": "assistant", "content": "Oi"}
]

formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

print("\n=== FORMATO GERADO ===")
print(repr(formatted))
print("======================\n")

instruction_part = "<|im_start|>user\n"
response_part = "<|im_start|>assistant\n"

print(f"Testando instruction_part='{repr(instruction_part)}': {'ENCONTRADO' if instruction_part in formatted else 'NÃO ENCONTRADO'}")
print(f"Testando response_part='{repr(response_part)}': {'ENCONTRADO' if response_part in formatted else 'NÃO ENCONTRADO'}")

# Verifica se há diferença de espaços ou quebras de linha
if instruction_part not in formatted:
    print("Investigando possíveis diferenças no instruction_part...")
    # Tenta variações comuns
    variations = [
        "<|im_start|>user",
        "<|im_start|>user ",
        "user\n",
    ]
    for v in variations:
         print(f"  Variação '{repr(v)}': {'ENCONTRADO' if v in formatted else 'N'}")

