import json
import pandas as pd
import random
import os

random.seed(42)

input_file = "puil_treinamento.txt"

with open(input_file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

data = []
for i, line in enumerate(lines):
    try:
        obj = json.loads(line)
        obj['id'] = f"puil_{i+1:03d}"
        data.append(obj)
    except Exception as e:
        print(f"Erro na linha {i+1}: {e}")

random.shuffle(data)

n_total = len(data)
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)

for i, obj in enumerate(data):
    if i < n_train:
        obj['alvo'] = 'treino'
    elif i < n_train + n_val:
        obj['alvo'] = 'validacao'
    else:
        obj['alvo'] = 'teste'

os.makedirs("dados", exist_ok=True)
os.makedirs("saidas", exist_ok=True)
os.makedirs("divisoes", exist_ok=True)
os.makedirs("jobs_logs", exist_ok=True)

df_entrada = pd.DataFrame([{'id': d['id'], 'prompt': d['prompt']} for d in data])
df_entrada.to_csv("dados/entrada_puil.csv", index=False)

df_saida = pd.DataFrame([{'id': d['id'], 'completion': d['completion']} for d in data])
df_saida.to_csv("saidas/saida_puil.csv", index=False)

df_divisao = pd.DataFrame([{'id': d['id'], 'alvo': d['alvo']} for d in data])
df_divisao.to_csv("divisoes/divisao_puil.csv", index=False)

with open("dados/prompt_puil.txt", "w", encoding="utf-8") as f:
    f.write("<<--TEXTO-->>")

print("Datasets CSV gerados com sucesso!")
