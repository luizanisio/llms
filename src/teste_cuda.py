# Autor: Luiz Anísio
# Fonte: https://github.com/luizanisio/llms/tree/main/src

import torch
try:
    import unsloth
    unsloth_ok = True
except ImportError:
    print("Instale a biblioteca unsloth: pip install unsloth")
    unsloth_ok = False
import transformers

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
if unsloth_ok:
    print(f"Unsloth version: {unsloth.__version__}")
else:
    print("Unsloth não está instalado.")    

if torch.cuda.is_available():
    print("CUDA está disponível. GPU encontrada.")
else:
    print("CUDA não está disponível. Apenas CPU.")

try:
    from util_prompt import UtilLLM
    UtilLLM.verifica_versao()
except Exception as e:
    import traceback
    print('* Não foi possível retornar informações mais completas do pacote UtilLLM!')
    traceback.print_exc()
