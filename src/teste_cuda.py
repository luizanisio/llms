import torch
import unsloth
import transformers

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Unsloth version: {unsloth.__version__}")

if torch.cuda.is_available():
    print("CUDA está disponível. GPU encontrada.")
else:
    print("CUDA não está disponível. Apenas CPU.")