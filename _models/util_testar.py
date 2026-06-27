import os

def list_models(base_dir):
    models = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        # Considera como modelo os diretórios que não são pastas ocultas/sistema
        if os.path.isdir(item_path) and not item.startswith('.') and item != '__pycache__':
            models.append(item)
    return sorted(models)

def escolher_modelo(base_dir):
    models = list_models(base_dir)
    
    if not models:
        print("Nenhum modelo encontrado na pasta atual.")
        return None
    
    print("Modelos disponíveis:")
    for i, model in enumerate(models, 1):
        print(f"[{i}] {model}")
        
    choice = input("\nEscolha um modelo pelo número: ")
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(models):
            print("Opção inválida.")
            return None
    except ValueError:
        print("Opção inválida.")
        return None
        
    return os.path.join(base_dir, models[idx])

def imprimir_versoes():
    print("=" * 60)
    print("INFORMAÇÕES DE AMBIENTE E PACOTES:")
    print("-" * 60)
    try:
        import torch
        print(f"- PyTorch: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"- CUDA Available: {cuda_avail}")
        if cuda_avail:
            print(f"- CUDA Version (built with): {torch.version.cuda}")
            print(f"- GPUs detectadas: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("- PyTorch: não instalado")

    try:
        import transformers
        print(f"- Transformers: {transformers.__version__}")
    except ImportError:
        print("- Transformers: não instalado")

    try:
        import vllm
        print(f"- vLLM: {vllm.__version__}")
    except ImportError:
        print("- vLLM: não instalado")
        
    try:
        import triton
        print(f"- Triton: {triton.__version__}")
    except ImportError:
        print("- Triton: não instalado")
    print("=" * 40 + "\n")
