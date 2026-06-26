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
