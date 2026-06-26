import os
import sys
import tempfile
import subprocess
import yaml
from util_testar import escolher_modelo

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    model_path = escolher_modelo(base_dir)
    
    if not model_path:
        return
        
    print(f"Modelo selecionado: {model_path}")
    
    # Criar ambiente temporário para util_vllm_batch.py
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "entrada")
        output_dir = os.path.join(temp_dir, "saida")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        prompt_text = "Quem é você e qual é a capital do Brasil?"
        with open(os.path.join(input_dir, "teste.txt"), "w", encoding="utf-8") as f:
            f.write(prompt_text)
            
        config = {
            "modelo": {
                "caminho": model_path
            },
            "vllm": {
                "max_model_len": 512,
                "gpu_memory_utilization": 0.90,
                "device": "cuda"
            },
            "geracao": {
                "max_tokens": 256,
                "temperature": 0.01
            },
            "entrada": {
                "arquivo": input_dir,
                "system_prompt": "Você é um assistente útil e direto."
            },
            "saida": {
                "arquivo": output_dir,
                "tipo_saida": "str"
            }
        }
        
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)
            
        print("\nIniciando inferência com util_vllm_batch.py (vLLM)...")
        print(f"Prompt: '{prompt_text}'")
        
        util_script = os.path.abspath(os.path.join(base_dir, "..", "src", "util_vllm_batch.py"))
        
        try:
            # Executa o util_vllm_batch.py passando o config.yaml gerado
            subprocess.run([sys.executable, util_script, "--config", config_path], check=True)
            
            # Ler saída gerada
            output_file = os.path.join(output_dir, "teste.txt")
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    resultado = f.read()
                print("\n" + "="*50)
                print("RESPOSTA DO MODELO (vLLM):")
                print("="*50)
                print(resultado.strip())
                print("="*50 + "\n")
            else:
                print("\nNão foi possível encontrar o arquivo de saída gerado.")
                
        except subprocess.CalledProcessError as e:
            print(f"\nErro ao executar a inferência: {e}")
        except Exception as e:
            print(f"\nErro inesperado: {e}")

if __name__ == '__main__':
    main()
