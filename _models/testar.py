import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util_testar import escolher_modelo, imprimir_versoes

def main():
    imprimir_versoes()
    base_dir = os.path.abspath(os.path.dirname(__file__))
    model_path = escolher_modelo(base_dir)
    
    if not model_path:
        return
        
    print(f"Modelo selecionado: {model_path}")
    
    prompt_text = "Quem é você e qual é a capital do Brasil?"
    print(f"Prompt: '{prompt_text}'")
    print("\nCarregando modelo diretamente (Transformers)...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto"
        )
        
        messages = [
            {"role": "system", "content": "Você é um assistente útil e direto."},
            {"role": "user", "content": prompt_text}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        print("\nIniciando inferência...")
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.01,
            do_sample=True
        )
        
        # Ignora os tokens do prompt na hora de decodificar a resposta
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        resultado = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print("\n" + "="*50)
        print("RESPOSTA DO MODELO (Transformers):")
        print("="*50)
        print(resultado.strip())
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\nErro ao executar a inferência com Transformers: {e}")

if __name__ == '__main__':
    main()
