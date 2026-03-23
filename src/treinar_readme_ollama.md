# Do Treinamento ao Ollama — Guia Passo a Passo

Este guia mostra como exportar seu modelo fine-tuned em HF safetensors,
converter para GGUF com llama.cpp, e importar no Ollama.

---

## 1. Exportar modelo (HF safetensors)

Exporte em **16-bit** (recomendado para conversão GGUF):

```bash
python treinar_unsloth.py treina_meu_modelo.yaml --merge --quant 16bit
```

> **⚠️ Importante**: modelos exportados em 4-bit (BitsAndBytes) **não** são
> compatíveis com `convert_hf_to_gguf.py`. Use sempre **16-bit** se pretende
> converter para GGUF.

A pasta de saída será criada ao lado do `treina_meu_modelo.yaml`, por exemplo:
```
experimentos/summa_qualifica/merged_google_gemma-3-4b-it_(16bit)/
├── config.json
├── model*.safetensors
├── tokenizer.json
├── Modelfile           ⬅️ gerado automaticamente 
└── ...
```

---

## 2. Instalar dependências (uma vez)

### 2a. Compilador C++ (necessário para numpy/sentencepiece)

```bash
sudo apt update && sudo apt install -y g++ build-essential
```

### 2b. Clonar llama.cpp

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

### 2c. Instalar pacotes Python do llama.cpp

Instale apenas os pacotes essenciais para conversão:

```bash
pip install gguf sentencepiece protobuf
```

Ou, se preferir instalar tudo:

```bash
pip install -r requirements.txt
```

### 2d. Compilar llama-quantize (opcional, para quantização)

⚠️ não funcionou
```bash
cd ~/llama.cpp
cmake -B build
cmake --build build --config Release -t llama-quantize
```

O binário ficará em `build/bin/llama-quantize`.

---

## 3. Converter HF → GGUF

```bash
cd ~/llama.cpp

# Converter para GGUF F16 (precisão máxima)
python3 convert_hf_to_gguf.py /caminho/para/merged_modelo_(16bit) \
    --outfile /caminho/para/merged_modelo_(16bit)/modelo-f16.gguf \
    --outtype f16
```

> **Dica**: use `--outfile` apontando para a **mesma pasta** do merge.
> Assim o `.gguf` fica junto com o `Modelfile`.

---

## 4. Quantizar (opcional, reduz tamanho)

```bash
# Q4_K_M — bom equilíbrio tamanho/qualidade (~4 bits)
~/llama.cpp/build/bin/llama-quantize \
    /caminho/para/merged_modelo_(16bit)/modelo-f16.gguf \
    /caminho/para/merged_modelo_(16bit)/modelo-q4_k_m.gguf \
    Q4_K_M

# Q8_0 — maior qualidade (~8 bits)
~/llama.cpp/build/bin/llama-quantize \
    /caminho/para/merged_modelo_(16bit)/modelo-f16.gguf \
    /caminho/para/merged_modelo_(16bit)/modelo-q8_0.gguf \
    Q8_0
```

Tipos comuns de quantização:

| Tipo     | Bits | Qualidade        | Tamanho relativo |
|----------|------|------------------|------------------|
| F16      | 16   | Máxima           | 100%             |
| Q8_0     | 8    | Muito boa        | ~50%             |
| Q6_K     | 6    | Boa              | ~40%             |
| Q5_K_M   | 5    | Boa              | ~35%             |
| Q4_K_M   | 4    | Aceitável ⭐     | ~28%             |
| Q3_K_M   | 3    | Razoável         | ~22%             |
| Q2_K     | 2    | Baixa            | ~18%             |

---

## 5. Ajustar o Modelfile

O `Modelfile` gerado automaticamente contém um placeholder:

```
FROM <./MeuModelo.gguf>
```

Substitua pelo nome real do arquivo GGUF gerado:

```
FROM ./modelo-q4_k_m.gguf
```

---

## 6. Importar no Ollama

```bash
# entrar na pasta que contém o modelo e o Modelfile
cd /caminho/para/merged_modelo_(16bit)

# Criar modelo no Ollama
ollama create meu-modelo -f Modelfile

# Testar
ollama run meu-modelo
```

---

## 7. Testar via API (opcional)

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "meu-modelo",
  "messages": [{"role": "user", "content": "Olá, como você está?"}]
}'
```

---

## Resumo do fluxo

```
treinar_unsloth.py --merge --quant 16bit
          │
          ▼
   HF safetensors + Modelfile
          │
          ▼
  convert_hf_to_gguf.py (llama.cpp)
          │
          ▼
      modelo.gguf
          │
          ▼  (opcional)
  llama-quantize → modelo-q4_k_m.gguf
          │
          ▼
  Ajustar FROM no Modelfile
          │
          ▼
   ollama create → ollama run
```

---

## Dicas

- **VRAM limitada?** Use Q4_K_M ou Q3_K_M para economizar memória.
- **Qualidade máxima?** Use F16 ou Q8_0.
- **Vários GGUFs?** Gere múltiplas quantizações e teste qual atende melhor.
- **Atualizar modelo no Ollama?** Basta rodar `ollama create` novamente com o mesmo nome.
- **Remover modelo do Ollama?** `ollama rm meu-modelo`
