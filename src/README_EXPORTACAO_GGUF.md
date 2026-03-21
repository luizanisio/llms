# Exportação para GGUF (Ollama / llama.cpp)

## 📦 O que é GGUF?

GGUF é um formato de modelo otimizado para inferência em CPU e GPUs consumidoras, usado por:
- **Ollama** - Servidor local de LLMs
- **llama.cpp** - Motor de inferência C++
- **LM Studio** - Interface desktop para LLMs

## 🎯 Por que Manter Unsloth Apenas Para Isso?

O Unsloth tem uma excelente implementação de exportação GGUF que:
- ✅ Suporta múltiplos métodos de quantização
- ✅ Otimiza vocabulário e embeddings
- ✅ É muito mais simples que usar llama.cpp manualmente
- ✅ Gera arquivos prontos para uso

**Decisão:** Manter Unsloth **apenas** como dependência opcional para exportação GGUF, isolado do pipeline de treinamento.

---

## 🚀 Como Usar

### 1. Instalar Dependências (uma vez)

```bash
# Dependências opcionais para exportação GGUF
pip install -r requirements-gguf.txt
```

Isso instala:
- `unsloth>=2026.1.4`
- `unsloth_zoo>=2026.1.4`

### 2. Exportar Modelo para GGUF

#### **Caso A: Modelo já mesclado (merged)**

```bash
# Exportar com Q4_K_M (padrão, balanceado)
python exportar_gguf.py ./modelos/meu_modelo_merged

# Exportar com Q8_0 (maior qualidade)
python exportar_gguf.py ./modelos/meu_modelo_merged --quantization q8_0

# Exportar com F16 (precisão máxima)
python exportar_gguf.py ./modelos/meu_modelo_merged --quantization f16
```

#### **Caso B: Modelo LoRA (adaptadores)**

```bash
# O script faz merge automaticamente antes de exportar
python exportar_gguf.py ./modelos/meu_modelo_lora --merge --quantization q4_k_m
```

---

## 📊 Métodos de Quantização

| Método | Tamanho (7B) | Qualidade | Uso Recomendado |
|--------|--------------|-----------|-----------------|
| **q4_k_m** | ~3.5 GB | ⭐⭐⭐⭐ | **Recomendado** - Melhor custo-benefício |
| **q8_0** | ~7.0 GB | ⭐⭐⭐⭐⭐ | Alta qualidade, 2x maior |
| **f16** | ~14 GB | ⭐⭐⭐⭐⭐ | Máxima qualidade, uso em produção |
| **q4_0** | ~3.2 GB | ⭐⭐⭐ | Mais compacto, menor qualidade |
| **q5_k_m** | ~4.5 GB | ⭐⭐⭐⭐ | Balanceado intermediário |

**Recomendação geral:** Use `q4_k_m` para maioria dos casos (excelente qualidade, tamanho compacto).

---

## 📝 Exemplos Práticos

### Exemplo 1: Workflow Completo (Treino → Merge → GGUF)

```bash
# 1. Treinar modelo com LoRA
python src/treinar_unsloth.py config.yaml --treinar

# 2. Mesclar adaptadores LoRA (gera modelo completo)
python src/treinar_unsloth_avaliar.py config.yaml --merge

# 3. Exportar para GGUF
python src/exportar_gguf.py ./modelos/meu_modelo_merged --quantization q4_k_m

# 4. Usar no Ollama
ollama create meu_modelo -f ./modelos/meu_modelo_merged_gguf_q4_k_m/Modelfile
ollama run meu_modelo "Olá!"
```

### Exemplo 2: Exportação Direta de LoRA

```bash
# Exporta LoRA diretamente (faz merge automático)
python src/exportar_gguf.py ./modelos/meu_modelo_lora \
    --merge \
    --quantization q4_k_m \
    --output ./modelos/meu_modelo_gguf
```

### Exemplo 3: Múltiplas Quantizações

```bash
# Gera várias versões para comparação
python src/exportar_gguf.py ./modelos/meu_modelo_merged --quantization q4_k_m
python src/exportar_gguf.py ./modelos/meu_modelo_merged --quantization q8_0
python src/exportar_gguf.py ./modelos/meu_modelo_merged --quantization f16
```

---

## 🔧 Integração com Ollama

### Criar Modelfile

Após exportar, crie um `Modelfile` no diretório de saída:

```dockerfile
# Modelfile
FROM ./modelo-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
```

### Registrar no Ollama

```bash
cd ./modelos/meu_modelo_merged_gguf_q4_k_m
ollama create meu_modelo -f Modelfile
ollama run meu_modelo
```

---

## ⚠️ Notas Importantes

### 1. **Memória RAM**
A exportação GGUF pode consumir **muita memória RAM** (não VRAM):
- Modelo 7B: ~20-30 GB RAM
- Modelo 13B: ~40-60 GB RAM
- Modelo 70B: ~150+ GB RAM

**Solução:** Use um servidor com RAM suficiente ou exporte em uma máquina separada.

### 2. **Tempo de Exportação**
- Modelo 7B: ~5-15 minutos
- Modelo 13B: ~15-30 minutos
- Modelo 70B: ~1-2 horas

### 3. **Unsloth é Opcional**
Se você **não precisa** de GGUF (ex: usa apenas vLLM, TGI, HF Inference):
- ❌ Não instale `requirements-gguf.txt`
- ✅ Use apenas exportação nativa do HF (safetensors)

---

## 🆚 Alternativa: llama.cpp Manual

Se preferir não usar Unsloth, você pode usar llama.cpp diretamente:

```bash
# 1. Clonar llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 2. Converter modelo HF para GGUF
python convert-hf-to-gguf.py ../modelos/meu_modelo_merged --outtype f16

# 3. Quantizar
./quantize ../modelos/meu_modelo_merged/ggml-model-f16.gguf \
           ../modelos/meu_modelo_merged/ggml-model-q4_k_m.gguf q4_k_m
```

**Desvantagem:** Mais complexo, requer compilação, múltiplos passos.

---

## 📚 Referências

- [Unsloth GGUF Export](https://github.com/unslothai/unsloth)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Ollama](https://ollama.ai)
- [Formatos GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

---

**Última atualização:** 2026-03-21
