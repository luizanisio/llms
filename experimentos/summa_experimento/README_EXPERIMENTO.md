# Passo 01 - Dataset

O primeiro passo é extrair os textos dos acórdãos do portal de dados abertos do STJ. Utilizamos o CKAN para gerar um arquivo `.parquet` com os metadados e os textos selecionados.

Usando o script `util_ckan.py`, utilizamos o arquivo de configuração da extração (`config_extracao.yaml`) para gerar o arquivo `.parquet` contendo os textos extraídos e os metadados dos processos.

**Comando:**
```bash
python ../../src/util_ckan.py --config config_extracao.yaml
```

**Composição dos dados:**
- **Grupo 1:** Dados selecionados para o experimento com 22k documentos estratificados com distância do cosseno de no mínimo `0.15` usando o modelo Athos do STJ.
- **Grupo 2:** Novos documentos posteriores ao treinamento de qualquer um dos modelos utilizados. Conjunto escolhido: acórdãos publicados em `25/05/2026`.

> TODO: Avaliar se a extração pode considerar distância do cosseno e qtd de itens para extrair.


---

# Passo 02 - Geração de Dados de Treino e Teste

Nesta etapa, geramos os dados processados utilizando os diferentes modelos configurados.

## 1. Professor: OpenRouter (Qwen3-235B)
A extração de dados via API OpenRouter para a destilação de conhecimento do modelo Qwen3-235B assegura a propriedade intelectual ao pesquisador, permitindo o uso para treinamento de modelos, conforme os Termos de Serviço vigentes. Foram configuradas as diretrizes restritivas de privacidade, incluindo o opt-out de compartilhamento de dados e o uso de rotas de *Zero Data Retention*, para impedir o aprimoramento de modelos de terceiros.

### Configurações do modelo professor:
- **Versão do modelo:** `qwen/qwen3-235b-a22b-2507:poor`
- **Provider:** `{"quantizations": ["fp8"]}`
- o sufixo :poor ativa no OpenRouter a busca por deployments mais baratos para o mesmo modelo, reduzindo a prioridade do pedido e podendo aumentar o tempo para extração.

**Variáveis de Ambiente:**
- `PESSOAL_OPENROUTER_API_KEY`
- `OPENROUTER_EXTRA`

**Comando:**
```bash
python ../../src/util_vllm_batch.py --config config_batch_235b.yaml
```

## 2. Aluno: Modelo local com vLLM (CISIA - PUCPR)
- **Versão do modelo:** `qwen/Qwen2.5-7B-Instruct`
- **max_model_len:** 38912
- **tensor_parallel_size:** 1
- **dtype:** "auto"
- **quantization:** bitsandbytes
- **load_format:** bitsandbytes
- 1xGPU H100

**Variáveis de Ambiente (opcional):**
- `HF_TOKEN`

**Comando:**
```bash
python ../../src/util_vllm_batch.py --config config_batch.yaml
```

## 3. Referência: GPT-5
- **API:** Azure
- **Versão do modelo:** `gpt-5-2025-08-07`
- **Configurações:** Reasoning=Médio / Verbose=Low

**Variáveis de Ambiente:**
- `OA_KEY`
- `OA_CONTROLE`

**Comando:**
```bash
python ../../src/util_vllm_batch.py --config config_batch_gpt5.yaml
```
