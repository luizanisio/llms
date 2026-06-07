

# Dataset
Primeiro passo é extrair os textos dos acórdãos do portal de dados abertos do STJ. Utilizado o CKAN para gerar um arquivo .parquet com os metadados e textos utilizados.
- grupo 1: dados selecionados para o experimento com 22k documentos extratificados com distância do cosseno de no mínimo 0.15 usando o modelo Athos do STJ
- grupo 2: novos documentos posteriores ao treinamento de qualquer um dos modelos utilizados. Data escolhida 25/05/2026

# Professor: OpenRouter:
A extração de dados via API OpenRouter para a destilação de conhecimento do modelo Qwen3-235B assegura a propriedade intelectual ao pesquisador, permitindo o uso para treinamento de modelos, conforme os Termos de Serviço vigentes. Foram configuradas as diretrizes restritivas de privacidade, incluindo o opt-out de compartilhamento de dados e o uso de rotas de Zero Data Retention, para impedir o aprimoramento de modelos terceiros.

## Configurações do modelo professor:
- versão do modelo:qwen/qwen3-235b-a22b-2507:poor
- provider: {"quantizations": ["fp8"]}

# Aluno: Modelo local com vLLM (CISIA - PUCPR)
- versão do modelo:qwen/Qwen2.5-7B-Instruct
- max_model_len: 38912
- tensor_parallel_size: 1
- dtype: "auto"
- quantization: bitsandbytes
- load_format: bitsandbytes

# Referência: GPT5 Reasoning=Médio/ Verbose=Low
- api Azure
- versão do modelo: gpt-5-2025-08-07

