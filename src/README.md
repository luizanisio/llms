
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

## Contents

| File / Module | Role |
|---|---|
| `util.py` | General-purpose utilities (env loading, file I/O, logging) |
| `util_agentes.py` | Agent orchestration helpers (planning, memory, directed review) |
| `util_json.py` | JSON schema validation, field extraction, and normalization |
| `util_json_carga.py` | Dataset loading with field grouping and sub-level aggregation |
| `util_json_dados.py` | Data access and manipulation utilities for JSON records |
| `util_json_exemplos.py` | Few-shot example selection and formatting |
| `util_json_relatorio.py` | JSON-based evaluation report generation |
| `util_openai.py` | OpenAI API wrappers (chat, embeddings, retry logic) |
| `util_prompt.py` | Prompt template construction and token budget management |
| `util_sbert.py` | Sentence-BERT wrappers for semantic similarity |
| `util_bertscore.py` | BERTScore-like metric implementation for evaluation |
| `util_bertscore_service.py` | BERTScore as a local service for batch evaluation |
| `util_analise_estatistica.py` | Statistical analysis (Wilcoxon, bootstrap, effect sizes) |
| `util_graficos.py` | Visualization utilities for metrics and comparisons |
| `util_pandas.py` | DataFrame helpers and tabular processing utilities |
| `util_tiktoken.py` | Token counting via tiktoken for cost estimation |
| `util_sysinfo.py` | System resource monitoring (CPU, RAM, GPU) |
| `util_get_resposta.py` | LLM response retrieval and parsing helpers |
| `generate.py` | Entry point for running single-prompt and JAMEX pipelines |
| `treinar_unsloth.py` | Fine-tuning pipeline using Unsloth + QLoRA |
| `treinar_unsloth_*.py` | Supporting modules: dataset, logging, monitor, report, charts |
| `treinar_gemma3.py` | Gemma 3 specific fine-tuning configuration |
| `teste_*.py` | Unit and integration tests for core utilities |
| `requirements.txt` | Python dependencies for this snapshot |

## Important

- The `.env_template` file in this directory is a **template only** and contains no real credentials and need to be copied to `.env` and filled with the actual credentials.
