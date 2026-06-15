
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

## Contents

### Utilities

| File / Module | Role |
|---|---|
| `util.py` | General-purpose utilities (env loading, file I/O, logging) |
| `util_print.py` | Process-aware grouped output with color support, DataFrame-to-Excel export |
| `util_menu_opcoes.py` | Interactive CLI menus for selecting YAML files, folders, and generic files |
| `util_agentes.py` | Agent orchestration helpers (planning, memory, directed review) |
| `util_json.py` | JSON schema validation, field extraction, and normalization |
| `util_json_carga.py` | Dataset loading with field grouping and sub-level aggregation |
| `util_json_dados.py` | Data access and manipulation utilities for JSON records |
| `util_json_divisoes.py` | Train/validation/test splits by difficulty level based on F1 scores |
| `util_json_exemplos.py` | Few-shot example selection and formatting |
| `util_json_graficos.py` | Comparative performance plots, per-field and per-model graphics |
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
| `util_vllm_batch.py` | Batch inference utility using vLLM configured via YAML |

### Training Pipeline (Unsloth / HF / vLLM)

| File / Module | Role |
|---|---|
| `treinar_unsloth.py` | Main fine-tuning class (`LLMsTrainer`) using Unsloth + QLoRA/full |
| `treinar_unsloth_actions.py` | CLI action dispatcher (`--info`, `--stats`, `--treinar`, `--reset`, etc.) |
| `treinar_unsloth_pipeline.py` | Curriculum learning pipeline with unified state/metrics tracking |
| `treinar_unsloth_util.py` | YAML config loader/validator (`YamlTreinamento`) and folder-based dataset loader |
| `treinar_unsloth_dataset.py` | Dataset loading, validation, and preparation for curriculum-based training |
| `treinar_unsloth_chat.py` | Chat template manager (template selection, dataset formatting, train-on-responses-only) |
| `treinar_unsloth_export.py` | Export and inference engine — predict (HF/vLLM/Unsloth), interactive inference, LoRA merge |
| `treinar_unsloth_avaliar.py` | Evaluation and statistics for trained models |
| `treinar_unsloth_graficos.py` | Chart generation for training metrics (loss curves, token-distribution boxplots) |
| `treinar_unsloth_report.py` | Markdown report generator for training runs |
| `treinar_unsloth_historico.py` | Training history manager — versioned YAML snapshots, event log files |
| `treinar_unsloth_logging.py` | Centralized logging configuration (levels, timestamps, file + console output) |
| `treinar_unsloth_monitor.py` | Resource monitor — continuous RAM/GPU tracking during predictions |
| `treinar_unsloth_dicas.py` | Inline documentation hints injected into YAML configuration files |
| `treinar_model_loader.py` | Model loading via HF Transformers + PEFT (base & LoRA), 4-bit/8-bit quantization |
| `treinar_chat_templates.py` | Native HF chat-template module — auto-detects template by model family |
| `treinar_vllm_inference.py` | Fast inference engine using vLLM (PagedAttention, continuous batching, multi-GPU) |
| `treinar_realizar_predicoes.py` | Prediction and interactive inference classes for LLM models (HF, vLLM, Unsloth, Ollama) |
| `treinar_gemma3.py` | Gemma-3 specific fine-tuning configuration |
| `treinar_to_ollama.py` | Ollama Modelfile generator for importing merged HF models into Ollama |
| `exportar_gguf.py` | GGUF export utility using Unsloth for quantized conversion |
| `inst_deps_treina.py` | Automated dependency installer for training (Colab-aware) |

### Other Tools

| File / Module | Role |
|---|---|
| `generate.py` | Entry point for running single-prompt and JAMEX pipelines |
| `comparar_extracoes.py` | YAML-driven comparison of structured extractions (BERTScore, ROUGE, Levenshtein) |
| `debug_dataset_format.py` | Debug script — simulates dataset loading and chat-template formatting |
| `debug_template.py` | Debug script — applies chat template to inspect formatted conversation output |

### Tests

| File / Module | Role |
|---|---|
| `teste_analise_estatistica.py` | Tests for statistical analysis utilities |
| `teste_bert_rouge.py` | Tests for BERTScore and ROUGE metrics |
| `teste_cuda.py` | CUDA availability and GPU diagnostic tests |
| `teste_sbert.py` | Tests for Sentence-BERT wrappers |
| `teste_util_json.py` | Tests for JSON utilities |

### Documentation

| File | Description |
|---|---|
| `treinar_unsloth.md` | Full documentation for the training package (Gemma-3, Deepseek, Llama, Qwen fine-tuning via YAML) |
| `treinar_TODO_PLANEJAMENTO.md` | Development planning and task-tracking (completed and pending items) |
| `treinar_readme_ollama.md` | Step-by-step guide to export a model to HF safetensors, convert to GGUF, and import into Ollama |
| `comparar_extracoes.md` | Technical documentation for the YAML-driven extraction comparison tool |

### Requirements

| File | Description |
|---|---|
| `requirements.txt` | Python dependencies for the general utilities |
| `requirements-unsloth.txt` | Python dependencies specific to the Unsloth training pipeline |

## Important

- The `.env_template` file in this directory is a **template only** and contains no real credentials and need to be copied to `.env` and filled with the actual credentials.

# Como as métricas de comparação avaliam os campos

## Definição de "efetivamente vazio"

Um valor de campo é considerado **efetivamente vazio** quando é:
- `None` (valor nulo)
- `""` (string vazia exata)
- `[]` (lista vazia)
- `{}` (dicionário vazio)

> **Nota:** Strings com espaços (`"   "`), pontuação residual (`"."`) ou estruturas aninhadas com todos os valores nulos (`{"a": null}`) **não** são consideradas vazias — o cálculo normal das métricas resolve esses casos.

## Chave ausente = valor nulo

Se a referência (professor) tem uma chave com valor nulo e o aluno não possui essa chave, ambos são tratados como equivalentes — o resultado é **1.0** (match perfeito).

## Regras de comparação para campos vazios

| Cenário | Resultado | Motivo |
|---|---|---|
| Ambos efetivamente vazios | **1.0** (P, R, F1 ou SIM) | Concordância: ambos indicam ausência de informação |
| Apenas um efetivamente vazio | **0.0** (P, R, F1 ou SIM) | Discordância: um extraiu, outro não |
| Nenhum vazio | Cálculo normal da métrica | Comparação padrão entre valores |

## Aplicação uniforme

Essas regras se aplicam **uniformemente** a todas as métricas:
- BERTScore
- Sentence-BERT (SBERT) — todos os tamanhos (pequeno, médio, grande)
- ROUGE-L, ROUGE-1, ROUGE-2
- Levenshtein

## Camadas de proteção

A verificação é feita em **duas camadas**:

1. **Camada de comparação de campos** (`JsonAnalise.comparar()`): verifica os valores JSON originais antes da conversão para texto. Se ambos são efetivamente vazios, retorna 1.0 diretamente sem invocar a métrica.

2. **Camada de métricas individuais** (`bscore()`, `sbert_score()`, ROUGE, Levenshtein): proteção defensiva sobre strings já convertidas. Se ambas as strings são vazias após `strip()`, retorna 1.0.

## Diferença entre campo vazio e documento com erro

- **Campo vazio**: o documento foi extraído com sucesso, mas o campo tem valor nulo/vazio. Segue as regras acima.
- **Documento com erro**: a extração falhou (dict com chave `'erro'`). É tratado pela flag `ignorar_erro_extracao` no YAML — se `true`, o documento é excluído da comparação; se `false`, recebe métricas zeradas.