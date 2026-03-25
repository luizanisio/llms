# Documentação do Treinamento com treinar_unsloth.py

## Descrição Geral

O pacote `treinar_unsloth.py` é uma ferramenta completa para fine-tuning de modelos LLM (Gemma-3, Deepseek, Llama, Qwen) usando Unsloth + TRL-SFTTrainer com configuração via arquivo YAML.

---

## Arquivos do Projeto

| Arquivo | Descrição |
|---------|-----------|
| `treinar_unsloth.py` | Script de treinamento e CLI (`LLMsTrainer`, `--treinar`, `--reset`) |
| `treinar_unsloth_avaliar.py` | Script de avaliação e estatísticas (`--info`, `--stats`, `gerar_graficos_estatisticos`) + CLI/menu interativo de avaliação |
| `treinar_realizar_predicoes.py` | Predição e inferência OOP: `UtilPredicao` (ABC) → `UtilPredicaoHF`, `UtilPredicaoVLLM`, `UtilPredicaoUnsloth`, `UtilPredicaoOllama` + CLI (`--engine`) |
| `treinar_unsloth_export.py` | Exportação (`executar_merge`) e helpers de predição (`_registro_ja_exportado`, `_construir_mapa_etapas`, `_copiar_para_pastas_etapas`) |
| `treinar_unsloth_actions.py` | Ações de treinamento (`executar_treinar`, `executar_reset`, `executar_injetar_dicas`) |
| `treinar_unsloth_util.py` | Utilitários de configuração (`YamlTreinamento`, `ConfigCurriculum`, `ConfigSaida`, `ConfigEntrada`, `ConfigPredicao`) e helpers |
| `treinar_unsloth_dataset.py` | Gerenciamento de datasets: carga, divisão, validação, criptografia (`DatasetTreinamento`) |
| `treinar_unsloth_pipeline.py` | Pipeline Universal: construção de etapas e `CurriculumTracker` (trava de conclusão) |
| `treinar_unsloth_graficos.py` | Gráficos: loss, eficiência (tokens), hardware (memória), boxplots de tokens |
| `treinar_unsloth_historico.py` | Histórico de treinamento: exemplos, info do modelo, eventos e versionamento YAML |
| `treinar_unsloth_logging.py` | Sistema centralizado de logging com níveis configuráveis |
| `treinar_unsloth_monitor.py` | Monitoramento contínuo de RAM/GPU |
| `treinar_unsloth_report.py` | Geração de relatórios em Markdown |
| `treinar_chat_templates.py` | Chat templates automáticos por modelo e `train_on_responses_only` |
| `treinar_unsloth_dicas.py` | Injeção de comentários de dicas no YAML (`--dicas`) |
| `util_print.py` | Utilitários de impressão: cores ANSI, `print_cores`, `exibir_menu_opcoes` (menus padronizados) |
| `treinar_unsloth.md` | Esta documentação |
| `treinar_TODO_PLANEJAMENTO.md` | Planejamento e roadmap de desenvolvimento |

---

## Características Principais

### Implementadas ✅

- [x] Carregamento de modelo base com suporte a quantização (4-bit, 8-bit)
- [x] Suporte a LoRA (Low-Rank Adaptation) configurável
- [x] Detecção e continuação de modelos LoRA já treinados
- [x] Gerenciamento automático de checkpoints
- [x] Suporte a múltiplas GPUs (via CUDA_VISIBLE_DEVICES)
- [x] Modo infos detalhado (`--info`)
- [x] Modo de estatísticas de tokens com gráficos (`--stats`)
- [x] Modo teste para validar predições (`--modelo N`) com opção `--base`
- [x] Geração de predições em massa (`--predict`) com opção `--base`
- [x] Suporte a entrada/saída via pasta de arquivos (pares .txt) ou dataframe parquet, com criptografia granular por campo
- [x] Divisão de datasets (treino, validação, teste) via CSV de divisão
- [x] Chat template automático baseado no modelo
- [x] Criptografia granular de dados sensíveis (Fernet) — por campo: `entrada.texto_criptografado`, `entrada.prompt_criptografado`, `saida.texto_criptografado`
- [x] Validação fail-fast da chave de criptografia (erro fatal se `misc.env_chave_criptografia` aponta para variável de ambiente inexistente)
- [x] Validação automática de consistência de dados (IDs e arquivos pareados)
- [x] Implementar exportação de modelo (GGUF, Merge)
- [x] Gerar gráficos de evolução de Loss (treino vs validação) ao final do treino
- [x] Geração automática de gráficos estatísticos pós-treinamento (loss, tokens, hardware, relatório .md)
- [x] Limpeza automática de gráficos anteriores ao iniciar novo treinamento (evita confusão com dados antigos)
- [x] Trava de conclusão: impede reexecução de treino já finalizado (com liberação via YAML)
- [x] Curriculum Learning: treinamento multietapa com transições e legendas visuais nos gráficos
- [x] Cálculo automático de `grad_batch_size` para manter batch efetivo constante independente do nº de GPUs
- [x] Formato unificado: seção `curriculum` como único formato de configuração de dados (entrada/saída suportam pasta ou dataframe)
- [x] Predição/Inferência via Ollama (API local) com `modelo.ollama` e `modelo.ollama_base`
- [x] Hierarquia OOP de motores de predição (`UtilPredicao` → HF, vLLM, Unsloth, Ollama) com saída padronizada e CLI unificado

---

## Uso via Linha de Comando

O sistema é dividido em dois scripts independentes:

### treinar_unsloth.py — Treinamento

```bash
python treinar_unsloth.py [CONFIG.yaml] [AÇÃO]
```

| Ação | Descrição |
|------|-----------|
| **(nenhuma)** | Modo interativo: seleciona YAML e ação via menu |
| `--treinar` | Inicia ou continua o treinamento |
| `--reset` | Limpa o treinamento atual (checkpoints e adaptador) |
| `--reset --treinar` | Limpa e treina do zero |
| `--dicas` | Injeta comentários de dicas no YAML de configuração |
| `--log-level LEVEL` | Define nível de log (DEBUG, INFO, WARNING, ERROR) |

### treinar_realizar_predicoes.py — Predição e Inferência (4 motores)

```bash
python treinar_realizar_predicoes.py [CONFIG.yaml] --engine {hf,vllm,unsloth,ollama} [AÇÃO]
```

| Ação | Descrição |
|------|-----------|
| `--predict` | Gera predições em lote para subsets |
| `--modelo N` | Inferência interativa com N exemplos |
| `--engine ENGINE` | Motor: `hf` (padrão), `vllm`, `unsloth`, `ollama` |
| `--base` | Força uso do modelo base |
| `--subsets LISTA` | Subsets específicos (ex: `teste,validacao`) |

```bash
# Exemplos:
python treinar_realizar_predicoes.py config.yaml --engine ollama --predict
python treinar_realizar_predicoes.py config.yaml --engine ollama --modelo 3
python treinar_realizar_predicoes.py config.yaml --engine vllm --predict --subsets teste
python treinar_realizar_predicoes.py config.yaml --engine hf --modelo 1 --base
```

### treinar_unsloth_avaliar.py — Avaliação, Inferência e Exportação

```bash
python treinar_unsloth_avaliar.py [CONFIG.yaml] [AÇÃO]
```

| Ação | Descrição |
|------|-----------|
| **(nenhuma)** | Modo interativo: seleciona YAML e ação via menu |
| `--info` | Informações detalhadas da configuração, datasets e modelo |
| `--stats` | Relatório estatístico de tokens e métricas de treinamento |
| `--predict` | Gera predições para todos os subsets (treino, validação, teste) |
| `--predict-treino` | Gera predições apenas do subset de treino |
| `--predict-validacao` | Gera predições apenas do subset de validação |
| `--predict-teste` | Gera predições apenas do subset de teste |
| `--modelo N` | Testa inferência interativa com N exemplos (padrão: 1) |
| `--merge` | Exporta modelo (merge LoRA + Base) |
| `--base` | Força uso do modelo base (ignora LoRA treinado) |
| `--quant METODO` | Quantização para merge (`16bit`, `4bit`, `q4_k_m`, `q8_0`) |
| `--log-level LEVEL` | Define nível de log (DEBUG, INFO, WARNING, ERROR) |

> **Nota:** Ambos os scripts aceitam o YAML como argumento opcional. Se omitido, exibem menu interativo para seleção do arquivo YAML via `util_menu_opcoes.escolher_yaml()`. Os menus de ação são renderizados pela função padronizada `exibir_menu_opcoes()` de `util_print.py`, que formata as opções como tabela visual alinhada com suporte a sub-itens, cores e seções.

### Exemplos de Uso

```bash
# === TREINAMENTO ===
python treinar_unsloth.py                                 # Modo interativo completo
python treinar_unsloth.py config.yaml                     # Seleciona ação via menu
python treinar_unsloth.py config.yaml --treinar           # Inicia treinamento
python treinar_unsloth.py config.yaml --reset --treinar   # Limpa e treina do zero

# === AVALIAÇÃO ===
python treinar_unsloth_avaliar.py                          # Modo interativo completo
python treinar_unsloth_avaliar.py config.yaml --info       # Informações detalhadas
python treinar_unsloth_avaliar.py config.yaml --stats      # Relatório estatístico
python treinar_unsloth_avaliar.py config.yaml --predict    # Predições de todos os subsets
python treinar_unsloth_avaliar.py config.yaml --predict --base  # Predições com modelo base
python treinar_unsloth_avaliar.py config.yaml --modelo 3   # Testar 3 exemplos
python treinar_unsloth_avaliar.py config.yaml --merge      # Exportar modelo
```

---

## Detalhes das Funcionalidades

### Relatório de Estatísticas (`--stats`)
Gera análise detalhada do consumo de tokens (entrada e saída) por subset.
*   **Saída**: `{output_dir}/treinamento/relatorio_estatistico.md`
*   **Gráficos**: Gera `stats_tokens_boxplot.png` contendo boxplots comparativos de todos os subsets (Entrada e Saída).
*   **Tabelas**: No relatório MD, apresenta tabelas com Min, Max, Média, Mediana, Desvio Padrão e Total por subset.

### Predição em Massa (`--predict`)
Gera respostas do modelo para os datasets configurados.
*   **Limpeza Segura**: Antes de iniciar, remove apenas arquivos `.json` e `.txt` da pasta de destino do subset, preservando outros arquivos.
*   **Arquivos Gerados**:
    *   `{id}.txt`: O texto da resposta gerada.
    *   `{id}.json`: Metadados (tempo, tokens, preview do prompt).
    *   `resumo.json`: Estatísticas consolidadas da execução.

### Configuração YAML

A configuração é centralizada em arquivo YAML. A seção `curriculum` é o formato único de configuração de dados.

#### Curriculum: Entrada, Saída e Predição
O YAML organiza os dados em três seções dentro de `curriculum`:
- **`saida`**: Gold standard (saídas esperadas). Suporta pasta de arquivos OU dataframe parquet. Obrigatório.
- **`entrada`**: Textos de entrada. Suporta pasta de arquivos OU dataframe parquet.
- **`predicao`**: Saída das predições do modelo. Criada automaticamente se não existir.

```yaml
curriculum:
  saida:
    #| Gold dataset: saídas esperadas usadas como alvo no treino (OBRIGATÓRIO)
    #| Modo pasta: ID = nome do arquivo sem extensão
    pasta: ./saidas_fold11/ext_qwen235b_11
    mascara: "*.txt"
    formato: json              # 'json' ou 'texto' (padrão: texto)
    # OU modo dataframe:
    # dataframe: ./saidas/gold.parquet
    # dataframe_col: saida
    # dataframe_id: id_peca
    # texto_criptografado: false
  predicao:
    #| Pasta onde serão gravadas as predições do modelo para avaliação (criada automaticamente)
    pasta: ./treino_simples/predict/ext_qwen1_5b_11
  entrada:
    #| Modo dataframe (texto de entrada via parquet):
    dataframe: ./saidas/pecas_exportadas_textos.parquet
    dataframe_col: texto
    dataframe_id: id_peca
    texto_criptografado: false   # Se true, descriptografa o texto
    prompt_criptografado: false  # Se true, descriptografa o template
    # OU modo pasta:
    # pasta: ./saidas/textos
    # mascara: "*.txt"
    prompt_template: './saidas/prompt_summa_raw.txt'
    tag_texto: '<<--TEXTO-->>'
```

#### Divisão e Pipeline
A subchave `divisao` define as etapas do pipeline de treinamento. Uma única etapa = treinamento simples.
- **`validar_ids`**: Quando `true`, levanta erro fatal se existem IDs nos arquivos que não estejam no CSV de divisão. Quando `false`, apenas emite aviso.

```yaml
curriculum:
  divisao:
    # Etapa única (treinamento simples):
    - arquivo: "caminho/para/divisao.csv"
      alias: "Principal"
      tipo: "lora"
      pace_epochs: 3
      proporcao:
        - treino: 0.7
        - validacao: 0.1
        - teste: 0.2
      seed: 42
      validar_ids: true
    # Ou múltiplas etapas (Curriculum Learning):
    # - arquivo: ./divisao_facil.csv
    #   alias: "fácil"
    #   pace_epochs: 1
    # - arquivo: ./divisao_medio.csv
    #   alias: "médio"
    #   pace_epochs: 2
```

#### Criptografia de Dados (`misc.env_chave_criptografia`)
Quando os dados de entrada ou saída estão criptografados com Fernet, configure o nome da variável de ambiente que contém a chave e ative os flags nos campos correspondentes:
```yaml
misc:
  env_chave_criptografia: CHAVE_CRIPT  # Nome da variável de ambiente

curriculum:
  entrada:
    texto_criptografado: true   # Descriptografa texto de entrada
    prompt_criptografado: false  # Descriptografa template de prompt
  saida:
    texto_criptografado: true   # Descriptografa saídas esperadas
```
**Validação fail-fast:** Se qualquer flag `*_criptografado` estiver ativo, o sistema verifica imediatamente se a variável de ambiente (`misc.env_chave_criptografia`) está definida e não está vazia. Caso contrário, levanta `EnvironmentError` impedindo que o treinamento prossiga com texto criptografado.

#### Modelo Ollama (`modelo.ollama` / `modelo.ollama_base`)
Para usar o Ollama como motor de predição/inferência, configure os nomes dos modelos na seção `modelo`:
```yaml
modelo:
  base_model_name: Qwen/Qwen2.5-1.5B-Instruct
  saida: ./treino_simples/Qwen2.5-1.5B-Instruct_simples
  ollama: meu-modelo-treinado:latest   # Modelo treinado (convertido para GGUF e importado no Ollama)
  ollama_base: qwen2.5:1.5b            # Modelo base (não treinado, disponível no Ollama)
  ollama_url:                           # URL da API Ollama (opcional, padrão: http://localhost:11434/api)
```
- **`ollama`**: Nome do modelo treinado no Ollama. Ativa as opções `3o` (predict) e `5o` (modelo) no menu interativo.
- **`ollama_base`**: Nome do modelo base no Ollama. Ativa as opções `4o` (predict-base) e `6o` (modelo-base) no menu.
- **`ollama_url`**: URL customizada da API Ollama (para servidores remotos). Se vazio, usa `http://localhost:11434/api`.
- O timeout da API é configurável via variável de ambiente `OLLAMA_TIMEOUT` (padrão: 600 segundos).
- A janela de contexto (`num_ctx`) é automaticamente configurada como `max_seq_length × 2`.

#### Train on Responses Only
Treina o modelo apenas nas respostas do assistente, ignorando o loss dos prompts do usuário. (sugerido no unsloth)
```yaml
treinamento:
  train_on_responses_only: true
```

#### Curriculum Learning
A seção `curriculum` é o formato único de configuração de dados. Um treinamento de etapa única é configurado como curriculum com uma divisão. A subchave `divisao` contém a lista de etapas do pipeline, cada uma com seu próprio arquivo CSV de divisão e parâmetros de treino.
```yaml
curriculum:
  predicao:
    pasta: ./predict/output
  saida:
    pasta: ./saidas/gold
    mascara: "*.txt"
    formato: json
  entrada:
    dataframe: ./dados/textos.parquet
    dataframe_col: texto
    dataframe_id: id_peca
  validacao:
    exigir_json_valido: true
  divisao:
    - arquivo: ./divisao_facil.csv
      alias: "fácil"
      tipo: "lora"
      pace_epochs: 1
    - arquivo: ./divisao_medio.csv
      alias: "médio"
      tipo: "lora"
      pace_epochs: 2
```

##### Batch Size Automático (`treinamento.batch_size`)
Quando `treinamento.batch_size` é configurado como dicionário com `efetivo` e `batch_size`, o sistema calcula `grad_batch_size` automaticamente para atingir o batch efetivo desejado, independente do número de GPUs. O `batch_size` por GPU é fixo (determinado empiricamente para evitar OOM).

```yaml
treinamento:
  batch_size:
    efetivo: 16  # Batch efetivo (ajusta grad_batch_size automaticamente)
    batch_size: 2  # Batch por GPU (verificar empiricamente para evitar OOM)
```

**Fórmula:** `grad_batch_size = round(efetivo / (batch_size × n_gpus))`, mínimo 1.

| GPUs | batch_size | efetivo desejado | grad_batch_size calculado | efetivo real |
|------|-----------|-----------------|--------------------------|-------------|
| 1    | 2         | 16              | 8                        | 16          |
| 2    | 2         | 16              | 4                        | 16          |
| 3    | 2         | 16              | 3                        | 18 ≈ 16     |
| 4    | 2         | 16              | 2                        | 16          |

> **Nota:** Quando `batch_size.efetivo` está configurado, os valores de `treinamento.batch_size` e `treinamento.grad_batch_size` são sobrescritos automaticamente. Se a seção `batch_size` não existir ou `efetivo` for 0, os valores manuais de `treinamento` são usados normalmente.

---

## Arquivos de Saída e Métricas

Durante o treinamento e testes, diversos arquivos são gerados na pasta de saída (`modelo.saida`):

| Pasta/Arquivo | Conteúdo |
|---------------|----------|
| `adapter_model.safetensors` | Pesos do LoRA treinado |
| `treinamento/training_metrics.jsonl` | Métricas unificadas: treino (loss, lr, epoch) + hardware (CPU, RAM, GPU, Disco) |
| `treinamento/relatorio_estatistico.md` | Relatório estatístico (gerado automaticamente pós-treino e via `--stats`) |
| `treinamento/treinamento_loss.png` | Gráfico de evolução do loss (treino vs validação, com marcadores de curriculum) |
| `treinamento/treinamento_tokens.png` | Gráfico de custo computacional (tokens e instâncias acumulados) |
| `treinamento/hardware_memoria.png` | Gráfico de uso de memória (RAM + GPU VRAM) durante o treinamento |
| `treinamento/stats_tokens_boxplot.png` | Boxplots comparativos de tokens por subset (gerado apenas via `--stats`) |
| `treinamento/memoria_predicao.png` | Gráfico de uso de memória gerado durante teste (`--modelo`) |
| `chkpt/` | Checkpoints do treinamento (zero-padded: `checkpoint-00001`) |
| `predict/` | Resultados da predição com modelo treinado |
| `predict_base/` | Resultados da predição com modelo base (`--base`) |

> **Nota:** Ao iniciar um novo treinamento, os gráficos (`treinamento_loss.png`, `treinamento_tokens.png`, `hardware_memoria.png`) e o `relatorio_estatistico.md` são automaticamente removidos para evitar confusão com dados de treinos anteriores. Eles são regenerados ao final do treinamento.

---

## Monitoramento

O sistema inclui monitoramento de recursos em background (`treinar_unsloth_monitor.py`):
1.  **Durante Treino**: Registra em `training_metrics.jsonl` (arquivo unificado com métricas de treino e hardware).
2.  **Durante Teste (`--modelo`)**: Coleta dados em tempo real e gera gráfico ao final.
3.  **Logs**: Exibe consumo de VRAM e RAM nos logs de execução.
---

## Histórico de Treinamento

O módulo `treinar_unsloth_historico.py` organiza informações estruturadas na pasta `{modelo.saida}/treinamento/`:

| Arquivo | Conteúdo |
|---------|----------|
| `treinamento_exemplos.md` | Um exemplo de treino e um de validação (sem campos internos de tokenização) |
| `modelo_info.md` | Modelo base, tipo, configuração LoRA, parâmetros, chat template e arquitetura resumida |
| `treinamento_config/` | Cópias versionadas do YAML: `nome (v001).yaml`, `(v002).yaml`, etc. |
| `treinamento_eventos.md` | Eventos de alto nível com data/hora (checkpoints, etapas, conclusão) |

**Regras de criação:**
- **Novo treinamento** (sem `adapter_config.json`): todos os 4 arquivos são criados do zero.
- **Continuação**: apenas `treinamento_eventos.md` é atualizado com um separador de nova sessão. Se o YAML foi modificado desde a última cópia, uma nova versão `(vN+1).yaml` é criada automaticamente.
- **Reset** (`--reset`): todos os arquivos de histórico são apagados junto com checkpoints e modelo LoRA.

---

## Continuidade e Trava de Conclusão

Para evitar o retrabalho indevido ou degradação de um modelo que já teve sucesso, o sistema inclui um **rastreador integrado de status**.

- **Continuação Automática ("Resume")**: Sempre que um script de treinamento é reexecutado na mesma pasta, o sistema localiza automaticamente o último `checkpoint` e retoma de onde parou.
- **Trava de Segurança (Early Exit)**: Se o treinamento for reaberto em uma pasta cuja última etapa do Curriculum já foi encerrada, o sistema detecta que o **objetivo final foi atingido** e impede seu avanço com uma mensagem informacional de *Concluído*.
- **Como estender um modelo retido pela "Trava"**:
   - Para adicionar mais etapas, basta ir no `.yaml` e adicionar uma nova etapa na lista `curriculum.divisao` para que a trava reconheça o novo requisito e seja liberada.
   - Para treinar mais épocas, ao **aumentar o número de épocas** (`pace_epochs` na etapa ou `epochs` global), o sistema identificará que o alvo cresceu em relação ao que constava no `curriculum_state.json`, destravando automaticamente.
   - ⚠️ *Dica: passar o argumento `--reset` reinicia do Zero ABSOLUTO (apaga logs, histórico e os checkpoints LoRA gerados). Utilize o reset apenas se desejar remover todo o treinamento realizado e começar o treinamento novamente a partir do modelo base.*

---

## Limitações Conhecidas

### 1. vLLM + Modelos Instruct — Chat Template Obrigatório

Modelos instruct (e.g. Qwen2.5-7B-**Instruct**, Llama-3-**Instruct**) esperam receber o prompt formatado com tokens especiais de chat (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` no caso do Qwen). Se o texto bruto é enviado ao vLLM sem `apply_chat_template()`, o modelo gera EOS imediatamente — resultando em predições vazias com `output_tokens=1`.

**Mitigação implementada:** Todos os motores vLLM (OOP e legacy) aplicam `apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)` antes da inferência, filtrando mensagens de role `assistant` e preservando tokens especiais na truncagem (`skip_special_tokens=False`).

> **Nota:** Esse comportamento é específico de modelos **instruct**. Modelos base (sem sufixo Instruct/Chat) podem aceitar texto bruto, mas a prática recomendada é sempre usar o chat template do tokenizer.

### 2. vLLM + `max_position_embeddings` (Qwen e outros)

O vLLM, por padrão, recusa alocar `max_model_len` superior ao `max_position_embeddings` declarado no `config.json` do modelo. Se a estimativa de contexto (`estimar_contexto_predict()`) produz um valor maior — por exemplo, 33.792 contra 32.768 do Qwen2.5 — o vLLM levanta `ValueError`.

**Mitigação implementada:** A variável `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` é definida via `os.environ.setdefault()` antes da inicialização do `LLM()` em `treinar_vllm_inference.py`. Isso habilita RoPE scaling automático, permitindo que o vLLM aceite contextos acima do `max_position_embeddings` sem abortar. O `estimar_contexto_predict()` **não limita** o contexto estimado — apenas loga um aviso informativo quando o valor excede `max_position_embeddings`.

> **Nota:** O erro `RuntimeError: cancelled` que ocorria com contextos grandes era na verdade causado pelo `tensor_parallel_size > 1` (ver limitação 3 abaixo), não pelo tamanho do contexto em si.

**Valores de referência:**

| Modelo | `max_position_embeddings` |
|--------|--------------------------|
| Qwen2.5-7B-Instruct | 32.768 |
| Qwen2.5-1.5B-Instruct | 32.768 |
| Llama-3.1-8B-Instruct | 131.072 |
| Gemma-3-4B-IT | 131.072 |

### 3. vLLM + Tensor Parallelism (Multi-GPU) — `RuntimeError: cancelled`

Com `tensor_parallel_size > 1`, o vLLM utiliza o `multiproc_executor` para coordenar múltiplas GPUs via shared memory (NCCL + `shm_broadcast`). Durante a inicialização do KV cache (`_initialize_kv_caches` → `determine_available_memory`), se um dos processos worker falha, o processo pai recebe `RuntimeError: cancelled` seguido de vazamentos de semáforos e shared memory.

**Causa raiz:** Para modelos ≤7B, tensor parallelism é **desnecessário** — o modelo inteiro (~14 GB em FP16) cabe confortavelmente em uma única GPU moderna (≥16 GB). O TP>1 para modelos pequenos adiciona overhead de coordenação inter-GPU sem ganho real de throughput, e introduz pontos de falha na comunicação NCCL/shared memory.

**Sintoma típico:**
```
(EngineCore pid=...) File ".../vllm/v1/executor/multiproc_executor.py", line 397, in collective_rpc
(EngineCore pid=...) File ".../vllm/distributed/device_communicators/shm_broadcast.py", line 677, in acquire_read
(EngineCore pid=...)     raise RuntimeError("cancelled")
resource_tracker: There appear to be 2 leaked semaphore objects to clean up at shutdown
```

**Mitigação implementada:**

| Ação | Descrição |
|------|-----------|
| **TP=1 para 7B** | `get_recommended_config()` retorna `tensor_parallel_size=1` para modelos 7B independente do número de GPUs disponíveis |
| **`enforce_eager=True` para TP>1** | Configurações de modelos maiores (13B, 70B) que usam TP>1 desabilitam CUDA graphs (`enforce_eager=True`) para evitar picos de memória durante a captura dos grafos |

> **Nota:** Se o modelo crescer para 13B+ e TP>1 for necessário, verifique que todas as GPUs estejam livres antes de inicializar o vLLM. Processos residuais de execuções anteriores podem ocupar memória e causar falha na coordenação. Use `nvidia-smi` para verificar e `kill` para limpar processos órfãos.

### 4. `_ler_max_position_embeddings()` — Apenas Modelos Locais

O método `_ler_max_position_embeddings()` lê o arquivo `config.json` diretamente do caminho em `modelo.base`. Se esse caminho for um identificador do HuggingFace Hub (e.g. `Qwen/Qwen2.5-7B-Instruct`) em vez de um diretório local, o método retorna `0` e o contexto **não é limitado automaticamente**.

**Recomendação:** Para predição via vLLM, configure `modelo.base` apontando para o diretório local do modelo (onde o `config.json` está presente). Se estiver usando diretamente do Hub, a 2ª camada (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`) ainda oferece proteção parcial.

### 5. Estimativa de Contexto — Dependência de Colunas CSV

O método `estimar_contexto_predict()` depende das colunas `token_total` e `token_output` nos arquivos CSV de divisão do curriculum. Se essas colunas não existirem, o sistema utiliza o fallback `2× max_seq_length`, que pode ser insuficiente ou excessivo dependendo da distribuição real dos dados.

**Recomendação:** Sempre gere os CSVs de divisão com as colunas de contagem de tokens. O comando `--stats` exibe a distribuição de tokens e ajuda a validar se as estimativas estão adequadas.

---

## Desenvolvimento e Manutenção

### Pendências Concluídas Recentemente ✅
1.  **Refatoração de Ações**: Lógica CLI movida para `treinar_unsloth_actions.py`.
2.  **Separação de Dataset**: Lógica complexa movida para `treinar_unsloth_dataset.py`.
3.  **Segurança**: Limpeza seletiva em `--predict`.
4.  **Flexibilidade**: Adição de flag `--base` e suporte a múltiplos subsets em stats.
5.  **Qualidade**: Correção de logs duplicados e bugs de formatação em relatórios.
6.  **Separação Treino/Avaliação (Passo 1)**: Script `treinar_unsloth_avaliar.py` criado com lógica de avaliação, estatísticas e CLI interativo. Funções de exportação/inferência em `treinar_unsloth_export.py`.
7.  **Separação `saida` / `predicao` no YAML**: `ConfigSaida` para o gold dataset (saídas esperadas, obrigatório). `ConfigPredicao` é pasta de saída das predições (criada automaticamente se não existir).
8.  **Menu Interativo YAML**: Ambos os scripts usam `util_menu_opcoes.escolher_yaml()` quando YAML é omitido, com menus de ação específicos para cada script.
9.  **Validação Fail-Fast de Criptografia**: `ConfigMisc.__post_init__` e `_carregar_dataframe_entrada` levantam erro fatal se `misc.env_chave_criptografia` está configurada no YAML mas a variável de ambiente não existe, impedindo treinamento com dados criptografados.
10. **Geração Automática de Gráficos Pós-Treinamento**: Função `gerar_graficos_estatisticos()` extraída e reutilizada por `--stats` e `executar_treinar()`. Gera loss, tokens, hardware e relatório .md automaticamente ao final do treino.
11. **Limpeza de Artefatos Antigos**: `MetricsLoggerCallback` remove gráficos e relatório estatístico anteriores ao iniciar novo treinamento, evitando confusão com dados de treinos passados.
12. **Batch Size Automático**: Seção `treinamento.batch_size` (dict) com `efetivo` e `batch_size`. O sistema calcula `grad_batch_size = round(efetivo / (batch_size × n_gpus))` automaticamente.
13. **Simplificação do YAML — Curriculum como Formato Único**: Removidos os modos `pastas` e `dataset` (`formatos.tipo_entrada`). Seção `curriculum` é o único formato de configuração de dados. Entrada e saída suportam pasta de arquivos OU dataframe parquet. Criptografia granular por campo. Classes removidas: `ConfigFormatos`, `ConfigGold`, `ConfigPastas`, `ConfigDataset`. Classes adicionadas: `ConfigSaida`, `ConfigCurriculum`.
14. **Menus Padronizados com `exibir_menu_opcoes`**: Função reutilizável em `util_print.py` que renderiza menus como tabela visual alinhada. Menus de treinamento e avaliação unificados, eliminando formatação manual.
15. **Separação Exportação/Inferência**: Módulo `treinar_unsloth_export.py` criado com funções de merge/zip e helpers (skip de registros exportados, cópia para pastas de etapas do curriculum). `treinar_unsloth_avaliar.py` mantém apenas avaliação, estatísticas e CLI.
16. **Refatoração OOP de Predição/Inferência**: Módulo `treinar_realizar_predicoes.py` (~1100 linhas) com hierarquia de classes `UtilPredicao` (ABC) → `UtilPredicaoHF`, `UtilPredicaoVLLM`, `UtilPredicaoUnsloth`, `UtilPredicaoOllama`. Elimina duplicação de código entre os motores. Saída padronizada compatível com `get_resposta()`. CLI independente via `--engine`. `treinar_unsloth_avaliar.py` despacha para classes.
17. **Suporte a Ollama base model**: Chave `modelo.ollama_base` no YAML para inferência com modelo base via Ollama. Menu com opções `4o`/`6o` com detecção independente de disponibilidade.

### Histórico de Desenvolvimento
> Curriculum Learning, simplificação do código e unificação de formatos

**Objetivo:** Permitir um fluxo de treinamento em múltiplos estágios (Curriculum Learning) alternando dados, estratégias (LoRA vs Full Fine-Tuning) e critérios de parada dinâmicos (Pace), com formato de configuração unificado.

O desenvolvimento foi dividido nas seguintes etapas incrementais:

#### Passo 1: Separação de Preocupações e Melhoria do CLI ✅ CONCLUÍDO
**Objetivo:** Desacoplar a inferência do motor de treinamento para blindar e otimizar o código base do Treinador, centralizando o treinamento para focar *apenas Treinar e dar Merge*, e melhorar a experiência CLI.

**Implementado:**
1. ✅ **Extração da Avaliação/Inferência:** Funções de avaliação (`executar_info`, `executar_stats`, `gerar_graficos_estatisticos`) em `treinar_unsloth_avaliar.py`. Predição e inferência via hierarquia OOP em `treinar_realizar_predicoes.py` (`UtilPredicao` → HF, vLLM, Unsloth, Ollama). Exportação (`executar_merge`) e helpers em `treinar_unsloth_export.py`. Removidas de `treinar_unsloth_actions.py` (que agora contém apenas `executar_treinar`, `executar_reset`, `executar_injetar_dicas` e funções auxiliares compartilhadas).
2. ✅ **Três Módulos Independentes:** `treinar_unsloth_avaliar.py` (~950 linhas) com CLI/menu interativo e funções de avaliação/estatísticas. `treinar_realizar_predicoes.py` (~1100 linhas) com classes de predição/inferência (4 motores + CLI). `treinar_unsloth_export.py` com merge e helpers.
3. ✅ **Menu Interativo (CLI):** Ambos os scripts usam `util_menu_opcoes.escolher_yaml(chave_obrigatoria='modelo')`. Argumento `config` é `nargs='?'` (opcional). Menu de ações específico: treino (treinar, reset+treinar, reset) e avaliação (info, stats, predict, modelo, merge).
4. ✅ **Separação `dataset`/`predicao`:** Nova `ConfigSaida` para o gold dataset (saídas esperadas, obrigatório, validação de existência). `ConfigPredicao` é pasta de saída (auto-criada). `parear_arquivos()` em `treinar_unsloth_dataset.py` usa `curriculum.saida` como fonte do gold.
* **⏱️ Teste Intermediário:** `--help` de ambos os scripts funciona. `--info` executa corretamente. Import de todos os módulos validado. Falta: teste completo de treinamento end-to-end e predição em massa.

#### Passo 2: O "Pipeline Universal" e Ajustes Finos (Pré Curriculum) ✅ CONCLUÍDO
**Objetivo:** Unificar a base de código do sistema atual antes de construir o Curriculum Learning multicamadas, assim o processo opera o mesmo sistema de logs (como se fosse de "apenas uma etapa").

1. **Pipeline Universal:** A seção `curriculum` é o formato único. Um treinamento simples (etapa única) é representado como curriculum com uma divisão. Toda parte de tracking funciona em cima desta lista universal.
2. **Log de Rastreamento Unificado e Resumo:** Todo salvamento utiliza métricas gravadas no esquema universal (`curriculum_state.json` e `curriculum_metrics.jsonl`).
3. **Mixando Modelos (LoRA vs Full):**
    * *Transição `[LoRA -> Full]`*: Mesclar base + lora via instanciador e usar o Merge como "o novo `FastLanguageModel` pleno" da segunda fase.
    * *Transição `[Full -> LoRA]`*: A requantização p/ nbits deve ser estritamente reacendida e embutida na modelagem ` FastLanguageModel.get_peft_model()` que sucede a transição.
4. ✅ **Simplificação do `max_seq_length` e Remoção do Cache:** Cálculo automático (e _dados_automaticos.json) removidos. Substituídos por comportamento estrito (ver Passo 3).
* **⏱️ Teste Intermediário:** Rodar um treinamento normal de teste (`pastas`) exigindo que o código passe perfeitamente sem as checagens e cálculos automáticos de contexto e não trave por arquivos de cache.

#### Passo 3: Motor Multietapas do Curriculum Learning
**Objetivo:** Adicionar interpretador do YAML para Curriculum, transições e regras de `LoRA` \leftrightarrow `Full`.

1. ✅ **Estrutura YAML (Curriculum Unificado):** A seção `curriculum` é o único formato de configuração de dados:
```yaml
curriculum:
  predicao:
    pasta: ./predict/output
  saida:
    pasta: ./saidas/gold       # OU dataframe: ./saidas/gold.parquet
    mascara: "*.txt"
    formato: json
    texto_criptografado: false
  entrada:
    dataframe: ./dados/textos.parquet  # OU pasta: ./dados/textos
    dataframe_col: texto
    dataframe_id: id_peca
    prompt_template: './dados/prompt.txt'
    tag_texto: '<<--TEXTO-->>'
    texto_criptografado: false
    prompt_criptografado: false
  validacao:
    exigir_json_valido: true
    exigir_ids_pareados: true
  divisao:
    - arquivo: "./saidas/divisao_facil.csv"
      alias: "fácil"
      tipo: "full"
      pace_epochs: 1
      max_seq_length: 512
      learning_rate: 0.0003
    - arquivo: "./saidas/divisao_medio.csv"
      alias: "médio"
      tipo: "lora"
      pace_loss: 0.015
      pace_epochs: 2
```
2. ✅ **Divisão Dinâmica ("Fail Fast"):** O sistema aborta com erro fatal se os arquivos CSV de divisão parametrizados não existirem.
3. ✅ **Roteamento e Sobrevivência de Passos:**
    * Encapsular salvamentos no formato de roteamento (ex: `{modelo.saida}/curriculum/01_facil`). Onde um retoma o modelo do passado.
    * No caso de Resume (`--treinar` de checkpoint quebrado), utilizar do state vivo (`curriculum_state.json`) para instanciar subpastas `checkpoint-N` precisas resgatando o ponto exato.
4. ✅ **Gestão do `max_seq_length` por Estágio (Simplificado):**
    * **Remoção do Cache Complexo**: Cálculo automático e cache (`_dados_automaticos.json`) removidos.
    * **Parâmetro Global Obrigatório**: `max_seq_length` é obrigatório (> 0). Erro fatal se zerado ou ausente.
    * **Recarga Dinâmica**: Se `max_seq_length` muda entre etapas, o modelo e tokenizer são recarregados automaticamente.

#### Passo 4: Pace Dinâmico e Ajustes Visuais de Controle
**Objetivo:** Interpolação analítica final garantindo eficiência via parada prematura baseada no desempenho e legibilidade de análise das métricas via Gráficos evolutivos de múltiplas fases.

1. **Pacing / Early Stopping Configurable:**
    * Programar `TrainerCallback` customizado plugado no Unsloth, que intercepta pós gatilhos `on_evaluate()`.
    * Checar `eval_loss <= pace_loss` ou total epochs se aproximarem de `pace_epochs`, injetar `control.should_training_stop = True` finalizando imediatamente aquela frente do currículo para poupar horas de servidor.
2. ✅ **Métricas de Eficiência Analíticas Estendidas:** `MetricsLoggerCallback` registra em `training_metrics.jsonl`: loss, lr, grad_norm, eval_loss, instâncias acumuladas, tokens acumulados, hardware (CPU, RAM, GPU), etapa do curriculum, step/epoch global. Gráficos de eficiência (tokens/instâncias) e hardware (memória) são gerados automaticamente.
3. ✅ **Legendagem Visual do Gráfico (Loss):** Implementado em `GraficoTreinamento.evolucao_loss()` e `construir_marcadores_etapas()`. Linhas violeta marcam transições entre etapas do curriculum com legendas dos aliases. Gráficos de eficiência e hardware também exibem marcadores de etapas.
4. ✅ **Controle de Conclusão e Retomada:** Trava estrutural implementada via `CurriculumTracker`. Bloqueia continuações de treinos já concluídos. Liberação automática ao adicionar etapas ou aumentar épocas no YAML.
5. ✅ **Geração Automática de Gráficos Pós-Treinamento:** Função `gerar_graficos_estatisticos()` em `treinar_unsloth_avaliar.py` é chamada automaticamente ao final de `executar_treinar()`. Gera loss, tokens, hardware e relatório .md sem necessidade de rodar `--stats` manualmente.
6. ✅ **Limpeza de Artefatos Antigos:** Ao iniciar novo treinamento (`etapa_index == 0`), o `MetricsLoggerCallback` remove gráficos e relatório estatístico anteriores (`treinamento_loss.png`, `treinamento_tokens.png`, `hardware_memoria.png`, `relatorio_estatistico.md`) junto com a truncagem do `training_metrics.jsonl`.
7. ✅ **Validação Fail-Fast de Criptografia:** `ConfigMisc.__post_init__` verifica se a variável de ambiente de criptografia existe quando configurada no YAML. `_carregar_dataframe_entrada` levanta erros em vez de silenciosamente continuar com dados criptografados.
* **⏱️ Teste Intermediário Final:** Processar múltiplos estágios usando Curriculum completo. Embutir propositalmente uma meta `pace_loss = 1.5` de fácil alcance numa das passagens e testar os limites do Early-Stopping e o respectivo avanço para a etapa 2. Constatar a divisão formatada do gráfico unificado renderizado em `.png` ao encerramento pleno.