# Documentação do Treinamento com treinar_unsloth.py

## Descrição Geral

O pacote `treinar_unsloth.py` é uma ferramenta completa para fine-tuning de modelos LLM (Gemma-3, Deepseek, Llama, Qwen) usando Unsloth + TRL-SFTTrainer com configuração via arquivo YAML.

---

## Arquivos do Projeto

| Arquivo | Descrição |
|---------|-----------|
| `treinar_unsloth.py` | Script de treinamento e CLI (`LLMsTrainer`, `--treinar`, `--reset`) |
| `treinar_unsloth_avaliar.py` | Script de avaliação, inferência e exportação (`--info`, `--stats`, `--predict`, `--modelo`, `--merge`, `gerar_graficos_estatisticos`) |
| `treinar_unsloth_actions.py` | Ações de treinamento (`executar_treinar`, `executar_reset`, `executar_injetar_dicas`) |
| `treinar_unsloth_util.py` | Utilitários de configuração (`YamlTreinamento`, `ConfigMisc`, `ConfigGold`, `ConfigPredicao`) e helpers |
| `treinar_unsloth_dataset.py` | Gerenciamento de datasets: carga, divisão, validação, criptografia (`DatasetTreinamento`) |
| `treinar_unsloth_pipeline.py` | Pipeline Universal: construção de etapas e `CurriculumTracker` (trava de conclusão) |
| `treinar_unsloth_graficos.py` | Gráficos: loss, eficiência (tokens), hardware (memória), boxplots de tokens |
| `treinar_unsloth_historico.py` | Histórico de treinamento: exemplos, info do modelo, eventos e versionamento YAML |
| `treinar_unsloth_logging.py` | Sistema centralizado de logging com níveis configuráveis |
| `treinar_unsloth_monitor.py` | Monitoramento contínuo de RAM/GPU |
| `treinar_unsloth_report.py` | Geração de relatórios em Markdown |
| `treinar_chat_templates.py` | Chat templates automáticos por modelo e `train_on_responses_only` |
| `treinar_unsloth_dicas.py` | Injeção de comentários de dicas no YAML (`--dicas`) |
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
- [x] Suporte a datasets em formado de pastas (pares .txt), arquivos únicos (csv/parquet) e curriculum (múltiplas divisões)
- [x] Divisão automática de datasets (treino, validação, teste) via YAML ou CSV
- [x] Chat template automático baseado no modelo
- [x] Criptografia de dados sensíveis (Fernet)
- [x] Validação fail-fast da chave de criptografia (erro fatal se `misc.env_chave_criptografia` aponta para variável de ambiente inexistente)
- [x] Validação automática de consistência de dados (IDs e arquivos pareados)
- [x] Implementar exportação de modelo (GGUF, Merge)
- [x] Gerar gráficos de evolução de Loss (treino vs validação) ao final do treino
- [x] Geração automática de gráficos estatísticos pós-treinamento (loss, tokens, hardware, relatório .md)
- [x] Limpeza automática de gráficos anteriores ao iniciar novo treinamento (evita confusão com dados antigos)
- [x] Trava de conclusão: impede reexecução de treino já finalizado (com liberação via YAML)
- [x] Curriculum Learning: transições de etapas com legendas visuais nos gráficos
- [x] Cálculo automático de `grad_batch_size` para manter batch efetivo constante independente do nº de GPUs (todos os formatos: pastas, dataset, curriculum)

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

> **Nota:** Ambos os scripts aceitam o YAML como argumento opcional. Se omitido, exibem menu interativo para seleção do arquivo YAML via `util_menu_opcoes.escolher_yaml()`.

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

A configuração é centralizada em arquivo YAML. Principais seções:

#### Pastas: Dataset e Predição
O YAML separa claramente os dados de **entrada** (gold dataset para treino) dos de **saída** (predições geradas pelo modelo):

```yaml
pastas:
  dataset:
    #| Gold dataset: pasta com as saídas esperadas usadas como alvo no treino (OBRIGATÓRIO)
    #| O nome do arquivo sem extensão é o ID ligado ao dataframe e ao arquivo de divisão
    pasta: ./saidas_fold11/ext_qwen235b_11
    mascara: "*.txt"
  predicao:
    #| Pasta onde serão gravadas as predições do modelo para avaliação (criada automaticamente)
    pasta: ./treino_simples/predict/ext_qwen1_5b_11
  entrada:
    dataframe: ./saidas/pecas_exportadas_textos.parquet
    dataframe_col: texto
    dataframe_id: id_peca
    prompt_template: './saidas/prompt_summa_raw.txt'
    tag_texto: '<<--TEXTO-->>'
```

- **`pastas.dataset`**: Gold standard (saídas esperadas). Obrigatório. Erro se a pasta não existir.
- **`pastas.predicao`**: Saída das predições do modelo. Criada automaticamente se não existir.
- **`pastas.entrada`**: Textos de entrada (via `dataframe` ou `pasta` de arquivos).

#### Proporção e Divisão
Define como os dados são divididos em subconjuntos para treinamento se não houver um arquivo CSV de divisão prévia.
- **`validar_ids`**: Quando `true`, levanta um erro fatal caso existam arquivos nas pastas (pareados) que não estejam mapeados no arquivo CSV de divisão prévio. Quando `false`, processa normalmente e apenas emite um aviso sobre os arquivos ignorados – excelente para usar apenas uma amostra do conjunto total de arquivos em disco.

```yaml
pastas:
  divisao:
    arquivo: "caminho/para/divisao.csv" # Opcional: Caminho para fixar/ler a divisão. Se não existir, será criado.
    proporcao: # Lista com a proporção de divisão de cada conjunto (deve somar 1.0)
      - treino: 0.7
      - validacao: 0.1
      - teste: 0.2
    seed: 42 # Opcional: Semente aleatória (Padrão: 42)
    validar_ids: true # Opcional: Verifica a integridade dos IDs (Padrão: true)
```

#### Criptografia de Dados (`misc.env_chave_criptografia`)
Quando os dados de entrada (dataframe parquet) estão criptografados com Fernet, configure o nome da variável de ambiente que contém a chave:
```yaml
misc:
  env_chave_criptografia: CHAVE_CRIPT  # Nome da variável de ambiente
```
**Validação fail-fast:** Se esta configuração existir no YAML, o sistema verifica imediatamente (no carregamento da configuração) se a variável de ambiente está definida e não está vazia. Caso contrário, levanta `EnvironmentError` com mensagem orientando a correção, impedindo que o treinamento prossiga com texto criptografado (ilegível).

#### Train on Responses Only
Treina o modelo apenas nas respostas do assistente, ignorando o loss dos prompts do usuário. (sugerido no unsloth)
```yaml
treinamento:
  train_on_responses_only: true
```

#### Curriculum Learning
O modo `curriculum` tem sua própria seção no YAML com a mesma estrutura que `pastas` (predicao, dataset, entrada, validacao). A subchave `divisao` contém a lista de etapas do pipeline, cada uma com seu próprio arquivo CSV de divisão e parâmetros de treino.
```yaml
formatos:
  tipo_entrada: curriculum

curriculum:
  predicao:
    pasta: ./predict/output
  dataset:
    pasta: ./saidas/gold
    mascara: "*.txt"
  entrada:
    dataframe: ./dados/textos.parquet
    dataframe_col: texto
    dataframe_id: id_peca
  validacao:
    exigir_json_valido: true
  batch_size:
    efetivo: 16   # Batch efetivo desejado (ajusta grad_batch_size automaticamente)
    batch_size: 2  # Batch por GPU (fixo, determinado empiricamente para evitar OOM)
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

Esta funcionalidade é suportada em **todos os formatos** (pastas, dataset, curriculum).

```yaml
# Qualquer formato (pastas, dataset, curriculum)
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
- **Trava de Segurança (Early Exit)**: Se o treinamento for reaberto em uma pasta cuja última etapa do Curriculum (ou o total predefinido em um Treino Simples) já foi encerrada, o sistema detecta que o **objetivo final foi atingido** e impede seu avanço com uma mensagem informacional de *Concluído*.
- **Como estender um modelo retido pela "Trava"**:
   - Para modelos em **Curriculum** que já finalizaram, basta ir no `.yaml` e adicionar uma nova etapa para que a trava reconheça o novo requisito e seja liberada.
   - Para um **Treino Simples**, ao apenas ir no `.yaml` e **aumentar o número de épocas finais (`epochs`)**, o sistema identificará que o alvo cresceu em relação ao que constava no `curriculum_state.json` do checkpoint retido, destravando a execução automaticamente e treinando a diferença de steps.
   - ⚠️ *Dica: passar o argumento `--reset` reinicia do Zero ABSOLUTO (apaga logs, histórico e os checkpoints LoRA gerados). Utilize o reset apenas se desejar remover todo o treinamento realizado e começar o treinamento novamente a partir do modelo base.*

---

## Desenvolvimento e Manutenção

### Pendências Concluídas Recentemente ✅
1.  **Refatoração de Ações**: Lógica CLI movida para `treinar_unsloth_actions.py`.
2.  **Separação de Dataset**: Lógica complexa movida para `treinar_unsloth_dataset.py`.
3.  **Segurança**: Limpeza seletiva em `--predict`.
4.  **Flexibilidade**: Adição de flag `--base` e suporte a múltiplos subsets em stats.
5.  **Qualidade**: Correção de logs duplicados e bugs de formatação em relatórios.
6.  **Separação Treino/Avaliação (Passo 1)**: Script `treinar_unsloth_avaliar.py` criado com toda a lógica de avaliação, inferência e exportação. CLI de treino simplificado.
7.  **Separação `dataset` / `predicao` no YAML**: Nova seção `pastas.dataset` para o gold dataset (entrada obrigatória). `pastas.predicao` agora é apenas pasta de saída das predições (criada automaticamente se não existir).
8.  **Menu Interativo YAML**: Ambos os scripts usam `util_menu_opcoes.escolher_yaml()` quando YAML é omitido, com menus de ação específicos para cada script.
9.  **Validação Fail-Fast de Criptografia**: `ConfigMisc.__post_init__` e `_carregar_dataframe_entrada` levantam erro fatal se `misc.env_chave_criptografia` está configurada no YAML mas a variável de ambiente não existe, impedindo treinamento com dados criptografados.
10. **Geração Automática de Gráficos Pós-Treinamento**: Função `gerar_graficos_estatisticos()` extraída e reutilizada por `--stats` e `executar_treinar()`. Gera loss, tokens, hardware e relatório .md automaticamente ao final do treino.
11. **Limpeza de Artefatos Antigos**: `MetricsLoggerCallback` remove gráficos e relatório estatístico anteriores ao iniciar novo treinamento, evitando confusão com dados de treinos passados.
12. **Batch Size Automático**: Seção `treinamento.batch_size` (dict) com `efetivo` e `batch_size`, suportada em todos os formatos (pastas, dataset, curriculum). O sistema calcula `grad_batch_size = round(efetivo / (batch_size × n_gpus))` automaticamente.

### Próximo Passo de Desenvolvimento
> pace de treinamento (Curriculum Learning) e simplificação do código

**Objetivo:** Permitir um fluxo de treinamento em múltiplos estágios (Curriculum Learning) alternando dados, estratégias (LoRA vs Full Fine-Tuning) e critérios de parada dinâmicos (Pace).

Para garantir uma implementação segura e testável, o desenvolvimento será dividido nas seguintes etapas incrementais, permitindo validação e testes intermediários a cada avanço.

#### Passo 1: Separação de Preocupações e Melhoria do CLI ✅ CONCLUÍDO
**Objetivo:** Desacoplar a inferência do motor de treinamento para blindar e otimizar o código base do Treinador, centralizando o treinamento para focar *apenas Treinar e dar Merge*, e melhorar a experiência CLI.

**Implementado:**
1. ✅ **Extração da Avaliação/Inferência:** Funções `executar_info`, `executar_stats`, `executar_predict`, `executar_merge`, `executar_modelo` movidas para `treinar_unsloth_avaliar.py`. Removidas de `treinar_unsloth_actions.py` (que agora contém apenas `executar_treinar`, `executar_reset`, `executar_injetar_dicas` e funções auxiliares compartilhadas).
2. ✅ **Novo Script Independente:** `treinar_unsloth_avaliar.py` criado (~830 linhas) com CLI próprio, modo interativo e funções completas de avaliação.
3. ✅ **Menu Interativo (CLI):** Ambos os scripts usam `util_menu_opcoes.escolher_yaml(chave_obrigatoria='modelo')`. Argumento `config` é `nargs='?'` (opcional). Menu de ações específico: treino (treinar, reset+treinar, reset) e avaliação (info, stats, predict, modelo, merge).
4. ✅ **Separação `dataset`/`predicao`:** Nova `ConfigGold` para `pastas.dataset` (gold standard, obrigatório, validação de existência). `ConfigPredicao` agora é pasta de saída (auto-criada). `parear_arquivos()` em `treinar_unsloth_dataset.py` usa `pastas.dataset` como fonte do gold.
* **⏱️ Teste Intermediário:** `--help` de ambos os scripts funciona. `--info` executa corretamente. Import de todos os módulos validado. Falta: teste completo de treinamento end-to-end e predição em massa.

#### Passo 2: O "Pipeline Universal" e Ajustes Finos (Pré Curriculum) ✅ CONCLUÍDO
**Objetivo:** Unificar a base de código do sistema atual antes de construir o Curriculum Learning multicamadas, assim o processo opera o mesmo sistema de logs (como se fosse de "apenas uma etapa").

1. **Pipeline Universal:** Remover as lógicas apartadas. Se o YAML acionar apenas 1 dataset ou pastas (`tipo_entrada: dataset` ou `pastas`), o inicializador do sistema encapsulará isso convertendo automaticamente em uma lista `curriculum` de tamanho 1, definindo `alias` padrão como "Principal". Toda parte de tracking funcionará agora em cima desta lista universal.
2. **Log de Rastreiamento Unificado e Resumo:** Implementar que todo salvamento utilize métricas gravadas no esquema universal (`curriculum_state.json` constando `{"current_step": 0, "status": "running"}` e `curriculum_metrics.jsonl`), abandonando outros tipos de lógicas divergentes.
4. **Mixando Modelos (LoRA vs Full):**
    * *Transição `[LoRA -> Full]`*: Mesclar base + lora via instanciador e usar o Merge como "o novo `FastLanguageModel` pleno" da segunda fase.
    * *Transição `[Full -> LoRA]`*: A requantização p/ nbits deve ser estritamente reacendida e embutida na modelagem ` FastLanguageModel.get_peft_model()` que sucede a transição.
3. **[Pendente] Simplificação do `max_seq_length` e Remoção do Cache:** A lógica atual de cálculo automático (e _dados_automaticos.json) provou ser uma complicação desnecessária e será removida e substituída por um comportamento estrito (ver Passo 3).
* **⏱️ Teste Intermediário:** Rodar um treinamento normal de teste (`pastas`) exigindo que o código passe perfeitamente sem as checagens e cálculos automáticos de contexto e não trave por arquivos de cache.

#### Passo 3: Motor Multietapas do Curriculum Learning
**Objetivo:** Adicionar interpretador do YAML para Curriculum, transições e regras de `LoRA` \leftrightarrow `Full`.

1. ✅ **Estrutura YAML:** Integrar suporte a configuração `curriculum` no arquivo:
```yaml
formatos:
  tipo_entrada: curriculum # Opções: dataset, pastas, curriculum

pastas:
  # Compartilhado entre todas as etapas do curriculum
  predicao:
    pasta: ./predict/output
  dataset:
    pasta: ./saidas/gold
    mascara: "*.txt"
  entrada:
    dataframe: ./dados/textos.parquet
    dataframe_col: texto
    dataframe_id: id_peca
    prompt_template: './dados/prompt.txt'
    tag_texto: '<<--TEXTO-->>'
  divisao:
    validar_ids: false
    proporcao:
      - treino: 0.70
      - validacao: 0.10
      - teste: 0.20
    seed: 42
  validacao:
    exigir_json_valido: true
    skip_invalidos: false

# Cada etapa = uma divisão do dataset com parâmetros de treino
curriculum:
  - arquivo: "./saidas/divisao_facil.csv"
    alias: "fácil"
    tipo: "full"       # "full" ou "lora" (Se "lora", obedece configurações de `lora` raíz)
    pace_epochs: 1     # (Padrão) Transita após 1 época.
    max_seq_length: 512 # [Opcional] Pode sobrepor config geral.
    learning_rate: 0.0003 # [Opcional] Força LR independente para esta etapa
  - arquivo: "./saidas/divisao_medio.csv"
    alias: "médio"
    tipo: "lora"
    pace_loss: 0.015   # Transita se eval_loss <= 0.015
    pace_epochs: 2     # Limite de segurança. Padrão ao omitir = 1.
```
2. ✅ **Divisão Dinâmica ("Fail Fast"):** Evitar a autogeração baseada em divisões randômicas complexas ao usar curriculum. O sistema deve abortar prevenindo bugs se os subarquivos parametrizados (ex. `{arquivo}_facil.csv`) ikke existirem perfeitamente.
3. ✅ **Roteamento e Sobrevivência de Passos:**
    * Encapsular salvamentos no format de roteamento (ex: `{modelo.saida}/curriculum/01_facil`). Onde um retoma o modelo do passado.
    * No caso de Resume (`--treinar` de checkpoint quebrado), utilizar do state vivo (`curriculum_state.json`) extraído no passo 2 para instanciar subpastas `checkpoint-N` precisas resgatando o ponto cego daquela etapa exata.
5. **Gestão do `max_seq_length` por Estágio (Simplificado):**
    * **Remoção do Cache Complexo**: O cálculo automático e cache (`_dados_automaticos.json`) serão inteiramente removidos para descomplicar a base de código.
    * **Parâmetro Global Obrigatório**: O `max_seq_length` será um parâmetro obrigatório global. O sistema abortará com erro fatal nas validações se estiver zerado ou ausente.
    * **Recarga Dinâmica**: O valor pode ser sobreposto em cada etapa do `curriculum`. Caso haja variação do `max_seq_length` durante a transição de um estágio para o outro, o pipeline se encarregará de recarregar a modelagem/tokenizer com a nova configuração, permitindo dar sequência com a nova limitação estritamente alocada.

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