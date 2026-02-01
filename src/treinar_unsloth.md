# Documentação do Treinamento com treinar_unsloth.py

## Descrição Geral

O pacote `treinar_unsloth.py` é uma ferramenta completa para fine-tuning de modelos LLM (Gemma-3, Deepseek, Llama, Qwen) usando Unsloth + TRL-SFTTrainer com configuração via arquivo YAML.

---

## Arquivos do Projeto

| Arquivo | Descrição |
|---------|-----------|
| `treinar_unsloth.py` | Script principal de treinamento |
| `treinar_unsloth_util.py` | Utilitários de configuração (`YamlTreinamento`) e CLI interativa |
| `treinar_unsloth_dataset.py` | Gerenciamento de datasets: carga, validação e preparação (`DatasetTreinamento`) |
| `treinar_unsloth_logging.py` | Sistema centralizado de logging com níveis configuráveis |
| `treinar_unsloth_monitor.py` | Monitoramento contínuo de RAM/GPU com geração de gráficos |
| `treinar_unsloth_report.py` | Geração de relatórios em Markdown |
| `treinar_unsloth.md` | Esta documentação |
| `util.py` | Utilitários gerais (Util.mensagem_to_json, etc.) |

---

## Características Principais

### Implementadas ✅

- [x] Carregamento de modelo base com suporte a quantização (4-bit, 8-bit)
- [x] Suporte a LoRA (Low-Rank Adaptation) configurável
- [x] Detecção e continuação de modelos LoRA já treinados
- [x] Gerenciamento automático de checkpoints
- [x] Suporte a múltiplas GPUs (via CUDA_VISIBLE_DEVICES)
- [x] Modo debug detalhado para inspeção de datasets e configurações (`--debug`)
- [x] Modo teste para validar predições do modelo (`--modelo N`)
- [x] Log detalhado de processamento com métricas de memória GPU
- [x] Suporte a datasets em formato parquet, json, jsonl e txt
- [x] Detecção automática de formato do dataset (messages ou prompt/completion)
- [x] Chat template automático baseado no modelo (gemma, qwen, llama)
- [x] Criptografia de dados sensíveis em dataframes (via chave Fernet)
- [x] Validação interativa de configurações ausentes ou incorretas

---

## Nova Estrutura de Configuração YAML

O sistema utiliza uma estrutura hierárquica dividida em seções lógicas (`formatos`, `misc`, `pastas`/`dataset`, `modelo`, `treinamento`, `lora`).

### Seção misc (Configurações Gerais)

A seção `misc` contém configurações diversas do projeto:

```yaml
misc:
  log_level: INFO          # Nível de log: DEBUG, INFO, WARNING, ERROR
  env_chave_criptografia: CHAVE_CRIPT  # Variável de ambiente com chave Fernet
```

**Parâmetros:**
- `log_level`: Define o nível de log padrão. Pode ser sobrescrito pelo parâmetro `--log-level` da CLI
- `env_chave_criptografia`: Nome da variável de ambiente que contém a chave de criptografia Fernet para decriptografar dados

### Modos de Entrada

1.  **Modo "pastas"**:
    *   Carrega dados de arquivos de texto/JSON organizados em diretórios.
    *   Parea automaticamente arquivos de entrada (prompts) com arquivos de predição (respostas esperadas) pelo nome base.
    *   Permite definir um template de prompt.
    *   Permite leitura de input a partir de colunas de um DataFrame Parquet (com flag `dataframe: true` e `dataframe_col`).

2.  **Modo "dataset"**:
    *   Carrega dados de arquivos Parquet prontos com colunas formatadas (como `messages`).

### Configuração de Divisão (proporcao)

A chave `proporcao` dentro de `pastas.divisao` define a proporção de dados para cada subset. **O formato recomendado é usar chaves nomeadas**:

```yaml
proporcao:
  - treino: 0.7      # Usado para aprendizado dos pesos (backpropagation)
  - validacao: 0.1   # Monitorar métricas durante treino e early stopping
  - teste: 0.2       # Avaliação final imparcial APÓS o treinamento
```

**Regras:**
- A soma dos valores deve ser igual a 1.0
- Valores devem ser entre 0 e 1
- Os nomes aceitos são: `treino`/`train`, `validacao`/`validação`/`validation`, `teste`/`test`

### Configuração de Treinamento (train_on_responses_only)

A opção `train_on_responses_only` permite treinar o modelo apenas nas respostas do assistant, mascarando as instruções do usuário. Isso é recomendado para melhorar a qualidade do fine-tuning:

```yaml
treinamento:
  train_on_responses_only: true  # Recomendado (padrão: true)
```

**Como funciona:**
- Detecta automaticamente as tags de instrução e resposta baseado no modelo (Gemma, Qwen, Llama, DeepSeek)
- Mascara tudo exceto as respostas do assistant durante o cálculo do loss
- Resulta em modelos mais focados em gerar respostas de alta qualidade

### Arquivos de Métricas Gerados

Durante o treinamento, são gerados três arquivos de métricas na pasta do modelo:

| Arquivo | Localização | Conteúdo |
|---------|-------------|----------|
| `metrics_stream.jsonl` | `{output_dir}/` | Métricas brutas do trainer (loss, lr, etc) |
| `hardware_metrics.jsonl` | `{output_dir}/treinamento/` | Uso de CPU, RAM, Disco e GPU a cada 10 steps |
| `training_metrics.jsonl` | `{output_dir}/treinamento/` | Métricas detalhadas com médias móveis e progresso |

**Exemplo de hardware_metrics.jsonl:**
```json
{"timestamp": 1706789999.0, "step": 10, "epoch": 0.1, "fase": "train", 
 "cpu_uso_%": 45.2, "ram_usada_gb": 12.5, "gpu0_alocada_gb": 8.2}
```

**Exemplo de training_metrics.jsonl:**
```json
{"event": "log", "step": 10, "train_loss": 2.345, "learning_rate": 0.0001, 
 "train_loss_avg_10": 2.456, "progress_%": 5.0}
```

---

## Uso via Linha de Comando

### treinar_unsloth.py (Treinamento)

```bash
# Treinar modelo
python treinar_unsloth.py CONFIG.yaml

# Modo debug (super recomendado antes do treino)
# Mostra resumo da configuração, status dos arquivos e exibe exemplos reais do dataset (começo/fim)
python treinar_unsloth.py CONFIG.yaml --debug

# Testar predições/inferência com o modelo (treinado ou base) em N exemplos
# Gera gráfico de uso de memória (RAM e GPU) durante as predições
python treinar_unsloth.py CONFIG.yaml --modelo 5

# Definir nível de log (DEBUG mostra mais detalhes)
python treinar_unsloth.py CONFIG.yaml --log-level DEBUG

# Definir GPUs específicas
export CUDA_VISIBLE_DEVICES=0,1 python treinar_unsloth.py CONFIG.yaml
```

### treinar_unsloth_util.py (Validação e Helpers)

TODO: revisar se está funcionando
```bash
# Validar YAML e exibir configurações
python treinar_unsloth_util.py CONFIG.yaml

# Modo interativo: corrige problemas de configuração perguntando ao usuário
python treinar_unsloth_util.py CONFIG.yaml --interativo

# Listar arquivos que serão usados no treino (modo pastas)
python treinar_unsloth_util.py CONFIG.yaml --listar-arquivos

```

---

## Arquitetura e Classes
TODO: complementar classes e métodos mais importantes relacionados ao treinamento

### `DatasetTreinamento` (src/treinar_unsloth_dataset.py)
Responsável por toda a manipulação de dados.
*   **Carregamento**: Lê arquivos de texto, JSON ou DataFrames Parquet.
*   **Decriptografia**: Lida com dados cifrados se configurado.
*   **Preparação**: Parea arquivos de entrada/saída, aplica templates de prompt.
*   **Divisão**: Gerencia splits de treino/teste/validação (cria/lê CSV).
*   **Validação de Consistência**: Verifica se IDs do CSV de divisão e arquivos pareados estão sincronizados (método `_validar_consistencia_divisao`).
*   **Visualização**: Gera previews detalhados dos dados para debug.

### `YamlTreinamento` (src/treinar_unsloth_util.py)
Gerencia a configuração.
*   **Validação**: Garante que o YAML segue a estrutura correta.
*   **Delegação**: Encapsula uma instância de `DatasetTreinamento` e expõe métodos de conveniência.
*   **Compatibilidade**: Provê acesso estruturado (`.modelo`, `.treinamento`) e mapeamentos legados se necessário.

### `LLMsTrainer` (src/treinar_unsloth.py)
Orquestra o treinamento.
*   Setup do modelo Unsloth e Tokenizer.
*   Integração com TRL (`SFTTrainer`).
*   Configuração de callbacks e logging.
*   Execução de inferência para testes.

---

### Concluídas Recentemente ✅
- [x] Padronização de nomes de subsets: `treino`, `validacao`, `teste` (em vez de avaliação)
- [x] Migração automática de arquivos de divisão antigos
- [x] Chave `proporcao` suporta dicionário nomeado e lista de dicts (ex: `- treino: 0.8`)
- [x] Padronização de informações de treino na pasta `treinamento`
- [x] Geração automática de relatório `.md` com estatísticas e config
- [x] **Opção `train_on_responses_only`** do unsloth (treina apenas nas respostas do assistant)
- [x] **Registro contínuo de métricas de hardware** (RAM, GPU, CPU, disco em `hardware_metrics.jsonl`)
- [x] **Registro contínuo de métricas de treinamento/validação** (loss, lr em `training_metrics.jsonl`)
- [x] **Sistema de logging padronizado** com biblioteca `logging` e níveis configuráveis (DEBUG, INFO, WARNING, ERROR)
- [x] **Monitoramento de memória em modo --modelo** com gráfico de linha RAM/GPU (`memoria_predicao.png`)
- [x] **Verificação de modelo treinado** em modo --modelo com opção de usar modelo base

## Pendências de Integração (Próximos Passos)

### Funcionalidades de Configuração
- [x] ~~incluir no método Util.dados_hardware(...) informações sobre o uso de memória RAM e GPU~~ (Concluído: método agora retorna RAM usada/disponível e informações detalhadas de GPU via torch.cuda)

### Validações Adicionais
- [x] ~~Validar se todos os IDs do CSV de divisão existem nos arquivos pareados (retornar erro se falhar)~~ (Concluído: método `_validar_consistencia_divisao` em `DatasetTreinamento`)
- [x] ~~Validar se todos os arquivos pareados estão no arquivo de divisão (retornar erro se falhar)~~ (Concluído: validação bidirecional com mensagens de erro detalhadas) 

---

## Pendências de Novas Funcionalidades

### Dataset e Pré-processamento
- [ ] Suporte a HuggingFace datasets (carregar direto do Hub)
- [ ] Filtro de registros por comprimento máximo de tokens
- [ ] Estatísticas mais detalhadas de tokens (histograma, etc)

### Treinamento
- [x] ~~Opção de usar `train_on_responses_only` do unsloth~~ (Concluído: detecta automaticamente as tags baseado no modelo)
- [x] ~~Registro contínuo de métricas de uso de memória RAM e GPU, processador e disco~~ (Concluído: `hardware_metrics.jsonl`)
- [x] ~~Registro contínuo de métricas de treinamento e validação~~ (Concluído: `training_metrics.jsonl`)
- [ ] Gerar gráficos de métricas de treinamento e validação ao final, na pasta treinamento com prefixo `grafico_` ... (apagando gráficos antes de nova geração)
- [ ] Suporte a early stopping com patience configurável (exmplicar e perguntar antes de implementar)
- [ ] Suporte a gradient clipping configurável (exmplicar e perguntar antes de implementar)
- [ ] Suporte a mixed precision training explícito (fp16/bf16) (exmplicar e perguntar antes de implementar)
- [ ] Integração com Weights & Biases (wandb) opcional (exmplicar e perguntar antes de implementar)

### Monitoramento e Logs
- [x] ~~Registrar uso de memória RAM antes/depois do treinamento~~ (Concluído: incluído no relatório e hardware_metrics.jsonl)
- [x] ~~Registrar uso de memória (RAM e GPU) durante teste (modo `--modelo`)~~ (Concluído: `treinar_unsloth_monitor.py` com gráfico em `memoria_predicao.png`)
- [x] ~~Caso seja usado o parâmetro --modelo e não existir modelo treinado ainda, perguntar se deseja realizar a predição no modelo base~~ (Concluído: verificação automática com prompt interativo)

### Checkpoints
- [ ] Opção para limpar checkpoints antigos ao finalizar com sucesso

### Inferência/Predição
- [ ] Suporte a batch inference (múltiplos prompts em paralelo)
- [ ] Exportar resultados de predição para arquivo JSON

### Exportação de Modelo
- [ ] Exportar modelo para GGUF (llama.cpp)
- [ ] Exportar modelo merged (LoRA + base)
- [ ] Exportar para ONNX

---

## Pendências de Refatoração

### Concluídas ✅
1. Nova classe `YamlTreinamento` em arquivo separado
2. Dataclasses para estruturação tipada de configurações
3. Parâmetro `seed` configurável no YAML
4. Validador interativo para correção de configurações
5. Separação de `DatasetTreinamento` em módulo próprio

### Pendentes
1. ~~**Padronizar logging**~~ ✅ (Concluído em `treinar_unsloth_logging.py`)
   - Usar biblioteca `logging` ao invés de `print`
   - Níveis de log configuráveis (DEBUG, INFO, WARNING, ERROR)
   - Parâmetro CLI `--log-level` adicionado

2. **Testes unitários**
   - Criar testes para YamlTreinamento
   - Criar testes para ValidadorInterativo
   - Criar testes para DatasetTreinamento

---

## Arquivos de Saída (modo --modelo)

Quando usado o modo de teste `--modelo`, são gerados:

| Arquivo | Localização | Descrição |
|---------|-------------|----------|
| `memoria_predicao.jsonl` | `{output_dir}/treinamento/` | Métricas de RAM e GPU coletadas durante predições |
| `memoria_predicao.png` | `{output_dir}/treinamento/` | Gráfico de linha mostrando uso de memória ao longo do tempo |

**Exemplo de gráfico gerado:**
- Eixo X: Tempo em segundos
- Eixo Y: Memória em GB
- Linha azul: RAM usada
- Linha vermelha: GPU usada (soma de todas as GPUs)