# Documentação do Treinamento com treinar_unsloth.py

## Descrição Geral

O pacote `treinar_unsloth.py` é uma ferramenta completa para fine-tuning de modelos LLM (Gemma-3, Deepseek, Llama, Qwen) usando Unsloth + TRL-SFTTrainer com configuração via arquivo YAML.

---

## Arquivos do Projeto

| Arquivo | Descrição |
|---------|-----------|
| `treinar_unsloth.py` | Script principal de treinamento e CLI (`LLMsTrainer`) |
| `treinar_unsloth_actions.py` | Implementação das ações da CLI (`executar_treinar`, `executar_stats`, etc.) |
| `treinar_unsloth_util.py` | Utilitários de configuração (`YamlTreinamento`) e helpers |
| `treinar_unsloth_dataset.py` | Gerenciamento de datasets: carga, divisão, validação (`DatasetTreinamento`) |
| `treinar_unsloth_logging.py` | Sistema centralizado de logging com níveis configuráveis |
| `treinar_unsloth_monitor.py` | Monitoramento contínuo de RAM/GPU |
| `treinar_unsloth_report.py` | Geração de relatórios em Markdown |
| `treinar_unsloth.md` | Esta documentação |

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
- [x] Suporte a datasets em formado de pastas (pares .txt) e arquivos únicos (csv/parquet)
- [x] Divisão automática de datasets (treino, validação, teste) via YAML ou CSV
- [x] Chat template automático baseado no modelo
- [x] Criptografia de dados sensíveis (Fernet)
- [x] Validação automática de consistência de dados (IDs e arquivos pareados)

---

## Uso via Linha de Comando

```bash
# Formato geral
python treinar_unsloth.py CONFIG.yaml [AÇÃO] [OPÇÕES]
```

### Ações Principais

| Ação | Descrição |
|------|-----------|
| **(nenhuma)** | Modo interativo: exibe menu para escolher ação |
| `--info` | Exibe informações detalhadas sobre configuração e dataset (substitui `--debug`) |
| `--stats` | Gera relatório estatístico de tokens com gráficos boxplot por subset |
| `--treinar` | Inicia ou continua o treinamento |
| `--predict` | Gera predições para todos os subsets (treino, validação, teste) |
| `--reset` | Limpa o treinamento atual (checkpoints e modelo LoRA) -- *Requer confirmação* |

### Opções Modificadoras

| Opção | Descrição |
|-------|-----------|
| `--base` | **Força o uso do modelo base**. <br>- Em `--predict`: salva em `predict_base/` ignorando LoRA.<br>- Em `--modelo`: testa o modelo base ignorando LoRA treinado. |
| `--predict-treino` | Gera predições apenas para o subset de treino |
| `--predict-validacao`| Gera predições apenas para o subset de validação |
| `--predict-teste` | Gera predições apenas para o subset de teste |
| `--modelo N` | Testa inferência interativa com N exemplos (default: 1). Exibe métricas de memória. |
| `--log-level LEVEL` | Define nível de log (DEBUG, INFO, WARNING, ERROR) |

### Exemplos de Uso

```bash
# Ver informações do setup
python treinar_unsloth.py config.yaml --info

# Gerar estatísticas do dataset (tabelas e gráficos)
python treinar_unsloth.py config.yaml --stats

# Treinar (continuando de checkpoint se existir)
python treinar_unsloth.py config.yaml --treinar

# Limpar tudo e treinar do zero
python treinar_unsloth.py config.yaml --treinar --reset

# Gerar predições com modelo treinado (LoRA)
# Saída: {output_dir}/predict/{subset}/{id}.txt e {id}.json
python treinar_unsloth.py config.yaml --predict

# Gerar predições com modelo BASE (ignorando treino)
# Saída: {output_dir}/predict_base/...
python treinar_unsloth.py config.yaml --predict --base

# Testar inferência com 3 exemplos usando modelo treinado
python treinar_unsloth.py config.yaml --modelo 3

# Testar inferência com modelo BASE
python treinar_unsloth.py config.yaml --modelo 1 --base
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

#### Proporção e Divisão
Define como os dados são divididos se não houver arquivo CSV de divisão prévio.
```yaml
pastas:
  divisao:
    arquivo: "caminho/para/divisao.csv" # Opcional: fixa a divisão
    proporcao:
      - treino: 0.7
      - validacao: 0.1
      - teste: 0.2
```

#### Train on Responses Only
Treina o modelo apenas nas respostas do assistente, ignorando o loss dos prompts do usuário.
```yaml
treinamento:
  train_on_responses_only: true
```

---

## Arquivos de Saída e Métricas

Durante o treinamento e testes, diversos arquivos são gerados na pasta de saída (`modelo.saida`):

| Pasta/Arquivo | Conteúdo |
|---------------|----------|
| `adapter_model.safetensors` | Pesos do LoRA treinado |
| `treinamento/training_metrics.jsonl` | Métricas de treino (loss, learning rate, epoch) a cada step |
| `treinamento/hardware_metrics.jsonl` | Uso de recursos (CPU, RAM, GPU, Disco) coletado periodicamente |
| `treinamento/memoria_predicao.png` | Gráfico de uso de memória gerado durante teste (`--modelo`) |
| `predict/` | Resultados da predição com modelo treinado |
| `predict_base/` | Resultados da predição com modelo base (`--base`) |

---

## Monitoramento

O sistema inclui monitoramento de recursos em background (`treinar_unsloth_monitor.py`):
1.  **Durante Treino**: Registra em `hardware_metrics.jsonl`.
2.  **Durante Teste (`--modelo`)**: Coleta dados em tempo real e gera gráfico ao final.
3.  **Logs**: Exibe consumo de VRAM e RAM nos logs de execução.

---

## Desenvolvimento e Manutenção

### Pendências Concluídas Recentemente ✅
1.  **Refatoração de Ações**: Lógica CLI movida para `treinar_unsloth_actions.py`.
2.  **Separação de Dataset**: Lógica complexa movida para `treinar_unsloth_dataset.py`.
3.  **Segurança**: Limpeza seletiva em `--predict`.
4.  **Flexibilidade**: Adição de flag `--base` e suporte a múltiplos subsets em stats.
5.  **Qualidade**: Correção de logs duplicados e bugs de formatação em relatórios.

### Próximos Passos (Backlog)
- [ ] Implementar exportação de modelo (GGUF, Merge).
- [ ] Adicionar suporte a Early Stopping configurável.
- [ ] Gerar gráficos de evolução de Loss (treino vs validação) ao final do treino.