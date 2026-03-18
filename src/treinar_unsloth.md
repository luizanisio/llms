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
- [x] Implementar exportação de modelo (GGUF, Merge).
- [x] Gerar gráficos de evolução de Loss (treino vs validação) ao final do treino.

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
| `--merge`   | Merge LoRA + Base. Exporta em 16-bit (padrão) ou outro formato via `--quant`. |
| `--reset`   | Limpa o treinamento atual (checkpoints e adaptador) para reiniciar do zero |
### Opções Modificadoras

| Opção | Descrição |
|-------|-----------|
| `--base` | **Força o uso do modelo base**. <br>- Em `--predict`: salva em `predict_base/` ignorando LoRA.<br>- Em `--modelo`: testa o modelo base ignorando LoRA treinado. |
| `--predict-treino` | Gera predições apenas para o subset de treino |
| `--predict-validacao`| Gera predições apenas para o subset de validação |
| `--predict-teste` | Gera predições apenas para o subset de teste |
| `--modelo N` | Testa inferência interativa com N exemplos (default: 1). Exibe métricas de memória. |
| `--quant METODO` | Define método de quantização p/ merge (`16bit`, `4bit`, `q4_k_m`, `q8_0`). |
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

#### Train on Responses Only
Treina o modelo apenas nas respostas do assistente, ignorando o loss dos prompts do usuário. (sugerido no unsloth)
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

### Próximo Passo de Desenvolvimento
> pace de treinamento (Curriculum Learning) e simplificação do código

**Objetivo:** Permitir um fluxo de treinamento em múltiplos estágios (Curriculum Learning) alternando dados, estratégias (LoRA vs Full Fine-Tuning) e critérios de parada dinâmicos (Pace).

Para garantir uma implementação segura e testável, o desenvolvimento será dividido nas seguintes etapas incrementais, permitindo validação e testes intermediários a cada avanço.

#### Passo 1: Separação de Preocupações e Melhoria do CLI (Refatoração de Avaliação)
**Objetivo:** Desacoplar a inferência do motor de treinamento para blindar e otimizar o código base do Treinador, centralizando o treinamento para focar *apenas Treinar e dar Merge*, e melhorar a experiência CLI.

1. **Extração da Avaliação/Inferência:** Mover em definitivo as funções de `--predict` (Predição em massa), inferência simulada (`--modelo N`) e exportação customizada de modelo de dentro do `treinar_unsloth.py` e `treinar_unsloth_actions.py`.
2. **Novo Script Independente:** Mover toda essa lógica para um novo script `treinar_unsloth_avaliar.py` focando apenas na avaliação e uso do modelo.
3. **Menu Interativo (CLI):** Modificar o CLI em ambos os pontos para listar os arquivos `*.yaml` disponíveis interativamente quando omitido como parâmetro. O menu deve permitir:
    * *No Treino:* Executar treinamento, Info (imprimir parâmetros/módulos gerados) ou gerar YAML padrão.
    * *Na Avaliação:* Avaliar modelo, Exportar ou Gerenciar Predições em Massa.
* **⏱️ Teste Intermediário:** Ao fim do primeiro passo, executar o treinamento padrão (usando pasta/dataset pré existente e validando que nada falhou pela refatoração), e usar o recém-nascido `treinar_unsloth_avaliar.py` para injetar comandos numa predição em massa (para garantir sua conectividade aos parâmetros sem erros).

#### Passo 2: O "Pipeline Universal" e Ajustes Finos (Pré Curriculum)
**Objetivo:** Unificar a base de código do sistema atual antes de construir o Curriculum Learning multicamadas, assim o processo opera o mesmo sistema de logs (como se fosse de "apenas uma etapa").

1. **Pipeline Universal:** Remover as lógicas apartadas. Se o YAML acionar apenas 1 dataset ou pastas (`tipo_entrada: dataset` ou `pastas`), o inicializador do sistema encapsulará isso convertendo automaticamente em uma lista `curriculum` de tamanho 1, definindo `alias` padrão como "Principal". Toda parte de tracking funcionará agora em cima desta lista universal.
2. **Log de Rastreiamento Unificado e Resumo:** Implementar que todo salvamento utilize métricas gravadas no esquema universal (`curriculum_state.json` constando `{"current_step": 0, "status": "running"}` e `curriculum_metrics.jsonl`), abandonando outros tipos de lógicas divergentes.
3. **Ajuste Dinâmico de `max_seq_length`:** Antes de dar início ao processo, codificar a análise da extensão máxima de tokens. Caso o dado ultrapasse o valor configurado/suportado, levantar Exception. Se `max_seq_length` for 0 ou omitido, deve arredondar com margem o teto máximo (ex: múltiplos precisos de 512 de folga para alocar tudo e enxugar VRAM).
* **⏱️ Teste Intermediário:** Rodar um treinamento normal de teste (`pastas`) a partir do zero. As métricas exportadas, `.JSONL`, consolidações do estado de curriculum e a folga calculada do `max_seq_length` devem gerar relatórios perfeitamente adequados sem ter exigido yaml especializado, comprovando a blindagem base.

#### Passo 3: Motor Multietapas do Curriculum Learning
**Objetivo:** Adicionar interpretador do YAML para Curriculum, transições e regras de `LoRA` \leftrightarrow `Full`.

1. **Estrutura YAML:** Integrar suporte a configuração `curriculum` no arquivo:
```yaml
formatos:
  tipo_entrada: curriculum # Opções: dataset, pastas, curriculum

curriculum:
  - arquivo: "./saidas/ext_qwen_11_facil.csv"
    alias: "fácil"
    tipo: "full"       # "full" ou "lora" (Se "lora", obedece configurações de `lora` raíz)
    pace_epochs: 1     # (Padrão) Transita após 1 época.
    max_seq_length: 512 # [Opcional] Pode sobrepor config geral.
    learning_rate: 0.0003 # [Opcional] Força LR independente para esta etapa
  - arquivo: "./saidas/ext_qwen_11_medio.csv"
    alias: "médio"
    tipo: "lora"
    pace_loss: 0.015   # Transita se eval_loss <= 0.015
    pace_epochs: 2     # Limite de segurança. Padrão ao omitir = 1.
```
2. **Divisão Dinâmica ("Fail Fast"):** Evitar a autogeração baseada em divisões randômicas complexas ao usar curriculum. O sistema deve abortar prevenindo bugs se os subarquivos parametrizados (ex. `{arquivo}_facil.csv`) ikke existirem perfeitamente.
3. **Roteamento e Sobrevivência de Passos:**
    * Encapsular salvamentos no format de roteamento (ex: `{modelo.saida}/curriculum/01_facil`). Onde um retoma o modelo do passado.
    * No caso de Resume (`--treinar` de checkpoint quebrado), utilizar do state vivo (`curriculum_state.json`) extraído no passo 2 para instanciar subpastas `checkpoint-N` precisas resgatando o ponto cego daquela etapa exata.
4. **Mixando Modelos (LoRA vs Full):**
    * *Transição `[LoRA -> Full]`*: Mesclar base + lora via instanciador e usar o Merge como "o novo `FastLanguageModel` pleno" da segunda fase.
    * *Transição `[Full -> LoRA]`*: A requantização p/ nbits deve ser estritamente reacendida e embutida na modelagem ` FastLanguageModel.get_peft_model()` que sucede a transição.
* **⏱️ Teste Intermediário:** Simular um YAML com duas etapas restritas em LoRA. Processar passo 1 e pausar; retomar usando reexecução do script pelo CLI, e observar se os modelos são salvos nos seus compartimentos próprios no HD, e o passo 2 continua usando as raízes geradas. 

#### Passo 4: Pace Dinâmico e Ajustes Visuais de Controle
**Objetivo:** Interpolação analítica final garantindo eficiência via parada prematura baseada no desempenho e legibilidade de análise das métricas via Gráficos evolutivos de múltiplas fases.

1. **Pacing / Early Stopping Configurable:**
    * Programar `TrainerCallback` customizado plugado no Unsloth, que intercepta pós gatilhos `on_evaluate()`.
    * Checar `eval_loss <= pace_loss` ou total epochs se aproximarem de `pace_epochs`, injetar `control.should_training_stop = True` finalizando imediatamente aquela frente do currículo para poupar horas de servidor.
2. **Métricas de Eficiência Analíticas Estendidas:** Em cada finalização de passo (`curriculum_metrics.jsonl`), adicionar logs em cada etapa englobando *Tamanho em instâncias geradas (Utilização e Validação do target)*, *eval_loss absoluto que atestou o fim da etapa*, *Tempo Real Clocked Time*, *VRAM Peak* na fase cruzada.
3. **Legendagem Visual do Gráfico (Loss):** Redesenhar e adaptar o método `GraficoTreinamento.evolucao_loss()` usando a engine base. Ele deve ler o tracker consolidado lido, identificar onde ocorreu a transição dos aliases e traçar linhas demarcatórias cruzando e assinalando como legenda "Etapa Fácil", "Etapa Médio", nos boxes e boxplots (em `--stats`).
* **⏱️ Teste Intermediário Final:** Processar múltiplos estágios usando Curriculum completo. Embutir propositalmente uma meta `pace_loss = 1.5` de fácil alcance numa das passagens e testar os limites do Early-Stopping e o respectivo avanço para a etapa 2. Constatar a divisão formatada do gráfico unificado renderizado em `.png` ao encerramento pleno.