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

**Proposta de Estrutura YAML:**
A melhor abordagem de engenharia de software seria instituir `curriculum` como um terceiro e oficial `tipo_entrada` (junto de `dataset` e `pastas`). Se a pessoa desejar as etapas dinâmicas, basta declarar esse modo e listar as etapas de forma explícita, mantendo retrocompatibilidade, validação rígida via `dataclasses` (ou Pydantic) e altíssima legibilidade.

```yaml
formatos:
  tipo_entrada: curriculum # Opções: dataset, pastas, curriculum

curriculum:
  - arquivo: "./saidas/ext_qwen_11_facil.csv"
    alias: "fácil"
    tipo: "full"       # "full" ou "lora" (Se "lora", obedece os sub-parâmetros baseados em `lora` raíz)
    pace_epochs: 1     # (Padrão) Transita após 1 época. Sem `pace_loss`, o loss não causa parada precoce
    max_seq_length: 512 # [Opcional] Pode sobrepor config geral. Sequências curtas gastam menos VRAM na 1ª fase! se omitido ou 0 calcula automaticamente pelos dados
    learning_rate: 0.0003 # [Opcional] Pode forçar um LR independente para esta etapa do curriculum
  - arquivo: "./saidas/ext_qwen_11_medio.csv"
    alias: "médio"
    tipo: "lora"
    pace_loss: 0.015   # Transita se eval_loss <= 0.015
    pace_epochs: 2     # Limite de segurança. Se pace_loss e pace_epochs forem omitidos, pace_epochs=1 implicitamente.
```

**Mecânica de Transição, Modelos e Checkpoints:**
1. **Roteamento de Modelos Intermediários:**
   - Onde salvamos? Sempre que um passo encerra, seu modelo consolidado deve ser gravado numa subpasta controlada: ex. `{modelo.saida}/curriculum/01_facil`.
   - *Dica Arquitetural:* Ao iniciar a etapa de `médio` logo na sequência, o instanciador do Unsloth reconfigura seu `modelo.base` subitamente apontando para o diretório físico `{modelo.saida}/curriculum/01_facil` como nova base e assim por diante.
2. **Processamento LoRA vs Full Fine-Tuning:**
   - No Unsloth, treinar `full` equivale a **NÃO** fazer wrap do modelo primário via `FastLanguageModel.get_peft_model()`. Ele treinará os pesos absolutos.
   - **Transição `[LoRA -> Full]`:** Se um passo `LoRA` preceder um `Full` (ex: passo 1: LoRA, passo 2: Full), ao encerrar o passo `LoRA`, o adaptador DEVE ser "merged" (consolidado nativamente) em `{modelo.saida}/curriculum/01_medio` para gerar uma base fundida e limpa, que então será adotada pelo processo nativo (Full) a seguir.
   - **Transição `[Full -> LoRA]` (Cuidado Especial):** O fluxo contrário é mais natural, porém exige um cuidado: o output do treinamento `Full` será um modelo massivo com pesos absolutos modificados (geralmente salvo em FP16 ou BF16 nativamente pelo `save_pretrained`). Ao resgatá-lo como "modelo base" no passo `LoRA` seguinte, verifique se a quantização (`nbits`, ex: 4-bits) está efetivamente ligada na inicialização do Unsloth associada a ele. O Unsloth cuidará de re-quantizar dinamicamente para inflar os adaptadores PEFT.
3. **Pace (Critério de Continuação e Valores Padrão):**
   - Transita caso o alvo de épocas do passo OU o *target loss* seja batido (early pacing).
   - **Padrões (Defaults):** Caso `pace_loss` e `pace_epochs` não sejam definidos explicitamente em uma etapa, o sistema deve assumir `pace_epochs: 1` intrinsecamente, e ignorar o monitoramento antecipado por sinal de loss.
   - *Dica Arquitetural:* Construa um callback herdando da classe padrão `TrainerCallback` que execute a avaliação de Pace em todo gatilho `on_evaluate()`. Lá, resgate `metrics.get("eval_loss")` e se `<= pace_loss`, aplique forçadamente `control.should_training_stop = True`.
4. **Sobrevivendo às Paradas e Checkpoints:**
   - Ao longo dos dias, para dar "Resume", o pipeline deve gravar um arquivo vivo em `curriculum_state.json` constando `{"current_step": 0, "status": "running"}`.
   - Reiniciar (`--treinar`) lê esse log e aciona apenas o `step` respectivo resgatando os últimos `.bin` e subpasta do seu correspondente em `chkpt/`. Cada curriculum poderá ter uma segregação de seus checkpoints (`chkpt/01_facil/checkpoint-N`).
5. **Divisão Dinâmica ("Fail Fast"):**
   - Se o YAML cita `arquivo1_facil.csv` que inexiste, a divisão "automática" descrita no ticket pode não saber gerar subgrupos perfeitamente dosados para facilitar um curriculum, visto que necessita categorização externa de "dificuldade". O ideal é assumir um modelo "*Fail-Fast*", onde se usar "curriculum", o arquivo `.csv` referenciado é OBRIGATÓRIO (ou emitir Exception e abortar se a estrutura de amostragem não for auto-evidente).
6. **Gráficos e Alias:**
   - Ao trocar a etapa, a classe `treinar_unsloth_logging.py` grava meta-registros novos no log do loss.
   - Na renderização `GraficoTreinamento.evolucao_loss()`, analise este log ou rastreando as divisas de "passo atual", inserindo linhas demarcatórias e legendas baseadas no atributo genérico `alias` (ou `ARQ01` se não houver um).
7. **Métricas e Logs de Eficiência (Comparativo de Pipelines):**
   - Para analisar de modo preciso se um Pipeline de Curriculum foi mais eficiente que outro, um log de rastreio independente deve ser gerado ou estendido (`curriculum_metrics.jsonl`). A cada quebra de passo, o log de treinamento deve selar **categoricamente**:
     - *Utilização Real de Dados:* Quantidade absoluta de instâncias ingeridas, divididas explicitamente nas amostras alocadas (Treino / Validação / Teste).
     - *Custo-Benefício de Iteração:* O `eval_loss` exato da barreira batida.
     - *Uso de Tempo:* Duração daquela fase (`clock_time`) e estimativas de `tokens/segundo`.
     - *Consumo e Steps:* O teto máximo de VRAM atingido durante a fase, os tamanhos do dataset, e a quantidade de `global_steps` efetivamente consumidos para furar o `target_loss`.

### Simplificação e Manutenibilidade (Visão Unificada)

Para garantir que o código seja limpo, modular e de altíssima manutenibilidade (tanto para humanos testando quanto para LLMs escrevendo), devemos purgar complexidades desnecessárias e unificar o processamento na raiz do sistema antes de codificar o Curriculum:

1. **O "Pipeline Universal" (Tratar Pastas e Datasets como Curriculuns de 1 Passo):**
   - Não escreveremos lógicas de log/checkpoint apartadas para os 3 modos. Se a entrada no YAML for somente do tipo `dataset` ou `pastas`, o instanciador interno converterá isso em uma lista `curriculum` de tamanho 1, definindo o `alias` padrão como "Principal". 
   - Com isso, a engine central de registros (Logs, `.jsonl` de performance, marcações de gráficos de loss, resume via `curriculum_state.json`) usa **uma única e idêntica base de código estrita**, independentemente de quão complexo o pipeline seja.
2. **Padrão Rígido de Métricas e Gráficos:**
   - Todos os treinamentos vão despachar as métricas de VRAM, Loss, Tokens processados e Tempo de Execução para a raiz do formato. As análises e o PDF de gráficos (`GraficoTreinamento.evolucao_loss`, estatísticas) farão parser e visualização de um mesmo layout, quebrando por `alias`.
3. **Remover a Engenharia de Inferência do Módulo de Treino (Separação de Preocupações):**
   - A complexidade central hoje é que o *Script de Treinamento* também serve de *Script de Inferência*.
   - **Remoção de Código:** Devemos extratir definitivamente as funções de `--predict` (Predição em massa do dataset inteiro) e `--modelo N` (Teste da inferência simulada) de dentro de `treinar_unsloth.py` e `treinar_unsloth_actions.py`.
   - *Por quê?* Porque acopla severamente e eleva a mil a quantidade de linhas focadas em instanciar inferências limpas, parsing de formatação em "respostas json" das avaliações do `user/assistant` e tratamentos de métricas da predição em si. Para que o motor de Treinamento seja limpo, blindado a bugs, e perfeitamente gerenciado, ele deve **apenas Treinar e dar Merge**. 
   - A inferência e testes devem/serão transferidas para script próprio externo `treinar_unsloth_avaliar.py` focado apenas em avaliação e teste do modelo, aproveitando estruturas compartilhadas para carga dos dados, modelo e checkpoints. Um CLI próprio será feito para permitir exportar o modelo para outros formatos, realizar merge ou gerar predições em massa

### Ajustes finos
- [ ] **Ajuste Dinâmico de `max_seq_length`:** Antes de dar início ao processo, analisar a extensão máxima de tokens efetivamente presente no dataset atual da "fase". Caso os dados ultrapassem o valor configurado, deve ser levantada uma exceção, caso o valor se max_seq_length seja omitido ou seja 0, deve-se readequar o limite automaticamente utilizando aproximação por margens (arredondando o teto máximo para múltiplos de 256 ou 512 tokens). Ex: se uma etapa atinge um teto de 1.830 tokens nas análises prévias, arredondaremos o respectivo `max_seq_length` estritamente para 2.048. Deve sempre ser definido um valor com margem mínima de 512 de folga para o maior volume de tokens encontrado. Essa é uma estratégia importante por dois motivos: garante que todo treino caiba ileso (sem truncamento agressivo da resposta do assistente) e enxuga VRAM severa e precisamente.
- [ ] Adicionar suporte a Early Stopping configurável, seguindo a lógica de transição de etapas do curriculum.
- [ ] Melhorar o CLI de treinamento e avaliação, listando os arquivos yaml compatíveis para escolha se não for passado como parâmetro. Criando um menu com opções para escolher
 - treino: Treinar, info para imprimir parãmetros / módulos do modelo para análise ou gerar o yaml padrão
 - avaliacao: Avaliar, Exportar ou Gerar Predições em Massa.