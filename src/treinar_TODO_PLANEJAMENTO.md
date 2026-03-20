# Planejamento do Treinamento com treinar_unsloth.py

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
3. **[Pendente] Simplificação do `max_seq_length` e Remoção do Cache:** A lógica atual de cálculo automático (e _dados_automaticos.json) provou ser uma complicação desnecessária e será removida e substituída por um comportamento estrito (ver Passo 3).
* **⏱️ Teste Intermediário:** Rodar um treinamento normal de teste (`pastas`) exigindo que o código passe perfeitamente sem as checagens e cálculos automáticos de contexto e não trave por arquivos de cache.

#### Passo 3: Motor Multietapas do Curriculum Learning
**Objetivo:** Adicionar interpretador do YAML para Curriculum, transições e regras de `LoRA` \leftrightarrow `Full`.

1. ✅ **Estrutura YAML:** Integrar suporte a configuração `curriculum` no arquivo. A seção `curriculum` segue a mesma estrutura que `pastas` (predicao, dataset, entrada, validacao), mas a subchave `divisao` é uma lista de etapas do pipeline:
```yaml
formatos:
  tipo_entrada: curriculum # Opções: dataset, pastas, curriculum

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
    prompt_template: './dados/prompt.txt'
    tag_texto: '<<--TEXTO-->>'
  validacao:
    exigir_json_valido: true
    skip_invalidos: false
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
2. ✅ **Divisão Dinâmica ("Fail Fast"):** Evitar a autogeração baseada em divisões randômicas complexas ao usar curriculum. O sistema deve abortar prevenindo bugs se os subarquivos parametrizados (ex. `{arquivo}_facil.csv`) não existirem perfeitamente.
3. ✅ **Roteamento e Sobrevivência de Passos:**
    * Encapsular salvamentos no format de roteamento (ex: `{modelo.saida}/curriculum/01_facil`). Onde um retoma o modelo do passado.
    * No caso de Resume (`--treinar` de checkpoint quebrado), utilizar do state vivo (`curriculum_state.json`) extraído no passo 2 para instanciar subpastas `checkpoint-N` precisas resgatando o ponto cego daquela etapa exata.
4. **Mixando Modelos (LoRA vs Full):**
    * *Transição `[LoRA -> Full]`*: Mesclar base + lora via instanciador e usar o Merge como "o novo `FastLanguageModel` pleno" da segunda fase.
    * *Transição `[Full -> LoRA]`*: A requantização p/ nbits deve ser estritamente reacendida e embutida na modelagem ` FastLanguageModel.get_peft_model()` que sucede a transição.
5. **Gestão do `max_seq_length` por Estágio (Simplificado):**
    * **Remoção do Cache Complexo**: O cálculo automático e cache (`_dados_automaticos.json`) serão inteiramente removidos para descomplicar a base de código.
    * **Parâmetro Global Obrigatório**: O `max_seq_length` será um parâmetro obrigatório global. O sistema abortará com erro fatal nas validações se estiver zerado ou ausente.
    * **Recarga Dinâmica**: O valor pode ser sobreposto em cada etapa do `curriculum`. Caso haja variação do `max_seq_length` durante a transição de um estágio para o outro, o pipeline se encarregará de recarregar a modelagem/tokenizer com a nova configuração, permitindo dar sequência com a nova limitação estritamente alocada.

#### Passo 4: Pace Dinâmico e Ajustes Visuais de Controle
**Objetivo:** Interpolação analítica final garantindo eficiência via parada prematura baseada no desempenho e legibilidade de análise das métricas via Gráficos evolutivos de múltiplas fases.

1. **Pacing / Early Stopping Configurable:**
    * Programar `TrainerCallback` customizado plugado no Unsloth, que intercepta pós gatilhos `on_evaluate()`.
    * Checar `eval_loss <= pace_loss` ou total epochs se aproximarem de `pace_epochs`, injetar `control.should_training_stop = True` finalizando imediatamente aquela frente do currículo para poupar horas de servidor.
2. **Métricas de Eficiência Analíticas Estendidas:** Em cada finalização de passo (`curriculum_metrics.jsonl`), adicionar logs em cada etapa englobando *Tamanho em instâncias geradas (Utilização e Validação do target)*, *eval_loss absoluto que atestou o fim da etapa*, *Tempo Real Clocked Time*, *VRAM Peak* na fase cruzada.
3. **Legendagem Visual do Gráfico (Loss):** Redesenhar e adaptar o método `GraficoTreinamento.evolucao_loss()` usando a engine base. Ele deve ler o tracker consolidado lido, identificar onde ocorreu a transição dos aliases e traçar linhas demarcatórias cruzando e assinalando como legenda "Etapa Fácil", "Etapa Médio", nos boxes e boxplots (em `--stats`).
* **⏱️ Teste Intermediário Final:** Processar múltiplos estágios usando Curriculum completo. Embutir propositalmente uma meta `pace_loss = 1.5` de fácil alcance numa das passagens e testar os limites do Early-Stopping e o respectivo avanço para a etapa 2. Constatar a divisão formatada do gráfico unificado renderizado em `.png` ao encerramento pleno.