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

#### Passo 2: O "Pipeline Universal" e Ajustes Finos (Pré Curriculum)
**Objetivo:** Unificar a base de código do sistema atual antes de construir o Curriculum Learning multicamadas, assim o processo opera o mesmo sistema de logs (como se fosse de "apenas uma etapa").

1. **Pipeline Universal:** Remover as lógicas apartadas. Se o YAML acionar apenas 1 dataset ou pastas (`tipo_entrada: dataset` ou `pastas`), o inicializador do sistema encapsulará isso convertendo automaticamente em uma lista `curriculum` de tamanho 1, definindo `alias` padrão como "Principal". Toda parte de tracking funcionará agora em cima desta lista universal.
2. **Log de Rastreiamento Unificado e Resumo:** Implementar que todo salvamento utilize métricas gravadas no esquema universal (`curriculum_state.json` constando `{"current_step": 0, "status": "running"}` e `curriculum_metrics.jsonl`), abandonando outros tipos de lógicas divergentes.
3. **Ajuste Dinâmico de `max_seq_length` e Cache:** Antes de dar início ao processo, codificar a análise da extensão máxima de tokens, considerando que o cálculo deve abranger a **janela de contexto total (entrada + saída gerada)**.
    * **Conferência de Cache:** Iniciar o treinamento verificando se existe um cache em `{modelo.saida}/_dados_automaticos.json`. O sistema deve conferir se o cache está atualizado e consiste aos dados alvo (por ex. validando a data, tamanho ou hash do dataset) antes de confiar no seu conteúdo (como o `max_seq_length`).
    * O controle respeitará a flag global `treinamento.validar_max_seq_length` (presente em qualquer YAML). Se ela for `false` e o `max_seq_length > 0`, o sistema pula inteiramente o recálculo dos tokens para poupar tempo (confiança estrita no valor passado pelo yaml global ou pelas sobreposições dos aliases no Curriculum).
    * Caso contrário, ou caso explicitamente `max_seq_length = 0` (e o cache esteja desatualizado ou inexistente), varre-se o dataset. Caso os tokens ultrapassem o teto fornecido pelo LLM ou YAML, levanta `Exception`. Se a limitação não engessar, o sistema arredonda para cima com margem teto (múltiplos de 256 com no mínimo 256 de folga entre o máximo identificado e o valor final).
    * O cálculo pode variar conforme o modelo base. Gravar os valores descobertos em `{modelo.saida}/_dados_automaticos.json` para evitar recálculos nas próximas execuções. Exemplo de chaves:
      ```json
      {
        "max_seq_length": 4096,
        "yaml_hash": "abcdef...",
        "yaml_atualizacao": "2026-03-18T20:00:00",
        "alias_facil": {"max_seq_length": 1024},
        "alias_medio": {"max_seq_length": 2048},
        "alias_dificil": {"max_seq_length": 4096}
      }
      ```
* **⏱️ Teste Intermediário:** Rodar um treinamento normal de teste (`pastas`) a partir do zero com `max_seq_length: 0`. As métricas exportadas e a folga no arquivo `_dados_automaticos.json` devem atuar perfeitamente. No teste estressado usando valor absoluto com flag _false_, certificar o "bypass" de CPU/GPU logado.

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
2. **Divisão Dinâmica ("Fail Fast"):** Evitar a autogeração baseada em divisões randômicas complexas ao usar curriculum. O sistema deve abortar prevenindo bugs se os subarquivos parametrizados (ex. `{arquivo}_facil.csv`) não existirem perfeitamente.
3. **Roteamento e Sobrevivência de Passos:**
    * Encapsular salvamentos no format de roteamento (ex: `{modelo.saida}/curriculum/01_facil`). Onde um retoma o modelo do passado.
    * No caso de Resume (`--treinar` de checkpoint quebrado), utilizar do state vivo (`curriculum_state.json`) extraído no passo 2 para instanciar subpastas `checkpoint-N` precisas resgatando o ponto cego daquela etapa exata.
4. **Mixando Modelos (LoRA vs Full):**
    * *Transição `[LoRA -> Full]`*: Mesclar base + lora via instanciador e usar o Merge como "o novo `FastLanguageModel` pleno" da segunda fase.
    * *Transição `[Full -> LoRA]`*: A requantização p/ nbits deve ser estritamente reacendida e embutida na modelagem ` FastLanguageModel.get_peft_model()` que sucede a transição.
5. **Validação de `max_seq_length` por Estágio:**
    * Quando for treinamento por currículo, o pipeline deve obrigatoriamente realizar a validação e o cálculo do `max_seq_length` em **todas** as etapas sequencialmente. Isso ocorre a menos que a verificação global (`treinamento.validar_max_seq_length`) esteja desativada, sendo indispensável para assegurar que os datasets iterativos de cada fase não ultrapassem a memória antes da execução e mantendo o cache coerente por alias.
* **⏱️ Teste Intermediário:** Simular um YAML com duas etapas restritas em LoRA. Processar passo 1 e pausar; retomar usando reexecução do script pelo CLI, e observar se os modelos são salvos nos seus compartimentos próprios no HD, e o passo 2 continua usando as raízes geradas. Adicionalmente, verificar se os logs exibem explicitamente a validação (ou leitura do cache) de `max_seq_length` ao assumir a transição da etapa 2.

#### Passo 4: Pace Dinâmico e Ajustes Visuais de Controle
**Objetivo:** Interpolação analítica final garantindo eficiência via parada prematura baseada no desempenho e legibilidade de análise das métricas via Gráficos evolutivos de múltiplas fases.

1. **Pacing / Early Stopping Configurable:**
    * Programar `TrainerCallback` customizado plugado no Unsloth, que intercepta pós gatilhos `on_evaluate()`.
    * Checar `eval_loss <= pace_loss` ou total epochs se aproximarem de `pace_epochs`, injetar `control.should_training_stop = True` finalizando imediatamente aquela frente do currículo para poupar horas de servidor.
2. **Métricas de Eficiência Analíticas Estendidas:** Em cada finalização de passo (`curriculum_metrics.jsonl`), adicionar logs em cada etapa englobando *Tamanho em instâncias geradas (Utilização e Validação do target)*, *eval_loss absoluto que atestou o fim da etapa*, *Tempo Real Clocked Time*, *VRAM Peak* na fase cruzada.
3. **Legendagem Visual do Gráfico (Loss):** Redesenhar e adaptar o método `GraficoTreinamento.evolucao_loss()` usando a engine base. Ele deve ler o tracker consolidado lido, identificar onde ocorreu a transição dos aliases e traçar linhas demarcatórias cruzando e assinalando como legenda "Etapa Fácil", "Etapa Médio", nos boxes e boxplots (em `--stats`).
* **⏱️ Teste Intermediário Final:** Processar múltiplos estágios usando Curriculum completo. Embutir propositalmente uma meta `pace_loss = 1.5` de fácil alcance numa das passagens e testar os limites do Early-Stopping e o respectivo avanço para a etapa 2. Constatar a divisão formatada do gráfico unificado renderizado em `.png` ao encerramento pleno.