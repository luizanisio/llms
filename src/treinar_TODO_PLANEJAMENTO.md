# Planejamento do Treinamento com treinar_unsloth.py

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
13. **Simplificação do YAML — Curriculum como Formato Único**: Removidos os modos `pastas` e `dataset` (`formatos.tipo_entrada`). Agora o YAML possui apenas a seção `curriculum` com subcampos `entrada`, `saida`, `predicao`, `divisao` e `validacao`. Tanto `entrada` quanto `saida` suportam pasta de arquivos OU dataframe parquet. Criptografia por campo (`entrada.texto_criptografado`, `entrada.prompt_criptografado`, `saida.texto_criptografado`). Classes removidas: `ConfigFormatos`, `ConfigGold`, `ConfigPastas`, `ConfigDataset`. Classes adicionadas: `ConfigSaida`, `ConfigCurriculum`. CLI simplificado: `--criar-exemplo` substitui `--criar-exemplo-pastas`/`--criar-exemplo-dataset`.
14. **Menus Padronizados com `exibir_menu_opcoes`**: Função reutilizável em `util_print.py` que renderiza menus como tabela visual alinhada (colunas de tecla, nome e descrição). Suporta sub-itens indentados (nível 1), cores por item, cabeçalhos de seção e notas. Menus de treinamento (`treinar_unsloth.py`) e avaliação (`treinar_unsloth_avaliar.py`) unificados usando esta função, eliminando formatação manual com `logger.info()`.
15. **Separação Exportação/Inferência**: Módulo `treinar_unsloth_export.py` extraído de `treinar_unsloth_avaliar.py` com 13 funções de exportação e inferência: `executar_predict` (HF, vLLM, Unsloth), `executar_modelo` (HF, vLLM, Unsloth), `executar_merge`, `_compactar_modelo_zip`, `_executar_merge_hf` e helpers (`_perguntar_subsets_predict`, `_registro_ja_exportado`, `_construir_mapa_etapas`, `_copiar_para_pastas_etapas`). `treinar_unsloth_avaliar.py` mantém avaliação, estatísticas e CLI (~900 linhas, redução de ~65%).

### Histórico de Desenvolvimento
> Curriculum Learning, simplificação do código e unificação de formatos

**Objetivo:** Permitir um fluxo de treinamento em múltiplos estágios (Curriculum Learning) alternando dados, estratégias (LoRA vs Full Fine-Tuning) e critérios de parada dinâmicos (Pace), com formato de configuração unificado.

O desenvolvimento foi dividido nas seguintes etapas incrementais:

#### Passo 1: Separação de Preocupações e Melhoria do CLI ✅ CONCLUÍDO
**Objetivo:** Desacoplar a inferência do motor de treinamento para blindar e otimizar o código base do Treinador, centralizando o treinamento para focar *apenas Treinar e dar Merge*, e melhorar a experiência CLI.

**Implementado:**
1. ✅ **Extração da Avaliação/Inferência:** Funções de avaliação (`executar_info`, `executar_stats`, `gerar_graficos_estatisticos`) em `treinar_unsloth_avaliar.py`. Funções de exportação/inferência (`executar_predict`, `executar_merge`, `executar_modelo` — HF, vLLM, Unsloth) em `treinar_unsloth_export.py`. Removidas de `treinar_unsloth_actions.py` (que agora contém apenas `executar_treinar`, `executar_reset`, `executar_injetar_dicas` e funções auxiliares compartilhadas).
2. ✅ **Dois Módulos Independentes:** `treinar_unsloth_avaliar.py` (~900 linhas) com CLI/menu interativo e funções de avaliação/estatísticas. `treinar_unsloth_export.py` (~1750 linhas) com todas as funções de exportação e inferência (3 motores: HF, vLLM, Unsloth).
3. ✅ **Menu Interativo (CLI):** Ambos os scripts usam `util_menu_opcoes.escolher_yaml(chave_obrigatoria='modelo')`. Argumento `config` é `nargs='?'` (opcional). Menu de ações específico: treino (treinar, reset+treinar, reset) e avaliação (info, stats, predict, modelo, merge).
4. ✅ **Separação `dataset`/`predicao`:** Nova `ConfigSaida` para o gold dataset (saídas esperadas, obrigatório, validação de existência). `ConfigPredicao` é pasta de saída (auto-criada). `parear_arquivos()` em `treinar_unsloth_dataset.py` usa `curriculum.saida` como fonte do gold.
* **⏱️ Teste Intermediário:** `--help` de ambos os scripts funciona. `--info` executa corretamente. Import de todos os módulos validado. Falta: teste completo de treinamento end-to-end e predição em massa.

#### Passo 2: O "Pipeline Universal" e Ajustes Finos (Pré Curriculum) ✅ CONCLUÍDO
**Objetivo:** Unificar a base de código do sistema atual antes de construir o Curriculum Learning multicamadas, assim o processo opera o mesmo sistema de logs (como se fosse de "apenas uma etapa").

1. **Pipeline Universal:** A seção `curriculum` é o formato único. Um treinamento simples (etapa única) é representado como curriculum com uma divisão. Toda parte de tracking funciona em cima desta lista universal.
2. **Log de Rastreiamento Unificado e Resumo:** Implementar que todo salvamento utilize métricas gravadas no esquema universal (`curriculum_state.json` constando `{"current_step": 0, "status": "running"}` e `curriculum_metrics.jsonl`), abandonando outros tipos de lógicas divergentes.
3. ✅ **Simplificação do `max_seq_length` e Remoção do Cache:** A lógica de cálculo automático (e _dados_automaticos.json) provou ser uma complicação desnecessária e foi removida e substituída por um comportamento estrito (ver Passo 3).
* **⏱️ Teste Intermediário:** Rodar um treinamento normal de teste (`pastas`) exigindo que o código passe perfeitamente sem as checagens e cálculos automáticos de contexto e não trave por arquivos de cache.

#### Passo 3: Motor Multietapas do Curriculum Learning
**Objetivo:** Adicionar interpretador do YAML para Curriculum, transições e regras de `LoRA` \leftrightarrow `Full`.

1. ✅ **Estrutura YAML (Curriculum Unificado):** A seção `curriculum` é o único formato de configuração de dados. Entrada e saída suportam pasta OU dataframe. Criptografia granular por campo:
```yaml
curriculum:
  predicao:
    pasta: ./predict/output
  saida:
    pasta: ./saidas/gold       # OU dataframe: ./saidas/gold.parquet
    mascara: "*.txt"
    formato: json              # 'json' ou 'texto'
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
2. ✅ **Divisão Dinâmica ("Fail Fast"):** Evitar a autogeração baseada em divisões randômicas complexas ao usar curriculum. O sistema deve abortar prevenindo bugs se os subarquivos parametrizados (ex. `{arquivo}_facil.csv`) não existirem perfeitamente.
3. ✅ **Roteamento e Sobrevivência de Passos:**
    * Encapsular salvamentos no format de roteamento (ex: `{modelo.saida}/curriculum/01_facil`). Onde um retoma o modelo do passado.
    * No caso de Resume (`--treinar` de checkpoint quebrado), utilizar do state vivo (`curriculum_state.json`) extraído no passo 2 para instanciar subpastas `checkpoint-N` precisas resgatando o ponto cego daquela etapa exata.
4. **Mixando Modelos (LoRA vs Full):**
    * *Transição `[LoRA -> Full]`*: Mesclar base + lora via instanciador e usar o Merge como "o novo `FastLanguageModel` pleno" da segunda fase.
    * *Transição `[Full -> LoRA]`*: A requantização p/ nbits deve ser estritamente reacendida e embutida na modelagem ` FastLanguageModel.get_peft_model()` que sucede a transição.
5. ✅ **Gestão do `max_seq_length` por Estágio (Simplificado):**
    * ✅ **Remoção do Cache Complexo**: Classe `CacheSeqLength` e arquivo `_dados_automaticos.json` removidos. Métodos `calcular_max_seq_length()`, `resolver_max_seq_length()` e `_arredondar_seq_length()` removidos. Flag `validar_max_seq_length` eliminada do YAML e da `ConfigTreinamento`.
    * ✅ **Parâmetro Global Obrigatório**: `max_seq_length` é agora obrigatório (> 0) tanto no `__post_init__` quanto no `_processar_treinamento()`. O sistema aborta com erro fatal orientando o pesquisador a consultar a coluna `token_total` do CSV de divisão.
    * ✅ **Informações de Tokens por Etapa**: Novo método `_ler_info_tokens_divisao()` lê `token_total` do CSV de divisão (gerado pelo pacote de comparação). `validar_max_seq_length()` e `info()` exibem max/média de tokens por etapa e alertam se `max_seq_length` é insuficiente.
    * ✅ **Recarga Dinâmica**: Em `_aplicar_etapa_curriculum()`, se `max_seq_length` muda entre etapas, o modelo e tokenizer são recarregados automaticamente via `_load_model()` com o novo valor.

#### Passo 4: Pace Dinâmico e Ajustes Visuais de Controle
**Objetivo:** Interpolação analítica final garantindo eficiência via parada prematura baseada no desempenho e legibilidade de análise das métricas via Gráficos evolutivos de múltiplas fases.

1. **Pacing / Early Stopping Configurable:**
    * Programar `TrainerCallback` customizado plugado no Unsloth, que intercepta pós gatilhos `on_evaluate()`.
    * Checar `eval_loss <= pace_loss` ou total epochs se aproximarem de `pace_epochs`, injetar `control.should_training_stop = True` finalizando imediatamente aquela frente do currículo para poupar horas de servidor.
2. ✅ **Métricas de Eficiência Analíticas Estendidas:** `MetricsLoggerCallback` registra em `training_metrics.jsonl`: loss, lr, grad_norm, eval_loss, instâncias acumuladas, tokens acumulados, hardware (CPU, RAM, GPU), etapa do curriculum, step/epoch global. Gráficos de eficiência (tokens/instâncias) e hardware (memória) são gerados automaticamente.
3. ✅ **Legendagem Visual do Gráfico (Loss):** Implementado em `GraficoTreinamento.evolucao_loss()` e `construir_marcadores_etapas()`. Linhas violeta marcam transições entre etapas do curriculum com legendas dos aliases. Gráficos de eficiência e hardware também exibem marcadores de etapas.
4. ✅ **Controle de Conclusão e Retomada:** Trava estrutural implementada via `CurriculumTracker` em `treinar_unsloth_pipeline.py`. Bloqueia continuações de treinos já concluídos com `is_concluido()` + verificação de `target_epochs`. Liberação automática ao adicionar etapas ou aumentar épocas no YAML.
5. ✅ **Geração Automática de Gráficos Pós-Treinamento:** Função `gerar_graficos_estatisticos()` extraída em `treinar_unsloth_avaliar.py` e chamada automaticamente ao final de `executar_treinar()` em `treinar_unsloth_actions.py`. Gera loss, tokens, hardware e relatório .md sem necessidade de rodar `--stats` manualmente.
6. ✅ **Limpeza de Artefatos Antigos:** Ao iniciar novo treinamento (`etapa_index == 0`), o `MetricsLoggerCallback` remove gráficos e relatório estatístico anteriores junto com a truncagem do `training_metrics.jsonl`.
7. ✅ **Validação Fail-Fast de Criptografia:** `ConfigMisc.__post_init__` verifica se a variável de ambiente de criptografia existe quando configurada no YAML (`misc.env_chave_criptografia`). `_carregar_dataframe_entrada` em `treinar_unsloth_dataset.py` levanta `EnvironmentError` em vez de silenciosamente continuar com dados criptografados.
8. ✅ **Batch Size Automático:** Nova `ConfigBatchSize` em `treinar_unsloth_util.py`. Seção `treinamento.batch_size` (dict) com `efetivo` (batch desejado) e `batch_size` (por GPU). `_aplicar_batch_size_auto()` calcula `grad_batch_size` com base no nº de GPUs detectado via `torch.cuda.device_count()`, sobrescrevendo `treinamento.batch_size` e `treinamento.grad_batch_size` de forma transparente.
* **⏱️ Teste Intermediário Final:** Processar múltiplos estágios usando Curriculum completo. Embutir propositalmente uma meta `pace_loss = 1.5` de fácil alcance numa das passagens e testar os limites do Early-Stopping e o respectivo avanço para a etapa 2. Constatar a divisão formatada do gráfico unificado renderizado em `.png` ao encerramento pleno.

#### Backlog Múltiplas GPUs

##### Estado atual
O modelo é carregado em **GPU única** usando o default do Unsloth (`device_map="sequential"`). O parâmetro `device_map="auto"` foi **removido intencionalmente** de `FastModel.from_pretrained()` e `FastModel.get_peft_model()` em `treinar_unsloth.py`. Adicionalmente, as seguintes variáveis de ambiente são definidas no topo do script antes de qualquer import do Unsloth:
```python
os.environ["TORCH_COMPILE_DISABLE"] = "1"        # Desabilita Dynamo/Inductor global
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"       # Desabilita torch.compile na fused CE loss
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
```

##### Motivação: por que múltiplas GPUs estão desativadas
A fused cross-entropy loss do Unsloth (`unsloth_zoo/fused_losses/cross_entropy_loss.py`) utiliza internamente `torch.func.grad_and_value` (API do functorch) para computar loss e gradientes de forma chunked e memory-efficient sem materializar a matriz completa `[batch×seq_len, vocab_size]` de logits. Quando `device_map="auto"` é usado, o **Accelerate** instala hooks de dispatch (`accelerate/hooks.py`) que interceptam chamadas `forward` para mover tensores entre GPUs. Esses hooks são **fundamentalmente incompatíveis** com `torch.func.grad_and_value`, que cria objetos `TensorWrapper` internos (subclasse C++ de `TensorImpl`) — o Accelerate não sabe manipular esses wrappers, resultando no erro fatal:
```
NotImplementedError: Cannot access storage of TensorWrapper
```
O erro ocorria na segunda etapa do curriculum learning, quando o trainer era reconstruído e o modelo ainda carregava estado de autograd residual da etapa anterior, agravando a incompatibilidade.

##### Erros solucionados
1. **`NotImplementedError: Cannot access storage of TensorWrapper`** — Removido `device_map="auto"` de `_load_model()` (tanto no carregamento do modelo LoRA treinado via `FastModel.from_pretrained()` quanto na aplicação de adaptadores via `FastModel.get_peft_model()`). Com o default `"sequential"` do Unsloth, o modelo fica em GPU única e a fused loss funciona normalmente.
2. **Compilação dinâmica conflitante** — Adicionado `UNSLOTH_COMPILE_DISABLE=1` (lido em import-time pelo compiled cache e pelo `cross_entropy_loss.py`) para desabilitar `torch.compile(fullgraph=True)` na função `accumulate_chunk` da fused loss. Isso remove uma segunda camada de incompatibilidade entre `torch.compile` + `torch.func.grad_and_value`.
3. **Estado de autograd residual entre etapas** — Adicionada limpeza explícita entre etapas do curriculum (`model.zero_grad(set_to_none=True)`, `del trainer`, `gc.collect()`, `torch.cuda.empty_cache()`) para evitar que gradientes e grafos computacionais da etapa anterior contaminem a próxima.
4. **`metrics_stream.jsonl` apagado na etapa 2+** — O arquivo de métricas brutas era deletado a cada reconstrução do trainer. Corrigido para só limpar na `etapa_index == 0`.

##### O que precisa ser feito para usar múltiplas GPUs no futuro
A abordagem recomendada é **não** usar `device_map="auto"` (model parallelism do Accelerate), mas sim uma das estratégias de paralelismo nativas do HuggingFace Trainer:

1. **DeepSpeed ZeRO (recomendado para LoRA + QLoRA):**
   - Criar arquivo `ds_config.json` com configuração ZeRO Stage 2 (partição de gradientes e estados do otimizador entre GPUs).
   - Adicionar `deepspeed="ds_config.json"` no `SFTConfig` em `_build_trainer()`.
   - Lançar via `accelerate launch` ou `torchrun --nproc_per_node=N treinar_unsloth.py`.
   - **Cuidado:** Verificar compatibilidade do Unsloth com DeepSpeed — a fused loss pode precisar de ajustes ou ser desabilitada (`UNSLOTH_RETURN_LOGITS=1`).

2. **FSDP (Fully Sharded Data Parallel):**
   - Configurar via parâmetro `fsdp` e `fsdp_config` do `TrainingArguments`.
   - Requer que o modelo seja compatível com wrapping FSDP (pode conflitar com PEFT/LoRA).

3. **DataParallel simples (DDP via Trainer):**
   - O HuggingFace Trainer já suporta DDP automaticamente quando lançado via `torchrun`.
   - Cada GPU recebe uma réplica completa do modelo e processa batches diferentes.
   - **Limitação:** Exige que o modelo inteiro caiba em cada GPU individual.

4. **Investigar bypass da fused loss:**
   - O compiled cache (`unsloth_compiled_module_qwen2.py`) hardcoda a chamada a `unsloth_fused_ce_loss` sem escape via env var para o branch de treinamento (Branch 2). `UNSLOTH_RETURN_LOGITS=1` só afeta o Branch 1 (CCE para pesos frozen).
   - Uma solução futura seria contribuir upstream ao Unsloth um flag para desabilitar a fused loss, ou modificar o compiled cache para cair no Branch 3 (CE loss padrão do PyTorch) quando multi-GPU é detectado.

5. **Pré-requisitos de infraestrutura:**
   - Validar que `CUDA_VISIBLE_DEVICES` expõe as GPUs desejadas.
   - Garantir que o script seja compatível com lançamento distribuído (`if __name__ == "__main__"` + `torch.distributed.init_process_group()`).
   - Adaptar o sistema de callbacks e métricas para não duplicar registros em processos múltiplos (registrar apenas no `rank == 0`).