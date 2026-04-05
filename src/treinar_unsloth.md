# Processamento Completo de LLMs (Treinamento, Predição e Avaliação)

## 🎯 Objetivo do Módulo (Visão Geral)
O pacote `treinar_unsloth.py` forma o núcleo de um ecossistema completo para fine-tuning, inferência e avaliação estatística de Modelos LLM (como Gemma-3, Qwen, Deepseek e Llama). O foco central é permitir a orquestração desde o Treinamento SFT unificado rodando sob baixo custo (Unsloth) atravessando até um Pipeline de Predição local e remota (vLLM, Ollama), tudo sem a necessidade de codificação manual — orquestrado unicamente por perfis YAML de alto nível.

## 🚀 Funcionalidades Principais
- **Treinamento Multiestágios (Curriculum Learning)**: Defina arquivos diferentes para estágios subsequentes do seu aprendizado limitando o loss ou épocas de avanço automático para refino iterativo sem interrupção.
- **Inferência Multimotores OOP**: Motor plug-n-play para avaliação: usa nativamente o pipeline HuggingFace, aloca de forma extremamente rápida infraestruturas locais com motor compatível vLLM ou se integra para despachar a API Local do Ollama.
- **Ecossistema Resiliente a Checkpoints**: Gestão autônoma de checkpoints LoRA e travas de conclusão de Curriculum, impedindo que reexecuções apaguem acidentalmente treinamentos ou subescrevam frentes já avaliadas com sucesso.
- **Dados Sensíveis e Proteção**: Suporte granular para ingestão e decodificação na hora de Parquets Criptografados (Fernet). Fail-fast garantido para evitar treinamentos inteiros usando base mal formatada e protegida.

## 🛠️ Como Executar (Início Rápido)
A interface é unificada por uma TUI rica rodando com Menu Interativo caso você não informe os parâmetros.
```bash
# === MODO MENU INTERATIVO (Recomendado) ===
python src/treinar_unsloth.py
python src/treinar_unsloth_avaliar.py

# === TREINAMENTO ===
python src/treinar_unsloth.py meu_config.yaml --treinar

# === AVALIAÇÃO e ESTATÍSTICA ===
# Gera insights visuais profundos sobre o Dataset, Context Length ideal, uso de tokens.
python src/treinar_unsloth_avaliar.py meu_config.yaml --stats

# === PREDIÇÃO EM LOTE ===
# Faz o modelo prever respostas para todos os datasets de teste (usando o motor super-rápido vLLM por padrão)
python src/treinar_realizar_predicoes.py meu_config.yaml --engine vllm --predict
```

## ⚙️ Configuração (O Arquivo YAML)
O comportamento de todos os scripts transita em volta do seu YAML. Os principais pontos que os pesquisadores devem observar e configurar para um novo experimento são:

- **`modelo`**:
  - `base_model_name`: Identificador no Hugging Face (ex: `Qwen/Qwen2.5-1.5B-Instruct`) ou caminho local de um fallback preexistente.
  - `saida`: Pasta raiz onde **tudo** que seu modelo evolutivo produzir nascerá. Os checkpoints ficarão em `/chkpt`, os gráficos na pasta de treinamento, e as respostas na pasta de sua escolha.
  - `alias`: (Opcional) Alias descritivo do experimento (ex: `"Grupo01-MINI curriculum"`). Se preenchido, a pasta de relatórios e logs será nomeada `treinamento (<alias>)` em vez de `treinamento`. Isso permite reunir outputs de múltiplos experimentos numa mesma pasta para análise comparativa, pois cada um terá nome diferenciado. Se o alias for adicionado **depois** de um treinamento já iniciado, a pasta `treinamento` existente é **renomeada automaticamente** para `treinamento (<alias>)` na próxima execução.
  - `ollama`: Se for avaliar um modelo consolidado convertido lá.

- **`curriculum` (Fluxo de Entrada e Avaliação)**: A subchave principal de arquitetura experimental.
  - `saida`: Local (pasta/Parquet) de onde o `gold_standard` perfeito (resposta humana alvo) mora na máquina. Obrigatório.
  - `entrada`: Local das entradas não parseadas do Dataset. Suporta criptografia. Pode pular se o par parquet estiver completo em `saida`.
  - **`divisao`**: (Lista). Essencial. Descreve as Etapas. Se você quer treinar um LORA de X epocas, defina:
    - `arquivo`: CSV exato da gaveta fracionada alvo produzida lá no pacote `comparar_extracoes`
    - `tipo`: `lora` ou `full`
    - `pace_epochs`: Mínimo de épocas garantido para a etapa. Se `pace_loss` estiver ativo, o treinamento nunca para antes desse mínimo.
    - `pace_epochs_max`: (Opcional) Máximo de épocas para a etapa. Se `pace_loss` não for atingido até essa época, a etapa finaliza. O máximo é inclusivo (ex: `pace_epochs_max: 2` executa 2 épocas completas). Se omitido, o teto é `pace_epochs`.
    - `pace_loss`: (Opcional) Eval loss alvo (validation loss). Após completar `pace_epochs` mínimo, se o eval_loss cair abaixo desse valor ao final de uma época, a etapa avança automaticamente. Usar eval_loss (e não training loss) é o padrão acadêmico — evita decisões baseadas em overfitting. Requer que `eval_steps` esteja configurado para que avaliações ocorram durante o treinamento. Se `pace_epochs_max` estiver configurado, o treinamento é limitado a esse teto mesmo que o loss nunca atinja o alvo. Se `pace_loss` não for definido (ou 0), a etapa sempre treina exatamente `pace_epochs` épocas.
    - `max_seq_length`: (Opcional) Define o comprimento máximo de sequência para a etapa. Se omitido (ou 0), e o `max_seq_length` global também for omitido, o valor é **auto-estimado** a partir da coluna `token_total` do CSV de divisão (max + 10% margem, arredondado para múltiplo de 128). Se definido, funciona como **teto** que trunca instâncias maiores. Importante colocar fixo quando a memória GPU é limitada (ex: RTX 3060 12GB).
    - `learning_rate`: (Opcional) Sobrepõe o `learning_rate` global apenas nesta etapa.
    - `batch_size`: (Opcional) Sobrepõe o `batch_size` por GPU (`treinamento.batch_size.batch_size`) apenas nesta etapa. Útil quando etapas com `max_seq_length` menor permitem batch maior, ou etapas com sequências longas exigem batch reduzido para evitar OOM.

- **`treinamento`**:
  - `max_seq_length`: Comprimento máximo de sequência. Se **omitido ou 0**, o sistema auto-estima a partir de `max(token_total)` dos CSVs de divisão + 10% margem, arredondado para múltiplo de 128. Se o CSV não possuir a coluna `token_total`, falha com instrução para definir manualmente. Se **definido > 0**, funciona como teto que trunca instâncias maiores. Quando o global é auto-estimado, cada etapa do curriculum também recebe um valor auto-estimado a partir do seu próprio CSV (otimizando memória por etapa).
  - `batch_size`: Suporta `efetivo: N` para autoavaliar quantas GPUs o torch tem na ponta e calcular perfeitamente o Gradient Acceleration Substep garantindo reprodutibilidade independentemente da topologia física!
  - `train_on_responses_only`: (true/false) Se a perda da atenção deve pular o lado Prompter (Usuário). Ótimo para modelos instruct.

## 🔄 Como Replicar Experimentos e Reutilizar Código
- **Retomada Autônoma**: Se um experimento for interrompido, baste re-rodar `--treinar`. O script escaneará `/chkpt`, subirá o state de onde parou as loss das métricas, e continuará exatamente na Época ou Pace que foi interrompido.
- **Versionamento Embutido**: Sem precisar versionar pelo git. A cada iteração ou "Resume" válido da sua frente, dentro da pasta `saida` alçada na pasta de treinamento em `treinamento_config`, viverão `.yaml` prefixados como cópia física perfeita congelada em tempo dos specs do dia (`(v001)`, `(v002)...`).
- Todo **log** e **visualização** ficará eternizado perfeitamente grafado na pasta de treinamento (`treinamento` ou `treinamento (<alias>)` se `modelo.alias` estiver configurado). Os perfis de RAM consumida, Tokens Processados e curvas de convergência residirão lá.

---

## � Relatório Estatístico e Gráficos (`--stats`)

Ao final do treinamento (ou via `--stats`), o sistema gera automaticamente um relatório completo na pasta de treinamento (ex: `<saida>/treinamento/relatorio_estatistico.md` ou `<saida>/treinamento (Grupo01-curriculum)/relatorio_estatistico.md` se `modelo.alias` estiver configurado) com gráficos e tabelas:

### Gráficos Gerados

| Gráfico | Arquivo | Conteúdo |
|---------|---------|----------|
| Evolução do Loss | `treinamento_loss.png` | Train loss, eval loss por etapa, eval loss global, transições de etapa curriculum. Escala log automática quando o range dinâmico é grande. Raleamento de labels de época quando > 10 épocas. |
| Custo Computacional | `treinamento_tokens.png` | Tokens reais acumulados × instâncias treinadas ao longo dos steps |
| Eficiência Tokens/Loss | `treinamento_eficiencia_tokens.png` | eval_loss global × eficiência marginal suavizada ao longo dos tokens acumulados |
| Uso de Memória | `hardware_memoria.png` | RAM, GPU VRAM reservada (treino e avaliação) ao longo do treinamento |

### Contagem Real de Tokens

O campo `tokens_acumulados` registrado no `training_metrics.jsonl` reflete o **número real de tokens** do dataset tokenizado, não uma estimativa baseada em `max_seq_length`. A contagem é feita uma única vez por etapa, logo após o SFTTrainer tokenizar o dataset:

```
média_tokens = sum(len(input_ids) por instância) / num_instâncias
tokens_acumulados = instâncias_processadas × média_tokens
```

Isso garante que a métrica de custo computacional seja precisa mesmo em datasets com alta variância de comprimento (ex: instâncias de 100 a 900 tokens com `max_seq_length: 1024`). Para épocas completas, o valor é **exato**; para épocas parciais, é uma aproximação muito precisa.

### Eficiência de Tokens (tokens/Δloss)

Métrica que quantifica o **custo computacional por unidade de melhoria**:

```
tokens_por_delta_loss = tokens_processados / (eval_loss_inicial - eval_loss_final)
```

- **Referência:** Usa `eval_loss` (validation loss), não training loss — padrão acadêmico que evita decisões baseadas em overfitting.
- **Cálculo global:** Usa **eval_loss_global** (avaliação no dataset combinado de todas as etapas). Comparável entre etapas do curriculum porque avalia sempre o mesmo dataset. É o valor apresentado no gráfico e no info text.
- **Cálculo por etapa:** Usa **eval_loss por etapa** (avaliação no dataset específico daquela etapa). Permite comparar a eficiência de cada etapa isoladamente: etapas mais fáceis tipicamente têm melhor custo-benefício. Apresentado na tabela do relatório.
- **Gráfico:** O eixo X mostra tokens acumulados (total global desde o início do treinamento). Eixo Y esquerdo mostra **eval_loss global** (azul — onde o modelo está). Eixo Y direito mostra **eficiência marginal suavizada** (vermelho — |Δloss|/Δtokens entre avaliações, suavizado com média móvel janela=3). Marcadores violeta indicam transições de etapa.
- **Interpretação:**
  - **Eficiência alta** → aprendizado rápido (cada token processado contribui significativamente)
  - **Eficiência caindo** → retornos decrescentes (modelo está convergindo)
  - **Eficiência ≈ 0** → modelo parou de melhorar (sinal natural de parada)
  - Picos de eficiência em transições de etapa indicam que o curriculum introduziu exemplos que impulsionaram o aprendizado.
- **Caso sem melhoria:** Se eval_loss não diminui (Δloss ≤ 0), a métrica é reportada como "∞ (sem melhoria)".

### Visualização Adaptativa

Todos os gráficos de treinamento se adaptam automaticamente à densidade dos dados:

- **Modo denso (> 150 pontos):** Remove marcadores de ponto individuais das séries, usa linhas mais finas (1.5px) e alpha reduzido, semelhante à visualização do TensorBoard/W&B.
- **Escala log automática (gráfico de loss):** Quando o range dinâmico do loss é grande (max/min > 5×), ativa escala logarítmica no eixo Y, padrão acadêmico para curvas de loss com decaimento rápido seguido de plateau.
- **Raleamento de épocas:** Quando > 10 épocas, exibe labels apenas a cada N-ésima época (~8-10 visíveis), com as demais como linhas sutis sem texto.
- **Melhor checkpoint global:** Marcado em todos os gráficos como "Melhor global" (laranja), baseado no menor eval_loss_global (ou eval_loss se não houver avaliação global).

---

## �🔥 Decisões de Implementação: Liger Kernel e Flash Attention 2

### Contexto

Após a migração do Unsloth para HuggingFace Transformers + PEFT + TRL, duas otimizações de memória foram integradas como opções configuráveis no YAML:

```yaml
treinamento:
  flash_attention_2: true   # Atenção O(n) em VRAM em vez de O(n²)
  liger_kernel: true         # Fused cross-entropy, RoPE, RMSNorm (~40% menos pico de VRAM)
```

Ambas são habilitadas por padrão (`true`) e validadas no carregamento: se ativas no YAML mas o pacote não estiver instalado, o treinamento **falha imediatamente** com instruções de como instalar ou desativar (princípio fail-fast).

---

### Flash Attention 2

**O que faz:** Implementação de atenção com complexidade de memória O(n) em vez de O(n²), fundamental para sequências longas (ex: `max_seq_length: 35840`).

**Detecção:** Usa `transformers.utils.is_flash_attn_2_available()` no nível do módulo (`treinar_model_loader.py`). A detecção via `import flash_attn` diretamente não é confiável — a função do Transformers verifica a mesma coisa que é checada internamente quando se passa `attn_implementation="flash_attention_2"`.

**Fallback:** Se o modelo não suportar Flash Attention 2 (ex: arquitetura incompatível), o carregamento captura a exceção e retenta automaticamente com `attn_implementation="sdpa"` (Scaled Dot Product Attention do PyTorch).

**Arquivo:** `treinar_model_loader.py` → `ModelLoader.load_base_model()`.

---

### Liger Kernel (AutoLigerKernelForCausalLM)

**O que faz:** Substitui `AutoModelForCausalLM` por `AutoLigerKernelForCausalLM`, que aplica **fused kernels** para cross-entropy loss, RoPE e RMSNorm. A principal economia vem da fused cross-entropy: em vez de materializar o tensor completo de logits `(batch × seq_len × vocab_size)`, a loss é computada diretamente a partir dos hidden_states, reduzindo o pico de VRAM em ~40%.

**Consequência arquitetural:** `outputs.logits` retorna `None` quando o Liger Kernel está ativo (os logits nunca são materializados).

**Arquivo:** `treinar_model_loader.py` → `ModelLoader.load_base_model()` (carregamento), `treinar_unsloth.py` (validação e patches).

#### Decisão 1: Múltiplas GPUs — Fail-fast com RuntimeError

**Problema:** A fused cross-entropy loss do Liger Kernel exige que `hidden_states` e `lm_head.weight` estejam no **mesmo device**. Com `device_map="auto"` e múltiplas GPUs, o Accelerate distribui camadas entre GPUs (model parallelism), causando `RuntimeError: Expected all tensors to be on the same device`.

**Decisão anterior (descartada):** Silenciosamente forçar `device_map="cuda:0"`, desperdiçando GPUs disponíveis sem o usuário saber.

**Decisão atual:** Interromper o treinamento com `RuntimeError` contendo:
- Explicação do problema (fused loss + device mismatch)
- Estado atual (quantas GPUs, valor de `CUDA_VISIBLE_DEVICES`)
- Três soluções concretas:
  1. `export CUDA_VISIBLE_DEVICES=0` — restringir a uma GPU
  2. `torchrun --nproc_per_node=N` — usar DDP (cada processo enxerga 1 GPU)
  3. `liger_kernel: false` no YAML — desativar e permitir model parallelism

**Justificativa:** O usuário deve tomar a decisão conscientemente. Forçar silenciosamente uma GPU pode levar a treinamentos subótimos em infraestrutura multi-GPU sem que o pesquisador perceba.

#### Decisão 2: Informar o TRL via `use_liger_kernel` no SFTConfig

**Problema:** O TRL SFTTrainer (`compute_loss`) possui dois blocos que acessam `outputs.logits`:
1. **Entropia**: `entropy_from_logits(outputs.logits)`
2. **Token accuracy**: `outputs.logits[..., :-1, :].contiguous()`

Ambos estão protegidos por `if not self.args.use_liger_kernel`, mas esse parâmetro precisa ser passado ao `SFTConfig`.

**Decisão:** Adicionar `use_liger_kernel=treino_cfg.liger_kernel and _LIGER_DISPONIVEL` na construção do `SFTConfig`. Assim o TRL pula nativamente os blocos de entropia e token accuracy quando o Liger está ativo.

**Rede de segurança:** Um monkey-patch de `entropy_from_logits` permanece ativo no nível do módulo (`treinar_unsloth.py`, linhas 108-130), retornando `torch.tensor(0.0)` quando `logits is None`. Este patch é seguro mesmo sem Liger — é um passthrough transparente (único `is None` check, overhead de nanossegundos). Serve como proteção contra futuras versões do TRL que possam adicionar novos acessos a logits fora das guardas existentes.

#### Decisão 3: Defaults e Validação

**Defaults no YAML:** Ambas as otimizações são `true` por padrão no template YAML gerado automaticamente. A lógica é que se o ambiente suporta, devem estar ativas — a economia de VRAM é significativa sem impacto na qualidade do treinamento.

**Validação fail-fast:** Se `flash_attention_2: true` no YAML mas o pacote não está instalado, ou `liger_kernel: true` mas `liger-kernel` não está no ambiente, o treinamento **não inicia** e exibe `RuntimeError` com instruções de `pip install` ou como desativar no YAML. Isso evita que o pesquisador descubra o problema após horas de preprocessamento de dados.

---

### Resumo das Dependências Opcionais

| Otimização | Pacote | Instalação | Flag YAML | Detecção |
|---|---|---|---|---|
| Flash Attention 2 | `flash-attn` | `pip install flash-attn --no-build-isolation` | `flash_attention_2: true` | `is_flash_attn_2_available()` |
| Liger Kernel | `liger-kernel` | `pip install liger-kernel` | `liger_kernel: true` | `import AutoLigerKernelForCausalLM` |