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
  - `saida`: Pasta raiz onde **tudo** que seu modelo evolutivo produzir nascerá. Os checkpoints ficarão em `/chkpt`, os gráficos em `/treinamento`, e as respostas na pasta de sua escolha.
  - `ollama`: Se for avaliar um modelo consolidado convertido lá.

- **`curriculum` (Fluxo de Entrada e Avaliação)**: A subchave principal de arquitetura experimental.
  - `saida`: Local (pasta/Parquet) de onde o `gold_standard` perfeito (resposta humana alvo) mora na máquina. Obrigatório.
  - `entrada`: Local das entradas não parseadas do Dataset. Suporta criptografia. Pode pular se o par parquet estiver completo em `saida`.
  - **`divisao`**: (Lista). Essencial. Descreve as Etapas. Se você quer treinar um LORA de X epocas, defina:
    - `arquivo`: CSV exato da gaveta fracionada alvo produzida lá no pacote `comparar_extracoes`
    - `tipo`: lora ou full
    - `pace_epochs`: (ou pace_loss) quando que a roda deste subconjunto deve encerrar e o Early-Stopping atrelado pular.
    - `max_seq_length`: Importante colocar fixo de acordo com colunas `token_total` geradas.

- **`treinamento`**:
  - `batch_size`: Suporta `efetivo: N` para autoavaliar quantas GPUs o torch tem na ponta e calcular perfeitamente o Gradient Acceleration Substep garantindo reprodutibilidade independentemente da topologia física!
  - `train_on_responses_only`: (true/false) Se a perda da atenção deve pular o lado Prompter (Usuário). Ótimo para modelos instruct.

## 🔄 Como Replicar Experimentos e Reutilizar Código
- **Retomada Autônoma**: Se um experimento for interrompido, baste re-rodar `--treinar`. O script escaneará `/chkpt`, subirá o state de onde parou as loss das métricas, e engatará exatamente na Época ou Pace que sucumbiu.
- **Versionamento Embutido**: Sem precisar versionar pelo git. A cada iteração ou "Resume" válido da sua frente, dentro da pasta `saida` alçada em `treinamento/treinamento_config`, viverão `.yaml` prefixados como cópia física perfeita congelada em tempo dos specs do dia (`(v001)`, `(v002)...`).
- Todo **log** e **visualização** ficará eternizado perfeitamente grafado em `<saida>/treinamento`. Os perfis de RAM consumida, Tokens Processados e curvas de convergência residirão em `/treinamento/*hardware*, *loss*`.

---

## 🔥 Decisões de Implementação: Liger Kernel e Flash Attention 2

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