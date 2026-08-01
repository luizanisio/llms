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

- **`misc`**: Configurações diversas do projeto.
  - `pasta_base`: (Opcional) Diretório base (absoluto ou relativo ao YAML) que atua como prefixo para que todos os caminhos subsequentes do arquivo de configuração (datasets, modelos, saídas) sejam resolvidos dinamicamente a partir dele. Facilita a portabilidade do projeto entre diferentes máquinas.

- **`modelo`**:
  - `base_model_name`: Identificador no Hugging Face (ex: `Qwen/Qwen2.5-1.5B-Instruct`) ou caminho local de um fallback preexistente.
  - `saida`: Pasta raiz onde **tudo** que seu modelo evolutivo produzir nascerá. Os checkpoints ficarão em `/chkpt`, os gráficos na pasta de treinamento, e as respostas na pasta de sua escolha.
  - `alias`: (Opcional) Alias descritivo do experimento (ex: `"Grupo01-MINI curriculum"`). Se preenchido, a pasta de relatórios e logs será nomeada `treinamento (<alias>)` em vez de `treinamento`. Isso permite reunir outputs de múltiplos experimentos numa mesma pasta para análise comparativa, pois cada um terá nome diferenciado. Se o alias for adicionado **depois** de um treinamento já iniciado, a pasta `treinamento` existente é **renomeada automaticamente** para `treinamento (<alias>)` na próxima execução.
  - `ollama`: Se for avaliar um modelo consolidado convertido lá.

- **`curriculum` (Fluxo de Entrada e Avaliação)**: A subchave principal de arquitetura experimental.
  - `saida`: Local (pasta/Parquet) de onde o `gold_standard` perfeito (resposta humana alvo) mora na máquina. Obrigatório.
  - `entrada`: Local das entradas não parseadas do Dataset. Suporta criptografia. Pode pular se o par parquet estiver completo em `saida`.
    - `dataset_filtro`: (Opcional) Dicionário (em formato JSON inline no YAML) para filtrar os dados de entrada na memória antes do treinamento. Suporta operadores relacionais como prefixo do valor (`!=`, `>`, `>=`, `<`, `<=`, `=`). Ex: `{"fold": "!=12", "dificuldade": "facil", "ano": ">=2020"}`. Quando utilizado, garante a exclusão automática dos registros de saída sem par na entrada filtrada e ignora os IDs sobressalentes no arquivo de divisão sem levantar falsos erros de inconsistência.
  - **`divisao`**: (Lista). Essencial. Descreve as Etapas. Se você quer treinar um LORA de X epocas, defina:
    - `arquivo`: CSV exato da gaveta fracionada alvo produzida lá no pacote `comparar_extracoes`
    - `tipo`: `lora` ou `full`. **Nota sobre Quantização:** O treinamento suporta recarregamento automático da quantização. Se a sua etapa possuir tipo `full`, o modelo será automaticamente recarregado em memória utilizando `16-bits`, ignorando a restrição de quantização global do YAML momentaneamente (uma vez que o ajuste total dos pesos necessita float16/bfloat16). Ao avançar para uma próxima etapa `lora`, ele recarrega o modelo com os pesos treinados obedecendo novamente os bits globais do YAML (ex: 4-bits ou 8-bits).
    - `pace_epochs`: Mínimo de épocas garantido para a etapa. Se `pace_loss` estiver ativo, o treinamento nunca para antes desse mínimo.
    - `pace_epochs_max`: (Opcional) Máximo de épocas para a etapa. Se `pace_loss` não for atingido até essa época, a etapa finaliza. O máximo é inclusivo (ex: `pace_epochs_max: 2` executa 2 épocas completas). Se omitido, o teto é `pace_epochs`.
    - `pace_loss`: (Opcional) Eval loss alvo (validation loss). Após completar `pace_epochs` mínimo, se o eval_loss cair abaixo desse valor ao final de uma época, a etapa avança automaticamente. Usar eval_loss (e não training loss) é o padrão acadêmico — evita decisões baseadas em overfitting. Requer que `eval_steps` esteja configurado para que avaliações ocorram durante o treinamento. Se `pace_epochs_max` estiver configurado, o treinamento é limitado a esse teto mesmo que o loss nunca atinja o alvo. Se `pace_loss` não for definido (ou 0), a etapa sempre treina exatamente `pace_epochs` épocas.
    - `max_seq_length`: (Opcional) Define o comprimento máximo de sequência para a etapa. Se omitido (ou 0), e o `max_seq_length` global também for omitido, o valor é **auto-estimado** a partir da coluna `token_total` do CSV de divisão (max + 10% margem, arredondado para múltiplo de 128). Se definido, funciona como **teto** que trunca instâncias maiores. Importante colocar fixo quando a memória GPU é limitada (ex: RTX 3060 12GB).
    - `learning_rate`: (Opcional) Sobrepõe o `learning_rate` global apenas nesta etapa.
    - `batch_size`: (Opcional) Sobrepõe o `batch_size` por GPU (`treinamento.batch_size.batch_size`) apenas nesta etapa. Útil quando etapas com `max_seq_length` menor permitem batch maior, ou etapas com sequências longas exigem batch reduzido para evitar OOM.
    - `warmup_steps`: (Opcional) Sobrepõe o `warmup_steps` global apenas nesta etapa. Cada etapa reconstrói o `SFTTrainer` do zero, o que **zera os momentos do Adam e reinicia o scheduler cosine** — uma rampa de aquecimento por etapa (~50-100 steps, recomendação da literatura de ReLoRA) mitiga o degrau de loss na fronteira. Atenção à sentinela: o valor "não informado" é `-1`, não `0`, porque `warmup_steps: 0` é uma escolha legítima ("sem warmup nesta etapa"). O global é restaurado a cada etapa, então um override não vaza para as seguintes.
    - `unfreeze_layers_from`: (Opcional) **Descongelamento progressivo** — treina apenas os blocos transformer a partir de um corte, congelando os inferiores. Aceita dois formatos: índice absoluto (`21`) ou percentual de blocos **congelados** (`75%`, que congela os primeiros 75% e treina os 25% finais). Válido **apenas** em etapas `tipo: "full"` — em etapas LoRA a capacidade já é controlada pelo rank do adaptador, e a tentativa levanta `ValueError` na carga do YAML. Ver a seção [Descongelamento Progressivo](#-descongelamento-progressivo-progressive-unfreezing) para a política completa.

  > **Proteção contra typos:** as chaves aceitas em cada item de `divisao` são validadas contra uma whitelist (`_CHAVES_ETAPA_VALIDAS` em `treinar_unsloth_pipeline.py`). Qualquer chave desconhecida gera um **warning explícito** no log, listando as válidas. Sem isso, um typo em `unfreeze_layers_from` desativaria o recurso silenciosamente e o experimento rodaria como full tradicional sem ninguém perceber.

- **`treinamento`**:
  - `max_seq_length`: Comprimento máximo de sequência. Se **omitido ou 0**, o sistema auto-estima a partir de `max(token_total)` dos CSVs de divisão + 10% margem, arredondado para múltiplo de 128. Se o CSV não possuir a coluna `token_total`, falha com instrução para definir manualmente. Se **definido > 0**, funciona como teto que trunca instâncias maiores. Quando o global é auto-estimado, cada etapa do curriculum também recebe um valor auto-estimado a partir do seu próprio CSV (otimizando memória por etapa).
  - `filtrar_max_seq_length`: (true/false) Se definido como true, os exemplos de treinamento e validação que ultrapassam o `max_seq_length` efetivo (da divisão ou global) serão **removidos** do dataset. Diferente do padrão do modelo (que apenas trunca as instâncias excedentes), esta flag garante que o modelo treine apenas com instâncias que cabem perfeitamente na janela de contexto delimitada. (Padrão: false).
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

## 🔄 Transições Curriculum: LoRA ↔ Full (Preservação de Pesos)

### Contexto

Quando o curriculum combina etapas `lora` e `full`, os pesos base do modelo são atualizados durante etapas `full`. Porém, `PeftModel.save_pretrained()` do PEFT salva **apenas os adaptadores LoRA**, descartando silenciosamente os pesos base treinados. Sem tratamento especial, transições `full→lora` perderiam todo o treinamento de pesos base.

### Regras

1. **Full sempre em 16 bits**: Etapas `full` forçam `nbits=16` automaticamente (pesos int4/int8 não suportam gradientes).
2. **Merge ao recarregar ou finalizar**: Em pipelines mistos (`lora` + `full`), o modelo é SEMPRE salvo como **full auto-contido** ao final do treinamento (`_save_model()`). Além disso, durante as transições de etapa que exigem recarregamento (ex: muda a quantização de `nbits=4` para `nbits=16` ou a atenção), o sistema faz o merge (`merge_adapter()`) e salva o modelo full no disco antes de limpar a memória.
3. **Remoção de adapters**: Após qualquer merge+save para a pasta de saída, os arquivos de adaptadores (`adapter_config.json`, `adapter_model.safetensors`, etc.) são excluídos do diretório final para não gerar conflito no carregamento e garantir que o modelo foi efetivado como *full*.
4. **Modelo full local como base no resume**: Ao retomar treinamento, se existir modelo full local na pasta de saída, o sistema usa o modelo local como base (não o HuggingFace), preservando os pesos da etapa anterior. Os checkpoints do HF Trainer (`chkpt/`) guardam o estado do otimizador e pesos intermediários.

### Fluxo Visual (Exemplo: LoRA int4 → Full 16 bits → LoRA int4)

```
Pipeline: lora (4-bit) → full (16-bit) → lora (4-bit)

lora (etapa 1, 4-bit)
  │
  └─ Fim da etapa: Transição para 16-bit exige recarregamento da memória
     Ação: MERGE LoRA→base + save full (~3GB) + remove adapter files da saída
     Disco (saída): model.safetensors (modelo completo com pesos atualizados)

full (etapa 2, 16-bit)
  │
  └─ Fim da etapa: Transição para 4-bit exige recarregamento da memória
     Ação: MERGE LoRA→base (se houver LoRA ativo) + save full + remove adapter files
     Disco (saída): model.safetensors (pesos atualizados com o treino full)

lora (etapa 3, 4-bit)
  │
  └─ Fim do pipeline: Chamada final ao _save_model()
     Ação: MERGE LoRA→base (pois é um pipeline misto) + save full + remove adapter files
     Disco (saída final): model.safetensors (modelo pronto, auto-contido)
```

### Comportamento de Transição (em Memória)

| Transição | Recarga? | Ação na Transição |
|---|---|---|
| lora→lora | Não (se nbits e attn iguais) | Mantém modelo na VRAM, apenas reconfigura `requires_grad` (se o target_modules não mudar) |
| full→full | Não (se attn igual) | Mantém modelo na VRAM, apenas reconfigura `requires_grad` |
| lora→full | Sim (se nbits muda de 4 para 16) | Merge + salva full na saída antes de descartar; recarrega em 16-bit e libera base |
| full→lora | Sim (se nbits muda de 16 para 4) | Salva full na saída antes de descartar; recarrega em 4-bit e reaplica LoRA (fresco) |

**Nota sobre LoRA fresco:** Quando ocorre uma transição que recarrega o modelo (ou quando entra num LoRA após um Full sem recarregar), os adaptadores LoRA são inicializados do zero. Isso é o comportamento correto do curriculum — a cada etapa `lora` inicia-se uma nova fase sobre os pesos base já consolidados. No caso de *resume* via checkpoint no meio da etapa, os pesos do adaptador são restaurados a partir do checkpoint respectivo (`chkpt/`).

### ⚠️ Erro de Requantização no Merge com `nbits: 4`

Quando a etapa LoRA anterior rodou em 4 bits, `merge_adapter()` opera sobre camadas bitsandbytes: o PEFT **dequantiza, soma o delta e requantiza** cada camada. Isso injeta erro de quantização nos pesos salvos — e é a origem do degrau de loss que **não retorna ao patamar anterior** nas transições LoRA→FF.

O sistema não corrige isso automaticamente (alterar o fluxo de merge seria invasivo), mas **emite um warning explícito no log** sempre que o merge acontece com `nbits_memoria == 4`, para que o fenômeno fique registrado nos artefatos do experimento:

```
⚠️  Merge LoRA→base com modelo em 4 bits: o merge dequantiza e REQUANTIZA cada camada,
    introduzindo erro de quantização nos pesos salvos (fonte do degrau de loss na
    transição de regime). Para eliminar o efeito, rode as etapas LoRA com nbits: 16.
```

**Como eliminar:** rodar as etapas LoRA com `nbits: 16`. O custo é VRAM (sem NF4), o benefício é remover por construção uma variável confundidora dos contrastes experimentais. Protocolos que comparam regimes de treinamento devem preferir 16 bits; protocolos focados em custo de inferência podem manter 4 bits e citar este warning como limitação conhecida.

### Arquivos Chave

- `treinar_unsloth.py` → `_save_model()` (merge incondicional para mistos), `_load_model()` (uso do full local), `_aplicar_etapa_curriculum()` (merge preventivo antes de descarte da memória).
- `treinar_unsloth_actions.py` → `_detectar_tipo_modelo_saida()` (identifica se a saída é Lora puro ou Full auto-contido).

---

## 🧊 Descongelamento Progressivo (Progressive Unfreezing)

### Contexto e motivação

O parâmetro por etapa `unfreeze_layers_from` permite que a **capacidade do modelo cresça ao longo do curriculum** sem trocar de técnica. É uma alternativa ao escalonamento LoRA→Full que evita três variáveis confundidoras de uma vez: **sem merge de adaptadores, sem requantização, sem troca de espaço de busca**.

A propriedade central é que as transições são ***function-preserving***: na fronteira entre etapas, **nenhum peso muda** — apenas mais blocos passam a ter `requires_grad = True`. O modelo que termina a etapa N é bit-a-bit o mesmo que começa a etapa N+1. Isso não acontece nas transições LoRA→FF, onde o merge altera os pesos.

Os blocos inferiores congelados funcionam como **âncora do conhecimento pré-treinado**, cumprindo papel análogo à regularização implícita do LoRA, mas sem restringir o espaço de busca a atualizações de baixo rank.

### Por que a implementação é pequena

O loop de `train()` reconstrói o `SFTTrainer` do zero a cada etapa, e o `Trainer` do HF monta o otimizador **filtrando `p.requires_grad`**. Basta portanto ajustar os flags antes de `_build_trainer` — exatamente o que `_aplicar_etapa_curriculum()` já fazia para alternar `full`/`lora`. O unfreeze é um refinamento do ramo `full`.

Entre etapas `full` consecutivas com o mesmo `nbits` (16), `precisa_recarregar` é `False`: o modelo **permanece em memória** e a transição é puramente uma reconfiguração de flags.

### Os dois formatos

| Formato | Exemplo | Significado |
|---|---|---|
| Índice absoluto | `unfreeze_layers_from: 21` | Treina os blocos com índice ≥ 21 |
| Percentual | `unfreeze_layers_from: 75%` | Congela os primeiros 75% dos blocos, treina os 25% finais |

**Decisão de design:** o percentual **não** é convertido no parse do YAML — o número de blocos só é conhecido após o carregamento do modelo. A etapa carrega a especificação bruta em `unfreeze_layers_pct` e a resolução para índice absoluto acontece em `_aplicar_unfreeze_parcial()`, contra `model.config.num_hidden_layers`.

O ganho é **portabilidade**: `75%` significa a mesma fração de profundidade em qualquer modelo. No Qwen 2.5, tanto o 1.5B quanto o 7B têm 28 blocos, então `75/50/25/0%` resolvem para `21/14/7/0` nos dois — o mesmo YAML descreve o protocolo em ambas as escalas.

> Nota YAML: `unfreeze_layers_from: 75%` sem aspas já é lido como a string `"75%"` pelo `yaml.safe_load` (o `%` impede a interpretação numérica). Aspas são opcionais.

### Política de congelamento

| Grupo de parâmetros | Regra |
|---|---|
| Bloco `layers.N.*` | Treinável se `N >= from_layer` |
| `embed_tokens` e `lm_head` | Treináveis **apenas** se `from_layer == 0` |
| Demais float (norm final, etc.) | Sempre treináveis |
| Parâmetros quantizados (int4/int8) | Intocados — não suportam gradiente |

#### Decisão: `lm_head` congelado junto com os embeddings

**Problema:** no Qwen 2.5 **1.5B**, `lm_head.weight` compartilha o tensor de `embed_tokens.weight` (`tie_word_embeddings: true`) — o par é **um parâmetro só**. No **7B** eles são tensores **separados** (untied, ~545M cada).

Se `lm_head` caísse na regra "demais float → sempre treinável", o mesmo protocolo treinaria a cabeça desde a primeira etapa no 7B e não no 1.5B. Os 545M da cabeça do 7B inflariam a etapa de 75% congelado para ~28% dos parâmetros, contra ~21% no 1.5B, **quebrando a comparabilidade entre experimentos**.

**Decisão:** congelar `lm_head` junto com `embed_tokens` até `from_layer == 0`. A progressão de capacidade fica praticamente idêntica nos dois modelos (medido nas arquiteturas reais):

| Corte | Qwen 2.5 1.5B (tied) | Qwen 2.5 7B (untied) |
|---|---|---|
| `75%` | 327.586.304 / 1.543.714.304 = **21,2%** | 1.631.408.128 / 7.615.616.512 = **21,4%** |
| `50%` | 655.171.072 = **42,4%** | 3.262.812.672 = **42,8%** |
| `25%` | 982.755.840 = **63,7%** | 4.894.217.216 = **64,3%** |
| `0%` | 1.543.714.304 = **100%** | 7.615.616.512 = **100%** |

A diferença residual (21,2% vs 21,4%) vem apenas da proporção entre blocos e embeddings ser ligeiramente diferente nas duas escalas — não de assimetria na política.

O gradiente continua fluindo **através** da cabeça congelada para os blocos superiores — isso é esperado e não impede o treinamento. Para a dissertação, a decisão se documenta como escolha de desenho: embeddings e cabeça como âncora do conhecimento pré-treinado.

#### Gradient checkpointing com camadas iniciais congeladas

Com as primeiras camadas congeladas, a entrada do primeiro bloco *checkpointed* não exige gradiente e o backward é cortado. A solução é `model.enable_input_require_grads()` — o mesmo mecanismo que o PEFT usa ao treinar adaptadores sobre base congelada.

**Guarda de idempotência:** a implementação do HF registra um forward hook nos embeddings e sobrescreve `_require_grads_hook` **sem remover o anterior**. Como o método roda uma vez por etapa (4× nos protocolos de unfreeze) sobre um modelo que permanece em memória, sem a guarda os hooks se acumulariam. O código só chama o método se ainda não houver hook registrado.

### Restrições e validações

1. **Só com `tipo: "full"`** — em etapas LoRA a capacidade já é controlada pelo rank; a tentativa levanta `ValueError` na carga do YAML (falha antes de alocar GPU).
2. **Incompatível com LoRA aplicado em memória** — `_aplicar_unfreeze_parcial()` levanta `ValueError` se `_lora_applied` for `True`. Protocolos de unfreeze devem ser pipelines 100% `full`.
3. **Percentual fora de 0–100%** ou valor não-numérico → `ValueError` com mensagem orientativa no parse.
4. **`from_layer >= num_hidden_layers`** (ex.: `100%`) → warning explícito; nenhum bloco treinável, apenas a norm final.
5. **`"75"` sem `%`** → interpretado como bloco absoluto 75, caindo no warning do item 4. Comportamento definido, não silencioso.

### Leitura dos logs

O resumo do curriculum mostra o corte **como declarado no YAML** (o percentual ainda não foi resolvido nesse ponto):

```
📋 Curriculum: 4 etapa(s) configurada(s)
   [0] alias='fácil-uf75pct', tipo=full, epochs=2, unfreeze_from=75%, arquivo=divisao_....csv
   [1] alias='médio-uf50pct', tipo=full, epochs=2, unfreeze_from=50%, arquivo=divisao_....csv
```

**Decisão de nomenclatura:** o rótulo é `unfreeze_from=`, não `unfreeze=`. Sem o `from`, "unfreeze=75%" se lê naturalmente como "75% descongelado" — o **inverso** do significado real (75% congelado, 25% final treinável). O `from` transforma o número numa *posição* na profundidade do modelo, e de quebra espelha o nome da chave YAML. O formato absoluto usa o mesmo prefixo: `unfreeze_from>=21`.

No início de cada etapa, o log resolve o corte e mostra o efeito concreto:

```
🧊 Unfreeze 75% congelado (7 camadas finais treináveis ≈ 25%): blocos 21-27 de 28 (etapa 'fácil-uf75pct')
ℹ️  Âncora de conhecimento pré-treinado: tensor único tied embed_tokens/lm_head permanece(m) CONGELADO(S) até unfreeze_layers_from: 0.
🧊 Unfreeze parcial (etapa 'fácil-uf75pct'): blocos >= 21 treináveis (327.586.304 params) | blocos congelados: 982.752.768 | embeddings+cabeça congelados: 233.373.696 | norm (sempre treinável): 1.536
🔓 Modo FULL: 327.586.304/1.543.714.304 parâmetros desbloqueados para treinamento
```

O contador `Modo FULL: X/Y` deve **crescer etapa a etapa**. Quando o percentual não divide exato (ex.: 30% de 28 = 8,4 → bloco 8), o log registra o índice efetivo, deixando o arredondamento documentado nos artefatos.

### Validação recomendada antes de submeter uma fila longa

Rodar `--datasets` (dry-run) confirma o parse e a listagem das etapas sem alocar GPU. No início da etapa 1, verificar nos primeiros ~50 steps que o loss desce e que não há warning de gradiente nulo — barato e conclusivo.

### Exemplo: currículo alinhado à capacidade

```yaml
curriculum:
  divisao:
  - dataset_filtro: {"dificuldade": "facil"}
    alias: "fácil-uf75pct"
    tipo: "full"
    unfreeze_layers_from: 75%  # 75% congelado = bloco 21 (28 blocos): treina 21-27 (25% finais)
    pace_epochs: 2
    learning_rate: 5e-06
  - dataset_filtro: {"dificuldade": "medio"}
    alias: "médio-uf50pct"
    tipo: "full"
    unfreeze_layers_from: 50%  # 50% congelado = bloco 14 (28 blocos): treina 14-27 (50% finais)
    pace_epochs: 2
    learning_rate: 5e-06
  - alias: "completo-uf0pct"
    tipo: "full"
    unfreeze_layers_from: 0%   # 0% congelado = tudo treinável, inclui embeddings/lm_head
    pace_epochs: 2
    learning_rate: 3e-06       # LR menor na capacidade plena (params recém-liberados
                               # entram com momentos Adam zerados)

treinamento:
  nbits: 16                    # Full FT: sempre 16 bits em todas as etapas
  max_grad_norm: 0.3           # Aperta o clip: estabiliza os primeiros steps após cada
                               # descongelamento (momentos zerados + sequências longas)
  warmup_steps: 100            # Rampa após cada reset de otimizador
```

Dados e capacidade progridem **juntos**: pouca capacidade para conteúdo simples, capacidade plena para o conjunto completo. O comentário ao lado de cada percentual registra o índice absoluto resolvido, para leitura direta sem precisar consultar a arquitetura.

### Arquivos Chave

- `treinar_unsloth_pipeline.py` → `EtapaCurriculum` (campos `unfreeze_layers_from` / `unfreeze_layers_pct` / `warmup_steps`), `construir_etapas()` (parse dual, whitelist de chaves, validação de tipo), `_CHAVES_ETAPA_VALIDAS`.
- `treinar_unsloth.py` → `_aplicar_unfreeze_parcial()` (resolução do percentual + política de congelamento), `_aplicar_etapa_curriculum()` (integração no ramo `full`, override de `warmup_steps`), `train()` (`global_defaults` com anti-vazamento de overrides entre etapas).

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

**Fallback:** Se o modelo não suportar Flash Attention 2 (ex: arquitetura incompatível) ou o pacote não estiver instalado, o fallback padrão é para `attn_implementation="eager"` (atenção padrão do PyTorch). **Nota:** O kernel SDPA fused do PyTorch (fallback anterior) causa overflow e NaN loss no step 0 quando combinado com LoRA em bfloat16 em GPUs Hopper (H100). Por isso o fallback é `"eager"`, que é numericamente estável em todos os cenários. 
Se você estiver rodando **Full FT** e precisar contornar a limitação de VRAM sem poder instalar o `flash-attn`, você pode adicionar `full_com_sdpa: true` na chave `treinamento` do YAML. Isso forçará o fallback para SDPA, assumindo que para Full FT o SDPA não apresentará os erros de NaN vistos no LoRA. Como camada extra de segurança, mesmo com a flag ativa, o sistema verificará o currículo: se houver **qualquer etapa LoRA**, o SDPA será bloqueado e o `eager` será mantido para evitar falha catastrófica.

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

**Decisão inicial (revertida):** passar `use_liger_kernel=treino_cfg.liger_kernel and _LIGER_DISPONIVEL` ao `SFTConfig`, para que o TRL pulasse nativamente os dois blocos.

**Problema descoberto — double patching:** `use_liger_kernel=True` no `SFTConfig` faz o **TRL re-aplicar** os patches do Liger por conta própria, incluindo a **fused cross-entropy**. Quando o model loader já carregou o modelo com `AutoLigerKernelForCausalLM` tendo *propositalmente desativado* a fused CE (o que ele faz sempre que `attn_implementation != "flash_attention_2"`, para evitar NaN com SDPA), o TRL a reabilita por cima — e o loss vira NaN imediatamente.

**Decisão atual:** `use_liger_kernel=False` fixo no `SFTConfig` ([treinar_unsloth.py:1880](treinar_unsloth.py#L1880)). O Liger é aplicado **num único lugar** — o model loader, via `AutoLigerKernelForCausalLM` — e o TRL não participa dessa decisão.

**Consequência a conhecer:** com `use_liger_kernel=False`, os blocos de entropia e token accuracy do TRL **sempre executam**. Se a fused CE estiver ativa, `outputs.logits` é `None` e o monkey-patch abaixo cobre o caso. Se a fused CE estiver **desativada** (sem `flash-attn` real), os logits são materializados e esses blocos passam a operar sobre um tensor grande — ver [Custo de VRAM quando a fused CE está desativada](#custo-de-vram-quando-a-fused-cross-entropy-está-desativada).

**Rede de segurança:** Um monkey-patch de `entropy_from_logits` permanece ativo no nível do módulo (`treinar_unsloth.py`, linhas 108-130), retornando `torch.tensor(0.0)` quando `logits is None`. Este patch é seguro mesmo sem Liger — é um passthrough transparente (único `is None` check, overhead de nanossegundos). Com `use_liger_kernel=False` no `SFTConfig`, ele deixou de ser apenas uma proteção defensiva e passou a ser o mecanismo **efetivo** que evita o crash quando a fused CE está ativa.

---

### Custo de VRAM quando a fused cross-entropy está desativada

O loader desativa `cross_entropy` e `fused_linear_cross_entropy` do Liger sempre que `attn_implementation != "flash_attention_2"` ([treinar_model_loader.py:179-185](treinar_model_loader.py#L179-L185)) — ou seja, **em qualquer máquina sem o pacote `flash-attn` instalado**, mesmo com `flash_attention_2: true` no YAML (o YAML expressa intenção; o fallback é `eager`/SDPA). O aviso aparece no log:

```
⚠️  Liger Kernel: desativando cross_entropy pois flash_attention_2 está desativado (evita NaN no eval loss com SDPA)
```

Sem a fused CE, `outputs.logits` é materializado inteiro, e o tensor escala com o **vocabulário**:

```
logits = batch × seq_len × vocab_size
```

No Qwen 2.5 (`vocab_size: 151936`), em bf16 **e** no upcast fp32 que a cross-entropy e a entropia fazem:

| batch × seq | logits bf16 | + upcast fp32 | pico da cadeia |
|---|---|---|---|
| 1 × 1024 | 0,29 GiB | 0,58 GiB | **0,87 GiB** |
| 1 × 2048 | 0,58 GiB | 1,16 GiB | **1,74 GiB** |
| 2 × 8192 | 4,64 GiB | 9,27 GiB | **13,91 GiB** |

Para comparação, o custo **estático** de um Full FT do Qwen 1.5B (pesos bf16 2,88 + grads 2,88 + `adamw_8bit` 2,88) é 8,63 GiB. A última linha da tabela sozinha já estoura qualquer GPU de consumo.

**Sintoma sob WSL2 — não é um OOM limpo:** quando a VRAM se esgota, o driver WDDM pagina para a RAM do host via PCIe em vez de falhar. O treino fica ordens de magnitude mais lento (ex.: 37 s/it) e eventualmente aborta com:

```
torch.AcceleratorError: CUDA error: unknown error
  ...
  File ".../trl/trainer/sft_trainer.py", line 1297, in compute_loss
    entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
```

O `.item()` é apenas o ponto de **sincronização** onde o erro assíncrono aflora — não a linha culpada. Diagnóstico por eliminação:

| Sintoma | Causa provável |
|---|---|
| `CUDA error: device-side assert triggered` + `indexSelectLargeIndex` | Índice fora do vocabulário (OOV real) |
| `torch.OutOfMemoryError: CUDA out of memory` | Falta de VRAM, reportada limpa (típico em Linux nativo) |
| `CUDA error: unknown error` + s/it muito alto | Exaustão de VRAM sob WSL2/WDDM (paginação PCIe) |

**Mitigações, em ordem de eficácia:**
1. Reduzir `max_seq_length` — ataca a causa, pois o tensor é linear em `seq_len`.
2. Reduzir `batch_size` por GPU e compensar com `grad_batch_size` (mantém o batch efetivo).
3. Instalar `flash-attn`, que reativa a fused CE e elimina a materialização dos logits.

O YAML `04_treinar_d_mini_uf.yaml` (pubmed-experimento) documenta esse orçamento no cabeçalho e serve de molde para rodadas locais em GPUs de 12 GB.

#### Decisão 3: Defaults e Validação

**Defaults no YAML:** Ambas as otimizações são `true` por padrão no template YAML gerado automaticamente. A lógica é que se o ambiente suporta, devem estar ativas — a economia de VRAM é significativa sem impacto na qualidade do treinamento.

**Validação fail-fast:** Se `flash_attention_2: true` no YAML mas o pacote não está instalado, ou `liger_kernel: true` mas `liger-kernel` não está no ambiente, o treinamento **não inicia** e exibe `RuntimeError` com instruções de `pip install` ou como desativar no YAML. Isso evita que o pesquisador descubra o problema após horas de preprocessamento de dados.

---

### Resumo das Dependências Opcionais

| Otimização | Pacote | Instalação | Flag YAML | Detecção |
|---|---|---|---|---|
| Flash Attention 2 | `flash-attn` | `pip install flash-attn --no-build-isolation` | `flash_attention_2: true` | `is_flash_attn_2_available()` |
| Liger Kernel | `liger-kernel` | `pip install liger-kernel` | `liger_kernel: true` | `import AutoLigerKernelForCausalLM` |