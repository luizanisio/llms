# Visão Geral dos Experimentos e Protocolos (Curriculum Learning + Progressive Training)

> 🛠️ **Para instruções técnicas de setup de ambiente, instalação (CUDA, Flash-Attn, vLLM) e execução em background, consulte o [README.md](./README.md).**

Este documento centraliza a arquitetura dos experimentos: os objetivos de cada frente, o mapa completo dos protocolos (`b`, `c`, `d1`–`d25`) e quais relatórios de comparação respondem a cada pergunta de pesquisa.

---

## 1. Resumo dos Experimentos

### ⚖️ Summa-Experimento *(experimento principal)*
- **Objetivo:** Extração de metadados jurídicos de acórdãos judiciais.
- **Modelo Base:** `Qwen2.5-7B-Instruct` (28 blocos) — professor: `Qwen3-235B`
- **Ambiente de Treino:** Cluster Slurm (H100 80GB) e H200 em treinos FF
- **Detalhes Técnicos:** Textos muito extensos, exigindo contexto longo (`max_seq_length: 32768`), processados com lotes cuidadosos (`batch_size: 1`, `grad_batch_size: 16`).
- **Protocolos disponíveis:** `b`, `c`, `d1`–`d25`

### 🩺 Pubmed-Experimento *(generalização do framework)*
- **Objetivo:** Extração e classificação de partes (Background, Methods, Results, Conclusions) em abstracts médicos (RCTs) sobre o PubMed 20k.
- **Modelo Base:** `Qwen2.5-1.5B-Instruct` (28 blocos)
- **Ambiente de Treino:** Cluster Slurm (H100 80GB)
- **Detalhes Técnicos:** Menor comprimento de contexto (`max_seq_length: 8192`), mas grande volume de amostras (`batch_size: 2`, `grad_batch_size: 8`).
- **Protocolos disponíveis:** `b`, `b16`, `b16r8`, `c`, `d1`–`d25`

### 🔬 Puil-Mini-Experimento
- **Objetivo:** Validação local rápida da máquina de estados do código-fonte (transições entre QLoRA e Full Fine-tuning, recarregamentos de modelo, modo fusão, scripts utilitários).
- **Modelo Base:** `Qwen2.5-1.5B-Instruct`
- **Ambiente de Treino:** Local via WSL em GPU comercial (`RTX 3060 12GB` / `RTX 2060`).
- **Detalhes Técnicos:** Pipeline simplificado com contexto mínimo e batch unitário, para debugar os *engineers* sem gastar horas do cluster.

---

## 2. Os Dois Eixos do Framework

Todos os protocolos combinam (ou isolam) dois eixos independentes:

**Eixo CL — progressão dos DADOS.** As instâncias são ordenadas por `dificuldade_int` e apresentadas em etapas. Variantes: *por etapas disjuntas* (fácil → médio → difícil → completo), *acumulado* com replay (fácil → fácil+médio → tudo) e *granular* (incremento unitário, `≤1 → ≤2 → … → ≤9 → tudo`).

**Eixo PT — progressão da CAPACIDADE.** Três instanciações distintas, e a diferença entre elas é o que a matriz de protocolos mede:

| Instanciação | Como a capacidade cresce | Custo de transição | Protocolos |
| :-- | :-- | :-- | :-- |
| **Troca de regime** | LoRA ↔ Full FT entre etapas | Alto: merge do adaptador, **recarga do modelo** e requantização | d1–d6, d11, d12, d14, d15 |
| **Descongelamento** | Mais blocos ficam treináveis a cada etapa (full 16b) | Nulo em pesos: a transição preserva a função — só muda o que recebe gradiente | d19, d20 |
| **Gating de LR** | Grupos de camadas entram no otimizador com LR=0 e "acordam" em rampa | Nulo: otimizador nunca reseta; grupos adormecidos já acumulam momentos Adam | d22, d23, d24, d25 |

> **Mecanismo relevante para a Q6:** no pipeline segmentado, `nbits` alvo é 16 sempre que a etapa é `full`; quando o valor difere do que está em memória, o modelo é **descarregado e recarregado** ([`src/treinar_unsloth.py`](../src/treinar_unsloth.py), `alvo_nbits`). Toda etapa também reinicia otimizador Adam e scheduler cosine. O **modo fusão** (d21–d25) elimina os dois: as etapas viram *spans* de um dataset concatenado em um único `trainer.train()`, com um cosine só, e as fronteiras passam a ser marcadores virtuais de log.

---

## 3. Mapa Completo dos Protocolos

Legenda de fronteiras: **real** = nova execução de trainer (reset de otimizador/scheduler, possível recarga do modelo) · **virtual** = span dentro de um único `trainer.train()`.

### Baselines

| ID | Papel | Configuração | Disp. |
| :-- | :-- | :-- | :-- |
| **A** | Referência | Zero-shot, sem ajuste fino | ambos |
| **b** | Baseline FT | LoRA **4 bits**, r=16, direto, dados completos, execução única | ambos |
| **b16** | Controle de `b` | LoRA **16 bits**, r=16 — isola o erro de quantização NF4 | pubmed |
| **b16r8** | Controle de `b` | LoRA **16 bits**, r=8 — isola o posto do adaptador | pubmed |
| **c** | Baseline FT | Full FT 16 bits direto, dados completos, execução única | ambos |

> Os controles `b16`/`b16r8` só existem no pubmed: são específicos para calibrar o baseline `b`, que é a referência de comparação da maioria dos protocolos. `b16` é também a **execução única em LoRA 16b r=16** — a referência que faltava para ler `d16` e `d17` sem confundir fronteira/ordenação com precisão.

### CL + escalonamento por troca de regime (fronteiras reais)

| ID | Etapas | O que testa | Disp. |
| :-- | :-- | :-- | :-- |
| **d1** | FF-fácil → LoRA-médio → LoRA-difícil → LoRA-completo | CL por etapas com FF precoce (capacidade decrescente) | ambos |
| **d2** | LoRA-fácil → LoRA-médio → LoRA-difícil → FF-completo | CL por etapas com FF tardio (consolidação final) | ambos |
| **d3** | FF-fácil → LoRA-(fácil+médio) → LoRA-tudo | Variante **acumulada** do d1 (replay elimina a etapa de consolidação) | ambos |
| **d4** | LoRA-fácil → LoRA-(fácil+médio) → FF-tudo | Variante **acumulada** do d2 | ambos |
| **d11** | FF(≤1) → LoRA(≤2) → … → LoRA(≤9) → LoRA-tudo | d3 com **pace granular**: 10 etapas, incremento unitário | ambos |
| **d12** | LoRA(≤1) → LoRA(≤2) → … → LoRA(≤9) → FF-tudo | d4 com pace granular | ambos |

### Ablações — isolam um eixo de cada vez (fronteiras reais)

| ID | Etapas | O que isola | Disp. |
| :-- | :-- | :-- | :-- |
| **d5** | FF-completo → LoRA-completo | Escalonamento FF→LoRA **sem CL** | ambos |
| **d6** | LoRA-completo → FF-completo | Escalonamento LoRA→FF **sem CL** | ambos |
| **d7** | LoRA fácil → médio → difícil → completo (4b) | **CL puro**, sem escalonamento | ambos |
| **d8** | LoRA fácil → (fácil+médio) → tudo (4b) | CL puro acumulado | ambos |
| **d13** | FF fácil → médio → difícil → completo (16b) | Contraparte FF-only do d7 | ambos |
| **d17** | LoRA fácil → médio → difícil → completo (**16b**) | d7 **sem a variável de quantização NF4** | ambos |

### Direção do currículo (controle negativo)

| ID | Etapas | O que testa | Disp. |
| :-- | :-- | :-- | :-- |
| **d9** | LoRA completo → difícil → médio → fácil | **Anti-CL** por etapas (espelho reverso do d7) | ambos |
| **d10** | LoRA (>7) → (>3) → tudo | Anti-CL acumulado (espelho reverso do d8) | ambos |
| **d18** | LoRA b1 → b2 → b3 → completo (16b) | **Terços aleatórios**: separa "ordenar por dificuldade" de "treinar em blocos com recência" | ambos |

### PT como pré-treino / estabilização

| ID | Etapas | O que testa | Disp. |
| :-- | :-- | :-- | :-- |
| **d14** | LoRA-completo → FF fácil → médio → difícil → completo | Warm-up LoRA no dataset inteiro **antes** do CL FF-only (d13) | ambos |
| **d15** | d14 + etapa de estabilização FF (LR 1e-6) após o merge | A estabilização suaviza o *spike* de loss da transição LoRA→FF? | ambos |

### Controles de fronteira e de capacidade (16 bits, orçamento 4N)

| ID | Etapas | O que testa | Disp. |
| :-- | :-- | :-- | :-- |
| **d16** | 4 × LoRA-completo (16b), 1 época cada | **Custo puro de fronteira**: mesma configuração de `b16`, só que segmentada em 4 execuções | ambos |
| **d19** | FF fácil(75% congelado) → médio(50%) → difícil(25%) → completo(0%) | **CL + PT por descongelamento** — sinergia (tese central) | ambos |
| **d20** | FF completo(75%) → completo(50%) → completo(25%) → completo(0%) | Descongelamento **sem CL** (controle do d19) | ambos |

### Protocolos fundidos — fronteiras virtuais

| ID | Spans | O que testa | Disp. |
| :-- | :-- | :-- | :-- |
| **d21** | fácil ×2 → médio ×2 → difícil ×2 → completo ×2 (LoRA 16b) | **Espelho exato do d17 sem fronteiras reais** — a medida mais limpa de custo de fronteira | ambos |
| **d22** | fácil(uf 75%) ×2 → médio(50%) ×2 → difícil(25%) ×2 → completo(0%) ×2 | CL + PT por **gating de LR**, otimizador contínuo | ambos |
| **d23** | completo(uf 75%) → (50%) → (25%) → (0%), ×1 cada | Gating **sem CL** (controle do d22) | ambos |
| **d24** | ≤1(uf 90%) → ≤2(80%) → … → tudo(0%), LoRA **4b** | Espelho fundido do **d12**: CL granular sem fronteiras, gating em passos de 10% | ambos |
| **d25** | idem d24, em **Full FT 16b** | Mesma topologia do d24 em regime full — isola "regime" dado CL+PT granular fundido | ambos |

> **Gating ≠ congelamento.** No d19/d20 os blocos congelados não entram no otimizador (economia real de VRAM). No gating (d22–d25) todos os grupos estão no otimizador desde o step 0, com LR 0 até acordarem — não há economia de memória. Isso pesa no **d25 do summa**: full FT 16 bits no 7B com contexto 32768 mantém pesos, gradientes e estados Adam de todos os parâmetros simultaneamente. Preferir H200 e validar o pico de VRAM nos primeiros steps.

---

## 4. Perguntas de Pesquisa → Protocolos → Relatório

| # | Pergunta | Protocolos | Relatório |
| :-- | :-- | :-- | :-- |
| **Q1** | O ajuste fino produz ganho sobre o zero-shot? | todos vs **A** | `06_compara_experimentais` |
| **Q2** | A progressão de dificuldade melhora sobre o FT direto? | b, c, d1–d4, d13 (+d7, d8, d17) | `06_compara_experimentais`, `06_compara_ablacoes` |
| **Q3** | FF→LoRA e LoRA→FF produzem desempenhos distintos? | d1–d4, d5, d6 | `06_compara_ordem_pt` |
| **Q4** | A ordem fácil→difícil importa vs difícil→fácil? | d7–d10 (+d17, d18) | `06_compara_ordem_cl` |
| **Q5** | O pace suave (unitário) melhora sobre a progressão rápida? | d11, d12 (vs d3, d4) | `06_compara_experimentais` |
| **Q6** | Há perda de eficiência ao transitar entre etapas LoRA/Full e durante a progressão de dificuldade? | d5, d6, d12, d16–d18, d21, d24, d25 | `06_compara_fronteiras` |
| **Q7** | *(proposta)* A capacidade pode escalar **sem** troca de regime, e isso sinergia com o CL? | d13, d17, d19–d25 | `06_compara_capacidade` |
| — | *(controle transversal)* Quanto do baseline `b` é limitado por quantização e posto? | b, b16, b16r8 | `06_compara_controle_b` *(pubmed)* |

> **Nota de numeração:** alguns cabeçalhos de `04_treinar_*.yaml` ainda usam Q6/Q7/Q8 com outro sentido (warm-up LoRA no d14, estabilização no d15, regime full vs LoRA no d24/d25). Os `06_compara_*.yaml` seguem a numeração desta tabela.

---

## 5. Relatórios de Comparação

Cada `06_compara_*.yaml` é uma **visão** — protocolos aparecem em mais de um relatório de propósito.

| Relatório | Modelos | Serve a |
| :-- | :-- | :-- |
| `06_compara_experimentais` | A, b, c, d1–d4, d11, d12 | Q1, Q2, Q3, Q5 (experimento principal, segmentado) |
| `06_compara_ablacoes` | A, b, c, d5–d8, d13, d14, d15, d17 | Isolamento de cada componente |
| `06_compara_ordem_cl` | A, b, d7–d10, d17, d18 | Q4 (crescente × decrescente × aleatório) |
| `06_compara_ordem_pt` | A, b, c, d1–d6 | Q3 (direção, com e sem CL) |
| `06_compara_fronteiras` | A, b, c, d5, d6, d12, d16–d18, d21, d24, d25 *(+b16 no pubmed)* | Q6 |
| `06_compara_capacidade` | A, b, c, d13, d17, d19–d25 | Q7 (unfreeze e gating) |
| `06_compara_controle_b` | b, b16, b16r8 — **só pubmed** | Calibração do baseline (quantização e posto) |
| `06_compara_todos` | tudo | Panorama, ranking, Friedman/Nemenyi global |

Execução em lote no cluster: `sbatch job_compara_testes.sh` (roda os relatórios em sequência).

---

## 6. Cruzamentos Essenciais

**O currículo funciona?**
`d17` × `b` (CL puro em 16b vs direto) — e o controle honesto é `d17` × `d18`: se a ordenação não bate os terços aleatórios, o ganho era efeito de bloco, não de currículo.

**A ordem importa?**
`d7` × `d9` e `d8` × `d10` (espelhos reversos). Comparar sempre dentro da mesma precisão.

**Vale consolidar no final?**
`d7` × `d8` — etapas disjuntas com consolidação final vs acumulado com replay.

**O escalonamento por troca de regime compensa o custo?**
`d1`/`d2` × `d7` (com CL) e `d5`/`d6` × `b`/`c` (sem CL). A posição do FF muda o desfecho: `d1` × `d2`, confirmado contra `d5` × `d6`.

**Quanto custa uma fronteira?**
`d21` × `d17` é a leitura mais limpa: mesma progressão de dados, mesma precisão, mesmo orçamento — só a fronteira muda. `d24` × `d12` repete a leitura na granularidade 10. Para a fronteira *sem* currículo, o par é `d16` × `b16` no pubmed (mesma precisão e posto, só segmentação); no summa, onde `b16` não existe, `d16` × `b` carrega também a quantização.

**Há sinergia CL+PT?**
Dois 2×2 independentes. Descongelamento: `d19` (CL+PT) × `d20` (PT) × `d13` (CL) × `c`. Gating: `d22` (CL+PT) × `d23` (PT) × `d21` (CL) × `b`. O padrão `d19 > d20 > c` e `d22 > d23 > b` sustenta a sinergia; `d19 ≈ d20` indicaria que o ganho é do escalonamento e o CL é passageiro.

---

## 7. Notas de Comparabilidade

- **Precisão.** `b` e d1–d15 rodam LoRA em 4 bits (NF4); d16–d23 rodam em 16 bits; d24 volta a 4 bits de propósito (espelho do d12) e d25 é full 16 bits. Cruzamentos entre grupos carregam **{efeito estudado + quantização}** como diferença conjunta — por isso existem `d17` (d7 em 16b), `d16` (b segmentado em 16b) e, no pubmed, `b16` (b em 16b, execução única). O contraste `b` × `b16` dá o fator de correção para ler os cruzamentos 4b×16b restantes. Etapas `full` sempre executam em 16 bits, independentemente do `nbits` global.
- **Orçamento.** d16–d25 são calibrados em **4N instâncias** (mesmo total de `b` com 4 épocas), o que torna suas comparações pareadas. Protocolos anteriores não seguem essa paridade estrita.
- **Posição das ativações no d22 × d23.** O d23 usa 1 época por span (dataset completo) e o d22 usa 2 (fatias), para equalizar o orçamento — os grupos de camadas acordam em pontos diferentes do stream. Registrar como limitação.
- **Evidência de eficiência.** As comparações leem `training_metrics.jsonl` via `pasta_treinamento`: o *spike* de loss em cada fronteira dos segmentados e sua ausência nos fundidos é a evidência primária da Q6, não apenas o score final.
