# Desenho dos Experimentos e Protocolos (Curriculum Learning + Progressive Training)

Este documento descreve **o que** os experimentos testam: o papel de cada experimento, os dois
eixos do framework, o mapa dos protocolos (`b`, `c`, `d1`–`d25`), quais recortes de comparação
respondem a cada pergunta de pesquisa e o que limita cada contraste.

> **Como executar:** [README_PASSO_A_PASSO.md](./README_PASSO_A_PASSO.md) ·
> **Números, implementação e estado de cada protocolo:** [INFORMACOES_TREINAMENTO.md](./INFORMACOES_TREINAMENTO.md) ·
> **Hiperparâmetros por protocolo (gerado dos YAMLs):** [RESUMO_EXPERIMENTOS.md](./RESUMO_EXPERIMENTOS.md)

---

## 1. Os Experimentos

| Experimento | Tarefa | Aluno | Referência | Papel na pesquisa |
| :-- | :-- | :-- | :-- | :-- |
| ⚖️ **Summa** | Extração de metadados jurídicos de acórdãos do STJ (textos longos, contexto 32k) | Qwen2.5-7B-Instruct | Professor Qwen3-235B-A22B-2507 (destilação) | Experimento principal; único com conjunto out-of-time (fold 12, 2026) e juiz LLM |
| 🩺 **PubMed** | Extração das seções (Background, Objective, Methods, Results, Conclusions) e metadados de abstracts do PubMed 20k RCT | Qwen2.5-1.5B-Instruct | Gold do dataset | Generalização para outro domínio, idioma e porte de modelo |
| 🏥 **SemClinBr** | NER + RE clínico em notas pt-br (Oliveira et al., 2022) | Qwen2.5-7B-Instruct | Gold humano | Generalização com corpus pequeno (697 docs de treino) e comparação externa (CRF, BioBERTpt) |
| 🔬 **Puil-Mini** | Extração simples (contexto 256 tokens) | Qwen2.5-1.5B-Instruct | — | Validação local da máquina de estados do código (troca QLoRA↔Full, recargas, fusão) em GPU de 12 GB |
| 📝 **Summa-Qualifica** | Versão preliminar do Summa | — | — | Referência histórica da qualificação |

Summa, PubMed e SemClinBr executam a **mesma malha de 31 protocolos treinados** (`b`, `b16`,
`b16r8`, `c`, `d1`–`d25`, `d1a`, `d1b`) mais o zero-shot `A`. Os detalhes de cada um ficam no
README da pasta do experimento.

---

## 2. Os Dois Eixos do Framework

Todos os protocolos combinam (ou isolam) dois eixos independentes:

**Eixo CL — progressão dos DADOS.** As instâncias são ordenadas por `dificuldade_int` (1–10) e
apresentadas em etapas. Variantes: *por etapas disjuntas* (fácil → médio → difícil → completo),
*acumulado* com replay (fácil → fácil+médio → tudo) e *granular* (incremento unitário,
`≤1 → ≤2 → … → ≤9 → tudo`).

**Eixo PT — progressão da CAPACIDADE.** Três instanciações distintas, e a diferença entre elas é o
que a matriz de protocolos mede:

| Instanciação | Como a capacidade cresce | Custo de transição | Protocolos |
| :-- | :-- | :-- | :-- |
| **Troca de regime** | LoRA ↔ Full FT entre etapas | Alto: merge do adaptador, **recarga do modelo** e requantização | d1–d6, d11, d12, d14, d15 |
| **Descongelamento** | Mais blocos ficam treináveis a cada etapa (full 16b) | Nulo em pesos: a transição preserva a função — só muda o que recebe gradiente | d19, d20 |
| **Gating de LR** | Grupos de camadas entram no otimizador com LR=0 e "acordam" em rampa | Nulo: otimizador nunca reseta; grupos adormecidos já acumulam momentos Adam | d22, d23, d24, d25 |

**Fronteiras.** Em uma fronteira **real** começa uma nova execução de trainer: Adam e scheduler
cosine reiniciam e, se a precisão muda, o modelo é recarregado. No **modo fusão** (d21–d25) as
etapas viram *spans* de um único `trainer.train()`, com um cosine só, e as fronteiras passam a ser
**virtuais** (marcadores de log). Essa diferença é o objeto da Q6. A mecânica de recarga, merge e
quantização está em [INFORMACOES_TREINAMENTO.md §2](./INFORMACOES_TREINAMENTO.md#2-recarga-merge-e-quantização-nos-pipelines-com-troca-de-regime).

---

## 3. Mapa Completo dos Protocolos

Todos os protocolos estão disponíveis nos três experimentos (Summa, PubMed, SemClinBr).

### Baselines

| ID | Papel | Configuração |
| :-- | :-- | :-- |
| **A** | Referência | Zero-shot, sem ajuste fino |
| **b** | Baseline FT | LoRA **4 bits**, r=16, direto, dados completos, execução única |
| **b16** | Controle de `b` | LoRA **16 bits**, r=16 — isola o erro de quantização NF4 |
| **b16r8** | Controle de `b` | LoRA **16 bits**, r=8 — isola o posto do adaptador |
| **c** | Baseline FT | Full FT 16 bits direto, dados completos, execução única |

> `b16` entra nos recortes `Q2a_cl_controlado` e `Q6a_fusao`: é a **execução única em LoRA 16b
> r=16** que permite ler `d16` e `d17` sem confundir fronteira/ordenação com precisão. `b16r8`
> (posto 8) é auxiliar e entra só no `Panorama_Geral`.

### CL + escalonamento por troca de regime (fronteiras reais)

| ID | Etapas | O que testa |
| :-- | :-- | :-- |
| **d1** | FF-fácil → LoRA-médio → LoRA-difícil → LoRA-completo | CL por etapas com FF precoce (capacidade decrescente) |
| **d2** | LoRA-fácil → LoRA-médio → LoRA-difícil → FF-completo | CL por etapas com FF tardio (consolidação final) |
| **d3** | FF-fácil → LoRA-(fácil+médio) → LoRA-tudo | Variante **acumulada** do d1 (replay elimina a etapa de consolidação) |
| **d4** | LoRA-fácil → LoRA-(fácil+médio) → FF-tudo | Variante **acumulada** do d2 |
| **d11** | FF(≤1) → LoRA(≤2) → … → LoRA(≤9) → LoRA-tudo | d3 com **pace granular**: 10 etapas, incremento unitário |
| **d12** | LoRA(≤1) → LoRA(≤2) → … → LoRA(≤9) → FF-tudo | d4 com pace granular |

### Ablações — isolam um eixo de cada vez (fronteiras reais)

| ID | Etapas | O que isola |
| :-- | :-- | :-- |
| **d5** | FF-completo → LoRA-completo | Escalonamento FF→LoRA **sem CL** |
| **d6** | LoRA-completo → FF-completo | Escalonamento LoRA→FF **sem CL** |
| **d7** | LoRA fácil → médio → difícil → completo (4b) | **CL puro**, sem escalonamento |
| **d8** | LoRA fácil → (fácil+médio) → tudo (4b) | CL puro acumulado |
| **d13** | FF fácil → médio → difícil → completo (16b) | Contraparte FF-only do d7 |
| **d17** | LoRA fácil → médio → difícil → completo (**16b**) | d7 **sem a variável de quantização NF4** |

### Direção do currículo (controle negativo)

| ID | Etapas | O que testa |
| :-- | :-- | :-- |
| **d9** | LoRA completo → difícil → médio → fácil | **Anti-CL** por etapas (espelho reverso do d7) |
| **d10** | LoRA (>7) → (>3) → tudo | Anti-CL acumulado (espelho reverso do d8) |
| **d18** | LoRA b1 → b2 → b3 → completo (16b) | **Terços aleatórios**: separa "ordenar por dificuldade" de "treinar em blocos com recência" |

### PT como pré-treino / estabilização

| ID | Etapas | O que testa |
| :-- | :-- | :-- |
| **d14** | LoRA-completo → FF fácil → médio → difícil → completo | Warm-up LoRA no dataset inteiro **antes** do CL FF-only (d13) |
| **d15** | d14 + etapa de estabilização FF (LR 1e-6) após o merge | A estabilização suaviza o *spike* de loss da transição LoRA→FF? |

### Controles de fronteira e de capacidade (16 bits, orçamento 4N)

| ID | Etapas | O que testa |
| :-- | :-- | :-- |
| **d16** | 4 × LoRA-completo (16b), 1 época cada | **Custo puro de fronteira**: mesma configuração de `b16`, só que segmentada em 4 execuções |
| **d19** | FF fácil(75% congelado) → médio(50%) → difícil(25%) → completo(0%) | **CL + PT por descongelamento** — sinergia (tese central) |
| **d20** | FF completo(75%) → completo(50%) → completo(25%) → completo(0%) | Descongelamento **sem CL** (controle do d19) |

### Protocolos fundidos — fronteiras virtuais

| ID | Spans | O que testa |
| :-- | :-- | :-- |
| **d21** | fácil ×2 → médio ×2 → difícil ×2 → completo ×2 (LoRA 16b) | **Espelho exato do d17 sem fronteiras reais** — a medida mais limpa de custo de fronteira |
| **d22** | fácil(uf 75%) ×2 → médio(50%) ×2 → difícil(25%) ×2 → completo(0%) ×2 | CL + PT por **gating de LR**, otimizador contínuo |
| **d23** | completo(uf 75%) → (50%) → (25%) → (0%), ×1 cada | Gating **sem CL** (controle do d22) |
| **d24** | ≤1(uf 90%) → ≤2(80%) → … → tudo(0%), LoRA **4b** | Espelho fundido do **d12**: CL granular sem fronteiras, gating em passos de 10% |
| **d25** | idem d24, em **Full FT 16b** | Mesma topologia do d24 em regime full — isola "regime" dado CL+PT granular fundido |

> **Gating ≠ congelamento.** No d19/d20 os blocos congelados não entram no otimizador (economia
> real de VRAM). No gating (d22–d25) todos os grupos estão no otimizador desde o step 0, com LR 0
> até acordarem — não há economia de memória. Isso pesa no **d25 do summa**: full FT 16 bits no 7B
> com contexto 32768 mantém pesos, gradientes e estados Adam de todos os parâmetros
> simultaneamente (treinado em H200 NVL).

### Réplicas (calibração da ROPE)

| ID | Etapas | O que testa |
| :-- | :-- | :-- |
| **d1a** | Idêntico ao d1 | **Réplica 2** do d1 — mede variância não-determinística do treinamento |
| **d1b** | Idêntico ao d1 | **Réplica 3** do d1 — mesmos hiperparâmetros, saída distinta |

> A comparação `d1` × `d1a` × `d1b` (`06_compara_d1ab.yaml`) calibra a ROPE bayesiana e separa
> efeito real de ruído estocástico. No Summa, as mesmas réplicas calibram também a ROPE do juiz LLM.

---

## 4. Perguntas de Pesquisa → Protocolos → Recortes

As perguntas de pesquisa são respondidas por **recortes** (subconjuntos de modelos) dentro de um
único relatório consolidado (`06_compara_todos.yaml`, bloco `estatistica.protocolos`). No Summa, o
`07_avaliacao_llm.py --recorte` repete os mesmos recortes sobre as notas do juiz.

| Recorte | Pergunta | Protocolos |
| :-- | :-- | :-- |
| **Q1_ajuste_fino** | O ajuste fino produz ganho sobre o zero-shot? | A, B, C |
| **Q2a_cl_controlado** | O CL melhora sobre FT direto? (controles: fronteira, blocos aleatórios) | B16, D16, D18, D17 |
| **Q2b_cl_puro** | CL sem escalonamento, em cada regime | B, D7, D8, D13, D14 |
| **Q2c_granularidade** | Progressão granular (10 etapas) vs saltos (3 etapas) | D3, D4, D11, D12 |
| **Q3a_direcao_com_cl** | FF→LoRA vs LoRA→FF com CL | D1, D2, D3, D4 |
| **Q3b_direcao_sem_cl** | Direção do escalonamento sem CL (controle) | B, C, D5, D6 |
| **Q4a_decomposicao** | Decomposição 2×2: CL só, escalonamento só, ambos, nenhum | B, C, D1, D5, D7 |
| **Q4b_unfreeze** | Sinergia CL+PT por descongelamento progressivo | B, C, D20, D19 |
| **Q5_anti_curriculo** | Direção do currículo: CL vs anti-CL | B, D7, D8, D9, D10 |
| **Q6a_fusao** | Fundido vs segmentado (custo de fronteira) | B, C, B16, D17, D21, D22, D23 |
| **Q6b_fusao_granular** | Fusão granular: regime full vs LoRA | B, C, D12, D24, D25 |
| **Q6c_transicao_regime** | Pré-treino LoRA e estabilização pós-merge | B, C, D13, D14, D15 |
| **Panorama_Geral** | Visão global descritiva (precisões e orçamentos misturados) — nunca fonte de conclusão | TODOS |

> **Nota de numeração:** alguns cabeçalhos de `04_treinar_*.yaml` ainda usam Q6/Q7/Q8 com outro
> sentido (warm-up LoRA no d14, estabilização no d15, regime full vs LoRA no d24/d25). Vale a
> numeração desta tabela.

---

## 5. Relatórios de Comparação

Dois YAMLs por experimento:

| Arquivo | Conteúdo | Uso |
| :-- | :-- | :-- |
| `06_compara_d1ab.yaml` | D1, D1a, D1b (`calibracao_rope: true`) | Calibração da ROPE por campo (`estatisticas/00_rope_sugerido.md`) |
| `06_compara_todos.yaml` | A, B, B16, B16R8, C, D1–D25 e os recortes da seção 4 | Relatório completo |

Cada recorte gera seu próprio conjunto de figuras/tabelas, com o nome do recorte como prefixo dos
arquivos e uma seção própria no markdown final. A análise principal é **bayesiana**
(`baycomp.CorrelatedTTest`, ROPE calibrada por campo, limiar 0,95); a frequentista
(Friedman → Wilcoxon-Holm → Nemenyi) é exploratória.

Execução em lote no cluster: `sbatch job_compara_testes.sh`. O procedimento completo está no
[README_PASSO_A_PASSO.md, etapa 06](./README_PASSO_A_PASSO.md#06--comparação-dos-protocolos).

---

## 6. Cruzamentos Essenciais

**O currículo funciona?**
`d17` × `b16` (CL puro vs direto, ambos LoRA 16b) — e o controle honesto é `d17` × `d18`: se a
ordenação não bate os terços aleatórios, o ganho era efeito de bloco, não de currículo.

**A ordem importa?**
`d7` × `d9` e `d8` × `d10` (espelhos reversos). Comparar sempre dentro da mesma precisão.

**Vale consolidar no final?**
`d7` × `d8` — etapas disjuntas com consolidação final vs acumulado com replay.

**O escalonamento por troca de regime compensa o custo?**
`d1`/`d2` × `d7` (com CL) e `d5`/`d6` × `b`/`c` (sem CL). A posição do FF muda o desfecho:
`d1` × `d2`, confirmado contra `d5` × `d6`.

**Quanto custa uma fronteira?**
`d21` × `d17` é a leitura mais limpa: mesma progressão de dados, mesma precisão, mesmo orçamento —
só a fronteira muda. `d24` × `d12` repete a leitura na granularidade 10. Para a fronteira *sem*
currículo, o par é `d16` × `b16` (mesma precisão e posto, só segmentação).

**Há sinergia CL+PT?**
Dois 2×2 independentes. Descongelamento: `d19` (CL+PT) × `d20` (PT) × `d13` (CL) × `c`. Gating:
`d22` (CL+PT) × `d23` (PT) × `d21` (CL) × `b`. O padrão `d19 > d20 > c` e `d22 > d23 > b` sustenta
a sinergia; `d19 ≈ d20` indicaria que o ganho é do escalonamento e o CL é passageiro.

---

## 7. Notas de Comparabilidade

- **Precisão.** `b` e d1–d15 rodam LoRA em 4 bits (NF4); d16–d23 rodam em 16 bits; d24 volta a
  4 bits de propósito (espelho do d12) e d25 é full 16 bits. Etapas `full` sempre executam em 16
  bits. Cruzamentos entre grupos carregam **{efeito estudado + quantização}** como diferença
  conjunta — por isso existem `d17` (d7 em 16b), `d16` (b segmentado em 16b) e `b16` (b em 16b,
  execução única); o contraste `b` × `b16` dá o fator de correção para os cruzamentos 4b×16b
  restantes. Na inferência, Summa e SemClinBr requantizam todos os modelos em 4 bits
  ([INFORMACOES_TREINAMENTO.md §1](./INFORMACOES_TREINAMENTO.md#1-precisão-na-inferência-vllm)).
- **Orçamento.** Em dataset-equivalentes (fração × épocas), os protocolos caem em quatro faixas:
  **4N** (b, b16, b16r8, c, d1–d10, d13, d16–d23, d1a, d1b), **8N** (d14 — warm-up LoRA sobre o
  dataset inteiro antes do CL), **9N** (d15 — d14 mais a etapa de estabilização) e **11N** (d11,
  d12, d24, d25 — acúmulo granular). Contrastes dentro de uma faixa são limpos; entre faixas
  carregam o orçamento junto e são declarados, com custo por instâncias/tokens ao lado.
- **Exposição por item.** Com o mesmo orçamento 4N, o pace por etapas expõe cada item 4 vezes
  (2 na sua faixa + 2 no completo), e o acumulado expõe fáceis 6×, médios 4× e difíceis 2×
  (*replay*). É propriedade do pacing acumulado, não confundidor; equalizar exigiria mudar as
  épocas por etapa.
- **Recortes pareados em orçamento.** Oito dos doze recortes têm todos os protocolos em 4N: Q1,
  Q2a, Q3a, Q3b, Q4a, Q4b, Q5 e Q6a. Os quatro restantes são heterogêneos **por desenho** — Q2c e
  Q6b comparam 4N com a progressão granular (11N), Q6c inclui o warm-up (8N) e a estabilização
  (9N), e Q2b inclui o d14 (8N).
- **Warmup e clipping.** D16–D25 usam `warmup_steps: 100` (rampa após cada reset de otimizador) e
  D19, D20, D25 usam `max_grad_norm: 0.3`, contra warmup 5 e clip 1.0 nos demais. No PubMed e no
  Summa isso confunde Q2a, Q4b, Q6a e Q6b; no SemClinBr o warmup é 5 em todos (escala
  proporcional) e só o clip continua confundido. Lista por recorte em
  [RESUMO_EXPERIMENTOS.md §5](./RESUMO_EXPERIMENTOS.md).
- **Posição das ativações no d22 × d23.** O d23 usa 1 época por span (dataset completo) e o d22
  usa 2 (fatias), para equalizar o orçamento — os grupos de camadas acordam em pontos diferentes do
  stream. Registrar como limitação.
- **Evidência de eficiência.** As comparações leem `training_metrics.jsonl` via
  `pasta_treinamento`: o *spike* de loss em cada fronteira dos segmentados e sua ausência nos
  fundidos é a evidência primária da Q6, não apenas o score final.
- **Referência.** No Summa a referência é um professor, não humana: as métricas medem
  concordância com o professor, e o juiz LLM mede qualidade. No PubMed e no SemClinBr o gold é
  humano e concordar com ele é acertar.
