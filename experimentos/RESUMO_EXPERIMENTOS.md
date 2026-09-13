# Resumo dos Experimentos — Configurações dos Protocolos

> Gerado automaticamente por `tabela_experimentos.py`. Não editar manualmente.

## 1. Parâmetros Constantes (todos os protocolos)

| Parâmetro | Valor |
|:--|:--|
| Batch efetivo | 16 |
| Seed | 3407 |
| Otimizador | adamw_8bit |
| Weight decay | 0.01 |
| LR scheduler | cosine |
| LoRA dropout | 0.05 |
| LoRA α | 2×r |
| LoRA target_modules | q, k, v, o, gate, up, down (7 projeções) |
| train_on_responses_only | true |

## 2. Parâmetros por Experimento

| Parâmetro | Pubmed | SemClinBR | Summa |
|:--|:--|:--|:--|
| Modelo base | Qwen2.5-1.5B-Instruct | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct |
| max_seq_length | 8192 / 16384 | 12288 | 32768 |
| batch_size/GPU | 2 | 1 | 1 |

## 3. Tabela de Protocolos

**Legenda:**
- **CL**: ↑ ascendente · ↓ anti-CL · ∼ aleatório · — sem CL
- **Modo CL**: disj.=disjunto · acum.=acumulado · gran.=granular
- **PT**: —=sem · troca=merge LoRA↔FF · unfreeze=descongelamento · gating=gating de LR
- **Fronteira**: real=reset otimizador · virtual=um único train() · N/A=etapa única
- **Precisão**: 4b=NF4 QLoRA · 16b=bf16 · misto=etapas FF 16b + LoRA 4b
- **P/S/Su**: disponibilidade em Pubmed/SemClinBR/Summa

| ID | Grupo | Descrição | #Et. | Sequência | #FF | #L | CL | Modo CL | PT | Fronteira | Precisão | LR | LoRA r | grad_norm | warmup | P | S | Su |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| b | Q1 | Baseline LoRA 4b direto | 1 | L | 0 | 1 | — | N/A | — | N/A | 4b | 2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| b16 | Q2a | Controle LoRA 16b r=16 | 1 | L | 0 | 1 | — | N/A | — | N/A | 16b | 2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| b16r8 | Q2a | Controle LoRA 16b r=8 | 1 | L | 0 | 1 | — | N/A | — | N/A | 16b | 2e-5 | 8 | 1 | 5 | ✓ | ✓ | ✓ |
| c | Q1 | Baseline Full FT 16b direto | 1 | F | 1 | 0 | — | N/A | — | N/A | 16b | 5e-6 | N/A | 1 | 5 | ✓ | ✓ | ✓ |
| d1 | Q2a,Q3a | CL disj. FF→LoRA (FF precoce) | 4 | F→L→L→L | 1 | 3 | ↑ | disj. | troca | real | misto | 5e-6/2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d2 | Q2a,Q3a | CL disj. LoRA→FF (FF tardio) | 4 | L→L→L→F | 1 | 3 | ↑ | disj. | troca | real | misto | 5e-6/2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d3 | Q2c,Q3a | CL acum. FF→LoRA (replay) | 3 | F→L→L | 1 | 2 | ↑ | acum. | troca | real | misto | 5e-6/2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d4 | Q2c,Q3a | CL acum. LoRA→FF | 3 | L→L→F | 1 | 2 | ↑ | acum. | troca | real | misto | 5e-6/2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d5 | Q3b,Q4a | FF→LoRA sem CL | 2 | F→L | 1 | 1 | — | N/A | troca | real | misto | 5e-6/2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d6 | Q3b,Q4a | LoRA→FF sem CL | 2 | L→F | 1 | 1 | — | N/A | troca | real | misto | 5e-6/2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d7 | Q2b,Q4a,Q5 | CL puro disj. LoRA 4b | 4 | L→L→L→L | 0 | 4 | ↑ | disj. | — | real | 4b | 2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d8 | Q2b | CL puro acum. LoRA 4b | 3 | L→L→L | 0 | 3 | ↑ | acum. | — | real | 4b | 2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d9 | Q5 | Anti-CL disj. LoRA 4b | 4 | L→L→L→L | 0 | 4 | ↓ | disj. | — | real | 4b | 2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d10 | Q5 | Anti-CL acum. LoRA 4b | 3 | L→L→L | 0 | 3 | ↓ | acum. | — | real | 4b | 2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d11 | Q2c | CL gran. FF→LoRA (10 etapas) | 10 | F→L→L→L→L→L→L→L→L→L | 1 | 9 | ↑ | gran. | troca | real | misto | 5e-6/2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d12 | Q2c,Q6b | CL gran. LoRA→FF (10 etapas) | 10 | L→L→L→L→L→L→L→L→L→F | 1 | 9 | ↑ | gran. | troca | real | misto | 5e-6/2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d13 | Q2b,Q4a | CL puro disj. FF 16b | 4 | F→F→F→F | 4 | 0 | ↑ | disj. | — | real | 16b | 5e-6 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d14 | Q6c | Warm-up LoRA + CL FF | 5 | L→F→F→F→F | 4 | 1 | ↑ | disj. | troca | real | misto | 5e-6/2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d15 | Q6c | Warm-up LoRA + estab. + CL FF | 6 | L→F→F→F→F→F | 5 | 1 | ↑ | disj. | troca | real | misto | 1e-6/5e-6/2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d16 | Q6a | B segmentado 16b (custo fronteira) | 4 | L→L→L→L | 0 | 4 | — | N/A | — | real | 16b | 2e-5 | 16 | 1 | 100 | ✓ | ✓ | ✓ |
| d17 | Q2a,Q6a | CL puro disj. LoRA 16b | 4 | L→L→L→L | 0 | 4 | ↑ | disj. | — | real | 16b | 2e-5 | 16 | 1 | 100 | ✓ | ✓ | ✓ |
| d18 | Q2a,Q5 | Blocos aleatórios LoRA 16b | 4 | L→L→L→L | 0 | 4 | ∼ | disj. | — | real | 16b | 2e-5 | 16 | 1 | 100 | ✓ | ✓ | ✓ |
| d19 | Q4b | CL + unfreeze (tese central) | 4 | F→F→F→F | 4 | 0 | ↑ | disj. | unfreeze | real | 16b | 3e-6/5e-6 | N/A | 0.3 | 100 | ✓ | ✓ | ✓ |
| d20 | Q4b | Unfreeze sem CL (controle d19) | 4 | F→F→F→F | 4 | 0 | — | N/A | unfreeze | real | 16b | 3e-6/5e-6 | N/A | 0.3 | 100 | ✓ | ✓ | ✓ |
| d21 | Q6a | CL fundido LoRA 16b (espelho d17) | 4 | L→L→L→L | 0 | 4 | ↑ | disj. | — | virtual | 16b | 2e-5 | 16 | 1 | 100 | ✓ | ✓ | ✓ |
| d22 | Q6a,Q4b | CL + gating fundido LoRA 16b | 4 | L→L→L→L | 0 | 4 | ↑ | disj. | gating | virtual | 16b | 2e-5 | 16 | 1 | 100 | ✓ | ✓ | ✓ |
| d23 | Q6a | Gating fundido sem CL (controle d22) | 4 | L→L→L→L | 0 | 4 | — | N/A | gating | virtual | 16b | 2e-5 | 16 | 1 | 100 | ✓ | ✓ | ✓ |
| d24 | Q6b | CL gran. + gating fundido LoRA 4b | 10 | L→L→L→L→L→L→L→L→L→L | 0 | 10 | ↑ | gran. | gating | virtual | 4b | 2e-5 | 16 | 1 | 100 | ✓ | ✓ | ✓ |
| d25 | Q6b | CL gran. + gating fundido FF 16b | 10 | F→F→F→F→F→F→F→F→F→F | 10 | 0 | ↑ | gran. | gating | virtual | 16b | 5e-6 | N/A | 0.3 | 100 | ✓ | ✓ | ✓ |
| d1a | Réplica | Réplica 2 do d1 | 4 | F→L→L→L | 1 | 3 | ↑ | disj. | troca | real | misto | 5e-6/2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |
| d1b | Réplica | Réplica 3 do d1 | 4 | F→L→L→L | 1 | 3 | ↑ | disj. | troca | real | misto | 5e-6/2e-5 | 16 | 1 | 5 | ✓ | ✓ | ✓ |

## 4. Notas de Consistência

✅ Todos os parâmetros diferenciadores são **consistentes** entre experimentos.

✅ Todos os parâmetros constantes estão nos valores esperados.

## 5. Notas de Comparabilidade

- **Precisão:** Protocolos com precisão "misto" têm etapas FF (sempre 16b) e etapas LoRA (4b NF4). Cruzamentos entre grupos de precisão carregam {efeito estudado + quantização} como diferença conjunta.
- **Orçamento:** d16–d25 são calibrados em 4N instâncias (mesmo total de B com 4 épocas). Protocolos anteriores podem não seguir essa paridade.
- **Fronteiras reais** resetam otimizador Adam e scheduler cosine. Fronteiras virtuais (protocolos fundidos) mantêm trajetória contínua.
- **Gating ≠ Congelamento:** No d19/d20 blocos congelados não entram no otimizador. No gating (d22–d25) todos os grupos estão no otimizador desde o step 0, com LR 0 até acordarem.
