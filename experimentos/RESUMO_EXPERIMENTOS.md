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
| train_on_responses_only | true |
| LoRA target_modules | down, gate, k, o, q, up, v (7 projeções) |
| LoRA α | 2×r |

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
- **N-eq**: orçamento de dados em dataset-equivalentes (Σ fração × épocas)
- **LR / warmup**: valor efetivo de cada etapa treinável (com o fallback real do treinador); um valor só quando todas as etapas são iguais, senão sequência por etapa · `Ng` = rampa de N steps por grupo (fusão)
- **grad_norm**: `max_grad_norm` (clipping)
- **Grupo**: recortes de contraste de `06_compara_todos.yaml` · — = fora de todo recorte (só `Panorama_Geral`)
- **P/S/Su**: disponibilidade em Pubmed/SemClinBR/Summa

| ID | Grupo | Descrição | #Et. | N-eq | Sequência | #FF | #L | CL | Modo CL | PT | Fronteira | Precisão | LR | LoRA r | grad_norm | warmup | P | S | Su |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| b | Q1,Q2b,Q3b,Q4a,Q4b,Q5,Q6a,Q6b,Q6c | Baseline LoRA 4b direto | 1 | 4N | L | 0 | 1 | — | N/A | — | N/A | 4b | 2e-5 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| b16 | Q2a,Q6a | Controle LoRA 16b r=16 | 1 | 4N | L | 0 | 1 | — | N/A | — | N/A | 16b | 2e-5 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| b16r8 | — | Controle LoRA 16b r=8 | 1 | 4N | L | 0 | 1 | — | N/A | — | N/A | 16b | 2e-5 | 8 | 1.0 | 5 | ✓ | ✓ | ✓ |
| c | Q1,Q3b,Q4a,Q4b,Q6a,Q6b,Q6c | Baseline Full FT 16b direto | 1 | 4N | F | 1 | 0 | — | N/A | — | N/A | 16b | 5e-6 | N/A | 1.0 | 5 | ✓ | ✓ | ✓ |
| d1 | Q3a,Q4a | CL disj. FF→LoRA (FF precoce) | 4 | 4N | F→L→L→L | 1 | 3 | ↑ | disj. | troca | real | misto | 5e-6→2e-5→2e-5→2e-5 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d2 | Q3a | CL disj. LoRA→FF (FF tardio) | 4 | 4N | L→L→L→F | 1 | 3 | ↑ | disj. | troca | real | misto | 2e-5→2e-5→2e-5→5e-6 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d3 | Q2c,Q3a | CL acum. FF→LoRA (replay) | 3 | 4N | F→L→L | 1 | 2 | ↑ | acum. | troca | real | misto | 5e-6→2e-5→2e-5 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d4 | Q2c,Q3a | CL acum. LoRA→FF | 3 | 4N | L→L→F | 1 | 2 | ↑ | acum. | troca | real | misto | 2e-5→2e-5→5e-6 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d5 | Q3b,Q4a | FF→LoRA sem CL | 2 | 4N | F→L | 1 | 1 | — | N/A | troca | real | misto | 5e-6→2e-5 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d6 | Q3b | LoRA→FF sem CL | 2 | 4N | L→F | 1 | 1 | — | N/A | troca | real | misto | 2e-5→5e-6 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d7 | Q2b,Q4a,Q5 | CL puro disj. LoRA 4b | 4 | 4N | L→L→L→L | 0 | 4 | ↑ | disj. | — | real | 4b | 2e-5 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d8 | Q2b,Q5 | CL puro acum. LoRA 4b | 3 | 4N | L→L→L | 0 | 3 | ↑ | acum. | — | real | 4b | 2e-5 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d9 | Q5 | Anti-CL disj. LoRA 4b | 4 | 4N | L→L→L→L | 0 | 4 | ↓ | disj. | — | real | 4b | 2e-5 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d10 | Q5 | Anti-CL acum. LoRA 4b | 3 | 4N | L→L→L | 0 | 3 | ↓ | acum. | — | real | 4b | 2e-5 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d11 | Q2c | CL gran. FF→LoRA (10 etapas) | 10 | 11N | F→L→L→L→L→L→L→L→L→L | 1 | 9 | ↑ | gran. | troca | real | misto | 5e-6→2e-5→2e-5→2e-5→2e-5→2e-5→2e-5→2e-5→2e-5→2e-5 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d12 | Q2c,Q6b | CL gran. LoRA→FF (10 etapas) | 10 | 11N | L→L→L→L→L→L→L→L→L→F | 1 | 9 | ↑ | gran. | troca | real | misto | 2e-5→2e-5→2e-5→2e-5→2e-5→2e-5→2e-5→2e-5→2e-5→5e-6 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d13 | Q2b,Q6c | CL puro disj. FF 16b | 4 | 4N | F→F→F→F | 4 | 0 | ↑ | disj. | — | real | 16b | 5e-6 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d14 | Q2b,Q6c | Warm-up LoRA + CL FF | 5 | 8N | L→F→F→F→F | 4 | 1 | ↑ | disj. | troca | real | misto | 2e-5→5e-6→5e-6→5e-6→5e-6 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d15 | Q6c | Warm-up LoRA + estab. + CL FF | 6 | 9N | L→F→F→F→F→F | 5 | 1 | ↑ | disj. | troca | real | misto | 2e-5→1e-6→5e-6→5e-6→5e-6→5e-6 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d16 | Q2a | B segmentado 16b (custo fronteira) | 4 | 4N | L→L→L→L | 0 | 4 | — | N/A | — | real | 16b | 2e-5 | 16 | 1.0 | 100 | ✓ | ✓ | ✓ |
| d17 | Q2a,Q6a | CL puro disj. LoRA 16b | 4 | 4N | L→L→L→L | 0 | 4 | ↑ | disj. | — | real | 16b | 2e-5 | 16 | 1.0 | 100 | ✓ | ✓ | ✓ |
| d18 | Q2a | Blocos aleatórios LoRA 16b | 4 | 4N | L→L→L→L | 0 | 4 | ∼ | disj. | — | real | 16b | 2e-5 | 16 | 1.0 | 100 | ✓ | ✓ | ✓ |
| d19 | Q4b | CL + unfreeze (tese central) | 4 | 4N | F→F→F→F | 4 | 0 | ↑ | disj. | unfreeze | real | 16b | 5e-6→5e-6→5e-6→3e-6 | N/A | 0.3 | 100 | ✓ | ✓ | ✓ |
| d20 | Q4b | Unfreeze sem CL (controle d19) | 4 | 4N | F→F→F→F | 4 | 0 | — | N/A | unfreeze | real | 16b | 5e-6→5e-6→5e-6→3e-6 | N/A | 0.3 | 100 | ✓ | ✓ | ✓ |
| d21 | Q6a | CL fundido LoRA 16b (espelho d17) | 4 | 4N | L→L→L→L | 0 | 4 | ↑ | disj. | — | virtual | 16b | 2e-5 | 16 | 1.0 | 100+100g | ✓ | ✓ | ✓ |
| d22 | Q6a | CL + gating fundido LoRA 16b | 4 | 4N | L→L→L→L | 0 | 4 | ↑ | disj. | gating | virtual | 16b | 2e-5 | 16 | 1.0 | 100+100g | ✓ | ✓ | ✓ |
| d23 | Q6a | Gating fundido sem CL (controle d22) | 4 | 4N | L→L→L→L | 0 | 4 | — | N/A | gating | virtual | 16b | 2e-5 | 16 | 1.0 | 100+100g | ✓ | ✓ | ✓ |
| d24 | Q6b | CL gran. + gating fundido LoRA 4b | 10 | 11N | L→L→L→L→L→L→L→L→L→L | 0 | 10 | ↑ | gran. | gating | virtual | 4b | 2e-5 | 16 | 1.0 | 100+100g | ✓ | ✓ | ✓ |
| d25 | Q6b | CL gran. + gating fundido FF 16b | 10 | 11N | F→F→F→F→F→F→F→F→F→F | 10 | 0 | ↑ | gran. | gating | virtual | 16b | 5e-6 | N/A | 0.3 | 100+100g | ✓ | ✓ | ✓ |
| d1a | Réplica | Réplica 2 do d1 | 4 | 4N | F→L→L→L | 1 | 3 | ↑ | disj. | troca | real | misto | 5e-6→2e-5→2e-5→2e-5 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |
| d1b | Réplica | Réplica 3 do d1 | 4 | 4N | F→L→L→L | 1 | 3 | ↑ | disj. | troca | real | misto | 5e-6→2e-5→2e-5→2e-5 | 16 | 1.0 | 5 | ✓ | ✓ | ✓ |

## 4. Famílias de Configuração

Protocolos agrupados por warmup, clipping e precisão. Dentro de uma família esses fatores são iguais; entre famílias, a diferença vem junto com o efeito estudado.

| warmup | max_grad_norm | Precisão | Protocolos |
|:--|:--|:--|:--|
| 5 | 1.0 | 4b | b, d7, d8, d9, d10 |
| 5 | 1.0 | 16b | b16, b16r8, c, d13 |
| 5 | 1.0 | misto | d1, d2, d3, d4, d5, d6, d11, d12, d14, d15, d1a, d1b |
| 100 | 1.0 | 16b | d16, d17, d18 |
| 100 | 0.3 | 16b | d19, d20 |
| 100+100g | 1.0 | 16b | d21, d22, d23 |
| 100+100g | 1.0 | 4b | d24 |
| 100+100g | 0.3 | 16b | d25 |

> Warmup na escala de referência (Pubmed/Summa). Escala proporcional ao nº de steps — SemClinBR: d16, d17, d18, d19, d20, d21, d22, d23, d24, d25 usam 5 (≡ 100), 5+5g (≡ 100+100g). Dentro desses experimentos, o confundimento de warmup entre famílias não se aplica.

## 5. Contrastes com Confusão

Recortes de `06_compara_todos.yaml` cujos protocolos diferem em fatores além do eixo estudado (se o fator listado **é** o eixo do recorte, ignore):

- **Q1_ajuste_fino** — precisão: 4b (b) × 16b (c)
- **Q2a_cl_controlado** — warmup: 5 (b16) × 100 (d16, d18, d17)
- **Q2b_cl_puro** — precisão: 4b (b, d7, d8) × 16b (d13) × misto (d14) · N-eq: 4N (b, d7, d8, d13) × 8N (d14)
- **Q2c_granularidade** — N-eq: 4N (d3, d4) × 11N (d11, d12)
- **Q3b_direcao_sem_cl** — precisão: 4b (b) × 16b (c) × misto (d5, d6)
- **Q4a_decomposicao** — precisão: 4b (b, d7) × 16b (c) × misto (d1, d5)
- **Q4b_unfreeze** — warmup: 5 (b, c) × 100 (d20, d19) · max_grad_norm: 1.0 (b, c) × 0.3 (d20, d19) · precisão: 4b (b) × 16b (c, d20, d19)
- **Q6a_fusao** — warmup: 5 (b, c, b16) × 100 (d17) × 100+100g (d21, d22, d23) · precisão: 4b (b) × 16b (c, b16, d17, d21, d22, d23)
- **Q6b_fusao_granular** — warmup: 5 (b, c, d12) × 100+100g (d24, d25) · max_grad_norm: 1.0 (b, c, d12, d24) × 0.3 (d25) · precisão: 4b (b, d24) × 16b (c, d25) × misto (d12) · N-eq: 4N (b, c) × 11N (d12, d24, d25)
- **Q6c_transicao_regime** — precisão: 4b (b) × 16b (c, d13) × misto (d14, d15) · N-eq: 4N (b, c, d13) × 8N (d14) × 9N (d15)

> Warmup na escala de referência (Pubmed/Summa). Escala proporcional ao nº de steps — SemClinBR: d16, d17, d18, d19, d20, d21, d22, d23, d24, d25 usam 5 (≡ 100), 5+5g (≡ 100+100g). Dentro desses experimentos, o confundimento de warmup entre famílias não se aplica.

## 6. Notas de Consistência

### Treino × YAML atual (retreino pendente)

15 treino(s) feitos com uma versão do YAML diferente da atual (cópia em `treinamento/treinamento_config/`). As conferências abaixo usam o YAML do treino:

- **Summa/d1** (treinado com `04_treinar_d1 (v001).yaml`): `lr` 5e-6→2e-5→5e-6→5e-6 → 5e-6→2e-5→2e-5→2e-5
- **Summa/d3** (treinado com `04_treinar_d3 (v001).yaml`): `lr` 5e-6→2e-5→5e-6 → 5e-6→2e-5→2e-5
- **Summa/d5** (treinado com `04_treinar_d5 (v001).yaml`): `lr` 5e-6 → 5e-6→2e-5
- **SemClinBR/d16** (treinado com `04_treinar_d16 (v001).yaml`): `warmup` 100 → 5
- **SemClinBR/d17** (treinado com `04_treinar_d17 (v001).yaml`): `warmup` 100 → 5
- **SemClinBR/d18** (treinado com `04_treinar_d18 (v001).yaml`): `warmup` 100 → 5
- **SemClinBR/d19** (treinado com `04_treinar_d19 (v001).yaml`): `warmup` 100 → 5
- **SemClinBR/d20** (treinado com `04_treinar_d20 (v001).yaml`): `warmup` 100 → 5
- **SemClinBR/d21** (treinado com `04_treinar_d21 (v001).yaml`): `warmup` 100+100g → 5+5g
- **SemClinBR/d22** (treinado com `04_treinar_d22 (v001).yaml`): `warmup` 100+100g → 5+5g
- **SemClinBR/d23** (treinado com `04_treinar_d23 (v001).yaml`): `warmup` 100+100g → 5+5g
- **SemClinBR/d24** (treinado com `04_treinar_d24 (v001).yaml`): `warmup` 100+100g → 5+5g
- **SemClinBR/d25** (treinado com `04_treinar_d25 (v001).yaml`): `warmup` 100+100g → 5+5g
- **Summa/d1a** (treinado com `04_treinar_d1a (v001).yaml`): `lr` 5e-6→2e-5→5e-6→5e-6 → 5e-6→2e-5→2e-5→2e-5
- **Summa/d1b** (treinado com `04_treinar_d1b (v001).yaml`): `lr` 5e-6→2e-5→5e-6→5e-6 → 5e-6→2e-5→2e-5→2e-5

### LR herdado da 1ª etapa

✅ Nenhuma etapa depende do fallback de LR da 1ª etapa.

### LR do YAML × treino real (`training_metrics.jsonl`)

✅ LR treinado confere com o esperado em 76 treinos (fusão fora da conferência).

### Warmup × tamanho real da etapa

27 etapa(s) em que o warmup do treino ocupa mais de 25% dos steps reais (segundo o `train_begin` do jsonl):

- **SemClinBR/d11** etapa 0 `f1-FF`: warmup 5 de 10 steps (50%)
- **SemClinBR/d11** etapa 1 `f2`: warmup 5 de 18 steps (28%)
- **SemClinBR/d12** etapa 0 `f1`: warmup 5 de 10 steps (50%)
- **SemClinBR/d12** etapa 1 `f2`: warmup 5 de 18 steps (28%)
- **SemClinBR/d16** etapa 0 `completo-1`: warmup 100 de 44 steps (227%) — pico nunca atingido (máx ≈ 43% do LR)
- **SemClinBR/d16** etapa 1 `completo-2`: warmup 100 de 44 steps (227%) — pico nunca atingido (máx ≈ 43% do LR)
- **SemClinBR/d16** etapa 2 `completo-3`: warmup 100 de 44 steps (227%) — pico nunca atingido (máx ≈ 43% do LR)
- **SemClinBR/d16** etapa 3 `completo-4`: warmup 100 de 44 steps (227%) — pico nunca atingido (máx ≈ 43% do LR)
- **SemClinBR/d17** etapa 0 `fácil`: warmup 100 de 28 steps (357%) — pico nunca atingido (máx ≈ 27% do LR)
- **SemClinBR/d17** etapa 1 `médio`: warmup 100 de 36 steps (278%) — pico nunca atingido (máx ≈ 35% do LR)
- **SemClinBR/d17** etapa 2 `difícil`: warmup 100 de 28 steps (357%) — pico nunca atingido (máx ≈ 27% do LR)
- **SemClinBR/d17** etapa 3 `completo`: warmup 100 de 88 steps (114%) — pico nunca atingido (máx ≈ 87% do LR)
- **SemClinBR/d18** etapa 0 `bloco-1`: warmup 100 de 30 steps (333%) — pico nunca atingido (máx ≈ 29% do LR)
- **SemClinBR/d18** etapa 1 `bloco-2`: warmup 100 de 30 steps (333%) — pico nunca atingido (máx ≈ 29% do LR)
- **SemClinBR/d18** etapa 2 `bloco-3`: warmup 100 de 30 steps (333%) — pico nunca atingido (máx ≈ 29% do LR)
- **SemClinBR/d18** etapa 3 `completo`: warmup 100 de 88 steps (114%) — pico nunca atingido (máx ≈ 87% do LR)
- **SemClinBR/d19** etapa 0 `fácil-uf75pct`: warmup 100 de 28 steps (357%) — pico nunca atingido (máx ≈ 27% do LR)
- **SemClinBR/d19** etapa 1 `médio-uf50pct`: warmup 100 de 36 steps (278%) — pico nunca atingido (máx ≈ 35% do LR)
- **SemClinBR/d19** etapa 2 `difícil-uf25pct`: warmup 100 de 28 steps (357%) — pico nunca atingido (máx ≈ 27% do LR)
- **SemClinBR/d19** etapa 3 `completo-uf0pct`: warmup 100 de 88 steps (114%) — pico nunca atingido (máx ≈ 87% do LR)
- **SemClinBR/d20** etapa 0 `completo-uf75pct`: warmup 100 de 44 steps (227%) — pico nunca atingido (máx ≈ 43% do LR)
- **SemClinBR/d20** etapa 1 `completo-uf50pct`: warmup 100 de 44 steps (227%) — pico nunca atingido (máx ≈ 43% do LR)
- **SemClinBR/d20** etapa 2 `completo-uf25pct`: warmup 100 de 44 steps (227%) — pico nunca atingido (máx ≈ 43% do LR)
- **SemClinBR/d20** etapa 3 `completo-uf0pct`: warmup 100 de 44 steps (227%) — pico nunca atingido (máx ≈ 43% do LR)
- **SemClinBR/d21** fusão: warmup global 100 de 175 steps (57%)
- **SemClinBR/d22** fusão: warmup global 100 de 175 steps (57%)
- **SemClinBR/d23** fusão: warmup global 100 de 175 steps (57%)

✅ Todos os parâmetros diferenciadores são **consistentes** entre experimentos.

Divergências previstas (`WARMUP_ESCALADO`):

- ℹ️  **d16**: `warmup` Pubmed (100) × SemClinBR (5) — escala proporcional
- ℹ️  **d17**: `warmup` Pubmed (100) × SemClinBR (5) — escala proporcional
- ℹ️  **d18**: `warmup` Pubmed (100) × SemClinBR (5) — escala proporcional
- ℹ️  **d19**: `warmup` Pubmed (100) × SemClinBR (5) — escala proporcional
- ℹ️  **d20**: `warmup` Pubmed (100) × SemClinBR (5) — escala proporcional
- ℹ️  **d21**: `warmup` Pubmed (100+100g) × SemClinBR (5+5g) — escala proporcional
- ℹ️  **d22**: `warmup` Pubmed (100+100g) × SemClinBR (5+5g) — escala proporcional
- ℹ️  **d23**: `warmup` Pubmed (100+100g) × SemClinBR (5+5g) — escala proporcional
- ℹ️  **d24**: `warmup` Pubmed (100+100g) × SemClinBR (5+5g) — escala proporcional
- ℹ️  **d25**: `warmup` Pubmed (100+100g) × SemClinBR (5+5g) — escala proporcional

✅ Todos os parâmetros constantes estão nos valores esperados.

## 7. Notas de Comparabilidade

- **Precisão:** Protocolos com precisão "misto" têm etapas FF (sempre 16b) e etapas LoRA (4b NF4). Cruzamentos entre grupos de precisão carregam {efeito estudado + quantização} como diferença conjunta.
- **Orçamento (N-eq):** **4N** (b, b16, b16r8, c, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d13, d16, d17, d18, d19, d20, d21, d22, d23, d1a, d1b) · **8N** (d14) · **9N** (d15) · **11N** (d11, d12, d24, d25). Contrastes dentro de uma faixa são limpos; entre faixas carregam o orçamento junto e precisam declará-lo ao lado do resultado.
- **Fronteiras reais** resetam otimizador Adam e scheduler cosine. Fronteiras virtuais (protocolos fundidos) mantêm trajetória contínua.
- **Gating ≠ Congelamento:** No d19/d20 blocos congelados não entram no otimizador. No gating (d22–d25) todos os grupos estão no otimizador desde o step 0, com LR 0 até acordarem.
