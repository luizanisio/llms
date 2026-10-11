# Correções e verificações pendentes

> Diagnóstico: `python tabela_experimentos.py` → `RESUMO_EXPERIMENTOS.md` (§4–6).

**Concluído:** summa d1, d1a, d1b, d3, d5 com `learning_rate` explícito por etapa (retreino em andamento); pubmed d21–d23 com caminho de divisão atualizado (split do treino já era o mesmo arquivo); LR treinado = esperado nos demais treinos.

## 1. Após o retreino do summa

- [ ] Conferir no jsonl que as etapas LoRA treinaram com 2e-5 (`tabela_experimentos.py`).
- [ ] Predição, avaliação e `06_compara_*` dos recortes afetados.
- [ ] Manter as versões antigas renomeadas (ex.: `(d1)_lr5e-6`) como ablação de LR baixo.
- [ ] d1a/d1b: usar só se a variância entre seeds do d1 entrar na tese.

## 2. Retreino do semclinbr d16–d25 (warmup)

Warmup 100 era maior que as etapas (114–357% em d16–d20; o LR nunca chegava ao pico) e ocupava 57% (d21–d23) e 21% (d24–d25) do run na fusão. **YAMLs já corrigidos:** `warmup_steps: 5` e `fusao.warmup_grupo_steps: 5`. Assim o warmup ocupa a mesma fração dos steps que warmup 100 ocupa no pubmed/summa (d16–d20: 6–18% × 5–19%; fusão: 2,9% e 1,05% × 2,7–2,9% e 1,0–1,1%). Justificativa e tabela completa: `semclinbr-experimento/README_SEMCLIMBR_EXPERIMENTO.md` §2.4.

- [ ] Renomear as pastas de saída atuais (ex.: `(d16)_warmup100`) **antes** de treinar: com `resume_from_checkpoint: true`, o treino retomaria o checkpoint antigo.
- [ ] Retreinar d16–d25 (barato: 175–478 steps por protocolo).
- [ ] Conferir no `tabela_experimentos.py` que d16–d25 saíram de "Treino × YAML atual" e de "Warmup × tamanho real da etapa".
- [ ] Predição, avaliação NER e `06_compara_*`.

## 3. Correções de código (sem GPU)

- [ ] **C1 – Fallback de LR:** etapa sem `learning_rate` herda o da 1ª etapa (`treinar_unsloth_util.py` ~L509). Guardar os globais antes do override e restaurá-los entre etapas; depois `FALLBACK_DA_1A_ETAPA = False` em `tabela_experimentos.py`. Até lá: LR explícito em toda etapa.
- [ ] **C2 – Aviso de warmup:** alertar se `warmup_steps > 25%` dos steps da etapa/fusão (opcional: `warmup_ratio`).
- [ ] **C3 – `train_begin`:** gravar valores efetivos (LR, warmup, `max_grad_norm`, scheduler, `max_seq_length`, tipo, nbits, LoRA r/alpha, treináveis, `n_gpus`).
- [ ] **C4 – Report/gráficos:** hiperparâmetros por etapa e nº de GPUs vindos do jsonl (`treinar_unsloth_report.py` L296/L326/L430); pico de LR por etapa e linha de `max_grad_norm`; logar os primeiros `max(2×warmup, 50)` steps de cada etapa.

Regra para YAML novo: `learning_rate`, `warmup_steps` e `max_grad_norm` explícitos.

## 4. Contrastes confundidos (warmup 5 × 100; clip 1.0 × 0.3)

Afeta Q2a, Q4b, Q6a, Q6b (RESUMO §5) no pubmed e no summa. No semclinbr, depois do retreino, só o clip continua confundido (todos os protocolos com warmup 5). Ablações baratas no pubmed:

| Treino | Compara com | Responde |
|---|---|---|
| `b16-s2` | b16 | ruído entre seeds |
| `d16-w5` | d16, b16 | efeito do warmup |
| `d20-clip1` | d20 | efeito do clip |

Diferença ≤ ruído de seed → declarar como não confundidor. Diferença relevante → retreinar os protocolos envolvidos. Sem as ablações: comparar só dentro de cada família e declarar os contrastes como confundidos.
