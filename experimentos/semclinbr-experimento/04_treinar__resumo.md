# Protocolos de treinamento — experimento SemClinBr (Qwen2.5-7B-Instruct)

Índice dos 29 protocolos de `04_treinar_*.yaml`, agrupados pelos contrastes
que cada par de gêmeos isola. A estrutura é a mesma dos experimentos PubMed e
SUMMA — é isso que permite a leitura paralela do README §7.6.

## Parâmetros comuns a todos

| Item | Valor |
|---|---|
| Modelo base | `Qwen2.5-7B-Instruct` |
| Janela | `max_seq_length: 12288` (global e por etapa), `filtrar_max_seq_length: true` |
| Lote | `efetivo: 16`, `batch_size: 1` → acúmulo real **16**, lote efetivo **16** |
| LR | global `2e-05`; etapas `lora` → `2e-05`; etapas `full` → `5e-06` |
| Seed | `3407` |
| Entrada | `dados/semclinbr.parquet` (`texto`, `id`) + `dados/prompt_semclinbr.txt` |
| Gabarito | `saidas/saida_semclinbr_gold.parquet` (`chave`, `resposta`) |
| Divisão | `dados/divisao_Gold_Qwen7B.csv` (gerada pelo passo 03) |

> **`batch_size.efetivo` manda em `grad_batch_size`.** `_aplicar_batch_size_auto`
> (src/treinar_unsloth_util.py:958) recalcula `grad_batch_size = efetivo /
> (batch_size × n_gpus)` e **sobrescreve** o valor escrito no YAML — o número ao
> lado de `grad_batch_size:` é documentação, não configuração. Com `efetivo: 16`
> e `batch_size: 1` em 1 GPU, o acúmulo real é 16 e o lote efetivo é 16, o mesmo
> que os 17 protocolos já treinados do SUMMA registram em `modelo_info.md`.
>
> **O override por etapa não recalcula o acúmulo.** `treinar_unsloth.py:2641`
> troca `batch_size` da etapa mas mantém o `grad_batch_size` global. Aqui isso é
> inócuo porque o `batch_size` global já é 1 e toda etapa que declara
> `batch_size: 1` repete o mesmo valor — as 123 etapas rodam em lote efetivo 16.
> No PubMed, cujo `batch_size` global é 2, oito etapas `full` que declaram
> `batch_size: 1` acabam em lote efetivo 8 (ver README_PUBMED_EXPERIMENTO.md).

> **Uma seed por protocolo, por enquanto.** Com 700 documentos de treino, os
> protocolos granulares de 10 etapas (D11, D12, D24, D25) ficam com ~70
> instâncias por faixa e os de 3 faixas com ~230. O README §8.2 prevê alta
> variância entre seeds nesse regime e recomenda ≥3 seeds por protocolo. Os
> YAMLs saem com `seed: 3407`; a replicação por seed é uma segunda rodada.

### 1. Gêmeos de CL + Unfreeze (D19) vs. Fusão com Gating (D22)
* **D19:** CL por etapas + unfreeze progressivo, **FF 16b** *(Múltiplos estágios/trainers separados).*
* **D22:** FUSÃO: CL por etapas + gating de LR por camada, **LoRA 16b** *(Mesmo trainer, sem quebra de fronteira).*

### 2. Gêmeos de Dados Completos + Unfreeze (D20) vs. Fusão com Gating (D23)
* **D20:** Sem CL + unfreeze progressivo, **FF 16b** (dados completos) *(Múltiplos estágios).*
* **D23:** FUSÃO: gating de LR sem CL (dados completos), **LoRA 16b** *(Sem quebra de trainer).*

### 3. Gêmeos de Pré-treino + CL (Sem vs. Com Estabilização)
* **D14:** PT LoRA Completo $\rightarrow$ CL por etapas FF-only *(Passagem direta).*
* **D15:** Gêmeo **Com fase de Estabilização** *(Insere uma etapa FF com taxa de aprendizado baixa antes de iniciar o CL).*

### 4. Gêmeos de CL por Etapas (Progressão de Capacidade/Estratégia)
Todos seguem a sequência *Fácil $\rightarrow$ Médio $\rightarrow$ Difícil $\rightarrow$ Completo*, variando o tipo de ajuste:
* **D7:** Apenas LoRA (4b).
* **D13:** Apenas Full Fine-Tuning (FF 16b).
* **D17:** CL por etapas, LoRA **16b** (D7 sem quantização).
* **D21:** FUSÃO: CL por etapas, **LoRA 16b**, fronteiras virtuais *(espelho do D17, sem interrupções).*
* **D1 e D2:** Gêmeos espelhados de capacidade (D1 faz FF $\rightarrow$ LoRA; D2 faz LoRA $\rightarrow$ FF).

### 5. Gêmeos de CL Acumulado (Progressão de Capacidade)
Todos seguem a sequência acumulada *Fácil $\rightarrow$ Médio (Acumulado) $\rightarrow$ Completo*, variando apenas a alocação de capacidade:
* **D8:** Apenas LoRA (4b).
* **D3:** Capacidade decrescente (FF $\rightarrow$ LoRA).
* **D4:** Capacidade crescente (LoRA $\rightarrow$ FF).

### 6. Gêmeos de CL Granular Acumulado (10 etapas seriais)
* **D11:** CL acumulado granular (10 etapas) + escalonamento FF $\rightarrow$ LoRA.
* **D12:** CL acumulado granular (10 etapas) + escalonamento LoRA $\rightarrow$ FF.

### 7. Gêmeos de CL Granular com Fusão (Gating)
* **D24:** FUSÃO: CL granular (10 spans) + gating 10%, **LoRA 4b** *(espelho do D12).*
* **D25:** FUSÃO: CL granular (10 spans) + gating 10%, **Full FT 16b**.

### 8. Gêmeos de Escalonamento Simples (Sem progressão de dificuldade)
* **D5:** Dados completos (FF $\rightarrow$ LoRA).
* **D6:** Gêmeo invertido, dados completos (LoRA $\rightarrow$ FF).

### 9. Baselines de Treino Direto (Etapa Única)
* **B:** Treino direto (LoRA 4b).
* **C:** Gêmeo em (Full FT 16b).

### 10. Gêmeos de Anti-CL (Direção Invertida)
Sequência invertida *Difícil $\rightarrow$ Médio $\rightarrow$ Fácil*, LoRA-only 4b — controle de direção do currículo:
* **D9:** Anti-CL por etapas, LoRA-only 4b.
* **D10:** Anti-CL acumulado, LoRA-only 4b.

### 11. Controles 16 bits (Ablações de Fronteira e Bloco)
* **D16:** Sem CL, 4 execuções com dados completos, LoRA 16b *(fronteira pura — controle de custo de múltiplos trainers).*
* **D18:** Blocos aleatórios (terços), LoRA 16b *(controle de efeito de bloco — sem progressão de dificuldade).*
### 12. Controles do baseline B (quantização e posto do LoRA)
Precificam as duas escolhas de implementação embutidas no B, que de outra forma
contaminam toda comparação contra ele. Idênticos em dados, pace, orçamento e LR:
mudam só a precisão da base e o posto do adaptador.
* **B:** LoRA NF4 (4 bits), r = 16 — baseline usado nas demais comparações.
* **B16:** LoRA 16 bits, r = 16 — mesma capacidade, sem quantização.
* **B16R8:** LoRA 16 bits, r = 8 — mesma precisão, metade do posto.

Ler `b` vs `b16` dá o fator de correção entre os protocolos de 4 bits
(b, d1–d15, d24) e os de 16 bits (d16–d23, d25); `b16` vs `b16r8` diz se o posto
do adaptador está saturado. Ver `06_compara_controle_b.yaml`.
