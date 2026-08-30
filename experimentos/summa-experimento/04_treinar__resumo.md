
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