# Visão Geral dos Experimentos e Protocolos (Curriculum Learning)

> 🛠️ **Para instruções técnicas de setup de ambiente, instalação (CUDA, Flash-Attn, vLLM) e execução em background, consulte o [README.md](./README.md).**

Este documento centraliza a arquitetura dos experimentos, organizando os objetivos de cada frente de projeto e mapeando detalhadamente as configurações e propósitos das ablações de `d1` a `d8`.

---

## 1. Resumo dos Experimentos

### ⚖️ Summa-Experimento
- **Objetivo:** Extração de metadados jurídicos de acórdãos judiciais.
- **Modelo Base:** `Qwen2.5-7B-Instruct`
- **Ambiente de Treino:** Cluster Slurm (H100 80GB) e H200 em treinos FF
- **Detalhes Técnicos:** Textos muito extensos, exigindo contexto longo (`max_seq_length: 32768`), processados com lotes cuidadosos (`batch_size: 1`, `grad_batch_size: 16`).

### 🩺 Pubmed-Experimento
- **Objetivo:** Extração e classificação de partes (Background, Methods, Results, Conclusions) em abstracts médicos (RCTs) utilizando uma base de dados extensa de artigos biomédicos (PubMed 20k).
- **Modelo Base:** `Qwen2.5-1.5B-Instruct`
- **Ambiente de Treino:** Cluster Slurm (H100 80GB)
- **Detalhes Técnicos:** Menor comprimento de contexto (`max_seq_length: 8192`), mas grande volume de amostras, configurado com (`batch_size: 2`, `grad_batch_size: 8`).

### 🔬 Puil-Mini-Experimento
- **Objetivo:** Validação local rápida da máquina de estados do código-fonte (transições entre QLoRA e Full Fine-tuning, recarregamentos de modelo, scripts utilitários).
- **Modelo Base:** `Qwen2.5-1.5B-Instruct`
- **Ambiente de Treino:** Local via WSL em GPU padrão comercial (`RTX 3060 12GB VRAM`).
- **Detalhes Técnicos:** Pipeline simplificado com tamanho de contexto mínimo e batch unitário para rodar eficientemente e evitar Out-of-Memory, permitindo debugar os *engineers* sem gastar horas de computação no cluster pesado.

---

## 2. Comparativo dos Protocolos de Treinamento (Ablações)

A série `d1` a `d8` desenha o controle experimental para isolar três variáveis independentes: o **efeito progressivo do curriculum learning (CL)**, a **integração do Full Fine-tuning (FF)**, e a **direcionalidade do currículo (fácil-difícil vs. difícil-fácil)**. 

*(Nota: Todos os treinamentos LoRA operam em 4-bits globalmente, exceto versões explícitas de 16-bits para controle do efeito de quantização no summa).*

| ID | Ordem das Fases | O que este protocolo testa / Objetivo |
| :--- | :--- | :--- |
| **d1** | FF-fácil → LoRA-médio → LoRA-difícil → LoRA-completo | **CL + FF precoce**: Inicia forçando representações profundas na base com as tarefas fáceis em 16-bits. O resto do CL usa recursos computacionais mais leves (LoRA). |
| **d1(16bits)** | Igual ao d1 | *(Apenas Summa)* O LoRA opera em 16-bits sem quantização. **Testa:** O uso do 4-bits reduz a eficiência de aprendizagem final? |
| **d2** | LoRA-fácil → LoRA-médio → LoRA-difícil → FF-completo | **CL + FF tardio**: Deixa o modelo aprender sequencialmente em QLoRA e consolida todo o conhecimento alterando todo o modelo (FF 16-bits) quando vê todos os dados misturados na fase final. |
| **d2(16bits)** | Igual ao d2 | *(Apenas Summa)* Igual lógica do d1(16bits). |
| **d3** | LoRA-fácil → LoRA-médio → LoRA-difícil → LoRA-completo | **CL Progressivo Puro + Revisão**: Curriculum padrão, estritamente em LoRA. **Testa:** Isola o ganho estrutural do Curriculum retirando a "sujeira" analítica do Full FT. |
| **d4** | LoRA-completo → LoRA-difícil → LoRA-médio → LoRA-fácil | **Anti-currículo (Controle Negativo)**: Inverte a ordem cronológica estritamente em LoRA. **Testa:** A ordem "fácil→difícil" é essencial ou meramente expor o modelo aos dados fatiados já resulta em ganhos? (Espera-se d3 > d4). |
| **d5** | FF-completo → LoRA-completo | **Baseline FF Primeiro**: Sem curriculum learning (ambas as fases veem dados não filtrados). **Testa:** Se d1 superar d5, confirma-se o valor da progressão curricular precoce. |
| **d6** | LoRA-completo → FF-completo | **Baseline FF Último**: Sem curriculum learning. Análogo ao d2 em fases, mas sem fracionamento de dificuldade. **Testa:** Se d2 superar d6, confirma-se o valor da progressão curricular tardia. |
| **d7** | LoRA-fácil → LoRA-médio → LoRA-difícil | **CL Fragmentado**: Curriculum sem a etapa de consolidação mista final. **Testa:** Aprender partes separadas por si só ensina o conjunto inteiro (comparação com d3)? |
| **d8** | LoRA-completo *(etapa única)* | **O Grande Baseline**: Treinamento clássico. Os dados estão todos embaralhados do início ao fim em LoRA. **O Teste Final:** Qualquer modelo d1-d7 que não superar significativamente o d8 indica que a estruturação (currículo ou FF misto) não compensou o aumento de complexidade no treino. |

### Diagrama de Testes Essenciais

Para interpretar os resultados, os cruzamentos mais relevantes a se observar são:
1. **O Curriculum Funciona?** Compare `d3` (CL) x `d8` (Baseline misto).
2. **A Ordem Importa?** Compare `d3` (Crescente) x `d4` (Decrescente).
3. **Vale a pena consolidar no final?** Compare `d3` (CL com resumo final) x `d7` (CL sem etapa completa).
4. **O Full FT compensa o custo extra?** Compare `d1/d2` (CL+FF) x `d3` (CL LoRA only).
5. **A posição do Full FT muda o desfecho?** Compare `d1` (FF começo) x `d2` (FF final). E confirme se não é ilusão de hiperparâmetro avaliando contra `d5` (Sem CL, FF começo) e `d6` (Sem CL, FF final).
