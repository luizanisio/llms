# Passo 01 - Dataset

O primeiro passo é extrair os textos dos acórdãos do portal de dados abertos do STJ. Utilizamos o CKAN para gerar um arquivo `.parquet` com os metadados e os textos selecionados.

Usando o script `util_ckan.py`, utilizamos o arquivo de configuração da extração (`config_extracao.yaml`) para gerar o arquivo `.parquet` contendo os textos extraídos e os metadados dos processos.

**Comando:**
```bash
python ../../src/util_ckan.py --config config_extracao.yaml
```

**Composição dos dados:**
- **Grupo 1:** Dados selecionados para o experimento com 22k documentos estratificados com distância do cosseno de no mínimo `0.15` usando o modelo Athos do STJ.
- **Grupo 2:** Novos documentos posteriores ao treinamento de qualquer um dos modelos utilizados. Conjunto escolhido: acórdãos publicados em `25/05/2026`.

> TODO: Implementar na extração a opção de considerar a distância do cosseno e qtd de itens para extrair, permitindo reprodutibilidade de diversidade diretamente no script de extração.

---

# Passo 02 - Geração de Dados de Treino e Teste

Nesta etapa, geramos os dados processados utilizando os diferentes modelos configurados.

## 1. Professor: OpenRouter (Qwen3-235B)
A extração de dados via API OpenRouter para a destilação de conhecimento do modelo Qwen3-235B assegura a propriedade intelectual ao pesquisador, permitindo o uso para treinamento de modelos, conforme os Termos de Serviço vigentes. Foram configuradas as diretrizes restritivas de privacidade, incluindo o opt-out de compartilhamento de dados e o uso de rotas de *Zero Data Retention*, para impedir o aprimoramento de modelos de terceiros.

### Configurações do modelo professor:
- **Versão do modelo:** `qwen/qwen3-235b-a22b-2507:poor`
- **Provider:** `{"quantizations": ["fp8"]}`
- o sufixo :poor ativa no OpenRouter a busca por deployments mais baratos para o mesmo modelo, reduzindo a prioridade do pedido e podendo aumentar o tempo para extração.

**Variáveis de Ambiente:**
- `PESSOAL_OPENROUTER_API_KEY`
- `OPENROUTER_EXTRA`

**Comando:**
```bash
python ../../src/util_vllm_batch.py --config config_batch_235b.yaml
```

## 2. Aluno: Modelo local com vLLM (CISIA - PUCPR)
- **Versão do modelo:** `qwen/Qwen2.5-7B-Instruct`
- **max_model_len:** 38912
- **tensor_parallel_size:** 1
- **dtype:** "auto"
- **quantization:** bitsandbytes
- **load_format:** bitsandbytes
- 1xGPU H100

**Variáveis de Ambiente (opcional):**
- `HF_TOKEN`

**Comando:**
```bash
python ../../src/util_vllm_batch.py --config arquivo_config_batch.yaml
```

## 3. Referência: GPT-5
- **API:** Azure
- **Versão do modelo:** `gpt-5-2025-08-07`
- **Configurações:** Reasoning=Médio / Verbose=Low (extração de referência); como **juiz LLM** usa Reasoning=High / Verbose=Low (ver `avaliacao_llm_humana/02_extracao_70.yaml`)

**Variáveis de Ambiente:**
- `OA_KEY`
- `OA_CONTROLE`

**Comando:**
```bash
python ../../src/util_vllm_batch.py --config config_batch_gpt5.yaml
```

## 4. Comparação Sabiá 4
- **API:** Maritaca
- **Versão do modelo:** `sabiá-4`
- **Treinamento:** até 08/2024
- Janela 128k
- Nome: sabia-4 ou sabia-4-2026-01-06

# Passo 03 - Realizar comparação entre o professor (ou dados raw) e o modelo base sem treinamento
A realização da comparação gera um arquivo de divisão de treino, teste e validação, que será utilizado nas próximas etapas. 
Outro resultado da comparação é o nível de dificuldade de cada instância de acordo com a performance do modelo base e da quantidade de chaves do item.
Esse arquivo pode ser usado diretamente para o arquivo yaml de treinamento para configuração dos níveis de dificuldade e alvo (treino, teste e avaliação)

---

# Passo 04 - Treinamento

O modelo-alvo é o **Qwen2.5-7B-Instruct**, que permite validar o framework CL+PT em um modelo de maior capacidade que o experimento PubMed (1.5B), testando escalabilidade.

> 📖 **A malha completa é a mesma dos três experimentos**: 28 protocolos de treinamento (`b`, `c`, `b16` e `d1`–`d25`) + o modelo sem ajuste (`A`), mais o controle auxiliar `b16r8` e as réplicas de calibração `d1a`/`d1b`. O mapa completo, as seis questões de pesquisa e os 12 recortes estão em [README_protocolos.md](../README_protocolos.md) e na tabela gerada automaticamente [RESUMO_EXPERIMENTOS.md](../RESUMO_EXPERIMENTOS.md). As tabelas abaixo cobrem apenas as cinco camadas iniciais (`b`, `c`, `d1`–`d10`); os protocolos `b16`, `b16r8`, `d11`–`d25`, `d1a` e `d1b` seguem os YAMLs homônimos do PubMed/SemClinBr.

As cinco camadas iniciais:

## Perguntas de pesquisa

| Pergunta | Descrição | Recortes (`06_compara_todos.yaml`) |
|---|---|---|
| **Q1** | Efeito do ajuste fino: FT (qualquer variante) produz ganho sobre baseline zero-shot? | `Q1_ajuste_fino` |
| **Q2** | Efeito do CL: a progressão de dificuldade melhora sobre FT direto — e o ganho persiste quando a segmentação é controlada? | `Q2a_cl_controlado`, `Q2b_cl_puro`, `Q2c_granularidade` |
| **Q3** | Direção do escalonamento: FF→LoRA vs LoRA→FF produz desempenhos distintos, com e sem CL? | `Q3a_direcao_com_cl`, `Q3b_direcao_sem_cl` |
| **Q4** | Decomposição: o ganho vem do CL, do escalonamento ou da combinação? | `Q4a_decomposicao`, `Q4b_unfreeze` |
| **Q5** | Direção do currículo: a ordem fácil→difícil importa vs difícil→fácil? | `Q5_anti_curriculo` |
| **Q6** | Custo da fronteira entre etapas: desaparece quando a fronteira é virtual? | `Q6a_fusao`, `Q6b_fusao_granular`, `Q6c_transicao_regime` |

## Camada 1 — Baselines (sem CL, sem escalonamento)

| Proto | Modo | Etapas | Arquivo treino |
|---|---|---|---|
| **A** | Zero-shot (sem treino) | — | — |
| **b** | LoRA direto (dataset completo) | LoRA-completo | `04_treinar_b.yaml` |
| **c** | FF direto (dataset completo) | FF-completo | `04_treinar_c.yaml` |

## Camada 2 — Experimentais (CL + escalonamento de capacidade)

| Proto | Pace | Direção | Etapas | Arquivo treino |
|---|---|---|---|---|
| **D1** | etapas | FF→LoRA | FF-fácil → LoRA-médio → LoRA-difícil → LoRA-completo | `04_treinar_d1.yaml` |
| **D2** | etapas | LoRA→FF | LoRA-fácil → LoRA-médio → LoRA-difícil → FF-completo | `04_treinar_d2.yaml` |
| **D3** | acumulado | FF→LoRA | FF-(≤3) → LoRA-(≤7) → LoRA-tudo | `04_treinar_d3.yaml` |
| **D4** | acumulado | LoRA→FF | LoRA-(≤3) → LoRA-(≤7) → FF-tudo | `04_treinar_d4.yaml` |

## Camada 3 — Ablação: escalonamento sem CL

| Proto | Modo | Etapas | Arquivo treino |
|---|---|---|---|
| **D5** | FF→LoRA, sem progressão | FF-completo → LoRA-completo | `04_treinar_d5.yaml` |
| **D6** | LoRA→FF, sem progressão | LoRA-completo → FF-completo | `04_treinar_d6.yaml` |

## Camada 4 — Ablação: CL sem escalonamento (LoRA-only)

| Proto | Pace | Etapas | Arquivo treino |
|---|---|---|---|
| **D7** | etapas | LoRA-fácil → LoRA-médio → LoRA-difícil → LoRA-completo | `04_treinar_d7.yaml` |
| **D8** | acumulado | LoRA-(≤3) → LoRA-(≤7) → LoRA-tudo | `04_treinar_d8.yaml` |

## Camada 5 — Ablação: anti-currículo (direção inversa, LoRA-only)

| Proto | Pace | Etapas | Arquivo treino |
|---|---|---|---|
| **D9** | etapas | LoRA-completo → LoRA-difícil → LoRA-médio → LoRA-fácil | `04_treinar_d9.yaml` |
| **D10** | acumulado | LoRA-(>7) → LoRA-(>3) → LoRA-tudo | `04_treinar_d10.yaml` |

## Design fatorial

|  | Sem escal. | FF→LoRA | LoRA→FF |
|---|---|---|---|
| **Sem CL** | b, c | D5 | D6 |
| **CL por etapas** | D7 | D1 | D2 |
| **CL acumulado** | D8 | D3 | D4 |
| **Anti-CL etapas** | D9 | — | — |
| **Anti-CL acumulado** | D10 | — | — |

## Arquivos de comparação

Os relatórios temáticos foram consolidados em um único arquivo com recortes internos (ver [README_protocolos.md §5](../README_protocolos.md)):

| Arquivo | Modelos incluídos | Propósito |
|---|---|---|
| `06_compara_todos.yaml` | A, b, b16, b16r8, c, D1–D25 | Relatório completo: 12 recortes (Q1–Q6) + `Panorama_Geral` (descritivo) |
| `06_compara_todos_parcial.yaml` | idem, com os protocolos ainda em treinamento comentados | Relatório incremental |
| `06_compara_d1ab.yaml` | D1, D1a, D1b | Calibração da ROPE por campo (`00_rope_sugerido.md`) — transcrever para `rope_por_campo` |

---

# Passo 05 - Extração com Modelos Treinados

Após a conclusão do treinamento, os modelos ajustados estão salvos em `treinos/`. Para extrair as informações do conjunto de teste com esses pesos treinados, usamos as configurações `05_extracao_*_teste.yaml`.

```bash
# Exemplo para protocolo D1:
python ../../src/util_vllm_batch.py --config 05_extracao_d1_teste.yaml
```

---

# Passo 06 - Comparação dos Resultados

Comparamos as extrações geradas contra o gabarito do professor (Qwen3-235B-A22B-2507). O processo lê os parquets de saída, aplica métricas automáticas (BERTScore e SBERT com o encoder jurídico `stjiris/bert-large-portuguese-cased-legal-mlm-mkd-nli-sts-v1`, ROUGE-L/2, Levenshtein) sobre os campos e os agregados virtuais `Likert` (6 campos críticos) e `MetaDados` (4 campos literais), e roda a análise estatística por recorte: **bayesiana** (`baycomp.CorrelatedTTest`, ROPE calibrada por campo, limiar 0,95) como análise principal e frequentista (Friedman/Wilcoxon-Holm/Nemenyi) apenas exploratória. A avaliação de qualidade pelo juiz LLM (Likert 1–4, 3 rodadas) entra quando `campos_parquet.avaliacao` apontar para as avaliações (ver `avaliacao_llm_humana/`).

```bash
# Exemplo:
python ../../src/comparar_extracoes.py --config 06_compara_todos.yaml
```

---

# Passo 07 - Validação out-of-time (documentos de 2026)

O SUMMA é o único dos experimentos com um conjunto posterior ao corte de
treinamento dos modelos envolvidos: o **fold 12**, com acórdãos publicados em
25/05/2026. Esta seção descreve como ele é usado.

## 7.1 Por que ele existe e onde ele está hoje

Todos os 32 YAMLs de treino e os 31 de extração de teste declaram
`dataset_filtro: {"fold": "<=10"}` no bloco `curriculum.entrada` / `entrada.filtro`.
O filtro atua no carregamento, antes de qualquer etapa, e é por isso que o corpus
efetivo do experimento é **19.712** e não as 22.155 linhas do arquivo de divisão:

| Conjunto | Na divisão | Após `fold <= 10` |
|---|---:|---:|
| Treino | 15.508 | **13.788** |
| Validação | 2.217 | **1.976** |
| Teste | 4.430 | **3.948** |
| **Total** | 22.155 | **19.712** |

Os 2.443 descartados são os folds 11 e 12. O **fold 12 tem 471 documentos**, todos
com gabarito do professor (Qwen3-235B) já gerado e JSON válido — nenhum deles
entrou em treino, validação ou teste de nenhum protocolo.

> ⚠️ **A coluna `alvo` do fold 12 é lixo.** O sorteio da divisão marcou 327 como
> treino, 48 como validação e 94 como teste, mas o filtro `fold <= 10` os removeu
> antes de qualquer uso. Os **471 são igualmente inéditos**. Filtrar por
> `alvo: teste` aqui reduziria o conjunto a 94 documentos e jogaria fora 80% do
> poder estatístico sem nenhum ganho.

## 7.2 Desenho: duas análises independentes

O fold 12 **não é somado** ao conjunto de teste. Somar os 471 aos 3.948 estreitaria
o posterior em apenas ~5% (√(3948/4419) = 0,945) e, em troca, eliminaria a
possibilidade de qualquer afirmação out-of-time. São duas análises separadas, cada
uma com sua própria ROPE:

| | Análise principal | Confirmatória out-of-time |
|---|---|---|
| Documentos | 3.948 (fold ≤ 10, `alvo: teste`) | 471 (fold 12, **todos**) |
| Período | 2022–2024 | 2026 (posterior ao cutoff) |
| Recortes | os 12 — 104 contrastes | 2 pré-especificados — 12 contrastes |
| Protocolos | todos | `B`, `B16`, `C`, `D16`, `D17`, `D18`, `D19`, `D20` |
| ROPE | calibrada em D1/D1a/D1b sobre **esses 3.948** | recalibrada em D1/D1a/D1b sobre **esses 471** |
| Papel | os vereditos do trabalho | replicação sob documentos não memorizáveis |

**Por que só dois recortes.** Rodar os 104 contrastes em 471 documentos gastaria o
holdout em multiplicidade. Os dois escolhidos carregam as afirmações centrais e já
trazem o próprio controle:

- **`Q2a_cl_controlado`** (`B16`, `D16`, `D18`, `D17`) — ordenação × efeito de bloco
  × custo de fronteira. O `D18` (blocos aleatórios) é o controle negativo: se a
  ordenação replicar em 2026 e os blocos aleatórios não, a confirmação é forte.
- **`Q4b_unfreeze`** (`B`, `C`, `D20`, `D19`) — sinergia CL+PT por descongelamento.

**Por que recalibrar a ROPE.** O desvio-padrão posterior escala com 1/√n: de 3.948
para 471 ele fica ~2,9× mais largo. Aplicar aos 471 a ROPE calibrada no conjunto
grande faria quase todos os pares saírem `incerto` — o que **pareceria falha de
replicação quando é só falta de poder**. A ROPE de 2026 será mais larga, e isso
precisa ser declarado antes de rodar: a análise confirmatória detecta efeitos
grandes e devolve `incerto` para os moderados.

## 7.3 Roteiro

1. **Criar os YAMLs de extração de 2026** para os 8 protocolos dos recortes
   (`B`, `B16`, `C`, `D16`, `D17`, `D18`, `D19`, `D20`) e para as 3 réplicas da
   calibração (`D1`, `D1a`, `D1b`) — 11 arquivos. Partir dos
   `05_extracao_*_teste.yaml` e trocar **apenas** o bloco de filtro e a saída:
   ```yaml
   entrada:
     filtro:
       dataset_filtro: {"fold": 12}   # sem filtro_externo: a coluna alvo não vale aqui
   saida:
     arquivo: "saida/saida_qwen7b(d19)_2026.parquet"
   ```
2. **Rodar as 11 extrações** sobre os 471 documentos (`util_vllm_batch.py`).
   Nenhum retreino: os pesos são os mesmos da análise principal.
3. **Calibrar a ROPE de 2026** com um `06_compara_d1ab_2026.yaml` — cópia do
   `06_compara_d1ab.yaml` apontando para as saídas `*_2026.parquet`, com
   `calibracao_rope: true` e o mesmo filtro `{"fold": 12}`. Transcrever
   `00_rope_sugerido.md` → `rope_por_campo`.
4. **Rodar a comparação** com um `06_compara_2026.yaml` contendo só os recortes
   `Q2a_cl_controlado` e `Q4b_unfreeze` e a ROPE do passo 3.
5. **Reportar os dois períodos lado a lado**, mesmos contrastes em duas colunas
   (2022–2024 / 2026), marcando que a coluna de 2026 opera sob ROPE mais larga.
   Distinguir no texto `incerto por poder` de `equivalente` — são leituras
   diferentes.

> O passo 4 é **de uso único**. O valor do fold 12 vem de ele nunca ter
> participado de nenhuma decisão. Rodar, não gostar do resultado e reselecionar
> protocolos ou recortes destrói o holdout.

## 7.4 Leitura dos resultados

**Deriva de distribuição não invalida os contrastes.** Acórdãos de 2026 podem
diferir dos de 2022–2024 por outros motivos além do tempo (matéria, redação,
composição do colegiado). Mas a análise é sobre **diferenças pareadas por
documento** entre protocolos: uma deriva que afete todos os protocolos igualmente
cancela no delta. O que ela muda são os escores absolutos, não os contrastes — a
menos que interaja com o protocolo, o que já seria achado. Reportar os escores
absolutos dos dois períodos torna a deriva visível e mensurável.

**Não comparar escores absolutos entre períodos sem normalizar por faixa.** A
estratificação de dificuldade do fold 12 é 162 fácil / 179 médio / 128 difícil
(34/38/27%), contra os 30/40/30 do conjunto principal, porque os percentis foram
calculados globalmente. Isso não afeta os contrastes (pareados dentro do mesmo
conjunto), mas desaconselha a leitura direta entre períodos.

**A referência de 2026 é o professor, não humano.** O que a análise confirma é que
*a concordância com o professor persiste em documentos posteriores ao cutoff* — não
que a qualidade persiste. Mesma ressalva de referência fraca da análise principal.

**O que isso entrega.** Um controle de contaminação out-of-time sobre documentos
que nenhum dos modelos envolvidos poderia ter memorizado, respondendo
antecipadamente à crítica de que os ganhos do currículo vieram do pré-treino.

---

# Observações Importantes (Dicas de Treinamento)

- **Comparação de Extrações**: Para gerar divisões completas e consistentes, configure `ignorar_erro_extracao: false`. Se estiver como `true`, arquivos com erro de extração pelo modelo base serão ignorados. Ao manter `false`, eles são contabilizados e classificados (geralmente como "difíceis"), o que é o comportamento desejado para garantir que o modelo aprenda com seus erros de formato.
- **Full Finetuning (Ex: Protocolo C)**: 
  - **Precisão Automática:** O treinamento exige que o modelo seja carregado em pesos nativos e destravados de meia precisão (bfloat16). Para facilitar, ao definir `tipo: "full"` na divisão de currículo do YAML, o framework tem inteligência de **automaticamente forçar `nbits=16`** e recarregar o modelo da VRAM na precisão correta para você, desativando a quantização.
  - **Learning Rate (CRÍTICO):** A taxa de aprendizado para Full FT deve ser rigorosamente menor que a de LoRA. Enquanto LoRA funciona perfeitamente com `2e-4`, um Full FT explodirá os gradientes (Loss NaN ou Inf) se usar essa taxa. **Sugestão: use `5e-6` ou no máximo `1e-5`** para treinamentos Full (protocolos C e D).
  - **Contexto (max_seq_length):** Em sequências muito longas (ex: 8192, 16384), a instabilidade numérica é amplificada. Certifique-se de que a `learning_rate` está correta na divisão do YAML, pois ela sobrescreve a global. Opcionalmente, pode configurar `max_grad_norm: 0.3` na seção `treinamento` para clipar os gradientes se a instabilidade persistir.
- **Liger Kernel Inteligente (Múltiplas GPUs e SDPA)**:
  - O framework possui uma inteligência embarcada para garantir que o **Liger Kernel** calcule a perda corretamente e sem erros (como os problemas de `NaN` no `eval_loss` ou travamentos por *device mismatch*).
  - **Uso sem Flash Attention 2:** Se o `flash_attention_2` não estiver disponível no servidor, a atenção padrão (SDPA) será usada. O sistema desativará automaticamente a *Fused Cross Entropy* do Liger, evitando o bug de `NaN` em 16-bits. É recomendado instalar o flash attention 2 para reduzir consideravelmente o uso de VRAM.
  - **Uso com Múltiplas GPUs:** Ao treinar em 2 ou mais GPUs (`device_map="auto"`), a *Fused Cross Entropy* também é desligada automaticamente para permitir que os parâmetros fluam de forma segura e paralela.
  - Nas duas situações de adaptação, você continua se beneficiando da **economia de VRAM** nas outras operações do Liger (RMSNorm, RoPE, SwiGLU) sem precisar mexer em nada, mas o flash attention 2 combinado com o liger kernel vão economizar VRAM e permitir o treino FF do Qwen 7b em uma H200.

  ## Anotações e Decisões de Projeto
  Apesar de ser possível ultrapassar o limite de 32k no treinamento, foram apenas 22 instâncias de treino e 4 de validação removidas por excederem o tamanho em função dos tokens especiais de formatação.
  A melhor estratégia parece ser focar no treino de instâncias que respeitam o limite base do modelo para evitar criar complicadores para replicação do experimento, dado que qualidade deve estar acima de quantidade.
  > será avaliado como os exemplos acima de 32k se comportam nos testes
  > também serão avaliados como os exemplos de baixa similaridade com o GPT5 se comportam nos testes.

  **Decisões e Parâmetros Extraídos para a Dissertação:**
  - **Estratégia de Amostragem:** O Grupo 1 foi estratificado utilizando distância do cosseno (mínimo de 0.15 via modelo Athos/STJ) para garantir diversidade. O Grupo 2 adotou um corte temporal posterior (acórdãos de 25/05/2026) para evitar *data leakage* durante a avaliação.
  - **Destilação Segura:** A escolha da API do OpenRouter com diretrizes de *Zero Data Retention* garantiu a não retenção de dados e a manutenção da propriedade intelectual. Aceitou-se intencionalmente um trade-off de tempo de resposta via sufixo `:poor` para viabilizar custos do projeto.
  - **Aprendizado por Erro de Formato:** Manter `ignorar_erro_extracao: false` foi uma decisão consciente para incluir falhas prévias de formatação do modelo base como casos "difíceis", forçando o aprendizado corretivo dessa estrutura no fine-tuning.
  - **Estabilidade do Full Finetuning:** Foi diagnosticado que taxas comuns de LoRA explodem o gradiente no Full FT. Firmou-se o uso de *learning rates* reduzidas (`5e-6` a `1e-5`) e o clip de gradientes (`max_grad_norm: 0.3`) como requisitos para estabilizar sequências longas.
  - **Engenharia de Hardware/VRAM:** Para gerenciar incompatibilidades de *device mismatch* e `NaN` em 16-bits, optou-se pela desativação cirúrgica da *Fused Cross Entropy* do Liger Kernel em setups multicore ou sem Flash Attention 2, sem perder o ganho de memória nas camadas subjacentes.