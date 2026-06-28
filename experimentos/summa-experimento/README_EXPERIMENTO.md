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

> TODO: Avaliar se a extração pode considerar distância do cosseno e qtd de itens para extrair.

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
- **Configurações:** Reasoning=Médio / Verbose=Low

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

# Observações Importantes (Dicas de Treinamento)

- **Comparação de Extrações**: Para gerar divisões completas e consistentes, configure `ignorar_erro_extracao: false`. Se estiver como `true`, arquivos com erro de extração pelo modelo base serão ignorados. Ao manter `false`, eles são contabilizados e classificados (geralmente como "difíceis"), o que é o comportamento desejado para garantir que o modelo aprenda com seus erros de formato.
- **Full Finetuning (Ex: Protocolo C)**: 
  - **Precisão Automática:** O treinamento exige que o modelo seja carregado em pesos nativos e destravados de meia precisão (bfloat16). Para facilitar, ao definir `tipo: "full"` na divisão de currículo do YAML, o framework tem inteligência de **automaticamente forçar `nbits=16`** e recarregar o modelo da VRAM na precisão correta para você, desativando a quantização.
  - **Learning Rate (CRÍTICO):** A taxa de aprendizado para Full FT deve ser rigorosamente menor que a de LoRA. Enquanto LoRA funciona perfeitamente com `2e-4`, um Full FT explodirá os gradientes (Loss NaN ou Inf) se usar essa taxa. **Sempre use `5e-6` ou no máximo `1e-5`** para treinamentos Full (protocolos C e D).
  - **Contexto (max_seq_length):** Em sequências muito longas (ex: 8192, 16384), a instabilidade numérica é amplificada. Certifique-se de que a `learning_rate` está correta na divisão do YAML, pois ela sobrescreve a global. Opcionalmente, pode configurar `max_grad_norm: 0.3` na seção `treinamento` para clipar os gradientes se a instabilidade persistir.
- **Liger Kernel Inteligente (Múltiplas GPUs e SDPA)**:
  - O framework possui uma inteligência embarcada para garantir que o **Liger Kernel** calcule a perda corretamente e sem erros (como os problemas de `NaN` no `eval_loss` ou travamentos por *device mismatch*).
  - **Uso sem Flash Attention 2:** Se o `flash_attention_2` não estiver disponível no servidor, a atenção padrão (SDPA) será usada. O sistema desativará automaticamente a *Fused Cross Entropy* do Liger, evitando o bug de `NaN` em 16-bits.
  - **Uso com Múltiplas GPUs:** Ao treinar em 2 ou mais GPUs (`device_map="auto"`), a *Fused Cross Entropy* também é desligada automaticamente para permitir que os parâmetros fluam de forma segura e paralela.
  - Nas duas situações de adaptação, você continua se beneficiando da **economia de VRAM** nas outras operações do Liger (RMSNorm, RoPE, SwiGLU) sem precisar mexer em nada!

  ## Anotações e Decisões de Projeto
  Apesar de ser possível ultrapassar o limite de 32k no treinamento, foram apenas 22 instâncias de treino e 4 de validação removidas por excederem o tamanho em função dos tokens especiais de formatação.
  A melhor estratégia parece ser focar no treino de instâncias que respeitam o limite base do modelo para evitar criar complicadores para replicação do experimento, dado que qualidade deve estar acima de quantidade.
  > será avaliado como os exemplos acima de 32k se comportam nos testes
  > também serão avaliados como os exemplos de baixa similaridade com o GPT5 se comportam nos testes.

  **Decisões e Parâmetros Extraídos para a Dissertação:**
  - **Estratégia de Amostragem:** O Grupo 1 foi estratificado utilizando distância do cosseno (mínimo de 0.15 via modelo Athos) para garantir diversidade. O Grupo 2 adotou um corte temporal posterior (acórdãos de 25/05/2026) para evitar *data leakage* durante a avaliação.
  - **Destilação Segura:** A escolha da API do OpenRouter com diretrizes de *Zero Data Retention* garantiu a não retenção de dados e a manutenção da propriedade intelectual. Aceitou-se intencionalmente um trade-off de tempo de resposta via sufixo `:poor` para viabilizar custos do projeto.
  - **Aprendizado por Erro de Formato:** Manter `ignorar_erro_extracao: false` foi uma decisão consciente para incluir falhas prévias de formatação do modelo base como casos "difíceis", forçando o aprendizado corretivo dessa estrutura no fine-tuning.
  - **Estabilidade do Full Finetuning:** Foi diagnosticado que taxas comuns de LoRA explodem o gradiente no Full FT. Firmou-se o uso de *learning rates* reduzidas (`5e-6` a `1e-5`) e o clip de gradientes (`max_grad_norm: 0.3`) como requisitos para estabilizar sequências longas.
  - **Engenharia de Hardware/VRAM:** Para gerenciar incompatibilidades de *device mismatch* e `NaN` em 16-bits, optou-se pela desativação cirúrgica da *Fused Cross Entropy* do Liger Kernel em setups multicore ou sem Flash Attention 2, sem perder o ganho de memória nas camadas subjacentes.