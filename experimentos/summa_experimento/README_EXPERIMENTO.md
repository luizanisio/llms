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
python ../../src/util_vllm_batch.py --config config_batch.yaml
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
- **Full Finetuning (Ex: Protocolo C)**: Ao realizar Full Finetuning, o treinamento exige que o modelo seja carregado em pesos completos ou meia precisão. É fundamental configurar `bits: 16` no YAML (vai dar erro se tentar aplicar Full Finetuning em um modelo quantizado em 4-bits ou 8-bits).
- **Liger Kernel Inteligente (Múltiplas GPUs e SDPA)**:
  - O framework possui uma inteligência embarcada para garantir que o **Liger Kernel** calcule a perda corretamente e sem erros (como os problemas de `NaN` no `eval_loss` ou travamentos por *device mismatch*).
  - **Uso sem Flash Attention 2:** Se o `flash_attention_2` não estiver disponível no servidor, a atenção padrão (SDPA) será usada. O sistema desativará automaticamente a *Fused Cross Entropy* do Liger, evitando o bug de `NaN` em 16-bits.
  - **Uso com Múltiplas GPUs:** Ao treinar em 2 ou mais GPUs (`device_map="auto"`), a *Fused Cross Entropy* também é desligada automaticamente para permitir que os parâmetros fluam de forma segura e paralela.
  - Nas duas situações de adaptação, você continua se beneficiando da **economia de VRAM** nas outras operações do Liger (RMSNorm, RoPE, SwiGLU) sem precisar mexer em nada!