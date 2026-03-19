# Documentação do Treinamento e Avaliação

## Descrição Geral

O pacote `treinar_unsloth.py` é uma ferramenta completa para fine-tuning de modelos LLM (Gemma-3, Deepseek, Llama, Qwen) usando Unsloth + TRL-SFTTrainer com configuração via arquivo YAML.

---

## Arquivos do Projeto

| Arquivo | Descrição |
|---------|-----------|
| `treinar_unsloth.py` | Script de treinamento e CLI (`LLMsTrainer`, `--treinar`, `--reset`) |
| `treinar_unsloth_avaliar.py` | Script de avaliação, inferência e exportação (`--info`, `--stats`, `--predict`, `--modelo`, `--merge`) |
| `treinar_unsloth_actions.py` | Ações de treinamento (`executar_treinar`, `executar_reset`, `executar_injetar_dicas`) |
| `treinar_unsloth_util.py` | Utilitários de configuração (`YamlTreinamento`, `ConfigGold`, `ConfigPredicao`) e helpers |
| `treinar_unsloth_dataset.py` | Gerenciamento de datasets: carga, divisão, validação (`DatasetTreinamento`) |
| `treinar_unsloth_logging.py` | Sistema centralizado de logging com níveis configuráveis |
| `treinar_unsloth_monitor.py` | Monitoramento contínuo de RAM/GPU |
| `treinar_unsloth_report.py` | Geração de relatórios em Markdown |
| `treinar_unsloth.md` | Esta documentação |

---

## Características Principais

### Implementadas ✅

- [x] Carregamento de modelo base com suporte a quantização (4-bit, 8-bit)
- [x] Suporte a LoRA (Low-Rank Adaptation) configurável
- [x] Detecção e continuação de modelos LoRA já treinados
- [x] Gerenciamento automático de checkpoints
- [x] Suporte a múltiplas GPUs (via CUDA_VISIBLE_DEVICES)
- [x] Modo infos detalhado (`--info`)
- [x] Modo de estatísticas de tokens com gráficos (`--stats`)
- [x] Modo teste para validar predições (`--modelo N`) com opção `--base`
- [x] Geração de predições em massa (`--predict`) com opção `--base`
- [x] Suporte a datasets em formado de pastas (pares .txt) e arquivos únicos (csv/parquet)
- [x] Divisão automática de datasets (treino, validação, teste) via YAML ou CSV
- [x] Chat template automático baseado no modelo
- [x] Criptografia de dados sensíveis (Fernet)
- [x] Validação automática de consistência de dados (IDs e arquivos pareados)
- [x] Implementar exportação de modelo (GGUF, Merge).
- [x] Gerar gráficos de evolução de Loss (treino vs validação) ao final do treino.

---

## Treinamento

O treinamento principal do modelo utiliza o banco de dados configurado no YAML para atualizar os pesos do modelo (criando adaptadores LoRA ou ajustando o modelo completo).

### Uso via Linha de Comando (`treinar_unsloth.py`)

```bash
python treinar_unsloth.py [CONFIG.yaml] [AÇÃO]
```

| Ação | Descrição |
|------|-----------|
| **(nenhuma)** | Modo interativo: seleciona YAML e ação via menu |
| `--treinar` | Inicia ou continua o treinamento |
| `--reset` | Limpa o treinamento atual (checkpoints e adaptador) |
| `--reset --treinar` | Limpa e treina do zero |
| `--dicas` | Injeta comentários de dicas no YAML de configuração |
| `--log-level LEVEL` | Define nível de log (DEBUG, INFO, WARNING, ERROR) |

**Exemplos de Uso:**
```bash
python treinar_unsloth.py                                 # Modo interativo completo
python treinar_unsloth.py config.yaml                     # Seleciona ação via menu
python treinar_unsloth.py config.yaml --treinar           # Inicia treinamento
python treinar_unsloth.py config.yaml --reset --treinar   # Limpa e treina do zero
```

### Arquivos de Saída e Monitoramento no Treinamento

| Pasta/Arquivo | Conteúdo |
|---------------|----------|
| `adapter_model.safetensors` | Pesos do LoRA treinado |
| `dados_automaticos.json` | Arquivo de cache interno (metadados resolvidos como `max_seq_length_auto` para acelerar reexecuções) |
| `treinamento/training_metrics.jsonl` | Métricas de treino (loss, learning rate, epoch) a cada step |
| `treinamento/hardware_metrics.jsonl` | Uso de recursos (CPU, RAM, GPU, Disco) coletado periodicamente |

**Monitoramento de Recursos:** O módulo `treinar_unsloth_monitor.py` é iniciado em background e registra ativamente uso de RAM, VRAM (GPU) e Disco, armazenados durante o tempo em que o treinamento estiver ativo.

---

## Avaliação

O script de avaliação tem o papel de analisar os dados processados e testar um modelo treinado (ou o base) mediante datasets específicos.

### Uso via Linha de Comando (`treinar_unsloth_avaliar.py`)

```bash
python treinar_unsloth_avaliar.py [CONFIG.yaml] [AÇÃO]
```

| Ação | Descrição |
|------|-----------|
| **(nenhuma)** | Modo interativo: seleciona YAML e ação via menu |
| `--info` | Informações detalhadas da configuração, datasets e modelo |
| `--stats` | Relatório estatístico de tokens e métricas de treinamento |
| `--predict` | Gera predições para todos os subsets (treino, validação, teste) |
| `--predict-treino` | Gera predições apenas do subset de treino |
| `--predict-validacao` | Gera predições apenas do subset de validação |
| `--predict-teste` | Gera predições apenas do subset de teste |
| `--modelo N` | Testa inferência interativa com N exemplos (padrão: 1) |
| `--merge` | Exporta modelo (merge LoRA + Base) |
| `--base` | Força uso do modelo base (ignora LoRA treinado) |
| `--quant METODO` | Quantização para merge (`16bit`, `4bit`, `q4_k_m`, `q8_0`) |
| `--log-level LEVEL` | Define nível de log (DEBUG, INFO, WARNING, ERROR) |

**Exemplos de Uso:**
```bash
python treinar_unsloth_avaliar.py                          # Modo interativo completo
python treinar_unsloth_avaliar.py config.yaml --info       # Informações detalhadas
python treinar_unsloth_avaliar.py config.yaml --stats      # Relatório estatístico
python treinar_unsloth_avaliar.py config.yaml --predict    # Predições de todos os subsets
python treinar_unsloth_avaliar.py config.yaml --predict --base  # Predições com modelo base
python treinar_unsloth_avaliar.py config.yaml --modelo 3   # Testar 3 exemplos
python treinar_unsloth_avaliar.py config.yaml --merge      # Exportar modelo
```

### Funcionalidades de Avaliação

#### Relatório de Estatísticas (`--stats`)
Gera análise detalhada do consumo de tokens (entrada e saída) por subset.
*   **Saída**: `{output_dir}/treinamento/relatorio_estatistico.md`
*   **Gráficos**: Gera `stats_tokens_boxplot.png` contendo boxplots comparativos de todos os subsets (Entrada e Saída).

#### Predição em Massa (`--predict`)
Gera respostas do modelo para os datasets configurados. Remove os arquivos da rodada anterior, mas apenas `.json` e `.txt` para fins de segurança, sem deletar outras coisas presentes na pasta de predição.
*   **Arquivos Gerados**:
    *   `{id}.txt`: O texto da resposta gerada.
    *   `{id}.json`: Metadados (tempo, tokens, preview do prompt).
    *   `resumo.json`: Estatísticas consolidadas da execução na respectiva pasta.

#### Inferência Interativa e Memória (`--modelo`)
* Gera gráfico `treinamento/memoria_predicao.png` constatando o uso de VRAM demandado pelo LLM ao instanciar as requisições gerativas.

---

## Principais Chaves do Arquivo YAML

O arquivo de configuração define todas as variáveis que controlam o treinamento e a avaliação. Para garantir compreensão fácil, as chaves mais utilizadas foram agrupadas e explicadas:

### 1. Modelo Base (`modelo`)
```yaml
modelo:
  nome: unsloth/Qwen2.5-1.5B-Instruct
  saida: ./modelos_treinados/qwen_padrao
```
*   `nome`: Repositório do modelo na no Hugging Face (preferencialmente os quantizados do _unsloth/_).
*   `saida`: Pasta principal local onde os pesos do LoRA, os logs de treino e as predições geradas serão salvos.

### 2. Configurações de Pastas (`pastas`)
```yaml
pastas:
  dataset:
    pasta: ./saidas_fold11/ext_qwen235b_11
    mascara: "*.txt"
  predicao:
    pasta: ./treino_simples/predict/ext_qwen1_5b_11
  entrada:
    dataframe: ./saidas/pecas_exportadas_textos.parquet
    dataframe_col: texto
    dataframe_id: id_peca
    prompt_template: './saidas/prompt_summa_raw.txt'
    tag_texto: '<<--TEXTO-->>'
```
*   `dataset.pasta`: **[Obrigatório]** É o seu *Gold Dataset* (respostas originais corretas). Usado para calcular o _loss_ (erro) que o modelo precisa aprender a reduzir.
*   `predicao.pasta`: Pasta onde as novas inferências/respostas escritas pelo seu modelo treinado serão registradas no modo `--predict`.
*   `entrada`: Especifica onde está localizado o texto de entrada do usuário. Geralmente um `dataframe` (.csv, .parquet) lincando as chaves em `dataframe_id` aos arquivos txt do `dataset.pasta` através de um super prompt definido pelo `prompt_template`.

### 3. Divisão do Dataset (`pastas.divisao`)
```yaml
  divisao:
    arquivo: "caminho/para/divisao.csv"
    proporcao:
      - treino: 0.7
      - validacao: 0.1
      - teste: 0.2
    validar_ids: true
```
*   `arquivo`: Se omitido, novos subconjuntos são embaralhados. Se preenchido com um arquivo que existe, fixa qual texto cai em qual subconjunto usando um CSV para reprodução determinística em outros testes.
*   `proporcao`: Define a porcentagem das amostras entre Treinamento, Validação e Teste (a soma obrigatoriamente é 1.0).
*   `validar_ids`: Se `true`, para a execução quando encontrar IDs no dataset que não constam no aquivo de divisão. Se `false`, o programa avança com um warning, servindo para trabalhar com fragmentos da base instalada localmente no repositório.

### 4. Modo de Aprendizagem e LoRA (`treinamento`)
```yaml
treinamento:
  train_on_responses_only: true
  lora:
    r: 16
    lora_alpha: 16
```
*   `train_on_responses_only`: Quando estabelecido `true`, o modelo recebe o prompt inteiro e a resposta esperada, mas a "nota" da punição do gradiente (o cálculo de loss) incidirá exclusivamente nos tokens respondidos por ele — evitando que o LoRA tente acidentalmente re-codificar as palavras dadas pelo usuário.
*   `lora.r` e `lora_alpha`: Controlam o tamanho (paramétricos ajustáveis) do seu adaptador (matriz injetada em subcamadas do Transformer), ajustando expressividade vs custo de memoria durante o refino do comportamento.

### 5. Janela de Contexto (`max_seq_length`)
```yaml
treinamento:
  max_seq_length: 0 # Opcional (se omitido ou 0, utiliza o Cálculo Dinâmico)
```
*   `max_seq_length`: Define a limitação de leitura máxima do modelo. **Regra de Ouro:** O tamanho do contexto reflete a janela _inteira_, logo, considera textualmente a soma do **Prompt/Entrada do Usuário + a Resposta/Saída do Assistente + formatação estrutural**. 
*   **Cálculo Dinâmico**: Se omitido ou definido como `0`, o módulo de carregamento lerá seu _Dataset_ inteiro prevendo as divisões para simular como determinado modelo formatará os tokens. Ele encontra o teto máximo de processamento exigido e enxuga as alocações de VRAM da GPU para aquele limite calculado. Os resultados dessa conta e de eventuais desdobramentos de *Curriculum Learning* são postos em um arquivo cache `dados_automaticos.json` ao lado do aprendizado.