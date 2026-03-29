# Processamento Completo de LLMs (Treinamento, Predição e Avaliação)

## 🎯 Objetivo do Módulo (Visão Geral)
O pacote `treinar_unsloth.py` forma o núcleo de um ecossistema completo para fine-tuning, inferência e avaliação estatística de Modelos LLM (como Gemma-3, Qwen, Deepseek e Llama). O foco central é permitir a orquestração desde o Treinamento SFT unificado rodando sob baixo custo (Unsloth) atravessando até um Pipeline de Predição local e remota (vLLM, Ollama), tudo sem a necessidade de codificação manual — orquestrado unicamente por perfis YAML de alto nível.

## 🚀 Funcionalidades Principais
- **Treinamento Multiestágios (Curriculum Learning)**: Defina arquivos diferentes para estágios subsequentes do seu aprendizado limitando o loss ou épocas de avanço automático para refino iterativo sem interrupção.
- **Inferência Multimotores OOP**: Motor plug-n-play para avaliação: usa nativamente o pipeline HuggingFace, aloca de forma extremamente rápida infraestruturas locais com motor compatível vLLM ou se integra para despachar a API Local do Ollama.
- **Ecossistema Resiliente a Checkpoints**: Gestão autônoma de checkpoints LoRA e travas de conclusão de Curriculum, impedindo que reexecuções apaguem acidentalmente treinamentos ou subescrevam frentes já avaliadas com sucesso.
- **Dados Sensíveis e Proteção**: Suporte granular para ingestão e decodificação na hora de Parquets Criptografados (Fernet). Fail-fast garantido para evitar treinamentos inteiros usando base mal formatada e protegida.

## 🛠️ Como Executar (Início Rápido)
A interface é unificada por uma TUI rica rodando com Menu Interativo caso você não informe os parâmetros.
```bash
# === MODO MENU INTERATIVO (Recomendado) ===
python src/treinar_unsloth.py
python src/treinar_unsloth_avaliar.py

# === TREINAMENTO ===
python src/treinar_unsloth.py meu_config.yaml --treinar

# === AVALIAÇÃO e ESTATÍSTICA ===
# Gera insights visuais profundos sobre o Dataset, Context Length ideal, uso de tokens.
python src/treinar_unsloth_avaliar.py meu_config.yaml --stats

# === PREDIÇÃO EM LOTE ===
# Faz o modelo prever respostas para todos os datasets de teste (usando o motor super-rápido vLLM por padrão)
python src/treinar_realizar_predicoes.py meu_config.yaml --engine vllm --predict
```

## ⚙️ Configuração (O Arquivo YAML)
O comportamento de todos os scripts transita em volta do seu YAML. Os principais pontos que os pesquisadores devem observar e configurar para um novo experimento são:

- **`modelo`**:
  - `base_model_name`: Identificador no Hugging Face (ex: `Qwen/Qwen2.5-1.5B-Instruct`) ou caminho local de um fallback preexistente.
  - `saida`: Pasta raiz onde **tudo** que seu modelo evolutivo produzir nascerá. Os checkpoints ficarão em `/chkpt`, os gráficos em `/treinamento`, e as respostas na pasta de sua escolha.
  - `ollama`: Se for avaliar um modelo consolidado convertido lá.

- **`curriculum` (Fluxo de Entrada e Avaliação)**: A subchave principal de arquitetura experimental.
  - `saida`: Local (pasta/Parquet) de onde o `gold_standard` perfeito (resposta humana alvo) mora na máquina. Obrigatório.
  - `entrada`: Local das entradas não parseadas do Dataset. Suporta criptografia. Pode pular se o par parquet estiver completo em `saida`.
  - **`divisao`**: (Lista). Essencial. Descreve as Etapas. Se você quer treinar um LORA de X epocas, defina:
    - `arquivo`: CSV exato da gaveta fracionada alvo produzida lá no pacote `comparar_extracoes`
    - `tipo`: lora ou full
    - `pace_epochs`: (ou pace_loss) quando que a roda deste subconjunto deve encerrar e o Early-Stopping atrelado pular.
    - `max_seq_length`: Importante colocar fixo de acordo com colunas `token_total` geradas.

- **`treinamento`**:
  - `batch_size`: Suporta `efetivo: N` para autoavaliar quantas GPUs o torch tem na ponta e calcular perfeitamente o Gradient Acceleration Substep garantindo reprodutibilidade independentemente da topologia física!
  - `train_on_responses_only`: (true/false) Se a perda da atenção deve pular o lado Prompter (Usuário). Ótimo para modelos instruct.

## 🔄 Como Replicar Experimentos e Reutilizar Código
- **Retomada Autônoma**: Se um experimento for interrompido, baste re-rodar `--treinar`. O script escaneará `/chkpt`, subirá o state de onde parou as loss das métricas, e engatará exatamente na Época ou Pace que sucumbiu.
- **Versionamento Embutido**: Sem precisar versionar pelo git. A cada iteração ou "Resume" válido da sua frente, dentro da pasta `saida` alçada em `treinamento/treinamento_config`, viverão `.yaml` prefixados como cópia física perfeita congelada em tempo dos specs do dia (`(v001)`, `(v002)...`).
- Todo **log** e **visualização** ficará eternizado perfeitamente grafado em `<saida>/treinamento`. Os perfis de RAM consumida, Tokens Processados e curvas de convergência residirão em `/treinamento/*hardware*, *loss*`.