# Puil Mini Experimento

Este diretório contém os scripts e configurações para um mini experimento rápido com modelos locais treinados ou base (como o Qwen2.5) para analisar extrações e respostas abertas no contexto jurídico (Classe Processual PUIL).

Esse projeto é muito simples e tem como objetivo principal **testar o ambiente e validar a corretude dos códigos de treinamento, extração e comparação**. Por ser pequeno, com uma janela de contexto de apenas 256 tokens, pode ser rodado em uma GPU de 12GB (sendo o maior limitador a etapa Full Fine-tuning, que fica no limite da RTX 3060).

---

## 🚀 Roteiro de Replicação (Passo a Passo)

### 1. Configuração do Ambiente
> ⚠️ **Antes de começar:** Certifique-se de ter configurado o ambiente Python e as bibliotecas (PyTorch, vLLM, Flash-Attention, etc.).  
👉 **Leia as instruções completas no arquivo central: [experimentos/README.md](../README.md)**

### 2. Preparação do Dataset
Diferente de experimentos maiores, os dados do PUIL já vêm pré-processados e encontram-se prontos na pasta `dados/` (ex: `entrada_puil.csv` e `prompt_puil.txt`). Não há necessidade de executar nenhum script complexo de pré-processamento para iniciar os testes.

### 3. Extração de Dados com Modelos Base (Baseline)
Embora os arquivos atuais foquem na extração após o treinamento, uma extração baseline (usando o modelo base original em zero-shot) pode ser facilmente realizada. Para isso, crie um YAML similar aos outros de extração (como o da etapa 05), apenas removendo o caminho do adaptador LoRA.
*Nota: Como o objetivo deste mini-experimento é validar a pipeline de execução dos códigos, a etapa de baseline pode ser pulada sem prejuízos técnicos se o seu foco for apenas checar o fluxo de fine-tuning.*

### 4. Treinamento
Você pode rodar o ajuste do modelo tanto usando adaptação LoRA quanto Full Fine-tuning. Os scripts estão preparados para rodar localmente, o que é ótimo para testar a infraestrutura de treino de forma ágil.
Exemplos de configurações disponíveis:
- `04_treinar_puil_lora.yaml`
- `04_treinar_puil_full.yaml`

Para rodar (exemplo em shell):
```bash
bash job_treinar_puil.sh
# O script internamente chamará comandos como:
# python ../../src/treinar_unsloth.py --config 04_treinar_puil_lora.yaml
```

### 5. Extração de Dados com Modelos Treinados
Após o treino, use o motor de inferência (vLLM em batch) para extrair as respostas utilizando os novos pesos treinados. O processamento em lote é invocado através das configurações da etapa 05.
```bash
python ../../src/util_vllm_batch.py --config 05_extracao_treino_puil_lora.yaml
python ../../src/util_vllm_batch.py --config 05_extracao_treino_puil_full.yaml
```
Os arquivos gerados (`.parquet`) contendo as predições do modelo serão salvos na pasta `saidas/`.

*Dica rápida:* Não é necessário colocar caminhos absolutos nas configurações. O experimento pode aproveitar a chave `misc.pastas_modelos_treinados` nos arquivos `.yaml` para buscar dinamicamente as saídas na sua rede local ou máquina!

### 6. Geração das Comparações dos Resultados
Por último, para validar se a lógica de pontuação está operando bem, use o arquivo `06_comparar_treinos.yaml` para comparar os resultados gerados nos arquivos `.parquet` com o gabarito oficial (a coluna *target* do dataset original).
```bash
python ../../src/comparar_extracoes.py --config 06_comparar_treinos.yaml
```

---

## 🛠️ Comparação de Saídas de Texto Livre (Não-JSON)

Se o modelo foi treinado para gerar um texto livre em vez de um JSON estruturado (como é o caso proeminente desse mini experimento), a engine de avaliação possui suporte nativo para converter a saída de texto puro em um JSON padronizado no formato `{"resposta": <texto puro>}`, permitindo o uso imediato do pacote de comparação. 

Para habilitar isso, certifique-se de configurar a flag `saida_json: false` no YAML da comparação (`06_comparar_treinos.yaml`), por exemplo:

```yaml
configuracao_comparacao:
  campos_parquet:
    saida_json: false
  campos:
    bertscore:
      - "(global)"
```

**Como funciona internamente?**
- Ao ativar `saida_json: false`, o motor de extração lê os arquivos de resultados dos modelos e injeta todo o conteúdo bruto dentro de uma chave virtual chamada `"resposta"`.
- A engine usará esse conteúdo íntegro para calcular as métricas agregadas sob a diretiva de campo `(global)`, como a similaridade semântica (BERTScore, SBERT) e métricas lexicais (ROUGE), ignorando eventuais falhas comuns de parsing de JSON.