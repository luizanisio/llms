# Puil Mini Experimento

Este diretório contém os scripts e configurações para um mini experimento rápido com modelos locais treinados ou base (como o Qwen2.5) para analisar extrações e respostas abertas no contexto jurídico (Classe Processual PUIL).

Esse projeto é muito simples, mas visa exemplificar o pipeline de extração, treinamento e comparação de resultados.
Por ser pequeno, com uma janela de contexto de 256 tokens, pode ser rodado em uma GPU de 12Gb, sendo o maior limitador a etapa Full Finetuning que fica no limite da RTX 3060.

## Visão Geral do Fluxo

O mini experimento envolve as seguintes etapas:

1. **Preparação dos Dados**: Utiliza-se um dataset de entrada (`entrada_puil.csv`) com o prompt base (`prompt_puil.txt`) para processar e formatar as requisições. 
2. **Treinamento (LoRA / Full)**: Scripts como `04_treinar_puil_lora.yaml` definem as hiperconfigurações do treinamento, rodando em ambiente local com o framework Unsloth.
3. **Extração de Respostas**: O `05_extracao_treino_puil_full.yaml` e o `05_extracao_treino_puil_lora.yaml` executam a inferência em lote via vLLM, usando os pesos treinados para gerar as respostas na pasta `saidas/`.
4. **Avaliação e Comparação**: Usa-se o `06_comparar_treinos.yaml` para comparar os resultados gerados nos arquivos `.parquet` extraídos na etapa anterior com o gabarito.

## Comparação de Saídas de Texto Livre (Não-JSON)

Se o modelo foi treinado para gerar um texto livre em vez de um JSON estruturado, como é o caso desse mini experimento, a engine de avaliação possui suporte nativo convertendo a saída de texto puro em um json no formato `{"resposta": <texto puro>}` para usar o pacote de comparação. 

Para habilitar isso, configure a flag `saida_json: false` no YAML da comparação (`06_comparar_treinos.yaml`), por exemplo:

```yaml
configuracao_comparacao:
  campos_parquet:
    saida_json: false
  campos:
    bertscore:
      - "(global)"
```

### Como funciona internamente?
- Ao ativar `saida_json: false`, o motor de extração lê os arquivos de resultados dos modelos e injeta todo o conteúdo bruto dentro de uma chave virtual chamada `"resposta"`.
- A engine usará esse conteúdo íntegro para calcular as métricas agregadas sob a diretiva de campo `(global)`, como a similaridade semântica (BERTScore, SBERT) e métricas lexicais (ROUGE), ignorando falhas comuns de parsing de JSON.

## Dicas Rápidas
* **Caminhos Dinâmicos**: Não precisa colocar caminhos absolutos nas configurações. O experimento pode aproveitar a chave `misc.pastas_modelos_treinados` para buscar dinamicamente as saídas na sua rede local!
* **Ajuste de Avaliações**: Verifique em `compara/analises_comparacao` os logs de execução e planilhas comparativas após rodar os testes.