# Comparação de Extrações de Documentos (Genérico via YAML)

> Documentação técnica para desenvolvedores e agentes de IA (Copilot).

## 1. Visão Geral e Propósito
O script `comparar_extracoes.py` é uma **ferramenta genérica e agnóstica** para comparar extrações de dados estruturados (JSON) provenientes de diferentes fontes (Modelos LLM, OCR, Parsers).

O objetivo é validar a qualidade de extração de "Modelos Candidatos" (Agentes/LLMs) contra um "Modelo Base" (Gabarito/Ground Truth/Referência), utilizando métricas textuais (ROUGE, BERTScore, Levenshtein) adaptadas semanticamente para cada tipo de campo.

## 2. Arquitetura e Fluxo de Execução

O fluxo é controlado inteiramente por um arquivo de configuração YAML, eliminando a necessidade de alteração de código para novos experimentos.

### Fluxo de Dados:
1.  **Carga de Configuração**: O script lê o YAML (`--config`) para definir caminhos, rótulos e regras.
2.  **Carga de Dados (`CargaDadosComparacao`)**:
    *   Lê os JSONs da pasta `modelo_base`.
    *   Lê os JSONs das pastas `modelos_comparacao`.
    *   Alinha os documentos pelo ID (padrão regex no nome do arquivo).
3.  **Configuração de Métricas**:
    *   Define quais métricas (BERTScore, ROUGE-L, etc.) se aplicam a quais campos.
4.  **Processamento (`JsonAnaliseDataFrame`)**:
    *   Calcula métricas par-a-par (Base vs Modelo X).
    *   Gera vetores de similaridade.
5.  **Pós-Processamento e Saída**:
    *   **Excel**: Matriz colorida (heatmap) com os scores.
    *   **CSV**: Dados brutos para análise externa.
    *   **Estatísticas**: Agregação (Mean/Std/Median) por modelo e métrica.
    *   **LLM-as-a-Judge (Opcional)**: Usa um LLM para decidir qual extração é melhor em casos complexos.
    *   **Gráficos**: Gera visualizações comparativas.

## 3. Configuração (Arquivo YAML)

A estrutura do YAML é dividida em 4 seções principais:

### 3.1 `saida`
Controla onde os artefatos serão salvos e se deve reprocessar dados existentes.
```yaml
saida:
  pasta: "./saida_experimento_X"
  regerar_planilha_base: false # Cache: se true, não recalcula métricas pesadas se o excel existir
```

> **Nota sobre caminhos**: Todos os caminhos definidos no YAML são resolvidos **relativamente à localização do próprio arquivo YAML**. 
> 
> **Organização de Arquivos**: O script estrutura a saída da seguinte forma:
> *   **Raiz (`saida.pasta`)**: Arquivos de visão geral (Excel, CSV consolidado e Gráficos de Status).
> *   **Subpasta `jsons/`**: Arquivos de detalhe (JSON de análise por documento, Relatórios Markdown e outros artefatos granulares).

### 3.2 `execucao`
Flags de controle de fluxo (substituem as antigas constantes globais `SO_GRAFICOS`, etc.).
```yaml
execucao:
  max_workers: 10
  teste_rapido: false      # Se true, desativa BERTScore (lento) e usa apenas ROUGE/Levenshtein
  gerar_graficos: true     # Apenas gera gráficos baseados no excel existente
  llm_as_a_judge: false    # Executa avaliação qualitativa via LLM
  analise_estatistica: false # Executa testes de hipótese/comparação estatística profunda
```

### 3.3 Definição de Modelos
Define quem é a verdade (Base) e quem são os desafiantes.
```yaml
modelo_base:
  pasta: "./gabarito"
  rotulo: "Humano"

modelos_comparacao:
  - pasta: "./modelo_A"
    rotulo: "GPT-4"
    familia: "OpenAI" # Usado para agrupamento em gráficos/stats
    ativo: true       # Opcional (padrão true). Se false, ignora este modelo completamente.
```

### 3.4 `configuracao_comparacao` (Mapa de Métricas)
Mapeia campos do JSON para algoritmos de similaridade específicos.
*   **bertscore**: Para textos longos, argumentativos ou com paráfrases (ex: `teseJuridica`).
*   **rouge_l**: Para listas ou textos onde a ordem importa (ex: `jurisprudenciaCitada`).
*   **rouge_2**: Para textos curtos, termos técnicos ou precisão de n-gramas.
*   **levenshtein**: Para identificadores, datas ou valores exatos.

## 4. Checklist de Implementação e Status

O código foi refatorado para suportar o modelo genérico. Abaixo o status atual e próximos passos.

- [x] **Leitura de YAML**: Implementado suporte a arquivo de configuração como argumento posicional.
- [x] **Remoção de Hardcoding**: Cenários fixos (ex: `base_gpt5`) removidos em favor do carregamento dinâmico.
- [x] **Mapeamento Dinâmico de Métricas**: Lógica para converter config YAML em config do `JsonAnaliseDataFrame`.
- [x] **Controle de Execução**: Flags `regerar_planilha_base`, `teste_rapido` implementadas.
- [x] **Análise Estatística Dinâmica**: `processar_analise_estatistica` adaptada para iterar sobre os pares definidos no YAML.
- [x] **Validação de Caminhos Relativos**: Caminhos no YAML são resolvidos relativamente ao arquivo de configuração.
- [x] **Validação de Dependências**: Verificar se todos os imports em `src/` estão alinhados com a nova estrutura de dados (ex: `JsonAnaliseDados`).
- [x] **Teste de Integração**: Executar o script com `config_summa.yaml` para validar ponta-a-ponta.

Ajustes e conferências:
- [x] Gráfico boxplot (global)
- [x] Gráfico boxplot (estrutura)
- [x] Revisão dos nomes dos gráficos para melhor compreensão
- [x] Nos gráficos e planilha, a ordem dos campos deve ser a mesma que no YAML
- [x] Verificar geração de gráficos de estrutura e global nos testes
- [x] Quando ignorar_erro_extração=true, ignora os itens que deram erro na extração de qualquer modelo (base ou outros), comparando apenas itens onde todos os modelos deram sucesso. Gera um gráfico de barras com sucesso e erro por modelo.
- [x] Verificar se todos os gráficos usam o componente de gráficos em util_graficos.py
- [x] Quando (global) não está definido, é usado Rouge-L
- [x] Quando (estrutura) não está definido, é usado Rouge-1

