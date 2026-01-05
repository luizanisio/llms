# CONTEXTO COMPLETO DO EXPERIMENTO

Experimento realizado para obtenção do título de Especialista em Ciência de Dados pela PUCPR (Nov/2025).
Atualmente (dez/2025), o experimento está sendo replicado em escala ampliada, utilizando todo o **dataset** utilizado no experimento inicial que avaliou apenas 300 instâncias na avaliação LLM-as-a-judge, contando agora com **1.225 instâncias para avaliação via LLM-as-a-judge**. O estudo contempla ajustes nos prompts dos agentes e no avaliador, mantendo o prompt base inalterado para fins de *baseline*.

## 📋 Descrição Geral
O experimento investiga se uma orquestração de **agentes especialistas** operando modelos menores (**Gemma 3 12B** e **27B**) alcança qualidade de extração comparável a um **prompt unificado (*few-shot*)** executado com os mesmos modelos. A avaliação final (**LLM-as-a-Judge**) é realizada pelo **GPT-5**, escolhido por ser um modelo robusto e de fronteira, garantindo uma referência de alta qualidade para o julgamento.

**Importante:** Não há treino supervisionado (*fine-tuning*); todas as abordagens utilizam *In-Context Learning* sobre o mesmo conjunto de acórdãos.

## 🏗️ Configurações Comparadas

1.  **Prompt Base (Baseline)**: Prompt único processando o inteiro teor para extrair todos os campos. Executado em GPT-5 e Gemma 3.
2.  **Pipeline Agêntico (Orquestração)**: Sistema coordenado de agentes especialistas onde:
    *   Um **Agente de Campos** planeja a extração.
    *   **Agentes Especialistas** extraem dados (sequencialmente ou em paralelo).
    *   Um **Agente de Validação** verifica consistência e regras de negócio, podendo solicitar **revisão e retrabalho** (até 3 iterações, configurável).

## ⚙️ Fluxo de Execução dos Agentes
O pipeline agêntico segue uma lógica estrita de dependências para maximizar contexto e eficiência:

1.  **AgenteCampos**: Identifica quais campos estão presentes no acórdão.
2.  **AgenteTeses**: Extrai as teses jurídicas (base para outras extrações).
3.  **AgenteJurisprudenciasCitadas**: Extrai precedentes, utilizando as teses extraídas como contexto (execução sequencial).
4.  **Execução Paralela**: Agentes independentes executam simultaneamente:
    *   `AgenteNotas`
    *   `AgenteInformacoesComplementares` (ICE)
    *   `AgenteTermosAuxiliares` (TAP)
    *   `AgenteTema`
    *   `AgenteReferenciasLegislativas`
5.  **AgenteValidacaoFinal**: Consolida os dados e valida estrutura, tipos de dados e regras de negócio.
6.  **Loop de Revisão**: Se inconsistências são detectadas, o orquestrador reativa os agentes específicos com notas de correção.

**Escopo dos Campos (Chaves Canônicas):**
`teseJuridica`, `referenciasLegislativas`, `jurisprudenciaCitada`, `tema`, `termosAuxiliares`, `notas`, `informacoesComplementares`.

## 📊 Pipeline de Consolidação e Comparação
A análise de qualidade compara as extrações geradas (Base vs. Agentes) utilizando métricas específicas para a natureza de cada campo, avaliando a fidelidade ao texto original e às regras do Manual:

### Estratégia de Seleção de Métricas
*   **BERTScore**: Análise semântica profunda para textos longos e interpretativos (ex: `teseJuridica`, `notas`, `informacoesComplementares`).
*   **ROUGE-L**: Validação de estruturas sequenciais e ordenadas (ex: `jurisprudenciaCitada`, `referenciasLegislativas`).
*   **ROUGE-2**: Precisão de bigramas para termos técnicos e curtos (ex: `termosAuxiliares`, `tema`).
*   **ROUGE-1**: Análise estrutural geral do documento.
*   **Levenshtein**: Comparações exatas (usado pontualmente em testes).

O sistema gera planilhas multidimensionais com mapas de calor (Excel), estatísticas globais e relatórios de exemplos (Markdown).

## ⚖️ Avaliação e Observabilidade

### 1. Qualidade Textual e Semântica
*   **Métricas Clássicas**: ROUGE-1/2/L e BERTScore calculados por campo.
*   **LLM-as-a-Judge**: Modelo GPT-5 (Temperatura 0.0) atuando como "Analista Judiciário". Avalia **Precisão**, **Cobertura (Recall)** e **F1-Score**, fornecendo justificativas baseadas no Manual de Inclusão de Acórdãos do STJ.

### 2. Eficiência e Custos
*   **Telemetria de Tokens**: Contagem detalhada de tokens de entrada, saída, *cache* e *reasoning* (pensamento).
*   **Observabilidade Operacional**: Monitoramento de iterações, contagem de loops de revisão (retrabalho), tempo de execução e pontos de falha por agente.

## 🎯 Objetivo Final
Verificar métricas quantitativas e qualitativas para determinar se a **arquitetura de agentes especialistas** (com modelos abertos e menores) oferece uma alternativa viável ao uso de **modelos proprietários gigantes**, considerando não apenas a qualidade final do espelho, mas também a robustez operacional (consistência JSON) e a eficiência de custos.
