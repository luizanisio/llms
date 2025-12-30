# Diagramas de Fluxo - Experimento de Extração de Espelhos

**[← Voltar para README principal](README.md)**

## Índice de Diagramas

1. [Geração de Espelho Base (Prompt Único)](#1-geração-de-espelho-base-prompt-único)
2. [Fluxo de Orquestração Completo (Sistema de Agentes)](#2-fluxo-de-orquestração-completo-sistema-de-agentes)
3. [Geração de Espelho com Agentes Especializados (Visão Simplificada)](#3-geração-de-espelho-com-agentes-especializados-visão-simplificada)
4. [Estrutura de Agentes](#4-estrutura-de-agentes-agentes_orquestradorpy)
5. [Avaliação LLM-as-a-Judge](#5-avaliação-llm-as-a-judge)
6. [Comparação de Extrações (Métricas de Similaridade)](#6-comparação-de-extrações-métricas-de-similaridade)
7. [Fluxo de Métricas por Campo](#7-fluxo-de-métricas-por-campo)
8. [Observabilidade do Sistema de Agentes](#8-observabilidade-do-sistema-de-agentes)
9. [Principais Classes Utilitárias](#9-principais-classes-utilitárias)

---

## 1. Geração de Espelho Base (Prompt Único)

```mermaid
flowchart TD
    A[DataFrame Parquet<br/>espelhos_acordaos_consolidado_textos.parquet] --> B[gerar_espelhos_base.py]
    B --> C{Texto Criptografado}
    C -->|Decripta| D[CRIPT.decriptografar]
    D --> E[Prompt Base Único<br/>PROMPT_ESPELHO_BASE]
    E --> F{Modelo LLM}
    F -->|GPT-5| G[get_resposta<br/>util_get_resposta.py]
    F -->|Gemma-3| H[get_resposta<br/>util_openai.py]
    G --> I[Espelho JSON]
    H --> I
    I --> J[Gravação<br/>espelhos_base_modelo/id_peca.json]
    
    style B fill:#e1f5ff
    style E fill:#fff4e1
    style I fill:#e8f5e9
```

## 2. Fluxo de Orquestração Completo (Sistema de Agentes)

### Diagrama Detalhado com Todas as Etapas

```mermaid
flowchart TD
    Start([Início]) --> E1[ETAPA 1: AgenteCampos<br/>Identifica campos necessários]
    E1 --> CheckErro{Teve erro?}
    
    CheckErro -->|Sim| End
    CheckErro -->|Não| CheckCampos{Campos<br/>identificados?}
    
    CheckCampos -->|Não| E15[ETAPA 1.5: Revisão AgenteCampos<br/>Solicita conferência detalhada]
    E15 --> ReexecCampos[Reexecuta AgenteCampos<br/>com instruções de revisão]
    ReexecCampos --> CheckRevisao{Identificou<br/>campos?}
    
    CheckRevisao -->|Não| ConfirmaSem[Confirma sem campos<br/>para extração]
    CheckRevisao -->|Sim| E2
    CheckCampos -->|Sim| E2[ETAPA 2: AgenteTeses<br/>Extrai teses jurídicas]
    
    E2 --> E25[ETAPA 2.5: AgenteJurisprudenciasCitadas<br/>Extrai precedentes com contexto de teses]
    
    E25 --> E3[ETAPA 3: Execução Paralela<br/>AgenteNotas, AgenteICE, AgenteTAP,<br/>AgenteTema, AgenteRefLeg]
    
    E3 --> E4[ETAPA 4: AgenteValidacaoFinal<br/>Valida todas as extrações]
    
    E4 --> E5{ETAPA 5:<br/>Loop de Revisão<br/>max 2 ciclos}
    
    E5 -->|Há erros ou<br/>revisões| CheckErros[Detecta agentes com erro]
    CheckErros --> AddRevisao[Adiciona instruções de<br/>revisão automática]
    AddRevisao --> ProcessaRev[Reexecuta agentes com revisão]
    ProcessaRev --> RevalAgentes[Atualiza resultados e revalida]
    RevalAgentes --> E5
    
    E5 -->|Validação<br/>aprovada ou<br/>limite atingido| Consolidacao[Consolidação Final:<br/>Monta espelho_final]
    
    ConfirmaSem --> Consolidacao
    
    Consolidacao --> VerificaErros{Há erros<br/>remanescentes?}
    
    VerificaErros -->|Sim| NaoGrava[NÃO grava arquivos<br/>Permite nova tentativa]
    VerificaErros -->|Não| Grava[Grava arquivos:<br/>- espelho.json<br/>- resumo.json<br/>- observabilidade]
    
    NaoGrava --> End([Fim])
    Grava --> End
    
    style E1 fill:#e1f5ff
    style E15 fill:#fff3cd
    style E2 fill:#d4edda
    style E25 fill:#d4edda
    style E3 fill:#d1ecf1
    style E4 fill:#f8d7da
    style E5 fill:#e7e7e7
    style Consolidacao fill:#d6d8db
    style Grava fill:#28a745,color:#fff
    style NaoGrava fill:#dc3545,color:#fff
```

**Pipeline de Execução Detalhado:**

1. **ETAPA 1**: `AgenteCampos` - Identifica quais campos existem no acórdão
2. **ETAPA 1.5**: Revisão do `AgenteCampos` - Se não identificou campos, solicita revisão com instrução específica para conferir atentamente
3. **ETAPA 2**: `AgenteTeses` - Extrai as teses jurídicas (dependência primária)
4. **ETAPA 2.5**: `AgenteJurisprudenciasCitadas` - Extrai precedentes baseados nas teses extraídas
5. **ETAPA 3**: Execução Paralela - `AgenteNotas`, `AgenteInformacoesComplementares`, `AgenteTermosAuxiliares`, `AgenteTema` e `AgenteReferenciasLegislativas` rodam simultaneamente
6. **ETAPA 4**: `AgenteValidacaoFinal` - Consolida e valida todas as extrações
7. **ETAPA 5**: Loop de Revisão - Processa até 2 ciclos de revisões conforme necessário, reexecutando agentes com erros ou que precisam de ajustes
8. **Consolidação Final**: Monta o espelho final com todos os campos extraídos e metadados
9. **Verificação de Erros**: Apenas grava arquivos se não houver erros remanescentes, permitindo novas tentativas em caso de falha

---

## 3. Geração de Espelho com Agentes Especializados (Visão Simplificada)

```mermaid
flowchart TD
    A[DataFrame Parquet] --> B[agentes_gerar_espelhos.py]
    B --> C[AgenteOrquestradorEspelho]
    C --> D[Pipeline de Agentes]
    
    D --> E1[1. AgenteCampos<br/>Identifica campos necessários]
    E1 --> E2[2. AgenteTeses<br/>Extrai teses jurídicas]
    E2 --> E3[3. AgenteJurisprudenciasCitadas<br/>Extrai jurisprudências + contexto teses]
    E3 --> E4[4. Agentes Paralelos]
    
    E4 --> F1[AgenteNotas]
    E4 --> F2[AgenteInformacoesComplementares]
    E4 --> F3[AgenteTermosAuxiliares]
    E4 --> F4[AgenteTema]
    E4 --> F5[AgenteReferenciasLegislativas]
    
    F1 & F2 & F3 & F4 & F5 --> G[5. AgenteValidacaoFinal]
    G --> H{Validação Aprovada?}
    H -->|Não| I[Loop Revisão<br/>max 2 iterações]
    I --> E2
    H -->|Sim| J[Espelho Final]
    J --> K[espelhos_agentes_modelo/id_peca.json]
    J --> L[Observabilidade<br/>*.obs.json, *.obs.md, *.txt]
    
    style C fill:#e1f5ff
    style D fill:#fff4e1
    style E4 fill:#f3e5f5
    style G fill:#ffebee
    style J fill:#e8f5e9
```

**Nota:** Este é um diagrama simplificado. [Ver diagrama completo detalhado acima](#2-fluxo-de-orquestração-completo-sistema-de-agentes).

---

## 4. Estrutura de Agentes (agentes_orquestrador.py)

```mermaid
classDiagram
    class Agente {
        +nome: str
        +prompt_base: str
        +modelo: str
        +iteracoes: int
        +preparar_prompt()
        +executar()
        +get_resposta()
    }
    
    class AgenteOrquestradorEspelho {
        +id_peca: str
        +texto_peca: str
        +callable_modelo
        +executar()
        +_executar_agente_unico()
        +_executar_agentes_paralelo()
        +_processar_revisao()
    }
    
    class AgenteCampos {
        PROMPT_AGENTE_CAMPOS
    }
    
    class AgenteTeses {
        PROMPT_AGENTE_TESES
    }
    
    class AgenteJurisprudenciasCitadas {
        PROMPT_AGENTE_JURIS_CITADA
        preparar_prompt(contexto_teses)
    }
    
    class AgenteValidacaoFinal {
        PROMPT_VALIDACAO_FINAL
        preparar_prompt(saidas_agentes)
    }
    
    Agente <|-- AgenteCampos
    Agente <|-- AgenteTeses
    Agente <|-- AgenteJurisprudenciasCitadas
    Agente <|-- AgenteValidacaoFinal
    AgenteOrquestradorEspelho --> Agente : coordena
```

---

## 5. Avaliação LLM-as-a-Judge

```mermaid
flowchart TD
    A[DataFrame com Textos] --> B[avaliacao_llm_as_a_judge.py]
    B --> C{Para cada id_peca}
    C --> D[Carrega Extração JSON<br/>espelhos_*/id_peca.json]
    C --> E[Decripta Texto Original]
    
    D & E --> F[Monta Prompt<br/>PROMPT_LLM_AS_A_JUDGE]
    F --> G[GPT-5 como Juiz<br/>PAPEL_LLM_AS_A_JUDGE]
    G --> H[Resposta JSON]
    
    H --> I{Campos Avaliados}
    I --> J[precision: float]
    I --> K[recall: float]
    I --> L[f1_score: float]
    I --> M[explicacao: str]
    
    J & K & L & M --> N[id_peca.avaliacao.json]
    N --> O[id_peca.avaliacao.log]
    
    style B fill:#e1f5ff
    style F fill:#fff4e1
    style G fill:#ffebee
    style N fill:#e8f5e9
```

---

## 6. Comparação de Extrações (Métricas de Similaridade)

```mermaid
flowchart TD
    A[Definição Origem/Destinos] --> B[comparar_extracoes.py]
    B --> C[CargaDadosComparacao<br/>util_json_carga.py]
    C --> D[JsonAnaliseDados]
    
    D --> E[JsonAnaliseDataFrame]
    E --> F{Configuração Métricas<br/>CONFIG_COMPARACAO}
    
    F --> G1[BERTScore<br/>textos longos semânticos]
    F --> G2[ROUGE-L<br/>sequências estruturadas]
    F --> G3[ROUGE-2<br/>bigramas precisos]
    F --> G4[Levenshtein<br/>textos curtos exatos]
    
    G1 & G2 & G3 & G4 --> H[Cálculo Paralelo<br/>max_workers]
    H --> I[DataFrame Comparação]
    
    I --> J1[CSV<br/>comparacao_extracoes.csv]
    I --> J2[Excel com Mapas de Calor<br/>comparacao_extracoes.xlsx]
    I --> J3[Estatísticas Globais<br/>*.estatisticas.csv]
    I --> J4[Markdown Exemplos<br/>*.exemplos.md]
    
    J2 --> K[Aba: Avaliação LLM<br/>atualizar_avaliacao_llm_no_excel]
    
    style C fill:#e1f5ff
    style E fill:#fff4e1
    style F fill:#f3e5f5
    style H fill:#ffebee
    style J2 fill:#e8f5e9
```

---

## 7. Fluxo de Métricas por Campo

```mermaid
flowchart LR
    A[Campo] --> B{Tipo de Campo}
    
    B -->|Textos Longos| C[teseJuridica<br/>notas<br/>informacoesComplementares]
    B -->|Estruturados| D[jurisprudenciaCitada<br/>referenciasLegislativas]
    B -->|Termos Técnicos| E[termosAuxiliares<br/>tema]
    B -->|Global| F[documento completo]
    
    C --> G1[BERTScore<br/>semântica profunda]
    C --> G2[ROUGE-L<br/>precisão fraseamento]
    
    D --> H1[ROUGE-L<br/>estrutura sequencial]
    D --> H2[ROUGE-2<br/>bigramas]
    
    E --> I1[BERTScore<br/>contexto técnico]
    E --> I2[ROUGE-2<br/>termos exatos]
    
    F --> J1[ROUGE-2<br/>métrica padrão]
    F --> J2[todas disponíveis<br/>análise multidimensional]
    
    style C fill:#e8f5e9
    style D fill:#fff4e1
    style E fill:#f3e5f5
    style F fill:#ffebee
```

---

## 8. Observabilidade do Sistema de Agentes

```mermaid
flowchart TD
    A[AgenteOrquestradorEspelho] --> B[_soma_observabilidade]
    B --> C[Lock Thread-Safe]
    
    C --> D{Dados Coletados}
    D --> E1[Duração por Agente]
    D --> E2[Iterações/Revisões]
    D --> E3[Tokens Consumidos]
    D --> E4[Sucesso/Erro]
    
    E1 & E2 & E3 & E4 --> F[observabilidade/]
    
    F --> G1[id_peca.obs.json<br/>dados estruturados]
    F --> G2[id_peca.obs.md<br/>relatório markdown]
    F --> G3[id_peca.AgenteTeses.txt<br/>prompts completos]
    F --> G4[id_peca.resumo.json<br/>tokens por campo]
    
    G4 --> H[Estatísticas]
    H --> I1[prompt_tokens]
    H --> I2[completion_tokens]
    H --> I3[cached_tokens]
    H --> I4[reasoning_tokens]
    H --> I5[time real vs linear]
    
    style A fill:#e1f5ff
    style F fill:#fff4e1
    style G4 fill:#e8f5e9
```

---

## 9. Principais Classes Utilitárias

```mermaid
classDiagram
    class CargaDadosComparacao {
        +pasta_origem: str
        +pastas_destinos: list
        +carregar() JsonAnaliseDados
    }
    
    class JsonAnaliseDataFrame {
        +to_df() DataFrame
        +exportar_csv()
        +exportar_excel()
        +atualizar_avaliacao_llm_no_excel()
        +gerar_graficos_de_excel()
    }
    
    class UtilCriptografia {
        +decriptografar()
    }
    
    class STJOpenAIA {
        +prompt() dict
    }
    
    CargaDadosComparacao --> JsonAnaliseDataFrame : fornece dados
    JsonAnaliseDataFrame --> UtilPandasExcel : formatação
    AgenteOrquestradorEspelho --> UtilCriptografia : textos
    AgenteOrquestradorEspelho --> STJOpenAIA : chamadas LLM
```

## Legenda de Cores

- 🔵 **Azul Claro**: Entrada de dados / Carregamento
- 🟡 **Amarelo**: Processamento / Transformação
- 🟣 **Roxo**: Execução Paralela / Múltiplos Agentes
- 🔴 **Vermelho**: Validação / Decisão Crítica
- 🟢 **Verde**: Saída de Dados / Resultado Final

## Arquivos Python Principais

| Arquivo | Função Principal |
|---------|-----------------|
| `gerar_espelhos_base.py` | Extração com prompt único (baseline) |
| `agentes_gerar_espelhos.py` | Extração com sistema de agentes |
| `agentes_orquestrador.py` | Orquestração e coordenação de agentes |
| `prompt_espelho_agentes.py` | Definição de todos os prompts |
| `avaliacao_llm_as_a_judge.py` | Avaliação com GPT-5 como juiz |
| `comparar_extracoes.py` | Comparação com múltiplas métricas |
| `util_json_carga.py` | Carregamento de dados para comparação |
| `util_json.py` | Análise e exportação de resultados |
| `util_bertscore.py` | Configuração e cálculo de BERTScore |
