## Contém anotações importantes de aprendizado e ajustes no experimento

### Introdução
A orquestração de agentes não garante automaticamente resultados superiores a um prompt único (monolítico). O sucesso depende da especialização correta, da consistência entre as instruções e de um mecanismo robusto de validação. Abaixo estão os aprendizados essenciais para replicar uma orquestração bem-sucedida.

## 1. Passo a Passo de Revisão da Orquestração de Agentes

Para converter um prompt base monolítico em uma orquestração eficiente:

1.  **Decomposição Orientada a Responsabilidade**:
    *   Quebre o prompt base em "domínios de conhecimento". Cada agente deve ser especialista em *uma* coisa (ex: AgenteTeses, AgenteDatas).
    *   **Regra de Ouro**: Se um agente precisa ler a saída de outro para trabalhar, defina claramente essa dependência no fluxo (ex: AgenteJurisprudencia só roda após AgenteTeses identificar os temas).

2.  **Sincronização da "Fonte da Verdade"**:
    *   Mantenha um "Manual" ou prompt de referência central.
    *   **Verificação Cruzada**: Garanta que *todas* as regras do prompt base (ex: gabaritos de formatação, listas de termos proibidos) foram migradas explicitamente para o prompt do agente específico. Um erro comum é esquecer regras de "borda" (ex: como tratar valores nulos) ao criar o prompt especializado.

3.  **Protocolo de Validação e Tolerância**:
    *   Implemente um **Agente Validador** que atua como revisor, não como re-executor. Ele deve criticar *se* a saída obedece às regras, não tentar refazer o trabalho.
    *   **Critério de Parada Dinâmico**: Evite loops infinitos de correção por "pedantismo". Defina um limiar (ex: após 3 tentativas) onde o validador aceita erros menores (formatação) e só bloqueia erros críticos (alucinações).
    *   **Feedback Consolidado**: O validador deve listar *todos* os erros de uma vez para que o agente corrija em lote, economizando tokens e tempo.

## 2. Cuidados com a Avaliação (LLM-as-a-Judge)

Como avaliar se a orquestração superou o baseline sem criar viés?

1.  **Isenção do Gabarito**: O Juiz **não** deve receber a resposta do modelo base como "gabarito absoluto". Ele deve avaliar a extração do agente contra o **Texto Original** e as **Regras do Manual**.
2.  **Alinhamento de Critérios**: O prompt do Juiz deve ser uma versão mais rigorosa dos prompts dos agentes. Se o agente foi instruído a ignorar acentos, o Juiz não pode penalizar falta de acentos.
    *   *Dica Prática*: Sempre que atualizar o prompt de um agente, atualize imediatamente o critério correspondente no prompt do Juiz.
3.  **Métricas Granulares**: Avalie *Precision* (alucinou dados?) e *Recall* (esqueceu dados?) separadamente. Uma média única (F1) pode mascarar um agente que fala pouco mas acerta tudo (alta precisão, baixo recall).

## 3. Cuidados com a Observabilidade

Debugging de sistemas multi-agentes é complexo. O que fazer:

1.  **Rastreamento de Passo a Passo**: Salve os prompts reais enviados e as respostas brutas de *cada* agente em arquivos separados ou logs estruturados. Saber *o que* o agente recebeu é crucial para entender por que ele errou.
2.  **Logs de Revisão**: Registre claramente as iterações de correção. "O que o Validador pediu?" vs "O que o Agente entregou na V2?". Isso ajuda a calibrar a "rabugice" do validador.
3.  **Iteração Visível**: Injete no prompt de revisão o número da iteração atual (ex: `<ESTADO_VALIDACAO> Iteração: 2/3`). Isso permite que o próprio modelo mude de comportamento (seja mais conservador) quando percebe que está falhando repetidamente ou tente concluir de forma mais objetiva quando identifica que é a última iteração possível.

## 4. Cuidados com a Qualidade dos Dados

1.  **Gabaritos de Formatação**: Para campos estruturados (datas, valores, leis), forneça exemplos de *few-shot* negativos e positivos no prompt. Mostre o que *não* fazer.
2.  **Tratamento de Nulos**: Defina explicitamente como o agente deve responder quando não encontra a informação. Arrays vazios `[]` são preferíveis a alucinações de "Não consta". O modelo tem que saber que uma resposta vazia é possível e que isso não é, necessariamente, um erro.

## 5. Cuidados com a Interação entre Agentes
1.  **Limpeza de Metadados**: Ao passar a saída de um agente para outro (ex: AgentTesis -> AgenteValidador), remova chaves de metadados internos (como `usage`, `model`, `contribuição`). Isso economiza tokens e reduz "ruído" que pode confundir o prompt do próximo agente.
2.  **Memória do Validador**: Para evitar revisões circulares, o Validador precisa saber o que ele mesmo pediu anteriormente. Envie o histórico de revisões (`revisao_solicitada`) e a lista de campos já aprovados (`campos_aprovados`) para que o ele não "esqueça" o progresso e peça a mesma correção duas vezes ou critique algo que ele já aprovou.
3.  **Sanitização de Exemplos**: Use dados genéricos/fictícios nos exemplos de *few-shot*. Modelos menores (como 12B) tendem a copiar dados dos exemplos se estes parecerem muito reais (ex: citar um número de processo específico). Use placeholders (ex: `000.000/UF`) e instruções explícitas para **NUNCA** copiar dados dos exemplos.

## 6. Recomendações para Reprodução ou Extensão

1.  **Mantenha o Baseline Intocado**: Ao iterar sobre os prompts dos agentes, *nunca* altere o prompt base simultaneamente. O baseline deve permanecer estável para isolar o efeito das mudanças.
2.  **Versionamento de Prompts**: Trate os prompts como código. Versione cada alteração para rastrear o impacto de ajustes específicos.
3.  **Amostragem Consistente**: Use `random_state` fixo ao selecionar amostras para avaliação, garantindo reprodutibilidade entre rodadas.
4.  **Documentação de Decisões**: Ao ajustar um critério no Validador ou no Juiz (ex: "ignorar erros de formatação após iteração 2"), registre *por que* a decisão foi tomada.

---
**Conclusão**: Orquestração de agentes é uma técnica poderosa, mas não mágica. Seu sucesso depende de engenharia cuidadosa de prompts, validação robusta e observabilidade detalhada. Este documento serve como checklist para evitar armadilhas comuns.
