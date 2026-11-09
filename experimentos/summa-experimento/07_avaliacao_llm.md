# 07_avaliacao_llm.py — Avaliação LLM-as-a-judge

Envia cada extração do conjunto de teste ao modelo juiz com o prompt
`avaliacao_llm_humana/prompt_juiz_llm.txt` e grava a nota Likert (1–4) na coluna `avaliacao`
do parquet de saída do protocolo (`saida_qwen7b({protocolo})_teste.parquet`).

O juiz é escolhido pelo perfil `--juiz` (ver [Juízes](#juízes---juiz)). O perfil padrão, `gpt5`,
grava nos parquets de `saida/`. O perfil `sabia4` é isolado: grava cópias desses parquets em
`avaliacao_llm/sabia4/saida/` e não escreve em `saida/` nem nos artefatos do `gpt5`.

Extrações inválidas (erro na extração, JSON inválido, sem temas ou sem íntegra) não vão ao juiz:
recebem nota 1 direta (`{"nota": 1, "problemas": ["json_invalido"]}`).

## Parâmetros

| Parâmetro | O que faz | Chama a API? |
|---|---|:---:|
| *(nenhum)* | Avalia só as instâncias com a coluna `avaliacao` vazia. Retoma de onde parou. | sim |
| `--juiz sabia4` | Escolhe o perfil do juiz (`JUIZES` no script): modelo, pasta e parâmetros. Sem ele, usa `gpt5`. Vale para todos os modos abaixo. | — |
| `--protocolos d1 d7 ...` | Escolhe os protocolos. Sem ele, usa a lista `protocolos` do perfil ou, se vazia, `PROTOCOLOS` do script. Aceita os especiais `a` (zero-shot), `prof` (professor) e `base` (GPT5). | — |
| `--refazer` | Descarta as avaliações existentes e reenvia **tudo** ao juiz. | sim |
| `--continuar` | Retoma uma execução do juiz que foi interrompida, usando a entrada que ela já tinha gerado. | sim |
| `--reconstruir` | Refaz a coluna `avaliacao` a partir de `saida_juiz_{p}.parquet` da pasta do juiz, sem chamar o juiz. Gera também o resumo. | não |
| `--resumo` | Gera `RESUMO_AVALIACAO_LLM.md`, os CSVs e os gráficos bayesianos na pasta do juiz. | não |
| `--recorte Q1_ajuste_fino` | Com `--resumo`/`--reconstruir` (exige `--protocolos`): grava o resumo em `<pasta do juiz>/recortes/<nome>/`, um por questão de pesquisa, sem sobrescrever o resumo geral. Comandos de cada recorte no cabeçalho do script. | não |
| `--tokens` | Estima os tokens de entrada e saída do que seria enviado. | não |
| `--rope 0.15` | ROPE fixa da análise bayesiana Likert. Tem prioridade sobre a calibração. | — |
| `--rope-protocolos d1 d1a d1b` | Réplicas usadas para calibrar a ROPE. Sem ele, usa `ROPE_PROTOCOLOS` do script. | — |

`--refazer`, `--continuar` e `--reconstruir` são mutuamente exclusivos.
Os modos que chamam a API mostram um resumo (juiz, modelo e onde as notas serão gravadas) e pedem confirmação (`Continuar? [S/n]`) antes de começar.

## Juízes (`--juiz`)

Cada perfil de `JUIZES` (no início do script) define o modelo juiz e a pasta dos artefatos dele.
Assim o modelo e o destino das notas ficam sempre juntos, e notas de juízes diferentes nunca se misturam.

| Perfil | Modelo | Pasta dos artefatos | Onde as notas são gravadas | Isolado |
|---|---|---|---|:---:|
| `gpt5` *(padrão)* | `oa:gpt5:m:l` | `avaliacao_llm/` | coluna `avaliacao` dos parquets de `saida/` | não |
| `sabia4` | `mt:sabia-4` | `avaliacao_llm/sabia4/` | coluna `avaliacao` das cópias em `avaliacao_llm/sabia4/saida/` | sim |

**A coluna se chama `avaliacao` em todos os perfis.** O juiz é identificado pela pasta, não pelo nome da coluna.
Por isso `comparar_extracoes.py` e `realizar_avaliacoes.py` leem as notas de qualquer juiz da mesma forma.

**Como funciona o perfil isolado:**

- O parquet de destino tem o mesmo nome do original em `saida/`
  (ex.: `avaliacao_llm/sabia4/saida/saida_qwen7b(d1)_teste.parquet`, `.../saida_or_235b(prof)_teste.parquet`).
- Enquanto a cópia não existe, a extração é lida de `saida/` e a coluna `avaliacao` herdada (do `gpt5`) é descartada:
  todas as instâncias começam pendentes. A cópia é criada na primeira gravação; daí em diante a leitura vem dela.
- Não escreve em `saida/`, não cria `.bak` lá e não sincroniza o parquet completo dos protocolos especiais
  (`saida_or_235b.parquet`, `saida_oa_gpt5.parquet`, `saida_qwen7b.parquet`).
- Retomada, `--continuar`, `--reconstruir` e `--resumo` funcionam exatamente como no `gpt5`, só que dentro de `avaliacao_llm/sabia4/`.
- Pode rodar em paralelo com o `gpt5`: os dois não compartilham nenhum arquivo gravado.
  Duas execuções do **mesmo** juiz no mesmo protocolo continuam não suportadas (gravam os mesmos arquivos).

**ROPE do sabia4.** A ROPE é calibrada pelas réplicas (`ROPE_PROTOCOLOS` = `d1`, `d1a`, `d1b`) e depende do juiz.
Para o resumo bayesiano do sabia4, avalie também as réplicas com ele
(`--juiz sabia4 --protocolos d1 d1a d1b`) ou informe `--rope`. Reaproveitar a ROPE do `gpt5` mistura juízes.

**Modelo.** `mt:sabia-4` é o mesmo modelo comparado aos avaliadores humanos em `avaliacao_llm_humana/saida_sabia_0*`.
O sufixo `:flex` (`mt:sabia-4:flex`) muda só o service tier (mais barato e mais lento), não o modelo.

**Incluir outro juiz:** acrescente um perfil em `JUIZES` com `modelo`, `pasta` (própria, ex.: `./avaliacao_llm/<nome>`),
`isolado: True`, `workers`, `tentativas`, `tokens_saida_estimado` (só para o `--tokens`) e `protocolos` (`None` = `PROTOCOLOS`).
Mais nada no script precisa mudar: as funções leem as variáveis do juiz ativo definidas por `aplicar_juiz()`.

## O que usar em cada situação

| Situação | Comando |
|---|---|
| Avaliar um protocolo pela primeira vez | `python 07_avaliacao_llm.py --protocolos d1` |
| Avaliar instâncias novas/pendentes de um protocolo já avaliado | `python 07_avaliacao_llm.py --protocolos d1` |
| Parei uma execução **sem** `--refazer` | Rode o mesmo comando de novo (ou use `--continuar`) |
| Parei uma execução **com** `--refazer` | `python 07_avaliacao_llm.py --protocolos d1 --continuar` |
| Quero substituir avaliações antigas (mudou o modelo juiz, o prompt ou a extração) | `python 07_avaliacao_llm.py --protocolos d1 --refazer` |
| A coluna `avaliacao` foi perdida ou o parquet de saída foi regenerado, mas a saída do juiz está completa | `python 07_avaliacao_llm.py --protocolos d1 --reconstruir` |
| Atualizar o relatório comparativo (ROPE calibrada por `ROPE_PROTOCOLOS`) | `python 07_avaliacao_llm.py --resumo` |
| Calibrar a ROPE com outras réplicas | `python 07_avaliacao_llm.py --resumo --rope-protocolos d1 d1b` |
| Usar uma ROPE fixa, sem calibrar | `python 07_avaliacao_llm.py --resumo --rope 0.15` |
| Saber quanto vai custar antes de rodar | `python 07_avaliacao_llm.py --protocolos d1 --tokens` (com `--refazer`, estima o reprocessamento completo) |
| Avaliar protocolos com o sabia4, sem tocar nas notas do gpt5 | `python 07_avaliacao_llm.py --juiz sabia4 --protocolos d1 d7` |
| Qualquer situação acima com o sabia4 | O mesmo comando com `--juiz sabia4` (ex.: `--juiz sabia4 --protocolos d1 --continuar`) |
| Resumo de uma questão de pesquisa (mesmos recortes de `estatistica.protocolos` do `06_compara_todos.yaml`) | `python 07_avaliacao_llm.py --resumo --recorte Q4b_unfreeze --protocolos b c d20 d19` (com `--juiz sabia5` vai para `avaliacao_llm/sabia5/recortes/`) |
| Resumo do sabia4 com ROPE calibrada nas réplicas dele | `python 07_avaliacao_llm.py --juiz sabia4 --protocolos d1 d1a d1b` e depois `--juiz sabia4 --resumo --protocolos d1 d7 ...` |

### Por que a interrupção com `--refazer` precisa do `--continuar`

A coluna `avaliacao` só é gravada no parquet de saída quando o juiz termina.
Se o `--refazer` é interrompido, o parquet continua com as notas **antigas** preenchidas e parece completo:
rodar sem parâmetros responde "Nada a fazer", e rodar `--refazer` de novo apaga o que já foi avaliado e recomeça do zero.

O `--continuar` reaproveita `entrada_juiz_{p}.parquet` e `saida_juiz_{p}.parquet`, envia ao juiz só o que falta
(e o que deu erro) e, no fim, aplica o resultado na coluna `avaliacao`.

O modo padrão avisa quando encontra uma execução incompleta:

```
⚠️  execução anterior do juiz incompleta (1726/3689 concluídas) — para retomá-la use --continuar
```

### Validações do `--continuar`

Antes de chamar a API, confere para cada protocolo:

1. existem `entrada_juiz_{p}.parquet` e `config_juiz_{p}.yaml` da execução anterior (se não existem, o protocolo é ignorado);
2. todas as chaves da entrada existem no parquet de extração;
3. os prompts da entrada são idênticos aos montados agora (mesma extração, íntegra e template).

Se alguma falhar, nada é processado. Corrija a causa ou descarte a execução anterior com `--refazer`.

**Modelo juiz diferente.** Se o modelo da execução anterior (registrado no `config_juiz_{p}.yaml`) não for o
modelo atual do perfil e houver chaves pendentes, o script mostra os dois nomes e pergunta
`Continuar {p} com modelos diferentes? [s/N]`. Respondendo sim, as pendentes são avaliadas pelo modelo atual
e o protocolo fica com notas dos dois modelos; respondendo não, o protocolo é ignorado.
Sem chaves pendentes, nada vai ao juiz: o resultado é só aplicado e o YAML não é alterado.
Como cada perfil tem pasta própria, isso só ocorre quando o `modelo` do perfil é alterado entre execuções
(ex.: `oa:gpt5-chat:m:l` → `oa:gpt5:m:l`); trocar de juiz é trocar de `--juiz`.

Ao aplicar o resultado:

- chaves enviadas ao juiz recebem a nota dele (nota 1 se o juiz deu erro em todas as tentativas);
- extrações inválidas recebem nota 1 direta;
- as demais chaves mantêm a avaliação que já tinham.

### `--continuar` × `--reconstruir`

O `--reconstruir` dá nota 1 a **toda** chave que não está na saída do juiz.
Isso só é correto quando a saída do juiz cobre todas as extrações válidas, por exemplo depois de um `--refazer` concluído.
Depois de execuções incrementais ou interrompidas, use `--continuar`, que preserva as avaliações já existentes.

## ROPE Likert (`--resumo` e `--reconstruir`)

A ROPE é a margem de equivalência prática da análise bayesiana sobre as notas 1–4. Ela é definida nesta ordem:

1. `--rope <valor>`, se informado;
2. calibração pelas réplicas de `--rope-protocolos` (ou, sem ele, de `ROPE_PROTOCOLOS`);
3. `ROPE_LIKERT` do script;
4. nenhuma: o resumo é gerado sem a análise bayesiana.

**Calibração.** As réplicas (ex.: `d1`, `d1a`, `d1b`) são execuções idênticas do mesmo protocolo,
um controle negativo: a diferença entre as notas do juiz nelas é ruído, não efeito.
A calibração usa o `calibrar_rope()` do `src/realizar_avaliacoes.py`, o mesmo usado na avaliação humana:
para cada par de réplicas, calcula a menor margem em que o par seria `equivalente` ao limiar (0,95), e a ROPE é a maior delas.
É o mesmo critério da calibração das métricas automáticas, **sem piso**: o valor calibrado é usado como sai.
A única exceção é a calibração degenerar a zero (réplicas com notas idênticas em todos os itens), caso em que
o script trata como problema de validação (ver abaixo) e pede `--rope` ou outras réplicas.

**Sensibilidade.** O resumo repete a comparação com ROPE/2, ROPE e 2·ROPE e conta quantos pares mudam de
categoria (seção 4.3 e `resumo_bayesiana_sensibilidade_rope.csv`), como as métricas automáticas.
Reporte essa tabela junto das conclusões: ela mostra quais delas dependem da margem.

**Validação das réplicas.** Antes de calibrar, cada réplica precisa:

- ter o parquet de saída do juiz ativo com a coluna `avaliacao` (no `sabia4`, a cópia em `avaliacao_llm/sabia4/saida/`);
- ter todas as instâncias avaliadas;
- ter a última execução do juiz concluída (sem chaves pendentes em `saida_juiz_{p}.parquet`).

Se alguma falhar, ou se a calibração degenerar a zero, o script mostra os problemas e o que fará se continuar
(usar `ROPE_LIKERT` ou gerar o resumo sem a análise bayesiana) e pergunta `Continuar mesmo assim? [s/N]`.
Réplicas avaliadas por modelos juiz diferentes geram só um aviso.

As réplicas servem apenas à calibração: o resumo compara só os protocolos de `--protocolos` (ou da lista padrão do perfil).
A calibração usa as notas do juiz ativo: com `--juiz sabia4`, as réplicas precisam ter sido avaliadas pelo sabia4.

No `RESUMO_AVALIACAO_LLM.md`, a seção **3. ROPE Likert** mostra o valor, a origem, os avisos e a tabela da calibração
por par de réplicas (também em `resumo_rope_calibracao.csv`).

## Integração com a comparação

A coluna `avaliacao` guarda o JSON do juiz (`{"nota": 1..4, "problemas": [...]}`), no mesmo formato da avaliação humana
(`converter_label_studio_para_parquet.py`); o `realizar_avaliacoes.py` lê as duas fontes da mesma forma.
Para o `comparar_extracoes.py` usar as notas, inclua no YAML de comparação:

```yaml
llm_as_a_judge: true
campos_parquet:
  avaliacao: "avaliacao"   # nome da coluna do parquet
```

Para usar as notas do **sabia4**, mantenha `campos_parquet` como acima e aponte o `arquivo` de cada modelo
para a cópia do juiz, em um YAML de comparação próprio (ex.: `06_compara_sabia4.yaml`):

```yaml
modelos_comparacao:
  - arquivo: "avaliacao_llm/sabia4/saida/saida_qwen7b(d1)_teste.parquet"   # em vez de saida/...
```

As cópias têm as mesmas colunas da extração (`chave`, `resposta`, `erro`, ...), então as métricas automáticas
não mudam; só as notas são do sabia4. Para medir a concordância entre os juízes por instância,
junte `saida/X.parquet` e `avaliacao_llm/sabia4/saida/X.parquet` pela `chave`.

## Arquivos

Na pasta do juiz (`avaliacao_llm/` para o `gpt5`, `avaliacao_llm/sabia4/` para o `sabia4`), por protocolo:

| Arquivo | Conteúdo |
|---|---|
| `entrada_juiz_{p}.parquet` | Prompts enviados ao juiz na última execução (`chave`, `texto`) |
| `saida_juiz_{p}.parquet` | Respostas do juiz. Acumula entre execuções; serve para retomar. |
| `config_juiz_{p}.yaml` | Configuração passada ao `util_vllm_batch.py` (inclui o modelo juiz usado) |
| `saida_juiz_{p}.parquet.bkp_refazer` | Cópia da saída do juiz feita pelo `--refazer` antes de apagá-la |
| `saida_juiz_{p}_*.log/.csv/.json/.png/.md` | Logs, tempos, tokens e resumo da execução |

Gerados pelo `--resumo`: `RESUMO_AVALIACAO_LLM.md`, `resumo_avaliacoes.csv`, `resumo_rope_calibracao.csv`,
`resumo_bayesiana_sintese.csv`, `resumo_bayesiana_pares.csv`, `resumo_bayesiana_sensibilidade_rope.csv` e os gráficos `avaliacao_llm_heatmap.png` e `avaliacao_llm_diferencas.png`.
Com `--recorte <nome>`, os mesmos arquivos vão para `<pasta do juiz>/recortes/<nome>/`. A ROPE é a mesma em todos os recortes
(calibrada pelas réplicas, não pelos protocolos do recorte), e a ordem do `--protocolos` define a ordem do heatmap.

**Juiz `gpt5` (padrão):** em `saida/`, o parquet de cada protocolo ganha a coluna `avaliacao`; antes de cada gravação é criado um `.bak`.
Para `a`, `prof` e `base`, o resultado é gravado no parquet `_teste` e sincronizado no parquet completo de origem
(`base`: `saida_oa_gpt5.parquet` → `saida_oa_gpt5(base)_teste.parquet`).

**Juiz `sabia4` (isolado):** as notas ficam em `avaliacao_llm/sabia4/saida/`, uma cópia por protocolo com o mesmo nome
do parquet de `saida/` (para `a`, `prof` e `base`, a cópia é do `_teste`, já filtrado), com seu próprio `.bak`.
Nada é gravado em `saida/` nem nos parquets completos.

```
avaliacao_llm/
├── config_juiz_{p}.yaml, entrada_juiz_{p}.parquet, saida_juiz_{p}.parquet, RESUMO_AVALIACAO_LLM.md, ...   ← gpt5
└── sabia4/
    ├── config_juiz_{p}.yaml, entrada_juiz_{p}.parquet, saida_juiz_{p}.parquet, RESUMO_AVALIACAO_LLM.md, ...   ← sabia4
    └── saida/saida_qwen7b({p})_teste.parquet, saida_or_235b(prof)_teste.parquet, ...   ← notas do sabia4
```

## Cuidados

- **Modelo juiz consistente.** Compare na análise bayesiana apenas protocolos avaliados pelo mesmo modelo juiz;
  o modelo usado em cada protocolo está em `config_juiz_{p}.yaml` da pasta do juiz (campo `caminho`).
  Cada `--resumo` usa só as notas de um juiz (o do `--juiz`), e o cabeçalho do `RESUMO_AVALIACAO_LLM.md` indica qual.
  `oa:gpt5:m:l` e `oa:gpt5-chat:m:l` são hoje o mesmo modelo em implantações diferentes; mesmo assim, o
  `--continuar` pergunta e a calibração da ROPE avisa quando os nomes diferem, para o pesquisador decidir.
  Depois de continuar com modelos diferentes, o YAML registra só o modelo atual.
- **Cópia do sabia4 e extração regenerada.** Se a extração de um protocolo em `saida/` for regenerada, a cópia
  em `avaliacao_llm/sabia4/saida/` fica desatualizada (ela é lida no lugar do original). Apague a cópia e rode
  `--juiz sabia4 --refazer` para esse protocolo. O script não detecta esse caso: enquanto a cópia existir,
  o original de `saida/` não é lido.
- **`--refazer` sobrescreve o backup.** O `.bkp_refazer` guarda só a saída imediatamente anterior;
  um segundo `--refazer` substitui o backup do primeiro.
- **`--reconstruir` regenera o resumo só com os protocolos informados.** Depois de reconstruir um subconjunto,
  rode `--resumo` sem `--protocolos` para refazer o relatório com todos.
- **Ambiente Python.** Use o ambiente `transf13` (`/root/anaconda3/envs/transf13/bin/python`);
  o Python base do anaconda falha ao importar o pandas (incompatibilidade com NumPy 2).
