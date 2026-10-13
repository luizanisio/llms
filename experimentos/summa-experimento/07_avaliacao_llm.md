# 07_avaliacao_llm.py — Avaliação LLM-as-a-judge

Envia cada extração do conjunto de teste ao modelo juiz (`MODELO_JUIZ`) com o prompt
`avaliacao_llm_humana/prompt_juiz_llm.txt` e grava a nota Likert (1–4) na coluna `avaliacao`
do parquet de saída do protocolo (`saida/saida_qwen7b({protocolo})_teste.parquet`).

Extrações inválidas (erro na extração, JSON inválido, sem temas ou sem íntegra) não vão ao juiz:
recebem nota 1 direta (`{"nota": 1, "problemas": ["json_invalido"]}`).

## Parâmetros

| Parâmetro | O que faz | Chama a API? |
|---|---|:---:|
| *(nenhum)* | Avalia só as instâncias com a coluna `avaliacao` vazia. Retoma de onde parou. | sim |
| `--protocolos d1 d7 ...` | Escolhe os protocolos. Sem ele, usa a lista `PROTOCOLOS` do script. Aceita os especiais `a` (zero-shot) e `prof` (professor). | — |
| `--refazer` | Descarta as avaliações existentes e reenvia **tudo** ao juiz. | sim |
| `--continuar` | Retoma uma execução do juiz que foi interrompida, usando a entrada que ela já tinha gerado. | sim |
| `--reconstruir` | Refaz a coluna `avaliacao` a partir de `avaliacao_llm/saida_juiz_{p}.parquet`, sem chamar o juiz. Gera também o resumo. | não |
| `--resumo` | Gera `avaliacao_llm/RESUMO_AVALIACAO_LLM.md`, os CSVs e os gráficos bayesianos. | não |
| `--tokens` | Estima os tokens de entrada e saída do que seria enviado. | não |
| `--rope 0.15` | ROPE da análise bayesiana Likert (usado por `--resumo` e `--reconstruir`). | — |

`--refazer`, `--continuar` e `--reconstruir` são mutuamente exclusivos.
Os modos que chamam a API mostram um resumo e pedem confirmação (`Continuar? [S/n]`) antes de começar.

## O que usar em cada situação

| Situação | Comando |
|---|---|
| Avaliar um protocolo pela primeira vez | `python 07_avaliacao_llm.py --protocolos d1` |
| Avaliar instâncias novas/pendentes de um protocolo já avaliado | `python 07_avaliacao_llm.py --protocolos d1` |
| Parei uma execução **sem** `--refazer` | Rode o mesmo comando de novo (ou use `--continuar`) |
| Parei uma execução **com** `--refazer` | `python 07_avaliacao_llm.py --protocolos d1 --continuar` |
| Quero substituir avaliações antigas (mudou o modelo juiz, o prompt ou a extração) | `python 07_avaliacao_llm.py --protocolos d1 --refazer` |
| A coluna `avaliacao` foi perdida ou o parquet de saída foi regenerado, mas a saída do juiz está completa | `python 07_avaliacao_llm.py --protocolos d1 --reconstruir` |
| Atualizar o relatório comparativo | `python 07_avaliacao_llm.py --resumo` |
| Saber quanto vai custar antes de rodar | `python 07_avaliacao_llm.py --protocolos d1 --tokens` (com `--refazer`, estima o reprocessamento completo) |

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
2. o modelo juiz da execução anterior é o mesmo `MODELO_JUIZ` atual;
3. todas as chaves da entrada existem no parquet de extração;
4. os prompts da entrada são idênticos aos montados agora (mesma extração, íntegra e template).

Se alguma falhar, nada é processado. Corrija a causa (por exemplo, volte `MODELO_JUIZ` ao modelo usado)
ou descarte a execução anterior com `--refazer`.

Ao aplicar o resultado:

- chaves enviadas ao juiz recebem a nota dele (nota 1 se o juiz deu erro em todas as tentativas);
- extrações inválidas recebem nota 1 direta;
- as demais chaves mantêm a avaliação que já tinham.

### `--continuar` × `--reconstruir`

O `--reconstruir` dá nota 1 a **toda** chave que não está na saída do juiz.
Isso só é correto quando a saída do juiz cobre todas as extrações válidas, por exemplo depois de um `--refazer` concluído.
Depois de execuções incrementais ou interrompidas, use `--continuar`, que preserva as avaliações já existentes.

## Arquivos

Em `avaliacao_llm/`, por protocolo:

| Arquivo | Conteúdo |
|---|---|
| `entrada_juiz_{p}.parquet` | Prompts enviados ao juiz na última execução (`chave`, `texto`) |
| `saida_juiz_{p}.parquet` | Respostas do juiz. Acumula entre execuções; serve para retomar. |
| `config_juiz_{p}.yaml` | Configuração passada ao `util_vllm_batch.py` (inclui o modelo juiz usado) |
| `saida_juiz_{p}.parquet.bkp_refazer` | Cópia da saída do juiz feita pelo `--refazer` antes de apagá-la |
| `saida_juiz_{p}_*.log/.csv/.json/.png/.md` | Logs, tempos, tokens e resumo da execução |

Em `saida/`, o parquet de cada protocolo ganha a coluna `avaliacao`; antes de cada gravação é criado um `.bak`.
Para `a` e `prof`, o resultado é gravado no parquet `_teste` e sincronizado no parquet completo de origem.

## Cuidados

- **Modelo juiz consistente.** Compare na análise bayesiana apenas protocolos avaliados pelo mesmo `MODELO_JUIZ`;
  o modelo usado em cada protocolo está em `avaliacao_llm/config_juiz_{p}.yaml` (campo `caminho`).
- **`--refazer` sobrescreve o backup.** O `.bkp_refazer` guarda só a saída imediatamente anterior;
  um segundo `--refazer` substitui o backup do primeiro.
- **`--reconstruir` regenera o resumo só com os protocolos informados.** Depois de reconstruir um subconjunto,
  rode `--resumo` sem `--protocolos` para refazer o relatório com todos.
- **Ambiente Python.** Use o ambiente `transf13` (`/root/anaconda3/envs/transf13/bin/python`);
  o Python base do anaconda falha ao importar o pandas (incompatibilidade com NumPy 2).
