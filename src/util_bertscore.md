# BERTScore — Estratégias e Notas Técnicas

**Autor:** Luiz Anísio  
**Fonte:** [github.com/luizanisio/llms/tree/main/src](https://github.com/luizanisio/llms/tree/main/src)

---

## Visão Geral

O módulo `util_bertscore.py` calcula BERTScore (Precision, Recall e F1) comparando textos usando embeddings contextuais de modelos BERT. Inclui:

- **Cache em disco** baseado em MD5 para evitar recálculos
- **Janela deslizante** para textos que excedem o limite do modelo
- **Suporte a modelos personalizados** (ex: modelos jurídicos)

---

## Janela Deslizante para Textos Longos

### O Problema

Modelos BERT possuem um limite rígido de tokens definido por `max_position_embeddings` (geralmente 512, mas pode variar — ex: LongBERT usa 4096). Textos que ultrapassam esse limite causam erro de runtime.

### A Solução

Quando um texto excede o limite do modelo, ele é dividido automaticamente em **janelas sobrepostas** (sliding windows) de tokens:

1. O texto é tokenizado com o tokenizer do modelo
2. Se o número de tokens excede `max_position_embeddings - 2` (descontando `[CLS]` e `[SEP]`), o texto é dividido em janelas
3. Cada janela tem o tamanho máximo permitido, com sobreposição configurável (padrão: 50%)
4. Os scores são calculados para cada par de janelas alinhadas
5. O resultado final é a **média ponderada** pelo número de tokens de cada janela

### Alinhamento de Janelas

Quando ambos os textos (pred e true) são longos, as janelas são alinhadas na ordem:
- Janela 1 do pred ↔ Janela 1 do true
- Janela 2 do pred ↔ Janela 2 do true
- etc.

Se um texto gera mais janelas que o outro, o texto mais curto repete sua última janela para alinhar. Isso significa que se um texto é curto (cabe em 1 janela), cada janela do texto longo é comparada com o texto completo.

### Média Ponderada

A combinação dos scores entre janelas usa média ponderada. O peso de cada par de janelas é a soma do número de tokens das duas janelas (pred + true). Assim, janelas com mais conteúdo (tipicamente as primeiras) têm mais influência no score final do que a última janela, que pode ser menor.

### Heurística de Eficiência

Para evitar tokenização desnecessária em textos curtos (que são a maioria), uma heurística rápida verifica primeiro o número de palavras. Apenas textos com mais de `max_tokens × 0.7` palavras são tokenizados para confirmação. Textos claramente curtos seguem o caminho rápido (batch único sem janelamento).

---

## Detecção Automática do Limite do Modelo

O limite de tokens é lido automaticamente do campo `max_position_embeddings` do config do modelo via `transformers.AutoConfig`. Isso garante compatibilidade com qualquer modelo BERT.

### Margem de Segurança Proporcional

Em vez de subtrair um valor fixo (como 2 para `[CLS]`/`[SEP]`), o módulo aplica uma **margem proporcional** controlada pela constante `BERTSCORE_MAX_POSITION_PERCENTAGE` (padrão: 95%).

Isso é mais robusto porque:
- Tokens especiais (`[CLS]`, `[SEP]`) são adicionados pela biblioteca `bert_score` internamente
- O ciclo encode → decode → re-encode (necessário para a janela deslizante) pode gerar tokens extras
- A margem proporcional escala automaticamente para modelos com contextos maiores

| Modelo | max_position_embeddings | Margem (95%) | Max Tokens Efetivo |
|--------|------------------------|-------------|--------------------|
| bert-base-multilingual-cased | 512 | 5% = ~26 | 486 |
| bert-large-portuguese-cased-legal | 512 | 5% = ~26 | 486 |
| LongBERT / BigBird | 4096 | 5% = ~205 | 3891 |

---

## Cache em Disco

Resultados são armazenados em arquivos JSON individuais baseados no hash MD5 dos textos:

- A chave de cache é baseada no texto **original** (antes de qualquer janelamento)
- A ordem dos textos não importa: `(A, B)` e `(B, A)` geram a mesma chave, com P e R invertidos automaticamente
- O cache é segregado por modelo para evitar colisão entre modelos diferentes

### Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `BERTSCORE_DEVICE` | `auto` | Device: `auto`, `cuda`, `cpu` |
| `BERTSCORE_CACHE_PATH` | `./_bertmodels/bs_cache/` | Diretório do cache |
| `BERTSCORE_OVERLAP` | `0.5` | Fração de overlap da janela deslizante (0.0 a 0.9) |
| `BERTSCORE_MAX_POSITION_PERCENTAGE` | `0.95` | Percentual do `max_position_embeddings` a usar (margem de segurança) |

---

## Uso Rápido

```python
from util_bertscore import bscore

# Cálculo básico
P, R, F1 = bscore(preds, trues, lang='pt', verbose=True)

# Com modelo personalizado (jurídico)
P, R, F1 = bscore(preds, trues, model_type='stjiris/bert-large-portuguese-cased-legal-mlm-mkd-nli-sts-v1')

# Textos longos são tratados automaticamente — não é necessária nenhuma configuração adicional
```
