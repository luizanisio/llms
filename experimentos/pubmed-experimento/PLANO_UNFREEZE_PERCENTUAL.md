# Roteiro — `unfreeze_layers_from` com percentual ("75%" = congela 75%, treina os 25% finais)

**Semântica:** `unfreeze_layers_from: 75%` congela os primeiros 75% dos blocos transformer e
treina os 25% finais. No Qwen 1.5B (28 blocos): `round(28 × 0.75) = 21` → blocos 21–27
treináveis (7 camadas finais) — idêntico ao `unfreeze_layers_from: 21` atual.

**Decisão de design:** o percentual NÃO é convertido no parse do YAML (o número de blocos só
é conhecido após o carregamento do modelo). A etapa carrega a especificação bruta em um campo
novo (`unfreeze_layers_pct`) e a resolução para índice absoluto acontece em
`_aplicar_unfreeze_parcial`, via `model.config.num_hidden_layers`. Os dois formatos (inteiro
absoluto e percentual) coexistem na mesma chave YAML.

---

## Passo 1 — `treinar_unsloth_pipeline.py`

### 1.1 `EtapaCurriculum`: campo adicional

```python
    unfreeze_layers_from: int = -1    # -1 = desativado. N >= 0 = índice absoluto do 1º bloco treinável
    unfreeze_layers_pct: float = -1.0 # -1 = desativado. 0..100 = % de blocos CONGELADOS a partir da base
                                      # (resolvido para índice absoluto após o load do modelo)
```

### 1.2 `construir_etapas()`: parse dual (int ou "NN%")

Substituir a linha `unfreeze_layers_from=int(item.get("unfreeze_layers_from", -1)),` por um
parse antes da construção do `EtapaCurriculum`:

```python
        # --- unfreeze_layers_from: aceita índice absoluto (int) ou percentual ("75%") ---
        _uf_raw = item.get("unfreeze_layers_from", -1)
        _uf_abs, _uf_pct = -1, -1.0
        if isinstance(_uf_raw, str) and _uf_raw.strip().endswith("%"):
            try:
                _uf_pct = float(_uf_raw.strip().rstrip("%").strip())
            except ValueError:
                raise ValueError(
                    f"Etapa '{alias}': unfreeze_layers_from='{_uf_raw}' inválido. "
                    f"Use um inteiro (índice do bloco) ou percentual como '75%'."
                )
            if not (0.0 <= _uf_pct <= 100.0):
                raise ValueError(
                    f"Etapa '{alias}': unfreeze_layers_from percentual deve estar em 0%..100%, "
                    f"recebido: {_uf_pct}%"
                )
        else:
            _uf_abs = int(_uf_raw)
```

E na construção do dataclass:

```python
            unfreeze_layers_from=_uf_abs,
            unfreeze_layers_pct=_uf_pct,
```

### 1.3 Validação de tipo (ajustar a existente)

A validação "só com tipo full" passa a cobrir os dois formatos:

```python
        if (etapa.unfreeze_layers_from >= 0 or etapa.unfreeze_layers_pct >= 0) and etapa.tipo != "full":
            raise ValueError(
                f"Etapa '{alias}': 'unfreeze_layers_from' só é válido com tipo: \"full\" "
                f"(recebido tipo='{etapa.tipo or '(vazio)'}')."
            )
```

### 1.4 Log do curriculum (ajustar a linha do `pace_info`)

```python
        if e.unfreeze_layers_pct >= 0:
            pace_info += f", unfreeze={e.unfreeze_layers_pct:g}%"
        elif e.unfreeze_layers_from >= 0:
            pace_info += f", unfreeze>={e.unfreeze_layers_from}"
```

> Nota YAML: `unfreeze_layers_from: 75%` sem aspas já é lido como string `"75%"` pelo
> `yaml.safe_load` (o `%` impede interpretação numérica). Aspas são opcionais.

---

## Passo 2 — `treinar_unsloth.py`

### 2.1 Resolução percentual → índice em `_aplicar_unfreeze_parcial`

Mudar a assinatura para receber a etapa (ou os dois valores) e resolver no início do método:

```python
    def _aplicar_unfreeze_parcial(self, etapa) -> None:
        """Descongelamento progressivo. Aceita índice absoluto (unfreeze_layers_from)
        ou percentual de blocos congelados (unfreeze_layers_pct), resolvido aqui
        contra num_hidden_layers do modelo carregado."""
        alias = etapa.alias
        n_layers = getattr(self.model.config, "num_hidden_layers", None)

        if etapa.unfreeze_layers_pct >= 0:
            if n_layers is None:
                raise ValueError(
                    f"Etapa '{alias}': unfreeze percentual requer model.config.num_hidden_layers, "
                    f"não encontrado neste modelo. Use o formato absoluto (índice do bloco)."
                )
            from_layer = int(round(n_layers * etapa.unfreeze_layers_pct / 100.0))
            from_layer = max(0, min(from_layer, n_layers))  # clamp defensivo
            n_finais = n_layers - from_layer
            pct_finais = 100.0 - etapa.unfreeze_layers_pct
            logger.info(
                f"🧊 Unfreeze {etapa.unfreeze_layers_pct:g}% congelado "
                f"({n_finais} camadas finais treináveis ≈ {pct_finais:g}%): "
                f"blocos {from_layer}-{n_layers - 1} de {n_layers} (etapa '{alias}')"
            )
        else:
            from_layer = etapa.unfreeze_layers_from

        # ... restante do método permanece igual, usando `from_layer` ...
```

O corpo existente (regex, embeddings, tied, `enable_input_require_grads`, stats) não muda —
apenas passa a operar sobre o `from_layer` resolvido.

### 2.2 Chamada no ramo `full` de `_aplicar_etapa_curriculum`

Atualizar a condição e a chamada:

```python
            if getattr(etapa, "unfreeze_layers_from", -1) >= 0 or getattr(etapa, "unfreeze_layers_pct", -1) >= 0:
                self._aplicar_unfreeze_parcial(etapa)
```

(Se preferir preservar a assinatura antiga `(from_layer, alias)` para não tocar em chamadas
existentes, resolva o percentual dentro de `_aplicar_etapa_curriculum` e mantenha o método
como está — as duas formas são equivalentes; a do 2.1 concentra a lógica num único ponto.)

---

## Passo 3 — Casos de borda (comportamento definido)

| Entrada | Resolução (28 blocos) | Comportamento |
|---|---|---|
| `75%` | from_layer=21 | 7 camadas finais treináveis |
| `50%` | from_layer=14 | 14 finais |
| `0%` | from_layer=0 | tudo treinável (= full tradicional, inclui embeddings tied) |
| `100%` | from_layer=28 | nenhum bloco treinável → warning existente ("apenas norm/lm_head") |
| `"setenta%"` | — | ValueError no parse (mensagem clara) |
| `150%` / `-5%` | — | ValueError no parse (fora de 0..100) |
| `21` (int) | from_layer=21 | comportamento absoluto atual, inalterado |

O arredondamento usa `round()` — em modelos onde o percentual não divide exato (ex.: 25% de
28 = 7.0, mas 30% de 28 = 8.4 → 8), o log registra o valor efetivo, então a aproximação fica
documentada nos artefatos do experimento.

---

## Passo 4 — Checklist de validação

1. `--datasets` com `unfreeze_layers_from: 75%` → log do curriculum mostra `unfreeze=75%`, sem erro.
2. Início da etapa → log `🧊 Unfreeze 75% congelado (7 camadas finais treináveis ≈ 25%): blocos 21-27 de 28`.
3. `unfreeze_layers_from: 21` (int) continua funcionando idêntico ao antes (regressão zero).
4. Percentual em etapa `tipo: "lora"` → ValueError na carga do YAML.
5. `"75"` sem `%` → interpretado como bloco absoluto 75 → cai no warning existente de
   `from_layer >= num_hidden_layers` (comportamento definido, não silencioso).
6. Contador `Modo FULL: X/Y parâmetros` cresce entre etapas como no formato absoluto.

## Exemplo de uso (D19 reescrito em percentual — portável entre modelos)

```yaml
  divisao:
  - dataset_filtro: {"dificuldade": "facil"}
    alias: "fácil-uf75"
    tipo: "full"
    unfreeze_layers_from: 75%    # 1.5B: blocos 21-27 | 7B: blocos 21-27 (28 camadas em ambos)
    pace_epochs: 2
    learning_rate: 5e-06
  - dataset_filtro: {"dificuldade": "medio"}
    alias: "médio-uf50"
    tipo: "full"
    unfreeze_layers_from: 50%
    pace_epochs: 2
    learning_rate: 5e-06
  - dataset_filtro: {"dificuldade": "dificil"}
    alias: "difícil-uf25"
    tipo: "full"
    unfreeze_layers_from: 25%
    pace_epochs: 2
    learning_rate: 5e-06
  - alias: "completo-uf0"
    tipo: "full"
    unfreeze_layers_from: 0%     # equivale a full completo
    pace_epochs: 2
    learning_rate: 3e-06
```