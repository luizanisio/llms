#!/usr/bin/env python3
"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Modo FUSÃO — execução contínua de curriculum (CL) e progressão de capacidade (PT).

No modo SEGMENTADO (padrão), cada etapa de ``curriculum.divisao`` roda um
``trainer.train()`` próprio, resetando otimizador e scheduler a cada fronteira.
No modo FUNDIDO (``curriculum.fusao.ativo: true``), todas as etapas viram *spans*
de um único dataset concatenado percorrido por UM único ``trainer.train()``:

- as fronteiras entre etapas são apenas marcadores virtuais de log/gráfico;
- o scheduler é um único cosine (warmup global → decaimento até ~0);
- ``unfreeze_layers_from`` deixa de congelar pesos e vira *gating de learning
  rate por grupos de camadas*: todos os parâmetros treináveis entram no
  otimizador desde o step 0, mas grupos "adormecidos" recebem LR=0 até o span
  que os liberta (acumulando momentos Adam enquanto dormem).

Este módulo agrupa todo o código específico da fusão:
- ``ConfigFusao``: dataclass do bloco ``curriculum.fusao``;
- ``validar_fusao``: contrato de validação (roda na carga do YAML);
- ``construir_dataset_fundido``: concatenação dos spans com épocas virtuais;
- ``resolver_corte`` / ``montar_grupos_gating``: grupos de parâmetros por faixa
  de camadas e steps de ativação;
- ``get_fusao_trainer_cls``: subclasse do SFTTrainer (sampler sequencial,
  otimizador com param groups e LambdaLR de gating);
- ``get_marcadores_virtuais_cls``: callback de eventos de span + log de LR
  por grupo;
- ``hash_fusao`` / ``salvar_fusao_spans`` / ``verificar_hash_resume``: guarda de
  consistência do resume.

Imports pesados (trl/transformers/torch/datasets) são feitos de forma tardia
para que a carga do YAML (que importa ConfigFusao via treinar_unsloth_util)
continue leve.
"""

import os
import re
import json
import math
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from treinar_unsloth_logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

@dataclass
class ConfigFusao:
    """Bloco ``curriculum.fusao`` do YAML (irmão de ``divisao``)."""
    ativo: bool = False
    tipo: str = "lora"              # regime único do run: "lora" | "full"
    learning_rate: float = 2e-5     # pico do cosine global
    warmup_grupo_steps: int = 100   # rampa de cada grupo ao acordar
    seed_shuffle: int = 3407        # shuffle intra-span determinístico

    def __post_init__(self):
        if self.tipo not in ("lora", "full"):
            raise ValueError(f"fusao.tipo deve ser 'lora' ou 'full', recebido: {self.tipo}")
        if self.learning_rate <= 0:
            raise ValueError("fusao.learning_rate deve ser > 0")
        if self.warmup_grupo_steps < 0:
            raise ValueError("fusao.warmup_grupo_steps deve ser >= 0")


_CHAVES_FUSAO_VALIDAS = {"ativo", "tipo", "learning_rate", "warmup_grupo_steps", "seed_shuffle"}


def processar_config_fusao(curriculum_raw: dict) -> ConfigFusao:
    """Lê ``curriculum.fusao`` do YAML bruto; ausente → ConfigFusao(ativo=False)."""
    fusao_raw = (curriculum_raw or {}).get("fusao", None)
    if not fusao_raw:
        return ConfigFusao()
    if not isinstance(fusao_raw, dict):
        raise ValueError(f"curriculum.fusao deve ser um dicionário, recebido: {type(fusao_raw).__name__}")

    desconhecidas = set(fusao_raw.keys()) - _CHAVES_FUSAO_VALIDAS
    if desconhecidas:
        logger.warning(
            f"⚠️  curriculum.fusao: chaves desconhecidas IGNORADAS: {sorted(desconhecidas)}. "
            f"Verifique typos (chaves válidas: {sorted(_CHAVES_FUSAO_VALIDAS)})"
        )

    return ConfigFusao(
        ativo=fusao_raw.get("ativo", False) in {True, "true", "True", 1, "1", "sim"},
        tipo=str(fusao_raw.get("tipo", "lora")),
        learning_rate=float(fusao_raw.get("learning_rate", 2e-5)),
        warmup_grupo_steps=int(fusao_raw.get("warmup_grupo_steps", 100)),
        seed_shuffle=int(fusao_raw.get("seed_shuffle", 3407)),
    )


# ---------------------------------------------------------------------------
# Validador do contrato (roda na carga do YAML, antes de tocar GPU)
# ---------------------------------------------------------------------------

# No modo fundido esses valores são GLOBAIS (curriculum.fusao / treinamento);
# declará-los por etapa indicaria expectativa de comportamento segmentado.
_CHAVES_PROIBIDAS_EM_FUSAO = {"tipo", "learning_rate", "batch_size", "max_seq_length",
                              "warmup_steps", "pace_loss", "pace_epochs_max"}


def corte_da_etapa(etapa) -> float:
    """Corte de unfreeze declarado na etapa (percentual tem precedência); -1 = sem corte."""
    pct = getattr(etapa, "unfreeze_layers_pct", -1.0)
    if pct >= 0:
        return pct
    return float(getattr(etapa, "unfreeze_layers_from", -1))


def validar_fusao(fusao: ConfigFusao, etapas: list) -> None:
    """Valida o contrato do modo fundido sobre as etapas TREINÁVEIS do curriculum.

    Regras (§7 do plano): chaves por-etapa proibidas, pace_epochs >= 1,
    cortes de unfreeze monotônicos não crescentes e primeira etapa com corte
    quando há gating. Falha com mensagem citando etapa e chave.
    """
    if not fusao or not fusao.ativo:
        return
    if not etapas:
        raise ValueError("Modo FUSÃO: nenhuma etapa treinável em curriculum.divisao")

    cortes = []
    for i, e in enumerate(etapas):
        declaradas = getattr(e, "chaves_yaml", set()) or set()
        proibidas = _CHAVES_PROIBIDAS_EM_FUSAO & declaradas
        if proibidas:
            raise ValueError(
                f"Modo FUSÃO: etapa {i} ('{e.alias}') declara {sorted(proibidas)} — "
                f"no modo fundido esses valores são globais (curriculum.fusao / treinamento). "
                f"Remova-os da etapa ou desative a fusão."
            )
        if e.pace_epochs < 1:
            raise ValueError(f"Modo FUSÃO: etapa {i} ('{e.alias}') requer pace_epochs >= 1")
        cortes.append(corte_da_etapa(e))

    declarados = [c for c in cortes if c >= 0]
    if declarados:
        # gating: cortes devem ser monotônicos não crescentes (re-congelar não é suportado)
        if any(b > a for a, b in zip(declarados, declarados[1:])):
            raise ValueError(
                "Modo FUSÃO: unfreeze_layers_from deve ser monotônico não crescente "
                f"ao longo dos spans; recebido: {declarados}"
            )
        # se alguma etapa declara, exigir que a PRIMEIRA declare (define o G1 sempre-ativo)
        if cortes[0] < 0:
            raise ValueError(
                "Modo FUSÃO com gating: a primeira etapa deve declarar "
                "unfreeze_layers_from (define o grupo ativo desde o step 0)."
            )


# ---------------------------------------------------------------------------
# Dataset concatenado
# ---------------------------------------------------------------------------

def construir_dataset_fundido(etapas: list, fusao: ConfigFusao,
                              carregar_span_fn: Callable, batch_efetivo: int):
    """Concatena os spans na ordem do currículo, com épocas virtuais e shuffle intra-span.

    Cada etapa vira um *span* do stream; ``pace_epochs: k`` insere o span k vezes,
    com shuffle independente por repetição (seed derivada:
    ``fusao.seed_shuffle + idx_etapa*100 + idx_repeticao`` — determinístico,
    requisito do resume por fast-forward).

    Decisões fixadas:
    - s_g (step de ativação de cada span) é medido em **optimizer steps**:
      ``instancias_inicio // batch_efetivo`` com ``batch_efetivo =
      batch_size × grad_batch_size × n_gpus``. Batches que cruzam fronteira de
      span são aceitos sem tratamento (afeta ≤1 batch por transição).
    - ``spans_meta`` é serializado em ``<saida>/fusao_spans.json`` junto com o
      hash da configuração (ver ``hash_fusao``) — insumo de relatórios, callback
      de marcadores e guarda de resume.

    Returns:
        (dataset_concatenado, spans_meta)
    """
    from datasets import concatenate_datasets

    partes, spans_meta, inst_acum = [], [], 0
    for idx, etapa in enumerate(etapas):
        ds = carregar_span_fn(etapa)          # mesma lógica de filtro/validação atual
        if ds is None or len(ds) == 0:
            raise ValueError(f"Modo FUSÃO: span {idx} ('{etapa.alias}') sem instâncias de treino")
        inicio_inst = inst_acum
        for rep in range(etapa.pace_epochs):
            seed = fusao.seed_shuffle + idx * 100 + rep
            partes.append(ds.shuffle(seed=seed))
            inst_acum += len(ds)
        spans_meta.append({
            "alias": etapa.alias,
            "idx": idx,
            "instancias_inicio": inicio_inst,
            "instancias_fim": inst_acum,                       # exclusivo
            "step_inicio": inicio_inst // batch_efetivo,       # optimizer steps (floor)
            "unfreeze_corte": corte_da_etapa(etapa),           # -1 = sem gating novo
            "pace_epochs": etapa.pace_epochs,
        })
        logger.info(
            f"<cinza>   • span {idx} ('{etapa.alias}'): {len(ds)} instâncias × "
            f"pace_epochs={etapa.pace_epochs} → stream [{inicio_inst}, {inst_acum}) "
            f"(step_inicio={spans_meta[-1]['step_inicio']}, "
            f"corte={spans_meta[-1]['unfreeze_corte']:g})</cinza>"
        )
    return concatenate_datasets(partes), spans_meta


# ---------------------------------------------------------------------------
# Grupos de parâmetros e gating
# ---------------------------------------------------------------------------

# Regex idêntico ao de treinar_unsloth.py: casa "model.layers.N." e também o
# prefixo PEFT "base_model.model.model.layers.N.".
_RE_IDX_CAMADA = re.compile(r"\.layers\.(\d+)\.")


def resolver_corte_meta(corte: float, eh_percentual: bool, n_layers: int) -> int:
    """Percentual de blocos congelados OU índice absoluto → índice do bloco de corte."""
    if eh_percentual:
        from_layer = int(round(n_layers * corte / 100.0))
    else:
        from_layer = int(corte)
    return max(0, min(from_layer, n_layers))  # clamp defensivo


def montar_grupos_gating(model, spans_meta: List[Dict], n_layers: int,
                         fusao_tipo: str, etapas: list) -> Optional[List[Dict]]:
    """Monta os grupos de gating em ordem DETERMINÍSTICA (requisito de resume).

    Os grupos derivam do conjunto de cortes distintos declarados
    (ex.: 75/50/25/0 → 4 grupos), ordenados do corte mais alto para o mais
    baixo. Faixas de camadas: G1=[lim_0, n_layers), G2=[lim_1, lim_0), ...
    Cabeça/norm (não-layers) sempre no primeiro grupo; ``embed_tokens`` (tied
    no Qwen 1.5B) pertence ao grupo do corte 0 — se nenhum span declara 0,
    fica FORA do otimizador (nunca treina no run).

    A ordem de montagem depende apenas do YAML (cortes/spans) e da arquitetura
    (named_parameters), nunca de estado de runtime — o resume do HF mapeia o
    estado do otimizador POR POSIÇÃO nos param_groups.

    Regime "lora": chamar APÓS aplicar LoRA (só entram parâmetros com
    ``requires_grad=True``, i.e. os adapters). Regime "full": chamar após
    marcar ``requires_grad=True`` nos float params (todos entram; nada é
    congelado — o gating substitui o congelamento).

    Args:
        model: modelo já preparado (LoRA aplicado ou full destravado)
        spans_meta: saída de construir_dataset_fundido
        n_layers: model.config.num_hidden_layers
        fusao_tipo: "lora" | "full"
        etapas: etapas treináveis (para saber se o corte é percentual ou absoluto)

    Returns:
        Lista de grupos [{nome, corte, faixa, ativacao, params: [(name, p), ...]}]
        em ordem determinística, ou None se nenhum span declara corte.
    """
    # 1) cortes distintos em ordem de declaração + step de ativação (primeiro span que declara)
    cortes: List[Tuple[float, int, bool]] = []   # (corte, step_ativacao, eh_percentual)
    vistos = set()
    for s in spans_meta:
        c = s["unfreeze_corte"]
        if c >= 0 and c not in vistos:
            vistos.add(c)
            etapa = etapas[s["idx"]]
            eh_pct = getattr(etapa, "unfreeze_layers_pct", -1.0) >= 0
            cortes.append((c, s["step_inicio"], eh_pct))
    if not cortes:
        return None

    cortes.sort(key=lambda x: -x[0])                              # 75, 50, 25, 0
    limites = [resolver_corte_meta(c, pct, n_layers) for c, _, pct in cortes]  # ex.: [21, 14, 7, 0]

    # 2) faixas de camadas por grupo (decrescentes a partir do topo)
    faixas = []
    topo = n_layers
    for lim in limites:
        faixas.append(range(lim, topo))
        topo = lim

    grupos_params: List[List[Tuple[str, Any]]] = [[] for _ in faixas]
    extras_g1: List[Tuple[str, Any]] = []     # norm final, lm_head não-tied etc.
    embeds: List[Tuple[str, Any]] = []        # embed_tokens (tied) → grupo do corte 0
    fora = 0                                   # camadas abaixo do último corte (se corte mínimo > 0)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            # LoRA: só adapters treináveis entram; full: tudo float já está True
            continue
        m = _RE_IDX_CAMADA.search(name)
        if m:
            idx = int(m.group(1))
            for gi, fx in enumerate(faixas):
                if idx in fx:
                    grupos_params[gi].append((name, p))
                    break
            else:
                # camada abaixo do menor corte declarado: fica fora do otimizador
                # (nunca acorda neste run; requires_grad permanece — nada é congelado)
                fora += p.numel()
        elif "embed_tokens" in name:
            embeds.append((name, p))
        else:
            extras_g1.append((name, p))       # norm/lm_head não-tied → G1
    grupos_params[0] = extras_g1 + grupos_params[0]
    if embeds:
        if limites[-1] == 0:
            grupos_params[-1].extend(embeds)  # acorda com o corte 0
        else:
            logger.info(
                "ℹ️  Gating: nenhum span declara corte 0 — embed_tokens fica FORA do "
                "otimizador (nunca treina neste run)."
            )
    if fora > 0:
        logger.info(
            f"ℹ️  Gating: {fora:,} parâmetros de camadas abaixo do menor corte "
            f"declarado ficam fora do otimizador (nunca acordam neste run)."
        )

    grupos = []
    for (c, s_g, eh_pct), lim, fx, params in zip(cortes, limites, faixas, grupos_params):
        rotulo = f"{c:g}%" if eh_pct else f"{c:g}"
        grupos.append({
            "nome": f"gate_{rotulo}",
            "corte": c,
            "faixa": (fx.start, fx.stop),     # [inicio, fim)
            "ativacao": s_g,
            "params": params,
        })

    if grupos[0]["ativacao"] != 0:
        # O validador (§1.2) garante corte na primeira etapa; o primeiro span
        # começa no step 0, logo o grupo do corte mais alto ativa em 0.
        logger.warning(
            f"⚠️  Gating: grupo inicial '{grupos[0]['nome']}' com ativação em "
            f"step {grupos[0]['ativacao']} (esperado 0)."
        )
    return grupos


def tabela_grupos_gating(grupos: List[Dict], warmup_grupo_steps: int) -> str:
    """Tabela markdown com grupo, faixa de camadas, nº de params e step de ativação."""
    linhas = ["| Grupo | Faixa de camadas | Parâmetros | Step de ativação | Rampa |",
              "|---|---|---|---|---|"]
    for g in grupos:
        n_params = sum(p.numel() for _, p in g["params"])
        ini, fim = g["faixa"]
        faixa = f"{ini}-{fim - 1}" if fim > ini else "(vazia)"
        extra = " (+cabeça/norm)" if g is grupos[0] else ""
        linhas.append(
            f"| {g['nome']} | {faixa}{extra} | {n_params:,} | "
            f"{g['ativacao']} | +{warmup_grupo_steps} steps |"
        )
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Trainer fundido (lazy: importa trl/transformers apenas quando necessário)
# ---------------------------------------------------------------------------

_FUSAO_TRAINER_CLS = None


def get_fusao_trainer_cls():
    """Constrói (uma vez) e retorna a classe FusaoTrainer(SFTTrainer)."""
    global _FUSAO_TRAINER_CLS
    if _FUSAO_TRAINER_CLS is not None:
        return _FUSAO_TRAINER_CLS

    from trl import SFTTrainer
    from transformers import Trainer

    class FusaoTrainer(SFTTrainer):
        """SFTTrainer do modo FUSÃO: sampler sequencial (a ordem do stream É o
        currículo), otimizador com param groups de gating e scheduler LambdaLR
        (cosine global × gate por grupo).

        ``fusao_ctx`` (dict):
            ativa: bool
            grupos: saída de montar_grupos_gating (ou None — sem gating, D21)
            warmup_grupo: fusao.warmup_grupo_steps
        """

        def __init__(self, *args, fusao_ctx=None, **kwargs):
            self._fusao = fusao_ctx or {}
            super().__init__(*args, **kwargs)

        # --- sampler sequencial -------------------------------------------
        # Assinatura varia entre versões do transformers:
        # _get_train_sampler(self) e _get_train_sampler(self, dataset).
        def _get_train_sampler(self, *args, **kwargs):
            if self._fusao.get("ativa"):
                from torch.utils.data import SequentialSampler
                dataset = args[0] if args else self.train_dataset
                return SequentialSampler(dataset)
            return super()._get_train_sampler(*args, **kwargs)

        # --- otimizador com grupos de gating -------------------------------
        def create_optimizer(self):
            if not (self._fusao.get("ativa") and self._fusao.get("grupos")):
                return super().create_optimizer()
            if self.optimizer is None:
                cls, kw = Trainer.get_optimizer_cls_and_kwargs(self.args)
                kw.pop("params", None)
                param_groups, lambdas = self._montar_param_groups_com_decay()
                self._fusao["lambdas"] = lambdas
                self.optimizer = cls(param_groups, **kw)
            return self.optimizer

        def _montar_param_groups_com_decay(self):
            """Subdivide cada grupo de gating em (decay, no-decay), preservando a
            ordem (gate_75-decay, gate_75-nodecay, gate_50-decay, ...) e duplicando
            a lambda correspondente. Ordem determinística: depende só de
            YAML + arquitetura (requisito do resume, mapeado por posição).

            ``get_optimizer_cls_and_kwargs`` não aplica a separação decay/no-decay
            do Trainer padrão quando passamos grupos prontos — replicamos aqui.
            """
            decay_names = set(self.get_decay_parameter_names(self.model))
            wd = self.args.weight_decay
            W = self.args.warmup_steps
            WG = self._fusao["warmup_grupo"]

            param_groups, lambdas = [], []
            for g in self._fusao["grupos"]:
                com_decay = [p for n, p in g["params"] if n in decay_names]
                sem_decay = [p for n, p in g["params"] if n not in decay_names]
                lam = self._make_lambda(g["ativacao"], W, WG)
                if com_decay:
                    param_groups.append({"params": com_decay, "weight_decay": wd,
                                         "name": f"{g['nome']}-decay"})
                    lambdas.append(lam)
                if sem_decay:
                    param_groups.append({"params": sem_decay, "weight_decay": 0.0,
                                         "name": f"{g['nome']}-nodecay"})
                    lambdas.append(lam)
            return param_groups, lambdas

        def _make_lambda(self, s_g: int, W: int, WG: int):
            """LR_grupo(t) = f_global(t) × gate_g(t).

            f_global: warmup linear até W, depois cosine até ~0 em T steps.
            gate_g: 0 antes de s_g, rampa linear por WG steps, satura em 1.
            Grupos que acordam tarde rampam até o valor JÁ DECAÍDO do cosine
            (discriminative LR emergente); enquanto dormem (LR=0), não há update
            nem weight decay no AdamW desacoplado, mas exp_avg/exp_avg_sq
            continuam acumulando.
            """
            def lam(t):
                T = self._fusao.get("total_steps", 0) or 1
                # f_global(t): warmup + cosine único do pico a ~0
                if t < W:
                    f_global = t / max(1, W)
                else:
                    p = (t - W) / max(1, T - W)
                    p = min(max(p, 0.0), 1.0)
                    f_global = 0.5 * (1.0 + math.cos(math.pi * p))
                # gate_g(t)
                if s_g == 0:
                    gate = 1.0
                elif t < s_g:
                    gate = 0.0
                elif t < s_g + WG:
                    gate = (t - s_g) / max(1, WG)
                else:
                    gate = 1.0
                return f_global * gate
            return lam

        def create_scheduler(self, num_training_steps: int, optimizer=None):
            if not (self._fusao.get("ativa") and self._fusao.get("grupos")):
                return super().create_scheduler(num_training_steps, optimizer)
            if self.lr_scheduler is None:
                opt = optimizer if optimizer is not None else self.optimizer
                # total_steps entra nas lambdas via contexto (create_optimizer roda antes)
                self._fusao["total_steps"] = num_training_steps
                lambdas = self._fusao.get("lambdas")
                if lambdas is None or len(lambdas) != len(opt.param_groups):
                    raise RuntimeError(
                        "FusaoTrainer: lambdas e param_groups desalinhados "
                        f"({0 if lambdas is None else len(lambdas)} vs {len(opt.param_groups)}). "
                        "create_optimizer deve rodar antes de create_scheduler."
                    )
                from torch.optim.lr_scheduler import LambdaLR
                self.lr_scheduler = LambdaLR(opt, lr_lambda=lambdas)
                self._created_lr_scheduler = True
            return self.lr_scheduler

    _FUSAO_TRAINER_CLS = FusaoTrainer
    return _FUSAO_TRAINER_CLS


# ---------------------------------------------------------------------------
# Callback de marcadores virtuais
# ---------------------------------------------------------------------------

_MARCADORES_CLS = None


def get_marcadores_virtuais_cls():
    """Constrói (uma vez) e retorna a classe MarcadoresVirtuaisCallback."""
    global _MARCADORES_CLS
    if _MARCADORES_CLS is not None:
        return _MARCADORES_CLS

    from transformers import TrainerCallback

    class MarcadoresVirtuaisCallback(TrainerCallback):
        """Emite eventos de span ao cruzar cada ``step_inicio`` (fronteiras
        virtuais — nenhum reset acontece) e registra a curva de LR por grupo
        em ``fusao_lr_grupos.jsonl`` (evidência do gating para os relatórios).
        """

        def __init__(self, spans_meta, historico=None, lr_log_path: str = "",
                     grupos_info: str = ""):
            self.spans = sorted(spans_meta, key=lambda s: s["step_inicio"])
            self.proximo = 0
            self.historico = historico
            self.lr_log_path = lr_log_path
            self.grupos_info = grupos_info

        def on_train_begin(self, args, state, control, **kw):
            # resume: avança o ponteiro para além dos spans já cruzados
            while (self.proximo < len(self.spans)
                   and self.spans[self.proximo]["step_inicio"] < state.global_step):
                self.proximo += 1

        def on_step_end(self, args, state, control, **kw):
            while (self.proximo < len(self.spans)
                   and state.global_step >= self.spans[self.proximo]["step_inicio"]):
                s = self.spans[self.proximo]
                corte = s.get("unfreeze_corte", -1)
                corte_txt = f"{corte:g}" if corte >= 0 else "(sem gating novo)"
                logger.info(
                    f"<azul>🪧 SPAN INICIADO (virtual): '{s['alias']}' "
                    f"@ step {state.global_step} (corte={corte_txt})</azul>"
                )
                if self.historico is not None:
                    self.historico.registrar_evento(
                        "SPAN INICIADO (virtual)",
                        f"- **alias:** {s['alias']}\n"
                        f"- **step_global:** {state.global_step}\n"
                        f"- **instancias_inicio:** {s['instancias_inicio']}\n"
                        f"- **unfreeze_corte:** {corte_txt}"
                    )
                self.proximo += 1

        def on_log(self, args, state, control, logs=None, **kw):
            """Registra LR por grupo (scheduler.get_last_lr(), um por param_group)."""
            if not self.lr_log_path:
                return
            scheduler = kw.get("lr_scheduler")
            optimizer = kw.get("optimizer")
            try:
                if scheduler is not None:
                    lrs = list(scheduler.get_last_lr())
                elif optimizer is not None:
                    lrs = [g.get("lr", 0.0) for g in optimizer.param_groups]
                else:
                    return
                nomes = []
                if optimizer is not None:
                    nomes = [g.get("name", f"g{i}") for i, g in enumerate(optimizer.param_groups)]
                registro = {"step": state.global_step,
                            "lrs": {(nomes[i] if i < len(nomes) else f"g{i}"): lr
                                    for i, lr in enumerate(lrs)}}
                with open(self.lr_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(registro, ensure_ascii=False) + "\n")
            except Exception:
                pass  # log de LR é auxiliar; nunca derruba o treino

        def on_train_end(self, args, state, control, **kw):
            if self.lr_log_path:
                try:
                    gerar_grafico_lr_grupos(self.lr_log_path)
                except Exception as e:
                    logger.warning(f"⚠️  Erro ao gerar gráfico de LR por grupo: {e}")

    _MARCADORES_CLS = MarcadoresVirtuaisCallback
    return _MARCADORES_CLS


def gerar_grafico_lr_grupos(lr_log_path: str) -> Optional[str]:
    """Gera ``treinamento_lr_grupos.png`` a partir de ``fusao_lr_grupos.jsonl``.

    Uma linha por grupo de gating (apenas grupos '-decay' para não duplicar a
    curva — a lambda é a mesma do par '-nodecay'). É a evidência visual do
    gating: grupos dormem (LR=0), rampam e seguem o envelope do cosine.
    """
    if not os.path.isfile(lr_log_path):
        return None
    registros = []
    with open(lr_log_path, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                try:
                    registros.append(json.loads(linha))
                except json.JSONDecodeError:
                    continue
    if not registros:
        return None

    # séries por grupo (dedup por nome-base sem sufixo -decay/-nodecay)
    series: Dict[str, Dict[str, list]] = {}
    for r in registros:
        step = r.get("step", 0)
        for nome, lr in (r.get("lrs") or {}).items():
            base = nome.replace("-nodecay", "").replace("-decay", "")
            if nome.endswith("-nodecay") and f"{base}-decay" in (r.get("lrs") or {}):
                continue  # mesma lambda do par -decay
            s = series.setdefault(base, {"x": [], "y": []})
            s["x"].append(step)
            s["y"].append(lr)
    if not series:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for nome, s in series.items():
        ax.plot(s["x"], s["y"], label=nome, linewidth=1.6)
    ax.set_xlabel("Step (otimizador)")
    ax.set_ylabel("Learning rate")
    ax.set_title("LR por grupo de camadas — gating do modo FUSÃO")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(os.path.dirname(lr_log_path), "treinamento_lr_grupos.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"📈 Gráfico de LR por grupo salvo em: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Guarda de consistência do resume (hash de configuração)
# ---------------------------------------------------------------------------

ARQ_FUSAO_SPANS = "fusao_spans.json"


def hash_fusao(fusao: ConfigFusao, spans_meta: List[Dict], batch_efetivo: int,
               lr: float, warmup_global: int) -> str:
    """Hash SHA-256 da configuração efetiva da fusão (guarda do resume)."""
    payload = {
        "tipo": fusao.tipo, "lr": lr, "warmup_global": warmup_global,
        "warmup_grupo": fusao.warmup_grupo_steps, "seed": fusao.seed_shuffle,
        "batch_efetivo": batch_efetivo,
        "spans": [{k: s[k] for k in ("alias", "instancias_inicio", "instancias_fim",
                                     "unfreeze_corte", "pace_epochs")} for s in spans_meta],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def salvar_fusao_spans(output_dir: str, spans_meta: List[Dict], hash_cfg: str) -> str:
    """Serializa spans_meta + hash em <saida>/fusao_spans.json."""
    caminho = os.path.join(output_dir, ARQ_FUSAO_SPANS)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump({"hash": hash_cfg, "spans": spans_meta}, f, ensure_ascii=False, indent=2)
    return caminho


def carregar_fusao_spans(pasta: str) -> Optional[Dict[str, Any]]:
    """Lê fusao_spans.json de uma pasta; None se ausente/corrompido."""
    caminho = os.path.join(pasta, ARQ_FUSAO_SPANS)
    if not os.path.isfile(caminho):
        return None
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)
        if isinstance(dados, dict) and isinstance(dados.get("spans"), list):
            return dados
    except Exception as e:
        logger.warning(f"⚠️  Erro ao ler {caminho}: {e}")
    return None


def verificar_hash_resume(output_dir: str, hash_atual: str, tem_checkpoint: bool) -> None:
    """No resume, recomputa e compara o hash da fusão; divergência ⇒ erro.

    Sem checkpoint (run novo), o arquivo é simplesmente sobrescrito depois.
    """
    if not tem_checkpoint:
        return
    dados = carregar_fusao_spans(output_dir)
    if dados is None:
        logger.warning(
            "⚠️  Resume de fusão sem fusao_spans.json anterior — impossível conferir "
            "a consistência da configuração; prosseguindo com a atual."
        )
        return
    hash_salvo = dados.get("hash", "")
    if hash_salvo and hash_salvo != hash_atual:
        raise ValueError(
            "Configuração de fusão mudou desde o checkpoint; scheduler/grupos seriam "
            "inconsistentes. Apague os checkpoints ou restaure o YAML.\n"
            f"   hash salvo:  {hash_salvo}\n"
            f"   hash atual:  {hash_atual}"
        )
