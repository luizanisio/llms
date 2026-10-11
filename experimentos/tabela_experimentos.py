'''
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Varre todos os experimentos principais (pubmed-experimento, semclinbr-experimento, summa-experimento)
e gera um RESUMO_EXPERIMENTOS.md com as principais configurações dos protocolos.

O objetivo é verificar se todos os experimentos estão usando parâmetros iguais entre si
e quais parâmetros diferenciam um protocolo do outro.

Fonte de dados híbrida:
  - Automática: leitura dos 04_treinar_*.yaml para extrair parâmetros numéricos reais
  - Estática:   mapa semântico (PROTOCOLOS) com metadados que não estão nos YAMLs
                (grupo de pesquisa, direção CL, modo CL, tipo de PT, etc.)
  - Conferência: quando existe training_metrics.jsonl, compara o LR logado com o
                 LR efetivo esperado de cada etapa.

Saída:
  - RESUMO_EXPERIMENTOS.md na raiz de /experimentos/
  - Warnings no console para divergências entre experimentos
'''

import os
import sys
import json
import math
import yaml
from pathlib import Path
from collections import OrderedDict

# ===========================================================================
# CONFIGURAÇÃO
# ===========================================================================

# Diretório base (onde este script vive)
BASE_DIR = Path(__file__).resolve().parent

# Experimentos a varrer (nome, diretório relativo)
EXPERIMENTOS = OrderedDict([
    ('Pubmed',    BASE_DIR / 'pubmed-experimento'),
    ('SemClinBR', BASE_DIR / 'semclinbr-experimento'),
    ('Summa',     BASE_DIR / 'summa-experimento'),
])

# IDs dos protocolos na ordem desejada para a tabela
IDS_PROTOCOLOS = [
    'b', 'b16', 'b16r8', 'c',
    'd1', 'd2', 'd3', 'd4',
    'd5', 'd6', 'd7', 'd8',
    'd9', 'd10',
    'd11', 'd12',
    'd13', 'd14', 'd15',
    'd16', 'd17', 'd18',
    'd19', 'd20',
    'd21', 'd22', 'd23',
    'd24', 'd25',
    'd1a', 'd1b',
]

# Espelha o comportamento ATUAL de treinar_unsloth_util.py: fora da FUSÃO, o
# learning_rate/pace_epochs da 1ª etapa treinável sobrescreve o global e vira o
# fallback das etapas que não definem o seu. Quando o bug for corrigido no
# treinador, mude para False — a conferência com o jsonl passará a acusar os
# treinos antigos feitos com o fallback da 1ª etapa.
FALLBACK_DA_1A_ETAPA = True

# Warmup escalado de propósito pelo tamanho do corpus: ocupa a mesma fração dos
# steps que o valor de referência ocupa nos demais experimentos (protocolo
# idêntico = mesma forma da curva de LR ao longo do treino).
# {experimento: ({protocolos}, {warmup no experimento: warmup de referência})}.
# Essas divergências saem como nota, não como aviso.
# Ver semclinbr-experimento/README_SEMCLIMBR_EXPERIMENTO.md §2.4.
WARMUP_ESCALADO = {
    'SemClinBR': ({f'd{i}' for i in range(16, 26)}, {'5': '100', '5+5g': '100+100g'}),
}


def _warmup_referencia(exp_nome: str, proto_id: str, warmup: str) -> str:
    """Warmup na escala de referência (desfaz WARMUP_ESCALADO)."""
    protos, mapa = WARMUP_ESCALADO.get(exp_nome, (set(), {}))
    return mapa.get(warmup, warmup) if proto_id in protos else warmup

# ===========================================================================
# MAPA ESTÁTICO SEMÂNTICO
# ===========================================================================
# Campos que NÃO podem ser extraídos dos YAMLs: direção do currículo (CL),
# modo do currículo, tipo de Progressive Training (PT) e descrição curta.
#
# A coluna 'grupo' NÃO é mantida aqui: é derivada em tempo de execução dos
# recortes de `06_compara_todos.yaml` (ver derivar_grupos()). Mantê-la à mão
# fazia a tabela divergir silenciosamente da definição real dos contrastes.
#
# Legenda CL: ↑=ascendente  ↓=anti-CL  ∼=aleatório  —=sem CL
# Legenda Modo CL: disj.=disjunto  acum.=acumulado  gran.=granular
# Legenda PT: —=sem  troca=merge LoRA↔FF  unfreeze=descongelamento  gating=gating de LR

PROTOCOLOS = {
    # === Baselines ===
    'b':     {'grupo': 'Q1',              'cl': '—', 'modo_cl': 'N/A',   'pt': '—',       'descricao': 'Baseline LoRA 4b direto'},
    'b16':   {'grupo': 'Q2a',             'cl': '—', 'modo_cl': 'N/A',   'pt': '—',       'descricao': 'Controle LoRA 16b r=16'},
    'b16r8': {'grupo': 'Q2a',             'cl': '—', 'modo_cl': 'N/A',   'pt': '—',       'descricao': 'Controle LoRA 16b r=8'},
    'c':     {'grupo': 'Q1',              'cl': '—', 'modo_cl': 'N/A',   'pt': '—',       'descricao': 'Baseline Full FT 16b direto'},

    # === CL + escalonamento por troca de regime (fronteiras reais) ===
    'd1':    {'grupo': 'Q2a,Q3a',         'cl': '↑', 'modo_cl': 'disj.', 'pt': 'troca',   'descricao': 'CL disj. FF→LoRA (FF precoce)'},
    'd2':    {'grupo': 'Q2a,Q3a',         'cl': '↑', 'modo_cl': 'disj.', 'pt': 'troca',   'descricao': 'CL disj. LoRA→FF (FF tardio)'},
    'd3':    {'grupo': 'Q2c,Q3a',         'cl': '↑', 'modo_cl': 'acum.', 'pt': 'troca',   'descricao': 'CL acum. FF→LoRA (replay)'},
    'd4':    {'grupo': 'Q2c,Q3a',         'cl': '↑', 'modo_cl': 'acum.', 'pt': 'troca',   'descricao': 'CL acum. LoRA→FF'},

    # === Ablações — isolam um eixo de cada vez ===
    'd5':    {'grupo': 'Q3b,Q4a',         'cl': '—', 'modo_cl': 'N/A',   'pt': 'troca',   'descricao': 'FF→LoRA sem CL'},
    'd6':    {'grupo': 'Q3b,Q4a',         'cl': '—', 'modo_cl': 'N/A',   'pt': 'troca',   'descricao': 'LoRA→FF sem CL'},
    'd7':    {'grupo': 'Q2b,Q4a,Q5',      'cl': '↑', 'modo_cl': 'disj.', 'pt': '—',       'descricao': 'CL puro disj. LoRA 4b'},
    'd8':    {'grupo': 'Q2b',             'cl': '↑', 'modo_cl': 'acum.', 'pt': '—',       'descricao': 'CL puro acum. LoRA 4b'},

    # === Direção do currículo (controle negativo) ===
    'd9':    {'grupo': 'Q5',              'cl': '↓', 'modo_cl': 'disj.', 'pt': '—',       'descricao': 'Anti-CL disj. LoRA 4b'},
    'd10':   {'grupo': 'Q5',              'cl': '↓', 'modo_cl': 'acum.', 'pt': '—',       'descricao': 'Anti-CL acum. LoRA 4b'},

    # === Granularidade (10 etapas) ===
    'd11':   {'grupo': 'Q2c',             'cl': '↑', 'modo_cl': 'gran.', 'pt': 'troca',   'descricao': 'CL gran. FF→LoRA (10 etapas)'},
    'd12':   {'grupo': 'Q2c,Q6b',         'cl': '↑', 'modo_cl': 'gran.', 'pt': 'troca',   'descricao': 'CL gran. LoRA→FF (10 etapas)'},

    # === Ablações FF-only ===
    'd13':   {'grupo': 'Q2b,Q4a',         'cl': '↑', 'modo_cl': 'disj.', 'pt': '—',       'descricao': 'CL puro disj. FF 16b'},
    'd14':   {'grupo': 'Q6c',             'cl': '↑', 'modo_cl': 'disj.', 'pt': 'troca',   'descricao': 'Warm-up LoRA + CL FF'},
    'd15':   {'grupo': 'Q6c',             'cl': '↑', 'modo_cl': 'disj.', 'pt': 'troca',   'descricao': 'Warm-up LoRA + estab. + CL FF'},

    # === Controles de fronteira e precisão ===
    'd16':   {'grupo': 'Q6a',             'cl': '—', 'modo_cl': 'N/A',   'pt': '—',       'descricao': 'B segmentado 16b (custo fronteira)'},
    'd17':   {'grupo': 'Q2a,Q6a',         'cl': '↑', 'modo_cl': 'disj.', 'pt': '—',       'descricao': 'CL puro disj. LoRA 16b'},
    'd18':   {'grupo': 'Q2a,Q5',          'cl': '∼', 'modo_cl': 'disj.', 'pt': '—',       'descricao': 'Blocos aleatórios LoRA 16b'},

    # === PT por descongelamento progressivo ===
    'd19':   {'grupo': 'Q4b',             'cl': '↑', 'modo_cl': 'disj.', 'pt': 'unfreeze','descricao': 'CL + unfreeze (tese central)'},
    'd20':   {'grupo': 'Q4b',             'cl': '—', 'modo_cl': 'N/A',   'pt': 'unfreeze','descricao': 'Unfreeze sem CL (controle d19)'},

    # === Protocolos fundidos — fronteiras virtuais ===
    'd21':   {'grupo': 'Q6a',             'cl': '↑', 'modo_cl': 'disj.', 'pt': '—',       'descricao': 'CL fundido LoRA 16b (espelho d17)'},
    'd22':   {'grupo': 'Q6a,Q4b',         'cl': '↑', 'modo_cl': 'disj.', 'pt': 'gating',  'descricao': 'CL + gating fundido LoRA 16b'},
    'd23':   {'grupo': 'Q6a',             'cl': '—', 'modo_cl': 'N/A',   'pt': 'gating',  'descricao': 'Gating fundido sem CL (controle d22)'},
    'd24':   {'grupo': 'Q6b',             'cl': '↑', 'modo_cl': 'gran.', 'pt': 'gating',  'descricao': 'CL gran. + gating fundido LoRA 4b'},
    'd25':   {'grupo': 'Q6b',             'cl': '↑', 'modo_cl': 'gran.', 'pt': 'gating',  'descricao': 'CL gran. + gating fundido FF 16b'},

    # === Réplicas ===
    'd1a':   {'grupo': 'Réplica',         'cl': '↑', 'modo_cl': 'disj.', 'pt': 'troca',   'descricao': 'Réplica 2 do d1'},
    'd1b':   {'grupo': 'Réplica',         'cl': '↑', 'modo_cl': 'disj.', 'pt': 'troca',   'descricao': 'Réplica 3 do d1'},
}


# ===========================================================================
# RECORTES E ORÇAMENTO (derivados, nunca mantidos à mão)
# ===========================================================================

def carregar_recortes() -> dict:
    """Recortes de `06_compara_todos.yaml`: {recorte: [protocolos em minúsculas]}.

    Os três experimentos declaram os mesmos recortes — divergência é erro e
    vira warning. Retorna os recortes do primeiro experimento encontrado.
    """
    mapas = {}
    for exp_nome, exp_dir in EXPERIMENTOS.items():
        caminho = exp_dir / '06_compara_todos.yaml'
        if not caminho.exists():
            continue
        cfg = carregar_yaml(caminho) or {}
        recortes = (cfg.get('estatistica') or {}).get('protocolos') or {}
        mapas[exp_nome] = {k: v for k, v in recortes.items() if isinstance(v, list)}

    if not mapas:
        return {}

    ref_nome, ref = next(iter(mapas.items()))
    for exp_nome, m in mapas.items():
        if m != ref:
            print(f'⚠️  Recortes de {exp_nome} divergem de {ref_nome} '
                  f'(06_compara_todos.yaml).', file=sys.stderr)
    return {k: [str(p).lower() for p in v] for k, v in ref.items()}


def derivar_grupos(recortes: dict) -> dict:
    """Inverte os recortes de `06_compara_todos.yaml`: protocolo -> recortes.

    Os recortes são a definição normativa dos contrastes; a coluna 'Grupo' da
    tabela precisa sair deles, senão passa a contradizer a própria análise.
    """
    grupos = {}
    for recorte, protos in recortes.items():
        curto = recorte.split('_')[0]          # Q2a_cl_controlado -> Q2a
        for proto in protos:
            grupos.setdefault(proto.lower(), []).append(curto)
    return {k: ','.join(v) for k, v in grupos.items()}


def _fracao_divisao(exp_dir, arquivo, filtro):
    """Fração do split de treino coberta por um `dataset_filtro` de etapa."""
    import csv
    caminho = exp_dir / arquivo
    if not caminho.exists():
        return None
    with open(caminho, encoding='utf-8') as f:
        linhas = [r for r in csv.DictReader(f) if r.get('alvo') == 'treino']
    if not linhas:
        return None
    if not filtro:
        return 1.0
    sel = linhas
    for campo, valor in filtro.items():
        if isinstance(valor, str) and valor[:2] in ('<=', '>=') or \
           (isinstance(valor, str) and valor[:1] in ('<', '>', '=')):
            op = valor[:2] if valor[:2] in ('<=', '>=', '==') else valor[:1]
            alvo = float(valor[len(op):])
            cmp = {'<=': lambda a: a <= alvo, '>=': lambda a: a >= alvo,
                   '==': lambda a: a == alvo, '<': lambda a: a < alvo,
                   '>': lambda a: a > alvo}[op]
            sel = [r for r in sel if r.get(campo) not in (None, '') and cmp(float(r[campo]))]
        else:
            sel = [r for r in sel if r.get(campo) == str(valor)]
    return len(sel) / len(linhas)


def _fallback_efetivo(cfg: dict):
    """Valores que o treinador usa quando a etapa não define o seu.

    Ver FALLBACK_DA_1A_ETAPA. Retorna (eh_fusao, etapas_treinaveis,
    lr_fallback, epochs_fallback).
    """
    curriculum = cfg.get('curriculum', {}) or {}
    treinamento = cfg.get('treinamento', {}) or {}
    fusao = curriculum.get('fusao') or {}
    divisao = curriculum.get('divisao') or []
    eh_fusao = bool(fusao.get('ativo', False))
    etapas = [e for e in divisao if isinstance(e, dict) and (eh_fusao or e.get('tipo'))]
    lr = treinamento.get('learning_rate')
    lr = float(lr) if lr is not None else None
    ep = treinamento.get('epochs') or treinamento.get('num_train_epochs', 1)
    if FALLBACK_DA_1A_ETAPA and not eh_fusao and etapas:
        primeira = etapas[0]
        if float(primeira.get('learning_rate') or 0) > 0:
            lr = float(primeira['learning_rate'])
        if float(primeira.get('pace_epochs') or 0) > 0:
            ep = primeira['pace_epochs']
    return eh_fusao, etapas, lr, ep


def _compactar(valores: list) -> str:
    """'2e-5' se todas as etapas iguais; senão '5e-6→2e-5→…' por etapa."""
    if not valores:
        return ''
    if len(set(valores)) == 1:
        return valores[0]
    return '→'.join(valores)


def calcular_orcamento(cfg: dict, exp_dir) -> float:
    """Orçamento de dados em dataset-equivalentes (N-eq): Σ (fração × épocas).

    É o número que torna auditável se um contraste compara protocolos sob o
    mesmo volume de dados. Sem ele, uma divergência de `pace_epochs` entre
    experimentos passa despercebida na tabela.
    """
    curriculum = cfg.get('curriculum', {}) or {}
    treinamento = cfg.get('treinamento', {}) or {}
    divisao = curriculum.get('divisao') or []
    epochs_global = treinamento.get('num_train_epochs', 1)
    if not divisao:
        return float(epochs_global)
    _, _, _, epochs_fallback = _fallback_efetivo(cfg)
    total = 0.0
    for etapa in divisao:
        frac = _fracao_divisao(exp_dir, etapa.get('arquivo', ''), etapa.get('dataset_filtro'))
        if frac is None:
            return None
        total += frac * float(etapa.get('pace_epochs') or epochs_fallback)
    return total


# ===========================================================================
# PARSER DE YAML
# ===========================================================================

def carregar_yaml(caminho: Path) -> dict:
    """Carrega um YAML de treinamento e retorna o dict."""
    with open(caminho, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extrair_parametros(cfg: dict, exp_dir=None) -> dict:
    """
    Extrai parâmetros diferenciadores de um YAML de treinamento já carregado.
    Retorna um dict com os campos normalizados.
    """
    curriculum = cfg.get('curriculum', {})
    treinamento = cfg.get('treinamento', {})
    lora_cfg = cfg.get('lora', {})
    modelo = cfg.get('modelo', {})
    fusao = curriculum.get('fusao', {})
    divisao = curriculum.get('divisao', [])
    if divisao is None:
        divisao = []

    # --- Número de etapas ---
    n_etapas = len(divisao)

    # --- Sequência de tipos e contagens ---
    tipos_por_etapa = []
    for etapa in divisao:
        tipo = etapa.get('tipo', None)
        if tipo is None:
            # Em modo fundido, o tipo vem do bloco fusao
            tipo = fusao.get('tipo', 'lora')
        tipos_por_etapa.append(tipo)

    seq_tipos = '→'.join('F' if t == 'full' else 'L' for t in tipos_por_etapa)
    n_ff = sum(1 for t in tipos_por_etapa if t == 'full')
    n_lora = sum(1 for t in tipos_por_etapa if t == 'lora')

    # --- Fronteira ---
    eh_fusao = fusao.get('ativo', False)
    if n_etapas <= 1:
        fronteira = 'N/A'
    elif eh_fusao:
        fronteira = 'virtual'
    else:
        fronteira = 'real'

    # --- Precisão ---
    nbits_global = treinamento.get('nbits', 4)
    tipos_set = set(tipos_por_etapa)
    if tipos_set == {'full'}:
        precisao = '16b'
    elif tipos_set == {'lora'}:
        precisao = f'{nbits_global}b'
    else:
        # Misto: tem etapas FF (sempre 16b) e LoRA (nbits_global)
        if nbits_global == 16:
            precisao = '16b'
        else:
            precisao = 'misto'

    # --- Learning Rate efetivo por etapa ---
    # Usa o fallback REAL do treinador (ver FALLBACK_DA_1A_ETAPA), não o
    # global do YAML — assim a tabela mostra o que de fato foi treinado.
    _, etapas_treino, lr_fallback, _ = _fallback_efetivo(cfg)
    lr_global = treinamento.get('learning_rate', None)
    lr_global = float(lr_global) if lr_global is not None else None
    avisos_lr = []
    if eh_fusao and fusao.get('learning_rate') is not None:
        # Etapas fundidas não têm LR próprio (proibido pelo validador)
        lr_etapas = [float(fusao['learning_rate'])] * len(etapas_treino)
    else:
        lr_etapas = []
        for i, etapa in enumerate(etapas_treino):
            proprio = etapa.get('learning_rate')
            if proprio is not None and float(proprio) > 0:
                lr_etapas.append(float(proprio))
            else:
                lr_etapas.append(lr_fallback)
                if lr_fallback != lr_global:
                    avisos_lr.append(
                        f"etapa {i} `{etapa.get('alias', i)}` sem `learning_rate`: "
                        f"herda {formatar_lr(lr_fallback)} da 1ª etapa "
                        f"(global = {formatar_lr(lr_global)})")
    lr_str = _compactar([formatar_lr(x) for x in lr_etapas]) or formatar_lr(lr_global)

    # --- LoRA r ---
    lora_r = lora_cfg.get('r', 0)
    lora_r_str = str(lora_r) if lora_r > 0 else 'N/A'

    # --- max_grad_norm ---
    grad_norm = f"{float(treinamento.get('max_grad_norm', 1.0)):.1f}"

    # --- warmup_steps efetivo por etapa ---
    # Fora da fusão, cada etapa reinicia o scheduler (override por etapa: -1 = global).
    # Na fusão: warmup global único + rampa por grupo ('g').
    w_global = int(treinamento.get('warmup_steps', 5))
    if eh_fusao:
        warmup_etapas = [w_global] * len(etapas_treino)
        warmup = f"{w_global}+{int(fusao.get('warmup_grupo_steps', 100))}g"
    else:
        warmup_etapas = [int(e['warmup_steps']) if int(e.get('warmup_steps', -1)) >= 0 else w_global
                         for e in etapas_treino]
        warmup = _compactar([str(w) for w in warmup_etapas]) or str(w_global)

    # --- Modelo base ---
    modelo_base = modelo.get('base_model_name', '?')

    # --- max_seq_length ---
    max_seq = treinamento.get('max_seq_length', '?')

    # --- batch_size por GPU ---
    batch_cfg = treinamento.get('batch_size', {})
    if isinstance(batch_cfg, dict):
        batch_gpu = batch_cfg.get('batch_size', '?')
        batch_efetivo = batch_cfg.get('efetivo', '?')
    else:
        batch_gpu = batch_cfg
        batch_efetivo = '?'

    # --- Parâmetros constantes (para validação) ---
    seed = treinamento.get('seed', '?')
    optim = treinamento.get('optim', '?')
    weight_decay = treinamento.get('weight_decay', '?')
    lr_scheduler = treinamento.get('lr_scheduler_type', '?')
    lora_dropout = lora_cfg.get('dropout', '?')
    lora_alpha = lora_cfg.get('alpha', '?')
    target_modules = lora_cfg.get('target_modules', [])
    train_resp_only = treinamento.get('train_on_responses_only', '?')
    n_eq = calcular_orcamento(cfg, exp_dir) if exp_dir is not None else None


    return {
        'n_eq': n_eq,
        'n_etapas': n_etapas,
        'sequencia': seq_tipos,
        'n_ff': n_ff,
        'n_lora': n_lora,
        'fronteira': fronteira,
        'precisao': precisao,
        'lr': lr_str,
        'lora_r': lora_r_str,
        'grad_norm': grad_norm,
        'warmup': warmup,
        'modelo_base': modelo_base,
        'max_seq_length': max_seq,
        'batch_gpu': batch_gpu,
        'batch_efetivo': batch_efetivo,
        # Detalhe por etapa treinável (conferência com jsonl e avisos)
        'eh_fusao': eh_fusao,
        'aliases': [str(e.get('alias', i)) for i, e in enumerate(etapas_treino)],
        'lr_etapas': lr_etapas,
        'warmup_etapas': warmup_etapas,
        'avisos_lr': avisos_lr,
        'saida': modelo.get('saida'),
        # Constantes para validação cruzada
        '_seed': seed,
        '_optim': optim,
        '_weight_decay': weight_decay,
        '_lr_scheduler': lr_scheduler,
        '_lora_dropout': lora_dropout,
        '_lora_alpha': lora_alpha,
        '_target_modules': sorted(target_modules) if isinstance(target_modules, list) else target_modules,
        '_train_resp_only': train_resp_only,
    }


def chave_ordem(pid: str):
    """Ordena b, b16, c, d1, d2, ... d25, d1a — numérico, não lexicográfico."""
    try:
        return (IDS_PROTOCOLOS.index(pid), pid)
    except ValueError:
        return (len(IDS_PROTOCOLOS), pid)


def formatar_n_eq(v) -> str:
    """N-equivalente arredondado: '4N', '11N'. '?' quando a divisão não foi lida."""
    if v is None:
        return '?'
    return f'{round(v)}N'


def formatar_lr(lr) -> str:
    """Formata um learning rate de forma compacta."""
    if lr is None:
        return '?'
    lr = float(lr)
    if lr == 0:
        return '0'
    # Formatar como notação científica compacta
    exp = 0
    val = lr
    while val < 1 and exp > -10:
        val *= 10
        exp -= 1
    # Formatos comuns
    if lr == 0.0002 or lr == 2e-4:
        return '2e-4'
    elif lr == 2e-5 or lr == 0.00002:
        return '2e-5'
    elif lr == 5e-6 or lr == 0.000005:
        return '5e-6'
    elif lr == 3e-6:
        return '3e-6'
    elif lr == 1e-6:
        return '1e-6'
    else:
        return f'{lr:.0e}'


# ===========================================================================
# VARREDURA DE EXPERIMENTOS
# ===========================================================================

def _yaml_treinado(exp_dir: Path, saida: str):
    """Cópia do YAML salva pelo treinador na pasta de saída (última versão)."""
    if not saida:
        return None
    pasta = exp_dir / saida / 'treinamento' / 'treinamento_config'
    copias = sorted(pasta.glob('*.yaml')) if pasta.is_dir() else []
    return copias[-1] if copias else None


def varrer_experimentos() -> dict:
    """
    Varre todos os experimentos e retorna um dict:
      {protocolo_id: {exp_nome: parametros_extraidos, ...}, ...}

    Quando existe treino, `params['treinado']` traz os parâmetros da cópia do
    YAML que o treinador de fato usou (pode diferir do YAML atual).
    """
    resultados = {}
    for proto_id in IDS_PROTOCOLOS:
        resultados[proto_id] = {}
        for exp_nome, exp_dir in EXPERIMENTOS.items():
            yaml_path = exp_dir / f'04_treinar_{proto_id}.yaml'
            if yaml_path.exists():
                try:
                    cfg = carregar_yaml(yaml_path)
                    params = extrair_parametros(cfg, exp_dir)
                    copia = _yaml_treinado(exp_dir, params.get('saida'))
                    params['treinado'] = extrair_parametros(carregar_yaml(copia)) if copia else None
                    params['yaml_treinado'] = copia
                    resultados[proto_id][exp_nome] = params
                except Exception as e:
                    print(f'⚠️  Erro ao processar {yaml_path}: {e}', file=sys.stderr)
                    resultados[proto_id][exp_nome] = None
    return resultados


# ===========================================================================
# VERIFICAÇÃO DE CONSISTÊNCIA
# ===========================================================================

# Parâmetros que DEVEM ser iguais entre experimentos para o mesmo protocolo
# (excluindo modelo_base, max_seq_length, batch_gpu que variam por design)
PARAMS_CONSISTENCIA = [
    'n_etapas', 'sequencia', 'n_ff', 'n_lora', 'fronteira', 'precisao',
    'lr', 'lora_r', 'grad_norm', 'warmup',
]

# Parâmetros que devem ser constantes em TODOS os protocolos de cada experimento
PARAMS_CONSTANTES_ESPERADOS = {
    'batch_efetivo': 16,
    '_seed': 3407,
    '_optim': 'adamw_8bit',
    '_weight_decay': 0.01,
    '_lr_scheduler': 'cosine',
    '_lora_dropout': 0.05,
    '_train_resp_only': True,
    '_target_modules': sorted(['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']),
}

# Rótulos da seção 1 (gerada a partir de PARAMS_CONSTANTES_ESPERADOS)
ROTULOS_CONSTANTES = {
    'batch_efetivo': 'Batch efetivo',
    '_seed': 'Seed',
    '_optim': 'Otimizador',
    '_weight_decay': 'Weight decay',
    '_lr_scheduler': 'LR scheduler',
    '_lora_dropout': 'LoRA dropout',
    '_target_modules': 'LoRA target_modules',
    '_train_resp_only': 'train_on_responses_only',
}


def _formatar_constante(valor) -> str:
    if isinstance(valor, bool):
        return str(valor).lower()
    if isinstance(valor, list):
        return f"{', '.join(v.replace('_proj', '') for v in valor)} ({len(valor)} projeções)"
    return str(valor)


def verificar_consistencia(resultados: dict):
    """
    Verifica consistência entre experimentos para cada protocolo.
    Retorna (warnings, notas): divergências de warmup previstas em
    WARMUP_ESCALADO viram notas.
    """
    warnings, notas = [], []

    for proto_id in IDS_PROTOCOLOS:
        exps = resultados.get(proto_id, {})
        # Filtrar apenas experimentos onde o YAML existe e foi parseado com sucesso
        exps_validos = {k: v for k, v in exps.items() if v is not None}
        if len(exps_validos) < 2:
            continue

        nomes_exp = list(exps_validos.keys())
        ref_nome = nomes_exp[0]
        ref_params = exps_validos[ref_nome]

        for exp_nome in nomes_exp[1:]:
            params = exps_validos[exp_nome]
            for campo in PARAMS_CONSISTENCIA:
                val_ref = ref_params.get(campo)
                val_exp = params.get(campo)
                if val_ref == val_exp:
                    continue
                if campo == 'warmup' and (_warmup_referencia(ref_nome, proto_id, val_ref)
                                          == _warmup_referencia(exp_nome, proto_id, val_exp)):
                    notas.append(f'ℹ️  **{proto_id}**: `warmup` {ref_nome} ({val_ref}) × '
                                 f'{exp_nome} ({val_exp}) — escala proporcional')
                    continue
                warnings.append(
                    f'⚠️  **{proto_id}**: `{campo}` diverge entre '
                    f'{ref_nome} ({val_ref}) e {exp_nome} ({val_exp})'
                )

    return warnings, notas


# Campos comparados entre o YAML atual e a cópia usada no treino
PARAMS_TREINO = ['n_etapas', 'sequencia', 'precisao', 'lr', 'lora_r', 'grad_norm', 'warmup']


def verificar_treino_desatualizado(resultados: dict) -> list:
    """Treinos cuja cópia do YAML difere do YAML atual (retreino pendente)."""
    avisos = []
    for proto_id in IDS_PROTOCOLOS:
        for exp_nome in EXPERIMENTOS:
            p = resultados.get(proto_id, {}).get(exp_nome)
            if not p or not p.get('treinado'):
                continue
            t = p['treinado']
            difs = [f'`{c}` {t.get(c)} → {p.get(c)}' for c in PARAMS_TREINO if t.get(c) != p.get(c)]
            if difs:
                avisos.append(f'**{exp_nome}/{proto_id}** (treinado com '
                              f'`{p["yaml_treinado"].name}`): ' + ' · '.join(difs))
    return avisos


# Parâmetros LoRA que só se aplicam quando o protocolo usa LoRA
PARAMS_CONSTANTES_LORA = {'_lora_dropout', '_target_modules'}


def verificar_constantes(resultados: dict) -> list:
    """
    Verifica se os parâmetros esperados como constantes realmente são
    constantes em todos os protocolos de cada experimento.
    Retorna lista de strings de warning.
    """
    warnings = []

    for exp_nome in EXPERIMENTOS:
        for proto_id in IDS_PROTOCOLOS:
            params = resultados.get(proto_id, {}).get(exp_nome)
            if params is None:
                continue

            # Detectar se o protocolo usa LoRA (r > 0)
            usa_lora = params.get('lora_r', 'N/A') != 'N/A'

            for campo, val_esperado in PARAMS_CONSTANTES_ESPERADOS.items():
                # Pular verificações LoRA para protocolos 100% Full FT
                if campo in PARAMS_CONSTANTES_LORA and not usa_lora:
                    continue

                val_real = params.get(campo)
                if val_real != val_esperado:
                    warnings.append(
                        f'⚠️  **{exp_nome}/{proto_id}**: constante `{campo}` '
                        f'esperada={val_esperado}, encontrada={val_real}'
                    )

    return warnings


# ===========================================================================
# CONFERÊNCIA COM O TREINO REAL (training_metrics.jsonl)
# ===========================================================================

def _fator_cosine(step: int, total: int, warmup: int) -> float:
    """Fator do LR no step (warmup linear + cosine até 0, como no HF)."""
    if warmup > 0 and step < warmup:
        return step / max(1, warmup)
    progresso = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progresso)))


def _primeiro_lr_por_etapa(caminho: Path) -> dict:
    """{etapa_index: (step_local, lr, total_steps)} do 1º log de LR de cada etapa."""
    total_por_idx, obs = {}, {}
    with open(caminho, encoding='utf-8') as f:
        for linha in f:
            if '"train_begin"' in linha:
                r = json.loads(linha)
                total_por_idx[r.get('etapa_index', 0)] = r.get('total_steps')
            elif '"learning_rate"' in linha:
                r = json.loads(linha)
                idx = r.get('etapa_index', 0)
                if idx in obs or r.get('learning_rate') is None or (r.get('step') or 0) < 2:
                    continue
                total = total_por_idx.get(idx)
                if total:
                    obs[idx] = (int(r['step']), float(r['learning_rate']), int(total))
    return obs


def verificar_lr_real(resultados: dict, tolerancia: float = 0.05, limite_warmup: float = 0.25):
    """Compara o pico de LR inferido do jsonl com o LR efetivo esperado.

    O esperado vem da cópia do YAML usada no treino (`params['treinado']`),
    não do YAML atual — divergências entre os dois ficam com
    verificar_treino_desatualizado().

    Pico observado = LR logado / fator do scheduler. O LR logado no step `s`
    é o do scheduler em `s-1` (o log sai antes do `scheduler.step()`).
    Protocolos fundidos ficam fora da conferência de LR (LR multiplicado pelo
    gate de cada grupo), mas entram na de warmup (warmup global × run inteiro).

    Também acusa etapas em que o warmup ocupa mais que `limite_warmup` dos
    steps reais (com warmup ≥ etapa, o pico nunca é atingido).
    Retorna (avisos_lr, avisos_warmup, n_treinos_verificados).
    """
    avisos, avisos_warmup, n_verif = [], [], 0
    for proto_id in IDS_PROTOCOLOS:
        for exp_nome, exp_dir in EXPERIMENTOS.items():
            p_atual = resultados.get(proto_id, {}).get(exp_nome)
            if not p_atual or not p_atual.get('saida'):
                continue
            p = p_atual.get('treinado') or p_atual
            caminho = exp_dir / p_atual['saida'] / 'treinamento' / 'training_metrics.jsonl'
            if not caminho.exists():
                continue
            try:
                obs = _primeiro_lr_por_etapa(caminho)
            except Exception as e:
                print(f'⚠️  Erro lendo {caminho}: {e}', file=sys.stderr)
                continue
            if not obs:
                continue
            if p['eh_fusao']:
                # Um único train(): warmup global sobre o run inteiro
                total, w = obs[min(obs)][2], p['warmup_etapas'][0] if p['warmup_etapas'] else 0
                if w > limite_warmup * total:
                    avisos_warmup.append(f'**{exp_nome}/{proto_id}** fusão: warmup global '
                                         f'{w} de {total} steps ({100 * w / total:.0f}%)')
                continue
            n_verif += 1
            for idx, (step, lr, total) in sorted(obs.items()):
                if idx >= len(p['lr_etapas']) or not p['lr_etapas'][idx]:
                    continue
                pico = p['lr_etapas'][idx]
                w = p['warmup_etapas'][idx]
                if w > limite_warmup * total:
                    alcancado = min(1.0, (total - 1) / w) if w else 1.0
                    avisos_warmup.append(
                        f'**{exp_nome}/{proto_id}** etapa {idx} `{p["aliases"][idx]}`: '
                        f'warmup {w} de {total} steps ({100 * w / total:.0f}%)'
                        + (f' — pico nunca atingido (máx ≈ {100 * alcancado:.0f}% do LR)'
                           if alcancado < 1 else ''))
                fator = _fator_cosine(step - 1, total, w)
                if fator <= 0:
                    continue
                pico_obs = lr / fator
                if abs(pico_obs / pico - 1) > tolerancia:
                    avisos.append(
                        f'⚠️  **{exp_nome}/{proto_id}** etapa {idx} `{p["aliases"][idx]}`: '
                        f'LR esperado {formatar_lr(pico)}, treinado ≈ {pico_obs:.1e} '
                        f'(step {step}/{total})')
    return avisos, avisos_warmup, n_verif


# ===========================================================================
# FAMÍLIAS E CONTRASTES
# ===========================================================================

def _params_ref(resultados: dict, proto_id: str):
    """Parâmetros do primeiro experimento disponível para o protocolo."""
    for exp_nome in EXPERIMENTOS:
        p = resultados.get(proto_id, {}).get(exp_nome)
        if p is not None:
            return p
    return None


def agrupar_familias(resultados: dict) -> OrderedDict:
    """{(warmup, max_grad_norm, precisão): [protocolos]} na ordem da tabela."""
    familias = OrderedDict()
    for proto_id in IDS_PROTOCOLOS:
        p = _params_ref(resultados, proto_id)
        if p is None:
            continue
        familias.setdefault((p['warmup'], p['grad_norm'], p['precisao']), []).append(proto_id)
    return familias


# Fatores que não deveriam variar dentro de um contraste (a menos que sejam o eixo estudado)
FATORES_CONFUSAO = [('warmup', 'warmup'), ('grad_norm', 'max_grad_norm'),
                    ('precisao', 'precisão'), ('n_eq', 'N-eq')]


def contrastes_confundidos(resultados: dict, recortes: dict) -> list:
    """Uma linha por recorte cujos protocolos diferem em algum FATORES_CONFUSAO."""
    saida = []
    for recorte, protos in recortes.items():
        presentes = [pid for pid in protos if _params_ref(resultados, pid)]
        partes = []
        for campo, rotulo in FATORES_CONFUSAO:
            grupos = OrderedDict()
            for pid in presentes:
                v = _params_ref(resultados, pid).get(campo)
                v = formatar_n_eq(v) if campo == 'n_eq' else str(v)
                grupos.setdefault(v, []).append(pid)
            if len(grupos) > 1:
                partes.append(f'{rotulo}: ' + ' × '.join(
                    f'{v} ({", ".join(ids)})' for v, ids in grupos.items()))
        if partes:
            saida.append(f'- **{recorte}** — ' + ' · '.join(partes))
    return saida


# ===========================================================================
# GERAÇÃO DO MARKDOWN
# ===========================================================================

def obter_parametros_por_experimento(resultados: dict) -> dict:
    """
    Extrai os parâmetros que variam entre experimentos (não entre protocolos).
    Retorna {exp_nome: {param: valor}}.
    """
    info = {}
    for exp_nome in EXPERIMENTOS:
        modelos = set()
        seqs = set()
        batches = set()
        for proto_id in IDS_PROTOCOLOS:
            params = resultados.get(proto_id, {}).get(exp_nome)
            if params is None:
                continue
            modelos.add(params['modelo_base'])
            seqs.add(str(params['max_seq_length']))
            batches.add(str(params['batch_gpu']))

        info[exp_nome] = {
            'modelo_base': ' / '.join(sorted(modelos)),
            'max_seq_length': ' / '.join(sorted(seqs, key=lambda x: int(x) if x.isdigit() else 0)),
            'batch_gpu': ' / '.join(sorted(batches)),
        }
    return info


def _nota_warmup_escalado() -> str:
    """Linha explicando WARMUP_ESCALADO nas seções 4 e 5 (vazia se não houver)."""
    partes = [f'{exp}: {", ".join(sorted(protos, key=chave_ordem))} usam '
              + ', '.join(f'{v} (≡ {ref})' for v, ref in mapa.items())
              for exp, (protos, mapa) in WARMUP_ESCALADO.items()]
    if not partes:
        return ''
    return ('> Warmup na escala de referência (Pubmed/Summa). Escala proporcional ao nº de '
            'steps — ' + ' · '.join(partes) + '. Dentro desses experimentos, o confundimento '
            'de warmup entre famílias não se aplica.')


def gerar_markdown(resultados: dict, warnings_consistencia: list, warnings_constantes: list,
                   recortes: dict = None, avisos_lr: list = None,
                   avisos_real: list = None, n_verif: int = 0,
                   avisos_warmup: list = None, notas_consistencia: list = None,
                   avisos_desatualizados: list = None) -> str:
    """Gera o conteúdo do RESUMO_EXPERIMENTOS.md."""
    recortes = recortes or {}
    avisos_lr = avisos_lr or []
    avisos_real = avisos_real or []
    avisos_warmup = avisos_warmup or []
    notas_consistencia = notas_consistencia or []
    avisos_desatualizados = avisos_desatualizados or []
    nota_warmup = _nota_warmup_escalado()
    linhas = []
    linhas.append('# Resumo dos Experimentos — Configurações dos Protocolos')
    linhas.append('')
    linhas.append('> Gerado automaticamente por `tabela_experimentos.py`. '
                  'Não editar manualmente.')
    linhas.append('')

    # --- Seção 1: Parâmetros constantes (validados em verificar_constantes) ---
    linhas.append('## 1. Parâmetros Constantes (todos os protocolos)')
    linhas.append('')
    linhas.append('| Parâmetro | Valor |')
    linhas.append('|:--|:--|')
    for campo, valor in PARAMS_CONSTANTES_ESPERADOS.items():
        linhas.append(f'| {ROTULOS_CONSTANTES.get(campo, campo)} | {_formatar_constante(valor)} |')
    linhas.append('| LoRA α | 2×r |')
    linhas.append('')

    # --- Seção 2: Parâmetros por experimento ---
    linhas.append('## 2. Parâmetros por Experimento')
    linhas.append('')
    info_exp = obter_parametros_por_experimento(resultados)
    exp_nomes = list(EXPERIMENTOS.keys())
    header = '| Parâmetro | ' + ' | '.join(exp_nomes) + ' |'
    sep = '|:--|' + '|'.join(':--' for _ in exp_nomes) + '|'
    linhas.append(header)
    linhas.append(sep)
    for param, label in [('modelo_base', 'Modelo base'), ('max_seq_length', 'max_seq_length'), ('batch_gpu', 'batch_size/GPU')]:
        vals = ' | '.join(info_exp[e].get(param, '?') for e in exp_nomes)
        linhas.append(f'| {label} | {vals} |')
    linhas.append('')

    # --- Seção 3: Tabela principal ---
    linhas.append('## 3. Tabela de Protocolos')
    linhas.append('')
    linhas.append('**Legenda:**')
    linhas.append('- **CL**: ↑ ascendente · ↓ anti-CL · ∼ aleatório · — sem CL')
    linhas.append('- **Modo CL**: disj.=disjunto · acum.=acumulado · gran.=granular')
    linhas.append('- **PT**: —=sem · troca=merge LoRA↔FF · unfreeze=descongelamento · gating=gating de LR')
    linhas.append('- **Fronteira**: real=reset otimizador · virtual=um único train() · N/A=etapa única')
    linhas.append('- **Precisão**: 4b=NF4 QLoRA · 16b=bf16 · misto=etapas FF 16b + LoRA 4b')
    linhas.append('- **N-eq**: orçamento de dados em dataset-equivalentes (Σ fração × épocas)')
    linhas.append('- **LR / warmup**: valor efetivo de cada etapa treinável (com o fallback real do '
                  'treinador); um valor só quando todas as etapas são iguais, senão sequência por etapa · '
                  '`Ng` = rampa de N steps por grupo (fusão)')
    linhas.append('- **grad_norm**: `max_grad_norm` (clipping)')
    linhas.append('- **Grupo**: recortes de contraste de `06_compara_todos.yaml` · '
                  '— = fora de todo recorte (só `Panorama_Geral`)')
    linhas.append('- **P/S/Su**: disponibilidade em Pubmed/SemClinBR/Summa')
    linhas.append('')

    # Cabeçalho da tabela
    colunas = ['ID', 'Grupo', 'Descrição', '#Et.', 'N-eq', 'Sequência', '#FF', '#L',
               'CL', 'Modo CL', 'PT', 'Fronteira', 'Precisão', 'LR',
               'LoRA r', 'grad_norm', 'warmup', 'P', 'S', 'Su']
    linhas.append('| ' + ' | '.join(colunas) + ' |')
    linhas.append('|' + '|'.join(':--' for _ in colunas) + '|')

    # Linhas da tabela
    for proto_id in IDS_PROTOCOLOS:
        meta = PROTOCOLOS.get(proto_id, {})
        exps = resultados.get(proto_id, {})

        # Pegar parâmetros do primeiro experimento disponível
        params = None
        for exp_nome in EXPERIMENTOS:
            p = exps.get(exp_nome)
            if p is not None:
                params = p
                break

        if params is None:
            # Protocolo sem YAML em nenhum experimento — preencher com '?'
            disponibilidade = ['✗'] * len(EXPERIMENTOS)
            linhas.append(
                f'| {proto_id} | {meta.get("grupo","?")} | {meta.get("descricao","?")} | '
                f'? | ? | ? | ? | ? | {meta.get("cl","?")} | {meta.get("modo_cl","?")} | '
                f'{meta.get("pt","?")} | ? | ? | ? | ? | ? | ? | '
                + ' | '.join(disponibilidade) + ' |'
            )
            continue

        # Disponibilidade por experimento
        disp = []
        for exp_nome in EXPERIMENTOS:
            disp.append('✓' if exps.get(exp_nome) is not None else '✗')

        linha = (
            f'| {proto_id} '
            f'| {meta.get("grupo", "?")} '
            f'| {meta.get("descricao", "?")} '
            f'| {params["n_etapas"]} '
            f'| {formatar_n_eq(params.get("n_eq"))} '
            f'| {params["sequencia"]} '
            f'| {params["n_ff"]} '
            f'| {params["n_lora"]} '
            f'| {meta.get("cl", "?")} '
            f'| {meta.get("modo_cl", "?")} '
            f'| {meta.get("pt", "?")} '
            f'| {params["fronteira"]} '
            f'| {params["precisao"]} '
            f'| {params["lr"]} '
            f'| {params["lora_r"]} '
            f'| {params["grad_norm"]} '
            f'| {params["warmup"]} '
            f'| {" | ".join(disp)} |'
        )
        linhas.append(linha)

    linhas.append('')

    # --- Seção 4: Famílias de configuração ---
    linhas.append('## 4. Famílias de Configuração')
    linhas.append('')
    linhas.append('Protocolos agrupados por warmup, clipping e precisão. Dentro de uma família '
                  'esses fatores são iguais; entre famílias, a diferença vem junto com o efeito estudado.')
    linhas.append('')
    linhas.append('| warmup | max_grad_norm | Precisão | Protocolos |')
    linhas.append('|:--|:--|:--|:--|')
    for (w, g, prec), ids in agrupar_familias(resultados).items():
        linhas.append(f'| {w} | {g} | {prec} | {", ".join(ids)} |')
    linhas.append('')
    if nota_warmup:
        linhas.extend([nota_warmup, ''])

    # --- Seção 5: Contrastes com confusão ---
    linhas.append('## 5. Contrastes com Confusão')
    linhas.append('')
    confusos = contrastes_confundidos(resultados, recortes)
    if not recortes:
        linhas.append('_Recortes de `06_compara_todos.yaml` não encontrados._')
    elif confusos:
        linhas.append('Recortes de `06_compara_todos.yaml` cujos protocolos diferem em fatores '
                      'além do eixo estudado (se o fator listado **é** o eixo do recorte, ignore):')
        linhas.append('')
        linhas.extend(confusos)
        if nota_warmup:
            linhas.extend(['', nota_warmup])
    else:
        linhas.append('✅ Nenhum recorte mistura warmup, clipping, precisão ou orçamento.')
    linhas.append('')

    # --- Seção 6: Notas de consistência ---
    linhas.append('## 6. Notas de Consistência')
    linhas.append('')

    linhas.append('### Treino × YAML atual (retreino pendente)')
    linhas.append('')
    if avisos_desatualizados:
        linhas.append(f'{len(avisos_desatualizados)} treino(s) feitos com uma versão do YAML '
                      'diferente da atual (cópia em `treinamento/treinamento_config/`). '
                      'As conferências abaixo usam o YAML do treino:')
        linhas.append('')
        linhas.extend(f'- {a}' for a in avisos_desatualizados)
    else:
        linhas.append('✅ Todos os treinos usaram a versão atual do YAML.')
    linhas.append('')

    linhas.append('### LR herdado da 1ª etapa')
    linhas.append('')
    if avisos_lr:
        linhas.append('Etapas sem `learning_rate` próprio recebem o LR da 1ª etapa treinável '
                      '(comportamento atual do treinador), **não** o global:')
        linhas.append('')
        linhas.extend(f'- {a}' for a in avisos_lr)
    else:
        linhas.append('✅ Nenhuma etapa depende do fallback de LR da 1ª etapa.')
    linhas.append('')

    linhas.append('### LR do YAML × treino real (`training_metrics.jsonl`)')
    linhas.append('')
    if n_verif == 0:
        linhas.append('_Nenhum `training_metrics.jsonl` encontrado para conferência._')
    elif avisos_real:
        linhas.append(f'{len(avisos_real)} etapa(s) com LR treinado diferente do esperado '
                      f'({n_verif} treinos conferidos; fusão fora da conferência):')
        linhas.append('')
        linhas.extend(f'- {a}' for a in avisos_real)
    else:
        linhas.append(f'✅ LR treinado confere com o esperado em {n_verif} treinos '
                      '(fusão fora da conferência).')
    linhas.append('')

    if avisos_warmup:
        linhas.append('### Warmup × tamanho real da etapa')
        linhas.append('')
        linhas.append(f'{len(avisos_warmup)} etapa(s) em que o warmup do treino ocupa mais de 25% '
                      'dos steps reais (segundo o `train_begin` do jsonl):')
        linhas.append('')
        linhas.extend(f'- {a}' for a in avisos_warmup)
        linhas.append('')

    if warnings_consistencia:
        linhas.append('### Divergências entre Experimentos')
        linhas.append('')
        linhas.append('Os seguintes parâmetros diferem entre experimentos para o mesmo protocolo:')
        linhas.append('')
        for w in warnings_consistencia:
            linhas.append(f'- {w}')
        linhas.append('')
    else:
        linhas.append('✅ Todos os parâmetros diferenciadores são **consistentes** entre experimentos.')
        linhas.append('')
    if notas_consistencia:
        linhas.append('Divergências previstas (`WARMUP_ESCALADO`):')
        linhas.append('')
        linhas.extend(f'- {n}' for n in notas_consistencia)
        linhas.append('')

    if warnings_constantes:
        linhas.append('### Constantes Inesperadas')
        linhas.append('')
        linhas.append('Os seguintes valores divergem do padrão esperado:')
        linhas.append('')
        for w in warnings_constantes:
            linhas.append(f'- {w}')
        linhas.append('')
    else:
        linhas.append('✅ Todos os parâmetros constantes estão nos valores esperados.')
        linhas.append('')

    # --- Seção 7: Notas de comparabilidade ---
    linhas.append('## 7. Notas de Comparabilidade')
    linhas.append('')
    linhas.append('- **Precisão:** Protocolos com precisão "misto" têm etapas FF (sempre 16b) '
                  'e etapas LoRA (4b NF4). Cruzamentos entre grupos de precisão carregam '
                  '{efeito estudado + quantização} como diferença conjunta.')
    faixas = {}
    for proto_id in IDS_PROTOCOLOS:
        for exp_nome in EXPERIMENTOS:
            p = resultados.get(proto_id, {}).get(exp_nome)
            if p and p.get('n_eq') is not None:
                faixas.setdefault(round(p['n_eq']), set()).add(proto_id)
                break
    desc = ' · '.join(f'**{n}N** ({", ".join(sorted(ids, key=chave_ordem))})'
                      for n, ids in sorted(faixas.items()))
    linhas.append(f'- **Orçamento (N-eq):** {desc}. Contrastes dentro de uma faixa '
                  'são limpos; entre faixas carregam o orçamento junto e precisam '
                  'declará-lo ao lado do resultado.')
    linhas.append('- **Fronteiras reais** resetam otimizador Adam e scheduler cosine. '
                  'Fronteiras virtuais (protocolos fundidos) mantêm trajetória contínua.')
    linhas.append('- **Gating ≠ Congelamento:** No d19/d20 blocos congelados não entram no '
                  'otimizador. No gating (d22–d25) todos os grupos estão no otimizador desde '
                  'o step 0, com LR 0 até acordarem.')
    linhas.append('')

    return '\n'.join(linhas)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print('🔍 Varrendo YAMLs de treinamento...')
    print()

    # 1. Varrer todos os experimentos
    resultados = varrer_experimentos()

    # 2. Contar protocolos encontrados
    total = 0
    for proto_id in IDS_PROTOCOLOS:
        exps = resultados.get(proto_id, {})
        n_encontrados = sum(1 for v in exps.values() if v is not None)
        if n_encontrados > 0:
            total += 1

    print(f'📊 Protocolos encontrados: {total}/{len(IDS_PROTOCOLOS)}')
    for exp_nome in EXPERIMENTOS:
        n = sum(1 for pid in IDS_PROTOCOLOS
                if resultados.get(pid, {}).get(exp_nome) is not None)
        print(f'   {exp_nome}: {n} YAMLs')
    print()

    # 3. Verificar consistência
    recortes = carregar_recortes()
    grupos = derivar_grupos(recortes)
    if grupos:
        for pid, meta in PROTOCOLOS.items():
            if pid in grupos:
                meta['grupo'] = grupos[pid]
            elif meta.get('grupo') != 'Réplica':
                # Sem recorte de contraste: entra apenas no Panorama_Geral.
                # Nunca cair de volta num rótulo manual — foi assim que a
                # coluna passou a contradizer os recortes.
                meta['grupo'] = '—'
        sem_recorte = [p for p in IDS_PROTOCOLOS if p not in grupos]
        if sem_recorte:
            print(f'   ℹ️  Fora de qualquer recorte de contraste: {", ".join(sem_recorte)}')

    print('🔎 Verificando consistência entre experimentos...')
    warnings_consistencia, notas_consistencia = verificar_consistencia(resultados)
    warnings_constantes = verificar_constantes(resultados)
    avisos_desatualizados = verificar_treino_desatualizado(resultados)

    if warnings_consistencia:
        print(f'\n⚠️  {len(warnings_consistencia)} divergência(s) entre experimentos:')
        for w in warnings_consistencia:
            print(f'   {w}')
    else:
        print('   ✅ Consistente entre experimentos.')
    if notas_consistencia:
        print(f'   ℹ️  {len(notas_consistencia)} divergência(s) de warmup prevista(s) (WARMUP_ESCALADO).')

    if avisos_desatualizados:
        print(f'\n⚠️  {len(avisos_desatualizados)} treino(s) com YAML diferente do atual (retreino pendente):')
        for a in avisos_desatualizados:
            print(f'   {a}')

    if warnings_constantes:
        print(f'\n⚠️  {len(warnings_constantes)} constante(s) inesperada(s):')
        for w in warnings_constantes[:10]:  # Mostrar no máximo 10
            print(f'   {w}')
        if len(warnings_constantes) > 10:
            print(f'   ... e mais {len(warnings_constantes) - 10}')
    else:
        print('   ✅ Constantes OK.')

    avisos_lr = [f'**{exp_nome}/{pid}**: {a}'
                 for pid in IDS_PROTOCOLOS for exp_nome in EXPERIMENTOS
                 for a in ((resultados.get(pid, {}).get(exp_nome) or {}).get('avisos_lr') or [])]
    if avisos_lr:
        print(f'\n⚠️  {len(avisos_lr)} etapa(s) herdando LR da 1ª etapa:')
        for a in avisos_lr:
            print(f'   {a}')

    print('\n🔎 Conferindo LR com training_metrics.jsonl...')
    avisos_real, avisos_warmup, n_verif = verificar_lr_real(resultados)
    if avisos_real:
        print(f'⚠️  {len(avisos_real)} etapa(s) com LR treinado ≠ esperado ({n_verif} treinos):')
        for a in avisos_real:
            print(f'   {a}')
    else:
        print(f'   ✅ {n_verif} treinos conferidos.')
    if avisos_warmup:
        print(f'⚠️  {len(avisos_warmup)} etapa(s) com warmup > 25% da etapa:')
        for a in avisos_warmup:
            print(f'   {a}')
    print()

    # 4. Gerar Markdown
    md_content = gerar_markdown(resultados, warnings_consistencia, warnings_constantes,
                                recortes, avisos_lr, avisos_real, n_verif, avisos_warmup,
                                notas_consistencia, avisos_desatualizados)
    md_path = BASE_DIR / 'RESUMO_EXPERIMENTOS.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f'✅ Arquivo gerado: {md_path}')
    print(f'   Tamanho: {len(md_content)} bytes')


if __name__ == '__main__':
    main()
