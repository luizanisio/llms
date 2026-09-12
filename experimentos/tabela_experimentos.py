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

Saída:
  - RESUMO_EXPERIMENTOS.md na raiz de /experimentos/
  - Warnings no console para divergências entre experimentos
'''

import os
import sys
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

# ===========================================================================
# MAPA ESTÁTICO SEMÂNTICO
# ===========================================================================
# Campos que NÃO podem ser extraídos dos YAMLs: grupo de pesquisa, direção
# do currículo (CL), modo do currículo, tipo de Progressive Training (PT),
# e descrição curta.
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
# PARSER DE YAML
# ===========================================================================

def carregar_yaml(caminho: Path) -> dict:
    """Carrega um YAML de treinamento e retorna o dict."""
    with open(caminho, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extrair_parametros(cfg: dict) -> dict:
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

    # --- Learning Rate ---
    # Coletar LRs distintos: global + por etapa + fusao
    lrs = set()
    lr_global = treinamento.get('learning_rate', None)
    if lr_global is not None:
        lrs.add(float(lr_global))

    lr_fusao = fusao.get('learning_rate', None)
    if lr_fusao is not None:
        lrs.add(float(lr_fusao))

    for etapa in divisao:
        lr_etapa = etapa.get('learning_rate', None)
        if lr_etapa is not None:
            lrs.add(float(lr_etapa))

    # Se há fusão, o LR efetivo é o da fusão (sobrescreve o global)
    if eh_fusao and lr_fusao is not None:
        lrs_efetivos = {float(lr_fusao)}
        # Etapas fundidas não têm LR próprio (proibido pelo validador)
    else:
        # Para protocolos segmentados, o que importa são os LRs de etapa
        # (quando existem) e o global como fallback
        lrs_efetivos = set()
        for etapa in divisao:
            lr_etapa = etapa.get('learning_rate', None)
            if lr_etapa is not None:
                lrs_efetivos.add(float(lr_etapa))
            elif lr_global is not None:
                lrs_efetivos.add(float(lr_global))

    lr_str = '/'.join(formatar_lr(lr) for lr in sorted(lrs_efetivos))
    if not lr_str:
        lr_str = formatar_lr(lr_global) if lr_global else '?'

    # --- LoRA r ---
    lora_r = lora_cfg.get('r', 0)
    lora_r_str = str(lora_r) if lora_r > 0 else 'N/A'

    # --- max_grad_norm ---
    grad_norm = treinamento.get('max_grad_norm', 1)

    # --- warmup_steps ---
    warmup = treinamento.get('warmup_steps', 5)

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

    return {
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

def varrer_experimentos() -> dict:
    """
    Varre todos os experimentos e retorna um dict:
      {protocolo_id: {exp_nome: parametros_extraidos, ...}, ...}
    """
    resultados = {}
    for proto_id in IDS_PROTOCOLOS:
        resultados[proto_id] = {}
        for exp_nome, exp_dir in EXPERIMENTOS.items():
            yaml_path = exp_dir / f'04_treinar_{proto_id}.yaml'
            if yaml_path.exists():
                try:
                    cfg = carregar_yaml(yaml_path)
                    params = extrair_parametros(cfg)
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


def verificar_consistencia(resultados: dict) -> list:
    """
    Verifica consistência entre experimentos para cada protocolo.
    Retorna lista de strings de warning.
    """
    warnings = []

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
                if val_ref != val_exp:
                    warnings.append(
                        f'⚠️  **{proto_id}**: `{campo}` diverge entre '
                        f'{ref_nome} ({val_ref}) e {exp_nome} ({val_exp})'
                    )

    return warnings


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


def gerar_markdown(resultados: dict, warnings_consistencia: list, warnings_constantes: list) -> str:
    """Gera o conteúdo do RESUMO_EXPERIMENTOS.md."""
    linhas = []
    linhas.append('# Resumo dos Experimentos — Configurações dos Protocolos')
    linhas.append('')
    linhas.append('> Gerado automaticamente por `tabela_experimentos.py`. '
                  'Não editar manualmente.')
    linhas.append('')

    # --- Seção 1: Parâmetros constantes ---
    linhas.append('## 1. Parâmetros Constantes (todos os protocolos)')
    linhas.append('')
    linhas.append('| Parâmetro | Valor |')
    linhas.append('|:--|:--|')
    linhas.append('| Batch efetivo | 16 |')
    linhas.append('| Seed | 3407 |')
    linhas.append('| Otimizador | adamw_8bit |')
    linhas.append('| Weight decay | 0.01 |')
    linhas.append('| LR scheduler | cosine |')
    linhas.append('| LoRA dropout | 0.05 |')
    linhas.append('| LoRA α | 2×r |')
    linhas.append('| LoRA target_modules | q, k, v, o, gate, up, down (7 projeções) |')
    linhas.append('| train_on_responses_only | true |')
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
    linhas.append('- **P/S/Su**: disponibilidade em Pubmed/SemClinBR/Summa')
    linhas.append('')

    # Cabeçalho da tabela
    colunas = ['ID', 'Grupo', 'Descrição', '#Et.', 'Sequência', '#FF', '#L',
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
                f'? | ? | ? | ? | {meta.get("cl","?")} | {meta.get("modo_cl","?")} | '
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

    # --- Seção 4: Notas de consistência ---
    linhas.append('## 4. Notas de Consistência')
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

    # --- Seção 5: Notas de comparabilidade ---
    linhas.append('## 5. Notas de Comparabilidade')
    linhas.append('')
    linhas.append('- **Precisão:** Protocolos com precisão "misto" têm etapas FF (sempre 16b) '
                  'e etapas LoRA (4b NF4). Cruzamentos entre grupos de precisão carregam '
                  '{efeito estudado + quantização} como diferença conjunta.')
    linhas.append('- **Orçamento:** d16–d25 são calibrados em 4N instâncias (mesmo total de '
                  'B com 4 épocas). Protocolos anteriores podem não seguir essa paridade.')
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
    print('🔎 Verificando consistência entre experimentos...')
    warnings_consistencia = verificar_consistencia(resultados)
    warnings_constantes = verificar_constantes(resultados)

    if warnings_consistencia:
        print(f'\n⚠️  {len(warnings_consistencia)} divergência(s) entre experimentos:')
        for w in warnings_consistencia:
            print(f'   {w}')
    else:
        print('   ✅ Consistente entre experimentos.')

    if warnings_constantes:
        print(f'\n⚠️  {len(warnings_constantes)} constante(s) inesperada(s):')
        for w in warnings_constantes[:10]:  # Mostrar no máximo 10
            print(f'   {w}')
        if len(warnings_constantes) > 10:
            print(f'   ... e mais {len(warnings_constantes) - 10}')
    else:
        print('   ✅ Constantes OK.')
    print()

    # 4. Gerar Markdown
    md_content = gerar_markdown(resultados, warnings_consistencia, warnings_constantes)
    md_path = BASE_DIR / 'RESUMO_EXPERIMENTOS.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f'✅ Arquivo gerado: {md_path}')
    print(f'   Tamanho: {len(md_content)} bytes')


if __name__ == '__main__':
    main()
