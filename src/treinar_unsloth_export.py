#!/usr/bin/env python3

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Módulo de Exportação e Inferência de modelos LLM treinados.
Contém funções de predict (HF, vLLM, Unsloth), merge/exportação
e inferência interativa com exemplos.

Funções principais:
    - executar_predict()          — Predições com HuggingFace Transformers
    - executar_predict_vllm()     — Predições com vLLM (batch rápido)
    - executar_predict_unsloth()  — Predições com Unsloth (2x mais rápido)
    - executar_modelo()           — Inferência interativa com HF
    - executar_modelo_vllm()      — Inferência interativa com vLLM
    - executar_modelo_unsloth()   — Inferência interativa com Unsloth
    - executar_merge()            — Merge LoRA + Base (HF safetensors)

Helpers internos:
    - _perguntar_subsets_predict() — Menu interativo de seleção de subsets
    - _registro_ja_exportado()     — Verifica se registro já foi exportado
    - _construir_mapa_etapas()     — Mapa de etapas do curriculum para cópias
    - _copiar_para_pastas_etapas() — Copia predições para pastas de etapa
    - _compactar_modelo_zip()      — Compacta modelo merged em .zip
    - _executar_merge_hf()         — Merge HF safetensors (transformers + PEFT)
"""

import os
import sys
import json
import shutil
import gc
import time
from typing import Optional
from datetime import datetime

import torch

from treinar_unsloth_logging import get_logger, log_separador
from util_print import exibir_menu_opcoes
from treinar_unsloth_util import (
    YamlTreinamento, FORMATO_SAIDA_JSON
)
from treinar_unsloth_actions import (
    _exibir_cabecalho_modelo,
    _verificar_modelo_treinado,
    _perguntar_confirmacao,
)
from util import UtilEnv
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers de Predict (menu de subsets + skip de itens já exportados)
# ---------------------------------------------------------------------------

def _perguntar_subsets_predict() -> Optional[list]:
    """Exibe menu interativo para selecionar quais subsets exportar.
    
    Returns:
        Lista de subsets selecionados, ou None se cancelou.
    """
    itens = [
        ('1', 'teste',     'Apenas teste (padrão)'),
        ('2', 'validacao', 'Apenas validação'),
        ('3', 'treino',    'Apenas treino'),
        ('4', 'todos',     'Todos os subsets (treino + validação + teste)'),
        ('---',),
        ('0', 'cancelar',  'Cancelar'),
    ]
    
    try:
        escolha = exibir_menu_opcoes(
            titulo='<azul>📂 SUBSETS PARA EXPORTAÇÃO:</azul>',
            itens=itens,
            prompt="❓ Escolha o subset [1]",
        )
        
        if not escolha or escolha in ('1', 'teste', ''):
            return ['teste']
        elif escolha in ('2', 'validacao', 'validação'):
            return ['validacao']
        elif escolha in ('3', 'treino'):
            return ['treino']
        elif escolha in ('4', 'todos'):
            return ['treino', 'validacao', 'teste']
        elif escolha in ('0', 'cancelar', 'sair'):
            return None
        else:
            logger.warning(f"Opção inválida: '{escolha}'. Usando padrão (teste).")
            return ['teste']
    except (KeyboardInterrupt, EOFError):
        return None


def _registro_ja_exportado(subset_dir: str, registro_id: str, min_bytes: int = 10, formato_json: bool = False) -> bool:
    """Verifica se um registro já foi exportado com sucesso (txt + json válidos).
    
    Um registro é considerado exportado se AMBOS os arquivos existem
    e têm tamanho >= min_bytes (evita considerar arquivos vazios/corrompidos).
    Quando ``formato_json=True``, também valida que o conteúdo do .txt é JSON
    válido — permitindo que registros com resposta não-JSON sejam reexportados.
    
    Args:
        subset_dir: Diretório do subset (ex: predict/teste/)
        registro_id: ID do registro (ex: 'teste_0001')
        min_bytes: Tamanho mínimo em bytes para considerar válido (padrão: 10)
        formato_json: Se True, valida se o conteúdo do .txt é JSON válido.
    
    Returns:
        True se ambos .txt e .json existem com tamanho válido (e JSON válido
        quando formato_json=True).
    """
    txt_path = os.path.join(subset_dir, f"{registro_id}.txt")
    json_path = os.path.join(subset_dir, f"{registro_id}.json")
    
    try:
        if not os.path.exists(txt_path) or not os.path.exists(json_path):
            return False
        if os.path.getsize(txt_path) < min_bytes or os.path.getsize(json_path) < min_bytes:
            return False
        if formato_json:
            with open(txt_path, 'r', encoding='utf-8') as f:
                conteudo = f.read()
            try:
                json.loads(conteudo)
            except (json.JSONDecodeError, ValueError):
                return False
        return True
    except OSError:
        return False


def _construir_mapa_etapas(divisao_dict: dict) -> dict:
    """Extrai mapa de etapas do dicionário de divisão unificado.

    Converte o dicionário retornado por ``carregar_divisao_completa`` em um
    mapa ``{(id_arquivo, alvo): [alias1, alias2, ...]}`` usado por
    ``_copiar_para_pastas_etapas``.

    Args:
        divisao_dict: Dicionário ``{id: {"alvo": str, "divisoes": list, "etapas": int}}``.

    Returns:
        Dicionário de mapeamento ou None se não há multi-etapa (etapas <= 1
        ou nenhum ID com divisões preenchidas).
    """
    if not divisao_dict:
        return None

    mapa = {}
    for id_arq, info in divisao_dict.items():
        if info["divisoes"]:
            mapa[(id_arq, info["alvo"])] = info["divisoes"]

    if not mapa:
        return None

    # Descobre aliases únicos para log (preserva ordem de aparição)
    aliases_set = []
    for als in mapa.values():
        for a in als:
            if a not in aliases_set:
                aliases_set.append(a)
    logger.info(f"<cinza>📋 Curriculum multi-etapa ({' → '.join(aliases_set)}): cópias por etapa habilitadas</cinza>")
    return mapa


def _copiar_para_pastas_etapas(
    predict_dir: str, subset: str, registro_id: str, mapa_etapas: dict
) -> None:
    """Copia arquivos de predição para pastas específicas de cada etapa.

    Se ``mapa_etapas`` for None ou o registro não pertencer a nenhuma etapa,
    não faz nada. Cria as pastas automaticamente e só copia se o destino
    ainda não existir (idempotente).

    Args:
        predict_dir: Diretório raiz de predições (ex: output/predict/).
        subset: Nome do subset (treino, validacao, teste).
        registro_id: ID do registro exportado.
        mapa_etapas: Mapa ``{(id, alvo): [alias, ...]}`` ou None.
    """
    if mapa_etapas is None:
        return

    aliases = mapa_etapas.get((registro_id, subset))
    if not aliases:
        return

    for alias in aliases:
        pasta_etapa = os.path.join(predict_dir, f"{subset} ({alias})")
        os.makedirs(pasta_etapa, exist_ok=True)
        for ext in ('.txt', '.json'):
            src = os.path.join(predict_dir, subset, f"{registro_id}{ext}")
            dst = os.path.join(pasta_etapa, f"{registro_id}{ext}")
            if os.path.isfile(src) and not os.path.isfile(dst):
                shutil.copy2(src, dst)


def _todas_predicoes_exportadas(
    predict_dir: str, subsets: list, divisao_dict: dict, mapa_etapas: dict,
    formato_json: bool = False
) -> bool:
    """Verifica se todas as predições já foram exportadas (pré-check rápido).

    Itera pelos IDs esperados em cada subset (extraídos do ``divisao_dict``)
    e verifica se os arquivos ``.txt`` e ``.json`` já existem.  Quando todos
    existem, garante também as cópias para pastas de etapa (curriculum
    multi-etapa) e retorna ``True`` para permitir saída antecipada **sem
    carregar o modelo**.

    Args:
        predict_dir: Diretório raiz de predições.
        subsets: Lista de subsets a verificar.
        divisao_dict: Dicionário de divisão unificada.
        mapa_etapas: Mapa de etapas ou None.
        formato_json: Se True, valida conteúdo JSON dos .txt ao verificar.

    Returns:
        True se todas as predições já existem (nada a gerar).
    """
    total_exportados = 0
    total_pendentes = 0

    for subset in subsets:
        subset_dir = os.path.join(predict_dir, subset)
        ids_subset = [id_arq for id_arq, info in divisao_dict.items()
                      if info["alvo"] == subset]
        for id_arq in ids_subset:
            if _registro_ja_exportado(subset_dir, id_arq, formato_json=formato_json):
                total_exportados += 1
                _copiar_para_pastas_etapas(predict_dir, subset, id_arq, mapa_etapas)
            else:
                total_pendentes += 1

    if total_pendentes == 0 and total_exportados > 0:
        logger.info(f"<verde>\n✅ Todas as {total_exportados} predições já exportadas — nada a gerar.</verde>")
        return True

    if total_exportados > 0:
        logger.info(f"<cinza>📋 Pré-check: {total_exportados} já exportadas, {total_pendentes} pendentes</cinza>")

    return False


def gerar_estatisticas_predicoes(pasta: str) -> bool:
    """Varre JSONs de predição em uma pasta e gera CSV + gráficos de análise.

    Lê todos os arquivos ``{id}.json`` da *pasta* (excluindo ``resumo.json``
    e ``resumo_geral.json``), extrai ``id``, ``input_tokens``, ``output_tokens``
    e ``time_s``, e produz:

    * ``predicoes.csv``        — tabela com id, input_tokens, output_tokens, time_s
    * ``predicoes_tokens.png`` — boxplots lado-a-lado de tokens de entrada e saída
    * ``predicoes_tempo.png``  — boxplot com tempos de geração (segundos)

    É seguro chamar várias vezes (sobrescreve artefatos anteriores).

    Args:
        pasta: Diretório contendo os ``{id}.json`` de predição.

    Returns:
        True se gerou ao menos o CSV, False se a pasta está vazia ou falhou.
    """
    if not os.path.isdir(pasta):
        return False

    # Coleta dados dos JSONs
    registros = []
    for nome in sorted(os.listdir(pasta)):
        if not nome.endswith('.json'):
            continue
        if nome in ('resumo.json', 'resumo_geral.json'):
            continue
        caminho = os.path.join(pasta, nome)
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            registros.append({
                'id': dados.get('id', nome.replace('.json', '')),
                'input_tokens': int(dados.get('input_tokens', 0)),
                'output_tokens': int(dados.get('output_tokens', 0)),
                'time_s': float(dados.get('time_s', 0)),
            })
        except Exception:
            continue

    if not registros:
        return False

    # --- CSV ---
    import csv as _csv
    csv_path = os.path.join(pasta, 'predicoes.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = _csv.DictWriter(f, fieldnames=['id', 'input_tokens', 'output_tokens', 'time_s'])
        writer.writeheader()
        writer.writerows(registros)
    logger.info(f"<cinza>   📄 CSV gerado: {csv_path} ({len(registros)} registros)</cinza>")

    # --- Gráficos (boxplots) ---
    try:
        from util_graficos import UtilGraficos

        inputs  = [r['input_tokens'] for r in registros]
        outputs = [r['output_tokens'] for r in registros]
        totais  = [r['input_tokens'] + r['output_tokens'] for r in registros]
        tempos  = [r['time_s'] for r in registros]

        # Boxplot de tokens (entrada + saída + total)
        nota_maximos = (f"max(entrada)={max(inputs):,}  "
                        f"max(saída)={max(outputs):,}  "
                        f"max(total)={max(totais):,}").replace(',', '.')
        tokens_png = os.path.join(pasta, 'predicoes_tokens.png')
        UtilGraficos.gerar_boxplot(
            dados={'Entrada': inputs, 'Saída': outputs, 'Total': totais},
            titulo='Distribuição de Tokens — Predições',
            ylabel='Tokens',
            arquivo_saida=tokens_png,
            nota=nota_maximos,
        )
        logger.info(f"<cinza>   📊 Boxplot tokens: {tokens_png}</cinza>")

        # Boxplot de tempo
        t_sorted = sorted(tempos)
        q1_tempo = t_sorted[len(t_sorted) // 4] if len(t_sorted) >= 4 else t_sorted[0]
        nota_tempo = (f"min={min(tempos):.1f}s  "
                      f"Q1={q1_tempo:.1f}s (75% acima)  "
                      f"max={max(tempos):.1f}s")
        tempo_png = os.path.join(pasta, 'predicoes_tempo.png')
        UtilGraficos.gerar_boxplot(
            dados={'Tempo (s)': tempos},
            titulo='Distribuição de Tempo de Geração — Predições',
            ylabel='Segundos',
            arquivo_saida=tempo_png,
            nota=nota_tempo,
        )
        logger.info(f"<cinza>   ⏱️  Boxplot tempo: {tempo_png}</cinza>")
    except Exception as e:
        logger.warning(f"<amarelo>   ⚠️ Não foi possível gerar gráficos: {e}</amarelo>")

    return True


def gerar_estatisticas_predicoes_etapas(predict_dir: str, subset: str) -> None:
    """Gera estatísticas para pastas de etapa do curriculum (se existirem).

    Procura subpastas com padrão ``{subset} ({alias})`` dentro de
    *predict_dir* e chama :func:`gerar_estatisticas_predicoes` em cada uma.

    Args:
        predict_dir: Diretório raiz de predições.
        subset: Nome do subset (treino, validacao, teste).
    """
    if not os.path.isdir(predict_dir):
        return
    prefixo = f"{subset} ("
    for nome in sorted(os.listdir(predict_dir)):
        if nome.startswith(prefixo) and nome.endswith(')'):
            pasta_etapa = os.path.join(predict_dir, nome)
            if os.path.isdir(pasta_etapa):
                logger.info(f"<cinza>   📂 Estatísticas etapa: {nome}</cinza>")
                gerar_estatisticas_predicoes(pasta_etapa)


# ---------------------------------------------------------------------------
# Predições com HuggingFace Transformers (LLMsTrainer)
# ---------------------------------------------------------------------------

def executar_predict(yaml_path: str, subsets: list = None, usar_base: bool = False) -> None:
    """
    Gera predições do modelo para os subsets especificados.
    Salva os resultados na pasta {output_dir}/predict/{subset}/
    Se usar_base=True, salva em {output_dir}/predict_base/{subset}/
    
    Usa o mesmo caminho de inferência (LLMsTrainer.prompt) que executar_modelo,
    garantindo resultados consistentes entre predição em lote e exemplos interativos.
    
    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
        subsets: Lista de subsets para processar ('treino', 'validacao', 'teste').
                 Se None, processa todos.
        usar_base: Se True, usa o modelo base original em vez do treinado.
    """
    from treinar_unsloth import LLMsTrainer
    
    logger.info("\n")
    log_separador(caractere="=", largura=80)
    logger.info("<azul>>> MODO PREDICT - GERANDO PREDIÇÕES</azul>")
    log_separador(caractere="=", largura=80)
    
    # Carrega configuração para validações iniciais
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=True)
    yaml_config.validar_max_seq_length()
    _exibir_cabecalho_modelo(yaml_config)
    
    # Verifica formato de saída
    formato_json = yaml_config.formato_saida == FORMATO_SAIDA_JSON
    logger.info(f"<cinza>\n📋 Formato de saída: {yaml_config.formato_saida}</cinza>")
    
    # Verifica se há modelo treinado
    tem_modelo_treinado = _verificar_modelo_treinado(yaml_config)
    
    if usar_base:
        logger.info(f"<cinza>ℹ️  Opção --base ativada: Forçando uso do modelo base.</cinza>")
    elif not tem_modelo_treinado:
        logger.warning("<amarelo>\n⚠️ Não foi encontrado modelo LoRA treinado.</amarelo>")
        if not _perguntar_confirmacao("Deseja usar o modelo base para predição?", padrao=False):
            logger.info("Operação cancelada.")
            return
        usar_base = True
        logger.info("Usando modelo base para predição...\n")
    else:
        logger.info(f"<verde>✅ Usando modelo treinado: {yaml_config.modelo.saida}</verde>")
    
    modelo_usado = yaml_config.modelo.base if usar_base else yaml_config.modelo.saida
    
    # Define subsets a processar (padrão: apenas teste)
    if subsets is None:
        subsets = ['teste']
    
    logger.info(f"<cinza>\n📋 Subsets a processar: {', '.join(subsets)}</cinza>")
    
    # Cria diretório de predições
    nome_pasta = "predict_base" if usar_base else "predict"
    predict_dir = os.path.join(yaml_config.modelo.saida, nome_pasta)
    os.makedirs(predict_dir, exist_ok=True)
    
    # Divisão unificada: dicionário {id: {alvo, divisoes, etapas}}
    divisao_dict = yaml_config.dataset_manager.carregar_divisao_completa(yaml_config.curriculum)
    mapa_etapas = _construir_mapa_etapas(divisao_dict)

    # Pré-check: se todas as predições já existem, sai sem carregar o modelo
    if _todas_predicoes_exportadas(predict_dir, subsets, divisao_dict, mapa_etapas, formato_json=formato_json):
        return
    
    # Carrega modelo via LLMsTrainer (mesmo caminho que executar_modelo)
    logger.info("<azul>\n🔄 Carregando modelo via LLMsTrainer...</azul>")
    ini_carga = time.time()
    
    try:
        trainer = LLMsTrainer(yaml_path, force_base=usar_base)
        logger.info(f"<verde>   ✅ Modelo carregado em {time.time() - ini_carga:.1f}s</verde>")
    except Exception as e:
        logger.error(f"<vermelho>❌ Erro ao carregar modelo: {e}</vermelho>")
        return
    
    max_new_tokens = yaml_config.treinamento.max_seq_length
    
    # Estatísticas globais
    uso_total = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_registros': 0,
        'tempo_total_s': 0,
        'por_subset': {}
    }
    
    try:
        # Processa cada subset
        for subset in subsets:
            logger.info(f"<azul>\n📂 Processando subset: {subset}</azul>")
            log_separador(caractere="-", largura=60)
            
            try:
                mensagens = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo=subset, divisao=divisao_dict)
                if not mensagens:
                    logger.warning(f"<amarelo>   ⚠️ Nenhum dado encontrado para {subset}</amarelo>")
                    continue
                logger.info(f"<cinza>   📊 {len(mensagens)} registros encontrados</cinza>")
            except Exception as e:
                logger.error(f"<vermelho>   ❌ Erro ao carregar {subset}: {e}</vermelho>")
                continue
            
            # Cria diretório do subset (preserva arquivos existentes para continuação)
            subset_dir = os.path.join(predict_dir, subset)
            os.makedirs(subset_dir, exist_ok=True)
            
            subset_stats = {
                'input_tokens': 0,
                'output_tokens': 0,
                'registros_ok': 0,
                'registros_erro': 0,
                'registros_skip': 0,
                'tempo_s': 0
            }
            
            total = len(mensagens)
            ini_subset = time.time()
            
            for idx, msg in enumerate(mensagens):
                try:
                    if isinstance(msg, dict) and 'messages' in msg:
                        messages = msg['messages']
                        
                        prompt_texto = ""
                        for m in messages:
                            if m.get('role') == 'user':
                                prompt_texto = m.get('content', '')
                                break
                    else:
                        logger.warning(f"   ⚠️ Formato não reconhecido no registro {idx}")
                        subset_stats['registros_erro'] += 1
                        continue
                    
                    if not prompt_texto:
                        logger.warning(f"   ⚠️ Prompt vazio no registro {idx}")
                        subset_stats['registros_erro'] += 1
                        continue
                    
                    registro_id = msg.get('id', f'{subset}_{idx:04d}')
                    
                    # Skip se já exportado (permite continuação de exportações incompletas)
                    if _registro_ja_exportado(subset_dir, registro_id, formato_json=formato_json):
                        subset_stats['registros_skip'] += 1
                        _copiar_para_pastas_etapas(predict_dir, subset, registro_id, mapa_etapas)
                        continue
                    
                    # Usa LLMsTrainer.prompt() — mesmo caminho de executar_modelo
                    tempo_inicio = time.time()
                    resultado = trainer.prompt(
                        prompt_texto,
                        temperatura=0.01,
                        max_new_tokens=max_new_tokens
                    )
                    tempo_pred = time.time() - tempo_inicio
                    
                    resposta_modelo = resultado.get('texto', '')
                    input_tokens = resultado.get('prompt_tokens', 0)
                    output_tokens = resultado.get('completion_tokens', 0)
                    
                    # Se formato JSON, tenta parsear a resposta
                    if formato_json and resposta_modelo.strip():
                        try:
                            from util import UtilTextos
                            json_obj = UtilTextos.mensagem_to_json(resposta_modelo)
                            resposta_modelo = json.dumps(json_obj, ensure_ascii=False, indent=2)
                        except Exception:
                            pass  # mantém resposta como texto se não for JSON válido
                    
                    output_txt = os.path.join(subset_dir, f"{registro_id}.txt")
                    with open(output_txt, 'w', encoding='utf-8') as f:
                        f.write(resposta_modelo)
                    
                    tam_preview = 100
                    if len(prompt_texto) > tam_preview * 2:
                        prompt_preview = f"{prompt_texto[:tam_preview]} [...] {prompt_texto[-tam_preview:]}"
                    else:
                        prompt_preview = prompt_texto

                    usage_data = {
                        'id': registro_id,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'time_s': round(tempo_pred, 3),
                        'prompt_preview': prompt_preview,
                    }
                    output_json = os.path.join(subset_dir, f"{registro_id}.json")
                    with open(output_json, 'w', encoding='utf-8') as f:
                        json.dump(usage_data, f, ensure_ascii=False, indent=2)
                    
                    # Copia para pastas de etapa (curriculum multi-etapa)
                    _copiar_para_pastas_etapas(predict_dir, subset, registro_id, mapa_etapas)
                    
                    subset_stats['input_tokens'] += input_tokens
                    subset_stats['output_tokens'] += output_tokens
                    subset_stats['registros_ok'] += 1
                    
                    if (idx + 1) % 10 == 0 or (idx + 1) == total:
                        logger.info(f"   Progresso: {idx + 1}/{total} ({100*(idx+1)//total}%)")
                        
                except Exception as e:
                    logger.error(f"<vermelho>   ❌ Erro no registro {idx}: {e}</vermelho>")
                    subset_stats['registros_erro'] += 1
                    continue
            
            subset_stats['tempo_s'] = time.time() - ini_subset
            
            resumo_subset = {
                'subset': subset,
                'total_registros': total,
                'registros_ok': subset_stats['registros_ok'],
                'registros_erro': subset_stats['registros_erro'],
                'input_tokens_total': subset_stats['input_tokens'],
                'output_tokens_total': subset_stats['output_tokens'],
                'tempo_processamento_s': round(subset_stats['tempo_s'], 2),
                'data_geracao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'modelo': modelo_usado,
                'formato_saida': yaml_config.formato_saida,
            }
            resumo_file = os.path.join(subset_dir, "resumo.json")
            with open(resumo_file, 'w', encoding='utf-8') as f:
                json.dump(resumo_subset, f, ensure_ascii=False, indent=2)
            
            skip_msg = f", {subset_stats['registros_skip']} já exportados" if subset_stats['registros_skip'] else ""
            logger.info(f"<verde>   ✅ {subset_stats['registros_ok']} predições salvas em: {subset_dir}</verde>")
            logger.info(f"<cinza>   📊 Tokens: {subset_stats['input_tokens']} entrada, {subset_stats['output_tokens']} saída{skip_msg}</cinza>")
            gerar_estatisticas_predicoes(subset_dir)
            gerar_estatisticas_predicoes_etapas(predict_dir, subset)
            
            uso_total['input_tokens'] += subset_stats['input_tokens']
            uso_total['output_tokens'] += subset_stats['output_tokens']
            uso_total['total_registros'] += subset_stats['registros_ok']
            uso_total['tempo_total_s'] += subset_stats['tempo_s']
            uso_total['por_subset'][subset] = resumo_subset
    
    finally:
        # Libera memória do trainer
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Salva resumo geral
    resumo_geral = {
        'data_geracao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'modelo_base': yaml_config.modelo.base,
        'modelo_saida': yaml_config.modelo.saida,
        'modelo_usado': modelo_usado,
        'formato_saida': yaml_config.formato_saida,
        'total_registros': uso_total['total_registros'],
        'input_tokens_total': uso_total['input_tokens'],
        'output_tokens_total': uso_total['output_tokens'],
        'tempo_total_s': round(uso_total['tempo_total_s'], 2),
        'subsets': uso_total['por_subset']
    }
    resumo_geral_file = os.path.join(predict_dir, "resumo_geral.json")
    with open(resumo_geral_file, 'w', encoding='utf-8') as f:
        json.dump(resumo_geral, f, ensure_ascii=False, indent=2)
    
    log_separador(caractere="=", largura=80)
    logger.info(f"<verde>✅ PREDICT COMPLETO - Resultados em: {predict_dir}</verde>")
    logger.info(f"<cinza>📊 Total: {uso_total['total_registros']} registros, {uso_total['input_tokens']} + {uso_total['output_tokens']} tokens</cinza>")
    log_separador(caractere="=", largura=80)


# ---------------------------------------------------------------------------
# Merge / Exportação de Modelos
# ---------------------------------------------------------------------------

def executar_merge(yaml_path: str, quantizacao: str = None, gerar_zip: bool = False) -> None:
    """
    Realiza merge do modelo treinado com o base.
    Salva em {output_dir}(merged_FORMATO)/ ou {output_dir}(merged_FORMATO).zip
    
    Exporta HF safetensors (16bit ou 4bit) usando transformers + PEFT.
    Sem dependência de unsloth. Para converter para GGUF (Ollama), use llama.cpp.
    Gera Modelfile e README_OLLAMA.md com instruções de conversão.
    
    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
        quantizacao: Opcional. '16bit', '4bit', '16bit_zip', '4bit_zip'.
        gerar_zip: Se True, compacta e remove o diretório temporário.
    """
    logger.info("\n")
    log_separador(caractere="=", largura=80)
    logger.info("<azul>>> MODO MERGE - INTEGRANDO LORA ADAPTERS</azul>")
    log_separador(caractere="=", largura=80)
    
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=True)
    _exibir_cabecalho_modelo(yaml_config)
    
    output_dir = yaml_config.modelo.saida
    
    if not _verificar_modelo_treinado(yaml_config):
        logger.error(f"<vermelho>❌ Erro: Não foi encontrado modelo treinado em {output_dir}</vermelho>")
        return
    
    mapa_quant = {
        '1': '16bit',
        '2': '4bit',
        '1z': '16bit_zip',
        '2z': '4bit_zip',
    }
    
    if not quantizacao:
        if sys.stdin.isatty():
            itens_quant = [
                ('1',  '16bit',     '☁️  HF safetensors 16-bit (Padrão - vLLM/HF/llama.cpp) ⭐'),
                ('1z', '16bit+zip', '📦 HF 16-bit → .zip (merge temporário, exporta só o .zip)'),
                ('2',  '4bit',      '📉 HF safetensors 4-bit (Compacto)'),
                ('2z', '4bit+zip',  '📦 HF 4-bit  → .zip (merge temporário, exporta só o .zip)'),
            ]
            try:
                escolha = exibir_menu_opcoes(
                    titulo='<azul>📦 Formato de Exportação/Merge:</azul>',
                    itens=itens_quant,
                    prompt='Opção [1]',
                )
                if not escolha:
                    escolha = '1'
                quantizacao = mapa_quant.get(escolha, '16bit')
            except (KeyboardInterrupt, EOFError):
                quantizacao = '16bit'
        else:
            quantizacao = '16bit'
            logger.info(f"Iniciando merge padrão ({quantizacao})")

    quantizacao = quantizacao.lower()
    
    # Detecta modo zip (via menu _zip ou flag --zip)
    if quantizacao.endswith('_zip'):
        gerar_zip = True
        quantizacao = quantizacao.replace('_zip', '')  # '16bit_zip' → '16bit'
    
    dirname = f"{output_dir}(merged_{quantizacao})"
    zip_path = f"{dirname}.zip"
    
    # Modo zip: merge em tempfile.TemporaryDirectory → zip → limpeza automática
    # Modo normal: merge direto no destino final
    if gerar_zip:
        logger.info(f"<cinza>\n📦 Modo ZIP: merge em diretório temporário → compactação</cinza>")
        logger.info(f"<cinza>   Destino final: {zip_path}</cinza>")
        logger.info(f"<cinza>⚙️  Formato: {quantizacao} (HF safetensors)</cinza>")
        logger.info("<cinza>📦 Motor: transformers + PEFT (sem unsloth)</cinza>")
        
        if os.path.exists(zip_path):
            logger.warning(f"⚠️  Arquivo .zip já existe: {zip_path}")
            if not _perguntar_confirmacao("Deseja sobrescrever?", padrao=False):
                logger.info("Operação cancelada.")
                return
            os.remove(zip_path)
        
        logger.info(f"<azul>\n🔄 Carregando modelo LoRA de: {output_dir}</azul>")
        logger.info("   Isso pode levar alguns instantes...")
        
        import tempfile
        # Nome base que aparecerá como raiz dentro do .zip
        base_name = os.path.basename(dirname)
        
        try:
            with tempfile.TemporaryDirectory(prefix="merge_") as tmpdir:
                merge_dir = os.path.join(tmpdir, base_name)
                os.makedirs(merge_dir, exist_ok=True)
                
                logger.info(f"<cinza>   📁 Diretório temporário: {tmpdir}</cinza>")
                
                _executar_merge_hf(yaml_config, output_dir, merge_dir, quantizacao)
                
                src_treinamento = os.path.join(output_dir, "treinamento")
                dst_treinamento = os.path.join(merge_dir, "treinamento")
                
                if os.path.exists(src_treinamento):
                    logger.info("📋 Copiando relatórios e gráficos...")
                    shutil.copytree(src_treinamento, dst_treinamento, dirs_exist_ok=True)
                
                try:
                    from treinar_to_ollama import gerar_modelfile_ollama
                    gerar_modelfile_ollama(merge_dir, yaml_config, quantizacao)
                except Exception as e:
                    logger.warning(f"<amarelo>⚠️  Erro ao gerar Modelfile: {e}</amarelo>")
                
                logger.info("<verde>✅ Merge concluído com sucesso!</verde>")
                
                # Compacta do tmpdir para o destino final
                _compactar_modelo_zip(merge_dir, zip_destino=zip_path)
                logger.info(f"<verde>📦 Exportação finalizada: {zip_path}</verde>")
                # TemporaryDirectory limpa automaticamente ao sair do with
                
        except Exception as e:
            logger.error(f"<vermelho>❌ Erro ao realizar merge/exportação: {e}</vermelho>")
        
        return
    
    # --- Modo normal (sem zip): merge direto no destino final ---
    logger.info(f"<cinza>\n📂 Diretório de destino: {dirname}</cinza>")
    logger.info(f"<cinza>⚙️  Formato: {quantizacao} (HF safetensors)</cinza>")
    logger.info("<cinza>📦 Motor: transformers + PEFT (sem unsloth)</cinza>")

    if os.path.exists(dirname):
        logger.warning(f"⚠️  O diretório já existe.")
        if not _perguntar_confirmacao("Deseja sobrescrever (isso apagará o conteúdo atual)?", padrao=False):
            logger.info("Operação cancelada.")
            return
        try:
            shutil.rmtree(dirname)
            logger.info("   Diretório antigo removido.")
        except Exception as e:
            logger.error(f"Erro ao remover diretório: {e}")
            return
            
    logger.info(f"<azul>\n🔄 Carregando modelo LoRA de: {output_dir}</azul>")
    logger.info("   Isso pode levar alguns instantes...")
    
    try:
        _executar_merge_hf(yaml_config, output_dir, dirname, quantizacao)
        
        src_treinamento = os.path.join(output_dir, "treinamento")
        dst_treinamento = os.path.join(dirname, "treinamento")
        
        if os.path.exists(src_treinamento):
            logger.info("📋 Copiando relatórios e gráficos...")
            shutil.copytree(src_treinamento, dst_treinamento, dirs_exist_ok=True)
        
        # Gera Modelfile e README_OLLAMA.md para facilitar importação no Ollama
        try:
            from treinar_to_ollama import gerar_modelfile_ollama
            gerar_modelfile_ollama(dirname, yaml_config, quantizacao)
        except Exception as e:
            logger.warning(f"<amarelo>⚠️  Erro ao gerar Modelfile: {e}</amarelo>")
        
        logger.info("<verde>✅ Merge concluído com sucesso!</verde>")
        logger.info(f"<cinza>   Modelo pronto em: {dirname}</cinza>")
        
    except Exception as e:
        logger.error(f"<vermelho>❌ Erro ao realizar merge/exportação: {e}</vermelho>")


def _compactar_modelo_zip(dirname: str, zip_destino: str = None) -> None:
    """Compacta o diretório merged em um arquivo .zip.
    
    Args:
        dirname: Diretório de origem contendo o modelo merged.
        zip_destino: Caminho completo do .zip de destino. Se None, cria
                     no mesmo nível do diretório com o mesmo nome.
                     Ex: ./modelo(merged_16bit)/ → ./modelo(merged_16bit).zip
    
    Usa zipfile com compressão ZIP_DEFLATED (compatível com qualquer OS).
    Safetensors já são bastante compactos, então o ganho de compressão é mínimo,
    mas o .zip facilita o transporte (arquivo único) e verificação de integridade.
    """
    import zipfile
    
    zip_path = zip_destino or f"{dirname}.zip"
    
    if os.path.exists(zip_path):
        logger.warning(f"⚠️  Arquivo já existe: {zip_path}")
        if not _perguntar_confirmacao("Deseja sobrescrever?", padrao=False):
            logger.info("Compactação cancelada.")
            return
        os.remove(zip_path)
    
    logger.info(f"<azul>\n📦 Compactando modelo em .zip...</azul>")
    logger.info(f"<cinza>   Origem: {dirname}</cinza>")
    logger.info(f"<cinza>   Destino: {zip_path}</cinza>")
    
    ini = time.time()
    total_bytes = 0
    n_arquivos = 0
    
    try:
        # Calcula tamanho total para progresso
        todos_arquivos = []
        for root, dirs, files in os.walk(dirname):
            for f in files:
                filepath = os.path.join(root, f)
                todos_arquivos.append(filepath)
                total_bytes += os.path.getsize(filepath)
        
        logger.info(f"<cinza>   Arquivos: {len(todos_arquivos)} | Tamanho: {total_bytes / (1024**3):.2f} GB</cinza>")
        
        # Nome base do diretório (para preservar a estrutura relativa dentro do .zip)
        base_name = os.path.basename(dirname)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for filepath in todos_arquivos:
                # Caminho relativo dentro do zip: base_name/subdir/arquivo
                arcname = os.path.join(base_name, os.path.relpath(filepath, dirname))
                zf.write(filepath, arcname)
                n_arquivos += 1
                
                if n_arquivos % 5 == 0 or n_arquivos == len(todos_arquivos):
                    logger.info(f"   Progresso: {n_arquivos}/{len(todos_arquivos)} arquivos")
        
        zip_size = os.path.getsize(zip_path)
        tempo = time.time() - ini
        ratio = (zip_size / total_bytes * 100) if total_bytes > 0 else 100
        
        logger.info(f"<verde>✅ .zip criado com sucesso!</verde>")
        logger.info(f"<cinza>   📦 {zip_path}</cinza>")
        logger.info(f"<cinza>   📊 {zip_size / (1024**3):.2f} GB ({ratio:.0f}% do original) | {tempo:.1f}s</cinza>")
        
    except Exception as e:
        logger.error(f"<vermelho>❌ Erro ao compactar: {e}</vermelho>")
        # Remove zip parcial se existir
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass


def _executar_merge_hf(yaml_config, output_dir: str, dirname: str, quantizacao: str) -> None:
    """Merge HF safetensors usando transformers + PEFT (sem dependência do unsloth).
    
    Carrega o modelo em full precision (16-bit) para o merge,
    independente da quantização usada no treinamento.
    Para converter para GGUF (Ollama), use llama.cpp após o merge.
    Veja treinar_ollama_readme.md para instruções detalhadas.
    """
    from treinar_model_loader import ModelLoader, QuantizationConfig
    
    # Para 4bit, usa BitsAndBytes para reduzir tamanho do safetensors
    # Para 16bit (padrão), carrega em full precision
    if quantizacao == '4bit':
        quant_cfg = QuantizationConfig(nbits=4)
        logger.info("💾 Merge com quantização 4-bit (BitsAndBytes)...")
    else:
        quant_cfg = None
        logger.info("💾 Merge em full precision (16-bit safetensors)...")
    
    model, tokenizer = ModelLoader.load_lora_model(
        base_model_name=yaml_config.modelo.base,
        lora_model_path=output_dir,
        max_seq_length=yaml_config.treinamento.max_seq_length,
        quant_config=quant_cfg,
    )
    
    ModelLoader.save_merged_model(model, dirname, tokenizer)


# ---------------------------------------------------------------------------
# Inferência interativa com HuggingFace Transformers (LLMsTrainer)
# ---------------------------------------------------------------------------

def executar_modelo(yaml_path: str, n_exemplos: int = 1, usar_base: bool = False) -> None:
    """
    Testa inferência interativa com N exemplos do dataset de treino.
    
    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
        n_exemplos: Número de exemplos para testar
        usar_base: Se True, usa o modelo base (ignora LoRA treinado)
    """
    from treinar_unsloth import LLMsTrainer
    
    logger.info("\n")
    log_separador(caractere="=", largura=80)
    logger.info(f"<azul>>> MODO MODELO - TESTANDO INFERÊNCIA ({n_exemplos} exemplo(s))</azul>")
    log_separador(caractere="=", largura=80)
    
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=False)
    
    if not usar_base:
        if not _verificar_modelo_treinado(yaml_config):
            logger.warning("<amarelo>\n⚠️  Não foi encontrado modelo LoRA treinado na pasta de saída.</amarelo>")
            if not _perguntar_confirmacao("Deseja continuar com o modelo base?", padrao=False):
                logger.info("Operação cancelada.")
                return
            usar_base = True
            logger.info("Continuando com modelo base (sem fine-tuning)...\n")
        else:
            logger.info(f"<verde>✅ Modelo LoRA treinado encontrado em: {yaml_config.modelo.saida}</verde>")
    else:
        logger.info("<cinza>ℹ️  Opção --base ativada: Forçando uso do modelo base.</cinza>")

    trainer = LLMsTrainer(yaml_path, force_base=usar_base)
    resultado = trainer.testar_predicoes(
        n_exemplos=n_exemplos, 
        temperatura=0.01, 
        max_new_tokens=512
    )
    
    # Exibe resumo de memória
    if resultado.get('metricas_memoria'):
        metricas = resultado['metricas_memoria']
        logger.info("\n📊 RESUMO DE USO DE MEMÓRIA:")
        if 'ram' in metricas:
            logger.info(f"   RAM: máx={metricas['ram'].get('max_gb', 0):.1f} GB, média={metricas['ram'].get('media_gb', 0):.1f} GB")
        if 'gpu' in metricas and metricas['gpu'].get('num_gpus', 0) > 0:
            logger.info(f"   GPU: máx={metricas['gpu'].get('max_gb', 0):.1f} GB, média={metricas['gpu'].get('media_gb', 0):.1f} GB ({metricas['gpu'].get('num_gpus', 0)} GPU(s))")
    
    # Libera memória
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    log_separador(caractere="=", largura=80)
    logger.info("<verde>✅ TESTE DE INFERÊNCIA COMPLETO</verde>")
    log_separador(caractere="=", largura=80)


# ---------------------------------------------------------------------------
# Predições com vLLM (inferência rápida)
# ---------------------------------------------------------------------------

def executar_predict_vllm(yaml_path: str, subsets: list = None, usar_base: bool = False) -> None:
    """Executa predições usando vLLM para inferência de alta performance.

    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
        subsets: Lista de subsets a processar (None = todos)
        usar_base: Se True, forca o uso do modelo base sem LoRA
    """
    try:
        from treinar_vllm_inference import VLLMInferenceEngine, VLLM_AVAILABLE, get_recommended_config
    except ImportError:
        logger.error("❌ Módulo vLLM não encontrado!")
        return

    if not VLLM_AVAILABLE:
        logger.error("❌ vLLM não está instalado!")
        logger.info("   Para usar predições rápidas, instale: pip install vllm")
        return

    logger.info("\n")
    log_separador(caractere="=", largura=80)
    logger.info("<azul>>> MODO PREDICT - VLLM (INFERÊNCIA RÁPIDA 🚀)</azul>")
    log_separador(caractere="=", largura=80)

    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=True)
    yaml_config.validar_max_seq_length()
    _exibir_cabecalho_modelo(yaml_config)

    output_dir = yaml_config.modelo.saida
    modelo_base_path = yaml_config.modelo.base
    max_seq_length = yaml_config.treinamento.max_seq_length

    # Verifica formato de saída
    formato_json = yaml_config.formato_saida == FORMATO_SAIDA_JSON
    logger.info(f"<cinza>\n📋 Formato de saída: {yaml_config.formato_saida}</cinza>")

    # Decide o caminho do modelo a utilizar
    lora_adapter_path = None
    if usar_base:
        logger.info(f"<cinza>ℹ️  Usando modelo BASE: {modelo_base_path}</cinza>")
    else:
        if not _verificar_modelo_treinado(yaml_config):
            logger.warning("<amarelo>\n⚠️ Não foi encontrado modelo treinado em {output_dir}</amarelo>")
            if not _perguntar_confirmacao("Deseja usar o modelo base para predição?", padrao=False):
                return
            usar_base = True
            logger.info("Continuando com modelo base...\n")
        else:
            lora_adapter_path = output_dir
            logger.info(f"<verde>✅ Modelo treinado (LoRA): {lora_adapter_path}</verde>")
            logger.info(f"<cinza>   Modelo base para vLLM: {modelo_base_path}</cinza>")

    # Detecta número de GPUs
    import torch
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logger.info(f"🎮 GPUs disponíveis: {num_gpus}")

    # Configuração recomendada
    # Contexto estimado a partir dos dados reais de tokens (CSVs do curriculum)
    ctx_info = yaml_config.estimar_contexto_predict()
    vllm_context = ctx_info["contexto"]
    vllm_max_new_tokens = ctx_info["max_new_tokens"]
    config = get_recommended_config(num_gpus=num_gpus, model_size="7B")
    config.max_model_len = vllm_context
    logger.info(f"⚙️  Tensor Parallel: {config.tensor_parallel_size} GPU(s)")
    logger.info(f"⚙️  GPU Memory Utilization: {config.gpu_memory_utilization*100:.0f}%")
    logger.info(f"⚙️  Max Model Len: {config.max_model_len} ({ctx_info['fonte']})")
    logger.info(f"⚙️  Max New Tokens: {vllm_max_new_tokens}\n")

    # Define subsets a processar
    if subsets is None:
        subsets = ['treino', 'validacao', 'teste']

    logger.info(f"<cinza>\n📋 Subsets a processar: {', '.join(subsets)}</cinza>")

    # Cria diretório de predições
    nome_pasta = "predict_base_vllm" if usar_base else "predict_vllm"
    predict_dir = os.path.join(output_dir, nome_pasta)
    os.makedirs(predict_dir, exist_ok=True)

    # Divisão unificada: dicionário {id: {alvo, divisoes, etapas}}
    divisao_dict = yaml_config.dataset_manager.carregar_divisao_completa(yaml_config.curriculum)
    mapa_etapas = _construir_mapa_etapas(divisao_dict)

    # Pré-check: se todas as predições já existem, sai sem carregar o modelo
    if _todas_predicoes_exportadas(predict_dir, subsets, divisao_dict, mapa_etapas, formato_json=formato_json):
        return

    # Inicializa vLLM (modelo base + LoRA adapter opcional)
    try:
        engine = VLLMInferenceEngine(
            model_path=modelo_base_path,
            config=config,
            lora_path=lora_adapter_path,
        )
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar vLLM: {e}")
        return

    # O engine expõe seu tokenizer base para uso generalizado (como get_tokenizer)
    vllm_tokenizer = engine.llm.get_tokenizer()
    if getattr(vllm_tokenizer, "pad_token", None) is None:
        vllm_tokenizer.pad_token = vllm_tokenizer.eos_token

    uso_total = {'input_tokens': 0, 'output_tokens': 0, 'total_registros': 0, 'tempo_total_s': 0}

    try:
        for subset in subsets:
            logger.info(f"<azul>\n📂 Processando subset: {subset}</azul>")
            log_separador(caractere="-", largura=60)

            try:
                mensagens = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo=subset, divisao=divisao_dict)
                if not mensagens:
                    logger.warning(f"<amarelo>   ⚠️ Nenhum dado encontrado para {subset}</amarelo>")
                    continue
                logger.info(f"<cinza>   📊 {len(mensagens)} registros encontrados</cinza>")
            except Exception as e:
                logger.error(f"<vermelho>   ❌ Erro ao carregar {subset}: {e}</vermelho>")
                continue

            # Cria diretório do subset (preserva arquivos existentes para continuação)
            subset_dir = os.path.join(predict_dir, subset)
            os.makedirs(subset_dir, exist_ok=True)

            total = len(mensagens)
            subset_stats = {'input_tokens': 0, 'output_tokens': 0, 'registros_ok': 0, 'registros_erro': 0, 'registros_skip': 0}
            ini_subset = time.time()

            max_input_len = max_seq_length - 256
            prompts_batch = []
            registros_batch = []

            for idx, msg in enumerate(mensagens):
                try:
                    if isinstance(msg, dict) and 'messages' in msg:
                        messages = msg['messages']
                        prompt_texto = ""
                        for m in messages:
                            if m.get('role') == 'user':
                                prompt_texto = m.get('content', '')
                                break
                    else:
                        subset_stats['registros_erro'] += 1
                        continue

                    if not prompt_texto:
                        subset_stats['registros_erro'] += 1
                        continue

                    registro_id = msg.get('id', f'{subset}_{idx:04d}')
                    
                    # Skip se já exportado (permite continuação de exportações incompletas)
                    if _registro_ja_exportado(subset_dir, registro_id, formato_json=formato_json):
                        subset_stats['registros_skip'] += 1
                        _copiar_para_pastas_etapas(predict_dir, subset, registro_id, mapa_etapas)
                        continue

                    # Formata com chat template (exclui assistant = resposta esperada)
                    chat_msgs = [m for m in messages if m.get('role') != 'assistant']
                    formatted_prompt = vllm_tokenizer.apply_chat_template(
                        chat_msgs, tokenize=False, add_generation_prompt=True
                    )
                    prompt_ids = vllm_tokenizer.encode(formatted_prompt)
                    if len(prompt_ids) > max_input_len:
                        prompt_ids = prompt_ids[:max_input_len]
                        formatted_prompt = vllm_tokenizer.decode(prompt_ids, skip_special_tokens=False)

                    registro_id = msg.get('id', f'{subset}_{idx:04d}')
                    prompts_batch.append(formatted_prompt)
                    registros_batch.append({
                        'id': registro_id,
                        'input_tokens': len(prompt_ids),
                        'idx': idx
                    })
                except Exception as e:
                    logger.error(f"<vermelho>   ❌ Erro gerando prompt: {e}</vermelho>")
                    subset_stats['registros_erro'] += 1

            if not prompts_batch:
                continue

            # Processamento em lote real com vLLM
            try:
                # vLLM lidando com o fluxo - pegamos min do limite para gerar
                # Para inferir tokens_para_gerar (tamanho seguro) iteramos maximos
                max_len_batch = max(reg['input_tokens'] for reg in registros_batch)
                tokens_para_gerar = min(vllm_max_new_tokens, config.max_model_len - max_len_batch)

                tempo_inicio = time.time()
                resultados_vllm = engine.generate_batch(
                    prompts=prompts_batch,
                    max_tokens=tokens_para_gerar,
                    temperature=0.01,
                    top_k=2,
                    n=1
                )
                tempo_pred = time.time() - tempo_inicio

                # Processar as respostas e salvar
                for idx_vllm, res in enumerate(resultados_vllm):
                    reg_meta = registros_batch[idx_vllm]
                    subset_stats['input_tokens'] += reg_meta['input_tokens']

                    resposta_modelo = res["output"]
                    if formato_json and resposta_modelo.strip():
                        try:
                            from util import UtilTextos
                            json_obj = UtilTextos.mensagem_to_json(resposta_modelo)
                            resposta_modelo = json.dumps(json_obj, ensure_ascii=False, indent=2)
                        except Exception:
                            pass

                    output_txt = os.path.join(subset_dir, f"{reg_meta['id']}.txt")
                    with open(output_txt, 'w', encoding='utf-8') as f:
                        f.write(resposta_modelo)

                    usage_data = {
                        'id': reg_meta['id'],
                        'input_tokens': reg_meta['input_tokens'],
                        'output_tokens': res["tokens"],
                        'time_s': round(tempo_pred / len(prompts_batch), 3),
                    }
                    output_json = os.path.join(subset_dir, f"{reg_meta['id']}.json")
                    with open(output_json, 'w', encoding='utf-8') as f:
                        json.dump(usage_data, f, ensure_ascii=False, indent=2)

                    # Copia para pastas de etapa (curriculum multi-etapa)
                    _copiar_para_pastas_etapas(predict_dir, subset, reg_meta['id'], mapa_etapas)

                    subset_stats['output_tokens'] += res["tokens"]
                    subset_stats['registros_ok'] += 1

            except Exception as e:
                logger.error(f"<vermelho>   ❌ Erro no batch inference: {e}</vermelho>")

            tempo_subset = time.time() - ini_subset
            skip_msg = f", {subset_stats['registros_skip']} já exportados" if subset_stats['registros_skip'] else ""
            logger.info(f"<verde>   ✅ {subset_stats['registros_ok']} predições salvas em: {subset_dir}</verde>")
            logger.info(f"<cinza>   📊 Tokens: {subset_stats['input_tokens']} entrada, {subset_stats['output_tokens']} saída ({tempo_subset:.1f}s){skip_msg}</cinza>")
            gerar_estatisticas_predicoes(subset_dir)
            gerar_estatisticas_predicoes_etapas(predict_dir, subset)

            uso_total['input_tokens'] += subset_stats['input_tokens']
            uso_total['output_tokens'] += subset_stats['output_tokens']
            uso_total['total_registros'] += subset_stats['registros_ok']
            uso_total['tempo_total_s'] += tempo_subset

    finally:
        pass

    log_separador(caractere="=", largura=80)
    logger.info(f"<verde>✅ PREDICT VLLM COMPLETO - Resultados em: {predict_dir}</verde>")
    logger.info(f"<cinza>📊 Total: {uso_total['total_registros']} registros, {uso_total['input_tokens']} + {uso_total['output_tokens']} tokens ({uso_total['tempo_total_s']:.1f}s)</cinza>")
    log_separador(caractere="=", largura=80)


# ---------------------------------------------------------------------------
# Inferência interativa com vLLM
# ---------------------------------------------------------------------------

def executar_modelo_vllm(yaml_path: str, n_exemplos: int = 1, usar_base: bool = False) -> None:
    """Testa inferência interativa com N exemplos usando vLLM (inferência rápida).

    Segue o mesmo fluxo de executar_modelo, mas utiliza o VLLMInferenceEngine
    para gerar as predições, beneficiando-se de PagedAttention e batching otimizado.

    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
        n_exemplos: Número de exemplos para testar
        usar_base: Se True, usa o modelo base (ignora LoRA treinado)
    """
    try:
        from treinar_vllm_inference import VLLMInferenceEngine, VLLM_AVAILABLE, get_recommended_config
    except ImportError:
        logger.error("❌ Módulo treinar_vllm_inference não encontrado!")
        return

    if not VLLM_AVAILABLE:
        logger.error("❌ vLLM não está instalado!")
        logger.info("   Instale com: pip install vllm")
        return

    logger.info("\n")
    log_separador(caractere="=", largura=80)
    logger.info(f"<azul>>> MODO MODELO - VLLM 🚀 TESTANDO INFERÊNCIA ({n_exemplos} exemplo(s))</azul>")
    log_separador(caractere="=", largura=80)

    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=True)
    _exibir_cabecalho_modelo(yaml_config)

    # Decide o caminho do modelo a utilizar
    # vLLM com LoRA: carrega modelo BASE + adapter LoRA separado
    modelo_base_path = yaml_config.modelo.base
    lora_adapter_path = None

    if usar_base:
        logger.info(f"<cinza>ℹ️  Usando modelo BASE: {modelo_base_path}</cinza>")
    else:
        if not _verificar_modelo_treinado(yaml_config):
            logger.warning("<amarelo>\n⚠️  Não foi encontrado modelo LoRA treinado na pasta de saída.</amarelo>")
            if not _perguntar_confirmacao("Deseja continuar com o modelo base?", padrao=False):
                logger.info("Operação cancelada.")
                return
            usar_base = True
            logger.info("Continuando com modelo base (sem fine-tuning)...\n")
        else:
            lora_adapter_path = yaml_config.modelo.saida
            logger.info(f"<verde>✅ Modelo treinado (LoRA): {lora_adapter_path}</verde>")
            logger.info(f"<cinza>   Modelo base para vLLM: {modelo_base_path}</cinza>")

    # ---- Carrega exemplos do dataset ----
    logger.info("<azul>\n📂 Carregando exemplos do dataset de treino...</azul>")

    try:
        mensagens = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo="treino")
        if not mensagens:
            logger.error("<vermelho>❌ Nenhum dado de treino encontrado.</vermelho>")
            return
    except Exception as e:
        logger.error(f"<vermelho>❌ Erro ao carregar dados de treino: {e}</vermelho>")
        return

    n_exemplos = min(n_exemplos, len(mensagens))
    logger.info(f"<cinza>   📊 {len(mensagens)} registros disponíveis, testando {n_exemplos}</cinza>")

    # ---- Inicializa vLLM ----
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logger.info(f"\n🎮 GPUs disponíveis: {num_gpus}")

    config = get_recommended_config(num_gpus=num_gpus, model_size="7B")
    # Contexto estimado a partir dos dados reais de tokens (CSVs do curriculum)
    ctx_info = yaml_config.estimar_contexto_predict()
    config.max_model_len = ctx_info["contexto"]
    logger.info(f"⚙️  Tensor Parallel: {config.tensor_parallel_size} GPU(s)")
    logger.info(f"⚙️  GPU Memory Utilization: {config.gpu_memory_utilization*100:.0f}%")
    logger.info(f"⚙️  Max Model Len: {config.max_model_len} ({ctx_info['fonte']})")
    logger.info(f"⚙️  Max New Tokens: {ctx_info['max_new_tokens']}")

    logger.info("<azul>\n🚀 Inicializando vLLM...</azul>")
    try:
        engine = VLLMInferenceEngine(
            model_path=modelo_base_path,
            config=config,
            lora_path=lora_adapter_path,
        )
    except Exception as e:
        logger.error(f"<vermelho>❌ Erro ao inicializar vLLM: {e}</vermelho>")
        return

    # ---- Processa exemplos ----
    max_new_tokens = ctx_info["max_new_tokens"]
    temperatura = 0.01
    resultados = []

    for i in range(n_exemplos):
        log_separador(caractere="-", largura=60)
        logger.info(f">> EXEMPLO {i+1}/{n_exemplos}")
        log_separador(caractere="-", largura=60)

        msg = mensagens[i]
        if not isinstance(msg, dict) or 'messages' not in msg:
            logger.warning(f"   ⚠️ Formato não reconhecido no registro {i}")
            continue

        messages = msg['messages']
        prompt_texto = ""
        resposta_esperada = ""
        for m in messages:
            if m.get('role') == 'user':
                prompt_texto = m.get('content', '')
            elif m.get('role') == 'assistant':
                resposta_esperada = m.get('content', '')

        if not prompt_texto:
            logger.warning(f"   ⚠️ Prompt vazio no registro {i}")
            continue

        # Exibe prompt
        logger.info(f">> PROMPT:")
        if len(prompt_texto) > 500:
            logger.info(f"   {prompt_texto[:250]} [...] {prompt_texto[-250:]}")
        else:
            logger.info(f"   {prompt_texto}")

        # Exibe resposta esperada
        logger.info(f"\n>> RESPOSTA ESPERADA:")
        if len(resposta_esperada) > 500:
            logger.info(f"   {resposta_esperada[:250]} [...] {resposta_esperada[-250:]}")
        else:
            logger.info(f"   {resposta_esperada}")

        # Gera predição com vLLM
        try:
            # Formata com chat template (exclui assistant = resposta esperada)
            vllm_tokenizer = engine.llm.get_tokenizer()
            chat_msgs = [m for m in messages if m.get('role') != 'assistant']
            formatted_prompt = vllm_tokenizer.apply_chat_template(
                chat_msgs, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = vllm_tokenizer.encode(formatted_prompt)
            max_input_len = config.max_model_len - 256
            if len(prompt_ids) > max_input_len:
                logger.warning(f"   ⚠️ Prompt truncado: {len(prompt_ids)} → {max_input_len} tokens")
                prompt_ids = prompt_ids[:max_input_len]
                formatted_prompt = vllm_tokenizer.decode(prompt_ids, skip_special_tokens=False)

            # vLLM restringe: prompt_len + max_tokens <= max_model_len
            tokens_para_gerar = min(max_new_tokens, config.max_model_len - len(prompt_ids))

            tempo_inicio = time.time()
            resultado = engine.generate_batch(
                prompts=[formatted_prompt],
                max_tokens=tokens_para_gerar,
                temperature=max(temperatura, 0.01),
                top_k=20 if temperatura > 0.3 else 2,
                n=1,
            )
            tempo_pred = time.time() - tempo_inicio

            if resultado:
                resposta_modelo = resultado[0]["output"]
                output_tokens = resultado[0]["tokens"]
                finish_reason = resultado[0].get("finish_reason", "?")
            else:
                resposta_modelo = "(sem resposta)"
                output_tokens = 0
                finish_reason = "error"

            logger.info(f"\n>> RESPOSTA DO MODELO (vLLM 🚀):")
            if len(resposta_modelo) > 500:
                logger.info(f"   {resposta_modelo[:250]} [...] {resposta_modelo[-250:]}")
            else:
                logger.info(f"   {resposta_modelo}")

            logger.info(f"\n>> ESTATÍSTICAS:")
            logger.info(f"   - Tokens da resposta: {output_tokens}")
            logger.info(f"   - Finish reason: {finish_reason}")
            logger.info(f"   - Temperatura: {temperatura}")
            logger.info(f"   - Tempo de predição: {tempo_pred:.2f}s")
            if output_tokens > 0 and tempo_pred > 0:
                logger.info(f"   - Velocidade: {output_tokens / tempo_pred:.1f} tokens/s")

            resultados.append({
                "exemplo": i + 1,
                "output_tokens": output_tokens,
                "tempo_segundos": round(tempo_pred, 2),
                "finish_reason": finish_reason,
            })

        except Exception as e:
            logger.error(f"<vermelho>   ❌ Erro ao gerar predição: {e}</vermelho>")

    # ---- Resumo final ----
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if resultados:
        total_tokens = sum(r["output_tokens"] for r in resultados)
        total_tempo = sum(r["tempo_segundos"] for r in resultados)
        logger.info(f"\n📊 RESUMO VLLM:")
        logger.info(f"   Exemplos processados: {len(resultados)}/{n_exemplos}")
        logger.info(f"   Tokens gerados: {total_tokens}")
        logger.info(f"   Tempo total: {total_tempo:.2f}s")
        if total_tempo > 0:
            logger.info(f"   Throughput médio: {total_tokens / total_tempo:.1f} tokens/s")

    log_separador(caractere="=", largura=80)
    logger.info("<verde>✅ TESTE DE INFERÊNCIA VLLM COMPLETO</verde>")
    log_separador(caractere="=", largura=80)


# ---------------------------------------------------------------------------
# Inferência com Unsloth (FastLanguageModel.for_inference)
# ---------------------------------------------------------------------------

def executar_modelo_unsloth(yaml_path: str, n_exemplos: int = 1, usar_base: bool = False) -> None:
    """Testa inferência com N exemplos usando unsloth (2x mais rápido que HF padrão).

    Carrega o modelo via FastLanguageModel.from_pretrained e ativa
    FastLanguageModel.for_inference() para inferência otimizada.

    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
        n_exemplos: Número de exemplos para testar
        usar_base: Se True, usa o modelo base (ignora LoRA treinado)
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("❌ Módulo unsloth não encontrado!")
        logger.info("   Instale com: pip install unsloth")
        return

    # Garante que triton encontre um compilador C
    import shutil as _shutil
    if not os.environ.get('CC'):
        cc_path = _shutil.which('gcc') or _shutil.which('cc')
        if cc_path:
            os.environ['CC'] = cc_path

    logger.info("\n")
    log_separador(caractere="=", largura=80)
    logger.info(f"<azul>>> MODO MODELO - UNSLOTH ⚡ TESTANDO INFERÊNCIA ({n_exemplos} exemplo(s))</azul>")
    log_separador(caractere="=", largura=80)

    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=True)
    _exibir_cabecalho_modelo(yaml_config)

    output_dir = yaml_config.modelo.saida
    base_model = yaml_config.modelo.base
    max_seq_length = yaml_config.treinamento.max_seq_length

    # Para inferência com Unsloth, o contexto total (input + output) deve caber
    # no max_seq_length passado a from_pretrained.
    # Estima a partir dos dados reais de tokens (CSVs do curriculum).
    ctx_info = yaml_config.estimar_contexto_predict()
    unsloth_context = ctx_info["contexto"]

    # Decide qual modelo carregar
    if usar_base:
        model_name = base_model
        logger.info(f"<cinza>ℹ️  Usando modelo BASE: {model_name}</cinza>")
    else:
        if not _verificar_modelo_treinado(yaml_config):
            logger.warning("<amarelo>\n⚠️  Não foi encontrado modelo LoRA treinado na pasta de saída.</amarelo>")
            if not _perguntar_confirmacao("Deseja continuar com o modelo base?", padrao=False):
                logger.info("Operação cancelada.")
                return
            usar_base = True
            model_name = base_model
            logger.info("Continuando com modelo base (sem fine-tuning)...\n")
        else:
            model_name = output_dir
            logger.info(f"<verde>✅ Modelo treinado: {model_name}</verde>")

    # ---- Carrega exemplos do dataset ----
    logger.info("<azul>\n📂 Carregando exemplos do dataset de treino...</azul>")

    try:
        mensagens = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo="treino")
        if not mensagens:
            logger.error("<vermelho>❌ Nenhum dado de treino encontrado.</vermelho>")
            return
    except Exception as e:
        logger.error(f"<vermelho>❌ Erro ao carregar dados de treino: {e}</vermelho>")
        return

    n_exemplos = min(n_exemplos, len(mensagens))
    logger.info(f"<cinza>   📊 {len(mensagens)} registros disponíveis, testando {n_exemplos}</cinza>")

    # ---- Carrega modelo com unsloth ----
    logger.info("<azul>\n⚡ Carregando modelo com unsloth...</azul>")
    logger.info(f"<cinza>   Contexto Unsloth: {unsloth_context} ({ctx_info['fonte']})</cinza>")
    ini_carga = time.time()
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=unsloth_context,
            dtype=None,
            load_in_4bit=yaml_config.treinamento.nbits == 4,
        )
        # Ativa modo de inferência otimizado (2x mais rápido)
        FastLanguageModel.for_inference(model)
        logger.info(f"<verde>   ✅ Modelo carregado com unsloth em {time.time() - ini_carga:.1f}s</verde>")
        logger.info(f"<cinza>   ⚡ FastLanguageModel.for_inference() ativado</cinza>")
    except Exception as e:
        logger.error(f"<vermelho>❌ Erro ao carregar modelo com unsloth: {e}</vermelho>")
        return

    # Configura pad_token se necessário
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Processa exemplos ----
    temperatura = 0.01
    resultados = []

    for i in range(n_exemplos):
        log_separador(caractere="-", largura=60)
        logger.info(f">> EXEMPLO {i+1}/{n_exemplos}")
        log_separador(caractere="-", largura=60)

        msg = mensagens[i]
        if not isinstance(msg, dict) or 'messages' not in msg:
            logger.warning(f"   ⚠️ Formato não reconhecido no registro {i}")
            continue

        messages = msg['messages']
        prompt_texto = ""
        resposta_esperada = ""
        for m in messages:
            if m.get('role') == 'user':
                prompt_texto = m.get('content', '')
            elif m.get('role') == 'assistant':
                resposta_esperada = m.get('content', '')

        if not prompt_texto:
            logger.warning(f"   ⚠️ Prompt vazio no registro {i}")
            continue

        # Exibe prompt
        logger.info(f">> PROMPT:")
        if len(prompt_texto) > 500:
            logger.info(f"   {prompt_texto[:250]} [...] {prompt_texto[-250:]}")
        else:
            logger.info(f"   {prompt_texto}")

        # Exibe resposta esperada
        logger.info(f"\n>> RESPOSTA ESPERADA:")
        if len(resposta_esperada) > 500:
            logger.info(f"   {resposta_esperada[:250]} [...] {resposta_esperada[-250:]}")
        else:
            logger.info(f"   {resposta_esperada}")

        # Gera predição com unsloth
        try:
            # Tokeniza o prompt usando chat template
            chat_msgs = [{"role": "user", "content": prompt_texto}]
            inputs = tokenizer.apply_chat_template(
                chat_msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            # Trunca se exceder contexto (reservando max_seq_length para geração)
            max_input_len = unsloth_context - max_seq_length  # = max_seq_length
            if inputs.shape[1] > max_input_len:
                logger.warning(f"   ⚠️ Prompt truncado: {inputs.shape[1]} → {max_input_len} tokens")
                inputs = inputs[:, :max_input_len]

            input_length = inputs.shape[1]
            attention_mask = torch.ones_like(inputs)

            # max_new_tokens = espaço restante no contexto Unsloth (input + output ≤ unsloth_context)
            max_new_tokens = max(256, unsloth_context - input_length)

            tempo_inicio = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperatura, 0.01),
                    top_k=20 if temperatura > 0.3 else 2,
                    do_sample=bool(temperatura > 0.3),
                )
            tempo_pred = time.time() - tempo_inicio

            # Decode apenas a resposta (exclui tokens de entrada)
            resposta_modelo = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            output_tokens = len(outputs[0]) - input_length

            logger.info(f"\n>> RESPOSTA DO MODELO (unsloth ⚡):")
            if len(resposta_modelo) > 500:
                logger.info(f"   {resposta_modelo[:250]} [...] {resposta_modelo[-250:]}")
            else:
                logger.info(f"   {resposta_modelo}")

            logger.info(f"\n>> ESTATÍSTICAS:")
            logger.info(f"   - Tokens do prompt: {input_length}")
            logger.info(f"   - Tokens da resposta: {output_tokens}")
            logger.info(f"   - Temperatura: {temperatura}")
            logger.info(f"   - Tempo de predição: {tempo_pred:.2f}s")
            if output_tokens > 0 and tempo_pred > 0:
                logger.info(f"   - Velocidade: {output_tokens / tempo_pred:.1f} tokens/s")

            resultados.append({
                "exemplo": i + 1,
                "prompt_tokens": input_length,
                "output_tokens": output_tokens,
                "tempo_segundos": round(tempo_pred, 2),
            })

        except Exception as e:
            logger.error(f"<vermelho>   ❌ Erro ao gerar predição: {e}</vermelho>")

    # ---- Resumo final ----
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if resultados:
        total_tokens = sum(r["output_tokens"] for r in resultados)
        total_tempo = sum(r["tempo_segundos"] for r in resultados)
        logger.info(f"\n📊 RESUMO UNSLOTH:")
        logger.info(f"   Exemplos processados: {len(resultados)}/{n_exemplos}")
        logger.info(f"   Tokens gerados: {total_tokens}")
        logger.info(f"   Tempo total: {total_tempo:.2f}s")
        if total_tempo > 0:
            logger.info(f"   Throughput médio: {total_tokens / total_tempo:.1f} tokens/s")

    log_separador(caractere="=", largura=80)
    logger.info("<verde>✅ TESTE DE INFERÊNCIA UNSLOTH COMPLETO</verde>")
    log_separador(caractere="=", largura=80)


# ---------------------------------------------------------------------------
# Predições com Ollama (API local)
# ---------------------------------------------------------------------------

def executar_predict_ollama(yaml_path: str, subsets: list = None) -> None:
    """Gera predições usando Ollama via API local.

    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
        subsets: Lista de subsets para processar ('treino', 'validacao', 'teste').
                 Se None, processa todos.
    """
    from util_openai import UtilOllama

    logger.info("\n")
    log_separador(caractere="=", largura=80)
    logger.info("<azul>>> MODO PREDICT - OLLAMA 🦙 (API LOCAL)</azul>")
    log_separador(caractere="=", largura=80)

    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=True)
    yaml_config.validar_max_seq_length()
    _exibir_cabecalho_modelo(yaml_config)

    # Verifica se o modelo Ollama está configurado
    if not hasattr(yaml_config.modelo, 'ollama') or not yaml_config.modelo.ollama:
        logger.error("<vermelho>❌ Chave 'modelo.ollama' não configurada no YAML</vermelho>")
        logger.info("   Configure o nome do modelo Ollama no YAML (ex: modelo.ollama: QwenDireto)")
        return

    modelo_ollama = yaml_config.modelo.ollama
    ollama_url = getattr(yaml_config.modelo, 'ollama_url', None) or None

    # Verifica formato de saída
    formato_json = yaml_config.formato_saida == FORMATO_SAIDA_JSON
    logger.info(f"<cinza>\n📋 Formato de saída: {yaml_config.formato_saida}</cinza>")
    logger.info(f"<cinza>🦙 Modelo Ollama: {modelo_ollama}</cinza>")
    if ollama_url:
        logger.info(f"<cinza>🌐 API URL: {ollama_url}</cinza>")

    # Define subsets a processar (padrão: apenas teste)
    if subsets is None:
        subsets = ['teste']

    logger.info(f"<cinza>\n📋 Subsets a processar: {', '.join(subsets)}</cinza>")

    # Cria diretório de predições
    predict_dir = os.path.join(yaml_config.modelo.saida, "predict_ollama")
    os.makedirs(predict_dir, exist_ok=True)

    max_seq_length = yaml_config.treinamento.max_seq_length

    # Contexto estimado a partir dos dados reais de tokens (CSVs do curriculum)
    ctx_info = yaml_config.estimar_contexto_predict()
    ollama_context = ctx_info["contexto"]
    ollama_max_tokens = ctx_info["max_new_tokens"]
    logger.info(f"<cinza>⚙️  Contexto: num_ctx={ollama_context}, max_tokens={ollama_max_tokens} ({ctx_info['fonte']})</cinza>")

    # Divisão unificada: dicionário {id: {alvo, divisoes, etapas}}
    divisao_dict = yaml_config.dataset_manager.carregar_divisao_completa(yaml_config.curriculum)
    mapa_etapas = _construir_mapa_etapas(divisao_dict)

    # Pré-check: se todas as predições já existem, sai sem conectar ao Ollama
    if _todas_predicoes_exportadas(predict_dir, subsets, divisao_dict, mapa_etapas, formato_json=formato_json):
        return

    # Verifica status do Ollama
    try:
        status = UtilOllama.status(api_url=ollama_url)
        if not status.get('api'):
            logger.error("<vermelho>❌ Ollama API não está disponível</vermelho>")
            logger.info(f"   Verifique se o Ollama está rodando em {ollama_url or 'http://localhost:11434/api'}")
            return
        logger.info(f"<verde>✅ Ollama versão: {status.get('versao', '?')}</verde>")
        if modelo_ollama not in status.get('modelos', []):
            logger.warning(f"<amarelo>⚠️ Modelo '{modelo_ollama}' não encontrado localmente</amarelo>")
            logger.info(f"   Modelos disponíveis: {', '.join(status.get('modelos', []))}")
            if not _perguntar_confirmacao("Deseja continuar mesmo assim?", padrao=False):
                return
    except Exception as e:
        logger.error(f"<vermelho>❌ Erro ao verificar status do Ollama: {e}</vermelho>")
        return

    uso_total = {'input_tokens': 0, 'output_tokens': 0, 'total_registros': 0, 'tempo_total_s': 0}

    try:
        for subset in subsets:
            logger.info(f"<azul>\n📂 Processando subset: {subset}</azul>")
            log_separador(caractere="-", largura=60)

            try:
                mensagens = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo=subset, divisao=divisao_dict)
                if not mensagens:
                    logger.warning(f"<amarelo>   ⚠️ Nenhum dado encontrado para {subset}</amarelo>")
                    continue
                logger.info(f"<cinza>   📊 {len(mensagens)} registros encontrados</cinza>")
            except Exception as e:
                logger.error(f"<vermelho>   ❌ Erro ao carregar {subset}: {e}</vermelho>")
                continue

            # Cria diretório do subset (preserva arquivos existentes para continuação)
            subset_dir = os.path.join(predict_dir, subset)
            os.makedirs(subset_dir, exist_ok=True)

            total = len(mensagens)
            subset_stats = {'input_tokens': 0, 'output_tokens': 0, 'registros_ok': 0, 'registros_erro': 0, 'registros_skip': 0}
            ini_subset = time.time()

            for idx, msg in enumerate(mensagens):
                try:
                    if isinstance(msg, dict) and 'messages' in msg:
                        messages = msg['messages']
                        prompt_texto = ""
                        for m in messages:
                            if m.get('role') == 'user':
                                prompt_texto = m.get('content', '')
                                break
                    else:
                        subset_stats['registros_erro'] += 1
                        continue

                    if not prompt_texto:
                        subset_stats['registros_erro'] += 1
                        continue

                    registro_id = msg.get('id', f'{subset}_{idx:04d}')

                    # Skip se já exportado (permite continuação de exportações incompletas)
                    if _registro_ja_exportado(subset_dir, registro_id, formato_json=formato_json):
                        subset_stats['registros_skip'] += 1
                        _copiar_para_pastas_etapas(predict_dir, subset, registro_id, mapa_etapas)
                        continue

                    # Chama Ollama via UtilOllama
                    tempo_inicio = time.time()
                    resultado = UtilOllama.chat_completion_padronizado(
                        messages=messages,
                        modelo=modelo_ollama,
                        temperature=0.01,
                        max_tokens=ollama_max_tokens,
                        num_ctx=ollama_context,
                        as_json=formato_json,
                        raw=False,
                        timeout=300,
                        api_url=ollama_url,
                    )
                    tempo_pred = time.time() - tempo_inicio

                    # Verifica se houve erro
                    if 'erro' in resultado:
                        logger.error(f"<vermelho>   ❌ Erro no registro {idx}: {resultado['erro']}</vermelho>")
                        subset_stats['registros_erro'] += 1
                        continue

                    resposta_modelo = resultado.get('resposta', '')
                    if isinstance(resposta_modelo, dict) and formato_json:
                        resposta_modelo = json.dumps(resposta_modelo, ensure_ascii=False, indent=2)
                    elif not isinstance(resposta_modelo, str):
                        resposta_modelo = str(resposta_modelo)

                    input_tokens = resultado.get('usage', {}).get('prompt_tokens', 0)
                    output_tokens = resultado.get('usage', {}).get('completion_tokens', 0)

                    output_txt = os.path.join(subset_dir, f"{registro_id}.txt")
                    with open(output_txt, 'w', encoding='utf-8') as f:
                        f.write(resposta_modelo)

                    usage_data = {
                        'id': registro_id,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'time_s': round(tempo_pred, 3),
                        'model': resultado.get('model', modelo_ollama),
                    }
                    output_json = os.path.join(subset_dir, f"{registro_id}.json")
                    with open(output_json, 'w', encoding='utf-8') as f:
                        json.dump(usage_data, f, ensure_ascii=False, indent=2)

                    # Copia para pastas de etapa (curriculum multi-etapa)
                    _copiar_para_pastas_etapas(predict_dir, subset, registro_id, mapa_etapas)

                    subset_stats['input_tokens'] += input_tokens
                    subset_stats['output_tokens'] += output_tokens
                    subset_stats['registros_ok'] += 1

                    if (idx + 1) % 10 == 0 or (idx + 1) == total:
                        logger.info(f"   Progresso: {idx + 1}/{total} ({100*(idx+1)//total}%)")

                except Exception as e:
                    logger.error(f"<vermelho>   ❌ Erro no registro {idx}: {e}</vermelho>")
                    subset_stats['registros_erro'] += 1
                    continue

            tempo_subset = time.time() - ini_subset
            skip_msg = f", {subset_stats['registros_skip']} já exportados" if subset_stats['registros_skip'] else ""
            logger.info(f"<verde>   ✅ {subset_stats['registros_ok']} predições salvas em: {subset_dir}</verde>")
            logger.info(f"<cinza>   📊 Tokens: {subset_stats['input_tokens']} entrada, {subset_stats['output_tokens']} saída ({tempo_subset:.1f}s){skip_msg}</cinza>")
            gerar_estatisticas_predicoes(subset_dir)
            gerar_estatisticas_predicoes_etapas(predict_dir, subset)

            uso_total['input_tokens'] += subset_stats['input_tokens']
            uso_total['output_tokens'] += subset_stats['output_tokens']
            uso_total['total_registros'] += subset_stats['registros_ok']
            uso_total['tempo_total_s'] += tempo_subset

    finally:
        pass

    log_separador(caractere="=", largura=80)
    logger.info(f"<verde>✅ PREDICT OLLAMA COMPLETO - Resultados em: {predict_dir}</verde>")
    logger.info(f"<cinza>📊 Total: {uso_total['total_registros']} registros, {uso_total['input_tokens']} + {uso_total['output_tokens']} tokens ({uso_total['tempo_total_s']:.1f}s)</cinza>")
    log_separador(caractere="=", largura=80)


# ---------------------------------------------------------------------------
# Inferência interativa com Ollama (API local)
# ---------------------------------------------------------------------------

def executar_modelo_ollama(yaml_path: str, n_exemplos: int = 1) -> None:
    """Testa inferência interativa com N exemplos usando Ollama via API local.

    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
        n_exemplos: Número de exemplos para testar
    """
    from util_openai import UtilOllama

    logger.info("\n")
    log_separador(caractere="=", largura=80)
    logger.info(f"<azul>>> MODO MODELO - OLLAMA 🦙 TESTANDO INFERÊNCIA ({n_exemplos} exemplo(s))</azul>")
    log_separador(caractere="=", largura=80)

    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=False)
    _exibir_cabecalho_modelo(yaml_config)

    if not hasattr(yaml_config.modelo, 'ollama') or not yaml_config.modelo.ollama:
        logger.error("<vermelho>❌ Chave 'modelo.ollama' não configurada no YAML</vermelho>")
        return

    modelo_ollama = yaml_config.modelo.ollama
    ollama_url = getattr(yaml_config.modelo, 'ollama_url', None) or None
    max_seq_length = yaml_config.treinamento.max_seq_length
    formato_json = yaml_config.formato_saida == FORMATO_SAIDA_JSON

    # Contexto estimado a partir dos dados reais de tokens (CSVs do curriculum)
    ctx_info = yaml_config.estimar_contexto_predict()
    ollama_context = ctx_info["contexto"]
    ollama_max_tokens = ctx_info["max_new_tokens"]

    logger.info(f"<cinza>🦙 Modelo Ollama: {modelo_ollama}</cinza>")
    if ollama_url:
        logger.info(f"<cinza>🌐 API URL: {ollama_url}</cinza>")
    logger.info(f"<cinza>⚙️  Contexto: num_ctx={ollama_context}, max_tokens={ollama_max_tokens} ({ctx_info['fonte']})</cinza>")

    # Verifica status do Ollama
    try:
        status = UtilOllama.status(api_url=ollama_url)
        if not status.get('api'):
            logger.error("<vermelho>❌ Ollama API não está disponível</vermelho>")
            logger.info(f"   Verifique se o Ollama está rodando em {ollama_url or 'http://localhost:11434/api'}")
            return
        logger.info(f"<verde>✅ Ollama versão: {status.get('versao', '?')}</verde>")
        if modelo_ollama not in status.get('modelos', []):
            logger.warning(f"<amarelo>⚠️ Modelo '{modelo_ollama}' não encontrado localmente</amarelo>")
            logger.info(f"   Modelos disponíveis: {', '.join(status.get('modelos', []))}")
            if not _perguntar_confirmacao("Deseja continuar mesmo assim?", padrao=False):
                return
    except Exception as e:
        logger.error(f"<vermelho>❌ Erro ao verificar status do Ollama: {e}</vermelho>")
        return

    # Carrega exemplos do dataset de treino
    logger.info("<azul>\n📂 Carregando exemplos do dataset de treino...</azul>")
    try:
        mensagens = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo="treino")
        if not mensagens:
            logger.error("<vermelho>❌ Nenhum dado de treino encontrado.</vermelho>")
            return
    except Exception as e:
        logger.error(f"<vermelho>❌ Erro ao carregar dados de treino: {e}</vermelho>")
        return

    n_exemplos = min(n_exemplos, len(mensagens))
    logger.info(f"<cinza>   📊 {len(mensagens)} registros disponíveis, testando {n_exemplos}</cinza>")

    for idx in range(n_exemplos):
        log_separador(caractere="-", largura=60)
        msg = mensagens[idx]
        logger.info(f"<azul>📌 Exemplo {idx + 1}/{n_exemplos}</azul>")

        if not (isinstance(msg, dict) and 'messages' in msg):
            logger.error(f"<vermelho>   ❌ Formato de mensagem inválido no exemplo {idx}</vermelho>")
            continue

        messages = msg['messages']

        for m in messages:
            if m.get('role') == 'user':
                conteudo = m.get('content', '')
                preview = conteudo[:300] + ('...' if len(conteudo) > 300 else '')
                logger.info(f"<cinza>📥 Prompt (user):\n{preview}</cinza>")
                break

        for m in messages:
            if m.get('role') == 'assistant':
                conteudo = m.get('content', '')
                preview = conteudo[:300] + ('...' if len(conteudo) > 300 else '')
                logger.info(f"<cinza>🎯 Esperado:\n{preview}</cinza>")
                break

        logger.info(f"<azul>🚀 Gerando resposta via Ollama ({modelo_ollama})...</azul>")
        tempo_inicio = time.time()
        try:
            resultado = UtilOllama.chat_completion_padronizado(
                messages=messages,
                modelo=modelo_ollama,
                temperature=0.01,
                max_tokens=ollama_max_tokens,
                num_ctx=ollama_context,
                as_json=formato_json,
                raw=False,
                timeout=UtilEnv.get_int('OLLAMA_TIMEOUT', 600),
                api_url=ollama_url,
            )
        except Exception as e:
            logger.error(f"<vermelho>   ❌ Erro na chamada Ollama: {e}</vermelho>")
            continue

        tempo_pred = time.time() - tempo_inicio

        if 'erro' in resultado:
            logger.error(f"<vermelho>   ❌ Erro Ollama: {resultado['erro']}</vermelho>")
            continue

        resposta = resultado.get('resposta', '')
        if isinstance(resposta, dict):
            resposta = json.dumps(resposta, ensure_ascii=False, indent=2)
        elif not isinstance(resposta, str):
            resposta = str(resposta)

        input_tokens  = resultado.get('usage', {}).get('prompt_tokens', 0)
        output_tokens = resultado.get('usage', {}).get('completion_tokens', 0)

        preview_resp = resposta[:500] + ('...' if len(resposta) > 500 else '')
        logger.info(f"<verde>📤 Resposta Ollama:\n{preview_resp}</verde>")
        logger.info(f"<cinza>   ⏱️  {tempo_pred:.2f}s | tokens: {input_tokens} entrada + {output_tokens} saída</cinza>")

    log_separador(caractere="=", largura=80)
    logger.info("<verde>✅ TESTE DE INFERÊNCIA OLLAMA COMPLETO</verde>")
    log_separador(caractere="=", largura=80)


# ---------------------------------------------------------------------------
# Predições com Unsloth (FastLanguageModel.for_inference)
# ---------------------------------------------------------------------------

def executar_predict_unsloth(yaml_path: str, subsets: list = None, usar_base: bool = False) -> None:
    """Gera predições usando unsloth FastLanguageModel.for_inference().

    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
        subsets: Lista de subsets para processar ('treino', 'validacao', 'teste').
                 Se None, processa todos.
        usar_base: Se True, usa o modelo base original.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("❌ Módulo unsloth não encontrado!")
        logger.info("   Instale com: pip install unsloth")
        return

    # Garante que triton encontre um compilador C
    import shutil as _shutil
    if not os.environ.get('CC'):
        cc_path = _shutil.which('gcc') or _shutil.which('cc')
        if cc_path:
            os.environ['CC'] = cc_path

    logger.info("\n")
    log_separador(caractere="=", largura=80)
    logger.info("<azul>>> MODO PREDICT - UNSLOTH ⚡ (INFERÊNCIA RÁPIDA)</azul>")
    log_separador(caractere="=", largura=80)

    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=True)
    yaml_config.validar_max_seq_length()
    _exibir_cabecalho_modelo(yaml_config)

    output_dir = yaml_config.modelo.saida
    base_model = yaml_config.modelo.base
    max_seq_length = yaml_config.treinamento.max_seq_length

    # Para inferência com Unsloth, o contexto total (input + output) deve caber
    # no max_seq_length passado a from_pretrained.
    # Estima a partir dos dados reais de tokens (CSVs do curriculum).
    ctx_info = yaml_config.estimar_contexto_predict()
    unsloth_context = ctx_info["contexto"]
    unsloth_max_new_tokens = ctx_info["max_new_tokens"]

    # Verifica formato de saída
    formato_json = yaml_config.formato_saida == FORMATO_SAIDA_JSON
    logger.info(f"<cinza>\n📋 Formato de saída: {yaml_config.formato_saida}</cinza>")

    # Decide qual modelo carregar
    tem_modelo_treinado = _verificar_modelo_treinado(yaml_config)

    if usar_base:
        model_name = base_model
        logger.info(f"<cinza>ℹ️  Opção --base ativada: Forçando uso do modelo base.</cinza>")
    elif not tem_modelo_treinado:
        logger.warning("<amarelo>\n⚠️ Não foi encontrado modelo LoRA treinado.</amarelo>")
        if not _perguntar_confirmacao("Deseja usar o modelo base para predição?", padrao=False):
            logger.info("Operação cancelada.")
            return
        usar_base = True
        model_name = base_model
    else:
        model_name = output_dir
        logger.info(f"<verde>✅ Usando modelo treinado: {model_name}</verde>")

    # Define subsets a processar (padrão: apenas teste)
    if subsets is None:
        subsets = ['teste']

    logger.info(f"<cinza>\n📋 Subsets a processar: {', '.join(subsets)}</cinza>")

    # Cria diretório de predições
    nome_pasta = "predict_base_unsloth" if usar_base else "predict_unsloth"
    predict_dir = os.path.join(output_dir, nome_pasta)
    os.makedirs(predict_dir, exist_ok=True)

    # Divisão unificada: dicionário {id: {alvo, divisoes, etapas}}
    divisao_dict = yaml_config.dataset_manager.carregar_divisao_completa(yaml_config.curriculum)
    mapa_etapas = _construir_mapa_etapas(divisao_dict)

    # Pré-check: se todas as predições já existem, sai sem carregar o modelo
    if _todas_predicoes_exportadas(predict_dir, subsets, divisao_dict, mapa_etapas, formato_json=formato_json):
        return

    # Carrega modelo com unsloth
    logger.info("<azul>\n⚡ Carregando modelo com unsloth...</azul>")
    logger.info(f"<cinza>   Contexto Unsloth: {unsloth_context} ({ctx_info['fonte']})</cinza>")
    ini_carga = time.time()
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=unsloth_context,
            dtype=None,
            load_in_4bit=yaml_config.treinamento.nbits == 4,
        )
        FastLanguageModel.for_inference(model)
        logger.info(f"<verde>   ✅ Modelo carregado com unsloth em {time.time() - ini_carga:.1f}s</verde>")
    except Exception as e:
        logger.error(f"<vermelho>❌ Erro ao carregar modelo com unsloth: {e}</vermelho>")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    uso_total = {'input_tokens': 0, 'output_tokens': 0, 'total_registros': 0, 'tempo_total_s': 0}

    try:
        for subset in subsets:
            logger.info(f"<azul>\n📂 Processando subset: {subset}</azul>")
            log_separador(caractere="-", largura=60)

            try:
                mensagens = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo=subset, divisao=divisao_dict)
                if not mensagens:
                    logger.warning(f"<amarelo>   ⚠️ Nenhum dado encontrado para {subset}</amarelo>")
                    continue
                logger.info(f"<cinza>   📊 {len(mensagens)} registros encontrados</cinza>")
            except Exception as e:
                logger.error(f"<vermelho>   ❌ Erro ao carregar {subset}: {e}</vermelho>")
                continue

            # Cria diretório do subset (preserva arquivos existentes para continuação)
            subset_dir = os.path.join(predict_dir, subset)
            os.makedirs(subset_dir, exist_ok=True)

            total = len(mensagens)
            subset_stats = {'input_tokens': 0, 'output_tokens': 0, 'registros_ok': 0, 'registros_erro': 0, 'registros_skip': 0}
            ini_subset = time.time()

            for idx, msg in enumerate(mensagens):
                try:
                    if isinstance(msg, dict) and 'messages' in msg:
                        messages = msg['messages']
                        prompt_texto = ""
                        for m in messages:
                            if m.get('role') == 'user':
                                prompt_texto = m.get('content', '')
                                break
                    else:
                        subset_stats['registros_erro'] += 1
                        continue

                    if not prompt_texto:
                        subset_stats['registros_erro'] += 1
                        continue

                    registro_id = msg.get('id', f'{subset}_{idx:04d}')

                    # Skip se já exportado (permite continuação de exportações incompletas)
                    if _registro_ja_exportado(subset_dir, registro_id, formato_json=formato_json):
                        subset_stats['registros_skip'] += 1
                        _copiar_para_pastas_etapas(predict_dir, subset, registro_id, mapa_etapas)
                        continue

                    # Tokeniza usando chat template
                    chat_msgs = [{"role": "user", "content": prompt_texto}]
                    inputs = tokenizer.apply_chat_template(
                        chat_msgs,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(model.device)

                    # Trunca se exceder contexto (reservando max_new_tokens estimados para geração)
                    max_input_len = max(128, unsloth_context - unsloth_max_new_tokens)
                    if inputs.shape[1] > max_input_len:
                        inputs = inputs[:, :max_input_len]

                    input_length = inputs.shape[1]
                    attention_mask = torch.ones_like(inputs)

                    # max_new_tokens = espaço restante no contexto Unsloth (input + output ≤ unsloth_context)
                    max_new_tokens = max(256, unsloth_context - input_length)

                    tempo_inicio = time.time()
                    with torch.inference_mode():
                        outputs = model.generate(
                            input_ids=inputs,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            temperature=0.01,
                            top_k=2,
                            do_sample=False,
                        )
                    tempo_pred = time.time() - tempo_inicio

                    resposta_modelo = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                    output_tokens = len(outputs[0]) - input_length

                    # Se formato JSON, tenta parsear
                    if formato_json and resposta_modelo.strip():
                        try:
                            from util import UtilTextos
                            json_obj = UtilTextos.mensagem_to_json(resposta_modelo)
                            resposta_modelo = json.dumps(json_obj, ensure_ascii=False, indent=2)
                        except Exception:
                            pass

                    output_txt = os.path.join(subset_dir, f"{registro_id}.txt")
                    with open(output_txt, 'w', encoding='utf-8') as f:
                        f.write(resposta_modelo)

                    usage_data = {
                        'id': registro_id,
                        'input_tokens': input_length,
                        'output_tokens': output_tokens,
                        'time_s': round(tempo_pred, 3),
                    }
                    output_json = os.path.join(subset_dir, f"{registro_id}.json")
                    with open(output_json, 'w', encoding='utf-8') as f:
                        json.dump(usage_data, f, ensure_ascii=False, indent=2)

                    # Copia para pastas de etapa (curriculum multi-etapa)
                    _copiar_para_pastas_etapas(predict_dir, subset, registro_id, mapa_etapas)

                    subset_stats['input_tokens'] += input_length
                    subset_stats['output_tokens'] += output_tokens
                    subset_stats['registros_ok'] += 1

                    if (idx + 1) % 10 == 0 or (idx + 1) == total:
                        logger.info(f"   Progresso: {idx + 1}/{total} ({100*(idx+1)//total}%)")

                except Exception as e:
                    logger.error(f"<vermelho>   ❌ Erro no registro {idx}: {e}</vermelho>")
                    subset_stats['registros_erro'] += 1
                    continue

            tempo_subset = time.time() - ini_subset
            skip_msg = f", {subset_stats['registros_skip']} já exportados" if subset_stats['registros_skip'] else ""
            logger.info(f"<verde>   ✅ {subset_stats['registros_ok']} predições salvas em: {subset_dir}</verde>")
            logger.info(f"<cinza>   📊 Tokens: {subset_stats['input_tokens']} entrada, {subset_stats['output_tokens']} saída ({tempo_subset:.1f}s){skip_msg}</cinza>")
            gerar_estatisticas_predicoes(subset_dir)
            gerar_estatisticas_predicoes_etapas(predict_dir, subset)

            uso_total['input_tokens'] += subset_stats['input_tokens']
            uso_total['output_tokens'] += subset_stats['output_tokens']
            uso_total['total_registros'] += subset_stats['registros_ok']
            uso_total['tempo_total_s'] += tempo_subset

    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_separador(caractere="=", largura=80)
    logger.info(f"<verde>✅ PREDICT UNSLOTH COMPLETO - Resultados em: {predict_dir}</verde>")
    logger.info(f"<cinza>📊 Total: {uso_total['total_registros']} registros, {uso_total['input_tokens']} + {uso_total['output_tokens']} tokens ({uso_total['tempo_total_s']:.1f}s)</cinza>")
    log_separador(caractere="=", largura=80)
