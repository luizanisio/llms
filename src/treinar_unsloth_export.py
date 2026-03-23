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

import csv
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


def _registro_ja_exportado(subset_dir: str, registro_id: str, min_bytes: int = 10) -> bool:
    """Verifica se um registro já foi exportado com sucesso (txt + json válidos).
    
    Um registro é considerado exportado se AMBOS os arquivos existem
    e têm tamanho >= min_bytes (evita considerar arquivos vazios/corrompidos).
    
    Args:
        subset_dir: Diretório do subset (ex: predict/teste/)
        registro_id: ID do registro (ex: 'teste_0001')
        min_bytes: Tamanho mínimo em bytes para considerar válido (padrão: 10)
    
    Returns:
        True se ambos .txt e .json existem com tamanho válido.
    """
    txt_path = os.path.join(subset_dir, f"{registro_id}.txt")
    json_path = os.path.join(subset_dir, f"{registro_id}.json")
    
    try:
        if not os.path.exists(txt_path) or not os.path.exists(json_path):
            return False
        return os.path.getsize(txt_path) >= min_bytes and os.path.getsize(json_path) >= min_bytes
    except OSError:
        return False


def _construir_mapa_etapas(yaml_config) -> dict:
    """Constrói mapa de etapas do curriculum para cópia de predições.

    Para curriculum multi-etapa, lê os CSVs de divisão de cada etapa e
    monta um dicionário ``{(id_arquivo, alvo): [alias1, alias2, ...]}``.
    Se o curriculum tiver apenas uma etapa, retorna ``None`` (sem cópias).

    Args:
        yaml_config: Instância de YamlTreinamento.

    Returns:
        Dicionário de mapeamento ou None se etapa única.
    """
    etapas = yaml_config.curriculum
    if len(etapas) <= 1:
        return None

    mapa = {}  # {(id_arquivo, alvo): [alias, ...]}

    for etapa in etapas:
        if not etapa.arquivo or not os.path.isfile(etapa.arquivo):
            continue

        try:
            with open(etapa.arquivo, 'r', encoding='utf-8-sig') as f:
                amostra = f.read(4096)
                f.seek(0)
                try:
                    dialeto = csv.Sniffer().sniff(amostra)
                except csv.Error:
                    dialeto = 'excel'
                reader = csv.DictReader(f, dialect=dialeto)

                # Detecta colunas (com tolerância a nomes antigos)
                campos = {c.strip(): c for c in (reader.fieldnames or [])}
                col_id = None
                col_alvo = None
                for nome_limpo, nome_original in campos.items():
                    if nome_limpo in ('id_arquivo', 'id') and col_id is None:
                        col_id = nome_original
                    if nome_limpo in ('alvo', 'divisão', 'divisao', 'grupo') and col_alvo is None:
                        col_alvo = nome_original

                if not col_id or not col_alvo:
                    logger.warning(f"⚠️  CSV de etapa '{etapa.alias}' sem colunas id/alvo: {etapa.arquivo}")
                    continue

                for row in reader:
                    id_val = str(row.get(col_id, '')).strip()
                    alvo_val = str(row.get(col_alvo, '')).strip()

                    # Normaliza alvos antigos
                    if alvo_val in ('avaliacao', 'avaliação', 'eval'):
                        alvo_val = 'validacao'

                    if id_val and alvo_val:
                        chave = (id_val, alvo_val)
                        if chave not in mapa:
                            mapa[chave] = []
                        if etapa.alias not in mapa[chave]:
                            mapa[chave].append(etapa.alias)

        except Exception as e:
            logger.warning(f"⚠️  Erro ao ler CSV da etapa '{etapa.alias}': {e}")
            continue

    if not mapa:
        return None

    aliases = [et.alias for et in etapas]
    logger.info(f"<cinza>📋 Curriculum multi-etapa ({' → '.join(aliases)}): cópias por etapa habilitadas</cinza>")
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
    
    # Mapa de etapas para cópia em pastas por divisão (curriculum multi-etapa)
    mapa_etapas = _construir_mapa_etapas(yaml_config)
    
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
                mensagens = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo=subset)
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
                    if _registro_ja_exportado(subset_dir, registro_id):
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
    config = get_recommended_config(num_gpus=num_gpus, model_size="7B")
    config.max_model_len = max_seq_length
    logger.info(f"⚙️  Tensor Parallel: {config.tensor_parallel_size} GPU(s)")
    logger.info(f"⚙️  GPU Memory Utilization: {config.gpu_memory_utilization*100:.0f}%\n")

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

    # Define subsets a processar
    if subsets is None:
        subsets = ['treino', 'validacao', 'teste']

    logger.info(f"<cinza>\n📋 Subsets a processar: {', '.join(subsets)}</cinza>")

    # Cria diretório de predições
    nome_pasta = "predict_base_vllm" if usar_base else "predict_vllm"
    predict_dir = os.path.join(output_dir, nome_pasta)
    os.makedirs(predict_dir, exist_ok=True)

    # Mapa de etapas para cópia em pastas por divisão (curriculum multi-etapa)
    mapa_etapas = _construir_mapa_etapas(yaml_config)

    uso_total = {'input_tokens': 0, 'output_tokens': 0, 'total_registros': 0, 'tempo_total_s': 0}

    try:
        for subset in subsets:
            logger.info(f"<azul>\n📂 Processando subset: {subset}</azul>")
            log_separador(caractere="-", largura=60)

            try:
                mensagens = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo=subset)
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
                    if _registro_ja_exportado(subset_dir, registro_id):
                        subset_stats['registros_skip'] += 1
                        _copiar_para_pastas_etapas(predict_dir, subset, registro_id, mapa_etapas)
                        continue

                    # Estruturar prompt simples (estilo chat_template já feito na base caso necessário)
                    prompt_ids = vllm_tokenizer.encode(prompt_texto)
                    if len(prompt_ids) > max_input_len:
                        prompt_ids = prompt_ids[:max_input_len]
                        prompt_texto = vllm_tokenizer.decode(prompt_ids, skip_special_tokens=True)

                    registro_id = msg.get('id', f'{subset}_{idx:04d}')
                    prompts_batch.append(prompt_texto)
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
                tokens_para_gerar = min(max_seq_length, config.max_model_len - max_len_batch)

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
    config.max_model_len = yaml_config.treinamento.max_seq_length
    logger.info(f"⚙️  Tensor Parallel: {config.tensor_parallel_size} GPU(s)")
    logger.info(f"⚙️  GPU Memory Utilization: {config.gpu_memory_utilization*100:.0f}%")
    logger.info(f"⚙️  Max Model Len: {config.max_model_len}")

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
    max_new_tokens = yaml_config.treinamento.max_seq_length
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
            # Trunca prompt se exceder max_seq_length (reserva 256 tokens para resposta)
            max_input_len = config.max_model_len - 256
            vllm_tokenizer = engine.llm.get_tokenizer()
            prompt_ids = vllm_tokenizer.encode(prompt_texto)
            if len(prompt_ids) > max_input_len:
                logger.warning(f"   ⚠️ Prompt truncado: {len(prompt_ids)} → {max_input_len} tokens")
                prompt_ids = prompt_ids[:max_input_len]
                prompt_texto = vllm_tokenizer.decode(prompt_ids, skip_special_tokens=True)

            # vLLM restringe: prompt_len + max_tokens <= max_model_len
            tokens_para_gerar = min(max_new_tokens, config.max_model_len - len(prompt_ids))

            tempo_inicio = time.time()
            resultado = engine.generate_batch(
                prompts=[prompt_texto],
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
    # no max_seq_length passado a from_pretrained. Usamos 2× para acomodar
    # prompts longos + respostas completas (Unsloth faz RoPE scaling interno).
    unsloth_context = max_seq_length * 2

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
    logger.info(f"<cinza>   Contexto Unsloth: {unsloth_context} (2× max_seq_length para acomodar input+output)</cinza>")
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
    # no max_seq_length passado a from_pretrained. Usamos 2× para acomodar
    # prompts longos + respostas completas (Unsloth faz RoPE scaling interno).
    unsloth_context = max_seq_length * 2

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

    # Carrega modelo com unsloth
    logger.info("<azul>\n⚡ Carregando modelo com unsloth...</azul>")
    logger.info(f"<cinza>   Contexto Unsloth: {unsloth_context} (2× max_seq_length para acomodar input+output)</cinza>")
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

    # Define subsets a processar (padrão: apenas teste)
    if subsets is None:
        subsets = ['teste']

    logger.info(f"<cinza>\n📋 Subsets a processar: {', '.join(subsets)}</cinza>")

    # Cria diretório de predições
    nome_pasta = "predict_base_unsloth" if usar_base else "predict_unsloth"
    predict_dir = os.path.join(output_dir, nome_pasta)
    os.makedirs(predict_dir, exist_ok=True)

    # Mapa de etapas para cópia em pastas por divisão (curriculum multi-etapa)
    mapa_etapas = _construir_mapa_etapas(yaml_config)

    uso_total = {'input_tokens': 0, 'output_tokens': 0, 'total_registros': 0, 'tempo_total_s': 0}

    try:
        for subset in subsets:
            logger.info(f"<azul>\n📂 Processando subset: {subset}</azul>")
            log_separador(caractere="-", largura=60)

            try:
                mensagens = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo=subset)
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
                    if _registro_ja_exportado(subset_dir, registro_id):
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

                    # Trunca se exceder contexto (reservando max_seq_length para geração)
                    max_input_len = unsloth_context - max_seq_length  # = max_seq_length
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
