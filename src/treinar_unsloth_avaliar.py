#!/usr/bin/env python3

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Script de Avaliação, Inferência e Exportação de modelos LLM treinados.

Uso:
    python treinar_unsloth_avaliar.py [CONFIG.yaml] [AÇÃO] [OPÇÕES]

* Se CONFIG.yaml for omitido, exibe menu interativo para seleção do arquivo.
* Se nenhuma ação for informada, exibe menu interativo de ações.

Ações:
    --info              Informações da configuração, datasets e modelo
    --stats             Relatório estatístico (tokens, loss, hardware) com gráficos
    --predict           Predições em massa para todos os subsets (treino, validação, teste)
    --predict-treino    Predições apenas do subset de treino
    --predict-validacao Predições apenas do subset de validação
    --predict-teste     Predições apenas do subset de teste
    --modelo N          Inferência interativa com N exemplos (default: 1)
    --merge             Exportação: merge LoRA + Base

Opções:
    --base              Força uso do modelo base (ignora LoRA treinado)
    --quant METODO      Quantização para merge (16bit, 4bit, q4_k_m, q8_0, f16)
    --log-level LEVEL   Nível de log (DEBUG, INFO, WARNING, ERROR)
"""

import argparse
import os
import sys
import json
import shutil
import gc
import time
import statistics
from typing import Optional
from datetime import datetime

import torch

import util  # garante que a pasta src está no sys.path
from util import UtilEnv, Util
from treinar_unsloth_logging import get_logger, configurar_logging, log_separador, log_bloco
from treinar_unsloth_util import (
    YamlTreinamento, TIPO_ENTRADA_PASTAS, TIPO_ENTRADA_DATASET, TIPOS_BASEADOS_EM_PASTAS, FORMATO_SAIDA_JSON
)
from treinar_unsloth_actions import (
    _exibir_cabecalho_modelo,
    _verificar_modelo_treinado,
    _verificar_checkpoints_existem,
    _perguntar_confirmacao,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Ações de Avaliação
# ---------------------------------------------------------------------------

def executar_info(yaml_path: str) -> None:
    """
    Exibe informações detalhadas sobre configuração, datasets e modelo.
    Realiza verificação e, se necessário, recálculo de max_seq_length.
    Salva o resultado como info.md na pasta de treinamento.
    
    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
    """
    import io
    import contextlib
    from treinar_unsloth import LLMsTrainer
    
    # Carrega config com validação de caminhos para permitir resolver max_seq_length
    try:
        yaml_config = YamlTreinamento(yaml_path, validar_caminhos=True)
    except Exception:
        # Fallback sem validação de caminhos (ex: pastas inexistentes)
        yaml_config = YamlTreinamento(yaml_path, validar_caminhos=False)
    
    # Verifica/recalcula max_seq_length antes de exibir info
    try:
        yaml_config.resolver_max_seq_length()
    except Exception as e:
        logger.warning(f"⚠️ Não foi possível resolver max_seq_length: {e}")
    
    # Captura stdout para gravar em arquivo (inclui saída do debug_info)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        LLMsTrainer.debug_info(yaml_path, yaml_config=yaml_config)
    
    conteudo = buffer.getvalue()
    
    # Exibe no console normalmente
    print(conteudo, end="")
    
    # Salva info.md na pasta de treinamento
    try:
        report_dir = os.path.join(yaml_config.modelo.saida, "treinamento")
        os.makedirs(report_dir, exist_ok=True)
        info_path = os.path.join(report_dir, "info.md")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"# Info - {os.path.basename(yaml_path)}\n\n")
            f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("```\n")
            f.write(conteudo)
            f.write("```\n")
        logger.info(f"\n📝 Info salvo em: {info_path}")
    except Exception as e:
        logger.warning(f"⚠️ Não foi possível salvar info.md: {e}")


def executar_stats(yaml_path: str) -> None:
    """
    Gera relatório estatístico com informações sobre uso de tokens.
    Gera tabelas e gráficos separados por subset (treino, validação, teste).
    
    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
    """
    from treinar_unsloth_report import GeradorRelatorio
    
    logger.info("\n")
    log_separador(caractere="=", largura=80)
    logger.info(">> MODO STATS - RELATÓRIO ESTATÍSTICO DE TOKENS")
    log_separador(caractere="=", largura=80)
    
    # Carrega configuração
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=True)
    _exibir_cabecalho_modelo(yaml_config)
    
    # Cria diretório de saída para relatório
    report_dir = os.path.join(yaml_config.modelo.saida, "treinamento")
    os.makedirs(report_dir, exist_ok=True)
    
    # Carrega dados para estatísticas
    logger.info("\n📊 Carregando dados para estatísticas...")
    
    # Estrutura para armazenar dados por subset
    stats_por_subset = {}
    
    if yaml_config.tipo_entrada in TIPOS_BASEADOS_EM_PASTAS:
        # Modo pastas/curriculum
        for alvo, nome in [("treino", "Treino"), ("validacao", "Validação"), ("teste", "Teste")]:
            try:
                mensagens = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo=alvo)
                if mensagens:
                    stats_por_subset[alvo] = {
                        'nome': nome,
                        'registros': len(mensagens),
                        'entrada': [],
                        'saida': []
                    }
                    logger.info(f"   {nome}: {len(mensagens)} registros")
                    
                    for msg in mensagens:
                        if isinstance(msg, dict) and 'messages' in msg:
                            for m in msg['messages']:
                                texto = m.get('content', '')
                                tokens = len(texto.split())
                                if m.get('role') == 'user':
                                    stats_por_subset[alvo]['entrada'].append(tokens)
                                elif m.get('role') == 'assistant':
                                    stats_por_subset[alvo]['saida'].append(tokens)
            except Exception as e:
                logger.warning(f"   ⚠️ Erro ao carregar {alvo}: {e}")
    else:
        logger.info("   Modo dataset: use --info para ver estatísticas do dataset")
        return
    
    if not stats_por_subset:
        logger.warning("   Nenhum dado encontrado para gerar estatísticas.")
        return
    
    def _tabela_stats(lista, titulo):
        if not lista:
            return f"\n#### {titulo}\nNenhum dado disponível.\n"
        
        stdev_val = statistics.stdev(lista) if len(lista) > 1 else 0
        return f"""
#### {titulo}

| Métrica | Valor |
|---------|-------|
| Mínimo | {min(lista)} |
| Máximo | {max(lista)} |
| Média | {statistics.mean(lista):.1f} |
| Mediana | {statistics.median(lista):.1f} |
| Desvio Padrão | {stdev_val:.1f} |
| Total Tokens | {sum(lista)} |
"""

    # Gera relatório
    stats_report = []
    stats_report.append("# Relatório Estatístico de Tokens por Subset\n")
    stats_report.append(f"**Modelo Base:** `{yaml_config.modelo.base}`\n")
    stats_report.append(f"**Modelo Saída:** `{yaml_config.modelo.saida}`\n")
    stats_report.append(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    stats_report.append("\n## Resumo de Registros\n")
    stats_report.append("| Subset | Registros |")
    stats_report.append("|--------|-----------|")
    for alvo, dados in stats_por_subset.items():
        stats_report.append(f"| {dados['nome']} | {dados['registros']} |")
    
    logger.info("\n📈 Gerando relatórios e preparando gráficos...")
    
    dados_grafico = {}
    subsets_ordem = ['treino', 'validacao', 'teste']
    
    for alvo in subsets_ordem:
        if alvo in stats_por_subset and stats_por_subset[alvo]['entrada']:
            nome = stats_por_subset[alvo]['nome']
            dados_grafico[f"{nome} (In)"] = stats_por_subset[alvo]['entrada']
            
    for alvo in subsets_ordem:
        if alvo in stats_por_subset and stats_por_subset[alvo]['saida']:
            nome = stats_por_subset[alvo]['nome']
            dados_grafico[f"{nome} (Out)"] = stats_por_subset[alvo]['saida']

    for alvo, dados in stats_por_subset.items():
        nome_subset = dados['nome']
        entrada = dados['entrada']
        saida = dados['saida']
        
        titulo_secao = f"\n## Estatísticas: {nome_subset}\n"
        stats_report.append(titulo_secao)
        
        stats_report.append(_tabela_stats(entrada, "Tokens de Entrada (User)"))
        stats_report.append(_tabela_stats(saida, "Tokens de Saída (Assistant)"))

    # Gera Gráfico de Tokens
    if dados_grafico:
        from treinar_unsloth_graficos import GraficoTokens
        
        nome_arquivo_grafico = "stats_tokens_boxplot.png"
        boxplot_path = os.path.join(report_dir, nome_arquivo_grafico)
        
        if GraficoTokens.boxplot_comparativo(dados_grafico, boxplot_path):
            logger.info(f"   ✅ Gráfico consolidado salvo: {nome_arquivo_grafico}")
            stats_report.append(f"\n## Gráfico Comparativo\n")
            stats_report.append(f"![Boxplot Comparativo]({nome_arquivo_grafico})\n")
        else:
            logger.warning("   ⚠️ Erro ao gerar gráfico de tokens.")

    # Métricas de treinamento (loss)
    from treinar_unsloth_graficos import GraficoTreinamento
    
    chkpt_dir = os.path.join(yaml_config.modelo.saida, "chkpt")
    if not os.path.exists(chkpt_dir) or not any(
        d.startswith("checkpoint-") for d in os.listdir(chkpt_dir) 
        if os.path.isdir(os.path.join(chkpt_dir, d))
    ):
        chkpt_dir = yaml_config.modelo.saida
    
    trainer_state = GraficoTreinamento.carregar_trainer_state(chkpt_dir)
    
    if trainer_state:
        logger.info("\n📈 Processando métricas de treinamento...")
        
        train_data, eval_data = GraficoTreinamento.extrair_metricas(trainer_state)
        checkpoints = GraficoTreinamento.listar_checkpoints(chkpt_dir)
        
        if train_data or eval_data:
            stats_report.append("\n## Métricas de Treinamento\n")
            stats_report.append(f"**Checkpoints encontrados:** {len(checkpoints)}\n")
            stats_report.append(f"**Épocas treinadas:** {trainer_state.get('epoch', 0)}\n")
            stats_report.append(f"**Steps totais:** {trainer_state.get('global_step', 0)}\n")
            
            stats_report.append("\n### Evolução do Loss\n")
            tabela_loss = GraficoTreinamento.tabela_loss_markdown(train_data, eval_data)
            stats_report.extend(tabela_loss)
            
            logger.info("   📊 Gerando gráfico de loss...")
            loss_graph_path = os.path.join(report_dir, "treinamento_loss.png")
            
            if GraficoTreinamento.evolucao_loss(train_data, eval_data, checkpoints, loss_graph_path):
                logger.info("   ✅ Gráfico de loss salvo: treinamento_loss.png")
                stats_report.append(f"\n### Gráfico de Evolução do Loss\n")
                stats_report.append(f"![Loss de Treinamento](treinamento_loss.png)\n")
                stats_report.append("*Linhas verdes tracejadas: fim de época | Linhas cinzas pontilhadas: checkpoints*\n")
            else:
                logger.warning("   ⚠️ Erro ao gerar gráfico de loss.")
        else:
            logger.info("   ℹ️ Nenhum dado de loss encontrado nos checkpoints.")
    else:
        logger.info("\n📊 Nenhum treinamento realizado ainda (pasta chkpt não encontrada ou sem trainer_state).")

    # Métricas de hardware
    from treinar_unsloth_graficos import GraficoHardware
    
    treinamento_dir = os.path.join(yaml_config.modelo.saida, "treinamento")
    hardware_metricas = GraficoHardware.carregar_metricas(treinamento_dir)
    
    if hardware_metricas:
        logger.info("\n📊 Processando métricas de hardware...")
        
        stats_report.append("\n## Métricas de Hardware\n")
        stats_report.append(f"**Amostras coletadas:** {len(hardware_metricas)}\n")
        
        stats_report.append("\n### Resumo de Uso de Recursos\n")
        tabela_hw = GraficoHardware.tabela_resumo_markdown(hardware_metricas)
        stats_report.extend(tabela_hw)
        
        logger.info("   📊 Gerando gráfico de memória...")
        mem_graph_path = os.path.join(report_dir, "hardware_memoria.png")
        
        if GraficoHardware.evolucao_memoria(hardware_metricas, mem_graph_path):
            logger.info("   ✅ Gráfico de memória salvo: hardware_memoria.png")
            stats_report.append(f"\n### Gráfico de Uso de Memória\n")
            stats_report.append(f"![Uso de Memória](hardware_memoria.png)\n")
        else:
            logger.warning("   ⚠️ Erro ao gerar gráfico de memória.")
    else:
        logger.info("\n📊 Nenhuma métrica de hardware disponível (arquivo hardware_metrics.jsonl não encontrado).")

    # Salva relatório
    report_path = os.path.join(report_dir, "relatorio_estatistico.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_report))
    
    logger.info(f"\n📝 Relatório estatístico salvo em: {report_path}")
    
    log_separador(caractere="=", largura=80)
    logger.info("✅ STATS COMPLETO - RELATÓRIO GERADO")
    log_separador(caractere="=", largura=80)


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
    logger.info(">> MODO PREDICT - GERANDO PREDIÇÕES")
    log_separador(caractere="=", largura=80)
    
    # Carrega configuração para validações iniciais
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=True)
    yaml_config.resolver_max_seq_length()
    _exibir_cabecalho_modelo(yaml_config)
    
    # Verifica formato de saída
    formato_json = yaml_config.formato_saida == FORMATO_SAIDA_JSON
    logger.info(f"\n📋 Formato de saída: {yaml_config.formato_saida}")
    
    # Verifica se há modelo treinado
    tem_modelo_treinado = _verificar_modelo_treinado(yaml_config)
    
    if usar_base:
        logger.info(f"ℹ️  Opção --base ativada: Forçando uso do modelo base.")
    elif not tem_modelo_treinado:
        logger.warning("\n⚠️ Não foi encontrado modelo LoRA treinado.")
        if not _perguntar_confirmacao("Deseja usar o modelo base para predição?", padrao=False):
            logger.info("Operação cancelada.")
            return
        usar_base = True
        logger.info("Usando modelo base para predição...\n")
    else:
        logger.info(f"✅ Usando modelo treinado: {yaml_config.modelo.saida}")
    
    modelo_usado = yaml_config.modelo.base if usar_base else yaml_config.modelo.saida
    
    # Define subsets a processar
    if subsets is None:
        subsets = ['treino', 'validacao', 'teste']
    
    logger.info(f"\n📋 Subsets a processar: {', '.join(subsets)}")
    
    # Cria diretório de predições
    nome_pasta = "predict_base" if usar_base else "predict"
    predict_dir = os.path.join(yaml_config.modelo.saida, nome_pasta)
    os.makedirs(predict_dir, exist_ok=True)
    
    # Carrega modelo via LLMsTrainer (mesmo caminho que executar_modelo)
    logger.info("\n🔄 Carregando modelo via LLMsTrainer...")
    ini_carga = time.time()
    
    try:
        trainer = LLMsTrainer(yaml_path, force_base=usar_base)
        logger.info(f"   ✅ Modelo carregado em {time.time() - ini_carga:.1f}s")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
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
            logger.info(f"\n📂 Processando subset: {subset}")
            log_separador(caractere="-", largura=60)
            
            if yaml_config.tipo_entrada in TIPOS_BASEADOS_EM_PASTAS:
                try:
                    mensagens = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo=subset)
                    if not mensagens:
                        logger.warning(f"   ⚠️ Nenhum dado encontrado para {subset}")
                        continue
                    logger.info(f"   📊 {len(mensagens)} registros encontrados")
                except Exception as e:
                    logger.error(f"   ❌ Erro ao carregar {subset}: {e}")
                    continue
            else:
                logger.warning(f"   ⚠️ Modo dataset não suportado para predict ainda")
                continue
            
            # Cria diretório do subset (limpa arquivos .json e .txt se já existir)
            subset_dir = os.path.join(predict_dir, subset)
            if os.path.exists(subset_dir):
                for f in os.listdir(subset_dir):
                    if f.endswith('.json') or f.endswith('.txt'):
                        try:
                            os.remove(os.path.join(subset_dir, f))
                        except Exception as e:
                            logger.warning(f"   ⚠️ Não foi possível remover {f}: {e}")
            os.makedirs(subset_dir, exist_ok=True)
            
            subset_stats = {
                'input_tokens': 0,
                'output_tokens': 0,
                'registros_ok': 0,
                'registros_erro': 0,
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
                    
                    # Usa LLMsTrainer.prompt() — mesmo caminho de executar_modelo
                    tempo_inicio = time.time()
                    resultado = trainer.prompt(
                        prompt_texto,
                        temperatura=0.02,
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
                    
                    subset_stats['input_tokens'] += input_tokens
                    subset_stats['output_tokens'] += output_tokens
                    subset_stats['registros_ok'] += 1
                    
                    if (idx + 1) % 10 == 0 or (idx + 1) == total:
                        logger.info(f"   Progresso: {idx + 1}/{total} ({100*(idx+1)//total}%)")
                        
                except Exception as e:
                    logger.error(f"   ❌ Erro no registro {idx}: {e}")
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
            
            logger.info(f"   ✅ {subset_stats['registros_ok']} predições salvas em: {subset_dir}")
            logger.info(f"   📊 Tokens: {subset_stats['input_tokens']} entrada, {subset_stats['output_tokens']} saída")
            
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
    logger.info(f"✅ PREDICT COMPLETO - Resultados em: {predict_dir}")
    logger.info(f"📊 Total: {uso_total['total_registros']} registros, {uso_total['input_tokens']} + {uso_total['output_tokens']} tokens")
    log_separador(caractere="=", largura=80)


def executar_merge(yaml_path: str, quantizacao: str = None) -> None:
    """
    Realiza merge do modelo treinado com o base.
    Salva em {output_dir}(merged_FORMATO)/
    
    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
        quantizacao: Opcional. '16bit', '4bit', 'q4_k_m', 'q8_0', 'f16'.
    """
    from unsloth import FastLanguageModel
    
    logger.info("\n")
    log_separador(caractere="=", largura=80)
    logger.info(">> MODO MERGE - INTEGRANDO LORA ADAPTERS")
    log_separador(caractere="=", largura=80)
    
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=True)
    _exibir_cabecalho_modelo(yaml_config)
    
    output_dir = yaml_config.modelo.saida
    
    if not _verificar_modelo_treinado(yaml_config):
        logger.error(f"❌ Erro: Não foi encontrado modelo treinado em {output_dir}")
        return
    
    mapa_quant = {
        '1': '16bit',
        '2': '4bit',
        '3': 'q4_k_m',
        '4': 'q8_0',
        '5': 'f16'
    }
    
    if not quantizacao:
        if sys.stdin.isatty():
            logger.info("\n📦 Escolha o formato de Exportação/Merge:")
            print("   1) ☁️  16-bit .safetensors (Padrão - vLLM/HF)")
            print("   2) 📉 4-bit  .safetensors (Compacto - vLLM/HF)")
            print("   3) 🦙 GGUF Q4_K_M (Ollama - Balanceado)")
            print("   4) 🦙 GGUF Q8_0   (Ollama - Alta Qualidade)")
            print("   5) 🦙 GGUF F16    (Ollama - Full Precision)")
            
            try:
                escolha = input("\nOpção [1]: ").strip()
                if not escolha:
                    escolha = '1'
                quantizacao = mapa_quant.get(escolha, '16bit')
            except (KeyboardInterrupt, EOFError):
                quantizacao = '16bit'
        else:
            quantizacao = '16bit'
            logger.info(f"Iniciando merge padrão ({quantizacao})")

    quantizacao = quantizacao.lower()
    
    dirname = f"{output_dir}(merged_{quantizacao})"
        
    logger.info(f"\n📂 Diretório de destino: {dirname}")
    logger.info(f"⚙️  Formato: {quantizacao}")

    # Gera stats antes do merge
    logger.info("\n📊 Verificando estatísticas...")
    try:
        executar_stats(yaml_path)
    except Exception as e:
        logger.warning(f"⚠️  Erro ao gerar estatísticas (o merge continuará): {e}")

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
            
    logger.info(f"\n🔄 Carregando modelo LoRA de: {output_dir}")
    logger.info("   Isso pode levar alguns instantes...")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=output_dir,
            max_seq_length=yaml_config.treinamento.max_seq_length,
            dtype=None,
            load_in_4bit=yaml_config.treinamento.nbits == 4,
        )
        
        is_gguf = quantizacao in ['q4_k_m', 'q8_0', 'f16'] or 'gguf' in quantizacao
        
        if is_gguf:
            logger.info(f"💾 Convertendo para GGUF ({quantizacao})...")
            logger.warning("   ⚠️  Isso pode demorar e consumir memória significativa.")
            model.save_pretrained_gguf(dirname, tokenizer, quantization_method=quantizacao)
        else:
            method = "merged_4bit_forced" if quantizacao == '4bit' else "merged_16bit"
            logger.info(f"💾 Salvando merge ({method})...")
            model.save_pretrained_merged(dirname, tokenizer, save_method=method)
        
        src_treinamento = os.path.join(output_dir, "treinamento")
        dst_treinamento = os.path.join(dirname, "treinamento")
        
        if os.path.exists(src_treinamento):
            logger.info("📋 Copiando relatórios e gráficos...")
            shutil.copytree(src_treinamento, dst_treinamento, dirs_exist_ok=True)
        
        logger.info("✅ Merge concluído com sucesso!")
        logger.info(f"   Modelo pronto em: {dirname}")
        
    except Exception as e:
        logger.error(f"❌ Erro ao realizar merge/exportação: {e}")


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
    logger.info(f">> MODO MODELO - TESTANDO INFERÊNCIA ({n_exemplos} exemplo(s))")
    log_separador(caractere="=", largura=80)
    
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=False)
    
    if not usar_base:
        if not _verificar_modelo_treinado(yaml_config):
            logger.warning("\n⚠️  Não foi encontrado modelo LoRA treinado na pasta de saída.")
            if not _perguntar_confirmacao("Deseja continuar com o modelo base?", padrao=False):
                logger.info("Operação cancelada.")
                return
            usar_base = True
            logger.info("Continuando com modelo base (sem fine-tuning)...\n")
        else:
            logger.info(f"✅ Modelo LoRA treinado encontrado em: {yaml_config.modelo.saida}")
    else:
        logger.info("ℹ️  Opção --base ativada: Forçando uso do modelo base.")

    trainer = LLMsTrainer(yaml_path, force_base=usar_base)
    resultado = trainer.testar_predicoes(
        n_exemplos=n_exemplos, 
        temperatura=0.2, 
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
    logger.info("✅ TESTE DE INFERÊNCIA COMPLETO")
    log_separador(caractere="=", largura=80)


# ---------------------------------------------------------------------------
# Menu interativo de avaliação
# ---------------------------------------------------------------------------

def _modo_interativo_avaliar(yaml_path: str) -> Optional[str]:
    """
    Exibe menu interativo com ações de avaliação.
    
    Returns:
        Nome da ação escolhida ou None se cancelou
    """
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=False)
    _exibir_cabecalho_modelo(yaml_config)
    
    # Status atual
    tem_modelo = _verificar_modelo_treinado(yaml_config)
    tem_checkpoints, qtd_checkpoints = _verificar_checkpoints_existem(yaml_config)
    
    logger.info("\n📊 STATUS ATUAL:")
    if tem_modelo:
        logger.info(f"   ✅ Modelo LoRA treinado encontrado")
    else:
        logger.info(f"   ❌ Nenhum modelo treinado encontrado")
    
    if tem_checkpoints:
        logger.info(f"   💾 {qtd_checkpoints} checkpoint(s) disponível(is)")
    else:
        logger.info(f"   💾 Nenhum checkpoint encontrado")
    
    # Menu
    logger.info("\n📋 AÇÕES DE AVALIAÇÃO:")
    logger.info("   1. info         - Informações detalhadas da configuração e datasets")
    logger.info("   2. stats        - Relatório estatístico (tokens, loss, hardware) com gráficos")
    if tem_modelo:
        logger.info("   3. predict      - Gerar predições com modelo treinado (todos os subsets)")
        logger.info("   4. predict-base - Gerar predições com modelo BASE (todos os subsets)")
        logger.info("   5. modelo       - Testar inferência com modelo treinado (N exemplos)")
        logger.info("   6. modelo-base  - Testar inferência com modelo BASE (N exemplos)")
        logger.info("   7. merge        - Exportar modelo (merge LoRA + Base)")
    else:
        logger.info("   3. predict-base - Gerar predições com modelo BASE (todos os subsets)")
        logger.info("   4. modelo-base  - Testar inferência com modelo BASE (N exemplos)")
    logger.info("   0. sair         - Cancelar e sair")
    
    try:
        escolha = input("\n❓ Digite o número ou nome da ação: ").strip().lower()
        
        if tem_modelo:
            mapa_acoes = {
                '1': 'info', 'info': 'info',
                '2': 'stats', 'stats': 'stats',
                '3': 'predict', 'predict': 'predict',
                '4': 'predict-base', 'predict-base': 'predict-base',
                '5': 'modelo', 'modelo': 'modelo',
                '6': 'modelo-base', 'modelo-base': 'modelo-base',
                '7': 'merge', 'merge': 'merge', 'export': 'merge',
                '0': None, 'sair': None, 'exit': None, 'quit': None,
            }
        else:
            mapa_acoes = {
                '1': 'info', 'info': 'info',
                '2': 'stats', 'stats': 'stats',
                '3': 'predict-base', 'predict-base': 'predict-base',
                '4': 'modelo-base', 'modelo-base': 'modelo-base',
                '0': None, 'sair': None, 'exit': None, 'quit': None,
            }
        
        acao = mapa_acoes.get(escolha)
        
        if escolha not in mapa_acoes:
            logger.warning(f"Opção inválida: '{escolha}'")
            return None
        
        return acao
        
    except (KeyboardInterrupt, EOFError):
        logger.info("\nOperação cancelada pelo usuário.")
        return None


def _perguntar_n_exemplos() -> int:
    """Pergunta ao usuário quantos exemplos testar."""
    try:
        n_str = input("Número de exemplos [1]: ").strip()
        return int(n_str) if n_str else 1
    except (ValueError, KeyboardInterrupt, EOFError):
        return 1


def _executar_acao_avaliar(acao: str, yaml_path: str, usar_base: bool = False, 
                           predict_subsets: list = None, quant: str = None) -> None:
    """Despacha a ação de avaliação escolhida."""
    if acao == 'info':
        executar_info(yaml_path)
    elif acao == 'stats':
        executar_stats(yaml_path)
    elif acao == 'predict':
        executar_predict(yaml_path, subsets=predict_subsets, usar_base=usar_base)
    elif acao == 'predict-base':
        executar_predict(yaml_path, subsets=predict_subsets, usar_base=True)
    elif acao == 'modelo':
        executar_modelo(yaml_path, n_exemplos=_perguntar_n_exemplos(), usar_base=usar_base)
    elif acao == 'modelo-base':
        executar_modelo(yaml_path, n_exemplos=_perguntar_n_exemplos(), usar_base=True)
    elif acao == 'merge':
        executar_merge(yaml_path, quantizacao=quant)
    else:
        logger.error(f"Ação desconhecida: '{acao}'")



# ---------------------------------------------------------------------------
# Seleção de YAML
# ---------------------------------------------------------------------------

def _selecionar_yaml() -> Optional[str]:
    """Exibe menu de seleção de YAML na pasta atual."""
    from util_menu_opcoes import escolher_yaml
    
    return escolher_yaml(
        pasta='./',
        chave_obrigatoria='modelo',
        titulo='Selecione o arquivo de configuração',
        padrao_recente=True,
        opcoes_extras=[
            ("Sair", None),
        ]
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_avaliar() -> None:
    parser = argparse.ArgumentParser(
        description="Avaliação, Inferência e Exportação de modelos LLM treinados.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ações disponíveis:
  --info            Informações detalhadas da configuração, datasets e modelo
  --stats           Relatório estatístico (tokens, loss, hardware) com gráficos
  --predict         Gera predições para todos os subsets (treino, validacao, teste)
  --predict-treino  Gera predições apenas do subset de treino
  --predict-validacao  Gera predições apenas do subset de validação
  --predict-teste   Gera predições apenas do subset de teste
  --modelo N        Testa inferência interativa com N exemplos (padrão: 1)
  --merge           Exporta modelo (merge LoRA + Base)
  
Sem argumentos: modo interativo (seleciona YAML e ação via menu).

Exemplos:
  %(prog)s                          # Modo interativo completo
  %(prog)s config.yaml              # Seleciona ação via menu
  %(prog)s config.yaml --info       # Informações detalhadas
  %(prog)s config.yaml --stats      # Relatório estatístico
  %(prog)s config.yaml --predict    # Gera predições de todos os subsets
  %(prog)s config.yaml --predict --base  # Predições com modelo base
  %(prog)s config.yaml --modelo 5   # Testa 5 predições interativas
  %(prog)s config.yaml --merge --quant q4_k_m  # Exporta GGUF Q4
"""
    )
    parser.add_argument("config", nargs='?', default=None,
                        help="Arquivo YAML com as configurações (opcional: se omitido, exibe menu)")
    
    # Ações de avaliação
    parser.add_argument("--info", action="store_true", 
                        help="Informações detalhadas da configuração e datasets")
    parser.add_argument("--stats", action="store_true",
                        help="Relatório estatístico (tokens, loss, hardware) com gráficos")
    parser.add_argument("--predict", action="store_true",
                        help="Gera predições para todos os subsets")
    parser.add_argument("--predict-treino", action="store_true",
                        help="Gera predições apenas do subset de treino")
    parser.add_argument("--predict-validacao", action="store_true",
                        help="Gera predições apenas do subset de validação")
    parser.add_argument("--predict-teste", action="store_true",
                        help="Gera predições apenas do subset de teste")
    parser.add_argument("--modelo", type=int, nargs='?', const=1,
                        help="Testa inferência interativa com N exemplos (padrão: 1)")
    parser.add_argument("--merge", action="store_true",
                        help="Exporta modelo (merge LoRA + Base)")
    
    # Opções modificadoras
    parser.add_argument("--base", action="store_true",
                        help="Força o uso do modelo base (ignora LoRA treinado)")
    parser.add_argument("--quant", type=str, default=None,
                        help="Método de quantização para merge: 16bit, 4bit, q4_k_m, q8_0, f16")
    parser.add_argument("--log-level", type=str, default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Nível de log (sobrescreve misc.log_level do YAML)")
    
    args = parser.parse_args()
    
    # --- Resolve YAML (menu se não fornecido) ---
    cfg_path = args.config
    
    if cfg_path is None:
        cfg_path = _selecionar_yaml()
        if cfg_path is None:
            sys.exit(0)
    
    if not os.path.exists(cfg_path):
        logger.error(f"❌ Arquivo não encontrado: {cfg_path}")
        sys.exit(1)
    
    # --- Configura logging ---
    log_level_padrao = "INFO"
    try:
        yaml_config = YamlTreinamento(cfg_path, validar_caminhos=False)
        log_level_padrao = yaml_config.misc.log_level
    except Exception:
        pass
    
    nivel_log = args.log_level if args.log_level else log_level_padrao
    configurar_logging(nivel=nivel_log)
    
    # --- Info CUDA ---
    if torch.cuda.is_available():
        logger.info(f"CUDA disponível — {torch.cuda.device_count()} GPU(s) detectada(s)")
    else:
        logger.warning("CUDA não disponível — inferência será na CPU (muito mais lento)")
    
    # --- Identifica se há ação explícita na CLI ---
    tem_acao_explicita = any([
        args.info, args.stats, args.predict, 
        getattr(args, 'predict_treino', False),
        getattr(args, 'predict_validacao', False),
        getattr(args, 'predict_teste', False),
        args.modelo is not None, args.merge
    ])
    
    # --- Roteamento ---
    if not tem_acao_explicita:
        # Modo interativo: menu de ações
        acao = _modo_interativo_avaliar(cfg_path)
        if acao:
            _executar_acao_avaliar(acao, cfg_path, usar_base=args.base, quant=args.quant)
        sys.exit(0)
    
    # Ações explícitas via CLI
    if args.info:
        executar_info(cfg_path)
    elif args.stats:
        executar_stats(cfg_path)
    elif args.merge:
        executar_merge(cfg_path, quantizacao=args.quant)
    elif args.modelo is not None:
        n_exemplos = args.modelo if isinstance(args.modelo, int) else 1
        executar_modelo(cfg_path, n_exemplos=n_exemplos, usar_base=args.base)
    else:
        # Predições
        predict_subsets = None
        if getattr(args, 'predict_treino', False):
            predict_subsets = predict_subsets or []
            predict_subsets.append('treino')
        if getattr(args, 'predict_validacao', False):
            predict_subsets = predict_subsets or []
            predict_subsets.append('validacao')
        if getattr(args, 'predict_teste', False):
            predict_subsets = predict_subsets or []
            predict_subsets.append('teste')
        # --predict sem subset específico → todos
        
        executar_predict(cfg_path, subsets=predict_subsets, usar_base=args.base)


if __name__ == "__main__":
    UtilEnv.carregar_env(pastas=['./', '../', '../src/'])
    _cli_avaliar()
