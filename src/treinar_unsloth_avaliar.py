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
    --quant METODO      Formato de merge (16bit, 4bit)
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
from util_print import print_cores, exibir_menu_opcoes
from treinar_unsloth_util import (
    YamlTreinamento, FORMATO_SAIDA_JSON
)
from treinar_unsloth_actions import (
    _exibir_cabecalho_modelo,
    _verificar_modelo_treinado,
    _verificar_checkpoints_existem,
    _perguntar_confirmacao,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Geração de gráficos/estatísticas reutilizável (pós-treinamento e --stats)
# ---------------------------------------------------------------------------

def gerar_graficos_estatisticos(yaml_config, silencioso: bool = False,
                                stats_report: list = None) -> Optional[str]:
    """
    Gera gráficos e relatório estatístico a partir das métricas de treinamento
    já salvas em disco (training_metrics.jsonl, checkpoints, hardware).

    Pode ser chamada tanto pelo --stats quanto automaticamente ao final do
    treinamento, sem necessidade de recarregar o dataset.

    Args:
        yaml_config: Instância de YamlTreinamento já carregada.
        silencioso: Se True, reduz mensagens de log (útil no pós-treinamento).
        stats_report: Lista existente de linhas do relatório para estender.
                      Se None, cria um relatório novo com cabeçalho próprio.

    Returns:
        Caminho do relatório gerado ou None se não havia dados suficientes.
    """
    from treinar_unsloth_graficos import GraficoTreinamento, GraficoEficiencia, GraficoHardware

    report_dir = os.path.join(yaml_config.modelo.saida, "treinamento")
    os.makedirs(report_dir, exist_ok=True)

    if not silencioso:
        logger.info("<azul>\n📈 Gerando gráficos estatísticos do treinamento...</azul>")

    # ---- Métricas de loss (prioriza training_metrics.jsonl) ----
    treinamento_dir = report_dir
    metricas_jsonl = GraficoTreinamento.carregar_training_metrics(treinamento_dir)

    chkpt_dir = os.path.join(yaml_config.modelo.saida, "chkpt")
    if not os.path.exists(chkpt_dir) or not any(
        d.startswith("checkpoint-") for d in os.listdir(chkpt_dir)
        if os.path.isdir(os.path.join(chkpt_dir, d))
    ):
        chkpt_dir = yaml_config.modelo.saida
    checkpoints = GraficoTreinamento.listar_checkpoints(chkpt_dir)

    if metricas_jsonl:
        train_data = metricas_jsonl["train_data"]
        eval_data = metricas_jsonl["eval_data"]
        etapas_curriculum = metricas_jsonl["etapas"]
    else:
        train_data, eval_data, etapas_curriculum = [], [], []
        trainer_state = GraficoTreinamento.carregar_trainer_state(chkpt_dir)
        if trainer_state:
            train_data, eval_data = GraficoTreinamento.extrair_metricas(trainer_state)

    if not train_data and not eval_data:
        if not silencioso:
            logger.info("📊 Nenhum dado de loss encontrado (sem training_metrics.jsonl nem checkpoints).")
        return None

    # ---- Monta relatório ----
    report_proprio = stats_report is None
    if stats_report is None:
        stats_report = []
        stats_report.append("# Relatório Estatístico do Treinamento\n")
        stats_report.append(f"**Modelo Base:** `{yaml_config.modelo.base}`\n")
        stats_report.append(f"**Modelo Saída:** `{yaml_config.modelo.saida}`\n")
        stats_report.append(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ---- Gráfico de Loss ----
    if train_data or eval_data:
        if not silencioso:
            logger.info("<cinza>   📊 Gerando gráfico de loss...</cinza>")

        stats_report.append("\n## Métricas de Treinamento\n")
        stats_report.append(f"**Checkpoints encontrados:** {len(checkpoints)}\n")
        if etapas_curriculum:
            nomes = [et["alias"] for et in etapas_curriculum]
            stats_report.append(f"**Etapas do curriculum:** {' → '.join(nomes)}\n")

        stats_report.append("\n### Evolução do Loss\n")
        tabela_loss = GraficoTreinamento.tabela_loss_markdown(train_data, eval_data)
        stats_report.extend(tabela_loss)

        loss_graph_path = os.path.join(report_dir, "treinamento_loss.png")
        if GraficoTreinamento.evolucao_loss(
            train_data, eval_data, checkpoints, loss_graph_path,
            etapas_curriculum=etapas_curriculum
        ):
            logger.info("<verde>   ✅ Gráfico de loss salvo: treinamento_loss.png</verde>")
            stats_report.append("\n### Gráfico de Evolução do Loss\n")
            stats_report.append("![Loss de Treinamento](treinamento_loss.png)\n")
            legenda = "*Linhas verdes: fim de época | Cinzas: checkpoints"
            if etapas_curriculum and len(etapas_curriculum) > 1:
                legenda += " | Violeta: transição de etapa curriculum"
            legenda += "*\n"
            stats_report.append(legenda)
        else:
            logger.warning("<amarelo>   ⚠️ Erro ao gerar gráfico de loss.</amarelo>")

    # ---- Gráfico de Eficiência (tokens/instâncias acumulados) ----
    if train_data:
        if not silencioso:
            logger.info("<cinza>   📊 Gerando gráfico de eficiência (tokens)...</cinza>")

        tokens_graph_path = os.path.join(report_dir, "treinamento_tokens.png")
        if GraficoEficiencia.evolucao_tokens(
            train_data, tokens_graph_path,
            etapas_curriculum=etapas_curriculum
        ):
            logger.info("<verde>   ✅ Gráfico de tokens salvo: treinamento_tokens.png</verde>")
            stats_report.append("\n### Custo Computacional\n")
            stats_report.append("![Tokens Acumulados](treinamento_tokens.png)\n")
            stats_report.append("*Azul: tokens processados (custo computacional) | Laranja: instâncias treinadas*\n")
        else:
            if not silencioso:
                logger.info("<cinza>   ℹ️ Gráfico de tokens não gerado (sem dados de tokens_acumulados).</cinza>")

    # ---- Métricas de Hardware ----
    hardware_metricas = GraficoHardware.carregar_metricas(treinamento_dir)

    if hardware_metricas:
        if not silencioso:
            logger.info("<azul>\n📊 Processando métricas de hardware...</azul>")

        stats_report.append("\n## Métricas de Hardware\n")
        stats_report.append(f"**Amostras coletadas:** {len(hardware_metricas)}\n")

        stats_report.append("\n### Resumo de Uso de Recursos\n")
        tabela_hw = GraficoHardware.tabela_resumo_markdown(hardware_metricas)
        stats_report.extend(tabela_hw)

        if not silencioso:
            logger.info("<cinza>   📊 Gerando gráfico de memória...</cinza>")
        mem_graph_path = os.path.join(report_dir, "hardware_memoria.png")

        if GraficoHardware.evolucao_memoria(hardware_metricas, mem_graph_path, train_data=train_data, etapas_curriculum=etapas_curriculum):
            logger.info("<verde>   ✅ Gráfico de memória salvo: hardware_memoria.png</verde>")
            stats_report.append("\n### Gráfico de Uso de Memória\n")
            stats_report.append("![Uso de Memória](hardware_memoria.png)\n")
        else:
            logger.warning("<amarelo>   ⚠️ Erro ao gerar gráfico de memória.</amarelo>")
    else:
        if not silencioso:
            logger.info("\n📊 Nenhuma métrica de hardware disponível (sem dados em training_metrics.jsonl).")

    # ---- Resumo Consolidado ----
    _gerar_resumo_consolidado(stats_report, train_data, eval_data, etapas_curriculum, hardware_metricas)

    # ---- Salva relatório ----
    report_path = os.path.join(report_dir, "relatorio_estatistico.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_report))

    logger.info(f"<verde>📝 Relatório estatístico salvo em: {report_path}</verde>")
    return report_path


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
    
    # Valida max_seq_length (obrigatório) e exibe info de tokens
    try:
        yaml_config.validar_max_seq_length()
    except Exception as e:
        logger.warning(f"<amarelo>⚠️ Não foi possível validar max_seq_length: {e}</amarelo>")
    
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
        logger.info(f"<verde>\n📝 Info salvo em: {info_path}</verde>")
    except Exception as e:
        logger.warning(f"<amarelo>⚠️ Não foi possível salvar info.md: {e}</amarelo>")


def _gerar_resumo_consolidado(
    stats_report: list,
    train_data: list,
    eval_data: list,
    etapas_curriculum: list,
    hardware_metricas: list,
) -> None:
    """Adiciona seção 'Resumo Consolidado' ao relatório com dados-chave dos três gráficos."""
    linhas = ["\n---\n", "\n## Resumo Consolidado do Treinamento\n"]

    # --- Loss ---
    if train_data:
        total_steps = len(train_data)
        loss_inicial = train_data[0].get("loss", 0)
        loss_final = train_data[-1].get("loss", 0)
        reducao_pct = ((loss_inicial - loss_final) / loss_inicial * 100) if loss_inicial > 0 else 0

        linhas.append("### Desempenho (Loss)\n")
        linhas.append("| Indicador | Valor |")
        linhas.append("|-----------|-------|")
        linhas.append(f"| Loss inicial | {loss_inicial:.4f} |")
        linhas.append(f"| Loss final | {loss_final:.4f} |")
        linhas.append(f"| Redução | {reducao_pct:.1f}% |")
        linhas.append(f"| Steps de treino | {total_steps} |")

        if eval_data:
            best_eval = min(eval_data, key=lambda e: e.get("eval_loss", float("inf")))
            linhas.append(f"| Melhor eval loss | {best_eval.get('eval_loss', 0):.4f} (step {best_eval.get('step', '?')}) |")
            linhas.append(f"| Avaliações realizadas | {len(eval_data)} |")

        if etapas_curriculum and len(etapas_curriculum) > 1:
            nomes = [et["alias"] for et in etapas_curriculum]
            linhas.append(f"| Etapas curriculum | {' → '.join(nomes)} |")

        # Épocas globais
        last_epoch = train_data[-1].get("epoch_global", train_data[-1].get("epoch", 0))
        if last_epoch:
            linhas.append(f"| Épocas totais | {int(last_epoch)} |")

        linhas.append("")

    # --- Tokens / eficiência ---
    if train_data:
        tokens_final = train_data[-1].get("tokens_acumulados", 0)
        instancias_final = train_data[-1].get("instancias_acumuladas", 0)
        elapsed = train_data[-1].get("elapsed_seconds", 0)

        if tokens_final > 0 or instancias_final > 0:
            linhas.append("### Custo Computacional\n")
            linhas.append("| Indicador | Valor |")
            linhas.append("|-----------|-------|")
            if tokens_final > 0:
                linhas.append(f"| Tokens processados | {tokens_final:,.0f} |")
            if instancias_final > 0:
                linhas.append(f"| Instâncias treinadas | {instancias_final:,.0f} |")
            if elapsed > 0:
                mins = elapsed / 60
                linhas.append(f"| Tempo de treino | {mins:.1f} min |")
                if tokens_final > 0:
                    tok_per_sec = tokens_final / elapsed
                    linhas.append(f"| Throughput | {tok_per_sec:,.0f} tok/s |")
            # Eficiência: redução de loss por milhão de tokens
            if tokens_final > 0 and train_data:
                loss_ini = train_data[0].get("loss", 0)
                loss_fim = train_data[-1].get("loss", 0)
                delta_loss = loss_ini - loss_fim
                mtok = tokens_final / 1_000_000
                if mtok > 0:
                    linhas.append(f"| Eficiência (Δloss/Mtok) | {delta_loss / mtok:.4f} |")
            linhas.append("")

    # --- Hardware ---
    if hardware_metricas:
        ram_usadas = [m.get("ram_usada_gb", 0) for m in hardware_metricas]
        gpu_picos = []
        for m in hardware_metricas:
            gpu_total = 0
            for key in m:
                if key.startswith("gpu") and "reservada_gb" in key and "max" not in key:
                    gpu_total += m.get(key, 0)
            if gpu_total > 0:
                gpu_picos.append(gpu_total)

        linhas.append("### Recursos de Hardware\n")
        linhas.append("| Indicador | Valor |")
        linhas.append("|-----------|-------|")
        if ram_usadas:
            linhas.append(f"| RAM pico | {max(ram_usadas):.1f} GB |")
        if gpu_picos:
            linhas.append(f"| GPU VRAM pico | {max(gpu_picos):.1f} GB |")
            linhas.append(f"| GPU VRAM média | {sum(gpu_picos)/len(gpu_picos):.1f} GB |")
        cpu_usos = [m.get("cpu_uso_%", 0) for m in hardware_metricas if m.get("cpu_uso_%", 0) > 0]
        if cpu_usos:
            linhas.append(f"| CPU médio | {sum(cpu_usos)/len(cpu_usos):.0f}% |")
        linhas.append("")

    if len(linhas) > 2:  # Tem conteúdo além do separador e título
        stats_report.extend(linhas)


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
    logger.info("<azul>>> MODO STATS - RELATÓRIO ESTATÍSTICO DE TOKENS</azul>")
    log_separador(caractere="=", largura=80)
    
    # Carrega configuração
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=True)
    _exibir_cabecalho_modelo(yaml_config)
    
    # Cria diretório de saída para relatório
    report_dir = os.path.join(yaml_config.modelo.saida, "treinamento")
    os.makedirs(report_dir, exist_ok=True)
    
    # Carrega dados para estatísticas
    logger.info("<azul>\n📊 Carregando dados para estatísticas...</azul>")
    
    # Estrutura para armazenar dados por subset
    stats_por_subset = {}
    
    # Carrega dados do curriculum por subset
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
                logger.info(f"<cinza>   {nome}: {len(mensagens)} registros</cinza>")
                
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
            logger.warning(f"<amarelo>   ⚠️ Erro ao carregar {alvo}: {e}</amarelo>")
    
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
    
    logger.info("<azul>\n📈 Gerando relatórios e preparando gráficos...</azul>")
    
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
            logger.info(f"<verde>   ✅ Gráfico consolidado salvo: {nome_arquivo_grafico}</verde>")
            stats_report.append(f"\n## Gráfico Comparativo\n")
            stats_report.append(f"![Boxplot Comparativo]({nome_arquivo_grafico})\n")
        else:
            logger.warning("<amarelo>   ⚠️ Erro ao gerar gráfico de tokens.</amarelo>")

    # Gera gráficos de treinamento (loss, eficiência, hardware, resumo consolidado)
    # via função reutilizável (mesma usada no pós-treinamento automático)
    # Passa stats_report para estender o relatório existente (tokens + treino num só arquivo)
    gerar_graficos_estatisticos(yaml_config, stats_report=stats_report)
    
    log_separador(caractere="=", largura=80)
    logger.info("<verde>✅ STATS COMPLETO - RELATÓRIO GERADO</verde>")
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
    
    # Define subsets a processar
    if subsets is None:
        subsets = ['treino', 'validacao', 'teste']
    
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
            
            logger.info(f"<verde>   ✅ {subset_stats['registros_ok']} predições salvas em: {subset_dir}</verde>")
            logger.info(f"<cinza>   📊 Tokens: {subset_stats['input_tokens']} entrada, {subset_stats['output_tokens']} saída</cinza>")
            
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

            # Cria diretório do subset
            subset_dir = os.path.join(predict_dir, subset)
            if os.path.exists(subset_dir):
                for f in os.listdir(subset_dir):
                    if f.endswith('.json') or f.endswith('.txt'):
                        try:
                            os.remove(os.path.join(subset_dir, f))
                        except Exception:
                            pass
            os.makedirs(subset_dir, exist_ok=True)

            total = len(mensagens)
            subset_stats = {'input_tokens': 0, 'output_tokens': 0, 'registros_ok': 0, 'registros_erro': 0}
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

                    subset_stats['output_tokens'] += res["tokens"]
                    subset_stats['registros_ok'] += 1

            except Exception as e:
                logger.error(f"<vermelho>   ❌ Erro no batch inference: {e}</vermelho>")

            tempo_subset = time.time() - ini_subset
            logger.info(f"<verde>   ✅ {subset_stats['registros_ok']} predições salvas em: {subset_dir}</verde>")
            logger.info(f"<cinza>   📊 Tokens: {subset_stats['input_tokens']} entrada, {subset_stats['output_tokens']} saída ({tempo_subset:.1f}s)</cinza>")

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
    import shutil
    if not os.environ.get('CC'):
        cc_path = shutil.which('gcc') or shutil.which('cc')
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
    import shutil
    if not os.environ.get('CC'):
        cc_path = shutil.which('gcc') or shutil.which('cc')
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

    # Define subsets a processar
    if subsets is None:
        subsets = ['treino', 'validacao', 'teste']

    logger.info(f"<cinza>\n📋 Subsets a processar: {', '.join(subsets)}</cinza>")

    # Cria diretório de predições
    nome_pasta = "predict_base_unsloth" if usar_base else "predict_unsloth"
    predict_dir = os.path.join(output_dir, nome_pasta)
    os.makedirs(predict_dir, exist_ok=True)

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

            # Cria diretório do subset
            subset_dir = os.path.join(predict_dir, subset)
            if os.path.exists(subset_dir):
                for f in os.listdir(subset_dir):
                    if f.endswith('.json') or f.endswith('.txt'):
                        try:
                            os.remove(os.path.join(subset_dir, f))
                        except Exception:
                            pass
            os.makedirs(subset_dir, exist_ok=True)

            total = len(mensagens)
            subset_stats = {'input_tokens': 0, 'output_tokens': 0, 'registros_ok': 0, 'registros_erro': 0}
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
            logger.info(f"<verde>   ✅ {subset_stats['registros_ok']} predições salvas em: {subset_dir}</verde>")
            logger.info(f"<cinza>   📊 Tokens: {subset_stats['input_tokens']} entrada, {subset_stats['output_tokens']} saída ({tempo_subset:.1f}s)</cinza>")

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
    
    print_cores("\n📊 STATUS ATUAL:", color_auto=False)
    if tem_modelo:
        print_cores("   ✅ Modelo LoRA treinado encontrado", color_auto=False)
    else:
        print_cores("   ❌ Nenhum modelo treinado encontrado", color_auto=False)
    
    if tem_checkpoints:
        print_cores(f"   💾 {qtd_checkpoints} checkpoint(s) disponível(is)", color_auto=False)
    else:
        print_cores("   💾 Nenhum checkpoint encontrado", color_auto=False)
    
    # Verifica disponibilidade sem importar módulos pesados
    import importlib.util
    vllm_ok = importlib.util.find_spec("vllm") is not None
    unsloth_ok = importlib.util.find_spec("unsloth") is not None

    # --- Monta itens do menu dinamicamente ---
    itens = [
        ('1', 'info',    'Informações detalhadas da configuração e datasets'),
        ('2', 'stats',   'Relatório estatístico (tokens, loss, hardware) com gráficos'),
    ]

    if tem_modelo:
        itens.append(('3', 'predict',      'Gerar predições com modelo treinado (todos os subsets)'))
        if unsloth_ok:
            itens.append(('3u', 'predict-unsloth', '⚡ Predições com unsloth (2x mais rápido)', 'amarelo', 1))
        if vllm_ok:
            itens.append(('3v', 'predict-vllm',    '🚀 Predições RÁPIDAS com vLLM (até 24x mais rápido)', 'verde', 1))
        itens.append(('4', 'predict-base',  'Gerar predições com modelo BASE (todos os subsets)'))
        if unsloth_ok:
            itens.append(('4u', 'predict-base-unsloth', '⚡ Predições BASE com unsloth (2x mais rápido)', 'amarelo', 1))
        if vllm_ok:
            itens.append(('4v', 'predict-base-vllm',    '🚀 Predições BASE com vLLM (até 24x mais rápido)', 'verde', 1))
        itens.append(('5', 'modelo',        'Testar inferência com modelo treinado (N exemplos)'))
        if unsloth_ok:
            itens.append(('5u', 'modelo-unsloth', '⚡ Testar inferência com unsloth (2x mais rápido)', 'amarelo', 1))
        else:
            itens.append(('5u', 'modelo-unsloth', '(Inativo) Depende do pacote unsloth', 'cinza', 1))
        if vllm_ok:
            itens.append(('5v', 'modelo-vllm',    '🚀 Testar inferência com vLLM (RÁPIDO)', 'verde', 1))
        else:
            itens.append(('5v', 'modelo-vllm',    '(Inativo) Depende do pacote vLLM', 'cinza', 1))
        itens.append(('6', 'modelo-base',   'Testar inferência com modelo BASE (N exemplos)'))
        if unsloth_ok:
            itens.append(('6u', 'modelo-base-unsloth', '⚡ Testar BASE com unsloth (2x mais rápido)', 'amarelo', 1))
        else:
            itens.append(('6u', 'modelo-base-unsloth', '(Inativo) Depende do pacote unsloth', 'cinza', 1))
        if vllm_ok:
            itens.append(('6v', 'modelo-base-vllm',    '🚀 Testar BASE com vLLM (RÁPIDO)', 'verde', 1))
        else:
            itens.append(('6v', 'modelo-base-vllm',    '(Inativo) Depende do pacote vLLM', 'cinza', 1))

        itens.append(('---', '<azul>📦 EXPORTAÇÃO:</azul>'))
        itens.append(('7', 'merge',  'Exportar modelo (HF safetensors → converta para GGUF/Ollama)'))
    else:
        itens.append(('3', 'predict-base', 'Gerar predições com modelo BASE (todos os subsets)'))
        itens.append(('4', 'modelo-base',  'Testar inferência com modelo BASE (N exemplos)'))
        if unsloth_ok:
            itens.append(('4u', 'modelo-base-unsloth', '⚡ Testar BASE com unsloth (2x mais rápido)', 'amarelo', 1))
        else:
            itens.append(('4u', 'modelo-base-unsloth', '(Inativo) Depende do pacote unsloth', 'cinza', 1))
        if vllm_ok:
            itens.append(('4v', 'modelo-base-vllm',    '🚀 Testar BASE com vLLM (RÁPIDO)', 'verde', 1))
        else:
            itens.append(('4v', 'modelo-base-vllm',    '(Inativo) Depende do pacote vLLM', 'cinza', 1))

    itens.append(('---',))
    itens.append(('0', 'sair', 'Cancelar e sair'))

    # --- Notas de disponibilidade ---
    notas = []
    if not unsloth_ok:
        notas.append("unsloth não instalado. Instale-o para ativar opções ⚡ (2x).")
    if not vllm_ok:
        notas.append("vLLM não instalado. Use 'pip install vllm' para ativar as opções 🚀.")

    try:
        escolha = exibir_menu_opcoes(
            titulo='<azul>📋 AÇÕES DE AVALIAÇÃO:</azul>',
            itens=itens,
            notas=notas if notas else None,
        )
        
        if tem_modelo:
            mapa_acoes = {
                '1': 'info', 'info': 'info',
                '2': 'stats', 'stats': 'stats',
                '3': 'predict', 'predict': 'predict',
                '3u': 'predict-unsloth', 'predict-unsloth': 'predict-unsloth',
                '3v': 'predict-vllm', 'predict-vllm': 'predict-vllm',
                '4': 'predict-base', 'predict-base': 'predict-base',
                '4u': 'predict-base-unsloth', 'predict-base-unsloth': 'predict-base-unsloth',
                '4v': 'predict-base-vllm', 'predict-base-vllm': 'predict-base-vllm',
                '5': 'modelo', 'modelo': 'modelo',
                '5u': 'modelo-unsloth', 'modelo-unsloth': 'modelo-unsloth',
                '5v': 'modelo-vllm', 'modelo-vllm': 'modelo-vllm',
                '6': 'modelo-base', 'modelo-base': 'modelo-base',
                '6u': 'modelo-base-unsloth', 'modelo-base-unsloth': 'modelo-base-unsloth',
                '6v': 'modelo-base-vllm', 'modelo-base-vllm': 'modelo-base-vllm',
                '7': 'merge', 'merge': 'merge', 'export': 'merge',
                '0': None, 'sair': None, 'exit': None, 'quit': None,
            }
        else:
            mapa_acoes = {
                '1': 'info', 'info': 'info',
                '2': 'stats', 'stats': 'stats',
                '3': 'predict-base', 'predict-base': 'predict-base',
                '4': 'modelo-base', 'modelo-base': 'modelo-base',
                '4u': 'modelo-base-unsloth', 'modelo-base-unsloth': 'modelo-base-unsloth',
                '4v': 'modelo-base-vllm', 'modelo-base-vllm': 'modelo-base-vllm',
                '0': None, 'sair': None, 'exit': None, 'quit': None,
            }
        
        acao = mapa_acoes.get(escolha)
        
        if acao in ['predict-vllm', 'predict-base-vllm', 'modelo-vllm', 'modelo-base-vllm'] and not vllm_ok:
            logger.warning("Opção indisponível: vLLM não está instalado.")
            return None

        if acao in ['predict-unsloth', 'predict-base-unsloth', 'modelo-unsloth', 'modelo-base-unsloth'] and not unsloth_ok:
            logger.warning("Opção indisponível: unsloth não está instalado.")
            return None
        
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
                           predict_subsets: list = None, quant: str = None,
                           gerar_zip: bool = False) -> None:
    """Despacha a ação de avaliação escolhida."""
    if acao == 'info':
        executar_info(yaml_path)
    elif acao == 'stats':
        executar_stats(yaml_path)
    elif acao == 'predict':
        executar_predict(yaml_path, subsets=predict_subsets, usar_base=usar_base)
    elif acao == 'predict-vllm':
        executar_predict_vllm(yaml_path, subsets=predict_subsets)
    elif acao == 'predict-unsloth':
        executar_predict_unsloth(yaml_path, subsets=predict_subsets, usar_base=usar_base)
    elif acao == 'predict-base':
        executar_predict(yaml_path, subsets=predict_subsets, usar_base=True)
    elif acao == 'predict-base-unsloth':
        executar_predict_unsloth(yaml_path, subsets=predict_subsets, usar_base=True)
    elif acao == 'predict-base-vllm':
        executar_predict_vllm(yaml_path, subsets=predict_subsets, usar_base=True)
    elif acao == 'modelo':
        executar_modelo(yaml_path, n_exemplos=_perguntar_n_exemplos(), usar_base=usar_base)
    elif acao == 'modelo-unsloth':
        executar_modelo_unsloth(yaml_path, n_exemplos=_perguntar_n_exemplos(), usar_base=usar_base)
    elif acao == 'modelo-vllm':
        executar_modelo_vllm(yaml_path, n_exemplos=_perguntar_n_exemplos(), usar_base=usar_base)
    elif acao == 'modelo-base':
        executar_modelo(yaml_path, n_exemplos=_perguntar_n_exemplos(), usar_base=True)
    elif acao == 'modelo-base-unsloth':
        executar_modelo_unsloth(yaml_path, n_exemplos=_perguntar_n_exemplos(), usar_base=True)
    elif acao == 'modelo-base-vllm':
        executar_modelo_vllm(yaml_path, n_exemplos=_perguntar_n_exemplos(), usar_base=True)
    elif acao == 'merge':
        executar_merge(yaml_path, quantizacao=quant, gerar_zip=gerar_zip)
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
  %(prog)s config.yaml --merge --quant 16bit   # Exporta safetensors 16-bit
  %(prog)s config.yaml --merge --zip           # Exporta e compacta em .zip
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
                        help="Formato de merge: 16bit (padrão, recomendado), 4bit (compacto)")
    parser.add_argument("--zip", action="store_true",
                        help="Compacta o modelo merged em .zip após exportação")
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
            _executar_acao_avaliar(acao, cfg_path, usar_base=args.base, quant=args.quant,
                                   gerar_zip=getattr(args, 'zip', False))
        sys.exit(0)
    
    # Ações explícitas via CLI
    if args.info:
        executar_info(cfg_path)
    elif args.stats:
        executar_stats(cfg_path)
    elif args.merge:
        executar_merge(cfg_path, quantizacao=args.quant, gerar_zip=getattr(args, 'zip', False))
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
