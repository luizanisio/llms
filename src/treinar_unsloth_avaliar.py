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
    --predict           Exportar predições (padrão: subset teste; menu interativo no modo sem args)
    --predict-treino    Predições apenas do subset de treino
    --predict-validacao Predições apenas do subset de validação
    --predict-teste     Predições apenas do subset de teste
    --modelo N          Inferência interativa com N exemplos (default: 1)
    --merge             Exportação: merge LoRA + Base

Opções:
    --base              Força uso do modelo base (ignora LoRA treinado)
    --quant METODO      Formato de merge (16bit, 4bit)
    --log-level LEVEL   Nível de log (DEBUG, INFO, WARNING, ERROR)

Nota: Itens já exportados (com .txt e .json válidos) são ignorados
      automaticamente, permitindo continuação de exportações incompletas.
"""

import argparse
import os
import sys
import json
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
from treinar_unsloth_export import (
    _perguntar_subsets_predict,
    executar_predict,
    executar_predict_vllm,
    executar_predict_unsloth,
    executar_merge,
    executar_modelo,
    executar_modelo_vllm,
    executar_modelo_unsloth,
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
        itens.append(('3', 'predict',      'Exportar predições com modelo treinado'))
        if unsloth_ok:
            itens.append(('3u', 'predict-unsloth', '⚡ Predições com unsloth (2x mais rápido)', 'amarelo', 1))
        if vllm_ok:
            itens.append(('3v', 'predict-vllm',    '🚀 Predições RÁPIDAS com vLLM (até 24x mais rápido)', 'verde', 1))
        itens.append(('4', 'predict-base',  'Exportar predições com modelo BASE'))
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
        itens.append(('3', 'predict-base', 'Exportar predições com modelo BASE'))
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
    """Despacha a ação de avaliação escolhida.
    
    Para ações de predict no modo interativo (predict_subsets=None),
    exibe menu de seleção de subsets antes de executar.
    """
    # Para ações de predict, pergunta qual subset se não veio da CLI
    if acao in ('predict', 'predict-vllm', 'predict-unsloth',
                'predict-base', 'predict-base-vllm', 'predict-base-unsloth'):
        if predict_subsets is None:
            predict_subsets = _perguntar_subsets_predict()
            if predict_subsets is None:
                logger.info("Operação cancelada.")
                return
    
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
  --predict         Exportar predições (padrão: teste; no interativo exibe menu de subsets)
  --predict-treino  Predições apenas do subset de treino
  --predict-validacao  Predições apenas do subset de validação
  --predict-teste   Predições apenas do subset de teste
  --modelo N        Testa inferência interativa com N exemplos (padrão: 1)
  --merge           Exporta modelo (merge LoRA + Base)
  
Sem argumentos: modo interativo (seleciona YAML e ação via menu).
Itens já exportados (com .txt e .json válidos) são ignorados, permitindo
continuação de exportações interrompidas.

Exemplos:
  %(prog)s                          # Modo interativo completo
  %(prog)s config.yaml              # Seleciona ação via menu
  %(prog)s config.yaml --info       # Informações detalhadas
  %(prog)s config.yaml --stats      # Relatório estatístico
  %(prog)s config.yaml --predict    # Exporta predições (subset teste)
  %(prog)s config.yaml --predict-treino --predict-teste  # Treino + teste
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
                        help="Exportar predições (padrão: subset teste)")
    parser.add_argument("--predict-treino", action="store_true",
                        help="Exportar predições do subset de treino")
    parser.add_argument("--predict-validacao", action="store_true",
                        help="Exportar predições do subset de validação")
    parser.add_argument("--predict-teste", action="store_true",
                        help="Exportar predições do subset de teste")
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
        # --predict sem subset específico → padrão ['teste']
        
        executar_predict(cfg_path, subsets=predict_subsets, usar_base=args.base)


if __name__ == "__main__":
    UtilEnv.carregar_env(pastas=['./', '../', '../src/'])
    _cli_avaliar()
