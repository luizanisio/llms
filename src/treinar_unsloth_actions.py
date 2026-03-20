#!/usr/bin/env python3

"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Módulo de ações para o pacote treinar_unsloth.py

Ações disponíveis:
    --info    : Informações gerais do treinamento e modelo
    --stats   : Relatório estatístico com tokens de entrada/saída e boxplots
    --treinar : Inicia ou reinicia o treinamento
    --reset   : Limpa o treinamento atual (com confirmação)

Uso:
    python treinar_unsloth.py CONFIG.yaml [--info | --stats | --treinar | --reset]
    
    Sem ação: modo interativo
"""

import os
import sys
import shutil
from typing import Optional
import gc
import torch

# Configuração de path para permitir execução de qualquer diretório
import util  # garante que a pasta src está no sys.path

from treinar_unsloth_logging import get_logger, log_separador, log_bloco

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _exibir_cabecalho_modelo(yaml_config) -> None:
    """Exibe cabeçalho com informações do modelo base e de saída."""
    log_separador(caractere="=", largura=70)
    logger.info("📋 CONFIGURAÇÃO DO TREINAMENTO")
    log_separador(caractere="-", largura=70)
    logger.info(f"  Modelo Base:  {yaml_config.modelo.base}")
    logger.info(f"  Modelo Saída: {yaml_config.modelo.saida}")
    log_separador(caractere="=", largura=70)


def _verificar_modelo_treinado(yaml_config) -> bool:
    """
    Verifica se existe modelo LoRA treinado na pasta de saída.
    
    Returns:
        True se existir modelo treinado, False caso contrário
    """
    output_dir = yaml_config.modelo.saida
    arq_lora = os.path.join(output_dir, 'adapter_config.json')
    arq_model = os.path.join(output_dir, 'adapter_model.safetensors')
    arq_pytorch = os.path.join(output_dir, 'pytorch_model.bin')
    
    return os.path.exists(arq_lora) and (os.path.exists(arq_model) or os.path.exists(arq_pytorch))


def _verificar_checkpoints_existem(yaml_config) -> tuple[bool, int]:
    """
    Verifica se existem checkpoints na pasta de treinamento.
    
    Returns:
        Tupla (existe_checkpoint, quantidade)
    """
    checkpoint_dir = os.path.join(yaml_config.modelo.saida, "chkpt")
    if not os.path.exists(checkpoint_dir):
        return False, 0
    
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            checkpoints.append(item)
    
    return len(checkpoints) > 0, len(checkpoints)


def _perguntar_confirmacao(mensagem: str, padrao: bool = False) -> bool:
    """
    Pergunta confirmação ao usuário.
    
    Args:
        mensagem: Pergunta a ser exibida
        padrao: Resposta padrão se apenas Enter for pressionado
        
    Returns:
        True se confirmou, False caso contrário
    """
    sufixo = "[S/n]" if padrao else "[s/N]"
    try:
        resposta = input(f"{mensagem} {sufixo}: ").strip().lower()
        if not resposta:
            return padrao
        return resposta in ('s', 'sim', 'y', 'yes')
    except (KeyboardInterrupt, EOFError):
        logger.info("\nOperação cancelada pelo usuário.")
        return False


# ---------------------------------------------------------------------------
# Ações Principais
# ---------------------------------------------------------------------------


def executar_injetar_dicas(cfg_path: str) -> None:
    """
    Injeta comentários de dicas no YAML de configuração.
    
    Args:
        cfg_path: Caminho para o arquivo YAML de configuração
    """
    if not os.path.exists(cfg_path):
        logger.error(f"Arquivo '{cfg_path}' não encontrado para injetar dicas.")
        sys.exit(1)
        
    logger.info(f"ℹ️  Injetando dicas no arquivo: {cfg_path}")
    from treinar_unsloth_dicas import injetar_dicas_yaml, DICAS_YAML
    
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = injetar_dicas_yaml(content, DICAS_YAML)
        
        with open(cfg_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        logger.info(f"✅ Dicas injetadas com sucesso em '{cfg_path}'.")
    except Exception as e:
        logger.error(f"❌ Erro ao injetar dicas: {e}")
        import traceback
        traceback.print_exc()
        
    sys.exit(0)


# ---------------------------------------------------------------------------
# Funções de avaliação (info, stats, predict, merge, modelo) foram movidas
# para treinar_unsloth_avaliar.py
# ---------------------------------------------------------------------------
    """
    Gera relatório estatístico com informações sobre uso de tokens.
    Gera tabelas e gráficos separados por subset (treino, validação, teste).
    
    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
    """
    from treinar_unsloth_util import YamlTreinamento, TIPO_ENTRADA_PASTAS, TIPOS_BASEADOS_EM_PASTAS
    from treinar_unsloth_report import GeradorRelatorio
    import json
    import statistics
    from datetime import datetime
    
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
        # Modo pastas
        for alvo, nome in [("treino", "Treino"), ("validacao", "Validação"), ("teste", "Teste")]:
            try:
                mensagens = yaml_config.dataset_manager.carregar_mensagens_de_pastas(alvo=alvo)
                if mensagens:
                    # Inicializa listas para este subset
                    stats_por_subset[alvo] = {
                        'nome': nome,
                        'registros': len(mensagens),
                        'entrada': [],
                        'saida': []
                    }
                    logger.info(f"   {nome}: {len(mensagens)} registros")
                    
                    # Coleta tokens de entrada e saída
                    for msg in mensagens:
                        if isinstance(msg, dict) and 'messages' in msg:
                            for m in msg['messages']:
                                texto = m.get('content', '')
                                tokens = len(texto.split())  # Aproximação simples
                                if m.get('role') == 'user':
                                    stats_por_subset[alvo]['entrada'].append(tokens)
                                elif m.get('role') == 'assistant':
                                    stats_por_subset[alvo]['saida'].append(tokens)
            except Exception as e:
                logger.warning(f"   ⚠️ Erro ao carregar {alvo}: {e}")
    else:
        # Modo dataset
        logger.info("   Modo dataset: use --info para ver estatísticas do dataset")
        return
    
    if not stats_por_subset:
        logger.warning("   Nenhum dado encontrado para gerar estatísticas.")
        return
    
    # Função auxiliar para gerar tabela Markdown
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
    
    # Gera seções por subset e prepara dados para gráfico
    logger.info("\n📈 Gerando relatórios e preparando gráficos...")
    
    dados_grafico = {} # Dicionário ordenado para o gráfico
    
    # Ordem desejada para o gráfico: Entrada primeiro, depois Saída
    subsets_ordem = ['treino', 'validacao', 'teste']
    
    # Coleta dados para o gráfico na ordem correta
    # Primeiro Entrada
    for alvo in subsets_ordem:
        if alvo in stats_por_subset and stats_por_subset[alvo]['entrada']:
            nome = stats_por_subset[alvo]['nome']
            dados_grafico[f"{nome} (In)"] = stats_por_subset[alvo]['entrada']
            
    # Depois Saída
    for alvo in subsets_ordem:
        if alvo in stats_por_subset and stats_por_subset[alvo]['saida']:
            nome = stats_por_subset[alvo]['nome']
            dados_grafico[f"{nome} (Out)"] = stats_por_subset[alvo]['saida']

    # Gera tabelas no relatório (mantém ordem de iteração)
    for alvo, dados in stats_por_subset.items():
        nome_subset = dados['nome']
        entrada = dados['entrada']
        saida = dados['saida']
        
        titulo_secao = f"\n## Estatísticas: {nome_subset}\n"
        stats_report.append(titulo_secao)
        
        # Tabelas
        stats_report.append(_tabela_stats(entrada, "Tokens de Entrada (User)"))
        stats_report.append(_tabela_stats(saida, "Tokens de Saída (Assistant)"))

    # Gera Gráfico de Tokens (usando classe refatorada)
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

    # ==========================================================================
    # MÉTRICAS DE TREINAMENTO
    # Prioriza training_metrics.jsonl (contém etapa curriculum e instâncias acumuladas)
    # Fallback: trainer_state.json do último checkpoint
    # ==========================================================================
    from treinar_unsloth_graficos import GraficoTreinamento
    
    treinamento_dir = os.path.join(yaml_config.modelo.saida, "treinamento")
    metricas_jsonl = GraficoTreinamento.carregar_training_metrics(treinamento_dir)
    
    # Tenta encontrar checkpoints em chkpt/ ou no diretório raiz do modelo
    chkpt_dir = os.path.join(yaml_config.modelo.saida, "chkpt")
    if not os.path.exists(chkpt_dir) or not any(d.startswith("checkpoint-") for d in os.listdir(chkpt_dir) if os.path.isdir(os.path.join(chkpt_dir, d))):
        chkpt_dir = yaml_config.modelo.saida
    checkpoints = GraficoTreinamento.listar_checkpoints(chkpt_dir)
    
    if metricas_jsonl:
        # Fonte preferida: training_metrics.jsonl com dados enriquecidos
        train_data = metricas_jsonl["train_data"]
        eval_data = metricas_jsonl["eval_data"]
        etapas_curriculum = metricas_jsonl["etapas"]
    else:
        # Fallback: trainer_state.json do checkpoint
        train_data, eval_data, etapas_curriculum = [], [], []
        trainer_state = GraficoTreinamento.carregar_trainer_state(chkpt_dir)
        if trainer_state:
            train_data, eval_data = GraficoTreinamento.extrair_metricas(trainer_state)
    
    if train_data or eval_data:
        logger.info("\n📈 Processando métricas de treinamento...")
        
        stats_report.append("\n## Métricas de Treinamento\n")
        stats_report.append(f"**Checkpoints encontrados:** {len(checkpoints)}\n")
        if etapas_curriculum:
            nomes = [et["alias"] for et in etapas_curriculum]
            stats_report.append(f"**Etapas do curriculum:** {' → '.join(nomes)}\n")
        
        # Tabela de Loss por Step
        stats_report.append("\n### Evolução do Loss\n")
        tabela_loss = GraficoTreinamento.tabela_loss_markdown(train_data, eval_data)
        stats_report.extend(tabela_loss)
        
        # Gera gráfico de Loss
        logger.info("   📊 Gerando gráfico de loss...")
        loss_graph_path = os.path.join(report_dir, "treinamento_loss.png")
        
        if GraficoTreinamento.evolucao_loss(
            train_data, eval_data, checkpoints, loss_graph_path,
            etapas_curriculum=etapas_curriculum
        ):
            logger.info("   ✅ Gráfico de loss salvo: treinamento_loss.png")
            stats_report.append(f"\n### Gráfico de Evolução do Loss\n")
            stats_report.append(f"![Loss de Treinamento](treinamento_loss.png)\n")
            legenda = "*Linhas verdes: fim de época | Cinzas: checkpoints"
            if etapas_curriculum and len(etapas_curriculum) > 1:
                legenda += " | Violeta: transição de etapa curriculum"
            legenda += "*\n"
            stats_report.append(legenda)
        else:
            logger.warning("   ⚠️ Erro ao gerar gráfico de loss.")
    else:
        logger.info("\n📊 Nenhum dado de loss encontrado (sem training_metrics.jsonl nem checkpoints).")

    # ==========================================================================
    # MÉTRICAS DE HARDWARE (RAM, GPU, CPU)
    # ==========================================================================
    from treinar_unsloth_graficos import GraficoHardware
    
    treinamento_dir = os.path.join(yaml_config.modelo.saida, "treinamento")
    hardware_metricas = GraficoHardware.carregar_metricas(treinamento_dir)
    
    if hardware_metricas:
        logger.info("\n📊 Processando métricas de hardware...")
        
        # Adiciona seção ao relatório
        stats_report.append("\n## Métricas de Hardware\n")
        stats_report.append(f"**Amostras coletadas:** {len(hardware_metricas)}\n")
        
        # Tabela resumo
        stats_report.append("\n### Resumo de Uso de Recursos\n")
        tabela_hw = GraficoHardware.tabela_resumo_markdown(hardware_metricas)
        stats_report.extend(tabela_hw)
        
        # Gera gráfico de memória
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
    
def executar_reset(yaml_path: str, confirmar: bool = True) -> bool:
    """
    Limpa o treinamento atual (checkpoints e modelo LoRA).
    
    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
        confirmar: Se True, pede confirmação antes de apagar
        
    Returns:
        True se limpou, False se cancelou ou não havia nada para limpar
    """
    from treinar_unsloth_util import YamlTreinamento
    
    logger.info("\n")
    log_separador(caractere="=", largura=80)
    logger.info(">> MODO RESET - LIMPAR TREINAMENTO ATUAL")
    log_separador(caractere="=", largura=80)
    
    # Carrega configuração
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=False)
    _exibir_cabecalho_modelo(yaml_config)
    
    output_dir = yaml_config.modelo.saida
    
    # Verifica o que existe para limpar
    tem_modelo = _verificar_modelo_treinado(yaml_config)
    tem_checkpoints, qtd_checkpoints = _verificar_checkpoints_existem(yaml_config)
    
    if not tem_modelo and not tem_checkpoints:
        logger.info("\n✅ Nada para limpar - não existe modelo treinado nem checkpoints.")
        return False
    
    # Exibe o que será removido
    logger.info("\n⚠️  Os seguintes itens serão REMOVIDOS:")
    if tem_modelo:
        logger.info(f"   • Modelo LoRA treinado em: {output_dir}")
    if tem_checkpoints:
        checkpoint_dir = os.path.join(output_dir, "chkpt")
        logger.info(f"   • {qtd_checkpoints} checkpoint(s) em: {checkpoint_dir}")
    
    # Pede confirmação
    if confirmar:
        if not _perguntar_confirmacao("\n❓ Deseja continuar com a limpeza?", padrao=False):
            logger.info("Operação cancelada.")
            return False
    
    # Executa limpeza
    logger.info("\n🗑️  Limpando...")
    
    try:
        # Remove checkpoints
        if tem_checkpoints:
            checkpoint_dir = os.path.join(output_dir, "chkpt")
            shutil.rmtree(checkpoint_dir)
            logger.info(f"   ✅ Checkpoints removidos: {checkpoint_dir}")
        
        # Remove modelo LoRA (apenas arquivos do adapter, não a pasta inteira)
        if tem_modelo:
            arquivos_lora = [
                'adapter_config.json',
                'adapter_model.safetensors',
                'pytorch_model.bin',
                'training_args.bin',
                'trainer_state.json',
                'tokenizer_config.json',
                'special_tokens_map.json',
                'tokenizer.json',
            ]
            for arq in arquivos_lora:
                arq_path = os.path.join(output_dir, arq)
                if os.path.exists(arq_path):
                    os.remove(arq_path)
                    logger.debug(f"   Removido: {arq}")
            logger.info(f"   ✅ Modelo LoRA removido de: {output_dir}")
            
    except Exception as e:
        logger.error(f"❌ Erro ao limpar: {e}")
        return False
    
    log_separador(caractere="=", largura=80)
    logger.info("✅ RESET COMPLETO - TREINAMENTO LIMPO")
    log_separador(caractere="=", largura=80)
    
    return True


def executar_treinar(yaml_path: str, reset: bool = False) -> None:
    """
    Executa o treinamento do modelo.
    
    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
        reset: Se True, limpa treinamento anterior antes de iniciar
    """
    # Importa aqui para evitar imports circulares
    from treinar_unsloth import LLMsTrainer
    
    logger.info("\n")
    log_separador(caractere="=", largura=80)
    logger.info(">> MODO TREINAR - EXECUTANDO TREINAMENTO")
    log_separador(caractere="=", largura=80)
    
    # Executa reset se solicitado
    if reset:
        if not executar_reset(yaml_path, confirmar=True):
            # Se cancelou o reset ou não havia nada para limpar, continua
            pass
    
    # Inicializa e executa treinamento
    trainer = LLMsTrainer(yaml_path)
    trainer.train()
    
    # Libera memória para permitir execução subsequente (ex: predict)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    log_separador(caractere="=", largura=80)
    logger.info("✅ TREINAMENTO COMPLETO")
    log_separador(caractere="=", largura=80)

