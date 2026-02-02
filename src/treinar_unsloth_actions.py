#!/usr/bin/env python3
"""
Autor: Luiz Anisio 02/2026

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

# Configuração de path para permitir execução de qualquer diretório
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

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

def executar_info(yaml_path: str) -> None:
    """
    Exibe informações detalhadas sobre configuração e datasets.
    Substitui o antigo modo --debug.
    
    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
    """
    # Importa aqui para evitar imports circulares
    from treinar_unsloth import LLMsTrainer
    
    # LLMsTrainer.debug_info já imprime cabeçalhos e rodapés
    LLMsTrainer.debug_info(yaml_path)


def executar_stats(yaml_path: str) -> None:
    """
    Gera relatório estatístico com informações sobre uso de tokens.
    Gera tabelas e gráficos separados por subset (treino, validação, teste).
    
    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
    """
    from treinar_unsloth_util import YamlTreinamento, TIPO_ENTRADA_PASTAS
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
    
    if yaml_config.tipo_entrada == TIPO_ENTRADA_PASTAS:
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
    # MÉTRICAS DE TREINAMENTO (se houver checkpoints)
    # ==========================================================================
    from treinar_unsloth_graficos import GraficoTreinamento
    
    # Tenta encontrar checkpoints em chkpt/ ou no diretório raiz do modelo
    chkpt_dir = os.path.join(yaml_config.modelo.saida, "chkpt")
    if not os.path.exists(chkpt_dir) or not any(d.startswith("checkpoint-") for d in os.listdir(chkpt_dir) if os.path.isdir(os.path.join(chkpt_dir, d))):
        # Fallback: checkpoints no diretório raiz do modelo
        chkpt_dir = yaml_config.modelo.saida
    
    trainer_state = GraficoTreinamento.carregar_trainer_state(chkpt_dir)
    
    if trainer_state:
        logger.info("\n📈 Processando métricas de treinamento...")
        
        # Extrai métricas
        train_data, eval_data = GraficoTreinamento.extrair_metricas(trainer_state)
        checkpoints = GraficoTreinamento.listar_checkpoints(chkpt_dir)
        
        if train_data or eval_data:
            # Adiciona seção ao relatório
            stats_report.append("\n## Métricas de Treinamento\n")
            stats_report.append(f"**Checkpoints encontrados:** {len(checkpoints)}\n")
            stats_report.append(f"**Épocas treinadas:** {trainer_state.get('epoch', 0)}\n")
            stats_report.append(f"**Steps totais:** {trainer_state.get('global_step', 0)}\n")
            
            # Tabela de Loss por Step
            stats_report.append("\n### Evolução do Loss\n")
            tabela_loss = GraficoTreinamento.tabela_loss_markdown(train_data, eval_data)
            stats_report.extend(tabela_loss)
            
            # Gera gráfico de Loss
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
    
    log_separador(caractere="=", largura=80)
    logger.info("✅ STATS COMPLETO - RELATÓRIO GERADO")
    log_separador(caractere="=", largura=80)


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
    
    log_separador(caractere="=", largura=80)
    logger.info("✅ TREINAMENTO COMPLETO")
    log_separador(caractere="=", largura=80)


def executar_predict(yaml_path: str, subsets: list = None, usar_base: bool = False) -> None:
    """
    Gera predições do modelo para os subsets especificados.
    Salva os resultados na pasta {output_dir}/predict/{subset}/
    Se usar_base=True, salva em {output_dir}/predict_base/{subset}/
    
    Estrutura de saída:
        predict/{subset}/{id}.txt          - Resposta do modelo
        predict/{subset}/{id}.json         - Métricas de tokens e preview
        predict/{subset}/resumo.json       - Resumo do processamento
    
    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
        subsets: Lista de subsets para processar ('treino', 'validacao', 'teste').
                 Se None, processa todos.
        usar_base: Se True, usa o modelo base original em vez do treinado.
    """
    from treinar_unsloth_util import YamlTreinamento, TIPO_ENTRADA_PASTAS, FORMATO_SAIDA_JSON
    from util_prompt import Prompt, UtilLLM
    import json
    from datetime import datetime
    from time import time
    
    logger.info("\n")
    log_separador(caractere="=", largura=80)
    logger.info(">> MODO PREDICT - GERANDO PREDIÇÕES")
    log_separador(caractere="=", largura=80)
    
    # Carrega configuração
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=True)
    _exibir_cabecalho_modelo(yaml_config)
    
    # Verifica formato de saída
    formato_json = yaml_config.formato_saida == FORMATO_SAIDA_JSON
    logger.info(f"\n📋 Formato de saída: {yaml_config.formato_saida}")
    
    # Verifica se há modelo treinado
    modelo_path = yaml_config.modelo.saida
    tem_modelo_treinado = _verificar_modelo_treinado(yaml_config)
    
    if usar_base:
        logger.info(f"ℹ️  Opção --base ativada: Forçando uso do modelo base.")
        modelo_path = yaml_config.modelo.base
        # Se for usar base, não importa se tem modelo treinado ou não
    elif not tem_modelo_treinado:
        logger.warning("\n⚠️ Não foi encontrado modelo LoRA treinado.")
        if not _perguntar_confirmacao("Deseja usar o modelo base para predição?", padrao=False):
            logger.info("Operação cancelada.")
            return
        logger.info("Usando modelo base para predição...\n")
        modelo_path = yaml_config.modelo.base
    else:
        logger.info(f"✅ Usando modelo treinado: {modelo_path}")
    
    # Define subsets a processar
    if subsets is None:
        subsets = ['treino', 'validacao', 'teste']
    
    logger.info(f"\n📋 Subsets a processar: {', '.join(subsets)}")
    
    # Cria diretório de predições
    nome_pasta = "predict_base" if usar_base else "predict"
    predict_dir = os.path.join(yaml_config.modelo.saida, nome_pasta)
    os.makedirs(predict_dir, exist_ok=True)
    
    # Inicializa modelo usando classe Prompt
    logger.info("\n🔄 Carregando modelo...")
    ini_carga = time()
    
    try:
        # Usa Prompt para carregar o modelo (suporta LoRA)
        # Se usar_base=True, passamos usar_unsloth=False para garantir loading puro?
        # A classe Prompt com usar_unsloth=True carrega LoRA se o modelo for o diretório de saída
        # Como passamos modelo_path = base na opção --base, ele não vai carregar o LoRA do adapter.
        
        prompt_handler = Prompt(
            modelo=modelo_path,
            max_seq_length=yaml_config.treinamento.max_seq_length,
            usar_unsloth=True # Unsloth funciona com base model também
        )
        logger.info(f"   ✅ Modelo carregado em {time() - ini_carga:.1f}s")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        return
    
    # Estatísticas globais de uso de tokens
    uso_total = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_registros': 0,
        'tempo_total_s': 0,
        'por_subset': {}
    }
    
    # Processa cada subset
    for subset in subsets:
        logger.info(f"\n📂 Processando subset: {subset}")
        log_separador(caractere="-", largura=60)
        
        # Carrega dados do subset
        if yaml_config.tipo_entrada == TIPO_ENTRADA_PASTAS:
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
        
        # Estatísticas do subset
        subset_stats = {
            'input_tokens': 0,
            'output_tokens': 0,
            'registros_ok': 0,
            'registros_erro': 0,
            'tempo_s': 0
        }
        
        total = len(mensagens)
        ini_subset = time()
        
        for idx, msg in enumerate(mensagens):
            try:
                # Extrai prompt (mensagem do usuário)
                if isinstance(msg, dict) and 'messages' in msg:
                    messages = msg['messages']
                    
                    # Pega o conteúdo do user para o prompt
                    prompt_texto = ""
                    for m in messages:
                        if m.get('role') == 'user':
                            prompt_texto = m.get('content', '')
                            break
                    
                    # Pega resposta esperada (assistant) para comparação
                    resposta_esperada = ""
                    for m in messages:
                        if m.get('role') == 'assistant':
                            resposta_esperada = m.get('content', '')
                            break
                else:
                    logger.warning(f"   ⚠️ Formato não reconhecido no registro {idx}")
                    subset_stats['registros_erro'] += 1
                    continue
                
                if not prompt_texto:
                    logger.warning(f"   ⚠️ Prompt vazio no registro {idx}")
                    subset_stats['registros_erro'] += 1
                    continue
                
                # Gera predição usando Prompt
                registro_id = msg.get('id', f'{subset}_{idx:04d}')
                
                if formato_json:
                    # Usa prompt_to_json para saída JSON
                    resultado = prompt_handler.prompt_to_json(
                        prompt=prompt_texto,
                        max_new_tokens=yaml_config.treinamento.max_seq_length,
                        temperatura=0.02
                    )
                    
                    # Extrai resposta e usage
                    usage = resultado.pop('usage', {})
                    if 'erro' in resultado:
                        resposta_modelo = resultado.get('response', str(resultado))
                    else:
                        resposta_modelo = json.dumps(resultado, ensure_ascii=False, indent=2)
                else:
                    # Usa prompt normal para saída texto
                    resultado = prompt_handler.prompt(
                        prompt=prompt_texto,
                        max_new_tokens=yaml_config.treinamento.max_seq_length,
                        temperatura=0.02,
                        detalhar=True
                    )
                    
                    if isinstance(resultado, dict):
                        resposta_modelo = resultado.get('texto', str(resultado))
                        usage = {
                            'input_tokens': resultado.get('input_tokens', 0),
                            'output_tokens': resultado.get('output_tokens', 0),
                            'time': resultado.get('time', 0)
                        }
                    else:
                        resposta_modelo = str(resultado)
                        usage = {}
                
                # Salva resposta do modelo em .txt
                output_txt = os.path.join(subset_dir, f"{registro_id}.txt")
                with open(output_txt, 'w', encoding='utf-8') as f:
                    f.write(resposta_modelo)
                
                # Prepara preview do prompt (início...fim)
                tam_preview = 100
                if len(prompt_texto) > tam_preview * 2:
                    prompt_preview = f"{prompt_texto[:tam_preview]} [...] {prompt_texto[-tam_preview:]}"
                else:
                    prompt_preview = prompt_texto

                # Salva dados de uso de tokens em .json (mesmo nome do arquivo de resposta)
                usage_data = {
                    'id': registro_id,
                    'input_tokens': usage.get('input_tokens', 0),
                    'output_tokens': usage.get('output_tokens', 0),
                    'time_s': usage.get('time', 0),
                    'prompt_preview': prompt_preview,
                }
                output_json = os.path.join(subset_dir, f"{registro_id}.json")
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(usage_data, f, ensure_ascii=False, indent=2)
                
                # Atualiza estatísticas
                subset_stats['input_tokens'] += usage.get('input_tokens', 0)
                subset_stats['output_tokens'] += usage.get('output_tokens', 0)
                subset_stats['registros_ok'] += 1
                
                # Log de progresso
                if (idx + 1) % 10 == 0 or (idx + 1) == total:
                    logger.info(f"   Progresso: {idx + 1}/{total} ({100*(idx+1)//total}%)")
                    
            except Exception as e:
                logger.error(f"   ❌ Erro no registro {idx}: {e}")
                subset_stats['registros_erro'] += 1
                continue
        
        # Tempo do subset
        subset_stats['tempo_s'] = time() - ini_subset
        
        # Salva resumo do subset
        resumo_subset = {
            'subset': subset,
            'total_registros': total,
            'registros_ok': subset_stats['registros_ok'],
            'registros_erro': subset_stats['registros_erro'],
            'input_tokens_total': subset_stats['input_tokens'],
            'output_tokens_total': subset_stats['output_tokens'],
            'tempo_processamento_s': round(subset_stats['tempo_s'], 2),
            'data_geracao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'modelo': modelo_path,
            'formato_saida': yaml_config.formato_saida,
        }
        resumo_file = os.path.join(subset_dir, "resumo.json")
        with open(resumo_file, 'w', encoding='utf-8') as f:
            json.dump(resumo_subset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"   ✅ {subset_stats['registros_ok']} predições salvas em: {subset_dir}")
        logger.info(f"   📊 Tokens: {subset_stats['input_tokens']} entrada, {subset_stats['output_tokens']} saída")
        
        # Atualiza estatísticas globais
        uso_total['input_tokens'] += subset_stats['input_tokens']
        uso_total['output_tokens'] += subset_stats['output_tokens']
        uso_total['total_registros'] += subset_stats['registros_ok']
        uso_total['tempo_total_s'] += subset_stats['tempo_s']
        uso_total['por_subset'][subset] = resumo_subset
    
    # Salva resumo geral
    resumo_geral = {
        'data_geracao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'modelo_base': yaml_config.modelo.base,
        'modelo_saida': yaml_config.modelo.saida,
        'modelo_usado': modelo_path,
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



def modo_interativo(yaml_path: str) -> Optional[str]:
    """
    Modo interativo: exibe informações do modelo e pergunta qual ação executar.
    
    Args:
        yaml_path: Caminho para o arquivo YAML de configuração
        
    Returns:
        Nome da ação escolhida ou None se cancelou
    """
    from treinar_unsloth_util import YamlTreinamento
    
    # Carrega configuração
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=False)
    
    # Exibe cabeçalho
    _exibir_cabecalho_modelo(yaml_config)
    
    # Verifica status atual
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
    
    # Menu de ações
    logger.info("\n📋 AÇÕES DISPONÍVEIS:")
    logger.info("   1. info    - Informações detalhadas da configuração e datasets")
    logger.info("   2. stats   - Relatório estatístico de tokens com gráficos")
    logger.info("   3. treinar - Iniciar ou continuar treinamento")
    logger.info("   4. predict - Gerar predições para todos os subsets")
    logger.info("   5. reset   - Limpar treinamento atual")
    logger.info("   0. sair    - Cancelar e sair")
    
    try:
        escolha = input("\n❓ Digite o número ou nome da ação: ").strip().lower()
        
        mapa_acoes = {
            '1': 'info', 'info': 'info',
            '2': 'stats', 'stats': 'stats',
            '3': 'treinar', 'treinar': 'treinar', 'train': 'treinar',
            '4': 'predict', 'predict': 'predict',
            '5': 'reset', 'reset': 'reset',
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


def executar_acao(acao: str, yaml_path: str, reset: bool = False, predict_subsets: list = None) -> None:
    """
    Executa a ação especificada.
    
    Args:
        acao: Nome da ação ('info', 'stats', 'treinar', 'reset', 'predict')
        yaml_path: Caminho para o arquivo YAML
        reset: Se True e acao='treinar', limpa antes de treinar
        predict_subsets: Lista de subsets para predict (None = todos)
    """
    if acao == 'info':
        executar_info(yaml_path)
    elif acao == 'stats':
        executar_stats(yaml_path)
    elif acao == 'treinar':
        executar_treinar(yaml_path, reset=reset)
    elif acao == 'predict':
        executar_predict(yaml_path, subsets=predict_subsets)
    elif acao == 'reset':
        executar_reset(yaml_path, confirmar=True)
    else:
        logger.error(f"Ação desconhecida: '{acao}'")

