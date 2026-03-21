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
    logger.info("<azul>📋 CONFIGURAÇÃO DO TREINAMENTO</azul>")
    log_separador(caractere="-", largura=70)
    logger.info(f"<cinza>  Modelo Base:  {yaml_config.modelo.base}</cinza>")
    logger.info(f"<cinza>  Modelo Saída: {yaml_config.modelo.saida}</cinza>")
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
        
    logger.info(f"<cinza>ℹ️  Injetando dicas no arquivo: {cfg_path}</cinza>")
    from treinar_unsloth_dicas import injetar_dicas_yaml, DICAS_YAML
    
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = injetar_dicas_yaml(content, DICAS_YAML)
        
        with open(cfg_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        logger.info(f"<verde>✅ Dicas injetadas com sucesso em '{cfg_path}'.</verde>")
    except Exception as e:
        logger.error(f"<vermelho>❌ Erro ao injetar dicas: {e}</vermelho>")
        import traceback
        traceback.print_exc()
        
    sys.exit(0)


# ---------------------------------------------------------------------------
# Funções de avaliação (info, stats, predict, merge, modelo) foram movidas
# para treinar_unsloth_avaliar.py
# ---------------------------------------------------------------------------

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
    logger.info("<azul>>> MODO RESET - LIMPAR TREINAMENTO ATUAL</azul>")
    log_separador(caractere="=", largura=80)
    
    # Carrega configuração
    yaml_config = YamlTreinamento(yaml_path, validar_caminhos=False)
    _exibir_cabecalho_modelo(yaml_config)
    
    output_dir = yaml_config.modelo.saida
    
    # Verifica o que existe para limpar
    tem_modelo = _verificar_modelo_treinado(yaml_config)
    tem_checkpoints, qtd_checkpoints = _verificar_checkpoints_existem(yaml_config)
    
    if not tem_modelo and not tem_checkpoints:
        logger.info("<verde>\n✅ Nada para limpar - não existe modelo treinado nem checkpoints.</verde>")
        return False
    
    # Exibe o que será removido
    logger.info("<amarelo>\n⚠️  Os seguintes itens serão REMOVIDOS:</amarelo>")
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
            removidos = 0
            for item in os.listdir(checkpoint_dir):
                item_path = os.path.join(checkpoint_dir, item)
                if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                    shutil.rmtree(item_path)
                    removidos += 1
                    logger.debug(f"   Removido: {item}")
            logger.info(f"<verde>   ✅ {removidos} checkpoint(s) removido(s) de: {checkpoint_dir}</verde>")
        
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
            logger.info(f"<verde>   ✅ Modelo LoRA removido de: {output_dir}</verde>")
        
        # Limpa arquivos de histórico (serão recriados no próximo treinamento)
        treino_dir = os.path.join(output_dir, "treinamento")
        config_dir = os.path.join(treino_dir, "treinamento_config")
        
        # Remove cópias versionadas do YAML (apenas arquivos .yaml/.yml)
        if os.path.isdir(config_dir):
            removidos_yaml = 0
            for arq in os.listdir(config_dir):
                if arq.lower().endswith(('.yaml', '.yml')):
                    os.remove(os.path.join(config_dir, arq))
                    removidos_yaml += 1
                    logger.debug(f"   Removido: {arq}")
            if removidos_yaml > 0:
                logger.info(f"<verde>   ✅ {removidos_yaml} cópia(s) de configuração removida(s)</verde>")
        
        # Remove arquivos de histórico que serão recriados
        arquivos_historico = [
            os.path.join(treino_dir, "treinamento_exemplos.md"),
            os.path.join(treino_dir, "modelo_info.md"),
            os.path.join(treino_dir, "treinamento_eventos.md"),
        ]
        for arq_h in arquivos_historico:
            if os.path.exists(arq_h):
                os.remove(arq_h)
                logger.debug(f"   Removido: {os.path.basename(arq_h)}")
        
        # Limpa logs de processamento antigos
        for log_file in ["metrics_stream.jsonl"]:
            log_path = os.path.join(output_dir, log_file)
            if os.path.exists(log_path):
                os.remove(log_path)
                logger.debug(f"   Removido: {log_file}")
        
        logger.info(f"<verde>   ✅ Histórico de treinamento limpo</verde>")
            
    except Exception as e:
        logger.error(f"<vermelho>❌ Erro ao limpar: {e}</vermelho>")
        return False
    
    log_separador(caractere="=", largura=80)
    logger.info("<verde>✅ RESET COMPLETO - TREINAMENTO LIMPO</verde>")
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
    logger.info("<azul>>> MODO TREINAR - EXECUTANDO TREINAMENTO</azul>")
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
    logger.info("<verde>✅ TREINAMENTO COMPLETO</verde>")
    log_separador(caractere="=", largura=80)

