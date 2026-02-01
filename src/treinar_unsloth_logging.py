#!/usr/bin/env python3
"""
Módulo centralizado de logging para o projeto treinar_unsloth.

Fornece:
- Logger configurável com níveis (DEBUG, INFO, WARNING, ERROR)
- Formatação consistente com timestamps
- Suporte para saída em arquivo e console
- Configuração via variável de ambiente ou parâmetro

Uso:
    from treinar_unsloth_logging import get_logger, configurar_logging
    
    # Configura nível global (opcional, padrão INFO)
    configurar_logging(nivel="DEBUG")
    
    # Obtém logger para módulo
    logger = get_logger(__name__)
    
    logger.info("Mensagem informativa")
    logger.debug("Mensagem de debug (só aparece se nível >= DEBUG)")
    logger.warning("Mensagem de aviso")
    logger.error("Mensagem de erro")
"""

import logging
import os
import sys
from typing import Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Nome do logger raiz do projeto
LOGGER_NAME = "treinar_unsloth"

# Variável de ambiente para configurar nível de log
ENV_LOG_LEVEL = "UNSLOTH_LOG_LEVEL"

# Níveis de log válidos
NIVEIS_VALIDOS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Formato padrão das mensagens
FORMATO_CONSOLE = "%(message)s"
FORMATO_CONSOLE_DEBUG = "[%(levelname).1s] %(message)s"
FORMATO_ARQUIVO = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


# ---------------------------------------------------------------------------
# Configuração global
# ---------------------------------------------------------------------------

_logger_configurado = False
_nivel_global = logging.INFO
_arquivo_log: Optional[str] = None


def _get_nivel_from_env() -> int:
    """Obtém nível de log da variável de ambiente."""
    nivel_env = os.getenv(ENV_LOG_LEVEL, "").upper()
    return NIVEIS_VALIDOS.get(nivel_env, logging.INFO)


def configurar_logging(
    nivel: str = "INFO",
    arquivo: Optional[str] = None,
    formato_debug: bool = None
) -> None:
    """
    Configura o sistema de logging global do projeto.
    
    Args:
        nivel: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        arquivo: Caminho opcional para arquivo de log
        formato_debug: Se True, usa formato com prefixo de nível mesmo em INFO
    
    Exemplo:
        configurar_logging("DEBUG")  # Mostra todas as mensagens
        configurar_logging("WARNING")  # Só mostra warnings e erros
    """
    global _logger_configurado, _nivel_global, _arquivo_log
    
    # Determina nível (prioridade: parâmetro > env > INFO)
    nivel_env = _get_nivel_from_env()
    nivel_param = NIVEIS_VALIDOS.get(nivel.upper(), logging.INFO)
    
    # Usa o mais baixo (mais verbose) entre env e parâmetro
    _nivel_global = min(nivel_env, nivel_param)
    
    if arquivo:
        _arquivo_log = arquivo
    
    # Obtém ou cria logger raiz do projeto
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_nivel_global)
    logger.propagate = False  # Evita duplicação de logs (não propaga para root)
    
    # Remove handlers existentes para reconfiguração
    logger.handlers.clear()
    
    # Handler de console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(_nivel_global)
    
    # Formato depende do nível
    if _nivel_global <= logging.DEBUG or formato_debug:
        console_format = FORMATO_CONSOLE_DEBUG
    else:
        console_format = FORMATO_CONSOLE
    
    console_handler.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console_handler)
    
    # Handler de arquivo (se especificado)
    if _arquivo_log:
        os.makedirs(os.path.dirname(_arquivo_log) or ".", exist_ok=True)
        file_handler = logging.FileHandler(_arquivo_log, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Arquivo sempre recebe tudo
        file_handler.setFormatter(logging.Formatter(FORMATO_ARQUIVO))
        logger.addHandler(file_handler)
    
    _logger_configurado = True


def get_logger(name: str = None) -> logging.Logger:
    """
    Obtém um logger configurado para o módulo especificado.
    
    Args:
        name: Nome do módulo (use __name__ para obter automaticamente)
    
    Returns:
        Logger configurado
    
    Exemplo:
        logger = get_logger(__name__)
        logger.info("Processando dados...")
    """
    global _logger_configurado
    
    # Configura automaticamente se ainda não foi feito
    if not _logger_configurado:
        configurar_logging()
    
    # Cria logger filho do logger raiz do projeto
    if name:
        logger_name = f"{LOGGER_NAME}.{name}"
    else:
        logger_name = LOGGER_NAME
    
    return logging.getLogger(logger_name)


def set_nivel(nivel: str) -> None:
    """
    Altera dinamicamente o nível de log.
    
    Args:
        nivel: Novo nível (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global _nivel_global
    
    novo_nivel = NIVEIS_VALIDOS.get(nivel.upper(), logging.INFO)
    _nivel_global = novo_nivel
    
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(novo_nivel)
    
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(novo_nivel)


def get_nivel() -> str:
    """Retorna o nível de log atual como string."""
    for nome, valor in NIVEIS_VALIDOS.items():
        if valor == _nivel_global:
            return nome
    return "INFO"


# ---------------------------------------------------------------------------
# Utilitários de formatação
# ---------------------------------------------------------------------------

class LoggerContexto:
    """
    Context manager para adicionar contexto temporário às mensagens de log.
    
    Exemplo:
        with LoggerContexto("[GPU 0]"):
            logger.info("Alocando memória")  # Imprime: [GPU 0] Alocando memória
    """
    
    def __init__(self, prefixo: str):
        self.prefixo = prefixo
        self._original_format = None
    
    def __enter__(self):
        logger = logging.getLogger(LOGGER_NAME)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                self._original_format = handler.formatter._fmt
                novo_fmt = f"{self.prefixo} {self._original_format}"
                handler.setFormatter(logging.Formatter(novo_fmt))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_format:
            logger = logging.getLogger(LOGGER_NAME)
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setFormatter(logging.Formatter(self._original_format))


def log_separador(titulo: str = "", caractere: str = "=", largura: int = 60) -> None:
    """
    Imprime uma linha separadora para organização visual.
    
    Args:
        titulo: Texto centralizado na linha (opcional)
        caractere: Caractere usado para a linha
        largura: Largura total da linha
    """
    logger = get_logger()
    
    if titulo:
        padding = (largura - len(titulo) - 2) // 2
        linha = f"{caractere * padding} {titulo} {caractere * padding}"
        # Ajusta se largura for ímpar
        if len(linha) < largura:
            linha += caractere
    else:
        linha = caractere * largura
    
    logger.info(linha)


def log_bloco(titulo: str, conteudo: str, nivel: str = "INFO") -> None:
    """
    Imprime um bloco de informação formatado.
    
    Args:
        titulo: Título do bloco
        conteudo: Conteúdo do bloco
        nivel: Nível de log (DEBUG, INFO, etc)
    """
    logger = get_logger()
    nivel_log = NIVEIS_VALIDOS.get(nivel.upper(), logging.INFO)
    
    logger.log(nivel_log, f"\n{'='*60}")
    logger.log(nivel_log, f"📋 {titulo}")
    logger.log(nivel_log, f"{'-'*60}")
    for linha in conteudo.split("\n"):
        logger.log(nivel_log, f"  {linha}")
    logger.log(nivel_log, f"{'='*60}")
