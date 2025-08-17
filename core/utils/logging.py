"""
Sistema de logging para o LLM Playground
"""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None, log_dir: str = "logs"
) -> logging.Logger:
    """
    Configura o sistema de logging

    Args:
        log_level: Nível de logging (DEBUG, INFO, WARNING, ERROR)
        log_file: Nome do arquivo de log (opcional)
        log_dir: Diretório para salvar logs

    Returns:
        Logger configurado
    """

    # Criar diretório de logs se não existir
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Configurar logger
    logger = logging.getLogger("LLMPlayground")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Limpar handlers existentes
    logger.handlers.clear()

    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Formato do log
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para arquivo (se especificado)
    if log_file:
        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "LLMPlayground") -> logging.Logger:
    """
    Retorna um logger específico

    Args:
        name: Nome do logger

    Returns:
        Logger configurado
    """
    return logging.getLogger(name)


# Logger padrão
logger = get_logger()
