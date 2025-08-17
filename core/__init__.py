"""
LLM Playground - Core Package
Uma implementação educacional do modelo GPT em PyTorch
"""

from .configs import GPT2_LARGE, GPT2_MEDIUM, GPT2_SMALL
from .data import GPTDataLoader, GPTDataset, generate_text_simple

# Imports principais
from .models import GPTModel
from .training import GPTTrainer
from .utils import GPTConfig, setup_logging

# Versão do pacote
__version__ = "1.0.0"

# O que será importado quando alguém fizer: from core import *
__all__ = [
    # Modelos
    "GPTModel",
    # Treinamento
    "GPTTrainer",
    # Dados
    "GPTDataset",
    "GPTDataLoader",
    "generate_text_simple",
    # Utilitários
    "GPTConfig",
    "setup_logging",
    # Configurações
    "GPT2_SMALL",
    "GPT2_MEDIUM",
    "GPT2_LARGE",
]
