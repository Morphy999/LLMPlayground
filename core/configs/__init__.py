"""
Configurações dos modelos e treinamento
"""

from .model_configs import GPT2_LARGE, GPT2_MEDIUM, GPT2_SMALL
from .training_configs import DEFAULT_TRAINING_CONFIG

__all__ = [
    "GPT2_SMALL",
    "GPT2_MEDIUM",
    "GPT2_LARGE",
    "DEFAULT_TRAINING_CONFIG",
]
