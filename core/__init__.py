"""
LLM Playground - Core Package
Uma implementação educacional do modelo GPT em PyTorch
"""

from .GPTModel import GPTModel
from .Train import GPTTrainer

__version__ = "1.0.0"

__all__ = [
    "GPTModel",
    "GPTTrainer",
]
