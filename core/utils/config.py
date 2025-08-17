"""
Configurações centralizadas do LLM Playground
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GPTConfig:
    """Configuração do modelo GPT"""

    # Dimensões do modelo
    vocab_size: int = 50257
    emb_dim: int = 768
    context_length: int = 1024
    n_layers: int = 12
    n_heads: int = 12

    # Hiperparâmetros
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Configurações de treinamento
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GPTConfig":
        """Cria instância a partir de dicionário"""
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """Configuração de treinamento"""

    # Hiperparâmetros básicos
    batch_size: int = 32
    max_epochs: int = 100
    eval_freq: int = 100
    eval_iter: int = 10

    # Otimização
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000

    # Checkpointing
    save_freq: int = 1000
    checkpoint_dir: str = "checkpoints"

    # Logging
    log_freq: int = 10
    use_tensorboard: bool = True
    use_wandb: bool = False


def get_model_config(model_size: str = "small") -> GPTConfig:
    """Retorna configuração pré-definida baseada no tamanho do modelo"""

    configs = {
        "tiny": GPTConfig(
            vocab_size=50257, emb_dim=384, context_length=512, n_layers=6, n_heads=6
        ),
        "small": GPTConfig(
            vocab_size=50257, emb_dim=768, context_length=1024, n_layers=12, n_heads=12
        ),
        "medium": GPTConfig(
            vocab_size=50257, emb_dim=1024, context_length=1024, n_layers=24, n_heads=16
        ),
        "large": GPTConfig(
            vocab_size=50257, emb_dim=1280, context_length=1024, n_layers=36, n_heads=20
        ),
    }

    if model_size not in configs:
        raise ValueError(f"Tamanho de modelo inválido: {model_size}. Use: {list(configs.keys())}")

    return configs[model_size]
