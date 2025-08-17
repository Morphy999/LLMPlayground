"""
Configurações de treinamento pré-definidas
"""

from ..utils.config import TrainingConfig

# Configuração padrão de treinamento
DEFAULT_TRAINING_CONFIG = TrainingConfig(
    batch_size=32,
    max_epochs=100,
    eval_freq=100,
    eval_iter=10,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=1000,
    save_freq=1000,
    checkpoint_dir="checkpoints",
    log_freq=10,
    use_tensorboard=True,
    use_wandb=False,
)

# Configuração para treinamento rápido (testes)
FAST_TRAINING_CONFIG = TrainingConfig(
    batch_size=16,
    max_epochs=10,
    eval_freq=50,
    eval_iter=5,
    learning_rate=1e-3,
    weight_decay=0.01,
    warmup_steps=100,
    save_freq=500,
    checkpoint_dir="checkpoints_fast",
    log_freq=5,
    use_tensorboard=False,
    use_wandb=False,
)

# Configuração para treinamento longo (produção)
PRODUCTION_TRAINING_CONFIG = TrainingConfig(
    batch_size=64,
    max_epochs=1000,
    eval_freq=500,
    eval_iter=20,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=5000,
    save_freq=2000,
    checkpoint_dir="checkpoints_production",
    log_freq=20,
    use_tensorboard=True,
    use_wandb=True,
)

# Configuração para fine-tuning
FINE_TUNING_CONFIG = TrainingConfig(
    batch_size=8,
    max_epochs=50,
    eval_freq=25,
    eval_iter=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    save_freq=500,
    checkpoint_dir="checkpoints_finetune",
    log_freq=10,
    use_tensorboard=True,
    use_wandb=False,
)

# Dicionário de todas as configurações de treinamento
ALL_TRAINING_CONFIGS = {
    "default": DEFAULT_TRAINING_CONFIG,
    "fast": FAST_TRAINING_CONFIG,
    "production": PRODUCTION_TRAINING_CONFIG,
    "fine_tuning": FINE_TUNING_CONFIG,
}


def get_training_config_by_name(name: str) -> TrainingConfig:
    """
    Retorna configuração de treinamento pelo nome

    Args:
        name: Nome da configuração

    Returns:
        Configuração de treinamento

    Raises:
        ValueError: Se o nome não for encontrado
    """
    if name not in ALL_TRAINING_CONFIGS:
        available = list(ALL_TRAINING_CONFIGS.keys())
        raise ValueError(
            f"Configuração de treinamento '{name}' não encontrada. Disponíveis: {available}"
        )

    return ALL_TRAINING_CONFIGS[name]


def list_available_training_configs() -> list:
    """
    Lista todas as configurações de treinamento disponíveis

    Returns:
        Lista de nomes de configurações
    """
    return list(ALL_TRAINING_CONFIGS.keys())
