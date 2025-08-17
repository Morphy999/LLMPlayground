"""
Configurações pré-definidas dos modelos GPT
"""

from ..utils.config import GPTConfig

# Configurações baseadas no GPT-2
GPT2_TINY = GPTConfig(
    vocab_size=50257, emb_dim=384, context_length=512, n_layers=6, n_heads=6, dropout=0.1
)

GPT2_SMALL = GPTConfig(
    vocab_size=50257, emb_dim=768, context_length=1024, n_layers=12, n_heads=12, dropout=0.1
)

GPT2_MEDIUM = GPTConfig(
    vocab_size=50257, emb_dim=1024, context_length=1024, n_layers=24, n_heads=16, dropout=0.1
)

GPT2_LARGE = GPTConfig(
    vocab_size=50257, emb_dim=1280, context_length=1024, n_layers=36, n_heads=20, dropout=0.1
)

# Configurações customizadas
GPT_CUSTOM_SMALL = GPTConfig(
    vocab_size=50257, emb_dim=512, context_length=512, n_layers=8, n_heads=8, dropout=0.1
)

GPT_CUSTOM_MEDIUM = GPTConfig(
    vocab_size=50257, emb_dim=768, context_length=1024, n_layers=16, n_heads=12, dropout=0.1
)

# Dicionário de todas as configurações
ALL_CONFIGS = {
    "gpt2_tiny": GPT2_TINY,
    "gpt2_small": GPT2_SMALL,
    "gpt2_medium": GPT2_MEDIUM,
    "gpt2_large": GPT2_LARGE,
    "gpt_custom_small": GPT_CUSTOM_SMALL,
    "gpt_custom_medium": GPT_CUSTOM_MEDIUM,
}


def get_config_by_name(name: str) -> GPTConfig:
    """
    Retorna configuração pelo nome

    Args:
        name: Nome da configuração

    Returns:
        Configuração do modelo

    Raises:
        ValueError: Se o nome não for encontrado
    """
    if name not in ALL_CONFIGS:
        available = list(ALL_CONFIGS.keys())
        raise ValueError(f"Configuração '{name}' não encontrada. Disponíveis: {available}")

    return ALL_CONFIGS[name]


def list_available_configs() -> list:
    """
    Lista todas as configurações disponíveis

    Returns:
        Lista de nomes de configurações
    """
    return list(ALL_CONFIGS.keys())
