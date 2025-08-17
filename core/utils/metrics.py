"""
Métricas para avaliação do modelo GPT
"""

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator


def compute_loss_metrics(train_losses: List[float], val_losses: List[float]) -> Dict[str, float]:
    """
    Computa métricas de loss

    Args:
        train_losses: Lista de losses de treinamento
        val_losses: Lista de losses de validação

    Returns:
        Dicionário com métricas
    """

    if not train_losses or not val_losses:
        return {}

    metrics = {
        "train_loss_final": train_losses[-1],
        "val_loss_final": val_losses[-1],
        "train_loss_min": min(train_losses),
        "val_loss_min": min(val_losses),
        "train_loss_mean": np.mean(train_losses),
        "val_loss_mean": np.mean(val_losses),
        "overfitting_ratio": train_losses[-1] / val_losses[-1]
        if val_losses[-1] > 0
        else float("inf"),
    }

    return metrics


def compute_perplexity(loss: float) -> float:
    """
    Computa perplexidade a partir do loss

    Args:
        loss: Loss do modelo

    Returns:
        Perplexidade
    """
    return torch.exp(torch.tensor(loss)).item()


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computa acurácia do modelo

    Args:
        logits: Logits do modelo (batch_size, seq_len, vocab_size)
        targets: Targets reais (batch_size, seq_len)

    Returns:
        Acurácia
    """
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def compute_metrics_batch(
    logits: torch.Tensor, targets: torch.Tensor, loss: float
) -> Dict[str, float]:
    """
    Computa todas as métricas para um batch

    Args:
        logits: Logits do modelo
        targets: Targets reais
        loss: Loss computado

    Returns:
        Dicionário com todas as métricas
    """

    metrics = {
        "loss": loss,
        "perplexity": compute_perplexity(loss),
        "accuracy": compute_accuracy(logits, targets),
    }

    return metrics


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()
