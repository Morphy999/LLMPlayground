"""
Métricas para avaliação do modelo GPT
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch


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


def plot_training_curves(
    train_losses: List[float], val_losses: List[float], save_path: str = None
) -> None:
    """
    Plota curvas de treinamento

    Args:
        train_losses: Losses de treinamento
        val_losses: Losses de validação
        save_path: Caminho para salvar o gráfico
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss", alpha=0.8)
        plt.plot(val_losses, label="Validation Loss", alpha=0.8)
        plt.xlabel("Evaluation Steps")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    except ImportError:
        print("Matplotlib não está instalado. Instale com: pip install matplotlib")
