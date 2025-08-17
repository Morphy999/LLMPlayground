"""
Callbacks úteis para o GPTTrainer
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .trainer import GPTTrainer, TrainingCallback, TrainingMetrics


class EarlyStoppingCallback(TrainingCallback):
    """
    Callback para early stopping baseado na loss de validação
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def on_evaluation(self, metrics: TrainingMetrics, trainer: "GPTTrainer") -> None:
        if metrics.val_loss < self.best_loss - self.min_delta:
            self.best_loss = metrics.val_loss
            self.counter = 0
            if self.restore_best:
                self.best_state = trainer.model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best and self.best_state is not None:
                trainer.model.load_state_dict(self.best_state)
            trainer.logger.info(f"Early stopping at epoch {metrics.epoch + 1}")
            # Aqui você poderia implementar uma forma de parar o treinamento
            # Por exemplo, definindo um flag ou usando uma exceção


class ModelCheckpointCallback(TrainingCallback):
    """
    Callback para salvar checkpoints do modelo
    """

    def __init__(
        self, save_dir: str = "./checkpoints", save_freq: int = 1, save_best_only: bool = True
    ):
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_best_only = save_best_only
        self.best_loss = float("inf")

        # Criar diretório se não existir
        import os

        os.makedirs(save_dir, exist_ok=True)

    def on_evaluation(self, metrics: TrainingMetrics, trainer: "GPTTrainer") -> None:
        # Salvar checkpoint periódico
        if metrics.epoch % self.save_freq == 0:
            self._save_checkpoint(trainer, metrics, f"epoch_{metrics.epoch + 1}")

        # Salvar melhor modelo
        if self.save_best_only and metrics.val_loss < self.best_loss:
            self.best_loss = metrics.val_loss
            self._save_checkpoint(trainer, metrics, "best")

    def _save_checkpoint(self, trainer: "GPTTrainer", metrics: TrainingMetrics, name: str) -> None:
        import os

        checkpoint = {
            "epoch": metrics.epoch,
            "global_step": metrics.global_step,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "train_loss": metrics.train_loss,
            "val_loss": metrics.val_loss,
            "tokens_seen": metrics.tokens_seen,
            "training_state": trainer.get_training_state(),
        }

        filepath = os.path.join(self.save_dir, f"{name}.pt")
        torch.save(checkpoint, filepath)
        trainer.logger.info(f"Checkpoint salvo: {filepath}")


class LearningRateSchedulerCallback(TrainingCallback):
    """
    Callback para agendar a learning rate
    """

    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler):
        self.scheduler = scheduler

    def on_step_end(self, step: int, trainer: "GPTTrainer") -> None:
        self.scheduler.step()

    def on_evaluation(self, metrics: TrainingMetrics, trainer: "GPTTrainer") -> None:
        # Alguns schedulers são chamados por época
        if hasattr(self.scheduler, "step") and callable(getattr(self.scheduler, "step", None)):
            self.scheduler.step()


class ProgressBarCallback(TrainingCallback):
    """
    Callback para mostrar barra de progresso (requer tqdm)
    """

    def __init__(self):
        try:
            from tqdm import tqdm

            self.tqdm = tqdm
            self.available = True
        except ImportError:
            self.available = False
            print("tqdm não disponível. Instale com: pip install tqdm")

    def on_epoch_begin(self, epoch: int, trainer: "GPTTrainer") -> None:
        if self.available:
            self.pbar = self.tqdm(
                total=len(trainer.train_loader),
                desc=f"Epoch {epoch + 1}/{trainer.epochs}",
                leave=True,
            )

    def on_step_end(self, step: int, trainer: "GPTTrainer") -> None:
        if self.available and hasattr(self, "pbar"):
            self.pbar.update(1)

    def on_epoch_end(self, epoch: int, trainer: "GPTTrainer") -> None:
        if self.available and hasattr(self, "pbar"):
            self.pbar.close()


class MetricsPlottingCallback(TrainingCallback):
    """
    Callback para plotar métricas de treinamento
    """

    def __init__(self, plot_freq: int = 5, save_plots: bool = True, save_dir: str = "./plots"):
        self.plot_freq = plot_freq
        self.save_plots = save_plots
        self.save_dir = save_dir

        if self.save_plots:
            import os

            os.makedirs(save_dir, exist_ok=True)

    def on_evaluation(self, metrics: TrainingMetrics, trainer: "GPTTrainer") -> None:
        if len(trainer.train_losses) % self.plot_freq == 0:
            self._plot_metrics(trainer)

    def _plot_metrics(self, trainer: "GPTTrainer") -> None:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Plot das losses
            ax1.plot(trainer.train_losses, label="Train Loss", color="blue")
            ax1.plot(trainer.val_losses, label="Val Loss", color="red")
            ax1.set_xlabel("Evaluation Step")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training and Validation Loss")
            ax1.legend()
            ax1.grid(True)

            # Plot de tokens vistos
            ax2.plot(trainer.track_tokens_seen, label="Tokens Seen", color="green")
            ax2.set_xlabel("Evaluation Step")
            ax2.set_ylabel("Tokens")
            ax2.set_title("Tokens Processed")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()

            if self.save_plots:
                import os

                filepath = os.path.join(self.save_dir, f"metrics_step_{trainer.global_step}.png")
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                trainer.logger.info(f"Plot salvo: {filepath}")

            plt.show()

        except Exception as e:
            trainer.logger.warning(f"Erro ao plotar métricas: {e}")


class GradientClippingCallback(TrainingCallback):
    """
    Callback para clipping de gradientes
    """

    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm

    def on_step_end(self, step: int, trainer: "GPTTrainer") -> None:
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), self.max_norm)


class MemoryMonitoringCallback(TrainingCallback):
    """
    Callback para monitorar uso de memória (GPU)
    """

    def __init__(self, log_freq: int = 100):
        self.log_freq = log_freq

    def on_step_end(self, step: int, trainer: "GPTTrainer") -> None:
        if step % self.log_freq == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(trainer.device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(trainer.device) / 1024**3  # GB
            trainer.logger.info(
                f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
            )
