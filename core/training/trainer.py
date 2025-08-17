import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..data.utils import generate_text, text_to_token_ids, token_ids_to_text


@dataclass
class TrainingMetrics:
    """Container para métricas de treinamento"""

    train_loss: float
    val_loss: float
    tokens_seen: int
    global_step: int
    epoch: int


class TrainingCallback:
    """Classe base para callbacks de treinamento"""

    def on_epoch_begin(self, epoch: int, trainer: "GPTTrainer") -> None:
        """Chamado no início de cada época"""
        pass

    def on_epoch_end(self, epoch: int, trainer: "GPTTrainer") -> None:
        """Chamado no final de cada época"""
        pass

    def on_step_end(self, step: int, trainer: "GPTTrainer") -> None:
        """Chamado após cada step de treinamento"""
        pass

    def on_evaluation(self, metrics: TrainingMetrics, trainer: "GPTTrainer") -> None:
        """Chamado após cada avaliação"""
        pass


class LoggingCallback(TrainingCallback):
    """Callback para logging das métricas"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def on_evaluation(self, metrics: TrainingMetrics, trainer: "GPTTrainer") -> None:
        self.logger.info(
            f"Epoch {metrics.epoch+1} (Step {metrics.global_step:06d}): "
            f"Train loss {metrics.train_loss:.4f}, "
            f"Val loss {metrics.val_loss:.4f}, "
            f"Tokens seen {metrics.tokens_seen:,}"
        )


class GPTTrainer:
    """
    Trainer melhorado para modelos GPT com sistema de callbacks e melhor estrutura
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        eval_freq: int = 100,
        eval_iter: Optional[int] = None,
        device: Optional[torch.device] = None,
        start_context: str = "Every effort moves you",
        tokenizer=None,
        callbacks: Optional[List[TrainingCallback]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        # Validação de entrada
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if callbacks is None:
            callbacks = []

        # Configuração de logging
        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Atributos principais
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.eval_freq = eval_freq
        self.eval_iter = eval_iter
        self.start_context = start_context
        self.tokenizer = tokenizer

        # Callbacks
        self.callbacks = callbacks
        if not any(isinstance(cb, LoggingCallback) for cb in callbacks):
            self.callbacks.append(LoggingCallback(self.logger))

        # Estado de treinamento
        self.reset_training_state()

        # Mover modelo para device
        self.model.to(self.device)

        self.logger.info(f"GPTTrainer inicializado com device: {self.device}")
        self.logger.info(
            f"Modelo tem {sum(p.numel() for p in self.model.parameters()):,} parâmetros"
        )

    def reset_training_state(self) -> None:
        """Reseta o estado de treinamento"""
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.track_tokens_seen: List[int] = []
        self.tokens_seen = 0
        self.global_step = 0

    @contextmanager
    def _model_context(self, training: bool = True):
        """Context manager para gerenciar o estado do modelo"""
        was_training = self.model.training
        try:
            if training:
                self.model.train()
            else:
                self.model.eval()
            yield
        finally:
            if was_training:
                self.model.train()
            else:
                self.model.eval()

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computa a loss entre logits e targets"""
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    def train_step(self, input_batch: torch.Tensor, target_batch: torch.Tensor) -> float:
        """
        Executa um step de treinamento

        Args:
            input_batch: Batch de entrada
            target_batch: Batch de targets

        Returns:
            Loss do step
        """
        # Mover dados para device
        inputs = input_batch.to(self.device)
        targets = target_batch.to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        loss = self._compute_batch_loss(inputs, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _compute_batch_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computa a loss para um batch"""
        logits = self.model(inputs)
        return self.compute_loss(logits, targets)

    def train(self) -> Tuple[List[float], List[float], List[int]]:
        """
        Executa o treinamento completo

        Returns:
            Tuple com (train_losses, val_losses, track_tokens_seen)
        """
        self.logger.info(f"Iniciando treinamento por {self.epochs} épocas")

        for epoch in range(self.epochs):
            # Callback de início de época
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, self)

            # Treinamento da época
            self._train_epoch(epoch)

            # Callback de fim de época
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self)

            # Geração de exemplo
            self._generate_sample()

        self.logger.info("Treinamento concluído")
        return self.train_losses, self.val_losses, self.track_tokens_seen

    def _train_epoch(self, epoch: int) -> None:
        """Treina uma época completa"""
        with self._model_context(training=True):
            for input_batch, target_batch in self.train_loader:
                # Step de treinamento
                loss = self.train_step(input_batch, target_batch)

                # Atualizar contadores
                self.tokens_seen += input_batch.numel()
                self.global_step += 1

                # Avaliação periódica
                if self.global_step % self.eval_freq == 0:
                    self._evaluate_and_log(epoch)

                # Callback de step
                for callback in self.callbacks:
                    callback.on_step_end(self.global_step, self)

    def _evaluate_and_log(self, epoch: int) -> None:
        """Executa avaliação e logging"""
        train_loss, val_loss = self.evaluate_model()

        # Armazenar métricas
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.track_tokens_seen.append(self.tokens_seen)

        # Criar objeto de métricas
        metrics = TrainingMetrics(
            train_loss=train_loss,
            val_loss=val_loss,
            tokens_seen=self.tokens_seen,
            global_step=self.global_step,
            epoch=epoch,
        )

        # Notificar callbacks
        for callback in self.callbacks:
            callback.on_evaluation(metrics, self)

    def evaluate_model(self) -> Tuple[float, float]:
        """
        Avalia o modelo no conjunto de treino e validação

        Returns:
            Tuple com (train_loss, val_loss)
        """
        self.logger.info("Avaliando modelo...")

        with self._model_context(training=False):
            with torch.no_grad():
                train_loss = self._compute_loader_loss(self.train_loader, self.eval_iter)
                val_loss = self._compute_loader_loss(self.validation_loader, self.eval_iter)

        return train_loss, val_loss

    def _compute_loader_loss(
        self, data_loader: torch.utils.data.DataLoader, num_batches: Optional[int] = None
    ) -> float:
        """
        Computa a loss média para um data loader

        Args:
            data_loader: DataLoader para avaliar
            num_batches: Número de batches para avaliar (None para todos)

        Returns:
            Loss média
        """
        if len(data_loader) == 0:
            return float("nan")

        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        total_loss = 0.0
        batch_count = 0

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break

            loss = self._compute_batch_loss(
                input_batch.to(self.device), target_batch.to(self.device)
            )
            total_loss += loss.item()
            batch_count += 1

        return total_loss / batch_count if batch_count > 0 else float("nan")

    def _generate_sample(self) -> None:
        """Gera e imprime um exemplo de texto"""
        if not self.tokenizer:
            self.logger.warning("Tokenizer não disponível, pulando geração de exemplo")
            return

        try:
            with self._model_context(training=False):
                context_size = self.model.pos_emb_layer.weight.shape[0]

                # Codificar contexto inicial
                encoded = text_to_token_ids(self.start_context, self.tokenizer).to(self.device)
                encoded = encoded.unsqueeze(0)

                # Gerar texto
                with torch.no_grad():
                    token_ids = generate_text(
                        model=self.model,
                        idx=encoded,
                        max_new_tokens=50,
                        context_size=context_size,
                        temperature=0.0,
                        top_k=3,
                    )

                # Decodificar e imprimir
                decoded_text = token_ids_to_text(token_ids.squeeze(0), self.tokenizer)
                self.logger.info(f"Exemplo gerado: {decoded_text.replace(chr(10), ' ')}")

        except Exception as e:
            self.logger.error(f"Erro ao gerar exemplo: {e}")

    def get_training_state(self) -> Dict[str, Any]:
        """Retorna o estado atual do treinamento"""
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "track_tokens_seen": self.track_tokens_seen,
            "tokens_seen": self.tokens_seen,
            "global_step": self.global_step,
            "epochs": self.epochs,
        }

    def add_callback(self, callback: TrainingCallback) -> None:
        """Adiciona um callback ao trainer"""
        self.callbacks.append(callback)

    def remove_callback(self, callback: TrainingCallback) -> None:
        """Remove um callback do trainer"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
