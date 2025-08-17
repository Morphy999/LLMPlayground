import torch
import torch.nn.functional as F

from .DataUtils.utils import generate_text_simple, text_to_token_ids, token_ids_to_text


class GPTTrainer:
    def __init__(
        self,
        model,
        epochs,
        eval_freq,
        eval_iter,
        train_loader,
        validation_loader,
        optimizer,
        device=None,
        start_context: str = "Every effort moves you",
        tokenizer=None,
    ):
        self.start_context = start_context
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.eval_freq = eval_freq
        self.eval_iter = eval_iter
        self.train_losses, self.val_losses, self.track_tokens_seen = [], [], []
        self.tokens_seen, self.global_step = 0, -1
        self.tokenizer = tokenizer

    def compute_loss(self, logits, targets):
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    def train_step(self, input_batch, target_batch):

        inputs = input_batch.to(self.device)

        targets = target_batch.to(self.device)

        self.optimizer.zero_grad()

        loss = self.calc_loss_batch(inputs, targets)

        loss.backward()

        self.optimizer.step()

    def train(self):

        for epoch in range(self.epochs):
            self.model.train()
            for input_batch, target_batch in self.train_loader:

                self.train_step(input_batch, target_batch)

                self.tokens_seen += input_batch.numel()
                self.global_step += 1

                if self.global_step % self.eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model()
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)
                    self.track_tokens_seen.append(self.tokens_seen)
                    print(
                        f"Ep {epoch+1} (Step {self.global_step:06d}): "
                        f"Train loss {self.train_losses[-1]:.3f}, "
                        f"Val loss {self.val_losses[-1]:.3f}"
                    )

            self.generate_and_print_sample(self.start_context)

        return self.train_losses, self.val_losses, self.track_tokens_seen

    def evaluate_model(self):
        self.model.eval()
        print("Evaluating model...")
        with torch.no_grad():
            train_loss = self.calc_loss_loader(self.train_loader, self.eval_iter)
            val_loss = self.calc_loss_loader(self.validation_loader, self.eval_iter)

        self.model.train()
        return train_loss, val_loss

    def generate_and_print_sample(self, start_context: str):
        self.model.eval()
        context_size = self.model.pos_emb_layer.weight.shape[0]

        encoded = text_to_token_ids(start_context, self.tokenizer).to(self.device)

        encoded = encoded.unsqueeze(0)

        with torch.no_grad():
            token_ids = generate_text_simple(
                model=self.model, idx=encoded, max_new_tokens=50, context_size=context_size
            )

        decoded_text = token_ids_to_text(token_ids.squeeze(0), self.tokenizer)

        print(decoded_text.replace("\n", " "))

        self.model.train()

    def calc_loss_loader(self, data_loader, num_batches=None):
        total_loss = 0.0
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
            for i, (input_batch, target_batch) in enumerate(data_loader):
                if i < num_batches:
                    loss = self.calc_loss_batch(input_batch, target_batch)
                    total_loss += loss.item()
                else:
                    break
        return total_loss / num_batches

    def calc_loss_batch(self, input_batch, target_batch):

        input_batch = input_batch.to(self.device)

        target_batch = target_batch.to(self.device)

        logits = self.model(input_batch)

        loss = self.compute_loss(logits, target_batch)

        return loss
