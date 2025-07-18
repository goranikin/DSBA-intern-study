from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch


class VaeTrainer:
    def __init__(
        self,
        model,
        optimizer,
        x_dim,
        loss_fn,
        epochs,
        device,
        train_loader: DataLoader,
        model_name: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.x_dim = x_dim
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device
        self.train_loader = train_loader

    def save_model(self, path="./saved_model/vae_model.pth"):
        torch.save(self.model.state_dict(), path)

    def train_epochs(self):
        self.model.train()

        for epoch in range(self.epochs):
            overall_loss = 0.0

            progress_bar = tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}"
            )

            for batch_idx, (x, _) in enumerate(progress_bar):
                x = x.view(self.train_loader.batch_size, self.x_dim).to(self.device)

                self.optimizer.zero_grad()

                x_hat, mean, log_variance = self.model(x)
                loss = self.loss_fn(x, x_hat, mean, log_variance)

                overall_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                avg_loss = overall_loss / (
                    (batch_idx + 1) * self.train_loader.batch_size
                )
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        print("Trainig completed.")

        save_path = "./saved_model/vae_model.pth"
        self.save_model(path=save_path)
        print(f"Model saved to {save_path}")
