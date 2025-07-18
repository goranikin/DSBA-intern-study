from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F


class VaeTrainer:
    def __init__(
        self,
        model,
        optimizer,
        x_dim,
        epochs,
        device,
        train_loader: DataLoader,
        model_name: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.x_dim = x_dim
        self.epochs = epochs
        self.device = device
        self.train_loader = train_loader
        self.model_name = model_name

    def save_model(self, path="./saved_model/vq_vae_model"):
        torch.save(self.model.state_dict(), path + self.model_name + ".pth")

    def train_epochs(self):
        self.model.train()

        for epoch in range(self.epochs):
            overall_loss = 0.0

            progress_bar = tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}"
            )

            for batch_idx, (x, _) in enumerate(progress_bar):
                x = x.view(x.size(0), self.x_dim).to(self.device)

                self.optimizer.zero_grad()

                x_hat, vq_loss, z_e, z_q, encoding_indices = self.model(x)
                recon_loss = F.mse_loss(x_hat, x)
                loss = recon_loss + vq_loss

                overall_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                avg_loss = overall_loss / ((batch_idx + 1) * x.size(0))
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        print("Trainig completed.")

        save_path = "./saved_model/vq_vae_model"
        self.save_model(path=save_path)
        print("Model saved")
