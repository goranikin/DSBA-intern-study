import torch
from .train import load_model
from .data import load_dataset, load_dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def show_image(x, idx):
    x = x.view(batch_size, 28, 28)

    fig = plt.figure()
    plt.imshow(x[idx].cpu().numpy())


if __name__ == "__main__":
    dataset_path = "./datasets"
    batch_size = 100
    x_dim = 784
    hidden_dim = 400
    latent_dim = 200
    lr = 1e-3
    epochs = 30

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    model_path = os.path.join(project_root, "saved_model", "vae_model.pth")
    dataset_path = os.path.join(project_root, dataset_path)

    model = load_model(
        input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, is_training=False
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    _, test_dataset = load_dataset(dataset_path)
    test_dataloader = load_dataloader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model.eval()

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(test_dataloader)):
            x = x.view(batch_size, x_dim).to(device)
            x = x.to(device)

            x_hat, _, _ = model(x)
            break  # Only process the first batch for evaluation

    n = 8
    plt.figure(figsize=(16, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(x[i].cpu().reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.subplot(2, n, n + i + 1)
        plt.imshow(x_hat[i].cpu().reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
