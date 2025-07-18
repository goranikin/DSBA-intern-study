import torch
import os
from .train import load_model
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
from .utils import set_seed


set_seed(42)


def show_image(x, idx):
    x = x.view(batch_size, 28, 28)

    fig = plt.figure()
    plt.imshow(x[idx].cpu().numpy())


if __name__ == "__main__":
    dataset_path = "./datasets"
    model_path = "./saved_model/vae_model"
    model_name = "_5_layers"
    batch_size = 100
    x_dim = 784
    hidden_dim = 400
    latent_dim = 200

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    noise = torch.randn(batch_size, latent_dim).to(device)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    model_path = os.path.join(project_root, model_path)

    model = load_model(
        input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, is_training=False
    ).to(device)
    model.load_state_dict(
        torch.load(model_path + model_name + ".pth", map_location=torch.device(device))
    )

    model.eval()

    # Generate random images
    with torch.no_grad():
        generated_images = model.decoder(noise)

    save_image(
        generated_images.view(batch_size, 1, 28, 28),
        project_root + "/results/generated_sample" + model_name + ".png",
    )

    z1 = torch.randn(1, latent_dim).to(device)
    z2 = torch.randn(1, latent_dim).to(device)

    num_steps = 10  # 보간 단계 수
    interpolated = []
    for alpha in np.linspace(0, 1, num_steps):
        z = (1 - alpha) * z1 + alpha * z2
        interpolated.append(z)
    interpolated = torch.cat(interpolated, dim=0)  # (num_steps, latent_dim)

    with torch.no_grad():
        generated_images = model.decoder(interpolated)

    # 한 줄로 이미지 저장 (num_steps, 1, 28, 28)
    save_image(
        generated_images.view(num_steps, 1, 28, 28),
        project_root + "/results/interpolated" + model_name + ".png",
        nrow=num_steps,
    )
