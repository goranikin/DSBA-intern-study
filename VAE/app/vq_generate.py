import torch
import os
from .vq_train import load_model
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
from .utils import set_seed

set_seed(42)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    dataset_path = "./datasets"
    model_path = "saved_model/"
    model_name = "vq_vae_model_64d_512emb"  # 학습 시 사용한 model_name과 일치해야 함
    batch_size = 128
    x_dim = 784
    hidden_dim = 400
    latent_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    model_path = os.path.join(project_root, model_path)
    results_dir = os.path.join(project_root, "results")
    ensure_dir(results_dir)

    # VQ-VAE 모델 로드
    model = load_model(
        input_dim=x_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        is_training=False,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path + model_name + ".pth", map_location=torch.device(device))
    )

    model.eval()

    # 1. Random latent vector로부터 이미지 생성
    with torch.no_grad():
        # latent vector를 코드북 인덱스 대신 연속적으로 샘플링하는 것은 VQ-VAE의 원래 방식은 아니지만,
        # 실험적으로 코드북에서 무작위로 인덱스를 뽑아 생성할 수도 있음
        # 여기서는 latent_dim 크기의 코드북 인덱스를 무작위로 뽑아 생성
        random_indices = torch.randint(0, num_embeddings, (batch_size,))
        z_q = model.vector_quantizer.embeddings(
            random_indices.to(device)
        )  # (batch_size, latent_dim)
        generated_images = model.decoder(z_q)

    save_image(
        generated_images.view(batch_size, 1, 28, 28),
        os.path.join(results_dir, f"generated_sample{model_name}.png"),
    )

    # 2. Latent space interpolation (코드북 인덱스 공간에서 보간)
    z1_idx = torch.randint(0, num_embeddings, (1,))
    z2_idx = torch.randint(0, num_embeddings, (1,))
    z1 = model.vector_quantizer.embeddings(z1_idx.to(device))  # (1, latent_dim)
    z2 = model.vector_quantizer.embeddings(z2_idx.to(device))  # (1, latent_dim)

    num_steps = 10  # 보간 단계 수
    interpolated = []
    for alpha in np.linspace(0, 1, num_steps):
        z = (1 - alpha) * z1 + alpha * z2
        interpolated.append(z)
    interpolated = torch.cat(interpolated, dim=0)  # (num_steps, latent_dim)

    with torch.no_grad():
        generated_images = model.decoder(interpolated)

    save_image(
        generated_images.view(num_steps, 1, 28, 28),
        os.path.join(results_dir, f"interpolated{model_name}.png"),
        nrow=num_steps,
    )

    # 3. (Optional) 시각화 함수
    def show_image(x, idx):
        x = x.view(-1, 28, 28)
        fig = plt.figure()
        plt.imshow(x[idx].cpu().numpy(), cmap="gray")
        plt.axis("off")
        plt.show()
