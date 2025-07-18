import torch
from .vq_train import load_model
from .data import load_dataset, load_dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from .utils import set_seed

set_seed(42)

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
    lr = 2e-4
    epochs = 30

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    model_path = os.path.join(project_root, model_path)
    dataset_path = os.path.join(project_root, dataset_path)

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

    _, test_dataset = load_dataset(dataset_path)
    test_dataloader = load_dataloader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model.eval()

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(test_dataloader)):
            x = x.view(x.size(0), x_dim).to(device)
            x = x.to(device)

            x_hat, vq_loss, z_e, z_q, encoding_indices = model(x)
            break  # 첫 배치만 시각화

    n = 8  # 시각화할 이미지 개수
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
