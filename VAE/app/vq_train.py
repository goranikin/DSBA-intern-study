from .vq_model import VAE, Encoder, Decoder, VectorQuantizer
import torch
from torch.optim import Adam
from .data import load_dataset, load_dataloader
from .vq_trainer import VaeTrainer
from .utils import set_seed

set_seed(42)


def load_model(
    input_dim, hidden_dim, latent_dim, num_embeddings, commitment_cost, is_training=True
):
    encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(
        latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim
    )
    vector_quantizer = VectorQuantizer(
        num_embeddings=num_embeddings,
        embedding_dim=latent_dim,
        commitment_cost=commitment_cost,
    )
    model = VAE(
        encoder=encoder,
        decoder=decoder,
        vector_quantizer=vector_quantizer,
        is_training=is_training,
    )
    return model


if __name__ == "__main__":
    dataset_path = "./datasets"
    batch_size = 128
    x_dim = 784
    hidden_dim = 400
    latent_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25  # 논문 기본값
    lr = 2e-4
    epochs = 30

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_dataset, test_dataset = load_dataset(dataset_path)
    train_dataloader = load_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = load_dataloader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model = load_model(
        input_dim=x_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        is_training=True,
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    trainer = VaeTrainer(
        model=model,
        optimizer=optimizer,
        x_dim=x_dim,
        epochs=epochs,
        device=device,
        train_loader=train_dataloader,
        model_name=f"_{latent_dim}d_{num_embeddings}emb",
    )
    trainer.train_epochs()
