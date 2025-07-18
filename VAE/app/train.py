from .model import VAE, Encoder, Decoder
import torch
from torch.optim import Adam
from .data import load_dataset, load_dataloader
from .trainer import VaeTrainer
from .loss import loss_function


def load_model(input_dim, hidden_dim, latent_dim, is_training=True):
    encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(
        latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim
    )
    model = VAE(encoder=encoder, decoder=decoder, is_training=is_training)

    return model


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

    train_dataset, test_dataset = load_dataset(dataset_path)
    train_dataloader = load_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = load_dataloader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model = load_model(
        input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, is_training=True
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    trainer = VaeTrainer(
        model=model,
        optimizer=optimizer,
        x_dim=x_dim,
        loss_fn=loss_function,
        epochs=epochs,
        device=device,
        train_loader=train_dataloader,
        model_name="_5_layers",
    )
    trainer.train_epochs()
