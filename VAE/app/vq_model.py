import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super().__init__()

        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.fc_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h_ = self.LeakyReLU(self.fc_input(x))
        h_ = self.LeakyReLU(self.fc_input2(h_))
        z_e = self.fc_latent(h_)

        return z_e


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, z_e):
        z_e_expanded = z_e.unsqueeze(1)  # (batch, 1, embedding_dim)
        embeddings_expanded = self.embeddings.weight.unsqueeze(
            0
        )  # (1, num_embeddings, embedding_dim)
        distances = torch.sum(
            (z_e_expanded - embeddings_expanded) ** 2, dim=2
        )  # (batch, num_embeddings)

        encoding_indices = torch.argmin(distances, dim=1)  # (batch,)
        z_q = self.embeddings(encoding_indices)  # (batch, embedding_dim)

        vq_loss = torch.mean(
            (z_q.detach() - z_e) ** 2
        ) + self.commitment_cost * torch.mean((z_q - z_e.detach()) ** 2)

        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, vq_loss, encoding_indices


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h_ = self.LeakyReLU(self.fc_hidden(x))
        h_ = self.LeakyReLU(self.fc_hidden2(h_))
        x_hat = torch.sigmoid(self.fc_output(h_))

        return x_hat


class VAE(nn.Module):
    def __init__(self, encoder, decoder, vector_quantizer, is_training=True):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vector_quantizer = vector_quantizer

    def forward(self, x):
        z_e = self.encoder(x)  # (batch, latent_dim)
        z_q, vq_loss, encoding_indices = self.vector_quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss, z_e, z_q, encoding_indices
