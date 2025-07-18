import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super().__init__()

        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.fc_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_input3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_input4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_input5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_variance = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h_ = self.LeakyReLU(self.fc_input(x))
        h_ = self.LeakyReLU(self.fc_input2(h_))
        h_ = self.LeakyReLU(self.fc_input3(h_))
        h_ = self.LeakyReLU(self.fc_input4(h_))
        h_ = self.LeakyReLU(self.fc_input5(h_))
        mean = self.fc_mean(h_)
        log_variance = self.fc_variance(h_)

        return mean, log_variance


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
    def __init__(self, encoder, decoder, is_training=True):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_variance):
        if self.training:
            std = torch.exp(0.5 * log_variance)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean

    def forward(self, x):
        mean, log_variance = self.encoder(x)
        z = self.reparameterize(mean, log_variance)
        x_hat = self.decoder(z)

        return x_hat, mean, log_variance
