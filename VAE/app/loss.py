import torch.nn as nn
import torch

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    """
    Compute the loss function for the VAE
    Args:
        x (torch.Tensor): Original input data.
        x_hat (torch.Tensor): Reconstructed output from the decoder.
        mean (torch.Tensor): Mean of the latent space distribution.
        log_var (torch.Tensor): Log variance of the latent space distribution.
    Returns:
        sum of the reconstruction loss and the Kullback-Leibler divergence.
    It calculates the binary cross-entropy loss and the Kullback-Leibler divergence.
    """
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD
