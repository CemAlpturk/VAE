from typing import Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


# Define models
class Encoder(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_latent: int,
        layer_dims: Sequence[int] = (),
    ) -> None:
        super().__init__()

        dims = (n_in,) + tuple(layer_dims)

        layers = []
        for d1, d2 in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d1, d2))

        self.layers = nn.ModuleList(layers)

        self.mu = nn.Linear(dims[-1], n_latent, bias=False)
        self.logvar = nn.Linear(dims[-1], n_latent)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        for layer in self.layers:
            x = F.relu(layer(x))

        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar


class Decoder(nn.Module):

    def __init__(
        self,
        n_latent: int,
        n_out: int,
        layer_dims: Sequence[int] = (),
    ) -> None:
        super().__init__()

        dims = (n_latent,) + tuple(layer_dims)

        layers = []
        for d1, d2 in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d1, d2))

        self.layers = nn.ModuleList(layers)

        self.mu = nn.Linear(dims[-1], n_out)

    def forward(self, z: Tensor) -> Tensor:
        for layer in self.layers:
            z = F.relu(layer(z))

        x = F.sigmoid(self.mu(z))

        return x


class VAE(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_latent: int,
        hidden_dims: Sequence[int] = (),
        name: str = "VAE",
    ) -> None:
        super().__init__()
        self.name = name

        # Build encoder
        self.encoder = Encoder(
            n_in=n_in,
            n_latent=n_latent,
            layer_dims=hidden_dims,
        )

        # Symmetric layers
        hidden_dims = hidden_dims[::-1]

        # Build decoder
        self.decoder = Decoder(
            n_latent=n_latent,
            n_out=n_in,
            layer_dims=hidden_dims,
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode the input and return latent codes.

        Args:
            x (Tensor): Inputs to encode [N x F]

        Returns:
            Tensor, torch.Tensor: mu and logvar.
        """

        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps given latent code to original space.

        Args:
            z (Tensor): Latent codes [N x D]

        Returns:
            Tensor: Generated sample [N x F]
        """
        x = self.decoder(z)
        return x

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparametrize trick to sample N(`mu`, `logvar`) from N(0, 1).

        Args:
            mu (Tensor): Latent mean. [N x D]
            logvar (Tensor): Latent log variance. [N x D]

        Returns:
            Tensor: Random sample from distribution N(`mu`, `logvar`). [N x D]
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)

        return eps * std + mu

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass"""

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)

        return x_hat, mu, logvar

    def sample(self, num_samples: int) -> Tensor:
        """
        Samples from the latent space and return generated values.

        Args:
            num_samples (int): Number of samples.

        Returns:
            Tensor: Generated values.
        """

        z = torch.randn(num_samples, self.n_latent)
        samples = self.decode(z)

        return samples

    @torch.no_grad()
    def generate(self, x: Tensor) -> Tensor:
        """
        Given a data point x, return the reconstructed image.
        """

        return self.forward(x)[0]
