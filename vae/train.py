from collections import defaultdict
import os

import numpy as np

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from vae.model import VAE
from vae.datasets import MNISTDataset
from vae.configs import ModelConfig, TrainConfig

MODEL_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "models",
    )
)


def loss_fn(
    x: Tensor,
    x_hat: Tensor,
    mu: Tensor,
    logvar: Tensor,
    M_N: float = 1.0,  # TODO: Find appropriate value for this
) -> dict[str, Tensor]:
    """ELBO loss"""

    recon_loss = F.mse_loss(x_hat, x)

    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
    )

    loss = recon_loss + M_N * kld_loss

    return {
        "loss": loss,
        "recon_loss": recon_loss.detach(),
        "kld_loss": kld_loss.detach(),
    }


def train(
    model: VAE,
    train_dataset: Dataset,
    test_dataset: Dataset,
    train_config: TrainConfig,
) -> dict[str, list[float]]:

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    opt = Adam(model.parameters(), lr=train_config.lr)

    metrics = defaultdict(list)
    step = 0
    batch_iter = iter(train_dataloader)

    # Training loop
    for step in tqdm(range(train_config.steps)):
        # Infinite batch
        try:
            xs, _ = next(batch_iter)
        except StopIteration:
            # Reinitialize batch
            batch_iter = iter(train_dataloader)
            xs, _ = next(batch_iter)

        x_hat, _, mu, logvar = model(xs)
        train_losses = loss_fn(xs, x_hat, mu, logvar)
        loss = train_losses["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % train_config.logging_steps == 0:
            metrics["steps"].append(step)
            metrics["train_loss"].append(loss.item())
            metrics["train_recon_loss"].append(train_losses["recon_loss"].item())
            metrics["train_kld_loss"].append(train_losses["kld_loss"].item())

        # TODO: Evaluate model

    return metrics
