from typing import Any

from collections import defaultdict
import os
import json
import concurrent.futures

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
        "../models",
    )
)


def setup_dir(model_name: str) -> str:
    model_path = os.path.join(MODEL_DIR, model_name)

    os.makedirs(model_path, exist_ok=True)

    return model_path


def save_metrics(metrics: dict[str, Any], filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(metrics, f)


def loss_fn(
    x: Tensor,
    x_hat: Tensor,
    mu: Tensor,
    logvar: Tensor,
) -> dict[str, Tensor]:
    """ELBO loss"""
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")

    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
    )

    loss = recon_loss + kld_loss

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
) -> None:

    # Set up model dir
    model_path = setup_dir(model.name)

    # Metrics path
    metrics_path = os.path.join(model_path, "metrics.json")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    opt = Adam(model.parameters(), lr=train_config.lr)

    metrics = defaultdict(list)
    async_saver = concurrent.futures.ThreadPoolExecutor()

    # Training loop
    for epoch in tqdm(range(train_config.max_epochs), position=0, desc="Epoch"):
        epoch_train_loss = 0
        epoch_train_recon_loss = 0
        epoch_train_kld_loss = 0

        for batch_idx, (xs, _) in tqdm(
            enumerate(train_dataloader),
            position=1,
            leave=False,
            total=len(train_dataloader),
            desc="Step",
        ):

            opt.zero_grad()

            x_hat, mu, logvar = model(xs)
            train_losses = loss_fn(xs, x_hat, mu, logvar)
            loss = train_losses["loss"]

            epoch_train_loss += loss.item()
            epoch_train_recon_loss += train_losses["recon_loss"].item()
            epoch_train_kld_loss += train_losses["kld_loss"].item()

            loss.backward()
            opt.step()

        # Average losses
        epoch_train_loss /= batch_idx * train_config.batch_size
        epoch_train_recon_loss /= batch_idx * train_config.batch_size
        epoch_train_kld_loss /= batch_idx * train_config.batch_size

        metrics["epochs"].append(epoch)
        metrics["epoch_train_loss"].append(epoch_train_loss)
        metrics["epoch_train_recon_loss"].append(epoch_train_recon_loss)
        metrics["epoch_train_kld_loss"].append(epoch_train_kld_loss)

        # Save metrics async
        async_saver.submit(save_metrics, dict(metrics), metrics_path)

        # TODO: Evaluate model

    async_saver.shutdown(wait=True)
