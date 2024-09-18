import numpy as np

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from tqdm import tqdm
from matplotlib import pyplot as plt

from dataset import mnist
from model import VAE


# Helper classes
class MNISTDataset(Dataset):
    def __init__(
        self,
        imgs: np.ndarray,
        labels: np.ndarray,
        num: int | None = None,
    ) -> None:

        n, w, h = imgs.shape
        n1 = len(labels)

        assert n == n1, "Images and shapes must have the same length"

        # Flatten the images
        imgs = imgs.reshape(n, w * h).astype(np.float32)

        # Normalize images to between [0, 1]
        imgs /= 255.0

        if num is not None:
            assert num >= 0 and num < 10

            idxs = labels == num
        else:
            idxs = np.ones_like(labels)

        self.imgs = torch.from_numpy(imgs[idxs].copy())

        self.labels = torch.from_numpy(labels[idxs].copy())

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        return self.imgs[idx], self.labels[idx]


def loss_fn(
    x: Tensor,
    x_hat: Tensor,
    mu: Tensor,
    logvar: Tensor,
    M_N: float = 1.0,
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
        "kld": kld_loss.detach(),
    }


def main() -> None:

    # Arguments
    epochs = 10
    batch_size = 64
    latent_size = 2
    lr = 0.0001
    sig_dec = 0.1
    n_samples = 20

    # Load mnist
    data_dir = "data"
    data = mnist(data_dir)

    train_dataset = MNISTDataset(
        imgs=data["train_images"],
        labels=data["train_labels"],
        num=None,
    )

    test_dataset = MNISTDataset(
        imgs=data["test_images"],
        labels=data["test_labels"],
    )

    # Model
    layers = (200, 100)
    model = VAE(
        n_in=28 * 28,
        n_latent=latent_size,
        hidden_dims=layers,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    opt = Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in tqdm(range(epochs), position=1, leave=True):
        for xs, _ in tqdm(train_dataloader, position=0, leave=False):

            x_hat, _, mu, logvar = model(xs)
            train_losses = loss_fn(xs, x_hat, mu, logvar)
            loss = train_losses["loss"]
            loss.backward()
            opt.step()

            opt.zero_grad()

            losses.append(loss.item())

    # Plot losses
    plt.plot(losses)
    plt.show()

    with torch.no_grad():
        # Plot reconstructed image
        xs, _ = next(iter(train_dataloader))
        x = xs[0]
        x_hat = model.generate(x)

    x_im = x.reshape(28, 28)
    x_hat_im = x_hat.reshape(28, 28)

    fig, axs = plt.subplots(2)
    axs[0].imshow(x_im)
    axs[1].imshow(x_hat_im)
    plt.show()


if __name__ == "__main__":
    main()
