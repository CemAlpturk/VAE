import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset


class MNISTDataset(Dataset):

    def __init__(
        self,
        imgs: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        n, w, h = imgs.shape

        assert len(labels) == n, "Images and labels must have the same length"

        # Flatten the images
        imgs = imgs.reshape(n, w * h).astype(np.float32)

        # Normalize images between [0, 1]
        imgs /= 255.0

        self.imgs = torch.from_numpy(imgs.copy())
        self.labels = torch.from_numpy(labels.copy())

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:

        return self.imgs[idx], self.labels[idx]
