from dataclasses import dataclass


@dataclass
class ModelConfig:
    latent_size: int
    hidden_layers: tuple[int, ...]


@dataclass
class TrainConfig:
    lr: float
    batch_size: int
    max_epochs: int
