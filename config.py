# config.py

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Данные ---
    n_points: int = 10_000
    window: int = 50
    train_split: float = 0.8

    # --- Архитектура ---
    hidden_dim: int = 64
    depths: list[int] = field(default_factory=lambda: [1, 5, 10, 15, 20])

    # --- Обучение ---
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-5

    # --- Рандом ---
    seed: int = 55

    # --- Пути ---
    save_dir: str = "experiments"


EXPERIMENTS: list[dict] = [
    {"model_type": "resnet", "norm_type": None, "name": "ResNet (no BN)"},
    {"model_type": "mlp", "norm_type": "postnorm", "name": "MLP + Post-Norm BN"},
    {"model_type": "mlp", "norm_type": None, "name": "MLP (no BN)"},
    {"model_type": "resnet", "norm_type": "postnorm", "name": "ResNet + Post-Norm BN"},
    {"model_type": "resnet", "norm_type": "prenorm", "name": "ResNet + Pre-Norm BN"},
]
