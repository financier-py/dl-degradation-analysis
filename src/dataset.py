# dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset


def generate_signal(n_points: int, seed: int = 57) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 100, n_points)

    signal = np.sin(0.5 * t) * np.cos(0.2 * t) + 0.5 * np.sin(2.0 * t)
    signal *= 1 + 0.5 * np.sin(t / 15)

    chaos = np.zeros(n_points)
    x = 0.5
    for i in range(n_points):
        x = 3.8 * x * (1 - x)
        chaos[i] = x
    signal += 0.2 * chaos * np.sin(t)

    signal += rng.normal(0, 0.02, n_points)
    return signal


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        n_points: int = 10_000,
        window: int = 50,
        seed: int = 67,
        train_split: float = 0.8,
    ):
        self.window = window

        signal = generate_signal(n_points, seed=seed)

        tr_end = int(n_points * train_split)
        tr_signal = signal[:tr_end]

        mn, mx = tr_signal.min(), tr_signal.max()

        self.data = ((signal - mn) / (mx - mn + 1e-8)).astype(float)

    def __len__(self):
        return len(self.data) - self.window

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window]
        y = self.data[idx + self.window]

        return torch.tensor(x), torch.tensor([y])


def get_dataloaders(
    n_points: int = 10_000,
    window: int = 50,
    batch_size: int = 128,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 12,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    dataset = TimeSeriesDataset(
        n_points=n_points, window=window, seed=seed, train_split=train_split
    )

    tot_len = len(dataset)
    tr_end = int(tot_len * train_split)
    val_end = int(tot_len * (train_split + val_split))

    tr_ind = list(range(0, tr_end))
    val_ind = list(range(tr_end, val_end))
    test_ind = list(range(val_end, tot_len))

    train_ds = Subset(dataset, tr_ind)
    val_ds = Subset(dataset, val_ind)
    test_ds = Subset(dataset, test_ind)

    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return tr_loader, val_loader, test_loader
