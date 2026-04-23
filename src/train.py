# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    tot_loss = 0.0
    tot_grad_norm = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()

        # ========== тут собираем градиенты ==========
        batch_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                batch_norm += param_norm.item() ** 2
        batch_norm = batch_norm**0.5
        tot_grad_norm += batch_norm
        # ========== конец ==========

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # let it be
        optimizer.step()
        tot_loss += loss.item()

    avg_loss = tot_loss / len(loader)
    avg_grad = tot_grad_norm / len(loader)
    return avg_loss, avg_grad


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    model.eval()
    tot_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        tot_loss += criterion(model(x), y).item()

    return tot_loss / len(loader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 40,
    lr: float = 1e-5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) # let it be

    model.to(device)

    history = {"train": [], "val": [], "grad_norm": []}

    pbar = tqdm(range(epochs), desc="Training", leave=True)
    for epoch in pbar:
        train_loss, grad_norm = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        # scheduler.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["grad_norm"].append(grad_norm)

        pbar.set_postfix(
            {
                "train": f"{train_loss:.4f}",
                "val": f"{val_loss:.4f}",
                "grad": f"{grad_norm:.4f}",
            }
        )

    return history
