# main.py

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from config import Config, EXPERIMENTS
from src.model import DeepModel
from src.train import train_model, evaluate
from src.dataset import get_dataloaders


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def save_results(results: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> сохранено: {path}")


def run_experiment(
    exp: dict,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
):
    print(f"Идет запуск {exp['name']} :)")
    set_seed(cfg.seed)

    model = DeepModel(
        input_dim=cfg.window,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.depth,
        model_type=exp["model_type"],
        norm_type=exp["norm_type"],
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    test_huber = evaluate(model, test_loader, criterion, device)

    print(f"MSE на тесте: {test_huber:.6f}")

    weights_path = os.path.join(cfg.save_dir, f"{exp['name'].replace(' ', '_')}.pth")
    torch.save(model.state_dict(), weights_path)

    return {"name": exp["name"], "history": history, "test_huber": test_huber}


def main():
    cfg = Config()
    set_seed(cfg.seed)

    train_loader, val_loader, test_loader = get_dataloaders(
        n_points=cfg.n_points,
        window=cfg.window,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    all_res = []

    for exp in EXPERIMENTS:
        result = run_experiment(exp, cfg, train_loader, val_loader, test_loader)
        all_res.append(result)

    save_results(
        {r["name"]: r["history"] for r in all_res},
        path=os.path.join(cfg.save_dir, "history.json"),
    )


if __name__ == "__main__":
    main()