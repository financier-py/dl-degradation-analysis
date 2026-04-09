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
    depth: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
):
    exp_name = f"{exp['name']} + depth {depth * 2}"
    print(f"Running {exp_name}")

    set_seed(cfg.seed)

    model = DeepModel(
        input_dim=cfg.window,
        hidden_dim=cfg.hidden_dim,
        num_layers=depth,
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
    test_mse = evaluate(model, test_loader, criterion, device)

    print(f"MSE на тестовой выборке: {test_mse:.6f}")

    weights_path = os.path.join(cfg.save_dir, f"{exp_name.replace(' ', '_')}.pth")
    torch.save(model.state_dict(), weights_path)

    return {
        "name": exp_name,
        "architecture": exp_name,
        "depth": depth,
        "history": history,
        "test_mse": test_mse,
    }


def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        n_points=cfg.n_points,
        window=cfg.window,
        batch_size=cfg.batch_size,
        train_split=cfg.train_split,
        seed=cfg.seed,
    )

    all_res = []

    for exp in EXPERIMENTS:
        for depth in cfg.depths:
            result = run_experiment(
                exp, cfg, depth, train_loader, val_loader, test_loader
            )
            all_res.append(result)

    final_results = {
        r["name"]: {
            "history": r["history"],
            "test_mse": r["test_mse"],
            "depth": r["depth"],
            "architecture": r["architecture"],
        }
        for r in all_res
    }

    save_results(
        final_results, path=os.path.join(cfg.save_dir, "history.json")
    )


if __name__ == "__main__":
    main()