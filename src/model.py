# model.py

import torch
import torch.nn as nn

# -------------------- MLP блоки --------------------


class MLPBlock(nn.Module):
    """Без нормализации"""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.relu(self.net(x))


class MLPBlockBN(nn.Module):
    """BN после каждого Linear"""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.relu(self.net(x))


# -------------------- ResNet блоки --------------------


class ResBlock(nn.Module):
    """Без нормализации -> Skip-connection работает верно"""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        nn.init.constant_(self.net[-1].weight, 0.0) # type: ignore
        if self.net[-1].bias is not None:
            nn.init.constant_(self.net[-1].bias, 0.0) # type: ignore

    def forward(self, x: torch.Tensor):
        return x + self.net(x)

class ResBlockPostNorm(nn.Module):
    """Post-Norm"""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        nn.init.constant_(self.net[-1].weight, 0.0) # type: ignore
        nn.init.constant_(self.net[-1].bias, 0.0) # type: ignore
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        out = x + out
        return self.relu(out)


class ResBlockPreNorm(nn.Module):
    """Pre-Norm"""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.BatchNorm1d(dim),  # батч-норм до лин. слоя
            nn.Linear(dim, dim),
        )

        nn.init.constant_(self.net[-1].weight, 0.0) # type: ignore
        if self.net[-1].bias is not None:
            nn.init.constant_(self.net[-1].bias, 0.0) # type: ignore

    def forward(self, x: torch.Tensor):
        return x + self.net(x)


# -------------------- Фабрика блоков --------------------

_BLOCK_REGISTRY = {
    ("mlp", None): MLPBlock,
    ("mlp", "postnorm"): MLPBlockBN,
    ("resnet", None): ResBlock,
    ("resnet", "postnorm"): ResBlockPostNorm,
    ("resnet", "prenorm"): ResBlockPreNorm,
}


def _get_block(model_type: str, norm_type: str | None, dim: int) -> nn.Module:
    key = (model_type, norm_type)
    if key not in _BLOCK_REGISTRY:
        raise ValueError("Нет такой модельки")
    return _BLOCK_REGISTRY[key](dim)


# -------------------- Основная модель --------------------


class DeepModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        model_type: str = "mlp",
        norm_type: str | None = None,
    ):
        super().__init__()

        self.inp_layer = nn.Linear(input_dim, hidden_dim)

        self.backbone = nn.Sequential(
            *[_get_block(model_type, norm_type, hidden_dim) for _ in range(num_layers)]
        )

        self.out_layer = nn.Linear(hidden_dim, 1)
        #self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu", a=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.inp_layer(x)
        x = self.backbone(x)
        return self.out_layer(x)
