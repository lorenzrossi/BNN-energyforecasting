"""Point-forecast networks. Architectures match the thesis notebooks."""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_inputs: int, hidden: int = 10, n_layers: int = 2):
        super().__init__()
        layers, width = [], n_inputs
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LSTM(nn.Module):
    """Stacked LSTM. All time steps are flattened into the output layer, as in
    Keras LSTM(return_sequences=True) -> Flatten -> Dense(1)."""

    def __init__(self, n_features: int, window: int, hidden: int = 10, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden * window, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out.flatten(1))


class CNN(nn.Module):
    """Two 1-D convolutions with kernel size 1, flattened into a linear output."""

    def __init__(
        self,
        n_features: int,
        window: int,
        filters: int = 16,
        kernel_size: int = 1,
        n_layers: int = 2,
    ):
        super().__init__()
        layers, channels = [], n_features
        for _ in range(n_layers):
            layers += [nn.Conv1d(channels, filters, kernel_size), nn.ReLU()]
            channels = filters
            window = window - kernel_size + 1
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(filters * window, 1)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2))  # (batch, channels, time)
        return self.head(x.flatten(1))


def build_model(kind: str, n_features: int, window: int, **kwargs) -> nn.Module:
    if kind == "mlp":
        return MLP(n_features, **kwargs)
    if kind == "lstm":
        return LSTM(n_features, window, **kwargs)
    if kind == "cnn":
        return CNN(n_features, window, **kwargs)
    raise ValueError(f"unknown model kind {kind!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pick_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
