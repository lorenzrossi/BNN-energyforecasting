"""Rolling-origin evaluation shared by the point and Bayesian models.

Protocol (same as the thesis notebooks): fit on the first n_train hours,
forecast the next `horizon` hours, then drop the oldest `horizon` hours from
the history, append the observed ones, continue training from the current
weights and repeat until the test set is exhausted.
"""

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .bayesian import BayesianMLP, elbo_loss, predict_samples
from .models import build_model, count_parameters, pick_device
from .windows import FEATURE_SETS, Split, make_split

SEQUENCE_MODELS = {"lstm", "cnn"}
DEFAULT_WINDOW = {"mlp": 24, "lstm": 1, "cnn": 2, "bmlp": 1}


@dataclass
class Result:
    model: str
    feature_set: str
    window: int
    predictions: np.ndarray
    truth: np.ndarray
    period_rmse: list = field(default_factory=list)
    std: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None

    @property
    def rmse(self) -> float:
        return sqrt(np.mean((self.truth - self.predictions) ** 2))

    @property
    def mae(self) -> float:
        return float(np.mean(np.abs(self.truth - self.predictions)))

    @property
    def r2(self) -> float:
        ss_res = np.sum((self.truth - self.predictions) ** 2)
        ss_tot = np.sum((self.truth - self.truth.mean()) ** 2)
        return float(1 - ss_res / ss_tot)

    @property
    def coverage(self) -> float | None:
        if self.lower is None:
            return None
        return float(np.mean((self.lower <= self.truth) & (self.truth <= self.upper)))

    def summary(self) -> dict:
        out = {
            "model": self.model,
            "features": self.feature_set,
            "window": self.window,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "periods": len(self.period_rmse),
        }
        if self.lower is not None:
            out["coverage"] = self.coverage
        return out

    def frame(self) -> pd.DataFrame:
        cols = {"truth": self.truth, "prediction": self.predictions}
        if self.lower is not None:
            cols.update(std=self.std, lower=self.lower, upper=self.upper)
        return pd.DataFrame(cols)


class RollingForecaster:
    """Point forecasts with an MLP, LSTM or CNN."""

    def __init__(
        self,
        model: str,
        feature_set: str = "ts_only",
        window: int | None = None,
        horizon: int = 24,
        epochs: int = 100,
        batch_size: int = 168,
        lr: float = 0.003,
        patience: int | None = 3,
        device: str | None = None,
        verbose: bool = True,
        seed: int | None = None,
        **model_kwargs,
    ):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {list(FEATURE_SETS)}")
        self.model_kind = model
        self.feature_set = feature_set
        self.window = DEFAULT_WINDOW[model] if window is None else window
        self.horizon = horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.device = pick_device(device)
        self.verbose = verbose
        self.seed = seed
        self.model_kwargs = model_kwargs
        self.model: nn.Module | None = None

    # Hooks overridden by the Bayesian subclass.

    def _build(self, n_features: int) -> nn.Module:
        return build_model(self.model_kind, n_features, self.window, **self.model_kwargs)

    def _loss(self, model, x, y, n_history):
        return F.mse_loss(model(x), y)

    def _predict(self, model, x: np.ndarray, split: Split):
        """Return (prediction, std) in the original units; std is None for point models."""
        model.eval()
        with torch.no_grad():
            out = model(torch.as_tensor(x, device=self.device))
        return split.unscale(out.cpu().numpy()), None

    # Shared machinery.

    def _fit(self, model, optimizer, X: torch.Tensor, y: torch.Tensor) -> int:
        """Train for up to self.epochs; stop early when the epoch loss stalls."""
        loader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=True)
        best, stale = float("inf"), 0
        for epoch in range(self.epochs):
            model.train()
            total = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = self._loss(model, xb, yb, len(X))
                loss.backward()
                optimizer.step()
                total += loss.item() * len(xb)
            epoch_loss = total / len(X)
            if epoch_loss < best:
                best, stale = epoch_loss, 0
            else:
                stale += 1
                if self.patience is not None and stale >= self.patience:
                    break
        return epoch + 1

    def run(
        self, data: pd.DataFrame, n_train: int = 35064, max_periods: int | None = None
    ) -> Result:
        """Evaluate on data[n_train:]; max_periods truncates the test set (for quick checks)."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        split = make_split(
            data,
            self.feature_set,
            n_train,
            self.window,
            sequence=self.model_kind in SEQUENCE_MODELS,
        )
        self.model = self._build(split.X_train.shape[-1]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.verbose:
            print(
                f"{self.model_kind} / {self.feature_set} / window {self.window} / "
                f"{count_parameters(self.model)} parameters / {self.device}"
            )

        X_hist = torch.as_tensor(split.X_train, device=self.device)
        y_hist = torch.as_tensor(split.y_train, device=self.device).unsqueeze(1)
        n_test = len(split.X_test)
        starts = range(0, n_test, self.horizon)
        if max_periods is not None:
            starts = starts[:max_periods]

        preds, stds, period_rmse = [], [], []
        for period, start in enumerate(starts, 1):
            stop = min(start + self.horizon, n_test)
            epochs_run = self._fit(self.model, optimizer, X_hist, y_hist)

            X_block = split.X_test[start:stop]
            pred, std = self._predict(self.model, X_block, split)
            block_rmse = sqrt(np.mean((split.truth[start:stop] - pred) ** 2))
            period_rmse.append(block_rmse)
            preds.append(pred)
            if std is not None:
                stds.append(std)
            if self.verbose:
                print(f"period {period}/{len(starts)}: {epochs_run} epochs, rmse {block_rmse:.1f}")

            X_new = torch.as_tensor(X_block, device=self.device)
            y_new = torch.as_tensor(split.y_test[start:stop], device=self.device).unsqueeze(1)
            X_hist = torch.cat([X_hist[stop - start :], X_new])
            y_hist = torch.cat([y_hist[stop - start :], y_new])

        predictions = np.concatenate(preds)
        result = Result(
            self.model_kind,
            self.feature_set,
            self.window,
            predictions,
            split.truth[: len(predictions)],
            period_rmse,
        )
        if stds:
            result.std = np.concatenate(stds)
            result.lower = predictions - self.std_multiplier * result.std
            result.upper = predictions + self.std_multiplier * result.std
        if self.verbose:
            print(
                f"rmse {result.rmse:.1f}  mae {result.mae:.1f}  r2 {result.r2:.4f}"
                + (f"  coverage {result.coverage:.3f}" if result.lower is not None else "")
            )
        return result


class BayesianRollingForecaster(RollingForecaster):
    """Bayes-by-backprop MLP. Predictions are the mean over `n_predict_samples`
    weight draws; the interval is mean +/- std_multiplier * std of the draws."""

    def __init__(
        self,
        feature_set: str = "ts_only",
        window: int | None = None,
        horizon: int = 24,
        epochs: int = 5,
        batch_size: int = 168,
        lr: float = 0.01,
        patience: int | None = None,
        device: str | None = None,
        verbose: bool = True,
        seed: int | None = None,
        n_train_samples: int = 5,
        n_predict_samples: int = 100,
        std_multiplier: float = 3.0,
        kl_weight: float | None = None,
        **model_kwargs,
    ):
        super().__init__(
            "bmlp",
            feature_set,
            window,
            horizon,
            epochs,
            batch_size,
            lr,
            patience,
            device,
            verbose,
            seed,
            **model_kwargs,
        )
        self.n_train_samples = n_train_samples
        self.n_predict_samples = n_predict_samples
        self.std_multiplier = std_multiplier
        self.kl_weight = kl_weight

    def _build(self, n_features):
        return BayesianMLP(n_features, **self.model_kwargs)

    def _loss(self, model, x, y, n_history):
        weight = 1.0 / n_history if self.kl_weight is None else self.kl_weight
        loss, _, _ = elbo_loss(model, x, y, self.n_train_samples, weight)
        return loss

    def _predict(self, model, x, split):
        model.eval()
        draws = predict_samples(
            model, torch.as_tensor(x, device=self.device), self.n_predict_samples
        )
        draws = draws.squeeze(-1).cpu().numpy()
        return split.unscale(draws.mean(axis=0)), draws.std(axis=0) * split.std


GRID = [
    ("mlp", "ts_only"),
    ("mlp", "weekend"),
    ("mlp", "business_hour"),
    ("lstm", "ts_only"),
    ("lstm", "weekend"),
    ("lstm", "business_hour"),
    ("cnn", "ts_only"),
    ("cnn", "weekend"),
    ("cnn", "business_hour"),
    ("bmlp", "ts_only"),
    ("bmlp", "weekend"),
    ("bmlp", "business_hour"),
]


def make_forecaster(model: str, feature_set: str, **kwargs) -> RollingForecaster:
    if model == "bmlp":
        return BayesianRollingForecaster(feature_set, **kwargs)
    return RollingForecaster(model, feature_set, **kwargs)


def run_grid(
    data: pd.DataFrame, grid=GRID, n_train: int = 35064, max_periods=None, **kwargs
) -> tuple[pd.DataFrame, dict]:
    """Run every (model, feature_set) pair; returns a summary table and the Result objects."""
    rows, results = [], {}
    for model, feature_set in grid:
        forecaster = make_forecaster(model, feature_set, **kwargs)
        result = forecaster.run(data, n_train=n_train, max_periods=max_periods)
        results[(model, feature_set)] = result
        rows.append(result.summary())
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True), results
