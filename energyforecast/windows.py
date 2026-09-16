"""Turn the hourly frame into supervised learning arrays."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TARGET = "total_aggregated"

# Exogenous dummies used in the three experimental settings.
FEATURE_SETS = {
    "ts_only": [],
    "weekend": ["saturday", "sunday"],
    "business_hour": ["business_hour"],
}


@dataclass
class Split:
    """Arrays for one train/test split.

    X_train, X_test: (n, lags [+ n_dummies]) for tabular models,
                     (n, window, 1 + n_dummies) for sequence models.
    y_train, y_test: scaled target, shape (n,).
    truth: unscaled target aligned with y_test.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    truth: np.ndarray
    mean: float
    std: float

    def unscale(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y).reshape(-1) * self.std + self.mean


def make_lags(target: np.ndarray, dummies: np.ndarray, lags: int):
    """Tabular design: [y_{t-1} ... y_{t-lags}, dummies_t] -> y_t."""
    X_lags = sliding_window_view(target[:-1], lags)[:, ::-1].copy()
    X = np.concatenate([X_lags, dummies[lags:]], axis=1) if dummies.shape[1] else X_lags
    return X, target[lags:]


def make_sequences(values: np.ndarray, window: int):
    """Sequence design: values[t-window:t] (all columns) -> values[t, 0]."""
    X = sliding_window_view(values[:-1], window, axis=0).transpose(0, 2, 1).copy()
    return X, values[window:, 0]


def make_split(
    data: pd.DataFrame,
    feature_set: str,
    n_train: int,
    window: int,
    sequence: bool,
) -> Split:
    """Standardise the target on the training part and build the arrays.

    Dummies are passed through unscaled. The scaler statistics come from the
    first n_train hours only.
    """
    cols = [TARGET] + FEATURE_SETS[feature_set]
    values = data[cols].to_numpy(dtype=np.float32)
    mean = float(values[:n_train, 0].mean())
    std = float(values[:n_train, 0].std())
    scaled = values.copy()
    scaled[:, 0] = (values[:, 0] - mean) / std

    train, test = scaled[:n_train], scaled[n_train:]
    if sequence:
        X_train, y_train = make_sequences(train, window)
        X_test, y_test = make_sequences(test, window)
    else:
        X_train, y_train = make_lags(train[:, 0], train[:, 1:], window)
        X_test, y_test = make_lags(test[:, 0], test[:, 1:], window)

    truth = values[n_train + window :, 0].astype(np.float64)
    return Split(X_train, y_train, X_test, y_test, truth, mean, std)
