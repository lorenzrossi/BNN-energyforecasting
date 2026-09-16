import numpy as np
import pandas as pd
import torch

from energyforecast.bayesian import BayesianMLP, elbo_loss, predict_samples
from energyforecast.data import add_calendar_features, aggregate_sources, hours_in_years
from energyforecast.models import build_model
from energyforecast.training import BayesianRollingForecaster, RollingForecaster
from energyforecast.windows import make_lags, make_sequences, make_split


def synthetic(n_hours=24 * 30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h", name="time")
    hour = idx.hour.to_numpy()
    level = 25000 + 5000 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 500, n_hours)
    df = pd.DataFrame({"total_aggregated": level}, index=idx)
    return add_calendar_features(df)


def test_lag_alignment():
    y = np.arange(10, dtype=float)
    dummies = np.arange(10, dtype=float).reshape(-1, 1) * 100
    X, target = make_lags(y, dummies, lags=3)
    assert X.shape == (7, 4) and target.shape == (7,)
    # row 0 predicts y[3] from y[2], y[1], y[0] and the dummy at t=3
    assert X[0].tolist() == [2, 1, 0, 300] and target[0] == 3


def test_sequence_alignment():
    v = np.stack([np.arange(10, dtype=float), np.arange(10, dtype=float) * 10], axis=1)
    X, target = make_sequences(v, window=2)
    assert X.shape == (8, 2, 2) and target.shape == (8,)
    assert X[0].tolist() == [[0, 0], [1, 10]] and target[0] == 2


def test_split_truth_matches_unscaled_target():
    data = synthetic()
    split = make_split(data, "weekend", n_train=400, window=5, sequence=False)
    np.testing.assert_allclose(split.unscale(split.y_test), split.truth, rtol=1e-5)
    assert split.X_train.shape[1] == 7  # 5 lags + saturday + sunday


def test_model_shapes():
    x_seq = torch.zeros(4, 3, 2)
    assert build_model("lstm", 2, 3).forward(x_seq).shape == (4, 1)
    assert build_model("cnn", 2, 3).forward(x_seq).shape == (4, 1)
    assert build_model("mlp", 6, 3).forward(torch.zeros(4, 6)).shape == (4, 1)


def test_elbo_backward_and_sampling():
    model = BayesianMLP(3)
    x, y = torch.randn(8, 3), torch.randn(8, 1)
    loss, mse, kl = elbo_loss(model, x, y, n_samples=2, kl_weight=0.1)
    loss.backward()
    assert kl.item() > 0 and model.net[0].weight_mu.grad is not None
    draws = predict_samples(model, x, n_samples=5)
    assert draws.shape == (5, 8, 1) and not torch.allclose(draws[0], draws[1])


def test_rolling_forecasters_run():
    data = synthetic()
    for model in ["mlp", "lstm", "cnn"]:
        result = RollingForecaster(model, "business_hour", epochs=2, verbose=False, seed=0).run(
            data, n_train=400, max_periods=2
        )
        assert len(result.predictions) == 48 and np.isfinite(result.rmse)
    result = BayesianRollingForecaster("weekend", epochs=2, verbose=False, seed=0).run(
        data, n_train=400, max_periods=2
    )
    assert result.lower is not None and 0 <= result.coverage <= 1


def test_aggregation_and_calendar():
    idx = pd.date_range("2016-01-01", periods=48, freq="h")
    cols = [
        "biomass",
        "coal_gas",
        "gas",
        "hard_coal",
        "oil",
        "geothermal",
        "hydro_pumped",
        "hydro_river",
        "hydro_reservoir",
        "other",
        "solar",
        "waste",
        "wind",
    ]
    df = pd.DataFrame(1.0, index=idx, columns=cols)
    out = add_calendar_features(aggregate_sources(df))
    assert (out["total_aggregated"] == 13).all()
    assert out["hydro_tot"].iloc[0] == 3 and out["gas_tot"].iloc[0] == 2
    assert out.loc["2016-01-02 09:00", "saturday"] == 1
    assert out.loc["2016-01-02 09:00", "business_hour"] == 1
    assert out.loc["2016-01-01 07:00", "business_hour"] == 0
    assert hours_in_years([2016, 2017, 2018, 2019]) == 35064
