# Forecasting Italian electricity generation with neural networks

Hourly forecasts of total electricity generation in Italy (ENTSO-E data, 2016-2021) with small
feed-forward, LSTM and 1-D convolutional networks, plus a Bayesian MLP trained with Bayes by
backprop that also gives a prediction interval. The code comes from my thesis; the
`energyforecast` package is a rewrite of the thesis notebooks into one PyTorch code path, and the
notebooks themselves are kept under `notebooks/original`.

## Data

The input files are the ENTSO-E "Actual Generation per Production Type" exports for Italy, one CSV
per year (`ITA2016.csv` ... `ITA2021.csv`). They are not in the repository. They are public and you can download them
from the transparency platform 

Put the files in `data/`. `energyforecast.data.load_dataset` then:

- parses the MTU column (`01.01.2016 00:00 - 01.01.2016 01:00 (CET/CEST)`) as local time,
- drops the duplicated hour of the autumn DST change and reindexes to a complete hourly range,
- interpolates missing values (about 2,200 out of 52,608 hours),
- sums the three hydro columns into `hydro_tot` and the two gas columns into `gas_tot`,
- adds `total_aggregated` (sum over all sources) and the calendar dummies `saturday`, `sunday`,
  `business_hour` (08:00-18:00).

The target everywhere is `total_aggregated`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .              # numpy, pandas, torch
pip install -e ".[notebooks]" # matplotlib, seaborn, statsmodels, jupyter, ...
```

The thesis notebooks in `notebooks/original` additionally need `tensorflow` (MLP, LSTM, CNN) or
`blitz-bayesian-pytorch` (Bayesian MLP); see the `tensorflow` and `blitz` extras. The updated documentation is entirely in PyTorch and doesn't need those packages/framework. 

## Usage

Command line:

```bash
energyforecast --model cnn --features business_hour        # one configuration
energyforecast --model bmlp --features weekend --window 24   # Bayesian MLP with 24 lags
energyforecast --model all --features all --seed 0           # full grid, writes results/summary.csv
energyforecast --max-periods 5 --epochs 10                   # quick check
```

Python:

```python
from energyforecast import load_dataset, RollingForecaster, BayesianRollingForecaster

data = load_dataset("data")

cnn = RollingForecaster("cnn", "business_hour", window=2, seed=0)
result = cnn.run(data, n_train=35064)
print(result.rmse, result.mae, result.r2)

bayes = BayesianRollingForecaster("business_hour", window=24, epochs=5)
result = bayes.run(data, n_train=35064)
print(result.rmse, result.coverage)
result.frame()  # truth, prediction, std, lower, upper
```

`notebooks/train_all_models.ipynb` runs the whole grid and plots the forecasts;
`notebooks/data_exploration.ipynb` is the descriptive analysis of the series.

## Method

Evaluation is rolling-origin. The first `n_train` hours (35,064 = 2016-2019) are the initial
training window. The model is fitted on that window, forecasts the next 24 hours, then the oldest
24 hours are dropped from the window, the 24 observed hours are appended, and training continues
from the current weights. This repeats until the end of 2021. The RMSE reported is over all
one-day-ahead forecasts of the test period, in MW.

Inputs are lagged values of the target, standardised with the mean and standard deviation of the
initial training window, optionally concatenated with unscaled calendar dummies. Three feature
sets are compared: `ts_only`, `weekend` (Saturday and Sunday dummies) and `business_hour`.

| Model | Input | Architecture | Optimiser |
|---|---|---|---|
| `mlp` | last `window` lags + dummies | 2 x Linear(10) + ReLU, Linear(1) | Adam 3e-3, MSE, early stopping (patience 3) |
| `lstm` | sequence of `window` steps, each (target + dummies) | 2 stacked LSTM(10), flatten, Linear(1) | same |
| `cnn` | same sequence layout | 2 x Conv1d(16, kernel 1) + ReLU, flatten, Linear(1) | same |
| `bmlp` | as `mlp` | 2 x BayesianLinear(10) + ReLU, BayesianLinear(1) | Adam 1e-2, ELBO (5 weight samples), 5 epochs per period |

The Bayesian layers follow Blundell et al. (2015): Gaussian variational posterior per weight,
scale-mixture Gaussian prior, KL term estimated by Monte Carlo and weighted by 1/N. Prediction is
the mean of 100 forward passes; the interval is mean +/- 3 standard deviations of those passes. The
interval captures uncertainty about the weights only, so its empirical coverage is well below the
nominal Gaussian value.

Default windows are 24 lags for `mlp` and 1 for `bmlp`, a sequence length of 1 for `lstm` and 2
for `cnn`; all can be changed with `--window`.

## Results from the thesis notebooks

RMSE in MW over the test period, as reported in `notebooks/original`. The MLP `ts_only` run used 24 lags only; the other MLP runs used the lags plus the dummies (`weekend`, `business_hour`). The Bayesian runs use the same distinctions (`ts_only`, `weekend`, `business_hour`). Coverage is the share of
observations inside the interval (5 standard deviations for the first two, 3 for the third).

| Model | ts_only | weekend | business_hour |
|---|---|---|---|
| MLP | 4907 | 2717 | 2939 |
| LSTM | 3181 | 2679 | 2569 |
| CNN | 2854 | 2393 | 1359 |
| Bayesian MLP | 2054 (coverage 0.42) | 2044 (0.30) | 1130 (0.52) |

Rerun `energyforecast --model all --features all` if you want to change the parameters and obtain new results.

## Layout

```
energyforecast/
  data.py        load and preprocess the ENTSO-E files
  windows.py     lag matrices, sequence windows, target scaling
  models.py      MLP, LSTM, CNN
  bayesian.py    BayesianLinear, BayesianMLP, ELBO
  training.py    rolling-origin trainer, result container, grid runner
  cli.py         command line interface
notebooks/
  data_exploration.ipynb    descriptive analysis (outputs kept)
  train_all_models.ipynb    runs the grid with the package
  original/                 thesis notebooks, outputs stripped, data loading replaced by the package
```

## Reference

Blundell, C., Cornebise, J., Kavukcuoglu, K., Wierstra, D. (2015). Weight uncertainty in neural
networks. ICML. https://arxiv.org/abs/1505.05424
