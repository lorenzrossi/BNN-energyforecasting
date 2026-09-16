"""Command line entry point: python -m energyforecast ..."""

import argparse
from pathlib import Path

from .data import load_dataset
from .training import GRID, make_forecaster, run_grid
from .windows import FEATURE_SETS


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="energyforecast", description="Rolling-origin forecasts of Italian hourly generation."
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--years", type=int, nargs="+", default=list(range(2016, 2022)))
    p.add_argument("--model", choices=["mlp", "lstm", "cnn", "bmlp", "all"], default="all")
    p.add_argument("--features", choices=[*FEATURE_SETS, "all"], default="all")
    p.add_argument(
        "--window", type=int, default=None, help="lags (mlp, bmlp) or sequence length (lstm, cnn)"
    )
    p.add_argument(
        "--n-train", type=int, default=35064, help="hours in the initial training window"
    )
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument(
        "--max-periods", type=int, default=None, help="stop after this many forecast periods"
    )
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default="results", help="directory for CSV outputs")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    data = load_dataset(args.data_dir, args.years)
    kwargs = dict(
        window=args.window,
        horizon=args.horizon,
        device=args.device,
        seed=args.seed,
        verbose=not args.quiet,
    )
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs
    if args.lr is not None:
        kwargs["lr"] = args.lr

    models = ["mlp", "lstm", "cnn", "bmlp"] if args.model == "all" else [args.model]
    features = list(FEATURE_SETS) if args.features == "all" else [args.features]
    grid = [(m, f) for m, f in GRID if m in models and f in features]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(grid) == 1:
        model, feature_set = grid[0]
        result = make_forecaster(model, feature_set, **kwargs).run(
            data, n_train=args.n_train, max_periods=args.max_periods
        )
        path = out / f"{model}_{feature_set}.csv"
        result.frame().to_csv(path, index=False)
        print(f"wrote {path}")
    else:
        table, results = run_grid(
            data, grid, n_train=args.n_train, max_periods=args.max_periods, **kwargs
        )
        for (model, feature_set), result in results.items():
            result.frame().to_csv(out / f"{model}_{feature_set}.csv", index=False)
        table.to_csv(out / "summary.csv", index=False)
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
