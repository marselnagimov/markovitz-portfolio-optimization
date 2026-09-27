"""Command-line entry point: python -m markowitz.cli [options]"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from markowitz.data import annualize_stats, compute_log_returns, download_prices
from markowitz.optimize import (
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    portfolio_performance,
    sharpe_ratio,
)
from markowitz.plotting import plot_efficient_frontier


def load_config(path: str | None) -> dict:
    defaults = {
        "tickers": ["SPY", "VXUS", "QTUM"],
        "start_date": "2019-01-01",
        "end_date": None,
        "risk_free_rate": 0.04,
        "trading_days": 252,
        "allow_short": False,
        "output_dir": "outputs",
    }
    if path and Path(path).exists():
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        defaults.update(user_cfg)
    return defaults


def parse_args():
    parser = argparse.ArgumentParser(description="Markowitz mean-variance portfolio optimizer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to a YAML config file")
    parser.add_argument("--tickers", nargs="+", help="Override tickers, e.g. --tickers SPY VXUS QTUM")
    parser.add_argument("--start", type=str, help="Override start date, YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="Override end date, YYYY-MM-DD")
    parser.add_argument("--rf", type=float, help="Override annualized risk-free rate")
    parser.add_argument("--allow-short", action="store_true", help="Allow negative (short) weights")
    parser.add_argument("--points", type=int, default=100, help="Number of efficient frontier points")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Evaluate a specific weight mix instead of/alongside optimizing, "
        "in the same order as --tickers (or the config's tickers). "
        "Must sum to 1.0, e.g. --weights 0.6 0.2 0.2",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output folder for the chart (defaults to config's output_dir, "
        "or outputs/custom when --weights is used without this flag)",
    )
    return parser.parse_args()


def print_portfolio(name: str, tickers, weights, ret, vol, rf):
    print(f"=== {name} ===")
    for t, w in zip(tickers, weights):
        print(f"  {t}: {w:.2%}")
    print(f"  Expected return: {ret:.2%}  |  Volatility: {vol:.2%}  |  Sharpe: {sharpe_ratio(ret, vol, rf):.2f}\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    tickers = args.tickers or cfg["tickers"]
    start = args.start or cfg["start_date"]
    end = args.end or cfg["end_date"]
    rf = args.rf if args.rf is not None else cfg["risk_free_rate"]
    allow_short = args.allow_short or cfg["allow_short"]
    trading_days = cfg["trading_days"]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.weights:
        output_dir = Path("outputs/custom")
    else:
        output_dir = Path(cfg["output_dir"])

    custom_weights = None
    if args.weights:
        if len(args.weights) != len(tickers):
            raise SystemExit(
                f"--weights has {len(args.weights)} values but there are {len(tickers)} tickers "
                f"({tickers}). Provide one weight per ticker, in that order."
            )
        total = sum(args.weights)
        if abs(total - 1.0) > 1e-6:
            raise SystemExit(f"--weights must sum to 1.0 (got {total:.4f}).")
        custom_weights = args.weights

    print(f"Downloading data for {tickers} from {start} to {end or 'today'}...")
    prices = download_prices(tickers, start, end)
    returns = compute_log_returns(prices)
    mean_returns, cov_matrix = annualize_stats(returns, trading_days)

    # Defensive: make absolutely sure everything downstream is indexed in the
    # caller's ticker order, regardless of what order the data came back in.
    mean_returns = mean_returns.reindex(tickers)
    cov_matrix = cov_matrix.reindex(index=tickers, columns=tickers)

    print("\nAnnualized mean returns:")
    for t, r in mean_returns.items():
        print(f"  {t}: {r:.2%}")
    asset_vols = cov_matrix.values.diagonal() ** 0.5
    print("\nAnnualized volatility:")
    for t, v in zip(tickers, asset_vols):
        print(f"  {t}: {v:.2%}")
    print("\nCorrelation matrix:")
    print(returns.corr().round(2).to_string())
    print()

    min_var_w = min_variance_portfolio(mean_returns, cov_matrix, allow_short)
    min_var_ret, min_var_vol = portfolio_performance(min_var_w, mean_returns, cov_matrix)

    max_sharpe_w = max_sharpe_portfolio(mean_returns, cov_matrix, rf, allow_short)
    max_sharpe_ret, max_sharpe_vol = portfolio_performance(max_sharpe_w, mean_returns, cov_matrix)

    print_portfolio("Minimum Variance Portfolio", tickers, min_var_w, min_var_ret, min_var_vol, rf)
    print_portfolio("Maximum Sharpe Ratio Portfolio", tickers, max_sharpe_w, max_sharpe_ret, max_sharpe_vol, rf)

    custom_point = None
    if custom_weights is not None:
        import numpy as np

        custom_ret, custom_vol = portfolio_performance(np.array(custom_weights), mean_returns, cov_matrix)
        print_portfolio(
            f"Custom Portfolio ({', '.join(f'{t}={w:.0%}' for t, w in zip(tickers, custom_weights))})",
            tickers,
            custom_weights,
            custom_ret,
            custom_vol,
            rf,
        )
        custom_point = (custom_ret, custom_vol)

    frontier_returns, frontier_vols = efficient_frontier(
        mean_returns, cov_matrix, n_points=args.points, allow_short=allow_short
    )

    out_path = plot_efficient_frontier(
        tickers,
        mean_returns,
        cov_matrix,
        frontier_returns,
        frontier_vols,
        (min_var_ret, min_var_vol),
        (max_sharpe_ret, max_sharpe_vol),
        output_dir / "efficient_frontier.png",
        custom_point=custom_point,
        custom_label="Your Portfolio" if custom_point else None,
    )
    print(f"Saved efficient frontier chart to {out_path}")


if __name__ == "__main__":
    main()
