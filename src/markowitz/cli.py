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
    output_dir = Path(cfg["output_dir"])

    print(f"Downloading data for {tickers} from {start} to {end or 'today'}...")
    prices = download_prices(tickers, start, end)
    returns = compute_log_returns(prices)
    mean_returns, cov_matrix = annualize_stats(returns, trading_days)

    print("\nAnnualized mean returns:")
    print(mean_returns.to_string())
    asset_vols = (cov_matrix.values.diagonal() ** 0.5)
    print("\nAnnualized volatility:")
    for t, v in zip(tickers, asset_vols):
        print(f"  {t}: {v:.2%}")
    print()

    min_var_w = min_variance_portfolio(mean_returns, cov_matrix, allow_short)
    min_var_ret, min_var_vol = portfolio_performance(min_var_w, mean_returns, cov_matrix)

    max_sharpe_w = max_sharpe_portfolio(mean_returns, cov_matrix, rf, allow_short)
    max_sharpe_ret, max_sharpe_vol = portfolio_performance(max_sharpe_w, mean_returns, cov_matrix)

    print_portfolio("Minimum Variance Portfolio", tickers, min_var_w, min_var_ret, min_var_vol, rf)
    print_portfolio("Maximum Sharpe Ratio Portfolio", tickers, max_sharpe_w, max_sharpe_ret, max_sharpe_vol, rf)

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
    )
    print(f"Saved efficient frontier chart to {out_path}")


if __name__ == "__main__":
    main()
