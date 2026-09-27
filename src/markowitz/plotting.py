"""Efficient frontier plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_efficient_frontier(
    tickers: list[str],
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    frontier_returns: np.ndarray,
    frontier_vols: np.ndarray,
    min_var_point: tuple[float, float],
    max_sharpe_point: tuple[float, float],
    output_path: str | Path,
) -> Path:
    """Render and save the efficient frontier chart. Returns the saved path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    plt.plot(frontier_vols, frontier_returns, "b--", label="Efficient Frontier")

    asset_vols = np.sqrt(np.diag(cov_matrix.values))
    plt.scatter(asset_vols, mean_returns.values, marker="o", s=80, c="gray", label="Individual assets")
    for i, ticker in enumerate(tickers):
        plt.annotate(
            ticker,
            (asset_vols[i], mean_returns.values[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.scatter(min_var_point[1], min_var_point[0],
                marker="*", color="green", s=300, label="Min Variance Portfolio")
    plt.scatter(max_sharpe_point[1], max_sharpe_point[0],
                marker="*", color="red", s=300, label="Max Sharpe Portfolio")

    plt.xlabel("Annualized Volatility (Risk)")
    plt.ylabel("Annualized Expected Return")
    plt.title(f"Markowitz Efficient Frontier: {', '.join(tickers)}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
