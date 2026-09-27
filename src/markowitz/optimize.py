"""Core Markowitz mean-variance optimization routines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_performance(
    weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame
) -> tuple[float, float]:
    """Return (expected_return, volatility) for a given weight vector."""
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(weights.T @ cov_matrix.values @ weights))
    return ret, vol


def _neg_sharpe(weights, mean_returns, cov_matrix, rf):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - rf) / vol


def _vol(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]


def _bounds(n_assets: int, allow_short: bool):
    return tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(n_assets))


def min_variance_portfolio(
    mean_returns: pd.Series, cov_matrix: pd.DataFrame, allow_short: bool = False
):
    """Solve for the global minimum-variance portfolio."""
    n = len(mean_returns)
    init = np.repeat(1 / n, n)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    result = minimize(
        _vol,
        init,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=_bounds(n, allow_short),
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Min-variance optimization failed: {result.message}")
    return result.x


def max_sharpe_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    allow_short: bool = False,
):
    """Solve for the maximum Sharpe ratio (tangency) portfolio."""
    n = len(mean_returns)
    init = np.repeat(1 / n, n)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    result = minimize(
        _neg_sharpe,
        init,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=_bounds(n, allow_short),
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {result.message}")
    return result.x


def efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 100,
    allow_short: bool = False,
):
    """Trace the efficient frontier by sweeping target returns.

    Returns
    -------
    (target_returns, volatilities) : tuple[np.ndarray, np.ndarray]
    """
    n = len(mean_returns)
    init = np.repeat(1 / n, n)
    targets = np.linspace(mean_returns.min(), mean_returns.max(), n_points)
    vols = []
    for target in targets:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) - t},
        )
        result = minimize(
            _vol,
            init,
            args=(mean_returns, cov_matrix),
            method="SLSQP",
            bounds=_bounds(n, allow_short),
            constraints=constraints,
        )
        vols.append(result.fun if result.success else np.nan)
    return targets, np.array(vols)


def sharpe_ratio(ret: float, vol: float, risk_free_rate: float) -> float:
    return (ret - risk_free_rate) / vol
