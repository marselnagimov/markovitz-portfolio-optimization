"""Unit tests for markowitz.optimize.

Uses a synthetic mean-return vector and covariance matrix so tests run
offline with no dependency on live market data.
"""

import numpy as np
import pandas as pd
import pytest

from markowitz.optimize import (
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    portfolio_performance,
    sharpe_ratio,
)

TICKERS = ["A", "B", "C"]


@pytest.fixture
def mean_returns():
    return pd.Series([0.08, 0.06, 0.15], index=TICKERS)


@pytest.fixture
def cov_matrix():
    # Positive semi-definite by construction (diagonal dominance)
    data = [
        [0.040, 0.010, 0.015],
        [0.010, 0.030, 0.005],
        [0.015, 0.005, 0.090],
    ]
    return pd.DataFrame(data, index=TICKERS, columns=TICKERS)


def test_weights_sum_to_one(mean_returns, cov_matrix):
    weights = min_variance_portfolio(mean_returns, cov_matrix)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-6)


def test_weights_within_bounds_long_only(mean_returns, cov_matrix):
    weights = min_variance_portfolio(mean_returns, cov_matrix, allow_short=False)
    assert np.all(weights >= -1e-8)
    assert np.all(weights <= 1 + 1e-8)


def test_min_variance_beats_equal_weight(mean_returns, cov_matrix):
    """The min-variance portfolio should have volatility <= the equal-weight portfolio."""
    min_var_w = min_variance_portfolio(mean_returns, cov_matrix)
    _, min_var_vol = portfolio_performance(min_var_w, mean_returns, cov_matrix)

    equal_w = np.repeat(1 / len(TICKERS), len(TICKERS))
    _, equal_vol = portfolio_performance(equal_w, mean_returns, cov_matrix)

    assert min_var_vol <= equal_vol + 1e-8


def test_max_sharpe_weights_sum_to_one(mean_returns, cov_matrix):
    weights = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=0.03)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-6)


def test_max_sharpe_has_highest_sharpe_among_candidates(mean_returns, cov_matrix):
    rf = 0.03
    max_sharpe_w = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=rf)
    ret, vol = portfolio_performance(max_sharpe_w, mean_returns, cov_matrix)
    best_sharpe = sharpe_ratio(ret, vol, rf)

    # Compare against several random long-only portfolios
    rng = np.random.default_rng(42)
    for _ in range(200):
        w = rng.dirichlet(np.ones(len(TICKERS)))
        r, v = portfolio_performance(w, mean_returns, cov_matrix)
        s = sharpe_ratio(r, v, rf)
        assert s <= best_sharpe + 1e-6


def test_efficient_frontier_shapes_match(mean_returns, cov_matrix):
    targets, vols = efficient_frontier(mean_returns, cov_matrix, n_points=20)
    assert len(targets) == len(vols) == 20
    assert np.all(np.diff(targets) >= 0)  # returns should be monotonically increasing


def test_portfolio_performance_basic():
    mean_returns = pd.Series([0.10, 0.10], index=["X", "Y"])
    cov_matrix = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=["X", "Y"], columns=["X", "Y"])
    weights = np.array([0.5, 0.5])
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    assert np.isclose(ret, 0.10)
    # Uncorrelated equal-variance assets: portfolio vol = sqrt(0.5^2*0.04 + 0.5^2*0.04)
    assert np.isclose(vol, np.sqrt(0.02), atol=1e-6)
