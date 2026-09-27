"""Price download and return calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols, e.g. ["SPY", "VXUS", "QTUM"].
    start : str
        Start date, "YYYY-MM-DD".
    end : str | None
        End date, "YYYY-MM-DD". None means up to today.

    Returns
    -------
    pd.DataFrame
        Daily adjusted close prices, one column per ticker, NaNs dropped.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.dropna(how="any")
    if prices.empty:
        raise ValueError(
            "No price data returned. Check ticker symbols and date range."
        )
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a price series into daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def annualize_stats(
    returns: pd.DataFrame, trading_days: int = 252
) -> tuple[pd.Series, pd.DataFrame]:
    """Annualize mean returns and the covariance matrix from daily log returns."""
    mean_returns = returns.mean() * trading_days
    cov_matrix = returns.cov() * trading_days
    return mean_returns, cov_matrix
