# Markowitz Portfolio Optimizer

Mean-variance portfolio optimization for a configurable basket of tickers
(defaults to **SPY**, **VXUS**, and **QTUM**), built on the classic
Markowitz framework.

Given historical prices, this project computes:

- Annualized expected returns and the covariance matrix
- The **Minimum Variance** portfolio
- The **Maximum Sharpe Ratio** (tangency) portfolio
- The full **Efficient Frontier**
- A plot of the frontier with individual assets and both special portfolios marked

## Project layout

```
markowitz-portfolio-optimizer/
├── config.yaml               # tickers, date range, risk-free rate
├── src/markowitz/
│   ├── data.py                # price download + return calculations
│   ├── optimize.py            # min-variance, max-Sharpe, efficient frontier
│   ├── plotting.py            # frontier chart
│   └── cli.py                 # command-line entry point
├── tests/
│   └── test_optimize.py       # unit tests (synthetic data, no network calls)
├── outputs/                   # generated charts land here (gitignored)
├── requirements.txt
├── pyproject.toml
└── .github/workflows/ci.yml   # lint + test on push
```

## Setup

```bash
git clone https://github.com/<your-username>/markowitz-portfolio-optimizer.git
cd markowitz-portfolio-optimizer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run with the default config (SPY, VXUS, QTUM):

```bash
python -m markowitz.cli
```

Or point it at your own config:

```bash
python -m markowitz.cli --config path/to/config.yaml
```

Useful flags:

```bash
python -m markowitz.cli --tickers SPY VXUS QTUM BND --start 2020-01-01 --rf 0.04
```

Run `python -m markowitz.cli --help` for the full list of options.

Output:
- Console summary of the minimum-variance and max-Sharpe portfolios (weights,
  expected return, volatility, Sharpe ratio)
- `outputs/efficient_frontier.png`

## Configuration

Edit `config.yaml`:

```yaml
tickers: ["SPY", "VXUS", "QTUM"]
start_date: "2019-01-01"
end_date: null          # null = today
risk_free_rate: 0.04    # annualized
trading_days: 252
allow_short: false      # true = weights can go negative
```

## Running tests

```bash
pip install -r requirements.txt
pytest
```

Tests use a synthetic covariance matrix and mean-return vector, so they run
offline with no market data dependency.

## Notes and caveats

- Markowitz optimization is highly sensitive to the sample window used to
  estimate expected returns and covariances; results are **not** predictive
  of future performance.
- QTUM's inception was September 2018, which caps how far back its history
  goes.
- This is an educational/analytical tool, not investment advice.

## License

MIT — see [LICENSE](LICENSE).
