# Financial Toolkit v2

A Streamlit application for transparent DCF valuation and constrained portfolio analytics.

## Features

- Three-scenario FCFF DCF, WACC / terminal-growth sensitivity grid, editable historical inputs and forecast export.
- Two-stage and three-stage dividend-discount models with explicit company-type suitability guidance.
- P/E, P/B, P/S and EV/EBITDA valuation models with editable market and fundamental inputs.
- Constrained maximum-Sharpe, minimum-volatility and risk-parity portfolios.
- Historical VaR, expected shortfall, maximum drawdown, correlation and risk-contribution views.
- Portfolio workbench with CSV holdings import/export, session portfolios, target weights and executable rebalance trade lists.
- Benchmark comparison, security-level return contribution and cost-aware walk-forward backtesting.
- Fixed-income holdings analytics with cash-flow YTM, duration, convexity, DV01, spread risk and rate/spread scenario testing.
- Blended valuation summary with editable model weights, dispersion and confidence indicators.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Market data is provided by Yahoo Finance when available. Outputs are analytical estimates and not investment advice.
