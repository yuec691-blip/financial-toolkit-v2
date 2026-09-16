# Financial Toolkit v2

A Streamlit application for transparent DCF valuation and constrained portfolio analytics.

## Features

- Three-scenario FCFF DCF, WACC / terminal-growth sensitivity grid, editable historical inputs and forecast export.
- Constrained maximum-Sharpe, minimum-volatility and risk-parity portfolios.
- Historical VaR, expected shortfall, maximum drawdown, correlation and risk-contribution views.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Market data is provided by Yahoo Finance when available. Outputs are analytical estimates and not investment advice.
