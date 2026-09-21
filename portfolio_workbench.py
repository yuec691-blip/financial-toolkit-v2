"""Portfolio analyst workbench: holdings, rebalance, attribution and walk-forward backtest."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize


@st.cache_data(ttl=600, show_spinner=False)
def _download_prices(tickers, benchmark, start, end):
    universe = list(dict.fromkeys([*tickers, benchmark]))
    raw = yf.download(universe, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("No price data was returned. Check the tickers and date range.")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(universe[0])
    return prices.ffill().dropna(axis=1, how="all")


def _optimize(returns, rf, objective, min_weight, max_weight):
    n = returns.shape[1]
    if min_weight * n > 1 or max_weight * n < 1:
        raise ValueError("The position limits cannot form a fully invested portfolio.")
    covariance = returns.cov().to_numpy() * 252
    expected = returns.mean().to_numpy() * 252
    initial = np.repeat(1 / n, n)
    bounds = [(min_weight, max_weight)] * n
    constraint = {"type": "eq", "fun": lambda w: w.sum() - 1}
    if objective == "Minimum volatility":
        function = lambda w: np.sqrt(w @ covariance @ w)
    elif objective == "Risk parity":
        def function(w):
            portfolio_variance = w @ covariance @ w
            contributions = w * (covariance @ w) / max(portfolio_variance, 1e-12)
            return np.square(contributions - 1 / n).sum()
    else:
        function = lambda w: -((w @ expected - rf) / max(np.sqrt(w @ covariance @ w), 1e-12))
    result = minimize(function, initial, method="SLSQP", bounds=bounds, constraints=constraint)
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    return result.x


def _annual_metrics(series, rf=0.0):
    series = series.dropna()
    if series.empty:
        return {"CAGR": np.nan, "Volatility": np.nan, "Sharpe": np.nan, "Max drawdown": np.nan}
    cumulative = (1 + series).cumprod()
    years = len(series) / 252
    cagr = cumulative.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    volatility = series.std() * np.sqrt(252)
    sharpe = (series.mean() * 252 - rf) / volatility if volatility else np.nan
    drawdown = cumulative / cumulative.cummax() - 1
    return {"CAGR": cagr, "Volatility": volatility, "Sharpe": sharpe, "Max drawdown": drawdown.min()}


def _walk_forward(returns, benchmark_returns, objective, rf, min_weight, max_weight, train_days, rebalance_days, cost_bps):
    portfolio_parts, weight_rows = [], []
    previous = np.repeat(1 / returns.shape[1], returns.shape[1])
    for start in range(train_days, len(returns), rebalance_days):
        train = returns.iloc[start - train_days:start]
        test = returns.iloc[start:min(start + rebalance_days, len(returns))]
        if len(train) < train_days or test.empty:
            continue
        weights = _optimize(train, rf, objective, min_weight, max_weight)
        period = test @ weights
        turnover = np.abs(weights - previous).sum() / 2
        period.iloc[0] -= turnover * cost_bps / 10000
        portfolio_parts.append(period)
        weight_rows.append(pd.Series(weights, index=returns.columns, name=test.index[0]))
        previous = weights
    if not portfolio_parts:
        raise ValueError("Not enough history for the selected walk-forward settings.")
    portfolio = pd.concat(portfolio_parts).sort_index()
    benchmark = benchmark_returns.reindex(portfolio.index).dropna()
    portfolio = portfolio.reindex(benchmark.index)
    weights = pd.DataFrame(weight_rows)
    return portfolio, benchmark, weights


def render_portfolio_workbench():
    st.title("Portfolio analyst workbench")
    st.caption("Import holdings, create a target portfolio, generate a trade list, compare with a benchmark and run a no-look-ahead walk-forward backtest.")

    with st.expander("📘 How to use this app", expanded=True):
        st.markdown(
            """
            **Quick start**

            1. **Enter your holdings** below or upload a CSV with `Ticker` and `Shares`; `Cost basis / share` is optional.
            2. **Choose the mandate** in the sidebar: benchmark, optimization objective, position limits, rebalancing threshold and estimated trading cost.
            3. Select **Run portfolio workbench** to generate target weights, suggested trades, performance attribution and a walk-forward backtest.
            4. **Review before acting** and download the holdings or rebalance CSV when you are satisfied with the assumptions.
            """
        )
        guide_left, guide_right = st.columns(2)
        with guide_left:
            st.markdown(
                """
                **Choose the right analysis**

                - **Portfolio workbench:** analyze an existing portfolio and produce a rebalance plan.
                - **Fixed income risk:** measure bond yield, duration, DV01, spread risk and scenario P&L.
                - **Portfolio construction:** build a new allocation from a list of securities.
                - **DCF valuation:** value a company from projected free cash flow.
                """
            )
        with guide_right:
            st.markdown(
                """
                **Valuation tools**

                - **DDM & relative valuation:** use dividends or market multiples to estimate fair value.
                - **Valuation summary:** combine several valuation methods into one weighted estimate.
                - Move between modules using the **Analysis** menu in the left sidebar.
                """
            )
        st.info(
            "Market data is sourced from Yahoo Finance and may be delayed or incomplete. "
            "Model outputs are sensitive to assumptions and are for analysis and education, not investment advice."
        )

    default_holdings = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "NVDA", "GOOGL"],
        "Shares": [100., 80., 60., 75.],
        "Cost basis / share": [180., 410., 150., 185.],
    })
    st.session_state.setdefault("saved_portfolios", {})
    saved_names = ["Working portfolio", *sorted(st.session_state.saved_portfolios)]
    selected_saved = st.selectbox("Saved portfolio", saved_names)
    base_holdings = st.session_state.saved_portfolios.get(selected_saved, default_holdings).copy()

    uploaded = st.file_uploader("Import holdings CSV", type="csv", help="Required columns: Ticker, Shares. Optional: Cost basis / share.")
    if uploaded is not None:
        try:
            base_holdings = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            st.stop()
    if "Cost basis / share" not in base_holdings:
        base_holdings["Cost basis / share"] = 0.0
    missing = {"Ticker", "Shares"} - set(base_holdings.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        st.stop()

    holdings = st.data_editor(
        base_holdings[["Ticker", "Shares", "Cost basis / share"]], num_rows="dynamic", hide_index=True,
        use_container_width=True, key=f"holdings_{selected_saved}_{getattr(uploaded, 'name', 'manual')}",
    )
    holdings["Ticker"] = holdings["Ticker"].astype(str).str.upper().str.strip()
    holdings["Shares"] = pd.to_numeric(holdings["Shares"], errors="coerce").fillna(0.)
    holdings["Cost basis / share"] = pd.to_numeric(holdings["Cost basis / share"], errors="coerce").fillna(0.)
    holdings = holdings[(holdings["Ticker"] != "") & (holdings["Shares"] > 0)].copy()
    holdings = holdings.groupby("Ticker", as_index=False).agg({"Shares": "sum", "Cost basis / share": "mean"})

    save_left, save_right = st.columns([2, 1])
    portfolio_name = save_left.text_input("Portfolio name", "Core portfolio")
    if save_right.button("Save in this session", use_container_width=True):
        st.session_state.saved_portfolios[portfolio_name] = holdings.copy()
        st.success(f"Saved “{portfolio_name}” for this browser session.")
    st.download_button("Export current holdings CSV", holdings.to_csv(index=False), f"{portfolio_name.replace(' ', '_')}_holdings.csv", "text/csv")

    st.sidebar.header("Workbench settings")
    benchmark = st.sidebar.text_input("Benchmark ticker", "SPY", key="workbench_benchmark").upper().strip()
    objective = st.sidebar.selectbox("Target portfolio objective", ["Maximum Sharpe ratio", "Minimum volatility", "Risk parity"], key="workbench_objective")
    rf = st.sidebar.number_input("Risk-free rate (%)", 0., 20., 4., .1, key="workbench_rf") / 100
    min_weight = st.sidebar.number_input("Minimum target weight (%)", 0., 100., 0., .5, key="workbench_min") / 100
    max_weight = st.sidebar.number_input("Maximum target weight (%)", 1., 100., 40., .5, key="workbench_max") / 100
    drift_threshold = st.sidebar.number_input("Rebalance threshold (%)", 0., 20., 2., .5) / 100
    cost_bps = st.sidebar.number_input("Estimated trading cost (bps)", 0., 200., 10., 1.)
    whole_shares = st.sidebar.checkbox("Trade whole shares", value=True)
    history_years = st.sidebar.slider("Price history (years)", 3, 10, 5)
    train_days = st.sidebar.select_slider("Backtest training window", options=[126, 252, 504, 756], value=252, format_func=lambda x: f"{x} trading days")
    rebalance_days = st.sidebar.select_slider("Rebalance frequency", options=[21, 63, 126, 252], value=63, format_func=lambda x: f"Every {x} trading days")

    if not st.button("Run portfolio workbench", type="primary"):
        st.info("Review the holdings and settings, then select **Run portfolio workbench**.")
        return
    if holdings.empty:
        st.error("Enter at least one holding.")
        return

    tickers = list(dict.fromkeys(holdings["Ticker"]))
    try:
        with st.spinner("Loading market history, optimizing targets and running the walk-forward test..."):
            end = date.today() + timedelta(days=1)
            start = end - timedelta(days=365 * history_years)
            prices = _download_prices(tickers, benchmark, start, end)
            unavailable = [ticker for ticker in tickers if ticker not in prices]
            if unavailable:
                raise ValueError(f"No usable price history for: {', '.join(unavailable)}")
            asset_prices = prices[tickers].dropna()
            returns = asset_prices.pct_change().dropna()
            benchmark_returns = prices[benchmark].pct_change().dropna() if benchmark in prices else None
            if benchmark_returns is None:
                raise ValueError(f"No usable benchmark history for {benchmark}.")
            target_weights = _optimize(returns, rf, objective, min_weight, max_weight)
    except Exception as exc:
        st.error(str(exc))
        return

    latest_prices = asset_prices.iloc[-1]
    position = holdings.set_index("Ticker").reindex(tickers)
    position["Latest price"] = latest_prices
    position["Market value"] = position["Shares"] * position["Latest price"]
    position["Cost value"] = position["Shares"] * position["Cost basis / share"]
    position["Unrealized P/L"] = position["Market value"] - position["Cost value"]
    total_value = position["Market value"].sum()
    position["Current weight"] = position["Market value"] / total_value
    position["Target weight"] = target_weights
    position["Target value"] = position["Target weight"] * total_value
    position["Raw trade value"] = position["Target value"] - position["Market value"]
    position["Drift"] = position["Target weight"] - position["Current weight"]
    position["Trade value"] = np.where(position["Drift"].abs() >= drift_threshold, position["Raw trade value"], 0.)
    shares_to_trade = position["Trade value"] / position["Latest price"]
    position["Shares to trade"] = np.round(shares_to_trade) if whole_shares else shares_to_trade
    position["Estimated cost"] = position["Trade value"].abs() * cost_bps / 10000
    position["Action"] = np.select([position["Shares to trade"] > 0, position["Shares to trade"] < 0], ["BUY", "SELL"], default="HOLD")
    turnover = position["Trade value"].abs().sum() / (2 * total_value)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Portfolio market value", f"${total_value:,.0f}")
    m2.metric("Unrealized P/L", f"${position['Unrealized P/L'].sum():+,.0f}")
    m3.metric("One-way turnover", f"{turnover:.1%}")
    m4.metric("Estimated trading cost", f"${position['Estimated cost'].sum():,.0f}")
    m5.metric("Positions requiring trades", f"{(position['Action'] != 'HOLD').sum()}")

    tab1, tab2, tab3, tab4 = st.tabs(["Rebalance", "Performance", "Attribution", "Walk-forward backtest"])
    with tab1:
        trade_table = position.reset_index()[["Ticker", "Shares", "Latest price", "Market value", "Unrealized P/L", "Current weight", "Target weight", "Drift", "Action", "Shares to trade", "Trade value", "Estimated cost"]]
        st.dataframe(trade_table.style.format({
            "Shares": "{:,.2f}", "Latest price": "${:,.2f}", "Market value": "${:,.0f}",
            "Unrealized P/L": "${:+,.0f}", "Current weight": "{:.1%}", "Target weight": "{:.1%}", "Drift": "{:+.1%}",
            "Shares to trade": "{:,.2f}", "Trade value": "${:+,.0f}", "Estimated cost": "${:,.0f}",
        }), use_container_width=True)
        st.download_button("Download rebalance trade list", trade_table.to_csv(index=False), "rebalance_trade_list.csv", "text/csv")
        comparison = position.reset_index().melt(id_vars="Ticker", value_vars=["Current weight", "Target weight"], var_name="Portfolio", value_name="Weight")
        fig = px.bar(comparison, x="Ticker", y="Weight", color="Portfolio", barmode="group", title="Current vs. target weights")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    current_weights = position["Current weight"].to_numpy()
    portfolio_returns = returns @ current_weights
    aligned = pd.concat([portfolio_returns.rename("Portfolio"), benchmark_returns.rename(benchmark)], axis=1).dropna()
    with tab2:
        portfolio_metrics = _annual_metrics(aligned["Portfolio"], rf)
        benchmark_metrics = _annual_metrics(aligned[benchmark], rf)
        active = aligned["Portfolio"] - aligned[benchmark]
        tracking_error = active.std() * np.sqrt(252)
        information_ratio = active.mean() * 252 / tracking_error if tracking_error else np.nan
        metrics = pd.DataFrame([portfolio_metrics, benchmark_metrics], index=["Portfolio", benchmark])
        metrics["Tracking error"] = [tracking_error, np.nan]
        metrics["Information ratio"] = [information_ratio, np.nan]
        st.dataframe(metrics.style.format({"CAGR": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}", "Max drawdown": "{:.1%}", "Tracking error": "{:.1%}", "Information ratio": "{:.2f}"}), use_container_width=True)
        growth = (1 + aligned).cumprod()
        st.plotly_chart(px.line(growth, title="Historical growth of $1", labels={"value": "Value", "index": "Date", "variable": "Series"}), use_container_width=True)

    with tab3:
        annual_asset_returns = returns.mean() * 252
        attribution = pd.DataFrame({"Ticker": tickers, "Current weight": current_weights, "Annualized asset return": annual_asset_returns.reindex(tickers).values})
        attribution["Annualized return contribution"] = attribution["Current weight"] * attribution["Annualized asset return"]
        st.dataframe(attribution.sort_values("Annualized return contribution", ascending=False).style.format({"Current weight": "{:.1%}", "Annualized asset return": "{:+.1%}", "Annualized return contribution": "{:+.1%}"}), use_container_width=True)
        fig = px.bar(attribution.sort_values("Annualized return contribution"), x="Ticker", y="Annualized return contribution", title="Security-level return contribution")
        fig.update_yaxes(tickformat="+.1%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This is fixed-weight security contribution analysis. Full Brinson allocation/selection attribution requires benchmark constituent weights.")

    with tab4:
        try:
            backtest, benchmark_test, weight_history = _walk_forward(returns, benchmark_returns, objective, rf, min_weight, max_weight, train_days, rebalance_days, cost_bps)
            backtest_frame = pd.concat([backtest.rename("Walk-forward portfolio"), benchmark_test.rename(benchmark)], axis=1).dropna()
            backtest_metrics = pd.DataFrame([_annual_metrics(backtest_frame.iloc[:, 0], rf), _annual_metrics(backtest_frame.iloc[:, 1], rf)], index=backtest_frame.columns)
            st.dataframe(backtest_metrics.style.format({"CAGR": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}", "Max drawdown": "{:.1%}"}), use_container_width=True)
            st.plotly_chart(px.line((1 + backtest_frame).cumprod(), title="Out-of-sample walk-forward performance", labels={"value": "Value", "index": "Date", "variable": "Series"}), use_container_width=True)
            if not weight_history.empty:
                st.plotly_chart(px.area(weight_history, title="Target weights through time", labels={"value": "Weight", "index": "Rebalance date", "variable": "Ticker"}), use_container_width=True)
            st.caption(f"Each rebalance uses only the preceding {train_days} trading days, holds for {rebalance_days} trading days and deducts {cost_bps:.0f} bps of estimated cost on turnover.")
        except Exception as exc:
            st.warning(str(exc))
