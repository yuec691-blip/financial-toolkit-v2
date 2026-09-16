"""Financial Toolkit v2 — scenario DCF and constrained portfolio analytics."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize


st.set_page_config(page_title="Financial Toolkit v2", page_icon="📊", layout="wide")

st.markdown("""
<style>
  .block-container {max-width: 1450px; padding-top: 2rem;}
  [data-testid="stMetric"] {
    background:#f7f9fc;
    border:1px solid #e7ebf1;
    border-radius:12px;
    padding:14px;
    color:#101828 !important;
  }
  [data-testid="stMetric"] * {
    color:#101828 !important;
  }
  .stAlert {border-radius:10px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def prices_for(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("No market-price data was returned. Check tickers and dates.")
    data = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    return data.dropna(axis=1, how="all").ffill().dropna()


@st.cache_data(ttl=300, show_spinner=False)
def company_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    financials, cashflow = stock.financials, stock.cashflow
    latest_price = None
    price_as_of = "Unavailable"
    try:
        latest_price = float(stock.fast_info.last_price)
    except Exception:
        pass
    try:
        quote_history = stock.history(period="5d", interval="1d", auto_adjust=False)
        if not quote_history.empty:
            if latest_price is None:
                latest_price = float(quote_history["Close"].dropna().iloc[-1])
            price_as_of = str(quote_history.index[-1].date())
    except Exception:
        pass
    fallback = pd.DataFrame({"Year": ["2022", "2023", "2024"], "Revenue": [1000., 1100., 1200.],
                             "EBIT": [150., 165., 180.], "D&A": [35., 38., 40.],
                             "CapEx": [-55., -58., -60.], "Change in WC": [-15., -16., -18.]})
    result = {
        "price": float(latest_price or info.get("currentPrice") or info.get("regularMarketPrice") or 100),
        "price_as_of": price_as_of,
        "shares": float(info.get("sharesOutstanding") or 1_000_000_000) / 1e6,
        "debt": float(info.get("totalDebt") or 0) / 1e6,
        "cash": float(info.get("totalCash") or 0) / 1e6,
        "beta": float(info.get("beta") or 1.0), "history": fallback, "live": not financials.empty,
    }
    if financials.empty:
        return result

    def get(frame, labels, col):
        for label in labels:
            if label in frame.index and pd.notna(frame.loc[label, col]):
                return float(frame.loc[label, col]) / 1e6
        return 0.0

    rows = []
    for col in reversed(financials.columns[:4]):
        rows.append({"Year": str(col.year), "Revenue": get(financials, ["Total Revenue", "Operating Revenue"], col),
                     "EBIT": get(financials, ["EBIT", "Operating Income"], col),
                     "D&A": get(cashflow, ["Depreciation And Amortization", "Depreciation"], col),
                     "CapEx": get(cashflow, ["Capital Expenditure"], col),
                     "Change in WC": get(cashflow, ["Change In Working Capital"], col)})
    result["history"] = pd.DataFrame(rows)
    return result


def portfolio_stats(weights, returns, rf):
    annual_return = float(returns.mean().to_numpy() @ weights * 252)
    annual_vol = float(np.sqrt(weights @ (returns.cov().to_numpy() * 252) @ weights))
    return annual_return, annual_vol, (annual_return - rf) / annual_vol if annual_vol else 0


def optimise(returns, rf, strategy, min_weight, max_weight):
    n = returns.shape[1]
    covariance = returns.cov().to_numpy() * 252
    average = returns.mean().to_numpy() * 252
    initial = np.repeat(1 / n, n)
    bounds = [(min_weight, max_weight)] * n
    constraint = {"type": "eq", "fun": lambda x: x.sum() - 1}
    if strategy == "Risk parity":
        def objective(w):
            risk = w * (covariance @ w)
            return np.square(risk - risk.mean()).sum()
    elif strategy == "Minimum volatility":
        objective = lambda w: np.sqrt(w @ covariance @ w)
    else:
        objective = lambda w: -((w @ average - rf) / np.sqrt(w @ covariance @ w))
    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraint)
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    return result.x


def dcf_value(base_revenue, assumptions, capital, scenario):
    growth, margin, wacc, terminal_growth = scenario
    years = assumptions["years"]
    forecast, revenue = [], base_revenue
    for year in range(1, years + 1):
        revenue *= 1 + growth
        ebit = revenue * margin
        nopat = ebit * (1 - assumptions["tax"])
        da = revenue * assumptions["da"]
        capex = revenue * abs(assumptions["capex"])
        nwc = revenue * abs(assumptions["nwc"])
        fcff = nopat + da - capex - nwc
        forecast.append([year, revenue, ebit, nopat, fcff, fcff / (1 + wacc) ** year])
    frame = pd.DataFrame(forecast, columns=["Year", "Revenue", "EBIT", "NOPAT", "FCFF", "PV of FCFF"])
    terminal = frame.iloc[-1]["FCFF"] * (1 + terminal_growth) / (wacc - terminal_growth)
    ev = frame["PV of FCFF"].sum() + terminal / (1 + wacc) ** years
    equity = ev + capital["cash"] - capital["debt"]
    return equity / capital["shares"], frame, ev, terminal / (1 + wacc) ** years


def app_header(title, subtitle):
    st.title(title)
    st.caption(subtitle)


mode = st.sidebar.radio("Analysis", ["Portfolio construction", "DCF valuation"], label_visibility="collapsed")
st.sidebar.divider()

if mode == "Portfolio construction":
    app_header("Portfolio construction", "Constrained allocation, risk attribution and historical stress testing. This is analysis, not investment advice.")
    st.sidebar.header("1. Investment universe")
    symbols = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,MSFT,NVDA,GOOGL,AMZN,META").upper()
    tickers = list(dict.fromkeys(x.strip() for x in symbols.split(",") if x.strip()))
    st.sidebar.header("2. Portfolio mandate")
    strategy = st.sidebar.selectbox("Objective", ["Maximum Sharpe ratio", "Minimum volatility", "Risk parity"])
    investment = st.sidebar.number_input("Portfolio value ($)", min_value=1000, value=100000, step=5000)
    rf = st.sidebar.number_input("Risk-free rate (%)", min_value=0., value=4.0, step=.1) / 100
    confidence = st.sidebar.slider("VaR / expected shortfall confidence", .90, .99, .95)
    st.sidebar.header("3. Constraints")
    min_weight = st.sidebar.number_input("Minimum position (%)", 0., 100., 0., .5) / 100
    max_weight = st.sidebar.number_input("Maximum position (%)", 1., 100., 35., .5) / 100
    end = date.today()
    start = st.sidebar.date_input("Historical start", end - timedelta(days=365 * 3))
    run = st.sidebar.button("Build portfolio", type="primary")

    if not run:
        st.info("Set the mandate in the sidebar, then select **Build portfolio**.")
        st.stop()
    if not tickers or min_weight * len(tickers) > 1 or max_weight * len(tickers) < 1:
        st.error("The position limits cannot form a fully invested portfolio. Adjust the universe or allocation limits.")
        st.stop()
    try:
        with st.spinner("Downloading prices and optimizing the portfolio..."):
            price_data = prices_for(tickers, start, end)
            daily = price_data.pct_change().dropna()
            weights = optimise(daily, rf, strategy, min_weight, max_weight)
        tickers = list(daily.columns)
        ret, vol, sharpe = portfolio_stats(weights, daily, rf)
        portfolio_returns = daily @ weights
        var = np.quantile(portfolio_returns, 1 - confidence)
        es = portfolio_returns[portfolio_returns <= var].mean()
        cov = daily.cov().to_numpy() * 252
        marginal = cov @ weights / vol
        risk_contrib = weights * marginal / vol
        allocation = pd.DataFrame({"Ticker": tickers, "Weight": weights, "Dollar allocation": weights * investment,
                                   "Risk contribution": risk_contrib}).sort_values("Weight", ascending=False)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected annual return", f"{ret:.1%}")
        c2.metric("Annualized volatility", f"{vol:.1%}")
        c3.metric("Sharpe ratio", f"{sharpe:.2f}")
        c4.metric(f"Expected shortfall ({confidence:.0%})", f"${abs(es) * investment:,.0f}")
        st.caption(f"Price history: {daily.index.min().date()} to {daily.index.max().date()} · {len(daily)} observations · Objective: {strategy}")
        tab1, tab2, tab3 = st.tabs(["Allocation", "Risk attribution", "Stress & history"])
        with tab1:
            left, right = st.columns([1, 1.2])
            with left:
                st.plotly_chart(px.pie(allocation, names="Ticker", values="Weight", hole=.55, title="Recommended weights"), use_container_width=True)
            with right:
                st.dataframe(allocation.style.format({"Weight": "{:.1%}", "Dollar allocation": "${:,.0f}", "Risk contribution": "{:.1%}"}), use_container_width=True)
            st.download_button("Download allocation CSV", allocation.to_csv(index=False), "portfolio_allocation.csv", "text/csv")
        with tab2:
            chart = px.bar(allocation.sort_values("Risk contribution"), x="Ticker", y="Risk contribution", title="Contribution to total portfolio risk")
            chart.update_yaxes(tickformat=".0%")
            st.plotly_chart(chart, use_container_width=True)
            st.plotly_chart(px.imshow(daily.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Return correlation"), use_container_width=True)
        with tab3:
            cumulative = (1 + portfolio_returns).cumprod()
            drawdown = cumulative / cumulative.cummax() - 1
            max_dd = drawdown.min()
            s1, s2, s3 = st.columns(3)
            s1.metric("Historical VaR (daily)", f"{var:.2%}")
            s2.metric("Max drawdown", f"{max_dd:.1%}")
            s3.metric("Worst daily return", f"{portfolio_returns.min():.2%}")
            st.plotly_chart(px.line(cumulative, title="Growth of $1", labels={"value": "Portfolio value", "index": "Date"}), use_container_width=True)
            st.plotly_chart(px.line(drawdown, title="Historical drawdown", labels={"value": "Drawdown", "index": "Date"}), use_container_width=True)
    except Exception as exc:
        st.error(str(exc))

else:
    app_header("DCF valuation", "A transparent FCFF valuation with downside, base and upside scenarios. Figures are estimates, not a price target or investment recommendation.")
    st.sidebar.header("1. Company and capital structure")
    ticker = st.sidebar.text_input("Ticker", "AAPL").upper().strip()
    if st.sidebar.button("Refresh latest market data"):
        company_data.clear()
        st.rerun()
    try:
        with st.spinner("Loading company data..."):
            data = company_data(ticker)
    except Exception:
        data = {"price": 100., "price_as_of": "Unavailable", "shares": 1000., "debt": 0., "cash": 0., "beta": 1., "history": pd.DataFrame() , "live": False}
    price = st.sidebar.number_input(
        "Current share price ($)", min_value=.01, value=float(data["price"]), key=f"current_price_{ticker}"
    )
    shares = st.sidebar.number_input("Diluted shares outstanding (millions)", min_value=.1, value=float(data["shares"]), step=10.)
    debt = st.sidebar.number_input("Total debt ($m)", min_value=0., value=float(data["debt"]), step=100.)
    cash = st.sidebar.number_input("Cash and investments ($m)", min_value=0., value=float(data["cash"]), step=100.)
    st.sidebar.header("2. Operating assumptions")
    years = st.sidebar.slider("Forecast years", 3, 10, 5)
    tax = st.sidebar.number_input("Cash tax rate (%)", 0., 60., 21., .5) / 100
    st.sidebar.header("3. Discount rate")
    beta = st.sidebar.number_input("Beta", 0., 5., float(data["beta"]), .05)
    rf = st.sidebar.number_input("Risk-free rate (%)", 0., 20., 4., .1) / 100
    erp = st.sidebar.number_input("Equity risk premium (%)", 0., 20., 5.5, .1) / 100
    cod = st.sidebar.number_input("Pre-tax cost of debt (%)", 0., 30., 4.5, .1) / 100
    market_cap = price * shares
    total_capital = market_cap + debt
    wacc = (market_cap / total_capital) * (rf + beta * erp) + (debt / total_capital) * cod * (1 - tax)

    history = st.data_editor(data["history"], num_rows="dynamic", use_container_width=True, key="financial_history")
    if history.empty or "Revenue" not in history or history["Revenue"].iloc[-1] <= 0:
        st.error("Enter at least one historical financial year with positive revenue.")
        st.stop()
    base_revenue = float(history["Revenue"].iloc[-1])
    implied_margin = float(history["EBIT"].iloc[-1] / base_revenue) if base_revenue else .15
    da = abs(float(history["D&A"].iloc[-1] / base_revenue))
    capex = abs(float(history["CapEx"].iloc[-1] / base_revenue))
    nwc = abs(float(history["Change in WC"].iloc[-1] / base_revenue))
    a, b, c = st.columns(3)
    with a:
        base_growth = st.number_input("Base revenue growth (%)", -50., 100., 8., .5) / 100
        base_margin = st.number_input("Base EBIT margin (%)", -50., 80., float(implied_margin * 100), .5) / 100
    with b:
        terminal_growth = st.number_input("Terminal growth (%)", -2., float(wacc * 100 - .1), 2.5, .1) / 100
        da_margin = st.number_input("D&A as % revenue", 0., 50., float(da * 100), .1) / 100
    with c:
        capex_margin = st.number_input("CapEx as % revenue", 0., 50., float(capex * 100), .1) / 100
        nwc_margin = st.number_input("Change in NWC as % revenue", 0., 50., float(nwc * 100), .1) / 100
    assumptions = {"years": years, "tax": tax, "da": da_margin, "capex": capex_margin, "nwc": nwc_margin}
    capital = {"shares": shares, "debt": debt, "cash": cash}
    scenarios = {"Downside": (base_growth - .04, base_margin - .03, wacc + .01, max(-.01, terminal_growth - .005)),
                 "Base": (base_growth, base_margin, wacc, terminal_growth),
                 "Upside": (base_growth + .04, base_margin + .03, max(.001, wacc - .01), terminal_growth + .005)}
    outputs = {name: dcf_value(base_revenue, assumptions, capital, inputs) for name, inputs in scenarios.items()}
    base_value, forecast, ev, pv_terminal = outputs["Base"]
    upside = base_value / price - 1
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"{ticker} latest market price", f"${price:,.2f}")
    m2.metric("Base fair value", f"${base_value:,.2f}", f"{upside:+.1%} vs. market")
    m3.metric("WACC", f"{wacc:.2%}")
    m4.metric("Terminal value / EV", f"{pv_terminal / ev:.1%}")
    m5.metric("Latest revenue", f"${base_revenue:,.0f}m")
    st.caption(
        f"Latest quoted session: {data.get('price_as_of', 'Unavailable')} · "
        "Market and financial data are sourced from Yahoo Finance where available and may be delayed. "
        "You can override the price and all key inputs in the sidebar."
    )
    tab1, tab2, tab3 = st.tabs(["Scenarios", "Sensitivity", "Forecast & bridge"])
    with tab1:
        scenario_table = pd.DataFrame({"Scenario": list(outputs), "Fair value / share": [x[0] for x in outputs.values()],
                                       "Upside / (downside)": [x[0] / price - 1 for x in outputs.values()],
                                       "Revenue growth": [x[0] for x in scenarios.values()], "EBIT margin": [x[1] for x in scenarios.values()], "WACC": [x[2] for x in scenarios.values()]})
        st.dataframe(scenario_table.style.format({"Fair value / share": "${:,.2f}", "Upside / (downside)": "{:+.1%}", "Revenue growth": "{:.1%}", "EBIT margin": "{:.1%}", "WACC": "{:.1%}"}), use_container_width=True)
        fig = go.Figure()
        for name, output in outputs.items():
            fig.add_trace(go.Scatter(x=output[1]["Year"], y=output[1]["FCFF"], mode="lines+markers", name=name))
        fig.update_layout(title="FCFF by scenario ($m)", xaxis_title="Forecast year", yaxis_title="FCFF ($m)")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        wacc_range = np.round(np.linspace(wacc - .015, wacc + .015, 7), 4)
        growth_range = np.round(np.linspace(terminal_growth - .01, terminal_growth + .01, 7), 4)
        grid = pd.DataFrame(index=[f"{x:.1%}" for x in wacc_range], columns=[f"{x:.1%}" for x in growth_range])
        for wr in wacc_range:
            for gr in growth_range:
                if gr < wr:
                    grid.loc[f"{wr:.1%}", f"{gr:.1%}"] = dcf_value(base_revenue, assumptions, capital, (base_growth, base_margin, wr, gr))[0]
        fig = px.imshow(grid.astype(float), text_auto="$.0f", color_continuous_scale="RdYlGn", aspect="auto", labels={"x": "Terminal growth", "y": "WACC", "color": "Fair value"}, title="DCF sensitivity: fair value per share")
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        left, right = st.columns([1.5, 1])
        with left:
            st.dataframe(forecast.style.format({x: "{:,.1f}" for x in ["Revenue", "EBIT", "NOPAT", "FCFF", "PV of FCFF"]}), use_container_width=True)
        with right:
            bridge = pd.DataFrame({"Item": ["PV of forecast FCFF", "PV of terminal value", "Enterprise value", "Cash", "Debt", "Equity value"],
                                   "$m": [forecast["PV of FCFF"].sum(), pv_terminal, ev, cash, -debt, ev + cash - debt]})
            st.dataframe(bridge.style.format({"$m": "{:,.1f}"}), use_container_width=True)
        st.download_button("Download base-case forecast CSV", forecast.to_csv(index=False), f"{ticker}_dcf_forecast.csv", "text/csv")
