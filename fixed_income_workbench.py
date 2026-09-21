"""Fixed-income holdings analytics, cash-flow risk and scenario testing."""
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq


REQUIRED_COLUMNS = {
    "Bond ID",
    "Issuer",
    "Sector",
    "Rating",
    "Coupon (%)",
    "Maturity",
    "Frequency",
    "Face value",
    "Clean price",
}


def sample_bond_portfolio(as_of=None):
    """Return a diversified demonstration portfolio with future maturity dates."""
    as_of = pd.Timestamp(as_of or date.today())
    years = [2, 4, 6, 8, 11, 15, 20]
    maturities = [(as_of + pd.DateOffset(years=year)).date().isoformat() for year in years]
    return pd.DataFrame(
        {
            "Bond ID": ["UST-2Y", "UST-5Y", "FNMA-6Y", "MSFT-8Y", "JPM-11Y", "FORD-15Y", "UST-20Y"],
            "Issuer": ["US Treasury", "US Treasury", "Fannie Mae", "Microsoft", "JPMorgan", "Ford", "US Treasury"],
            "Sector": ["Treasury", "Treasury", "Agency", "Technology", "Financials", "Consumer Cyclical", "Treasury"],
            "Rating": ["AAA", "AAA", "AA+", "AAA", "A-", "BB+", "AAA"],
            "Coupon (%)": [4.25, 4.00, 4.50, 4.30, 5.10, 6.60, 4.75],
            "Maturity": maturities,
            "Frequency": [2, 2, 2, 2, 2, 2, 2],
            "Face value": [1_000_000, 1_250_000, 800_000, 900_000, 850_000, 500_000, 700_000],
            "Clean price": [99.85, 99.10, 100.40, 99.50, 101.20, 97.30, 103.10],
            "OAS (bps)": [0, 0, 28, 52, 92, 285, 0],
            "Spread duration": [0.0, 0.0, 5.4, 6.8, 8.1, 9.2, 0.0],
        }
    )


def _cash_flows(coupon_rate, maturity, frequency, as_of):
    """Approximate a regular fixed-rate bond's remaining cash flows per $100 par."""
    maturity = pd.Timestamp(maturity)
    as_of = pd.Timestamp(as_of)
    years = max((maturity - as_of).days / 365.25, 0.0)
    if years <= 0:
        raise ValueError("Maturity must be after the analysis date.")
    frequency = int(frequency)
    if frequency not in {1, 2, 4, 12}:
        raise ValueError("Coupon frequency must be 1, 2, 4 or 12.")
    periods = max(1, int(np.ceil(years * frequency)))
    first_time = max(years - (periods - 1) / frequency, 1 / 365.25)
    times = first_time + np.arange(periods) / frequency
    coupon = 100 * coupon_rate / frequency
    cash_flows = np.repeat(coupon, periods)
    cash_flows[-1] += 100
    fraction_to_next_coupon = min(first_time * frequency, 1.0)
    accrued_per_100 = coupon * (1 - fraction_to_next_coupon)
    return times, cash_flows, accrued_per_100, years


def _price_from_yield(times, cash_flows, annual_yield, frequency):
    base = 1 + annual_yield / frequency
    if base <= 0:
        return np.inf
    return float(np.sum(cash_flows / np.power(base, times * frequency)))


def bond_analytics(row, as_of):
    """Calculate yield and interest-rate risk for one regular fixed-rate bond."""
    coupon_rate = float(row["Coupon (%)"]) / 100
    frequency = int(row["Frequency"])
    clean_price = float(row["Clean price"])
    face_value = float(row["Face value"])
    times, cash_flows, accrued, years = _cash_flows(coupon_rate, row["Maturity"], frequency, as_of)
    dirty_price = clean_price + accrued

    def difference(yield_value):
        return _price_from_yield(times, cash_flows, yield_value, frequency) - dirty_price

    try:
        ytm = brentq(difference, -0.95, 3.0)
    except (ValueError, RuntimeError):
        ytm = np.nan

    if np.isfinite(ytm):
        discount_base = 1 + ytm / frequency
        present_values = cash_flows / np.power(discount_base, times * frequency)
        macaulay = float(np.sum(times * present_values) / np.sum(present_values))
        modified = macaulay / discount_base
        convexity = float(
            np.sum(present_values * times * (times + 1 / frequency))
            / (np.sum(present_values) * discount_base**2)
        )
    else:
        macaulay = modified = convexity = np.nan

    market_value = face_value * clean_price / 100
    supplied_spread_duration = row.get("Spread duration", 0.0)
    spread_duration = float(supplied_spread_duration) if pd.notna(supplied_spread_duration) else 0.0
    return {
        "Years to maturity": years,
        "Accrued / 100": accrued,
        "Dirty price": dirty_price,
        "YTM": ytm,
        "Current yield": coupon_rate * 100 / clean_price,
        "Macaulay duration": macaulay,
        "Modified duration": modified,
        "Convexity": convexity,
        "Market value": market_value,
        "DV01": modified * market_value * 0.0001,
        "Spread DV01": spread_duration * market_value * 0.0001,
    }


def analyze_portfolio(holdings, as_of):
    """Validate and enrich a holdings table with fixed-income analytics."""
    missing = REQUIRED_COLUMNS - set(holdings.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    frame = holdings.copy()
    frame["Maturity"] = pd.to_datetime(frame["Maturity"], errors="coerce")
    numeric_columns = ["Coupon (%)", "Frequency", "Face value", "Clean price", "OAS (bps)", "Spread duration"]
    for column in numeric_columns:
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["OAS (bps)"] = frame["OAS (bps)"].fillna(0.0)
    frame["Spread duration"] = frame["Spread duration"].fillna(0.0)
    frame = frame.dropna(subset=["Maturity", "Coupon (%)", "Frequency", "Face value", "Clean price"])
    frame = frame[(frame["Face value"] > 0) & (frame["Clean price"] > 0)]
    frame = frame[frame["Maturity"] > pd.Timestamp(as_of)]
    if frame.empty:
        raise ValueError("No valid bonds remain after validation. Check prices, face values and maturity dates.")

    analytics = pd.DataFrame([bond_analytics(row, as_of) for _, row in frame.iterrows()], index=frame.index)
    result = pd.concat([frame, analytics], axis=1)
    result["Weight"] = result["Market value"] / result["Market value"].sum()
    result["Maturity bucket"] = pd.cut(
        result["Years to maturity"],
        bins=[0, 3, 5, 7, 10, 20, np.inf],
        labels=["0–3Y", "3–5Y", "5–7Y", "7–10Y", "10–20Y", "20Y+"],
        right=False,
    )
    return result.reset_index(drop=True)


def portfolio_summary(analyzed):
    weights = analyzed["Weight"]
    return {
        "Market value": analyzed["Market value"].sum(),
        "YTM": float(np.nansum(weights * analyzed["YTM"])),
        "Duration": float(np.nansum(weights * analyzed["Modified duration"])),
        "Convexity": float(np.nansum(weights * analyzed["Convexity"])),
        "DV01": analyzed["DV01"].sum(),
        "OAS": float(np.nansum(weights * analyzed["OAS (bps)"].fillna(0))),
        "WAM": float(np.nansum(weights * analyzed["Years to maturity"])),
    }


def scenario_result(analyzed, rate_shock_bps, spread_shock_bps):
    """Estimate security-level P&L using duration, spread duration and convexity."""
    shocked = analyzed.copy()
    rate_change = rate_shock_bps / 10_000
    shocked["Rates P&L"] = -shocked["DV01"] * rate_shock_bps
    shocked["Convexity P&L"] = 0.5 * shocked["Convexity"] * shocked["Market value"] * rate_change**2
    shocked["Spread P&L"] = -shocked["Spread DV01"] * spread_shock_bps
    shocked["Scenario P&L"] = shocked[["Rates P&L", "Convexity P&L", "Spread P&L"]].sum(axis=1)
    shocked["Scenario return"] = shocked["Scenario P&L"] / shocked["Market value"]
    return shocked


def _allocation_chart(analyzed, column, title):
    grouped = analyzed.groupby(column, observed=True)["Market value"].sum().reset_index()
    grouped["Weight"] = grouped["Market value"] / grouped["Market value"].sum()
    figure = px.bar(grouped, x=column, y="Weight", text_auto=".1%", title=title)
    figure.update_yaxes(tickformat=".0%")
    return figure


def render_fixed_income_workbench():
    st.title("Fixed income risk workbench")
    st.caption(
        "Price a bond portfolio from contractual cash flows, measure rates and spread risk, and test macro-style scenarios."
    )
    with st.expander("📘 Workflow and methodology", expanded=False):
        st.markdown(
            """
            1. Use the demonstration portfolio or upload a CSV and review the editable holdings.
            2. Select an analysis date, then run the workbench to calculate yield, duration, convexity and DV01.
            3. Review concentration by rating, sector and maturity before applying rates and credit-spread shocks.
            4. Export the enriched security-level analytics for an investment memo or further analysis.

            **Required CSV columns:** `Bond ID`, `Issuer`, `Sector`, `Rating`, `Coupon (%)`, `Maturity`,
            `Frequency`, `Face value`, `Clean price`. Optional columns: `OAS (bps)` and `Spread duration`.
            """
        )
        st.caption(
            "Methodology assumes regular fixed-rate bullet bonds and ACT/365.25 timing. It does not model calls, puts, defaults, "
            "prepayments, embedded options or full OAS paths."
        )

    as_of = st.sidebar.date_input("Fixed-income analysis date", date.today(), key="fi_as_of")
    uploaded = st.file_uploader("Upload bond holdings CSV", type="csv", key="fi_upload")
    source = sample_bond_portfolio(as_of)
    if uploaded is not None:
        try:
            source = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the bond CSV: {exc}")
            return
    if "OAS (bps)" not in source:
        source["OAS (bps)"] = 0.0
    if "Spread duration" not in source:
        source["Spread duration"] = 0.0

    st.subheader("Bond holdings")
    holdings = st.data_editor(source, num_rows="dynamic", hide_index=True, use_container_width=True, key="fi_holdings")
    run = st.button("Run fixed income workbench", type="primary")
    if not run:
        st.info("Review the demonstration holdings or upload your own CSV, then select **Run fixed income workbench**.")
        return

    try:
        analyzed = analyze_portfolio(holdings, as_of)
    except Exception as exc:
        st.error(str(exc))
        return
    summary = portfolio_summary(analyzed)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Market value", f"${summary['Market value']:,.0f}")
    m2.metric("Portfolio YTM", f"{summary['YTM']:.2%}")
    m3.metric("Modified duration", f"{summary['Duration']:.2f}")
    m4.metric("DV01", f"${summary['DV01']:,.0f}")
    m5.metric("Weighted OAS", f"{summary['OAS']:,.0f} bps")
    m6.metric("Avg. maturity", f"{summary['WAM']:.1f} years")
    st.caption(
        f"Analysis date: {pd.Timestamp(as_of).date()} · {len(analyzed)} securities · "
        "YTM is solved from clean price plus estimated accrued interest."
    )

    overview_tab, security_tab, risk_tab, scenario_tab = st.tabs(
        ["Portfolio overview", "Security analytics", "Risk decomposition", "Scenario lab"]
    )
    with overview_tab:
        left, middle, right = st.columns(3)
        with left:
            st.plotly_chart(_allocation_chart(analyzed, "Sector", "Allocation by sector"), use_container_width=True)
        with middle:
            st.plotly_chart(_allocation_chart(analyzed, "Rating", "Allocation by rating"), use_container_width=True)
        with right:
            st.plotly_chart(_allocation_chart(analyzed, "Maturity bucket", "Allocation by maturity"), use_container_width=True)
        income = (analyzed["Face value"] * analyzed["Coupon (%)"] / 100).sum()
        o1, o2, o3 = st.columns(3)
        o1.metric("Annual coupon income", f"${income:,.0f}")
        o2.metric("Weighted convexity", f"{summary['Convexity']:.1f}")
        o3.metric("Largest issuer", analyzed.groupby("Issuer")["Weight"].sum().idxmax())

    with security_tab:
        display_columns = [
            "Bond ID", "Issuer", "Rating", "Maturity", "Face value", "Clean price", "Dirty price", "YTM",
            "Current yield", "Modified duration", "Convexity", "Market value", "Weight", "DV01", "OAS (bps)",
            "Spread duration", "Spread DV01",
        ]
        table = analyzed[display_columns]
        st.dataframe(
            table.style.format(
                {
                    "Face value": "${:,.0f}", "Clean price": "{:,.2f}", "Dirty price": "{:,.2f}",
                    "YTM": "{:.2%}", "Current yield": "{:.2%}", "Modified duration": "{:.2f}",
                    "Convexity": "{:.1f}", "Market value": "${:,.0f}", "Weight": "{:.1%}",
                    "DV01": "${:,.0f}", "OAS (bps)": "{:,.0f}", "Spread duration": "{:.2f}",
                    "Spread DV01": "${:,.0f}",
                }
            ),
            use_container_width=True,
        )
        st.download_button(
            "Download fixed-income analytics CSV", analyzed.to_csv(index=False), "fixed_income_analytics.csv", "text/csv"
        )

    with risk_tab:
        risk = analyzed[["Bond ID", "DV01", "Spread DV01", "Market value"]].copy()
        risk_long = risk.melt(
            id_vars=["Bond ID", "Market value"], value_vars=["DV01", "Spread DV01"],
            var_name="Risk measure", value_name="Dollar risk per bp",
        )
        st.plotly_chart(
            px.bar(
                risk_long, x="Bond ID", y="Dollar risk per bp", color="Risk measure", barmode="group",
                title="Interest-rate and spread risk by security",
            ),
            use_container_width=True,
        )
        contribution = analyzed[["Bond ID", "DV01"]].copy()
        contribution["DV01 contribution"] = contribution["DV01"] / contribution["DV01"].sum()
        contribution = contribution.sort_values("DV01 contribution", ascending=False)
        figure = px.bar(contribution, x="Bond ID", y="DV01 contribution", title="Contribution to portfolio DV01")
        figure.update_yaxes(tickformat=".0%")
        st.plotly_chart(figure, use_container_width=True)
        st.info(
            "DV01 estimates the dollar change from a 1 bp parallel yield move. Spread DV01 isolates an approximate 1 bp "
            "credit-spread move using the supplied spread duration."
        )

    with scenario_tab:
        presets = {
            "Custom": (0, 0),
            "Rates rally": (-100, -15),
            "Soft landing": (-50, -25),
            "Inflation shock": (100, 35),
            "Credit sell-off": (25, 150),
            "Recession / flight to quality": (-125, 200),
        }
        preset = st.selectbox("Scenario", list(presets), key="fi_scenario")
        default_rate, default_spread = presets[preset]
        s1, s2 = st.columns(2)
        rate_shock = s1.number_input(
            "Parallel Treasury yield shock (bps)", -500, 500, int(default_rate), 25, key=f"fi_rate_{preset}"
        )
        spread_shock = s2.number_input(
            "Credit spread shock (bps)", -500, 1000, int(default_spread), 25, key=f"fi_spread_{preset}"
        )
        shocked = scenario_result(analyzed, rate_shock, spread_shock)
        scenario_pnl = shocked["Scenario P&L"].sum()
        scenario_return = scenario_pnl / summary["Market value"]
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Scenario P&L", f"${scenario_pnl:+,.0f}")
        q2.metric("Scenario return", f"{scenario_return:+.2%}")
        q3.metric("Rates contribution", f"${shocked['Rates P&L'].sum():+,.0f}")
        q4.metric("Spread contribution", f"${shocked['Spread P&L'].sum():+,.0f}")

        pnl_long = shocked.melt(
            id_vars="Bond ID", value_vars=["Rates P&L", "Convexity P&L", "Spread P&L"],
            var_name="Driver", value_name="P&L",
        )
        st.plotly_chart(
            px.bar(pnl_long, x="Bond ID", y="P&L", color="Driver", title="Scenario P&L by security and driver"),
            use_container_width=True,
        )

        rate_grid = [-100, -50, 0, 50, 100]
        spread_grid = [-50, 0, 50, 100, 200]
        matrix = pd.DataFrame(index=[f"{x:+} bps" for x in rate_grid], columns=[f"{x:+} bps" for x in spread_grid])
        for rate_value in rate_grid:
            for spread_value in spread_grid:
                result = scenario_result(analyzed, rate_value, spread_value)["Scenario P&L"].sum()
                matrix.loc[f"{rate_value:+} bps", f"{spread_value:+} bps"] = result / summary["Market value"]
        heatmap = go.Figure(
            data=go.Heatmap(
                z=matrix.astype(float).to_numpy(), x=matrix.columns, y=matrix.index,
                colorscale="RdYlGn", zmid=0, text=np.vectorize(lambda value: f"{value:+.1%}")(matrix.astype(float)),
                texttemplate="%{text}", colorbar_title="Return",
            )
        )
        heatmap.update_layout(
            title="Rate / spread scenario return matrix", xaxis_title="Credit spread shock", yaxis_title="Treasury yield shock"
        )
        st.plotly_chart(heatmap, use_container_width=True)
        st.caption(
            "Scenario estimates use duration, spread duration and convexity approximations. They are most reliable for "
            "moderate shocks and option-free bonds; they are not forecasts."
        )
