"""Blended valuation dashboard across intrinsic, dividend and relative models."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


APPLICABILITY = {
    "DCF": "Operating companies with forecastable free cash flow.",
    "Two-stage DDM": "Dividend payer transitioning from growth toward maturity.",
    "P/E": "Profitable company with positive, reasonably stable earnings.",
    "P/B": "Bank, insurer or asset-intensive company with meaningful book value.",
    "P/S": "Company with meaningful sales but negative or volatile earnings.",
    "EV/EBITDA": "Non-financial operating company with positive EBITDA.",
}


def render_valuation_summary(company_data, dcf_value, two_stage_ddm):
    st.title("Blended valuation summary")
    st.caption("Combine intrinsic, dividend and relative valuation methods while explicitly controlling model weights and applicability.")
    ticker = st.sidebar.text_input("Ticker", "AAPL", key="summary_ticker").upper().strip()
    if st.sidebar.button("Refresh latest market data", key="summary_refresh"):
        company_data.clear()
        st.rerun()
    try:
        with st.spinner("Building the valuation summary..."):
            data = company_data(ticker)
    except Exception as exc:
        st.error(f"Unable to retrieve company data: {exc}")
        return

    price, shares, debt, cash = data["price"], data["shares"], data["debt"], data["cash"]
    history = data["history"]
    if history.empty or history.iloc[-1]["Revenue"] <= 0:
        st.error("Positive historical revenue is required for the blended valuation.")
        return
    revenue = float(history.iloc[-1]["Revenue"])
    ebit_margin_default = float(history.iloc[-1]["EBIT"] / revenue)
    da_margin = abs(float(history.iloc[-1]["D&A"] / revenue))
    capex_margin = abs(float(history.iloc[-1]["CapEx"] / revenue))
    nwc_margin = abs(float(history.iloc[-1]["Change in WC"] / revenue))

    st.subheader(f"{data['company_name']} ({ticker})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest market price", f"${price:,.2f}")
    m2.metric("Sector", data["sector"])
    m3.metric("Quote session", data["price_as_of"])
    m4.metric("Models available", "6")

    with st.expander("Intrinsic-value assumptions", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        growth = c1.number_input("DCF revenue growth (%)", -30., 60., 8., .5) / 100
        margin = c2.number_input("DCF EBIT margin (%)", -30., 80., float(ebit_margin_default * 100), .5) / 100
        wacc = c3.number_input("DCF WACC (%)", .5, 30., 9.5, .1) / 100
        terminal_growth_max = float(wacc * 100 - .1)
        terminal_growth = c4.number_input("DCF terminal growth (%)", -3., terminal_growth_max, min(2.5, terminal_growth_max), .1) / 100
        dcf_years = st.slider("DCF forecast years", 3, 10, 5)
        assumptions = {"years": dcf_years, "tax": .21, "da": da_margin, "capex": capex_margin, "nwc": nwc_margin}
        capital = {"shares": shares, "debt": debt, "cash": cash}
        dcf_fair, _, _, _ = dcf_value(revenue, assumptions, capital, (growth, margin, wacc, terminal_growth))

    with st.expander("Dividend and market-multiple assumptions", expanded=False):
        c1, c2, c3 = st.columns(3)
        ddm_required = c1.number_input("DDM required return (%)", .5, 40., 9.5, .1) / 100
        ddm_high_growth = c2.number_input("DDM high growth (%)", -30., 60., 8., .5) / 100
        ddm_stable_max = float(ddm_required * 100 - .1)
        ddm_stable = c3.number_input("DDM stable growth (%)", -3., ddm_stable_max, min(2.5, ddm_stable_max), .1) / 100
        p1, p2, p3, p4 = st.columns(4)
        target_pe = p1.number_input("Target P/E", .1, 100., 20., .5)
        target_pb = p2.number_input("Target P/B", .1, 30., 3., .1)
        target_ps = p3.number_input("Target P/S", .1, 50., 4., .1)
        target_ev_ebitda = p4.number_input("Target EV/EBITDA", .1, 50., 12., .5)

    if data["annual_dividend"] > 0:
        ddm_fair, _, _ = two_stage_ddm(data["annual_dividend"], ddm_high_growth, 5, ddm_stable, ddm_required)
    else:
        ddm_fair = np.nan
    pe_fair = data["eps"] * target_pe if data["eps"] > 0 else np.nan
    pb_fair = data["book_value_per_share"] * target_pb if data["book_value_per_share"] > 0 else np.nan
    ps_fair = data["revenue_per_share"] * target_ps if data["revenue_per_share"] > 0 else np.nan
    ev_fair = ((data["ebitda"] * target_ev_ebitda) - debt + cash) / shares if data["ebitda"] > 0 and shares > 0 else np.nan

    defaults = {"DCF": .35, "Two-stage DDM": .10, "P/E": .20, "P/B": .10, "P/S": .10, "EV/EBITDA": .15}
    values = {"DCF": dcf_fair, "Two-stage DDM": ddm_fair, "P/E": pe_fair, "P/B": pb_fair, "P/S": ps_fair, "EV/EBITDA": ev_fair}
    table = pd.DataFrame({
        "Model": list(values), "Implied value / share": list(values.values()),
        "Weight": [defaults[name] if np.isfinite(values[name]) and values[name] > 0 else 0 for name in values],
        "Applicable company type": [APPLICABILITY[name] for name in values],
    })
    edited = st.data_editor(table, hide_index=True, use_container_width=True, disabled=["Model", "Applicable company type"],
                            column_config={"Weight": st.column_config.NumberColumn("Weight (0–1)", min_value=0., max_value=1., step=.05, format="%.2f")})
    valid = edited[np.isfinite(edited["Implied value / share"]) & (edited["Implied value / share"] > 0) & (edited["Weight"] > 0)].copy()
    if valid.empty or valid["Weight"].sum() <= 0:
        st.error("Assign a positive weight to at least one model with a valid implied value.")
        return
    valid["Normalized weight"] = valid["Weight"] / valid["Weight"].sum()
    valid["Weighted value"] = valid["Implied value / share"] * valid["Normalized weight"]
    blended = valid["Weighted value"].sum()
    upside = blended / price - 1
    dispersion = valid["Implied value / share"].std(ddof=0) / valid["Implied value / share"].mean() if len(valid) > 1 else 0
    confidence = "High" if dispersion < .15 else "Medium" if dispersion < .35 else "Low"

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Blended fair value", f"${blended:,.2f}", f"{upside:+.1%} vs. market")
    r2.metric("Active models", f"{len(valid)}")
    r3.metric("Model dispersion", f"{dispersion:.1%}")
    r4.metric("Valuation confidence", confidence)
    chart = valid[["Model", "Implied value / share"]].copy()
    chart.loc[len(chart)] = ["Current market price", price]
    fig = px.bar(chart, x="Model", y="Implied value / share", color="Model", title="Valuation range by method")
    fig.add_hline(y=blended, line_dash="dash", annotation_text=f"Blended ${blended:,.2f}")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(valid.style.format({"Implied value / share": "${:,.2f}", "Weight": "{:.0%}", "Normalized weight": "{:.0%}", "Weighted value": "${:,.2f}"}), use_container_width=True)
    st.download_button("Download valuation summary", valid.to_csv(index=False), f"{ticker}_valuation_summary.csv", "text/csv")
    st.caption("Invalid models are automatically assigned zero weight. Model dispersion is used as a simple confidence indicator; it is not a probability estimate.")
