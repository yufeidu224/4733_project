"""
ML Statistical Arbitrage — Performance Dashboard
Krauss et al. (2017) Replication + Extension

Launch:
    streamlit run dashboard.py
    python main.py --dashboard
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yaml

# ── Paths ─────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
CFG_PATH = ROOT / "config.yaml"

st.set_page_config(
    page_title="ML Stat-Arb Dashboard",
    page_icon="📈",
    layout="wide",
)


# ── Config ────────────────────────────────────────────────────
@st.cache_data
def load_config():
    if CFG_PATH.exists():
        with open(CFG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {"backtest": {"k_values": [10, 50, 100, 150, 200], "tc_bps": 5},
            "dashboard": {"default_k": 10, "show_net": True}}


# ── Data loaders ─────────────────────────────────────────────
@st.cache_data
def load_portfolio_returns():
    p = DATA_DIR / "ensemble_portfolio_returns.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    return df


@st.cache_data
def load_performance(sub: str):
    """sub: 'paper' or 'extended'"""
    p = DATA_DIR / f"ensemble_performance_{sub}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data
def load_factor_regression():
    p = DATA_DIR / "factor_regression_results.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data
def load_factor_summary():
    p = DATA_DIR / "factor_regression_summary.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data
def load_gbt_performance():
    p = DATA_DIR / "gbt_performance.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["test_start", "test_end"])


# ── Helpers ──────────────────────────────────────────────────
TDPY = 250

MODEL_COLORS = {
    "ENS1": "#1f77b4",
    "ENS2": "#ff7f0e",
    "ENS3": "#2ca02c",
    "DNN":  "#d62728",
    "GBT":  "#9467bd",
    "RAF":  "#8c564b",
}

def apply_tc(raw_series: pd.Series, tc_bps: int) -> pd.Series:
    """
    Daily TC = 4 half-turns × tc_bps bps
    (enter long, exit long, enter short, exit short)
    """
    daily_tc = 4 * tc_bps / 10_000
    return raw_series - daily_tc


def cumulative_return(rets: pd.Series) -> pd.Series:
    return (1 + rets).cumprod() - 1


def compute_metrics(rets: pd.Series) -> dict:
    n         = len(rets)
    if n == 0:
        return {}
    day_ret   = rets.mean()
    total_ret = (1 + rets).prod() - 1
    ann_ret   = (1 + total_ret) ** (TDPY / n) - 1
    ann_vol   = rets.std() * np.sqrt(TDPY)
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum       = (1 + rets).cumprod()
    max_dd    = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    return {
        "Day Ret (%)":   round(day_ret * 100,  4),
        "Ann. Ret (%)":  round(ann_ret * 100,  2),
        "Ann. Vol (%)":  round(ann_vol * 100,  2),
        "Sharpe":        round(sharpe,          3),
        "Max DD (%)":    round(max_dd * 100,    2),
        "Calmar":        round(calmar,           3),
        "Total Ret (%)": round(total_ret * 100, 1),
        "N days":        n,
    }


# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════
cfg = load_config()
k_options  = cfg["backtest"]["k_values"]
default_k  = cfg["dashboard"].get("default_k", 10)
tc_bps_cfg = cfg["backtest"]["tc_bps"]

with st.sidebar:
    st.title("⚙️ Settings")
    k_sel    = st.selectbox("Portfolio size k", k_options,
                            index=k_options.index(default_k))
    show_net = st.toggle("Show TC-adjusted (net) returns",
                         value=cfg["dashboard"].get("show_net", True))
    tc_bps   = st.number_input("TC (bps per half-turn)", value=tc_bps_cfg,
                                min_value=0, max_value=50, step=1)

    st.markdown("---")
    st.markdown("""
**Pipeline**
```
data → features → periods
  → DNN → GBT → RF
  → Ensemble → Factor Reg
```
**Paper:** Krauss et al. (2017)
*Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500*
""")


# ══════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════
port_df   = load_portfolio_returns()
perf_pap  = load_performance("paper")
perf_ext  = load_performance("extended")
fac_res   = load_factor_regression()
fac_sum   = load_factor_summary()
gbt_perf  = load_gbt_performance()

if port_df is None:
    st.error("ensemble_portfolio_returns.csv not found — run `python main.py --steps ensemble` first.")
    st.stop()


# ══════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Cumulative Returns",
    "🔢 Performance Tables",
    "🔬 Factor Regression",
    "⚠️ Risk Diagnostics",
])


# ─────────────────────────────────────────────────────────────
# Tab 1 — Overview
# ─────────────────────────────────────────────────────────────
with tab1:
    st.header("ML Statistical Arbitrage — Overview")

    paper_end      = cfg["periods"]["paper_end"]
    extended_start = cfg["periods"]["extended_start"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Paper Period (≤ {paper_end})")
        if perf_pap is not None:
            row = perf_pap[(perf_pap["model"] == "ENS1") & (perf_pap["k"] == k_sel)]
            if not row.empty:
                r = row.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Sharpe (ENS1)", f"{r['sharpe']:.3f}")
                c2.metric("Ann. Ret",      f"{r['ann_ret']*100:.1f}%")
                c3.metric("Max DD",        f"{r['max_drawdown']*100:.1f}%")
    with col2:
        st.subheader(f"Extended Period (≥ {extended_start})")
        if perf_ext is not None:
            row = perf_ext[(perf_ext["model"] == "ENS1") & (perf_ext["k"] == k_sel)]
            if not row.empty:
                r = row.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Sharpe (ENS1)", f"{r['sharpe']:.3f}")
                c2.metric("Ann. Ret",      f"{r['ann_ret']*100:.1f}%")
                c3.metric("Max DD",        f"{r['max_drawdown']*100:.1f}%")

    st.divider()

    # Quick comparison bar chart: Sharpe by model for selected k
    if perf_pap is not None and perf_ext is not None:
        st.subheader(f"Sharpe Ratio Comparison  (k={k_sel})")
        models = ["ENS1", "ENS2", "ENS3", "DNN", "GBT", "RAF"]

        def sharpe_row(df, model, k):
            r = df[(df["model"] == model) & (df["k"] == k)]
            return r["sharpe"].iloc[0] if not r.empty else None

        comp = []
        for m in models:
            sp = sharpe_row(perf_pap, m, k_sel)
            se = sharpe_row(perf_ext, m, k_sel)
            if sp is not None or se is not None:
                comp.append({"Model": m, "Paper Period": sp, "Extended Period": se})
        comp_df = pd.DataFrame(comp)

        fig = go.Figure()
        for period_col, color in [("Paper Period", "#1f77b4"), ("Extended Period", "#ff7f0e")]:
            fig.add_bar(
                name=period_col,
                x=comp_df["Model"],
                y=comp_df[period_col],
                marker_color=color,
            )
        fig.update_layout(
            barmode="group",
            yaxis_title="Sharpe Ratio",
            height=380,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Paper benchmarks (before TC, k=10): "
            "DNN ~2.44 | GBT ~3.16 | RAF ~3.09 | ENS1 ~3.40 | ENS2 ~3.43 | ENS3 ~3.46"
        )


# ─────────────────────────────────────────────────────────────
# Tab 2 — Cumulative Returns
# ─────────────────────────────────────────────────────────────
with tab2:
    st.header(f"Cumulative Returns  (k={k_sel})")

    paper_end_ts      = pd.Timestamp(paper_end)
    extended_start_ts = pd.Timestamp(extended_start)

    models_sel = st.multiselect(
        "Models", ["ENS1", "ENS2", "ENS3", "DNN", "GBT", "RAF"],
        default=["ENS1", "ENS2", "ENS3"],
    )

    sub = port_df[port_df["k"] == k_sel].copy()
    sub = sub.sort_values("date")

    fig = go.Figure()
    for model in models_sel:
        mdf  = sub[sub["ensemble"] == model].set_index("date")["portfolio_ret"]
        rets = apply_tc(mdf, tc_bps) if show_net else mdf
        cum  = cumulative_return(rets) * 100  # percent
        fig.add_scatter(
            x=cum.index, y=cum.values,
            name=model,
            line=dict(color=MODEL_COLORS.get(model)),
        )

    # Shade paper vs extended period
    if not sub.empty:
        x_min = sub["date"].min()
        x_max = sub["date"].max()
        fig.add_vrect(
            x0=x_min, x1=min(paper_end_ts, x_max),
            fillcolor="lightblue", opacity=0.08, line_width=0,
            annotation_text="Paper", annotation_position="top left",
        )
        if x_max > extended_start_ts:
            fig.add_vrect(
                x0=extended_start_ts, x1=x_max,
                fillcolor="lightyellow", opacity=0.12, line_width=0,
                annotation_text="Extended", annotation_position="top right",
            )

    fig.update_layout(
        yaxis_title="Cumulative Return (%)",
        height=480,
        legend=dict(orientation="h", y=1.05),
        hovermode="x unified",
    )
    label = "Net (after TC)" if show_net else "Raw (before TC)"
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Showing **{label}** returns.  TC = {tc_bps} bps/half-turn → {4*tc_bps} bps/day total.")

    # Rolling 1-year Sharpe
    st.subheader("Rolling 1-Year Sharpe Ratio")
    fig2 = go.Figure()
    for model in models_sel:
        mdf  = sub[sub["ensemble"] == model].set_index("date")["portfolio_ret"]
        rets = apply_tc(mdf, tc_bps) if show_net else mdf
        roll = rets.rolling(TDPY).apply(
            lambda x: (x.mean() / x.std() * np.sqrt(TDPY)) if x.std() > 0 else 0
        )
        fig2.add_scatter(
            x=roll.index, y=roll.values,
            name=model,
            line=dict(color=MODEL_COLORS.get(model)),
        )
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(yaxis_title="Sharpe (1-yr rolling)", height=360,
                       legend=dict(orientation="h", y=1.05), hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Tab 3 — Performance Tables
# ─────────────────────────────────────────────────────────────
with tab3:
    st.header("Performance Tables")

    def format_perf(df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()
        out = df.copy()
        for col in ["day_ret", "ann_ret", "ann_vol", "max_drawdown", "total_ret"]:
            if col in out.columns:
                out[col] = (out[col] * 100).round(3)
        if "sharpe" in out.columns:
            out["sharpe"] = out["sharpe"].round(3)
        out = out.rename(columns={
            "model": "Model", "k": "k", "n_days": "N days",
            "day_ret": "Day Ret %", "ann_ret": "Ann Ret %",
            "ann_vol": "Ann Vol %", "sharpe": "Sharpe",
            "max_drawdown": "Max DD %", "total_ret": "Total Ret %",
        })
        return out.drop(columns=["sub_period"], errors="ignore")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Paper Period (≤ {paper_end})")
        if perf_pap is not None:
            st.dataframe(format_perf(perf_pap), use_container_width=True, hide_index=True)
        else:
            st.info("ensemble_performance_paper.csv not found.")
    with col2:
        st.subheader(f"Extended Period (≥ {extended_start})")
        if perf_ext is not None:
            st.dataframe(format_perf(perf_ext), use_container_width=True, hide_index=True)
        else:
            st.info("ensemble_performance_extended.csv not found.")

    # Per-period Sharpe heatmap (GBT)
    if gbt_perf is not None:
        st.subheader("GBT Walk-Forward Sharpe by Period")
        fig_heat = px.bar(
            gbt_perf, x="period", y="sharpe_ratio",
            color="sharpe_ratio",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            labels={"sharpe_ratio": "Sharpe", "period": "Study Period"},
            height=320,
        )
        fig_heat.update_layout(coloraxis_showscale=True)
        st.plotly_chart(fig_heat, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Tab 4 — Factor Regression
# ─────────────────────────────────────────────────────────────
with tab4:
    st.header("Factor Regression  (Table 4 Replication)")
    st.caption("ENS1 k=10 after-TC returns regressed on Fama-French factors.")

    if fac_res is None:
        st.info("factor_regression_results.csv not found — run `python main.py --steps factors`.")
    else:
        # Alpha row highlight
        alpha_df = fac_res[fac_res["variable"].str.lower() == "const"].copy() if "variable" in fac_res.columns else fac_res.head(4)
        st.subheader("Daily Alpha (intercept) by Specification")
        st.dataframe(alpha_df, use_container_width=True, hide_index=True)

        st.subheader("Full Coefficient Table")
        st.dataframe(fac_res, use_container_width=True, hide_index=True)

    if fac_sum is not None:
        st.subheader("Model Fit Summary")
        st.dataframe(fac_sum, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# Tab 5 — Risk Diagnostics
# ─────────────────────────────────────────────────────────────
with tab5:
    st.header("Risk Diagnostics")

    models_risk = st.multiselect(
        "Models", ["ENS1", "ENS2", "ENS3", "DNN", "GBT", "RAF"],
        default=["ENS1", "GBT"],
        key="risk_models",
    )
    sub = port_df[port_df["k"] == k_sel].copy().sort_values("date")

    # Drawdown chart
    st.subheader("Drawdown")
    fig_dd = go.Figure()
    for model in models_risk:
        mdf  = sub[sub["ensemble"] == model].set_index("date")["portfolio_ret"]
        rets = apply_tc(mdf, tc_bps) if show_net else mdf
        cum  = (1 + rets).cumprod()
        dd   = (cum - cum.cummax()) / cum.cummax() * 100
        fig_dd.add_scatter(x=dd.index, y=dd.values, name=model,
                           line=dict(color=MODEL_COLORS.get(model)),
                           fill="tozeroy", fillcolor=MODEL_COLORS.get(model, "#888") + "22")
    fig_dd.update_layout(yaxis_title="Drawdown (%)", height=360,
                         legend=dict(orientation="h", y=1.05), hovermode="x unified")
    st.plotly_chart(fig_dd, use_container_width=True)

    # Return distribution
    st.subheader("Daily Return Distribution")
    fig_hist = go.Figure()
    for model in models_risk:
        mdf  = sub[sub["ensemble"] == model].set_index("date")["portfolio_ret"]
        rets = apply_tc(mdf, tc_bps) if show_net else mdf
        fig_hist.add_histogram(x=rets.values * 100, name=model, opacity=0.6,
                               marker_color=MODEL_COLORS.get(model), nbinsx=80)
    fig_hist.update_layout(barmode="overlay", xaxis_title="Daily Return (%)",
                           yaxis_title="Count", height=340,
                           legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig_hist, use_container_width=True)

    # Risk metrics table
    st.subheader("Risk Metrics Summary")
    risk_rows = []
    for model in models_risk:
        mdf  = sub[sub["ensemble"] == model].set_index("date")["portfolio_ret"]
        rets = apply_tc(mdf, tc_bps) if show_net else mdf

        # Split by period
        for period_name, mask in [
            (f"Paper (≤{paper_end})",        rets.index <= paper_end_ts),
            (f"Extended (≥{extended_start})", rets.index >= extended_start_ts),
        ]:
            s = rets[mask]
            if s.empty:
                continue
            m = compute_metrics(s)
            m["Model"]  = model
            m["Period"] = period_name
            risk_rows.append(m)

    if risk_rows:
        risk_df = pd.DataFrame(risk_rows)[
            ["Model", "Period", "Day Ret (%)", "Ann. Ret (%)",
             "Ann. Vol (%)", "Sharpe", "Max DD (%)", "Calmar", "N days"]
        ]
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
