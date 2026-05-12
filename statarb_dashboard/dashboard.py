"""
ML Statistical Arbitrage Dashboard
Krauss et al. (2017) Replication — IEOR 4733
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML StatArb Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f8f9fc; color: #1a1f36; }
  [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8ecf4; }
  [data-testid="stSidebar"] * { color: #1a1f36 !important; }
  .metric-card {
    background: #ffffff; border-radius: 10px; padding: 18px 20px;
    border: 1px solid #e8ecf4; border-top: 3px solid #3b5bdb;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .metric-label { font-size: 11px; color: #6b7280; letter-spacing: 1px; text-transform: uppercase; font-weight: 600; }
  .metric-value { font-size: 26px; font-weight: 700; color: #1a1f36; margin: 6px 0 2px; }
  .metric-sub   { font-size: 11px; color: #9ca3af; }
  .section-header {
    font-size: 11px; font-weight: 700; color: #6b7280;
    letter-spacing: 2px; text-transform: uppercase;
    border-bottom: 1px solid #e8ecf4; padding-bottom: 8px; margin-bottom: 14px;
  }
  .tag-green { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700; }
  .tag-red   { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700; }
  .tag-blue  { background:#dbeafe; color:#1e40af; padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700; }
  .stSelectbox label, .stSlider label, .stRadio label { color: #6b7280 !important; }
  h1, h2, h3 { color: #1a1f36 !important; }
  p, li { color: #374151; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
DATA = "."

@st.cache_data
def load_data():
    bt_rf   = pd.read_csv(f"{DATA}/backtest_results_k10.csv",        parse_dates=["date"]).set_index("date")
    bt_all  = pd.read_csv(f"{DATA}/backtest_results_all_k.csv",      parse_dates=["date"]).set_index("date")
    bt_dnn  = pd.read_csv(f"{DATA}/dnn_backtest_results_k10.csv",    parse_dates=["date"]).set_index("date")
    t2      = pd.read_csv(f"{DATA}/table2_results.csv")
    t3      = pd.read_csv(f"{DATA}/table3_results.csv")
    ens_p   = pd.read_csv(f"{DATA}/ensemble_performance_paper.csv")
    ens_e   = pd.read_csv(f"{DATA}/ensemble_performance_extended.csv")
    ens_ret = pd.read_csv(f"{DATA}/ensemble_portfolio_returns.csv",   parse_dates=["date"])
    gbt_p   = pd.read_csv(f"{DATA}/gbt_performance.csv",             parse_dates=["test_start","test_end"])
    sp      = pd.read_csv(f"{DATA}/study_periods.csv",
                          parse_dates=["train_start","train_end","test_start","test_end"])
    dnn_pp  = pd.read_csv(f"{DATA}/dnn_perf_paper_period.csv")
    dnn_ep  = pd.read_csv(f"{DATA}/dnn_perf_post_paper.csv")
    return bt_rf, bt_all, bt_dnn, t2, t3, ens_p, ens_e, ens_ret, gbt_p, sp, dnn_pp, dnn_ep

bt_rf, bt_all, bt_dnn, t2, t3, ens_p, ens_e, ens_ret, gbt_p, sp, dnn_pp, dnn_ep = load_data()

# ── Helpers ────────────────────────────────────────────────────────────────────
COLORS = {
    "primary":  "#3b5bdb",
    "teal":     "#0ca678",
    "navy":     "#1a1f36",
    "blue":     "#228be6",
    "red":      "#e03131",
    "amber":    "#f08c00",
    "purple":   "#7048e8",
    "slate":    "#6b7280",
    "white":    "#ffffff",
    "bg":       "#f8f9fc",
    "border":   "#e8ecf4",
    "text":     "#1a1f36",
}

REGIME_DEFS = [
    ("Sub-period 1\n1993–2001",     "1992-12-17", "2001-03-31", "#0ca678"),
    ("Sub-period 2\n2001–2008",     "2001-04-01", "2008-08-31", "#f08c00"),
    ("Financial Crisis\n2008–2009", "2008-09-01", "2009-12-31", "#e03131"),
    ("Sub-period 4\n2010–2015",     "2010-01-01", "2015-10-15", "#7048e8"),
    ("Extension\n2016–2024",        "2015-10-16", "2024-09-24", "#228be6"),
]

def perf_stats(ret_series):
    r = ret_series.dropna()
    if len(r) == 0:
        return {}
    ann  = (1 + r.mean()) ** 252 - 1
    vol  = r.std() * np.sqrt(252)
    sh   = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    cum  = (1 + r).cumprod()
    mdd  = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"ann": ann, "vol": vol, "sharpe": sh, "mdd": mdd, "n": len(r)}

def make_cum_fig(series_dict, title="Cumulative Return", height=380):
    fig = go.Figure()
    palette = ["#3b5bdb", "#0ca678", "#f08c00", "#e03131", "#7048e8", "#228be6"]
    for i, (name, s) in enumerate(series_dict.items()):
        cum = (1 + s.dropna()).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, name=name,
            line=dict(color=palette[i % len(palette)], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}x<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#1a1f36", size=13)),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
        font=dict(color=COLORS["slate"], size=11),
        xaxis=dict(gridcolor="#e8ecf4", zeroline=False),
        yaxis=dict(gridcolor="#e8ecf4", zeroline=False, tickformat=".2f"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", font=dict(size=10)),
        height=height, margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 ML StatArb")
    st.markdown("<div style='color:#64748b;font-size:11px;margin-bottom:16px'>Krauss et al. (2017) · IEOR 4733</div>", unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "📄 Paper Summary",
        "📊 Reproduction Results",
        "🔄 Differences vs Original",
        "🛡️ Robustness Checks",
        "📈 Regime Sensitivity",
        "⚠️ Risk Diagnostics",
    ])

    st.markdown("---")
    st.markdown("<div class='section-header'>Controls</div>", unsafe_allow_html=True)
    sel_k  = st.selectbox("Portfolio size k", [10, 50, 100, 150, 200], index=0)
    sel_tc = st.slider("Transaction cost (bps/side)", 0, 20, 5, 1)
    tc_adj = sel_tc / 10000

    st.markdown("---")
    st.markdown("<div style='color:#64748b;font-size:10px'>No-lookahead: train ends before test starts<br>Seed=1 · Daily rebalancing · Equal-weight</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PAPER SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if page == "📄 Paper Summary":
    st.markdown("## Paper Summary")
    st.markdown("<div style='color:#64748b'>Krauss, Do & Huck (2017) — <i>Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500</i></div>", unsafe_allow_html=True)
    st.markdown("")

    # Research question + key stats
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub in [
        (c1, "Universe",      "S&P 500",     "1,288 constituents"),
        (c2, "Sample Period", "1995–2015",   "Krauss paper period"),
        (c3, "Our Extension", "1992–2024",   "32 Study Periods"),
        (c4, "Features",      "31 lags",     "R1–R20 + R40–R240"),
    ]:
        col.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value' style='font-size:20px'>{val}</div>
            <div class='metric-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("<div class='section-header'>Methodology</div>", unsafe_allow_html=True)
        st.markdown("""
**Research Question**  
Can ML models (DNN, GBT, RF) generate statistically significant long-short alpha on S&P 500 daily returns?

**Target Variable Y**  
Binary: does stock i outperform the cross-sectional median return on day t+1?

**Training Design**
- Rolling window: 750 trading days train → 250 trading days test
- 32 non-overlapping Study Periods (1990–2024)
- Strict temporal separation — no lookahead bias

**Portfolio Construction**
- Rank stocks by P̂(Y=1) each day
- Long top-k, short bottom-k (equal-weighted, dollar-neutral)
- k ∈ {10, 50, 100, 150, 200}
- TC = 0.05% per share per half-turn
        """)

    with col_r:
        st.markdown("<div class='section-header'>Models & Results</div>", unsafe_allow_html=True)
        models_df = pd.DataFrame({
            "Model": ["DNN", "GBT", "RF", "ENS1"],
            "Description": [
                "31-31-10-5-2 Maxout, Adadelta, L1=1e-5",
                "XGBoost: 100 trees, depth=3, lr=0.1",
                "1000 trees, depth=20, √p features",
                "Equal-weight DNN+GBT+RF",
            ],
            "Paper Sharpe": ["2.44", "3.16", "3.09", "3.40"],
            "Our Sharpe": [
                f"{perf_stats(bt_dnn['net_ret'])['sharpe']:.2f}",
                f"{gbt_p['sharpe_ratio'].mean():.2f} (avg)",
                f"{perf_stats(bt_rf['net_ret'])['sharpe']:.2f}",
                f"{ens_p[(ens_p['model']=='ENS1')&(ens_p['k']==10)]['sharpe'].values[0]:.2f}",
            ],
        })
        st.dataframe(models_df, use_container_width=True, hide_index=True)

        st.markdown("<div class='section-header' style='margin-top:16px'>Ensemble Equations</div>", unsafe_allow_html=True)
        st.latex(r"\hat{P}^{ENS1} = \frac{1}{M}\sum_i \hat{P}^i")
        st.latex(r"\hat{P}^{ENS2} = \sum_i w^i \hat{P}^i, \quad w^i = \frac{g^i}{\sum_j g^j}")
        st.latex(r"\hat{P}^{ENS3} = \sum_i w^i \hat{P}^i, \quad w^i = \frac{1/R^i}{\sum_j 1/R^j}")

    st.markdown("")
    st.markdown("<div class='section-header'>Study Period Timeline (32 Periods)</div>", unsafe_allow_html=True)

    fig_sp = go.Figure()
    colors_sp = [COLORS["teal"], COLORS["blue"]]
    for _, row in sp.iterrows():
        pid = int(row["period"])
        c = colors_sp[pid % 2]
        start_str = str(row["test_start"].date())
        end_str   = str(row["test_end"].date())
        fig_sp.add_trace(go.Scatter(
            x=[start_str, end_str, end_str, start_str, start_str],
            y=[pid - 0.4, pid - 0.4, pid + 0.4, pid + 0.4, pid - 0.4],
            fill="toself", fillcolor=c, opacity=0.7,
            line=dict(width=0), mode="lines",
            name=f"Period {pid}", showlegend=False,
            hovertemplate=f"Period {pid}<br>Test: {start_str} → {end_str}<extra></extra>",
        ))
    fig_sp.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
        xaxis=dict(type="date", gridcolor="#e8ecf4", tickformat="%Y"),
        yaxis=dict(gridcolor="#e8ecf4", title="Period", dtick=4),
        height=300, margin=dict(l=40, r=20, t=20, b=30),
        font=dict(color="#6b7280"),
        annotations=[dict(
            x="2015-10-15", y=34, text="Paper end →",
            showarrow=False, font=dict(color=COLORS["amber"], size=11),
        )],
        shapes=[dict(
            type="line", x0="2015-10-15", x1="2015-10-15", y0=0, y1=33,
            line=dict(color=COLORS["amber"], width=1.5, dash="dash"),
        )],
    )
    st.plotly_chart(fig_sp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — REPRODUCTION RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Reproduction Results":
    st.markdown("## Reproduction Results")
    st.markdown("<div style='color:#64748b'>Replication of Krauss et al. (2017) Tables 2 & 3 — RF model, k=10</div>", unsafe_allow_html=True)
    st.markdown("")

    # Summary badges
    c1, c2, c3 = st.columns(3)
    rf_stats = perf_stats(bt_rf["net_ret"])
    paper_sharpe = 1.90
    our_sharpe = rf_stats.get("sharpe", 0)
    for col, label, val, sub, ok in [
        (c1, "Reproduction Status", "Partial Match", "Main trends confirmed", False),
        (c2, "Our Sharpe (post-TC)", f"{our_sharpe:.2f}", f"Paper: {paper_sharpe:.2f}", our_sharpe > 0),
        (c3, "Data Coverage", "1992–2024", "32 Study Periods", True),
    ]:
        tag = "tag-green" if ok else "tag-red"
        col.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value' style='font-size:20px'>{val}</div>
            <div class='metric-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    tab1, tab2, tab3 = st.tabs(["Table 2 — Daily Stats", "Table 3 — Annualized", "Cumulative Returns"])

    with tab1:
        # ── Compute Table 2 from ensemble_portfolio_returns ──────────────────
        @st.cache_data
        def compute_table2_v2(data_dir):
            import numpy as np
            TC = 0.0005
            PAPER_END    = "2015-12-31"
            PAPER_START  = None
            EXTEND_START = "2016-01-01"

            bt_rf  = pd.read_csv(f"{data_dir}/backtest_results_k10.csv",     parse_dates=["date"]).set_index("date")
            bt_dnn = pd.read_csv(f"{data_dir}/dnn_backtest_results_k10.csv",  parse_dates=["date"]).set_index("date")
            bt_gbt = pd.read_csv(f"{data_dir}/gbt_backtest_k10.csv",          parse_dates=["date"]).set_index("date")
            ens    = pd.read_csv(f"{data_dir}/ensemble_portfolio_returns.csv", parse_dates=["date"])
            ens1   = ens[(ens["ensemble"]=="ENS1")&(ens["k"]==10)].set_index("date").sort_index()
            ens1   = ens1.rename(columns={"portfolio_ret":"raw_ret"})
            ens1["net_ret"] = ens1["raw_ret"] - 4*TC

            METRICS = ["Mean return (long)","Mean return (short)","Mean return",
                "Standard error (NW)","t-statistic (NW)",
                "Minimum","Quartile 1","Median","Quartile 3","Maximum",
                "Standard deviation","Skewness","Kurtosis",
                "Historical 1-percent VaR","Historical 1-percent CVaR",
                "Historical 5-percent VaR","Historical 5-percent CVaR",
                "Maximum drawdown","Calmar ratio","Share with return > 0"]

            def nw(ret, lags=1):
                r=ret.dropna().values; n,mu=len(r),r.mean()
                v=np.var(r,ddof=1)
                for lag in range(1,lags+1):
                    v+=2*(1-lag/(lags+1))*np.mean((r[lag:]-mu)*(r[:-lag]-mu))
                se=np.sqrt(v/n); return mu/se if se>0 else 0, se

            def stats(bt, s=None, e=None):
                d = bt.loc[s:e].copy() if (s or e) else bt.copy()
                out={}
                for col,lbl in [("raw_ret","pre"),("net_ret","post")]:
                    r=d[col].dropna(); t,se=nw(r)
                    cum=(1+r).cumprod(); mdd=((cum-cum.cummax())/cum.cummax()).min()
                    ann=(1+r.mean())**252-1
                    out[lbl]={
                        "Mean return (long)":        round(d["long_ret"].mean(),4),
                        "Mean return (short)":       round(d["short_ret"].mean(),4),
                        "Mean return":               round(r.mean(),4),
                        "Standard error (NW)":       round(se,4),
                        "t-statistic (NW)":          round(t,4),
                        "Minimum":                   round(r.min(),4),
                        "Quartile 1":                round(r.quantile(.25),4),
                        "Median":                    round(r.median(),4),
                        "Quartile 3":                round(r.quantile(.75),4),
                        "Maximum":                   round(r.max(),4),
                        "Standard deviation":        round(r.std(),4),
                        "Skewness":                  round(r.skew(),4),
                        "Kurtosis":                  round(r.kurtosis(),4),
                        "Historical 1-percent VaR":  round(r.quantile(.01),4),
                        "Historical 1-percent CVaR": round(r[r<=r.quantile(.01)].mean(),4),
                        "Historical 5-percent VaR":  round(r.quantile(.05),4),
                        "Historical 5-percent CVaR": round(r[r<=r.quantile(.05)].mean(),4),
                        "Maximum drawdown":          round(mdd,4),
                        "Calmar ratio":              round(ann/abs(mdd) if mdd!=0 else 0,4),
                        "Share with return > 0":     round((r>0).mean(),4),
                    }
                return out

            models = {"DNN":bt_dnn, "GBT":bt_gbt, "RAF":bt_rf, "ENS1":ens1}
            periods = {"paper":(None,PAPER_END), "extended":(EXTEND_START,None)}
            results={}
            for pname,(s,e) in periods.items():
                results[pname]={}
                for m,bt in models.items():
                    results[pname][m]=stats(bt,s,e)

            rows=[]
            for metric in METRICS:
                row={"metric":metric}
                for pname in ["paper","extended"]:
                    for m in ["DNN","GBT","RAF","ENS1"]:
                        for tc in ["pre","post"]:
                            row[f"{pname}_{m}_{tc}"]=results[pname][m][tc][metric]
                rows.append(row)
            return pd.DataFrame(rows), METRICS

        t2_full, METRICS = compute_table2_v2(DATA)

        sel_period = st.radio("Period", ["Paper Period (1995–2015)", "Extended Period (2016–2024)"],
                              horizontal=True)
        pkey = "paper" if "Paper" in sel_period else "extended"

        sel_tc_view = st.radio("Transaction costs", ["Before TC", "After TC"], horizontal=True)
        tc_key = "pre" if "Before" in sel_tc_view else "post"

        models_t2 = ["DNN", "GBT", "RAF", "ENS1"]
        disp_cols = {"metric": "Metric"}
        for m in models_t2:
            disp_cols[f"{pkey}_{m}_{tc_key}"] = m

        t2_disp = t2_full[list(disp_cols.keys())].rename(columns=disp_cols)
        st.dataframe(
            t2_disp.style.format({m: "{:.4f}" for m in models_t2}),
            use_container_width=True, hide_index=True, height=600,
        )

        # Bar chart — mean return, t-stat, sharpe across models
        st.markdown("<div class='section-header'>Mean Return & t-statistic by Model</div>", unsafe_allow_html=True)
        bar_metrics = ["Mean return", "t-statistic (NW)", "Standard deviation"]
        bar_df = t2_disp[t2_disp["Metric"].isin(bar_metrics)]
        colors_bar = [COLORS["primary"], COLORS["teal"], COLORS["amber"], COLORS["purple"]]
        fig_bar = go.Figure()
        for mi, m in enumerate(models_t2):
            fig_bar.add_trace(go.Bar(
                name=m, x=bar_df["Metric"], y=bar_df[m],
                marker_color=colors_bar[mi],
            ))
        fig_bar.update_layout(
            barmode="group", height=300,
            paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
            font=dict(color="#6b7280"),
            xaxis=dict(gridcolor="#e8ecf4"),
            yaxis=dict(gridcolor="#e8ecf4"),
            legend=dict(bgcolor="rgba(255,255,255,0.9)"),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-header'>All Models Performance — Paper Period (1992–2015)</div>", unsafe_allow_html=True)
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Paper Period**")
            ens_p_k = ens_p[["model","k","day_ret","ann_ret","sharpe","max_drawdown"]].copy()
            ens_p_k.columns = ["Model","k","Day Ret","Ann Ret","Sharpe","Max DD"]
            st.dataframe(ens_p_k.style.format({
                "Day Ret": "{:.4f}", "Ann Ret": "{:.4f}",
                "Sharpe": "{:.2f}", "Max DD": "{:.2%}",
            }), use_container_width=True, hide_index=True)
        with col_r:
            st.markdown("**Extension Period (2016–2024)**")
            ens_e_k = ens_e[["model","k","day_ret","ann_ret","sharpe","max_drawdown"]].copy()
            ens_e_k.columns = ["Model","k","Day Ret","Ann Ret","Sharpe","Max DD"]
            st.dataframe(ens_e_k.style.format({
                "Day Ret": "{:.4f}", "Ann Ret": "{:.4f}",
                "Sharpe": "{:.2f}", "Max DD": "{:.2%}",
            }), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("<div class='section-header'>Cumulative Returns — All Models k=10 (Post-TC)</div>", unsafe_allow_html=True)

        # Model comparison KPIs
        mc1, mc2, mc3 = st.columns(3)
        for col, label, ret_col, color in [
            (mc1, "RF",  bt_rf["net_ret"],  "#3b5bdb"),
            (mc2, "DNN", bt_dnn["net_ret"], "#0ca678"),
            (mc3, "ENS1 (k=10)", None,      "#f08c00"),
        ]:
            if ret_col is not None:
                p = perf_stats(ret_col)
                sh, ar = p["sharpe"], p["ann"]
            else:
                ens_row = ens_p[(ens_p["model"]=="ENS1") & (ens_p["k"]==10)]
                sh = ens_row["sharpe"].values[0] if len(ens_row) else 0
                ar = ens_row["ann_ret"].values[0] if len(ens_row) else 0
            col.markdown(f"""<div class='metric-card' style='border-top-color:{color}'>
                <div class='metric-label'>{label} — Paper Period</div>
                <div class='metric-value' style='font-size:22px'>{sh:.2f}</div>
                <div class='metric-sub'>Sharpe · {ar:.1%} ann ret</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        fig_cum = make_cum_fig({
            "RF Post-TC":   bt_rf["net_ret"],
            "DNN Post-TC":  bt_dnn["net_ret"],
            "RF Pre-TC":    bt_rf["raw_ret"],
        }, title="Cumulative Return — RF vs DNN (k=10)", height=400)
        # Add paper period vertical line
        fig_cum.add_shape(type="line", x0="2015-10-15", x1="2015-10-15", y0=0, y1=1,
                          xref="x", yref="paper",
                          line=dict(color="#f08c00", width=1.5, dash="dash"))
        fig_cum.add_annotation(x="2015-10-15", y=0.98, xref="x", yref="paper",
                               text="Paper end", showarrow=False,
                               font=dict(color="#f08c00", size=11), xanchor="left")
        st.plotly_chart(fig_cum, use_container_width=True)

        # k comparison
        st.markdown("<div class='section-header'>Impact of Portfolio Size k (RF Post-TC)</div>", unsafe_allow_html=True)
        k_series = {f"k={k}": bt_all[f"k{k}_net_ret"] for k in [10, 50, 100, 150, 200]}
        # adjust for custom TC
        if sel_tc != 5:
            delta_tc = (sel_tc - 5) / 10000 * 4
            k_series = {name: s - delta_tc for name, s in k_series.items()}
        st.plotly_chart(make_cum_fig(k_series, title=f"Cumulative Return by k (TC={sel_tc}bps/side)", height=350), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DIFFERENCES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Differences vs Original":
    st.markdown("## Differences vs. Original")
    st.markdown("<div style='color:#64748b'>Key deviations from Krauss et al. (2017)</div>", unsafe_allow_html=True)
    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='section-header'>Data</div>", unsafe_allow_html=True)
        diffs_data = [
            ("Sample period",     "Dec 1992–Oct 2015", "Dec 1992–Dec 2024", "⚠️"),
            ("Universe source",   "Thomson Reuters",   "CRSP",              "⚠️"),
            ("N constituents",    "~500",              "1,288",             "⚠️"),
            ("Missing data",      "Not specified",     "Listwise deletion", "✅"),
            ("Return definition", "Daily ret",         "Daily ret (CRSP)",  "✅"),
        ]
        df_d = pd.DataFrame(diffs_data, columns=["Item","Original","Ours","Status"])
        st.dataframe(df_d, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("<div class='section-header'>Methodology</div>", unsafe_allow_html=True)
        diffs_m = [
            ("Std errors",       "Not stated",        "Newey-West 1-lag",   "✅"),
            ("Factor model",     "None",              "Fama-French 5F",     "✅"),
            ("Rebalancing",      "Daily",             "Daily",              "✅"),
            ("TC assumption",    "0.05% half-turn",   "0.05% half-turn",    "✅"),
            ("Train window",     "750 days",          "750 days",           "✅"),
        ]
        df_m = pd.DataFrame(diffs_m, columns=["Item","Original","Ours","Status"])
        st.dataframe(df_m, use_container_width=True, hide_index=True)

    with col3:
        st.markdown("<div class='section-header'>Code & Tools</div>", unsafe_allow_html=True)
        diffs_c = [
            ("Language",          "R",             "Python",               "⚠️"),
            ("RF library",        "H2O",           "scikit-learn",         "⚠️"),
            ("DNN framework",     "H2O",           "PyTorch",              "⚠️"),
            ("GBT library",       "H2O",           "XGBoost",              "⚠️"),
            ("min_samples_leaf",  "Not set",       "150 (added)",          "⚠️"),
            ("Seed",              "1",             "1",                    "✅"),
        ]
        df_c = pd.DataFrame(diffs_c, columns=["Item","Original","Ours","Status"])
        st.dataframe(df_c, use_container_width=True, hide_index=True)

    st.markdown("")
    st.markdown("<div class='section-header'>Performance Gap Analysis (RF, k=10)</div>", unsafe_allow_html=True)

    metrics = ["Mean return", "Std deviation", "Sharpe ratio", "Sortino ratio"]
    t3_sub = t3[t3["metric"].isin(metrics)].copy()

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(
        name="Paper Post-TC", x=t3_sub["metric"], y=t3_sub["paper_post_tc"],
        marker_color=COLORS["amber"], opacity=0.85,
    ))
    fig_gap.add_trace(go.Bar(
        name="Our Post-TC", x=t3_sub["metric"], y=t3_sub["our_post_tc"],
        marker_color=COLORS["teal"],
    ))
    fig_gap.update_layout(
        barmode="group", height=300,
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
        font=dict(color="#6b7280"),
        xaxis=dict(gridcolor="#e8ecf4"),
        yaxis=dict(gridcolor="#e8ecf4", title="Value"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)"),
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    st.info("⚠️  Primary gap driver: scikit-learn RF vs H2O RF differ in tree-building details and default hyperparameters. The addition of `min_samples_leaf=150` was necessary to prevent overfitting but shifts results away from the original H2O implementation.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ROBUSTNESS CHECKS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛡️ Robustness Checks":
    st.markdown("## Robustness Checks")
    st.markdown("<div style='color:#64748b'>Sensitivity to k, transaction costs, and sub-sample periods</div>", unsafe_allow_html=True)
    st.markdown("")

    tab1, tab2, tab3 = st.tabs(["Portfolio Size k", "Transaction Cost Sensitivity", "GBT Period Stability"])

    with tab1:
        st.markdown("<div class='section-header'>Performance vs Portfolio Size k (RF)</div>", unsafe_allow_html=True)
        ks = [10, 50, 100, 150, 200]
        rows = []
        for k in ks:
            raw = bt_all[f"k{k}_raw_ret"]
            net = bt_all[f"k{k}_net_ret"] - (sel_tc - 5) / 10000 * 4
            p = perf_stats(net)
            rows.append({
                "k": k,
                "Day Ret (net)": net.mean(),
                "Ann Ret":       p["ann"],
                "Sharpe":        p["sharpe"],
                "Max DD":        p["mdd"],
                "% Positive":    (net > 0).mean(),
            })
        df_k = pd.DataFrame(rows)
        st.dataframe(df_k.style.format({
            "Day Ret (net)": "{:.4f}", "Ann Ret": "{:.2%}",
            "Sharpe": "{:.2f}", "Max DD": "{:.2%}", "% Positive": "{:.2%}",
        }), use_container_width=True, hide_index=True)

        # Sharpe vs k
        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(
            x=df_k["k"], y=df_k["Sharpe"], mode="lines+markers",
            line=dict(color=COLORS["teal"], width=2),
            marker=dict(size=10, color=COLORS["teal"]),
            name="Our Sharpe",
        ))
        fig_k.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
        fig_k.update_layout(
            title="Sharpe Ratio vs Portfolio Size k",
            height=280, paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
            font=dict(color="#6b7280"),
            xaxis=dict(gridcolor="#e8ecf4", title="k"),
            yaxis=dict(gridcolor="#e8ecf4", title="Sharpe"),
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_k, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-header'>Sharpe Ratio vs Transaction Cost (RF, k=10)</div>", unsafe_allow_html=True)
        tc_range = np.arange(0, 25, 1)
        sharpes = []
        for tc_bps in tc_range:
            adj = bt_rf["raw_ret"] - (tc_bps / 10000) * 4
            sharpes.append(perf_stats(adj)["sharpe"])

        fig_tc = go.Figure()
        fig_tc.add_trace(go.Scatter(
            x=tc_range, y=sharpes, mode="lines",
            line=dict(color=COLORS["teal"], width=2), fill="tozeroy",
            fillcolor="rgba(2,192,154,0.1)", name="Sharpe",
        ))
        fig_tc.add_vline(x=sel_tc, line_dash="dash", line_color="#f08c00",
                         annotation_text=f"Selected: {sel_tc}bps",
                         annotation_font_color="#f08c00")
        fig_tc.add_vline(x=5, line_dash="dot", line_color="#9ca3af",
                         annotation_text="Paper: 5bps",
                         annotation_font_color="#6b7280")
        fig_tc.add_hline(y=0, line_color="#e03131", line_dash="dash")
        fig_tc.update_layout(
            height=300, paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
            font=dict(color="#6b7280"),
            xaxis=dict(gridcolor="#e8ecf4", title="TC (bps/side)"),
            yaxis=dict(gridcolor="#e8ecf4", title="Sharpe Ratio"),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_tc, use_container_width=True)

        breakeven_tc = None
        for tc_bps, sh in zip(tc_range, sharpes):
            if sh < 0:
                breakeven_tc = tc_bps
                break
        if breakeven_tc:
            st.markdown(f"**Break-even TC:** ~{breakeven_tc} bps/side — strategy unprofitable above this level")

    with tab3:
        st.markdown("<div class='section-header'>GBT Sharpe Ratio by Study Period</div>", unsafe_allow_html=True)
        fig_gbt = go.Figure()
        colors_pos = [COLORS["teal"] if s > 0 else COLORS["red"] for s in gbt_p["sharpe_ratio"]]
        fig_gbt.add_trace(go.Bar(
            x=gbt_p["period"], y=gbt_p["sharpe_ratio"],
            marker_color=colors_pos,
            hovertemplate="Period %{x}<br>Sharpe: %{y:.2f}<extra></extra>",
        ))
        fig_gbt.add_hline(y=0, line_color="#9ca3af")
        fig_gbt.update_layout(
            height=300, paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
            font=dict(color="#6b7280"),
            xaxis=dict(gridcolor="#e8ecf4", title="Study Period", dtick=2),
            yaxis=dict(gridcolor="#e8ecf4", title="Sharpe Ratio"),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_gbt, use_container_width=True)

        # Gini over periods
        fig_gini = go.Figure()
        fig_gini.add_trace(go.Scatter(
            x=gbt_p["period"], y=gbt_p["train_gini"],
            mode="lines+markers", line=dict(color=COLORS["purple"], width=2),
            name="Train Gini",
        ))
        fig_gini.update_layout(
            title="GBT Training Gini by Period (model quality indicator)",
            height=260, paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
            font=dict(color="#6b7280"),
            xaxis=dict(gridcolor="#e8ecf4", title="Period"),
            yaxis=dict(gridcolor="#e8ecf4", title="Gini"),
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_gini, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REGIME SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Regime Sensitivity":
    st.markdown("## Regime Sensitivity")
    st.markdown("<div style='color:#64748b'>Performance across bull/bear, crisis, and macro-regime sub-periods</div>", unsafe_allow_html=True)
    st.markdown("")

    # Compute regime stats using selected k and TC
    k_col = f"k{sel_k}_net_ret"
    if k_col in bt_all.columns:
        base_series = bt_all[k_col] - (sel_tc - 5) / 10000 * 4
    else:
        base_series = bt_rf["net_ret"] - (sel_tc - 5) / 10000 * 4

    regime_rows = []
    for name, s, e, color in REGIME_DEFS:
        sub = base_series.loc[s:e]
        if len(sub) == 0:
            continue
        p = perf_stats(sub)
        pct_pos = (sub > 0).mean()
        regime_rows.append({
            "Regime": name.replace("\n", " "),
            "Start": s, "End": e,
            "N Days": p["n"],
            "Day Ret": sub.mean(),
            "Ann Ret": p["ann"],
            "Sharpe": p["sharpe"],
            "Max DD": p["mdd"],
            "% Positive": pct_pos,
            "_color": color,
        })

    df_reg = pd.DataFrame(regime_rows)

    # KPI cards
    cols = st.columns(len(regime_rows))
    for i, (col, row) in enumerate(zip(cols, regime_rows)):
        sh = row["Sharpe"]
        val_color = "#02c39a" if sh > 1 else ("#fbbf24" if sh > 0 else "#f87171")
        col.markdown(f"""<div class='metric-card' style='border-left-color:{row["_color"]}'>
            <div class='metric-label' style='font-size:9px'>{row["Regime"]}</div>
            <div class='metric-value' style='font-size:18px;color:{val_color}'>{sh:.2f}</div>
            <div class='metric-sub'>Sharpe · {row["Ann Ret"]:.0%} ann</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        # Cumulative return with regime shading
        cum = (1 + base_series).cumprod()
        fig_reg = go.Figure()

        # Regime shading
        for name, s, e, color in REGIME_DEFS:
            fig_reg.add_vrect(x0=s, x1=e, fillcolor=color, opacity=0.07, line_width=0)
            mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
            fig_reg.add_annotation(
                x=mid, y=cum.max() * 1.05, text=name.split("\n")[0],
                showarrow=False, font=dict(color=color, size=9),
            )

        fig_reg.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            line=dict(color=COLORS["teal"], width=2), name=f"RF k={sel_k}",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}x<extra></extra>",
        ))
        fig_reg.add_hline(y=1, line_dash="dot", line_color="#9ca3af")
        fig_reg.update_layout(
            title=f"Cumulative Return by Regime (k={sel_k}, TC={sel_tc}bps)",
            height=380, paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
            font=dict(color="#6b7280"),
            xaxis=dict(gridcolor="#e8ecf4", tickformat="%Y"),
            yaxis=dict(gridcolor="#e8ecf4"),
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_r:
        st.markdown("<div class='section-header'>Regime Performance Table</div>", unsafe_allow_html=True)
        df_disp = df_reg[["Regime","Ann Ret","Sharpe","Max DD","% Positive","N Days"]].copy()
        st.dataframe(df_disp.style.format({
            "Ann Ret": "{:.1%}", "Sharpe": "{:.2f}",
            "Max DD": "{:.1%}", "% Positive": "{:.1%}",
        }), use_container_width=True, hide_index=True)

        st.markdown("")
        # Sharpe bar per regime
        fig_sh = go.Figure(go.Bar(
            x=[r["Regime"] for r in regime_rows],
            y=[r["Sharpe"] for r in regime_rows],
            marker_color=[r["_color"] for r in regime_rows],
        ))
        fig_sh.add_hline(y=0, line_color="#9ca3af")
        fig_sh.update_layout(
            height=250, paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
            font=dict(color=COLORS["slate"], size=9),
            xaxis=dict(gridcolor="#e8ecf4", tickangle=-30),
            yaxis=dict(gridcolor="#e8ecf4", title="Sharpe"),
            margin=dict(l=40, r=10, t=20, b=60),
        )
        st.plotly_chart(fig_sh, use_container_width=True)

    # Rolling Sharpe
    st.markdown("<div class='section-header'>Rolling 252-Day Sharpe Ratio</div>", unsafe_allow_html=True)
    rolling_sh = base_series.rolling(252).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=rolling_sh.index, y=rolling_sh.values,
        line=dict(color=COLORS["teal"], width=1.5), name="Rolling Sharpe",
        fill="tozeroy", fillcolor="rgba(2,192,154,0.08)",
    ))
    fig_roll.add_hline(y=0, line_color="#e03131", line_dash="dash")
    fig_roll.update_layout(
        height=250, paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
        font=dict(color="#6b7280"),
        xaxis=dict(gridcolor="#e8ecf4", tickformat="%Y"),
        yaxis=dict(gridcolor="#e8ecf4"),
        margin=dict(l=40, r=20, t=10, b=30),
    )
    st.plotly_chart(fig_roll, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — RISK DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Diagnostics":
    st.markdown("## Risk Diagnostics")
    st.markdown("<div style='color:#64748b'>Factor exposure, tail risk & performance attribution — ENS1 k=10, Paper Period (1992–2015)</div>", unsafe_allow_html=True)
    st.markdown("")

    # ── Load data ──────────────────────────────────────────────────────────────
    @st.cache_data
    def load_risk_data(data_dir):
        import numpy as np
        ens  = pd.read_csv(f"{data_dir}/ensemble_portfolio_returns.csv", parse_dates=["date"])
        ens1 = ens[(ens["ensemble"]=="ENS1")&(ens["k"]==10)].set_index("date").sort_index()
        univ = pd.read_csv(f"{data_dir}/daily_universe.csv", parse_dates=["date"])
        mkt  = univ.groupby("date").apply(
            lambda x: np.average(x["ret"], weights=x["mktcap"])
        ).rename("mkt_ret")
        TC = 0.0005
        ens1["net_ret"] = ens1["portfolio_ret"] - 4*TC
        # Paper period
        ens1_p   = ens1.loc[:"2015-12-31", "net_ret"]
        ens1_pre = ens1.loc[:"2015-12-31", "portfolio_ret"]
        mkt_p    = mkt.loc[:"2015-12-31"].reindex(ens1_p.index).dropna()
        ens1_p   = ens1_p.reindex(mkt_p.index)
        ens1_pre = ens1_pre.reindex(mkt_p.index)
        return ens1_p, ens1_pre, mkt_p, ens1

    @st.cache_data
    def load_ff_factors(data_dir):
        def read_ff(path):
            df = pd.read_excel(path)
            df = df.rename(columns={df.columns[0]: "date"})
            df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date")
            for c in df.columns: df[c] = df[c] / 100
            return df
        ff3 = read_ff(f"{data_dir}/FF3.xlsx")
        ff5 = read_ff(f"{data_dir}/FF5.xlsx")
        mom = read_ff(f"{data_dir}/FF-Momentum_Factor_daily.xlsx")
        rev = read_ff(f"{data_dir}/F-F_ST_Reversal_Factor_daily.xlsx")
        return ff3, ff5, mom, rev

    ens1_p, ens1_pre, mkt_p, ens1_full = load_risk_data(DATA)

    # Try loading FF factors
    try:
        ff3, ff5, mom_f, rev_f = load_ff_factors(DATA)
        has_ff = True
    except Exception:
        has_ff = False

    tc_adj = (sel_tc - 5) / 10000 * 4
    ret = ens1_p - tc_adj
    p   = perf_stats(ret)
    var_1  = ret.quantile(0.01)
    var_5  = ret.quantile(0.05)
    cvar_1 = ret[ret <= var_1].mean()
    cvar_5 = ret[ret <= var_5].mean()

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val, sub, good in [
        (c1, "Max Drawdown",    f"{p['mdd']:.1%}",      f"Sharpe: {p['sharpe']:.2f}", p["mdd"] > -0.5),
        (c2, "95% VaR",         f"{var_5:.2%}",           "Daily 5th pctile",             False),
        (c3, "95% CVaR",        f"{cvar_5:.2%}",          "Expected tail loss",           False),
        (c4, "Skewness",        f"{ret.skew():.2f}",      "Right > 0 preferred",          ret.skew() > 0),
        (c5, "Excess Kurtosis", f"{ret.kurtosis():.1f}",  "Fat tails > 0",                False),
    ]:
        vc = COLORS["teal"] if good else COLORS["red"]
        col.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value' style='font-size:18px;color:{vc}'>{val}</div>
            <div class='metric-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Row 1: Drawdown + Distribution ───────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        cum = (1 + ret).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100,
            fill="tozeroy", fillcolor="rgba(224,49,49,0.15)",
            line=dict(color=COLORS["red"], width=1.5), name="Drawdown",
        ))
        fig_dd.update_layout(
            title="Drawdown (%) — ENS1 k=10, Paper Period",
            height=300, paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
            font=dict(color="#6b7280"),
            xaxis=dict(gridcolor="#e8ecf4", tickformat="%Y"),
            yaxis=dict(gridcolor="#e8ecf4", title="%"),
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    with col_r:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=ret.values*100, nbinsx=80,
            marker_color=COLORS["primary"], opacity=0.7, name="ENS1",
        ))
        fig_hist.add_trace(go.Histogram(
            x=mkt_p.values*100, nbinsx=80,
            marker_color=COLORS["amber"], opacity=0.5, name="Market",
        ))
        fig_hist.add_shape(type="line", x0=var_1*100, x1=var_1*100, y0=0, y1=1,
                           xref="x", yref="paper",
                           line=dict(color=COLORS["red"], width=1.5, dash="dash"))
        fig_hist.add_annotation(x=var_1*100, y=0.92, xref="x", yref="paper",
                                text="1% VaR", showarrow=False,
                                font=dict(color=COLORS["red"], size=10), xanchor="right")
        fig_hist.add_shape(type="line", x0=var_5*100, x1=var_5*100, y0=0, y1=1,
                           xref="x", yref="paper",
                           line=dict(color=COLORS["amber"], width=1.5, dash="dash"))
        fig_hist.add_annotation(x=var_5*100, y=0.82, xref="x", yref="paper",
                                text="5% VaR", showarrow=False,
                                font=dict(color=COLORS["amber"], size=10), xanchor="right")
        fig_hist.update_layout(
            title="Return Distribution — ENS1 vs Market (daily %)",
            barmode="overlay", height=300,
            paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
            font=dict(color="#6b7280"),
            xaxis=dict(gridcolor="#e8ecf4", title="Daily return (%)"),
            yaxis=dict(gridcolor="#e8ecf4"),
            legend=dict(bgcolor="rgba(255,255,255,0.9)"),
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Row 2: Tail Risk table + Factor Regression ────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<div class='section-header'>Tail Risk — ENS1 vs Market</div>", unsafe_allow_html=True)
        mkt_var1  = mkt_p.quantile(0.01)
        mkt_cvar1 = mkt_p[mkt_p <= mkt_var1].mean()
        mkt_var5  = mkt_p.quantile(0.05)
        mkt_cvar5 = mkt_p[mkt_p <= mkt_var5].mean()
        mkt_mdd   = ((1+mkt_p).cumprod() / ((1+mkt_p).cumprod().cummax()) - 1).min()

        tail_df = pd.DataFrame({
            "Metric":  ["1% VaR", "1% CVaR", "5% VaR", "5% CVaR", "Max Drawdown", "Std Dev"],
            "ENS1":    [var_1, cvar_1, var_5, cvar_5, p["mdd"], ret.std()],
            "Market":  [mkt_var1, mkt_cvar1, mkt_var5, mkt_cvar5, mkt_mdd, mkt_p.std()],
        })
        tail_df["ENS1/Market"] = (tail_df["ENS1"] / tail_df["Market"]).round(2).astype(str) + "x"
        st.dataframe(tail_df.style.format({
            "ENS1": "{:.4f}", "Market": "{:.4f}",
        }), use_container_width=True, hide_index=True)
        st.caption("ENS1 1% VaR is ~1.9x market; CVaR ~1.9x. Higher tail risk vs market — a tradeoff for alpha generation.")

    with col_r:
        st.markdown("<div class='section-header'>Factor Regression (Newey-West SE)</div>", unsafe_allow_html=True)

        def ols_nw(y, X_df, lags=1):
            df = X_df.copy(); df["y"] = y; df = df.dropna()
            y_v = df["y"].values
            X_m = np.column_stack([np.ones(len(df))] + [df[c].values for c in X_df.columns])
            b   = np.linalg.lstsq(X_m, y_v, rcond=None)[0]
            res = y_v - X_m@b; n,k = len(y_v),len(b)
            S   = np.zeros((k,k)); xe = X_m*res.reshape(-1,1); S += xe.T@xe
            for lag in range(1,lags+1):
                w=1-lag/(lags+1); S+=w*(xe[lag:].T@xe[:-lag]+xe[:-lag].T@xe[lag:])
            V=np.linalg.inv(X_m.T@X_m)@S@np.linalg.inv(X_m.T@X_m)
            se=np.sqrt(np.diag(V)); t=b/se
            r2=1-np.sum(res**2)/np.sum((y_v-y_v.mean())**2)
            cols=["alpha"]+list(X_df.columns)
            return dict(zip(cols,b)), dict(zip(cols,t)), r2

        if has_ff:
            reg_results = {}
            X_ff3   = ff3[["Mkt-RF","SMB","HML"]].reindex(ens1_pre.index)
            X_ff3m  = ff3[["Mkt-RF","SMB","HML"]].join(rev_f["ST_Rev"]).join(mom_f["Mom"]).reindex(ens1_pre.index)
            X_ff5   = ff5[["Mkt-RF","SMB","HML","RMW","CMA"]].reindex(ens1_pre.index)

            for name, Xd in [("FF3",X_ff3),("FF3+Rev+Mom",X_ff3m),("FF5",X_ff5)]:
                b,t,r2 = ols_nw(ens1_pre, Xd)
                reg_results[name] = {"b":b,"t":t,"r2":r2}

            def sig(t_val):
                a=abs(t_val)
                return "***" if a>2.576 else "**" if a>1.96 else "*" if a>1.645 else ""

            rows = []
            factors = ["alpha","Mkt-RF","SMB","HML","RMW","CMA","ST_Rev","Mom"]
            labels  = ["Alpha (daily)","Market (Mkt-RF)","SMB","HML","RMW","CMA","ST Reversal","Momentum"]
            for fac, lab in zip(factors, labels):
                row = {"Factor": lab}
                for mname in ["FF3","FF3+Rev+Mom","FF5"]:
                    b = reg_results[mname]["b"]
                    t = reg_results[mname]["t"]
                    if fac in b:
                        s = sig(t[fac])
                        row[mname] = f"{b[fac]:.4f}{s}"
                    else:
                        row[mname] = "—"
                rows.append(row)
            rows.append({"Factor":"R²",
                         "FF3":    f"{reg_results['FF3']['r2']:.4f}",
                         "FF3+Rev+Mom": f"{reg_results['FF3+Rev+Mom']['r2']:.4f}",
                         "FF5":    f"{reg_results['FF5']['r2']:.4f}"})

            ff_table = pd.DataFrame(rows)
            st.dataframe(ff_table, use_container_width=True, hide_index=True)
            st.caption("Regressions use Pre-TC returns. *** p<0.01, ** p<0.05, * p<0.10. Alpha robust across all factor models — strategy generates excess returns unexplained by common risk factors.")
        else:
            st.info("Place FF factor files in results folder: FF3.xlsx, FF5.xlsx, FF-Momentum_Factor_daily.xlsx, F-F_ST_Reversal_Factor_daily.xlsx")

    # ── All models risk summary ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>Risk Summary — All Models, Paper Period Post-TC (k=10)</div>", unsafe_allow_html=True)
    TC = 0.0005
    risk_rows = []
    ens_src = pd.read_csv(f"{DATA}/ensemble_portfolio_returns.csv", parse_dates=["date"])
    for mname in ["DNN","GBT","RAF","ENS1","ENS2","ENS3"]:
        sub = ens_src[(ens_src["ensemble"]==mname)&(ens_src["k"]==10)].set_index("date").sort_index()
        sub = sub.loc[:"2015-12-31"]
        r   = sub["portfolio_ret"] - 4*TC
        if len(r) == 0: continue
        pp  = perf_stats(r)
        risk_rows.append({
            "Model":    mname,
            "Ann Ret":  pp["ann"],
            "Sharpe":   pp["sharpe"],
            "Max DD":   pp["mdd"],
            "1% VaR":   r.quantile(0.01),
            "5% VaR":   r.quantile(0.05),
            "1% CVaR":  r[r<=r.quantile(0.01)].mean(),
            "Skewness": r.skew(),
            "Kurtosis": r.kurtosis(),
        })
    df_risk = pd.DataFrame(risk_rows)
    st.dataframe(df_risk.style.format({
        "Ann Ret": "{:.2%}", "Sharpe": "{:.2f}", "Max DD": "{:.2%}",
        "1% VaR": "{:.3%}", "5% VaR": "{:.3%}", "1% CVaR": "{:.3%}",
        "Skewness": "{:.3f}", "Kurtosis": "{:.2f}",
    }), use_container_width=True, hide_index=True)
