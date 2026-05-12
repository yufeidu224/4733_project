"""
IEOR 4733 - ML Statistical Arbitrage on S&P 500
DNN Backtest — k=10, Table 2 replication (Krauss et al. 2017)

Outputs:
    data/dnn_backtest_results_k10.csv   — daily returns (same format as RF)
    data/dnn_perf_paper_period.csv      — Table-2 metrics, 1992-12-17 – 2015-10-15
    data/dnn_perf_post_paper.csv        — Table-2 metrics, 2015-10-16 – 2024-09-25
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

TC_PER_DAY = 0.002        # 20 bps: 0.05% per share per half-turn × 2 sides × 2 turns
TDPY       = 250

PAPER_START = "1992-12-17"
PAPER_END   = "2015-10-15"
EXT_START   = "2015-10-16"
EXT_END     = "2024-09-25"

DATA_DIR    = "Projects/data"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def nw_se(series: pd.Series, lags: int = 1) -> float:
    """Newey-West heteroscedasticity-and-autocorrelation-consistent SE."""
    y = series.values
    X = np.ones((len(y), 1))
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags}, use_t=False)
    return float(res.bse[0])


def pt_test(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Pesaran-Timmermann (1992) test statistic.
    Applied at the individual stock level:
      y_true : actual binary label (Y=1 if stock > cross-sectional median)
      y_score: predicted probability; predicted class = (y_score > 0.5)
    """
    f = (y_score > 0.5).astype(float)
    y = y_true.astype(float)
    N = len(y)
    P1   = y.mean()              # actual positive rate
    P2   = f.mean()              # predicted positive rate
    Phat = (f == y).mean()       # directional accuracy
    Pstar = P1 * P2 + (1 - P1) * (1 - P2)
    var_Phat  = Pstar * (1 - Pstar) / N
    var_Pstar = ((2*P2 - 1)**2 * P1*(1-P1) +
                 (2*P1 - 1)**2 * P2*(1-P2)) / N
    denom = var_Phat - var_Pstar
    if denom <= 0:
        return np.nan
    return float((Phat - Pstar) / np.sqrt(denom))


def compute_metrics(ret: pd.Series, long_ret: pd.Series, short_ret: pd.Series,
                    pt_stat: float) -> dict:
    """All metrics matching Krauss et al. (2017) Table 2."""
    n      = len(ret)
    mu     = ret.mean()
    sigma  = ret.std()

    # Annualised return via compounding
    total    = (1 + ret).prod() - 1
    ann_ret  = (1 + total) ** (TDPY / n) - 1

    # Newey-West SE & t-stat (1-lag, as in paper)
    se_nw  = nw_se(ret, lags=1)
    t_nw   = mu / se_nw

    # Drawdown series
    cum    = (1 + ret).cumprod()
    dd     = (cum - cum.cummax()) / cum.cummax()
    max_dd = abs(dd.min())        # stored as positive magnitude (paper convention)

    # Calmar = ann_ret / max_dd
    calmar = ann_ret / max_dd if max_dd > 0 else np.nan

    # VaR / CVaR (historical, negative = loss)
    var1  = np.percentile(ret, 1)
    cvar1 = ret[ret <= var1].mean()
    var5  = np.percentile(ret, 5)
    cvar5 = ret[ret <= var5].mean()

    return {
        "Mean return (long)":         round(long_ret.mean(), 6),
        "Mean return (short)":        round(short_ret.mean(), 6),
        "Mean return":                round(mu, 6),
        "Standard error (NW)":        round(se_nw, 6),
        "t-statistic (NW)":           round(t_nw, 4),
        "PT test statistic":          round(pt_stat, 4) if not np.isnan(pt_stat) else "",
        "Minimum":                    round(ret.min(), 4),
        "Quartile 1":                 round(np.percentile(ret, 25), 4),
        "Median":                     round(np.median(ret), 4),
        "Quartile 3":                 round(np.percentile(ret, 75), 4),
        "Maximum":                    round(ret.max(), 4),
        "Standard deviation":         round(sigma, 4),
        "Skewness":                   round(stats.skew(ret), 4),
        "Kurtosis":                   round(stats.kurtosis(ret), 4),   # excess (Fisher)
        "Historical 1-percent VaR":   round(var1, 4),
        "Historical 1-percent CVaR":  round(cvar1, 4),
        "Historical 5-percent VaR":   round(var5, 4),
        "Historical 5-percent CVaR":  round(cvar5, 4),
        "Maximum drawdown":           round(max_dd, 4),
        "Calmar ratio":               round(calmar, 4),
        "Share with return > 0":      round((ret > 0).mean(), 4),
    }


def build_table(port_slice: pd.DataFrame, pred_slice: pd.DataFrame) -> pd.DataFrame:
    """
    Build a Table-2 style DataFrame with columns [metric, before_TC, after_TC].
    port_slice : rows from dnn_backtest_results_k10 for the period
    pred_slice : rows from dnn_predictions merged with Y for the period
    """
    raw = port_slice["raw_ret"]
    net = port_slice["net_ret"]
    lon = port_slice["long_ret"]
    sho = port_slice["short_ret"]

    pt = pt_test(pred_slice["Y"].values, pred_slice["prob_outperform"].values)

    m_raw = compute_metrics(raw, lon, sho, pt)
    m_net = compute_metrics(net, lon - TC_PER_DAY / 2, sho + TC_PER_DAY / 2, np.nan)

    df = pd.DataFrame({
        "metric":     list(m_raw.keys()),
        "before_TC":  list(m_raw.values()),
        "after_TC":   list(m_net.values()),
    })
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data...")
port = pd.read_csv(f"{DATA_DIR}/dnn_portfolio_returns.csv", parse_dates=["date"])
univ = pd.read_csv(f"{DATA_DIR}/daily_universe.csv",        parse_dates=["date"])
pred = pd.read_csv(f"{DATA_DIR}/dnn_predictions.csv",       parse_dates=["date"])
feat = pd.read_csv(f"{DATA_DIR}/features.csv",              parse_dates=["date"])

# Merge actual Y into predictions
pred = pred.merge(feat[["permno", "date", "Y"]], on=["permno", "date"], how="left")

# Universe size per day
universe_size = univ.groupby("date").size().rename("n_stocks").reset_index()

# Build daily backtest file (k=10)
k10 = port[port["k"] == 10].copy()
k10 = k10.merge(universe_size, on="date", how="left")

out = pd.DataFrame({
    "date":      k10["date"].dt.strftime("%Y-%m-%d"),
    "long_ret":  k10["long_ret"].values,
    "short_ret": k10["short_ret"].values,
    "raw_ret":   k10["portfolio_ret"].values,
    "net_ret":   (k10["portfolio_ret"] - TC_PER_DAY).values,
    "n_stocks":  k10["n_stocks"].fillna(0).astype(int).values,
}).sort_values("date").reset_index(drop=True)

out.to_csv(f"{DATA_DIR}/dnn_backtest_results_k10.csv", index=False)
print(f"  Saved {len(out):,} rows → dnn_backtest_results_k10.csv")

out["date"] = pd.to_datetime(out["date"])

# ══════════════════════════════════════════════════════════════════════════════
# Split periods
# ══════════════════════════════════════════════════════════════════════════════

paper_port = out[(out["date"] >= PAPER_START) & (out["date"] <= PAPER_END)].reset_index(drop=True)
paper_pred = pred[(pred["date"] >= PAPER_START) & (pred["date"] <= PAPER_END)].reset_index(drop=True)

ext_port   = out[(out["date"] >= EXT_START) & (out["date"] <= EXT_END)].reset_index(drop=True)
ext_pred   = pred[(pred["date"] >= EXT_START) & (pred["date"] <= EXT_END)].reset_index(drop=True)

print(f"  Paper period : {PAPER_START} – {PAPER_END}  ({len(paper_port):,} days)")
print(f"  Extension    : {EXT_START}  – {EXT_END}  ({len(ext_port):,} days)")

# ══════════════════════════════════════════════════════════════════════════════
# Compute & save Table-2 metrics
# ══════════════════════════════════════════════════════════════════════════════

print("\nComputing Table-2 metrics...")

paper_tbl = build_table(paper_port, paper_pred)
ext_tbl   = build_table(ext_port,   ext_pred)

paper_tbl.to_csv(f"{DATA_DIR}/dnn_perf_paper_period.csv", index=False)
ext_tbl.to_csv(  f"{DATA_DIR}/dnn_perf_post_paper.csv",   index=False)

print(f"  Saved → dnn_perf_paper_period.csv")
print(f"  Saved → dnn_perf_post_paper.csv")

# ── Pretty print ──────────────────────────────────────────────────────────────
for label, tbl in [("PAPER PERIOD", paper_tbl), ("EXTENSION PERIOD", ext_tbl)]:
    print(f"\n{'='*60}")
    print(f"  DNN k=10 — {label}")
    print(f"{'='*60}")
    print(f"  {'Metric':<32} {'Before TC':>12} {'After TC':>12}")
    print("  " + "-"*56)
    for _, row in tbl.iterrows():
        print(f"  {row['metric']:<32} {str(row['before_TC']):>12} {str(row['after_TC']):>12}")
