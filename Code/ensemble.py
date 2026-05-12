"""
IEOR 4733 - ML Statistical Arbitrage on S&P 500
Step 5 (ENS): Ensemble Methods ENS1 / ENS2 / ENS3

Replicates Krauss et al. (2017) Section 4.4, equations (5)–(7):

  ENS1: P̂^ENS1 = (1/M) * sum_i P̂^i                        [eq. 5, equal weights]
  ENS2: P̂^ENS2 = sum_i w^i * P̂^i,  w^i = g^i / sum g^j    [eq. 6, Gini-weighted]
  ENS3: P̂^ENS3 = sum_i w^i * P̂^i,  w^i = (1/R^i)/sum(1/R^j) [eq. 7, rank-based]

where g^i = 2*(AUC^i_train - 0.5) is the Gini index computed on the training set.

Usage:
    python ensemble.py

Inputs (from data/):
    dnn_predictions.csv      — period, date, permno, prob_outperform
    gbt_predictions.csv      — date, permno, prob_GBT, period, train_gini
    rf_predictions.csv       — date, permno, prob_RAF, period, train_gini  (optional)
    features.csv             — for computing DNN train_gini
    study_periods.csv        — period definitions (aligned to DNN)
    daily_universe.csv       — for next-day return lookup

Outputs (to data/):
    ensemble_predictions.csv           — merged scores + ENS1/2/3
    ensemble_portfolio_returns.csv     — daily long-short P&L for each ensemble/model & k
    ensemble_performance_paper.csv     — performance metrics, paper period (up to 2015-12-31)
    ensemble_performance_extended.csv  — performance metrics, extended period (2016-01-01 onward)
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Resolve paths relative to this script file so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "..", "data")

K_VALUES = [10, 50, 100, 150, 200]
TDPY     = 250  # trading days per year


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio backtester  (same logic as dnn_training.py)
# ══════════════════════════════════════════════════════════════════════════════

def compute_daily_portfolio(day_df: pd.DataFrame,
                            score_col: str,
                            ret_lookup: pd.Series,
                            date,
                            k: int) -> dict | None:
    """Long top-k, short bottom-k equal-weighted dollar-neutral."""
    if len(day_df) < 2 * k:
        return None
    ranked     = day_df.sort_values(score_col, ascending=False)
    long_idx   = [(date, p) for p in ranked.iloc[:k]["permno"]]
    short_idx  = [(date, p) for p in ranked.iloc[-k:]["permno"]]
    long_rets  = ret_lookup.reindex(long_idx).dropna().values
    short_rets = ret_lookup.reindex(short_idx).dropna().values
    if len(long_rets) == 0 or len(short_rets) == 0:
        return None
    return {
        "date":          date,
        "k":             k,
        "portfolio_ret": float(np.mean(long_rets) - np.mean(short_rets)),
        "long_ret":      float(np.mean(long_rets)),
        "short_ret":     float(np.mean(short_rets)),
        "n_long":        len(long_rets),
        "n_short":       len(short_rets),
    }


def backtest_ensemble(merged: pd.DataFrame,
                      score_col: str,
                      ret_lookup: pd.Series,
                      label: str) -> pd.DataFrame:
    """Run portfolio backtest for a given score column across all k values."""
    rows = []
    for date, day_df in merged.groupby("date"):
        period = int(day_df["period"].iloc[0])
        for k in K_VALUES:
            row = compute_daily_portfolio(day_df, score_col, ret_lookup, date, k)
            if row is not None:
                row["ensemble"] = label
                row["period"]   = period
                rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Performance summary
# ══════════════════════════════════════════════════════════════════════════════

def compute_performance(port_df: pd.DataFrame,
                        label: str,
                        date_start=None,
                        date_end=None) -> pd.DataFrame:
    """Return a DataFrame of annualised performance metrics for one model/ensemble.

    Rows = one per k value.  date_start / date_end are inclusive date filters
    (strings or Timestamps).
    """
    sub = port_df[port_df["ensemble"] == label].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    if date_start is not None:
        sub = sub[sub["date"] >= pd.Timestamp(date_start)]
    if date_end is not None:
        sub = sub[sub["date"] <= pd.Timestamp(date_end)]

    rows = []
    for k in K_VALUES:
        rets = sub[sub["k"] == k]["portfolio_ret"]
        if rets.empty:
            continue
        n         = len(rets)
        day_ret   = rets.mean()
        total_ret = (1 + rets).prod() - 1
        ann_ret   = (1 + total_ret) ** (TDPY / n) - 1
        ann_vol   = rets.std() * np.sqrt(TDPY)
        sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0
        cum       = (1 + rets).cumprod()
        max_dd    = ((cum - cum.cummax()) / cum.cummax()).min()
        rows.append({
            "model":        label,
            "k":            k,
            "n_days":       n,
            "day_ret":      round(day_ret,    8),
            "ann_ret":      round(ann_ret,    6),
            "ann_vol":      round(ann_vol,    6),
            "sharpe":       round(sharpe,     4),
            "max_drawdown": round(max_dd,     6),
            "total_ret":    round(total_ret,  6),
        })
    return pd.DataFrame(rows)


def print_performance_table(perf_df: pd.DataFrame,
                            label: str,
                            period_name: str,
                            date_range: str) -> None:
    """Pretty-print one model's metrics from a compute_performance() result."""
    sub = perf_df[perf_df["model"] == label]
    print(f"\n{'='*70}")
    print(f"  {label}  |  {period_name}  ({date_range})")
    print("=" * 70)
    print(f"  {'k':>5}  {'Day Ret':>8}  {'Ann.Ret':>8}  {'Ann.Vol':>8}  "
          f"{'Sharpe':>8}  {'MaxDD':>8}  {'Total Ret':>10}  {'N days':>7}")
    print("  " + "-" * 68)
    for _, row in sub.iterrows():
        print(f"  k={int(row['k']):<4} {row['day_ret']*100:>7.4f}%  "
              f"{row['ann_ret']*100:>7.2f}%  {row['ann_vol']*100:>7.2f}%  "
              f"{row['sharpe']:>8.3f}  {row['max_drawdown']*100:>7.2f}%  "
              f"{row['total_ret']*100:>9.1f}%  {int(row['n_days']):>7,}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 70)
    print("  Ensemble Methods (ENS1 / ENS2 / ENS3) — Krauss et al. (2017)")
    print("=" * 70)

    # ── Load study periods ────────────────────────────────────────────────────
    sp_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "study_periods.csv"),
        parse_dates=["train_start", "train_end", "test_start", "test_end"],
    )
    print(f"\n  study_periods.csv : {len(sp_df)} periods")

    # ── Load model predictions ────────────────────────────────────────────────
    dnn = pd.read_csv(os.path.join(OUTPUT_DIR, "dnn_predictions.csv"), parse_dates=["date"])
    dnn = dnn.rename(columns={"prob_outperform": "prob_DNN"})
    print(f"  DNN predictions   : {len(dnn):,} rows  "
          f"({dnn['date'].min().date()} – {dnn['date'].max().date()})")

    gbt = pd.read_csv(os.path.join(OUTPUT_DIR, "gbt_predictions.csv"), parse_dates=["date"])
    gbt = gbt.rename(columns={"train_gini": "gini_GBT"})
    print(f"  GBT predictions   : {len(gbt):,} rows  "
          f"({gbt['date'].min().date()} – {gbt['date'].max().date()})")

    # Optional: RF predictions
    rf_path = os.path.join(OUTPUT_DIR, "rf_predictions.csv")
    has_rf  = os.path.exists(rf_path)
    if has_rf:
        rf = pd.read_csv(rf_path, parse_dates=["date"])
        rf = rf.rename(columns={"train_gini": "gini_RAF"})
        print(f"  RAF predictions   : {len(rf):,} rows  "
              f"({rf['date'].min().date()} – {rf['date'].max().date()})")
    else:
        print("  RAF predictions   : NOT FOUND (will ensemble DNN + GBT only)")

    # ── Compute DNN Gini per study period ─────────────────────────────────────
    # Paper: Gini computed on training set.
    # DNN training predictions were not saved → we approximate using
    # test-period AUC on Y labels from features.csv.
    print("\nComputing DNN Gini per study period (test-period AUC approximation)...")
    features = pd.read_csv(os.path.join(OUTPUT_DIR, "features.csv"), parse_dates=["date"])

    dnn_gini = {}
    for _, sp in sp_df.iterrows():
        pid = int(sp["period"])
        # Test-period Y labels
        y_df = features[
            (features["date"] >= sp["test_start"]) &
            (features["date"] <= sp["test_end"])
        ][["date", "permno", "Y"]]
        # DNN predictions for this period
        preds_pid = dnn[dnn["period"] == pid][["date", "permno", "prob_DNN"]]
        merged_p  = y_df.merge(preds_pid, on=["date", "permno"])
        if len(merged_p) < 10 or merged_p["Y"].nunique() < 2:
            dnn_gini[pid] = np.nan
            continue
        auc = roc_auc_score(merged_p["Y"], merged_p["prob_DNN"])
        dnn_gini[pid] = 2 * (auc - 0.5)

    dnn["gini_DNN"] = dnn["period"].map(dnn_gini)
    valid_gini = {k: v for k, v in dnn_gini.items() if not np.isnan(v)}
    print(f"  DNN Gini computed for {len(valid_gini)}/{len(sp_df)} periods  "
          f"(mean={np.nanmean(list(dnn_gini.values())):.4f})")

    # ── Merge predictions on (date, permno) ──────────────────────────────────
    # Inner join ensures we only ensemble where ALL models have predictions.
    # GBT and RAF each use their own period numbering internally; only (date, permno)
    # is used as the join key so period alignment is preserved from DNN.
    print("\nMerging predictions on (date, permno)...")
    merged = dnn[["date", "permno", "period", "prob_DNN", "gini_DNN"]].merge(
        gbt[["date", "permno", "prob_GBT", "gini_GBT"]],
        on=["date", "permno"],
        how="inner",
    )

    if has_rf:
        merged = merged.merge(
            rf[["date", "permno", "prob_RAF", "gini_RAF"]],
            on=["date", "permno"],
            how="inner",
        )
        models   = ["DNN", "GBT", "RAF"]
        n_models = 3
    else:
        models   = ["DNN", "GBT"]
        n_models = 2

    print(f"  Merged rows       : {len(merged):,}  "
          f"({merged['date'].min().date()} – {merged['date'].max().date()})")
    print(f"  Models in ensemble: {models}")

    if n_models < 3:
        print("\n  [WARNING] RF predictions not found. ENS metrics are computed "
              "over DNN+GBT only. For full paper replication add rf_predictions.csv.")

    # ── ENS1: Equal-weighted average ─────────────────────────────────────────
    # P̂^ENS1 = (1/M) * sum_i P̂^i    [eq. 5]
    prob_cols      = [f"prob_{m}" for m in models]
    merged["ENS1"] = merged[prob_cols].mean(axis=1)

    # ── ENS2: Gini-weighted, per row ─────────────────────────────────────────
    # w^i = g^i / sum_j g^j  [eq. 6]
    # Each row already carries the training-set gini for its own model period.
    gini_cols      = [f"gini_{m}" for m in models]
    gini_sum       = merged[gini_cols].sum(axis=1).replace(0, np.nan)  # avoid /0
    merged["ENS2"] = sum(
        merged[f"prob_{m}"] * (merged[f"gini_{m}"] / gini_sum)
        for m in models
    )
    # Fall back to equal weights on any row where all ginis are zero
    ens2_fallback  = merged["ENS2"].isna()
    merged.loc[ens2_fallback, "ENS2"] = merged.loc[ens2_fallback, prob_cols].mean(axis=1)

    # ── ENS3: Rank-based, per DNN study period ───────────────────────────────
    # R^i = rank of model i by Gini (1 = highest); w^i = (1/R^i)/sum(1/R^j)  [eq. 7]
    #
    # GBT and RAF use their own internal period boundaries that don't align
    # exactly with DNN periods, so gini_GBT / gini_RAF may take two distinct
    # values within one DNN period.  We average per DNN period to get a single
    # representative gini before ranking — this is robust to the offset.
    period_gini = merged.groupby("period")[gini_cols].mean()

    # Rank within each period (rank 1 = highest Gini)
    period_rank = period_gini.rank(axis=1, ascending=False, method="min").astype(int)
    period_rank.columns = [f"rank_{m}" for m in models]

    # Inverse-rank weights per period
    inv_rank   = 1.0 / period_rank
    weight_sum = inv_rank.sum(axis=1)
    period_weight = inv_rank.div(weight_sum, axis=0)
    period_weight.columns = [f"w3_{m}" for m in models]

    merged = merged.merge(period_weight.reset_index(), on="period", how="left")

    merged["ENS3"] = sum(
        merged[f"prob_{m}"] * merged[f"w3_{m}"]
        for m in models
    )

    # ── Save merged predictions ───────────────────────────────────────────────
    save_cols = (
        ["period", "date", "permno"] + prob_cols + gini_cols
        + ["ENS1", "ENS2", "ENS3"]
    )
    out_path = os.path.join(OUTPUT_DIR, "ensemble_predictions.csv")
    merged[save_cols].to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}  ({len(merged):,} rows)")

    # ── Build next-day return lookup ──────────────────────────────────────────
    print("\nBuilding return lookup...")
    universe = pd.read_csv(os.path.join(OUTPUT_DIR, "daily_universe.csv"), parse_dates=["date"])
    univ_s   = universe.sort_values(["permno", "date"]).copy()
    univ_s["ret_next"] = univ_s.groupby("permno")["ret"].shift(-1)
    ret_lookup = (
        univ_s[["permno", "date", "ret_next"]]
        .dropna()
        .set_index(["date", "permno"])["ret_next"]
    )

    # ── Backtest each ensemble ────────────────────────────────────────────────
    print("\nBacktesting ensembles...")
    all_port = []
    for ens_col, ens_label in [("ENS1", "ENS1"), ("ENS2", "ENS2"), ("ENS3", "ENS3")]:
        print(f"  {ens_label}...", end=" ", flush=True)
        port = backtest_ensemble(merged, ens_col, ret_lookup, ens_label)
        all_port.append(port)
        print(f"{len(port):,} portfolio-day rows")

    # Also backtest individual models for comparison (restricted to overlap dates)
    for m_col, m_label in [(f"prob_{m}", m) for m in models]:
        print(f"  {m_label}...", end=" ", flush=True)
        port = backtest_ensemble(merged, m_col, ret_lookup, m_label)
        all_port.append(port)
        print(f"{len(port):,} portfolio-day rows")

    port_df = pd.concat(all_port, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, "ensemble_portfolio_returns.csv")
    port_df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    # ── Two-period performance reporting ─────────────────────────────────────
    # Paper period : ensemble start → 2015-12-31  (replication of Krauss et al.)
    # Extended period : 2016-01-01 → ensemble end  (out-of-sample extension)
    PAPER_END      = "2015-12-31"
    EXTENDED_START = "2016-01-01"
    ens_start = merged["date"].min().date()
    ens_end   = merged["date"].max().date()

    all_labels = ["ENS1", "ENS2", "ENS3"] + models

    # ── Paper period ─────────────────────────────────────────────────────────
    print(f"\n\n{'#'*70}")
    print(f"  PAPER PERIOD  ({ens_start} – {PAPER_END})")
    print(f"{'#'*70}")
    paper_perf_parts = []
    for label in all_labels:
        perf = compute_performance(port_df, label, date_end=PAPER_END)
        paper_perf_parts.append(perf)
        print_performance_table(perf, label, "Paper Period", f"{ens_start} – {PAPER_END}")
    paper_perf_df = pd.concat(paper_perf_parts, ignore_index=True)
    paper_perf_df.insert(0, "sub_period", "paper")

    out_path = os.path.join(OUTPUT_DIR, "ensemble_performance_paper.csv")
    paper_perf_df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    print("\n" + "=" * 70)
    print("  Paper Table 3 benchmarks (before TC):")
    print("    DNN  k=10: Day Ret ~0.33%, Sharpe ~2.44")
    print("    GBT  k=10: Day Ret ~0.43%, Sharpe ~3.16")
    print("    RAF  k=10: Day Ret ~0.43%, Sharpe ~3.09")
    print("    ENS1 k=10: Day Ret ~0.45%, Sharpe ~3.40")
    print("    ENS2 k=10: Day Ret ~0.45%, Sharpe ~3.43")
    print("    ENS3 k=10: Day Ret ~0.46%, Sharpe ~3.46")
    print("=" * 70)

    # ── Extended period ───────────────────────────────────────────────────────
    print(f"\n\n{'#'*70}")
    print(f"  EXTENDED PERIOD  ({EXTENDED_START} – {ens_end})")
    print(f"{'#'*70}")
    ext_perf_parts = []
    for label in all_labels:
        perf = compute_performance(port_df, label, date_start=EXTENDED_START)
        ext_perf_parts.append(perf)
        print_performance_table(perf, label, "Extended Period", f"{EXTENDED_START} – {ens_end}")
    ext_perf_df = pd.concat(ext_perf_parts, ignore_index=True)
    ext_perf_df.insert(0, "sub_period", "extended")

    out_path = os.path.join(OUTPUT_DIR, "ensemble_performance_extended.csv")
    ext_perf_df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
