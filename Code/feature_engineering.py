"""
IEOR 4733 - ML Statistical Arbitrage on S&P 500
Step 2: Feature Engineering

What this script does:
    1. For each stock on each day, calculate 31 lagged return features (X)
    2. Generate the binary classification label (Y):
           Y = 1 if stock's next-day return > cross-sectional median
           Y = 0 otherwise

Input:
    - data/daily_universe.csv

Output:
    - data/features.csv : one row per (date, permno)
                          columns: permno, date, R1..R20, R40..R240, Y
"""

import pandas as pd
import numpy as np
import os
import sys

# ── Configuration ─────────────────────────────────────────────────────────────

# The 31 lookback periods from the paper
# First 20: daily resolution (past 1 to 20 trading days)
# Next 11:  monthly resolution (past 40, 60, ..., 240 trading days)
DAILY_LAGS   = list(range(1, 21))          # [1, 2, 3, ..., 20]
MONTHLY_LAGS = list(range(40, 241, 20))    # [40, 60, 80, ..., 240]
ALL_LAGS     = DAILY_LAGS + MONTHLY_LAGS   # 31 lags total

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading daily_universe.csv...")
print("=" * 60)

daily_universe = pd.read_csv(
    "data/daily_universe.csv",
    parse_dates=["date"]
)

print(f"  Rows loaded : {len(daily_universe):,}")
print(f"  Date range  : {daily_universe['date'].min().date()} "
      f"to {daily_universe['date'].max().date()}")
print(f"  Stocks      : {daily_universe['permno'].nunique():,}\n")

# Sort — critical for the shift operations below to work correctly
daily_universe = daily_universe.sort_values(["permno", "date"]).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Calculate 31 Lagged Return Features (X)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Calculating 31 lagged return features...")
print("=" * 60)

# For each lag m, the multi-period return is:
#     R(m) = price_today / price_(today - m) - 1
#
# Since we only have daily returns (ret), we can compute multi-period
# returns by compounding:
#     R(m) = product of (1 + ret) over the past m days - 1
#
# Implementation: use groupby + rolling to compute this efficiently.
# We group by permno so that returns from different stocks don't mix.

# First, compute the log return for efficient compounding
# log_ret = log(1 + ret), then sum over m days = log(cumulative return)
# cumulative return = exp(sum of log_rets) - 1

daily_universe["log_ret"] = np.log1p(daily_universe["ret"])

feature_list = []

grouped = daily_universe.groupby("permno")

print(f"  Computing features for {daily_universe['permno'].nunique():,} stocks...")
print("  (This may take a few minutes...)\n")

for lag in ALL_LAGS:
    col_name = f"R{lag}"

    # For each stock, sum the log returns over the past `lag` days.
    # At date t the rolling sum covers [t-lag+1 .. t], giving
    # R^s_{t,m} = P^s_t / P^s_{t-m} - 1  (paper eq. 1).
    # No extra shift: today's return is known at market close and is NOT
    # the thing we're predicting (Y is based on tomorrow's return).
    log_sum = grouped["log_ret"].transform(
        lambda x: x.rolling(window=lag, min_periods=lag).sum()
    )

    # Convert back from log return to simple return
    daily_universe[col_name] = np.expm1(log_sum)
    print(f"  Computed R{lag:>3d}  |  non-NaN rows: "
          f"{daily_universe[col_name].notna().sum():,}")

feature_cols = [f"R{lag}" for lag in ALL_LAGS]

print(f"\n  All 31 features computed.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Generate Binary Label Y
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Generating binary label Y...")
print("=" * 60)

# For each day, compute the cross-sectional median return of ALL stocks
# Y = 1 if stock's NEXT-DAY return > that day's cross-sectional median
# Y = 0 otherwise
#
# Important: we use next-day return (shift(-1) within each stock),
# and compare it against the next day's cross-sectional median.

# Next-day return for each stock
daily_universe["ret_next"] = grouped["ret"].transform(lambda x: x.shift(-1))

# Cross-sectional median of next-day returns (across all stocks, per day)
# We compute this on the next day's date to avoid lookahead
next_day_median = (
    daily_universe
    .groupby("date")["ret_next"]
    .median()
    .rename("median_ret_next")
)

daily_universe = daily_universe.merge(next_day_median, on="date", how="left")

# Binary label
daily_universe["Y"] = (
    daily_universe["ret_next"] > daily_universe["median_ret_next"]
).astype(int)

# Check label balance
y_counts = daily_universe["Y"].value_counts()
print(f"  Y = 1 (outperforms): {y_counts.get(1, 0):,} "
      f"({y_counts.get(1, 0)/len(daily_universe)*100:.1f}%)")
print(f"  Y = 0 (underperforms): {y_counts.get(0, 0):,} "
      f"({y_counts.get(0, 0)/len(daily_universe)*100:.1f}%)")
print("  (Should be close to 50/50 by construction)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Assemble Final Feature Dataset
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Assembling final feature dataset...")
print("=" * 60)

# Keep only the columns we need
keep_cols = ["permno", "date"] + feature_cols + ["Y"]
features = daily_universe[keep_cols].copy()

# Drop rows where ANY feature is NaN
# This happens at the start of each stock's history where we don't have
# enough past data to compute the longest lag (R240 = 240 trading days)
rows_before = len(features)
features = features.dropna()
rows_after  = len(features)
dropped     = rows_before - rows_after

print(f"  Rows before dropping NaN : {rows_before:,}")
print(f"  Rows dropped (NaN)       : {dropped:,}  "
      f"(expected — first ~240 days per stock have incomplete history)")
print(f"  Rows after dropping NaN  : {rows_after:,}")

# Final sort
features = features.sort_values(["date", "permno"]).reset_index(drop=True)

# ── Sanity checks ─────────────────────────────────────────────────────────────
print("\n  Sanity checks:")
print(f"  Date range  : {features['date'].min().date()} "
      f"to {features['date'].max().date()}")
print(f"  Stocks      : {features['permno'].nunique():,}")
print(f"  Avg stocks per day: "
      f"{features.groupby('date')['permno'].count().mean():.0f}")
print(f"  Missing values: {features.isnull().sum().sum()}")

# Preview
print("\n  First 3 rows (selected columns):")
preview_cols = ["permno", "date", "R1", "R5", "R20", "R240", "Y"]
print(features[preview_cols].head(3).to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────
features.to_csv(f"{OUTPUT_DIR}/features.csv", index=False)
print(f"\n  Saved -> {OUTPUT_DIR}/features.csv")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)
print(f"  Output : data/features.csv")
print(f"  Shape  : {features.shape[0]:,} rows x {features.shape[1]} columns")
print(f"  Columns: permno, date, R1-R20, R40-R240 (31 features), Y")
print("\nNext step: Train/Test Split and Model Training")
