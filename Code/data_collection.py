"""
IEOR 4733 - ML Statistical Arbitrage on S&P 500
Step 1: Data Collection from WRDS

Three steps:
    1. Pull historical S&P 500 constituent lists (crsp.msp500list)
    2. Pull daily price / return data (crsp.dsf)
    3. Merge and filter to get a clean daily universe

Output files:
    - constituents.csv   : raw membership records (permno, start, ending)
    - prices.csv         : raw daily returns for all ever-constituents
    - daily_universe.csv : clean merged dataset, one row per (date, permno)
                           containing only stocks that were IN the index that day
"""

import wrds
import pandas as pd
import numpy as np
import os

# ── Configuration ─────────────────────────────────────────────────────────────
START_DATE = "1990-01-01"
END_DATE   = "2025-12-31"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Pull Historical S&P 500 Constituent Lists
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Pulling S&P 500 constituent history...")
print("=" * 60)

# Connect to WRDS
conn = wrds.Connection()

# crsp.msp500list contains one row per stock per membership spell.
# 'start'  = date the stock entered the S&P 500
# 'ending' = date the stock left the S&P 500
# Includes stocks that were later delisted or removed — this is what
# eliminates survivorship bias.
constituents = conn.raw_sql("""
    SELECT permno, start, ending
    FROM crsp.msp500list
    WHERE start  <= '{end}'
    AND   ending >= '{start}'
    ORDER BY start, permno
""".format(start=START_DATE, end=END_DATE))

# Convert date columns
constituents["start"]  = pd.to_datetime(constituents["start"])
constituents["ending"] = pd.to_datetime(constituents["ending"])

print(f"  Total membership records : {len(constituents):,}")
print(f"  Unique stocks (permno)   : {constituents['permno'].nunique():,}")
print(f"  Date range               : {constituents['start'].min().date()} "
      f"to {constituents['ending'].max().date()}")

constituents.to_csv(f"{OUTPUT_DIR}/constituents.csv", index=False)
print(f"  Saved -> {OUTPUT_DIR}/constituents.csv\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Pull Daily Price / Return Data
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 2: Pulling daily return data from crsp.dsf...")
print("=" * 60)

# Get all unique permnos that ever appeared in the S&P 500
all_permnos = constituents["permno"].unique().tolist()
permno_str  = ",".join(map(str, all_permnos))

print(f"  Fetching data for {len(all_permnos):,} stocks...")
print("  (This may take a few minutes depending on your connection)")

# crsp.dsf columns used:
#   permno : unique stock identifier in CRSP
#   date   : trading date
#   ret    : daily total return (includes dividends), already adjusted
#            for splits and corporate actions
#   prc    : closing price (negative values mean bid/ask midpoint was used)
#   shrout : shares outstanding (thousands)
prices = conn.raw_sql("""
    SELECT permno, date, ret, prc, shrout
    FROM crsp.dsf
    WHERE permno IN ({permnos})
    AND   date BETWEEN '{start}' AND '{end}'
    ORDER BY permno, date
""".format(permnos=permno_str, start=START_DATE, end=END_DATE))

# Convert and clean
prices["date"] = pd.to_datetime(prices["date"])
prices["ret"]  = pd.to_numeric(prices["ret"],    errors="coerce")
prices["prc"]  = pd.to_numeric(prices["prc"],    errors="coerce").abs()
prices["shrout"] = pd.to_numeric(prices["shrout"], errors="coerce")

# Market cap (used later for sanity checks / weighting if needed)
prices["mktcap"] = prices["prc"] * prices["shrout"]

print(f"  Total rows pulled        : {len(prices):,}")
print(f"  Date range               : {prices['date'].min().date()} "
      f"to {prices['date'].max().date()}")
print(f"  Stocks with return data  : {prices['permno'].nunique():,}")
print(f"  Missing returns (NaN)    : {prices['ret'].isna().sum():,} "
      f"({prices['ret'].isna().mean()*100:.1f}%)")

prices.to_csv(f"{OUTPUT_DIR}/prices.csv", index=False)
print(f"  Saved -> {OUTPUT_DIR}/prices.csv\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Merge and Filter — Build the Clean Daily Universe
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 3: Building clean daily universe...")
print("=" * 60)

# For each row in prices, check whether that stock was actually
# IN the S&P 500 on that date.
#
# Approach: merge prices with constituents on permno, then keep only
# rows where the price date falls within [start, ending].
#
# This is the key operation that eliminates survivorship bias:
# even if we pulled price data for a stock, we only keep its data
# for the dates it was actually a constituent.

print("  Merging prices with constituent membership records...")

# Merge on permno (one price row can match multiple membership spells
# for the same stock if it left and re-entered the index)
merged = prices.merge(constituents, on="permno", how="left")

# Keep only rows where the trading date falls inside the membership spell
in_index = (merged["date"] >= merged["start"]) & (merged["date"] <= merged["ending"])
daily_universe = merged[in_index].copy()

# If a stock had multiple membership spells and a date matched more than
# one spell, drop duplicates (keep first match)
daily_universe = daily_universe.drop_duplicates(subset=["permno", "date"])

# Drop the membership date columns — no longer needed
daily_universe = daily_universe.drop(columns=["start", "ending"])

# Sort for readability and downstream feature engineering
daily_universe = daily_universe.sort_values(["date", "permno"]).reset_index(drop=True)

# ── Sanity checks ─────────────────────────────────────────────────────────────
print("\n  Sanity checks:")

# How many stocks per day on average?
stocks_per_day = daily_universe.groupby("date")["permno"].count()
print(f"  Avg stocks per trading day : {stocks_per_day.mean():.0f}")
print(f"  Min stocks on any day      : {stocks_per_day.min()} "
      f"({stocks_per_day.idxmin().date()})")
print(f"  Max stocks on any day      : {stocks_per_day.max()} "
      f"({stocks_per_day.idxmax().date()})")

# Missing returns after filtering
missing_ret = daily_universe["ret"].isna().sum()
total_rows  = len(daily_universe)
print(f"  Missing returns in universe: {missing_ret:,} "
      f"({missing_ret/total_rows*100:.1f}%)")

# Rows removed by the survivorship bias filter
rows_before = len(prices)
rows_after  = len(daily_universe)
removed     = rows_before - rows_after
print(f"\n  Rows before filtering      : {rows_before:,}")
print(f"  Rows after filtering       : {rows_after:,}")
print(f"  Rows removed (not in index): {removed:,} "
      f"({removed/rows_before*100:.1f}%)")

# ── Optional: drop rows with missing returns ──────────────────────────────────
# Rows with ret = NaN usually correspond to non-trading days, halted stocks,
# or data gaps. We drop them here; they will be excluded from feature
# calculation and model training automatically.
daily_universe = daily_universe.dropna(subset=["ret"])
print(f"\n  Final rows (ret not NaN)   : {len(daily_universe):,}")

# ── Save ──────────────────────────────────────────────────────────────────────
daily_universe.to_csv(f"{OUTPUT_DIR}/daily_universe.csv", index=False)
print(f"\n  Saved -> {OUTPUT_DIR}/daily_universe.csv")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DATA COLLECTION COMPLETE")
print("=" * 60)
print(f"  constituents.csv   : {len(constituents):,} membership records")
print(f"  prices.csv         : {len(prices):,} raw daily rows")
print(f"  daily_universe.csv : {len(daily_universe):,} clean daily rows")
print(f"\n  Columns in daily_universe:")
for col in daily_universe.columns:
    print(f"    - {col}")
print("\nNext step: Feature Engineering (calculate 31 lagged returns)")

conn.close()
