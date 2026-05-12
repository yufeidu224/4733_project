# ML Statistical Arbitrage on the S&P 500

Replication and extension of **Krauss, Do, and Huck (2017)** —
*Deep Neural Networks, Gradient-Boosted Trees, Random Forests: Statistical Arbitrage on the S&P 500*

**IEOR 4733 Algorithmic Trading · Columbia University**
Authors: An Wang · Bingjing Hong · Yufei Du

---

## Overview

This project implements a full machine-learning statistical arbitrage pipeline on S&P 500 constituents (1992–2024). Three base models — Deep Neural Networks (DNN), Gradient-Boosted Trees (GBT), and Random Forests (RAF) — are trained in a rolling walk-forward framework and combined into ensemble forecasts (ENS1/2/3). Performance is evaluated across two sub-periods:

| Period | Dates | Description |
|--------|-------|-------------|
| **Paper period** | Dec 1992 – Dec 2015 | Replication of Krauss et al. (2017) |
| **Extended period** | Jan 2016 – Sep 2024 | Out-of-sample extension |

---

## Project Structure

```
Projects/
├── config.yaml                  ← all parameters (TC, k values, date ranges, paths)
├── main.py                      ← pipeline entry point (CLI)
├── dashboard.py                 ← Streamlit performance dashboard
├── requirements.txt
│
├── Code/                        ← computation scripts
│   ├── data_collection.py       ← Step 1: pull CRSP data from WRDS
│   ├── feature_engineering.py   ← Step 2: 31 lagged return features + Y label
│   ├── study_period.py          ← Step 3: define 32 rolling walk-forward periods
│   ├── dnn_training.py          ← Step 4a: DNN model (PyTorch)
│   ├── gbt_model.py             ← Step 4b: GBT model (XGBoost)
│   ├── random_forest.ipynb      ← Step 4c: RAF model (scikit-learn, run in Jupyter)
│   ├── ensemble.py              ← Step 5: ENS1/2/3 ensemble + backtest
│   ├── factor_regression.py     ← Step 6: FF3/5 factor regression (Table 4)
│   ├── dnn_training_variants.py ← hyperparameter sensitivity (Table 3)
│   ├── sub_periods_fig.ipynb    ← regime sensitivity figures (Fig 3/4)
│   └── (other notebooks)
│
├── data/                        ← all CSV / Excel data files
│   ├── daily_universe.csv       ← S&P 500 constituents + daily returns
│   ├── features.csv             ← engineered features (R1–R240, Y)
│   ├── study_periods.csv        ← 32 period definitions
│   ├── dnn_predictions.csv      ← DNN probability forecasts
│   ├── gbt_predictions.csv      ← GBT probability forecasts
│   ├── rf_predictions.csv       ← RAF probability forecasts
│   ├── ensemble_predictions.csv ← merged forecasts + ENS1/2/3 scores
│   ├── ensemble_portfolio_returns.csv  ← daily P&L for all models & k values
│   ├── ensemble_performance_paper.csv  ← performance metrics, paper period
│   ├── ensemble_performance_extended.csv ← performance metrics, extended period
│   ├── factor_regression_results.csv   ← FF3/5 regression coefficients
│   ├── VIXCLS.csv               ← VIX index (for regime analysis)
│   ├── crsp_market_index.csv    ← CRSP value/equal-weighted market returns
│   └── FF3.xlsx, FF5.xlsx, ...  ← Fama-French factor data
│
├── figures/                     ← generated plots (from sub_periods_fig.ipynb)
│   ├── fig2a_paper.{png,pdf}    ← cumulative returns, paper period
│   ├── fig2b_extended.{png,pdf} ← cumulative returns, extended period
│   ├── regime_paper.{png,pdf}   ← regime sensitivity, paper period
│   └── regime_extended.{png,pdf}← regime sensitivity, extended period
│
└── statarb_dashboard/           ← standalone dashboard (self-contained, alternative UI)
    ├── dashboard.py
    ├── requirements.txt
    └── *.csv                    ← local data copies for this dashboard
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For PyTorch with GPU support (CUDA 12.x):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Check pipeline status

```bash
python main.py --list
```

### 3. Run a specific step

```bash
# Re-run ensemble only (all other outputs already exist)
python main.py --steps ensemble

# Force re-run GBT and ensemble
python main.py --force --steps gbt ensemble
```

### 4. Run full pipeline

```bash
python main.py
```

> **Note:** Step 1 (`data`) requires a WRDS institutional account.  
> Step 4c (`rf`) is a Jupyter notebook — open `Code/random_forest.ipynb` and run manually.  
> All other outputs are already pre-computed in `data/`.

### 5. Launch dashboard

```bash
python main.py --dashboard
# or
streamlit run dashboard.py
```

---

## Pipeline Steps

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Data | `data_collection.py` | WRDS/CRSP | `daily_universe.csv` |
| 2. Features | `feature_engineering.py` | `daily_universe.csv` | `features.csv` |
| 3. Periods | `study_period.py` | `daily_universe.csv` | `study_periods.csv` |
| 4a. DNN | `dnn_training.py` | `features.csv`, `study_periods.csv` | `dnn_predictions.csv` |
| 4b. GBT | `gbt_model.py` | `features.csv`, `study_periods.csv` | `gbt_predictions.csv` |
| 4c. RAF | `random_forest.ipynb` | `features.csv`, `study_periods.csv` | `rf_predictions.csv` |
| 5. Ensemble | `ensemble.py` | predictions + `daily_universe.csv` | `ensemble_*.csv` |
| 6. Factors | `factor_regression.py` | ensemble returns + FF factors | `factor_regression_*.csv` |

---

## Key Parameters (`config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backtest.k_values` | `[10, 50, 100, 150, 200]` | Portfolio sizes (long + short legs) |
| `backtest.tc_bps` | `5` | Transaction cost per half-turn (bps) |
| `periods.paper_end` | `2015-12-31` | Paper period end (Table 1 in report) |
| `periods.extended_start` | `2016-01-01` | Extension period start (Table 5 in report) |
| `periods.krauss_original_end` | `2015-10-14` | Original Krauss 2017 period 23 end |
| `dashboard.default_k` | `10` | Default portfolio size shown in dashboard |
| `dashboard.show_net` | `true` | Show TC-adjusted returns by default |

To change any parameter, edit `config.yaml` and re-run the relevant pipeline step.

---

## Methodology

### Feature Construction
31 lagged return features per stock per day:
- Short-term: R1–R20 (1 to 20 trading days)
- Long-term: R40, R60, ..., R240 (in 20-day steps)

Binary label: `Y = 1` if next-day return > cross-sectional median, else `Y = 0`.

### Walk-Forward Validation
- **Training window:** 750 trading days (~3 years)
- **Test window:** 250 trading days (~1 year)
- **Total periods:** 32 rolling windows (1990–2024)

### Portfolio Construction
Each day, rank all S&P 500 stocks by predicted outperformance probability:
- **Long:** top-k stocks (equal-weighted)
- **Short:** bottom-k stocks (equal-weighted)
- **Dollar-neutral:** long and short legs cancel

### Transaction Costs
Following Krauss et al. (2017): 5 bps per half-turn.
With daily full rebalancing: **20 bps per day** total (4 half-turns).

### Ensemble Methods
| Method | Formula | Description |
|--------|---------|-------------|
| ENS1 | `(1/3)(DNN + GBT + RAF)` | Equal-weighted average |
| ENS2 | `Σ wᵢ·pᵢ, wᵢ = gᵢ/Σgⱼ` | Gini-weighted (training AUC) |
| ENS3 | `Σ wᵢ·pᵢ, wᵢ = (1/Rᵢ)/Σ(1/Rⱼ)` | Rank-weighted (more robust to outliers) |

---

## Key Results

### Paper Period (Dec 1992 – Dec 2015), k=10, before TC

| Model | Mean Daily Ret | Sharpe |
|-------|---------------|--------|
| DNN | 0.26% | — |
| GBT | 0.37% | — |
| RAF | 0.28% | — |
| ENS1 | 0.27% | 2.40 |

### Performance Decay (Extended Period, Jan 2016 – Sep 2024)
After transaction costs, all models generate negative or near-zero net Sharpe ratios in the extension period, consistent with increased algorithmic competition and market efficiency.

---

## Dashboard

The Streamlit dashboard (`dashboard.py`) provides:

| Tab | Content |
|-----|---------|
| Overview | Sharpe ratio bar chart (paper vs. extended period) |
| Cumulative Returns | Equity curves + rolling 1-year Sharpe |
| Performance Tables | Full metrics for all models and k values |
| Factor Regression | FF3/FF5 alpha table (Table 4 replication) |
| Risk Diagnostics | Drawdown chart, return distribution, VaR/CVaR |

Sidebar controls: portfolio size `k`, TC toggle, TC bps input.

---

## References

Krauss, C., Do, X. A., & Huck, N. (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. *European Journal of Operational Research*, 259(2), 689–702.

Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
