"""
IEOR 4733 - ML Statistical Arbitrage on S&P 500
Step 3: Gradient-Boosted Trees (GBT) Model Training

论文参数 (Krauss et al., 2017):
    - n_estimators  : 100   (树的数量)
    - max_depth     : 3     (树的深度，允许二阶交互效应)
    - learning_rate : 0.1   (学习率)
    - colsample     : 15/31 (每次分裂随机选取约一半特征)
    - 目标           : 预测每只股票次日收益率跑赢截面中位数的概率

Input:
    - data/features.csv      : 特征数据 (permno, date, R1-R240, Y)
    - data/study_periods.csv : 29个Study Period的时间划分

Output:
    - data/gbt_predictions.csv : 每个交易期每只股票的预测概率
                                 列: date, permno, prob_GBT
    - data/gbt_performance.csv : 每个Period的策略绩效指标
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier

# ── Configuration ──────────────────────────────────────────────────────────────
# 论文中的31个特征列名
DAILY_LAGS   = list(range(1, 21))        # R1 - R20
MONTHLY_LAGS = list(range(40, 241, 20))  # R40, R60, ..., R240
ALL_LAGS     = DAILY_LAGS + MONTHLY_LAGS
FEATURE_COLS = [f"R{lag}" for lag in ALL_LAGS]

# 论文GBT超参数
GBT_PARAMS = {
    "n_estimators"      : 100,       # 树的数量 (MGBT)
    "max_depth"         : 3,         # 树的深度 (JGBT)，允许二阶交互
    "learning_rate"     : 0.1,       # 学习率 (λGBT)
    "colsample_bytree"  : 15/31,     # 每棵树随机选取特征比例 (mGBT = 15)
    "colsample_bylevel" : 1.0,
    "subsample"         : 1.0,       # 不做行采样，与论文一致
    "objective"         : "binary:logistic",
    "eval_metric"       : "auc",
    "use_label_encoder" : False,
    "random_state"      : 1,         # 论文seed=1
    "n_jobs"            : -1,
}

# 交易参数
K = 10           # 做多top-k，做空bottom-k
TC = 0.0005      # 单边交易成本 0.05% (论文设定)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data...")
print("=" * 60)

features = pd.read_csv("data/features.csv", parse_dates=["date"])
study_periods = pd.read_csv("data/study_periods.csv", parse_dates=[
    "train_start", "train_end", "test_start", "test_end"
])

features = features.sort_values(["date", "permno"]).reset_index(drop=True)

print(f"  Features shape    : {features.shape}")
print(f"  Study periods     : {len(study_periods)}")
print(f"  Feature columns   : {len(FEATURE_COLS)} ({FEATURE_COLS[0]} ... {FEATURE_COLS[-1]})")

# ══════════════════════════════════════════════════════════════════════════════
# 主循环：对每个Study Period训练GBT并生成预测
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Training GBT models across all Study Periods...")
print("=" * 60)

all_predictions = []   # 汇总所有交易期的预测结果
performance_records = []  # 每个Period的绩效指标
all_daily_returns = []    # 汇总所有交易期的逐日收益（用于计算总体指标）

for _, period in study_periods.iterrows():
    pid          = int(period["period"])
    train_start  = period["train_start"]
    train_end    = period["train_end"]
    test_start   = period["test_start"]
    test_end     = period["test_end"]

    print(f"\nPeriod {pid:>2d} | "
          f"Train: {train_start.date()} → {train_end.date()} | "
          f"Test : {test_start.date()} → {test_end.date()}")

    # ── 1. 切分训练集和测试集 ────────────────────────────────────────────────
    train_mask = (features["date"] >= train_start) & (features["date"] <= train_end)
    test_mask  = (features["date"] >= test_start)  & (features["date"] <= test_end)

    train_df = features[train_mask].copy()
    test_df  = features[test_mask].copy()

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["Y"].values
    X_test  = test_df[FEATURE_COLS].values

    print(f"         Train rows: {len(train_df):,} | "
          f"Test rows: {len(test_df):,} | "
          f"Train stocks: {train_df['permno'].nunique()}")

    # ── 2. 训练GBT模型 ───────────────────────────────────────────────────────
    model = XGBClassifier(**GBT_PARAMS)
    model.fit(X_train, y_train)

    # 训练集AUC（用于后续ENS2/ENS3集成时的权重计算）
    from sklearn.metrics import roc_auc_score
    train_prob = model.predict_proba(X_train)[:, 1]
    train_auc  = roc_auc_score(y_train, train_prob)
    train_gini = 2 * (train_auc - 0.5)
    print(f"         Train AUC : {train_auc:.4f} | Train Gini: {train_gini:.4f}")

    # ── 3. 生成测试集预测概率 ────────────────────────────────────────────────
    test_prob = model.predict_proba(X_test)[:, 1]

    pred_df = test_df[["date", "permno"]].copy()
    pred_df["prob_GBT"]   = test_prob
    pred_df["period"]     = pid
    pred_df["train_gini"] = train_gini

    all_predictions.append(pred_df)

    # ── 4. 每日排名并构建 long-short 组合 ────────────────────────────────────
    # 按预测概率排名，top-k做多，bottom-k做空
    daily_returns = []

    for date, day_df in test_df.assign(prob_GBT=test_prob).groupby("date"):
        n_stocks = len(day_df)
        if n_stocks < 2 * K:
            # 股票数不足时跳过
            continue

        day_sorted = day_df.sort_values("prob_GBT", ascending=False)

        long_ret  = day_sorted.iloc[:K]["ret_next"].mean()   if "ret_next" in day_df.columns else np.nan
        short_ret = day_sorted.iloc[-K:]["ret_next"].mean()  if "ret_next" in day_df.columns else np.nan

        # 如果features.csv里没有ret_next，需要用Y近似（此处做保护）
        if "ret_next" not in test_df.columns:
            break

        raw_return = long_ret - short_ret
        # 交易成本：每次开仓和平仓各收一次，共 4 * TC（多头开/平 + 空头开/平）
        tc_cost    = 4 * TC
        net_return = raw_return - tc_cost

        daily_returns.append({
            "date"       : date,
            "period"     : pid,
            "long_ret"   : long_ret,
            "short_ret"  : short_ret,
            "raw_return" : raw_return,
            "net_return" : net_return,
        })

    if daily_returns:
        period_ret_df = pd.DataFrame(daily_returns)

        # 每个Period内的统计（Period级别）
        mean_raw = period_ret_df["raw_return"].mean()
        mean_net = period_ret_df["net_return"].mean()
        sharpe   = (period_ret_df["net_return"].mean() /
                    period_ret_df["net_return"].std() * np.sqrt(252)
                    if period_ret_df["net_return"].std() > 0 else np.nan)

        print(f"         Mean raw return/day: {mean_raw*100:.4f}% | "
              f"Net: {mean_net*100:.4f}% | "
              f"Ann. Sharpe: {sharpe:.2f}")

        performance_records.append({
            "period"          : pid,
            "test_start"      : test_start,
            "test_end"        : test_end,
            "mean_raw_return" : mean_raw,
            "mean_net_return" : mean_net,
            "std_net_return"  : period_ret_df["net_return"].std(),
            "sharpe_ratio"    : sharpe,
            "train_gini"      : train_gini,
            "n_trading_days"  : len(period_ret_df),
        })

        # 把逐日收益追加到总列表（用于最终整体统计）
        all_daily_returns.append(period_ret_df)

# ══════════════════════════════════════════════════════════════════════════════
# 保存结果
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Saving results...")
print("=" * 60)

# 预测概率
predictions_df = pd.concat(all_predictions, ignore_index=True)
predictions_df = predictions_df.sort_values(["date", "permno"]).reset_index(drop=True)
predictions_df.to_csv(f"{OUTPUT_DIR}/gbt_predictions.csv", index=False)
print(f"  Saved -> {OUTPUT_DIR}/gbt_predictions.csv")
print(f"  Shape : {predictions_df.shape}")

# 绩效指标（Period级别）
if performance_records:
    perf_df = pd.DataFrame(performance_records)
    perf_df.to_csv(f"{OUTPUT_DIR}/gbt_performance.csv", index=False)
    print(f"  Saved -> {OUTPUT_DIR}/gbt_performance.csv")

# 逐日收益（完整序列）
if all_daily_returns:
    daily_df = pd.concat(all_daily_returns, ignore_index=True)
    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    daily_df.to_csv(f"{OUTPUT_DIR}/gbt_daily_returns.csv", index=False)
    print(f"  Saved -> {OUTPUT_DIR}/gbt_daily_returns.csv")
    print(f"  Shape : {daily_df.shape}")

    # ── 用完整逐日序列计算总体指标（与论文对齐）────────────────────────────
    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE SUMMARY (GBT, k=10)")
    print("  [基于完整逐日收益序列，与论文计算方式一致]")
    print("=" * 60)

    # 全部Period
    raw_all  = daily_df["raw_return"].mean()
    net_all  = daily_df["net_return"].mean()
    sharpe_all = (daily_df["net_return"].mean() /
                  daily_df["net_return"].std() * np.sqrt(252))
    print(f"  全部Period (1995-2024):")
    print(f"    Mean raw return/day : {raw_all*100:.4f}%")
    print(f"    Mean net return/day : {net_all*100:.4f}%")
    print(f"    Ann. Sharpe ratio   : {sharpe_all:.2f}")

    # 仅论文区间（测试期 <= 2015-10）
    paper_dates = perf_df[perf_df["test_start"] <= "2015-10-31"]["test_start"]
    paper_periods = perf_df[perf_df["test_start"] <= "2015-10-31"]["period"].tolist()
    daily_paper = daily_df[daily_df["period"].isin(paper_periods)]

    raw_paper   = daily_paper["raw_return"].mean()
    net_paper   = daily_paper["net_return"].mean()
    sharpe_paper = (daily_paper["net_return"].mean() /
                    daily_paper["net_return"].std() * np.sqrt(252))
    print(f"\n  论文区间 (1995-2015, Period 1-21):")
    print(f"    Mean raw return/day : {raw_paper*100:.4f}%")
    print(f"    Mean net return/day : {net_paper*100:.4f}%")
    print(f"    Ann. Sharpe ratio   : {sharpe_paper:.2f}")
    print(f"\n  论文基准 (GBT, k=10):")
    print(f"    Raw return/day      : 0.37%")
    print(f"    Net return/day      : 0.17%")
    print(f"    Ann. Sharpe         : 1.23")

print("\n" + "=" * 60)
print("GBT TRAINING COMPLETE")
print("=" * 60)
print(f"  gbt_predictions.csv : {len(predictions_df):,} rows")
print(f"  Columns             : date, permno, prob_GBT, period, train_gini")
print("\nNext step: Ensemble construction (ENS1/ENS2/ENS3)")
