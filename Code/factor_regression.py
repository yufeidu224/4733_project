"""
IEOR 4733 - ML Statistical Arbitrage on S&P 500
Factor Regression — Table 4 Replication (Krauss et al. 2017)

对 ENS1 k=10 after-TC 收益率做四个回归：
  FF3    : Mkt-RF, SMB, HML
  FF3+2  : Mkt-RF, SMB, HML, Momentum, Reversal
  FF5    : Mkt-RF, SMB5, HML5, RMW5, CMA5
  FF_VIX : Mkt-RF, SMB, HML, Momentum, Reversal, VIX_dummy

回归期间：1992-12-17 – 2015-10-15（论文复现期）
输出：
  data/factor_regression_results.csv   — 系数 + SE + 显著性标记
  data/factor_regression_summary.csv   — R², Adj.R², N, RMSE
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ── 配置 ────────────────────────────────────────────────────────────────────────
# 路径相对于本脚本所在目录（Code/），data 在 ../data/
DATA        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
START       = "1992-12-17"
END         = "2015-10-15"
TC          = 0.002
VIX_THRESH  = 30.0          # 论文：VIX dummy = 1 if VIX > 30 (90th pct)

OUT_COEF    = f"{DATA}/factor_regression_results.csv"
OUT_SUMM    = f"{DATA}/factor_regression_summary.csv"


# ══════════════════════════════════════════════════════════════════════════════
# 1. 加载并整理因子数据
# ══════════════════════════════════════════════════════════════════════════════

def load_ff(filename, cols_rename=None):
    df = pd.read_excel(f"{DATA}/{filename}.xlsx")
    df.columns = df.columns.str.strip()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    # 因子值为百分比形式，转换为小数
    for c in df.columns[1:]:
        df[c] = df[c] / 100.0
    if cols_rename:
        df = df.rename(columns=cols_rename)
    return df


print("加载因子数据...")
ff3  = load_ff("FF3",  {"Mkt-RF": "MktRF", "SMB": "SMB",  "HML": "HML",  "RF": "RF"})
ff5  = load_ff("FF5",  {"Mkt-RF": "MktRF", "SMB": "SMB5", "HML": "HML5",
                         "RMW": "RMW5", "CMA": "CMA5", "RF": "RF5"})
mom  = load_ff("FF_Momentum_Factor", {"Mom": "Mom"})
rev  = load_ff("FF_Reversal_Factor", {"ST_Rev": "Rev"})

vix  = pd.read_csv(f"{DATA}/vix.csv", parse_dates=["date"])
vix["VIX_dummy"] = (vix["VIX"] > VIX_THRESH).astype(float)

# ── 合并所有因子 ────────────────────────────────────────────────────────────────
factors = (ff3[["date", "MktRF", "SMB", "HML", "RF"]]
           .merge(ff5[["date", "SMB5", "HML5", "RMW5", "CMA5"]], on="date", how="left")
           .merge(mom[["date", "Mom"]],          on="date", how="left")
           .merge(rev[["date", "Rev"]],           on="date", how="left")
           .merge(vix[["date", "VIX", "VIX_dummy"]], on="date", how="left"))


# ══════════════════════════════════════════════════════════════════════════════
# 2. 加载 ENS1 k=10 after-TC 收益率
# ══════════════════════════════════════════════════════════════════════════════

print("加载 ENS1 收益率...")
ens_all = pd.read_csv(f"{DATA}/ensemble_portfolio_returns.csv", parse_dates=["date"])
ens1 = (ens_all[(ens_all["ensemble"] == "ENS1") & (ens_all["k"] == 10)]
        .assign(net_ret=lambda d: d["portfolio_ret"] - TC)
        [["date", "net_ret"]]
        .reset_index(drop=True))

# ── 合并 ────────────────────────────────────────────────────────────────────────
df = (ens1
      .merge(factors, on="date", how="left")
      .query(f"date >= '{START}' and date <= '{END}'")
      .dropna()
      .reset_index(drop=True))

# 超额收益 = net_ret - RF（无风险利率）
df["excess_ret"] = df["net_ret"] - df["RF"]

print(f"  合并后样本：{len(df)} 天 ({df['date'].min().date()} – {df['date'].max().date()})")
print(f"  VIX > {VIX_THRESH} 的天数：{df['VIX_dummy'].sum():.0f} "
      f"({df['VIX_dummy'].mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 回归函数
# ══════════════════════════════════════════════════════════════════════════════

def run_ols(y: pd.Series, X_df: pd.DataFrame, model_name: str) -> dict:
    """OLS 回归，返回系数、SE、显著性标记及汇总统计。"""
    X = sm.add_constant(X_df, prepend=True)
    X.columns = ["Intercept"] + list(X_df.columns)
    res = sm.OLS(y, X).fit(cov_type="HC3")   # 异方差稳健 SE

    def sig(p):
        if p < 0.001: return "***"
        elif p < 0.01:  return "**"
        elif p < 0.05:  return "*"
        return ""

    coef_rows = []
    for var in X.columns:
        coef_rows.append({
            "model":    model_name,
            "variable": var,
            "coef":     round(res.params[var], 4),
            "se":       round(res.bse[var], 4),
            "t":        round(res.tvalues[var], 4),
            "pval":     round(res.pvalues[var], 4),
            "sig":      sig(res.pvalues[var]),
        })

    summary = {
        "model":   model_name,
        "R2":      round(res.rsquared, 4),
        "Adj_R2":  round(res.rsquared_adj, 4),
        "N":       int(res.nobs),
        "RMSE":    round(np.sqrt(res.mse_resid), 4),
    }
    return {"coef": coef_rows, "summary": summary, "result": res}


# ══════════════════════════════════════════════════════════════════════════════
# 4. 四个回归
# ══════════════════════════════════════════════════════════════════════════════

y = df["excess_ret"]

models = {
    "FF3":    df[["MktRF", "SMB", "HML"]],
    "FF3+2":  df[["MktRF", "SMB", "HML", "Mom", "Rev"]],
    "FF5":    df[["MktRF", "SMB5", "HML5", "RMW5", "CMA5"]],
    "FF_VIX": df[["MktRF", "SMB", "HML", "Mom", "Rev", "VIX_dummy"]],
}

all_coef    = []
all_summary = []
results     = {}

for name, X_df in models.items():
    out = run_ols(y, X_df, name)
    all_coef.extend(out["coef"])
    all_summary.append(out["summary"])
    results[name] = out["result"]


# ══════════════════════════════════════════════════════════════════════════════
# 5. 保存 CSV
# ══════════════════════════════════════════════════════════════════════════════

coef_df = pd.DataFrame(all_coef)
summ_df = pd.DataFrame(all_summary)

coef_df.to_csv(OUT_COEF,  index=False)
summ_df.to_csv(OUT_SUMM,  index=False)
print(f"\n已保存 → {OUT_COEF}")
print(f"已保存 → {OUT_SUMM}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. 打印 Table 4 风格汇总
# ══════════════════════════════════════════════════════════════════════════════

# 所有出现过的变量，按顺序
VAR_ORDER = ["Intercept", "MktRF", "SMB", "HML", "Mom", "Rev",
             "SMB5", "HML5", "RMW5", "CMA5", "VIX_dummy"]
MODEL_ORDER = ["FF3", "FF3+2", "FF5", "FF_VIX"]

# 转成 pivot 方便打印
pivot = coef_df.pivot(index="variable", columns="model", values=["coef", "se", "sig"])

print("\n" + "=" * 75)
print("  Table 4 — Factor Regression: ENS1 k=10 After TC (1992-12-17 – 2015-10-15)")
print("=" * 75)
print(f"  {'':20s}", end="")
for m in MODEL_ORDER:
    print(f"  {m:>12s}", end="")
print()
print("  " + "-" * 70)

for var in VAR_ORDER:
    if var not in pivot.index:
        continue
    # 系数行
    coef_str = f"  {var:20s}"
    for m in MODEL_ORDER:
        try:
            c = pivot.loc[var, ("coef", m)]
            s = pivot.loc[var, ("sig",  m)]
            if pd.isna(c):
                coef_str += f"  {'':>12s}"
            else:
                s = "" if pd.isna(s) else str(s)
                coef_str += f"  {c:>8.4f}{s:<3s} "
        except KeyError:
            coef_str += f"  {'':>12s}"
    print(coef_str)
    # SE 行
    se_str = f"  {'':20s}"
    for m in MODEL_ORDER:
        try:
            se = pivot.loc[var, ("se", m)]
            if pd.isna(se):
                se_str += f"  {'':>12s}"
            else:
                se_str += f"  ({se:.4f})    "
        except KeyError:
            se_str += f"  {'':>12s}"
    print(se_str)

print("  " + "-" * 70)
print(f"  {'R2':20s}", end="")
for m in MODEL_ORDER:
    r = summ_df.loc[summ_df["model"] == m, "R2"].values[0]
    print(f"  {r:>12.4f}", end="")
print()
print(f"  {'Adj. R2':20s}", end="")
for m in MODEL_ORDER:
    r = summ_df.loc[summ_df["model"] == m, "Adj_R2"].values[0]
    print(f"  {r:>12.4f}", end="")
print()
print(f"  {'N':20s}", end="")
for m in MODEL_ORDER:
    n = summ_df.loc[summ_df["model"] == m, "N"].values[0]
    print(f"  {n:>12d}", end="")
print()
print(f"  {'RMSE':20s}", end="")
for m in MODEL_ORDER:
    r = summ_df.loc[summ_df["model"] == m, "RMSE"].values[0]
    print(f"  {r:>12.4f}", end="")
print()
print("  *** p<0.001  ** p<0.01  * p<0.05   (HC3 robust SE)")
