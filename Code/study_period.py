import pandas as pd
import numpy as np

features = pd.read_csv('data/features.csv', parse_dates=['date'])
features = features.sort_values(['date', 'permno']).reset_index(drop=True)

# 论文 Section 4.2 参数：
#   原始每个 study period = 1000 天（750 训练 + 250 测试）
#   前 240 天因 R240 特征预热丢失，已在 feature_engineering.py 中以 dropna() 处理
#   → 有效训练天数 = 750 - 240 = 510
#   → 测试天数     = 250
#   → 共 23 个 study periods（December 1992 – October 2015）
TRAIN_DAYS = 510
TEST_DAYS  = 250

trading_days = (features['date']
                .drop_duplicates()
                .sort_values()
                .reset_index(drop=True))

# 划分所有Study Periods
study_periods = []
i = 0

while True:
    train_start_idx = i * TEST_DAYS
    train_end_idx = train_start_idx + TRAIN_DAYS - 1
    test_start_idx = train_end_idx + 1
    test_end_idx = test_start_idx + TEST_DAYS - 1

    # 超出范围就停止
    if test_end_idx >= len(trading_days):
        break

    study_periods.append({
        'period': i + 1,
        'train_start': trading_days[train_start_idx],
        'train_end': trading_days[train_end_idx],
        'test_start': trading_days[test_start_idx],
        'test_end': trading_days[test_end_idx],
    })
    i += 1

# 打印所有Study Periods
sp_df = pd.DataFrame(study_periods)
print(f"共划分 {len(sp_df)} 个Study Periods\n")
print(sp_df.to_string(index=False))

# 保存
sp_df.to_csv('data/study_periods.csv', index=False)
print("\n已保存 data/study_periods.csv")