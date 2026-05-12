"""
IEOR 4733 - ML Statistical Arbitrage on S&P 500
Step 3 (DNN): Deep Neural Network Training and Prediction

Replicates Krauss et al. (2017) Section 4.3.1

Architecture: 31-31-10-5-2
    - Maxout activation (2 channels per neuron, paper eq. 3)
    - Input dropout : 10%
    - Hidden dropout: 50%
    - L1 regularization: lambda = 1e-5
    - Optimizer: ADADELTA (Zeiler 2012)
    - Max epochs: 400, early stopping on 5-epoch moving-avg training loss
    - Loss: cross-entropy (paper eq. 4)
    - Seed: 1, deterministic (single-threaded)

Parameter count: 2746  (matches paper footnote 3)

Outputs:
    data/dnn_predictions.csv     : (period, date, permno, prob_outperform)
    data/dnn_portfolio_returns.csv : daily long-short returns for k in {10,50,100,150,200}
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ── Configuration ──────────────────────────────────────────────────────────────
SEED          = 1
BATCH_SIZE    = 2048
MAX_EPOCHS    = 400
L1_LAMBDA     = 1e-5
INPUT_DROP    = 0.1
HIDDEN_DROP   = 0.5
K_VALUES      = [10, 50, 100, 150, 200]
OUTPUT_DIR    = "data"

# Reproducibility — paper: "run all calculations on a single core"
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda")         # RTX 5090 requires PyTorch nightly+cu128


# ══════════════════════════════════════════════════════════════════════════════
# Model Definition
# ══════════════════════════════════════════════════════════════════════════════

class MaxoutLayer(nn.Module):
    """
    Maxout unit (Goodfellow et al., 2013):
        f(alpha1, alpha2) = max(alpha1, alpha2)
    Each output neuron has `num_pieces` independent linear projections;
    the neuron fires the maximum.

    Linear weight shape: (in_features) -> (out_features * num_pieces)
    then reshape to (batch, out_features, num_pieces) and take max over pieces.
    """
    def __init__(self, in_features: int, out_features: int, num_pieces: int = 2):
        super().__init__()
        self.num_pieces  = num_pieces
        self.out_features = out_features
        # Single Linear covers both channels (matches H2O maxout implementation)
        self.linear = nn.Linear(in_features, out_features * num_pieces)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)                              # (batch, out * pieces)
        z = z.view(-1, self.out_features, self.num_pieces)  # (batch, out, pieces)
        return z.max(dim=2).values                      # (batch, out)


class DNN(nn.Module):
    """
    31 -> 31(maxout) -> 10(maxout) -> 5(maxout) -> 2(softmax)

    Parameter count verification (paper footnote 3):
        I->H1 : 2*31*31 + 2*31 = 1922 + 62  = 1984
        H1->H2: 2*10*31 + 2*10 =  620 + 20  =  640
        H2->H3: 2* 5*10 + 2* 5 =  100 + 10  =  110
        H3->O :   2* 5  +   2  =   10 +  2  =   12
        Total  = 2746  ✓
    """
    def __init__(
        self,
        input_dim:    int   = 31,
        hidden_dims:  list  = [31, 10, 5],
        output_dim:   int   = 2,
        input_drop:   float = INPUT_DROP,
        hidden_drop:  float = HIDDEN_DROP,
    ):
        super().__init__()
        self.input_dropout = nn.Dropout(p=input_drop)

        blocks = []
        in_dim = input_dim
        for h in hidden_dims:
            blocks.append(MaxoutLayer(in_dim, h, num_pieces=2))
            blocks.append(nn.Dropout(p=hidden_drop))
            in_dim = h
        self.hidden = nn.Sequential(*blocks)

        # Output layer: plain linear + softmax (no maxout, paper §4.3.1)
        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        x = self.hidden(x)
        x = self.output_layer(x)
        return torch.softmax(x, dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_dnn(X_train: np.ndarray, y_train: np.ndarray, period_id: int) -> DNN:
    """
    Train one DNN for a single study period.

    Early stopping (paper §4.3.1):
        Track 5-epoch moving average of training loss.
        If it fails to improve for 5 consecutive measurements, stop.
    """
    torch.manual_seed(SEED)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    g = torch.Generator()
    g.manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
    )

    model     = DNN().to(DEVICE)
    optimizer = optim.Adadelta(model.parameters())   # default rho=0.9, eps=1e-6
    criterion = nn.CrossEntropyLoss()

    # Collect all named parameters for L1
    all_params = [p for p in model.parameters()]

    loss_history    = []
    best_moving_avg = float("inf")
    no_improve      = 0
    t0              = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()

            preds = model(X_b)
            ce    = criterion(preds, y_b)

            # L1 regularization on all weights (paper: lambda_DNN = 1e-5)
            l1 = sum(p.abs().sum() for p in all_params)
            loss = ce + L1_LAMBDA * l1

            loss.backward()
            optimizer.step()

            epoch_loss += ce.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        # Early stopping check (start after 5 epochs)
        if len(loss_history) >= 5:
            moving_avg = float(np.mean(loss_history[-5:]))
            if moving_avg < best_moving_avg - 1e-7:
                best_moving_avg = moving_avg
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= 5:
                print(f"    [Period {period_id}] Early stop @ epoch {epoch} "
                      f"| loss_ma={moving_avg:.5f} | "
                      f"{time.time()-t0:.0f}s elapsed")
                break

        if epoch % 50 == 0 or epoch == 1:
            print(f"    [Period {period_id}] Epoch {epoch:>3}/{MAX_EPOCHS} "
                  f"| loss={avg_loss:.5f} | {time.time()-t0:.0f}s")

    return model


def predict_proba(model: DNN, X: np.ndarray) -> np.ndarray:
    """Return P(Y=1 | X) — probability of outperforming the cross-sectional median."""
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    return probs[:, 1].cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def compute_daily_portfolio(
    day_df: pd.DataFrame,
    ret_lookup: pd.Series,
    date,
    k: int,
) -> dict | None:
    """
    Long top-k, short bottom-k stocks on signal date `date`.
    Returns earned next-day return (equal-weighted, dollar-neutral).
    """
    if len(day_df) < 2 * k:
        return None

    ranked    = day_df.sort_values("prob_outperform", ascending=False)
    long_idx  = [(date, p) for p in ranked.iloc[:k]["permno"]]
    short_idx = [(date, p) for p in ranked.iloc[-k:]["permno"]]

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


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _paper_period_summary(port_df: pd.DataFrame, sp_df: pd.DataFrame) -> None:
    """Print performance metrics for the paper period (Dec 1992 – Oct 2015)."""
    TDPY = 250
    paper_end = sp_df[sp_df["test_end"] <= "2015-10-31"]["period"].max()
    paper_ret = port_df[port_df["period"] <= paper_end].copy()

    print("\n" + "=" * 65)
    print(f"  PAPER PERIOD (periods 1–{paper_end}): "
          f"{sp_df.loc[sp_df['period']==1, 'test_start'].iloc[0].date()} "
          f"– {sp_df.loc[sp_df['period']==paper_end, 'test_end'].iloc[0].date()}")
    print("=" * 65)
    print(f"  {'k':>5}  {'Day Ret':>8}  {'Ann.Ret':>8}  {'Ann.Vol':>8}  "
          f"{'Sharpe':>8}  {'MaxDD':>8}  {'Total Ret':>10}")
    print("  " + "-" * 68)
    for k in sorted(port_df["k"].unique()):
        sub = paper_ret[paper_ret["k"] == k]["portfolio_ret"]
        if sub.empty:
            continue
        n         = len(sub)
        day_ret   = sub.mean()
        total_ret = (1 + sub).prod() - 1
        ann_ret   = (1 + total_ret) ** (TDPY / n) - 1
        ann_vol   = sub.std() * np.sqrt(TDPY)
        sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0
        cum       = (1 + sub).cumprod()
        max_dd    = ((cum - cum.cummax()) / cum.cummax()).min()
        print(f"  k={k:<4} {day_ret*100:>7.4f}%  {ann_ret*100:>7.2f}%  {ann_vol*100:>7.2f}%  "
              f"{sharpe:>8.3f}  {max_dd*100:>7.2f}%  {total_ret*100:>9.1f}%")


if __name__ == "__main__":
    import sys
    if "--paper" in sys.argv:
        # ── 不重新训练，直接读 CSV 看论文区间绩效 ─────────────────────────────
        _port_df = pd.read_csv(f"{OUTPUT_DIR}/dnn_portfolio_returns.csv")
        _sp_df   = pd.read_csv(f"{OUTPUT_DIR}/study_periods.csv",
                               parse_dates=["train_start", "train_end",
                                            "test_start",  "test_end"])
        _paper_period_summary(_port_df, _sp_df)
        sys.exit(0)

    print("=" * 65)
    print("  DNN Training — Krauss et al. (2017) Replication")
    print(f"  Device : {DEVICE}")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading data...")
    features = pd.read_csv(f"{OUTPUT_DIR}/features.csv",
                           parse_dates=["date"])
    universe = pd.read_csv(f"{OUTPUT_DIR}/daily_universe.csv",
                           parse_dates=["date"])
    sp_df    = pd.read_csv(f"{OUTPUT_DIR}/study_periods.csv",
                           parse_dates=["train_start", "train_end",
                                        "test_start",  "test_end"])

    FEAT_COLS = [c for c in features.columns
                 if c.startswith("R") and c[1:].isdigit()]
    print(f"  features.csv  : {features.shape[0]:,} rows, "
          f"{len(FEAT_COLS)} feature cols")
    print(f"  study periods : {len(sp_df)}")
    print(f"  date range    : {features['date'].min().date()} – "
          f"{features['date'].max().date()}")

    # Build next-day return lookup:
    #   ret_lookup[(date_t, permno)] = actual return on date t+1
    #   Used to compute portfolio P&L for signal issued at close of date_t.
    univ_s = (universe
               .sort_values(["permno", "date"])
               .copy())
    univ_s["ret_next"] = univ_s.groupby("permno")["ret"].shift(-1)
    ret_lookup = (
        univ_s[["permno", "date", "ret_next"]]
        .dropna()
        .set_index(["date", "permno"])["ret_next"]
    )

    all_preds   = []
    port_rows   = []

    # ── Study period loop ──────────────────────────────────────────────────────
    for _, sp in sp_df.iterrows():
        pid = int(sp["period"])
        print(f"\n{'='*65}")
        print(f"  Period {pid:>2}/{len(sp_df)}  |  "
              f"Train {sp['train_start'].date()} → {sp['train_end'].date()}  |  "
              f"Test  {sp['test_start'].date()} → {sp['test_end'].date()}")
        print("=" * 65)

        # ── Split features ────────────────────────────────────────────────────
        tr = features[
            (features["date"] >= sp["train_start"]) &
            (features["date"] <= sp["train_end"])
        ].copy()
        te = features[
            (features["date"] >= sp["test_start"]) &
            (features["date"] <= sp["test_end"])
        ].copy()

        if len(tr) == 0 or len(te) == 0:
            print("  Skipping — no data in this window")
            continue

        X_tr = tr[FEAT_COLS].values.astype(np.float32)
        y_tr = tr["Y"].values.astype(np.int64)
        X_te = te[FEAT_COLS].values.astype(np.float32)

        print(f"  Train: {len(tr):,} rows  |  Test: {len(te):,} rows  |  "
              f"Y=1 ratio: {y_tr.mean():.3f}")

        # ── Standardize (fit on train only, as in H2O default) ────────────────
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        # ── Train DNN ─────────────────────────────────────────────────────────
        model = train_dnn(X_tr, y_tr, pid)

        # ── Predict ───────────────────────────────────────────────────────────
        probs  = predict_proba(model, X_te)
        te     = te.assign(prob_outperform=probs, period=pid)
        all_preds.append(te[["period", "date", "permno", "prob_outperform"]])

        # ── Portfolio returns for each k ──────────────────────────────────────
        for date, day_df in te.groupby("date"):
            for k in K_VALUES:
                row = compute_daily_portfolio(day_df, ret_lookup, date, k)
                if row is not None:
                    row["period"] = pid
                    port_rows.append(row)

        mean_p = probs.mean()
        dir_acc = (probs > 0.5).mean()
        print(f"  Done. mean P(outperform)={mean_p:.4f}  "
              f"pred dir acc={dir_acc:.3f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Saving outputs...")

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv(f"{OUTPUT_DIR}/dnn_predictions.csv", index=False)
    print(f"  Saved -> {OUTPUT_DIR}/dnn_predictions.csv  "
          f"({len(pred_df):,} rows)")

    port_df = pd.DataFrame(port_rows)
    port_df.to_csv(f"{OUTPUT_DIR}/dnn_portfolio_returns.csv", index=False)
    print(f"  Saved -> {OUTPUT_DIR}/dnn_portfolio_returns.csv  "
          f"({len(port_df):,} rows)")

    # ── Performance summary ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PERFORMANCE SUMMARY — DNN (before transaction costs)")
    print("=" * 65)
    print(f"  {'k':>5}  {'Mean ret/day':>12}  {'Std/day':>9}  "
          f"{'Ann.Sharpe':>10}  {'Dir.Acc':>8}  {'N days':>7}")
    print("  " + "-" * 57)

    for k in K_VALUES:
        sub = port_df[port_df["k"] == k]["portfolio_ret"]
        if sub.empty:
            continue
        mu     = sub.mean()
        sigma  = sub.std()
        sharpe = mu / sigma * np.sqrt(252) if sigma > 0 else 0.0
        dacc   = (sub > 0).mean()
        print(f"  k={k:<4} {mu*100:>11.4f}%  {sigma*100:>8.4f}%  "
              f"{sharpe:>10.4f}  {dacc*100:>7.2f}%  {len(sub):>7,}")

    print("\n  Files ready for ENS1 ensemble:")
    print("    data/dnn_predictions.csv  — merge with RAF & GBT predictions")
    print("    data/dnn_portfolio_returns.csv — for individual DNN performance")

    # ── Paper-period performance: Dec 1992 – Oct 2015 (periods 1–23) ──────────
    _paper_period_summary(port_df, sp_df)
