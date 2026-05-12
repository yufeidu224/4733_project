"""
IEOR 4733 - ML Statistical Arbitrage on S&P 500
Table 7 Robustness — DNN Architecture Variants

Usage:
    python Code/dnn_training_variants.py --arch alt1   # 31-15-10-5-2
    python Code/dnn_training_variants.py --arch alt2   # 31-62-10-5-2
    python Code/dnn_training_variants.py --arch alt3   # 31-31-2, tanh, no dropout

Variants:
    alt1 : 31-15-10-5-2  (lower parameterization, first hidden = 15)
    alt2 : 31-62-10-5-2  (higher parameterization, first hidden = 62)
    alt3 : 31-31-2       (standard NN, tanh activation, no dropout, no L1)

Outputs per variant (saved to data/):
    dnn_predictions_{arch}.csv
    dnn_portfolio_returns_{arch}.csv

After running all variants, compute Table 7 Panel B:
    python Code/dnn_training_variants.py --summary
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "data")

# ── Fixed hyperparameters (same as baseline) ───────────────────────────────────
SEED        = 1
BATCH_SIZE  = 2048
MAX_EPOCHS  = 400
L1_LAMBDA   = 1e-5
INPUT_DROP  = 0.1
HIDDEN_DROP = 0.5
K_VALUES    = [10, 50, 100, 150, 200]
PAPER_START = "1992-12-17"
PAPER_END   = "2015-10-15"

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Architecture definitions ───────────────────────────────────────────────────
VARIANTS = {
    "alt1": dict(hidden_dims=[15, 10, 5], use_maxout=True,  use_dropout=True,  use_l1=True),
    "alt2": dict(hidden_dims=[62, 10, 5], use_maxout=True,  use_dropout=True,  use_l1=True),
    "alt3": dict(hidden_dims=[31],        use_maxout=False, use_dropout=False, use_l1=False),
}


# ══════════════════════════════════════════════════════════════════════════════
# Model classes
# ══════════════════════════════════════════════════════════════════════════════

class MaxoutLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_pieces: int = 2):
        super().__init__()
        self.num_pieces   = num_pieces
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features * num_pieces)

    def forward(self, x):
        z = self.linear(x)
        z = z.view(-1, self.out_features, self.num_pieces)
        return z.max(dim=2).values


class DNNMaxout(nn.Module):
    """Maxout DNN — used for alt1 and alt2."""
    def __init__(self, input_dim=31, hidden_dims=[31, 10, 5], output_dim=2,
                 input_drop=INPUT_DROP, hidden_drop=HIDDEN_DROP):
        super().__init__()
        self.input_dropout = nn.Dropout(p=input_drop)
        blocks, in_dim = [], input_dim
        for h in hidden_dims:
            blocks.append(MaxoutLayer(in_dim, h, num_pieces=2))
            blocks.append(nn.Dropout(p=hidden_drop))
            in_dim = h
        self.hidden       = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.hidden(x)
        x = self.output_layer(x)
        return torch.softmax(x, dim=1)


class DNNTanh(nn.Module):
    """Standard NN with tanh — used for alt3 (31-31-2, no dropout, no L1)."""
    def __init__(self, input_dim=31, hidden_dims=[31], output_dim=2):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.hidden       = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output_layer(x)
        return torch.softmax(x, dim=1)


def build_model(cfg: dict) -> nn.Module:
    if cfg["use_maxout"]:
        return DNNMaxout(hidden_dims=cfg["hidden_dims"]).to(DEVICE)
    else:
        return DNNTanh(hidden_dims=cfg["hidden_dims"]).to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, cfg: dict, period_id: int):
    torch.manual_seed(SEED)
    g = torch.Generator(); g.manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train,  dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=True, generator=g,
    )
    model     = build_model(cfg)
    optimizer = optim.Adadelta(model.parameters())
    criterion = nn.CrossEntropyLoss()
    params    = list(model.parameters())

    loss_history, best_ma, no_improve = [], float("inf"), 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_b)
            ce    = criterion(preds, y_b)
            if cfg["use_l1"]:
                ce = ce + L1_LAMBDA * sum(p.abs().sum() for p in params)
            ce.backward()
            optimizer.step()
            epoch_loss += ce.item(); n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        loss_history.append(avg)

        if len(loss_history) >= 5:
            ma = float(np.mean(loss_history[-5:]))
            if ma < best_ma - 1e-7:
                best_ma = ma; no_improve = 0
            else:
                no_improve += 1
            if no_improve >= 5:
                print(f"    [Period {period_id}] Early stop @ epoch {epoch} "
                      f"| loss_ma={ma:.5f} | {time.time()-t0:.0f}s")
                break

        if epoch % 50 == 0 or epoch == 1:
            print(f"    [Period {period_id}] Epoch {epoch:>3}/{MAX_EPOCHS} "
                  f"| loss={avg:.5f} | {time.time()-t0:.0f}s")

    return model


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    return probs[:, 1].cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio
# ══════════════════════════════════════════════════════════════════════════════

def compute_daily_portfolio(day_df, ret_lookup, date, k):
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
# Run one variant
# ══════════════════════════════════════════════════════════════════════════════

def run_variant(arch: str):
    cfg = VARIANTS[arch]
    arch_str = "31-" + "-".join(str(h) for h in cfg["hidden_dims"]) + "-2"
    print("=" * 65)
    print(f"  DNN Variant: {arch}  |  Architecture: {arch_str}")
    print(f"  Maxout: {cfg['use_maxout']}  |  Dropout: {cfg['use_dropout']}  "
          f"|  L1: {cfg['use_l1']}  |  Device: {DEVICE}")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading data...")
    features = pd.read_csv(os.path.join(OUTPUT_DIR, "features.csv"),
                           parse_dates=["date"])
    universe = pd.read_csv(os.path.join(OUTPUT_DIR, "daily_universe.csv"),
                           parse_dates=["date"])
    sp_df    = pd.read_csv(os.path.join(OUTPUT_DIR, "study_periods.csv"),
                           parse_dates=["train_start", "train_end",
                                        "test_start",  "test_end"])

    FEAT_COLS = [c for c in features.columns if c.startswith("R") and c[1:].isdigit()]

    univ_s = universe.sort_values(["permno", "date"]).copy()
    univ_s["ret_next"] = univ_s.groupby("permno")["ret"].shift(-1)
    ret_lookup = (univ_s[["permno", "date", "ret_next"]]
                  .dropna()
                  .set_index(["date", "permno"])["ret_next"])

    all_preds, port_rows = [], []

    # ── Study period loop ──────────────────────────────────────────────────────
    for _, sp in sp_df.iterrows():
        pid = int(sp["period"])
        print(f"\n{'='*65}")
        print(f"  Period {pid:>2}/{len(sp_df)}  "
              f"Train {sp['train_start'].date()} -> {sp['train_end'].date()}  "
              f"Test  {sp['test_start'].date()} -> {sp['test_end'].date()}")
        print("=" * 65)

        tr = features[(features["date"] >= sp["train_start"]) &
                      (features["date"] <= sp["train_end"])].copy()
        te = features[(features["date"] >= sp["test_start"]) &
                      (features["date"] <= sp["test_end"])].copy()
        if len(tr) == 0 or len(te) == 0:
            continue

        X_tr = tr[FEAT_COLS].values.astype(np.float32)
        y_tr = tr["Y"].values.astype(np.int64)
        X_te = te[FEAT_COLS].values.astype(np.float32)

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        model = train_model(X_tr, y_tr, cfg, pid)
        probs = predict_proba(model, X_te)
        te    = te.assign(prob_outperform=probs, period=pid)
        all_preds.append(te[["period", "date", "permno", "prob_outperform"]])

        for date, day_df in te.groupby("date"):
            for k in K_VALUES:
                row = compute_daily_portfolio(day_df, ret_lookup, date, k)
                if row is not None:
                    row["period"] = pid
                    port_rows.append(row)

        print(f"  Done. mean P={probs.mean():.4f}  dir_acc={(probs>0.5).mean():.3f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    pred_df = pd.concat(all_preds, ignore_index=True)
    port_df = pd.DataFrame(port_rows)

    pred_path = os.path.join(OUTPUT_DIR, f"dnn_predictions_{arch}.csv")
    port_path = os.path.join(OUTPUT_DIR, f"dnn_portfolio_returns_{arch}.csv")
    pred_df.to_csv(pred_path, index=False)
    port_df.to_csv(port_path, index=False)
    print(f"\nSaved -> {pred_path}")
    print(f"Saved -> {port_path}")

    # ── Table 7 Panel B: mean return k=10 paper period ─────────────────────────
    k10 = port_df[port_df["k"] == 10]
    k10 = k10[k10["date"].between(PAPER_START, PAPER_END)]
    mean_ret = k10["portfolio_ret"].mean()
    print(f"\nTable 7 Panel B — {arch} ({arch_str})")
    print(f"  Mean return/day k=10 (paper period, before TC): {mean_ret:.4f}")

    return mean_ret


# ══════════════════════════════════════════════════════════════════════════════
# Summary: Table 7 Panel B across all variants
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    """Read saved portfolio returns and print Table 7 Panel B."""
    # baseline
    rows = {"baseline (31-31-10-5-2)": "dnn_portfolio_returns.csv"}
    for arch, cfg in VARIANTS.items():
        arch_str = "31-" + "-".join(str(h) for h in cfg["hidden_dims"]) + "-2"
        rows[f"{arch} ({arch_str})"] = f"dnn_portfolio_returns_{arch}.csv"

    print("\n" + "=" * 55)
    print("  Table 7 Panel B — Mean return/day, k=10, before TC")
    print("  Paper period: 1992-12-17 to 2015-10-15")
    print("=" * 55)
    print(f"  {'Architecture':<30} {'Our result':>12} {'Paper':>8}")
    print("  " + "-" * 52)
    paper_vals = {"baseline (31-31-10-5-2)": 0.0033,
                  "alt1 (31-15-10-5-2)":     0.0030,
                  "alt2 (31-62-10-5-2)":     0.0031,
                  "alt3 (31-31-2)":           0.0015}

    for label, fname in rows.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  {label:<30} {'(not run yet)':>12}")
            continue
        df = pd.read_csv(fpath, parse_dates=["date"])
        k10 = df[(df["k"] == 10) &
                 (df["date"] >= PAPER_START) &
                 (df["date"] <= PAPER_END)]
        mean_ret = k10["portfolio_ret"].mean()
        paper_v  = paper_vals.get(label, "-")
        print(f"  {label:<30} {mean_ret:>12.4f} {str(paper_v):>8}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["alt1", "alt2", "alt3"],
                        help="Which architecture variant to train")
    parser.add_argument("--summary", action="store_true",
                        help="Print Table 7 Panel B summary (no training)")
    args = parser.parse_args()

    if args.summary:
        print_summary()
    elif args.arch:
        run_variant(args.arch)
        print_summary()
    else:
        parser.print_help()
