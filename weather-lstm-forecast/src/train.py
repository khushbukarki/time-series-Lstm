"""Train the LSTM, benchmark it against persistence, moving-average and ARIMA baselines, save artifacts.

    python -m src.train --download          # fetch Auckland history from Open-Meteo, then train
    python -m src.train                     # train on data/auckland_daily.csv
    python -m src.train --synthetic --epochs 3 --skip-arima     # quick smoke test
"""
import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from .baselines import arima_one_step, moving_average, persistence  # noqa: E402
from .config import Config  # noqa: E402
from .data import (StandardScaler, add_features, chronological_split, download_history,  # noqa: E402
                   load_history, make_windows, synthetic_history)
from .metrics import mae, rmse  # noqa: E402
from .model import LSTMForecaster, predict  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "auckland_daily.csv"


def train_model(cfg: Config, X_tr, y_tr, X_val, y_val, max_epochs: int, log=print):
    torch.manual_seed(cfg.seed)
    model = LSTMForecaster(X_tr.shape[2], cfg.horizon, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=cfg.batch_size, shuffle=True)  # shuffling windows is fine; split is by time
    Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val)

    best_state, best_val, bad_epochs, history = None, float("inf"), 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        sched.step(val_loss)
        history.append({"epoch": epoch, "train_loss": total / len(X_tr), "val_loss": val_loss})
        log(f"epoch {epoch:3d}  train {total / len(X_tr):.4f}  val {val_loss:.4f}")
        if val_loss < best_val - 1e-5:
            best_val, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                log(f"early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return model, history


def run(cfg: Config, df, out_dir: Path, max_epochs: int, skip_arima: bool = False, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = add_features(df, cfg)
    t_idx = cfg.feature_names.index(cfg.target)
    train, val, test = chronological_split(feats, cfg)

    scaler = StandardScaler.fit(train.to_numpy())
    X_tr, y_tr = make_windows(scaler.transform(train.to_numpy()), cfg.lookback, cfg.horizon, t_idx)
    X_val, y_val = make_windows(scaler.transform(val.to_numpy()), cfg.lookback, cfg.horizon, t_idx)
    X_te, _ = make_windows(scaler.transform(test.to_numpy()), cfg.lookback, cfg.horizon, t_idx)
    X_te_raw, y_te = make_windows(test.to_numpy(), cfg.lookback, cfg.horizon, t_idx)  # unscaled truth

    start = time.time()
    model, history = train_model(cfg, X_tr, y_tr, X_val, y_val, max_epochs, log)
    lstm_pred = scaler.inverse_target(predict(model, X_te), t_idx)

    target_hist = X_te_raw[:, :, t_idx]
    preds = {
        "LSTM": lstm_pred,
        "Persistence": persistence(target_hist, cfg.horizon),
        "Moving average (7d)": moving_average(target_hist, cfg.horizon, 7),
    }
    per_horizon = {
        name: {"rmse": [rmse(y_te[:, h], p[:, h]) for h in range(cfg.horizon)],
               "mae": [mae(y_te[:, h], p[:, h]) for h in range(cfg.horizon)]}
        for name, p in preds.items()
    }
    day1 = {name: {"rmse": m["rmse"][0], "mae": m["mae"][0]} for name, m in per_horizon.items()}

    arima_order, arima_pred = None, None
    if not skip_arima:
        n = len(feats)
        series = feats[cfg.target].to_numpy()
        arima_pred, arima_order = arima_one_step(series, len(train), n - int(n * cfg.test_frac), len(y_te))
        day1["ARIMA"] = {"rmse": rmse(y_te[:, 0], arima_pred), "mae": mae(y_te[:, 0], arima_pred)}

    best_baseline = min((v["rmse"], k) for k, v in day1.items() if k != "LSTM")
    metrics = {
        "location": cfg.location,
        "test_period": [str(test.index[cfg.lookback].date()), str(test.index[-1].date())],
        "n_test_windows": int(len(y_te)),
        "day1": day1,
        "per_horizon": per_horizon,
        "arima_order": arima_order,
        "lstm_vs_best_baseline_day1_rmse_pct": round(100 * (1 - day1["LSTM"]["rmse"] / best_baseline[0]), 1),
        "best_baseline": best_baseline[1],
        "epochs_trained": len(history),
        "train_seconds": round(time.time() - start, 1),
    }

    torch.save(model.state_dict(), out_dir / "model.pt")
    scaler.save(out_dir / "scaler.json")
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history))
    plot_results(test.index[cfg.lookback : cfg.lookback + len(y_te)], y_te, preds, arima_pred, per_horizon,
                 history, out_dir)
    return metrics


def plot_results(dates, y_true, preds, arima_pred, per_horizon, history, out_dir: Path):
    last = slice(-120, None)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates[last], y_true[last, 0], color="black", lw=1.6, label="Actual")
    ax.plot(dates[last], preds["LSTM"][last, 0], color="#1f77b4", label="LSTM")
    if arima_pred is not None:
        ax.plot(dates[last], arima_pred[last], color="#ff7f0e", alpha=0.8, label="ARIMA")
    ax.set_title("Next-day mean temperature, last 120 days of the test set")
    ax.set_ylabel("°C")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "test_predictions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, m in per_horizon.items():
        ax.plot(range(1, len(m["rmse"]) + 1), m["rmse"], marker="o", label=name)
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Error by forecast horizon")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_by_horizon.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h["train_loss"] for h in history], label="train")
    ax.plot([h["val_loss"] for h in history], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled)")
    ax.set_title("Training curve")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="download history from Open-Meteo first")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (tests / no network)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--skip-arima", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    args = ap.parse_args()

    cfg = Config()
    if args.synthetic:
        df = synthetic_history()
    elif args.download or not DATA_PATH.exists():
        print(f"Downloading {cfg.location} history {cfg.start_date}..{cfg.end_date} from Open-Meteo")
        df = download_history(cfg, DATA_PATH)
    else:
        df = load_history(DATA_PATH)

    m = run(cfg, df, args.out, args.epochs or cfg.max_epochs, args.skip_arima)
    print(f"\nTest period {m['test_period'][0]} to {m['test_period'][1]} ({m['n_test_windows']} windows)")
    print(f"{'model':<22}{'RMSE':>8}{'MAE':>8}   (next-day mean temperature, °C)")
    for name, v in sorted(m["day1"].items(), key=lambda kv: kv[1]["rmse"]):
        print(f"{name:<22}{v['rmse']:>8.3f}{v['mae']:>8.3f}")
    print(f"\nLSTM vs best baseline ({m['best_baseline']}): {m['lstm_vs_best_baseline_day1_rmse_pct']:+.1f}% RMSE")
    print(f"Artifacts saved to {args.out}")


if __name__ == "__main__":
    main()
