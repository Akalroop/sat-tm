"""#1 Main entry point — LSTM autoencoder anomaly detection pipeline."""

import gc
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import load_and_scale, make_dataloaders
from model import LSTMAutoencoder
from trainer import train
from evaluation import (
    per_channel_errors, window_any_anomaly, estimate_clean_stats,
    channelwise_topk_score, tune_threshold, apply_persistence_filter,
    compute_metrics, compute_val_metrics_full,
)
from sklearn.metrics import precision_recall_curve


def _seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=False)


def main(config_path="../config/config.json"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # ---- config + seed ----
    with open(config_path) as f:
        cfg = json.load(f)

    EXPERIMENT_ID = cfg["experiment_id"]
    SEED = cfg["seed"]
    _seed(SEED)

    RESULTS_ROOT = Path("experiments")
    RUN_DIR = RESULTS_ROOT / EXPERIMENT_ID
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Experiment '{EXPERIMENT_ID}' | seed={SEED} -> {RUN_DIR.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- data ----
    X, Y, Z, valid_labels, test_labels, n_features, scaler = load_and_scale(cfg, run_dir=RUN_DIR)
    loaders = make_dataloaders(X, Y, Z, cfg)
    del X, Y, Z  # free raw arrays after dataloaders built
    gc.collect()

    #---- model ----
    HIDDEN_DIM = cfg["model"]["hidden_size"]
    NUM_LAYERS = cfg["model"]["num_layers"]

    assert n_features == cfg["model"]["parameter_size"], \
        f"Feature mismatch: data={n_features}, config={cfg['model']['parameter_size']}"

    model = LSTMAutoencoder(n_features, HIDDEN_DIM, NUM_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,} ({n_params*4/1e6:.2f} MB fp32)")

    # ---- train ----
    (model, train_loss_history, val_loss_history,
     best_val_loss, best_epoch, best_epoch_time, total_train_time) = train(
        model, loaders, device, cfg,
    )

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, RUN_DIR / "best_model.pt")
    print(f"Saved best model -> {RUN_DIR}/best_model.pt  "
          f"epoch={best_epoch} | val_loss={best_val_loss:.6f}")

    # ---- loss curves ----
    plt.plot(train_loss_history, label="train")
    plt.plot(val_loss_history, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(RUN_DIR / "loss_curves.png")
    plt.close()

    # ---- inference: per-channel errors ----
    val_pc_err = per_channel_errors(loaders["val"], model, device)
    test_pc_err = per_channel_errors(loaders["test"], model, device)

    valid_window_labels = window_any_anomaly(valid_labels, cfg["dataloader"]["window_size"])
    test_window_labels = window_any_anomaly(test_labels, cfg["dataloader"]["window_size"])

    mu_rec, sigma_rec = estimate_clean_stats(val_pc_err, valid_window_labels, cfg)
    TOP_K = cfg["metrics"]["top_k"]

    val_score = channelwise_topk_score(val_pc_err, mu_rec, sigma_rec, top_k=TOP_K)
    test_score = channelwise_topk_score(test_pc_err, mu_rec, sigma_rec, top_k=TOP_K)

    print(f"Validation Per-Channel Errors: {val_pc_err.shape}")
    print(f"Test Per-Channel Errors: {test_pc_err.shape}")
    print(f"val_score: shape={val_score.shape}, min={val_score.min():.4f}, mean={val_score.mean():.4f}, max={val_score.max():.4f}")
    print(f"test_score: shape={test_score.shape}, min={test_score.min():.4f}, mean={test_score.mean():.4f}, max={test_score.max():.4f}")

    # ---- save clean reconstruction stats for live inference ----
    np.savez(RUN_DIR / "clean_stats.npz",
             mu_rec_err=mu_rec,
             sigma_rec_err=sigma_rec,
             top_k=np.array(TOP_K),
             min_rec_sigma=np.array(cfg["metrics"]["min_rec_sigma"]))
    print(f"Saved clean stats to {RUN_DIR}/clean_stats.npz")

    # ---- threshold tuning on validation ----
    strategy = cfg.get("metrics", {}).get("threshold_strategy", "f1_optimal")
    target_recall = float(cfg.get("metrics", {}).get("target_recall", 0.95))
    persistence_window = int(cfg.get("metrics", {}).get("persistence_window", 1))

    threshold, candidates, indices, prec_val, rec_val = tune_threshold(
        val_score, valid_window_labels, strategy, target_recall,
    )

    print("Validation-only threshold candidates:")
    for candidate_name, candidate_threshold in candidates.items():
        idx = indices[candidate_name]
        print(f"  {candidate_name:14s} threshold={candidate_threshold:.6f} "
              f"PR precision={prec_val[idx]:.4f} recall={rec_val[idx]:.4f}")
    print(f"Selected strategy: {strategy}")
    print(f"Locked validation threshold: {threshold:.6f}")
    print(f"top_k={TOP_K} | min_rec_sigma={cfg['metrics']['min_rec_sigma']} | persistence_window={persistence_window}")

    # ---- validation metrics before/after persistence ----
    val_preds_raw = (val_score >= threshold).astype(np.int8)
    val_preds = apply_persistence_filter(val_preds_raw, persistence_window)

    v_prec_raw, v_rec_raw, v_f1_raw, v_f2_raw = compute_val_metrics_full(valid_window_labels, val_preds_raw)
    v_prec, v_rec, v_f1, v_f2 = compute_val_metrics_full(valid_window_labels, val_preds)

    print("Validation metrics BEFORE persistence: "
          f"precision={v_prec_raw:.4f}, recall={v_rec_raw:.4f}, F1={v_f1_raw:.4f}, F2={v_f2_raw:.4f}")
    print("Validation metrics AFTER persistence:  "
          f"precision={v_prec:.4f}, recall={v_rec:.4f}, F1={v_f1:.4f}, F2={v_f2:.4f}")
    print(f"Validation window-level recall: {v_rec:.4f} ({v_rec*100:.2f}%)")

    # ---- test predictions + metrics ----
    test_preds_raw = (test_score > threshold).astype(int)
    test_preds = apply_persistence_filter(test_preds_raw, persistence_window)

    recall, precision, f1, conf_matrix = compute_metrics(
        test_window_labels, test_preds,
    )
    print(f"Test: recall={recall:.4f} | precision={precision:.4f} | F1={f1:.4f}")
    print("Confusion matrix:\n", conf_matrix)

    # ---- PR curve (test) ----
    prec_curve, rec_curve, _ = precision_recall_curve(test_window_labels, test_score)
    plt.figure(figsize=(7, 5))
    plt.plot(rec_curve, prec_curve, label=f"PR (F1={f1:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test)")
    plt.legend()
    plt.grid(True)
    plt.savefig(RUN_DIR / "pr_curve.png")
    plt.close()

    # ---- save metrics ----
    EXPERIMENT_ID = cfg["experiment_id"]
    BASELINE_ID = cfg.get("baseline_id")
    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "baseline_id": BASELINE_ID,
        "threshold": float(threshold),
        "recall": recall, "precision": precision, "f1": f1,
        "confusion_matrix": conf_matrix.tolist(),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_pc_err.mean()),
        "total_train_time": float(total_train_time),
        "seed": SEED,
    }
    with open(RUN_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(RUN_DIR / "run_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- append ablation CSV ----
    master_csv = RESULTS_ROOT / "ablation_results.csv"
    summary_row = {
        **metrics,
        "experiment_type": cfg.get("experiment_type"),
        "window_size": cfg["dataloader"]["window_size"],
        "batch_size": cfg["training"]["batch_size"],
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "lr": cfg["training"]["learning_rate"],
        "use_amp": cfg["training"]["amp"] and torch.cuda.is_available(),
    }
    row_df = pd.DataFrame([summary_row])
    row_df.to_csv(master_csv, mode="a", header=not master_csv.exists(), index=False)
    print(f"Appended summary -> {master_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    main(args.config)
