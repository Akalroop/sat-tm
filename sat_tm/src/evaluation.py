import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score,
    f1_score, fbeta_score, confusion_matrix,
)


def per_channel_errors(loader, model, device):
    """MSE per-channel reconstruction error. Returns (n_windows, n_features)."""
    model.eval()
    errs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            recon = model(x)
            err = ((recon - x) ** 2).mean(dim=1)  # (batch, n_features)
            errs.append(err.cpu().numpy())
    if not errs:
        raise ValueError("No windows were produced by the loader")
    return np.concatenate(errs, axis=0)


def window_any_anomaly(raw_labels, window_size):
    """Mark a window anomalous if *any* timestep in it is anomalous."""
    return np.lib.stride_tricks.sliding_window_view(
        raw_labels, window_shape=window_size
    ).max(axis=1)


def estimate_clean_stats(val_pc_err, valid_window_labels, cfg):
    """Estimate mean/std of reconstruction error on clean (non-anomalous) val windows."""
    MIN_REC_SIGMA = float(cfg["metrics"]["min_rec_sigma"])
    TOP_K = cfg["metrics"]["top_k"]

    clean_mask = valid_window_labels == 0
    if not np.any(clean_mask):
        raise ValueError("No clean validation windows; cannot estimate statistics")

    mu = val_pc_err[clean_mask].mean(axis=0)
    raw_sigma = val_pc_err[clean_mask].std(axis=0)
    sigma = np.maximum(raw_sigma, MIN_REC_SIGMA)

    print(f"Clean val windows: {int(clean_mask.sum())}/{len(clean_mask)}")
    floored = (raw_sigma < MIN_REC_SIGMA).sum()
    print(f"Channels floored at MIN_REC_SIGMA={MIN_REC_SIGMA}: {floored} / {len(raw_sigma)}")

    return mu, sigma


def channelwise_topk_score(errors_per_channel, mu, sigma, top_k):
    """Mean of the largest *top_k* channel z-scores for each window."""
    z = (errors_per_channel - mu) / sigma
    topk_z = np.partition(z, -top_k, axis=1)[:, -top_k:]
    return topk_z.mean(axis=1)


def tune_threshold(val_score, valid_window_labels, strategy, target_recall):
    """Find threshold from PR curve using the chosen strategy.

    Returns (threshold, candidates_dict, indices, prec_val, rec_val).
    """
    prec_val, rec_val, thresh_val = precision_recall_curve(
        valid_window_labels, val_score
    )
    f1_curve = 2 * (prec_val * rec_val) / np.maximum(prec_val + rec_val, 1e-8)
    f2_curve = 5 * prec_val * rec_val / np.maximum(4 * prec_val + rec_val, 1e-12)

    idx_f1 = int(np.argmax(f1_curve[:-1]))
    idx_f2 = int(np.argmax(f2_curve[:-1]))

    target_idxs = np.flatnonzero(rec_val[:-1] >= target_recall)
    if len(target_idxs):
        idx_target = int(target_idxs[np.argmax(prec_val[target_idxs])])
    else:
        idx_target = int(np.argmax(rec_val[:-1]))
        print(f"WARNING: target recall {target_recall:.3f} not achievable; "
              f"best available recall is {rec_val[idx_target]:.4f}")

    candidates = {
        "f1_optimal": float(thresh_val[idx_f1]),
        "f2_recall": float(thresh_val[idx_f2]),
        "target_recall": float(thresh_val[idx_target]),
    }
    indices = {"f1_optimal": idx_f1, "f2_recall": idx_f2, "target_recall": idx_target}

    if strategy not in candidates:
        raise ValueError(f"Unsupported threshold_strategy={strategy!r}; "
                         f"choose from {list(candidates)}")

    return candidates[strategy], candidates, indices, prec_val, rec_val


def apply_persistence_filter(preds, window_len):
    """Retain positive predictions belonging to a run of >= window_len windows."""
    preds = np.asarray(preds, dtype=np.int8)
    if window_len <= 1:
        return preds.copy()
    s = pd.Series(preds)
    fwd = s.rolling(window_len, min_periods=window_len).sum() == window_len
    bwd = s.iloc[::-1].rolling(window_len, min_periods=window_len).sum().iloc[::-1] == window_len
    return (fwd | bwd).astype(np.int8).to_numpy()


def compute_metrics(labels, preds):
    """Return (recall, precision, f1, confusion_matrix)."""
    return (
        float(recall_score(labels, preds)),
        float(precision_score(labels, preds)),
        float(f1_score(labels, preds)),
        confusion_matrix(labels, preds),
    )


def compute_val_metrics_full(labels, preds):
    """Return (recall, precision, f1, f2) for validation reporting."""
    return (
        float(recall_score(labels, preds, zero_division=0)),
        float(precision_score(labels, preds, zero_division=0)),
        float(f1_score(labels, preds, zero_division=0)),
        float(fbeta_score(labels, preds, beta=2, zero_division=0)),
    )
