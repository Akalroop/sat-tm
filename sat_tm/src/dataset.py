from pathlib import Path

from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import torch


def load_and_scale(
    cfg,
    run_dir: Path | None = None,
    train_dir="data_fixed/train/",
    valid_dirs=("data_fixed/eval/day_26.parquet", "data_fixed/eval/day_27.parquet"),
    test_dirs=(
        "data_fixed/test/day_28.parquet",
        "data_fixed/test/day_29.parquet",
        "data_fixed/test/day_30.parquet",
    ),
):
    """Load parquet files, scale, return scaled numpy arrays and labels."""
    from sklearn.preprocessing import StandardScaler

    MIN_STD = cfg["metrics"]["min_std"]
    CLIP_SCALED_TO = cfg["metrics"]["clip_scaled_to"]

    train_df = pd.read_parquet(train_dir)

    valid_df = pd.concat([pd.read_parquet(d) for d in valid_dirs], ignore_index=True)
    test_df = pd.concat([pd.read_parquet(d) for d in test_dirs], ignore_index=True)

    for df in (train_df, valid_df, test_df):
        df.set_index("timestamp", inplace=True)

    # Labels
    valid_labels = valid_df["is_anomaly"].to_numpy(np.int8)
    test_labels = test_df["is_anomaly"].to_numpy(np.int8)

    # Features (drop is_anomaly for val/test, keep all for train)
    X = train_df
    Y = valid_df.drop(columns=["is_anomaly"])
    Z = test_df.drop(columns=["is_anomaly"])

    n_features = X.shape[1]
    print(f"Feature count: {n_features}")

    # Scale — fit on train, floor near-zero std, clip outliers
    scaler = StandardScaler()
    scaler.fit(X)

    near_constant = X.columns[scaler.scale_ < MIN_STD].tolist()
    if near_constant:
        print(f"{len(near_constant)} near-constant columns floored to scale=1.0")
    scaler.scale_ = np.where(scaler.scale_ < MIN_STD, 1.0, scaler.scale_)

    X_scaled = np.clip(scaler.transform(X).astype(np.float32), -CLIP_SCALED_TO, CLIP_SCALED_TO)
    Y_scaled = np.clip(scaler.transform(Y).astype(np.float32), -CLIP_SCALED_TO, CLIP_SCALED_TO)
    Z_scaled = np.clip(scaler.transform(Z).astype(np.float32), -CLIP_SCALED_TO, CLIP_SCALED_TO)

    import pickle
    if run_dir is not None:
        with open(run_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print(f"Saved scaler to {run_dir}/scaler.pkl")

    del X, Y, Z

    return X_scaled, Y_scaled, Z_scaled, valid_labels, test_labels, n_features, scaler


def make_dataloaders(X_scaled, Y_scaled, Z_scaled, cfg):
    """Create WindowedTimeSeriesDataset + DataLoader for each split."""
    WINDOW_SIZE = cfg["dataloader"]["window_size"]
    BATCH_SIZE = cfg["training"]["batch_size"]
    NUM_WORKERS = cfg["hardware"]["num_workers"]
    PIN_MEMORY = torch.cuda.is_available()

    X_dataset = WindowedTimeSeriesDataset(X_scaled, WINDOW_SIZE)
    val_dataset = WindowedTimeSeriesDataset(Y_scaled, WINDOW_SIZE)
    test_dataset = WindowedTimeSeriesDataset(Z_scaled, WINDOW_SIZE)

    loaders = {
        "train": DataLoader(
            X_dataset, batch_size=BATCH_SIZE, shuffle=cfg["dataloader"]["shuffle"],
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            persistent_workers=(NUM_WORKERS > 0),
        ),
        "val": DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            persistent_workers=(NUM_WORKERS > 0),
        ),
        "test": DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            persistent_workers=(NUM_WORKERS > 0),
        ),
    }

    x, _ = next(iter(loaders["train"]))
    print(f"Batch shape: {x.shape} {x.dtype}")
    return loaders


class WindowedTimeSeriesDataset(Dataset):
    """Lazily slices fixed-length windows from a flat (N, F) array."""

    def __init__(self, arr: np.ndarray, window_size: int):
        assert arr.ndim == 2, "expected (n_timesteps, n_features)"
        self.arr = np.ascontiguousarray(arr)
        self.window_size = int(window_size)
        self.n_windows = arr.shape[0] - self.window_size + 1
        if self.n_windows <= 0:
            raise ValueError("window_size larger than dataset length")

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        window = self.arr[idx : idx + self.window_size]  # (window_size, F) view
        tensor = torch.from_numpy(window.copy())          # copy only this window
        return tensor, tensor  # (input, target) for autoencoder
