#!/usr/bin/env python3
"""
dashboard.py — PyQt6 Satellite Telemetry Dashboard with ML Anomaly Pipeline

Live GUI dashboard for LEO satellite telemetry simulation with **model-based**
anomaly detection using a trained LSTM Autoencoder.

Pipeline per tick:
  live frame → extract feature columns → scaler.transform → push to window buffer
    → when buffer full (WINDOW_SIZE frames): pass through model
       → compute per-channel reconstruction error (MSE over window dim)
       → z-score channels using clean-validation stats (μ, σ)
       → channelwise_topk_score (mean of top-K z-scores)
       → compare to locked threshold → binary anomaly prediction
       → persistence filter (consecutive anomalous windows)

Usage:
    python dashboard.py <experiment_dir>

  Experiment directory must contain:
    - config.json / result_config.json  (pipeline hyper-parameters + model architecture)
    - best_model.pt                     (trained model state dict)
    - scaler.pkl                        (fitted StandardScaler)
    - clean_stats.npz                   (clean validation stats for z-score normalization)

  Simulator defaults to sat_tm/simdata/sim66.py.
  Results are saved to sat_tm/simdata/dashboard_<experiment_id>.json on stop.

Dependencies: PyQt6, matplotlib, numpy, torch, scikit-learn
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================
import sys
import os
import json
import math
import time
import pickle
import warnings
import importlib.util
from datetime import datetime
from collections import deque
from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFormLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QLineEdit, QFileDialog,
    QGroupBox, QSplitter, QScrollArea, QFrame, QMessageBox,
    QTabWidget, QStatusBar, QProgressBar,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QTime
from PyQt6.QtGui import QFont, QColor, QTextCursor

# Matplotlib integration
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    import matplotlib
    matplotlib.use('qtagg')
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    import matplotlib
    matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# ML
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ============================================================================
# SECTION 2: SUBSYSTEM DEFINITIONS
# ============================================================================

SUBSYSTEMS = {
    "Power": [
        "solar_panel_current_A", "solar_panel_watts", "solar_panel_voltage_V",
        "battery_voltage_V", "battery_current_A", "battery_soc", "battery_temperature_C",
        "bus_power_W", "total_power_W", "heater_bus_voltage_V",
    ],
    "Thermal": [
        "cpu_temperature_C", "rx_temperature_C", "bus_temperature_C", "ambient_temperature_C",
        "heater_cpu_current_A", "heater_batt_current_A", "heater_rx_current_A",
        "thermistor_cpu_setpoint_C", "thermistor_batt_setpoint_C", "thermistor_rx_setpoint_C",
    ],
    "ADCS": [
        "att_q0", "att_q1", "att_q2", "att_q3",
        "att_rate_x_rads", "att_rate_y_rads", "att_rate_z_rads",
        "rw_speed_1_rpm", "rw_speed_2_rpm", "rw_speed_3_rpm", "rw_speed_4_rpm",
        "rw_torque_1_nm", "rw_torque_2_nm", "rw_torque_3_nm", "rw_torque_4_nm",
        "gyro_rate_x_dps", "gyro_rate_y_dps", "gyro_rate_z_dps",
    ],
    "Orbit & Nav": [
        "altitude_km", "nav_pos_x_km", "nav_pos_y_km", "nav_pos_z_km",
        "nav_vel_x_kms", "nav_vel_y_kms", "nav_vel_z_kms",
        "sun_angle_deg", "eclipse_depth", "shadow_fraction", "sun_exposure",
        "sun_vector_x", "sun_vector_y", "sun_vector_z",
    ],
    "Comms": [
        "comm_snr_db", "comm_active", "ground_range_km",
    ],
    "Propulsion": [
        "thruster_active", "thrust_force_N", "thruster_propellant_kg", "thruster_burn_time_s",
        "leb_active", "leb_thrust_N", "leb_power_W", "leb_propellant_kg", "leb_burn_time_s",
    ],
    "System": [
        "system_load",
    ],
}

# Unit suffixes for display
UNITS = {
    # Power
    "solar_panel_current_A": "A", "solar_panel_watts": "W", "solar_panel_voltage_V": "V",
    "battery_voltage_V": "V", "battery_current_A": "A", "battery_soc": "",
    "battery_temperature_C": "°C", "bus_power_W": "W", "total_power_W": "W", "heater_bus_voltage_V": "V",
    # Thermal
    "cpu_temperature_C": "°C", "rx_temperature_C": "°C", "bus_temperature_C": "°C", "ambient_temperature_C": "°C",
    "heater_cpu_current_A": "A", "heater_batt_current_A": "A", "heater_rx_current_A": "A",
    "thermistor_cpu_setpoint_C": "°C", "thermistor_batt_setpoint_C": "°C", "thermistor_rx_setpoint_C": "°C",
    # ADCS
    "att_q0": "", "att_q1": "", "att_q2": "", "att_q3": "",
    "att_rate_x_rads": "rad/s", "att_rate_y_rads": "rad/s", "att_rate_z_rads": "rad/s",
    "rw_speed_1_rpm": "RPM", "rw_speed_2_rpm": "RPM", "rw_speed_3_rpm": "RPM", "rw_speed_4_rpm": "RPM",
    "rw_torque_1_nm": "Nm", "rw_torque_2_nm": "Nm", "rw_torque_3_nm": "Nm", "rw_torque_4_nm": "Nm",
    "gyro_rate_x_dps": "°/s", "gyro_rate_y_dps": "°/s", "gyro_rate_z_dps": "°/s",
    # Orbit & Nav
    "altitude_km": "km", "nav_pos_x_km": "km", "nav_pos_y_km": "km", "nav_pos_z_km": "km",
    "nav_vel_x_kms": "km/s", "nav_vel_y_kms": "km/s", "nav_vel_z_kms": "km/s",
    "sun_angle_deg": "°", "eclipse_depth": "", "shadow_fraction": "", "sun_exposure": "",
    "sun_vector_x": "", "sun_vector_y": "", "sun_vector_z": "",
    # Comms
    "comm_snr_db": "dB", "comm_active": "", "ground_range_km": "km",
    # Propulsion
    "thruster_active": "", "thrust_force_N": "N", "thruster_propellant_kg": "kg", "thruster_burn_time_s": "s",
    "leb_active": "", "leb_thrust_N": "N", "leb_power_W": "W", "leb_propellant_kg": "kg", "leb_burn_time_s": "s",
    # System
    "system_load": "",
}

# Known 65 feature columns (matching parquet data, excludes timestamp + is_anomaly)
FEATURE_COLS = [
    "altitude_km", "nav_pos_x_km", "nav_pos_y_km", "nav_pos_z_km",
    "nav_vel_x_kms", "nav_vel_y_kms", "nav_vel_z_kms",
    "att_q0", "att_q1", "att_q2", "att_q3",
    "att_rate_x_rads", "att_rate_y_rads", "att_rate_z_rads",
    "sun_angle_deg", "eclipse_depth", "shadow_fraction", "sun_exposure",
    "sun_vector_x", "sun_vector_y", "sun_vector_z",
    "battery_voltage_V", "battery_current_A", "battery_soc", "battery_temperature_C",
    "solar_panel_current_A", "solar_panel_watts", "solar_panel_voltage_V",
    "cpu_temperature_C", "rx_temperature_C", "bus_temperature_C", "ambient_temperature_C",
    "heater_cpu_current_A", "heater_batt_current_A", "heater_rx_current_A",
    "thermistor_cpu_setpoint_C", "thermistor_batt_setpoint_C", "thermistor_rx_setpoint_C",
    "heater_bus_voltage_V",
    "system_load", "bus_power_W", "total_power_W",
    "rw_speed_1_rpm", "rw_speed_2_rpm", "rw_speed_3_rpm", "rw_speed_4_rpm",
    "rw_torque_1_nm", "rw_torque_2_nm", "rw_torque_3_nm", "rw_torque_4_nm",
    "gyro_rate_x_dps", "gyro_rate_y_dps", "gyro_rate_z_dps",
    "comm_snr_db", "comm_active", "ground_range_km",
    "thruster_active", "thrust_force_N", "thruster_propellant_kg", "thruster_burn_time_s",
    "leb_active", "leb_thrust_N", "leb_power_W", "leb_propellant_kg", "leb_burn_time_s",
]

# Subsystem dims for the subsystem variant (from ablation notebook)
SUBSYSTEM_DIMS = [1, 6, 4, 3, 7, 4, 3, 4, 7, 3, 8, 3, 3, 4, 5]

# ============================================================================
# SECTION 3: MODEL DEFINITIONS (copied from notebooks)
# ============================================================================

class LSTMAutoencoder(nn.Module):
    """Standard LSTM Autoencoder — ph2_preprocessing_sim66.ipynb Cell 19."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        seq_len = x.size(1)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        return self.output_layer(decoded)


class SubsystemLSTMAutoencoder(nn.Module):
    """Per-subsystem LSTM Autoencoder — SUBSYS-001/ph2_ablation_subsystems.ipynb Cell 19."""
    def __init__(self, subsystem_dims, hidden_dim=128, num_layers=1):
        super().__init__()
        self.subsystem_dims = subsystem_dims
        self.hidden_dim = hidden_dim

        self.encoders = nn.ModuleList([
            nn.LSTM(input_size=dim, hidden_size=hidden_dim,
                    num_layers=num_layers, batch_first=True)
            for dim in subsystem_dims
        ])
        self.decoders = nn.ModuleList([
            nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                    num_layers=num_layers, batch_first=True)
            for dim in subsystem_dims
        ])
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, dim)
            for dim in subsystem_dims
        ])

    def forward(self, x):
        splits = torch.split(x, self.subsystem_dims, dim=2)
        outputs = []
        for i, split_x in enumerate(splits):
            _, (hidden, cell) = self.encoders[i](split_x)
            seq_len = split_x.size(1)
            decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
            decoded, _ = self.decoders[i](decoder_input, (hidden, cell))
            out = self.output_layers[i](decoded)
            outputs.append(out)
        return torch.cat(outputs, dim=2)


# ============================================================================
# SECTION 4: ANOMALY PIPELINE ENGINE
# ============================================================================

class AnomalyPipeline:
    """Manages the full ML anomaly detection pipeline.

    Responsibilities:
      - Load trained model + config from a results directory
      - Fit / load a StandardScaler on training data (with variance floor)
      - Compute clean-validation statistics (mu_rec_err, sigma_rec_err)
      - Maintain a rolling window buffer of scaled frames
      - Score each new window and apply thresholding

    Mirrors the notebook pipeline:
      ph2_preprocessing_sim66.ipynb  (Cells 14, 24, 26)
      SUBSYS-001/ph2_ablation_subsystems.ipynb
    """

    def __init__(self):
        self.model = None
        self.scaler: StandardScaler | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {}

        # Hyper-parameters (overridden when config loaded)
        self.window_size = 300
        self.top_k = 3
        self.min_rec_sigma = 0.05
        self.min_std = 0.001
        self.clip_scaled_to = 10.0
        self.persistence_window = 3

        # Threshold (loaded from metrics.json or user-set)
        self.threshold = 1.0

        # Clean-validation stats for z-scoring
        self.mu_rec_err: np.ndarray | None = None   # shape (n_features,)
        self.sigma_rec_err: np.ndarray | None = None

        # Feature column ordering (may be learned from training data)
        self.feature_cols: list[str] = FEATURE_COLS.copy()

        # Window buffer: deque of scaled feature vectors, max WINDOW_SIZE
        self.window_buffer: deque = deque(maxlen=300)

        # Persistence tracking
        self._recent_predictions: deque[int] = deque(maxlen=50)
        self._warmup_done = False   # True once window buffer is full once

        # Scaler ready flag
        self.scaler_ready = False
        self.model_ready = False
        self.stats_ready = False
        self.ready = False

    # ── Loading helpers ────────────────────────────────────────────────

    def load_config(self, config_path: str) -> dict:
        """Load config.json and extract pipeline hyper-parameters."""
        with open(config_path) as f:
            self.config = json.load(f)

        mdl = self.config.get("model", {})
        dl  = self.config.get("dataloader", {})
        met = self.config.get("metrics", {})

        self.window_size = int(dl.get("window_size", 300))
        self.top_k       = int(met.get("top_k", 3))
        self.min_rec_sigma = float(met.get("min_rec_sigma", 0.05))
        self.min_std     = float(met.get("min_std", 0.001))
        self.clip_scaled_to = float(met.get("clip_scaled_to", 10.0))
        self.persistence_window = int(met.get("persistence_window", 3))

        # Reset buffer to new window size
        self.window_buffer = deque(maxlen=self.window_size)

        return self.config

    def load_model(self, model_path: str, architecture: str | None = None):
        """Load trained model state dict into the correct architecture."""
        if architecture is None:
            architecture = self.config.get("model", {}).get(
                "architecture", "LSTM_Autoencoder"
            )

        input_dim   = int(self.config.get("model", {}).get("parameter_size", 65))
        hidden_dim  = int(self.config.get("model", {}).get("hidden_size", 128))
        num_layers  = int(self.config.get("model", {}).get("num_layers", 1))

        if architecture == "SubsystemLSTMAutoencoder":
            self.model = SubsystemLSTMAutoencoder(
                subsystem_dims=SUBSYSTEM_DIMS,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        else:
            # Default to LSTMAutoencoder
            self.model = LSTMAutoencoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.model_ready = True

    def load_threshold(self, metrics_path: str):
        """Load the locked threshold from metrics.json."""
        with open(metrics_path) as f:
            metrics = json.load(f)
        self.threshold = float(metrics.get("threshold", 1.0))

    # ── Scaler ────────────────────────────────────────────────────────

    def fit_scaler(self, train_df):
        """Fit a StandardScaler on training data with variance floor.

        Mirrors ph2_preprocessing_sim66.ipynb Cell 14.
        """
        # Use feature columns from the dataframe (drop timestamp + is_anomaly)
        drop_cols = {"timestamp", "is_anomaly"}
        self.feature_cols = [c for c in train_df.columns if c not in drop_cols]

        X = train_df[self.feature_cols].values.astype(np.float64)

        self.scaler = StandardScaler()
        self.scaler.fit(X)

        # Variance floor for near-constant columns
        self.scaler.scale_ = np.where(
            self.scaler.scale_ < self.min_std, 1.0, self.scaler.scale_
        )

        self.scaler_ready = True

    def load_scaler(self, scaler_path: str):
        """Load a pre-saved StandardScaler from a pickle file."""
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        if hasattr(self.scaler, "feature_names_in_"):
            self.feature_cols = list(self.scaler.feature_names_in_)
        self.scaler_ready = True

    def transform_frame(self, row_dict: dict) -> np.ndarray | None:
        """Extract features from a live frame dict, scale, and clip.

        Returns shape (n_features,) scaled vector, or None if scaler not ready.
        """
        if not self.scaler_ready or self.scaler is None:
            return None

        vals = []
        for i, col in enumerate(self.feature_cols):
            if col in row_dict:
                vals.append(float(row_dict[col]))
            else:
                # Fill missing features with training mean to avoid bias
                vals.append(float(self.scaler.mean_[i]))

        arr = np.array(vals, dtype=np.float64).reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            scaled = self.scaler.transform(arr)
        clipped = np.clip(scaled, -self.clip_scaled_to, self.clip_scaled_to)
        return clipped.astype(np.float32).ravel()

    # ── Clean-validation stats ────────────────────────────────────────

    def compute_clean_stats(self, valid_loader, device=None):
        """Compute mu_rec_err and sigma_rec_err from clean validation windows.

        Mirrors Cell 24 of ph2_preprocessing_sim66.ipynb:
          - Get per-channel errors on ALL validation windows
          - Clean mask = windows with NO anomaly label
          - mu_rec_err = mean error over clean windows per channel
          - sigma_rec_err = std, floored at min_rec_sigma

        Parameters
        ----------
        valid_loader : iterator yielding (x: Tensor[batch, window, feat], labels)
            If labels are per-frame, wrap them with window_any_anomaly first.
        device : torch.device | None

        Returns dict with mu_rec_err, sigma_rec_err arrays.
        """
        if not self.model_ready or self.scaler_ready is False:
            raise RuntimeError("Load model and scaler before computing clean stats")

        dev = device or self.device
        errs_all = []
        labels_all = []

        with torch.no_grad():
            for x, lbl in valid_loader:
                x = x.to(dev, non_blocking=True)
                recon = self.model(x)
                err = ((recon - x) ** 2).mean(dim=1)  # (batch, n_features)
                errs_all.append(err.cpu().numpy())
                labels_all.append(lbl)

        errs_all = np.concatenate(errs_all, axis=0)       # (n_windows, feat)
        labels_all = np.concatenate(labels_all, axis=0)   # (n_windows,) or (n_frames,)

        # If labels are per-frame and longer than errors, apply window_any_anomaly
        if len(labels_all) > len(errs_all):
            win_size = self.window_size
            raw = labels_all[:win_size * (len(errs_all) + win_size - 1)]
            # Sliding-window "any anomaly"
            swv = np.lib.stride_tricks.sliding_window_view(raw, window_shape=win_size)
            labels_all = swv.max(axis=1).astype(np.int8)

        clean_mask = (labels_all == 0)
        if not np.any(clean_mask):
            raise ValueError("No clean windows in validation data; cannot compute stats")

        mu = errs_all[clean_mask].mean(axis=0)
        raw_sigma = errs_all[clean_mask].std(axis=0)
        sigma = np.maximum(raw_sigma, self.min_rec_sigma)

        self.mu_rec_err = mu
        self.sigma_rec_err = sigma
        self.stats_ready = True

        return {"mu_rec_err": mu, "sigma_rec_err": sigma, "n_clean": int(clean_mask.sum())}

    def load_stats(self, stats_path: str):
        """Load pre-computed clean-stats (mu, sigma) from numpy .npz file."""
        data = np.load(stats_path)
        self.mu_rec_err = data["mu_rec_err"]
        self.sigma_rec_err = data["sigma_rec_err"]
        self.stats_ready = True

    def compute_clean_stats_from_df(self, valid_df):
        """Compute mu_rec_err and sigma_rec_err from a validation DataFrame.

        Mirrors Cell 24 of ph2_preprocessing_sim66.ipynb:
          - Drop is_anomaly column → features only
          - Scale with fitted scaler + clip
          - Build sliding windows (shape: n_windows, WINDOW, feat)
          - Run through model → per-channel MSE
          - Clean mask from is_anomaly labels
          - Compute mu, sigma over clean windows
        """
        if not self.model_ready or not self.scaler_ready:
            raise RuntimeError("Load model and scaler before computing clean stats")

        valid_labels = valid_df["is_anomaly"].to_numpy(dtype=np.int8)
        drop_cols = {"timestamp", "is_anomaly"}
        feat_cols = [c for c in valid_df.columns if c not in drop_cols]
        feat_data = valid_df[feat_cols].values.astype(np.float64)

        # Scale + clip (Cell 14)
        scaled = self.scaler.transform(feat_data)
        clipped = np.clip(scaled, -self.clip_scaled_to,
                          self.clip_scaled_to).astype(np.float32)

                # Build sliding windows lazily via stride tricks
        #   sliding_window_view with axis=0 returns (n_windows, F, W)
        #   so we transpose to (n_windows, W, F) for the LSTM input
        W = self.window_size
        n_windows = len(clipped) - W + 1
        windows = np.lib.stride_tricks.sliding_window_view(
            clipped, window_shape=(W,), axis=0
        )[:n_windows].transpose(0, 2, 1).astype(np.float32)

        # Run through model in batches (Cell 24 per_channel_errors)
        batch_size = 128
        errs_all = []
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                chunk = windows[i:i + batch_size]
                x_t = torch.from_numpy(chunk).to(self.device)
                recon = self.model(x_t)
                err = ((recon - x_t) ** 2).mean(dim=1)  # (batch, feat)
                errs_all.append(err.cpu().numpy())

        errs_all = np.concatenate(errs_all, axis=0)  # (n_windows, feat)

        # Window-level labels: any anomaly in window → anomalous
        swv = np.lib.stride_tricks.sliding_window_view(
            valid_labels, window_shape=W
        )
        window_labels = swv.max(axis=1).astype(np.int8)[:n_windows]

        clean_mask = (window_labels == 0)
        if not np.any(clean_mask):
            raise ValueError("No clean windows in validation data")

        mu = errs_all[clean_mask].mean(axis=0)
        raw_sigma = errs_all[clean_mask].std(axis=0)
        sigma = np.maximum(raw_sigma, self.min_rec_sigma)

        self.mu_rec_err = mu
        self.sigma_rec_err = sigma
        self.stats_ready = True

        return {"n_clean": int(clean_mask.sum()), "n_total": n_windows}

    # ── Scoring ───────────────────────────────────────────────────────

    @staticmethod
    def channelwise_topk_score(errors_per_channel: np.ndarray,
                                mu: np.ndarray, sigma: np.ndarray,
                                top_k: int) -> float:
        """Mean of the largest top_k channel z-scores.

        Mirrors Cell 24 in ph2_preprocessing_sim66.ipynb.
        """
        z = (errors_per_channel - mu) / sigma  # shape (feat,) for a single window
        topk_z = np.partition(z, -top_k)[-top_k:]
        return float(topk_z.mean())

    def score_window(self, window_array: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Pass a single window through the model and compute anomaly score.

        Parameters
        ----------
        window_array : np.ndarray of shape (window_size, n_features), dtype float32

        Returns
        -------
        (score, recon_err_mean, per_channel_err)
          score: float anomaly score (top-K z-score if stats loaded, else raw MSE)
          recon_err_mean: float mean MSE across all channels
          per_channel_err: np.ndarray of shape (n_features,) with per-channel MSE
        """
        if not self.model_ready:
            return 0.0, 0.0, np.zeros(len(self.feature_cols), dtype=np.float32)

        x = torch.from_numpy(window_array).unsqueeze(0).to(self.device)  # (1, W, F)

        with torch.no_grad():
            recon = self.model(x)
            per_channel_err = ((recon - x) ** 2).mean(dim=1).cpu().numpy()[0]  # (F,)

        recon_err_mean = float(per_channel_err.mean())

        if self.stats_ready:
            score = self.channelwise_topk_score(
                per_channel_err, self.mu_rec_err, self.sigma_rec_err, self.top_k
            )
        else:
            # Fallback: raw mean of top-K channel errors (no z-scoring)
            sorted_err = np.sort(per_channel_err)[-self.top_k:]
            score = float(sorted_err.mean())

        return score, recon_err_mean, per_channel_err

    def check_anomaly(self, score: float) -> bool:
        """Determine if a window is anomalous using threshold + persistence."""
        raw_pred = int(score >= self.threshold)
        self._recent_predictions.append(raw_pred)

        if self.persistence_window <= 1:
            return bool(raw_pred)

        if len(self._recent_predictions) < self.persistence_window:
            # Not enough history to confirm persistence
            return False

        # Forward + backward persistence check (mirrors notebook Cell 26)
        recent = list(self._recent_predictions)[-self.persistence_window:]
        if all(recent):
            return True
        return False

    @property
    def is_buffer_ready(self) -> bool:
        return len(self.window_buffer) >= self.window_size

    @property
    def current_window_array(self) -> np.ndarray | None:
        """Return the current window as (window_size, n_features) array."""
        if not self.is_buffer_ready:
            return None
        return np.array(list(self.window_buffer), dtype=np.float32)


# ============================================================================
# SECTION 5: LIVE PLOT WIDGET
# ============================================================================

class SubsystemPlotWidget(QWidget):
    """Matplotlib-based live plot widget for a selected subsystem."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(8, 4), dpi=100, facecolor='#181825')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self._style_axis(self.ax)
        self.ax.set_xlabel("Frame", fontsize=9, color="#cdd6f4")
        self.ax.set_ylabel("Value", fontsize=9, color="#cdd6f4")
        self.ax.set_title("Live Telemetry", fontsize=10, fontweight='bold', color="#cdd6f4")

        self._history = {}
        self._frame_nums = deque(maxlen=60)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    @staticmethod
    def _style_axis(ax):
        ax.set_facecolor('#1e1e2e')
        ax.grid(True, alpha=0.3, color="#45475a", linestyle="--")
        ax.tick_params(colors='#bac2de', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#313244')

    def set_subsystem(self, params):
        self.ax.clear()
        self._style_axis(self.ax)
        self.ax.set_xlabel("Frame", fontsize=9, color="#cdd6f4")
        self.ax.set_ylabel("Value", fontsize=9, color="#cdd6f4")
        self.ax.set_title(
            f"Live Telemetry — {' + '.join(params)}",
            fontsize=10, fontweight='bold', color="#cdd6f4",
        )
        self._history = {p: deque(maxlen=60) for p in params}
        self._frame_nums.clear()
        self.canvas.draw()

    def add_frame(self, frame, frame_num):
        self._frame_nums.append(frame_num)
        for param, values in self._history.items():
            if param in frame:
                values.append(frame[param])
            elif len(values) > 0:
                values.append(values[-1])

        self.ax.clear()
        self._style_axis(self.ax)
        self.ax.set_xlabel("Frame", fontsize=9, color="#cdd6f4")
        self.ax.set_ylabel("Value", fontsize=9, color="#cdd6f4")
        self.ax.set_title(
            f"Live Telemetry — {' + '.join(list(self._history.keys()))}",
            fontsize=10, fontweight='bold', color="#cdd6f4",
        )

        plotted = False
        colors = plt_colors()
        for i, (param, values) in enumerate(self._history.items()):
            if len(values) > 1:
                xs = list(self._frame_nums)[-len(values):]
                self.ax.plot(xs, values, color=colors[i % len(colors)],
                             linewidth=1.4, label=param, alpha=0.9)
                plotted = True

        if plotted:
            leg = self.ax.legend(loc='upper left', fontsize=8, facecolor="#1e1e2e", edgecolor="#313244")
            if leg:
                for text in leg.get_texts():
                    text.set_color("#cdd6f4")
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.canvas.draw()


class ScorePlotWidget(QWidget):
    """Live plot for anomaly scores with threshold line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(8, 2.5), dpi=100, facecolor='#181825')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.ax = self.fig.add_subplot(111)
        self._style_axis(self.ax)
        self.ax.set_xlabel("Frame", fontsize=9, color="#cdd6f4")
        self.ax.set_ylabel("Anomaly Score", fontsize=9, color="#cdd6f4")
        self.ax.set_title("Anomaly Score per Window", fontsize=10, fontweight='bold', color="#cdd6f4")

        self._scores = deque(maxlen=200)  # (frame_num, score, is_anomaly)
        self._threshold = 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    @staticmethod
    def _style_axis(ax):
        ax.set_facecolor('#1e1e2e')
        ax.grid(True, alpha=0.3, color="#45475a", linestyle="--")
        ax.tick_params(colors='#bac2de', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#313244')

    def add_point(self, frame_num, score, is_anom):
        self._scores.append((frame_num, score, is_anom))
        self._redraw()

    def set_threshold_line(self, threshold: float):
        self._threshold = threshold
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        self._style_axis(self.ax)
        self.ax.set_xlabel("Frame", fontsize=9, color="#cdd6f4")
        self.ax.set_ylabel("Anomaly Score", fontsize=9, color="#cdd6f4")
        self.ax.set_title("Anomaly Score per Window", fontsize=10, fontweight='bold', color="#cdd6f4")

        if not self._scores:
            self.canvas.draw()
            return

        frames = [s[0] for s in self._scores]
        scores = [s[1] for s in self._scores]

        # Plot score line
        self.ax.plot(frames, scores, color="#89b4fa", linewidth=1.5, alpha=0.85, label="Score")

        # Mark anomalous points
        anom_frames = [s[0] for s in self._scores if s[2]]
        anom_scores = [s[1] for s in self._scores if s[2]]
        if anom_frames:
            self.ax.scatter(anom_frames, anom_scores, color="#f38ba8", s=25, zorder=5, label="Anomaly")

        # Threshold line
        if hasattr(self, '_threshold') and self._threshold is not None:
            self.ax.axhline(y=self._threshold, color="#f9e2af", linestyle="--",
                            linewidth=1.5, label=f"Threshold ({self._threshold:.3f})")

        leg = self.ax.legend(loc='upper left', fontsize=7, facecolor="#1e1e2e", edgecolor="#313244")
        if leg:
            for text in leg.get_texts():
                text.set_color("#cdd6f4")
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        self.canvas.draw()


def plt_colors():
    return ['#89b4fa', '#a6e3a1', '#f9e2af', '#fab387', '#cba6f7',
            '#f38ba8', '#94e2d5', '#74c7ec', '#b4befe', '#f5c2e7']


# ============================================================================
# SECTION 6: MAIN DASHBOARD WINDOW
# ============================================================================

class SatelliteDashboard(QMainWindow):
    """Full PyQt6 dashboard with ML-based anomaly detection pipeline."""

    def __init__(self, experiment_dir):
        super().__init__()
        self.setWindowTitle("🛰️ LEO Satellite Telemetry Dashboard — ML Pipeline")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 950)

        # Paths ────────────────────────────────────────────────────────
        self.experiment_dir = Path(experiment_dir).resolve()
        self.experiment_id  = self.experiment_dir.name
        script_dir = Path(__file__).resolve().parent
        candidate_sim = script_dir / "sim66.py"
        if not candidate_sim.exists():
            candidate_sim = Path("sim66.py").resolve()
        if not candidate_sim.exists():
            candidate_sim = Path("simdata/sim66.py").resolve()
        self.sim_path       = str(candidate_sim)
        self.results_save_dir = script_dir
        self.simulator = None
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.start_time = QTime()

        # Pipeline engine
        self.pipeline = AnomalyPipeline()

        # UI state
        self._anomaly_count = 0
        self._ok_count = 0

        # Results collection (saved on stop)
        self._results = []  # list of {window, score, recon_err, anomaly, time}

        # ── Build UI ─────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        self._build_top_section(main_layout)     # Mission status + file selectors
        self._build_pipeline_panel(main_layout)   # Pipeline init controls + status
        self._build_middle_section(main_layout)   # Telemetry table + plots
        self._build_bottom_section(main_layout)   # Anomaly score table + log

        # Bottom-most status bar: pipeline ready status on left, anomaly score & count on right
        self.statusBar().setStyleSheet(
            "QStatusBar { background: #181825; color: #ffffff; font-weight: bold; border-top: 1px solid #313244; }"
            "QStatusBar::item { border: none; }"
        )
        self.status_score_label = QLabel("Anomaly Score: --")
        self.status_score_label.setStyleSheet(
            "color: #89b4fa; font-weight: bold; padding: 2px 10px; background: #1e1e2e; border-radius: 4px; margin-right: 6px;"
        )
        self.status_anom_label = QLabel("Anomalies: 0")
        self.status_anom_label.setStyleSheet(
            "color: #a6e3a1; font-weight: bold; padding: 2px 10px; background: #1e1e2e; border-radius: 4px; margin-right: 8px;"
        )
        self.statusBar().addPermanentWidget(self.status_score_label)
        self.statusBar().addPermanentWidget(self.status_anom_label)
        self.anom_count_label = self.status_anom_label

        self.statusBar().showMessage(
            "Ready — load a model and start the pipeline"
        )

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(1000)

        # Auto-initialize pipeline on startup
        self._init_pipeline(quiet=True)

    # ─── TOP SECTION: Mission Status + File Selectors ───────────────────
    def _build_top_section(self, parent_layout):
        top = QFrame()
        top.setFrameShape(QFrame.Shape.StyledPanel)
        top.setStyleSheet(
            "QFrame { background: #2D4A9A; color: #ffffff; border-radius: 6px; padding: 8px; }"
            "QLabel { color: #ffffff; }"
        )
        top_layout = QVBoxLayout(top)

        # Title
        title_row = QHBoxLayout()
        title_label = QLabel("🛰️ LEO SATELLITE TELEMETRY DASHBOARD — ML ANOMALY PIPELINE")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title_row.addWidget(title_label)
        title_row.addStretch()
        top_layout.addLayout(title_row)

        # Experiment ID display (read-only)
        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Experiment:"))
        self.experiment_label = QLabel(self.experiment_id)
        self.experiment_label.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 4px 12px; color: #89b4fa;"
        )
        exp_row.addWidget(self.experiment_label)
        exp_row.addStretch()
        top_layout.addLayout(exp_row)

        # Controls row
        ctrl = QHBoxLayout()

        self.status_label = QLabel("⏸ IDLE")
        self.status_label.setStyleSheet(
            "font-weight: bold; padding: 4px 12px; border-radius: 4px; background: #585b70; color: #ffffff;"
        )
        ctrl.addWidget(self.status_label)

        self.time_label = QLabel("Elapsed: 00:00:00")
        ctrl.addWidget(self.time_label)
        self.frame_label = QLabel("Frames: 0")
        ctrl.addWidget(self.frame_label)
        ctrl.addStretch()

        self.btn_start = QPushButton("▶ Start")
        self.btn_start.setStyleSheet(
            "padding: 6px 16px; font-weight: bold; background: #a6e3a1;"
            " color: #1e1e2e; border-radius: 4px;"
        )
        self.btn_start.clicked.connect(self._start_simulation)
        ctrl.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setStyleSheet(
            "padding: 6px 16px; font-weight: bold; background: #f9e2af;"
            " color: #1e1e2e; border-radius: 4px;"
        )
        self.btn_pause.clicked.connect(self._pause_simulation)
        self.btn_pause.setEnabled(False)
        ctrl.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setStyleSheet(
            "padding: 6px 16px; font-weight: bold; background: #f38ba8;"
            " color: #1e1e2e; border-radius: 4px;"
        )
        self.btn_stop.clicked.connect(self._stop_simulation)
        self.btn_stop.setEnabled(False)
        ctrl.addWidget(self.btn_stop)

        top_layout.addLayout(ctrl)

        # Threshold + init button row
        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Threshold:"))
        self.threshold_edit = QLineEdit("1.0")
        self.threshold_edit.setFixedWidth(80)
        self.threshold_edit.setStyleSheet("color: #ffffff; background: #1e1e2e; padding: 3px; border: 1px solid #45475a;")
        thr_row.addWidget(self.threshold_edit)

        btn_init = QPushButton("⚙ Init Pipeline")
        btn_init.setStyleSheet(
            "padding: 6px 16px; font-weight: bold; background: #89b4fa;"
            " color: #1e1e2e; border-radius: 4px;"
        )
        btn_init.clicked.connect(self._init_pipeline)
        thr_row.addWidget(btn_init)
        thr_row.addStretch()
        top_layout.addLayout(thr_row)

        parent_layout.addWidget(top, 3)

    # ─── PIPELINE STATUS PANEL ──────────────────────────────────────────
    def _build_pipeline_panel(self, parent_layout):
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setStyleSheet("""
            QFrame { background: #1e1e2e; color: #ffffff; border-radius: 6px; }
            QGroupBox { color: #ffffff; font-weight: bold; font-size: 11px; }
            QGroupBox::title { color: #ffffff; font-weight: bold; }
        """)
        p_layout = QHBoxLayout(panel)
        p_layout.setContentsMargins(4, 4, 4, 4)

        labels = [
            ("Config", self._pipeline_status_label("config")),
            ("Model", self._pipeline_status_label("model")),
            ("Scaler", self._pipeline_status_label("scaler")),
            ("Stats", self._pipeline_status_label("stats")),
            (f"Window ({self.pipeline.window_size})",
             self._pipeline_status_label("buffer")),
        ]

        self.pipe_group_boxes = {}
        for title, lbl in labels:
            grp = QGroupBox(title)
            g_layout = QVBoxLayout(grp)
            g_layout.addWidget(lbl)
            p_layout.addWidget(grp, 1)
            comp_name = lbl.objectName().replace("pipe_lbl_", "")
            self.pipe_group_boxes[comp_name] = grp

        parent_layout.addWidget(panel, 1)

    @staticmethod
    def _pipeline_status_label(component: str) -> QLabel:
        lbl = QLabel(f"⏸ {component.upper()} not loaded")
        lbl.setStyleSheet("color: #f9e2af; padding: 4px; font-weight: bold;")
        lbl.setObjectName(f"pipe_lbl_{component}")
        return lbl

    def _update_pipeline_status(self):
        p = self.pipeline
        if hasattr(self, "pipe_group_boxes") and "buffer" in self.pipe_group_boxes:
            self.pipe_group_boxes["buffer"].setTitle(f"Window ({p.window_size})")
        items = {
            "config": ("✅ Config loaded" if p.config else "⏸ Config not loaded",
                       "#a6e3a1" if p.config else "#f9e2af"),
            "model": (f"✅ {getattr(self.pipeline.model, '__class__', '???').__name__}" if p.model_ready else "⏸ Model not loaded",
                      "#a6e3a1" if p.model_ready else "#f9e2af"),
            "scaler": ("✅ Scaler fitted" if p.scaler_ready else "⏸ Scaler not fitted",
                       "#a6e3a1" if p.scaler_ready else "#f9e2af"),
            "stats": ("✅ Stats loaded" if p.stats_ready else "⏸ Stats not loaded",
                      "#a6e3a1" if p.stats_ready else "#f9e2af"),
            "buffer": (f"📦 Buffer {len(p.window_buffer)}/{p.window_size}",
                       "#89b4fa"),
        }

        for component, (text, color) in items.items():
            lbl = self.findChild(QLabel, f"pipe_lbl_{component}")
            if lbl:
                lbl.setText(text)
                lbl.setStyleSheet(f"color: {color}; padding: 4px; font-weight: bold;")

    # ─── MIDDLE SECTION ────────────────────────────────────────────────
    def _build_middle_section(self, parent_layout):
        mid = QFrame()
        mid.setFrameShape(QFrame.Shape.StyledPanel)
        mid.setStyleSheet("""
            QFrame { background: #2D4A9A; color: #ffffff; border-radius: 6px; padding: 6px; }
            QGroupBox { color: #ffffff; font-weight: bold; font-size: 12px; }
            QGroupBox::title { color: #ffffff; font-weight: bold; }
            QLabel { color: #ffffff; font-weight: bold; }
        """)
        mid_layout = QHBoxLayout(mid)

        # LEFT: Telemetry Stream
        stream_group = QGroupBox("Telemetry Stream")
        stream_layout = QVBoxLayout(stream_group)

        self.telem_table = QTableWidget()
        self.telem_table.setColumnCount(3)
        self.telem_table.setHorizontalHeaderLabels(["Subsystem", "Parameter", "Value"])
        self.telem_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.telem_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.telem_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.telem_table.setColumnWidth(0, 110)
        self.telem_table.setColumnWidth(1, 180)
        self.telem_table.verticalHeader().setVisible(False)
        self.telem_table.setStyleSheet("""
            QTableWidget { background: #1e1e2e; gridline-color: #313244; color: #ffffff; }
            QTableWidget::item { padding: 2px 4px; }
            QHeaderView::section { background: #313244; color: #ffffff; font-weight: bold; padding: 4px; }
        """)
        stream_layout.addWidget(self.telem_table)
        mid_layout.addWidget(stream_group, 1)

        # RIGHT: Plots + Score plot
        right_group = QWidget()
        right_layout = QVBoxLayout(right_group)

        # Subsystem selector
        plot_ctrl = QHBoxLayout()
        plot_ctrl.addWidget(QLabel("Subsystem:"))
        self.subsystem_combo = QComboBox()
        self.subsystem_combo.addItems(list(SUBSYSTEMS.keys()))
        self.subsystem_combo.currentTextChanged.connect(self._on_subsystem_changed)
        plot_ctrl.addWidget(self.subsystem_combo)
        plot_ctrl.addStretch()
        right_layout.addLayout(plot_ctrl)

        # Telemetry plot
        self.plot_widget = SubsystemPlotWidget()
        self.plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.plot_widget)

        # Anomaly score plot
        self.score_plot_widget = ScorePlotWidget()
        right_layout.addWidget(self.score_plot_widget)

        mid_layout.addWidget(right_group, 2)
        parent_layout.addWidget(mid, 9)

    # ─── BOTTOM SECTION ────────────────────────────────────────────────
    def _build_bottom_section(self, parent_layout):
        bot = QFrame()
        bot.setFrameShape(QFrame.Shape.StyledPanel)
        bot.setStyleSheet("""
            QFrame { background: #2D4A9A; color: #ffffff; border-radius: 6px; padding: 6px; }
            QGroupBox { color: #ffffff; font-weight: bold; font-size: 12px; }
            QGroupBox::title { color: #ffffff; font-weight: bold; }
            QLabel { color: #ffffff; font-weight: bold; }
        """)
        bot_layout = QVBoxLayout(bot)
        bot_layout.setContentsMargins(6, 6, 6, 6)

        # ── Anomaly Score Table ──
        anomaly_group = QGroupBox("Anomaly Scores (last 50 windows)")
        anomaly_layout = QVBoxLayout(anomaly_group)

        self.score_table = QTableWidget()
        self.score_table.setColumnCount(7)
        self.score_table.setHorizontalHeaderLabels([
            "Window", "Score", "Recon Err", "Status",
            "Prediction", "Time", "Details"
        ])
        for col in range(5):
            self.score_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.Fixed)
        self.score_table.setColumnWidth(0, 60)
        self.score_table.setColumnWidth(1, 80)
        self.score_table.setColumnWidth(2, 80)
        self.score_table.setColumnWidth(3, 80)
        self.score_table.setColumnWidth(4, 70)
        self.score_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch)
        self.score_table.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.ResizeMode.Stretch)
        self.score_table.verticalHeader().setVisible(False)
        self.score_table.setMaximumHeight(160)
        self.score_table.setStyleSheet("""
            QTableWidget { background: #181825; gridline-color: #313244; color: #ffffff; }
            QTableWidget::item { padding: 2px; }
            QHeaderView::section { background: #313244; color: #ffffff; font-weight: bold; padding: 3px; }
        """)
        anomaly_layout.addWidget(self.score_table)
        bot_layout.addWidget(anomaly_group, 1)

        # ── Anomaly Log ──
        log_group = QGroupBox("Anomaly Event Log")
        log_layout = QVBoxLayout(log_group)

        self.anomaly_log = QTableWidget()
        self.anomaly_log.setColumnCount(5)
        self.anomaly_log.setHorizontalHeaderLabels([
            "Time", "Window", "Score", "Recon Err", "Details"
        ])
        for col in range(4):
            self.anomaly_log.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.Fixed)
        self.anomaly_log.setColumnWidth(0, 80)
        self.anomaly_log.setColumnWidth(1, 60)
        self.anomaly_log.setColumnWidth(2, 70)
        self.anomaly_log.setColumnWidth(3, 80)
        self.anomaly_log.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch)
        self.anomaly_log.verticalHeader().setVisible(False)
        self.anomaly_log.setMaximumHeight(220)
        self.anomaly_log.setStyleSheet("""
            QTableWidget { background: #181825; gridline-color: #313244; color: #ffffff; }
            QTableWidget::item { padding: 2px; }
            QHeaderView::section { background: #313244; color: #ffffff; font-weight: bold; padding: 3px; }
        """)
        log_layout.addWidget(self.anomaly_log)
        bot_layout.addWidget(log_group, 1)

        parent_layout.addWidget(bot, 8)

    # ─── FILE SELECTORS & PIPELINE INIT ────────────────────────────────

    def _init_pipeline(self, quiet: bool = False) -> bool:
        """Load config + model + scaler + clean stats from the experiment directory."""
        exp_dir = self.experiment_dir

        # Support both config.json and result_config.json
        cfg_path = exp_dir / "config.json"
        if not cfg_path.exists():
            cfg_path = exp_dir / "result_config.json"

        model_path = exp_dir / "best_model.pt"
        scaler_path = exp_dir / "scaler.pkl"
        stats_path = exp_dir / "clean_stats.npz"
        metrics_path = exp_dir / "metrics.json"

        if not cfg_path.exists():
            if not quiet:
                QMessageBox.warning(self, "Config Missing",
                                    f"No config.json or result_config.json in:\n{exp_dir}")
            return False
        if not model_path.exists():
            if not quiet:
                QMessageBox.warning(self, "Model Missing",
                                    f"No best_model.pt in:\n{exp_dir}")
            return False
        if not scaler_path.exists():
            if not quiet:
                QMessageBox.warning(self, "Scaler Missing",
                                    f"No scaler.pkl in:\n{exp_dir}")
            return False

        try:
            # 1. Load config
            self.pipeline.load_config(str(cfg_path))

            # 2. Load model (architecture from config only — no manual override)
            self.pipeline.load_model(str(model_path), architecture=None)

            # 3. Load threshold from metrics.json if available
            if metrics_path.exists():
                self.pipeline.load_threshold(str(metrics_path))
                self.threshold_edit.setText(f"{self.pipeline.threshold:.4f}")

            # 4. Load pre-saved scaler
            self.pipeline.load_scaler(str(scaler_path))

            # 5. Load pre-computed clean validation stats (mu, sigma)
            if stats_path.exists():
                self.pipeline.load_stats(str(stats_path))

            self.pipeline.ready = True

            self._update_pipeline_status()
            self.score_plot_widget.set_threshold_line(self.pipeline.threshold)

            self.statusBar().showMessage(
                f"Pipeline ready: {self.experiment_id} | Window={self.pipeline.window_size} | "
                f"Threshold={self.pipeline.threshold:.4f} | Stats={'loaded' if self.pipeline.stats_ready else 'none'}"
            )

            if not quiet:
                QMessageBox.information(
                    self, "Pipeline Ready",
                    f"Experiment: {self.experiment_id}\n\n"
                    f"Config: {cfg_path.name} ✅\n"
                    f"Model: {getattr(self.pipeline.model, '__class__', '???').__name__} ✅\n"
                    f"Scaler: {scaler_path.name} ✅\n"
                    f"Clean Stats: {'clean_stats.npz ✅' if self.pipeline.stats_ready else 'Not found ⚠️'}\n"
                    f"Window size: {self.pipeline.window_size}\n"
                    f"Threshold: {self.pipeline.threshold:.4f}\n"
                    f"Top-K: {self.pipeline.top_k}\n"
                    f"Persistence: {self.pipeline.persistence_window}"
                )
            return True

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            if not quiet:
                QMessageBox.critical(self, "Init Error",
                                     f"Failed to initialise pipeline:\n{e}\n\n{tb[:500]}")
            else:
                self.statusBar().showMessage(f"Pipeline init failed: {e}")
            return False

    # ─── SIMULATION CONTROLS ──────────────────────────────────────────

    def _start_simulation(self):
        """Load simulator and start the loop."""
        # Ensure pipeline is ready before starting
        if not self.pipeline.model_ready or not self.pipeline.scaler_ready:
            if not self._init_pipeline(quiet=False):
                return
        try:
            spec = importlib.util.spec_from_file_location(
                "simulator", self.sim_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.simulator = mod.TelemetrySimulator()
        except Exception as e:
            QMessageBox.critical(self, "Import Error",
                                 f"Failed to load simulator:\n{e}")
            return

        # Parse threshold override
        try:
            user_thresh = float(self.threshold_edit.text())
            self.pipeline.threshold = user_thresh
        except ValueError:
            pass  # keep loaded threshold

        # Reset counters
        self.running = True
        self.paused = False
        self.frame_count = 0
        self.start_time = QTime.currentTime()
        self._anomaly_count = 0
        self._ok_count = 0
        self.pipeline.window_buffer.clear()
        self.pipeline._recent_predictions.clear()

        # Reset UI tables
        self.telem_table.setRowCount(0)
        self.score_table.setRowCount(0)
        self.anomaly_log.setRowCount(0)

        # Button states
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

        self.status_label.setText("▶ RUNNING")
        self.status_label.setStyleSheet(
            "font-weight: bold; padding: 4px 12px; border-radius: 4px;"
            " background: #a6e3a1; color: #1e1e2e;"
        )

        self.status_score_label.setText("Anomaly Score: --")
        self.status_score_label.setStyleSheet(
            "color: #89b4fa; font-weight: bold; padding: 2px 10px; background: #1e1e2e; border-radius: 4px; margin-right: 6px;"
        )
        self.status_anom_label.setText("Anomalies: 0")
        self.status_anom_label.setStyleSheet(
            "color: #a6e3a1; font-weight: bold; padding: 2px 10px; background: #1e1e2e; border-radius: 4px; margin-right: 8px;"
        )

        self.statusBar().showMessage(
            f"Started: {self.experiment_id} — window={self.pipeline.window_size}"
        )

        self.score_plot_widget.set_threshold_line(self.pipeline.threshold)
        self._update_plot_subsystem()
        self.timer.start()

    def _pause_simulation(self):
        if self.paused:
            self.paused = False
            self.btn_pause.setText("⏸ Pause")
            self.status_label.setText("▶ RUNNING")
            self.status_label.setStyleSheet(
                "font-weight: bold; padding: 4px 12px; border-radius: 4px;"
                " background: #a6e3a1; color: #1e1e2e;"
            )
        else:
            self.paused = True
            self.btn_pause.setText("▶ Resume")
            self.status_label.setText("⏸ PAUSED")
            self.status_label.setStyleSheet(
                "font-weight: bold; padding: 4px 12px; border-radius: 4px;"
                " background: #f9e2af; color: #1e1e2e;"
            )

    def _stop_simulation(self):
        self.timer.stop()
        self.running = False
        self.paused = False
        if self.simulator:
            self.simulator.running = False

        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        self.status_label.setText("⏸ IDLE")
        self.status_label.setStyleSheet(
            "font-weight: bold; padding: 4px 12px; border-radius: 4px;"
            " background: #585b70;"
        )
        self.statusBar().showMessage(
            f"Stopped. Frames: {self.frame_count}, "
            f"Anomalies: {self._anomaly_count}"
        )

        # Save results to simdata directory
        self._save_results()

    def _save_results(self):
        """Save dashboard results as JSON into sat_tm/simdata."""
        if not self._results:
            return

        out_path = str(self.results_save_dir / f"dashboard_{self.experiment_id}.json")
        payload = {
            "experiment": self.experiment_id,
            "experiment_dir": str(self.experiment_dir),
            "total_frames": self.frame_count,
            "total_anomalies": self._anomaly_count,
            "total_ok": self._ok_count,
            "window_size": self.pipeline.window_size,
            "threshold": self.pipeline.threshold,
            "top_k": self.pipeline.top_k,
            "persistence_window": self.pipeline.persistence_window,
            "results": self._results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        self.statusBar().showMessage(
            f"Results saved to {out_path}"
        )

    # ─── MAIN TICK — PIPELINE LOOP ──────────────────────────────────────

    def _tick(self):
        if not self.running or self.paused or not self.simulator:
            return

        # 1. Get frame from simulator
        try:
            frame = self.simulator.get_frame()
        except Exception as e:
            self.statusBar().showMessage(f"Error getting frame: {e}")
            return

        frame.pop('timestamp', None)
        self.frame_count += 1

        # 2. Update telemetry display
        self._update_telemetry_table(frame)
        self._update_plot(frame)

        # 3. Scale frame → push to window buffer
        scaled = self.pipeline.transform_frame(frame)
        if scaled is not None:
            self.pipeline.window_buffer.append(scaled)
            self._update_pipeline_status()

            # 4. When buffer full, score the window
            if self.pipeline.is_buffer_ready:
                window_arr = self.pipeline.current_window_array
                if window_arr is not None:
                    score, recon_err, pc_err = self.pipeline.score_window(window_arr)
                    is_anom = self.pipeline.check_anomaly(score)

                    if is_anom:
                        self._anomaly_count += 1
                    else:
                        self._ok_count += 1

                    # Update UI
                    window_idx = max(0, self.frame_count - self.pipeline.window_size + 1)
                    self._add_anomaly_score(window_idx, score, recon_err, is_anom, pc_err)
                    self.score_plot_widget.add_point(window_idx, score, is_anom)

                    # Collect results for saving
                    self._results.append({
                        "window": window_idx,
                        "score": round(score, 6),
                        "recon_err": round(recon_err, 6),
                        "anomaly": bool(is_anom),
                        "time": datetime.now().strftime("%H:%M:%S"),
                    })

                    if is_anom:
                        self._add_anomaly_log(
                            window_idx, score, recon_err, is_anom)

                    # Update bottom-most status bar with latest score & count
                    self.status_score_label.setText(f"Anomaly Score: {score:.4f}")
                    if is_anom:
                        self.status_score_label.setStyleSheet(
                            "color: #f38ba8; font-weight: bold; padding: 2px 10px; background: #3b222c; border-radius: 4px; margin-right: 6px;"
                        )
                        self.status_anom_label.setText(f"Anomalies: {self._anomaly_count}")
                        self.status_anom_label.setStyleSheet(
                            "color: #f38ba8; font-weight: bold; padding: 2px 10px; background: #3b222c; border-radius: 4px; margin-right: 8px;"
                        )
                    else:
                        self.status_score_label.setStyleSheet(
                            "color: #89b4fa; font-weight: bold; padding: 2px 10px; background: #1e1e2e; border-radius: 4px; margin-right: 6px;"
                        )
                        self.status_anom_label.setText(f"Anomalies: {self._anomaly_count}")
                        self.status_anom_label.setStyleSheet(
                            "color: #a6e3a1; font-weight: bold; padding: 2px 10px; background: #1e1e2e; border-radius: 4px; margin-right: 8px;"
                        )

        # Update elapsed time
        elapsed = self.start_time.msecsTo(QTime.currentTime())
        total_secs = elapsed // 1000
        hh = total_secs // 3600
        mm = (total_secs % 3600) // 60
        ss = total_secs % 60
        self.time_label.setText(f"Elapsed: {hh:02d}:{mm:02d}:{ss:02d}")
        self.frame_label.setText(f"Frames: {self.frame_count}")

    # ─── TELEMETRY STREAM ──────────────────────────────────────────────

    def _update_telemetry_table(self, frame):
        rows = []
        for subsystem, params in SUBSYSTEMS.items():
            for param in params:
                if param in frame:
                    val = frame[param]
                    unit = UNITS.get(param, "")
                    if param in ("comm_active", "thruster_active", "leb_active"):
                        val_str = "ACTIVE" if val > 0.5 else "IDLE"
                    elif param == "battery_soc":
                        val_str = f"{val:.1f}%"
                    elif isinstance(val, (float, np.floating)) and (abs(val) > 1000 or (0 < abs(val) < 0.001)):
                        val_str = f"{val:.3g}"
                    elif isinstance(val, (int, np.integer)):
                        val_str = f"{val}"
                    else:
                        val_str = f"{val:.3f}" if abs(val) < 10 else f"{val:.2f}"
                    if unit:
                        val_str += f" {unit}"
                    rows.append((subsystem, param, val_str, param == "battery_soc", frame.get("battery_soc", 100.0)))

        # Also add any leftover keys in frame that were not in SUBSYSTEMS
        added_params = {r[1] for r in rows}
        for k, val in frame.items():
            if k not in added_params and k not in ("timestamp",):
                val_str = f"{val:.3g}" if isinstance(val, (float, np.floating)) else str(val)
                rows.append(("Other", k, val_str, False, 1.0))

        # Re-use existing table items when row count matches for smooth rendering
        if self.telem_table.rowCount() != len(rows):
            self.telem_table.setRowCount(len(rows))
            for i, (subsys, param, val_str, is_soc, soc_val) in enumerate(rows):
                self.telem_table.setItem(i, 0, QTableWidgetItem(subsys))
                self.telem_table.setItem(i, 1, QTableWidgetItem(param))
                item_val = QTableWidgetItem(val_str)
                self.telem_table.setItem(i, 2, item_val)
        else:
            for i, (subsys, param, val_str, is_soc, soc_val) in enumerate(rows):
                self.telem_table.item(i, 0).setText(subsys)
                self.telem_table.item(i, 1).setText(param)
                self.telem_table.item(i, 2).setText(val_str)

        # Color-code battery SOC
        for i, (subsys, param, val_str, is_soc, soc_val) in enumerate(rows):
            if is_soc:
                item = self.telem_table.item(i, 2)
                if item:
                    if soc_val > 50:
                        item.setForeground(QColor("#a6e3a1"))
                    elif soc_val > 25:
                        item.setForeground(QColor("#f9e2af"))
                    else:
                        item.setForeground(QColor("#f38ba8"))

    # ─── PLOT UPDATES ──────────────────────────────────────────────────

    def _on_subsystem_changed(self):
        self._update_plot_subsystem()

    def _update_plot_subsystem(self):
        subsystem = self.subsystem_combo.currentText()
        params = SUBSYSTEMS.get(subsystem, [])
        self.plot_widget.set_subsystem(params)

    def _update_plot(self, frame):
        self.plot_widget.add_frame(frame, self.frame_count)

    # ─── ANOMALY TABLE UPDATES ─────────────────────────────────────────

    def _add_anomaly_score(self, window_idx, score, recon_err, is_anom, pc_err=None):
        self.score_table.insertRow(0)

        item_w = QTableWidgetItem(str(window_idx))
        self.score_table.setItem(0, 0, item_w)

        item_s = QTableWidgetItem(f"{score:.4f}")
        item_s.setForeground(QColor("#f38ba8") if is_anom else QColor("#a6e3a1"))
        self.score_table.setItem(0, 1, item_s)

        item_re = QTableWidgetItem(f"{recon_err:.4f}")
        self.score_table.setItem(0, 2, item_re)

        item_st = QTableWidgetItem("⚠ ANOMALY" if is_anom else "✓ OK")
        if is_anom:
            item_st.setForeground(QColor("#f38ba8"))
        self.score_table.setItem(0, 3, item_st)

        # Prediction (raw, before persistence)
        raw_pred = int(score >= self.pipeline.threshold)
        item_p = QTableWidgetItem(str(raw_pred))
        self.score_table.setItem(0, 4, item_p)

        now = datetime.now().strftime("%H:%M:%S")
        self.score_table.setItem(0, 5, QTableWidgetItem(now))

        # Details: top contributing channels (if stats available)
        if (self.pipeline.mu_rec_err is not None and
                self.pipeline.sigma_rec_err is not None and
                pc_err is not None):
            z = (pc_err - self.pipeline.mu_rec_err) / self.pipeline.sigma_rec_err
            top_k_idx = np.argsort(z)[-self.pipeline.top_k:]
            feature_cols_display = self.pipeline.feature_cols
            top_feats = [feature_cols_display[i] if i < len(feature_cols_display)
                         else f"ch_{i}" for i in top_k_idx]
            details_str = ", ".join(
                f"{f}(z={z[i]:.1f})" for f, i in zip(top_feats, top_k_idx)
            )
        elif not self.pipeline.stats_ready:
            details_str = "No stats"
        else:
            details_str = "N/A"

        self.score_table.setItem(0, 6, QTableWidgetItem(details_str))

        # Keep table manageable
        if self.score_table.rowCount() > 50:
            self.score_table.removeRow(self.score_table.rowCount() - 1)

    def _add_anomaly_log(self, window_idx, score, recon_err, is_anom):
        self.anomaly_log.insertRow(0)

        now = datetime.now().strftime("%H:%M:%S")
        self.anomaly_log.setItem(0, 0, QTableWidgetItem(now))

        item_w = QTableWidgetItem(str(window_idx))
        item_w.setForeground(QColor("#f38ba8"))
        self.anomaly_log.setItem(0, 1, item_w)

        item_s = QTableWidgetItem(f"{score:.4f}")
        item_s.setForeground(QColor("#f38ba8"))
        self.anomaly_log.setItem(0, 2, item_s)

        self.anomaly_log.setItem(0, 3, QTableWidgetItem(f"{recon_err:.4f}"))

        details = (
            f"score={score:.4f} thresh={self.pipeline.threshold:.4f} "
            f"persist={self.pipeline.persistence_window}"
        )
        self.anomaly_log.setItem(0, 4, QTableWidgetItem(details))

        if self.anomaly_log.rowCount() > 50:
            self.anomaly_log.removeRow(self.anomaly_log.rowCount() - 1)


# ============================================================================
# SECTION 7: MAIN ENTRY POINT
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Satellite Telemetry Dashboard — ML Anomaly Pipeline")
    parser.add_argument("experiment_dir",
                        help="Path to experiment results directory "
                             "(must contain config.json, best_model.pt, scaler.pkl, clean_stats.npz)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QGroupBox {
            color: #ffffff;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #ffffff;
            font-weight: bold;
        }
        QHeaderView::section {
            background-color: #313244;
            color: #ffffff;
            font-weight: bold;
            padding: 3px;
        }
        QLabel {
            color: #ffffff;
        }
    """)

    dashboard = SatelliteDashboard(experiment_dir=args.experiment_dir)
    dashboard.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
