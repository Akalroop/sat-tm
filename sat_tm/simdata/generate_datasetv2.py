"""
30-Day Time Series Dataset Generator
=====================================
Generates partitioned parquet files from TelemetrySimulator.

Splits:
  train  -- days 01-25 (clean, no anomalies)
  eval   -- days 26-27 (light injected anomalies, for detector
            threshold/recall tuning -- NOT a clean split; use
            eval/anomaly_labels.csv for ground truth)
  test   -- days 28-30 (injected anomalies at a higher rate, for
            final evaluation)

Output:
  data_fixed/train/day_XX.parquet   (~20 MB each)
  data_fixed/eval/day_XX.parquet
  data_fixed/eval/anomaly_labels.csv
  data_fixed/test/day_XX.parquet
  data_fixed/test/anomaly_labels.csv

Usage:
  python generate_dataset.py
"""

import os
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from simulator_updatedv5 import TelemetrySimulator, PARAMS


# ============================================================================
# CONFIGURATION
# ============================================================================
DAYS_TOTAL       = 30
DAYS_TRAIN       = 25
DAYS_EVAL        = 2
DAYS_TEST        = 3
DT               = 1.0          # 1 Hz sampling
STEPS_PER_DAY    = int(86400 / DT)
BASE_DIR         = "data_fixed"
SEED             = 42           # reproducibility

# Anomaly injection config (test split only)
ANOMALY_RATE     = 0.005        # ~0.5% of frames (test split)
ANOMALY_RATE_EVAL= 0.003        # ~0.3% for eval tuning
ANOMALY_DURATION = {"drift": (30, 120), "spike": (1, 5), "cascade": (60, 300)}
ANOMALY_MIX      = {"drift": 0.40, "spike": 0.30, "cascade": 0.20, "multi": 0.10}


# ============================================================================
# ANOMALY INJECTION
# ============================================================================
def generate_anomaly_schedule(total_steps, rng, anomaly_rate=ANOMALY_RATE):
    """
    Generate a list of anomaly events for an anomaly-bearing split.
    Returns list of dicts: {start, end, type, params, severity}

    `anomaly_rate` lets callers use different densities per split (e.g.
    ANOMALY_RATE for test, ANOMALY_RATE_EVAL for eval) -- previously
    ANOMALY_RATE_EVAL was defined but never actually used; both splits
    silently got the same (test) rate.
    """
    events = []
    used_ranges = set()
    num_events = int(total_steps * anomaly_rate) / 50  # avg event ~50 frames

    for _ in range(int(num_events)):
        # Pick a start position/duration that doesn't overlap any
        # existing event, retrying the whole placement (not just the
        # start point) on collision. Previously only the single start
        # index was retried; if the resulting *range* collided it was
        # silently dropped, so the actual event count could quietly fall
        # short of num_events.
        placed = False
        for _ in range(100):
            start = rng.integers(0, total_steps - 1)
            r = rng.random()
            if r < ANOMALY_MIX["drift"]:
                atype = "drift"
            elif r < ANOMALY_MIX["drift"] + ANOMALY_MIX["spike"]:
                atype = "spike"
            elif r < ANOMALY_MIX["drift"] + ANOMALY_MIX["spike"] + ANOMALY_MIX["cascade"]:
                atype = "cascade"
            else:
                atype = "multi"

            lo, hi = ANOMALY_DURATION.get(atype, ANOMALY_DURATION["drift"])
            duration = rng.integers(lo, hi + 1)
            end = min(start + duration, total_steps)

            event_range = set(range(start, end))
            if not (event_range & used_ranges):
                used_ranges |= event_range
                placed = True
                break
        if not placed:
            continue

        # Pick parameters
        if atype == "multi":
            params = list(rng.choice(PARAMS, size=rng.integers(2, 5), replace=False))
        else:
            params = [rng.choice(PARAMS)]

        severity = rng.uniform(5, 30)

        events.append({
            "start": int(start),
            "end": int(end),
            "type": atype,
            "params": params,
            "severity": float(severity),
        })

    events.sort(key=lambda e: e["start"])
    return events


def inject_anomalies(values, events, rng):
    """
    Apply anomaly events to a numpy array of shape (steps, num_params).
    Returns modified values and labels (0=normal, 1=anomaly).
    """
    labels = np.zeros(values.shape[0], dtype=np.int8)

    for event in events:
        start, end = event["start"], event["end"]
        atype = event["type"]
        severity = event["severity"]

        for param in event["params"]:
            try:
                col_idx = PARAMS.index(param)
            except ValueError:
                continue

            if atype == "spike":
                spike_frames = rng.choice(
                    range(start, end), size=min(3, end - start), replace=False
                )
                for sf in spike_frames:
                    values[sf, col_idx] += rng.choice([-1, 1]) * severity * 10
                    labels[sf] = 1

            elif atype == "drift":
                offset = rng.choice([-1, 1]) * severity * 5
                for i in range(start, end):
                    progress = (i - start) / max(1, end - start - 1)
                    envelope = np.sin(np.pi * progress)
                    values[i, col_idx] += offset * envelope
                    labels[i] = 1

            elif atype == "cascade":
                shift = rng.choice([-1, 1]) * severity * 3
                for i in range(start, end):
                    values[i, col_idx] += shift
                    labels[i] = 1

            elif atype == "multi":
                for i in range(start, end):
                    values[i, col_idx] += rng.normal(0, severity)
                    labels[i] = 1

    return values, labels


# ============================================================================
# DATA GENERATION
# ============================================================================
def generate_split(sim, split_name, num_days, inject_anomalies_flag, rng, day_offset,
                    anomaly_rate=ANOMALY_RATE):
    """Generate data for one split (train/eval/test). Returns anomaly labels or None."""
    total_steps = num_days * STEPS_PER_DAY
    print(f"\n{'='*60}")
    print(f"Generating {split_name}: {num_days} days ({total_steps:,} frames)")
    print(f"{'='*60}")

    # Pre-allocate
    timestamps = np.empty(total_steps, dtype="datetime64[s]")
    values = np.empty((total_steps, len(PARAMS)), dtype=np.float64)

    base_time = datetime(2025, 1, 1) + timedelta(days=day_offset)
    sim_start = time.time()

    for step in range(total_steps):
        frame = sim.get_frame()
        ts = base_time + timedelta(seconds=step)
        timestamps[step] = np.datetime64(ts)

        for j, param in enumerate(PARAMS):
            values[step, j] = frame.get(param, 0.0)

        # Progress every 100k steps
        if (step + 1) % 100000 == 0:
            elapsed = time.time() - sim_start
            fps = (step + 1) / elapsed
            remaining = (total_steps - step - 1) / fps
            print(f"  Step {step+1:>7,}/{total_steps:,}  "
                  f"{fps:.0f} fps  "
                  f"ETA {remaining/60:.1f} min")

    # Inject anomalies for test split
    anomaly_labels = None
    if inject_anomalies_flag:
        events = generate_anomaly_schedule(total_steps, rng, anomaly_rate=anomaly_rate)
        print(f"  Injecting {len(events)} anomaly events")
        values, anomaly_labels = inject_anomalies(values, events, rng)

        # Build event log
        event_rows = []
        for i, ev in enumerate(events):
            event_start_ts = base_time + timedelta(seconds=ev["start"])
            event_end_ts = base_time + timedelta(seconds=ev["end"])
            event_rows.append({
                "event_id": i,
                "start_step": ev["start"],
                "end_step": ev["end"],
                "start_time": event_start_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": event_end_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "type": ev["type"],
                "params": ", ".join(ev["params"]),
                "severity": f"{ev['severity']:.2f}",
            })

        return timestamps, values, anomaly_labels, event_rows
    return timestamps, values, None, None


def save_daily_parquet(split_dir, split_name, timestamps, values, labels, day_offset):
    """Save data as daily parquet files."""
    num_days = len(timestamps) // STEPS_PER_DAY

    for day in range(num_days):
        start = day * STEPS_PER_DAY
        end = start + STEPS_PER_DAY

        df = pd.DataFrame(values[start:end], columns=PARAMS)
        df.insert(0, "timestamp", pd.to_datetime(timestamps[start:end]))

        if labels is not None:
            df["is_anomaly"] = labels[start:end]

        day_num = day_offset + day + 1
        filepath = os.path.join(split_dir, f"day_{day_num:02d}.parquet")
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filepath)

        size_mb = os.path.getsize(filepath) / 1e6
        anomaly_count = df["is_anomaly"].sum() if "is_anomaly" in df.columns else 0
        print(f"  Saved {filepath} ({size_mb:.1f} MB, "
              f"{len(df):,} rows, {anomaly_count} anomalous frames)")


# ============================================================================
# MAIN
# ============================================================================
def main():
    # Seed BOTH the Generator used for anomaly scheduling AND the global
    # numpy RNG state. Previously only the former was seeded; all the
    # sensor/physics noise inside TelemetrySimulator uses bare
    # np.random.randn()/np.random.choice(), which reads the global state
    # -- so despite the "SEED = 42 # reproducibility" comment, two runs
    # would NOT produce the same physics noise, only the same anomaly
    # schedule. Seeding both makes the full pipeline reproducible.
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # Create output directories
    for split in ["train", "eval", "test"]:
        os.makedirs(os.path.join(BASE_DIR, split), exist_ok=True)

    sim = TelemetrySimulator()
    sim.dt = DT

    total_start = time.time()

    # --- Train (clean, no anomalies) ---
    ts, vals, labels, events = generate_split(
        sim, "train", DAYS_TRAIN, inject_anomalies_flag=False,
        rng=rng, day_offset=0
    )
    save_daily_parquet(os.path.join(BASE_DIR, "train"), "train",
                       ts, vals, labels, day_offset=0)

    # --- Eval (light anomalies, for detector threshold/recall tuning --
    #     NOT a clean split; see module docstring) ---
    ts, vals, labels, events = generate_split(
        sim, "eval", DAYS_EVAL, inject_anomalies_flag=True,
        rng=rng, day_offset=DAYS_TRAIN, anomaly_rate=ANOMALY_RATE_EVAL
    )
    save_daily_parquet(os.path.join(BASE_DIR, "eval"), "eval",
                       ts, vals, labels, day_offset=DAYS_TRAIN)

    # Save anomaly eval labels
    if events:
        eval_labels_df   = pd.DataFrame(events)
        eval_labels_path = os.path.join(BASE_DIR, "eval" ,"anomaly_labels.csv")
        eval_labels_df.to_csv(eval_labels_path, index=False)
        print(f"\n Saved eval anomaly labels: {eval_labels_path} ({len(events)} events)")

    # --- Test (higher-rate anomalies, final evaluation split) ---
    ts, vals, labels, events = generate_split(
        sim, "test", DAYS_TEST, inject_anomalies_flag=True,
        rng=rng, day_offset=DAYS_TRAIN + DAYS_EVAL, anomaly_rate=ANOMALY_RATE
    )
    save_daily_parquet(os.path.join(BASE_DIR, "test"), "test",
                       ts, vals, labels, day_offset=DAYS_TRAIN + DAYS_EVAL)

    # Save anomaly labels
    if events:
        labels_df = pd.DataFrame(events)
        labels_path = os.path.join(BASE_DIR, "test", "anomaly_labels.csv")
        labels_df.to_csv(labels_path, index=False)
        print(f"\n  Saved anomaly labels: {labels_path}")
        print(f"  Total events: {len(events)}")
        type_counts = labels_df["type"].value_counts()
        for atype, count in type_counts.items():
            print(f"    {atype}: {count}")

    elapsed = time.time() - total_start
    total_frames = DAYS_TOTAL * STEPS_PER_DAY
    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Total time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hrs)")
    print(f"  Average speed: {total_frames/elapsed:.0f} frames/sec")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
