# Anomaly Detection in Time Series Dataset using LSTM Autoencoder

sat-tm trains and evaluates an LSTM autoencoder to detect anomalies in simulated satellite telemetry. It also includes a PyQt dashboard for running the trained detector against a live telemetry simulator, plus notebooks and experiment configurations for model and parameter ablations.

## How it works

1. Load telemetry from the train, validation (`eval`), and test parquet splits.
2. Fit a `StandardScaler` on the clean training split, scale and clip the features, then split them into fixed-length windows.
3. Train an LSTM autoencoder to reconstruct normal telemetry windows.
4. Score windows using per-channel reconstruction errors, normalized with clean validation statistics. The detector averages the largest `top_k` channel scores.
5. Choose a threshold using validation labels, optionally apply a persistence filter, and report test metrics.

The default configuration uses 65 features, 60-step windows, a 128-unit, one-layer LSTM, and seed 42. See [`configs/config.json`](configs/config.json) for the full configuration.

## Repository layout

- `src/` — training, dataset preparation, evaluation, and model code.
- `configs/` — baseline and ablation JSON configurations.
- `simdata/` — telemetry simulator, dataset generation script, and live dashboard.
- `notebooks/` — exploratory and model-variant notebooks.
- `experiments/` — saved experiment artifacts and metrics (including the `BASE-001` example).
- `result/` — aggregate results.

## Data

Training expects parquet telemetry data at these paths, relative to the current working directory:

```text
data_fixed/
├── train/                 # clean training data
├── eval/day_26.parquet
├── eval/day_27.parquet
├── test/day_28.parquet
├── test/day_29.parquet
└── test/day_30.parquet
```

Validation and test parquet files must include an `is_anomaly` label column. The feature columns must match the configured feature count (65 for the baseline). The loader fits scaling parameters on training data only. The simulator script `simdata/generate_datasetv2.py` describes a 30-day generation flow, but currently imports `simulator_updatedv5`, which is not included in this repository; obtain that simulator module before using the generator.

## Setup

Use Python 3.10 or newer and install the dependencies used by the training and dashboard code:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas pyarrow scikit-learn matplotlib torch PyQt6
```

Choose a PyTorch build that matches your platform and CUDA setup. CUDA is optional; training falls back to CPU.

## Training and evaluation

The intended entry point is `src/main.py`, with a JSON config selected using `--config`. For example:

```bash
python src/main.py --config configs/config.json
```

Runs create an experiment directory under `experiments/<experiment_id>/` with a model checkpoint, fitted scaler, clean reconstruction statistics, metrics, run configuration, and plots. A summary row is appended to `experiments/ablation_results.csv`.

**Current checkout note:** the training entry point imports `LSTMAutoencoder` from a top-level `model` module, but the checked-in implementation is `src/models/onelstm.py`. The entry point's default config path is also `config.json`, while the baseline config is `configs/config.json`. The explicit `--config` above addresses the config path; the model import needs to be aligned before training will run as-is.

To run an ablation, pass its config, for example:

```bash
python src/main.py --config configs/ablations/config-win30.json
```

Each config supplies its own experiment identifier and training, windowing, and scoring parameters. Results are appended to the same aggregate CSV.

## Live dashboard

`simdata/dashboard.py` provides a PyQt6 dashboard that applies a trained model to simulator frames. An experiment directory needs the model checkpoint, fitted scaler, clean statistics, and a config file. Its usage is:

```bash
python simdata/dashboard.py experiments/BASE-001
```

The dashboard expects a simulator module, `sim66.py` alongside it by default. It writes a dashboard results JSON in `simdata/` when a session stops.

## Existing baseline result

The checked-in `experiments/BASE-001/metrics.json` reports test precision 0.9772, recall 0.7545, and F1 0.8516. These figures describe that saved run and may depend on the dataset and software environment used to produce it.
