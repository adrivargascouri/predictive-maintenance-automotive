"""
predict.py
==========
Product-style prediction CLI for the Predictive Maintenance System.

Usage
-----
    python src/predict.py --machine Machine_3 --model lstm
    python src/predict.py --machine Machine_1 --model xgboost

The script:
    1. Loads the trained model (LSTM by default, falls back to XGBoost).
    2. Reads the last N rows of the sensor CSV for the requested machine.
    3. Runs preprocessing and feature engineering on the input slice.
    4. Outputs a human-readable prediction report.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "sensor_data.csv"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "saved_models"

SEQUENCE_LENGTH: int = 60   # minutes look-back for LSTM
TARGET_COL: str = "failure_within_48h"
DROP_COLS: list[str] = [
    "timestamp", "machine_id", "failure_within_48h", "time_to_failure"
]

# Threshold above which we issue an ALERT
ALERT_THRESHOLD: float = 0.5
MEDIUM_RISK_THRESHOLD: float = 0.2


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_machine_data(machine_id: str, n_rows: int = 200) -> pd.DataFrame:
    """
    Load the last ``n_rows`` rows of sensor data for the given machine.

    Args:
        machine_id: e.g. ``"Machine_3"``.
        n_rows:     Number of recent rows to load.

    Returns:
        DataFrame slice for that machine.

    Raises:
        SystemExit: if the machine is not found in the data.
    """
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
    machine_df = df[df["machine_id"] == machine_id].sort_values("timestamp")

    if machine_df.empty:
        print(f"❌ Machine '{machine_id}' not found in {RAW_DATA_PATH}")
        sys.exit(1)

    return machine_df.tail(n_rows).reset_index(drop=True)


def _preprocess_slice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply lightweight preprocessing to the input slice (fill, scale).

    Args:
        df: Raw machine data slice.

    Returns:
        Scaled DataFrame.
    """
    sensor_cols = ["temperature", "vibration", "pressure", "rpm", "current"]

    # Forward / backward fill any missing values
    df[sensor_cols] = df[sensor_cols].ffill().bfill()

    # Load and apply the fitted scaler
    scaler_path = PROCESSED_DIR / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        df[sensor_cols] = scaler.transform(df[sensor_cols])

    return df


def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the input slice.

    Args:
        df: Preprocessed machine data slice.

    Returns:
        Enriched DataFrame (NaN rows dropped).
    """
    # Import here to avoid circular dependencies at module level
    from src.features.feature_engineering import engineer_features
    return engineer_features(df)


# ── Prediction helpers ────────────────────────────────────────────────────────

def _predict_sklearn(X: pd.DataFrame, model_path: Path) -> float:
    """
    Run inference with a saved scikit-learn–compatible model (e.g. Random
    Forest or XGBoost) that exposes ``predict_proba``.

    Args:
        X:          Feature DataFrame (last row used).
        model_path: Path to the ``.pkl`` file.

    Returns:
        Failure probability (float in [0, 1]).
    """
    model = joblib.load(model_path)
    drop = [c for c in DROP_COLS if c in X.columns]
    X_clean = X.drop(columns=drop)
    probs = model.predict_proba(X_clean.iloc[[-1]])
    classes = getattr(model, "classes_", np.array([0, 1]))

    if probs.ndim == 1:
        return float(probs[0])

    if probs.shape[1] == 1:
        return float(probs[0, 0]) if classes[0] == 1 else 0.0

    if 1 in classes:
        positive_idx = int(np.where(classes == 1)[0][0])
        return float(probs[0, positive_idx])

    return float(probs[0, -1])


def _predict_lstm(X: pd.DataFrame, model_path: Path) -> float:
    """
    Run inference with the saved LSTM model using the last SEQUENCE_LENGTH rows.

    Args:
        X:          Feature DataFrame.
        model_path: Path to the ``.h5`` file.

    Returns:
        Failure probability (float in [0, 1]).

    Raises:
        SystemExit: if fewer rows than SEQUENCE_LENGTH are available.
    """
    import tensorflow as tf

    drop = [c for c in DROP_COLS if c in X.columns]
    X_clean = X.drop(columns=drop).values.astype(np.float32)

    if len(X_clean) < SEQUENCE_LENGTH:
        print(
            f"❌ Not enough data for LSTM (need {SEQUENCE_LENGTH} rows, "
            f"got {len(X_clean)}). Use --model xgboost instead."
        )
        sys.exit(1)

    sequence = X_clean[-SEQUENCE_LENGTH:][np.newaxis, ...]  # (1, seq_len, n_features)
    model = tf.keras.models.load_model(str(model_path))
    prob = float(model.predict(sequence, verbose=0)[0, 0])
    return prob


# ── Output ────────────────────────────────────────────────────────────────────

def _print_report(
    machine_id: str,
    model_name: str,
    probability: float,
    timestamp: str,
) -> None:
    """
    Print the product-style prediction report.

    Args:
        machine_id:   e.g. ``"Machine_3"``.
        model_name:   ``"LSTM"`` or ``"XGBoost"``.
        probability:  Predicted failure probability (0–1).
        timestamp:    Human-readable timestamp string.
    """
    bar = "=" * 60
    sep = "-" * 60

    if probability >= ALERT_THRESHOLD:
        risk_level = "High"
        status = "⚠️  Business Status: Elevated failure risk detected."
        interpretation = (
            f"🧭 Interpretation: {machine_id} is likely to fail within the next 48 hours."
        )
        action = "🔧 Recommended Action: Schedule maintenance within 24 hours."
    elif probability >= MEDIUM_RISK_THRESHOLD:
        risk_level = "Medium"
        status = "🟡 Business Status: Warning pattern detected."
        interpretation = (
            f"🧭 Interpretation: {machine_id} shows early degradation signals that merit review."
        )
        action = "🔧 Recommended Action: Inspect the machine during the next maintenance window."
    else:
        risk_level = "Low"
        status = "✅ Business Status: Machine operating within normal parameters."
        interpretation = (
            f"🧭 Interpretation: {machine_id} shows low short-term failure risk."
        )
        action = "🔧 Recommended Action: Continue regular monitoring."

    print(f"\n{bar}")
    print("🔍 PREDICTIVE MAINTENANCE SYSTEM — Automotive Line")
    print(bar)
    print(f"📍 Machine:     {machine_id}")
    print(f"🕐 Timestamp:   {timestamp}")
    print(f"⚙️  Model Used:  {model_name}")
    print(sep)
    print(f"📈 Risk Level:  {risk_level}")
    print(status)
    print(f"📊 Failure Probability (next 48h): {probability * 100:.1f}%")
    print(interpretation)
    print(action)
    print(f"{bar}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed namespace with ``machine`` and ``model`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance — Failure Probability Prediction"
    )
    parser.add_argument(
        "--machine",
        type=str,
        default="Machine_1",
        help="Machine identifier (e.g. Machine_3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lstm", "xgboost", "random_forest"],
        default="lstm",
        help="Model to use for prediction (default: lstm)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load data, run inference, print report."""
    args = parse_args()
    machine_id: str = args.machine
    model_choice: str = args.model

    # ── Load data ──────────────────────────────────────────────────────────
    df_raw = _load_machine_data(machine_id, n_rows=200)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Preprocess & feature engineer ─────────────────────────────────────
    df_prep = _preprocess_slice(df_raw)
    df_eng = _apply_feature_engineering(df_prep)

    # ── Inference ──────────────────────────────────────────────────────────
    model_name_display: str = "Unknown"
    probability: float = 0.0

    if model_choice == "lstm":
        lstm_path = MODELS_DIR / "lstm_model.h5"
        if lstm_path.exists():
            model_name_display = "LSTM"
            probability = _predict_lstm(df_eng, lstm_path)
        else:
            print("⚠️  LSTM model not found — falling back to XGBoost.")
            model_choice = "xgboost"

    if model_choice == "xgboost":
        xgb_path = MODELS_DIR / "xgboost.pkl"
        if not xgb_path.exists():
            print(f"❌ XGBoost model not found at {xgb_path}. Train models first.")
            sys.exit(1)
        model_name_display = "XGBoost"
        probability = _predict_sklearn(df_eng, xgb_path)

    if model_choice == "random_forest":
        rf_path = MODELS_DIR / "random_forest.pkl"
        if not rf_path.exists():
            print(f"❌ Random Forest model not found at {rf_path}. Train models first.")
            sys.exit(1)
        model_name_display = "Random Forest"
        probability = _predict_sklearn(df_eng, rf_path)

    _print_report(machine_id, model_name_display, probability, timestamp)


if __name__ == "__main__":
    main()
