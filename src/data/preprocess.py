"""
preprocess.py
=============
Loads the raw sensor dataset, cleans it, and produces train / validation /
test splits ready for modelling.

Pipeline
--------
1. load_data()        — Read ``data/raw/sensor_data.csv``
2. handle_missing()   — Forward-fill then backward-fill
3. remove_outliers()  — IQR method per sensor per machine
4. split_data()       — Chronological 70 / 15 / 15 split (no shuffle)
5. scale_features()   — StandardScaler fitted on train only
6. run_pipeline()     — Orchestrate all steps and save outputs

Outputs (data/processed/)
--------------------------
    train.csv, val.csv, test.csv, scaler.pkl
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "sensor_data.csv"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

# Sensor columns that will be scaled
SENSOR_COLS: list[str] = [
    "temperature", "vibration", "pressure", "rpm", "current"
]

# Chronological split ratios
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
# TEST_RATIO is implicitly 1 - TRAIN_RATIO - VAL_RATIO = 0.15


# ── Step 1 ───────────────────────────────────────────────────────────────────

def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw sensor CSV into a DataFrame.

    Args:
        path: Path to ``sensor_data.csv``.

    Returns:
        DataFrame with ``timestamp`` parsed as datetime and sorted by
        (machine_id, timestamp).
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values(["machine_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"✅ Loaded data: {df.shape}")
    return df


# ── Step 2 ───────────────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing sensor values per machine using forward-fill then
    backward-fill (preserving temporal order).

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with no missing values in sensor columns.
    """
    before = df[SENSOR_COLS].isna().sum().sum()
    df = df.copy()
    df[SENSOR_COLS] = (
        df.groupby("machine_id")[SENSOR_COLS]
        .transform(lambda x: x.ffill().bfill())
    )
    after = df[SENSOR_COLS].isna().sum().sum()
    print(f"✅ Missing values: {before} → {after}")
    return df


# ── Step 3 ───────────────────────────────────────────────────────────────────

def remove_outliers(df: pd.DataFrame, iqr_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Remove rows where a sensor reading lies outside
    [Q1 - k*IQR, Q3 + k*IQR] for any sensor, computed per machine.

    A generous multiplier (3.0) is used to preserve genuine degradation
    signals while removing data-entry errors.

    Args:
        df:              Input DataFrame.
        iqr_multiplier:  Number of IQR widths beyond Q1/Q3 to tolerate.

    Returns:
        Cleaned DataFrame with outlier rows removed.
    """
    df = df.copy()
    mask = pd.Series(True, index=df.index)

    for machine_id, group in df.groupby("machine_id"):
        for col in SENSOR_COLS:
            q1 = group[col].quantile(0.25)
            q3 = group[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            outlier_mask = (df["machine_id"] == machine_id) & (
                (df[col] < lower) | (df[col] > upper)
            )
            mask &= ~outlier_mask

    removed = (~mask).sum()
    df = df[mask].reset_index(drop=True)
    print(f"✅ Outliers removed: {removed:,} rows → {len(df):,} rows remaining")
    return df


# ── Step 4 ───────────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: 70% train / 15% validation / 15% test.
    Split is done on the global timestamp (not per machine) to respect
    temporal ordering.

    Args:
        df: Cleaned DataFrame sorted by timestamp.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # Sort by timestamp globally
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(
        f"✅ Split — train: {len(train_df):,} | "
        f"val: {len(val_df):,} | test: {len(test_df):,}"
    )
    return train_df, val_df, test_df


# ── Step 5 ───────────────────────────────────────────────────────────────────

def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler_path: Path = PROCESSED_DIR / "scaler.pkl",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on the training set and apply it to all splits.
    Saves the fitted scaler to disk for later inference.

    Args:
        train_df:    Training split.
        val_df:      Validation split.
        test_df:     Test split.
        scaler_path: Where to persist the fitted scaler.

    Returns:
        Tuple of (scaled_train, scaled_val, scaled_test, fitted_scaler).
    """
    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[SENSOR_COLS] = scaler.fit_transform(train_df[SENSOR_COLS])
    val_df[SENSOR_COLS] = scaler.transform(val_df[SENSOR_COLS])
    test_df[SENSOR_COLS] = scaler.transform(test_df[SENSOR_COLS])

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")

    return train_df, val_df, test_df, scaler


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """
    Execute the full preprocessing pipeline and save outputs to
    ``data/processed/``.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = handle_missing(df)
    df = remove_outliers(df)
    train_df, val_df, test_df = split_data(df)
    train_df, val_df, test_df, _ = scale_features(train_df, val_df, test_df)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"✅ Processed splits saved to {PROCESSED_DIR}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
