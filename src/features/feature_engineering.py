"""
feature_engineering.py
=======================
Transforms the preprocessed sensor DataFrame into a rich feature matrix
suitable for machine-learning models.

Features created
----------------
- Rolling statistics (mean, std, min, max) for windows of 10, 30, 60 min
- Lag features (1, 5, 10, 30 steps) for each sensor
- Rate-of-change (first difference) for each sensor
- Cross-sensor interaction ratios: temp/vibration, pressure/current
- Time-based features: hour_of_day, day_of_week, is_weekend

NaN rows introduced by rolling/lag operations are dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

SENSOR_COLS: list[str] = [
    "temperature", "vibration", "pressure", "rpm", "current"
]

ROLLING_WINDOWS: list[int] = [10, 30, 60]    # minutes
LAG_STEPS: list[int] = [1, 5, 10, 30]        # time steps (minutes)


# ── Core engineering function ─────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to the input DataFrame.

    The input must be sorted by (machine_id, timestamp). Features are
    computed **per machine** to avoid information leakage across machines.

    Args:
        df: Preprocessed DataFrame containing at least the columns
            in ``SENSOR_COLS`` plus ``timestamp`` and ``machine_id``.

    Returns:
        Enriched DataFrame with new feature columns. Rows containing NaN
        values introduced by rolling / lag operations are dropped.
    """
    df = df.copy()
    base_cols = set(df.columns)

    # Ensure temporal ordering
    df.sort_values(["machine_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Per-machine groupby to avoid mixing machine boundaries
    grouped = df.groupby("machine_id", group_keys=False)

    # ── Rolling statistics ────────────────────────────────────────────────
    for window in ROLLING_WINDOWS:
        for sensor in SENSOR_COLS:
            prefix = f"{sensor}_roll{window}"
            df[f"{prefix}_mean"] = grouped[sensor].transform(
                lambda x, w=window: x.rolling(w, min_periods=w).mean()
            )
            df[f"{prefix}_std"] = grouped[sensor].transform(
                lambda x, w=window: x.rolling(w, min_periods=w).std()
            )
            df[f"{prefix}_min"] = grouped[sensor].transform(
                lambda x, w=window: x.rolling(w, min_periods=w).min()
            )
            df[f"{prefix}_max"] = grouped[sensor].transform(
                lambda x, w=window: x.rolling(w, min_periods=w).max()
            )

    # ── Lag features ─────────────────────────────────────────────────────
    for lag in LAG_STEPS:
        for sensor in SENSOR_COLS:
            df[f"{sensor}_lag{lag}"] = grouped[sensor].transform(
                lambda x, l=lag: x.shift(l)
            )

    # ── Rate-of-change (diff) ─────────────────────────────────────────────
    for sensor in SENSOR_COLS:
        df[f"{sensor}_diff1"] = grouped[sensor].transform(
            lambda x: x.diff(1)
        )

    # ── Cross-sensor interactions ─────────────────────────────────────────
    # Avoid division by zero with a small epsilon
    eps = 1e-9
    df["temp_vibration_ratio"] = df["temperature"] / (df["vibration"] + eps)
    df["pressure_current_ratio"] = df["pressure"] / (df["current"] + eps)

    # ── Time-based features ───────────────────────────────────────────────
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ── Drop NaN rows introduced by rolling / lag only ────────────────────
    # Keep original columns such as time_to_failure untouched, because they
    # may be intentionally NaN outside pre-failure windows.
    engineered_cols = [c for c in df.columns if c not in base_cols]
    before = len(df)
    if engineered_cols:
        df.dropna(subset=engineered_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(
        f"✅ Feature engineering complete. "
        f"Rows: {before:,} → {len(df):,} | "
        f"Features: {df.shape[1]}"
    )

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for split in ("train", "val", "test"):
        path = PROCESSED_DIR / f"{split}.csv"
        if not path.exists():
            print(f"⚠️  {path} not found — run preprocess.py first.")
            continue

        df = pd.read_csv(path, parse_dates=["timestamp"])
        df_eng = engineer_features(df)
        out_path = PROCESSED_DIR / f"{split}_engineered.csv"
        df_eng.to_csv(out_path, index=False)
        print(f"   Saved → {out_path}")
