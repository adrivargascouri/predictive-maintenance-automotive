"""
shap_xgboost.py
===============
Reproducible SHAP explainability script for the XGBoost baseline model.

Usage:
    python -m src.evaluation.shap_xgboost
    python -m src.evaluation.shap_xgboost --sample-size 3000 --export-assets
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "saved_models"
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"
ASSETS_DIR: Path = PROJECT_ROOT / "assets" / "images"

MODEL_PATH: Path = MODELS_DIR / "xgboost.pkl"
TEST_PATH: Path = PROCESSED_DIR / "test_engineered.csv"

DROP_COLS: list[str] = [
    "timestamp", "machine_id", "failure_within_48h", "time_to_failure"
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for SHAP run configuration."""
    parser = argparse.ArgumentParser(
        description="Run SHAP explainability for the XGBoost baseline model"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of test rows to sample for SHAP (default: 2000)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Sampled row index for local waterfall plot (default: 0)",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=20,
        help="Maximum number of features shown in SHAP plots (default: 20)",
    )
    parser.add_argument(
        "--export-assets",
        action="store_true",
        help="Copy generated SHAP figures from reports/figures to assets/images",
    )
    return parser.parse_args()


def _load_features() -> pd.DataFrame:
    """Load engineered test set and keep only model feature columns."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing engineered test split: {TEST_PATH}")

    df = pd.read_csv(TEST_PATH, parse_dates=["timestamp"])
    drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=drop)


def _resolve_shap_values(
    explainer: shap.TreeExplainer,
    X_sample: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """
    Return SHAP values and base value for positive class in a robust way.
    """
    raw_values = explainer.shap_values(X_sample)
    expected_value = explainer.expected_value

    if isinstance(raw_values, list):
        shap_values = raw_values[1] if len(raw_values) > 1 else raw_values[0]
        if isinstance(expected_value, (list, np.ndarray)):
            base_value = float(expected_value[1] if len(expected_value) > 1 else expected_value[0])
        else:
            base_value = float(expected_value)
        return np.asarray(shap_values), base_value

    shap_values = np.asarray(raw_values)

    # Some SHAP versions return shape (n_samples, n_features, n_classes).
    if shap_values.ndim == 3:
        class_index = 1 if shap_values.shape[2] > 1 else 0
        shap_values = shap_values[:, :, class_index]

    if isinstance(expected_value, (list, np.ndarray)):
        base_value = float(expected_value[1] if len(expected_value) > 1 else expected_value[0])
    else:
        base_value = float(expected_value)

    return shap_values, base_value


def _save_summary_plots(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    max_display: int,
) -> tuple[Path, Path]:
    """Save SHAP beeswarm and bar summary plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = FIGURES_DIR / "shap_summary_xgboost.png"
    bar_path = FIGURES_DIR / "shap_bar_xgboost.png"

    plt.figure(figsize=(11, 6))
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=140)
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(bar_path, dpi=140)
    plt.close()

    return summary_path, bar_path


def _save_waterfall_plot(
    shap_values: np.ndarray,
    base_value: float,
    X_sample: pd.DataFrame,
    row_index: int,
    max_display: int,
) -> Path:
    """Save local SHAP waterfall plot for one sampled row."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    waterfall_path = FIGURES_DIR / "shap_waterfall_xgboost_row0.png"

    safe_row_index = max(0, min(row_index, len(X_sample) - 1))
    row = X_sample.iloc[safe_row_index]
    explanation = shap.Explanation(
        values=shap_values[safe_row_index],
        base_values=base_value,
        data=row.values,
        feature_names=list(X_sample.columns),
    )

    plt.figure(figsize=(11, 6))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=140)
    plt.close()
    return waterfall_path


def _export_assets(paths: list[Path]) -> None:
    """Copy generated SHAP figures into tracked assets folder."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, ASSETS_DIR / path.name)


def main() -> None:
    """Run reproducible SHAP analysis for the XGBoost baseline model."""
    args = parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing XGBoost model: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    X = _load_features()

    sample_size = min(args.sample_size, len(X))
    X_sample = X.sample(n=sample_size, random_state=args.random_state)

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Loaded features: {TEST_PATH}")
    print(f"Sample size: {sample_size}")

    explainer = shap.TreeExplainer(model)
    shap_values, base_value = _resolve_shap_values(explainer, X_sample)

    summary_path, bar_path = _save_summary_plots(
        shap_values=shap_values,
        X_sample=X_sample,
        max_display=args.max_display,
    )
    waterfall_path = _save_waterfall_plot(
        shap_values=shap_values,
        base_value=base_value,
        X_sample=X_sample,
        row_index=args.row_index,
        max_display=args.max_display,
    )

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    print("\nTop 10 features by mean(|SHAP|):")
    for i, idx in enumerate(top_idx, start=1):
        print(f"{i}. {X_sample.columns[idx]}: {mean_abs[idx]:.6f}")

    print("\nGenerated SHAP artifacts:")
    for path in [summary_path, bar_path, waterfall_path]:
        print(f"- {path}")

    if args.export_assets:
        _export_assets([summary_path, bar_path, waterfall_path])
        print("\nCopied artifacts to assets/images:")
        for path in [summary_path, bar_path, waterfall_path]:
            print(f"- {ASSETS_DIR / path.name}")


if __name__ == "__main__":
    main()
