"""
metrics.py
==========
Evaluation utilities for the predictive maintenance models.

Provides functions to:
    - compute_classification_report() — full sklearn classification report
    - compute_roc_auc()               — ROC-AUC score
    - plot_confusion_matrix()          — confusion matrix heatmap
    - plot_roc_curve()                 — ROC curve
    - plot_precision_recall_curve()    — Precision-Recall curve
    - evaluate_all_models()            — load all saved models, run on test set,
                                         and print a side-by-side summary table

All figures are saved to ``reports/figures/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "saved_models"
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"

TARGET_COL: str = "failure_within_48h"
DROP_COLS: list[str] = [
    "timestamp", "machine_id", "failure_within_48h", "time_to_failure"
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_figures_dir() -> None:
    """Create the figures output directory if it does not exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_test_split() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the engineered test split.

    Returns:
        Tuple of (X DataFrame, y Series).
    """
    path = PROCESSED_DIR / "test_engineered.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop)
    y = df[TARGET_COL]
    return X, y


# ── Core metric functions ─────────────────────────────────────────────────────

def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> str:
    """
    Compute and print a full sklearn classification report.

    Args:
        y_true:     Ground-truth binary labels.
        y_pred:     Predicted binary labels.
        model_name: Label used in the print header.

    Returns:
        The report string.
    """
    report = classification_report(y_true, y_pred, target_names=["No Failure", "Failure"])
    print(f"\n📋 Classification Report — {model_name}\n{report}")
    return report


def compute_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> float:
    """
    Compute and print the ROC-AUC score.

    Args:
        y_true:     Ground-truth binary labels.
        y_prob:     Predicted positive-class probabilities.
        model_name: Label used in the print statement.

    Returns:
        ROC-AUC score (float).
    """
    auc = roc_auc_score(y_true, y_prob)
    print(f"   [{model_name}] ROC-AUC: {auc:.4f}")
    return auc


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> None:
    """
    Plot and save a confusion matrix heatmap.

    Args:
        y_true:     Ground-truth binary labels.
        y_pred:     Predicted binary labels.
        model_name: Used in the title and filename.
    """
    _ensure_figures_dir()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    out = FIGURES_DIR / f"confusion_matrix_{slug}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"   Confusion matrix saved → {out}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> None:
    """
    Plot and save the ROC curve.

    Args:
        y_true:     Ground-truth binary labels.
        y_prob:     Predicted positive-class probabilities.
        model_name: Used in the title and filename.
    """
    _ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=model_name)
    ax.set_title(f"ROC Curve — {model_name}")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.legend()
    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    out = FIGURES_DIR / f"roc_curve_{slug}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"   ROC curve saved → {out}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> None:
    """
    Plot and save the Precision-Recall curve.

    Args:
        y_true:     Ground-truth binary labels.
        y_prob:     Predicted positive-class probabilities.
        model_name: Used in the title and filename.
    """
    _ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax, name=model_name)
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    out = FIGURES_DIR / f"pr_curve_{slug}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"   PR curve saved → {out}")


# ── Aggregate evaluation ──────────────────────────────────────────────────────

def evaluate_all_models() -> pd.DataFrame:
    """
    Load all saved baseline models, run evaluation on the test set, and
    print a side-by-side metrics summary table.

    LSTM evaluation is skipped here to avoid a TensorFlow import unless
    the model file is present (to keep this module lightweight).

    Returns:
        DataFrame with Precision, Recall, F1, and ROC-AUC for each model.
    """
    X_test, y_test = _load_test_split()
    y_test_np = y_test.values

    rows = []

    for model_name, pkl_name in [
        ("Random Forest", "random_forest.pkl"),
        ("XGBoost", "xgboost.pkl"),
    ]:
        model_path = MODELS_DIR / pkl_name
        if not model_path.exists():
            print(f"⚠️  {model_path} not found — skipping.")
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        compute_classification_report(y_test_np, y_pred, model_name)
        compute_roc_auc(y_test_np, y_prob, model_name)
        plot_confusion_matrix(y_test_np, y_pred, model_name)
        plot_roc_curve(y_test_np, y_prob, model_name)
        plot_precision_recall_curve(y_test_np, y_prob, model_name)

        rows.append(
            {
                "Model": model_name,
                "Precision": precision_score(y_test_np, y_pred, zero_division=0),
                "Recall": recall_score(y_test_np, y_pred, zero_division=0),
                "F1": f1_score(y_test_np, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test_np, y_prob),
            }
        )

    # Optional: LSTM
    lstm_path = MODELS_DIR / "lstm_model.keras"
    if lstm_path.exists():
        try:
            import tensorflow as tf
            from src.models.lstm_model import SEQUENCE_LENGTH, create_sequences

            lstm = tf.keras.models.load_model(str(lstm_path))
            X_raw = X_test.values.astype(np.float32)
            X_seq, y_seq = create_sequences(X_raw, y_test_np, SEQUENCE_LENGTH)
            y_prob_lstm = lstm.predict(X_seq).ravel()
            y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)

            compute_classification_report(y_seq, y_pred_lstm, "LSTM")
            compute_roc_auc(y_seq, y_prob_lstm, "LSTM")
            plot_confusion_matrix(y_seq, y_pred_lstm, "LSTM")
            plot_roc_curve(y_seq, y_prob_lstm, "LSTM")
            plot_precision_recall_curve(y_seq, y_prob_lstm, "LSTM")

            rows.append(
                {
                    "Model": "LSTM",
                    "Precision": precision_score(y_seq, y_pred_lstm, zero_division=0),
                    "Recall": recall_score(y_seq, y_pred_lstm, zero_division=0),
                    "F1": f1_score(y_seq, y_pred_lstm, zero_division=0),
                    "ROC-AUC": roc_auc_score(y_seq, y_prob_lstm),
                }
            )
        except Exception as exc:
            print(f"⚠️  LSTM evaluation skipped: {exc}")

    summary = pd.DataFrame(rows).set_index("Model")
    print("\n📊 Model Comparison\n")
    print(summary.to_string(float_format="{:.4f}".format))
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate_all_models()
