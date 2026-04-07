"""
baseline.py
===========
Trains and evaluates two baseline classifiers for the failure-prediction task:
    - Random Forest  (scikit-learn, class_weight='balanced')
    - XGBoost        (scale_pos_weight handles class imbalance)

Inputs:  data/processed/train_engineered.csv
         data/processed/val_engineered.csv
Outputs: saved_models/random_forest.pkl
         saved_models/xgboost.pkl
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "saved_models"
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"

TARGET_COL: str = "failure_within_48h"
# Columns to drop before training
DROP_COLS: list[str] = [
    "timestamp", "machine_id", "failure_within_48h", "time_to_failure"
]

SEED: int = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_split(split: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load an engineered split and separate features from target.

    Args:
        split: One of ``'train'``, ``'val'``, or ``'test'``.

    Returns:
        Tuple of (X DataFrame, y Series).
    """
    path = PROCESSED_DIR / f"{split}_engineered.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop)
    y = df[TARGET_COL]
    return X, y


# ── Model training ────────────────────────────────────────────────────────────

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Train a balanced Random Forest classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Fitted RandomForestClassifier.
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print("✅ Random Forest trained.")
    return rf


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """
    Train an XGBoost classifier with scale_pos_weight to handle imbalance.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Fitted XGBClassifier.
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    print("✅ XGBoost trained.")
    return xgb


def _positive_class_proba(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return probabilities for the positive class (label=1).

    Handles edge cases where a model is trained on a single class and
    ``predict_proba`` returns only one column.
    """
    probs = model.predict_proba(X)
    if probs.ndim == 1:
        return probs.astype(float)

    if probs.shape[1] == 1:
        classes = getattr(model, "classes_", np.array([0]))
        return probs[:, 0].astype(float) if classes[0] == 1 else np.zeros(len(X))

    classes = getattr(model, "classes_", np.array([0, 1]))
    if 1 in classes:
        idx = int(np.where(classes == 1)[0][0])
        return probs[:, idx].astype(float)

    return probs[:, -1].astype(float)


# ── Evaluation ────────────────────────────────────────────────────────────────

def _print_metrics(
    model_name: str,
    split_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    """
    Print a concise metrics summary for one model on one split.

    Args:
        model_name: Display name of the model.
        split_name: ``'Train'`` or ``'Val'``.
        y_true:     Ground-truth labels.
        y_pred:     Predicted binary labels.
        y_prob:     Predicted positive-class probabilities.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = np.nan
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)

    print(
        f"  [{model_name}] {split_name} — "
        f"Precision: {precision:.3f} | Recall: {recall:.3f} | "
        f"F1: {f1:.3f} | ROC-AUC: {'n/a' if np.isnan(roc_auc) else f'{roc_auc:.3f}'}"
    )


# ── Feature importance plot ───────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list[str],
    model_name: str,
    top_n: int = 20,
) -> None:
    """
    Bar chart of the top ``top_n`` most important features.

    The figure is saved to ``reports/figures/``.

    Args:
        model:         Fitted model with a ``feature_importances_`` attribute.
        feature_names: List of feature column names in the same order as the
                       training matrix.
        model_name:    Used in the figure title and filename.
        top_n:         Number of top features to display.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).nlargest(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    importances.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    out_path = FIGURES_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"   Figure saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")

    print(f"\n📊 Training shape: {X_train.shape}")
    print(f"   Class balance (train): {y_train.value_counts().to_dict()}\n")

    # ── Random Forest ─────────────────────────────────────────────────────
    rf = train_random_forest(X_train, y_train)

    rf_train_prob = _positive_class_proba(rf, X_train)
    rf_val_prob = _positive_class_proba(rf, X_val)
    _print_metrics("Random Forest", "Train", y_train, rf.predict(X_train), rf_train_prob)
    _print_metrics("Random Forest", "Val", y_val, rf.predict(X_val), rf_val_prob)

    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")
    plot_feature_importance(rf, list(X_train.columns), "Random Forest")

    # ── XGBoost ──────────────────────────────────────────────────────────
    xgb = train_xgboost(X_train, y_train)

    xgb_train_prob = _positive_class_proba(xgb, X_train)
    xgb_val_prob = _positive_class_proba(xgb, X_val)
    _print_metrics("XGBoost", "Train", y_train, xgb.predict(X_train), xgb_train_prob)
    _print_metrics("XGBoost", "Val", y_val, xgb.predict(X_val), xgb_val_prob)

    joblib.dump(xgb, MODELS_DIR / "xgboost.pkl")
    plot_feature_importance(xgb, list(X_train.columns), "XGBoost")

    print("\n✅ Baseline models saved to", MODELS_DIR)
