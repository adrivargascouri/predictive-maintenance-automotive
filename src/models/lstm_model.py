"""
lstm_model.py
=============
Builds, trains, and evaluates a sequence-based LSTM model for the
failure-prediction task.

Architecture
------------
    Input  (sequence_length=60, n_features)
    → LSTM(128)  → Dropout(0.3)
    → LSTM(64)   → Dropout(0.2)
    → Dense(32, relu)
    → Dense(1, sigmoid)

Inputs:  data/processed/train_engineered.csv
         data/processed/val_engineered.csv
Output:  saved_models/lstm_model.h5
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "saved_models"

TARGET_COL: str = "failure_within_48h"
DROP_COLS: list[str] = [
    "timestamp", "machine_id", "failure_within_48h", "time_to_failure"
]

SEQUENCE_LENGTH: int = 60   # 60 minutes look-back window
SEED: int = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ── Data helpers ─────────────────────────────────────────────────────────────

def _load_split(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an engineered CSV split and return raw arrays.

    Args:
        split: One of ``'train'``, ``'val'``, or ``'test'``.

    Returns:
        Tuple of (X array shape=(N, n_features), y array shape=(N,)).
    """
    path = PROCESSED_DIR / f"{split}_engineered.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop).values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    return X, y


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert flat arrays into overlapping sequences for LSTM input.

    Args:
        X:               Feature matrix, shape (N, n_features).
        y:               Label vector, shape (N,).
        sequence_length: Number of time steps per sequence.

    Returns:
        Tuple of:
        - X_seq: shape (N - sequence_length, sequence_length, n_features)
        - y_seq: shape (N - sequence_length,) — label of the last step
    """
    X_seqs, y_seqs = [], []
    for i in range(sequence_length, len(X)):
        X_seqs.append(X[i - sequence_length:i])
        y_seqs.append(y[i])
    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)


# ── Model definition ──────────────────────────────────────────────────────────

def build_model(n_features: int) -> keras.Model:
    """
    Construct the LSTM model.

    Args:
        n_features: Number of input features (last dimension of sequences).

    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(SEQUENCE_LENGTH, n_features)),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="lstm_predictive_maintenance",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    checkpoint_path: Path,
    epochs: int = 30,
    batch_size: int = 256,
) -> keras.callbacks.History:
    """
    Train the LSTM model with early stopping and model checkpointing.

    Class weights are computed from the training labels to handle imbalance.

    Args:
        model:           Compiled Keras model.
        X_train:         Training sequences, shape (N, seq_len, n_features).
        y_train:         Training labels.
        X_val:           Validation sequences.
        y_val:           Validation labels.
        checkpoint_path: Where to save the best model weights.
        epochs:          Maximum number of training epochs.
        batch_size:      Mini-batch size.

    Returns:
        Keras History object with per-epoch metrics.
    """
    # Class weights for imbalanced data
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: neg / pos if pos > 0 else 1.0}
    print(f"   Class weights: {class_weight}")

    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "Test",
) -> dict:
    """
    Evaluate the model on a given split and print metrics.

    Args:
        model:      Trained Keras model.
        X:          Input sequences.
        y:          True labels.
        split_name: Display label for the split (e.g. ``'Test'``).

    Returns:
        Dictionary with metric names and values.
    """
    results = model.evaluate(X, y, verbose=0)
    metric_names = model.metrics_names
    metrics = dict(zip(metric_names, results))
    print(f"\n📊 {split_name} metrics:")
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / "lstm_model.h5"

    # Load and reshape data
    X_train_raw, y_train = _load_split("train")
    X_val_raw, y_val = _load_split("val")

    print(f"Raw train shape: {X_train_raw.shape}")
    X_train, y_train_seq = create_sequences(X_train_raw, y_train)
    X_val, y_val_seq = create_sequences(X_val_raw, y_val)
    print(f"Sequence train shape: {X_train.shape}")

    n_features = X_train.shape[2]
    model = build_model(n_features)
    model.summary()

    history = train_model(
        model, X_train, y_train_seq, X_val, y_val_seq, checkpoint_path
    )

    evaluate_model(model, X_val, y_val_seq, split_name="Validation")

    print(f"\n✅ LSTM model saved to {checkpoint_path}")
