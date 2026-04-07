"""
run_pipeline.py
===============
Single-entry orchestration script for the predictive-maintenance project.

Recommended usage:
    python -m src.run_pipeline --skip-lstm
    python -m src.run_pipeline

Pipeline order:
    1) Data simulation
    2) Preprocessing
    3) Feature engineering (train / val / test)
    4) Baseline models (Random Forest + XGBoost)
    5) Optional LSTM training
    6) Evaluation
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


def _run_module(module_name: str) -> None:
    """Run a Python module as a subprocess and fail fast on errors."""
    print(f"\n>>> Running: python -m {module_name}")
    subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description="Run full predictive-maintenance pipeline from one command"
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip LSTM training step to speed up the first run",
    )
    return parser.parse_args()


def main() -> None:
    """Execute project pipeline in the recommended order."""
    args = parse_args()

    print("=" * 72)
    print("PREDICTIVE MAINTENANCE PIPELINE")
    print("=" * 72)

    _run_module("src.data.simulate_data")
    _run_module("src.data.preprocess")
    _run_module("src.features.feature_engineering")
    _run_module("src.models.baseline")

    if not args.skip_lstm:
        _run_module("src.models.lstm_model")
    else:
        print("\n>>> Skipping LSTM step (--skip-lstm enabled)")

    _run_module("src.evaluation.metrics")

    print("\n" + "=" * 72)
    print("Pipeline finished successfully.")
    print("Next: run prediction with python -m src.predict --machine Machine_3 --model xgboost")
    print("=" * 72)


if __name__ == "__main__":
    main()
