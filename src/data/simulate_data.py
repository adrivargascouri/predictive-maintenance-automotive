"""
simulate_data.py
================
Simulates 6 months of 1-minute-interval sensor readings for 5 industrial machines
on an automotive production line.

Sensors generated:
    - temperature  (°C)   — normal ~75, spikes before failure
    - vibration    (mm/s) — normal ~2.5
    - pressure     (bar)  — normal ~5.0
    - rpm                 — normal ~1500
    - current      (A)    — normal ~15

Failure patterns
    - A random number of failure events are injected per machine.
    - In the 48 hours (2 880 minutes) before each failure, all sensors
      gradually degrade with added Gaussian noise.
    - Binary label  ``failure_within_48h``  = 1 if a failure occurs in the
      next 48 hours, 0 otherwise.
    - Continuous label  ``time_to_failure``  = hours until next failure
      (NaN when no failure is upcoming within the dataset window).

Output:
    data/raw/sensor_data.csv
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Constants ────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
OUTPUT_PATH: Path = PROJECT_ROOT / "data" / "raw" / "sensor_data.csv"

N_MACHINES: int = 5
START_DATE: str = "2025-01-01"
END_DATE: str = "2025-07-01"  # ~6 months
FREQ: str = "1min"

# Normal operating ranges (mean, std)
SENSOR_PARAMS: dict = {
    "temperature": {"mean": 75.0, "std": 2.0},
    "vibration":   {"mean": 2.5,  "std": 0.2},
    "pressure":    {"mean": 5.0,  "std": 0.3},
    "rpm":         {"mean": 1500.0, "std": 30.0},
    "current":     {"mean": 15.0,  "std": 0.5},
}

# Degradation multiplier applied linearly in the 48 h before failure
FAILURE_DEGRADATION: dict = {
    "temperature": 20.0,   # +20 °C at the moment of failure
    "vibration":   3.0,    # +3 mm/s
    "pressure":    2.0,    # +2 bar
    "rpm":         200.0,  # ±200 rpm oscillation
    "current":     5.0,    # +5 A
}

FAILURE_WINDOW_MINUTES: int = 48 * 60  # 2 880 minutes
AVG_FAILURES_PER_MACHINE: int = 15     # average failure events per machine


# ── Helper functions ─────────────────────────────────────────────────────────

def _generate_failure_times(timestamps: pd.DatetimeIndex, n_failures: int) -> list:
    """
    Randomly sample ``n_failures`` timestamps to act as failure events.

    Args:
        timestamps: Full datetime index of the simulation period.
        n_failures: Number of failure events to inject.

    Returns:
        Sorted list of integer positions (iloc) within ``timestamps`` where
        failures occur.
    """
    # Reserve the first 48 h so there is always a lead-up window
    start_idx: int = FAILURE_WINDOW_MINUTES
    positions = sorted(
        random.sample(range(start_idx, len(timestamps)), n_failures)
    )
    return positions


def _build_sensor_series(
    n_rows: int,
    failure_positions: list,
    sensor: str,
) -> np.ndarray:
    """
    Build a full sensor time-series for one machine, injecting gradual
    degradation in the 48 hours before each failure event.

    Args:
        n_rows:            Total number of time steps.
        failure_positions: Sorted list of failure positions (int index).
        sensor:            Name of the sensor (key in ``SENSOR_PARAMS``).

    Returns:
        1-D NumPy array of sensor readings (float64).
    """
    params = SENSOR_PARAMS[sensor]
    values: np.ndarray = np.random.normal(params["mean"], params["std"], n_rows)
    degradation_magnitude: float = FAILURE_DEGRADATION[sensor]

    for fail_pos in failure_positions:
        window_start: int = max(0, fail_pos - FAILURE_WINDOW_MINUTES)
        window_len: int = fail_pos - window_start

        # Linear ramp from 0 → degradation_magnitude
        ramp = np.linspace(0, degradation_magnitude, window_len)
        noise = np.random.normal(0, degradation_magnitude * 0.1, window_len)
        values[window_start:fail_pos] += ramp + noise

    return values


def simulate_machine(machine_id: str, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Simulate all sensor readings and labels for one machine.

    Args:
        machine_id: Human-readable identifier, e.g. ``"Machine_1"``.
        timestamps:  DatetimeIndex covering the full simulation period.

    Returns:
        DataFrame with columns:
        timestamp, machine_id, temperature, vibration, pressure, rpm,
        current, failure_within_48h, time_to_failure.
    """
    n_rows: int = len(timestamps)

    # Randomise number of failure events (±5 around average)
    n_failures: int = max(
        1, AVG_FAILURES_PER_MACHINE + random.randint(-5, 5)
    )
    failure_positions = _generate_failure_times(timestamps, n_failures)

    # Build sensor columns
    sensor_data: dict = {
        sensor: _build_sensor_series(n_rows, failure_positions, sensor)
        for sensor in SENSOR_PARAMS
    }

    df = pd.DataFrame(sensor_data, index=timestamps)
    df.index.name = "timestamp"
    df.reset_index(inplace=True)
    df.insert(1, "machine_id", machine_id)

    # ── Labels ──────────────────────────────────────────────────────────────
    # failure_within_48h: any minute inside a 48-h window before a failure
    failure_within_48h = np.zeros(n_rows, dtype=int)
    time_to_failure = np.full(n_rows, np.nan)

    for fail_pos in failure_positions:
        window_start = max(0, fail_pos - FAILURE_WINDOW_MINUTES)
        failure_within_48h[window_start:fail_pos] = 1

        # time_to_failure in hours (counts down toward 0 at fail_pos)
        for i in range(window_start, fail_pos):
            hours_left = (fail_pos - i) / 60.0
            # Keep minimum (closest upcoming failure)
            if np.isnan(time_to_failure[i]) or hours_left < time_to_failure[i]:
                time_to_failure[i] = hours_left

    df["failure_within_48h"] = failure_within_48h
    df["time_to_failure"] = time_to_failure

    return df


def simulate_all_machines() -> pd.DataFrame:
    """
    Run the simulation for all machines and return the combined DataFrame.

    Returns:
        Combined DataFrame for all machines, sorted by (machine_id, timestamp).
    """
    timestamps = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQ)
    print(
        f"Simulating {N_MACHINES} machines | "
        f"{len(timestamps):,} time steps each | "
        f"Sensors: {list(SENSOR_PARAMS)}"
    )

    frames = []
    for i in range(1, N_MACHINES + 1):
        machine_id = f"Machine_{i}"
        print(f"  → Generating {machine_id} …")
        frames.append(simulate_machine(machine_id, timestamps))

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["machine_id", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def save_dataset(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    """
    Persist the simulated dataset to CSV.

    Args:
        df:   DataFrame to save.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n✅ Dataset saved to {path}")
    print(f"   Shape: {df.shape}")
    print(f"   Failure rate: {df['failure_within_48h'].mean():.2%}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = simulate_all_machines()
    save_dataset(df)
    print(df.head())
