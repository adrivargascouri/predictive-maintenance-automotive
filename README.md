# Predictive Maintenance System — Automotive Production Line

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

## Description

End-to-end predictive maintenance system for industrial machinery using time-series sensor data
(temperature, vibration, pressure). Simulates an automotive production line with 5 machines,
training classical ML baselines (Random Forest, XGBoost) and a deep-learning LSTM model to predict
machine failures **48 hours in advance**.

---

## Motivation / Problem Statement

Unplanned equipment downtime in manufacturing costs billions of dollars annually. By predicting
failures before they occur, maintenance teams can act proactively, reducing costly unplanned
shutdowns, extending equipment life, and improving overall production efficiency.

This project demonstrates a realistic, end-to-end predictive-maintenance pipeline that could be
deployed on an automotive production line, from raw sensor ingestion all the way to actionable
human-readable alerts.

---

## Project Architecture

```
Raw Sensor Data (CSV)
        │
        ▼
┌─────────────────┐
│  Data Simulation │  ← src/data/simulate_data.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing   │  ← src/data/preprocess.py
│  (clean, scale,  │
│   split)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Engineer.│  ← src/features/feature_engineering.py
│ (rolling, lag,   │
│  interactions)   │
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌────────┐ ┌──────────┐
│Baseline│ │   LSTM   │  ← src/models/
│RF/XGB  │ │  model   │
└────┬───┘ └────┬─────┘
     │           │
     └─────┬─────┘
           │
           ▼
  ┌─────────────────┐
  │   Evaluation    │  ← src/evaluation/metrics.py
  │ (Precision,     │
  │  Recall, F1,    │
  │  ROC-AUC, SHAP) │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Prediction API │  ← src/predict.py
  │  (CLI output)   │
  └─────────────────┘
```

---

## Dataset

The dataset is **fully simulated** using `src/data/simulate_data.py`.

| Property      | Value                                                                |
| ------------- | -------------------------------------------------------------------- |
| Machines      | 5 (Machine_1 … Machine_5)                                            |
| Sensors       | temperature (°C), vibration (mm/s), pressure (bar), rpm, current (A) |
| Sampling rate | 1 minute                                                             |
| Duration      | 6 months (~260 000 rows per machine)                                 |
| Target        | `failure_within_48h` (binary: 1 if failure in next 48 h)             |
| Extra         | `time_to_failure` (hours until next failure event)                   |

Failure patterns are injected as **gradual degradation** in the 48 hours preceding each failure
event, with added Gaussian noise for realism.

---

## Pipeline Overview

| Step                   | Notebook                         | Script                                |
| ---------------------- | -------------------------------- | ------------------------------------- |
| 1. Data Generation     | `01_data_generation.ipynb`       | `src/data/simulate_data.py`           |
| 2. EDA & Preprocessing | `02_eda_and_preprocessing.ipynb` | `src/data/preprocess.py`              |
| 3. Feature Engineering | `03_feature_engineering.ipynb`   | `src/features/feature_engineering.py` |
| 4. Baseline Models     | `04_baseline_models.ipynb`       | `src/models/baseline.py`              |
| 5. LSTM Model          | `05_lstm_model.ipynb`            | `src/models/lstm_model.py`            |
| 6. Evaluation          | `06_evaluation.ipynb`            | `src/evaluation/metrics.py`           |
| 7. SHAP Explainability | `07_explainability_shap.ipynb`   | —                                     |

---

## Quick Start (Day 1)

Use this section to get an end-to-end baseline running as fast as possible.

### 1) Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

Optional (for LSTM):

```bash
pip install -r requirements-lstm.txt
```

### 3) Run full pipeline (baseline product flow)

```bash
python -m src.run_pipeline --skip-lstm
```

This executes:

- data simulation
- preprocessing
- feature engineering
- baseline training (Random Forest + XGBoost)
- evaluation metrics and figures

### 4) Optional: train LSTM

```bash
python -m src.run_pipeline
```

### 5) Product-style inference

```bash
python -m src.predict --machine Machine_3 --model xgboost
python -m src.predict --machine Machine_3 --model lstm
```

---

## Key Results

> Results will be filled in after training. Placeholder table below.

| Model         | Precision | Recall | F1  | ROC-AUC |
| ------------- | --------- | ------ | --- | ------- |
| Random Forest | —         | —      | —   | —       |
| XGBoost       | —         | —      | —   | —       |
| LSTM          | —         | —      | —   | —       |

---

## Product-Style Output Example

```
============================================================
🔍 PREDICTIVE MAINTENANCE SYSTEM — Automotive Line
============================================================
📍 Machine:     Machine_3
🕐 Timestamp:   2026-04-07 14:00:00
⚙️  Model Used:  LSTM
------------------------------------------------------------
⚠️  ALERT: High failure probability detected!
📊 Failure Probability (next 48h): 80.4%
🔧 Recommended Action: Schedule maintenance within 24 hours
============================================================
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/adrivargascouri/predictive-maintenance-automotive.git
cd predictive-maintenance-automotive

# 2. Install dependencies
pip install -r requirements.txt

# Optional: install deep-learning dependencies for LSTM
pip install -r requirements-lstm.txt

# 3. Run complete pipeline (recommended)
python -m src.run_pipeline --skip-lstm

# 4. Optional: include LSTM training
python -m src.run_pipeline

# 5. Or run notebooks in order (inside notebooks/)
jupyter notebook

# 6. Make a prediction
python -m src.predict --machine Machine_3 --model lstm
```

---

## Suggested 3-5 Week Plan

| Week         | Focus                                             | Deliverable                            |
| ------------ | ------------------------------------------------- | -------------------------------------- |
| 1            | Data simulation, labels, preprocessing validation | Clean train/val/test pipeline          |
| 2            | Feature engineering + baseline models             | Baseline benchmark with metrics        |
| 3            | LSTM training and comparison                      | Model comparison (ROC-AUC, F1, recall) |
| 4            | SHAP + product-style reporting                    | Explainable predictions per machine    |
| 5 (optional) | Refactor, tests, README polish                    | Interview-ready GitHub project         |

---

## Tech Stack

| Tool                     | Purpose                               |
| ------------------------ | ------------------------------------- |
| **Python 3.10+**         | Core language                         |
| **pandas / numpy**       | Data manipulation                     |
| **scikit-learn**         | Preprocessing, Random Forest, metrics |
| **XGBoost**              | Gradient boosted trees baseline       |
| **TensorFlow / Keras**   | LSTM deep learning model              |
| **SHAP**                 | Model explainability                  |
| **matplotlib / seaborn** | Visualisation                         |
| **Jupyter**              | Interactive notebooks                 |

---

## License

This project is licensed under the [MIT License](LICENSE).
