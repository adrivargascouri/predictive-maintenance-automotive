"""
Streamlit demo app for the Predictive Maintenance project.

Run:
    streamlit run demo_streamlit.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src import predict


def _risk_bucket(probability: float) -> tuple[str, str, str]:
    """Map probability to risk level and recommended action."""
    if probability >= predict.ALERT_THRESHOLD:
        return (
            "High",
            "Elevated failure risk detected.",
            "Schedule maintenance within 24 hours.",
        )
    if probability >= predict.MEDIUM_RISK_THRESHOLD:
        return (
            "Medium",
            "Warning pattern detected.",
            "Inspect during the next maintenance window.",
        )
    return (
        "Low",
        "Machine operating within normal parameters.",
        "Continue regular monitoring.",
    )


def _predict_probability(machine_id: str, model_choice: str, n_rows: int) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Run full preprocessing + feature engineering + prediction."""
    df_raw = predict._load_machine_data(machine_id, n_rows=n_rows)
    df_prep = predict._preprocess_slice(df_raw.copy())
    df_eng = predict._apply_feature_engineering(df_prep)

    if model_choice == "lstm":
        model_path = predict.MODELS_DIR / "lstm_model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"LSTM model not found: {model_path}")
        probability = predict._predict_lstm(df_eng, model_path)
    elif model_choice == "xgboost":
        model_path = predict.MODELS_DIR / "xgboost.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model not found: {model_path}")
        probability = predict._predict_sklearn(df_eng, model_path)
    else:
        model_path = predict.MODELS_DIR / "random_forest.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Random Forest model not found: {model_path}")
        probability = predict._predict_sklearn(df_eng, model_path)

    return probability, df_raw, df_eng


@st.cache_data(show_spinner=False)
def _load_machine_ids(raw_path: Path) -> list[str]:
    """Load unique machine ids once."""
    df = pd.read_csv(raw_path, usecols=["machine_id"])
    return sorted(df["machine_id"].dropna().unique().tolist())


def main() -> None:
    """Render Streamlit demo UI."""
    st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")
    st.title("Predictive Maintenance Demo")
    st.caption("Interactive inference demo for 48h failure probability")

    with st.sidebar:
        st.header("Inputs")
        machine_ids = _load_machine_ids(predict.RAW_DATA_PATH)
        machine_id = st.selectbox("Machine", machine_ids, index=0)
        model_choice = st.selectbox("Model", ["xgboost", "random_forest", "lstm"], index=0)
        n_rows = st.slider("Recent rows used", min_value=120, max_value=1000, value=240, step=20)
        run = st.button("Run prediction", type="primary")

    if not run:
        st.info("Select inputs and click Run prediction.")
        return

    try:
        probability, df_raw, _ = _predict_probability(machine_id, model_choice, n_rows)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    risk_level, business_status, action = _risk_bucket(probability)

    c1, c2, c3 = st.columns(3)
    c1.metric("Failure Probability (48h)", f"{probability * 100:.2f}%")
    c2.metric("Risk Level", risk_level)
    c3.metric("Model", model_choice.upper())

    st.progress(min(max(probability, 0.0), 1.0))
    st.write(f"Status: {business_status}")
    st.write(f"Recommended action: {action}")

    st.subheader("Recent Sensor Trends")
    chart_cols = ["temperature", "vibration", "pressure", "rpm", "current"]
    chart_df = df_raw[["timestamp", *chart_cols]].copy()
    chart_df = chart_df.set_index("timestamp")
    st.line_chart(chart_df)

    with st.expander("Show recent rows"):
        st.dataframe(df_raw.tail(20), use_container_width=True)


if __name__ == "__main__":
    main()
