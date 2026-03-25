import streamlit as st
import pickle
import pandas as pd
import os
import gdown

from utils import classify_traffic_level, preprocess_input, get_days_in_month

# -----------------------------
# CONFIG
# -----------------------------
FILE_ID = "1nIcH_Fl6X_A5rato_r1qpuRzasPYjO-A"
MODEL_PATH = "traffic_model.pkl"

# -----------------------------
# DOWNLOAD MODEL (if not exists)
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model


# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.set_page_config(page_title="Traffic Prediction", page_icon="🚦")

    st.title("🚦 Real-Time Traffic Prediction System")
    st.markdown("Predict vehicle count and traffic congestion level using Machine Learning.")

    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # -----------------------------
    # INPUT SECTION
    # -----------------------------
    st.header("Input Parameters")

    junction = st.selectbox("Junction", [1, 2, 3, 4])
    year = st.number_input("Year", min_value=2015, max_value=2030, value=2023)

    month = st.selectbox(
        "Month",
        list(range(1, 13)),
        format_func=lambda m: f"{m:02d}"
    )

    max_day = get_days_in_month(year, month)
    day = st.selectbox(
        "Day",
        list(range(1, max_day + 1)),
        format_func=lambda d: f"{d:02d}"
    )

    hour = st.selectbox(
        "Hour",
        list(range(24)),
        format_func=lambda h: f"{h:02d}:00"
    )

    # -----------------------------
    # WEEKDAY AUTO CALCULATION
    # -----------------------------
    try:
        date_obj = pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
        weekday = date_obj.weekday()
        weekday_name = date_obj.day_name()

        st.info(f"Weekday: {weekday_name} ({weekday})")

    except Exception as e:
        st.error(f"Invalid date: {e}")
        return

    # -----------------------------
    # PREDICTION
    # -----------------------------
    if st.button("Predict Traffic"):

        # Preprocess input
        input_df = preprocess_input(
            junction, year, month, day, weekday, hour
        )

        # Predict
        prediction = model.predict(input_df)[0]
        vehicles = int(round(prediction))
        level = classify_traffic_level(vehicles)

        # -----------------------------
        # OUTPUT
        # -----------------------------
        st.header("Prediction Results")

        col1, col2 = st.columns(2)

        col1.metric("Vehicles", vehicles)
        col2.metric("Traffic Level", level)

        # Color feedback
        if level == "Low":
            st.success(" Traffic is Low")
        elif level == "Medium":
            st.warning("Traffic is Medium")
        else:
            st.error("Traffic is High")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()