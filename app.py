import streamlit as st
import pickle
import pandas as pd
import os
import gdown
import matplotlib.pyplot as plt

from utils import classify_traffic_level, preprocess_input, get_days_in_month

# -----------------------------
# CONFIG
# -----------------------------
FILE_ID = "1nIcH_Fl6X_A5rato_r1qpuRzasPYjO"
MODEL_PATH = "traffic_model.pkl"

# -----------------------------
# LOAD MODEL
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

    # Load model
    model = load_model()

    # -----------------------------
    # INPUT
    # -----------------------------
    st.header("Input Parameters")

    junction = st.selectbox("Junction", [1, 2, 3, 4])
    year = st.number_input("Year", min_value=2015, max_value=2030, value=2023)

    month = st.selectbox("Month", list(range(1, 13)))
    max_day = get_days_in_month(year, month)

    day = st.selectbox("Day", list(range(1, max_day + 1)))

    # weekday calculation
    date_obj = pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
    weekday = date_obj.weekday()

    st.info(f"Weekday: {date_obj.day_name()}")

    # -----------------------------
    # SINGLE PREDICTION
    # -----------------------------
    hour = st.slider("Select Hour", 0, 23)

    if st.button("Predict Traffic"):

        input_df = preprocess_input(junction, year, month, day, weekday, hour)
        prediction = model.predict(input_df)[0]

        vehicles = int(round(prediction))
        level = classify_traffic_level(vehicles)

        st.subheader("Prediction Result")
        st.metric("Vehicles", vehicles)
        st.metric("Traffic Level", level)

    # -----------------------------
    # FULL DAY GRAPH
    # -----------------------------
    st.header("📊 Full Day Traffic Prediction")

    if st.button("Show Daily Traffic Graph"):

        hours = list(range(24))
        predictions = []

        for h in hours:
            input_df = preprocess_input(junction, year, month, day, weekday, h)
            pred = model.predict(input_df)[0]
            predictions.append(pred)

        # Create DataFrame
        df_plot = pd.DataFrame({
            "Hour": hours,
            "Vehicles": predictions
        })

        # Plot
        fig, ax = plt.subplots()

        ax.plot(df_plot["Hour"], df_plot["Vehicles"], marker='o')
        ax.set_title("Hourly Traffic Prediction")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Predicted Vehicles")

        # Highlight peak
        peak_hour = df_plot["Vehicles"].idxmax()
        ax.axvline(x=peak_hour, linestyle="--")

        st.pyplot(fig)

        st.success(f"Peak Traffic at Hour: {peak_hour}:00")


if __name__ == "__main__":
    main()