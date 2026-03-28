import streamlit as st
import pickle
import pandas as pd
import os
import gdown
import matplotlib.pyplot as plt

from utils import classify_traffic_level, preprocess_input, get_days_in_month, convert_to_24_hour

FILE_ID = "1nIcH_Fl6X_A5rato_r1qpuRzasPYjO"
MODEL_PATH = "traffic_model.pkl"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def main():
    st.set_page_config(page_title="Traffic Prediction", page_icon="🚦")
    st.title("🚦 Traffic Prediction System")

    model = load_model()

    st.header("Input")

    junction = st.selectbox("Junction", [1, 2, 3, 4])
    year = st.number_input("Year", 2015, 2030, 2023)

    month = st.selectbox("Month", list(range(1, 13)))
    max_day = get_days_in_month(year, month)
    day = st.selectbox("Day", list(range(1, max_day + 1)))

    date_obj = pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
    weekday = date_obj.weekday()
    st.info(f"{date_obj.day_name()}")

    col1, col2 = st.columns(2)
    hour_12 = col1.selectbox("Hour", list(range(1, 13)))
    period = col2.selectbox("AM/PM", ["AM", "PM"])

    hour = convert_to_24_hour(hour_12, period)

    if st.button("Predict Traffic"):
        input_df = preprocess_input(junction, year, month, day, weekday, hour)
        pred = model.predict(input_df)[0]

        vehicles = int(round(pred))
        level = classify_traffic_level(vehicles)

        st.subheader("Result")
        st.metric("Vehicles", vehicles)
        st.metric("Traffic Level", level)

    st.header("📊 Daily Traffic Trend")

    if st.button("Show Graph"):

        hours = list(range(24))
        predictions = []

        for h in hours:
            df_in = preprocess_input(junction, year, month, day, weekday, h)
            p = model.predict(df_in)[0]
            predictions.append(p)

        df_plot = pd.DataFrame({
            "Hour": hours,
            "Vehicles": predictions
        })

        fig, ax = plt.subplots()

        ax.plot(df_plot["Hour"], df_plot["Vehicles"], marker='o', linewidth=2)

        ax.set_title("Hourly Traffic Prediction")
        ax.set_xlabel("Time")
        ax.set_ylabel("Vehicles")

        ax.grid(True, linestyle='--', alpha=0.6)

        # AM/PM labels
        ticks = list(range(0, 24, 2))
        labels = [f"{(h%12 or 12)} {'AM' if h < 12 else 'PM'}" for h in ticks]

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45)

        peak_hour = df_plot["Vehicles"].idxmax()
        peak_value = df_plot["Vehicles"].max()

        ax.axvline(x=peak_hour, linestyle="--")
        ax.scatter(peak_hour, peak_value)

        st.pyplot(fig)

        peak_label = f"{(peak_hour%12 or 12)} {'AM' if peak_hour < 12 else 'PM'}"
        st.success(f"Peak Traffic at {peak_label}")


if __name__ == "__main__":
    main()