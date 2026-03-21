import streamlit as st
import pickle
import pandas as pd
from utils import classify_traffic_level, preprocess_input, get_days_in_month

# Load model and features
@st.cache_resource
def load_model_and_features():
    with open('model/traffic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

def main():
    st.title("Real-Time Traffic Prediction System")
    st.markdown("Predict vehicle count and traffic congestion level using machine learning.")
    
    # Load model
    try:
        model, features = load_model_and_features()
    except FileNotFoundError:
        st.error("Model files not found. Please run train.py first to train the model.")
        return
    
    st.header("Input Parameters")
    st.write("Set the date, time, and junction to predict traffic volume.")

    junction = st.selectbox("Junction", options=[1, 2, 3, 4])
    year = st.number_input("Year", min_value=2015, max_value=2025, value=2023, step=1)
    month = st.selectbox("Month", options=list(range(1, 13)), format_func=lambda m: f"{m:02d}")

    max_day = get_days_in_month(year, month)
    day = st.selectbox("Day", options=list(range(1, max_day + 1)), format_func=lambda d: f"{d:02d}")
    hour = st.selectbox("Hour", options=list(range(24)), format_func=lambda h: f"{h:02d}:00")

    # Calculate week day automatically from year/month/day
    try:
        date_obj = pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
        weekday = date_obj.weekday()
        weekday_name = date_obj.day_name()
        st.info(f"Weekday auto-calculated: {weekday_name} ({weekday})")
    except Exception as e:
        st.error(f"Invalid date selected: {e}")
        return
    
    if st.button("Predict Traffic"):
        # Preprocess input
        input_df = preprocess_input(junction, year, month, day, weekday, hour)
        
        # Predict
        prediction = model.predict(input_df)[0]
        predicted_vehicles = round(prediction)
        traffic_level = classify_traffic_level(predicted_vehicles)
        
        st.header("Prediction Results")
        st.metric("Predicted Vehicles", predicted_vehicles)
        st.metric("Traffic Level", traffic_level)
        
        # Color coding
        if traffic_level == 'Low':
            st.success("Traffic is Low")
        elif traffic_level == 'Medium':
            st.warning("Traffic is Medium")
        else:
            st.error("Traffic is High")

if __name__ == "__main__":
    main()