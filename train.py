import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import os

def load_and_preprocess_data(file_path):
    """
    Load the traffic data and preprocess it.
    """
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['Day'] = df['DateTime'].dt.day
    df['Hour'] = df['DateTime'].dt.hour
    df['Weekday'] = df['DateTime'].dt.weekday  # 0=Monday, 6=Sunday
    return df

def train_model(df):
    """
    Train the Random Forest model.
    """
    features = ['Junction', 'Year', 'Month', 'Day', 'Weekday', 'Hour']
    target = 'Vehicles'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score: {r2:.4f}")
    
    return model, features

def save_model_and_features(model, features, model_path, features_path):
    """
    Save the trained model and features list.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"Model saved to {model_path}")
    print(f"Features saved to {features_path}")

if __name__ == "__main__":
    data_path = 'data/traffic.csv'
    model_path = 'model/traffic_model.pkl'
    features_path = 'model/features.pkl'
    
    df = load_and_preprocess_data(data_path)
    model, features = train_model(df)
    save_model_and_features(model, features, model_path, features_path)