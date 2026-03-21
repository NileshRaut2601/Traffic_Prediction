# Real-Time Traffic Prediction System

A machine learning-powered web application that predicts vehicle count and traffic congestion levels in real-time using a trained Random Forest model.

## Features

- **Traffic Prediction**: Predict vehicle count based on junction, date, and time parameters
- **Traffic Classification**: Automatic classification into Low, Medium, or High traffic levels
- **Interactive Web Interface**: User-friendly Streamlit interface for easy input and visualization
- **Model Training**: Complete training pipeline with data preprocessing and model evaluation

## Tech Stack

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning library for Random Forest model
- **Streamlit**: Web framework for the user interface
- **Pickle**: Model serialization

## Project Structure

```
traffic_project/
│
├── app.py                # Streamlit web application
├── train.py              # Model training script
├── utils.py              # Helper functions
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── .gitignore           # Git ignore rules
│
├── data/
│   └── traffic.csv      # Traffic dataset
│
└── model/
    ├── traffic_model.pkl # Trained Random Forest model
    └── features.pkl      # Feature list for consistency
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd traffic_project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Train the Model

1. Ensure the dataset `data/traffic.csv` is present
2. Run the training script:
   ```bash
   python train.py
   ```
3. The script will:
   - Load and preprocess the data
   - Train a Random Forest model
   - Display the R² score
   - Save the model and features to the `model/` directory

## How to Run the Streamlit App

1. Ensure the model is trained (run `train.py` first)
2. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open your browser to the provided URL (usually http://localhost:8501)
4. Enter the prediction parameters and click "Predict Traffic"

## Usage

1. **Junction**: Select the junction number (1-4)
2. **Date/Time Parameters**:
   - Year: Select the year
   - Month: Select the month (1-12)
   - Day: Select the day (1-31)
   - Weekday: Select the day of the week
   - Hour: Select the hour (0-23)
3. Click "Predict Traffic" to get the prediction
4. View the predicted vehicle count and traffic level

## Traffic Classification

- **Low**: Less than 20 vehicles
- **Medium**: 20-50 vehicles
- **High**: More than 50 vehicles

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
