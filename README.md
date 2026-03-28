# 🚦 Real-Time Traffic Prediction System

A machine learning-powered web application that predicts vehicle count and traffic congestion levels using a trained Random Forest model. The system provides both single-time predictions and full-day traffic trend visualization.

---

## ✨ Features

* **Traffic Prediction**: Predict vehicle count based on junction, date, and time
* **Traffic Classification**: Automatically classifies traffic as Low, Medium, or High
* **12-Hour Input Support**: User-friendly AM/PM input (converted internally to 24-hour format)
* **Daily Traffic Visualization**: Graph showing Hour vs Predicted Vehicles
* **Peak Hour Detection**: Identifies busiest time of the day
* **Interactive UI**: Built using Streamlit
* **Cloud Deployment**: Hosted on Streamlit Cloud

---

## 🛠 Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn (Random Forest)**
* **Streamlit**
* **Matplotlib**
* **gdown (for model loading)**

---

## 📁 Project Structure

```
traffic_project/
│
├── app.py                # Streamlit app
├── train.py              # Model training
├── utils.py              # Helper functions
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── traffic.csv
│
└── model/ (ignored in Git)
    └── traffic_model.pkl
```

---

## ⚙️ Installation

```bash
git clone <repository-url>
cd traffic_project
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## 🧠 Model Training

```bash
python train.py
```

* Preprocesses data
* Trains Random Forest model
* Saves model locally

---

## 🚀 Run the App (Local)

```bash
streamlit run app.py
```

---

## 🌐 Live Deployment

The app is deployed on Streamlit Cloud.

👉 **Live App**: *(add your link here)*

---

## 📊 Usage

### Inputs:

* Junction (1–4)
* Year, Month, Day
* Hour (12-hour format with AM/PM)

### Outputs:

* Predicted Vehicles
* Traffic Level
* Full-day traffic graph
* Peak traffic hour

---

## 📈 Traffic Levels

| Vehicles | Level  |
| -------- | ------ |
| < 20     | Low    |
| 20–50    | Medium |
| > 50     | High   |

---

## 🔄 How It Works

```text
User Input → Convert (12hr → 24hr) → Model → Prediction → Visualization
```

---

## ⚠️ Note on Model File

* The trained model is **not stored in GitHub** (due to size limits)
* It is downloaded dynamically using Google Drive

---

## 🚀 Future Improvements

* Live traffic API integration
* Google Maps visualization
* Deep Learning (LSTM) model
* Multi-day comparison dashboard

---

## 🎓 Conclusion

This project demonstrates a complete end-to-end machine learning pipeline, including data preprocessing, model training, deployment, and interactive visualization using Streamlit.

---

## 📄 License

This project is licensed under the MIT License.
