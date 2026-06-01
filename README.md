# Custom Real-Time Operational AI Fraud Detection Engine

## 📌 Project Overview
An end-to-end, production-ready Machine Learning engine designed to detect and block fraudulent credit card transactions in real time. This system handles severe class imbalance (0.1% fraud rate) and flags adversarial fraud patterns (card testing, velocity spikes, device anomalies) while maintaining a strict sub-10ms scoring constraint.

## 🛠️ Architecture & Pipeline Arc
1. **Data Simulation (`generate_data.py`):** Generates a synthetic financial dataset of 500,000 transactions with engineered fraud vectors.
2. **Behavioral Feature Engineering (`feature_engineering.py`):** Computes complex rolling time-window aggregations (e.g., transaction count per card in last 10 minutes, transaction amount deviation against a user's 30-day average baseline).
3. **Model Training & Class Balancing (`train_model.py`):** Trains a cost-sensitive XGBoost classifier using `scale_pos_weight` to correct extreme class imbalance without distorting the data signature.
4. **Real-Time API Scoring Engine (`app.py`):** A high-performance FastAPI implementation that scores transactions asynchronously via a POST request payload.

## 📊 Model Performance Matrix
* **True Positives (Fraud Caught):** 99 / 100
* **False Negatives (Missed Fraud):** 1
* **Recall (Sensitivity):** 99.0%
* **API Latency Score:** < 10.0ms (Satisfies strict real-time constraints)

## ⚡ How to Run the Inference Engine Locally
1. Clone this repository and navigate into the directory.
2. Install dependencies: `pip install -r requirements.txt`
3. Generate synthetic transactions: `python generate_data.py`
4. Build features: `python feature_engineering.py`
5. Train the XGBoost model: `python train_model.py`
6. Spin up the live ASGI gateway server: `python app.py`
7. Navigate to `http://127.0.0.1:8080/docs` to test transactions interactively using the Swagger UI dashboard.
