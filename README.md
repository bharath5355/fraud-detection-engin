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
7. Navigate to `https://fraud-detection-engin.onrender.com/docs` to test transactions interactively using the Swagger UI dashboard.

## 🔌 API Reference & Payload Schema

### Production Endpoint
`POST https://fraud-detection-engin.onrender.com/docs`

### Request Body Schema (JSON)
| Field | Type | Description |
| :--- | :--- | :--- |
| `amount` | Float | The dollar value of the current transaction. |
| `network_risk_score` | Float | Network Layer Risk (0.0 to 1.0) tracking VPN, proxy, and IP threat signals. |
| `tx_count_last_10m` | Integer | Velocity tracking: transaction count on this card in the last 10 minutes. |
| `amount_to_avg_ratio` | Float | Current amount divided by the user's historical 30-day average spend. |
| `device_encoded` | Integer | Channel mapping: 0 = Mobile App, 1 = Desktop Web, 2 = Mobile Web. |

### Example 1: Making a Public Call via Terminal (cURL)
```bash
curl -X 'POST' \
  'https://fraud-detection-engin.onrender.com/docs' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "amount": 4500.00,
  "network_risk_score": 0.89,
  "tx_count_last_10m": 5,
  "amount_to_avg_ratio": 12.5,
  "device_encoded": 2
}'

import requests

url = "https://fraud-detection-engin.onrender.com/docs"
payload = {
    "amount": 4500.00,
    "network_risk_score": 0.89,
    "tx_count_last_10m": 5,
    "amount_to_avg_ratio": 12.5,
    "device_encoded": 2
}

response = requests.post(url, json=payload)
print(response.json())
