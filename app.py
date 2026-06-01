import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

# 1. Initialize FastAPI Application
app = FastAPI(title="Enterprise Real-Time Fraud Scoring Engine", version="1.0")

# 2. Load the trained ML Model Artifact
try:
    with open('fraud_model.pkl', 'rb') as f:
        artifact = pickle.load(f)
        model = artifact['model']
        feature_names = artifact['feature_names']
    print("Fraud model artifact loaded successfully into memory.")
except Exception as e:
    print(f"Error loading model artifact: {e}")
    raise e


# 3. Define the Expected Incoming Transaction JSON Structure
class TransactionPayload(BaseModel):
    amount: float
    network_risk_score: float
    tx_count_last_10m: int
    amount_to_avg_ratio: float
    device_encoded: int


# 4. Create the Scoring Endpoint
@app.post("/score")
async def score_transaction(payload: TransactionPayload):
    start_time = time.time()

    try:
        # Convert incoming JSON payload to a DataFrame matching feature names
        input_data = pd.DataFrame([payload.dict()])
        X = input_data[feature_names]

        # Execute Real-Time Inference
        fraud_probability = float(model.predict_proba(X)[0][1])
        prediction = int(model.predict(X)[0])

        # Calculate execution latency to prove real-time constraint capabilities
        latency_ms = (time.time() - start_time) * 1000

        # Define business response logic
        decision = "BLOCK" if prediction == 1 else "APPROVE"

        return {
            "status": "success",
            "decision": decision,
            "fraud_probability": round(fraud_probability, 4),
            "latency_ms": round(latency_ms, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Start the local production-grade ASGI web server
    uvicorn.run(app, host="127.0.0.1", port=8080)