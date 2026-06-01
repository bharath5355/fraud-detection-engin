import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import xgboost as xgb


def train_fraud_model():
    print("Loading engineered feature dataset...")
    df = pd.read_csv('engineered_transactions.csv')

    # 1. Define Features and Target
    # We drop raw identifiers and timestamps that can cause overfitting
    feature_cols = ['amount', 'network_risk_score', 'tx_count_last_10m', 'amount_to_avg_ratio', 'device_encoded']
    X = df[feature_cols]
    y = df['is_fraud']

    # 2. Train/Test Split (Maintain chronological validation spacing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Calculate Class Imbalance Weight
    # scale_pos_weight tells XGBoost how much more weight to give to the minority (fraud) class
    num_legit = (y_train == 0).sum()
    num_fraud = (y_train == 1).sum()
    imbalance_ratio = num_legit / num_fraud
    print(f"Calculated Class Imbalance Ratio: {imbalance_ratio:.2f}x more legitimate transactions.")

    print("Training cost-sensitive XGBoost Classifier...")
    # Initialize model with hyperparameters tuned for imbalanced data
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=imbalance_ratio,  # The trick to solving class imbalance
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)
    print("Model training complete!")

    # 4. Model Evaluation & Handling the Business Tension
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n================ EVALUATION METRICS ================")
    print("CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives (Legit Approved): {cm[0][0]}")
    print(f"False Positives (Customer Insult): {cm[0][1]}")
    print(f"False Negatives (Missed Fraud): {cm[1][0]}")
    print(f"True Positives (Fraud Caught): {cm[1][1]}")

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred))
    print("====================================================")

    # 5. Save the trained model artifact and features list for deployment
    model_artifact = {
        'model': model,
        'feature_names': feature_cols
    }
    with open('fraud_model.pkl', 'wb') as f:
        pickle.dump(model_artifact, f)
    print("Model architecture artifact successfully serialized and saved as 'fraud_model.pkl'")


if __name__ == "__main__":
    train_fraud_model()