import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_fraud_data(num_transactions=500000, random_state=42):
    np.random.seed(random_state)
    print(f"Generating {num_transactions} synthetic transactions...")

    # 1. Generate Baseline Features
    user_ids = [f"USER_{i:05d}" for i in np.random.randint(1000, 9999, size=num_transactions)]
    card_ids = [f"CARD_{i:05d}" for i in np.random.randint(1000, 9999, size=num_transactions)]

    # Generate sequential timestamps over a 30-day period
    start_date = datetime(2026, 1, 1)
    timestamps = [start_date + timedelta(seconds=int(i * (2592000 / num_transactions))) for i in
                  range(num_transactions)]

    # Standard transaction amounts (log-normal distribution: mostly small, few large)
    amounts = np.random.lognormal(mean=3.5, sigma=0.8, size=num_transactions).round(2)

    # Device Categories & Risk Scores
    devices = np.random.choice(['mobile_app', 'desktop_web', 'mobile_web'], size=num_transactions, p=[0.6, 0.3, 0.1])
    base_risk_scores = np.random.uniform(0.01, 0.15, size=num_transactions)

    # Construct baseline DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'user_id': user_ids,
        'card_id': card_ids,
        'amount': amounts,
        'device_type': devices,
        'network_risk_score': base_risk_scores,
        'is_fraud': 0  # Default all to legitimate
    })

    # 2. Inject Severe Class Imbalance (Targeting ~0.1% Fraud)
    # We will inject exactly 500 sophisticated fraud cases out of 500,000
    fraud_indices = np.random.choice(df.index, size=500, replace=False)
    df.loc[fraud_indices, 'is_fraud'] = 1

    # 3. Program Adversarial Fraud Patterns (The Traps for our ML Model)
    # 3. Program Adversarial Fraud Patterns (The Traps for our ML Model)
    for idx in fraud_indices:
        pattern_type = np.random.choice(['card_testing', 'high_value_burst', 'device_switch'])

        if pattern_type == 'card_testing':
            # Pattern 1: Rapid, uniform micro-transactions to see if a card works
            df.loc[idx, 'amount'] = round(float(np.random.uniform(1.00, 5.00)), 2)
            df.loc[idx, 'network_risk_score'] = float(np.random.uniform(0.4, 0.7))

        elif pattern_type == 'high_value_burst':
            # Pattern 2: Massive spike in normal purchasing behavior
            df.loc[idx, 'amount'] = round(float(np.random.uniform(1500.00, 5000.00)), 2)
            df.loc[idx, 'device_type'] = 'mobile_web'  # Often used by out-of-network fraudsters
            df.loc[idx, 'network_risk_score'] = float(np.random.uniform(0.6, 0.95))

        elif pattern_type == 'device_switch':
            # Pattern 3: Normal amount but executed through highly risky network signatures
            df.loc[idx, 'network_risk_score'] = float(np.random.uniform(0.85, 0.99))

    # Save to CSV for the next pipeline stages
    output_file = 'synthetic_transactions.csv'
    df.to_csv(output_file, index=False)
    print(f"Data generation complete! Saved to '{output_file}'")

    # Print Forensic Summary Metrics
    actual_fraud = df['is_fraud'].sum()
    print("\n--- FORENSIC DATA AUDIT ---")
    print(f"Total Legit Transactions: {len(df) - actual_fraud}")
    print(f"Total Fraud Transactions: {actual_fraud}")
    print(f"Empirical Class Imbalance: {(actual_fraud / len(df)) * 100:.3f}%")
    print("---------------------------")


if __name__ == "__main__":
    generate_fraud_data()