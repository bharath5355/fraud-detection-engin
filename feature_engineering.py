import pandas as pd
import numpy as np


def run_feature_engineering():
    print("Loading synthetic transaction dataset...")
    # Load the data we generated in Stage 1
    df = pd.read_csv('synthetic_transactions.csv')

    # Ensure timestamp is parsed as a true datetime object for rolling calculations
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort chronologically to prevent look-ahead bias during window aggregations
    df = df.sort_values('timestamp').reset_index(drop=True)

    print("Calculating rolling behavioral features (Velocity & Deviations)...")

    # --- Feature 1: Card Velocity (Transaction Count in Last 10 Minutes) ---
    # We use a rolling time window grouped by each unique card_id
    df['tx_count_last_10m'] = (
        df.groupby('card_id')
        .rolling('10min', on='timestamp')
        .user_id.count()
        .reset_index(level=0, drop=True)
    )

    # --- Feature 2: Historical User Baseline (30-Day Expanding Window) ---
    # Calculate the running historical average amount spent by each unique user
    df['user_avg_amount_30d'] = (
        df.groupby('user_id')
        .amount.expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    # --- Feature 3: Deviation Ratio ---
    # Quantify how much higher the current transaction is compared to their historical baseline
    # We add 1 to the denominator to prevent any accidental divide-by-zero errors
    df['amount_to_avg_ratio'] = df['amount'] / (df['user_avg_amount_30d'] + 1)

    # --- Feature 4: Device Signature Changes ---
    # Map device type strings to categorical numerical encodings for the ML model
    device_mapping = {'mobile_app': 0, 'desktop_web': 1, 'mobile_web': 2}
    df['device_encoded'] = df['device_type'].map(device_mapping)

    # Handle any missing values caused by expanding window baselines (e.g., first transactions)
    df = df.fillna({
        'user_avg_amount_30d': df['amount'],
        'amount_to_avg_ratio': 1.0
    })

    # Save our newly enriched dataset
    output_file = 'engineered_transactions.csv'
    df.to_csv(output_file, index=False)
    print(f"Feature engineering complete! Enriched dataset saved to '{output_file}'")

    # Display a sneak peek of the new feature matrix
    print("\n--- SAMPLE ENRICHED FEATURE MATRIX ---")
    print(df[['timestamp', 'amount', 'tx_count_last_10m', 'amount_to_avg_ratio', 'is_fraud']].tail(5))


if __name__ == "__main__":
    run_feature_engineering()