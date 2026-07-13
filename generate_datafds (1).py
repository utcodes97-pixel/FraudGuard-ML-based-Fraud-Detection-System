"""
generate_data.py — creates a synthetic credit card transaction dataset.

WHY SYNTHETIC DATA INSTEAD OF THE REAL KAGGLE DATASET?
The real Kaggle "Credit Card Fraud Detection" dataset is excellent, but its
features (V1-V28) are already PCA-transformed/anonymized by the dataset
publisher — meaning the raw, human-readable columns are hidden. That's great
for privacy but bad for *learning feature engineering*, since you'd just be
handed pre-engineered numbers with no way to understand what they represent.

This script generates RAW, human-readable transaction data (card ID,
timestamp, amount, merchant category) and injects realistic fraud patterns
into it. You then engineer features yourself in train.py — which is the
actual skill interviewers want to see.

Swapping this out for the real Kaggle CSV later is a natural upgrade path
(mentioned in the README) — the rest of the pipeline (train/serve/monitor)
works the same either way.

FRAUD PATTERNS INJECTED (so you know what "signal" exists to be learned):
1. Rapid-fire transactions: multiple transactions on the same card within
   a very short time window (card testing / stolen card behavior).
2. Odd-hour transactions: fraud disproportionately happens late at night.
3. Amount outliers: fraudulent transactions skew toward unusually high
   amounts relative to that card's typical spending.
4. New/rare merchant categories for that card.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
np.random.seed(RNG_SEED)

N_CARDS = 2000
N_LEGIT_TRANSACTIONS = 50000
N_FRAUD_TRANSACTIONS = 250  # ~0.5% fraud rate, realistic imbalance

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_retail",
    "electronics", "travel", "pharmacy", "entertainment",
]

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 6, 30)


def random_timestamp():
    delta = END_DATE - START_DATE
    random_seconds = np.random.randint(0, int(delta.total_seconds()))
    return START_DATE + timedelta(seconds=int(random_seconds))


def generate_legit_transactions(n, card_ids):
    rows = []
    for _ in range(n):
        card_id = np.random.choice(card_ids)
        ts = random_timestamp()
        # Legit transactions cluster around normal shopping hours (8am-10pm)
        # and cluster around modest amounts (log-normal distribution).
        hour_bias = np.random.normal(loc=15, scale=4)  # centered ~3pm
        hour = int(np.clip(hour_bias, 0, 23))
        ts = ts.replace(hour=hour)
        amount = round(float(np.random.lognormal(mean=3.2, sigma=0.9)), 2)
        amount = min(amount, 2000.0)
        merchant = np.random.choice(MERCHANT_CATEGORIES)
        rows.append({
            "card_id": card_id,
            "timestamp": ts,
            "amount": amount,
            "merchant_category": merchant,
            "is_fraud": 0,
        })
    return rows


def generate_fraud_transactions(n, card_ids):
    rows = []
    for _ in range(n):
        card_id = np.random.choice(card_ids)
        base_ts = random_timestamp()
        # Fraud pattern: a short burst of 2-5 rapid transactions on the
        # same card within a few minutes, at odd hours, higher amounts.
        burst_size = np.random.randint(2, 6)
        hour = int(np.random.choice([0, 1, 2, 3, 4, 23]))  # odd hours
        base_ts = base_ts.replace(hour=hour)
        for i in range(burst_size):
            ts = base_ts + timedelta(seconds=int(np.random.randint(10, 180)) * i)
            amount = round(float(np.random.lognormal(mean=5.0, sigma=0.7)), 2)
            amount = min(amount, 5000.0)
            merchant = np.random.choice(["electronics", "online_retail", "travel"])
            rows.append({
                "card_id": card_id,
                "timestamp": ts,
                "amount": amount,
                "merchant_category": merchant,
                "is_fraud": 1,
            })
    return rows


def main():
    card_ids = [f"card_{i:05d}" for i in range(N_CARDS)]

    legit = generate_legit_transactions(N_LEGIT_TRANSACTIONS, card_ids)
    fraud = generate_fraud_transactions(N_FRAUD_TRANSACTIONS, card_ids)

    df = pd.DataFrame(legit + fraud)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["transaction_id"] = [f"txn_{i:07d}" for i in range(len(df))]

    df = df[["transaction_id", "card_id", "timestamp", "amount", "merchant_category", "is_fraud"]]

    df.to_csv("data/transactions.csv", index=False)

    print(f"Generated {len(df)} transactions ({df['is_fraud'].sum()} fraudulent, "
          f"{df['is_fraud'].mean() * 100:.3f}% fraud rate)")
    print("Saved to data/transactions.csv")


if __name__ == "__main__":
    main()
