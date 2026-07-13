"""
monitor.py — drift detection.

THE CORE IDEA: a model is trained on a snapshot of data. Real-world data
changes over time (new fraud tactics, seasonal shopping patterns, etc.) —
this is called "drift." A model can silently degrade because the world
moved on, not because anything is "broken" in the code sense.

This script compares the distribution of features seen in LIVE predictions
(logged by main.py) against the distribution seen in TRAINING data, and
flags features that have drifted meaningfully.

METRIC USED: Population Stability Index (PSI) — a standard, industry metric
for this exact purpose (used heavily in credit risk / fraud modeling).
    PSI < 0.1  -> no significant drift
    0.1 - 0.25 -> moderate drift, worth watching
    > 0.25     -> significant drift, consider retraining

Run with:
    python monitor.py
"""

import json
import sqlite3

import numpy as np
import pandas as pd

from train import load_data, engineer_features, FEATURE_COLUMNS

DB_PATH = "logs/predictions.db"


def get_training_distribution():
    """The feature distribution the model was originally trained on."""
    df = load_data()
    df = engineer_features(df)
    return df[FEATURE_COLUMNS]


def get_live_distribution():
    """The feature distribution of recent live predictions.

    Note: we log the raw transaction fields in `predictions`, not the full
    engineered feature vector (to keep the log table simple/light). For
    features directly available (amount), we compare directly. For a fuller
    production system, you'd log the complete feature vector at prediction
    time — noted as a known simplification in the README.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()
    return df


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between two distributions of a single feature."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    bucket_edges = np.percentile(expected, breakpoints)
    bucket_edges[0] = -np.inf
    bucket_edges[-1] = np.inf
    bucket_edges = np.unique(bucket_edges)  # guard against duplicate edges

    expected_counts, _ = np.histogram(expected, bins=bucket_edges)
    actual_counts, _ = np.histogram(actual, bins=bucket_edges)

    expected_pct = expected_counts / max(len(expected), 1)
    actual_pct = actual_counts / max(len(actual), 1)

    # Avoid division by zero / log(0) by flooring tiny percentages
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def interpret_psi(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    elif psi < 0.25:
        return "moderate drift — watch closely"
    else:
        return "significant drift — retraining recommended"


def main():
    print("Loading training distribution (baseline)...")
    train_features = get_training_distribution()

    print("Loading live prediction log...")
    live_df = get_live_distribution()

    if len(live_df) < 30:
        print(f"Only {len(live_df)} live predictions logged so far — "
              f"need at least ~30 for meaningful drift stats. "
              f"Send more requests to /predict first.")
        return

    print(f"\nComparing {len(live_df)} live predictions against "
          f"{len(train_features)} training examples.\n")

    # Compare the features we directly have in both places.
    comparable_features = ["amount"]

    results = {}
    for feature in comparable_features:
        psi = calculate_psi(
            train_features[feature].values,
            live_df[feature].values,
        )
        results[feature] = {"psi": round(psi, 4), "verdict": interpret_psi(psi)}
        print(f"{feature:30s} PSI={psi:.4f}  ->  {interpret_psi(psi)}")

    # Also report the live fraud-flag rate vs. training fraud rate — a
    # simple, intuitive drift signal on its own.
    training_fraud_rate = None
    try:
        raw_df = load_data()
        training_fraud_rate = raw_df["is_fraud"].mean()
    except Exception:
        pass

    live_flag_rate = live_df["is_flagged"].mean()
    print(f"\nTraining fraud rate: {training_fraud_rate:.4%}" if training_fraud_rate else "")
    print(f"Live flagged rate:   {live_flag_rate:.4%}")
    if training_fraud_rate and abs(live_flag_rate - training_fraud_rate) > training_fraud_rate:
        print("⚠️  Live flag rate is more than 2x the training fraud rate — "
              "investigate before trusting current predictions.")

    with open("logs/drift_report.json", "w") as f:
        json.dump({
            "feature_drift": results,
            "training_fraud_rate": training_fraud_rate,
            "live_flag_rate": live_flag_rate,
        }, f, indent=2)
    print("\nSaved full report to logs/drift_report.json")


if __name__ == "__main__":
    main()
