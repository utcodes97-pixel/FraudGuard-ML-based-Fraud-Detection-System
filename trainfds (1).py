"""
train.py — feature engineering + model training + evaluation.

Run after generate_data.py. This is the file that shows actual ML judgment,
not just library calls — read every comment, this is your interview
ammunition.

USAGE:
    python train.py
"""

import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, confusion_matrix,
    classification_report, average_precision_score,
)
import xgboost as xgb


# ---------------------------------------------------------------------------
# 1. LOAD RAW DATA
# ---------------------------------------------------------------------------

def load_data(path="data/transactions.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
# This is the part that separates "ran a library" from "did ML engineering."
# Raw columns (card_id, timestamp, amount, merchant) aren't directly useful
# to a model — we derive signal from them.
#
# Each feature below maps to one of the fraud patterns we know exist in the
# data (see generate_data.py). In a real job, you wouldn't know the patterns
# in advance — you'd hypothesize, test, and iterate. Here we engineer
# features that SHOULD catch our known patterns, which is a good way to
# learn what "good" feature engineering looks like.

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Time-based features ---
    df["hour"] = df["timestamp"].dt.hour
    df["is_odd_hour"] = df["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # --- Per-card historical features ---
    # IMPORTANT: these must be computed using only PAST transactions for
    # each row (no peeking at the future) — this mirrors what's possible in
    # a real-time production system, where you only know history up to now.
    df = df.sort_values(["card_id", "timestamp"]).reset_index(drop=True)

    # Time since this card's previous transaction, in seconds.
    # A very small value = rapid-fire transactions = fraud signal.
    df["prev_txn_time"] = df.groupby("card_id")["timestamp"].shift(1)
    df["seconds_since_last_txn"] = (
        (df["timestamp"] - df["prev_txn_time"]).dt.total_seconds()
    )
    # First transaction for a card has no "previous" — fill with a large
    # number (treat as "not suspicious", since there's no burst to detect).
    df["seconds_since_last_txn"] = df["seconds_since_last_txn"].fillna(999999)

    # Rolling count of transactions by this card in the last hour.
    # We do this with a manual loop-free approach: for each row, count prior
    # rows for the same card within 3600 seconds. Using a groupby + rolling
    # window on time keeps this correct and reasonably efficient.
    df = df.set_index("timestamp")
    df["txn_count_last_hour"] = (
        df.groupby("card_id")["amount"]
        .rolling("1h", closed="left")  # closed='left' excludes current row -> no leakage
        .count()
        .reset_index(level=0, drop=True)
    )
    df = df.reset_index()
    df["txn_count_last_hour"] = df["txn_count_last_hour"].fillna(0)

    # --- Amount-based features ---
    # Card's historical average amount, computed using only past transactions.
    df["card_avg_amount_so_far"] = (
        df.groupby("card_id")["amount"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["card_avg_amount_so_far"] = df["card_avg_amount_so_far"].fillna(df["amount"].median())

    # How many standard deviations this transaction's amount is from what's
    # "normal" for this card — a simple z-score-style outlier signal.
    df["amount_vs_card_avg_ratio"] = df["amount"] / df["card_avg_amount_so_far"].replace(0, 1)

    # --- Merchant category (one-hot encode — models need numeric input) ---
    # dtype=int (not the pandas default bool) so training-time columns match
    # serving-time columns exactly, which build plain 0/1 ints in main.py.
    df = pd.get_dummies(df, columns=["merchant_category"], prefix="merchant", dtype=int)

    return df


FEATURE_COLUMNS = [
    "amount", "hour", "is_odd_hour", "day_of_week",
    "seconds_since_last_txn", "txn_count_last_hour",
    "card_avg_amount_so_far", "amount_vs_card_avg_ratio",
] + [c for c in ["merchant_grocery", "merchant_restaurant", "merchant_gas_station",
                 "merchant_online_retail", "merchant_electronics", "merchant_travel",
                 "merchant_pharmacy", "merchant_entertainment"]]


# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT — TIME-AWARE, NOT RANDOM
# ---------------------------------------------------------------------------
# A random shuffle split would let the model "see the future" indirectly
# (e.g. a card's later transactions influencing features used to predict
# earlier ones during cross-validation folds). Real fraud systems are always
# evaluated on a strict time cutoff: train on the past, test on the future.
# This is a very common thing candidates get wrong, so implementing it
# correctly and being able to explain it is a strong signal.

def time_aware_split(df: pd.DataFrame, test_fraction: float = 0.2):
    cutoff_idx = int(len(df) * (1 - test_fraction))
    cutoff_time = df.sort_values("timestamp").iloc[cutoff_idx]["timestamp"]
    train_df = df[df["timestamp"] < cutoff_time]
    test_df = df[df["timestamp"] >= cutoff_time]
    return train_df, test_df


# ---------------------------------------------------------------------------
# 4. HANDLE CLASS IMBALANCE
# ---------------------------------------------------------------------------
# Fraud is ~0.5% of transactions. A model can get 99.5% "accuracy" by
# predicting "not fraud" every single time — completely useless.
#
# We use XGBoost's built-in `scale_pos_weight` parameter, which upweights
# the minority class in the loss function during training. This is simpler
# than SMOTE (which synthesizes fake minority examples) and works well for
# tree-based models. Worth knowing SMOTE exists and why we chose weighting
# here: SMOTE can introduce unrealistic synthetic fraud patterns, whereas
# class weighting just tells the model "mistakes on fraud cost more,"
# without inventing fake data.

def compute_scale_pos_weight(y_train):
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    return n_negative / max(n_positive, 1)


# ---------------------------------------------------------------------------
# 5. TRAIN THE MODEL
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, scale_pos_weight):
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",  # area under precision-recall curve — the right
                               # metric for imbalanced classification, unlike
                               # plain accuracy or even ROC-AUC alone.
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 6. EVALUATE — WHY NOT ACCURACY
# ---------------------------------------------------------------------------
# With 0.5% fraud, a dumb "always predict legit" model scores 99.5% accuracy
# and catches ZERO fraud. So we report:
#   - Precision: of transactions we flagged as fraud, how many really were?
#   - Recall: of all real fraud, how much did we catch?
#   - AUC-PR: overall ranking quality across all thresholds (better than
#     ROC-AUC for heavily imbalanced problems, since ROC-AUC can look
#     deceptively good even when precision at useful thresholds is poor)
#   - Confusion matrix at our chosen decision threshold

def evaluate_model(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    auc_pr = average_precision_score(y_test, probs)
    auc_roc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=["legit", "fraud"], output_dict=True)

    print(f"AUC-PR (precision-recall, the metric that matters here): {auc_pr:.4f}")
    print(f"AUC-ROC (secondary metric): {auc_roc:.4f}")
    print(f"Confusion matrix at threshold={threshold}:")
    print(f"  [[TN={cm[0][0]}, FP={cm[0][1]}],")
    print(f"   [FN={cm[1][0]}, TP={cm[1][1]}]]")
    print(f"Precision (fraud class): {report['fraud']['precision']:.4f}")
    print(f"Recall (fraud class):    {report['fraud']['recall']:.4f}")

    return {
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "precision_fraud": report["fraud"]["precision"],
        "recall_fraud": report["fraud"]["recall"],
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# 7. SAVE MODEL WITH VERSIONING
# ---------------------------------------------------------------------------
# Production ML systems never overwrite a model in place — you version it,
# so you can roll back, compare, and audit what was deployed when.

def save_model(model, metrics, feature_columns, version_dir="models"):
    import os
    os.makedirs(version_dir, exist_ok=True)

    # Version = timestamp, so every training run is uniquely identifiable.
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = f"{version_dir}/model_{version}.pkl"
    meta_path = f"{version_dir}/model_{version}_meta.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "version": version,
        "trained_at": datetime.utcnow().isoformat(),
        "feature_columns": feature_columns,
        "metrics": metrics,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Also write/overwrite a "latest" pointer file so the serving app always
    # knows which model to load without hardcoding a version.
    with open(f"{version_dir}/latest.json", "w") as f:
        json.dump({"model_path": model_path, "meta_path": meta_path}, f, indent=2)

    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"'latest' pointer updated -> this is now the production model")

    return model_path, meta_path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    df = load_data()

    print("Engineering features...")
    df = engineer_features(df)

    print("Splitting train/test (time-aware, no shuffling)...")
    train_df, test_df = time_aware_split(df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["is_fraud"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["is_fraud"]

    print(f"Train size: {len(X_train)} ({y_train.sum()} fraud)")
    print(f"Test size:  {len(X_test)} ({y_test.sum()} fraud)")

    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.1f}")

    print("Training XGBoost model...")
    model = train_model(X_train, y_train, scale_pos_weight)

    print("\nEvaluating on held-out test set:")
    metrics = evaluate_model(model, X_test, y_test)

    save_model(model, metrics, FEATURE_COLUMNS)


if __name__ == "__main__":
    main()
