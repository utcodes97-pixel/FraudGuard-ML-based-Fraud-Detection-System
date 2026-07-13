"""
main.py — serving layer for the fraud detection model.

THE HARD PART THIS FILE SOLVES: ONLINE FEATURE COMPUTATION.
In train.py, features like "seconds since this card's last transaction" or
"this card's average amount so far" were computed by looking across the
whole historical dataset with pandas. In a live API, there IS no dataset —
just one incoming transaction at a time. So we need to maintain a small
running "card history" store (SQLite) that updates after every prediction,
and use it to compute the same features on the fly.

This train/serve feature mismatch is one of the most common real-world ML
bugs (often called "training-serving skew") — this file exists specifically
to handle it correctly, and it's worth being able to explain in an
interview.

Run with:
    uvicorn main:app --reload
Then visit http://localhost:8000/docs
"""

import json
import pickle
import sqlite3
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from train import FEATURE_COLUMNS  # reuse the exact same feature list used in training

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_retail",
    "electronics", "travel", "pharmacy", "entertainment",
]

DB_PATH = "logs/predictions.db"

app = FastAPI(title="Fraud Detection API")

# Loaded once at startup — NOT reloaded per request. Reloading a model file
# from disk on every request adds latency and is a common serving mistake.
model = None
model_version = None


# ---------------------------------------------------------------------------
# STARTUP: load latest model + init logging DB
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup():
    global model, model_version
    with open("models/latest.json") as f:
        pointer = json.load(f)
    with open(pointer["model_path"], "rb") as f:
        model = pickle.load(f)
    with open(pointer["meta_path"]) as f:
        meta = json.load(f)
    model_version = meta["version"]
    print(f"Loaded model version {model_version}")

    init_db()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    import os
    os.makedirs("logs", exist_ok=True)
    conn = get_db()
    # Card history: the minimal running state needed to compute online
    # features. In production this would live in Redis for speed, since
    # it's a simple key-value lookup by card_id.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS card_history (
            card_id TEXT PRIMARY KEY,
            last_txn_time TEXT,
            txn_count INTEGER DEFAULT 0,
            running_avg_amount REAL DEFAULT 0
        )
    """)
    # Predictions log: every request + response, the raw material for
    # monitor.py's drift detection.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            card_id TEXT,
            timestamp TEXT,
            amount REAL,
            merchant_category TEXT,
            fraud_probability REAL,
            is_flagged INTEGER,
            model_version TEXT,
            predicted_at TEXT
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    transaction_id: str
    card_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    amount: float
    merchant_category: str


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_flagged: bool
    model_version: str


# ---------------------------------------------------------------------------
# ONLINE FEATURE COMPUTATION
# ---------------------------------------------------------------------------

def compute_online_features(txn: TransactionRequest) -> dict:
    """Compute the same features used in training, but using only the
    running per-card state we've stored so far — mirroring what's actually
    knowable at prediction time in a live system."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM card_history WHERE card_id = ?", (txn.card_id,)
    ).fetchone()

    if row is None:
        seconds_since_last_txn = 999999.0
        txn_count_last_hour = 0  # simplified: true production would track a
                                  # rolling window, not just all-time count
        card_avg_amount_so_far = txn.amount  # no history yet
    else:
        last_txn_time = datetime.fromisoformat(row["last_txn_time"])
        seconds_since_last_txn = (txn.timestamp - last_txn_time).total_seconds()
        txn_count_last_hour = row["txn_count"]  # simplified for this project;
                                                  # see README limitations
        card_avg_amount_so_far = row["running_avg_amount"]

    features = {
        "amount": txn.amount,
        "hour": txn.timestamp.hour,
        "is_odd_hour": int(txn.timestamp.hour in [0, 1, 2, 3, 4, 23]),
        "day_of_week": txn.timestamp.weekday(),
        "seconds_since_last_txn": seconds_since_last_txn,
        "txn_count_last_hour": txn_count_last_hour,
        "card_avg_amount_so_far": card_avg_amount_so_far,
        "amount_vs_card_avg_ratio": txn.amount / max(card_avg_amount_so_far, 1),
    }

    # One-hot encode merchant category — must match training exactly
    for cat in MERCHANT_CATEGORIES:
        features[f"merchant_{cat}"] = int(txn.merchant_category == cat)

    conn.close()
    return features


def update_card_history(txn: TransactionRequest):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM card_history WHERE card_id = ?", (txn.card_id,)
    ).fetchone()

    if row is None:
        conn.execute(
            "INSERT INTO card_history (card_id, last_txn_time, txn_count, running_avg_amount) "
            "VALUES (?, ?, ?, ?)",
            (txn.card_id, txn.timestamp.isoformat(), 1, txn.amount),
        )
    else:
        new_count = row["txn_count"] + 1
        # Incremental running average — avoids storing full history just to
        # compute a mean.
        new_avg = row["running_avg_amount"] + (txn.amount - row["running_avg_amount"]) / new_count
        conn.execute(
            "UPDATE card_history SET last_txn_time = ?, txn_count = ?, running_avg_amount = ? "
            "WHERE card_id = ?",
            (txn.timestamp.isoformat(), new_count, new_avg, txn.card_id),
        )
    conn.commit()
    conn.close()


def log_prediction(txn: TransactionRequest, probability: float, flagged: bool):
    conn = get_db()
    conn.execute(
        "INSERT INTO predictions (transaction_id, card_id, timestamp, amount, "
        "merchant_category, fraud_probability, is_flagged, model_version, predicted_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            txn.transaction_id, txn.card_id, txn.timestamp.isoformat(), txn.amount,
            txn.merchant_category, probability, int(flagged), model_version,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

FRAUD_THRESHOLD = 0.5  # tune this based on the precision/recall tradeoff you want


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = compute_online_features(txn)
    # Build a single-row DataFrame with columns in the EXACT order used
    # during training — XGBoost is sensitive to column order/names mismatch.
    X = pd.DataFrame([features])[FEATURE_COLUMNS]

    probability = float(model.predict_proba(X)[0, 1])
    flagged = probability >= FRAUD_THRESHOLD

    log_prediction(txn, probability, flagged)
    update_card_history(txn)

    return PredictionResponse(
        transaction_id=txn.transaction_id,
        fraud_probability=round(probability, 4),
        is_flagged=flagged,
        model_version=model_version,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_version": model_version}


@app.get("/")
def root():
    return {"message": "Fraud Detection API — see /docs for interactive testing"}
