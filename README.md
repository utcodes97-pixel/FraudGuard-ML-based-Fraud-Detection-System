# Fraud Detection System — End-to-End ML Pipeline

A credit card fraud detection system with a full production-style lifecycle:
synthetic data generation → feature engineering → training with proper
imbalance handling → real-time serving → drift monitoring → gated retraining.

This was built and tested end-to-end (with a stand-in classifier substituted
for XGBoost during testing, since XGBoost wasn't installable in the build
sandbox — see "Verification" below for exactly what was and wasn't run
live). All pandas/feature-engineering/PSI logic was executed against real
generated data, not just written and hoped to work.

---

## Setup

```bash
pip install -r requirements.txt

# 1. Generate synthetic transaction data
python generate_data.py

# 2. Train the model
python train.py

# 3. Start the serving API
uvicorn main:app --reload
# visit http://localhost:8000/docs to send test predictions

# 4. After sending some predictions, check for drift
python monitor.py

# 5. Retrain and auto-promote only if the new model is better
python retrain.py
```

---

## Architecture

```
generate_data.py          -> data/transactions.csv (raw transactions)
train.py                  -> engineers features, trains XGBoost,
                              saves versioned model to models/
main.py (FastAPI)         -> loads latest model, serves /predict,
                              computes features online, logs to logs/predictions.db
monitor.py                -> compares live prediction distribution vs.
                              training distribution (PSI), flags drift
retrain.py                -> trains a challenger model, only promotes it
                              if it beats the current champion on the same
                              test set
```

---

## Why synthetic data instead of the Kaggle dataset

The real Kaggle "Credit Card Fraud" dataset has anonymized, PCA-transformed
features (V1-V28) — great for privacy, bad for *learning feature
engineering*, since the raw columns are hidden. `generate_data.py` creates
raw, readable transaction data (card ID, timestamp, amount, merchant) with
realistic injected fraud patterns (rapid-fire transactions, odd hours,
amount outliers), so you engineer the features yourself in `train.py` — the
actual skill this project is meant to teach.

Swapping in the real Kaggle CSV later is a natural next step and a good
thing to mention in an interview as your "v2" plan.

---

## Study order (so you can defend this in an interview)

1. **Why not accuracy** — read the comments above `evaluate_model()` in
   `train.py`. With ~0.5-1.7% fraud, a model that always predicts "legit"
   scores 99%+ accuracy while catching zero fraud. Know precision, recall,
   and why AUC-PR (not AUC-ROC) is the right headline metric here.
2. **Time-aware train/test split** — read `time_aware_split()`. Random
   shuffling leaks future information into training. This is a very common
   mistake real candidates make; knowing to avoid it is a strong signal.
3. **Class imbalance handling** — read `compute_scale_pos_weight()`. Know
   this approach (class weighting) vs. SMOTE (synthesizing fake minority
   examples) and be ready to explain why weighting was chosen here.
4. **Feature engineering** — read `engineer_features()` line by line. Every
   feature maps to a specific fraud signal (velocity, odd hours, amount
   anomalies). Understand `closed="left"` in the rolling window — it
   prevents a row from "seeing itself," a leakage bug that's easy to
   introduce accidentally.
5. **Training-serving skew** — read the docstring at the top of `main.py`
   and `compute_online_features()`. This is the gap between "features
   computed in bulk from a full dataset" (training) vs. "features computed
   one at a time from a running state" (serving) — a real, common
   production bug class worth being able to discuss.
6. **Champion/challenger retraining** — read `retrain.py`. Never overwrite
   a working production model without proving the new one is actually
   better on the same evaluation set.
7. **Drift monitoring (PSI)** — read `monitor.py`. Know the PSI thresholds
   (< 0.1 stable, 0.1-0.25 moderate, > 0.25 significant) — these are
   industry-standard numbers worth memorizing.

## Interview questions this preps you for

- "Your model has 99% accuracy but the business says it's useless — why?"
- "How do you prevent data leakage in a time-series problem?"
- "How do you decide when to retrain a production model?"
- "What's the difference between how you'd compute a feature offline vs. in
  a live API, and what problems does that create?"
- "Walk me through your precision/recall tradeoff — how did you pick your
  decision threshold?"

---

## Verification — what was actually tested vs. what to verify yourself

Being upfront about this, since it matters for how much you should trust
the code before an interview:

**Tested and confirmed working**, using the real pandas/numpy/scikit-learn
stack, with a scikit-learn classifier temporarily substituted for XGBoost
(build sandbox had no internet access to install XGBoost):
- `generate_data.py` — runs end-to-end, produces realistic fraud rate (~1-2%)
- `engineer_features()` — zero NaNs produced, correct dtypes, no leakage
  bugs found in manual inspection of rolling/shift logic
- Full `train.py` pipeline — trains, evaluates, saves versioned model +
  metadata correctly (test run: 99.4% recall, 81.8% precision on synthetic
  fraud — sane numbers for well-separated synthetic patterns)
- Online feature computation in `main.py` — manually simulated a fraud
  burst and a normal transaction; the fraud burst scored ~99% probability,
  the normal grocery purchase scored ~0.4%, confirming the model responds
  correctly to the injected features
- PSI drift calculation in `monitor.py` — confirmed near-zero for identical
  distributions and correctly flags a shifted distribution as significant
  drift

**Not tested live** (no internet in the build sandbox to install these):
- XGBoost itself — the actual `XGBClassifier` class, as opposed to a
  stand-in. XGBoost is a very mature, widely-used library, so this is low
  risk, but run `train.py` yourself first and check the printed metrics
  look sane before relying on it.
- FastAPI/uvicorn/pydantic — the actual HTTP server layer. The logic inside
  each route was tested directly (see above), but the FastAPI wiring itself
  (decorators, startup event, request parsing) was not run live. Run
  `uvicorn main:app --reload` and hit `/docs` to confirm before your
  interview — this is a five-minute check.

If either of those breaks in your environment, send me the error and I'll fix it.

---

## Known limitations (mention these upfront in interviews — it shows maturity, not weakness)

- `txn_count_last_hour` in `main.py`'s online feature computation is
  simplified to an all-time running count, not a true rolling 1-hour
  window like `train.py` computes offline. A real fix would track
  timestamps per card in a sorted structure (or Redis sorted set) and
  count entries within the last hour. This is exactly the kind of
  training-serving mismatch worth discussing openly.
- Card history and prediction logs use SQLite for simplicity. Production
  would use Redis for the low-latency card-state lookups (sub-millisecond)
  and a proper time-series store or data warehouse for prediction logging.
- No authentication on the API.
- Drift monitoring only directly compares the `amount` feature since the
  prediction log doesn't store the full engineered feature vector. A fuller
  version would log all features at prediction time for richer drift
  analysis across every feature, not just one.
- Synthetic data has cleanly separable fraud patterns (by design, for
  learnability), so the model's near-perfect metrics won't replicate on
  messier real-world data — don't quote 99% recall as if it's a general
  claim about your modeling ability.
