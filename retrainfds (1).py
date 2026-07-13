"""
retrain.py — retraining pipeline with promotion gating.

THE CORE IDEA: never blindly overwrite a production model with a freshly
retrained one. Train a "challenger" model, evaluate it on the same held-out
test set as the current "champion" model, and only promote the challenger
if it's actually better. This prevents a bad retraining run (bad data,
bug, overfitting) from silently degrading production.

This mirrors real MLOps practice — champion/challenger evaluation is a
standard pattern worth naming in an interview.

Run with:
    python retrain.py
"""

import json
import pickle

from train import (
    load_data, engineer_features, time_aware_split, FEATURE_COLUMNS,
    compute_scale_pos_weight, train_model, evaluate_model, save_model,
)


def load_current_champion():
    with open("models/latest.json") as f:
        pointer = json.load(f)
    with open(pointer["model_path"], "rb") as f:
        champion_model = pickle.load(f)
    with open(pointer["meta_path"]) as f:
        champion_meta = json.load(f)
    return champion_model, champion_meta


def main():
    print("Loading current production model (the 'champion')...")
    try:
        champion_model, champion_meta = load_current_champion()
        champion_auc_pr = champion_meta["metrics"]["auc_pr"]
        print(f"Champion version: {champion_meta['version']}, AUC-PR: {champion_auc_pr:.4f}")
    except FileNotFoundError:
        print("No existing model found — train.py must be run at least once first.")
        return

    print("\nLoading data and retraining a challenger model...")
    df = load_data()
    df = engineer_features(df)
    train_df, test_df = time_aware_split(df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["is_fraud"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["is_fraud"]

    scale_pos_weight = compute_scale_pos_weight(y_train)
    challenger_model = train_model(X_train, y_train, scale_pos_weight)

    print("\nEvaluating challenger on the same held-out test set...")
    challenger_metrics = evaluate_model(challenger_model, X_test, y_test)
    challenger_auc_pr = challenger_metrics["auc_pr"]

    print(f"\nChampion AUC-PR:   {champion_auc_pr:.4f}")
    print(f"Challenger AUC-PR: {challenger_auc_pr:.4f}")

    if challenger_auc_pr > champion_auc_pr:
        print("\n✅ Challenger beats champion — promoting to production.")
        save_model(challenger_model, challenger_metrics, FEATURE_COLUMNS)
    else:
        print("\n❌ Challenger did not beat champion — keeping current production model.")
        print("(The challenger was evaluated but NOT saved as the new 'latest' model.)")


if __name__ == "__main__":
    main()
