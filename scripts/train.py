"""
Master Training Pipeline Script for Aquanga
Orchestrates preprocessing, feature engineering, chronological splitting,
training of all 6 ML & DL models, evaluation, and model artifact persistence.
"""

import os
import sys
import json
import logging
import joblib
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.preprocessing import run_preprocessing
from ml.feature_engineering import run_feature_engineering
from ml.create_sequences import (
    get_chronological_split,
    fit_and_scale_data,
    reshape_for_sequences,
    get_feature_columns
)
from ml.train_baseline import train_linear_regression, train_random_forest
from ml.train_xgboost import train_xgboost
from ml.train_cnn import train_cnn
from ml.train_lstm import train_lstm
from ml.train_cnn_lstm import train_cnn_lstm
from ml.evaluate import evaluate_all_models, save_evaluation_reports

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_full_training_pipeline():
    """Runs end-to-end model training, evaluation, and artifact saving."""
    logger.info("=== STEP 1: Running Data Preprocessing ===")
    clean_df = run_preprocessing()

    logger.info("=== STEP 2: Running Feature Engineering ===")
    features_df = run_feature_engineering()

    logger.info("=== STEP 3: Chronological Splitting & Scaling ===")
    X_train, X_test, y_train, y_test = get_chronological_split(features_df)
    X_train_s, X_test_s, y_train_s, y_test_s, scaler_X, scaler_y = fit_and_scale_data(
        X_train, X_test, y_train, y_test
    )

    X_train_seq = reshape_for_sequences(X_train_s)
    X_test_seq = reshape_for_sequences(X_test_s)

    logger.info("=== STEP 4: Training All 6 Models ===")
    models = {}

    # 1. Linear Regression
    logger.info("Training Model 1/6: Linear Regression...")
    models["Linear Regression"] = train_linear_regression(X_train_s, y_train_s)

    # 2. Random Forest
    logger.info("Training Model 2/6: Random Forest...")
    models["Random Forest"] = train_random_forest(X_train_s, y_train_s)

    # 3. XGBoost
    logger.info("Training Model 3/6: XGBoost...")
    models["XGBoost"] = train_xgboost(X_train_s, y_train_s)

    # 4. 1D CNN
    logger.info("Training Model 4/6: 1D CNN...")
    models["1D CNN"] = train_cnn(X_train_seq, y_train_s)

    # 5. LSTM
    logger.info("Training Model 5/6: LSTM...")
    models["LSTM"] = train_lstm(X_train_seq, y_train_s)

    # 6. CNN + LSTM
    logger.info("Training Model 6/6: CNN + LSTM...")
    models["CNN + LSTM"] = train_cnn_lstm(X_train_seq, y_train_s)

    logger.info("=== STEP 5: Model Evaluation & Comparison ===")
    comparison_df = evaluate_all_models(
        models_dict=models,
        X_test_s=X_test_s,
        y_test=y_test.values,
        scaler_y=scaler_y
    )

    print("\n" + "=" * 60)
    print("           AQUANGA MODEL COMPARISON TABLE")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60 + "\n")

    save_evaluation_reports(comparison_df)

    # Best model selection by lowest MAE
    best_model_name = comparison_df.iloc[0]["Model"]
    logger.info(f"Best Model Selected: {best_model_name}")

    # Persist best model as the default production do_prediction_model
    best_model_info = {
        "best_model_name": best_model_name,
        "mae": float(comparison_df.iloc[0]["MAE"]),
        "rmse": float(comparison_df.iloc[0]["RMSE"]),
        "r2": float(comparison_df.iloc[0]["R2"]),
        "feature_columns": get_feature_columns()
    }

    with open("models/best_model_info.json", "w") as f:
        json.dump(best_model_info, f, indent=2)

    # Copy / save primary model
    if best_model_name in ["1D CNN", "LSTM", "CNN + LSTM"]:
        models[best_model_name].save("models/do_prediction_model.keras")
    else:
        joblib.dump(models[best_model_name], "models/do_prediction_model.pkl")

    logger.info(f"Saved primary production model: {best_model_name}")
    return comparison_df


if __name__ == "__main__":
    run_full_training_pipeline()
