"""
Model Evaluation & Benchmark Comparison Module for Aquanga
Compares Linear Regression, Random Forest, XGBoost, 1D CNN, LSTM, and CNN+LSTM.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from ml.create_sequences import get_chronological_split, fit_and_scale_data, reshape_for_sequences
from ml.feature_engineering import run_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_all_models(
    models_dict: Dict[str, Any],
    X_test_s: np.ndarray,
    y_test: np.ndarray,
    scaler_y: Any
) -> pd.DataFrame:
    """
    Evaluates all trained model instances on test set and computes MAE, RMSE, R2.
    """
    X_test_seq = reshape_for_sequences(X_test_s)
    results = []

    for name, model in models_dict.items():
        try:
            if name in ["1D CNN", "LSTM", "CNN + LSTM"]:
                pred_s = model.predict(X_test_seq, verbose=0).ravel()
            else:
                pred_s = model.predict(X_test_s).ravel()

            pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()

            mae = float(mean_absolute_error(y_test, pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            r2 = float(r2_score(y_test, pred))

            results.append({
                "Model": name,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4)
            })
            logger.info(f"Model {name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        except Exception as e:
            logger.error(f"Error evaluating model {name}: {e}")

    results_df = pd.DataFrame(results).sort_values(by="MAE").reset_index(drop=True)
    return results_df


def save_evaluation_reports(
    results_df: pd.DataFrame,
    output_dir: str = "models"
) -> Tuple[str, str]:
    """Saves model comparison table to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "model_comparison.json")
    csv_path = os.path.join(output_dir, "model_comparison.csv")

    results_df.to_json(json_path, orient="records", indent=2)
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved evaluation results to {json_path} and {csv_path}")
    return json_path, csv_path


if __name__ == "__main__":
    from scripts.train import run_full_training_pipeline
    run_full_training_pipeline()
