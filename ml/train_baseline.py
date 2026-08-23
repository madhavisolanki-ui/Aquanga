"""
Baseline Models Training Module (Linear Regression & Random Forest) for Aquanga
"""

import os
import joblib
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.create_sequences import get_chronological_split, fit_and_scale_data
from ml.feature_engineering import run_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_dir: str = "models/baseline"
) -> LinearRegression:
    """Trains and persists an unregularized Linear Regression baseline."""
    os.makedirs(model_dir, exist_ok=True)
    model = LinearRegression()
    model.fit(X_train, y_train)
    model_path = os.path.join(model_dir, "linear_regression.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Linear Regression trained and saved to {model_path}")
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_dir: str = "models/baseline",
    random_state: int = 42
) -> RandomForestRegressor:
    """Trains and persists a Random Forest Regressor baseline."""
    os.makedirs(model_dir, exist_ok=True)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=2,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    model_path = os.path.join(model_dir, "random_forest.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Random Forest Regressor trained and saved to {model_path}")
    return model


def run_baseline_training() -> Dict[str, Dict[str, float]]:
    """Runs data loading, preprocessing, scaling, and baseline model training."""
    df = run_feature_engineering()
    X_tr, X_te, y_tr, y_te = get_chronological_split(df)
    X_tr_s, X_te_s, y_tr_s, y_te_s, scaler_X, scaler_y = fit_and_scale_data(X_tr, X_te, y_tr, y_te)

    # 1. Linear Regression
    lr_model = train_linear_regression(X_tr_s, y_tr_s)
    lr_pred_s = lr_model.predict(X_te_s)
    lr_pred = scaler_y.inverse_transform(lr_pred_s.reshape(-1, 1)).ravel()

    lr_mae = float(mean_absolute_error(y_te, lr_pred))
    lr_rmse = float(np.sqrt(mean_squared_error(y_te, lr_pred)))
    lr_r2 = float(r2_score(y_te, lr_pred))

    # 2. Random Forest
    rf_model = train_random_forest(X_tr_s, y_tr_s)
    rf_pred_s = rf_model.predict(X_te_s)
    rf_pred = scaler_y.inverse_transform(rf_pred_s.reshape(-1, 1)).ravel()

    rf_mae = float(mean_absolute_error(y_te, rf_pred))
    rf_rmse = float(np.sqrt(mean_squared_error(y_te, rf_pred)))
    rf_r2 = float(r2_score(y_te, rf_pred))

    results = {
        "Linear Regression": {"MAE": lr_mae, "RMSE": lr_rmse, "R2": lr_r2},
        "Random Forest": {"MAE": rf_mae, "RMSE": rf_rmse, "R2": rf_r2}
    }
    logger.info(f"Baseline Results: {results}")
    return results


if __name__ == "__main__":
    run_baseline_training()
