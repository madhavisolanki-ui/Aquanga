"""
XGBoost Model Training Module for Aquanga
"""

import os
import joblib
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.create_sequences import get_chronological_split, fit_and_scale_data
from ml.feature_engineering import run_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_dir: str = "models/xgboost",
    random_state: int = 42
) -> XGBRegressor:
    """Trains and saves an XGBoost Regressor."""
    os.makedirs(model_dir, exist_ok=True)
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbosity=0
    )
    model.fit(X_train, y_train)
    model_path = os.path.join(model_dir, "xgboost_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"XGBoost Regressor trained and saved to {model_path}")
    return model


def run_xgboost_training() -> Dict[str, Dict[str, float]]:
    """Executes XGBoost model training and evaluation."""
    df = run_feature_engineering()
    X_tr, X_te, y_tr, y_te = get_chronological_split(df)
    X_tr_s, X_te_s, y_tr_s, y_te_s, scaler_X, scaler_y = fit_and_scale_data(X_tr, X_te, y_tr, y_te)

    model = train_xgboost(X_tr_s, y_tr_s)
    pred_s = model.predict(X_te_s)
    pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()

    mae = float(mean_absolute_error(y_te, pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    r2 = float(r2_score(y_te, pred))

    results = {
        "XGBoost": {"MAE": mae, "RMSE": rmse, "R2": r2}
    }
    logger.info(f"XGBoost Results: {results}")
    return results


if __name__ == "__main__":
    run_xgboost_training()
