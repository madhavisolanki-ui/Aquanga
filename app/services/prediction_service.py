"""
Prediction Service for Aquanga
Loads trained models and scalers, formats input features, performs inference,
and computes water-quality risk and warnings.
"""

import os
import joblib
import logging
from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd

from ml.create_sequences import get_feature_columns, reshape_for_sequences
from app.services.risk_service import calculate_risk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class WaterQualityPredictor:
    """Manages model loading and inference for Dissolved Oxygen forecasting."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.scaler_X = None
        self.scaler_y = None
        self.cached_models = {}
        self.feature_columns = get_feature_columns()
        self._load_scalers()

    def _load_scalers(self):
        """Loads fitted scalers."""
        scaler_x_path = os.path.join(self.models_dir, "scaler_X.pkl")
        scaler_y_path = os.path.join(self.models_dir, "scaler_y.pkl")

        if os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path):
            self.scaler_X = joblib.load(scaler_x_path)
            self.scaler_y = joblib.load(scaler_y_path)
            logger.info("Loaded scaler_X and scaler_y.")
        else:
            logger.warning("Scalers not found. Ensure models are trained first.")

    def load_model(self, model_name: str = "best") -> Any:
        """Dynamically loads requested model architecture."""
        if model_name in self.cached_models:
            return self.cached_models[model_name]

        model_name_lower = model_name.lower().replace(" ", "_").replace("+", "_")
        model = None

        if model_name_lower in ["linear_regression", "baseline"]:
            path = os.path.join(self.models_dir, "baseline", "linear_regression.pkl")
            if os.path.exists(path):
                model = joblib.load(path)
        elif model_name_lower in ["random_forest"]:
            path = os.path.join(self.models_dir, "baseline", "random_forest.pkl")
            if os.path.exists(path):
                model = joblib.load(path)
        elif model_name_lower in ["xgboost"]:
            path = os.path.join(self.models_dir, "xgboost", "xgboost_model.pkl")
            if os.path.exists(path):
                model = joblib.load(path)
        elif model_name_lower in ["1d_cnn", "cnn"]:
            path = os.path.join(self.models_dir, "cnn", "cnn_model.keras")
            if os.path.exists(path):
                import tensorflow as tf
                model = tf.keras.models.load_model(path)
        elif model_name_lower in ["lstm"]:
            path = os.path.join(self.models_dir, "lstm", "lstm_model.keras")
            if os.path.exists(path):
                import tensorflow as tf
                model = tf.keras.models.load_model(path)
        elif model_name_lower in ["cnn_lstm", "cnn_lstm_model"]:
            path = os.path.join(self.models_dir, "cnn_lstm", "cnn_lstm_model.keras")
            if os.path.exists(path):
                import tensorflow as tf
                model = tf.keras.models.load_model(path)
        
        # Fallback to default production model
        if model is None:
            default_pkl = os.path.join(self.models_dir, "do_prediction_model.pkl")
            default_keras = os.path.join(self.models_dir, "do_prediction_model.keras")
            if os.path.exists(default_pkl):
                model = joblib.load(default_pkl)
            elif os.path.exists(default_keras):
                import tensorflow as tf
                model = tf.keras.models.load_model(default_keras)

        if model is None:
            raise FileNotFoundError(f"Model '{model_name}' could not be loaded from {self.models_dir}.")

        self.cached_models[model_name] = model
        return model

    def build_feature_dataframe(
        self,
        do_lag1: float,
        bod_lag1: float,
        fecal_coliform_lag1: float,
        do_change: float = 0.0,
        bod_change: float = 0.0,
        fecal_coliform_change: float = 0.0,
        year_index: int = 5,
        station_id: int = 0
    ) -> pd.DataFrame:
        """Constructs a DataFrame matching training feature columns."""
        do_good = 1 if do_lag1 >= 5.0 else 0
        bod_good = 1 if bod_lag1 <= 3.0 else 0
        fecal_good = 1 if fecal_coliform_lag1 <= 2500.0 else 0
        water_quality_score = do_good + bod_good + fecal_good

        data = {
            "do_lag1": [do_lag1],
            "bod_lag1": [bod_lag1],
            "fecal_coliform_lag1": [fecal_coliform_lag1],
            "do_change": [do_change],
            "bod_change": [bod_change],
            "fecal_coliform_change": [fecal_coliform_change],
            "do_good": [do_good],
            "bod_good": [bod_good],
            "fecal_good": [fecal_good],
            "water_quality_score": [water_quality_score],
            "year_index": [year_index],
            "station_id": [station_id]
        }
        return pd.DataFrame(data)[self.feature_columns]

    def build_feature_vector(self, *args, **kwargs) -> np.ndarray:
        """Helper to return feature array directly."""
        return self.build_feature_dataframe(*args, **kwargs).values

    def predict(
        self,
        station_name: str,
        do_lag1: float,
        bod_lag1: float,
        fecal_coliform_lag1: float,
        do_change: float = 0.0,
        bod_change: float = 0.0,
        fecal_coliform_change: float = 0.0,
        year: int = 2016,
        station_id: int = 0,
        model_name: str = "best"
    ) -> Dict[str, Any]:
        """
        Executes prediction pipeline and returns formatted response with risk scores.
        """
        if self.scaler_X is None or self.scaler_y is None:
            self._load_scalers()

        model = self.load_model(model_name)
        year_index = year - 2011

        raw_df = self.build_feature_dataframe(
            do_lag1=do_lag1,
            bod_lag1=bod_lag1,
            fecal_coliform_lag1=fecal_coliform_lag1,
            do_change=do_change,
            bod_change=bod_change,
            fecal_coliform_change=fecal_coliform_change,
            year_index=year_index,
            station_id=station_id
        )

        features_scaled = self.scaler_X.transform(raw_df)

        # Check if deep learning sequence model
        is_seq_model = "keras" in str(type(model)).lower() or any(
            k in model_name.lower() for k in ["cnn", "lstm"]
        )

        if is_seq_model:
            features_seq = reshape_for_sequences(features_scaled)
            pred_scaled = model.predict(features_seq, verbose=0).ravel()
        else:
            pred_scaled = model.predict(features_scaled).ravel()

        pred_do = float(self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
        pred_do = max(0.0, round(pred_do, 3))

        # Risk assessment based on predicted DO and latest known BOD/Fecal Coliform
        risk_data = calculate_risk(
            do=pred_do,
            bod=bod_lag1,
            fecal_coliform=fecal_coliform_lag1
        )

        return {
            "station": station_name,
            "station_id": station_id,
            "forecast_year": year,
            "model_used": model_name,
            "predicted_do": pred_do,
            "risk_score": risk_data["risk_score"],
            "risk_level": risk_data["risk_level"],
            "warning": risk_data["warning"],
            "parameters": {
                "do_lag1": do_lag1,
                "bod_lag1": bod_lag1,
                "fecal_coliform_lag1": fecal_coliform_lag1
            }
        }


# Singleton predictor instance
predictor = WaterQualityPredictor()
