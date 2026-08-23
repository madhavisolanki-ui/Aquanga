"""
1D Convolutional Neural Network (1D CNN) Training Module for Aquanga
"""

import os
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.create_sequences import get_chronological_split, fit_and_scale_data, reshape_for_sequences
from ml.feature_engineering import run_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_cnn_model(input_shape: Tuple[int, int]):
    """Constructs a 1D CNN regression architecture."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout

    model = Sequential([
        Conv1D(filters=32, kernel_size=1, activation="relu", padding="same", input_shape=input_shape),
        Flatten(),
        Dense(32, activation="relu"),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=["mae"])
    return model


def train_cnn(
    X_train_seq: np.ndarray,
    y_train_s: np.ndarray,
    epochs: int = 60,
    batch_size: int = 4,
    model_dir: str = "models/cnn"
):
    """Trains and saves the 1D CNN model."""
    import tensorflow as tf
    os.makedirs(model_dir, exist_ok=True)
    tf.random.set_seed(42)
    np.random.seed(42)

    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = build_cnn_model(input_shape)
    
    model.fit(
        X_train_seq,
        y_train_s,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    model_path = os.path.join(model_dir, "cnn_model.keras")
    model.save(model_path)
    logger.info(f"1D CNN trained and saved to {model_path}")
    return model


def run_cnn_training() -> Dict[str, Dict[str, float]]:
    """Executes 1D CNN model training and evaluation."""
    df = run_feature_engineering()
    X_tr, X_te, y_tr, y_te = get_chronological_split(df)
    X_tr_s, X_te_s, y_tr_s, y_te_s, scaler_X, scaler_y = fit_and_scale_data(X_tr, X_te, y_tr, y_te)

    X_tr_seq = reshape_for_sequences(X_tr_s)
    X_te_seq = reshape_for_sequences(X_te_s)

    model = train_cnn(X_tr_seq, y_tr_s)
    pred_s = model.predict(X_te_seq, verbose=0).ravel()
    pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()

    mae = float(mean_absolute_error(y_te, pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    r2 = float(r2_score(y_te, pred))

    results = {
        "1D CNN": {"MAE": mae, "RMSE": rmse, "R2": r2}
    }
    logger.info(f"1D CNN Results: {results}")
    return results


if __name__ == "__main__":
    run_cnn_training()
