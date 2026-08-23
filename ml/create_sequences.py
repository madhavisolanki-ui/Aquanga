"""
Sequence & Dataset Preparation Module for Aquanga
Creates tabular and 3D time-series sliding window sequences for ML & DL architectures.
"""

import os
import joblib
import logging
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "do_lag1",
    "bod_lag1",
    "fecal_coliform_lag1",
    "do_change",
    "bod_change",
    "fecal_coliform_change",
    "do_good",
    "bod_good",
    "fecal_good",
    "water_quality_score",
    "year_index",
    "station_id"
]

TARGET_COLUMN = "do"


def get_feature_columns() -> List[str]:
    """Returns the standardized list of input feature columns."""
    return list(FEATURE_COLUMNS)


def get_chronological_split(
    df: pd.DataFrame,
    test_year: Optional[int] = None,
    target_col: str = TARGET_COLUMN,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data chronologically to prevent temporal data leakage.
    Default: Training on all years prior to the latest year, Testing on the latest year.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    df_sorted = df.sort_values(by=["year", "location"]).reset_index(drop=True)
    if test_year is None:
        test_year = int(df_sorted["year"].max())

    train_mask = df_sorted["year"] < test_year
    test_mask = df_sorted["year"] == test_year

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        # Fallback if only 1 year exists or custom split
        split_idx = int(len(df_sorted) * 0.8)
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
    else:
        train_df = df_sorted[train_mask]
        test_df = df_sorted[test_mask]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    logger.info(f"Chronological split at test_year={test_year}: Train shape={X_train.shape}, Test shape={X_test.shape}")
    return X_train, X_test, y_train, y_test


def fit_and_scale_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler_dir: str = "models"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """
    Fits standard scalers on training set ONLY to prevent leakage, scales train and test sets,
    and persists the scalers to disk.
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler_X, os.path.join(scaler_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(scaler_dir, "scaler_y.pkl"))
    logger.info(f"Fitted and saved scalers to {scaler_dir}/")

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y


def reshape_for_sequences(
    X_scaled: np.ndarray,
    time_steps: int = 1
) -> np.ndarray:
    """
    Reshapes 2D tabular features (N, features) into 3D sequence tensors (N, time_steps, features)
    suitable for 1D CNN, LSTM, and hybrid CNN-LSTM networks.
    """
    # X_scaled shape: (samples, features) -> (samples, time_steps, features // time_steps)
    if time_steps == 1:
        return X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    else:
        samples = X_scaled.shape[0]
        features = X_scaled.shape[1]
        return X_scaled.reshape((samples, time_steps, features // time_steps))


if __name__ == "__main__":
    from ml.feature_engineering import run_feature_engineering
    df = run_feature_engineering()
    X_tr, X_te, y_tr, y_te = get_chronological_split(df)
    X_tr_s, X_te_s, y_tr_s, y_te_s, sc_x, sc_y = fit_and_scale_data(X_tr, X_te, y_tr, y_te)
    X_tr_seq = reshape_for_sequences(X_tr_s)
    print("Tabular X_train shape:", X_tr_s.shape)
    print("Sequence X_train shape:", X_tr_seq.shape)
