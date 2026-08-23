"""
Feature Engineering Module for Aquanga
Generates temporal lags, rate-of-change, CPCB compliance indicators, and composite quality scores.
"""

import os
import logging
from typing import List, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_water_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes environmental standards compliance flags and composite quality scores.
    CPCB Guidelines:
    - DO >= 5.0 mg/L is Good / Safe for aquatic life & bathing
    - BOD <= 3.0 mg/L is Good / Low organic pollution
    - Fecal Coliform <= 2500 MPN/100ml is Good / Permissible limit
    """
    df = df.copy()

    # CPCB Criteria Boolean flags
    df["do_good"] = (df["do"] >= 5.0).astype(int)
    df["bod_good"] = (df["bod"] <= 3.0).astype(int)
    df["fecal_good"] = (df["fecal_coliform"] <= 2500.0).astype(int)

    # Composite Water Quality Score (0 to 3)
    df["water_quality_score"] = df["do_good"] + df["bod_good"] + df["fecal_good"]

    # Year Index (0, 1, 2, ...)
    min_year = df["year"].min()
    df["year_index"] = df["year"] - min_year

    # Stable station ID mapping
    unique_stations = sorted(df["location"].unique())
    station_to_id = {st: idx for idx, st in enumerate(unique_stations)}
    df["station_id"] = df["location"].map(station_to_id)

    return df


def generate_lag_features(df: pd.DataFrame, lags: List[int] = [1, 2]) -> pd.DataFrame:
    """
    Generates temporal lag features and historical rate-of-change (velocity) features per station.
    Note: To prevent target leakage, do_change is defined as (do_lag1 - do_lag2), reflecting
    the observable trend prior to the forecast target step.
    """
    df = df.copy()
    df = df.sort_values(by=["location", "year"]).reset_index(drop=True)

    # Lag 1 features
    df["do_lag1"] = df.groupby("location")["do"].shift(1)
    df["bod_lag1"] = df.groupby("location")["bod"].shift(1)
    df["fecal_coliform_lag1"] = df.groupby("location")["fecal_coliform"].shift(1)

    # Lag 2 features for trend velocity
    df["do_lag2"] = df.groupby("location")["do"].shift(2)
    df["bod_lag2"] = df.groupby("location")["bod"].shift(2)
    df["fecal_coliform_lag2"] = df.groupby("location")["fecal_coliform"].shift(2)

    # Observable rate of change prior to forecast period (Lag1 - Lag2)
    df["do_change"] = (df["do_lag1"] - df["do_lag2"]).fillna(0.0)
    df["bod_change"] = (df["bod_lag1"] - df["bod_lag2"]).fillna(0.0)
    df["fecal_coliform_change"] = (df["fecal_coliform_lag1"] - df["fecal_coliform_lag2"]).fillna(0.0)

    # Drop rows where lag1 is NaN (first year of observation)
    df_features = df.dropna(subset=["do_lag1"]).reset_index(drop=True)
    logger.info(f"Generated features dataset with {len(df_features)} records after applying lag features.")
    return df_features


def run_feature_engineering(
    input_path: str = "data/processed/ganga_water_quality_clean.csv",
    output_path: str = "data/processed/ganga_water_quality_features.csv"
) -> pd.DataFrame:
    """
    Executes the full feature engineering pipeline.
    """
    if not os.path.exists(input_path):
        from ml.preprocessing import run_preprocessing
        clean_df = run_preprocessing(output_path=input_path)
    else:
        clean_df = pd.read_csv(input_path)

    features_df = generate_water_quality_features(clean_df)
    final_df = generate_lag_features(features_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info(f"Feature-engineered dataset saved to {output_path} with columns: {list(final_df.columns)}")
    return final_df


if __name__ == "__main__":
    run_feature_engineering()
