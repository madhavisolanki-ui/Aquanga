"""
Data Preprocessing Module for Aquanga
Handles ingestion, wide-to-long transformation, missing value imputation, and validation.
"""

import os
import re
import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_column_name(col: str) -> str:
    """Normalize raw column names."""
    return re.sub(r"\s+", " ", col).strip()


def parse_raw_cpcb_csv(file_path: str) -> pd.DataFrame:
    """
    Parses wide-format CPCB Ganga water quality CSV into a normalized long-format DataFrame.
    Dynamically identifies stations, years, DO, BOD, and Fecal Coliform columns.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw dataset not found at {file_path}")

    raw_df = pd.read_csv(file_path)
    raw_df.columns = [clean_column_name(c) for c in raw_df.columns]

    # Identify location column
    location_col = None
    for col in raw_df.columns:
        if "location" in col.lower():
            location_col = col
            break

    if not location_col:
        raise ValueError("Could not identify station/location column in dataset.")

    # Extract parameter columns and years
    records = []
    # Identify unique years present in headers
    years = set()
    for col in raw_df.columns:
        match = re.search(r"\b(19\d\d|20\d\d)\b", col)
        if match:
            years.add(int(match.group(1)))

    years = sorted(list(years))
    logger.info(f"Discovered years in dataset: {years}")

    for _, row in raw_df.iterrows():
        station_name = str(row[location_col]).strip()
        if not station_name or station_name.lower() == "nan":
            continue

        for yr in years:
            # Find DO, BOD, Fecal Coliform columns for this year
            do_val = np.nan
            bod_val = np.nan
            fc_val = np.nan

            for col in raw_df.columns:
                col_lower = col.lower()
                if str(yr) in col:
                    val_str = str(row[col]).strip()
                    val = pd.to_numeric(val_str, errors="coerce")

                    if "d.o" in col_lower or "do (" in col_lower or col_lower.startswith("do"):
                        do_val = val
                    elif "b.o.d" in col_lower or "bod" in col_lower:
                        bod_val = val
                    elif "fecal" in col_lower or "coliform" in col_lower:
                        fc_val = val

            records.append({
                "location": station_name,
                "year": yr,
                "do": do_val,
                "bod": bod_val,
                "fecal_coliform": fc_val
            })

    long_df = pd.DataFrame(records)
    logger.info(f"Parsed {len(long_df)} records across {long_df['location'].nunique()} stations.")
    return long_df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies robust missing value imputation:
    1. Temporal interpolation per station.
    2. Forward fill and backward fill per station.
    3. Global parameter median fallback if station has all missing values for a parameter.
    """
    df = df.copy()
    df = df.sort_values(by=["location", "year"]).reset_index(drop=True)

    numeric_cols = ["do", "bod", "fecal_coliform"]
    
    for col in numeric_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            logger.info(f"Imputing {missing_count} missing values in '{col}'...")

            # 1. Per-station linear interpolation across time
            df[col] = df.groupby("location")[col].transform(
                lambda s: s.interpolate(method="linear", limit_direction="both")
            )
            # 2. Per-station ffill/bfill
            df[col] = df.groupby("location")[col].transform(
                lambda s: s.ffill().bfill()
            )
            # 3. Global median fallback for any remaining NaNs
            if df[col].isna().sum() > 0:
                global_med = df[col].median()
                logger.info(f"Filling remaining NaNs in '{col}' with global median: {global_med}")
                df[col] = df[col].fillna(global_med)

    return df


def validate_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates physical plausibility and data types.
    """
    df = df.copy()
    
    # Ensure types
    df["location"] = df["location"].astype(str)
    df["year"] = df["year"].astype(int)
    df["do"] = df["do"].astype(float).clip(lower=0.0)
    df["bod"] = df["bod"].astype(float).clip(lower=0.0)
    df["fecal_coliform"] = df["fecal_coliform"].astype(float).clip(lower=0.0)

    # Drop any exact duplicate rows
    df = df.drop_duplicates(subset=["location", "year"]).reset_index(drop=True)
    return df


def run_preprocessing(
    raw_path: str = "data/raw/ganga_water_quality_2011_2015.csv",
    output_path: str = "data/processed/ganga_water_quality_clean.csv"
) -> pd.DataFrame:
    """
    Executes the full preprocessing pipeline.
    """
    logger.info("Starting data preprocessing pipeline...")
    raw_df = parse_raw_cpcb_csv(raw_path)
    imputed_df = impute_missing_values(raw_df)
    clean_df = validate_clean_data(imputed_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    logger.info(f"Cleaned dataset saved successfully to {output_path} with shape {clean_df.shape}")
    return clean_df


if __name__ == "__main__":
    run_preprocessing()
