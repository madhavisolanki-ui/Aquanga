"""
Unit tests for data preprocessing and feature engineering pipelines.
"""

import pytest
import pandas as pd
import numpy as np

from ml.preprocessing import impute_missing_values, validate_clean_data
from ml.feature_engineering import generate_water_quality_features, generate_lag_features
from ml.create_sequences import get_chronological_split, reshape_for_sequences


def test_missing_value_imputation():
    """Verify that missing values in water quality data are properly imputed."""
    df_raw = pd.DataFrame({
        "location": ["Station A", "Station A", "Station A", "Station B", "Station B"],
        "year": [2011, 2012, 2013, 2011, 2012],
        "do": [6.5, np.nan, 7.5, 5.0, 5.2],
        "bod": [3.0, 3.5, np.nan, 4.0, 4.2],
        "fecal_coliform": [1000.0, np.nan, 1200.0, 2000.0, 2100.0]
    })

    df_imputed = impute_missing_values(df_raw)

    assert df_imputed["do"].isna().sum() == 0
    assert df_imputed["bod"].isna().sum() == 0
    assert df_imputed["fecal_coliform"].isna().sum() == 0
    # Station A year 2012 linear interpolation for DO between 6.5 and 7.5 should be 7.0
    assert np.isclose(df_imputed.loc[1, "do"], 7.0)


def test_feature_engineering_cpcb_flags():
    """Verify CPCB standard binary flags and composite score."""
    df = pd.DataFrame({
        "location": ["Station 1", "Station 2"],
        "year": [2011, 2012],
        "do": [6.0, 4.0],              # 6.0 >= 5.0 (Good), 4.0 < 5.0 (Bad)
        "bod": [2.5, 5.0],             # 2.5 <= 3.0 (Good), 5.0 > 3.0 (Bad)
        "fecal_coliform": [1500.0, 5000.0] # 1500 <= 2500 (Good), 5000 > 2500 (Bad)
    })

    df_feat = generate_water_quality_features(df)

    assert df_feat.loc[0, "do_good"] == 1
    assert df_feat.loc[0, "bod_good"] == 1
    assert df_feat.loc[0, "fecal_good"] == 1
    assert df_feat.loc[0, "water_quality_score"] == 3

    assert df_feat.loc[1, "do_good"] == 0
    assert df_feat.loc[1, "bod_good"] == 0
    assert df_feat.loc[1, "fecal_good"] == 0
    assert df_feat.loc[1, "water_quality_score"] == 0


def test_chronological_split_no_leakage():
    """Verify chronological splitting holds out latest year without mixing."""
    df = pd.DataFrame({
        "location": ["St1", "St1", "St1", "St2", "St2", "St2"],
        "year": [2013, 2014, 2015, 2013, 2014, 2015],
        "do": [7.0, 7.2, 7.5, 6.0, 6.2, 6.4],
        "bod": [3.0, 3.1, 3.2, 4.0, 4.1, 4.2],
        "fecal_coliform": [1000, 1100, 1200, 2000, 2100, 2200],
        "do_lag1": [6.8, 7.0, 7.2, 5.8, 6.0, 6.2],
        "bod_lag1": [2.9, 3.0, 3.1, 3.9, 4.0, 4.1],
        "fecal_coliform_lag1": [900, 1000, 1100, 1900, 2000, 2100],
        "do_lag2": [6.5, 6.8, 7.0, 5.5, 5.8, 6.0],
        "bod_lag2": [2.8, 2.9, 3.0, 3.8, 3.9, 4.0],
        "fecal_coliform_lag2": [800, 900, 1000, 1800, 1900, 2000],
        "do_change": [0.3, 0.2, 0.2, 0.3, 0.2, 0.2],
        "bod_change": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "fecal_coliform_change": [100, 100, 100, 100, 100, 100],
        "do_good": [1, 1, 1, 1, 1, 1],
        "bod_good": [1, 0, 0, 0, 0, 0],
        "fecal_good": [1, 1, 1, 1, 1, 1],
        "water_quality_score": [3, 2, 2, 2, 2, 2],
        "year_index": [2, 3, 4, 2, 3, 4],
        "station_id": [0, 0, 0, 1, 1, 1]
    })

    X_train, X_test, y_train, y_test = get_chronological_split(df, test_year=2015)

    assert len(X_train) == 4
    assert len(X_test) == 2
    assert y_test.tolist() == [7.5, 6.4]


def test_sequence_reshaping():
    """Verify 3D sequence array creation for DL architectures."""
    X_sample = np.random.randn(10, 12)
    seq = reshape_for_sequences(X_sample)
    assert seq.shape == (10, 1, 12)
