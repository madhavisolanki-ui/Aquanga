"""
Unit tests for Model Prediction Service.
"""

import pytest
from app.services.prediction_service import predictor


def test_predictor_feature_vector_builder():
    """Verify feature vector assembly shape and values."""
    feat = predictor.build_feature_vector(
        do_lag1=7.0,
        bod_lag1=4.0,
        fecal_coliform_lag1=3000.0,
        do_change=0.2,
        bod_change=-0.1,
        fecal_coliform_change=50.0,
        year_index=5,
        station_id=1
    )
    assert feat.shape == (1, 12)
    # Check that water quality score is computed (DO >= 5: 1, BOD <= 3: 0, FC <= 2500: 0 -> score: 1)
    assert feat[0, 9] == 1.0


def test_predictor_predict_output():
    """Verify prediction pipeline returns full schema fields."""
    res = predictor.predict(
        station_name="GANGA AT VARANASI D/S (MALVIYA BRIDGE), U.P",
        do_lag1=7.7,
        bod_lag1=4.4,
        fecal_coliform_lag1=34000.0,
        do_change=0.0,
        bod_change=0.0,
        fecal_coliform_change=0.0,
        year=2016,
        station_id=0,
        model_name="best"
    )

    assert "predicted_do" in res
    assert isinstance(res["predicted_do"], float)
    assert res["predicted_do"] >= 0.0
    assert res["risk_level"] in ["Low", "Medium", "High"]
    assert res["risk_score"] in [0, 1, 2, 3]
    assert len(res["warning"]) > 0
