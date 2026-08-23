"""
Unit tests for CPCB-aligned risk calculation service and alert logic.
"""

import pytest
from app.services.risk_service import calculate_risk
from app.services.alert_service import generate_station_alert


def test_calculate_risk_pristine_water():
    """Verify pristine water generates Low Risk and score 0 or 1."""
    res = calculate_risk(do=7.5, bod=2.0, fecal_coliform=1000.0)
    assert res["risk_level"] == "Low"
    assert res["risk_score"] in [0, 1]
    assert "permissible" in res["warning"].lower()


def test_calculate_risk_moderate_pollution():
    """Verify moderate pollution triggers Medium Risk."""
    # BOD is 4.5 (> 3.0), DO is 4.8 (< 5.0) -> Points = 2
    res = calculate_risk(do=4.8, bod=4.5, fecal_coliform=2000.0)
    assert res["risk_level"] == "Medium"
    assert res["risk_score"] == 2
    assert "requires monitoring" in res["warning"].lower()


def test_calculate_risk_critical_hypoxia_and_fecal():
    """Verify critical hypoxia (DO < 4.0) and severe coliform (>10,000) trigger High Risk."""
    res = calculate_risk(do=3.2, bod=7.0, fecal_coliform=45000.0)
    assert res["risk_level"] == "High"
    assert res["risk_score"] == 3
    assert "critical" in res["warning"].lower()


def test_generate_station_alert_active():
    """Verify high risk station triggers active alert."""
    alert = generate_station_alert(
        station_name="Test Station Jajmau",
        station_id=1,
        predicted_do=3.5,
        bod=7.5,
        fecal_coliform=30000.0
    )
    assert alert["is_active"] is True
    assert alert["risk_level"] == "High"
    assert alert["alert_type"] == "CRITICAL_POLLUTION"
