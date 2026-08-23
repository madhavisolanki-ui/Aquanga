"""
Integration tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database.database import init_db

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Ensure database schema is ready."""
    init_db()


def test_root_endpoint():
    """Verify GET / returns 200 and system information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data
    assert "endpoints" in data


def test_health_endpoint():
    """Verify GET /health returns 200 and health indicators."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "database" in data


def test_stations_list_endpoint():
    """Verify GET /stations returns list of stations."""
    response = client.get("/stations")
    assert response.status_code == 200
    stations = response.json()
    assert isinstance(stations, list)


def test_predict_endpoint_valid():
    """Verify POST /predict returns valid DO forecast and risk evaluation."""
    payload = {
        "station_name": "GANGA AT KANPUR D/S (JAJMAU PUMPING STATION)",
        "station_id": 6,
        "do_lag1": 7.3,
        "bod_lag1": 7.7,
        "fecal_coliform_lag1": 40000.0,
        "do_change": 0.6,
        "bod_change": 0.9,
        "fecal_coliform_change": 26433.0,
        "forecast_year": 2016,
        "model_name": "best"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_do" in data
    assert data["station"] == payload["station_name"]
    assert data["risk_score"] in [0, 1, 2, 3]
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert len(data["warning"]) > 0


def test_alerts_endpoint():
    """Verify GET /alerts returns active alert records."""
    response = client.get("/alerts")
    assert response.status_code == 200
    alerts = response.json()
    assert isinstance(alerts, list)
