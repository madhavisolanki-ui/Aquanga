"""
Alert Management Service for Aquanga
Generates and queries system alerts for stations with elevated environmental risk.
"""

from typing import List, Dict, Any, Optional
from app.services.risk_service import calculate_risk


def generate_station_alert(
    station_name: str,
    station_id: int,
    predicted_do: float,
    bod: Optional[float] = None,
    fecal_coliform: Optional[float] = None,
    year: int = 2016
) -> Dict[str, Any]:
    """Generates an alert object for a station evaluation."""
    risk_info = calculate_risk(predicted_do, bod, fecal_coliform)
    
    is_active = risk_info["risk_score"] >= 2  # Medium or High risk triggers active alert
    alert_type = "CRITICAL_POLLUTION" if risk_info["risk_level"] == "High" else "MONITORING_NOTICE"

    return {
        "station_id": station_id,
        "station_name": station_name,
        "year": year,
        "predicted_do": round(predicted_do, 3),
        "bod": round(bod, 2) if bod is not None else None,
        "fecal_coliform": fecal_coliform,
        "risk_score": risk_info["risk_score"],
        "risk_level": risk_info["risk_level"],
        "warning": risk_info["warning"],
        "alert_type": alert_type,
        "is_active": is_active
    }
