"""
Pydantic Schemas for Forecasting, Risk Assessment, and Alerts
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    station_name: str = Field(..., description="Target station name", json_schema_extra={"example": "GANGA AT VARANASI D/S (MALVIYA BRIDGE), U.P"})
    station_id: Optional[int] = Field(0, description="Station ID index")
    do_lag1: float = Field(..., description="Previous year DO value (mg/L)", json_schema_extra={"example": 7.7})
    bod_lag1: Optional[float] = Field(4.4, description="Previous year BOD value (mg/L)", json_schema_extra={"example": 4.4})
    fecal_coliform_lag1: Optional[float] = Field(34000.0, description="Previous year Fecal Coliform (MPN/100ml)", json_schema_extra={"example": 34000.0})
    do_change: Optional[float] = Field(0.0, description="DO annual change delta")
    bod_change: Optional[float] = Field(0.0, description="BOD annual change delta")
    fecal_coliform_change: Optional[float] = Field(0.0, description="Fecal Coliform change delta")
    forecast_year: Optional[int] = Field(2016, description="Target forecast year", json_schema_extra={"example": 2016})
    model_name: Optional[str] = Field("best", description="Model architecture to use (best, linear_regression, random_forest, xgboost, 1d_cnn, lstm, cnn_lstm)")


class PredictionResponse(BaseModel):
    station: str
    station_id: int
    forecast_year: int
    model_used: str
    predicted_do: float
    risk_score: int
    risk_level: str
    warning: str
    parameters: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class AlertResponse(BaseModel):
    id: int
    station_id: int
    alert_type: str
    severity: str
    message: str
    is_active: bool
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
