"""
Pydantic Schemas for Monitoring Stations and Water Quality Observations
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field


class WaterQualityBase(BaseModel):
    year: int = Field(..., description="Observation year (e.g. 2015)")
    do: float = Field(..., description="Dissolved Oxygen in mg/L")
    bod: Optional[float] = Field(None, description="Biochemical Oxygen Demand in mg/L")
    fecal_coliform: Optional[float] = Field(None, description="Fecal Coliform in MPN/100ml")


class WaterQualityResponse(WaterQualityBase):
    id: int
    station_id: int
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class StationBase(BaseModel):
    name: str = Field(..., description="Unique station name / location")
    river: str = Field("Ganga", description="Monitored river system")
    state: Optional[str] = None
    district: Optional[str] = None
    location_type: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class StationCreate(StationBase):
    pass


class StationResponse(StationBase):
    id: int
    created_at: Optional[datetime] = None
    water_records: Optional[List[WaterQualityResponse]] = []

    model_config = ConfigDict(from_attributes=True)
