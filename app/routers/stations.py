"""
Stations Router for Aquanga
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import crud
from app.schemas.station import StationResponse

router = APIRouter(prefix="/stations", tags=["Stations"])


@router.get("", response_model=List[StationResponse], summary="List all water monitoring stations")
def list_stations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Retrieves all registered Ganga river water quality monitoring stations."""
    stations = crud.get_stations(db, skip=skip, limit=limit)
    return stations


@router.get("/{station_id}", response_model=StationResponse, summary="Get station details and history")
def get_station_details(station_id: int, db: Session = Depends(get_db)):
    """Retrieves details and historical observations for a specific station ID."""
    station = crud.get_station_by_id(db, station_id=station_id)
    if not station:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Station with ID {station_id} not found."
        )
    return station
