"""
CRUD Operations for Aquanga Database
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from app.database import models


# --- Station Operations ---
def get_stations(db: Session, skip: int = 0, limit: int = 100) -> List[models.Station]:
    return db.query(models.Station).offset(skip).limit(limit).all()


def get_station_by_id(db: Session, station_id: int) -> Optional[models.Station]:
    return db.query(models.Station).filter(models.Station.id == station_id).first()


def get_station_by_name(db: Session, name: str) -> Optional[models.Station]:
    return db.query(models.Station).filter(models.Station.name == name).first()


def create_station(
    db: Session,
    name: str,
    river: str = "Ganga",
    state: Optional[str] = None,
    district: Optional[str] = None,
    location_type: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
) -> models.Station:
    station = models.Station(
        name=name,
        river=river,
        state=state,
        district=district,
        location_type=location_type,
        latitude=latitude,
        longitude=longitude
    )
    db.add(station)
    db.commit()
    db.refresh(station)
    return station


# --- Water Quality Operations ---
def get_water_records_by_station(db: Session, station_id: int) -> List[models.WaterQualityRecord]:
    return (
        db.query(models.WaterQualityRecord)
        .filter(models.WaterQualityRecord.station_id == station_id)
        .order_by(models.WaterQualityRecord.year.asc())
        .all()
    )


def create_water_record(
    db: Session,
    station_id: int,
    year: int,
    do: float,
    bod: Optional[float] = None,
    fecal_coliform: Optional[float] = None
) -> models.WaterQualityRecord:
    record = models.WaterQualityRecord(
        station_id=station_id,
        year=year,
        do=do,
        bod=bod,
        fecal_coliform=fecal_coliform
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# --- Prediction Operations ---
def get_predictions_by_station(db: Session, station_id: int) -> List[models.PredictionRecord]:
    return (
        db.query(models.PredictionRecord)
        .filter(models.PredictionRecord.station_id == station_id)
        .order_by(models.PredictionRecord.created_at.desc())
        .all()
    )


def create_prediction(
    db: Session,
    station_id: int,
    model_name: str,
    forecast_year: int,
    predicted_do: float,
    risk_score: int,
    risk_level: str,
    warning: Optional[str] = None
) -> models.PredictionRecord:
    pred = models.PredictionRecord(
        station_id=station_id,
        model_name=model_name,
        forecast_year=forecast_year,
        predicted_do=predicted_do,
        risk_score=risk_score,
        risk_level=risk_level,
        warning=warning
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


# --- Alert Operations ---
def get_alerts(db: Session, active_only: bool = True, limit: int = 100) -> List[models.AlertRecord]:
    query = db.query(models.AlertRecord)
    if active_only:
        query = query.filter(models.AlertRecord.is_active == True)
    return query.order_by(models.AlertRecord.created_at.desc()).limit(limit).all()


def create_alert(
    db: Session,
    station_id: int,
    alert_type: str,
    severity: str,
    message: str,
    is_active: bool = True
) -> models.AlertRecord:
    alert = models.AlertRecord(
        station_id=station_id,
        alert_type=alert_type,
        severity=severity,
        message=message,
        is_active=is_active
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert
