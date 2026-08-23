"""
SQLAlchemy Relational Models for Aquanga
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class Station(Base):
    """Monitoring Station entity."""
    __tablename__ = "stations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    river = Column(String(100), default="Ganga", nullable=False)
    state = Column(String(100), nullable=True)
    district = Column(String(100), nullable=True)
    location_type = Column(String(100), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    water_records = relationship("WaterQualityRecord", back_populates="station", cascade="all, delete-orphan")
    predictions = relationship("PredictionRecord", back_populates="station", cascade="all, delete-orphan")
    alerts = relationship("AlertRecord", back_populates="station", cascade="all, delete-orphan")


class WaterQualityRecord(Base):
    """Historical water quality observation."""
    __tablename__ = "water_quality"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(Integer, ForeignKey("stations.id"), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    do = Column(Float, nullable=False)
    bod = Column(Float, nullable=True)
    fecal_coliform = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    station = relationship("Station", back_populates="water_records")


class PredictionRecord(Base):
    """Model forecast record."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(Integer, ForeignKey("stations.id"), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    forecast_year = Column(Integer, nullable=False)
    predicted_do = Column(Float, nullable=False)
    risk_score = Column(Integer, nullable=False)
    risk_level = Column(String(50), nullable=False)
    warning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    station = relationship("Station", back_populates="predictions")


class AlertRecord(Base):
    """Active environmental alert entity."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(Integer, ForeignKey("stations.id"), nullable=False, index=True)
    alert_type = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)  # Low, Medium, High
    message = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=utc_now)

    station = relationship("Station", back_populates="alerts")
