"""
Database Seeding Script for Aquanga
Populates stations, historical water quality measurements, forecasts, and alerts.
"""

import os
import sys
import json
import logging
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database.database import init_db, SessionLocal
from app.database import crud, models
from app.services.risk_service import calculate_risk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def seed_database(
    clean_csv_path: str = "data/processed/ganga_water_quality_clean.csv",
    coords_json_path: str = "data/external/station_coordinates.json"
):
    """Seeds the database tables."""
    logger.info("Initializing database tables...")
    init_db()

    db = SessionLocal()
    try:
        # Load station coordinates
        coords_data = {}
        if os.path.exists(coords_json_path):
            with open(coords_json_path, "r") as f:
                coords_data = json.load(f)

        # Load clean dataset
        if not os.path.exists(clean_csv_path):
            from ml.preprocessing import run_preprocessing
            df = run_preprocessing(output_path=clean_csv_path)
        else:
            df = pd.read_csv(clean_csv_path)

        logger.info(f"Loaded dataset with {len(df)} records across {df['location'].nunique()} stations.")

        # Seed Stations
        station_map = {}
        unique_locations = df["location"].unique()

        for loc in unique_locations:
            loc_str = str(loc).strip()
            existing_station = crud.get_station_by_name(db, loc_str)
            if not existing_station:
                meta = coords_data.get(loc_str, {})
                station = crud.create_station(
                    db=db,
                    name=loc_str,
                    river="Ganga",
                    state=meta.get("state", "Uttar Pradesh"),
                    district=meta.get("district", "Unknown"),
                    location_type=meta.get("location_type", "Monitoring Station"),
                    latitude=meta.get("latitude", 25.5),
                    longitude=meta.get("longitude", 81.5)
                )
                logger.info(f"Created Station: {station.name} (ID: {station.id})")
                station_map[loc_str] = station.id
            else:
                station_map[loc_str] = existing_station.id

        # Seed Historical Water Quality Records
        record_count = 0
        for _, row in df.iterrows():
            st_id = station_map[str(row["location"]).strip()]
            yr = int(row["year"])
            
            # Check if record already exists
            existing = (
                db.query(models.WaterQualityRecord)
                .filter(models.WaterQualityRecord.station_id == st_id, models.WaterQualityRecord.year == yr)
                .first()
            )
            if not existing:
                crud.create_water_record(
                    db=db,
                    station_id=st_id,
                    year=yr,
                    do=float(row["do"]),
                    bod=float(row["bod"]) if pd.notna(row["bod"]) else None,
                    fecal_coliform=float(row["fecal_coliform"]) if pd.notna(row["fecal_coliform"]) else None
                )
                record_count += 1

        logger.info(f"Inserted {record_count} water quality records.")

        # Seed Sample Predictions & Alerts for latest year
        for loc, st_id in station_map.items():
            st_df = df[df["location"] == loc].sort_values(by="year")
            if len(st_df) > 0:
                latest_row = st_df.iloc[-1]
                latest_do = float(latest_row["do"])
                latest_bod = float(latest_row["bod"]) if pd.notna(latest_row["bod"]) else None
                latest_fc = float(latest_row["fecal_coliform"]) if pd.notna(latest_row["fecal_coliform"]) else None

                # Compute risk
                risk_info = calculate_risk(do=latest_do, bod=latest_bod, fecal_coliform=latest_fc)

                crud.create_prediction(
                    db=db,
                    station_id=st_id,
                    model_name="Ensemble / LSTM",
                    forecast_year=2016,
                    predicted_do=latest_do,
                    risk_score=risk_info["risk_score"],
                    risk_level=risk_info["risk_level"],
                    warning=risk_info["warning"]
                )

                if risk_info["risk_score"] >= 2:
                    crud.create_alert(
                        db=db,
                        station_id=st_id,
                        alert_type="CRITICAL_WATER_QUALITY" if risk_info["risk_level"] == "High" else "MONITORING_ALERT",
                        severity=risk_info["risk_level"],
                        message=risk_info["warning"],
                        is_active=True
                    )

        logger.info("Database seeding completed successfully.")

    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
