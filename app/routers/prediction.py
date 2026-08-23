"""
Prediction Router for Aquanga
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import crud
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import predictor

router = APIRouter(tags=["Predictions"])


@router.post("/predict", response_model=PredictionResponse, summary="Predict future DO and assess water risk")
def predict_water_quality(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Predicts future Dissolved Oxygen (DO) levels for a monitoring station and calculates
    comprehensive environmental risk scores and warning messages based on CPCB standards.
    """
    try:
        result = predictor.predict(
            station_name=request.station_name,
            do_lag1=request.do_lag1,
            bod_lag1=request.bod_lag1 or 3.0,
            fecal_coliform_lag1=request.fecal_coliform_lag1 or 2500.0,
            do_change=request.do_change or 0.0,
            bod_change=request.bod_change or 0.0,
            fecal_coliform_change=request.fecal_coliform_change or 0.0,
            year=request.forecast_year or 2016,
            station_id=request.station_id or 0,
            model_name=request.model_name or "best"
        )

        # Log prediction to DB if station exists
        station = crud.get_station_by_id(db, request.station_id) if request.station_id else None
        if not station:
            station = crud.get_station_by_name(db, request.station_name)

        if station:
            crud.create_prediction(
                db=db,
                station_id=station.id,
                model_name=result["model_used"],
                forecast_year=result["forecast_year"],
                predicted_do=result["predicted_do"],
                risk_score=result["risk_score"],
                risk_level=result["risk_level"],
                warning=result["warning"]
            )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )
