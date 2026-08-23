"""
Alerts Router for Aquanga
"""

from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import crud
from app.schemas.prediction import AlertResponse

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("", response_model=List[AlertResponse], summary="Get active pollution & risk alerts")
def list_active_alerts(active_only: bool = True, db: Session = Depends(get_db)):
    """Retrieves all active water quality alerts triggered for Ganga monitoring stations."""
    alerts = crud.get_alerts(db, active_only=active_only)
    return alerts
