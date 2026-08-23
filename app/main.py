"""
FastAPI Application Entry Point for Aquanga
Predictive Water Monitoring & Early Warning System for Ganga River
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import settings
from app.database.database import init_db, get_db
from app.routers import stations, prediction, alerts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aquanga")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events: initialize database and pre-warm models."""
    logger.info("Initializing Aquanga system on startup...")
    try:
        init_db()
        logger.info("Database schemas verified.")
    except Exception as e:
        logger.warning(f"Database auto-init notice: {e}")
    yield
    logger.info("Aquanga system shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="High-precision water quality forecasting and environmental early warning API for Ganga River monitoring stations.",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(stations.router)
app.include_router(prediction.router)
app.include_router(alerts.router)


@app.get("/", tags=["Root"], summary="System Status & Root Welcome")
def root():
    """Returns application name, status, version, and main endpoints."""
    return {
        "project": settings.PROJECT_NAME,
        "status": "online",
        "version": settings.VERSION,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "stations": "/stations",
            "predict": "/predict",
            "alerts": "/alerts"
        }
    }


@app.get("/health", tags=["Root"], summary="Health check endpoint")
def health_check(db: Session = Depends(get_db)):
    """System health check verifying database and service availability."""
    db_status = "healthy"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "version": settings.VERSION
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
