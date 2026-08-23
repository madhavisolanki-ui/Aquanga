"""
Database Connection & Session Management for Aquanga
Supports PostgreSQL with SQLite fallback for developer portability and tests.
"""

import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Build database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    DB_USER = os.getenv("POSTGRES_USER")
    DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    DB_HOST = os.getenv("POSTGRES_HOST")
    DB_PORT = os.getenv("POSTGRES_PORT", "5432")
    DB_NAME = os.getenv("POSTGRES_DB", "aquanga_db")

    if DB_USER and DB_PASSWORD and DB_HOST:
        DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    else:
        # Default local fallback
        DATABASE_URL = "sqlite:///./aquanga.db"

# Create engine with fallback verification
def create_app_engine(db_url: str):
    if db_url.startswith("sqlite"):
        return create_engine(db_url, connect_args={"check_same_thread": False})
    try:
        eng = create_engine(db_url, pool_pre_ping=True, pool_size=10, max_overflow=20)
        # Test connection
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Connected to database at {db_url.split('@')[-1] if '@' in db_url else db_url}")
        return eng
    except Exception as e:
        logger.warning(f"PostgreSQL connection failed ({e}). Falling back to SQLite.")
        return create_engine("sqlite:///./aquanga.db", connect_args={"check_same_thread": False})


engine = create_app_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI Dependency for database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initializes database tables."""
    from app.database import models
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")
