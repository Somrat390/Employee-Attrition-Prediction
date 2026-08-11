"""Database layer: every prediction the API serves gets a row here.

Uses SQLAlchemy so the same code works against PostgreSQL (production, via
docker-compose) or SQLite (for running the API without a database server at
all -- see DATABASE_URL below). Connection details come from the
DATABASE_URL environment variable, matching how the frontend already reads
API_BASE_URL rather than hardcoding a host.
"""

import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./predictions.db")

# SQLite needs this connect_arg for use across threads (FastAPI can serve
# requests on different threads); PostgreSQL doesn't need or accept it.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Store the raw input as JSON text rather than one column per feature --
    # the 30 input fields are already validated and typed by EmployeeFeatures
    # before they get here, so a second rigid schema for the same 30 columns
    # would just be duplicated maintenance every time schemas.py changes.
    input_features = Column(String)  # JSON-encoded dict

    prediction = Column(String, index=True)  # "Stay" or "Leave"
    probability_leave = Column(Float)
    threshold_used = Column(Float)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
