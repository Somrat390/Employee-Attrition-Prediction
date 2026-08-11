"""Shared fixtures for all tests.

The key thing here: tests must never touch the real database (whatever
DATABASE_URL points at in dev/production) or leave rows behind. `client`
overrides FastAPI's `get_db` dependency with a fresh in-memory SQLite
session per test, so /predict's logging still runs (exercising the real
code path) but against disposable data.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# So `import app...` works when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import Base, get_db
from app.main import app


@pytest.fixture()
def client():
    # StaticPool is required here: SQLAlchemy's default pool opens a new
    # connection per checkout, and each new connection to "sqlite:///:memory:"
    # is a *separate*, empty in-memory database -- without StaticPool, the
    # table created below would exist for one request and vanish for the
    # next (this is exactly what happened on the first run of this suite).
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def high_risk_payload():
    """An employee profile the model should score as high attrition risk:
    young, single, frequent travel, low satisfaction, overtime."""
    return {
        "Age": 29,
        "DailyRate": 500,
        "DistanceFromHome": 20,
        "HourlyRate": 55,
        "MonthlyIncome": 2500,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 4,
        "PercentSalaryHike": 12,
        "TotalWorkingYears": 3,
        "TrainingTimesLastYear": 1,
        "YearsAtCompany": 1,
        "YearsInCurrentRole": 0,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 0,
        "BusinessTravel": "Travel_Frequently",
        "Department": "Sales",
        "EducationField": "Marketing",
        "Gender": "Male",
        "JobRole": "Sales Representative",
        "MaritalStatus": "Single",
        "OverTime": "Yes",
        "Education": 2,
        "EnvironmentSatisfaction": 1,
        "JobInvolvement": 2,
        "JobLevel": 1,
        "JobSatisfaction": 1,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 2,
        "StockOptionLevel": 0,
        "WorkLifeBalance": 1,
    }


@pytest.fixture()
def low_risk_payload():
    """An employee profile the model should score as low attrition risk:
    senior, married, high satisfaction, no overtime."""
    return {
        "Age": 45,
        "DailyRate": 1200,
        "DistanceFromHome": 2,
        "HourlyRate": 80,
        "MonthlyIncome": 12000,
        "MonthlyRate": 20000,
        "NumCompaniesWorked": 1,
        "PercentSalaryHike": 20,
        "TotalWorkingYears": 20,
        "TrainingTimesLastYear": 4,
        "YearsAtCompany": 15,
        "YearsInCurrentRole": 10,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 8,
        "BusinessTravel": "Non-Travel",
        "Department": "Research & Development",
        "EducationField": "Life Sciences",
        "Gender": "Female",
        "JobRole": "Manager",
        "MaritalStatus": "Married",
        "OverTime": "No",
        "Education": 4,
        "EnvironmentSatisfaction": 4,
        "JobInvolvement": 4,
        "JobLevel": 4,
        "JobSatisfaction": 4,
        "PerformanceRating": 4,
        "RelationshipSatisfaction": 4,
        "StockOptionLevel": 2,
        "WorkLifeBalance": 3,
    }
