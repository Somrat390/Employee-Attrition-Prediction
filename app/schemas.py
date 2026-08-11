"""Request/response schemas for the Employee Attrition Prediction API."""

from typing import Literal

from pydantic import BaseModel, Field


class EmployeeFeatures(BaseModel):
    """
    Raw employee attributes the model needs. These are exactly the 30 columns
    the trained pipeline's ColumnTransformer consumes (14 numerical, 7 nominal,
    9 ordinal) -- nothing else is required, since the pipeline itself handles
    scaling and one-hot encoding.
    """

    # --- Numerical features ---
    Age: int = Field(..., ge=18, le=60, examples=[34])
    DailyRate: int = Field(..., ge=100, le=1500, examples=[800])
    DistanceFromHome: int = Field(..., ge=0, le=30, examples=[5])
    HourlyRate: int = Field(..., ge=30, le=100, examples=[65])
    MonthlyIncome: int = Field(..., ge=1000, le=20000, examples=[5500])
    MonthlyRate: int = Field(..., ge=2000, le=27000, examples=[14000])
    NumCompaniesWorked: int = Field(..., ge=0, le=10, examples=[2])
    PercentSalaryHike: int = Field(..., ge=10, le=25, examples=[15])
    TotalWorkingYears: int = Field(..., ge=0, le=40, examples=[8])
    TrainingTimesLastYear: int = Field(..., ge=0, le=6, examples=[3])
    YearsAtCompany: int = Field(..., ge=0, le=40, examples=[5])
    YearsInCurrentRole: int = Field(..., ge=0, le=20, examples=[3])
    YearsSinceLastPromotion: int = Field(..., ge=0, le=15, examples=[1])
    YearsWithCurrManager: int = Field(..., ge=0, le=20, examples=[3])

    # --- Nominal (categorical) features ---
    BusinessTravel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"] = Field(
        ..., examples=["Travel_Rarely"]
    )
    Department: Literal["Human Resources", "Research & Development", "Sales"] = Field(
        ..., examples=["Research & Development"]
    )
    EducationField: Literal[
        "Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"
    ] = Field(..., examples=["Life Sciences"])
    Gender: Literal["Female", "Male"] = Field(..., examples=["Female"])
    JobRole: Literal[
        "Healthcare Representative",
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative",
    ] = Field(..., examples=["Laboratory Technician"])
    MaritalStatus: Literal["Divorced", "Married", "Single"] = Field(..., examples=["Single"])
    OverTime: Literal["Yes", "No"] = Field(..., examples=["Yes"])

    # --- Ordinal features (kept as ints, matching training data encoding) ---
    Education: int = Field(..., ge=1, le=5, examples=[3])
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4, examples=[2])
    JobInvolvement: int = Field(..., ge=1, le=4, examples=[3])
    JobLevel: int = Field(..., ge=1, le=5, examples=[1])
    JobSatisfaction: int = Field(..., ge=1, le=4, examples=[2])
    PerformanceRating: int = Field(..., ge=3, le=4, examples=[3])
    RelationshipSatisfaction: int = Field(..., ge=1, le=4, examples=[3])
    StockOptionLevel: int = Field(..., ge=0, le=3, examples=[0])
    WorkLifeBalance: int = Field(..., ge=1, le=4, examples=[2])


class PredictionResponse(BaseModel):
    prediction: Literal["Stay", "Leave"]
    probability_leave: float = Field(..., ge=0.0, le=1.0)
    threshold_used: float


class PredictionLogEntry(BaseModel):
    id: int
    timestamp: str
    input_features: dict
    prediction: Literal["Stay", "Leave"]
    probability_leave: float
    threshold_used: float


class Contribution(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class ExplanationResponse(BaseModel):
    base_value: float
    contributions: list[Contribution]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
