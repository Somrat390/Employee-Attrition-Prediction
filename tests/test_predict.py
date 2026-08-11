"""Tests for app.predict / app.model_loader -- the model logic itself,
called directly as Python functions rather than through HTTP. These would
catch a broken pipeline even if the API layer were completely down.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.model_loader import get_pipeline, get_threshold, load_artifact
from app.predict import predict_attrition
from app.schemas import EmployeeFeatures


def test_model_artifact_loads():
    """The saved pipeline, threshold, and label encoder all load without error."""
    artifact = load_artifact()
    assert "pipeline" in artifact
    assert "threshold" in artifact
    assert "label_encoder" in artifact


def test_threshold_is_the_tuned_value_not_the_sklearn_default():
    """Guards against silently shipping a re-trained model that forgot to
    carry over threshold tuning -- 0.5 would mean nobody set it."""
    threshold = get_threshold()
    assert threshold != 0.5
    assert 0.0 < threshold < 1.0


def test_pipeline_has_expected_steps():
    pipeline = get_pipeline()
    assert list(pipeline.named_steps.keys()) == ["preprocess", "smote", "model"]


def test_preprocessor_output_shape():
    """The ColumnTransformer should turn 30 raw columns into the expected
    number of post-one-hot columns (14 numerical + 9 ordinal + one-hot
    expansion of 7 nominal columns = 51, given the categories in the
    training data)."""
    pipeline = get_pipeline()
    preprocessor = pipeline.named_steps["preprocess"]

    row = pd.DataFrame(
        [
            {
                "Age": 30,
                "DailyRate": 800,
                "DistanceFromHome": 5,
                "HourlyRate": 65,
                "MonthlyIncome": 5000,
                "MonthlyRate": 14000,
                "NumCompaniesWorked": 1,
                "PercentSalaryHike": 15,
                "TotalWorkingYears": 8,
                "TrainingTimesLastYear": 2,
                "YearsAtCompany": 5,
                "YearsInCurrentRole": 3,
                "YearsSinceLastPromotion": 1,
                "YearsWithCurrManager": 3,
                "BusinessTravel": "Travel_Rarely",
                "Department": "Sales",
                "EducationField": "Marketing",
                "Gender": "Male",
                "JobRole": "Sales Representative",
                "MaritalStatus": "Single",
                "OverTime": "No",
                "Education": 3,
                "EnvironmentSatisfaction": 3,
                "JobInvolvement": 3,
                "JobLevel": 1,
                "JobSatisfaction": 3,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 3,
                "StockOptionLevel": 0,
                "WorkLifeBalance": 3,
            }
        ]
    )
    transformed = preprocessor.transform(row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    assert transformed.shape == (1, 51)


def test_predict_attrition_high_risk_profile(high_risk_payload):
    """A textbook high-risk profile should be predicted as Leave with high
    probability -- catches a model/threshold regression, not just a crash."""
    features = EmployeeFeatures(**high_risk_payload)
    result = predict_attrition(features)
    assert result.prediction == "Leave"
    assert result.probability_leave > 0.9


def test_predict_attrition_low_risk_profile(low_risk_payload):
    """A textbook low-risk profile should be predicted as Stay with low
    probability."""
    features = EmployeeFeatures(**low_risk_payload)
    result = predict_attrition(features)
    assert result.prediction == "Stay"
    assert result.probability_leave < 0.1


def test_predict_attrition_returns_threshold_used(high_risk_payload):
    features = EmployeeFeatures(**high_risk_payload)
    result = predict_attrition(features)
    assert result.threshold_used == get_threshold()


def test_predict_attrition_probability_is_valid_range(high_risk_payload, low_risk_payload):
    for payload in (high_risk_payload, low_risk_payload):
        result = predict_attrition(EmployeeFeatures(**payload))
        assert 0.0 <= result.probability_leave <= 1.0


def test_predict_attrition_prediction_matches_threshold_logic(high_risk_payload):
    """The prediction label should be internally consistent with the
    probability and threshold it returns -- not just plausible-looking."""
    features = EmployeeFeatures(**high_risk_payload)
    result = predict_attrition(features)
    if result.probability_leave >= result.threshold_used:
        assert result.prediction == "Leave"
    else:
        assert result.prediction == "Stay"
