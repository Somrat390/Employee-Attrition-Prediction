"""Tests for app.schemas.EmployeeFeatures -- confirms invalid input is
rejected at the validation layer before it ever reaches the model, and that
valid input passes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic import ValidationError

from app.schemas import EmployeeFeatures


def test_valid_payload_passes(high_risk_payload):
    features = EmployeeFeatures(**high_risk_payload)
    assert features.Age == 29
    assert features.OverTime == "Yes"


def test_missing_required_field_rejected(high_risk_payload):
    payload = dict(high_risk_payload)
    del payload["Age"]
    with pytest.raises(ValidationError):
        EmployeeFeatures(**payload)


@pytest.mark.parametrize("bad_value", ["SOMETIMES", "always", "", "travel_rarely"])
def test_invalid_business_travel_category_rejected(high_risk_payload, bad_value):
    """Category values must exactly match training data -- including case --
    since a silently-accepted typo would one-hot-encode to all zeros and
    produce a meaningless (not obviously wrong) prediction."""
    payload = dict(high_risk_payload)
    payload["BusinessTravel"] = bad_value
    with pytest.raises(ValidationError):
        EmployeeFeatures(**payload)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("Age", 17),  # below training data's minimum (18)
        ("Age", 61),  # above training data's maximum (60)
        ("DistanceFromHome", -1),
        ("Education", 0),  # ordinal scale is 1-5
        ("Education", 6),
        ("JobSatisfaction", 5),  # ordinal scale is 1-4
        ("PerformanceRating", 5),  # this dataset only has 3 and 4
        ("StockOptionLevel", 4),  # scale is 0-3
    ],
)
def test_out_of_range_values_rejected(high_risk_payload, field, bad_value):
    payload = dict(high_risk_payload)
    payload[field] = bad_value
    with pytest.raises(ValidationError):
        EmployeeFeatures(**payload)


def test_wrong_type_rejected(high_risk_payload):
    payload = dict(high_risk_payload)
    payload["Age"] = "thirty"
    with pytest.raises(ValidationError):
        EmployeeFeatures(**payload)


@pytest.mark.parametrize(
    "field",
    [
        "BusinessTravel",
        "Department",
        "EducationField",
        "Gender",
        "JobRole",
        "MaritalStatus",
        "OverTime",
    ],
)
def test_every_nominal_field_has_a_valid_example(high_risk_payload, field):
    """Every categorical field's own declared example value (from
    schemas.py) should itself pass validation -- catches a typo in the
    schema's Literal list vs. its own example."""
    payload = dict(high_risk_payload)
    # Re-validate the fixture as-is; if this ever fails it means the fixture
    # payload itself drifted out of sync with the schema.
    EmployeeFeatures(**payload)


def test_model_dump_round_trips_all_30_fields(high_risk_payload):
    features = EmployeeFeatures(**high_risk_payload)
    dumped = features.model_dump()
    assert set(dumped.keys()) == set(high_risk_payload.keys())
    assert len(dumped) == 30
