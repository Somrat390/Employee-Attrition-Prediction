"""Tests against the actual FastAPI app via TestClient -- exercises routing,
request/response validation, and the database logging path together, using
the isolated in-memory DB from conftest.py's `client` fixture.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_high_risk(client, high_risk_payload):
    response = client.post("/predict", json=high_risk_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "Leave"
    assert body["probability_leave"] > 0.9
    assert 0.0 < body["threshold_used"] < 1.0


def test_predict_low_risk(client, low_risk_payload):
    response = client.post("/predict", json=low_risk_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "Stay"
    assert body["probability_leave"] < 0.1


def test_predict_missing_field_returns_422(client, high_risk_payload):
    payload = dict(high_risk_payload)
    del payload["MonthlyIncome"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_category_returns_422(client, high_risk_payload):
    payload = dict(high_risk_payload)
    payload["OverTime"] = "Maybe"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_response_has_request_id_header(client, high_risk_payload):
    response = client.post("/predict", json=high_risk_payload)
    assert "x-request-id" in response.headers


def test_predict_is_logged_and_retrievable(client, high_risk_payload):
    """Exercises the full /predict -> database write -> /predictions read
    path together, against the isolated in-memory DB."""
    predict_response = client.post("/predict", json=high_risk_payload)
    assert predict_response.status_code == 200

    history_response = client.get("/predictions", params={"limit": 10})
    assert history_response.status_code == 200
    rows = history_response.json()
    assert len(rows) == 1
    assert rows[0]["prediction"] == predict_response.json()["prediction"]
    assert rows[0]["input_features"]["Age"] == high_risk_payload["Age"]


def test_predictions_empty_when_none_logged_yet(client):
    response = client.get("/predictions")
    assert response.status_code == 200
    assert response.json() == []


def test_predictions_respects_limit(client, high_risk_payload, low_risk_payload):
    client.post("/predict", json=high_risk_payload)
    client.post("/predict", json=low_risk_payload)
    response = client.get("/predictions", params={"limit": 1})
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_explain_returns_contributions(client, high_risk_payload):
    response = client.post("/explain", json=high_risk_payload, params={"top_n": 5})
    assert response.status_code == 200
    body = response.json()
    assert "base_value" in body
    assert len(body["contributions"]) == 5
    for c in body["contributions"]:
        assert "feature" in c and "shap_value" in c and "feature_value" in c


def test_explain_summary_returns_png(client):
    response = client.get("/explain/summary")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_explain_waterfall_returns_png(client, high_risk_payload):
    response = client.post("/explain/waterfall", json=high_risk_payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
