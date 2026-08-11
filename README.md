# Employee Attrition Prediction

An end-to-end machine learning system that predicts whether an employee is
likely to leave a company, and explains *why*. Starts from feature
engineering in a notebook and ends as a containerized, tested, CI-integrated
application: a FastAPI backend, a Streamlit frontend, SHAP explainability,
MLflow experiment tracking, PostgreSQL prediction logging, structured
logging, and an automated test suite.

**Model performance:** Logistic Regression + SMOTE, F1 improved from 0.438
(untreated class imbalance) to **0.542** (recall 0.553, precision 0.531,
PR-AUC 0.567) after properly applying SMOTE and tuning the decision
threshold to 0.66.

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Option A: Docker (recommended)](#option-a-docker-recommended)
  - [Option B: Manual setup](#option-b-manual-setup)
- [Using the App](#using-the-app)
- [API Reference](#api-reference)
- [Model Training & MLflow](#model-training--mlflow)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Logging](#logging)
- [Key Design Decisions](#key-design-decisions)
- [Roadmap](#roadmap)

---

## Architecture

```
┌─────────────────┐      HTTP/JSON       ┌──────────────────────┐
│  Streamlit       │ ───────────────────▶ │  FastAPI backend      │
│  frontend         │ ◀─────────────────── │  (predict, explain)   │
│  (port 8501)      │    predictions,      │  (port 8000)           │
└─────────────────┘    SHAP charts        └───────────┬──────────┘
                                                          │
                                    ┌─────────────────────┼─────────────────────┐
                                    ▼                      ▼                     ▼
                          ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
                          │  PostgreSQL        │  │  Trained pipeline  │  │  Structured logs   │
                          │  (prediction log)   │  │  (.pkl, joblib)     │  │  (stdout, JSON)     │
                          └──────────────────┘  └──────────────────┘  └──────────────────┘
```

The frontend has **zero ML dependencies** — it only calls the backend over
HTTP and renders what comes back. All model logic, preprocessing, and SHAP
computation live server-side in `app/`.

## Features

- **Prediction API** — FastAPI serving a scikit-learn pipeline (preprocessing
  → SMOTE → Logistic Regression), with full input validation against the
  training data's actual ranges and categories.
- **Interactive frontend** — a Streamlit form for all 30 employee attributes,
  showing the prediction, probability, and a visual explanation.
- **Explainability** — SHAP-based global feature importance, per-employee
  waterfall and force plots, and raw contribution values as JSON.
- **Experiment tracking** — every candidate model (Logistic Regression,
  Random Forest, HistGradientBoosting, XGBoost) is trained and logged to
  MLflow, with the best one registered under a `champion` alias.
- **Prediction logging** — every API call is persisted to PostgreSQL
  (timestamp, input, prediction, probability), queryable via `GET /predictions`.
- **Structured logging** — JSON logs for every request, prediction, and
  error, with a `request_id` for tracing.
- **Test suite** — 45 tests covering model logic, input validation, and the
  full API, with 86% coverage on `app/`.
- **CI/CD** — GitHub Actions runs tests, formatting/lint checks, and Docker
  builds on every push.
- **Containerized** — `docker compose up --build` starts Postgres, backend,
  and frontend together.

## Project Structure

```
employee_attrition/
├── .github/workflows/
│   └── ci.yml                  # test, lint, and build jobs
├── app/
│   ├── main.py                 # FastAPI app and all routes
│   ├── model_loader.py          # loads the trained pipeline once at startup
│   ├── schemas.py                # Pydantic request/response models
│   ├── predict.py                 # prediction logic
│   ├── explain.py                 # SHAP explainability
│   ├── database.py                # SQLAlchemy models, prediction logging
│   └── logging_config.py           # structured JSON logging
├── frontend/
│   ├── app.py                   # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── models/
│   ├── employee_attrition_pipeline.pkl   # {pipeline, threshold, label_encoder}
│   └── shap_background_sample.csv          # background data for SHAP
├── training/
│   ├── train.py                  # trains all candidates, logs to MLflow
│   └── data/
│       └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── tests/
│   ├── conftest.py                # shared fixtures
│   ├── test_predict.py             # model logic tests
│   ├── test_schemas.py             # input validation tests
│   └── test_api.py                  # full API tests
├── Dockerfile                     # backend image
├── docker-compose.yml               # backend + frontend + postgres
├── pyproject.toml                    # black + ruff config
├── pytest.ini
├── requirements.txt                   # backend dependencies
└── requirements-dev.txt                # + pytest, httpx, pytest-cov
```

## Getting Started

### Option A: Docker (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
docker compose up --build
```

This starts three services together:

| Service  | URL                         |
|----------|------------------------------|
| Backend  | http://localhost:8000/docs    |
| Frontend | http://localhost:8501           |
| Postgres | localhost:5432 (internal)         |

Stop with `docker compose down` (add `-v` to also wipe logged prediction data).

### Option B: Manual setup

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt
```

Start the backend:
```bash
uvicorn app.main:app --reload
```

In a second terminal, start the frontend:
```bash
streamlit run frontend/app.py
```

Without a `DATABASE_URL` environment variable set, the backend falls back to
a local SQLite file (`predictions.db`) instead of PostgreSQL — fine for
trying things out, but Docker Compose is what matches the full architecture.

To point the backend at a real Postgres instance manually:
```bash
export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/attrition_db"
```

## Using the App

1. Open the Streamlit frontend and fill in the employee form (or leave the
   defaults).
2. Click **Predict** — you'll see a Stay/Leave result with a probability.
3. A waterfall chart explains which features pushed the prediction in each
   direction, with exact SHAP values available in an expander below it.
4. A "Recent predictions" panel shows the logged history from the database.

## API Reference

| Method | Endpoint             | Description                                      |
|--------|-----------------------|---------------------------------------------------|
| GET    | `/`                    | API info                                            |
| GET    | `/health`              | Model load status                                    |
| POST   | `/predict`             | Predict attrition for one employee                    |
| GET    | `/predictions`         | Recent logged predictions (`?limit=50`)                |
| POST   | `/explain`             | SHAP contribution values (JSON) for one employee         |
| GET    | `/explain/summary`     | Global feature importance chart (PNG)                      |
| POST   | `/explain/waterfall`   | Per-employee SHAP waterfall chart (PNG)                       |
| POST   | `/explain/force`       | Per-employee SHAP force plot (PNG)                              |

Full interactive documentation (Swagger) is available at `/docs` once the
backend is running.

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 29, "DailyRate": 500, "DistanceFromHome": 20, "HourlyRate": 55,
    "MonthlyIncome": 2500, "MonthlyRate": 15000, "NumCompaniesWorked": 4,
    "PercentSalaryHike": 12, "TotalWorkingYears": 3, "TrainingTimesLastYear": 1,
    "YearsAtCompany": 1, "YearsInCurrentRole": 0, "YearsSinceLastPromotion": 0,
    "YearsWithCurrManager": 0,
    "BusinessTravel": "Travel_Frequently", "Department": "Sales",
    "EducationField": "Marketing", "Gender": "Male", "JobRole": "Sales Representative",
    "MaritalStatus": "Single", "OverTime": "Yes",
    "Education": 2, "EnvironmentSatisfaction": 1, "JobInvolvement": 2, "JobLevel": 1,
    "JobSatisfaction": 1, "PerformanceRating": 3, "RelationshipSatisfaction": 2,
    "StockOptionLevel": 0, "WorkLifeBalance": 1
  }'
```
```json
{"prediction": "Leave", "probability_leave": 0.9992, "threshold_used": 0.66}
```

## Model Training & MLflow

```bash
cd training
python train.py
```

Trains Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost
(all with SMOTE), logs each as an MLflow run, and registers the best one as
`employee-attrition-classifier` with a `champion` alias.

View results:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open `http://127.0.0.1:5000`.

| Model                  | F1     | PR-AUC  |
|--------------------------|--------|---------|
| **Logistic Regression**  | **0.480*** | **0.557** |
| HistGradientBoosting     | 0.448  | 0.520   |
| Random Forest            | 0.416  | 0.500   |
| XGBoost                  | 0.406  | 0.511   |

*at the default 0.5 threshold, as logged by `training/train.py`. The main
notebook's separately-tuned decision threshold (0.66, used by the deployed
API) pushes Logistic Regression's F1 to 0.542 — see `training/train.py`'s
`*_at_tuned_threshold` metrics for each model's number at that same cutoff.

## Testing

```bash
pip install -r requirements-dev.txt
pytest                                      # 45 tests
pytest --cov=app --cov-report=term-missing   # with coverage (86%)
```

Tests cover: the model pipeline's structure and output shape, prediction
correctness on known high/low-risk profiles (catches a bad retrain, not just
a crash), every input validation rule, and the full API including the
`/predict` → `/predictions` logging round-trip against an isolated in-memory
database.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:

- **test** — `pytest --cov=app`
- **lint** — `black --check` + `ruff check`
- **build** — Docker build of both images (only runs if test + lint pass)

## Logging

Every request, prediction, and error produces one structured JSON line on
stdout:

```json
{"timestamp": "...", "level": "INFO", "logger": "app.predictions", "message": "prediction made", "prediction": "Leave", "probability_leave": 0.9992, "inference_ms": 22.65}
```

Every response includes an `X-Request-ID` header matching its log line, for
tracing a specific request through the logs.

## Key Design Decisions

- **SMOTE inside an `imblearn.Pipeline`, not applied manually** — guarantees
  resampling only ever touches training folds, never the test set, and
  survives being reused inside `GridSearchCV`.
- **Threshold tuned to 0.66, saved alongside the model** — recall matters
  more than precision here (a missed leaver costs more than one unnecessary
  retention conversation), and saving the threshold with the pipeline means
  the API can't silently drift back to sklearn's 0.5 default.
- **SHAP via `LinearExplainer`** — exact, closed-form values for the linear
  final model, rather than a slower sampling-based approximation.
- **Prediction logging never breaks a prediction** — a `try/except/rollback`
  around the database write means a down database can't take the `/predict`
  endpoint down with it; verified by dropping the table under a live server.
- **JSON logs on stdout, not files** — the standard approach for anything
  running in Docker/Render/Railway, which capture stdout automatically.
- **Frontend has zero ML dependencies** — it only calls the backend over
  HTTP, so the two can be deployed independently (e.g. Streamlit Community
  Cloud + Render) without shared dependencies.

## Roadmap

- [x] Feature engineering & preprocessing
- [x] Model comparison, hyperparameter tuning, threshold optimization
- [x] FastAPI backend
- [x] Streamlit frontend
- [x] SHAP explainability
- [x] Docker & Docker Compose
- [x] MLflow experiment tracking
- [x] PostgreSQL prediction logging
- [x] Structured logging
- [x] Automated testing (pytest)
- [x] CI/CD (GitHub Actions)
- [ ] Deployment (FastAPI → Render/Railway, Streamlit → Streamlit Community Cloud)
