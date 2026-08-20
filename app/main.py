"""Employee Attrition Prediction API.

Run locally with:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger docs.
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from app import explain
from app.database import PredictionLog, get_db, init_db
from app.logging_config import configure_logging, get_logger
from app.model_loader import load_artifact
from app.predict import predict_attrition
from app.schemas import (
    EmployeeFeatures,
    ExplanationResponse,
    HealthResponse,
    PredictionLogEntry,
    PredictionResponse,
)

configure_logging()
request_logger = get_logger("app.requests")
error_logger = get_logger("app.errors")
prediction_logger = get_logger("app.predictions")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts, not on the first request --
    # this way a broken model file fails fast at boot instead of on someone's
    # first prediction.
    load_artifact()
    # Create the prediction_logs table if it doesn't exist yet. Safe to call
    # on every startup -- create_all() is a no-op for tables that already
    # match.
    init_db()
    yield


app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Serves the trained Logistic Regression + SMOTE pipeline from the notebook.",
    version="1.4.0",
    lifespan=lifespan,
)

# The frontend and backend are deployed as separate services on separate
# domains (Streamlit Community Cloud + Render/Railway) once Week 10's
# deployment is live, so the browser treats every frontend->backend call as
# cross-origin -- without CORS headers, browsers block the response before
# Streamlit's Python code ever sees it (this doesn't affect server-to-server
# calls, only requests a browser JS runtime makes, so it wasn't needed while
# everything ran on localhost or inside one Docker network).
# ALLOWED_ORIGINS defaults to "*" since this is a portfolio project with a
# single, non-sensitive read-mostly endpoint set -- for anything handling
# real user data, set ALLOWED_ORIGINS to the exact frontend URL instead.
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One structured log line per HTTP request: method, path, status,
    duration, and a request_id that also gets echoed back as a response
    header -- so a person reporting 'the API was slow/broke at 3:14pm' can
    be matched to one exact log line instead of a timestamp guess."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    request_logger.info(
        "request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": request.client.host if request.client else None,
        },
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def log_unhandled_exceptions(request: Request, exc: Exception):
    """Catches anything a route didn't already turn into an HTTPException --
    logs the full traceback (exc_info=True) so an unexpected crash is
    diagnosable from logs alone, then returns a generic 500 rather than
    leaking an internal stack trace to the caller."""
    error_logger.error(
        "unhandled exception",
        extra={"method": request.method, "path": request.url.path},
        exc_info=exc,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/", tags=["Meta"])
def root():
    return {"message": "Employee Attrition Prediction API. See /docs for usage."}


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    try:
        load_artifact()
        return HealthResponse(status="ok", model_loaded=True)
    except FileNotFoundError:
        return HealthResponse(status="model missing", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: EmployeeFeatures, db: Session = Depends(get_db)):
    inference_start = time.perf_counter()
    try:
        result = predict_attrition(features)
    except FileNotFoundError as e:
        error_logger.error("model file missing", extra={"error": str(e)})
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        error_logger.error("prediction failed", extra={"error": str(e)}, exc_info=e)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    inference_ms = round((time.perf_counter() - inference_start) * 1000, 2)

    prediction_logger.info(
        "prediction made",
        extra={
            "prediction": result.prediction,
            "probability_leave": result.probability_leave,
            "threshold_used": result.threshold_used,
            "inference_ms": inference_ms,
        },
    )

    # Log the prediction, but never let a logging failure break the response
    # the caller is waiting on -- a down/misconfigured database shouldn't
    # take the prediction feature down with it.
    try:
        log_entry = PredictionLog(
            input_features=json.dumps(features.model_dump()),
            prediction=result.prediction,
            probability_leave=result.probability_leave,
            threshold_used=result.threshold_used,
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        db.rollback()
        error_logger.error(
            "failed to write prediction log to database", extra={"error": str(e)}, exc_info=e
        )

    return result


@app.get("/predictions", response_model=list[PredictionLogEntry], tags=["Prediction"])
def list_predictions(limit: int = 50, db: Session = Depends(get_db)):
    """Most recent logged predictions, newest first."""
    rows = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(limit).all()
    return [
        PredictionLogEntry(
            id=row.id,
            timestamp=row.timestamp.isoformat(),
            input_features=json.loads(row.input_features),
            prediction=row.prediction,
            probability_leave=row.probability_leave,
            threshold_used=row.threshold_used,
        )
        for row in rows
    ]


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
def explain_prediction(features: EmployeeFeatures, top_n: int = 10):
    """Top SHAP-value contributing features for one employee, as numbers
    (for a frontend that wants to render its own chart or table)."""
    try:
        return explain.explain_instance(features, top_n=top_n)
    except Exception as e:
        error_logger.error("explanation failed", extra={"error": str(e)}, exc_info=e)
        raise HTTPException(status_code=400, detail=f"Explanation failed: {e}")


@app.get("/explain/summary", tags=["Explainability"])
def explain_summary(top_n: int = 15):
    """Global feature-importance bar chart (mean |SHAP value| across a
    background sample) as a PNG image."""
    try:
        png_bytes = explain.summary_plot_png(top_n=top_n)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        error_logger.error("summary plot failed", extra={"error": str(e)}, exc_info=e)
        raise HTTPException(status_code=400, detail=f"Summary plot failed: {e}")


@app.post("/explain/waterfall", tags=["Explainability"])
def explain_waterfall(features: EmployeeFeatures, top_n: int = 10):
    """Per-employee SHAP waterfall plot as a PNG image."""
    try:
        png_bytes = explain.waterfall_plot_png(features, top_n=top_n)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        error_logger.error("waterfall plot failed", extra={"error": str(e)}, exc_info=e)
        raise HTTPException(status_code=400, detail=f"Waterfall plot failed: {e}")


@app.post("/explain/force", tags=["Explainability"])
def explain_force(features: EmployeeFeatures, top_n: int = 6):
    """Per-employee SHAP force plot as a PNG image (optional secondary view --
    see the note in app/explain.py about its label-crowding limitation)."""
    try:
        png_bytes = explain.force_plot_png(features, top_n=top_n)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        error_logger.error("force plot failed", extra={"error": str(e)}, exc_info=e)
        raise HTTPException(status_code=400, detail=f"Force plot failed: {e}")
