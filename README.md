# Employee Attrition Prediction API (Phase 2, Week 1)

FastAPI backend that serves the Logistic Regression + SMOTE pipeline trained in
`Feature_Engineering_Preprocessing_v2.ipynb`.

## Project structure

```
employee_attrition/
│
├── app/
│   ├── main.py            # FastAPI app, routes: /, /health, /predict
│   ├── model_loader.py     # Loads the .pkl once, exposes pipeline/threshold/label_encoder
│   ├── schemas.py          # Pydantic request/response models with validation ranges
│   ├── predict.py           # Turns a validated request into a PredictionResponse
│
├── models/
│   └── employee_attrition_pipeline.pkl   # {"pipeline", "threshold", "label_encoder"}
│
├── requirements.txt
└── README.md
```

## Setup

```bash
cd employee_attrition
python -m venv venv
venv\Scripts\activate            # source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

**Windows note:** `requirements.txt` uses `>=` version ranges rather than exact
pins. Older exact pins (e.g. `pandas==2.2.2`) can force pip to compile pandas
from source on newer Python versions where no prebuilt wheel exists yet — that
compile needs Microsoft's Visual C++ build tools and fails without them. Version
ranges let pip pick a version that already ships a wheel, so no compiler is
needed.

## Run

```bash
uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger docs, or:

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

Expected response shape:

```json
{"prediction": "Leave", "probability_leave": 0.9992, "threshold_used": 0.66}
```

## Design notes

- **Why a dict of {pipeline, threshold, label_encoder} instead of just the model?**
  The threshold (0.66) was tuned in the notebook and is not the sklearn default of
  0.5 — saving it alongside the pipeline means the API doesn't silently drift back
  to an untuned decision boundary. The label encoder is saved too so the API never
  has to *assume* which encoded class means "Yes"/"Leave" — it looks it up.
- **Why does `EmployeeFeatures` only have 30 fields, not all 35 original columns?**
  `EmployeeNumber`, `EmployeeCount`, `Over18`, `StandardHours` were dropped before
  training (identifiers/constants) and the fitted `ColumnTransformer` only
  references the 30 columns it was fit on — there's nothing for the API to gain by
  accepting and then discarding four extra fields.
- **Why validate ranges/categories in `schemas.py` instead of trusting the model to
  cope?** The model was trained on values inside these exact ranges (e.g. Age 18-60,
  OverTime "Yes"/"No"). An out-of-range or misspelled category wouldn't crash the
  pipeline (the one-hot encoder has `handle_unknown="ignore"`) but it would silently
  produce a meaningless prediction instead of a clear error — validation surfaces
  the mistake immediately instead of returning a confident-looking wrong answer.

## Tested

- `GET /health` → `{"status": "ok", "model_loaded": true}`
- `POST /predict` with a high-risk profile (young, single, frequent travel, low
  satisfaction, overtime) → `Leave`, probability 0.999
- `POST /predict` with a low-risk profile (senior, married, high satisfaction, no
  overtime) → `Stay`, probability 0.0004
- Invalid category (`"BusinessTravel": "SOMETIMES"`) → 422 with a clear message
- Missing fields → 422 listing every missing field

## Next (Week 2)

A Streamlit frontend that posts to this `/predict` endpoint and displays the result.
**Done — see `frontend/app.py` below.**

---

# Week 2 — Streamlit Frontend

A form for all 30 employee fields (grouped into Personal & Job Details,
Compensation, Tenure, and Satisfaction & Performance) that calls the FastAPI
`/predict` endpoint and displays the result as "Employee will Leave/Stay" with
a probability metric and progress bar.

## Project structure (updated)

```
employee_attrition/
│
├── app/                # FastAPI backend (Week 1)
│   ├── main.py
│   ├── model_loader.py
│   ├── schemas.py
│   ├── predict.py
│
├── models/
│   └── employee_attrition_pipeline.pkl
│
├── frontend/            # Streamlit frontend (Week 2)
│   ├── app.py
│   └── requirements.txt
│
├── requirements.txt      # backend requirements
└── README.md
```

## Running both together

Two terminals, both from the `employee_attrition/` folder:

**Terminal 1 — backend:**
```bash
uvicorn app.main:app --reload
```

**Terminal 2 — frontend:**
```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

Streamlit opens at `http://localhost:8501`. Fill in the form and click **Predict**.

If your backend runs somewhere other than `http://127.0.0.1:8000`, set an
environment variable before launching Streamlit:

```bash
# macOS/Linux
export API_URL="http://your-backend-host:8000/predict"
# Windows PowerShell
$env:API_URL = "http://your-backend-host:8000/predict"
```

## Design notes

- **Why does the frontend have zero scikit-learn/pandas imports?** It only
  imports `streamlit` and `requests`. All model logic stays in the backend —
  the frontend just collects form inputs into a dict and POSTs JSON. This
  means frontend and backend can be deployed on completely different
  services (e.g. Streamlit Community Cloud + Render) without either one
  needing the other's dependencies installed.
- **Why a single `st.form` instead of inputs that predict on every change?**
  `st.form` batches all 30 inputs and only triggers one rerun (and one API
  call) when "Predict" is clicked — without it, Streamlit would re-run the
  whole script (and hit the API) on every single slider or dropdown change.
- **Why do the dropdown option lists live in `frontend/app.py` as plain
  Python lists instead of being fetched from the backend?** They're small,
  fixed vocabularies straight from the training data (e.g. exactly 3
  `Department` values) — fetching them from an API endpoint would add a
  network round-trip and a failure mode for values that never change.

## Tested

- Backend and frontend started together locally; `streamlit run frontend/app.py`
  boots cleanly with no errors and serves the page (HTTP 200).
- Submitted the exact default values of every form widget as a JSON payload
  directly against the running backend — returned `200` with a sensible
  `{"prediction": "Stay", "probability_leave": 0.0465, "threshold_used": 0.66}`.
- Backend-down case: stopping the FastAPI server and clicking Predict shows a
  clear `st.error` message instead of an unhandled exception.

## Next (Week 3)

SHAP explainability: feature importance, summary plot, and a waterfall plot
showing why one specific employee was scored as high-risk.
**Done — see below.**

---

# Week 3 — SHAP Explainability

Adds four things: global feature importance, a per-employee SHAP waterfall
plot, a per-employee SHAP force plot, and the raw contribution numbers as
JSON (for a frontend that wants a table instead of an image).

## What's new

**Backend (`app/explain.py`, new endpoints in `app/main.py`):**
- `GET /explain/summary` — global feature-importance bar chart (PNG), mean
  `|SHAP value|` across a 150-row background sample of the training data
  (`models/shap_background_sample.csv`).
- `POST /explain` — same per-employee contributions as the waterfall plot,
  as JSON `{"base_value": ..., "contributions": [{"feature", "shap_value",
  "feature_value"}, ...]}` — for a frontend that wants numbers, not an image.
- `POST /explain/waterfall` — per-employee SHAP waterfall chart (PNG):
  which features pushed *this* employee's prediction up or down.
- `POST /explain/force` — per-employee SHAP force plot (PNG), a secondary/
  optional view of the same contributions.

**Frontend (`frontend/app.py`):**
- A "🌍 Global feature importance" expander at the top of the page, loaded
  from `/explain/summary`.
- After clicking Predict: a "Why this prediction?" section showing the
  waterfall image from `/explain/waterfall`, plus an expander with the exact
  SHAP values from `/explain`.

## Design notes

- **Why `shap.LinearExplainer` instead of `KernelExplainer` or `TreeExplainer`?**
  The final model is a `LogisticRegression`, which is linear — `LinearExplainer`
  computes exact SHAP values in closed form for linear models, instead of the
  sampling-based approximation `KernelExplainer` would need. Much faster, and
  exact rather than approximate.
- **Why a saved 150-row `shap_background_sample.csv` instead of loading the
  full training CSV at request time?** SHAP needs a background distribution
  to compare each employee against, but the API shouldn't depend on having
  the entire original dataset sitting on the server — a small fixed sample,
  shipped as a versioned file in `models/`, is enough for stable SHAP values
  and keeps the API self-contained.
- **Why does `explain_instance()` drop one-hot columns that are 0 for an
  employee?** Since the training pipeline's `OneHotEncoder` doesn't use
  `drop='first'`, every category gets its own column, and a linear model
  places a coefficient on all of them (e.g. both `OverTime_Yes` and
  `OverTime_No`). Showing only the *active* category (the one this employee
  is actually in) instead of every category they're *not* in is what makes
  the waterfall readable — otherwise half the bars are describing categories
  that don't apply to this person.
- **Why is the force plot capped at 6 features while the waterfall gets 10?**
  SHAP's matplotlib force plot renders all feature labels along one crowded
  horizontal strip with connecting lines; past about 6 features the labels
  start overlapping regardless of figure size. The waterfall plot is the
  primary explanation visual for this reason — the force plot is the
  optional secondary one, exactly as flagged in the original roadmap.
- **Why does the frontend now import more than just `streamlit`/`requests`?**
  It still doesn't — `frontend/app.py` only calls `requests.get`/`.post` and
  displays the returned PNG bytes with `st.image()`. All SHAP/matplotlib
  logic stays server-side in `app/explain.py`, preserving the Week 2 design
  choice that the frontend has zero ML dependencies.

## Tested

- `explain.py`'s explainer, `get_global_importance()`, and `explain_instance()`
  run directly against the real saved pipeline and produce sensible,
  correctly-signed contributions (e.g. low `EnvironmentSatisfaction` and
  `OverTime = Yes` push toward "Leave"; low `NumCompaniesWorked` and
  seniority push toward "Stay").
- All three plot functions (`summary_plot_png`, `waterfall_plot_png`,
  `force_plot_png`) generate valid PNGs, verified both by file-type check and
  by viewing the images directly.
- All four endpoints tested against a live running server:
  `GET /explain/summary` → 200, valid PNG; `POST /explain` → 200 with the
  expected JSON shape; `POST /explain/waterfall` → 200, valid PNG;
  `POST /explain/force` → 200, valid PNG.
- The exact sequence of calls the frontend makes on page load
  (`/explain/summary`) and on form submit (`/predict` → `/explain/waterfall`
  → `/explain`) was replayed directly against the live backend and every
  step returned `200` with the expected content.
- Backend and frontend started together; Streamlit boots cleanly with the
  new expanders and image displays present.

## Next (Week 4)

Docker: a `Dockerfile` and `docker-compose.yml` to containerize the backend
(and optionally the frontend), then build and run the image locally.
**Done — see below.**

---

# Week 4 — Docker

Two images (backend, frontend) wired together with `docker-compose.yml` so
`docker compose up --build` starts the whole app.

## What's new

```
employee_attrition/
├── Dockerfile              # backend image
├── docker-compose.yml       # runs backend + frontend together
├── .dockerignore
├── frontend/
│   └── Dockerfile           # frontend image
├── app/ ...
├── models/ ...
└── requirements.txt
```

## Running it

```bash
docker compose up --build
```

- Backend: `http://localhost:8000` (`/docs` for Swagger)
- Frontend: `http://localhost:8501`

`docker compose down` to stop both.

## Design notes

- **Why two separate Dockerfiles instead of one image running both?**
  Backend and frontend have different dependencies (the backend needs
  scikit-learn/xgboost/shap; the frontend needs only streamlit/requests) and
  different scaling needs in production — one container restarting shouldn't
  take the other down. `docker-compose.yml` is what ties them together for
  local development.
- **Why does the backend `Dockerfile` only `COPY app` and `COPY models`,
  not the whole project?** The image should contain exactly what
  `app/main.py` needs at runtime — copying `frontend/`, `notebooks/`, or the
  raw dataset CSV would bloat the image with nothing the running server ever
  touches. This was actually verified, not assumed (see Tested, below).
- **Why `depends_on: condition: service_healthy` instead of a plain
  `depends_on: [backend]`?** Plain `depends_on` only waits for the backend
  *container* to start, not for uvicorn inside it to finish loading the
  model and become ready to answer requests. The healthcheck polls
  `/health` and makes Compose wait for an actual `200` before starting the
  frontend, avoiding a race where Streamlit's first request hits a backend
  that's still booting.
- **Why `API_BASE_URL=http://backend:8000` and not `127.0.0.1:8000` in the
  compose file?** Inside Docker's network, each service is reachable by its
  service name as a hostname — `backend` resolves to the backend
  container's IP. `127.0.0.1` from inside the frontend container would mean
  "the frontend container itself," which has no server listening on 8000.
  This is exactly why `frontend/app.py` reads `API_BASE_URL` from an
  environment variable rather than hardcoding a URL — the same image runs
  correctly against `127.0.0.1:8000` locally (Week 2/3) or `backend:8000`
  in Compose, with no code change.

## Tested

**Honest caveat first:** this sandbox doesn't have a Docker daemon
available, so I could not literally run `docker compose up --build` here.
What I did instead, to catch real bugs rather than just eyeball the
Dockerfiles:

- Copied *only* the files each `COPY` instruction actually copies (`app/`,
  `models/`, `requirements.txt` for the backend; `app.py`,
  `requirements.txt` for the frontend) into clean, isolated directories —
  simulating exactly what ends up inside each image, with nothing extra
  from the rest of the project available.
- Installed each `requirements.txt` into a fresh virtual environment from
  scratch (mirroring the `RUN pip install` layer) and confirmed both
  installed cleanly.
- Ran the backend's exact `CMD` (`uvicorn app.main:app --host 0.0.0.0 --port
  8000`) from inside the isolated backend directory and hit `/health`,
  `/predict`, `/explain/summary`, and `/explain/waterfall` — all returned
  correct responses using only the copied files, confirming there's no
  hidden dependency on something outside `app/`/`models/`.
- Ran the frontend's exact `CMD` from inside the isolated frontend
  directory with `API_BASE_URL` pointed at the isolated backend, and
  confirmed Streamlit booted cleanly and served `HTTP 200`.

**What this does and doesn't prove:** it confirms the Dockerfiles' `COPY`
lists are complete and the `CMD`s are correct — the most common way a
"looks right" Dockerfile actually fails on first build. It does **not**
confirm the images build cleanly inside an actual Docker daemon (base image
pull, layer caching, Linux/amd64 vs. your host architecture, etc.) — that
part needs to run on your machine. If `docker compose up --build` errors
out, paste the output here and I'll debug it directly.

## Next (Week 5)

MLflow: tracking experiments, metrics, parameters, and models from the
training notebook.
**Done — see below.**

---

# Week 5 — MLflow

Every model from the comparison now gets logged as its own MLflow run
(parameters, metrics at both the default and tuned threshold, a confusion
matrix image, and the fitted pipeline itself), and the best one is
registered in the MLflow Model Registry.

## What's new

```
employee_attrition/
├── training/
│   ├── train.py            # trains all 4 candidates, logs each to MLflow
│   └── data/
│       └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── mlflow.db                 # SQLite tracking store (created by train.py)
├── mlruns/                    # local artifact storage (created by train.py)
├── .gitignore                 # excludes mlflow.db/ and mlruns/ from git
└── requirements.txt            # now includes mlflow
```

## Running it

```bash
cd training
python train.py
```

This trains Logistic Regression, Random Forest, HistGradientBoosting, and
XGBoost (all with SMOTE, matching the notebook), logs each as an MLflow run
under the `employee-attrition` experiment, and registers the best one (by
test F1) as `employee-attrition-classifier` with a `champion` alias.

Then view the results:

```bash
cd ..   # back to employee_attrition/
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000` — you'll see all 4 runs under the
`employee-attrition` experiment, each with its logged parameters, metrics,
confusion matrix image, and downloadable model artifact.

## Design notes

- **Why SQLite instead of the plain local file store?** MLflow 3.x put the
  old filesystem-only backend (`file:./mlruns` as the *tracking* store, not
  just artifacts) into maintenance mode — it now raises an error on first
  use unless you explicitly opt out. `sqlite:///mlflow.db` is the smallest
  step up: still a single local file, no server to run, but on the
  supported path. Artifacts (models, images) still live on the local
  filesystem under `mlruns/`; only the tracking *metadata* moved into SQLite.
- **Why `serialization_format="cloudpickle"` when logging the model?**
  MLflow's default format for scikit-learn models is `skops`, which refuses
  to serialize `imblearn`'s `SMOTE`/`Pipeline` classes by default (an
  allow-list of "trusted types" that doesn't include them yet). Cloudpickle
  has no such restriction — the tradeoff, and it's a real one, is that
  loading a cloudpickled model executes arbitrary code, so only load models
  you trained yourself or trust the source of (true for this project, worth
  knowing if this pattern gets reused elsewhere).
- **Why log metrics at both the default (0.5) and tuned (0.66) threshold?**
  The notebook's headline result — Logistic Regression's F1 improving from
  0.48 to over 0.51 — only shows up at the tuned threshold. Logging only the
  default-threshold numbers would make MLflow's comparison view disagree
  with the notebook's conclusion for no good reason.
- **Why register only the best run, not all four?** The Model Registry is
  for "here's the model we'd actually deploy," not a second copy of the
  experiment log — the experiment/run view already holds all four for
  comparison. Registering everything would just make the registry as
  cluttered as the run list it's supposed to simplify.
- **Why an alias (`champion`) instead of the old `Staging`/`Production`
  stage labels?** MLflow deprecated model stages in favor of aliases —
  aliases are arbitrary, so `champion`/`challenger` reads more clearly than
  the old fixed `Staging`/`Production` vocabulary anyway, and it's the
  path that won't get deprecated further.

## Tested

- Ran `train.py` for real: all four models trained, logged, and ranked
  identically to the notebook's conclusion (Logistic Regression wins on F1
  and PR-AUC; tree models trade recall for precision).
- Confirmed the registered `champion` model loads back correctly via
  `mlflow.sklearn.load_model("models:/employee-attrition-classifier@champion")`
  and reproduces the same 99.98% leave-probability prediction on the
  familiar high-risk test profile from earlier weeks.
- Started `mlflow ui` for real and confirmed via its own REST API
  (`/api/2.0/mlflow/experiments/search`) that the `employee-attrition`
  experiment and its 4 runs are actually present and queryable — not just
  that the process started.

## Next (Week 6)

PostgreSQL: instead of only predicting, save every prediction (timestamp,
input features, probability, final prediction) to a database.
**Done — see below.**

---

# Week 6 — PostgreSQL

Every call to `/predict` now writes a row to a `prediction_logs` table:
timestamp, the full input, the prediction, and the probability. A new
`GET /predictions` endpoint reads them back, and the frontend shows the most
recent ones in an expander.

## What's new

```
employee_attrition/
├── app/
│   └── database.py     # SQLAlchemy engine, PredictionLog model, init_db()
├── docker-compose.yml    # now has a `db` (postgres:16-alpine) service
```

`app/main.py` gained:
- Table creation on startup (`init_db()`, safe to call every boot)
- Logging inside `/predict` after a successful prediction
- `GET /predictions?limit=50` — most recent logged predictions, newest first

## Running it

**With Docker Compose (recommended — Postgres included):**
```bash
docker compose up --build
```
The backend automatically gets `DATABASE_URL=postgresql+psycopg2://postgres:postgres@db:5432/attrition_db`
pointing at the new `db` service.

**Running the backend directly (no Docker):**
```bash
export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/attrition_db"
uvicorn app.main:app --reload
```
If you don't set `DATABASE_URL` at all, the backend falls back to a local
SQLite file (`predictions.db`) — handy for quickly running the app without
installing Postgres, though Compose is what actually matches the roadmap.

## Design notes

- **Why SQLAlchemy instead of raw `psycopg2` queries?** The same code needs
  to run against SQLite (Weeks 1-5, zero setup) and PostgreSQL (this week,
  the real target) — SQLAlchemy's `create_engine(DATABASE_URL)` picks the
  right dialect from the URL itself, so `app/database.py` has no
  Postgres-specific code path to maintain separately from a SQLite one.
- **Why does `input_features` store JSON text in one column instead of one
  column per feature?** `EmployeeFeatures` in `schemas.py` already validates
  and types all 30 fields — mirroring that as 30 separate database columns
  would mean updating two schemas in lockstep every time one field changes.
  A single JSON column stores exactly what was actually validated, once.
- **Why does a logging failure not fail the `/predict` request?** A person
  calling the API wants their prediction, not a 500 error because the
  database happened to be down. The `try/except/rollback` around the log
  write means the prediction always returns as long as the model works,
  independent of whether logging succeeded — verified directly (see
  Tested, below), not just written defensively and hoped for.
- **Why a Docker named volume (`pgdata`) for Postgres?** Without it,
  `docker compose down` would delete all logged predictions along with the
  container. The volume persists the data on disk across
  `down`/`up --build` cycles; `docker compose down -v` is the explicit way
  to wipe it if you actually want a clean slate.

## Tested

**Postgres was actually installed and run for this** (not simulated) —
this sandbox can install Ubuntu packages, so a real `postgresql:16` instance
was started and used for every check below.

- Started the backend with `DATABASE_URL` pointed at a live PostgreSQL
  instance; confirmed via `psql` directly (independent of the API's own
  claims) that the `prediction_logs` table was auto-created on startup.
- Submitted two different employee profiles through `/predict` and
  confirmed both rows landed in the real Postgres table — again checked
  with a direct `SELECT` in `psql`, not just by trusting the API's response.
- Called `GET /predictions` and confirmed it reads back exactly what's in
  the table, correctly deserializing the JSON `input_features` column.
- **Resilience check:** dropped the `prediction_logs` table out from under
  a running server, then called `/predict` again — it still returned a
  correct `200` prediction (logging failed silently and rolled back, exactly
  as designed) instead of the whole endpoint breaking.
- Re-simulated the exact backend Docker image (only the files its
  `Dockerfile` copies, per the Week 4 method) with `psycopg2-binary` added,
  ran it against the same real Postgres instance, and confirmed predictions
  and the `/predictions` history endpoint both work from the isolated image
  — not just from the full development checkout.
- Ran backend + frontend together against real Postgres and replayed the
  exact request sequence the frontend's new "Recent predictions" expander
  makes — confirmed it returns and displays the logged rows correctly.

## Next (Week 7)

Logging: structured logs for API requests, errors, and prediction timing —
important groundwork for running this in production.
**Done — see below.**

---

# Week 7 — Logging

Every request, prediction, and error now produces one structured JSON log
line on stdout — the format a log aggregator (or `docker logs`, or Render's
log viewer) can actually parse and search, instead of loose print statements.

## What's new

```
app/
└── logging_config.py    # JSONFormatter + configure_logging()
```

`app/main.py` gained:
- A request-logging middleware — one line per HTTP request with method,
  path, status code, duration, and a `request_id`
- A global exception handler for anything a route didn't already turn into
  an `HTTPException` — logs the full traceback, returns a generic 500
- Error-path logging in every route's existing `except` blocks (previously
  these just raised an `HTTPException` silently)
- Prediction-specific logging: result, probability, and the model's own
  inference time (separate from full request time, which also includes
  request parsing and the database write)

Every response also gets an `X-Request-ID` header — the same ID that's in
the log line for that request, so "the API was slow around 3:14pm" becomes
"here's the exact request_id, grep the logs for it."

## What the logs look like

```json
{"timestamp": "2026-08-08T13:36:08.528Z", "level": "INFO", "logger": "app.predictions", "message": "prediction made", "prediction": "Leave", "probability_leave": 0.9992, "threshold_used": 0.66, "inference_ms": 22.65}
{"timestamp": "2026-08-08T13:36:08.539Z", "level": "INFO", "logger": "app.requests", "message": "request completed", "request_id": "88116fcc-...", "method": "POST", "path": "/predict", "status_code": 200, "duration_ms": 36.09, "client_ip": "127.0.0.1"}
```

## Design notes

- **Why JSON lines on stdout instead of a log file?** Docker, Kubernetes,
  Render, and Railway all capture a container's stdout automatically and
  expect to hand it to a log viewer or aggregator — writing to a file
  inside the container instead means that data is invisible to all of them
  unless you also mount and tail a volume. This is the standard
  ["12-factor app"](https://12factor.net/logs) approach: treat logs as an
  event stream, not a file the app manages.
- **Why JSON instead of human-readable text?** A line like `Predicted Leave
  with 99.9% probability in 22ms` reads nicely in a terminal but can't be
  filtered by a log platform ("show me every request over 500ms" or "show
  me every 500 in the last hour") without fragile regex. JSON with
  consistent field names (`status_code`, `duration_ms`) is what tools like
  CloudWatch, Datadog, or even `jq` on the command line can actually query.
- **Why three separate loggers (`app.requests`, `app.predictions`,
  `app.errors`) instead of one?** So a person can turn one category up or
  down independently later (e.g. `logging.getLogger("app.requests").setLevel(logging.WARNING)`
  to silence routine request logs while keeping prediction and error logs at
  INFO) without touching the others.
- **Why time model inference separately from the full request?** The full
  request duration (in the `app.requests` log line) includes JSON parsing,
  Pydantic validation, and the database write — bundling all of that into
  one number would hide whether a slow request is actually the model being
  slow, or something else in the request path.
- **Why a global exception handler *and* per-route try/except, rather than
  just one or the other?** The per-route blocks turn *expected* failure
  modes (missing model file, bad SHAP computation) into meaningful HTTP
  status codes with a useful message. The global handler is the safety net
  for anything nobody anticipated — without it, an unexpected crash would
  return FastAPI's default response (which can leak a raw stack trace to
  the caller) instead of a clean 500 with the details safely in the logs
  instead of the response body.

## Tested

- Ran the server for real and inspected the actual log output for: a
  successful `/health` call, a successful `/predict` call (confirmed both
  the `app.predictions` line *and* the `app.requests` line appear, with
  sensible timings), a 422 validation error, and a 404 on an unknown route
  — every case produced the expected structured line with the right status
  code.
- Verified the `X-Request-ID` response header is present and matches the
  `request_id` field in that request's log line.
- **Forced a genuine startup failure** (temporarily removed the model file
  and restarted the server) and confirmed it fails fast with a clear
  traceback in uvicorn's error output, rather than starting up in a broken
  state and failing confusingly on the first request — this is also why
  `/health` exists as a separate fast-failing check rather than something
  that could silently succeed with no model loaded.

## Next (Week 8)

Testing: unit tests for preprocessing, the prediction endpoint, and input
validation.
**Done — see below.**

---

# Week 8 — Testing

45 tests across three files, run with `pytest`. They exercise the model
logic directly, input validation, and the full API (routing + database
logging together) — including a `test_predict_attrition_low_risk_profile`-style
check that the model's *direction* is still correct, not just that it runs
without crashing.

## What's new

```
employee_attrition/
├── tests/
│   ├── conftest.py       # shared fixtures: test client, sample payloads
│   ├── test_predict.py    # model logic, called directly (no HTTP)
│   ├── test_schemas.py    # input validation
│   └── test_api.py        # full API via FastAPI's TestClient
├── pytest.ini
└── requirements-dev.txt    # requirements.txt + pytest, pytest-cov, httpx
```

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

With a coverage report:
```bash
pytest --cov=app --cov-report=term-missing
```

Current result: **45 passed**, 86% statement coverage on `app/` (the
uncovered lines are mostly error-handling branches — like a missing model
file or a force-plot rendering failure — that need the environment itself to
be broken to trigger, which Week 7's manual "remove the model file" test
already exercised by hand).

## What's actually tested

- **`test_predict.py`** (model logic, no HTTP): the saved pipeline loads
  correctly, the threshold is genuinely the tuned 0.66 (not accidentally
  reverted to sklearn's 0.5 default), the pipeline has the expected
  preprocess → SMOTE → model steps, the preprocessor produces the expected
  51-column output, and — the most important check — a textbook high-risk
  profile predicts "Leave" with >90% probability while a textbook low-risk
  profile predicts "Stay" with <10%. This is the test that would actually
  catch a bad retrain, not just a broken import.
- **`test_schemas.py`** (input validation): valid input passes; a missing
  field, a wrong type, an out-of-range ordinal value, and every variation
  of a mistyped category (`"SOMETIMES"`, wrong case, empty string) are all
  rejected. Also checks that every category listed as a schema example is
  itself valid — catching the kind of typo where the `Literal` list and the
  `examples=[...]` value quietly drift apart.
- **`test_api.py`** (full API, via `TestClient`): `/health`, `/predict`
  (both directions, plus 422s), the `X-Request-ID` header, and — the one
  that exercises the most machinery at once — `/predict` followed by
  `/predictions` against the same in-memory database, confirming the
  logged row actually matches what was just predicted. Also smoke-tests
  `/explain`, `/explain/summary`, and `/explain/waterfall` (valid PNG magic
  bytes, correct JSON shape).

## Design notes

- **Why does `client` in `conftest.py` override the database with an
  in-memory SQLite session instead of using the real `predictions.db` or
  Postgres?** Tests need to be repeatable and side-effect-free — hitting
  the real database would leave rows behind, make tests depend on run
  order, and risk polluting actual prediction history with test data. FastAPI's
  `dependency_overrides` swaps `get_db` for the duration of each test only.
- **Why `poolclass=StaticPool` on that in-memory engine?** This one cost
  real debugging time, worth explaining: SQLAlchemy's default connection
  pool hands out a *new* connection per checkout, and `sqlite:///:memory:`
  creates a fresh, empty database *per connection* — so without
  `StaticPool` (which pins the engine to one single connection), the table
  created at fixture setup would vanish by the time the next request came
  in. The first run of this suite failed with exactly that error
  (`no such table: prediction_logs`) until this was added.
- **Why test the model's prediction *direction* (Leave vs. Stay) instead of
  just checking the endpoint returns 200?** A 200-only test would still
  pass if a bad retrain flipped the model's predictions or reset the
  threshold to 0.5 — it proves the server didn't crash, not that the model
  still works. Asserting on the two familiar profiles from every previous
  week catches a silent regression, not just a loud one.
- **Why migrate `@app.on_event("startup")` to a `lifespan` context manager
  while doing this?** FastAPI's `TestClient` still triggers whichever
  startup mechanism is registered, so this wasn't required to make tests
  pass — but `on_event` is deprecated and was already printing warnings on
  every single test run, and fixing it took one small change while already
  in this file.

## Tested

- Ran the full suite for real: 45 passed, then again from a completely
  fresh virtual environment built only from `requirements-dev.txt` (not the
  dev environment these tests were written in), confirming the test suite
  doesn't depend on anything installed by accident.
- Confirmed the real running server (via `uvicorn`, not the test client)
  still boots and serves `/health` correctly after the lifespan migration.
- Generated an actual coverage report rather than assuming — 86% on `app/`,
  with the gaps reviewed and understood rather than just noted as a number.

## Next (Week 9)

GitHub Actions: run this test suite automatically on every push, check
code formatting, and build the project in CI.
**Done — see below.**

---

# Week 9 — GitHub Actions / CI

Three jobs run on every push or pull request to `main`: tests, formatting/
linting, and a Docker build of both images.

## What's new

```
employee_attrition/
├── .github/workflows/ci.yml
└── pyproject.toml            # black + ruff config (100-char line length)
```

## What the workflow does

```
.github/workflows/ci.yml
├── test   -- pytest --cov=app --cov-report=term-missing
├── lint   -- black --check, then ruff check
└── build  -- docker build (backend), docker build (frontend)
              (needs: [test, lint] -- only runs if both pass)
```

## Design notes

- **Why three separate jobs instead of one long script?** GitHub Actions
  runs independent jobs in parallel by default (test and lint run at the
  same time here, not one after another) and reports each as its own
  pass/fail check on a pull request — "tests passed but formatting failed"
  is a much clearer signal than one job that failed somewhere in the
  middle of a combined script.
- **Why does `build` `need: [test, lint]`?** No point spending CI minutes
  building Docker images for code that doesn't even pass its own tests —
  `needs` makes the build job wait for, and require, both to succeed first.
- **Why black *and* ruff, not just one?** They do different jobs: black
  rewrites code to one canonical style (so nobody argues about spacing in
  a PR review), while ruff catches actual mistakes — unused imports,
  undefined names, unsorted imports. `ruff format` alone can replace black,
  but the two together is still the more common pairing today, and this is
  a fine place to see both used deliberately.
- **Why `E501` (line-too-long) ignored in ruff but not in black?** Enforcing
  line length in two tools with potentially different limits produces
  contradictory fix suggestions. Black already enforces the 100-character
  limit set in `pyproject.toml`; ruff is told to defer to it instead of
  re-litigating the same rule.
- **Why does `training/train.py` get a `per-file-ignore` for `E402`?** It
  deliberately calls `warnings.filterwarnings("ignore")` *before* its other
  imports, to suppress noisy deprecation warnings some ML libraries print
  on import — that ordering is intentional and correct for what it's
  doing, not disorganized code, so the one rule that would otherwise
  flag it is silenced for that file only, not the whole project.

## Tested

**Honest scope note, matching how Week 4's Docker section was handled:**
this sandbox has no way to actually trigger a GitHub Actions run (that
needs a real GitHub repository) — but everything the workflow *does* was
run for real, directly, first:
- Ran `black app tests training frontend` for real — it reformatted 9
  files (the codebase had never been run through a formatter before this),
  then confirmed `black --check` passes cleanly.
- Ran `ruff check`, fixed what it could automatically (unused imports,
  import sorting), and added one deliberate, documented per-file ignore for
  the one remaining case that's correct as written — confirmed
  `ruff check` now passes cleanly with zero unaddressed issues.
- **Re-ran the full pytest suite after reformatting** (45 passed) and
  **restarted the actual server** to confirm black's reformatting didn't
  silently change any behavior — a formatter is supposed to be a no-op on
  behavior, and this checks that it actually was.
- Validated `ci.yml`'s YAML with a parser to catch syntax errors before you
  ever push it.
- The `test` and `lint` jobs are exact copies of commands already verified
  above, so they're expected to pass in CI too. The `build` job's `docker
  build` commands use the same Dockerfiles verified in Week 4 (by
  replicating their exact `COPY` contents locally, since this sandbox has
  no Docker daemon) — GitHub's `ubuntu-latest` runners do have Docker
  pre-installed, so this is the one piece that genuinely only gets its
  first real test the moment you push.

## Next (Week 10)

Deployment: FastAPI to Render or Railway, Streamlit to Streamlit Community
Cloud — a live application anyone can use.
