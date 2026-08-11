"""Employee Attrition — training with MLflow tracking.

Reproduces the model comparison from the notebook (Weeks 1-3), but every
candidate is now logged as its own MLflow run: parameters, metrics, the
confusion matrix, and the fitted pipeline itself as a model artifact. The
best run (by test F1) is registered in the MLflow Model Registry as
"employee-attrition-classifier" with a "champion" alias.

Run with:
    python train.py

Then view results with:
    mlflow ui --backend-store-uri ../mlruns
(from inside training/, or point --backend-store-uri at wherever mlruns/
ends up — see the printed path at the end of this script.)
"""

import warnings

warnings.filterwarnings("ignore")

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
THIS_DIR = Path(__file__).resolve().parent
DATA_PATH = THIS_DIR / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MLRUNS_PATH = THIS_DIR.parent / "mlruns"
MLFLOW_DB_PATH = THIS_DIR.parent / "mlflow.db"
EXPERIMENT_NAME = "employee-attrition"
REGISTERED_MODEL_NAME = "employee-attrition-classifier"
FINAL_THRESHOLD = 0.66  # carried over from the notebook's threshold tuning

NUMERICAL_FEATURES = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]
NOMINAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]
ORDINAL_FEATURES = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance",
]


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("nominal", OneHotEncoder(handle_unknown="ignore"), NOMINAL_FEATURES),
            ("ordinal", "passthrough", ORDINAL_FEATURES),
        ],
        remainder="drop",
    )


def confusion_matrix_png(y_true, y_pred, title) -> bytes:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stay", "Leave"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Stay", "Leave"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def main():
    # MLflow 3.x's plain filesystem tracking store ("file:./mlruns") is in
    # maintenance mode and raises on first use -- a local SQLite database is
    # the supported lightweight backend now. Artifacts (models, images) still
    # land on the local filesystem under mlruns/, just tracked via the DB.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")

    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=f"file:{MLRUNS_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)
    new_df = df.drop(["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"], axis=1)
    X = new_df.drop("Attrition", axis=1)
    y = new_df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    candidates = {
        "logistic_regression": (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, C=1.0),
            {"C": 1.0, "max_iter": 2000},
        ),
        "random_forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE, n_estimators=300, max_depth=8, min_samples_leaf=4
            ),
            {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 4},
        ),
        "hist_gradient_boosting": (HistGradientBoostingClassifier(random_state=RANDOM_STATE), {}),
        "xgboost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
            ),
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1},
        ),
    }

    results = []

    for name, (model, params) in candidates.items():
        with mlflow.start_run(run_name=name):
            preprocessor = build_preprocessor()
            pipeline = ImbPipeline(
                [
                    ("preprocess", preprocessor),
                    ("smote", SMOTE(random_state=RANDOM_STATE)),
                    ("model", model),
                ]
            )
            pipeline.fit(X_train, y_train_enc)

            pred = pipeline.predict(X_test)
            prob = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test_enc, pred, prob)

            # Also log metrics at the tuned threshold (0.66), not just default 0.5
            pred_at_threshold = (prob >= FINAL_THRESHOLD).astype(int)
            metrics_at_threshold = evaluate(y_test_enc, pred_at_threshold, prob)

            mlflow.log_param("model_type", name)
            mlflow.log_param("resampling", "SMOTE")
            mlflow.log_param("decision_threshold", 0.5)
            for k, v in params.items():
                mlflow.log_param(k, v)

            mlflow.log_metrics(metrics)
            mlflow.log_metrics(
                {f"{k}_at_tuned_threshold": v for k, v in metrics_at_threshold.items()}
            )

            cm_png = confusion_matrix_png(y_test_enc, pred, f"{name} (threshold=0.5)")
            mlflow.log_image(plt.imread(io.BytesIO(cm_png)), "confusion_matrix.png")

            mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")

            results.append({"run_name": name, "run_id": mlflow.active_run().info.run_id, **metrics})
            print(
                f"{name:25s} F1={metrics['f1']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
                f"(at tuned threshold: F1={metrics_at_threshold['f1']:.4f})"
            )

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\nRanked by test F1 (default threshold):")
    print(results_df.to_string(index=False))

    # Register the best run's model in the MLflow Model Registry
    best = results_df.iloc[0]
    best_run_id = best["run_id"]
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", registered.version)

    print(
        f"\nRegistered '{best['run_name']}' (run {best_run_id}) as "
        f"{REGISTERED_MODEL_NAME} v{registered.version}, alias 'champion'."
    )
    print(f"\nMLflow tracking DB: {MLFLOW_DB_PATH}")
    print(f"MLflow artifacts:   {MLRUNS_PATH}")
    print("View with: mlflow ui --backend-store-uri", f"sqlite:///{MLFLOW_DB_PATH}")


if __name__ == "__main__":
    main()
