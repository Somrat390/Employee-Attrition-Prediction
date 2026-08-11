"""SHAP explainability for the Employee Attrition model.

Uses shap.LinearExplainer against the fitted LogisticRegression step, with a
150-row sample of the training data (shap_background_sample.csv) as the
background distribution. Kept in its own module, like model_loader.py, so the
(fairly expensive) explainer is built once and reused across requests.
"""

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless -- no display available on a server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from app.model_loader import get_pipeline
from app.schemas import EmployeeFeatures

BACKGROUND_PATH = Path(__file__).resolve().parent.parent / "models" / "shap_background_sample.csv"

NOMINAL_PREFIXES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

_explainer = None
_feature_names = None
_background_processed = None


def _is_onehot_column(name: str) -> bool:
    return any(name.startswith(prefix + "_") for prefix in NOMINAL_PREFIXES)


def _build_feature_names(preprocessor, numerical_features, nominal_features, ordinal_features):
    onehot_names = list(
        preprocessor.named_transformers_["nominal"].get_feature_names_out(nominal_features)
    )
    return numerical_features + onehot_names + ordinal_features


def _get_explainer():
    """Build (once) and return (explainer, feature_names, background_processed)."""
    global _explainer, _feature_names, _background_processed
    if _explainer is not None:
        return _explainer, _feature_names, _background_processed

    pipeline = get_pipeline()
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    numerical_features = [
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
    nominal_features = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "Gender",
        "JobRole",
        "MaritalStatus",
        "OverTime",
    ]
    ordinal_features = [
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

    background_raw = pd.read_csv(BACKGROUND_PATH)
    background_processed = preprocessor.transform(background_raw)
    if hasattr(background_processed, "toarray"):
        background_processed = background_processed.toarray()

    feature_names = _build_feature_names(
        preprocessor, numerical_features, nominal_features, ordinal_features
    )

    masker = shap.maskers.Independent(
        background_processed, max_samples=background_processed.shape[0]
    )
    explainer = shap.LinearExplainer(model, masker)

    _explainer, _feature_names, _background_processed = (
        explainer,
        feature_names,
        background_processed,
    )
    return _explainer, _feature_names, _background_processed


def _preprocess_features(features: EmployeeFeatures):
    pipeline = get_pipeline()
    preprocessor = pipeline.named_steps["preprocess"]
    row = pd.DataFrame([features.model_dump()])
    processed = preprocessor.transform(row)
    if hasattr(processed, "toarray"):
        processed = processed.toarray()
    return processed


def get_global_importance(top_n: int = 15):
    """Mean |SHAP value| per feature across the background sample -- a global
    ranking of which features matter most to the model overall (not for any
    one employee)."""
    explainer, feature_names, background_processed = _get_explainer()
    shap_values = explainer.shap_values(background_processed)
    mean_abs = np.abs(shap_values).mean(axis=0)

    order = np.argsort(mean_abs)[::-1][:top_n]
    return [(feature_names[i], float(mean_abs[i])) for i in order]


def explain_instance(features: EmployeeFeatures, top_n: int = 10):
    """Per-employee SHAP contributions. One-hot columns that are 0 for this
    employee (i.e. categories they don't belong to) are dropped -- keeping
    only the category they're actually in makes the result readable."""
    explainer, feature_names, _ = _get_explainer()
    processed = _preprocess_features(features)
    shap_values = explainer.shap_values(processed)[0]
    feature_values = processed[0]

    contributions = []
    for name, val, fval in zip(feature_names, shap_values, feature_values):
        if _is_onehot_column(name) and fval < 0.5:
            continue  # skip categories this employee isn't in
        contributions.append(
            {"feature": name, "shap_value": float(val), "feature_value": float(fval)}
        )

    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    base_value = float(
        explainer.expected_value
        if np.ndim(explainer.expected_value) == 0
        else explainer.expected_value[0]
    )

    return {
        "base_value": base_value,
        "contributions": contributions[:top_n],
    }


def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def summary_plot_png(top_n: int = 15) -> bytes:
    """Global feature-importance bar chart (mean |SHAP value|)."""
    ranked = get_global_importance(top_n=top_n)
    names = [n for n, _ in ranked][::-1]
    values = [v for _, v in ranked][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, values, color="#6c5ce7")
    ax.set_xlabel("Mean |SHAP value| (impact on model output)")
    ax.set_title("Global Feature Importance (SHAP)")
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def waterfall_plot_png(features: EmployeeFeatures, top_n: int = 10) -> bytes:
    """Per-employee SHAP waterfall: how each feature pushed the prediction
    up or down from the model's base rate."""
    explanation = explain_instance(features, top_n=top_n)
    base_value = explanation["base_value"]
    contributions = explanation["contributions"]

    names = [c["feature"] for c in contributions][::-1]
    values = [c["shap_value"] for c in contributions][::-1]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (red = pushes toward Leave, blue = pushes toward Stay)")
    ax.set_title(f"Why this prediction? (base value: {base_value:.2f})")
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def force_plot_png(features: EmployeeFeatures, top_n: int = 6) -> bytes:
    """Static (matplotlib) SHAP force plot for one employee, using the same
    filtered top-N contributions as the waterfall plot. Kept to a smaller
    top_n than the waterfall (default 6, not 10) because SHAP's matplotlib
    force plot renders feature labels along one crowded horizontal strip --
    more than ~6 makes them overlap and become unreadable regardless of
    figure size. The waterfall plot above is the primary explanation visual
    for this reason; the force plot is the optional secondary one."""
    explanation = explain_instance(features, top_n=top_n)
    base_value = explanation["base_value"]
    contributions = explanation["contributions"]

    names = [c["feature"] for c in contributions]
    shap_vals = np.array([c["shap_value"] for c in contributions])
    feature_vals = np.array([c["feature_value"] for c in contributions])

    fig = plt.figure(figsize=(13, 3))
    shap.force_plot(
        base_value,
        shap_vals,
        feature_vals,
        feature_names=names,
        matplotlib=True,
        show=False,
    )
    fig = plt.gcf()
    return _fig_to_png_bytes(fig)
