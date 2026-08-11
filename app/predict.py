"""Core prediction logic, kept separate from the FastAPI route so it can be
unit tested without spinning up the app (see tests/ in a later phase).
"""

import pandas as pd

from app.model_loader import get_label_encoder, get_pipeline, get_threshold
from app.schemas import EmployeeFeatures, PredictionResponse


def predict_attrition(features: EmployeeFeatures) -> PredictionResponse:
    pipeline = get_pipeline()
    threshold = get_threshold()
    label_encoder = get_label_encoder()

    # The pipeline's ColumnTransformer expects a DataFrame with the training
    # column names -- a single-row DataFrame from the validated request body.
    row = pd.DataFrame([features.model_dump()])

    # predict_proba()[:, 1] is P(class == 1). The label encoder was fit on the
    # training target ("No"/"Yes"), so we confirm which encoded value is "Yes"
    # rather than assuming 1 == "Yes".
    leave_class_index = list(label_encoder.classes_).index("Yes")
    probability_leave = float(pipeline.predict_proba(row)[0, leave_class_index])

    prediction = "Leave" if probability_leave >= threshold else "Stay"

    return PredictionResponse(
        prediction=prediction,
        probability_leave=round(probability_leave, 4),
        threshold_used=threshold,
    )
