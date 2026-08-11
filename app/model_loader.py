"""Loads the trained pipeline + threshold + label encoder once, at import time.

Keeping this in its own module means main.py and predict.py both get the same
already-loaded objects rather than re-reading the .pkl file from disk on every
request.
"""

from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "employee_attrition_pipeline.pkl"

_artifact = None


def load_artifact():
    """Load (once) and return the dict saved by the training notebook:
    {"pipeline": ..., "threshold": ..., "label_encoder": ...}
    """
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Copy employee_attrition_pipeline.pkl into the models/ folder."
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


def get_pipeline():
    return load_artifact()["pipeline"]


def get_threshold() -> float:
    return load_artifact()["threshold"]


def get_label_encoder():
    return load_artifact()["label_encoder"]
