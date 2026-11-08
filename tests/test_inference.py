"""Dobles explícitos del estimador y SHAP: pruebas de contrato, no de rendimiento."""

import numpy as np
import pandas as pd
import pytest

import src.inference as inference
from src.config import CONTRACT_VERSION, DEFAULT_RISK_LIMITS
from src.schema import FEATURE_COLUMNS


class SyntheticEstimator:
    classes_ = np.array([1, 0])

    def predict_proba(self, frame):
        return np.tile([0.7, 0.3], (len(frame), 1))


def synthetic_bundle() -> dict:
    return {
        "pipeline": SyntheticEstimator(),
        "feature_columns": list(FEATURE_COLUMNS),
        "model_name": "synthetic_test_only",
        "model_version": "test",
        "contract_version": CONTRACT_VERSION,
        "decision_threshold": 0.7,
        "risk_limits": DEFAULT_RISK_LIMITS.copy(),
    }


def test_score_matches_row_and_batch_with_explicit_shap_double(customer, monkeypatch):
    monkeypatch.setattr(
        inference,
        "explain_customer",
        lambda row, bundle: [
            {
                "feature": "Contract",
                "value": row.iloc[0]["Contract"],
                "direction": "aumenta",
                "contribution": 0.1,
                "label": "Contrato",
                "scale": "synthetic_test_only",
            },
        ],
    )
    frame = pd.DataFrame([{**customer, "customerID": "same"}] * 2)
    batch = inference.score_customers(frame, synthetic_bundle())
    individual = inference.score_customers(frame.iloc[[0]], synthetic_bundle())
    assert batch.loc[0, "churn_probability"] == 0.7
    assert batch.loc[0, "predicted_churn"] == 1
    assert batch.loc[0, "risk_level"] == "Alto"
    assert batch.loc[0, "priority_score"] == 56
    pd.testing.assert_series_equal(batch.iloc[0], individual.iloc[0])
    assert len(batch.attrs["validation_warnings"]) == 2


def test_scoring_does_not_silently_fake_explanations(customer_frame):
    with pytest.raises(NotImplementedError, match="SHAP"):
        inference.score_customers(customer_frame, synthetic_bundle())


def test_bundle_validation_and_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        inference.load_model_bundle(str(tmp_path / "absent.joblib"))
    bundle = synthetic_bundle()
    bundle["risk_limits"]["medium"] = 0.9
    with pytest.raises(ValueError, match="rango"):
        inference.validate_model_bundle(bundle)
