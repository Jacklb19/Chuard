"""Web boundary checks using synthetic records, never the reserved test set."""

import io
import json
from unittest.mock import patch

import pandas as pd
import pytest

from src.application import (
    EXAMPLE_PATH,
    UI_BATCH_LIMIT,
    export_customers,
    filter_customers,
    load_application_model,
)
from src.config import MODEL_MANIFEST_PATH
from src.data import read_customer_csv
from src.schema import InputValidationError


def test_corrupt_artifact_is_rejected_before_deserialization(tmp_path):
    artifact = tmp_path / "corrupt.joblib"
    artifact.write_bytes(b"untrusted executable artifact")
    with patch("src.application.load_model_bundle") as loader:
        with pytest.raises(ValueError, match="huella"):
            load_application_model(artifact, MODEL_MANIFEST_PATH)
        loader.assert_not_called()


def test_export_preserves_numbers_and_nested_results_and_escapes_formulas():
    frame = pd.DataFrame(
        {
            "customerID": ["=HYPERLINK(1)", " \t@danger", "normal"],
            "priority_score": [1.5, 0.0, -2.0],
            "recommendations": [["Revisar contrato, sin promesas"], [], ["Sí"]],
        }
    )
    exported = pd.read_csv(io.BytesIO(export_customers(frame)))
    assert exported.customerID.tolist() == ["'=HYPERLINK(1)", "' \t@danger", "normal"]
    assert exported.priority_score.tolist() == [1.5, 0.0, -2.0]
    assert json.loads(exported.recommendations.iloc[0]) == frame.recommendations.iloc[0]
    assert frame.customerID.iloc[0] == "=HYPERLINK(1)"


def test_filters_preserve_duplicate_id_row_identity_and_exclude_observed_churn():
    frame = pd.DataFrame(
        {
            "customerID": ["same", "same", "[literal]"],
            "risk_level": ["Alto", "Bajo", "Alto"],
            "Churn": ["Yes", "No", "No"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "InternetService": ["Fiber optic", "DSL", "No"],
        },
        index=[9, 17, 23],
    )
    assert filter_customers(frame, search="same").index.tolist() == [9, 17]
    assert filter_customers(frame, active_only=True).index.tolist() == [17, 23]
    assert filter_customers(frame, search="[literal]", risk="Alto").index.tolist() == [23]
    assert filter_customers(frame, contract="One year", internet="DSL").index.tolist() == [17]


def test_synthetic_example_and_ui_row_limit():
    validated = read_customer_csv(EXAMPLE_PATH, max_rows=UI_BATCH_LIMIT)
    assert validated.data.customerID.str.startswith("DEMO-").all()
    oversized = pd.concat([validated.data] * UI_BATCH_LIMIT, ignore_index=True)
    with pytest.raises(InputValidationError, match="Demasiadas filas"):
        read_customer_csv(
            io.BytesIO(oversized.to_csv(index=False).encode()), max_rows=UI_BATCH_LIMIT
        )
