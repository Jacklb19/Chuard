"""Protecciones de trazabilidad con clientes y resultados exclusivamente sintéticos."""

import hashlib
import json

import pandas as pd
import pytest

from src.config import CONTRACT_VERSION, DEFAULT_RISK_LIMITS
from src.data import dataset_sha256
from src.evaluate import classification_metrics, probability_metrics
from src.explain import prepare_explanation_context
from src.reporting import (
    FrozenResources,
    generate_frozen_reports,
    load_frozen_resources,
    restore_splits,
)
from src.reproduce import compare_reproduction
from src.schema import FEATURE_COLUMNS, validate_customers
from src.train import make_pipeline, split_positions


@pytest.fixture
def synthetic_data(customer):
    records = [
        {
            **customer,
            "customerID": f"sintetico_{i}",
            "tenure": i + 1,
            "MonthlyCharges": 20 + i,
            "TotalCharges": (20 + i) * (i + 1),
            "Contract": "Month-to-month" if i % 2 else "One year",
            "Churn": "Yes" if i % 2 else "No",
        }
        for i in range(50)
    ]
    return validate_customers(pd.DataFrame(records), mode="training").data


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_restore_requires_exact_hash_without_overwriting(synthetic_data, tmp_path, newline):
    data = tmp_path / "data.csv"
    synthetic_data.to_csv(data, index=False)
    splits = split_positions(synthetic_data.Churn.map({"No": 0, "Yes": 1}))
    record = {
        "dataset_sha256": dataset_sha256(data),
        "random_state": 42,
        "index_kind": "posición normalizada desde 0",
        "splits": splits,
    }
    payload = json.dumps(record, indent=2).replace("\n", newline).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    manifest = tmp_path / "model.manifest.json"
    manifest.write_text(json.dumps({**record, "split_manifest_sha256": digest}), "utf-8")
    output = tmp_path / "splits.json"
    assert restore_splits(data, manifest, output) == digest
    assert output.read_bytes() == payload
    output.write_bytes(b"cambio ajeno")
    with pytest.raises(ValueError, match="Huella"):
        restore_splits(data, manifest, output)
    assert output.read_bytes() == b"cambio ajeno"


def test_corrupted_inputs_fail_before_joblib_load(tmp_path, monkeypatch):
    data = tmp_path / "data.csv"
    data.write_text("corrupto", "utf-8")
    meta = {
        "dataset_sha256": "no_coincide",
        "model_sha256": "x",
        "split_manifest_sha256": "x",
        "validation_report_sha256": "x",
        "reserved_test": {"report_sha256": "x"},
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(meta), "utf-8")

    def forbidden(*args):
        pytest.fail("No debe deserializar un artefacto si una huella no coincide.")

    monkeypatch.setattr("src.reporting.load_model_bundle", forbidden)
    with pytest.raises(ValueError, match="Huella"):
        load_frozen_resources(data_path=data, model_manifest=manifest)


def test_reports_do_not_train_or_reevaluate_reserved_test(synthetic_data, tmp_path, monkeypatch):
    training, validation = synthetic_data.iloc[:40], synthetic_data.iloc[40:]
    estimator = make_pipeline().fit(
        training.loc[:, list(FEATURE_COLUMNS)], training.Churn.map({"No": 0, "Yes": 1})
    )
    probabilities = estimator.predict_proba(validation.loc[:, list(FEATURE_COLUMNS)])[:, 1]
    metrics = {
        **classification_metrics(
            validation.Churn.map({"No": 0, "Yes": 1}), probabilities, threshold=0.5
        ),
        **probability_metrics(validation.Churn.map({"No": 0, "Yes": 1}), probabilities),
    }

    def forbidden(*args, **kwargs):
        pytest.fail("El informe no puede entrenar.")

    monkeypatch.setattr(estimator, "fit", forbidden)
    bundle = {
        "pipeline": estimator,
        "feature_columns": list(FEATURE_COLUMNS),
        "model_name": "sintetico_0",
        "model_version": "test",
        "contract_version": CONTRACT_VERSION,
        "decision_threshold": 0.5,
        "risk_limits": DEFAULT_RISK_LIMITS,
        "explanation_context": prepare_explanation_context(
            training, method="permutation", background_size=5, max_evals=39
        ),
    }
    matrix = metrics["confusion_matrix"]
    comparison = pd.DataFrame(
        [
            {
                "candidate": f"sintetico_{i}",
                "decision_threshold": 0.5,
                **metrics,
                "true_negatives": matrix[0][0],
                "false_positives": matrix[0][1],
                "false_negatives": matrix[1][0],
                "true_positives": matrix[1][1],
            }
            for i in range(10)
        ]
    )
    # No hay filas de prueba en los recursos. Sus conteos guardados se usan solo para dibujar.
    saved_test = pd.DataFrame(
        [{"true_negatives": 9, "false_positives": 2, "false_negatives": 1, "true_positives": 4}]
    )
    resources = FrozenResources(
        bundle,
        {
            "model_name": "sintetico_0",
            "model_sha256": "sintetico",
            "dataset_sha256": "sintetico",
            "split_manifest_sha256": "sintetico",
            "metrics": {"validation": metrics},
        },
        training,
        validation,
        comparison,
        saved_test,
    )
    output = tmp_path / "evaluation"
    result = generate_frozen_reports(
        resources, evaluation_dir=output, explanation_dir=tmp_path / "explain", global_rows=2
    )
    assert result["validation_reproduced"] is True
    assert result["test_reevaluated"] is False
    assert result["shap_max_reconstruction_error"] < 1e-6
    assert (output / "matriz_prueba_final_guardada.png").is_file()
    assert not (output / "roc_prueba.png").exists()


def test_reproduction_reports_mismatch_without_hiding_it():
    original = pd.DataFrame(
        [
            {
                "candidate": "sintetico",
                "calibration": "sigmoid",
                "decision_threshold": 0.5,
                **{
                    key: 0.7
                    for key in (
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                        "average_precision",
                        "brier",
                    )
                },
            }
        ]
    )
    reproduced = original.rename(
        columns={
            "candidate": "candidato",
            "calibration": "calibracion",
            "decision_threshold": "umbral",
        }
    ).copy()
    reproduced["elegible"] = True
    assert compare_reproduction(original, reproduced)["all_candidates_match"]
    reproduced.loc[0, "f1"] = 0.6
    assert not compare_reproduction(original, reproduced)["all_candidates_match"]
