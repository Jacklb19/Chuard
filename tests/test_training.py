"""Registros sintéticos: verifican el protocolo, no el rendimiento con Telco."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluate import classification_metrics, select_threshold
from src.inference import load_model_bundle
from src.schema import CATEGORIES, FEATURE_COLUMNS, validate_customers
from src.train import (
    CALIBRATION_METHOD,
    FAMILY_ORDER,
    UNCALIBRATED,
    Partitions,
    base_estimator,
    build_candidate,
    calibration_decision,
    make_pipeline,
    positive_probability,
    run_experiment,
    run_final,
    search_space,
    select_winner,
    split_positions,
)


def test_split_is_disjoint_stratified_and_reproducible():
    y = np.array([0, 1] * 50)
    first = split_positions(y)
    assert first == split_positions(y)
    assert {name: len(values) for name, values in first.items()} == {
        "train": 60,
        "validation": 20,
        "test": 20,
    }
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert sorted(sum(first.values(), [])) == list(range(100))
    assert all(y[positions].mean() == 0.5 for positions in first.values())


@pytest.mark.parametrize("use_derived", [False, True])
def test_synthetic_pipeline_persistence_and_train_only_statistics(customer, tmp_path, use_derived):
    frame = pd.DataFrame(
        [
            {
                **customer,
                "tenure": i + 1,
                "MonthlyCharges": 20 + i,
                "TotalCharges": (20 + i) * (i + 1),
            }
            for i in range(20)
        ]
    )
    data = validate_customers(frame).data.loc[:, list(FEATURE_COLUMNS)]
    pipeline = make_pipeline(use_derived=use_derived).fit(data, np.array([0, 1] * 10))
    scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_[2], 29.5)
    before_mean = scaler.mean_.copy()
    changed = data.iloc[[0]].copy()
    changed["MonthlyCharges"] = 10000
    pipeline.predict_proba(changed)
    np.testing.assert_array_equal(scaler.mean_, before_mean)
    path = tmp_path / "synthetic_pipeline.joblib"
    joblib.dump(pipeline, path)
    np.testing.assert_allclose(pipeline.predict_proba(data), joblib.load(path).predict_proba(data))


def test_metrics_use_probability_for_auc_and_threshold_for_labels():
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.4, 0.8], threshold=0.4)
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["f1"] == 1
    assert metrics["average_precision"] == 1
    with pytest.raises(ValueError):
        classification_metrics([0, 0], [0.1, 0.2], threshold=0.5)
    # Sin predicciones positivas, precision/F1 valen 0 de forma explícita y sin aviso.
    assert classification_metrics([0, 1], [0.1, 0.2], threshold=0.9)["f1"] == 0


def test_threshold_rule_breaks_ties_towards_half_then_lower():
    chosen = select_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    # Cualquier umbral en (0.2, 0.8] separa perfectamente: gana el más cercano a 0.5.
    assert chosen["threshold"] == 0.5 and chosen["f1"] == 1 and chosen["tied_thresholds"] > 1
    with pytest.raises(ValueError):
        select_threshold([0, 1], [0.1, 0.9], grid=(0.0, 0.5))


@pytest.mark.parametrize("family", FAMILY_ORDER)
def test_search_spaces_are_valid_parameters(family):
    space = search_space(family, 2.5)
    pipeline = make_pipeline(base_estimator(family)).set_params(
        **{name: values[-1] for name, values in space.items()}
    )
    assert pipeline.named_steps["classifier"].get_params()
    with pytest.raises(ValueError):
        search_space(family, float("nan"))


def test_svm_requires_platt_probabilities():
    with pytest.raises(ValueError, match="Platt"):
        build_candidate("svm", False, {}, UNCALIBRATED)
    assert calibration_decision({}, family="svm") == CALIBRATION_METHOD


def test_calibration_requires_both_metrics_to_improve():
    raw = {"brier": 0.20, "log_loss": 0.50}
    both = {UNCALIBRATED: raw, CALIBRATION_METHOD: {"brier": 0.19, "log_loss": 0.48}}
    only_brier = {UNCALIBRATED: raw, CALIBRATION_METHOD: {"brier": 0.19, "log_loss": 0.499}}
    assert calibration_decision(both, family="random_forest") == CALIBRATION_METHOD
    assert calibration_decision(only_brier, family="random_forest") == UNCALIBRATED


def _candidate(key, family, f1, auc, recall, std, derived=False, eligible=True):
    return {
        "key": key,
        "family": family,
        "use_derived": derived,
        "eligible": eligible,
        "validation_f1": f1,
        "validation_roc_auc": auc,
        "validation_recall": recall,
        "cv_f1_std": std,
    }


def test_selection_uses_tolerance_then_documented_tie_breakers():
    candidates = [
        _candidate("dummy", "dummy", 0.99, 0.5, 1.0, 0.0, eligible=False),
        _candidate("svm", "svm", 0.640, 0.80, 0.70, 0.01),
        _candidate("xgb", "xgboost", 0.637, 0.84, 0.70, 0.02),
        _candidate("rf", "random_forest", 0.638, 0.84, 0.72, 0.03),
        _candidate("lr", "logistic_regression", 0.600, 0.90, 0.90, 0.01),
    ]
    winner, trace = select_winner(candidates)
    # F1 empata (±0.005) svm/xgb/rf; ROC-AUC deja xgb/rf; Recall prefiere rf.
    assert winner["key"] == "rf"
    assert len(trace) == 3
    tied = [
        _candidate("rf_con", "random_forest", 0.64, 0.84, 0.72, 0.02, derived=True),
        _candidate("lr", "logistic_regression", 0.64, 0.84, 0.72, 0.02),
    ]
    assert select_winner(tied)[0]["key"] == "lr"
    with pytest.raises(ValueError):
        select_winner([candidates[0]])


class ReversedClasses:
    classes_ = np.array([1, 0])

    def predict_proba(self, frame):
        return np.tile([0.8, 0.2], (len(frame), 1))


def test_positive_probability_follows_class_order(customer_frame):
    np.testing.assert_allclose(positive_probability(ReversedClasses(), customer_frame), [0.8])


@pytest.fixture
def synthetic_partitions(customer) -> Partitions:
    """Clientes inventados con señal simple; no representan al dataset Telco."""
    rng = np.random.default_rng(0)
    records = []
    for i in range(150):
        monthly = bool(i % 3)
        tenure = int(rng.integers(0, 72))
        charges = round(float(rng.uniform(20, 110)), 2)
        churn = monthly and (tenure < 24 or rng.random() < 0.15)
        records.append(
            {
                **customer,
                "customerID": f"sintetico_{i}",
                "tenure": tenure,
                "MonthlyCharges": charges,
                "TotalCharges": round(charges * tenure, 2) if tenure else "",
                "Contract": "Month-to-month" if monthly else ("One year", "Two year")[i % 2],
                "PaymentMethod": CATEGORIES["PaymentMethod"][i % 4],
                "Churn": "Yes" if churn else "No",
            }
        )
    data = validate_customers(pd.DataFrame(records), mode="training").data
    splits = split_positions(data["Churn"].map({"No": 0, "Yes": 1}))
    frames = {name: data.iloc[positions].copy() for name, positions in splits.items()}
    return Partitions(frames, "sintetico_dataset", "sintetico_manifiesto")


def test_experiment_then_single_final_evaluation(synthetic_partitions, tmp_path):
    table = tmp_path / "metricas.csv"
    record = run_experiment(
        synthetic_partitions,
        tmp_path,
        metrics_table=table,
        families=("logistic_regression", "svm"),
        n_iter=1,
        n_jobs=1,
    )
    keys = [candidate["key"] for candidate in record["candidates"]]
    assert keys[-1] == "dummy_prior" and len(keys) == 5
    assert record["winner"] != "dummy_prior"
    assert record["status"] == "validacion_completa_prueba_sin_usar"
    assert not (tmp_path / "test_report.json").exists()
    assert pd.read_csv(table)["ganador"].sum() == 1
    assert (tmp_path / "figures" / "roc_validacion.png").is_file()
    svm = next(item for item in record["candidates"] if item["family"] == "svm")
    assert svm["calibration"] == CALIBRATION_METHOD

    probabilities_path = tmp_path / "validation_probabilities.csv"
    original = pd.read_csv(probabilities_path)
    tampered = original.copy()
    tampered[record["winner"]] = 0.5
    tampered.to_csv(probabilities_path, index=False)
    with pytest.raises(ValueError, match="no reproduce"):
        run_final(synthetic_partitions, tmp_path, tmp_path / "model.joblib")
    original.to_csv(probabilities_path, index=False)

    report = run_final(synthetic_partitions, tmp_path, tmp_path / "model.joblib")
    assert report["status"] == "prueba_final_evaluada_una_vez"
    bundle = load_model_bundle(str(tmp_path / "model.joblib"))
    assert bundle["model_name"] == record["winner"]
    assert bundle["metrics"]["test"] == report["test"]
    assert bundle["explanation_context"]["origin"] == "training"
    assert set(bundle["training_ranges"]) == {
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    }
    manifest = json.loads((tmp_path / "model.manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_sha256"] == report["model_sha256"]
    assert "pipeline" not in manifest
    with pytest.raises(FileExistsError):
        run_final(synthetic_partitions, tmp_path, tmp_path / "other.joblib")
    with pytest.raises(FileExistsError, match="prueba final"):
        run_experiment(synthetic_partitions, tmp_path, metrics_table=table, n_iter=1, n_jobs=1)
