"""SHAP real sobre modelos sintéticos: prueba técnica, no resultado Telco."""

import json

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.explain import (
    ExplanationService,
    ExplanationUnavailableError,
    clear_explanation_cache,
    explain_customer,
    get_explanation_service,
    prepare_explanation_context,
    save_explanation_plots,
)
from src.schema import CATEGORIES, FEATURE_COLUMNS, validate_customers
from src.train import make_pipeline


@pytest.fixture
def synthetic_training(customer):
    records = []
    for i in range(20):
        records.append(
            {
                **customer,
                "tenure": i + 1,
                "MonthlyCharges": 20 + i * 3,
                "TotalCharges": (20 + i * 3) * (i + 1),
                "Contract": "Month-to-month" if i % 2 else "One year",
                "PaymentMethod": CATEGORIES["PaymentMethod"][i % 4],
                "Partner": "Yes" if i % 3 else "No",
                "Churn": "Yes" if i % 2 else "No",
            }
        )
    return validate_customers(pd.DataFrame(records), mode="training").data


def _bundle(training, *, backend="auto", classifier=None, calibrated=False):
    x = training.loc[:, list(FEATURE_COLUMNS)]
    y = training["Churn"].map({"No": 0, "Yes": 1})
    estimator = make_pipeline(classifier, use_derived=True)
    if calibrated:
        estimator = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=2)
    estimator.fit(x, y)
    return {
        "pipeline": estimator,
        "model_name": "synthetic_test_only",
        "model_version": "test1",
        "explanation_context": prepare_explanation_context(
            training,
            method=backend,
            background_size=5,
            max_evals=39,
            max_rows=20,
        ),
    }


@pytest.mark.parametrize(
    "backend", ["linear", "tree", "xgboost", "lightgbm", "svm", "permutation", "calibrated"]
)
def test_real_shap_reconstructs_correct_output(synthetic_training, backend):
    classifiers = {
        "tree": RandomForestClassifier(n_estimators=8, max_depth=3, random_state=42),
        "xgboost": XGBClassifier(n_estimators=5, max_depth=2, n_jobs=1, random_state=42),
        "lightgbm": LGBMClassifier(
            n_estimators=5,
            num_leaves=4,
            min_child_samples=2,
            verbosity=-1,
            n_jobs=1,
            random_state=42,
        ),
        "svm": SVC(random_state=42),
    }
    classifier = classifiers.get(backend)
    bundle = _bundle(
        synthetic_training,
        classifier=classifier,
        backend="permutation" if backend == "permutation" else "auto",
        calibrated=backend in {"calibrated", "svm"},
    )
    service = ExplanationService(bundle)
    result = service.explain(synthetic_training.iloc[:2])
    np.testing.assert_allclose(
        result.base_values + result.values.sum(axis=1), result.explained_values, atol=1e-6
    )
    if backend == "linear":
        assert result.scale == "log_odds"
        assert result.explained_output == "decision_function"
        assert result.values.shape[1] == 25  # One-Hot agrupado + derivadas.
    else:
        assert result.scale == "probability"
        np.testing.assert_allclose(result.explained_values, result.probabilities)
    if backend == "calibrated":
        assert result.method == "permutation"
        assert result.model_scope == "complete_estimator"
    factors = result.factors(0)
    assert 1 <= len(factors) <= 5
    assert all(item["direction"] in {"aumenta", "disminuye"} for item in factors)
    json.dumps(factors, allow_nan=False)


def test_permutation_is_independent_of_batch_order_and_preserves_rng(synthetic_training):
    bundle = _bundle(synthetic_training, backend="permutation")
    service = ExplanationService(bundle)
    state_before = np.random.get_state()
    first = service.explain(synthetic_training.iloc[[0, 1]])
    second = service.explain(synthetic_training.iloc[[1, 0]])
    single = service.explain(synthetic_training.iloc[[0]])
    np.testing.assert_allclose(first.values.iloc[0], second.values.iloc[1])
    np.testing.assert_allclose(first.values.iloc[0], single.values.iloc[0])
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_cache_invalidates_on_model_version_and_background(synthetic_training):
    clear_explanation_cache()
    bundle = _bundle(synthetic_training)
    original = get_explanation_service(bundle)
    assert get_explanation_service(bundle) is original
    bundle["model_version"] = "test2"
    current = get_explanation_service(bundle)
    assert current is not original
    bundle["explanation_context"]["background"].iloc[0, 4] += 1
    assert get_explanation_service(bundle) is not current


def test_missing_resources_and_invalid_background_are_explicit(synthetic_training):
    with pytest.raises(ExplanationUnavailableError, match="SHAP"):
        explain_customer(synthetic_training.iloc[[0]], {})
    bundle = _bundle(synthetic_training)
    bundle["explanation_context"]["origin"] = "test"
    with pytest.raises(ValueError, match="origen"):
        get_explanation_service(bundle)
    with pytest.raises(ValueError, match="exactamente"):
        explain_customer(synthetic_training.iloc[:2], bundle)
    with pytest.raises(ValueError, match="permutación"):
        prepare_explanation_context(synthetic_training, max_evals=2)


def test_native_explanation_cannot_misrepresent_calibrated_model(synthetic_training):
    bundle = _bundle(synthetic_training, calibrated=True, backend="linear_log_odds")
    with pytest.raises(ValueError, match="no calibrado"):
        ExplanationService(bundle)


def test_plots_and_metadata_are_real_artifacts(synthetic_training, tmp_path):
    result = ExplanationService(_bundle(synthetic_training)).explain(synthetic_training.iloc[:4])
    metadata = save_explanation_plots(result, tmp_path)
    assert metadata["scale"] == "log_odds"
    assert metadata["causal_effects_proven"] is False
    assert (tmp_path / "shap_global.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "shap_individual.png").read_bytes().startswith(b"\x89PNG")
    assert pd.read_csv(tmp_path / "shap_values.csv").shape == result.values.shape
