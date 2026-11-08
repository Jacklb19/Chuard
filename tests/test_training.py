import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluate import classification_metrics
from src.schema import FEATURE_COLUMNS, validate_customers
from src.train import make_pipeline, split_positions


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
