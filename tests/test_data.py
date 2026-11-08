import json

import pandas as pd
import pytest

from src.data import (
    audit_dataset,
    dataset_sha256,
    download_telco_dataset,
    segment_churn_rates,
    summarize_training,
    training_subset_from_manifest,
)
from src.schema import validate_customers


def _dataset_and_manifest(customer, tmp_path):
    frame = pd.DataFrame(
        [
            {**customer, "customerID": f"synthetic_{i}", "Churn": "Yes" if i % 2 else "No"}
            for i in range(10)
        ]
    )
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    manifest = {
        "dataset_sha256": dataset_sha256(path),
        "splits": {"train": [0, 1, 2, 3, 4, 5], "validation": [6, 7], "test": [8, 9]},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest_path, manifest


def test_structural_audit_does_not_drop_different_customers(customer, tmp_path):
    path, _, _ = _dataset_and_manifest(customer, tmp_path)
    report = audit_dataset(path)
    assert report["duplicate_rows"] == 0
    assert report["rows_sharing_profile_and_label"] == 10
    assert report["duplicates_removed"] == 0
    assert report["matches_expected_shape"] is False
    assert report["sha256"] == dataset_sha256(path)


def test_manifest_uses_only_training_and_rejects_overlap_or_wrong_origin(customer, tmp_path):
    path, manifest_path, manifest = _dataset_and_manifest(customer, tmp_path)
    result = training_subset_from_manifest(path, manifest_path)
    assert result["customerID"].tolist() == [f"synthetic_{i}" for i in range(6)]
    manifest["splits"]["test"] = [0, 9]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disjuntas"):
        training_subset_from_manifest(path, manifest_path)
    manifest["dataset_sha256"] = "wrong_source"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="huella"):
        training_subset_from_manifest(path, manifest_path)


def test_training_statistics_and_segment_sizes_do_not_use_other_rows(customer, tmp_path):
    path, manifest_path, _ = _dataset_and_manifest(customer, tmp_path)
    training = training_subset_from_manifest(path, manifest_path)
    summary = summarize_training(training)
    assert summary["rows"] == 6
    assert summary["training_ranges"]["tenure"] == {"min": 6, "max": 6}
    assert summary["high_monthly_charges"] == 80
    assert summary["numeric"]["SeniorCitizen"]["outside_iqr"] is None
    table = segment_churn_rates(training, "Contract")
    assert table["n"].sum() == 6
    assert table.iloc[0]["churn_rate"] == 0.5
    assert table.iloc[0]["small_segment"]


def test_missing_training_values_are_not_imputed_during_validation(customer):
    frame = pd.DataFrame([{**customer, "tenure": None}])
    normalized = validate_customers(frame, mode="training", require_target=False).data
    assert pd.isna(normalized.loc[0, "tenure"])


def test_download_never_overwrites_or_uses_unapproved_source(tmp_path):
    target = tmp_path / "original.csv"
    target.write_bytes(b"immutable")
    with pytest.raises(FileExistsError):
        download_telco_dataset(target)
    assert target.read_bytes() == b"immutable"
    with pytest.raises(ValueError, match="Kaggle"):
        download_telco_dataset(tmp_path / "new.csv", download_url="http://example.com/data")
