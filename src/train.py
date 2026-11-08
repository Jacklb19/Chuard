"""Base experimental: particiones y pipelines; no entrena al importar."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    LOGISTIC_MAX_ITER,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SPLIT_MANIFEST_PATH,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)
from src.data import dataset_sha256, read_customer_csv
from src.features import FeatureBuilder
from src.schema import DERIVED_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS


def split_positions(y, *, random_state: int = RANDOM_STATE) -> dict[str, list[int]]:
    """60/20/20 con redondeo de scikit-learn; reutilizar esta única asignación."""
    fractions = [TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION]
    if not all(value > 0 for value in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("Las fracciones deben ser positivas y sumar 1.")
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("La partición necesita ambas clases binarias.")
    positions = np.arange(len(y))
    train, remaining = train_test_split(
        positions,
        test_size=1 - TRAIN_FRACTION,
        random_state=random_state,
        stratify=y,
    )
    validation, test = train_test_split(
        remaining,
        test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
        random_state=random_state,
        stratify=y[remaining],
    )
    return {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()}


def make_pipeline(estimator=None, *, use_derived: bool = False, scale: bool = True) -> Pipeline:
    """Pipeline sin ajustar; la limpieza determinística de entrada precede a esta API."""
    numeric = list(NUMERIC_COLUMNS) + (list(DERIVED_COLUMNS) if use_derived else [])
    categorical = [name for name in FEATURE_COLUMNS if name not in NUMERIC_COLUMNS]
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    if estimator is None:
        estimator = LogisticRegression(random_state=RANDOM_STATE, max_iter=LOGISTIC_MAX_ITER)
    return Pipeline(
        [
            ("features", FeatureBuilder(enabled=use_derived)),
            ("preprocess", preprocessor),
            ("classifier", estimator),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preparar particiones de Chuard; no ejecuta selección ni entrenamiento."
    )
    parser.add_argument("--data", default=str(RAW_DATA_PATH))
    parser.add_argument("--output", default=str(SPLIT_MANIFEST_PATH))
    args = parser.parse_args()
    dataset_path, output_path = Path(args.data), Path(args.output)
    if not dataset_path.is_file():
        parser.error("Falta el CSV original. Consulta README.md.")
    data = read_customer_csv(dataset_path, mode="training").data
    if data["customerID"].duplicated().any():
        parser.error("Resolver IDs duplicados y documentar el criterio antes de dividir.")
    if output_path.exists():
        parser.error("El manifiesto ya existe; no se sobrescribe una partición congelada.")
    manifest = {
        "dataset_sha256": dataset_sha256(dataset_path),
        "random_state": RANDOM_STATE,
        "index_kind": "posición normalizada desde 0",
        "splits": split_positions(data["Churn"].map({"No": 0, "Yes": 1})),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Particiones guardadas; entrenamiento y selección siguen pendientes.")


if __name__ == "__main__":
    main()
