"""Inferencia sin fit: valida, puntúa y explica con el bundle congelado del proyecto."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import CONTRACT_VERSION
from src.explain import explain_customer
from src.recommendations import recommend_retention
from src.schema import FEATURE_COLUMNS, validate_customers


def validate_model_bundle(bundle: dict) -> None:
    if not isinstance(bundle, dict):
        raise ValueError("El bundle debe ser un diccionario del proyecto.")
    required = {
        "pipeline",
        "feature_columns",
        "model_name",
        "model_version",
        "contract_version",
        "decision_threshold",
        "risk_limits",
    }
    if required - bundle.keys():
        raise ValueError(f"Bundle incompleto: {sorted(required - bundle.keys())}.")
    if bundle["contract_version"] != CONTRACT_VERSION:
        raise ValueError("Versión de contrato incompatible.")
    if list(bundle["feature_columns"]) != list(FEATURE_COLUMNS):
        raise ValueError("Columnas de entrada incompatibles.")
    if not callable(getattr(bundle["pipeline"], "predict_proba", None)):
        raise ValueError("El estimador no ofrece predict_proba.")
    try:
        threshold = float(bundle["decision_threshold"])
        medium = float(bundle["risk_limits"]["medium"])
        high = float(bundle["risk_limits"]["high"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Umbral o bandas inválidos.") from exc
    if not np.isfinite([threshold, medium, high]).all():
        raise ValueError("Umbral y bandas deben ser finitos.")
    if not (0 <= threshold <= 1 and 0 <= medium < high <= 1):
        raise ValueError("Umbral o bandas fuera de rango.")


def load_model_bundle(path: str) -> dict:
    """Solo archivos propios de procedencia verificada: Joblib puede ejecutar código."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError("Falta el modelo entrenado de Chuard; consulte README.md.")
    bundle = joblib.load(model_path)
    validate_model_bundle(bundle)
    return bundle


def score_customers(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    validate_model_bundle(bundle)
    # Rangos aprendidos solo en entrenamiento: advierten, nunca recortan entradas.
    validated = validate_customers(df, training_ranges=bundle.get("training_ranges"))
    data = validated.data
    estimator = bundle["pipeline"]
    classes = np.asarray(getattr(estimator, "classes_", []))
    positive = np.flatnonzero(classes == 1)
    if (
        classes.ndim != 1
        or len(classes) != 2
        or len(positive) != 1
        or not np.isin(classes, [0, 1]).all()
    ):
        raise ValueError("Se requiere estimador binario con clase positiva 1.")
    probabilities = np.asarray(estimator.predict_proba(data.loc[:, FEATURE_COLUMNS]), dtype=float)
    if (
        probabilities.shape != (len(data), len(classes))
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError("El modelo devolvió probabilidades inválidas.")
    probability = probabilities[:, positive[0]]
    output = data.copy()
    output["churn_probability"] = probability
    output["predicted_churn"] = (probability >= float(bundle["decision_threshold"])).astype(int)
    limits = bundle["risk_limits"]
    output["risk_level"] = np.select(
        [probability >= float(limits["high"]), probability >= float(limits["medium"])],
        ["Alto", "Medio"],
        default="Bajo",
    )
    output["priority_score"] = probability * output["MonthlyCharges"]
    if not np.isfinite(output["priority_score"]).all():
        raise ValueError("La exposición mensual no es finita.")
    factors, recommendations = [], []
    for position in range(len(data)):
        customer_factors = explain_customer(data.iloc[[position]], bundle)
        factors.append(customer_factors)
        recommendations.append(
            recommend_retention(
                output.iloc[position].to_dict(),
                customer_factors,
                context=bundle.get("recommendation_context"),
            )
        )
    output["top_factors"] = factors
    output["recommendations"] = recommendations
    output.attrs["validation_warnings"] = validated.warnings
    return output
