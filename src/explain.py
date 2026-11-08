"""Contrato de SHAP: implementación definitiva a cargo del frente 1."""

import pandas as pd


def explain_customer(customer_df: pd.DataFrame, bundle: dict) -> list[dict]:
    """Pendiente del ganador y de su escala/calibración; nunca inventa factores."""
    raise NotImplementedError(
        "Explicabilidad pendiente: implementar SHAP del ganador con el contrato acordado."
    )
