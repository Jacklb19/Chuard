"""Contrato de entrada y normalización sin aprendizaje estadístico."""

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

INTERNET_FIELDS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
YES_NO = ("Yes", "No")
CATEGORIES = {
    "gender": ("Male", "Female"),
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ("Yes", "No", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    **{name: ("Yes", "No", "No internet service") for name in INTERNET_FIELDS},
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": YES_NO,
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}
FEATURE_COLUMNS = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    *INTERNET_FIELDS,
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
)
NUMERIC_COLUMNS = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
DERIVED_COLUMNS = (
    "numero_servicios",
    "cliente_nuevo",
    "contrato_mensual",
    "pago_automatico",
    "sin_soporte_tecnico",
    "cargo_por_servicio",
)
OUTPUT_COLUMNS = (
    "customerID",
    "churn_probability",
    "predicted_churn",
    "risk_level",
    "priority_score",
    "top_factors",
    "recommendations",
)


@dataclass(frozen=True)
class ValidationIssue:
    """row es posición de datos desde 1; no es índice ni línea física del CSV."""

    row: int | None
    column: str
    reason: str


class InputValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        detail = "; ".join(
            f"fila {item.row or '-'}, {item.column}: {item.reason}" for item in issues
        )
        super().__init__(detail)

    def to_records(self) -> list[dict]:
        return [asdict(issue) for issue in self.issues]


@dataclass
class ValidatedCustomers:
    data: pd.DataFrame
    warnings: list[ValidationIssue]


def validate_customers(
    df: pd.DataFrame, *, mode: Literal["inference", "training"] = "inference"
) -> ValidatedCustomers:
    """Valida el lote completo; no retorna puntuaciones parciales ni altera df."""
    if mode not in ("training", "inference"):
        raise ValueError("Modo de validación desconocido.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se requiere un DataFrame de Pandas.")
    issues: list[ValidationIssue] = []
    for column in df.columns[df.columns.duplicated()].unique():
        issues.append(ValidationIssue(None, str(column), "Encabezado duplicado."))
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            issues.append(ValidationIssue(None, column, "Columna requerida ausente."))
    allowed = set(FEATURE_COLUMNS) | {"customerID", "Churn"}
    for column in df.columns:
        if column not in allowed:
            issues.append(ValidationIssue(None, str(column), "Columna no admitida."))
    if df.empty:
        issues.append(ValidationIssue(None, "*", "El lote no contiene registros."))
    if issues:
        raise InputValidationError(issues)
    # Índice técnico interno: posición estable incluso con índices/IDs repetidos.
    data = df.copy(deep=True).reset_index(drop=True)

    def add(mask: pd.Series, column: str, reason: str) -> None:
        for position in np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool)):
            issues.append(ValidationIssue(int(position) + 1, column, reason))

    for column, domain in CATEGORIES.items():
        add(~data[column].isin(domain), column, f"Valor requerido: {', '.join(domain)}.")
    for column in NUMERIC_COLUMNS:
        original = data[column]
        missing = original.isna() | original.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        numeric = pd.to_numeric(original.mask(missing), errors="coerce").astype(float)
        add(~missing & numeric.isna(), column, "Se requiere un número válido.")
        add(
            original.map(lambda value: isinstance(value, (bool, np.bool_))),
            column,
            "No se admiten booleanos como números.",
        )
        add(numeric.notna() & ~np.isfinite(numeric), column, "El valor debe ser finito.")
        add(numeric.lt(0), column, "No se admiten negativos.")
        data[column] = numeric
        if column in ("SeniorCitizen", "tenure"):
            finite = numeric.where(np.isfinite(numeric))
            add(finite.notna() & finite.mod(1).ne(0), column, "Se requiere un entero.")
        if column == "SeniorCitizen":
            add(~numeric.isin([0, 1]), column, "Solo se admite 0 o 1.")
        elif mode == "inference" and column != "TotalCharges":
            add(missing, column, "El campo es obligatorio.")

    total_missing = data["TotalCharges"].isna()
    # Solo se normalizan vacíos reales. Texto corrupto ya genera un error arriba.
    data.loc[total_missing & data["tenure"].eq(0), "TotalCharges"] = 0.0
    if mode == "inference":
        add(
            data["TotalCharges"].isna(), "TotalCharges", "Solo puede estar vacío cuando tenure = 0."
        )

    no_phone = data["PhoneService"].eq("No")
    add(
        no_phone & data["MultipleLines"].ne("No phone service"),
        "MultipleLines",
        "Sin teléfono, debe ser No phone service.",
    )
    add(
        data["PhoneService"].eq("Yes") & ~data["MultipleLines"].isin(YES_NO),
        "MultipleLines",
        "Con teléfono, debe ser Yes o No.",
    )
    for column in INTERNET_FIELDS:
        add(
            data["InternetService"].eq("No") & data[column].ne("No internet service"),
            column,
            "Sin internet, debe ser No internet service.",
        )
        add(
            data["InternetService"].isin(["DSL", "Fiber optic"]) & ~data[column].isin(YES_NO),
            column,
            "Con internet, debe ser Yes o No.",
        )
    if "Churn" in data:
        add(~data["Churn"].isin(YES_NO), "Churn", "Etiqueta histórica inválida.")
    elif mode == "training":
        issues.append(ValidationIssue(None, "Churn", "Se requiere para entrenamiento."))
    if issues:
        raise InputValidationError(issues)

    ids = (
        data["customerID"].astype("string").str.strip()
        if "customerID" in data
        else pd.Series(pd.NA, index=data.index, dtype="string")
    )
    reserved = set(ids.dropna()) - {""}
    for position in data.index[ids.isna() | ids.eq("")]:
        candidate = f"fila_{position + 1}"
        while candidate in reserved:
            candidate += "_temporal"
        ids.loc[position] = candidate
        reserved.add(candidate)
    data["customerID"] = ids
    warnings = [
        ValidationIssue(int(position) + 1, "customerID", "ID repetido; use posición técnica.")
        for position in data.index[ids.duplicated(keep=False)]
    ]
    return ValidatedCustomers(data=data, warnings=warnings)
