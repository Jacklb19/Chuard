"""Características compartidas, sin ajuste de estadísticas ni acceso al objetivo."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.config import AUTOMATIC_PAYMENT_METHODS, NEW_CUSTOMER_MONTHS
from src.schema import DERIVED_COLUMNS, FEATURE_COLUMNS, INTERNET_FIELDS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe las entradas normalizadas; retorna originales y seis derivadas."""
    _check_columns(df)
    result = df.loc[:, list(FEATURE_COLUMNS)].copy()
    count_columns = ["PhoneService", "MultipleLines", *INTERNET_FIELDS]
    result["numero_servicios"] = result[count_columns].eq("Yes").sum(axis=1) + result[
        "InternetService"
    ].ne("No").astype(int)
    result["cliente_nuevo"] = np.where(
        result["tenure"].isna(), np.nan, result["tenure"].le(NEW_CUSTOMER_MONTHS).astype(int)
    )
    result["contrato_mensual"] = result["Contract"].eq("Month-to-month").astype(int)
    result["pago_automatico"] = result["PaymentMethod"].isin(AUTOMATIC_PAYMENT_METHODS).astype(int)
    result["sin_soporte_tecnico"] = (
        result["InternetService"].ne("No") & result["TechSupport"].eq("No")
    ).astype(int)
    result["cargo_por_servicio"] = result["MonthlyCharges"] / result["numero_servicios"].clip(
        lower=1
    )
    return result


def _check_columns(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Las características requieren un DataFrame normalizado.")
    if df.columns.duplicated().any():
        raise ValueError("No se admiten columnas duplicadas en las características.")
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan predictores: {', '.join(sorted(missing))}.")


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Transformador importable y clonable; activación fijada por experimento."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):
        _check_columns(X)
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled debe ser booleano.")
        self.feature_names_in_ = np.array(FEATURE_COLUMNS, dtype=object)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        _check_columns(X)
        if self.enabled:
            return build_features(X)
        return X.loc[:, list(FEATURE_COLUMNS)].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("Los nombres de entrada no coinciden con las 19 columnas originales.")
        return np.array(
            (*FEATURE_COLUMNS, *DERIVED_COLUMNS) if self.enabled else FEATURE_COLUMNS,
            dtype=object,
        )
