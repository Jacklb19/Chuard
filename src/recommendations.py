"""Reglas determinísticas de retención; sugerencias sin efecto causal demostrado.

No usan género ni edad para asignar beneficios. Ninguna acción se envía a clientes.
"""

import math
from numbers import Real

from src.config import (
    AUTOMATIC_PAYMENT_METHODS,
    MAX_TOP_FACTORS,
    NEW_CUSTOMER_MONTHS,
    POSITIVE_DIRECTION,
)

ANNUAL_CONTRACT = "Evaluar una oferta de contrato anual con beneficios."
PLAN_REVIEW = "Revisar alternativas de tarifa o paquete."
TECH_SUPPORT = "Ofrecer atención técnica y acompañamiento."
WELCOME = "Proponer bienvenida y seguimiento inicial."
PAYMENT_OPTIONS = "Sugerir alternativas de pago más cómodas."
PERSONAL_CONTACT = "Priorizar contacto personalizado."

REQUIRED_FIELDS = (
    "Contract",
    "MonthlyCharges",
    "InternetService",
    "TechSupport",
    "tenure",
    "PaymentMethod",
)
# Variable original o derivada cuya contribución positiva justifica cada regla.
RELEVANT_FEATURES = {
    ANNUAL_CONTRACT: {"Contract", "contrato_mensual"},
    PLAN_REVIEW: {"MonthlyCharges"},
    TECH_SUPPORT: {"TechSupport", "sin_soporte_tecnico"},
    PAYMENT_OPTIONS: {"PaymentMethod", "pago_automatico"},
}


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito.")
    return float(value)


def _positive_top_features(factors: list[dict]) -> set[str]:
    """Factores principales (en el orden recibido) que aumentan el riesgo."""
    if not isinstance(factors, list):
        raise ValueError("Los factores deben ser una lista.")
    relevant = set()
    for factor in factors[:MAX_TOP_FACTORS]:
        if not isinstance(factor, dict) or not isinstance(factor.get("feature"), str):
            raise ValueError("Cada factor requiere el nombre de su variable.")
        contribution = factor.get("contribution")
        consistent = contribution is None or _number(contribution, "contribution") > 0
        if factor.get("direction") == POSITIVE_DIRECTION and consistent:
            relevant.add(factor["feature"])
    return relevant


def _high_charges_limit(context: dict | None) -> float | None:
    """Límite aprendido en entrenamiento y entregado en el bundle; nunca del lote cargado."""
    if context is None:
        return None
    if not isinstance(context, dict):
        raise ValueError("El contexto de recomendaciones debe ser un diccionario.")
    limit = context.get("high_monthly_charges")
    return None if limit is None else _number(limit, "high_monthly_charges")


def recommend_retention(
    customer: dict, factors: list[dict], *, context: dict | None = None
) -> list[str]:
    """Contexto opcional: `high_monthly_charges` aprendido en entrenamiento."""
    if not isinstance(customer, dict):
        raise ValueError("El cliente debe ser un diccionario.")
    missing = [name for name in REQUIRED_FIELDS if name not in customer]
    if missing:
        raise ValueError(f"Faltan campos para recomendar: {', '.join(missing)}.")
    monthly = _number(customer["MonthlyCharges"], "MonthlyCharges")
    tenure = _number(customer["tenure"], "tenure")
    high_charges = _high_charges_limit(context)
    is_high_charge = high_charges is not None and monthly >= high_charges
    relevant = _positive_top_features(factors)

    def supported(action: str) -> bool:
        return bool(relevant & RELEVANT_FEATURES[action])

    candidates = [
        (ANNUAL_CONTRACT, customer["Contract"] == "Month-to-month" and supported(ANNUAL_CONTRACT)),
        (PLAN_REVIEW, is_high_charge and supported(PLAN_REVIEW)),
        (
            TECH_SUPPORT,
            customer["InternetService"] != "No"
            and customer["TechSupport"] == "No"
            and supported(TECH_SUPPORT),
        ),
        (WELCOME, tenure <= NEW_CUSTOMER_MONTHS),
        (
            PAYMENT_OPTIONS,
            customer["PaymentMethod"] not in AUTOMATIC_PAYMENT_METHODS
            and supported(PAYMENT_OPTIONS),
        ),
        # Exposición mensual: riesgo alto y cargo alto concentran el contacto humano.
        (PERSONAL_CONTACT, customer.get("risk_level") == "Alto" and is_high_charge),
    ]
    return list(dict.fromkeys(action for action, applies in candidates if applies))
