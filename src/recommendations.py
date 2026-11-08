"""Reglas determinísticas iniciales; no prueban efectos causales de retención."""

from src.config import (
    AUTOMATIC_PAYMENT_METHODS,
    NEW_CUSTOMER_MONTHS,
    POSITIVE_DIRECTION,
)


def recommend_retention(
    customer: dict, factors: list[dict], *, context: dict | None = None
) -> list[str]:
    """Contexto opcional para límites aprendidos y bandas del bundle."""
    context = context or {}
    relevant = {
        factor["feature"] for factor in factors if factor.get("direction") == POSITIVE_DIRECTION
    }
    recommendations = []
    if customer["Contract"] == "Month-to-month" and relevant & {"Contract", "contrato_mensual"}:
        recommendations.append("Evaluar una oferta de contrato anual con beneficios.")
    high_charges = context.get("high_monthly_charges")
    if (
        high_charges is not None
        and customer["MonthlyCharges"] >= high_charges
        and "MonthlyCharges" in relevant
    ):
        recommendations.append("Revisar alternativas de tarifa o paquete.")
    if (
        customer["InternetService"] != "No"
        and customer["TechSupport"] == "No"
        and relevant & {"TechSupport", "sin_soporte_tecnico"}
    ):
        recommendations.append("Ofrecer atención técnica y acompañamiento.")
    if customer["tenure"] <= NEW_CUSTOMER_MONTHS:
        recommendations.append("Proponer bienvenida y seguimiento inicial.")
    if customer["PaymentMethod"] not in AUTOMATIC_PAYMENT_METHODS and relevant & {
        "PaymentMethod",
        "pago_automatico",
    }:
        recommendations.append("Sugerir alternativas de pago más cómodas.")
    if (
        customer.get("risk_level") == "Alto"
        and high_charges is not None
        and customer["MonthlyCharges"] >= high_charges
    ):
        recommendations.append("Priorizar contacto personalizado.")
    return recommendations
