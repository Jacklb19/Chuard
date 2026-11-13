"""Reglas de retención con clientes y factores sintéticos; no miden efecto comercial."""

import math

import numpy as np
import pytest

from src.recommendations import (
    ANNUAL_CONTRACT,
    PAYMENT_OPTIONS,
    PERSONAL_CONTACT,
    PLAN_REVIEW,
    TECH_SUPPORT,
    WELCOME,
    recommend_retention,
)

CONTEXT = {"high_monthly_charges": 60}


def up(feature: str, contribution: float = 0.05) -> dict:
    return {"feature": feature, "direction": "aumenta", "contribution": contribution}


def down(feature: str, contribution: float = -0.05) -> dict:
    return {"feature": feature, "direction": "disminuye", "contribution": contribution}


@pytest.fixture
def veteran(customer):
    """Sintético: cliente antiguo, para aislar reglas distintas a bienvenida."""
    return {**customer, "tenure": 24}


def test_relevance_direction_and_context_are_respected(customer):
    factors = [up("Contract"), down("MonthlyCharges")]
    result = recommend_retention(customer, factors, context=CONTEXT)
    assert result == [ANNUAL_CONTRACT, WELCOME]


def test_personal_attributes_do_not_assign_different_benefits(customer):
    factors = [up(name) for name in ("Contract", "MonthlyCharges", "TechSupport")]
    first = recommend_retention(customer, factors, context=CONTEXT)
    for changes in ({"gender": "Male"}, {"SeniorCitizen": 1}, {"Partner": "No"}):
        assert recommend_retention({**customer, **changes}, factors, context=CONTEXT) == first


def test_all_rules_fire_once_in_stable_order(customer):
    factors = [up(name) for name in ("PaymentMethod", "TechSupport", "MonthlyCharges", "Contract")]
    factors += [up("contrato_mensual")]
    result = recommend_retention(
        {**customer, "risk_level": "Alto"}, factors + factors, context=CONTEXT
    )
    assert result == [
        ANNUAL_CONTRACT,
        PLAN_REVIEW,
        TECH_SUPPORT,
        WELCOME,
        PAYMENT_OPTIONS,
        PERSONAL_CONTACT,
    ]
    assert len(result) == len(set(result))


@pytest.mark.parametrize(
    ("feature", "action"),
    [
        ("contrato_mensual", ANNUAL_CONTRACT),
        ("sin_soporte_tecnico", TECH_SUPPORT),
        ("pago_automatico", PAYMENT_OPTIONS),
    ],
)
def test_derived_features_count_as_evidence(veteran, feature, action):
    assert recommend_retention(veteran, [up(feature)]) == [action]


def test_profile_must_be_compatible_even_with_relevant_factor(veteran):
    factors = [up("Contract"), up("TechSupport"), up("PaymentMethod"), up("MonthlyCharges")]
    compatible_none = {
        **veteran,
        "Contract": "Two year",
        "TechSupport": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 59.99,
    }
    assert recommend_retention(compatible_none, factors, context=CONTEXT) == []


def test_no_internet_never_receives_tech_support(veteran):
    customer = {**veteran, "InternetService": "No", "TechSupport": "No internet service"}
    assert TECH_SUPPORT not in recommend_retention(customer, [up("TechSupport")])
    customer = {**veteran, "InternetService": "No", "TechSupport": "No"}
    assert TECH_SUPPORT not in recommend_retention(customer, [up("sin_soporte_tecnico")])


def test_only_top_factors_are_considered(veteran):
    filler = [up(f"otro_{index}") for index in range(5)]
    assert recommend_retention(veteran, [*filler, up("Contract")]) == []
    assert recommend_retention(veteran, [*filler[:4], up("Contract")]) == [ANNUAL_CONTRACT]


def test_direction_must_match_positive_contribution(veteran):
    inconsistent = {"feature": "Contract", "direction": "aumenta", "contribution": -0.1}
    assert recommend_retention(veteran, [inconsistent]) == []
    assert recommend_retention(veteran, [{"feature": "Contract", "direction": "aumenta"}]) == [
        ANNUAL_CONTRACT
    ]


@pytest.mark.parametrize(("tenure", "welcome"), [(0, True), (6, True), (6.0, True), (7, False)])
def test_welcome_campaign_uses_tenure_limit(veteran, tenure, welcome):
    assert (WELCOME in recommend_retention({**veteran, "tenure": tenure}, [])) is welcome


def test_high_charges_require_context_and_limit_is_inclusive(veteran):
    factors = [up("MonthlyCharges")]
    assert recommend_retention(veteran, factors) == []
    assert recommend_retention({**veteran, "MonthlyCharges": 60}, factors, context=CONTEXT) == [
        PLAN_REVIEW
    ]
    assert recommend_retention({**veteran, "MonthlyCharges": 59.9}, factors, context=CONTEXT) == []


def test_personal_contact_requires_high_risk_and_high_charges(veteran):
    high_risk = {**veteran, "risk_level": "Alto"}
    assert recommend_retention(high_risk, [], context=CONTEXT) == [PERSONAL_CONTACT]
    assert recommend_retention({**veteran, "risk_level": "Medio"}, [], context=CONTEXT) == []
    assert recommend_retention(high_risk, []) == []


def test_numpy_scalars_from_scored_rows_are_accepted(veteran):
    row = {**veteran, "tenure": np.float64(3), "MonthlyCharges": np.float64(80)}
    assert recommend_retention(row, [up("MonthlyCharges")], context=CONTEXT) == [
        PLAN_REVIEW,
        WELCOME,
    ]


@pytest.mark.parametrize(
    ("customer_changes", "factors", "context", "message"),
    [
        ({"MonthlyCharges": "80"}, [], None, "MonthlyCharges"),
        ({"tenure": math.nan}, [], None, "tenure"),
        ({"tenure": True}, [], None, "tenure"),
        ({}, "Contract", None, "lista"),
        ({}, [{"direction": "aumenta"}], None, "variable"),
        ({}, [], {"high_monthly_charges": "alto"}, "high_monthly_charges"),
        ({}, [], {"high_monthly_charges": math.inf}, "high_monthly_charges"),
        ({}, [], [90], "diccionario"),
    ],
)
def test_invalid_inputs_fail_explicitly(veteran, customer_changes, factors, context, message):
    with pytest.raises(ValueError, match=message):
        recommend_retention({**veteran, **customer_changes}, factors, context=context)


def test_missing_required_field_is_named(veteran):
    incomplete = {key: value for key, value in veteran.items() if key != "Contract"}
    with pytest.raises(ValueError, match="Contract"):
        recommend_retention(incomplete, [])
