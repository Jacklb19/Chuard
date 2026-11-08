from src.recommendations import recommend_retention


def test_relevance_direction_and_context_are_respected(customer):
    factors = [
        {"feature": "Contract", "direction": "aumenta"},
        {"feature": "MonthlyCharges", "direction": "disminuye"},
    ]
    result = recommend_retention(customer, factors, context={"high_monthly_charges": 60})
    assert len(result) == 2
    assert any("contrato" in item for item in result)
    assert any("bienvenida" in item for item in result)
    assert not any("tarifa" in item for item in result)


def test_personal_attributes_do_not_assign_different_benefits(customer):
    first = recommend_retention(customer, [])
    changed = {**customer, "gender": "Male", "SeniorCitizen": 1}
    assert recommend_retention(changed, []) == first
