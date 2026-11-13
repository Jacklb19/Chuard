"""Exercise the real Streamlit application with the approved frozen model."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.application import EXAMPLE_PATH
from src.schema import InputValidationError


def click_label(app, label):
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception


@pytest.fixture
def app():
    instance = AppTest.from_file(str(Path("app.py").resolve()), default_timeout=90).run()
    assert not instance.exception
    return instance


def test_individual_prediction_is_invalidated_on_edit_and_services_stay_coherent(app):
    app.switch_page("ui/views/prediction.py").run()
    click_label(app, "Calcular predicción")
    result = app.session_state["individual_result"].iloc[0]
    assert 0 <= result.churn_probability <= 1
    assert len(result.top_factors) == 5
    app.number_input(key="individual_MonthlyCharges").set_value(101.0).run()
    assert "individual_result" not in app.session_state
    app.selectbox(key="individual_InternetService").select("No").run()
    assert app.selectbox(key="individual_TechSupport").value == "No internet service"
    app.selectbox(key="individual_PhoneService").select("No").run()
    assert app.selectbox(key="individual_MultipleLines").value == "No phone service"
    assert not app.exception


def test_portfolio_explanation_retention_navigation_and_session_clear(app):
    app.switch_page("ui/views/portfolio.py").run()
    click_label(app, "Probar con datos ficticios")
    click_label(app, "Analizar cartera")
    scored = app.session_state["scored"]
    assert len(scored) == len(pd.read_csv(EXAMPLE_PATH))
    assert all(len(factors) == 5 for factors in scored.top_factors)
    for segment in ("Internet", "Antigüedad", "Contrato"):
        app.radio[0].set_value(segment).run()
        assert not app.exception
    app.selectbox(key="portfolio_contract").select("Two year").run()
    assert app.dataframe[0].value.Contract.eq("Dos años").all()
    app.selectbox(key="portfolio_contract").select("Todos").run()
    app.switch_page("ui/views/explanation.py").run()
    app.radio[0].set_value("Cliente").run()
    assert not app.exception
    explanation = app.session_state["local_explanation"][1]
    assert abs(explanation.explained_values[0] - scored.churn_probability.iloc[0]) < 1e-6
    app.switch_page("ui/views/retention.py").run()
    assert not app.exception
    displayed = app.dataframe[0].value
    assert len(displayed) == scored.Churn.eq("No").sum()
    assert displayed.priority_score.is_monotonic_decreasing
    app.switch_page("ui/views/portfolio.py").run()
    assert len(app.session_state["scored"]) == len(scored)
    click_label(app, "Borrar datos de esta sesión")
    assert "scored" not in app.session_state
    assert "local_explanation" not in app.session_state


def test_invalid_new_source_removes_old_results_before_parsing(app):
    # Run the source transition inside a real Streamlit script context.
    transition = AppTest.from_string(
        "from ui.pages import replace_source\n"
        "import streamlit as st\n"
        "st.session_state.scored = 'previous customer results'\n"
        "st.session_state.selected_row = 5\n"
        "replace_source(b'broken,csv\\n1,2\\n', 'invalid.csv')\n"
    ).run()
    assert not transition.exception
    assert "scored" not in transition.session_state
    assert "selected_row" not in transition.session_state
    assert isinstance(transition.session_state["source_error"], InputValidationError)
