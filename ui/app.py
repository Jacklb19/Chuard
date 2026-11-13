"""Application composition; private customer records stay in the current session."""

import importlib.metadata

import streamlit as st

from src.application import load_application_model, sha256_file
from src.config import MODEL_MANIFEST_PATH, MODEL_PATH
from ui.theme import apply_theme

NAVIGATION = {
    "overview": ("Resumen", "dashboard"),
    "prediction": ("Predicción", "person_search"),
    "portfolio": ("Cartera CSV", "table_view"),
    "explanation": ("Explicabilidad", "analytics"),
    "retention": ("Retención", "handshake"),
}


@st.cache_resource(show_spinner="Verificando el modelo…")
def cached_model(model_digest: str, manifest_digest: str, environment: tuple) -> dict:
    """Only the public model is shared, never uploaded data or scored customers."""
    return load_application_model()


def main() -> None:
    st.set_page_config(page_title="Chuard · Análisis de abandono", page_icon="🌿", layout="wide")
    apply_theme()
    registered_pages = [
        st.Page(
            f"ui/views/{page_id}.py",
            title=label,
            icon=f":material/{icon}:",
            default=page_id == "overview",
        )
        for page_id, (label, icon) in NAVIGATION.items()
    ]
    navigation = st.navigation(registered_pages, position="hidden")
    with st.sidebar:
        st.html('<div class="brand">Chuard<span>ANÁLISIS DE ABANDONO</span></div>')
        for page in registered_pages:
            st.page_link(page, label=page.title, icon=page.icon, width="stretch")
        st.divider()
        st.caption("PROYECTO ACADÉMICO · TELCO")
        st.caption("Modelo 1.0.0 · XGBoost calibrado\n\n19 variables · sin reentrenamiento")
        st.caption(
            "Usa registros ficticios en la demostración pública. No cargues datos sensibles."
        )
        if st.button("Borrar datos de esta sesión", icon=":material/delete:"):
            for key in list(st.session_state):
                del st.session_state[key]
            st.rerun()
    navigation.run()
    st.divider()
    st.caption(
        "Chuard · Estimaciones académicas, sin horizonte temporal garantizado. "
        "Las recomendaciones necesitan revisión humana y no prueban efectos causales."
    )


def application_model() -> dict:
    """Return the verified public model or present an actionable startup error."""
    try:
        environment = tuple(
            (name, importlib.metadata.version(name))
            for name in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "shap", "joblib")
        )
        bundle = cached_model(
            sha256_file(MODEL_PATH), sha256_file(MODEL_MANIFEST_PATH), environment
        )
    except (OSError, ValueError, KeyError, importlib.metadata.PackageNotFoundError) as error:
        st.error(f"No se puede cargar el modelo de forma segura: {error}")
        st.info("Consulta las versiones y la instalación en el README del repositorio.")
        st.stop()
    return bundle
