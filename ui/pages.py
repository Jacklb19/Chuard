"""Five working views backed by the frozen scientific pipeline."""

import hashlib
import io
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.application import (
    EXAMPLE_PATH,
    UI_BATCH_LIMIT,
    export_customers,
    filter_customers,
    frame_fingerprint,
)
from src.config import PROJECT_ROOT
from src.data import read_customer_csv
from src.explain import get_explanation_service
from src.inference import score_customers
from src.schema import (
    CATEGORIES,
    CATEGORY_LABELS,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    INTERNET_FIELDS,
    InputValidationError,
)
from ui.components import (
    bar_chart,
    customer_table,
    metrics,
    page_header,
    portfolio_charts,
    result_card,
)
from ui.theme import ORANGE, PINE


def read_report(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / relative_path)


def overview(bundle: dict) -> None:
    page_header(
        "Resumen",
        "Entender el abandono.",
        "Resultados históricos, evidencia y contexto del modelo publicado.",
    )
    test = read_report("reports/evaluation/prueba_final.csv").iloc[0]
    metrics(
        [
            ("Clientes del dataset", "7.043", "4.225 / 1.409 / 1.409 por partición"),
            ("F1 · prueba final", f"{test['f1']:.3f}", "Clase positiva: abandono"),
            ("Recall · prueba final", f"{test['recall']:.1%}", "300 de 374 abandonos detectados"),
            (
                "ROC-AUC · prueba final",
                f"{test['roc_auc']:.3f}",
                "Evaluación guardada · 09 nov 2026",
            ),
        ]
    )
    left, right = st.columns([1.4, 1], gap="large")
    with left, st.container(border=True):
        st.subheader("Abandono según el contrato")
        st.caption("Solo entrenamiento · 4.225 clientes. Asociación histórica, no causalidad.")
        segments = read_report("reports/data/segments_Contract.csv")
        bar_chart(
            segments.Contract.map(CATEGORY_LABELS),
            segments.churn_rate,
            title="Tasa de abandono por contrato",
            percent=True,
        )
    with right, st.container(border=True):
        st.subheader("Un modelo, un protocolo")
        st.write("**XGBoost · sin variables derivadas**")
        st.write("Calibración sigmoid con cinco folds, aprendida solo en entrenamiento.")
        st.write(f"Umbral binario **{bundle['decision_threshold']:.0%}**, elegido en validación.")
        st.caption(
            "El ganador se decidió por F1, ROC-AUC, recall, estabilidad y parsimonia, con "
            "tolerancia de empate 0,005."
        )
        st.write(
            "**Prueba final reservada:** consultamos su reporte guardado; la web no "
            "vuelve a evaluarla."
        )
    st.subheader("Comparación de candidatos")
    st.caption(
        "Validación · cinco familias, con y sin seis variables derivadas. No son "
        "métricas de prueba final."
    )
    candidates = read_report("reports/evaluation/validacion_candidatos.csv")
    table = candidates[
        ["candidate", "accuracy", "precision", "recall", "f1", "roc_auc", "decision_threshold"]
    ].rename(
        columns={
            "candidate": "Candidato",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "roc_auc": "ROC-AUC",
            "decision_threshold": "Umbral",
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    with st.expander("Ver errores y resultados de la prueba final"):
        st.image(
            str(PROJECT_ROOT / "reports/evaluation/matriz_prueba_final_guardada.png"),
            width="stretch",
        )
        st.write(
            "745 permanencias y 300 abandonos acertados; 290 falsas alertas y 74 "
            "abandonos no detectados."
        )
        st.write(
            f"Accuracy {test['accuracy']:.3f} · Precision {test['precision']:.3f} · "
            f"Brier {test['brier']:.3f} · Log-loss {test['log_loss']:.3f}."
        )
        st.caption(
            "No se reconstruye una ROC de prueba: las probabilidades originales de ese "
            "conjunto no están disponibles."
        )


def show_warnings(scored: pd.DataFrame) -> None:
    warnings = scored.attrs.get("validation_warnings", [])
    if warnings:
        with st.expander(f"Advertencias de validación ({len(warnings)})", expanded=True):
            st.dataframe(pd.DataFrame([vars(item) for item in warnings]), hide_index=True)


def prediction(bundle: dict) -> None:
    page_header(
        "Predicción",
        "Un cliente, con contexto.",
        "Completa sus datos y consulta la estimación y los factores que la explican.",
    )
    default = pd.read_csv(EXAMPLE_PATH).iloc[0].to_dict()
    left, right = st.columns([1.35, 1], gap="large")
    values = {}
    with left, st.container(border=True):
        st.subheader("Perfil del cliente")
        values["customerID"] = st.text_input(
            "Identificador (opcional)", value="DEMO-INDIVIDUAL", key="individual_id"
        )
        groups = {
            "Contrato y facturación": (
                "tenure",
                "Contract",
                "MonthlyCharges",
                "TotalCharges",
                "PaymentMethod",
                "PaperlessBilling",
            ),
            "Servicios contratados": (
                "PhoneService",
                "MultipleLines",
                "InternetService",
                *INTERNET_FIELDS,
            ),
            "Datos del perfil": ("gender", "SeniorCitizen", "Partner", "Dependents"),
        }
        for heading, fields in groups.items():
            st.markdown(f"**{heading}**")
            columns = st.columns(2)
            for position, field in enumerate(fields):
                with columns[position % 2]:
                    key = f"individual_{field}"
                    if field in CATEGORIES:
                        options = list(CATEGORIES[field])
                        if field == "MultipleLines":
                            options = (
                                ["No phone service"]
                                if values["PhoneService"] == "No"
                                else ["Yes", "No"]
                            )
                        if field in INTERNET_FIELDS:
                            options = (
                                ["No internet service"]
                                if values["InternetService"] == "No"
                                else ["Yes", "No"]
                            )
                        if st.session_state.get(key) not in options:
                            st.session_state[key] = (
                                default[field] if default[field] in options else options[0]
                            )
                        values[field] = st.selectbox(
                            FEATURE_LABELS[field],
                            options,
                            format_func=lambda value: CATEGORY_LABELS[value],
                            key=key,
                        )
                    elif field == "SeniorCitizen":
                        values[field] = st.selectbox(
                            FEATURE_LABELS[field],
                            [0, 1],
                            format_func=lambda value: "Sí" if value else "No",
                            key=key,
                        )
                    else:
                        integer = field == "tenure"
                        values[field] = st.number_input(
                            FEATURE_LABELS[field],
                            min_value=0 if integer else 0.0,
                            value=int(default[field]) if integer else float(default[field]),
                            step=1 if integer else 0.5,
                            key=key,
                        )
        frame = pd.DataFrame([values])
        fingerprint = frame_fingerprint(frame)
        if st.session_state.get("individual_fingerprint") != fingerprint:
            st.session_state.pop("individual_result", None)
        if st.button(
            "Calcular predicción", type="primary", icon=":material/arrow_forward:", width="stretch"
        ):
            try:
                with st.spinner("Calculando probabilidad y explicación SHAP real…"):
                    scored = score_customers(frame, bundle)
                st.session_state.individual_result = scored
                st.session_state.individual_fingerprint = fingerprint
            except InputValidationError as error:
                st.error("Revisa los datos antes de continuar.")
                st.dataframe(pd.DataFrame(error.to_records()), hide_index=True)
            except (ValueError, RuntimeError) as error:
                st.error(f"No se pudo calcular la predicción: {error}")
    with right, st.container(border=True):
        if "individual_result" in st.session_state:
            scored = st.session_state.individual_result
            result_card(scored.iloc[0], bundle)
            show_warnings(scored)
            with st.expander("Sugerencias para revisión humana"):
                for recommendation in scored.iloc[0]["recommendations"]:
                    st.write(f"• {recommendation}")
            st.download_button(
                "Descargar resultado CSV",
                export_customers(scored),
                "chuard_cliente.csv",
                "text/csv",
                on_click="ignore",
            )
        else:
            st.subheader("El resultado aparecerá aquí")
            st.write(
                "Calcularemos la probabilidad, la banda de riesgo y los cinco factores con "
                "mayor contribución absoluta."
            )
            st.caption(
                "Los valores iniciales son ficticios. El modelo y las explicaciones son reales."
            )


def replace_source(payload: bytes, source_name: str) -> None:
    digest = hashlib.sha256(payload).hexdigest()
    if st.session_state.get("source_digest") != digest:
        for key in ("scored", "source", "selected_row", "local_explanation", "individual_result"):
            st.session_state.pop(key, None)
        st.session_state.source_digest = digest
        st.session_state.source_name = source_name
        st.session_state.pop("source_error", None)
        try:
            st.session_state.source = read_customer_csv(
                io.BytesIO(payload), max_rows=UI_BATCH_LIMIT
            ).data
        except (InputValidationError, ValueError, UnicodeError) as error:
            st.session_state.source_error = error


def upload_changed() -> None:
    """An explicit removal clears results; navigating between pages does not."""
    upload = st.session_state.get("customer_upload")
    if upload is None:
        for key in (
            "source",
            "scored",
            "source_digest",
            "source_error",
            "selected_row",
            "local_explanation",
            "last_upload_digest",
        ):
            st.session_state.pop(key, None)


def portfolio(bundle: dict) -> None:
    page_header(
        "Cartera CSV",
        "Una cartera, prioridades claras.",
        "Carga registros, revisa los errores y calcula resultados con explicaciones "
        "para cada cliente.",
    )
    with st.container(border=True):
        st.caption(
            f"UTF-8, separado por comas · hasta {UI_BATCH_LIMIT} registros · "
            "máximo 10 MB · 19 variables obligatorias."
        )
        upload = st.file_uploader(
            "Selecciona un archivo CSV",
            type="csv",
            key="customer_upload",
            on_change=upload_changed,
        )
        if upload is not None:
            digest = hashlib.sha256(upload.getvalue()).hexdigest()
            if st.session_state.get("last_upload_digest") != digest:
                replace_source(upload.getvalue(), upload.name)
                st.session_state.last_upload_digest = digest
        actions = st.columns(2)
        with actions[0]:
            if st.button("Probar con datos ficticios", icon=":material/science:", width="stretch"):
                replace_source(EXAMPLE_PATH.read_bytes(), "Ejemplo ficticio")
        with actions[1]:
            st.download_button(
                "Descargar CSV de ejemplo",
                EXAMPLE_PATH.read_bytes(),
                "chuard_ejemplo_ficticio.csv",
                "text/csv",
                on_click="ignore",
                width="stretch",
            )
        if "source_error" in st.session_state:
            error = st.session_state.source_error
            st.error(f"Se rechazó el archivo completo: {error}")
            if isinstance(error, InputValidationError):
                st.dataframe(pd.DataFrame(error.to_records()), hide_index=True)
        if "source" in st.session_state:
            st.caption(
                f"Fuente actual: {st.session_state.source_name} · "
                f"{len(st.session_state.source)} registros válidos."
            )
            if st.button("Analizar cartera", type="primary", width="stretch"):
                st.session_state.pop("scored", None)
                try:
                    with st.spinner(
                        "Calculando probabilidad, SHAP y recomendaciones de cada registro…"
                    ):
                        st.session_state.scored = score_customers(st.session_state.source, bundle)
                except (ValueError, RuntimeError) as error:
                    st.error(f"No se pudo completar el lote: {error}")
    if "scored" not in st.session_state:
        st.info("Carga tu CSV o usa el ejemplo ficticio y pulsa Analizar cartera.")
        return
    scored = st.session_state.scored
    show_warnings(scored)
    metrics(
        [
            ("Clientes analizados", str(len(scored)), "Resultados de esta sesión"),
            (
                "Proporción en riesgo alto",
                f"{scored.risk_level.eq('Alto').mean():.1%}",
                f"{scored.risk_level.eq('Alto').sum()} clientes · "
                f"desde {bundle['risk_limits']['high']:.0%}",
            ),
            (
                "Probabilidad media",
                f"{scored.churn_probability.mean():.1%}",
                "Estimación, no tasa observada",
            ),
            (
                "Exposición mensual",
                f"{scored.priority_score.sum():,.2f}",
                "Σ probabilidad × cargo; no pérdida real",
            ),
        ]
    )
    view = filtered_view(scored, "portfolio")
    if view.empty:
        st.info("No hay clientes que coincidan con esos filtros.")
        return
    portfolio_charts(view)
    st.subheader("Detalle de clientes")
    selection = customer_table(
        view,
        key=f"portfolio_table_{frame_fingerprint(view[['customerID']])}_{st.session_state.source_digest}",
    )
    if selection.selection.rows:
        st.session_state.selected_row = int(view.index[selection.selection.rows[0]])
    st.download_button(
        "Exportar resultados filtrados",
        export_customers(view),
        "chuard_cartera.csv",
        "text/csv",
        on_click="ignore",
    )
    st.caption(
        "Selecciona una fila para usarla en Explicabilidad. Los identificadores "
        "repetidos conservan su posición original."
    )


def filtered_view(scored: pd.DataFrame, key: str, *, active_only: bool = False) -> pd.DataFrame:
    left, right = st.columns([2, 1])
    search = left.text_input("Buscar identificador", key=f"{key}_search")
    risk = right.selectbox("Banda de riesgo", ["Todos", "Alto", "Medio", "Bajo"], key=f"{key}_risk")
    left, right = st.columns(2)
    contract = left.selectbox(
        "Contrato",
        ["Todos", *CATEGORIES["Contract"]],
        format_func=lambda value: CATEGORY_LABELS.get(value, value),
        key=f"{key}_contract",
    )
    internet = right.selectbox(
        "Servicio de internet",
        ["Todos", *CATEGORIES["InternetService"]],
        format_func=lambda value: CATEGORY_LABELS.get(value, value),
        key=f"{key}_internet",
    )
    return filter_customers(
        scored,
        search=search,
        risk=risk,
        active_only=active_only,
        contract=contract,
        internet=internet,
    )


def explanation(bundle: dict) -> None:
    page_header(
        "Explicabilidad",
        "De la probabilidad a sus factores.",
        "Contribuciones SHAP sobre la probabilidad calibrada del modelo completo.",
    )
    mode = st.radio("Vista de explicación", ["Global", "Cliente"], horizontal=True)
    if mode == "Global":
        st.caption(
            "100 registros de entrenamiento · referencia fija de entrenamiento · no es "
            "una explicación de la cartera cargada."
        )
        st.image(str(PROJECT_ROOT / "reports/explain/shap_global.png"), width="stretch")
        st.write(
            "Cada punto corresponde a un registro. La posición muestra cuánto aumenta o "
            "disminuye la probabilidad respecto a la referencia; el color representa el "
            "valor de la variable."
        )
        with st.expander("Método y límites"):
            metadata = json.loads(
                (PROJECT_ROOT / "reports/explain/explanation_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            st.json(metadata)
            st.write(
                "Permutation SHAP explica el pipeline calibrado completo. Las contribuciones "
                "son asociaciones y dependen de la referencia. No son efectos causales ni "
                "recomendaciones automáticas de trato al cliente."
            )
        return
    if "scored" not in st.session_state:
        st.info(
            "Analiza primero una cartera en Cartera CSV. Para un solo cliente, Predicción "
            "ya muestra sus factores."
        )
        return
    scored = st.session_state.scored
    row_ids = list(scored.index)
    initial = st.session_state.get("selected_row", row_ids[0])
    selected = st.selectbox(
        "Cliente a explicar",
        row_ids,
        index=row_ids.index(initial) if initial in row_ids else 0,
        format_func=lambda index: f"{scored.loc[index, 'customerID']} · fila {index + 1}",
        key=f"explain_customer_{st.session_state.source_digest}_{initial}",
    )
    row = scored.loc[selected]
    cache_key = (st.session_state.source_digest, selected)
    cached = st.session_state.get("local_explanation")
    if cached is None or cached[0] != cache_key:
        with st.spinner("Calculando la explicación completa de este cliente…"):
            result = get_explanation_service(bundle).explain(
                scored.loc[[selected], list(FEATURE_COLUMNS)]
            )
        st.session_state.local_explanation = (cache_key, result)
    else:
        result = cached[1]
    left, right = st.columns([1.5, 1])
    with left, st.container(border=True):
        st.subheader("Contribuciones individuales")
        values = result.values.iloc[0].sort_values()
        fig = go.Figure(
            go.Bar(
                x=values * 100,
                y=[FEATURE_LABELS[field] for field in values.index],
                orientation="h",
                marker_color=[ORANGE if value > 0 else PINE for value in values],
            )
        )
        fig.update_layout(
            height=620,
            margin=dict(l=0, r=10, t=10, b=35),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Contribución (puntos de probabilidad)",
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(
            fig, theme=None, width="stretch", alt="Las 19 contribuciones SHAP del cliente"
        )
        st.caption(
            f"Referencia {result.base_values[0]:.4f} + contribuciones "
            f"{result.values.iloc[0].sum():+.4f} = {result.explained_values[0]:.4f}. "
            "Tolerancia de reconstrucción 0,000001."
        )
    with right, st.container(border=True):
        result_card(row, bundle)


def retention(bundle: dict) -> None:
    page_header(
        "Retención",
        "Decidir a quién revisar primero.",
        "Una lista de trabajo basada en exposición mensual y factores del modelo.",
    )
    st.info(
        "Simulación académica: las sugerencias no garantizan retención ni se envían a los clientes."
    )
    if "scored" not in st.session_state:
        st.write("Analiza una cartera en Cartera CSV para crear la lista de revisión.")
        return
    scored = st.session_state.scored
    if "Churn" in scored:
        st.caption(
            "Se incluyen solo registros Churn=No. Los abandonos ya observados se excluyen "
            "de esta simulación."
        )
    else:
        st.warning(
            "El CSV no incluye Churn: se presupone que sus clientes están activos. "
            "Confírmalo antes de actuar."
        )
    view = filtered_view(scored, "retention", active_only=True).sort_values(
        "priority_score", ascending=False, kind="stable"
    )
    st.caption(
        "Prioridad = probabilidad × cargo mensual. Es exposición estimada en unidades "
        "del dataset, sin moneda confirmada; no es pérdida demostrada."
    )
    if view.empty:
        st.info("No hay clientes elegibles con estos filtros.")
        return
    selection = customer_table(
        view,
        key=f"retention_table_{frame_fingerprint(view[['customerID']])}_{st.session_state.source_digest}",
    )
    selected_position = selection.selection.rows[0] if selection.selection.rows else 0
    row = view.iloc[selected_position]
    with st.container(border=True):
        st.subheader(
            f"Sugerencias para {row['customerID']} · fila {view.index[selected_position] + 1}"
        )
        for recommendation in row["recommendations"]:
            st.write(f"• {recommendation}")
        st.caption(
            "Revisar viabilidad, consentimiento, costos y equidad antes de una acción "
            "comercial. Género y edad no se usan para asignar beneficios."
        )
    st.download_button(
        "Exportar lista priorizada",
        export_customers(view),
        "chuard_retencion.csv",
        "text/csv",
        on_click="ignore",
    )
