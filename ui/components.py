"""Shared headings, charts and model result presentation."""

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.schema import CATEGORY_LABELS, FEATURE_LABELS
from ui.theme import ORANGE, PINE


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.html(
        f'<div class="eyebrow">CHUARD / {escape(eyebrow.upper())}</div>'
        f"<h1>{escape(title)}</h1><p class='page-description'>{escape(description)}</p>"
    )


def metrics(items: list[tuple[str, str, str]]) -> None:
    with st.container(key="metric_strip"):
        for column, (label, value, note) in zip(st.columns(len(items)), items, strict=True):
            with column:
                st.html(
                    f'<div class="metric-label">{escape(label)}</div>'
                    f'<div class="metric-value">{escape(value)}</div>'
                    f'<div class="metric-note">{escape(note)}</div>'
                )


def bar_chart(labels, values, *, title: str, percent: bool = False) -> None:
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=PINE))
    fig.update_layout(
        height=max(240, 33 * len(labels) + 65),
        margin=dict(l=0, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#30423a"),
        yaxis=dict(autorange="reversed", automargin=True),
        xaxis=dict(tickformat=".0%" if percent else None, gridcolor="#e5e9e2"),
    )
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False}, alt=title)


def result_card(row: pd.Series, bundle: dict) -> None:
    probability = float(row["churn_probability"])
    risk = str(row["risk_level"])
    color = ORANGE if risk == "Alto" else "#896715" if risk == "Medio" else PINE
    st.html(
        '<div class="eyebrow">PROBABILIDAD ESTIMADA DE ABANDONO</div>'
        f'<div class="probability" style="color:{color}">{probability:.1%}</div>'
        f'<div class="risk-tag" style="color:{color}">Riesgo {escape(risk.lower())}</div>'
    )
    st.progress(probability)
    prediction = "Abandono" if row["predicted_churn"] else "Permanencia"
    st.write(f"**Decisión binaria:** {prediction} · umbral {bundle['decision_threshold']:.0%}")
    st.caption(
        f"Bandas operativas: bajo < {bundle['risk_limits']['medium']:.0%}; "
        f"medio < {bundle['risk_limits']['high']:.0%}; alto desde "
        f"{bundle['risk_limits']['high']:.0%}. No equivalen al umbral de decisión."
    )
    st.divider()
    st.subheader("Qué influyó en este resultado")
    for factor in row["top_factors"]:
        direction = "+" if factor["contribution"] >= 0 else ""
        st.write(f"**{FEATURE_LABELS[factor['feature']]}** · {factor['display_value']}")
        st.caption(f"{direction}{factor['contribution'] * 100:.2f} puntos de probabilidad")
    st.caption("SHAP describe asociaciones del modelo; no demuestra causas.")


def customer_table(frame: pd.DataFrame, *, key: str):
    columns = [
        "customerID",
        "tenure",
        "Contract",
        "MonthlyCharges",
        "churn_probability",
        "risk_level",
        "priority_score",
        "recommendations",
    ]
    display = frame.loc[:, columns].copy()
    display["recommendations"] = display["recommendations"].map(lambda items: " · ".join(items))
    display["Contract"] = display["Contract"].map(CATEGORY_LABELS)
    return st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=key,
        column_config={
            "customerID": st.column_config.TextColumn("Cliente"),
            "tenure": "Antigüedad (meses)",
            "Contract": "Contrato",
            "recommendations": st.column_config.TextColumn("Sugerencias", width="large"),
            "churn_probability": st.column_config.NumberColumn("P. abandono", format="percent"),
            "risk_level": "Riesgo",
            "MonthlyCharges": st.column_config.NumberColumn("Cargo mensual", format="%.2f"),
            "priority_score": st.column_config.NumberColumn("Exposición mensual", format="%.2f"),
        },
    )


def portfolio_charts(frame: pd.DataFrame) -> None:
    """All figures describe predictions in the currently filtered portfolio."""
    st.subheader("Cómo se distribuye el riesgo")
    st.caption("Predicciones de la selección actual; no son tasas históricas de abandono.")
    left, right = st.columns(2)
    with left:
        counts = frame.risk_level.value_counts().reindex(["Bajo", "Medio", "Alto"], fill_value=0)
        bar_chart(counts.index, counts.values, title="Número de clientes por banda de riesgo")
    with right:
        fig = go.Figure(
            go.Histogram(
                x=frame.churn_probability,
                xbins=dict(start=0, end=1, size=0.1),
                marker_color=PINE,
            )
        )
        fig.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(range=[0, 1], tickformat=".0%", title="Probabilidad estimada"),
            yaxis=dict(title="Clientes", dtick=1 if len(frame) < 10 else None),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#30423a"),
        )
        st.plotly_chart(fig, theme=None, width="stretch", alt="Distribución de probabilidades")
    segment = st.radio(
        "Probabilidad media por segmento",
        ["Contrato", "Internet", "Antigüedad"],
        horizontal=True,
    )
    if segment == "Antigüedad":
        labels = pd.cut(
            frame.tenure,
            bins=[0, 6, 12, 24, 48, float("inf")],
            include_lowest=True,
            labels=["0-6 meses", "7-12 meses", "13-24 meses", "25-48 meses", "49+ meses"],
        )
    else:
        labels = frame["Contract" if segment == "Contrato" else "InternetService"].map(
            CATEGORY_LABELS
        )
    grouped = frame.churn_probability.groupby(labels, observed=True).agg(["mean", "size"])
    bar_chart(
        [f"{label} · n={row['size']:.0f}" for label, row in grouped.iterrows()],
        grouped["mean"],
        title=f"Probabilidad media por {segment.lower()}",
        percent=True,
    )
