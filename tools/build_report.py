"""Build the academic PDF from saved evidence; never train or score test data."""

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
PINE = colors.HexColor("#25664f")
INK = colors.HexColor("#1c302a")
MUTED = colors.HexColor("#65736b")
WIDTH = A4[0] - 100


def records(relative: str) -> list[dict]:
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build_report(destination: Path, application_url: str | None) -> None:
    deployment_path = ROOT / "reports/application/deployment.json"
    if application_url is None and deployment_path.is_file():
        deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
        if deployment.get("status") == "verified":
            application_url = deployment["application_url"]
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "TitleChuard",
            fontName="Times-Roman",
            fontSize=42,
            leading=46,
            textColor=INK,
            spaceAfter=22,
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionChuard",
            fontName="Times-Roman",
            fontSize=25,
            leading=29,
            textColor=INK,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodyChuard",
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            textColor=INK,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "SmallChuard",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=MUTED,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "CellChuard",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=INK,
            alignment=TA_LEFT,
        )
    )
    story = []

    def paragraph(text: str, style: str = "BodyChuard") -> None:
        story.append(Paragraph(text, styles[style]))

    def section(number: str, title: str) -> None:
        if story:
            story.append(PageBreak())
        paragraph(f"CHUARD / INFORME FINAL / {number}", "SmallChuard")
        paragraph(title, "SectionChuard")

    def image(relative: str, max_height: float = 300) -> None:
        path = ROOT / relative
        width, height = ImageReader(str(path)).getSize()
        scale = min(WIDTH / width, max_height / height)
        story.append(Image(str(path), width=width * scale, height=height * scale))
        story.append(Spacer(1, 10))

    def table(rows: list[list], widths: list[float]) -> None:
        formatted = [
            [Paragraph(escape(str(cell)), styles["CellChuard"]) for cell in row] for row in rows
        ]
        result = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f2eb")),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, PINE),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#dce2da")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.extend([result, Spacer(1, 14)])

    manifest = json.loads((ROOT / "models/best_model.manifest.json").read_text("utf-8"))
    test = records("reports/evaluation/prueba_final.csv")[0]
    candidates = records("reports/evaluation/validacion_candidatos.csv")
    section("01", "Predicción y análisis de abandono")
    paragraph("Chuard", "TitleChuard")
    paragraph("Proyecto final · Minería de Datos · 13 de noviembre de 2026", "SmallChuard")
    paragraph(
        "<b>Objetivo.</b> Estimar el abandono de clientes de telecomunicaciones, "
        "comparar cinco familias de clasificación y ofrecer explicaciones y una lista "
        "de revisión para retención mediante una aplicación web."
    )
    paragraph(
        "<b>Resultado.</b> Se publican los datos de procedencia y calidad, tres notebooks "
        "con sus ejecuciones, diez candidatos evaluados, el pipeline congelado XGBoost "
        "sin derivadas, explicabilidad SHAP, recomendaciones determinísticas y una web "
        "Streamlit con cinco secciones. La aplicación utiliza el modelo; no lo entrena."
    )
    table(
        [
            ["Evidencia", "Resultado guardado"],
            [
                "Prueba final: 1.409 registros",
                f"F1 {float(test['f1']):.3f} · "
                f"Recall {float(test['recall']):.3f} · ROC-AUC {float(test['roc_auc']):.3f}",
            ],
            ["Ganador versión 1.0.0", "XGBoost calibrado sigmoid · umbral binario 0,25"],
            [
                "Fuentes de revisión",
                "README, notebooks, pruebas, manifiestos y resultados versionados",
            ],
        ],
        [180, WIDTH - 180],
    )
    paragraph(
        "<b>Interpretación.</b> El recall favorece detectar abandonos, con 290 falsas "
        "alertas frente a 300 aciertos positivos. El sistema apoya revisión humana; "
        "no garantiza abandono futuro, ahorros ni efectividad causal de las sugerencias."
    )
    paragraph(
        'Repositorio: <link href="https://github.com/Jacklb19/Chuard" color="#25664f">'
        "github.com/Jacklb19/Chuard</link>"
    )
    paragraph(
        "Aplicación: "
        + (
            f'<link href={quoteattr(application_url)} color="#25664f">'
            f"{escape(application_url)}</link>"
            if application_url
            else "implementación local verificada; publicación en la cuenta existente de "
            "Streamlit Community Cloud pendiente de acceso. Consultar el estado en README."
        )
    )

    section("02", "Datos, limpieza y exploración")
    paragraph(
        "Telco Customer Churn (Kaggle): 7.043 filas, 21 columnas, 19 predictores "
        "originales, customerID y Churn. La clase positiva es Churn=Yes: "
        "1.869 registros (26,54%); 5.174 no abandonos. customerID y Churn nunca "
        "entran como predictores."
    )
    paragraph(
        "La revisión detectó 11 cargos acumulados vacíos, todos con antigüedad cero; "
        "se normalizan a cero. No hay filas completas ni identificadores duplicados "
        "en el original. Se conservan perfiles iguales con identificadores diferentes. "
        "Se validan categorías, números finitos no negativos y coherencia entre "
        "telefonía, internet y servicios asociados."
    )
    paragraph(
        "El EDA estadístico se limita a entrenamiento (4.225 registros). El contrato "
        "mensual presenta 43,06% de abandono, frente a 11,05% en un año y 2,93% "
        "en dos años. Estas diferencias son asociaciones de la muestra."
    )
    image("reports/data/02_abandono_Contract.png", 255)
    paragraph(
        "La limpieza y las nueve figuras del EDA están en el notebook 01 y reports/data. "
        "Los rangos observados generan advertencias en inferencia; no recortan valores."
    )
    paragraph(
        "El CSV crudo se obtiene desde Kaggle mediante el comando documentado. "
        "No se redistribuye en el repositorio; el ejemplo de la web contiene solo "
        "seis perfiles ficticios, identificados con DEMO-.",
        "SmallChuard",
    )

    section("03", "Protocolo experimental y prevención de fuga")
    table(
        [
            ["Etapa", "Regla registrada"],
            [
                "Partición",
                "60/20/20 estratificada; semilla 42. Entrenamiento 4.225; "
                "validación 1.409; prueba final 1.409.",
            ],
            [
                "Transformación",
                "Imputación, codificación y escalado según la familia dentro "
                "del pipeline. Sus estadísticas se aprenden solo en cada entrenamiento.",
            ],
            [
                "Búsqueda",
                "RandomizedSearchCV: 20 combinaciones, cinco folds "
                "estratificados; F1 de la clase positiva.",
            ],
            [
                "Calibración",
                "Predicciones fuera de fold de entrenamiento. Sigmoid si mejora "
                "Brier y log-loss al menos 1%; SVM utiliza Platt.",
            ],
            [
                "Umbral",
                "Rejilla 0,05-0,95 en validación: maximizar F1; empate, cercanía a "
                "0,5 y luego menor umbral.",
            ],
            [
                "Selección",
                "F1, ROC-AUC y recall con tolerancia 0,005; estabilidad CV y "
                "parsimonia como desempate. No se elige por el mayor F1 aislado.",
            ],
            [
                "Prueba final",
                "Una evaluación del ganador congelado el 09/11/2026. "
                "No se reutiliza para decidir variables, parámetros, calibración o umbral.",
            ],
        ],
        [100, WIDTH - 100],
    )
    paragraph(
        "<b>Ingeniería de variables.</b> Se comparó cada familia con y sin seis "
        "derivadas: número de servicios; cliente nuevo (antigüedad &lt;=6); contrato "
        "mensual; pago automático; internet sin soporte; y cargo mensual dividido "
        "por servicios, con denominador mínimo uno. La creación es determinística "
        "y forma parte del pipeline."
    )
    paragraph(
        "Las cinco familias son Regresión Logística, Random Forest, XGBoost, "
        "LightGBM y SVM. El vencedor conserva los 19 predictores originales."
    )

    section("04", "Comparación y selección del ganador")
    paragraph(
        "Resultados de <b>validación</b>. Cada fila representa una familia y variante. "
        "Los umbrales se fijan en validación; estas cifras no son el resultado final reservado."
    )
    names = {
        "random_forest": "Random Forest",
        "logistic_regression": "Reg. logística",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "svm": "SVM",
    }
    rows = [["Candidato", "Acc.", "Prec.", "Recall", "F1", "AUC", "Umbral"]]
    for candidate in candidates:
        family, variant = candidate["candidate"].split("__")
        label = names[family] + (" + derivadas" if variant == "con_derivadas" else " original")
        rows.append(
            [
                label,
                *[
                    f"{float(candidate[key]):.3f}"
                    for key in (
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                        "decision_threshold",
                    )
                ],
            ]
        )
    table(rows, [165, *[(WIDTH - 165) / 6] * 6])
    paragraph(
        "Random Forest original alcanza F1 0,656 y XGBoost original F1 0,651. "
        "La tolerancia de 0,005 mantiene varios candidatos; el protocolo continúa "
        "con AUC, recall, estabilidad y parsimonia. El ganador registrado es "
        "XGBoost sin derivadas; su recall de validación es 0,824, AUC 0,856 y "
        "umbral 0,25. No es una selección retrospectiva con la prueba final."
    )
    paragraph(
        "<b>Configuración ganadora.</b> 200 árboles, profundidad 6, learning_rate=0,01, "
        "subsample=0,7 y scale_pos_weight=2,769. Calibración sigmoid, cinco folds, "
        "ensemble=False; ajuste exclusivo sobre entrenamiento."
    )

    section("05", "Evaluación final e implicaciones")
    table(
        [
            ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            [
                f"{float(test[key]):.3f}"
                for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
            ],
        ],
        [WIDTH / 5] * 5,
    )
    image("reports/evaluation/matriz_prueba_final_guardada.png", 300)
    paragraph(
        "De 374 abandonos, se detectan 300 y se omiten 74. De 1.035 permanencias, "
        "745 se clasifican correctamente y 290 reciben una alerta falsa. Precision "
        "0,508 implica que alrededor de la mitad de las alertas positivas son "
        "abandonos observados. Una campaña debe considerar el costo de contactar "
        "y de no detectar, sin asumir que todos los clientes señalados abandonarán."
    )
    paragraph(
        f"Average Precision {float(test['average_precision']):.3f}; "
        f"Brier {float(test['brier']):.3f}; log-loss {float(test['log_loss']):.3f}. "
        "Son resultados guardados del conjunto final; no se recalculan desde la web."
    )
    paragraph(
        "Las curvas ROC y calibración publicadas corresponden a validación. "
        "No se fabrica una curva ROC final: no se conservan las probabilidades "
        "originales de esa evaluación.",
        "SmallChuard",
    )

    section("06", "Explicabilidad y sugerencias de retención")
    paragraph(
        "Permutation SHAP explica la probabilidad calibrada del estimador completo, "
        "con referencia fija de 100 registros de entrenamiento. Las contribuciones "
        "se agrupan en los 19 predictores y reconstruyen la probabilidad con tolerancia "
        "absoluta 0,000001. El gráfico global usa 100 registros de entrenamiento."
    )
    image("reports/explain/shap_global.png", 330)
    paragraph(
        "En cada predicción se muestran los cinco factores de mayor contribución "
        "absoluta; la vista individual permite consultar los 19. El signo indica "
        "aumento o reducción respecto a la referencia, en puntos de probabilidad."
    )
    paragraph(
        "Las sugerencias son reglas determinísticas apoyadas por factores positivos "
        "y contexto del cliente: revisar contrato, facturación, soporte o servicios. "
        "La prioridad es probabilidad × cargo mensual. Las bandas 0,35 y 0,65 son "
        "convenciones operativas, diferentes del umbral binario 0,25. Género y edad "
        "no se usan para asignar beneficios; el predictor aún requiere una auditoría de equidad."
    )

    section("07", "Aplicación, arquitectura y seguridad")
    table(
        [
            ["Sección", "Flujo implementado"],
            [
                "Resumen",
                "Contexto del dataset, abandono por contrato, comparación de "
                "candidatos y matriz final guardada.",
            ],
            [
                "Predicción",
                "19 variables, coherencia de servicios, probabilidad, decisión "
                "binaria, banda de riesgo, SHAP y exportación. Se invalida al editar.",
            ],
            [
                "Cartera CSV",
                "Carga UTF-8/coma, validación completa, cálculo por cliente, "
                "filtros, búsqueda, selección de fila y exportación.",
            ],
            [
                "Explicabilidad",
                "Gráfico global y contribuciones individuales del cliente "
                "seleccionado, con base y reconstrucción numérica.",
            ],
            [
                "Retención",
                "Lista ordenada por exposición, sugerencias y exportación. "
                "Si existe Churn se incluyen solo activos Churn=No.",
            ],
        ],
        [105, WIDTH - 105],
    )
    paragraph(
        "<b>Separación.</b> app.py inicia ui/app.py; ui/pages.py presenta los flujos "
        "con controles nativos. src/application.py verifica recursos y prepara "
        "exportaciones; src/schema, inference, explain y recommendations concentran "
        "las reglas. El modelo se comparte como recurso público; los datos del "
        "usuario y resultados permanecen en su sesión, sin caché global de carteras."
    )
    paragraph(
        "<b>Seguridad.</b> Se comprueba SHA-256 antes de deserializar el Joblib "
        "propio y se rechazan dependencias incompatibles. No se permite cargar "
        "modelos arbitrarios. Un CSV inválido elimina resultados de la fuente "
        "anterior; las selecciones conservan índices técnicos aunque se repitan "
        "IDs. Se neutralizan fórmulas en celdas de texto exportadas. No hay "
        "credenciales ni datos personales en los ejemplos."
    )
    paragraph(
        "<b>Capacidad medida.</b> Límite web: 100 registros y 10 MB. Un lote local "
        "de 100 perfiles ficticios con probabilidad, SHAP y recomendaciones tardó "
        "34,93 segundos en Windows/Python 3.11.16. No es una cifra de Streamlit "
        "Cloud ni una garantía de latencia; el tiempo depende del servidor y los datos."
    )

    section("08", "La aplicación en funcionamiento")
    paragraph(
        "Capturas de la aplicación Streamlit implementada y verificada en navegador. "
        "La primera muestra el resumen científico; la segunda, una predicción real "
        "del modelo congelado sobre un perfil ficticio. No son pantallas simuladas."
    )
    image(
        "reports/application/publicacion.jpg"
        if application_url
        else "reports/application/resumen.jpg",
        245,
    )
    paragraph(
        "Resumen publicado en Streamlit Community Cloud: acceso público verificado."
        if application_url
        else "Resumen: contexto, métricas guardadas y análisis por contrato.",
        "SmallChuard",
    )
    image("reports/application/prediccion.jpg", 245)
    paragraph(
        "Ejemplo ficticio DEMO-INDIVIDUAL: probabilidad estimada 72,2%, "
        "riesgo alto y decisión de abandono al umbral 25%. No es una predicción "
        "sobre una persona real ni una medición de desempeño del modelo.",
        "SmallChuard",
    )

    section("09", "Verificación, reproducción y demostración")
    paragraph(
        "<b>Verificación integrada.</b> 103 pruebas pasan, incluidas entrada inválida, "
        "huella corrupta antes de deserializar, CSV seguro, IDs repetidos, predicción "
        "real, invalidación al editar, coherencia de servicios, navegación, SHAP, "
        "exclusión de abandonos y borrado de sesión. Ruff y formato se comprueban "
        "con los comandos del README. Son pruebas técnicas, no estimaciones "
        "adicionales de desempeño predictivo."
    )
    paragraph(
        "<b>Para ejecutar.</b> Clonar el repositorio, crear Python 3.11 e instalar "
        "requirements-dev-lock.txt para desarrollo o requirements.txt para servir "
        "la web. Ejecutar python -m streamlit run app.py. El dataset crudo no es "
        "necesario para usar la aplicación con el modelo y reportes publicados."
    )
    if application_url:
        paragraph(
            "<b>Publicación verificada.</b> La web abre sin iniciar sesión. Se comprobaron "
            "predicción, carga del CSV ficticio de seis filas, scoring, descarga, SHAP "
            "global e individual, cambio de cliente y retención de cinco activos. "
            "También se verificaron aislamiento entre sesiones y borrado de resultados. "
            "La descarga coincide numéricamente con Windows: diferencia máxima de "
            "probabilidad 5,56e-17 y de valores SHAP 1,60e-10, dentro de la tolerancia "
            "SHAP existente de 1e-6. Se conservan las mismas recomendaciones. "
            "La comparación usa perfiles ficticios, sin reevaluar la prueba final."
        )
    paragraph(
        "<b>Demostración sugerida (5-7 minutos).</b><br/>"
        "1. Resumen: explicar el problema, las particiones y la diferencia entre "
        "validación y prueba final.<br/>"
        "2. Predicción: calcular el perfil ficticio inicial y explicar SHAP, "
        "umbral binario y bandas; editar un dato para mostrar la invalidación.<br/>"
        "3. Cartera CSV: probar los seis perfiles ficticios y analizar; filtrar "
        "y exportar. El CSV de ejemplo se descarga desde la misma pantalla.<br/>"
        "4. Explicabilidad: mostrar global y cliente, verificar referencia más "
        "contribuciones y aclarar que no es causalidad.<br/>"
        "5. Retención: verificar orden por exposición y que DEMO-006, "
        "con Churn=Yes, queda fuera; explicar límites de las sugerencias.<br/>"
        "6. Borrar la sesión y presentar los entregables del README."
    )
    paragraph(
        "<b>Trazabilidad.</b> La reproducción del 10/11 coincide con los diez "
        "candidatos y el ganador usando solo entrenamiento/validación. No sustituye "
        "el experimento original: los archivos históricos detallados referidos "
        "por sus huellas no están disponibles y se conserva esa limitación."
    )
    paragraph(
        "La reproducción científica exige Python 3.11.16 y versiones exactas. "
        "La web comprueba Python 3.11 (permite parches) y las siete bibliotecas "
        "del manifiesto exactamente. Las huellas y .gitattributes conservan los "
        "bytes del modelo y los reportes entre sistemas."
    )

    section("10", "Limitaciones, conclusiones y fuentes")
    paragraph(
        "El dataset es una muestra histórica sin secuencia temporal suficiente "
        "para prometer riesgo en 30/60/90 días. La validez fuera de la población "
        "Telco, estabilidad temporal y equidad por subgrupos necesitan evaluación "
        "adicional. La web advierte valores fuera de rangos de entrenamiento, "
        "pero no verifica que un nuevo negocio tenga la misma distribución."
    )
    paragraph(
        "La exposición mensual no es ingreso perdido ni ingreso recuperable "
        "demostrado; no se atribuye moneda al dataset. Las recomendaciones son "
        "hipótesis comerciales, no acciones ejecutadas. Harían falta revisión "
        "operativa y experimentos controlados para estimar su efectividad."
    )
    paragraph(
        "El ganador ofrece un compromiso favorable a detección de abandonos, "
        "con costo de falsas alertas explícito. Chuard conserva un protocolo "
        "reproducible y separa evidencia científica de sugerencias de operación. "
        "La aplicación presenta las estimaciones con sus factores y límites."
    )
    paragraph("<b>Fuentes y evidencia.</b>")
    for text, url in [
        (
            "Telco Customer Churn · Kaggle",
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
        ),
        (
            "Código, notebooks, manifiestos y resultados · Chuard",
            "https://github.com/Jacklb19/Chuard",
        ),
        (
            "Validación cruzada y pipelines · scikit-learn",
            "https://scikit-learn.org/stable/modules/cross_validation.html",
        ),
        (
            "PermutationExplainer · SHAP",
            "https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html",
        ),
        (
            "Despliegue y dependencias · Streamlit",
            "https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies",
        ),
    ]:
        paragraph(f'<link href="{url}" color="#25664f">{escape(text)}</link>', "SmallChuard")
    paragraph("Huella del modelo congelado (SHA-256):", "SmallChuard")
    digest = manifest["model_sha256"]
    paragraph(digest[:32] + "<br/>" + digest[32:], "SmallChuard")

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#dce2da"))
        canvas.line(50, 45, A4[0] - 50, 45)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(50, 31, "CHUARD  /  MINERÍA DE DATOS  /  INFORME ACADÉMICO")
        canvas.drawRightString(A4[0] - 50, 31, f"{document.page:02d}")
        canvas.restoreState()

    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=45,
        bottomMargin=65,
        title="Chuard - Informe final",
        author="Equipo Chuard",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Created {destination} on {date.today().isoformat()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/informe_final.pdf")
    parser.add_argument("--application-url")
    arguments = parser.parse_args()
    build_report(arguments.output, arguments.application_url)
