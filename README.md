# Chuard

Proyecto académico de minería de datos para estimar abandono de clientes de telecomunicaciones, comparar cinco algoritmos y presentar explicaciones y sugerencias de retención.

**Estado:** base de desarrollo. Todavía no incluye dataset, modelo ganador, resultados reales, SHAP funcional ni interfaz web implementada.

## Instalación

Requisitos: Git y Python 3.11.

```powershell
git clone https://github.com/Jacklb19/Chuard.git
cd Chuard
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev-lock.txt
```

Si `python` no corresponde a Python 3.11, usa la ruta de esa instalación o `py -3.11` si tienes el lanzador. En Linux/macOS usa `.venv/bin/python` como ejecutable del entorno.

Alternativa con uv: `uv venv --python 3.11` y `uv pip install --python .venv/Scripts/python.exe -r requirements-dev-lock.txt`.

`requirements-dev-lock.txt` registra el entorno comprobado en Windows/Python 3.11. Si otro sistema necesita resolver versiones distintas, usa `requirements-dev.txt` y documenta los cambios antes de actualizar los archivos comunes. `requirements-lock.txt` contiene las dependencias de ejecución.

## Qué incluye la base

- Validación de las 19 variables originales y las columnas auxiliares `customerID` y `Churn`.
- Lectura CSV con límites de tamaño/filas, detección de encabezados duplicados y errores por registro.
- Reglas de coherencia entre telefonía, internet y servicios asociados.
- Seis características derivadas y un transformador compatible con pipelines de scikit-learn.
- Constructor de pipeline sin ajustar, particiones estratificadas reproducibles y métricas binarias.
- Validación inicial de artefactos, contrato de inferencia y recomendaciones determinísticas.
- Pruebas con registros sintéticos identificados como tales, sin cifras de desempeño inventadas.

El scoring completo llama a `explain_customer`, que todavía lanza `NotImplementedError`. No puede considerarse funcional hasta implementar SHAP con el ganador real. `app.py` informa que la interfaz está pendiente.

## Estructura

```text
src/
  config.py            configuración compartida
  schema.py            entradas, dominios y validación
  data.py              lectura y huella del dataset
  features.py          características derivadas
  train.py             pipeline y particiones; búsqueda pendiente
  evaluate.py          métricas de clasificación
  inference.py         carga del artefacto y scoring
  explain.py           contrato de SHAP pendiente
  recommendations.py   sugerencias determinísticas
tests/                 comprobaciones técnicas
app.py                 punto de entrada reservado para Streamlit
requirements*.txt      dependencias y versiones comprobadas
pyproject.toml         configuración del proyecto y herramientas
```

Los directorios de datos, artefactos y resultados se crean localmente cuando se necesiten. No se distribuyen archivos personales, credenciales, entornos virtuales ni material ajeno al proyecto.

## Dataset y preparación

Fuente prevista: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

Crea `data/raw/` y coloca allí `WA_Fn-UseC_-Telco-Customer-Churn.csv` sin modificarlo. La referencia esperada es 7.043 registros y 21 columnas: verificar esos valores, las categorías, nulos y duplicados contra el CSV real. No incluir credenciales de Kaggle. El dataset y los artefactos permanecen excluidos de Git por defecto.

Después de verificar calidad y duplicados:

```powershell
.venv\Scripts\python.exe -m src.train --help
.venv\Scripts\python.exe -m src.train
```

Este comando **solo guarda las particiones**, no entrena modelos. Falla si falta el CSV o ya existe un manifiesto. La asignación 60/20/20 usa semilla 42 y se conserva en `reports/modeling/splits.json`, junto con la huella del dataset.

El objetivo es comparar Regresión Logística, Random Forest, XGBoost, LightGBM y SVM, con/sin derivadas. Búsqueda mediante validación cruzada estratificada de cinco folds solo dentro de entrenamiento; calibración y umbral se cierran antes de la prueba final. Comparar en validación y evaluar el ganador congelado en prueba. La aplicación nunca entrena.

## Contrato técnico inicial

`src/schema.py` define nombres y dominios. `validate_customers(df, mode="inference")` devuelve una copia normalizada y advertencias; un lote inválido se rechaza completo con errores por posición de datos, columna y motivo. Los identificadores repetidos no se usan como clave única.

El pipeline recibe las 19 variables originales normalizadas. Las derivadas se calculan dentro del estimador cuando se activa esa variante; no se duplican en la interfaz. `customerID` y `Churn` nunca son predictores. `TotalCharges` vacío se admite en inferencia solo con `tenure=0`; texto corrupto, negativos e infinitos se rechazan.

El artefacto propio `models/best_model.joblib` deberá contener estimador/preprocesamiento, columnas, nombre/versión del modelo, versión del contrato, umbral, bandas y metadatos de procedencia/calibración. No cargar Joblib de procedencia desconocida: puede ejecutar código.

`score_customers` conserva las entradas y añade probabilidad de clase 1, decisión binaria, banda de riesgo, exposición mensual, factores y recomendaciones. La exposición `probabilidad × MonthlyCharges` es exploratoria, no pérdida confirmada. SHAP debe identificar clase, escala y salida explicada, especialmente si se calibra el modelo. Las recomendaciones no garantizan retención.

## Desarrollo en equipo

Cada persona usa su propia copia local y una rama distinta:

| Frente | Rama sugerida | Archivos principales |
|---|---|---|
| Datos y explicabilidad | `datos-explicabilidad` | `schema.py`, `config.py`, `data.py`, `features.py`, `explain.py` |
| Modelos e inferencia | `modelos-inferencia` | `train.py`, `evaluate.py`, `inference.py` |
| Web y retención | `web-retencion` | `app.py`, futuros componentes visuales y `recommendations.py` |

Antes de modificar interfaces, acordar entradas, salidas y metadatos entre los tres frentes. Preservar cambios existentes y comprobar integración por hitos. No publicar archivos ajenos, secretos, entornos, cachés o temporales; revisar las rutas y el contenido preparado, añadiendo archivos explícitos. La información del repositorio se centraliza en este README.

## Comprobaciones

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
```

La base cuenta con 35 pruebas comprobadas. Cubren validación, características, particiones, persistencia de un pipeline sintético, métricas, contrato de inferencia y recomendaciones; no miden rendimiento con Telco.

## Trabajo pendiente

Obtener y analizar el CSV real; entrenar/comparar las cinco familias; seleccionar/calibrar el ganador y su umbral; exportar el artefacto; implementar SHAP global/individual y las pantallas Streamlit; validar el flujo completo y presentar los resultados académicos. El sitio web todavía no está desplegado.
