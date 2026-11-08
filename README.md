# Chuard

Proyecto académico de minería de datos para estimar abandono de clientes de telecomunicaciones, comparar cinco algoritmos y presentar explicaciones y sugerencias de retención.

**Estado:** módulos de datos, EDA y explicabilidad desarrollados. El CSV se obtiene localmente; no se distribuye en Git. Todavía no hay modelo ganador ni interfaz web implementada, y falta validar SHAP con el ganador real.

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

`explain_customer` calcula SHAP cuando el bundle contiene un estimador entrenado y `explanation_context` con referencia exclusiva de entrenamiento. Sin esos recursos falla explícitamente, sin simular factores. El scoring del ganador aún debe validarse cuando se entregue su artefacto. `app.py` informa que la interfaz está pendiente.

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

Para obtenerlo explícitamente por el endpoint público de Kaggle, solo si no existe el original:

```powershell
.venv\Scripts\python.exe -m src.data download
.venv\Scripts\python.exe -m src.data audit
```

La descarga no crea cuentas, no usa credenciales y no sobrescribe el original. Si Kaggle requiere autenticación, descarga manualmente el CSV y colócalo en `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`. No incluyas secretos.

El archivo comprobado tiene 7.043 registros, 21 columnas, 1.869 etiquetas Yes y 5.174 No; no hay filas ni IDs duplicados. Sus 11 vacíos de `TotalCharges` corresponden a `tenure=0` y se normalizan a cero en una copia. Hay 42 registros que comparten perfil y etiqueta con otro cliente, pero tienen IDs distintos: no se eliminan. Huella SHA-256: `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`.

Después de verificar calidad y duplicados:

```powershell
.venv\Scripts\python.exe -m src.train --help
.venv\Scripts\python.exe -m src.train
```

Este comando **solo guarda las particiones**, no entrena modelos. Falla si falta el CSV o ya existe un manifiesto. La asignación 60/20/20 usa semilla 42 y se conserva en `reports/modeling/splits.json`, junto con la huella del dataset.

El objetivo es comparar Regresión Logística, Random Forest, XGBoost, LightGBM y SVM, con/sin derivadas. Búsqueda mediante validación cruzada estratificada de cinco folds solo dentro de entrenamiento; calibración y umbral se cierran antes de la prueba final. Comparar en validación y evaluar el ganador congelado en prueba. La aplicación nunca entrena.

## Análisis exploratorio reproducible

Con el manifiesto acordado del frente de modelos:

```powershell
.venv\Scripts\python.exe -m src.data eda --manifest reports/modeling/splits.json
```

El lector verifica la huella del CSV, posiciones válidas, cobertura completa y ausencia de solapamiento. El EDA genera nueve PNG, tablas, conclusiones y metadatos en `reports/data/`; usa solo entrenamiento. Una asignación de desarrollo se mantiene como propuesta local hasta confirmarla con el frente de modelos. No se modifica su manifiesto final desde el EDA.

Se describen abandono, contrato, pagos, internet/soporte, antigüedad, cargos y correlaciones. Las gráficas indican unidades y tamaños de segmento. Los diagnósticos IQR no eliminan ni truncan valores; no se aplican como detección de anomalías al indicador binario SeniorCitizen. Las asociaciones no demuestran causalidad y la moneda no está documentada.

## Contrato técnico inicial

`src/schema.py` define nombres y dominios. `validate_customers(df, mode="inference")` devuelve una copia normalizada y advertencias; un lote inválido se rechaza completo con errores por posición de datos, columna y motivo. Los identificadores repetidos no se usan como clave única.

El pipeline recibe las 19 variables originales normalizadas. Las derivadas se calculan dentro del estimador cuando se activa esa variante; no se duplican en la interfaz. `customerID` y `Churn` nunca son predictores. `TotalCharges` vacío se admite en inferencia solo con `tenure=0`; texto corrupto, negativos e infinitos se rechazan.

El artefacto propio `models/best_model.joblib` deberá contener estimador/preprocesamiento, columnas, nombre/versión del modelo, versión del contrato, umbral, bandas y metadatos de procedencia/calibración. No cargar Joblib de procedencia desconocida: puede ejecutar código.

`score_customers` conserva las entradas y añade probabilidad de clase 1, decisión binaria, banda de riesgo, exposición mensual, factores y recomendaciones. La exposición `probabilidad × MonthlyCharges` es exploratoria, no pérdida confirmada. SHAP debe identificar clase, escala y salida explicada, especialmente si se calibra el modelo. Las recomendaciones no garantizan retención.

### Conectar SHAP al artefacto

Durante la preparación del bundle, con el conjunto de entrenamiento normalizado y el estimador ya congelado:

```python
from src.explain import prepare_explanation_context, explain_customer

bundle["explanation_context"] = prepare_explanation_context(X_train)
factors = explain_customer(customer_df, bundle)
```

`prepare_explanation_context` acepta los 19 predictores con auxiliares opcionales; no ajusta el modelo. La referencia se muestrea reproduciblemente solo del entrenamiento y se guarda en el bundle. Registra también huellas del CSV/manifiesto y versiones compatibles. Nunca construyas esa referencia a partir del CSV subido a la aplicación.

El método `auto` usa LinearExplainer en log-odds para Regresión Logística sin calibrar, TreeExplainer en probabilidad para árboles compatibles sin calibrar y PermutationExplainer sobre la probabilidad del estimador completo para calibradores/SVM. Para exigir siempre la probabilidad final, pasa `method="permutation"` al preparar el contexto.

One-Hot se agrupa por variable original sin perder la suma de contribuciones. Las permutaciones sobre entradas originales recalculan las derivadas; sus híbridos pueden romper dependencias entre servicios, limitación que debe explicarse. El valor base más todas las contribuciones debe reconstruir la salida explicada; cinco factores principales no son la descomposición completa.

`get_explanation_service(bundle).explain(df)` devuelve contribuciones completas, valores base, probabilidad y escala; `save_explanation_plots(result, output_dir)` genera beeswarm y waterfall reales. La caché es acotada e identifica modelo, versión, referencia y configuración. No reajustar un modelo ya cacheado conservando su misma versión. Medir el coste con el ganador antes de prometer tiempos o tamaños de lote.

`validate_customers(..., training_ranges=bundle.get("training_ranges"))` permite advertir rangos fuera de entrenamiento sin recortar entradas. El frente de inferencia debe conectar ese contexto. Los límites se obtienen con `summarize_training` solo del entrenamiento; el umbral de cargos altos es una propuesta comercial basada en su Q3.

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

Las pruebas cubren validación, características, particiones, persistencia, métricas, contrato de inferencia, recomendaciones y SHAP real sobre modelos sintéticos de las cinco familias y calibradores. Comprueban reconstrucción, escala, consistencia individual/lote y gráficos; no miden rendimiento predictivo con Telco. Las advertencias siguen tratándose como errores.

Matplotlib se limita a versiones 3.9–3.10 por una API obsoleta usada por SHAP 0.51 en Matplotlib 3.11. No se silencian advertencias para pasar las pruebas. En scikit-learn 1.9, `SVC(probability=True)` está obsoleto: para SVM usar calibración explícita sobre el pipeline con particiones de entrenamiento separadas.

## Trabajo pendiente

Confirmar entre frentes la partición y metadatos; entrenar/comparar las cinco familias; seleccionar/calibrar el ganador y su umbral; exportar el artefacto con sus recursos SHAP; comprobar las explicaciones del ganador; implementar las pantallas Streamlit y validar el flujo completo. El sitio web todavía no está desplegado y no hay métricas predictivas reales publicadas.
