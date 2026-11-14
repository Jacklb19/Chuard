# Chuard

Proyecto académico de minería de datos para estimar abandono de clientes de telecomunicaciones, comparar cinco algoritmos y presentar explicaciones y sugerencias de retención.

**Estado:** aplicación Streamlit implementada e integrada con datos, EDA, entrenamiento, evaluación, inferencia, SHAP y retención. Ganador verificado XGBoost sin derivadas, calibrado sigmoid, versión 1.0.0 y umbral 0.25. Se incluyen tres notebooks con sus ejecuciones, resultados científicos e informe final PDF. **103 pruebas pasan**; Ruff y formato correctos. La aplicación pública está verificada en [chuard.streamlit.app](https://chuard.streamlit.app/), sin necesidad de iniciar sesión. Datos crudos y documentos privados permanecen fuera de GitHub.

## Entrega académica al docente

Esta lista sigue la especificación del proyecto Chuard; no sustituye una rúbrica adicional del docente. La parte científica, la aplicación y el informe están disponibles. La publicación remota también está comprobada con perfiles ficticios y acceso sin iniciar sesión.

| Entregable | Ubicación | Estado comprobado |
|---|---|---|
| Código, dependencias y pruebas | [`src/`](src/), [`tests/`](tests/), `requirements*.txt` | Implementados; 103 pruebas pasan, Ruff y formato correctos |
| Dataset documentado y calidad | [Fuente y descarga](#dataset-y-preparación), [reporte de calidad](reports/data/quality_report.json), [procedencia](reports/data/source_manifest.json) | CSV original obtenido aparte; fuente, huella, esquema, nulos y duplicados documentados |
| Limpieza, EDA y derivadas | [Notebook 01](notebooks/01_eda_limpieza.ipynb), [ejecución](reports/data/01_eda_limpieza_ejecutado.ipynb), [figuras y tablas](reports/data/) | Ejecutado; nueve figuras basadas solo en entrenamiento |
| Cinco modelos y evaluación | [Notebook 02](notebooks/02_modelos_evaluacion.ipynb), [ejecución](reports/evaluation/02_modelos_evaluacion_ejecutado.ipynb), [métricas](reports/metricas_modelos.csv), [reproducción](reports/modeling/reproduction_20261110_reanudada/) | Diez candidatos, CV, calibración, umbrales, selección y prueba final guardada |
| Modelo exportado | [Ganador](models/best_model.joblib), [manifiesto](models/best_model.manifest.json) | Pipeline congelado, versiones, umbral y huellas verificados |
| Explicabilidad | [Notebook 03](notebooks/03_interpretabilidad.ipynb), [ejecución](reports/explain/03_interpretabilidad_ejecutado.ipynb), [SHAP](reports/explain/) | Global e individual sobre probabilidad calibrada; reconstrucción comprobada |
| Aplicación y aceptación | [`app.py`](app.py), [`ui/`](ui/), [`tests/test_web.py`](tests/test_web.py) | Cinco secciones funcionales, pruebas de flujos y revisión visual local |
| Informe final | [PDF académico](reports/informe_final.pdf), [generador](tools/build_report.py) | Diez páginas: metodología, EDA, comparación, evaluación, SHAP, web, límites y demo |
| Publicación y demostración | [Guion de demostración](#demostración-al-docente) y despliegue | [Aplicación pública](https://chuard.streamlit.app/) verificada: cinco vistas, carga/descarga y sesiones separadas |

Para la sustentación, recorrer notebooks 01 → 02 → 03: problema y objetivo; calidad/limpieza de datos; prevención de fuga; comparación de cinco familias con/sin derivadas; selección del ganador y umbral; errores FP/FN; explicación SHAP y límites. Después demostrar la aplicación con el ejemplo ficticio incluido. No presentar las recomendaciones como efectos de retención comprobados ni las métricas históricas como garantías futuras.

La documentación académica disponible se centraliza aquí y en los notebooks. Los documentos privados de coordinación, prompts, archivos de agentes, entornos y cachés permanecen fuera del repositorio.

## Instalación

Requisitos: Git y Python 3.11. Para cargar el artefacto congelado comprobado, usar Python **3.11.16** y las versiones fijadas; `src.reporting` rechaza versiones diferentes antes de deserializarlo.

```powershell
git clone https://github.com/Jacklb19/Chuard.git
cd Chuard
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev-lock.txt
```

Si `python` no corresponde a Python 3.11, usa la ruta de esa instalación o `py -3.11` si tienes el lanzador. En Linux/macOS usa `.venv/bin/python` como ejecutable del entorno.

Alternativa con uv: `uv venv --python 3.11` y `uv pip install --python .venv/Scripts/python.exe -r requirements-dev-lock.txt`.

`requirements-dev-lock.txt` registra el entorno comprobado en Windows/Python 3.11. Si otro sistema necesita resolver versiones distintas, usa `requirements-dev.txt` y documenta los cambios antes de actualizar los archivos comunes. `requirements-lock.txt` contiene las dependencias de ejecución y `requirements.txt` lo incluye, también en Streamlit Cloud. `requirements.in` declara las dependencias directas para empaquetado y mantenimiento del lock. No actualizar versiones sin comprobar compatibilidad con el artefacto.

## Ejecutar la aplicación

No necesitas descargar el CSV crudo para usar la web: el modelo y los reportes publicados son suficientes.

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Abre la dirección local que indique Streamlit. Las cinco secciones son:

- **Resumen:** métricas históricas, contratos, diez candidatos de validación y matriz final guardada.
- **Predicción:** formulario de 19 variables; probabilidad real, decisión binaria, banda, cinco factores SHAP y exportación. Editar un dato invalida el resultado anterior. Los valores iniciales son ficticios.
- **Cartera CSV:** UTF-8 con coma, hasta **100 registros y 10 MB**; valida el lote completo, calcula todas las explicaciones, muestra distribuciones y segmentos, permite buscar, filtrar por riesgo/contrato/internet y exportar. Hay un [CSV ficticio](examples/clientes_ejemplo.csv) descargable desde la web.
- **Explicabilidad:** SHAP global sobre entrenamiento y contribuciones completas del cliente seleccionado por su posición técnica, incluso con IDs repetidos.
- **Retención:** prioridad `probabilidad × MonthlyCharges`, recomendaciones y exportación. Con `Churn`, excluye los `Yes`; sin esa columna, advierte que se presupone una cartera activa.

El umbral **0.25** produce la decisión binaria. Las bandas **0.35/0.65** son convenciones operativas y no equivalen a esa decisión. La exposición usa unidades del dataset, sin moneda confirmada; no es pérdida demostrada.

Las carteras y sus resultados se conservan en la sesión, sin caché global de datos de usuarios ni escritura en disco. El botón **Borrar datos de esta sesión** los elimina. Cambiar la fuente invalida los resultados previos incluso si el archivo nuevo es inválido. Las exportaciones conservan el orden, codifican listas como JSON y neutralizan fórmulas de hoja de cálculo en texto. El modelo compartido se verifica por SHA-256 **antes** de abrir Joblib y exige las versiones de sus siete bibliotecas; para servir la web permite parches de Python 3.11. La reproducción científica mantiene la exigencia exacta 3.11.16. No se aceptan modelos subidos por usuarios y la aplicación nunca entrena.

### Publicar en Streamlit Community Cloud

**Aplicación publicada:** [https://chuard.streamlit.app/](https://chuard.streamlit.app/). Acceso público comprobado sin iniciar sesión. Se usa la cuenta existente de la usuaria. Configuración según la [documentación oficial](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies): repositorio **Jacklb19/Chuard**, rama **main**, archivo **app.py**, Python **3.11** en Advanced settings. `requirements.txt` instala el lock comprobado; no se necesitan secretos ni el dataset original. El servidor cargó el modelo propio y se comprobaron los cinco flujos. La evidencia de publicación y sus límites están en [deployment.json](reports/application/deployment.json). Una actualización futura de main puede reiniciar la aplicación; hay que volver a verificarla después de cambios funcionales o de dependencias.

### Demostración al docente

1. Mostrar Resumen y distinguir validación de prueba final guardada.
2. En Predicción, calcular el perfil ficticio inicial y explicar probabilidad, bandas y factores. Cambiar un dato para mostrar que desaparece el resultado anterior.
3. En Cartera CSV, pulsar **Probar con datos ficticios** y **Analizar cartera**; filtrar, seleccionar una fila y exportar resultados.
4. En Explicabilidad, revisar Global y Cliente; comprobar referencia más contribuciones y explicar que SHAP no demuestra causalidad.
5. En Retención, mostrar orden por exposición y sugerencias. **DEMO-006**, etiquetado `Churn=Yes`, queda excluido.
6. Borrar los datos de sesión y presentar notebooks, PDF y repositorio. No usar información personal en la demostración pública.

### Regenerar el informe

El PDF se construye a partir de reportes guardados, sin entrenar ni evaluar de nuevo la prueba final. En un entorno separado de autoría instala `requirements-report.txt` y ejecuta `python tools/build_report.py`. El generador toma la URL comprobada de `reports/application/deployment.json`; para una publicación distinta usa `--application-url URL`. Revisa visualmente las diez páginas antes de sustituir la entrega.

## Qué incluye la base

- Validación de las 19 variables originales y las columnas auxiliares `customerID` y `Churn`.
- Lectura CSV con límites de tamaño/filas, detección de encabezados duplicados y errores por registro.
- Reglas de coherencia entre telefonía, internet y servicios asociados.
- Seis características derivadas y un transformador compatible con pipelines de scikit-learn.
- Pipelines de cinco familias con/sin derivadas, búsqueda CV, calibración OOF y selección documentada.
- Particiones estratificadas reproducibles, métricas, curvas ROC, calibración y matrices de confusión.
- Verificación de recursos congelados, inferencia y generación de explicaciones reales sin entrenar.
- Recomendaciones determinísticas y lista de retención ordenada por exposición mensual, sin ejecutar campañas.
- Pruebas con registros sintéticos identificados como tales, sin cifras de desempeño inventadas.

`explain_customer` calcula SHAP cuando el bundle contiene un estimador entrenado y `explanation_context` con referencia exclusiva de entrenamiento. Sin esos recursos falla explícitamente, sin simular factores. El ganador real está comprobado en validación y mediante scoring individual/lote; la aplicación nunca debe llamar a `fit` ni entrenar.

## Estructura

```text
src/
  config.py            configuración compartida
  schema.py            entradas, dominios y validación
  data.py              lectura y huella del dataset
  features.py          características derivadas
  train.py             particiones, búsqueda, calibración, selección y exportación
  evaluate.py          métricas, umbral y gráficos
  inference.py         carga del artefacto y scoring
  explain.py           SHAP, factores agrupados y gráficos
  reporting.py         verificación y entregables del ganador congelado
  reproduce.py         reproducción de entrenamiento/validación sin prueba
  recommendations.py   sugerencias determinísticas
  application.py       carga verificada, filtros y exportación segura
tests/                 comprobaciones técnicas
ui/                    páginas, componentes, estilos y fuentes locales
examples/              CSV de seis perfiles ficticios
tools/build_report.py  generador del informe académico
app.py                 punto de entrada Streamlit
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

Sin subcomando, `src.train` **solo guarda las particiones**, no entrena modelos. Falla si falta el CSV o ya existe un manifiesto. La asignación 60/20/20 usa semilla 42 y se conserva en `reports/modeling/splits.json`, junto con la huella del dataset. En la entrega existente ya está congelada: no crear otra asignación para sustituirla.

Se compararon Regresión Logística, Random Forest, XGBoost, LightGBM y SVM, con/sin derivadas. Cada variante usa 20 combinaciones en validación cruzada estratificada de cinco folds solo dentro de entrenamiento: 200 combinaciones en total. Los pipelines incluyen todas las transformaciones aprendidas.

La calibración se estudia con probabilidades fuera de fold de entrenamiento. Sigmoid se adopta si Brier y log-loss mejoran al menos 1%; SVM siempre usa Platt. Esas probabilidades reutilizan hiperparámetros seleccionados dentro del entrenamiento y no son una evaluación externa anidada del proceso completo. El umbral maximiza F1 positivo en validación, rejilla 0.05–0.95 en pasos de 0.01; empates: proximidad a 0.50 y luego menor umbral.

Para seleccionar familia/variante se comparan F1, ROC-AUC y recall con tolerancia operativa 0.005; después, estabilidad CV y parsimonia. La tolerancia no implica equivalencia estadística. Random Forest sin derivadas obtuvo el F1 máximo de validación (0.6557); XGBoost sin derivadas quedó dentro de la tolerancia de F1/ROC-AUC y ganó por recall (0.8235). Esta regla se conserva y no se cambió con los resultados de prueba.

## Ganador y resultados verificados

XGBoost sin derivadas, sigmoid mediante CV de cinco folds en entrenamiento (`ensemble=False`), umbral **0.25**. Hiperparámetros: 200 árboles, profundidad 6, learning rate 0.01, subsample 0.7 y scale_pos_weight 2.769. División: 4.225/1.409/1.409 registros.

| Partición | Accuracy | Precisión | Recall | F1 positivo | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| Validación, usada para selección | 0.7658 | 0.5385 | 0.8235 | 0.6512 | 0.8561 | 0.6689 |
| Prueba final, reporte guardado | 0.7417 | 0.5085 | 0.8021 | 0.6224 | 0.8297 | 0.6230 |

La prueba final fue realizada el **9 de noviembre de 2026**; aquí se verifica y lee su reporte sin reevaluarla. Matriz guardada, filas reales y columnas predichas `[No, Yes]`: `[[745, 290], [74, 300]]`. Se detectaron 300 de los 374 abandonos; hubo 74 falsos negativos y 290 falsos positivos. AP es Average Precision, no el área trapezoidal PR. Las cifras describen este dataset histórico y no garantizan resultados de campañas o predicciones futuras.

**Trazabilidad:** el manifiesto de particiones se reconstruyó con bytes idénticos a su huella registrada. Los archivos originales detallados de búsqueda/experimento no estaban disponibles. Una reproducción independiente del 10 de noviembre coincide con los diez candidatos en las métricas guardadas, conteos, calibración y umbrales; también coinciden hiperparámetros y los cinco folds del ganador. Sus archivos se identifican como reproducción y no sustituyen el historial original. No se dispone de las probabilidades originales de prueba para recuperar su curva ROC; no se vuelve a usar prueba para producirla.

## Usar la entrega sin reentrenar

La clonación o actualización de `main` incluye estos archivos propios y comprobados:

```text
models/best_model.joblib
models/best_model.manifest.json
reports/modeling/splits.json
reports/evaluation/validacion_candidatos.csv
reports/evaluation/prueba_final.csv
```

Obtener el CSV original por el procedimiento anterior. SHA-256 del modelo: `0bb0a8542c14f3e23e2fd779fc8b3dd2c488c6c28db069411d23e84b28d1cea1`; del manifiesto de particiones: `ea6ea1b7706099c28df9be2eb3a7e5fd30ff4cdb732d692467ef8d466c862b82`. Joblib puede ejecutar código: las huellas detectan cambios, pero no hacen seguro un archivo de procedencia desconocida.

Para actualizar una copia sin cambios pendientes, estando en `main`:

```powershell
git status --short
git pull --ff-only origin main
.venv\Scripts\python.exe -m pip install -r requirements-dev-lock.txt
.venv\Scripts\python.exe -m src.data download  # solo si todavía falta el CSV
```

Si la copia tiene cambios propios, conservarlos primero en su rama; no descartarlos ni forzar la actualización. Quien esté trabajando en otra rama debe integrar `main` en ella según el acuerdo del equipo, resolviendo los cambios compartidos. `pull` actualiza los archivos versionados, no el entorno Python ni los datos locales.

```powershell
# Si falta splits.json, solo lo restaura si reproduce exactamente su huella:
.venv\Scripts\python.exe -m src.reporting --restore-splits
# Verifica insumos/versiones y genera entregables; no entrena ni puntúa prueba:
.venv\Scripts\python.exe -m src.reporting
```

`reports/metricas_modelos.csv` identifica explícitamente validación y el ganador congelado. `reports/evaluation/` contiene comparación, matrices de los diez candidatos, ROC/calibración del ganador y comprobaciones. `reports/explain/` contiene beeswarm, waterfall, contribuciones completas y metadatos. Los CSV originales no se sobrescriben.

Para repetir la búsqueda como comprobación del protocolo previo, usar una **carpeta nueva**; el comando no recibe filas de prueba y no sustituye el ganador:

```powershell
.venv\Scripts\python.exe -m src.reproduce --output-dir reports/modeling/mi_reproduccion --jobs 2
```

La reproducción entrega `experiment.json`, `search_cv_results.csv`, `validation_probabilities.csv`, `metricas_modelos.csv`, curvas y `reproduction_provenance.json`. Este último compara los candidatos y verifica que no cambiaron el artefacto ni el reporte final. No modificar parámetros para forzar coincidencia. La reproducción verificada de esta entrega está en `reports/modeling/reproduction_20261110_reanudada/`.

`src.train --help`, `experiment --help` y `final --help` documentan el flujo de desarrollo. **No ejecutar `final` de nuevo para esta entrega:** prueba ya se utilizó y el artefacto está congelado. Un experimento futuro requiere su propio protocolo acordado antes de usar una prueba reservada.

Ejemplo de inferencia con el recurso verificado, sin interfaz:

```python
from src.reporting import load_frozen_resources
from src.inference import score_customers

resources = load_frozen_resources()
clientes = resources.training.head(2)  # demostración con datos históricos de entrenamiento
resultado = score_customers(clientes, resources.bundle)
print(resultado[["customerID", "churn_probability", "predicted_churn", "risk_level"]])
```

## Análisis exploratorio reproducible

Con el manifiesto acordado del frente de modelos:

```powershell
.venv\Scripts\python.exe -m src.data eda --manifest reports/modeling/splits.json
```

El lector verifica la huella del CSV, posiciones válidas, cobertura completa y ausencia de solapamiento. El EDA genera nueve PNG, tablas, conclusiones y metadatos en `reports/data/`; usa solo entrenamiento. La asignación definitiva recuperada coincide con la huella del ganador y es la usada por los análisis actuales. La propuesta anterior se conserva como antecedente local y no sustituye ese manifiesto.

Se describen abandono, contrato, pagos, internet/soporte, antigüedad, cargos y correlaciones. Las gráficas indican unidades y tamaños de segmento. Los diagnósticos IQR no eliminan ni truncan valores; no se aplican como detección de anomalías al indicador binario SeniorCitizen. Las asociaciones no demuestran causalidad y la moneda no está documentada.

## Contrato técnico inicial

`src/schema.py` define nombres y dominios. `validate_customers(df, mode="inference")` devuelve una copia normalizada y advertencias; un lote inválido se rechaza completo con errores por posición de datos, columna y motivo. Los identificadores repetidos no se usan como clave única.

El pipeline recibe las 19 variables originales normalizadas. Las derivadas se calculan dentro del estimador cuando se activa esa variante; no se duplican en la interfaz. `customerID` y `Churn` nunca son predictores. `TotalCharges` vacío se admite en inferencia solo con `tenure=0`; texto corrupto, negativos e infinitos se rechazan.

El artefacto propio `models/best_model.joblib` contiene estimador/preprocesamiento, columnas, nombre/versión del modelo, versión del contrato, umbral, bandas, rangos, referencia SHAP y metadatos de procedencia/calibración. `load_model_bundle` valida el contrato; `load_frozen_resources` añade comprobación previa de huellas, versiones y coherencia con los reportes. No cargar Joblib de procedencia desconocida.

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

Inferencia conecta `validate_customers(..., training_ranges=bundle.get("training_ranges"))`: advierte rangos fuera de entrenamiento sin recortar entradas. Los límites proceden solo del entrenamiento; el umbral de cargos altos 90.1 es una propuesta comercial basada en su Q3. Las bandas 0.35/0.65 siguen siendo convenciones operativas y no se presentan como optimizadas ni como la decisión binaria de umbral 0.25.

El ganador se explica con permutation sobre su probabilidad calibrada completa, referencia de 100 filas y muestra global de 100 clientes de entrenamiento. Se comprobó reconstrucción con error máximo aproximado de `2.11e-15` y consistencia individual/lote. El coste observado para 100 explicaciones en la ejecución verificada fue de unos 49 segundos; no garantiza latencia en otra máquina ni capacidad para lotes de 10.000 clientes. La web limita cada lote a 100 registros y 10 MB. La medición posterior de 100 perfiles ficticios con scoring completo tardó 34,93 segundos en esta máquina; no es una garantía de latencia en Cloud. La evidencia está en `reports/application/benchmark.json`. El lector técnico conserva su límite de 10.000 filas para otros flujos; la interfaz aplica el límite operativo menor de forma explícita.

## Notebooks y ruta de aprendizaje

El repositorio incluye los tres notebooks y copias con sus salidas verificadas, para que el docente pueda revisar tanto el procedimiento como los resultados:

| Orden | Fuente reproducible | Copia ejecutada y propósito |
|---|---|---|
| 1 | `notebooks/01_eda_limpieza.ipynb` | `reports/data/01_eda_limpieza_ejecutado.ipynb`: calidad, limpieza, derivadas y nueve gráficos de entrenamiento |
| 2 | `notebooks/02_modelos_evaluacion.ipynb` | `reports/evaluation/02_modelos_evaluacion_ejecutado.ipynb`: protocolo, CV/calibración, comparación, selección y prueba guardada |
| 3 | `notebooks/03_interpretabilidad.ipynb` | `reports/explain/03_interpretabilidad_ejecutado.ipynb`: SHAP global/individual, reconstrucción y límites |

Abrirlos con el kernel de `.venv` desde la raíz o `notebooks/`. Las fuentes no incluyen outputs. El notebook 02 consume la reproducción verificada configurada en `src/config.py`; no lanza búsquedas ni prueba final al ejecutar sus celdas. Para aprender el código, seguir el mismo orden: `schema/data/features`, `train/evaluate/reproduce`, `inference/explain/reporting`. Las reglas comerciales se mantienen en `recommendations.py`.

El código, ganador, resultados, notebooks y figuras EDA se actualizan por Git; el equipo y el docente no necesitan un ZIP separado para esos materiales. El CSV original y el entorno Python se obtienen por los comandos anteriores. Las fuentes de notebooks no contienen salidas; sus copias ejecutadas conservan tablas, figuras y advertencias, con rutas del proyecto en lugar de rutas personales.

## Desarrollo en equipo

Cada persona usa su propia copia local y una rama distinta:

| Frente | Rama sugerida | Archivos principales |
|---|---|---|
| Datos, evaluación, ganador, inferencia y SHAP | `datos-explicabilidad` | `schema.py`, `config.py`, `data.py`, `features.py`, `explain.py`, `evaluate.py`, `inference.py` |
| Entrenamiento y calibración de candidatos | `modelos-inferencia` | `train.py`, candidatos de modelos y resultados de CV |
| Web y retención | `web-retencion` | `app.py`, `ui/`, `application.py` y `recommendations.py` |

El frente 2 entrega candidatos entrenados con sus pipelines y metadatos; el frente 1 calcula métricas/curvas en validación, selecciona umbral y ganador, realiza la prueba final, exporta el artefacto y completa inferencia. Los tres frentes quedaron integrados en main; la web, retención y aceptación se completaron con aprobación posterior de la usuaria. Para nuevos cambios, cada persona debe partir del main actualizado y trabajar en una rama propia. Antes de modificar interfaces, acordar entradas, salidas y metadatos entre los tres frentes. Preservar cambios existentes y comprobar integración por hitos. No publicar archivos ajenos, secretos, entornos, cachés o temporales; revisar las rutas y el contenido preparado, añadiendo archivos explícitos. La información del repositorio se centraliza en este README. Se aprobó publicar el ganador, reportes verificados, notebooks y EDA como entrega académica; `.gitignore` permite únicamente esas rutas, no cualquier archivo futuro. `.gitattributes` conserva los bytes exactos de los artefactos para mantener las huellas entre sistemas.

## Comprobaciones

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
```

Las pruebas cubren validación, características, particiones, búsqueda/calibración/selección, persistencia, métricas, contrato de inferencia, recomendaciones, trazabilidad y SHAP real sobre modelos sintéticos de las cinco familias y calibradores. Comprueban reconstrucción, escala, consistencia individual/lote, gráficos, rechazo de insumos alterados y protección de prueba final; no miden rendimiento predictivo con Telco. La comprobación real separada es la reproducción y los notebooks de esta entrega. Las advertencias siguen tratándose como errores.

Última comprobación local (13 de noviembre de 2026): **103 pruebas pasan**, Ruff y formato correctos; flujos de la aplicación comprobados con AppTest y navegador. Los tres notebooks y sus gráficos se verificaron el 10 de noviembre; sus resultados científicos no se modificaron al completar la web.

Matplotlib se limita a versiones 3.9–3.10 por una API obsoleta usada por SHAP 0.51 en Matplotlib 3.11. No se silencian advertencias para pasar las pruebas. En scikit-learn 1.9, `SVC(probability=True)` está obsoleto: para SVM usar calibración explícita sobre el pipeline con particiones de entrenamiento separadas.

## Estado de entrega y límites

La aplicación, las recomendaciones, las pruebas de aceptación, el informe y el despliegue están completados y verificados. Para presentar el proyecto al docente, comparte este repositorio, el [informe final](reports/informe_final.pdf) y [la aplicación pública](https://chuard.streamlit.app/); el guion de demostración está arriba. Comprueba requisitos adicionales de la rúbrica del docente, si existen.

La limitación histórica es la ausencia de los archivos originales detallados del experimento y probabilidades originales de prueba; la reproducción se conserva como evidencia nueva identificada. No se inventan esos archivos ni se reutiliza prueba para compensar su ausencia. Acordar con el equipo los criterios comerciales de riesgo y los límites operativos de la aplicación. Push, merges y despliegues requieren autorización expresa.

Las pruebas de web ejecutan el modelo congelado sobre perfiles ficticios y verifican los flujos con Streamlit AppTest; no evalúan otra vez la prueba final. La revisión en navegador comprueba la presentación de escritorio y móvil.

Comprobación remota del 13 de noviembre de 2026: acceso anónimo, predicción, carga real del CSV ficticio, scoring de seis filas, descarga, SHAP global/individual, selección de otro cliente, cola de cinco activos, aislamiento y borrado de sesiones. La comparación del CSV Cloud/Linux con Windows arroja una diferencia máxima de probabilidad de `5,56e-17` y de valores SHAP de `1,60e-10` (no idénticos bit a bit; dentro de la tolerancia SHAP existente de `1e-6`), con iguales recomendaciones y orden de factores. No se cambiaron pruebas, tolerancias, modelo ni evaluación final. Los tiempos medidos localmente no garantizan latencia de Cloud.
