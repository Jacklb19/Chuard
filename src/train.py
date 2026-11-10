"""Particiones, búsqueda, calibración, selección y exportación; no entrena al importar."""

import argparse
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import (
    CONTRACT_VERSION,
    CV_FOLDS,
    DEFAULT_RISK_LIMITS,
    LOGISTIC_MAX_ITER,
    MODEL_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SPLIT_MANIFEST_PATH,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)
from src.data import dataset_sha256, read_customer_csv, summarize_training
from src.data import training_subset_from_manifest as _training_subset
from src.evaluate import (
    THRESHOLD_GRID,
    calibration_summary,
    classification_metrics,
    probability_metrics,
    save_calibration_curves,
    save_confusion_matrix,
    save_roc_curves,
    select_threshold,
)
from src.explain import prepare_explanation_context
from src.features import FeatureBuilder
from src.schema import DERIVED_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS

# --- Protocolo experimental (frente 2). Fijado antes de abrir validación o prueba. ---
SEARCH_ITERATIONS = 20  # combinaciones aleatorias por familia y variante
SEARCH_N_JOBS = -2  # paralelismo solo en búsqueda/CV; estimadores internos con n_jobs=1
SELECTION_TIE_TOLERANCE = 0.005  # diferencias menores se consideran empate
CALIBRATION_METHOD = "sigmoid"  # Platt; isotónica descartada por el tamaño de los folds
CALIBRATION_MIN_RELATIVE_GAIN = 0.01  # calibrar solo si Brier y log-loss mejoran ≥ 1 %
# Lote vs. fila: BLAS suma en otro orden según el tamaño (diferencias ~1e-16 observadas);
# 1e-12 queda muy por debajo del paso 0.01 de la rejilla de umbrales.
SCORING_ATOL = 1e-12
SCORING_CHECK_ROWS = 3  # filas de prueba reutilizadas solo como chequeo técnico de scoring
MODEL_VERSION = "1.0"
MODELING_DIR = SPLIT_MANIFEST_PATH.parent
EXPERIMENT_PATH = MODELING_DIR / "experiment.json"
VALIDATION_PROBABILITIES_PATH = MODELING_DIR / "validation_probabilities.csv"
TEST_REPORT_PATH = MODELING_DIR / "test_report.json"
METRICS_TABLE_PATH = PROJECT_ROOT / "reports/metricas_modelos.csv"
RUN_HISTORY_PATH = MODELING_DIR / "run_history.jsonl"
# Orden de parsimonia para el último desempate (más simple primero).
FAMILY_ORDER = ("logistic_regression", "random_forest", "lightgbm", "xgboost", "svm")
FAMILY_LABELS = {
    "logistic_regression": "Regresión Logística",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "svm": "SVM",
}
# Escalado para modelos sensibles a magnitudes; los árboles no lo necesitan.
SCALED_FAMILIES = {"logistic_regression", "svm"}
# SVC no tiene probabilidades propias: Platt (sigmoid) es su único mecanismo.
FAMILIES_WITHOUT_PROBABILITY = {"svm"}
VARIANTS = {False: "sin_derivadas", True: "con_derivadas"}
UNCALIBRATED = "sin_calibrar"
LIBRARIES = ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "shap", "joblib")


def split_positions(y, *, random_state: int = RANDOM_STATE) -> dict[str, list[int]]:
    """60/20/20 con redondeo de scikit-learn; reutilizar esta única asignación."""
    fractions = [TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION]
    if not all(value > 0 for value in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("Las fracciones deben ser positivas y sumar 1.")
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("La partición necesita ambas clases binarias.")
    positions = np.arange(len(y))
    train, remaining = train_test_split(
        positions,
        test_size=1 - TRAIN_FRACTION,
        random_state=random_state,
        stratify=y,
    )
    validation, test = train_test_split(
        remaining,
        test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
        random_state=random_state,
        stratify=y[remaining],
    )
    return {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()}


def make_pipeline(estimator=None, *, use_derived: bool = False, scale: bool = True) -> Pipeline:
    """Pipeline sin ajustar; la limpieza determinística de entrada precede a esta API."""
    numeric = list(NUMERIC_COLUMNS) + (list(DERIVED_COLUMNS) if use_derived else [])
    categorical = [name for name in FEATURE_COLUMNS if name not in NUMERIC_COLUMNS]
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    if estimator is None:
        estimator = LogisticRegression(random_state=RANDOM_STATE, max_iter=LOGISTIC_MAX_ITER)
    return Pipeline(
        [
            ("features", FeatureBuilder(enabled=use_derived)),
            ("preprocess", preprocessor),
            ("classifier", estimator),
        ]
    )


def base_estimator(family: str, *, random_state: int = RANDOM_STATE):
    """Estimador sin ajustar y con un solo hilo, para no anidar paralelismo."""
    if family == "logistic_regression":
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=random_state)
    if family == "random_forest":
        return RandomForestClassifier(random_state=random_state, n_jobs=1)
    if family == "xgboost":
        return XGBClassifier(
            tree_method="hist", eval_metric="logloss", random_state=random_state, n_jobs=1
        )
    if family == "lightgbm":
        return LGBMClassifier(random_state=random_state, n_jobs=1, verbosity=-1)
    if family == "svm":
        return SVC()
    raise ValueError(f"Familia de modelo desconocida: {family}.")


def search_space(family: str, positive_weight: float) -> dict[str, list]:
    """Rejillas discretas: muestreo reproducible y parámetros serializables en JSON."""

    def logspace(low: int, high: int, count: int) -> list[float]:
        return [float(value) for value in np.round(np.logspace(low, high, count), 6)]

    if not np.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("El peso de la clase positiva debe ser finito y positivo.")
    spaces = {
        "logistic_regression": {
            "C": logspace(-3, 2, 26),
            "class_weight": [None, "balanced"],
        },
        "random_forest": {
            "n_estimators": list(range(200, 601, 100)),
            "max_depth": [None, 4, 6, 8, 10, 12, 16],
            "min_samples_leaf": list(range(1, 21)),
            "class_weight": [None, "balanced", "balanced_subsample"],
        },
        "xgboost": {
            "n_estimators": list(range(100, 601, 50)),
            "max_depth": list(range(2, 9)),
            "learning_rate": logspace(-2, np.log10(0.3), 12),
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "scale_pos_weight": [1.0, round(float(positive_weight), 4)],
        },
        "lightgbm": {
            "n_estimators": list(range(100, 601, 50)),
            "num_leaves": list(range(8, 65, 4)),
            "learning_rate": logspace(-2, np.log10(0.3), 12),
            "min_child_samples": list(range(5, 101, 5)),
            "class_weight": [None, "balanced"],
        },
        "svm": {
            "kernel": ["rbf", "linear"],
            "C": logspace(-2, 1, 13),
            "gamma": logspace(-3, 0, 13),
            "class_weight": [None, "balanced"],
        },
    }
    if family not in spaces:
        raise ValueError(f"Familia de modelo desconocida: {family}.")
    return {f"classifier__{name}": values for name, values in spaces[family].items()}


def build_candidate(
    family: str,
    use_derived: bool,
    params: dict,
    calibration: str,
    *,
    random_state: int = RANDOM_STATE,
):
    """Estimador completo sobre las 19 entradas; el calibrador envuelve todo el pipeline."""
    if calibration not in (UNCALIBRATED, CALIBRATION_METHOD):
        raise ValueError("Calibración no soportada por el protocolo.")
    if family in FAMILIES_WITHOUT_PROBABILITY and calibration == UNCALIBRATED:
        raise ValueError("SVM necesita Platt para producir probabilidades.")
    pipeline = make_pipeline(
        base_estimator(family, random_state=random_state),
        use_derived=use_derived,
        scale=family in SCALED_FAMILIES,
    ).set_params(**params)
    if calibration == UNCALIBRATED:
        return pipeline
    # ensemble=False: un único pipeline ajustado en todo el entrenamiento y un calibrador
    # aprendido con predicciones fuera de fold internas (reemplazo de SVC(probability=True)).
    return CalibratedClassifierCV(
        pipeline,
        method=CALIBRATION_METHOD,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=random_state),
        ensemble=False,
    )


def positive_probability(estimator, frame: pd.DataFrame) -> np.ndarray:
    """Probabilidad de Churn=1 según el orden real de classes_, nunca por posición fija."""
    classes = np.asarray(estimator.classes_)
    positive = np.flatnonzero(classes == 1)
    if classes.shape != (2,) or len(positive) != 1 or not np.isin(classes, [0, 1]).all():
        raise ValueError("Se requiere un estimador binario con clase positiva 1.")
    probabilities = np.asarray(estimator.predict_proba(frame), dtype=float)[:, positive[0]]
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("El estimador devolvió probabilidades inválidas.")
    return probabilities


def calibration_decision(oof: dict[str, dict], *, family: str) -> str:
    """Regla fija: Platt solo si Brier y log-loss fuera de fold mejoran a la vez."""
    if family in FAMILIES_WITHOUT_PROBABILITY:
        return CALIBRATION_METHOD
    raw, calibrated = oof[UNCALIBRATED], oof[CALIBRATION_METHOD]
    gains = [(raw[key] - calibrated[key]) / raw[key] for key in ("brier", "log_loss")]
    if all(gain >= CALIBRATION_MIN_RELATIVE_GAIN for gain in gains):
        return CALIBRATION_METHOD
    return UNCALIBRATED


def select_winner(
    candidates: list[dict], *, tolerance: float = SELECTION_TIE_TOLERANCE
) -> tuple[dict, list[str]]:
    """F1 de validación; empates → ROC-AUC → Recall → estabilidad CV → parsimonia."""
    pool = [candidate for candidate in candidates if candidate["eligible"]]
    if not pool:
        raise ValueError("No hay candidatos elegibles para seleccionar.")
    trace = []
    for key, label in (
        ("validation_f1", "F1 validación"),
        ("validation_roc_auc", "ROC-AUC validación"),
        ("validation_recall", "Recall validación"),
    ):
        best = max(candidate[key] for candidate in pool)
        pool = [candidate for candidate in pool if candidate[key] >= best - tolerance]
        trace.append(
            f"{label}: mejor {best:.4f}; quedan {len(pool)} dentro de ±{tolerance} "
            f"({', '.join(candidate['key'] for candidate in pool)})."
        )
    pool.sort(
        key=lambda candidate: (
            candidate["cv_f1_std"],
            FAMILY_ORDER.index(candidate["family"]),
            candidate["use_derived"],
        )
    )
    if len(pool) > 1:
        trace.append("Desempate final: menor desviación de F1 en CV, luego familia más simple.")
    return pool[0], trace


@dataclass
class Partitions:
    """Datos normalizados de la asignación congelada; train conserva su procedencia."""

    frames: dict[str, pd.DataFrame]
    dataset_sha256: str
    manifest_sha256: str

    def features(self, name: str) -> pd.DataFrame:
        return self.frames[name].loc[:, list(FEATURE_COLUMNS)]

    def target(self, name: str) -> np.ndarray:
        return self.frames[name]["Churn"].map({"No": 0, "Yes": 1}).to_numpy(dtype=int)


def load_partitions(data_path: Path, manifest_path: Path) -> Partitions:
    """Verifica huella, cobertura y disyunción mediante el lector del frente 1."""
    training = _training_subset(data_path, manifest_path)
    data = read_customer_csv(data_path, mode="training").data
    splits = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["splits"]
    frames = {
        "train": training,
        "validation": data.iloc[splits["validation"]].copy(),
        "test": data.iloc[splits["test"]].copy(),
    }
    return Partitions(frames, training.attrs["dataset_sha256"], training.attrs["manifest_sha256"])


def _search_scoring() -> dict:
    # zero_division explícito: un fold sin positivos predichos puntúa F1 = 0 sin aviso.
    return {"f1": make_scorer(f1_score, zero_division=0), "roc_auc": "roc_auc", "recall": "recall"}


def _cv_summary(search: RandomizedSearchCV) -> dict:
    results, best = search.cv_results_, search.best_index_
    summary = {}
    for metric in ("f1", "roc_auc", "recall"):
        folds = [float(results[f"split{i}_test_{metric}"][best]) for i in range(CV_FOLDS)]
        summary[metric] = {
            "mean": float(np.mean(folds)),
            "std": float(np.std(folds)),
            "folds": folds,
        }
    return summary


def _protocol(n_iter: int) -> dict:
    return {
        "partition": "60/20/20 estratificada, semilla única del manifiesto",
        "search": f"RandomizedSearchCV, {n_iter} combinaciones, StratifiedKFold({CV_FOLDS}) "
        "solo en entrenamiento; métrica F1 de Churn=1",
        "calibration": f"Predicciones fuera de fold en entrenamiento; {CALIBRATION_METHOD} "
        f"(CalibratedClassifierCV ensemble=False, CV interna {CV_FOLDS}) si Brier y log-loss "
        f"mejoran ≥ {CALIBRATION_MIN_RELATIVE_GAIN:.0%}; SVM siempre usa Platt",
        "threshold": select_threshold.__doc__.strip(),
        "selection": select_winner.__doc__.strip()
        + f" Tolerancia de empate {SELECTION_TIE_TOLERANCE}.",
        "test": "Una sola evaluación del ganador congelado; sin reajuste posterior.",
        "random_state": RANDOM_STATE,
        "threshold_grid": [THRESHOLD_GRID[0], THRESHOLD_GRID[-1], len(THRESHOLD_GRID)],
    }


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {name: metadata.version(name) for name in LIBRARIES},
    }


def _write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")


def run_experiment(
    partitions: Partitions,
    output_dir: Path = MODELING_DIR,
    *,
    metrics_table: Path = METRICS_TABLE_PATH,
    families: tuple[str, ...] = FAMILY_ORDER,
    n_iter: int = SEARCH_ITERATIONS,
    n_jobs: int = SEARCH_N_JOBS,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Búsqueda y calibración en entrenamiento, umbral y comparación en validación."""
    output_dir = Path(output_dir)
    if (output_dir / TEST_REPORT_PATH.name).exists():
        raise FileExistsError("La prueba final ya se usó; no se vuelve a seleccionar.")
    if not families or any(family not in FAMILY_LABELS for family in families):
        raise ValueError("Familias desconocidas o vacías.")
    if n_iter < 1:
        raise ValueError("Se requiere al menos una combinación por búsqueda.")
    x_train, y_train = partitions.features("train"), partitions.target("train")
    x_val, y_val = partitions.features("validation"), partitions.target("validation")
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=random_state)
    positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    candidates, search_rows, probabilities = [], [], {}
    roc_curves, calibration_curves = {}, {}
    for family in families:
        for use_derived in (False, True):
            key = f"{family}__{VARIANTS[use_derived]}"
            started = time.perf_counter()
            search = RandomizedSearchCV(
                make_pipeline(
                    base_estimator(family, random_state=random_state),
                    use_derived=use_derived,
                    scale=family in SCALED_FAMILIES,
                ),
                search_space(family, positive_weight),
                n_iter=n_iter,
                scoring=_search_scoring(),
                refit="f1",
                cv=cv,
                n_jobs=n_jobs,
                random_state=random_state,
                error_score=np.nan,
            )
            search.fit(x_train, y_train)
            results = search.cv_results_
            failed = int(np.isnan(results["mean_test_f1"]).sum())
            for index, params in enumerate(results["params"]):
                search_rows.append(
                    {
                        "candidate": key,
                        "params": json.dumps(params, sort_keys=True),
                        **{
                            f"cv_{metric}_{stat}": float(results[f"{stat}_test_{metric}"][index])
                            for metric in ("f1", "roc_auc", "recall")
                            for stat in ("mean", "std")
                        },
                        "rank_f1": int(results["rank_test_f1"][index]),
                    }
                )
            params = dict(search.best_params_)
            oof = {}
            mechanisms = (
                (CALIBRATION_METHOD,)
                if family in FAMILIES_WITHOUT_PROBABILITY
                else (UNCALIBRATED, CALIBRATION_METHOD)
            )
            oof_probabilities = {}
            for mechanism in mechanisms:
                predicted = cross_val_predict(
                    build_candidate(
                        family, use_derived, params, mechanism, random_state=random_state
                    ),
                    x_train,
                    y_train,
                    cv=cv,
                    method="predict_proba",
                    n_jobs=n_jobs,
                )[:, 1]
                oof_probabilities[mechanism] = predicted
                oof[mechanism] = {
                    **probability_metrics(y_train, predicted),
                    "curve": calibration_summary(y_train, predicted),
                }
            calibration = calibration_decision(oof, family=family)
            calibration_curves[f"{key} ({calibration})"] = (
                y_train,
                oof_probabilities[calibration],
            )
            estimator = build_candidate(
                family, use_derived, params, calibration, random_state=random_state
            ).fit(x_train, y_train)
            validation_probability = positive_probability(estimator, x_val)
            threshold = select_threshold(y_val, validation_probability)
            metrics = classification_metrics(
                y_val, validation_probability, threshold=threshold["threshold"]
            )
            cv_summary = _cv_summary(search)
            probabilities[key] = validation_probability
            roc_curves[key] = (y_val, validation_probability)
            candidates.append(
                {
                    "key": key,
                    "family": family,
                    "label": FAMILY_LABELS[family],
                    "use_derived": use_derived,
                    "variant": VARIANTS[use_derived],
                    "eligible": True,
                    "params": params,
                    "failed_parameter_sets": failed,
                    "cv": cv_summary,
                    "cv_f1_mean": cv_summary["f1"]["mean"],
                    "cv_f1_std": cv_summary["f1"]["std"],
                    "oof_calibration": oof,
                    "calibration": calibration,
                    "threshold": threshold,
                    "validation": {**metrics, **probability_metrics(y_val, validation_probability)},
                    "validation_f1": metrics["f1"],
                    "validation_roc_auc": metrics["roc_auc"],
                    "validation_recall": metrics["recall"],
                    "seconds": round(time.perf_counter() - started, 1),
                }
            )
            print(f"{key}: F1 validación {metrics['f1']:.4f} (umbral {threshold['threshold']}).")
    baseline = DummyClassifier(strategy="prior").fit(x_train, y_train)
    baseline_probability = positive_probability(baseline, x_val)
    baseline_threshold = select_threshold(y_val, baseline_probability)
    baseline_metrics = classification_metrics(
        y_val, baseline_probability, threshold=baseline_threshold["threshold"]
    )
    candidates.append(
        {
            "key": "dummy_prior",
            "family": "dummy",
            "label": "Referencia (proporción de clases)",
            "use_derived": False,
            "variant": "referencia",
            "eligible": False,
            "threshold": baseline_threshold,
            "validation": baseline_metrics,
            "validation_f1": baseline_metrics["f1"],
            "validation_roc_auc": baseline_metrics["roc_auc"],
            "validation_recall": baseline_metrics["recall"],
        }
    )
    winner, trace = select_winner(candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    save_roc_curves(roc_curves, figures / "roc_validacion.png", title="Curvas ROC — validación")
    save_calibration_curves(
        calibration_curves,
        figures / "calibracion_entrenamiento_oof.png",
        title="Calibración fuera de fold — entrenamiento",
    )
    save_confusion_matrix(
        winner["validation"]["confusion_matrix"],
        figures / "matriz_validacion_ganador.png",
        title=f"{winner['label']} ({winner['variant']}) — validación, umbral "
        f"{winner['threshold']['threshold']:.2f}",
    )
    pd.DataFrame(probabilities).to_csv(output_dir / VALIDATION_PROBABILITIES_PATH.name, index=False)
    pd.DataFrame(search_rows).to_csv(output_dir / "search_cv_results.csv", index=False)
    table = pd.DataFrame(
        [
            {
                "particion": "validacion",
                "candidato": candidate["key"],
                "modelo": candidate["label"],
                "variante": candidate["variant"],
                "elegible": candidate["eligible"],
                "calibracion": candidate.get("calibration"),
                "umbral": candidate["threshold"]["threshold"],
                **{
                    metric: candidate["validation"][metric]
                    for metric in (
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                        "average_precision",
                    )
                },
                "brier": candidate["validation"].get("brier"),
                "cv_f1_media": candidate.get("cv_f1_mean"),
                "cv_f1_desviacion": candidate.get("cv_f1_std"),
                "ganador": candidate["key"] == winner["key"],
            }
            for candidate in candidates
        ]
    )
    metrics_table = Path(metrics_table)
    metrics_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_table, index=False)
    record = {
        "status": "validacion_completa_prueba_sin_usar",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_sha256": partitions.dataset_sha256,
        "split_manifest_sha256": partitions.manifest_sha256,
        "rows": {name: len(frame) for name, frame in partitions.frames.items()},
        "positive_weight_train": positive_weight,
        "protocol": _protocol(n_iter),
        "environment": _environment(),
        "candidates": candidates,
        "winner": winner["key"],
        "selection_trace": trace,
        "warning": "Las métricas de validación sirvieron para elegir umbral y modelo: son "
        "optimistas y no sustituyen la evaluación final en prueba.",
    }
    _write_json(output_dir / EXPERIMENT_PATH.name, record)
    return record


def _transformed_names(estimator) -> list[str]:
    pipeline = (
        estimator.calibrated_classifiers_[0].estimator
        if isinstance(estimator, CalibratedClassifierCV)
        else estimator
    )
    return pipeline[:-1].get_feature_names_out().tolist()


def build_bundle(estimator, winner: dict, experiment: dict, partitions: Partitions, test: dict):
    """Bundle con los campos mínimos del contrato y metadatos propuestos para ampliarlo."""
    summary = summarize_training(partitions.frames["train"])
    return {
        "pipeline": estimator,
        "feature_columns": list(FEATURE_COLUMNS),
        "model_name": winner["key"],
        "model_version": MODEL_VERSION,
        "contract_version": CONTRACT_VERSION,
        "decision_threshold": winner["threshold"]["threshold"],
        "risk_limits": dict(DEFAULT_RISK_LIMITS),
        "risk_limits_origin": "Valores iniciales de config.py; no ajustados experimentalmente.",
        "model_family": winner["family"],
        "model_label": winner["label"],
        "model_variant": winner["variant"],
        "use_derived": winner["use_derived"],
        "classes": np.asarray(estimator.classes_).tolist(),
        "positive_class": 1,
        "calibration": {
            "method": winner["calibration"],
            "cv_folds": CV_FOLDS if winner["calibration"] != UNCALIBRATED else None,
            "ensemble": False if winner["calibration"] != UNCALIBRATED else None,
            "fitted_on": "train",
        },
        "hyperparameters": winner["params"],
        "transformed_feature_names": _transformed_names(estimator),
        "random_state": RANDOM_STATE,
        "environment": _environment(),
        "dataset_sha256": partitions.dataset_sha256,
        "split_manifest_sha256": partitions.manifest_sha256,
        "metrics": {
            "cv_train": winner["cv"],
            "validation": winner["validation"],
            "test": test,
        },
        "training_ranges": summary["training_ranges"],
        "recommendation_context": {
            "high_monthly_charges": summary["high_monthly_charges"],
            "high_monthly_charges_rule": summary["high_monthly_charges_rule"],
        },
        "explanation_context": prepare_explanation_context(partitions.frames["train"]),
        "protocol": experiment["protocol"],
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def _check_single_equals_batch(sample: pd.DataFrame, bundle: dict, score) -> None:
    """Misma fila, mismo resultado; la probabilidad admite solo ruido de redondeo."""
    batch = score(sample, bundle)
    single = pd.concat([score(sample.iloc[[i]], bundle) for i in range(len(sample))])
    probability_ok = np.allclose(
        batch["churn_probability"].to_numpy(),
        single["churn_probability"].to_numpy(),
        rtol=0,
        atol=SCORING_ATOL,
    )
    # SHAP se calcula fila a fila en ambos casos: factores y decisiones deben ser idénticos.
    exact = all(
        batch[column].tolist() == single[column].tolist()
        for column in (
            "customerID",
            "predicted_churn",
            "risk_level",
            "top_factors",
            "recommendations",
        )
    )
    if not (probability_ok and exact):
        raise ValueError("El scoring individual y por lote no coinciden.")


def run_final(
    partitions: Partitions,
    output_dir: Path = MODELING_DIR,
    model_path: Path = MODEL_PATH,
    *,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Congela el ganador, lo evalúa una vez en prueba y exporta exactamente ese estimador."""
    from src.inference import load_model_bundle, score_customers

    output_dir, model_path = Path(output_dir), Path(model_path)
    test_report = output_dir / TEST_REPORT_PATH.name
    if test_report.exists() or model_path.exists():
        raise FileExistsError(
            "La prueba final o el modelo ya existen; no se repite ni sobrescribe."
        )
    experiment = json.loads((output_dir / EXPERIMENT_PATH.name).read_text(encoding="utf-8"))
    if (
        experiment["dataset_sha256"] != partitions.dataset_sha256
        or experiment["split_manifest_sha256"] != partitions.manifest_sha256
    ):
        raise ValueError("El experimento no corresponde al CSV o al manifiesto actuales.")
    winner = next(item for item in experiment["candidates"] if item["key"] == experiment["winner"])
    estimator = build_candidate(
        winner["family"],
        winner["use_derived"],
        winner["params"],
        winner["calibration"],
        random_state=random_state,
    ).fit(partitions.features("train"), partitions.target("train"))
    recorded = pd.read_csv(output_dir / VALIDATION_PROBABILITIES_PATH.name)[winner["key"]]
    validation_probability = positive_probability(estimator, partitions.features("validation"))
    if not np.allclose(validation_probability, recorded.to_numpy(), rtol=0, atol=1e-12):
        raise ValueError("El estimador reconstruido no reproduce la validación registrada.")
    x_test, y_test = partitions.features("test"), partitions.target("test")
    test_probability = positive_probability(estimator, x_test)
    test = {
        **classification_metrics(
            y_test, test_probability, threshold=winner["threshold"]["threshold"]
        ),
        **probability_metrics(y_test, test_probability),
    }
    bundle = build_bundle(estimator, winner, experiment, partitions, test)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Archivo temporal: el bundle definitivo solo aparece si pasa todas las comprobaciones.
    pending = model_path.with_name(model_path.name + ".pendiente")
    try:
        with pending.open("xb") as stream:
            joblib.dump(bundle, stream)
        reloaded = load_model_bundle(str(pending))
        if not np.array_equal(positive_probability(reloaded["pipeline"], x_test), test_probability):
            raise ValueError("El bundle recargado no reproduce las probabilidades de prueba.")
        _check_single_equals_batch(
            partitions.frames["test"].iloc[:SCORING_CHECK_ROWS], reloaded, score_customers
        )
        pending.replace(model_path)
    finally:
        pending.unlink(missing_ok=True)
    figures = output_dir / "figures"
    save_roc_curves(
        {winner["key"]: (y_test, test_probability)},
        figures / "roc_prueba_ganador.png",
        title="Curva ROC — prueba final (ganador congelado)",
    )
    save_confusion_matrix(
        test["confusion_matrix"],
        figures / "matriz_prueba_ganador.png",
        title=f"{winner['label']} ({winner['variant']}) — prueba, umbral "
        f"{winner['threshold']['threshold']:.2f}",
    )
    record = {
        "status": "prueba_final_evaluada_una_vez",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "winner": winner["key"],
        "decision_threshold": winner["threshold"]["threshold"],
        "calibration": winner["calibration"],
        "test": test,
        "model_path": model_path.name,
        "model_sha256": dataset_sha256(model_path),
        "checks": {
            "validation_reproduced": True,
            "reload_identical_on_test": True,
            "single_equals_batch_scoring_rows": SCORING_CHECK_ROWS,
            "single_equals_batch_probability_atol": SCORING_ATOL,
        },
    }
    _write_json(test_report, record)
    manifest = {
        key: value
        for key, value in bundle.items()
        if key not in {"pipeline", "explanation_context"}
    } | {"model_sha256": record["model_sha256"], "file": model_path.name}
    _write_json(model_path.with_suffix(".manifest.json"), manifest)
    return record


def _log_run(action: str, status: str, started: float, detail: str = "") -> None:
    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "at_utc": datetime.now(UTC).isoformat(),
        "action": action,
        "status": status,
        "seconds": round(time.perf_counter() - started, 1),
        "detail": detail,
    }
    with RUN_HISTORY_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _save_splits(dataset_path: Path, output_path: Path, parser: argparse.ArgumentParser) -> None:
    data = read_customer_csv(dataset_path, mode="training").data
    if data["customerID"].duplicated().any():
        parser.error("Resolver IDs duplicados y documentar el criterio antes de dividir.")
    if output_path.exists():
        parser.error("El manifiesto ya existe; no se sobrescribe una partición congelada.")
    manifest = {
        "dataset_sha256": dataset_sha256(dataset_path),
        "random_state": RANDOM_STATE,
        "index_kind": "posición normalizada desde 0",
        "splits": split_positions(data["Churn"].map({"No": 0, "Yes": 1})),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Particiones guardadas.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimentos de Chuard. Sin subcomando solo guarda las particiones."
    )
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH)
    parser.add_argument(
        "--manifest", "--output", dest="manifest", type=Path, default=SPLIT_MANIFEST_PATH
    )
    parser.set_defaults(action="splits")
    actions = parser.add_subparsers(dest="action")
    actions.add_parser("splits", help="Guarda la asignación 60/20/20; no la sobrescribe.")
    experiment = actions.add_parser(
        "experiment", help="Búsqueda/calibración en entrenamiento y selección en validación."
    )
    experiment.add_argument("--families", nargs="+", choices=FAMILY_ORDER, default=FAMILY_ORDER)
    experiment.add_argument("--n-iter", type=int, default=SEARCH_ITERATIONS)
    experiment.add_argument("--n-jobs", type=int, default=SEARCH_N_JOBS)
    final = actions.add_parser(
        "final", help="Evalúa una vez el ganador en prueba y exporta models/best_model.joblib."
    )
    final.add_argument(
        "--protocol-agreed",
        action="store_true",
        help="Confirma que el equipo aprobó el protocolo antes de usar prueba.",
    )
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("Falta el CSV original. Consulta README.md.")
    if args.action == "splits":
        _save_splits(args.data, args.manifest, parser)
        return
    if not args.manifest.is_file():
        parser.error("Falta el manifiesto de particiones; ejecuta primero el subcomando splits.")
    if args.action == "final" and not args.protocol_agreed:
        parser.error("La prueba final requiere --protocol-agreed tras acordar el protocolo.")
    started = time.perf_counter()
    try:
        partitions = load_partitions(args.data, args.manifest)
        if args.action == "experiment":
            record = run_experiment(
                partitions, families=tuple(args.families), n_iter=args.n_iter, n_jobs=args.n_jobs
            )
            detail = f"ganador en validación: {record['winner']}"
        else:
            record = run_final(partitions)
            detail = f"prueba evaluada: {record['winner']}"
    except Exception as exc:
        _log_run(args.action, "fallido", started, f"{type(exc).__name__}: {exc}")
        raise
    _log_run(args.action, "completado", started, detail)
    print(detail)


if __name__ == "__main__":
    main()
