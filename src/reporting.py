"""Entrega científica del ganador propio: verifica insumos, no entrena ni puntúa prueba."""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    EVALUATION_REPORT_DIR,
    EXPLANATION_REPORT_DIR,
    FIGURE_DPI,
    MODEL_MANIFEST_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORT_METRICS_ATOL,
    SHAP_ADDITIVITY_ATOL,
    SHAP_MAX_GLOBAL_ROWS,
    SPLIT_MANIFEST_PATH,
)
from src.data import dataset_sha256, read_customer_csv, training_subset_from_manifest
from src.evaluate import (
    classification_metrics,
    probability_metrics,
    save_calibration_curves,
    save_confusion_matrix,
    save_roc_curves,
)
from src.explain import get_explanation_service, save_explanation_plots
from src.inference import load_model_bundle, score_customers
from src.schema import FEATURE_COLUMNS
from src.train import positive_probability, split_positions


def verify_file(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Falta el recurso local: {path}.")
    if dataset_sha256(path) != expected:
        raise ValueError(f"Huella incompatible: {path.name}.")


def restore_splits(data_path: Path, model_manifest: Path, output: Path) -> str:
    """Reconstruye únicamente si los bytes coinciden con la huella congelada."""
    meta = json.loads(model_manifest.read_text(encoding="utf-8"))
    verify_file(data_path, meta["dataset_sha256"])
    expected = meta["split_manifest_sha256"]
    if output.exists():
        verify_file(output, expected)
        return expected
    data = read_customer_csv(data_path, mode="training").data
    record = {
        "dataset_sha256": meta["dataset_sha256"],
        "random_state": meta["random_state"],
        "index_kind": "posición normalizada desde 0",
        "splits": split_positions(
            data["Churn"].map({"No": 0, "Yes": 1}), random_state=meta["random_state"]
        ),
    }
    text = json.dumps(record, indent=2)
    for newline in ("\n", "\r\n"):
        payload = text.replace("\n", newline).encode("utf-8")
        if hashlib.sha256(payload).hexdigest() == expected:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                stream.write(payload)
            return expected
    raise ValueError("No se pudo reconstruir el manifiesto exacto; solicite el original al equipo.")


@dataclass
class FrozenResources:
    bundle: dict
    manifest: dict
    training: pd.DataFrame
    validation: pd.DataFrame
    comparison: pd.DataFrame
    final_report: pd.DataFrame


def _assert_metrics(actual: dict, expected: dict) -> None:
    for name, value in actual.items():
        if name not in expected or not np.allclose(
            value, expected[name], atol=REPORT_METRICS_ATOL, rtol=0
        ):
            raise ValueError(f"La métrica {name} no reproduce el reporte congelado.")


def load_frozen_resources(
    *,
    data_path: Path = RAW_DATA_PATH,
    split_path: Path = SPLIT_MANIFEST_PATH,
    model_path: Path = MODEL_PATH,
    model_manifest: Path = MODEL_MANIFEST_PATH,
    report_dir: Path = EVALUATION_REPORT_DIR,
) -> FrozenResources:
    """Solo artefactos propios. Verifica SHA/versiones antes de deserializar Joblib."""
    meta = json.loads(model_manifest.read_text(encoding="utf-8"))
    for path, expected in (
        (data_path, meta["dataset_sha256"]),
        (split_path, meta["split_manifest_sha256"]),
        (model_path, meta["model_sha256"]),
        (report_dir / "validacion_candidatos.csv", meta["validation_report_sha256"]),
        (report_dir / "prueba_final.csv", meta["reserved_test"]["report_sha256"]),
    ):
        verify_file(path, expected)
    environment = meta["environment"]
    if platform.python_version() != environment["python"]:
        raise ValueError("Versión de Python distinta de la del artefacto congelado.")
    for name, version in environment["libraries"].items():
        if metadata.version(name) != version:
            raise ValueError(f"Versión incompatible con el artefacto: {name}.")
    bundle = load_model_bundle(str(model_path))
    for key in (
        "model_name",
        "model_version",
        "contract_version",
        "decision_threshold",
        "dataset_sha256",
        "split_manifest_sha256",
        "hyperparameters",
        "risk_limits",
    ):
        if bundle[key] != meta[key]:
            raise ValueError(f"Bundle y manifiesto discrepan en {key}.")
    training = training_subset_from_manifest(data_path, split_path)
    splits = json.loads(split_path.read_text(encoding="utf-8"))["splits"]
    validation = read_customer_csv(data_path, mode="training").data.iloc[splits["validation"]]
    context = bundle["explanation_context"]
    if context["origin"] != "training" or context["training_rows"] != len(training):
        raise ValueError("La referencia SHAP no acredita origen de entrenamiento.")
    expected_background = training.iloc[context["background_positions_in_training"]]
    try:
        pd.testing.assert_frame_equal(
            expected_background.loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True),
            context["background"].reset_index(drop=True),
        )
    except AssertionError as exc:
        raise ValueError("La referencia SHAP no coincide con el entrenamiento congelado.") from exc
    comparison = pd.read_csv(report_dir / "validacion_candidatos.csv")
    final = pd.read_csv(report_dir / "prueba_final.csv")
    if len(comparison) != 10 or comparison["candidate"].duplicated().any() or len(final) != 1:
        raise ValueError("Se requieren diez candidatos distintos y un único reporte final.")
    for key, expected in meta["metrics"]["test"].items():
        actual = final.iloc[0][key]
        if isinstance(expected, str):
            if actual != expected:
                raise ValueError(f"El reporte final discrepa en {key}.")
        else:
            _assert_metrics({key: actual}, {key: expected})
    return FrozenResources(bundle, meta, training, validation, comparison, final)


def save_comparison(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = table.sort_values("f1")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for axis, metric, label in zip(
        axes,
        ("f1", "roc_auc", "recall"),
        ("F1 positivo", "ROC-AUC", "Recall positivo"),
        strict=True,
    ):
        axis.barh(ordered["candidate"], ordered[metric], color="#256d85")
        axis.set(xlabel=f"{label} (proporción)", xlim=(0, 1))
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Diez candidatos — validación usada para selección (no prueba)")
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def generate_frozen_reports(
    resources: FrozenResources,
    *,
    evaluation_dir: Path = EVALUATION_REPORT_DIR,
    explanation_dir: Path = EXPLANATION_REPORT_DIR,
    global_rows: int = SHAP_MAX_GLOBAL_ROWS,
) -> dict:
    """Reproduce validación; prueba solo se lee del reporte verificado existente."""
    if not 1 <= global_rows <= SHAP_MAX_GLOBAL_ROWS:
        raise ValueError(f"La muestra SHAP debe tener entre 1 y {SHAP_MAX_GLOBAL_ROWS} filas.")
    bundle, meta = resources.bundle, resources.manifest
    sample = resources.training.sample(
        n=min(global_rows, len(resources.training)), random_state=RANDOM_STATE
    )
    y = resources.validation["Churn"].map({"No": 0, "Yes": 1}).to_numpy()
    x = resources.validation.loc[:, list(FEATURE_COLUMNS)]
    probabilities = positive_probability(bundle["pipeline"], x)
    metrics = {
        **classification_metrics(y, probabilities, threshold=bundle["decision_threshold"]),
        **probability_metrics(y, probabilities),
    }
    _assert_metrics(metrics, meta["metrics"]["validation"])
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    save_comparison(resources.comparison, evaluation_dir / "comparacion_validacion.png")
    pd.DataFrame({"target": y, "churn_probability": probabilities}).to_csv(
        evaluation_dir / "probabilidades_validacion_ganador.csv", index=False
    )
    save_roc_curves(
        {bundle["model_name"]: (y, probabilities)},
        evaluation_dir / "roc_ganador_validacion.png",
        title="Ganador congelado — ROC de validación (no prueba)",
    )
    save_calibration_curves(
        {bundle["model_name"]: (y, probabilities)},
        evaluation_dir / "calibracion_ganador_validacion.png",
        title="Ganador congelado — calibración en validación",
    )
    for _, row in resources.comparison.iterrows():
        matrix = [
            [row.true_negatives, row.false_positives],
            [row.false_negatives, row.true_positives],
        ]
        save_confusion_matrix(
            matrix,
            evaluation_dir / f"matriz_validacion_{row.candidate}.png",
            title=f"{row.candidate}\nValidación, umbral {row.decision_threshold:.2f}",
        )
    final = resources.final_report.iloc[0]
    save_confusion_matrix(
        [
            [final.true_negatives, final.false_positives],
            [final.false_negatives, final.true_positives],
        ],
        evaluation_dir / "matriz_prueba_final_guardada.png",
        title="Prueba final — conteos del reporte ya guardado",
    )
    # La tabla original se conserva íntegra. La copia estándar explicita su partición.
    table = resources.comparison.rename(columns={"candidate": "candidato"}).copy()
    table.insert(0, "particion", "validacion")
    table = table.drop(columns=["provisional_winner"], errors="ignore")
    table["ganador_congelado"] = table["candidato"].eq(meta["model_name"])
    table.to_csv(evaluation_dir.parent / "metricas_modelos.csv", index=False)
    started = time.perf_counter()
    result = get_explanation_service(bundle).explain(sample)
    elapsed = time.perf_counter() - started
    shap_metadata = save_explanation_plots(result, explanation_dir)
    reconstructed = result.values.sum(axis=1).to_numpy() + result.base_values
    error = float(np.max(np.abs(reconstructed - result.probabilities)))
    if error > SHAP_ADDITIVITY_ATOL:
        raise ValueError("SHAP no reconstruye la probabilidad calibrada.")
    checks = sample.head(3)
    scored = score_customers(checks, bundle)
    for position in range(len(checks)):
        individual = score_customers(checks.iloc[[position]], bundle)
        np.testing.assert_allclose(
            individual["churn_probability"],
            scored["churn_probability"].iloc[[position]],
            atol=REPORT_METRICS_ATOL,
            rtol=0,
        )
        if individual["top_factors"].iloc[0] != scored["top_factors"].iloc[position]:
            raise ValueError("Las explicaciones individual/lote no coinciden.")
    pd.DataFrame(result.factors(0)).to_csv(
        explanation_dir / "factores_individuales.csv", index=False
    )
    sample["customerID"].reset_index(drop=True).to_csv(
        explanation_dir / "clientes_muestra.csv", index=False
    )
    record = {
        "model_name": bundle["model_name"],
        "model_sha256": meta["model_sha256"],
        "dataset_sha256": meta["dataset_sha256"],
        "split_manifest_sha256": meta["split_manifest_sha256"],
        "validation_reproduced": True,
        "test_reevaluated": False,
        "test_source": "prueba_final.csv verificado; no se predicen filas de prueba",
        "validation": metrics,
        "shap": shap_metadata,
        "shap_seconds": elapsed,
        "shap_max_reconstruction_error": error,
        "shap_sample_origin": "training",
        "individual_batch_rows": len(checks),
        "limitations": [
            "Las bandas 0.35/0.65 son convenciones operativas, no umbrales optimizados.",
            "SHAP describe el modelo; las coaliciones pueden romper dependencias entre servicios.",
            "El coste SHAP medido en esta máquina no garantiza tiempos de la web.",
            "No se dispone de probabilidades originales de prueba para reconstruir su curva ROC.",
            "Los archivos originales de CV no se recuperaron; la reproducción se entrega aparte.",
        ],
    }
    (evaluation_dir / "verificacion_ganador.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-splits", action="store_true")
    parser.add_argument("--global-rows", type=int, default=SHAP_MAX_GLOBAL_ROWS)
    args = parser.parse_args()
    if args.restore_splits:
        print(restore_splits(RAW_DATA_PATH, MODEL_MANIFEST_PATH, SPLIT_MANIFEST_PATH))
        return
    record = generate_frozen_reports(load_frozen_resources(), global_rows=args.global_rows)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"Resultados locales: {PROJECT_ROOT / 'reports'}")


if __name__ == "__main__":
    main()
