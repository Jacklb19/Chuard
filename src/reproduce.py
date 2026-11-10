"""Reproduce CV y validación del protocolo congelado; no modifica el modelo ni usa prueba."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODEL_PATH, PROJECT_ROOT, REPORT_METRICS_ATOL, REPRODUCTION_DIR
from src.data import dataset_sha256
from src.reporting import FrozenResources, load_frozen_resources
from src.train import SEARCH_ITERATIONS, Partitions, run_experiment


def compare_reproduction(original: pd.DataFrame, reproduced: pd.DataFrame) -> dict:
    """Compara resultados sin corregir hiperparámetros para forzar coincidencia."""
    original = original.set_index("candidate")
    eligible = reproduced.loc[reproduced["elegible"]].set_index("candidato")
    if set(original.index) != set(eligible.index):
        raise ValueError("La reproducción no contiene los diez candidatos originales.")
    comparisons = []
    for key, row in original.iterrows():
        new = eligible.loc[key]
        differences = {
            metric: float(new[metric] - row[metric])
            for metric in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "average_precision",
                "brier",
            )
        }
        same = all(abs(value) <= REPORT_METRICS_ATOL for value in differences.values())
        same = same and np.isclose(new["umbral"], row["decision_threshold"], atol=0, rtol=0)
        same = same and new["calibracion"] == row["calibration"]
        comparisons.append({"candidate": key, "matches": bool(same), "differences": differences})
    return {
        "all_candidates_match": all(row["matches"] for row in comparisons),
        "candidates": comparisons,
    }


def reproduce_frozen_protocol(
    resources: FrozenResources, output_dir: Path, *, n_jobs: int = 2
) -> dict:
    """Nueva evidencia de reproducción, conservando los reportes históricos íntegros."""
    if n_jobs < 1:
        raise ValueError("Indique un número positivo y acotado de trabajadores.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("La salida ya contiene resultados; elija una carpeta nueva.")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_before = dataset_sha256(MODEL_PATH)
    original_test = PROJECT_ROOT / "reports/evaluation/prueba_final.csv"
    test_before = dataset_sha256(original_test)
    source_hashes = {
        str(path.relative_to(PROJECT_ROOT)): dataset_sha256(path)
        for path in sorted((PROJECT_ROOT / "src").glob("*.py"))
    }
    provenance = {
        "purpose": "Reproducción del protocolo previo, sin nueva selección de producción.",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "dataset_sha256": resources.manifest["dataset_sha256"],
        "split_manifest_sha256": resources.manifest["split_manifest_sha256"],
        "model_sha256": model_before,
        "test_report_sha256": test_before,
        "test_rows_available_to_experiment": False,
        "n_iter": SEARCH_ITERATIONS,
        "n_jobs": n_jobs,
        "source_sha256": source_hashes,
    }
    development = Partitions(
        {"train": resources.training, "validation": resources.validation},
        resources.manifest["dataset_sha256"],
        resources.manifest["split_manifest_sha256"],
    )
    path = output_dir / "reproduction_provenance.json"
    path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    record = run_experiment(
        development,
        output_dir,
        metrics_table=output_dir / "metricas_modelos.csv",
        n_iter=SEARCH_ITERATIONS,
        n_jobs=n_jobs,
    )
    comparison = compare_reproduction(
        resources.comparison, pd.read_csv(output_dir / "metricas_modelos.csv")
    )
    record["status"] = "reproduccion_completa_sin_reevaluar_prueba"
    record["historical_experiment_sha256"] = resources.manifest["experiment_sha256"]
    record["warning"] += (
        " Esta ejecución reproduce un protocolo previo: no reemplaza su historial original "
        "ni su ganador; la prueba ya fue evaluada anteriormente y aquí no se vuelve a usar."
    )
    (output_dir / "experiment.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    provenance.update(
        completed_at_utc=datetime.now(UTC).isoformat(),
        reproduced_winner=record["winner"],
        production_winner=resources.bundle["model_name"],
        **comparison,
    )
    if dataset_sha256(MODEL_PATH) != model_before or dataset_sha256(original_test) != test_before:
        raise ValueError("Se modificó un insumo congelado durante la reproducción.")
    provenance["immutable_inputs_verified"] = True
    path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPRODUCTION_DIR)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    result = reproduce_frozen_protocol(load_frozen_resources(), args.output_dir, n_jobs=args.jobs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
