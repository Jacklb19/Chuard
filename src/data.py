"""Lectura explícita de CSV sin descarga ni modificación de datos de origen."""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Literal, TextIO
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from src.config import (
    DATA_REPORT_DIR,
    DATASET_DOWNLOAD_URL,
    DATASET_URL,
    DOWNLOAD_TIMEOUT_SECONDS,
    EDA_IQR_MULTIPLIER,
    EDA_MIN_SEGMENT_SIZE,
    FIGURE_DPI,
    MAX_BATCH_ROWS,
    MAX_UPLOAD_BYTES,
    RAW_DATA_PATH,
)
from src.schema import (
    CATEGORIES,
    CATEGORY_LABELS,
    FEATURE_LABELS,
    NUMERIC_COLUMNS,
    InputValidationError,
    ValidatedCustomers,
    ValidationIssue,
    validate_customers,
)


def read_customer_csv(
    source: str | Path | BinaryIO | TextIO,
    *,
    mode: Literal["inference", "training"] = "inference",
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_rows: int = MAX_BATCH_ROWS,
) -> ValidatedCustomers:
    """CSV UTF-8 separado por comas; valida encabezado antes del mangling de Pandas."""
    if max_bytes < 1 or max_rows < 1:
        raise ValueError("Los límites de carga deben ser positivos.")
    if isinstance(source, (str, Path)):
        with Path(source).open("rb") as stream:
            payload = stream.read(max_bytes + 1)
    else:
        payload = source.read(max_bytes + 1)
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if len(payload) > max_bytes:
        raise InputValidationError([ValidationIssue(None, "*", "Archivo demasiado grande.")])
    try:
        text = payload.decode("utf-8-sig")
        rows = csv.reader(io.StringIO(text), strict=True)
        header = next(rows, [])
        duplicates = sorted({column for column in header if header.count(column) > 1})
        if duplicates:
            raise InputValidationError(
                [ValidationIssue(None, column, "Encabezado duplicado.") for column in duplicates]
            )
        row_number = 0
        for row in rows:
            if not row:
                continue
            row_number += 1
            if len(row) != len(header):
                raise InputValidationError(
                    [ValidationIssue(row_number, "*", "Cantidad de columnas inconsistente.")]
                )
        data = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, nrows=max_rows + 1)
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as exc:
        raise InputValidationError(
            [ValidationIssue(None, "*", "CSV UTF-8 con estructura inválida.")]
        ) from exc
    if len(data) > max_rows:
        raise InputValidationError([ValidationIssue(None, "*", "Demasiadas filas en el lote.")])
    return validate_customers(data, mode=mode)


def dataset_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_telco_dataset(
    destination: str | Path = RAW_DATA_PATH,
    *,
    download_url: str = DATASET_DOWNLOAD_URL,
    timeout: float = DOWNLOAD_TIMEOUT_SECONDS,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> dict:
    """Descarga pública explícita; no autentica, sobrescribe ni extrae rutas del ZIP."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("El CSV original ya existe; no se sobrescribe.")
    if not download_url.startswith("https://www.kaggle.com/"):
        raise ValueError("La descarga debe provenir de Kaggle mediante HTTPS.")
    if timeout <= 0 or max_bytes <= 0:
        raise ValueError("Timeout y tamaño máximo deben ser positivos.")
    request = Request(download_url, headers={"User-Agent": "Chuard/0.1 data-quality"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("La descarga excede el tamaño permitido.")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [
                item
                for item in archive.infolist()
                if Path(item.filename).name == RAW_DATA_PATH.name
            ]
            if len(members) != 1 or members[0].file_size > max_bytes:
                raise ValueError("Se requiere un único CSV Telco dentro del tamaño permitido.")
            csv_bytes = archive.read(members[0])
    except zipfile.BadZipFile as exc:
        raise ValueError("Kaggle no devolvió un ZIP válido; puede requerir acceso manual.") from exc
    # No guardar un HTML/login ni un CSV ajeno como el original solicitado.
    read_customer_csv(io.BytesIO(csv_bytes), mode="training")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(csv_bytes)
    return {
        "source_url": DATASET_URL,
        "download_url": download_url,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "filename": destination.name,
        "bytes": len(csv_bytes),
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
    }


def audit_dataset(path: str | Path) -> dict:
    """Inspección estructural del original; no aprende reglas estadísticas."""
    path = Path(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    normalized = read_customer_csv(path, mode="training").data
    missing = {
        column: int((raw[column].isna() | raw[column].str.strip().eq("")).sum())
        for column in raw.columns
    }
    blank_total = raw["TotalCharges"].str.strip().eq("")
    duplicates_without_id = raw.drop(columns="customerID").duplicated(keep=False)
    return {
        "filename": path.name,
        "source_url": DATASET_URL,
        "sha256": dataset_sha256(path),
        "rows": len(raw),
        "columns": raw.columns.tolist(),
        "column_count": len(raw.columns),
        "matches_expected_shape": raw.shape == (7043, 21),
        "raw_dtypes": {column: str(dtype) for column, dtype in raw.dtypes.items()},
        "normalized_dtypes": {column: str(dtype) for column, dtype in normalized.dtypes.items()},
        "empty_values": missing,
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_customer_ids": int(raw["customerID"].duplicated().sum()),
        "rows_sharing_profile_and_label": int(duplicates_without_id.sum()),
        "duplicates_removed": 0,
        "duplicate_policy": (
            "No eliminar perfiles iguales de clientes con identificadores distintos."
        ),
        "target_counts": {
            str(key): int(value) for key, value in raw["Churn"].value_counts().items()
        },
        "categories": {column: sorted(raw[column].unique().tolist()) for column in CATEGORIES},
        "total_charges_empty_rows": (np.flatnonzero(blank_total.to_numpy()) + 1).tolist(),
        "total_charges_empty_tenure": raw.loc[blank_total, "tenure"].value_counts().to_dict(),
        "total_charges_zero_normalizations": int((blank_total & normalized["tenure"].eq(0)).sum()),
        "validation_warnings": [
            {"row": issue.row, "column": issue.column, "reason": issue.reason}
            for issue in read_customer_csv(path, mode="training").warnings
        ],
        "scope": (
            "Calidad estructural completa; no usar para seleccionar modelos o reglas estadísticas."
        ),
    }


def training_subset_from_manifest(path: str | Path, manifest_path: str | Path) -> pd.DataFrame:
    """Verifica procedencia/cobertura y retorna solo entrenamiento, con orden trazable."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != dataset_sha256(path):
        raise ValueError("El manifiesto no corresponde a la huella del CSV original.")
    data = read_customer_csv(path, mode="training").data
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("El manifiesto debe definir train, validation y test.")
    combined = []
    for name, values in splits.items():
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int for value in values)
        ):
            raise ValueError(f"Posiciones inválidas en {name}.")
        if any(position < 0 or position >= len(data) for position in values):
            raise ValueError(f"Posiciones fuera del dataset en {name}.")
        if data.iloc[values]["Churn"].nunique() != 2:
            raise ValueError(f"La partición {name} necesita ambas clases.")
        combined.extend(values)
    if len(combined) != len(data) or sorted(combined) != list(range(len(data))):
        raise ValueError("Las particiones deben ser disjuntas y cubrir el CSV completo.")
    if data["customerID"].duplicated().any():
        raise ValueError("Resolver identificadores duplicados antes de dividir los datos.")
    training = data.iloc[splits["train"]].copy()
    training.attrs["dataset_sha256"] = manifest["dataset_sha256"]
    training.attrs["manifest_sha256"] = dataset_sha256(manifest_path)
    training.attrs["split_status"] = manifest.get("status", "estado_no_documentado")
    return training


def segment_churn_rates(
    training: pd.DataFrame, column: str, *, min_size: int = EDA_MIN_SEGMENT_SIZE
) -> pd.DataFrame:
    if column not in CATEGORIES:
        raise ValueError("El segmento debe ser una categoría original del contrato.")
    if min_size < 1:
        raise ValueError("El tamaño mínimo de segmento debe ser positivo.")
    frame = training.assign(churn=training["Churn"].map({"No": 0, "Yes": 1}))
    result = frame.groupby(column, observed=True)["churn"].agg(n="size", churn_count="sum")
    result["churn_rate"] = result["churn_count"] / result["n"]
    result["small_segment"] = result["n"] < min_size
    return result.reset_index()


def summarize_training(training: pd.DataFrame) -> dict:
    """Estadísticas descriptivas y límites diagnósticos calculados solo aquí."""
    if training.empty or "Churn" not in training:
        raise ValueError("Se necesita entrenamiento no vacío con etiqueta histórica.")
    summary = {}
    for column in NUMERIC_COLUMNS:
        values = training[column].dropna()
        if values.empty:
            raise ValueError(f"No hay valores observados en entrenamiento para {column}.")
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - EDA_IQR_MULTIPLIER * iqr, q3 + EDA_IQR_MULTIPLIER * iqr
        summary[column] = {
            "observed": int(values.size),
            "missing": int(training[column].isna().sum()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "mean": float(values.mean()),
            "q1": float(q1),
            "q3": float(q3),
            "iqr_applicable": column != "SeniorCitizen",
            "iqr_lower": float(lower) if column != "SeniorCitizen" else None,
            "iqr_upper": float(upper) if column != "SeniorCitizen" else None,
            "outside_iqr": (
                int((values.lt(lower) | values.gt(upper)).sum())
                if column != "SeniorCitizen"
                else None
            ),
        }
    return {
        "rows": len(training),
        "target_counts": training["Churn"].value_counts().to_dict(),
        "numeric": summary,
        "training_ranges": {
            name: {"min": value["min"], "max": value["max"]} for name, value in summary.items()
        },
        "high_monthly_charges": float(training["MonthlyCharges"].quantile(0.75)),
        "high_monthly_charges_rule": "Q3 de MonthlyCharges del entrenamiento; propuesta comercial.",
        "scope": "Entrenamiento exclusivamente; IQR es diagnóstico y no elimina/trunca valores.",
    }


def generate_eda_report(
    dataset_path: str | Path, manifest_path: str | Path, output_dir: str | Path
) -> dict:
    """Genera tablas/PNG e interpretaciones verificables sin abrir validación/prueba."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    training = training_subset_from_manifest(dataset_path, manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics = summarize_training(training)
    quality = audit_dataset(dataset_path)
    for filename, record in (
        ("quality_report.json", quality),
        ("training_summary.json", statistics),
    ):
        (output_dir / filename).write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    numeric = training.loc[:, NUMERIC_COLUMNS].copy()
    numeric["Churn"] = training["Churn"].map({"No": 0, "Yes": 1})
    correlation = numeric.corr()
    correlation.to_csv(output_dir / "correlations_training.csv")
    conclusions = []
    figures = []

    def save(fig, filename: str, conclusion: str) -> None:
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        figures.append(filename)
        conclusions.append({"figure": filename, "conclusion": conclusion})

    counts = training["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
    fig, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(["No abandonó", "Abandonó"], counts, color=["#176B87", "#C8553D"])
    axis.bar_label(bars, labels=[f"{n:,} ({n / len(training):.1%})" for n in counts], padding=3)
    axis.set(
        title="Distribución histórica del abandono — entrenamiento",
        ylabel="Clientes (conteo)",
        xlabel="Abandono histórico",
    )
    save(
        fig,
        "01_distribucion_churn.png",
        f"Abandonaron {counts['Yes']} de {len(training)} clientes "
        f"({counts['Yes'] / len(training):.1%}); hay desbalance de clases.",
    )

    for index, column in enumerate(
        ("Contract", "PaymentMethod", "InternetService", "TechSupport"), 2
    ):
        table = segment_churn_rates(training, column)
        table.to_csv(output_dir / f"segments_{column}.csv", index=False)
        ordered = table.sort_values("churn_rate", ascending=False)
        fig, axis = plt.subplots(figsize=(9, 4.8))
        labels = ordered[column].map(lambda value: CATEGORY_LABELS.get(value, value))
        bars = axis.barh(labels, ordered["churn_rate"] * 100, color="#176B87")
        axis.bar_label(bars, labels=[f"n={n}" for n in ordered["n"]], padding=4)
        axis.set(
            title=f"Abandono por {FEATURE_LABELS[column].lower()} — entrenamiento",
            xlabel="Clientes que abandonaron (%)",
            ylabel=FEATURE_LABELS[column],
            xlim=(0, 105),
        )
        axis.invert_yaxis()
        highest, lowest = ordered.iloc[0], ordered.iloc[-1]
        conclusion = (
            f"{highest[column]}: {highest['churn_rate']:.1%} (n={highest['n']}); "
            f"{lowest[column]}: {lowest['churn_rate']:.1%} (n={lowest['n']}). "
            "Es una asociación histórica, no una causa demostrada."
        )
        if table["small_segment"].any():
            conclusion += f" Hay segmentos con menos de {EDA_MIN_SEGMENT_SIZE} clientes."
        save(fig, f"{index:02d}_abandono_{column}.png", conclusion)

    for index, (column, unit) in enumerate(
        (
            ("tenure", "meses"),
            ("MonthlyCharges", "unidades monetarias/mes"),
            ("TotalCharges", "unidades monetarias acumuladas"),
        ),
        6,
    ):
        groups = [
            training.loc[training["Churn"].eq(label), column].dropna() for label in ("No", "Yes")
        ]
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.boxplot(groups, tick_labels=["No abandonó", "Abandonó"], showfliers=True)
        axis.set(
            title=f"{column} por abandono — entrenamiento",
            ylabel=f"{column} ({unit})",
            xlabel="Abandono histórico",
        )
        save(
            fig,
            f"{index:02d}_{column}_por_clase.png",
            f"Mediana de {column}: No={groups[0].median():.2f}, "
            f"Yes={groups[1].median():.2f} {unit}. "
            "Los valores atípicos se muestran, no se eliminan.",
        )

    fig, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation)), correlation.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(correlation)), correlation.index)
    for row in range(len(correlation)):
        for col in range(len(correlation)):
            axis.text(col, row, f"{correlation.iloc[row, col]:.2f}", ha="center", va="center")
    axis.set_title("Correlación de Pearson — entrenamiento")
    fig.colorbar(image, ax=axis, label="Coeficiente de Pearson (sin unidad)")
    save(
        fig,
        "09_correlaciones.png",
        "Las correlaciones son descriptivas y no prueban causalidad. "
        "Churn se codifica No=0, Yes=1; SeniorCitizen es un indicador binario.",
    )

    report = {
        "dataset_sha256": quality["sha256"],
        "manifest_sha256": dataset_sha256(manifest_path),
        "split_status": training.attrs["split_status"],
        "training_rows": len(training),
        "validation_and_test_used_for_eda": False,
        "figures": figures,
        "conclusions": conclusions,
    }
    (output_dir / "eda_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# EDA de entrenamiento de Chuard",
        "",
        f"Registros: {len(training)}.",
        f"Estado de partición: {training.attrs['split_status']}.",
        "No se utilizó validación/prueba para las conclusiones. Moneda no documentada.",
        "",
    ]
    for item in conclusions:
        lines.extend([f"![{item['figure']}]({item['figure']})", "", item["conclusion"], ""])
    (output_dir / "EDA_entrenamiento.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Datos y EDA reproducibles de Chuard.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    download = subparsers.add_parser(
        "download", help="Descarga pública explícita sin sobrescribir."
    )
    download.add_argument("--data", type=Path, default=RAW_DATA_PATH)
    download.add_argument("--output", type=Path, default=DATA_REPORT_DIR)
    audit = subparsers.add_parser("audit", help="Calidad estructural del original completo.")
    audit.add_argument("--data", type=Path, default=RAW_DATA_PATH)
    audit.add_argument("--output", type=Path, default=DATA_REPORT_DIR)
    eda = subparsers.add_parser("eda", help="EDA solo de entrenamiento mediante manifiesto.")
    eda.add_argument("--data", type=Path, default=RAW_DATA_PATH)
    eda.add_argument("--output", type=Path, default=DATA_REPORT_DIR)
    eda.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "download":
        record = download_telco_dataset(args.data)
        filename = "source_manifest.json"
    elif args.action == "audit":
        record = audit_dataset(args.data)
        filename = "quality_report.json"
    else:
        record = generate_eda_report(args.data, args.manifest, args.output)
        print(f"EDA generado sobre {record['training_rows']} filas de entrenamiento.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / filename).write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(f"Datos verificados. Reporte: {filename}.")


if __name__ == "__main__":
    main()
