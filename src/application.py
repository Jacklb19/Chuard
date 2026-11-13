"""Verified resources and pure presentation operations for the web application."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd

from src.config import MODEL_MANIFEST_PATH, MODEL_PATH, PROJECT_ROOT
from src.inference import load_model_bundle

UI_BATCH_LIMIT = 100
EXAMPLE_PATH = PROJECT_ROOT / "examples/clientes_ejemplo.csv"
MODEL_SHA256 = "0bb0a8542c14f3e23e2fd779fc8b3dd2c488c6c28db069411d23e84b28d1cea1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_application_model(
    model_path: Path = MODEL_PATH, manifest_path: Path = MODEL_MANIFEST_PATH
) -> dict:
    """Verify before deserialization; allow Python patches, require exact libraries."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256_file(model_path)
    if digest != MODEL_SHA256 or digest != manifest["model_sha256"]:
        raise ValueError("La huella del modelo no coincide con el artefacto aprobado.")
    environment = manifest["environment"]
    if platform.python_version_tuple()[:2] != tuple(environment["python"].split(".")[:2]):
        raise ValueError("El modelo requiere Python 3.11 y las dependencias fijadas.")
    for library, expected in environment["libraries"].items():
        if importlib.metadata.version(library) != expected:
            raise ValueError(f"Versión incompatible de {library}; se requiere {expected}.")
    bundle = load_model_bundle(str(model_path))
    for key in ("model_version", "model_name", "decision_threshold", "risk_limits"):
        if bundle[key] != manifest[key]:
            raise ValueError(f"El manifiesto y el modelo difieren en {key}.")
    return bundle


def frame_fingerprint(frame: pd.DataFrame) -> str:
    payload = frame.to_json(orient="split", date_format="iso", double_precision=15)
    return hashlib.sha256(payload.encode()).hexdigest()


def filter_customers(
    scored: pd.DataFrame,
    *,
    search: str = "",
    risk: str = "Todos",
    active_only: bool = False,
    contract: str = "Todos",
    internet: str = "Todos",
) -> pd.DataFrame:
    """Preserve technical indices when customer IDs are duplicated."""
    selected = scored.copy()
    if active_only and "Churn" in selected:
        selected = selected.loc[selected["Churn"].eq("No")]
    if risk != "Todos":
        selected = selected.loc[selected["risk_level"].eq(risk)]
    for column, value in (("Contract", contract), ("InternetService", internet)):
        if value != "Todos":
            selected = selected.loc[selected[column].eq(value)]
    if search:
        selected = selected.loc[
            selected["customerID"].astype(str).str.contains(search, case=False, regex=False)
        ]
    return selected


def export_customers(frame: pd.DataFrame) -> bytes:
    """Serialize nested results and neutralize spreadsheet formulas in text cells."""
    output = frame.copy()

    def safe_cell(value):
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False, allow_nan=False)
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    for column in output.select_dtypes(include=["object", "string"]).columns:
        output[column] = output[column].map(safe_cell)
    return output.to_csv(index=False).encode("utf-8-sig")
