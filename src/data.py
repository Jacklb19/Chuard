"""Lectura explícita de CSV sin descarga ni modificación de datos de origen."""

import csv
import hashlib
import io
from pathlib import Path
from typing import BinaryIO, Literal, TextIO

import pandas as pd

from src.config import MAX_BATCH_ROWS, MAX_UPLOAD_BYTES
from src.schema import InputValidationError, ValidatedCustomers, ValidationIssue, validate_customers


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
