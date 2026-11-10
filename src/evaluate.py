"""Métricas, umbral, calibración y gráficos para etiquetas y probabilidades separadas."""

from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURE_DPI

# Regla común de umbral: rejilla fija definida antes de mirar validación.
THRESHOLD_GRID = tuple(round(value / 100, 2) for value in range(5, 96))
THRESHOLD_TIE_TARGET = 0.5
CALIBRATION_BINS = 10


def _checked(y_true, probabilities) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    probability = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or probability.ndim != 1 or len(y) != len(probability) or not len(y):
        raise ValueError("Etiquetas y probabilidades deben ser vectores del mismo tamaño.")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Se requieren ambas clases binarias para estas métricas.")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Las probabilidades deben estar entre 0 y 1 y ser finitas.")
    return y.astype(int), probability


def classification_metrics(y_true, probabilities, *, threshold: float) -> dict:
    """Precision y F1 valen 0 cuando no hay predicciones positivas (caso documentado)."""
    y, probability = _checked(y_true, probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("El umbral debe estar entre 0 y 1 y ser finito.")
    predicted = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "decision_threshold": float(threshold),
        "positive_predictions": int(predicted.sum()),
        "rows": len(y),
    }


def probability_metrics(y_true, probabilities) -> dict:
    """Brier y log-loss juntos: ninguna medida aislada demuestra buena calibración."""
    y, probability = _checked(y_true, probabilities)
    return {
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def select_threshold(
    y_true,
    probabilities,
    *,
    grid: tuple[float, ...] = THRESHOLD_GRID,
    tie_target: float = THRESHOLD_TIE_TARGET,
) -> dict:
    """Máximo F1 positivo en la rejilla; empate → más cercano a tie_target, luego menor."""
    y, probability = _checked(y_true, probabilities)
    if not grid or any(not 0 < value < 1 for value in grid):
        raise ValueError("La rejilla de umbrales debe contener valores en (0, 1).")
    scores = [
        (float(f1_score(y, (probability >= value).astype(int), zero_division=0)), value)
        for value in grid
    ]
    best_f1 = max(score for score, _ in scores)
    tied = [value for score, value in scores if score == best_f1]
    chosen = min(tied, key=lambda value: (abs(value - tie_target), value))
    return {
        "threshold": float(chosen),
        "f1": best_f1,
        "tied_thresholds": len(tied),
        "rule": (
            f"Máximo F1 de Churn=1 en rejilla {grid[0]:.2f}-{grid[-1]:.2f} "
            f"({len(grid)} valores); empate: más cercano a {tie_target}, luego menor."
        ),
    }


def calibration_summary(y_true, probabilities, *, n_bins: int = CALIBRATION_BINS) -> dict:
    """Curva por cuantiles; los tramos se reportan tal como quedan tras unir duplicados."""
    y, probability = _checked(y_true, probabilities)
    observed, predicted = calibration_curve(y, probability, n_bins=n_bins, strategy="quantile")
    return {
        "strategy": "quantile",
        "requested_bins": n_bins,
        "mean_predicted": predicted.tolist(),
        "observed_rate": observed.tolist(),
        "max_abs_gap": float(np.max(np.abs(observed - predicted))),
    }


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_roc_curves(curves: dict[str, tuple], path: str | Path, *, title: str) -> None:
    """curves: nombre → (y_true, probabilidades)."""
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7, 6))
    for name, (y_true, probabilities) in curves.items():
        y, probability = _checked(y_true, probabilities)
        false_positive, true_positive, _ = roc_curve(y, probability)
        auc = roc_auc_score(y, probability)
        axis.plot(false_positive, true_positive, linewidth=1.4, label=f"{name} (AUC {auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1, label="Azar")
    axis.set(
        title=title,
        xlabel="Tasa de falsos positivos (proporción)",
        ylabel="Tasa de verdaderos positivos (proporción)",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axis.legend(fontsize=7, loc="lower right")
    _save(plt, fig, path)


def save_calibration_curves(curves: dict[str, tuple], path: str | Path, *, title: str) -> None:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7, 6))
    for name, (y_true, probabilities) in curves.items():
        summary = calibration_summary(y_true, probabilities)
        axis.plot(
            summary["mean_predicted"],
            summary["observed_rate"],
            marker="o",
            markersize=3,
            label=name,
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1, label="Ideal")
    axis.set(
        title=title,
        xlabel="Probabilidad media predicha por tramo",
        ylabel="Proporción observada de Churn=1",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(fontsize=7, loc="upper left")
    _save(plt, fig, path)


def save_confusion_matrix(matrix, path: str | Path, *, title: str) -> None:
    plt = _pyplot()
    values = np.asarray(matrix)
    if values.shape != (2, 2):
        raise ValueError("Se requiere una matriz de confusión binaria 2x2.")
    fig, axis = plt.subplots(figsize=(5, 4.5))
    axis.imshow(values, cmap="Blues")
    labels = ["No abandona (0)", "Abandona (1)"]
    axis.set_xticks([0, 1], labels=labels)
    axis.set_yticks([0, 1], labels=labels)
    total = values.sum()
    for row in range(2):
        for column in range(2):
            count = int(values[row, column])
            color = "white" if count > values.max() / 2 else "black"
            axis.text(
                column,
                row,
                f"{count}\n({count / total:.1%})",
                ha="center",
                va="center",
                color=color,
            )
    axis.set(title=title, xlabel="Predicción (clientes)", ylabel="Valor real (clientes)")
    _save(plt, fig, path)


def _save(plt, fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
