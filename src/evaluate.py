"""Métricas binarias explícitas para etiquetas y probabilidades separadas."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, probabilities, *, threshold: float) -> dict:
    y = np.asarray(y_true)
    probability = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or probability.ndim != 1 or len(y) != len(probability) or not len(y):
        raise ValueError("Etiquetas y probabilidades deben ser vectores del mismo tamaño.")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Se requieren ambas clases binarias para estas métricas.")
    if (
        not np.isfinite(probability).all()
        or (probability < 0).any()
        or (probability > 1).any()
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Probabilidades y umbral deben estar entre 0 y 1 y ser finitos.")
    predicted = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted)),
        "f1": float(f1_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "decision_threshold": float(threshold),
    }
