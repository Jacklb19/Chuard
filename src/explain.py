"""SHAP real con referencias de entrenamiento y salida/escala explícitas."""

import hashlib
import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    FIGURE_DPI,
    MAX_TOP_FACTORS,
    NEGATIVE_DIRECTION,
    POSITIVE_DIRECTION,
    RANDOM_STATE,
    SHAP_ADDITIVITY_ATOL,
    SHAP_BACKGROUND_SIZE,
    SHAP_CACHE_SIZE,
    SHAP_GLOBAL_DISPLAY_FEATURES,
    SHAP_LOCAL_DISPLAY_FEATURES,
    SHAP_MAX_EVALS,
    SHAP_MAX_GLOBAL_ROWS,
    SHAP_ZERO_TOLERANCE,
)
from src.features import build_features
from src.schema import (
    CATEGORIES,
    CATEGORY_LABELS,
    DERIVED_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    validate_customers,
)

_SHAP_LOCK = threading.RLock()
_SERVICES: OrderedDict[tuple, "ExplanationService"] = OrderedDict()


class ExplanationUnavailableError(NotImplementedError):
    """Recursos SHAP todavía no entregados por entrenamiento; nunca simularlos."""


def prepare_explanation_context(
    training_df: pd.DataFrame,
    *,
    method: str = "auto",
    background_size: int = SHAP_BACKGROUND_SIZE,
    max_evals: int = SHAP_MAX_EVALS,
    max_rows: int = SHAP_MAX_GLOBAL_ROWS,
    random_state: int = RANDOM_STATE,
) -> dict:
    """El llamador aporta exclusivamente entrenamiento, con o sin etiqueta auxiliar."""
    if method not in {"auto", "permutation", "tree_probability", "linear_log_odds"}:
        raise ValueError("Método SHAP no soportado.")
    if (
        type(background_size) is not int
        or background_size < 1
        or background_size > SHAP_BACKGROUND_SIZE
    ):
        raise ValueError(f"La referencia SHAP admite entre 1 y {SHAP_BACKGROUND_SIZE} filas.")
    if type(max_evals) is not int or max_evals < 2 * len(FEATURE_COLUMNS) + 1:
        raise ValueError("El presupuesto SHAP debe cubrir una permutación completa.")
    if type(max_rows) is not int or not 1 <= max_rows <= SHAP_MAX_GLOBAL_ROWS:
        raise ValueError(f"Se admiten entre 1 y {SHAP_MAX_GLOBAL_ROWS} filas por explicación.")
    normalized = validate_customers(training_df, mode="training", require_target=False).data
    selected = np.random.default_rng(random_state).choice(
        len(normalized), size=min(background_size, len(normalized)), replace=False
    )
    selected.sort()
    return {
        "version": "1",
        "origin": "training",
        "method": method,
        "background": normalized.iloc[selected].loc[:, FEATURE_COLUMNS].copy(),
        "background_positions_in_training": selected.tolist(),
        "training_rows": len(normalized),
        "max_evals": max_evals,
        "max_rows": max_rows,
        "random_state": random_state,
        "dataset_sha256": training_df.attrs.get("dataset_sha256"),
        "split_manifest_sha256": training_df.attrs.get("manifest_sha256"),
    }


def _encoded(frame: pd.DataFrame) -> np.ndarray:
    encoded = frame.loc[:, FEATURE_COLUMNS].copy()
    for column, domain in CATEGORIES.items():
        encoded[column] = encoded[column].map({value: i for i, value in enumerate(domain)})
    return encoded.to_numpy(dtype=float)


def _decoded(matrix: np.ndarray) -> pd.DataFrame:
    result = pd.DataFrame(matrix, columns=FEATURE_COLUMNS)
    for column, domain in CATEGORIES.items():
        values = result[column].to_numpy()
        if (
            not np.isfinite(values).all()
            or (values != values.astype(int)).any()
            or (values < 0).any()
            or (values >= len(domain)).any()
        ):
            raise ValueError(f"Código categórico SHAP inválido: {column}.")
        result[column] = np.asarray(domain, dtype=object)[values.astype(int)]
    return result


def _native_feature_groups(names: list[str]) -> list[str]:
    """Agrupa One-Hot por su columna de origen, sin confundir prefijos/categorías."""
    groups = []
    for transformed_name in names:
        name = transformed_name.split("__", 1)[-1]
        if name in FEATURE_LABELS:
            groups.append(name)
            continue
        matches = [
            column
            for column, domain in CATEGORIES.items()
            if name in {f"{column}_{value}" for value in domain}
        ]
        if len(matches) != 1:
            raise ValueError(f"No se puede traducir la característica: {transformed_name}.")
        groups.append(matches[0])
    return groups


@dataclass
class ExplanationResult:
    values: pd.DataFrame
    base_values: np.ndarray
    explained_values: np.ndarray
    probabilities: np.ndarray
    feature_values: pd.DataFrame
    method: str
    scale: str
    explained_output: str
    model_scope: str

    def factors(self, position: int, *, limit: int = MAX_TOP_FACTORS) -> list[dict]:
        if not 0 <= position < len(self.values) or not 1 <= limit <= MAX_TOP_FACTORS:
            raise ValueError("Posición o número de factores fuera de rango.")
        contributions = self.values.iloc[position]
        ordered = np.argsort(-np.abs(contributions.to_numpy()), kind="stable")
        result = []
        for index in ordered:
            feature = contributions.index[index]
            contribution = float(contributions.iloc[index])
            if abs(contribution) <= SHAP_ZERO_TOLERANCE:
                continue
            value = self.feature_values.iloc[position][feature]
            value = (
                None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value
            )
            result.append(
                {
                    "feature": feature,
                    "value": value,
                    "label": FEATURE_LABELS[feature],
                    "display_value": CATEGORY_LABELS.get(value, value),
                    "direction": POSITIVE_DIRECTION if contribution > 0 else NEGATIVE_DIRECTION,
                    "contribution": contribution,
                    "scale": self.scale,
                    "explained_output": self.explained_output,
                    "model_scope": self.model_scope,
                    "base_value": float(self.base_values[position]),
                    "explained_value": float(self.explained_values[position]),
                }
            )
            if len(result) == limit:
                break
        remaining = float(contributions.sum() - sum(item["contribution"] for item in result))
        for item in result:
            item["remaining_contribution"] = remaining
        return result


class ExplanationService:
    """Instancia reutilizable para un estimador congelado y una referencia fija."""

    def __init__(self, bundle: dict):
        import shap

        self.estimator = bundle.get("pipeline")
        if not callable(getattr(self.estimator, "predict_proba", None)):
            raise ValueError("El modelo a explicar debe ofrecer predict_proba.")
        classes = np.asarray(getattr(self.estimator, "classes_", []))
        if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
            raise ValueError("SHAP requiere clases binarias 0 y 1.")
        self.positive_index = int(np.flatnonzero(classes == 1)[0])
        context = bundle.get("explanation_context")
        if not isinstance(context, dict):
            raise ExplanationUnavailableError(
                "Falta explanation_context con referencia SHAP de entrenamiento."
            )
        if context.get("origin") != "training" or context.get("version") != "1":
            raise ValueError("Referencia SHAP sin origen de entrenamiento o versión compatible.")
        background = context.get("background")
        if not isinstance(background, pd.DataFrame):
            raise ValueError("La referencia SHAP debe ser un DataFrame de las 19 entradas.")
        # Revalidar recursos recibidos del bundle; no confiar solo en los metadatos.
        checked = prepare_explanation_context(
            background,
            method=context.get("method", "auto"),
            background_size=len(background),
            max_evals=context.get("max_evals", SHAP_MAX_EVALS),
            max_rows=context.get("max_rows", SHAP_MAX_GLOBAL_ROWS),
            random_state=context.get("random_state", RANDOM_STATE),
        )
        self.background = checked["background"]
        self.max_evals = checked["max_evals"]
        self.max_rows = checked["max_rows"]
        self.random_state = checked["random_state"]
        self.method = checked["method"]
        native = isinstance(self.estimator, Pipeline) and list(self.estimator.named_steps) == [
            "features",
            "preprocess",
            "classifier",
        ]
        classifier = self.estimator.named_steps["classifier"] if native else None
        tree_supported = classifier is not None and (
            type(classifier).__name__
            in {"RandomForestClassifier", "XGBClassifier", "LGBMClassifier"}
            and type(classifier).__module__.split(".")[0] in {"sklearn", "xgboost", "lightgbm"}
        )
        if self.method == "auto":
            if native and isinstance(classifier, LogisticRegression):
                self.method = "linear_log_odds"
            elif native and tree_supported:
                self.method = "tree_probability"
            else:
                self.method = "permutation"
        if self.method != "permutation" and (not native or self.positive_index != 1):
            raise ValueError(
                "El método nativo requiere un pipeline no calibrado con clases [0, 1]."
            )

        self.scale = "log_odds" if self.method == "linear_log_odds" else "probability"
        self.explained_output = (
            "decision_function" if self.scale == "log_odds" else "churn_probability"
        )
        self.model_scope = (
            "complete_estimator" if self.method == "permutation" else "uncalibrated_pipeline"
        )
        self.groups = list(FEATURE_COLUMNS)
        with _SHAP_LOCK:
            state = np.random.get_state()
            try:
                if self.method == "linear_log_odds":
                    if not isinstance(classifier, LogisticRegression):
                        raise ValueError(
                            "linear_log_odds requiere Regresión Logística sin calibrador."
                        )
                    transformed = self.estimator[:-1].transform(self.background)
                    self.explainer = shap.LinearExplainer(classifier, transformed)
                elif self.method == "tree_probability":
                    if not tree_supported:
                        raise ValueError(
                            "tree_probability requiere un árbol compatible sin calibrador."
                        )
                    transformed = self.estimator[:-1].transform(self.background)
                    self.explainer = shap.TreeExplainer(
                        classifier,
                        data=transformed,
                        feature_perturbation="interventional",
                        model_output="probability",
                    )
                else:
                    matrix = _encoded(self.background)
                    masker = shap.maskers.Independent(matrix, max_samples=len(matrix))
                    self.explainer = shap.PermutationExplainer(
                        self._predict_encoded,
                        masker,
                        feature_names=list(FEATURE_COLUMNS),
                        seed=self.random_state,
                    )
            finally:
                np.random.set_state(state)
        if self.method != "permutation":
            names = self.estimator[:-1].get_feature_names_out().tolist()
            self.groups = _native_feature_groups(names)

    def _probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(self.estimator.predict_proba(frame), dtype=float)
        if (
            probabilities.shape != (len(frame), 2)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1)
        ):
            raise ValueError("El modelo devolvió probabilidades inválidas para SHAP.")
        return probabilities[:, self.positive_index]

    def _predict_encoded(self, matrix: np.ndarray) -> np.ndarray:
        # Las coaliciones SHAP son híbridos; no se validan como clientes reales.
        # El pipeline recomputa las derivadas al evaluar cada híbrido.
        return self._probabilities(_decoded(matrix))

    def explain(self, customers: pd.DataFrame) -> ExplanationResult:
        if not 1 <= len(customers) <= self.max_rows:
            raise ValueError(f"SHAP admite entre 1 y {self.max_rows} filas por llamada.")
        frame = validate_customers(customers, mode="training", require_target=False).data
        frame = frame.loc[:, FEATURE_COLUMNS]
        probabilities = self._probabilities(frame)
        rows, bases = [], []
        for position in range(len(frame)):
            row = frame.iloc[[position]]
            with _SHAP_LOCK:
                state = np.random.get_state()
                try:
                    if self.method == "permutation":
                        encoded = _encoded(row)
                        digest = hashlib.sha256(encoded.tobytes()).digest()
                        seed = (int.from_bytes(digest[:4], "little") + self.random_state) % 2**32
                        np.random.seed(seed)
                        explanation = self.explainer(encoded, max_evals=self.max_evals, silent=True)
                    else:
                        explanation = self.explainer(self.estimator[:-1].transform(row))
                finally:
                    np.random.set_state(state)
            values = np.asarray(explanation.values)
            base = np.asarray(explanation.base_values)
            if values.ndim == 3 and values.shape[-1] == 2:
                values = values[:, :, self.positive_index]
                base = base[:, self.positive_index]
            if values.shape != (1, len(self.groups)) or base.size != 1:
                raise ValueError("Dimensiones SHAP incompatibles con clase/variables del modelo.")
            rows.append(values[0])
            bases.append(float(base.reshape(-1)[0]))
        raw_values = np.vstack(rows)
        ordered_groups = [
            name for name in (*FEATURE_COLUMNS, *DERIVED_COLUMNS) if name in self.groups
        ]
        grouped = pd.DataFrame(
            {
                name: raw_values[:, np.asarray(self.groups) == name].sum(axis=1)
                for name in ordered_groups
            }
        )
        explained = (
            np.asarray(self.estimator.decision_function(frame), dtype=float)
            if self.scale == "log_odds"
            else probabilities
        )
        bases = np.asarray(bases)
        if (
            not np.isfinite(grouped.to_numpy()).all()
            or not np.isfinite(bases).all()
            or not np.allclose(
                bases + grouped.sum(axis=1),
                explained,
                atol=SHAP_ADDITIVITY_ATOL,
                rtol=SHAP_ADDITIVITY_ATOL,
            )
        ):
            raise ValueError("SHAP no reconstruye la salida explicada dentro de la tolerancia.")
        feature_values = build_features(frame).loc[:, ordered_groups]
        return ExplanationResult(
            grouped,
            bases,
            explained,
            probabilities,
            feature_values,
            self.method,
            self.scale,
            self.explained_output,
            self.model_scope,
        )


def get_explanation_service(bundle: dict) -> ExplanationService:
    """Caché acotada; modelo/versión/referencia/configuración identifican la instancia."""
    context = bundle.get("explanation_context")
    if not isinstance(context, dict) or not isinstance(context.get("background"), pd.DataFrame):
        raise ExplanationUnavailableError(
            "Falta referencia SHAP real de entrenamiento en el bundle."
        )
    background_hash = hashlib.sha256(
        pd.util.hash_pandas_object(context["background"], index=True).to_numpy().tobytes()
    ).hexdigest()
    parameters = {key: value for key, value in context.items() if key != "background"}
    key = (
        id(bundle.get("pipeline")),
        str(bundle.get("model_version")),
        background_hash,
        json.dumps(parameters, sort_keys=True, allow_nan=False),
    )
    with _SHAP_LOCK:
        if key not in _SERVICES:
            _SERVICES[key] = ExplanationService(bundle)
            while len(_SERVICES) > SHAP_CACHE_SIZE:
                _SERVICES.popitem(last=False)
        _SERVICES.move_to_end(key)
        return _SERVICES[key]


def clear_explanation_cache() -> None:
    with _SHAP_LOCK:
        _SERVICES.clear()


def explain_customer(customer_df: pd.DataFrame, bundle: dict) -> list[dict]:
    if len(customer_df) != 1:
        raise ValueError("explain_customer requiere exactamente un cliente.")
    return get_explanation_service(bundle).explain(customer_df).factors(0)


def save_explanation_plots(
    result: ExplanationResult, output_dir: str | Path, *, customer_position: int = 0
) -> dict:
    """Gráficos SHAP global/individual y valores íntegros, sin reentrenar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    if not 0 <= customer_position < len(result.values):
        raise ValueError("Cliente fuera del resultado de explicabilidad.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    explanation = shap.Explanation(
        values=result.values.to_numpy(),
        base_values=result.base_values,
        data=result.feature_values.map(lambda value: CATEGORY_LABELS.get(value, value)).to_numpy(),
        feature_names=[FEATURE_LABELS[name] for name in result.values.columns],
    )
    unit = "log-odds" if result.scale == "log_odds" else "probabilidad"
    axis = shap.plots.beeswarm(explanation, max_display=SHAP_GLOBAL_DISPLAY_FEATURES, show=False)
    axis.set_title(f"SHAP global — {unit}; n={len(result.values)}")
    axis.set_xlabel(f"Contribución SHAP ({unit})")
    _translate_shap_axes(axis.figure)
    axis.figure.savefig(output / "shap_global.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(axis.figure)
    axis = shap.plots.waterfall(
        explanation[customer_position], max_display=SHAP_LOCAL_DISPLAY_FEATURES, show=False
    )
    axis.set_title(f"Explicación individual — {unit}")
    _translate_shap_axes(axis.figure)
    axis.figure.savefig(output / "shap_individual.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(axis.figure)
    result.values.to_csv(output / "shap_values.csv", index=False)
    metadata = {
        "method": result.method,
        "scale": result.scale,
        "explained_output": result.explained_output,
        "model_scope": result.model_scope,
        "rows": len(result.values),
        "customer_position": customer_position,
        "causal_effects_proven": False,
        "base_value": float(result.base_values[customer_position]),
        "explained_value": float(result.explained_values[customer_position]),
        "churn_probability": float(result.probabilities[customer_position]),
    }
    (output / "explanation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def _translate_shap_axes(figure) -> None:
    """Traduce etiquetas de la biblioteca sin alterar los valores ni su escala."""
    for axis in figure.axes:
        labels = []
        for item in axis.get_yticklabels():
            label = item.get_text()
            label = re.sub(r"Sum of (\d+) other features", r"Suma de \1 variables restantes", label)
            label = label.replace("other features", "variables restantes")
            label = {"High": "Alto", "Low": "Bajo"}.get(label, label)
            labels.append(label)
        axis.set_yticks(axis.get_yticks(), labels=labels)
        if axis.get_ylabel() == "Feature value":
            axis.set_ylabel("Valor de la variable")
