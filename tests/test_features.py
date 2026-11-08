import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.features import FeatureBuilder, build_features
from src.schema import DERIVED_COLUMNS, FEATURE_COLUMNS, INTERNET_FIELDS


def test_exact_features_without_input_mutation(customer_frame):
    original = customer_frame.copy(deep=True)
    result = build_features(customer_frame)
    assert result.loc[0, list(DERIVED_COLUMNS)].tolist() == [4, 1, 1, 0, 1, 20]
    pd.testing.assert_frame_equal(customer_frame, original)


def test_service_zero_denominator_and_new_customer_boundary(customer):
    customer.update(PhoneService="No", MultipleLines="No phone service", InternetService="No")
    customer.update({name: "No internet service" for name in INTERNET_FIELDS})
    customer.update(tenure=7, MonthlyCharges=0)
    result = build_features(pd.DataFrame([customer]))
    assert result.loc[0, "numero_servicios"] == 0
    assert result.loc[0, "cargo_por_servicio"] == 0
    assert result.loc[0, "cliente_nuevo"] == 0
    assert result.loc[0, "sin_soporte_tecnico"] == 0


def test_missing_tenure_does_not_become_old_customer(customer_frame):
    customer_frame["tenure"] = np.nan
    assert pd.isna(build_features(customer_frame).loc[0, "cliente_nuevo"])


def test_transformer_clone_feature_names_and_fit_contract(customer_frame):
    builder = FeatureBuilder(enabled=False)
    with pytest.raises(NotFittedError):
        builder.transform(customer_frame)
    copied = clone(builder).fit(customer_frame)
    assert copied.get_feature_names_out().tolist() == list(FEATURE_COLUMNS)
    assert copied.transform(customer_frame).columns.tolist() == list(FEATURE_COLUMNS)
    assert FeatureBuilder().fit_transform(customer_frame).shape[1] == 25
