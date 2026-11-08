import io

import numpy as np
import pandas as pd
import pytest

from src.data import read_customer_csv
from src.schema import FEATURE_COLUMNS, INTERNET_FIELDS, InputValidationError, validate_customers


def test_contract_has_19_distinct_predictors():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == 19
    assert "Churn" not in FEATURE_COLUMNS
    assert "customerID" not in FEATURE_COLUMNS


def test_valid_input_is_copied_and_given_id(customer_frame):
    original = customer_frame.copy(deep=True)
    result = validate_customers(customer_frame)
    assert result.data.loc[0, "customerID"] == "fila_1"
    pd.testing.assert_frame_equal(customer_frame, original)


@pytest.mark.parametrize(
    "column,value",
    [
        ("gender", "Other"),
        ("tenure", -1),
        ("tenure", 1.2),
        ("tenure", True),
        ("MonthlyCharges", np.inf),
        ("tenure", np.inf),
        ("TotalCharges", "texto"),
        ("SeniorCitizen", 2),
        ("PhoneService", ""),
        ("MonthlyCharges", ""),
        ("TotalCharges", ""),
    ],
)
def test_invalid_rows_are_reported(customer, column, value):
    customer[column] = value
    with pytest.raises(InputValidationError) as raised:
        validate_customers(pd.DataFrame([customer]))
    assert any(issue.column == column and issue.row == 1 for issue in raised.value.issues)


def test_only_legitimate_zero_tenure_blank_is_normalized(customer):
    customer.update(tenure=0, TotalCharges="  ")
    assert validate_customers(pd.DataFrame([customer])).data.loc[0, "TotalCharges"] == 0
    customer["TotalCharges"] = "error"
    with pytest.raises(InputValidationError):
        validate_customers(pd.DataFrame([customer]))


def test_training_numeric_missing_is_not_silently_imputed(customer):
    customer.update(Churn="No", MonthlyCharges="")
    data = validate_customers(pd.DataFrame([customer]), mode="training").data
    assert pd.isna(data.loc[0, "MonthlyCharges"])


def test_missing_training_label_is_reported(customer_frame):
    with pytest.raises(InputValidationError, match="Churn"):
        validate_customers(customer_frame, mode="training")


def test_no_services_requires_consistent_categories(customer):
    customer.update(PhoneService="No", InternetService="No")
    with pytest.raises(InputValidationError):
        validate_customers(pd.DataFrame([customer]))
    customer["MultipleLines"] = "No phone service"
    customer.update({name: "No internet service" for name in INTERNET_FIELDS})
    assert len(validate_customers(pd.DataFrame([customer])).data) == 1


def test_ids_preserve_order_avoid_collisions_and_warn_duplicates(customer):
    frame = pd.DataFrame(
        [
            {**customer, "customerID": ""},
            {**customer, "customerID": "fila_1"},
            {**customer, "customerID": "repetido"},
            {**customer, "customerID": "repetido"},
        ],
        index=[7, 7, 7, 7],
    )
    result = validate_customers(frame)
    assert result.data["customerID"].tolist() == [
        "fila_1_temporal",
        "fila_1",
        "repetido",
        "repetido",
    ]
    assert [issue.row for issue in result.warnings] == [3, 4]


def test_rejects_unknown_and_duplicate_columns(customer_frame):
    with pytest.raises(InputValidationError, match="Columna no admitida"):
        validate_customers(customer_frame.assign(churn_probability=0.5))
    duplicate = pd.concat([customer_frame, customer_frame[["gender"]]], axis=1)
    with pytest.raises(InputValidationError, match="Encabezado duplicado"):
        validate_customers(duplicate)


def test_csv_limits_and_duplicate_header(customer_frame):
    csv_bytes = customer_frame.to_csv(index=False).encode("utf-8")
    assert len(read_customer_csv(io.BytesIO(csv_bytes)).data) == 1
    with pytest.raises(InputValidationError, match="grande"):
        read_customer_csv(io.BytesIO(csv_bytes), max_bytes=3)
    doubled = pd.concat([customer_frame, customer_frame]).to_csv(index=False).encode()
    with pytest.raises(InputValidationError, match="Demasiadas filas"):
        read_customer_csv(io.BytesIO(doubled), max_rows=1)
    with pytest.raises(InputValidationError, match="Encabezado duplicado"):
        read_customer_csv(io.BytesIO(b"gender,gender\nFemale,Female\n"))


def test_csv_rejects_malformed_empty_and_wrong_encoding():
    for payload in (b"", b"\xff", b'gender\n"unterminated'):
        with pytest.raises(InputValidationError):
            read_customer_csv(io.BytesIO(payload))


def test_csv_rejects_extra_cell_instead_of_using_it_as_an_index(customer_frame):
    header, row = customer_frame.to_csv(index=False).splitlines()
    malformed = f"{header}\ncontenido_ajeno,{row}\n".encode()
    with pytest.raises(InputValidationError, match="Cantidad de columnas"):
        read_customer_csv(io.BytesIO(malformed))
