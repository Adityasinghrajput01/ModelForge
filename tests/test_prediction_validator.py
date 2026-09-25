import pandas as pd
import pytest

from modelforge.prediction_validator import (
    PredictionSchemaError,
    PredictionSchemaValidator,
)


def make_data():
    return pd.DataFrame(
        {
            "age": [20, 30, 40],
            "income": [30000, 50000, 70000],
            "experience": [1, 5, 10],
        }
    )


def test_valid_schema():
    validator = PredictionSchemaValidator()

    data = make_data()

    report = validator.validate(
        data,
        [
            "age",
            "income",
            "experience",
        ],
    )

    assert report["valid"] is True
    assert report["missing_columns"] == []
    assert report["unexpected_columns"] == []


def test_missing_column():
    validator = PredictionSchemaValidator()

    data = make_data().drop(
        columns=["experience"]
    )

    with pytest.raises(
        PredictionSchemaError,
        match="Missing required columns",
    ):
        validator.validate(
            data,
            [
                "age",
                "income",
                "experience",
            ],
        )


def test_extra_columns_allowed():
    validator = PredictionSchemaValidator(
        allow_extra_columns=True
    )

    data = make_data()

    data["city"] = [
        "Delhi",
        "Mumbai",
        "Pune",
    ]

    report = validator.validate(
        data,
        [
            "age",
            "income",
            "experience",
        ],
    )

    assert report["valid"] is True
    assert report["unexpected_columns"] == [
        "city"
    ]


def test_extra_columns_rejected():
    validator = PredictionSchemaValidator(
        allow_extra_columns=False
    )

    data = make_data()

    data["city"] = [
        "Delhi",
        "Mumbai",
        "Pune",
    ]

    with pytest.raises(
        PredictionSchemaError,
        match="Unexpected columns",
    ):
        validator.validate(
            data,
            [
                "age",
                "income",
                "experience",
            ],
        )


def test_column_order():
    validator = PredictionSchemaValidator(
        enforce_column_order=True
    )

    data = make_data()[
        [
            "income",
            "age",
            "experience",
        ]
    ]

    with pytest.raises(
        PredictionSchemaError,
        match="Column order",
    ):
        validator.validate(
            data,
            [
                "age",
                "income",
                "experience",
            ],
        )


def test_column_order_report():
    validator = PredictionSchemaValidator()

    data = make_data()[
        [
            "income",
            "age",
            "experience",
        ]
    ]

    report = validator.validate(
        data,
        [
            "age",
            "income",
            "experience",
        ],
    )

    assert report["order_matches"] is False
    assert report["valid"] is True


def test_validate_and_align():
    validator = PredictionSchemaValidator()

    data = make_data()

    data["extra"] = [1, 2, 3]

    aligned = validator.validate_and_align(
        data,
        [
            "experience",
            "age",
            "income",
        ],
    )

    assert list(aligned.columns) == [
        "experience",
        "age",
        "income",
    ]


def test_empty_data_rejected():
    validator = PredictionSchemaValidator()

    data = pd.DataFrame(
        columns=[
            "age",
            "income",
        ]
    )

    with pytest.raises(
        PredictionSchemaError,
        match="empty",
    ):
        validator.validate(
            data,
            [
                "age",
                "income",
            ],
        )


def test_dtype_validation():
    validator = PredictionSchemaValidator(
        enforce_dtypes=True
    )

    data = make_data()

    expected_dtypes = {
        "age": "int64",
        "income": "int64",
        "experience": "int64",
    }

    report = validator.validate(
        data,
        [
            "age",
            "income",
            "experience",
        ],
        expected_dtypes,
    )

    assert report["valid"] is True
    assert report["dtype_mismatches"] == {}


def test_dtype_mismatch():
    validator = PredictionSchemaValidator(
        enforce_dtypes=True
    )

    data = make_data()

    expected_dtypes = {
        "age": "float64",
        "income": "int64",
        "experience": "int64",
    }

    with pytest.raises(
        PredictionSchemaError,
        match="Data type mismatch",
    ):
        validator.validate(
            data,
            [
                "age",
                "income",
                "experience",
            ],
            expected_dtypes,
        )


def test_infer_schema():
    validator = PredictionSchemaValidator()

    data = make_data()

    schema = validator.infer_schema(
        data
    )

    assert schema["n_features"] == 3

    assert schema["columns"] == [
        "age",
        "income",
        "experience",
    ]

    assert schema["dtypes"]["age"] == "int64"


def test_invalid_dataframe():
    validator = PredictionSchemaValidator()

    with pytest.raises(TypeError):
        validator.validate(
            "not a dataframe",
            ["age"],
        )


def test_invalid_expected_columns():
    validator = PredictionSchemaValidator()

    data = make_data()

    with pytest.raises(TypeError):
        validator.validate(
            data,
            "age",
        )


def test_invalid_dtype_schema():
    validator = PredictionSchemaValidator()

    data = make_data()

    with pytest.raises(TypeError):
        validator.validate(
            data,
            [
                "age",
                "income",
                "experience",
            ],
            expected_dtypes="invalid",
        )


def test_format_error():
    error = PredictionSchemaError(
        "Missing required columns: experience"
    )

    message = (
        PredictionSchemaValidator.format_error(
            error
        )
    )

    assert (
        message
        == "Missing required columns: experience"
    )