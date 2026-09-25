import numpy as np
import pandas as pd
import pytest

from modelforge.feature_engineering import (
    DateTimeFeatureExtractor,
    FeatureEngineeringEngine,
    NumericalFeatureTransformer,
)


def test_datetime_features():

    data = pd.DataFrame(
        {
            "signup_date": pd.to_datetime(
                [
                    "2026-09-25 14:30:00",
                    "2026-09-26 09:15:00",
                ]
            ),
            "age": [20, 25],
        }
    )

    extractor = DateTimeFeatureExtractor(
        datetime_columns=["signup_date"]
    )

    result = extractor.fit_transform(data)

    assert "signup_date" not in result.columns

    assert "signup_date_year" in result.columns
    assert "signup_date_month" in result.columns
    assert "signup_date_day" in result.columns
    assert "signup_date_weekday" in result.columns
    assert "signup_date_quarter" in result.columns
    assert "signup_date_hour" in result.columns
    assert "signup_date_is_weekend" in result.columns

    assert result.loc[0, "signup_date_year"] == 2026
    assert result.loc[0, "signup_date_month"] == 9
    assert result.loc[0, "signup_date_hour"] == 14


def test_weekend_detection():

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-09-25",
                    "2026-09-26",
                ]
            )
        }
    )

    extractor = DateTimeFeatureExtractor(
        datetime_columns=["date"]
    )

    result = extractor.fit_transform(data)

    assert bool(result.loc[0, "date_is_weekend"]) is False
    assert bool(result.loc[1, "date_is_weekend"]) is True


def test_numeric_square():

    data = pd.DataFrame(
        {
            "age": [2, 3, 4],
        }
    )

    transformer = NumericalFeatureTransformer(
        numerical_columns=["age"],
        add_square=True,
    )

    result = transformer.fit_transform(data)

    assert "age_squared" in result.columns
    assert result["age_squared"].tolist() == [
        4,
        9,
        16,
    ]


def test_numeric_log():

    data = pd.DataFrame(
        {
            "income": [0, 9, 99],
        }
    )

    transformer = NumericalFeatureTransformer(
        numerical_columns=["income"],
        add_log=True,
    )

    result = transformer.fit_transform(data)

    assert "income_log1p" in result.columns

    assert np.isclose(
        result.loc[0, "income_log1p"],
        0,
    )

    assert np.isclose(
        result.loc[1, "income_log1p"],
        np.log1p(9),
    )


def test_numeric_sqrt():

    data = pd.DataFrame(
        {
            "value": [0, 4, 9],
        }
    )

    transformer = NumericalFeatureTransformer(
        numerical_columns=["value"],
        add_sqrt=True,
    )

    result = transformer.fit_transform(data)

    assert "value_sqrt" in result.columns

    assert result["value_sqrt"].tolist() == [
        0,
        2,
        3,
    ]


def test_negative_values_are_handled_safely():

    data = pd.DataFrame(
        {
            "value": [-5, 0, 4],
        }
    )

    transformer = NumericalFeatureTransformer(
        numerical_columns=["value"],
        add_log=True,
        add_sqrt=True,
    )

    result = transformer.fit_transform(data)

    assert np.isnan(
        result.loc[0, "value_log1p"]
    )

    assert np.isnan(
        result.loc[0, "value_sqrt"]
    )

    assert np.isclose(
        result.loc[2, "value_sqrt"],
        2,
    )


def test_engine_datetime():

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01"]
            )
        }
    )

    engine = FeatureEngineeringEngine()

    result = engine.transform_datetime(data)

    assert "date_year" in result.columns
    assert result.loc[0, "date_year"] == 2026


def test_engine_numeric():

    data = pd.DataFrame(
        {
            "age": [2, 3, 4]
        }
    )

    engine = FeatureEngineeringEngine()

    result = engine.transform_numeric(
        data,
        numerical_columns=["age"],
        add_square=True,
    )

    assert "age_squared" in result.columns


def test_invalid_input():

    extractor = DateTimeFeatureExtractor()

    with pytest.raises(TypeError):
        extractor.fit([1, 2, 3])