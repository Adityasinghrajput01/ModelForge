import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from modelforge.explainability import (
    ExplainabilityEngine,
)
from modelforge.preprocessing import (
    PreprocessingEngine,
)


@pytest.fixture
def regression_data():
    return pd.DataFrame(
        {
            "age": [
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                55,
            ],
            "income": [
                20000,
                25000,
                30000,
                35000,
                40000,
                45000,
                50000,
                55000,
            ],
            "city": [
                "A",
                "A",
                "B",
                "B",
                "A",
                "B",
                "A",
                "B",
            ],
            "target": [
                200,
                250,
                300,
                350,
                400,
                450,
                500,
                550,
            ],
        }
    )


@pytest.fixture
def fitted_linear_pipeline(
    regression_data,
):
    preprocessing = PreprocessingEngine().build(
        data=regression_data,
        target="target",
        model_type="linear_regression",
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    X = regression_data.drop(
        columns=["target"]
    )

    y = regression_data["target"]

    pipeline.fit(X, y)

    return pipeline


def test_feature_importance_from_linear_model(
    fitted_linear_pipeline,
):
    engine = ExplainabilityEngine()

    result = engine.feature_importance(
        fitted_linear_pipeline
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert not result.empty

    assert "feature" in result.columns
    assert "importance" in result.columns
    assert "source" in result.columns
    assert "absolute_importance" in result.columns

    assert (
        result["source"] == "coefficient"
    ).all()


def test_feature_names_are_extracted(
    fitted_linear_pipeline,
):
    engine = ExplainabilityEngine()

    result = engine.feature_importance(
        fitted_linear_pipeline
    )

    assert len(result) > 0

    assert all(
        isinstance(
            feature,
            str,
        )
        for feature in result["feature"]
    )


def test_importance_sorted_descending(
    fitted_linear_pipeline,
):
    engine = ExplainabilityEngine()

    result = engine.feature_importance(
        fitted_linear_pipeline
    )

    values = result[
        "absolute_importance"
    ].to_numpy()

    assert np.all(
        values[:-1] >= values[1:]
    )


def test_permutation_importance(
    fitted_linear_pipeline,
    regression_data,
):
    engine = ExplainabilityEngine()

    X = regression_data.drop(
        columns=["target"]
    )

    y = regression_data["target"]

    result = engine.permutation_importance(
        pipeline=fitted_linear_pipeline,
        X=X,
        y=y,
        n_repeats=3,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert len(result) == X.shape[1]

    assert "feature" in result.columns
    assert "importance" in result.columns
    assert "std" in result.columns
    assert "source" in result.columns

    assert (
        result["source"] == "permutation"
    ).all()


def test_explain_prediction(
    fitted_linear_pipeline,
    regression_data,
):
    engine = ExplainabilityEngine()

    X = regression_data.drop(
        columns=["target"]
    ).head(2)

    result = engine.explain_prediction(
        pipeline=fitted_linear_pipeline,
        X=X,
    )

    assert isinstance(
        result,
        dict,
    )

    assert (
        result["prediction_count"] == 2
    )

    assert len(
        result["predictions"]
    ) == 2

    assert result[
        "feature_importance"
    ] is not None


def test_top_features(
    fitted_linear_pipeline,
):
    engine = ExplainabilityEngine()

    importance = engine.feature_importance(
        fitted_linear_pipeline
    )

    result = engine.top_features(
        importance,
        n=2,
    )

    assert len(result) == 2

    assert (
        result["absolute_importance"]
        .iloc[0]
        >= result[
            "absolute_importance"
        ].iloc[1]
    )


def test_top_features_cannot_be_empty(
    fitted_linear_pipeline,
):
    engine = ExplainabilityEngine()

    importance = engine.feature_importance(
        fitted_linear_pipeline
    )

    with pytest.raises(ValueError):
        engine.top_features(
            importance,
            n=0,
        )


def test_invalid_pipeline():
    engine = ExplainabilityEngine()

    with pytest.raises(TypeError):
        engine.feature_importance(
            "invalid"
        )


def test_pipeline_without_model_step():
    engine = ExplainabilityEngine()

    pipeline = Pipeline(
        steps=[
            (
                "something",
                LinearRegression(),
            )
        ]
    )

    with pytest.raises(ValueError):
        engine.feature_importance(
            pipeline
        )


def test_pipeline_without_preprocessing():
    engine = ExplainabilityEngine()

    pipeline = Pipeline(
        steps=[
            (
                "model",
                LinearRegression(),
            )
        ]
    )

    pipeline.fit(
        np.array(
            [
                [1],
                [2],
                [3],
            ]
        ),
        np.array(
            [
                1,
                2,
                3,
            ]
        ),
    )

    with pytest.raises(ValueError):
        engine.feature_importance(
            pipeline
        )


def test_invalid_permutation_input(
    fitted_linear_pipeline,
):
    engine = ExplainabilityEngine()

    with pytest.raises(TypeError):
        engine.permutation_importance(
            pipeline=fitted_linear_pipeline,
            X="invalid",
            y=[1, 2],
        )


def test_empty_permutation_input(
    fitted_linear_pipeline,
):
    engine = ExplainabilityEngine()

    X = pd.DataFrame()
    y = []

    with pytest.raises(ValueError):
        engine.permutation_importance(
            pipeline=fitted_linear_pipeline,
            X=X,
            y=y,
        )


def test_mismatched_permutation_lengths(
    fitted_linear_pipeline,
):
    engine = ExplainabilityEngine()

    X = pd.DataFrame(
        {
            "age": [20, 25],
            "income": [20000, 25000],
            "city": ["A", "B"],
        }
    )

    y = [1]

    with pytest.raises(ValueError):
        engine.permutation_importance(
            pipeline=fitted_linear_pipeline,
            X=X,
            y=y,
        )


def test_invalid_n_repeats(
    fitted_linear_pipeline,
    regression_data,
):
    engine = ExplainabilityEngine()

    X = regression_data.drop(
        columns=["target"]
    )

    y = regression_data["target"]

    with pytest.raises(ValueError):
        engine.permutation_importance(
            pipeline=fitted_linear_pipeline,
            X=X,
            y=y,
            n_repeats=0,
        )


def test_top_features_invalid_dataframe():
    engine = ExplainabilityEngine()

    with pytest.raises(TypeError):
        engine.top_features(
            "invalid",
            n=2,
        )


def test_top_features_empty_dataframe():
    engine = ExplainabilityEngine()

    with pytest.raises(ValueError):
        engine.top_features(
            pd.DataFrame(),
            n=2,
        )