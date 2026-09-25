import pandas as pd
import pytest

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modelforge.hyperparameter_optimization import (
    HyperparameterOptimizationEngine,
)


def make_regression_data():
    data = pd.DataFrame(
        {
            "feature_0": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
                2.0,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
                2.6,
                2.7,
                2.8,
                2.9,
                3.0,
                3.1,
                3.2,
                3.3,
                3.4,
                3.5,
                3.6,
                3.7,
                3.8,
                3.9,
                4.0,
            ],
            "feature_1": [
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
                2.0,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
                2.6,
                2.7,
                2.8,
                2.9,
                3.0,
                3.1,
                3.2,
                3.3,
                3.4,
                3.5,
                3.6,
                3.7,
                3.8,
                3.9,
                4.0,
                4.1,
                4.2,
                4.3,
                4.4,
                4.5,
                4.6,
                4.7,
                4.8,
                4.9,
            ],
            "feature_2": [
                2.0,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
                2.6,
                2.7,
                2.8,
                2.9,
                3.0,
                3.1,
                3.2,
                3.3,
                3.4,
                3.5,
                3.6,
                3.7,
                3.8,
                3.9,
                4.0,
                4.1,
                4.2,
                4.3,
                4.4,
                4.5,
                4.6,
                4.7,
                4.8,
                4.9,
                5.0,
                5.1,
                5.2,
                5.3,
                5.4,
                5.5,
                5.6,
                5.7,
                5.8,
                5.9,
            ],
        }
    )

    data["target"] = (
        3.0 * data["feature_0"]
        + 2.0 * data["feature_1"]
        + 0.5 * data["feature_2"]
    )

    return data


def make_classification_data():
    iris = load_iris()

    data = pd.DataFrame(
        iris.data,
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
    )

    data["target"] = iris.target

    return data


def test_optimizer_regression():
    data = make_regression_data()

    pipeline = Pipeline(
        [
            (
                "model",
                LinearRegression(),
            )
        ]
    )

    engine = HyperparameterOptimizationEngine(
        cv=3,
        random_state=42,
    )

    result = engine.optimize(
        data=data,
        target="target",
        pipeline=pipeline,
        parameter_space={},
        task_type="regression",
    )

    assert "best_pipeline" in result
    assert result["best_params"] == {}
    assert result["successful_trials"] == 1
    assert result["failed_trials"] == 0


def test_optimizer_classification():
    data = make_classification_data()

    pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                ),
            ),
        ]
    )

    engine = HyperparameterOptimizationEngine(
        cv=3,
        random_state=42,
    )

    result = engine.optimize(
        data=data,
        target="target",
        pipeline=pipeline,
        parameter_space={
            "model__C": [
                0.1,
                1.0,
                10.0,
            ]
        },
        task_type="classification",
    )

    assert result["successful_trials"] == 3
    assert result["failed_trials"] == 0
    assert len(result["trials"]) == 3
    assert result["best_params"]["model__C"] in {
        0.1,
        1.0,
        10.0,
    }


def test_optimizer_respects_max_trials():
    data = make_classification_data()

    pipeline = Pipeline(
        [
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                ),
            )
        ]
    )

    engine = HyperparameterOptimizationEngine(
        cv=3,
        random_state=42,
        max_trials=2,
    )

    result = engine.optimize(
        data=data,
        target="target",
        pipeline=pipeline,
        parameter_space={
            "model__C": [
                0.01,
                0.1,
                1.0,
                10.0,
            ]
        },
        task_type="classification",
    )

    assert len(result["trials"]) == 2


def test_optimizer_all_failed_trials_raise_error():
    data = make_classification_data()

    pipeline = Pipeline(
        [
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                ),
            )
        ]
    )

    engine = HyperparameterOptimizationEngine(
        cv=3,
        random_state=42,
    )

    with pytest.raises(RuntimeError, match="All hyperparameter"):
        engine.optimize(
            data=data,
            target="target",
            pipeline=pipeline,
            parameter_space={
                "model__does_not_exist": [
                    1,
                    2,
                ]
            },
            task_type="classification",
        )


def test_optimizer_invalid_cv():
    with pytest.raises(ValueError):
        HyperparameterOptimizationEngine(
            cv=1
        )


def test_optimizer_invalid_max_trials():
    with pytest.raises(ValueError):
        HyperparameterOptimizationEngine(
            max_trials=0
        )


def test_optimizer_invalid_task():
    data = make_regression_data()

    pipeline = Pipeline(
        [
            (
                "model",
                LinearRegression(),
            )
        ]
    )

    engine = HyperparameterOptimizationEngine(
        cv=3
    )

    with pytest.raises(ValueError):
        engine.optimize(
            data=data,
            target="target",
            pipeline=pipeline,
            parameter_space={},
            task_type="invalid",
        )