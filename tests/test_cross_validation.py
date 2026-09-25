import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from modelforge.cross_validation import (
    CrossValidationEngine,
)
from modelforge.pipeline_generator import (
    PipelineGenerator,
)


@pytest.fixture
def regression_data():
    return pd.DataFrame(
        {
            "feature_1": range(1, 21),
            "feature_2": [
                value * 2
                for value in range(1, 21)
            ],
            "category": [
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
            ],
            "target": [
                value * 4 + 1
                for value in range(1, 21)
            ],
        }
    )


@pytest.fixture
def classification_data():
    return pd.DataFrame(
        {
            "feature_1": range(1, 21),
            "feature_2": [
                value * 2
                for value in range(1, 21)
            ],
            "category": [
                "A",
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
                "B",
                "A",
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
                "B",
            ],
            "target": [
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
            ],
        }
    )


def test_regression_cross_validation(
    regression_data,
):
    generator = PipelineGenerator()

    pipelines = {
        "linear_regression": generator.build(
            data=regression_data,
            target="target",
            model_name="linear_regression",
            task_type="regression",
        ),
        "decision_tree": generator.build(
            data=regression_data,
            target="target",
            model_name="decision_tree_regressor",
            task_type="regression",
        ),
    }

    engine = CrossValidationEngine(
        cv=3
    )

    results = engine.evaluate(
        data=regression_data,
        target="target",
        pipelines=pipelines,
        task_type="regression",
    )

    assert len(results) == 2

    assert all(
        results["status"] == "success"
    )

    assert all(
        results["cv_folds"] == 3
    )

    assert all(
        results["successful_folds"] == 3
    )

    assert "cv_mean_r2" in results.columns
    assert "cv_std_r2" in results.columns
    assert "cv_mean_mae" in results.columns
    assert "cv_mean_rmse" in results.columns


def test_classification_cross_validation(
    classification_data,
):
    generator = PipelineGenerator()

    pipelines = {
        "logistic_regression": generator.build(
            data=classification_data,
            target="target",
            model_name="logistic_regression",
            task_type="classification",
        ),
        "decision_tree": generator.build(
            data=classification_data,
            target="target",
            model_name="decision_tree_classifier",
            task_type="classification",
        ),
    }

    engine = CrossValidationEngine(
        cv=3
    )

    results = engine.evaluate(
        data=classification_data,
        target="target",
        pipelines=pipelines,
        task_type="classification",
    )

    assert len(results) == 2

    assert all(
        results["status"] == "success"
    )

    assert "cv_mean_accuracy" in results.columns
    assert "cv_mean_precision" in results.columns
    assert "cv_mean_recall" in results.columns
    assert "cv_mean_f1" in results.columns
    assert "cv_mean_roc_auc" in results.columns


def test_cross_validation_uses_correct_number_of_folds(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    engine = CrossValidationEngine(
        cv=4
    )

    results = engine.evaluate(
        data=regression_data,
        target="target",
        pipelines={
            "linear_regression": pipeline
        },
        task_type="regression",
    )

    assert results.iloc[0]["cv_folds"] == 4
    assert (
        results.iloc[0]["successful_folds"]
        == 4
    )


def test_cv_standard_deviation_is_recorded(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    engine = CrossValidationEngine(
        cv=3
    )

    results = engine.evaluate(
        data=regression_data,
        target="target",
        pipelines={
            "linear_regression": pipeline
        },
        task_type="regression",
    )

    assert (
        results.iloc[0]["cv_std_r2"]
        is not None
    )

    assert (
        results.iloc[0]["cv_std_rmse"]
        is not None
    )


def test_total_time_is_recorded(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    engine = CrossValidationEngine(
        cv=3
    )

    results = engine.evaluate(
        data=regression_data,
        target="target",
        pipelines={
            "linear_regression": pipeline
        },
        task_type="regression",
    )

    assert (
        results.iloc[0]["total_time_seconds"]
        >= 0
    )


def test_unknown_target_raises_error(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    engine = CrossValidationEngine(
        cv=3
    )

    with pytest.raises(ValueError):
        engine.evaluate(
            data=regression_data,
            target="unknown",
            pipelines={
                "model": pipeline
            },
            task_type="regression",
        )


def test_empty_pipeline_dictionary_raises_error(
    regression_data,
):
    engine = CrossValidationEngine(
        cv=3
    )

    with pytest.raises(ValueError):
        engine.evaluate(
            data=regression_data,
            target="target",
            pipelines={},
            task_type="regression",
        )


def test_invalid_cv_value():
    with pytest.raises(ValueError):
        CrossValidationEngine(
            cv=1
        )


def test_invalid_cv_type():
    with pytest.raises(TypeError):
        CrossValidationEngine(
            cv="5"
        )


def test_invalid_random_state():
    with pytest.raises(TypeError):
        CrossValidationEngine(
            random_state="42"
        )


def test_invalid_shuffle():
    with pytest.raises(TypeError):
        CrossValidationEngine(
            shuffle="yes"
        )


def test_invalid_task_type(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    engine = CrossValidationEngine(
        cv=3
    )

    with pytest.raises(ValueError):
        engine.evaluate(
            data=regression_data,
            target="target",
            pipelines={
                "model": pipeline
            },
            task_type="clustering",
        )


def test_non_dataframe_raises_error():
    engine = CrossValidationEngine(
        cv=3
    )

    with pytest.raises(TypeError):
        engine.evaluate(
            data="invalid",
            target="target",
            pipelines={},
            task_type="regression",
        )


def test_non_pipeline_raises_error(
    regression_data,
):
    engine = CrossValidationEngine(
        cv=3
    )

    with pytest.raises(TypeError):
        engine.evaluate(
            data=regression_data,
            target="target",
            pipelines={
                "invalid": "not a pipeline"
            },
            task_type="regression",
        )