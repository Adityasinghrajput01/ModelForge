import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from modelforge.model_screening import (
    ModelScreeningEngine,
)
from modelforge.pipeline_generator import (
    PipelineGenerator,
)


@pytest.fixture
def regression_data():
    return pd.DataFrame(
        {
            "feature_1": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
            ],
            "feature_2": [
                2,
                4,
                6,
                8,
                10,
                12,
                14,
                16,
                18,
                20,
                22,
                24,
                26,
                28,
                30,
                32,
                34,
                36,
                38,
                40,
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
                5,
                9,
                13,
                17,
                21,
                25,
                29,
                33,
                37,
                41,
                45,
                49,
                53,
                57,
                61,
                65,
                69,
                73,
                77,
                81,
            ],
        }
    )


@pytest.fixture
def classification_data():
    return pd.DataFrame(
        {
            "feature_1": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
            ],
            "feature_2": [
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
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


def test_regression_screening(
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

    engine = ModelScreeningEngine()

    results = engine.screen(
        data=regression_data,
        target="target",
        pipelines=pipelines,
        task_type="regression",
    )

    assert len(results) == 2

    assert set(
        results["model"]
    ) == {
        "linear_regression",
        "decision_tree",
    }

    assert all(
        results["status"] == "success"
    )

    assert "r2" in results.columns
    assert "mae" in results.columns
    assert "mse" in results.columns
    assert "rmse" in results.columns
    assert (
        "training_time_seconds"
        in results.columns
    )


def test_classification_screening(
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

    engine = ModelScreeningEngine()

    results = engine.screen(
        data=classification_data,
        target="target",
        pipelines=pipelines,
        task_type="classification",
    )

    assert len(results) == 2

    assert all(
        results["status"] == "success"
    )

    assert "accuracy" in results.columns
    assert "precision" in results.columns
    assert "recall" in results.columns
    assert "f1" in results.columns
    assert "roc_auc" in results.columns


def test_training_time_is_recorded(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    engine = ModelScreeningEngine()

    results = engine.screen(
        data=regression_data,
        target="target",
        pipelines={
            "linear_regression": pipeline
        },
        task_type="regression",
    )

    training_time = results.iloc[0][
        "training_time_seconds"
    ]

    assert training_time >= 0


def test_regression_metrics_are_numeric(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    engine = ModelScreeningEngine()

    results = engine.screen(
        data=regression_data,
        target="target",
        pipelines={
            "linear_regression": pipeline
        },
        task_type="regression",
    )

    row = results.iloc[0]

    assert isinstance(
        row["r2"],
        float,
    )

    assert isinstance(
        row["mae"],
        float,
    )

    assert isinstance(
        row["mse"],
        float,
    )

    assert isinstance(
        row["rmse"],
        float,
    )


def test_classification_metrics_are_numeric(
    classification_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=classification_data,
        target="target",
        model_name="logistic_regression",
        task_type="classification",
    )

    engine = ModelScreeningEngine()

    results = engine.screen(
        data=classification_data,
        target="target",
        pipelines={
            "logistic_regression": pipeline
        },
        task_type="classification",
    )

    row = results.iloc[0]

    assert isinstance(
        row["accuracy"],
        float,
    )

    assert isinstance(
        row["precision"],
        float,
    )

    assert isinstance(
        row["recall"],
        float,
    )

    assert isinstance(
        row["f1"],
        float,
    )


def test_failed_pipeline_does_not_crash_screening(
    regression_data,
):
    generator = PipelineGenerator()

    valid_pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    invalid_pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    invalid_pipeline.set_params(
        model__positive="invalid"
    )

    pipelines = {
        "valid_model": valid_pipeline,
        "invalid_model": invalid_pipeline,
    }

    engine = ModelScreeningEngine()

    results = engine.screen(
        data=regression_data,
        target="target",
        pipelines=pipelines,
        task_type="regression",
    )

    assert len(results) == 2

    valid_result = results[
        results["model"] == "valid_model"
    ].iloc[0]

    invalid_result = results[
        results["model"] == "invalid_model"
    ].iloc[0]

    assert valid_result["status"] == "success"
    assert invalid_result["status"] == "failed"
    assert invalid_result["error"] is not None


def test_empty_pipelines_raise_error(
    regression_data,
):
    engine = ModelScreeningEngine()

    with pytest.raises(ValueError):
        engine.screen(
            data=regression_data,
            target="target",
            pipelines={},
            task_type="regression",
        )


def test_invalid_task_type_raises_error(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    engine = ModelScreeningEngine()

    with pytest.raises(ValueError):
        engine.screen(
            data=regression_data,
            target="target",
            pipelines={
                "model": pipeline
            },
            task_type="clustering",
        )


def test_invalid_target_raises_error(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    engine = ModelScreeningEngine()

    with pytest.raises(ValueError):
        engine.screen(
            data=regression_data,
            target="unknown",
            pipelines={
                "model": pipeline
            },
            task_type="regression",
        )


def test_invalid_test_size_raises_error():
    with pytest.raises(ValueError):
        ModelScreeningEngine(
            test_size=1.0
        )


def test_invalid_random_state_raises_error():
    with pytest.raises(TypeError):
        ModelScreeningEngine(
            random_state="42"
        )


def test_screening_preserves_pipeline_names(
    regression_data,
):
    generator = PipelineGenerator()

    first_pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    second_pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="decision_tree_regressor",
        task_type="regression",
    )

    engine = ModelScreeningEngine()

    results = engine.screen(
        data=regression_data,
        target="target",
        pipelines={
            "LinearModel": first_pipeline,
            "TreeModel": second_pipeline,
        },
        task_type="regression",
    )

    assert set(
        results["model"]
    ) == {
        "LinearModel",
        "TreeModel",
    }


def test_results_are_dataframe(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    engine = ModelScreeningEngine()

    results = engine.screen(
        data=regression_data,
        target="target",
        pipelines={
            "linear_regression": pipeline
        },
        task_type="regression",
    )

    assert isinstance(
        results,
        pd.DataFrame,
    )