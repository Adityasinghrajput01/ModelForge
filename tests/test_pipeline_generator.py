import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.neighbors import (
    KNeighborsRegressor,
)

from modelforge.pipeline_generator import (
    PipelineGenerator,
)


@pytest.fixture
def regression_data():
    return pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45],
            "experience": [1, 3, 5, 7, 10, 12],
            "city": [
                "Delhi",
                "Mumbai",
                "Delhi",
                "Pune",
                "Mumbai",
                "Delhi",
            ],
            "salary": [
                30000,
                40000,
                50000,
                60000,
                75000,
                90000,
            ],
        }
    )


@pytest.fixture
def classification_data():
    return pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45],
            "experience": [1, 3, 5, 7, 10, 12],
            "city": [
                "Delhi",
                "Mumbai",
                "Delhi",
                "Pune",
                "Mumbai",
                "Delhi",
            ],
            "approved": [
                0,
                0,
                0,
                1,
                1,
                1,
            ],
        }
    )


def test_build_regression_pipeline(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="salary",
        model_name="linear_regression",
        task_type="regression",
    )

    assert isinstance(
        pipeline,
        Pipeline,
    )

    assert (
        "preprocessing"
        in pipeline.named_steps
    )

    assert (
        "model"
        in pipeline.named_steps
    )

    assert isinstance(
        pipeline.named_steps["model"],
        LinearRegression,
    )


def test_build_classification_pipeline(
    classification_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=classification_data,
        target="approved",
        model_name="logistic_regression",
        task_type="classification",
    )

    assert isinstance(
        pipeline.named_steps["model"],
        LogisticRegression,
    )


def test_build_random_forest_regression_pipeline(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="salary",
        model_name="random_forest_regressor",
        task_type="regression",
    )

    assert isinstance(
        pipeline.named_steps["model"],
        RandomForestRegressor,
    )


def test_build_random_forest_classification_pipeline(
    classification_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=classification_data,
        target="approved",
        model_name="random_forest_classifier",
        task_type="classification",
    )

    assert isinstance(
        pipeline.named_steps["model"],
        RandomForestClassifier,
    )


def test_build_knn_pipeline(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="salary",
        model_name="knn_regressor",
        task_type="regression",
    )

    assert isinstance(
        pipeline.named_steps["model"],
        KNeighborsRegressor,
    )


def test_pipeline_can_fit(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="salary",
        model_name="linear_regression",
        task_type="regression",
    )

    X = regression_data.drop(
        columns=["salary"]
    )

    y = regression_data["salary"]

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)


def test_pipeline_with_variance_selection(
    regression_data,
):
    data = regression_data.copy()

    data["constant"] = 1

    generator = PipelineGenerator()

    pipeline = generator.build(
        data=data,
        target="salary",
        model_name="linear_regression",
        task_type="regression",
        variance_threshold=0.0,
    )

    assert (
        "variance_selection"
        in pipeline.named_steps
    )


def test_pipeline_with_correlation_selection(
    regression_data,
):
    data = regression_data.copy()

    data["age_copy"] = (
        data["age"] * 2
    )

    generator = PipelineGenerator()

    pipeline = generator.build(
        data=data,
        target="salary",
        model_name="linear_regression",
        task_type="regression",
        correlation_threshold=0.95,
    )

    assert (
        "correlation_selection"
        in pipeline.named_steps
    )


def test_excluded_columns(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="salary",
        model_name="linear_regression",
        task_type="regression",
        excluded_columns=["age"],
    )

    preprocessing = (
        pipeline.named_steps[
            "preprocessing"
        ]
    )

    transformed_columns = []

    for _, _, columns in preprocessing.transformers:
        transformed_columns.extend(
            columns
        )

    assert "age" not in transformed_columns


def test_custom_model_parameters(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="salary",
        model_name="random_forest_regressor",
        task_type="regression",
        model_params={
            "n_estimators": 25,
            "max_depth": 5,
        },
    )

    model = pipeline.named_steps[
        "model"
    ]

    assert model.n_estimators == 25
    assert model.max_depth == 5


def test_wrong_task_type_raises_error(
    regression_data,
):
    generator = PipelineGenerator()

    with pytest.raises(ValueError):
        generator.build(
            data=regression_data,
            target="salary",
            model_name="linear_regression",
            task_type="classification",
        )


def test_unknown_task_type_raises_error(
    regression_data,
):
    generator = PipelineGenerator()

    with pytest.raises(ValueError):
        generator.build(
            data=regression_data,
            target="salary",
            model_name="linear_regression",
            task_type="clustering",
        )


def test_unknown_target_raises_error(
    regression_data,
):
    generator = PipelineGenerator()

    with pytest.raises(ValueError):
        generator.build(
            data=regression_data,
            target="unknown",
            model_name="linear_regression",
            task_type="regression",
        )


def test_empty_dataset_raises_error():
    generator = PipelineGenerator()

    data = pd.DataFrame()

    with pytest.raises(ValueError):
        generator.build(
            data=data,
            target="salary",
            model_name="linear_regression",
            task_type="regression",
        )


def test_pipeline_steps_order(
    regression_data,
):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="salary",
        model_name="linear_regression",
        task_type="regression",
        variance_threshold=0.0,
        correlation_threshold=0.95,
    )

    step_names = [
        name
        for name, _ in pipeline.steps
    ]

    assert step_names == [
        "preprocessing",
        "variance_selection",
        "correlation_selection",
        "model",
    ]