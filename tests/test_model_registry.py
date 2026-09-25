import pytest

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.neighbors import KNeighborsRegressor

from modelforge.model_registry import (
    ModelRegistry,
    ModelSpec,
)


def test_registry_contains_models():
    registry = ModelRegistry()

    assert registry.count() > 0


def test_registry_contains_regression_models():
    registry = ModelRegistry()

    models = registry.list_models(
        "regression"
    )

    assert "linear_regression" in models
    assert "random_forest_regressor" in models
    assert "knn_regressor" in models


def test_registry_contains_classification_models():
    registry = ModelRegistry()

    models = registry.list_models(
        "classification"
    )

    assert "logistic_regression" in models
    assert "random_forest_classifier" in models


def test_regression_and_classification_are_separated():
    registry = ModelRegistry()

    regression_models = registry.get_models(
        "regression"
    )

    classification_models = registry.get_models(
        "classification"
    )

    assert all(
        spec.task_type == "regression"
        for spec in regression_models.values()
    )

    assert all(
        spec.task_type == "classification"
        for spec in classification_models.values()
    )


def test_get_model_spec():
    registry = ModelRegistry()

    spec = registry.get(
        "random_forest_regressor"
    )

    assert isinstance(spec, ModelSpec)
    assert spec.task_type == "regression"
    assert spec.category == "ensemble"


def test_create_linear_regression():
    registry = ModelRegistry()

    model = registry.create(
        "linear_regression"
    )

    assert isinstance(
        model,
        LinearRegression,
    )


def test_create_random_forest_regressor():
    registry = ModelRegistry()

    model = registry.create(
        "random_forest_regressor"
    )

    assert isinstance(
        model,
        RandomForestRegressor,
    )

    assert model.random_state == 42
    assert model.n_jobs == -1


def test_create_random_forest_classifier():
    registry = ModelRegistry()

    model = registry.create(
        "random_forest_classifier"
    )

    assert isinstance(
        model,
        RandomForestClassifier,
    )

    assert model.random_state == 42
    assert model.n_jobs == -1


def test_create_model_with_custom_parameters():
    registry = ModelRegistry()

    model = registry.create(
        "random_forest_regressor",
        n_estimators=250,
        max_depth=8,
    )

    assert model.n_estimators == 250
    assert model.max_depth == 8


def test_logistic_regression_requires_scaling():
    registry = ModelRegistry()

    spec = registry.get(
        "logistic_regression"
    )

    assert isinstance(
        spec.estimator,
        type,
    )

    assert spec.requires_scaling is True
    assert spec.supports_probability is True


def test_tree_models_do_not_require_scaling():
    registry = ModelRegistry()

    regressor = registry.get(
        "random_forest_regressor"
    )

    classifier = registry.get(
        "random_forest_classifier"
    )

    assert regressor.requires_scaling is False
    assert classifier.requires_scaling is False


def test_knn_requires_scaling():
    registry = ModelRegistry()

    spec = registry.get(
        "knn_regressor"
    )

    assert isinstance(
        registry.create("knn_regressor"),
        KNeighborsRegressor,
    )

    assert spec.requires_scaling is True


def test_unknown_model_raises_error():
    registry = ModelRegistry()

    with pytest.raises(KeyError):
        registry.get(
            "unknown_model"
        )


def test_invalid_task_type_raises_error():
    registry = ModelRegistry()

    with pytest.raises(ValueError):
        registry.get_models(
            "clustering"
        )


def test_hyperparameter_space_exists():
    registry = ModelRegistry()

    spec = registry.get(
        "random_forest_regressor"
    )

    assert "n_estimators" in (
        spec.hyperparameter_space
    )

    assert "max_depth" in (
        spec.hyperparameter_space
    )


def test_logistic_regression_model():
    registry = ModelRegistry()

    model = registry.create(
        "logistic_regression"
    )

    assert isinstance(
        model,
        LogisticRegression,
    )

    assert model.max_iter == 1000