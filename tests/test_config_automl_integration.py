import pandas as pd
import pytest

from modelforge.automl import AutoML
from modelforge.config import ModelForgeConfig


def test_automl_accepts_model_forge_config():
    config = ModelForgeConfig(
        {
            "target": "target",
            "cv": 3,
            "random_state": 123,
            "objective": "performance",
            "experiment_directory": ".modelforge/test-config",
        }
    )

    automl = AutoML(config=config)

    assert automl.config is config
    assert automl.test_size == 0.2
    assert automl.cv == 3
    assert automl.random_state == 123
    assert automl.objective == "performance"


def test_config_controls_feature_selection():
    config = ModelForgeConfig(
        {
            "feature_selection": {
                "variance_threshold": 0.01,
                "correlation_threshold": 0.9,
            }
        }
    )

    automl = AutoML(config=config)

    assert automl.variance_threshold == 0.01
    assert automl.correlation_threshold == 0.9


def test_config_controls_optimization():
    config = ModelForgeConfig(
        {
            "optimization": {
                "enabled": True,
                "models": 2,
                "max_trials": 7,
            }
        }
    )

    automl = AutoML(config=config)

    assert automl.enable_optimization is True
    assert automl.optimization_models == 2
    assert automl.optimization_max_trials == 7


def test_config_controls_experiment_directory(tmp_path):
    experiment_directory = (
        tmp_path / "config-experiments"
    )

    config = ModelForgeConfig(
        {
            "experiment_directory": str(
                experiment_directory
            )
        }
    )

    automl = AutoML(config=config)

    assert (
        automl.experiment_directory
        == experiment_directory
    )


def test_fit_can_use_target_from_config(monkeypatch):
    config = ModelForgeConfig(
        {
            "target": "price",
            "task_type": "regression",
        }
    )

    automl = AutoML(config=config)

    captured = {}

    def fake_workflow(
        data,
        target,
        task_type,
        model_names,
        excluded_columns,
    ):
        captured["target"] = target
        captured["task_type"] = task_type
        captured["model_names"] = model_names
        captured["excluded_columns"] = excluded_columns

        automl.target = target
        automl.task_type = task_type
        automl.best_model = "linear_regression"

        return {
            "models_evaluated": 1,
        }

    monkeypatch.setattr(
        automl,
        "_fit_workflow",
        fake_workflow,
    )

    result = automl.fit(
        data=pd.DataFrame(
            {
                "price": [1, 2, 3],
                "feature": [4, 5, 6],
            }
        ),
        print_report=False,
    )

    assert captured["target"] == "price"
    assert captured["task_type"] == "regression"
    assert captured["model_names"] is None
    assert captured["excluded_columns"] == []
    assert result["run_id"] is not None


def test_explicit_fit_values_override_config():
    config = ModelForgeConfig(
        {
            "target": "config_target",
            "task_type": "regression",
            "models": ["linear_regression"],
            "excluded_columns": ["config_column"],
        }
    )

    automl = AutoML(config=config)

    captured = {}

    def fake_workflow(
        data,
        target,
        task_type,
        model_names,
        excluded_columns,
    ):
        captured["target"] = target
        captured["task_type"] = task_type
        captured["model_names"] = model_names
        captured["excluded_columns"] = excluded_columns

        automl.target = target
        automl.task_type = task_type
        automl.best_model = "linear_regression"

        return {
            "models_evaluated": 1,
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        automl,
        "_fit_workflow",
        fake_workflow,
    )

    try:
        automl.fit(
            data=pd.DataFrame(
                {
                    "explicit_target": [1, 2, 3],
                    "feature": [4, 5, 6],
                }
            ),
            target="explicit_target",
            task_type="regression",
            model_names=["ridge"],
            excluded_columns=["explicit_column"],
            print_report=False,
        )
    finally:
        monkeypatch.undo()

    assert captured["target"] == "explicit_target"
    assert captured["task_type"] == "regression"
    assert captured["model_names"] == ["ridge"]
    assert captured["excluded_columns"] == [
        "explicit_column"
    ]


def test_fit_requires_target_when_config_has_no_target():
    automl = AutoML()

    with pytest.raises(
        ValueError,
        match="target must be provided",
    ):
        automl.fit(
            data=pd.DataFrame(
                {
                    "feature": [1, 2, 3],
                    "value": [4, 5, 6],
                }
            )
        )


def test_configuration_contains_config_driven_values():
    config = ModelForgeConfig(
        {
            "target": "price",
            "task_type": "regression",
            "models": ["linear_regression"],
            "excluded_columns": ["id"],
            "experiment_directory": ".modelforge/custom",
        }
    )

    automl = AutoML(config=config)

    automl.target = "price"
    automl.task_type = "regression"

    configuration = automl._configuration()

    assert configuration["target"] == "price"
    assert configuration["task_type"] == "regression"
    assert configuration["models"] == [
        "linear_regression"
    ]
    assert configuration["excluded_columns"] == ["id"]
    assert (
        configuration["experiment_directory"]
        == ".modelforge/custom"
    )


def test_invalid_config_type_is_rejected():
    with pytest.raises(
        TypeError,
        match="config must be a ModelForgeConfig",
    ):
        AutoML(config={})


def test_default_automl_api_still_works():
    automl = AutoML(
        test_size=0.25,
        cv=3,
        random_state=99,
        objective="speed",
        variance_threshold=0.02,
        correlation_threshold=0.95,
        enable_optimization=True,
        optimization_models=2,
        optimization_max_trials=4,
    )

    assert automl.test_size == 0.25
    assert automl.cv == 3
    assert automl.random_state == 99
    assert automl.objective == "speed"
    assert automl.variance_threshold == 0.02
    assert automl.correlation_threshold == 0.95
    assert automl.enable_optimization is True
    assert automl.optimization_models == 2
    assert automl.optimization_max_trials == 4