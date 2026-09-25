import pytest

from modelforge.config import (
    ModelForgeConfig,
)


def test_default_configuration():
    config = ModelForgeConfig()

    assert (
        config.get("objective")
        == "balanced"
    )

    assert (
        config.get("test_size")
        == 0.2
    )

    assert (
        config.get("cv")
        == 5
    )

    assert (
        config.get("random_state")
        == 42
    )


def test_custom_configuration():
    config = ModelForgeConfig(
        {
            "target": "price",
            "task_type": "regression",
            "objective": "performance",
            "test_size": 0.25,
            "cv": 3,
            "random_state": 100,
            "models": [
                "ridge",
                "random_forest_regressor",
            ],
            "excluded_columns": [
                "id",
            ],
        }
    )

    assert (
        config.get("target")
        == "price"
    )

    assert (
        config.get("task_type")
        == "regression"
    )

    assert (
        config.get("objective")
        == "performance"
    )

    assert (
        config.get("test_size")
        == 0.25
    )

    assert (
        config.get("cv")
        == 3
    )

    assert (
        config.get("random_state")
        == 100
    )

    assert config.get(
        "models"
    ) == [
        "ridge",
        "random_forest_regressor",
    ]

    assert config.get(
        "excluded_columns"
    ) == ["id"]


def test_feature_selection_configuration():
    config = ModelForgeConfig(
        {
            "feature_selection": {
                "variance_threshold": 0.01,
                "correlation_threshold": 0.90,
            }
        }
    )

    feature_selection = config.get(
        "feature_selection"
    )

    assert (
        feature_selection[
            "variance_threshold"
        ]
        == 0.01
    )

    assert (
        feature_selection[
            "correlation_threshold"
        ]
        == 0.90
    )


def test_to_dict_returns_copy():
    config = ModelForgeConfig(
        {
            "target": "price"
        }
    )

    values = config.to_dict()

    values["target"] = "changed"

    assert (
        config.get("target")
        == "price"
    )


def test_unknown_configuration_key():
    with pytest.raises(ValueError):
        ModelForgeConfig(
            {
                "unknown": "value"
            }
        )


def test_invalid_task_type():
    with pytest.raises(ValueError):
        ModelForgeConfig(
            {
                "task_type": "clustering"
            }
        )


def test_invalid_objective():
    with pytest.raises(ValueError):
        ModelForgeConfig(
            {
                "objective": "accuracy"
            }
        )


def test_invalid_test_size():
    with pytest.raises(ValueError):
        ModelForgeConfig(
            {
                "test_size": 1.0
            }
        )


def test_invalid_cv():
    with pytest.raises(ValueError):
        ModelForgeConfig(
            {
                "cv": 1
            }
        )


def test_invalid_models_type():
    with pytest.raises(TypeError):
        ModelForgeConfig(
            {
                "models": "ridge"
            }
        )


def test_empty_models():
    with pytest.raises(ValueError):
        ModelForgeConfig(
            {
                "models": []
            }
        )


def test_invalid_excluded_columns():
    with pytest.raises(TypeError):
        ModelForgeConfig(
            {
                "excluded_columns": "id"
            }
        )


def test_invalid_variance_threshold():
    with pytest.raises(ValueError):
        ModelForgeConfig(
            {
                "feature_selection": {
                    "variance_threshold": -1
                }
            }
        )


def test_invalid_correlation_threshold():
    with pytest.raises(ValueError):
        ModelForgeConfig(
            {
                "feature_selection": {
                    "correlation_threshold": 1.5
                }
            }
        )


def test_save_and_load_config(
    tmp_path,
):
    config = ModelForgeConfig(
        {
            "target": "price",
            "task_type": "regression",
            "objective": "performance",
            "cv": 3,
        }
    )

    config_path = (
        tmp_path / "config.yaml"
    )

    saved_path = config.save(
        str(config_path)
    )

    assert config_path.exists()

    assert saved_path == str(
        config_path.resolve()
    )

    loaded = (
        ModelForgeConfig.from_file(
            str(config_path)
        )
    )

    assert (
        loaded.get("target")
        == "price"
    )

    assert (
        loaded.get("task_type")
        == "regression"
    )

    assert (
        loaded.get("objective")
        == "performance"
    )

    assert (
        loaded.get("cv")
        == 3
    )


def test_missing_config_file(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError
    ):
        ModelForgeConfig.from_file(
            str(
                tmp_path
                / "missing.yaml"
            )
        )


def test_invalid_config_extension(
    tmp_path,
):
    config_path = (
        tmp_path / "config.txt"
    )

    config_path.write_text(
        "target: price"
    )

    with pytest.raises(ValueError):
        ModelForgeConfig.from_file(
            str(config_path)
        )


def test_invalid_yaml_root(
    tmp_path,
):
    config_path = (
        tmp_path / "config.yaml"
    )

    config_path.write_text(
        "- item1\n- item2"
    )

    with pytest.raises(ValueError):
        ModelForgeConfig.from_file(
            str(config_path)
        )


def test_save_existing_config(
    tmp_path,
):
    config = ModelForgeConfig()

    config_path = (
        tmp_path / "config.yaml"
    )

    config.save(
        str(config_path)
    )

    with pytest.raises(
        FileExistsError
    ):
        config.save(
            str(config_path)
        )


def test_save_existing_config_overwrite(
    tmp_path,
):
    config = ModelForgeConfig(
        {
            "target": "price"
        }
    )

    config_path = (
        tmp_path / "config.yaml"
    )

    config.save(
        str(config_path)
    )

    config.save(
        str(config_path),
        overwrite=True,
    )

    loaded = (
        ModelForgeConfig.from_file(
            str(config_path)
        )
    )

    assert (
        loaded.get("target")
        == "price"
    )