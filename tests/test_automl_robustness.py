import numpy as np
import pandas as pd
import pytest

from modelforge.automl import AutoML


def regression_data(rows=30):
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "feature_1": rng.normal(size=rows),
            "feature_2": rng.normal(size=rows),
            "feature_3": rng.normal(size=rows),
            "target": rng.normal(size=rows),
        }
    )


def classification_data(rows=30):
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "feature_1": rng.normal(size=rows),
            "feature_2": rng.normal(size=rows),
            "feature_3": rng.normal(size=rows),
            "target": np.tile([0, 1], rows // 2 + 1)[:rows],
        }
    )


def test_automl_rejects_missing_target():
    data = regression_data()

    automl = AutoML()

    with pytest.raises(ValueError, match="target must be provided"):
        automl.fit(data)


def test_automl_rejects_invalid_target_column():
    data = regression_data()

    automl = AutoML()

    with pytest.raises((ValueError, KeyError), match="target"):
        automl.fit(data, target="missing_target")


def test_automl_rejects_invalid_model_name():
    data = regression_data()

    automl = AutoML()

    with pytest.raises((ValueError, KeyError)):
        automl.fit(
            data,
            target="target",
            model_names=["model_that_does_not_exist"],
        )


def test_automl_rejects_unsupported_task_type():
    data = regression_data()

    automl = AutoML()

    with pytest.raises(ValueError):
        automl.fit(
            data,
            target="target",
            task_type="unsupported_task",
        )


def test_automl_handles_constant_regression_target():
    data = regression_data()
    data["target"] = 10.0

    automl = AutoML(cv=2)

    result = automl.fit(
        data,
        target="target",
        model_names=["linear_regression"],
    )

    assert result is not None
    assert isinstance(result, dict)


def test_automl_rejects_dataset_with_only_target_column():
    data = pd.DataFrame(
        {
            "target": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    automl = AutoML(cv=2)

    with pytest.raises((ValueError, RuntimeError)):
        automl.fit(
            data,
            target="target",
            model_names=["linear_regression"],
        )


def test_automl_rejects_too_few_samples_for_cv():
    data = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
            "target": [2.0, 4.0, 6.0],
        }
    )

    automl = AutoML(cv=5)

    with pytest.raises((ValueError, RuntimeError)):
        automl.fit(
            data,
            target="target",
            model_names=["linear_regression"],
        )


def test_automl_handles_single_class_classification_cleanly():
    data = classification_data()
    data["target"] = 1

    automl = AutoML(cv=2)

    with pytest.raises((ValueError, RuntimeError)):
        automl.fit(
            data,
            target="target",
            task_type="classification",
            model_names=["logistic_regression"],
        )


def test_automl_handles_insufficient_class_samples():
    data = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5, 6],
            "feature_2": [6, 5, 4, 3, 2, 1],
            "target": [0, 0, 0, 0, 0, 1],
        }
    )

    automl = AutoML(cv=3)

    with pytest.raises((ValueError, RuntimeError)):
        automl.fit(
            data,
            target="target",
            task_type="classification",
            model_names=["logistic_regression"],
        )


def test_automl_handles_missing_values_in_features():
    data = regression_data()

    data.loc[0, "feature_1"] = np.nan
    data.loc[1, "feature_2"] = np.nan
    data.loc[2, "feature_3"] = np.nan

    automl = AutoML(cv=2)

    result = automl.fit(
        data,
        target="target",
        model_names=["linear_regression"],
    )

    assert result is not None