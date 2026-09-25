import numpy as np
import pandas as pd
import pytest

from modelforge.automl import AutoML
from modelforge.persistence import ModelPersistence


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


def categorical_data(rows=30):
    rng = np.random.default_rng(42)

    categories = np.array(["A", "B", "C"])

    return pd.DataFrame(
        {
            "feature_1": rng.normal(size=rows),
            "category": categories[np.arange(rows) % len(categories)],
            "feature_2": rng.normal(size=rows),
            "target": rng.normal(size=rows),
        }
    )


def test_prediction_rejects_missing_feature():
    data = regression_data()

    automl = AutoML(cv=2)

    result = automl.fit(
        data,
        target="target",
        model_names=["linear_regression"],
    )

    assert result is not None

    prediction_data = data.drop(columns=["feature_2", "target"])

    with pytest.raises((ValueError, KeyError)):
        automl.predict(prediction_data)


def test_prediction_rejects_extra_feature():
    data = regression_data()

    automl = AutoML(cv=2)

    automl.fit(
        data,
        target="target",
        model_names=["linear_regression"],
    )

    prediction_data = data.drop(columns=["target"]).copy()
    prediction_data["unexpected_feature"] = 1.0

    with pytest.raises((ValueError, KeyError)):
        automl.predict(prediction_data)


def test_prediction_accepts_different_column_order():
    data = regression_data()

    automl = AutoML(cv=2)

    automl.fit(
        data,
        target="target",
        model_names=["linear_regression"],
    )

    prediction_data = data.drop(columns=["target"])[
        ["feature_3", "feature_1", "feature_2"]
    ]

    predictions = automl.predict(prediction_data)

    assert predictions is not None
    assert len(predictions) == len(prediction_data)


def test_prediction_handles_missing_values():
    data = regression_data()

    automl = AutoML(cv=2)

    automl.fit(
        data,
        target="target",
        model_names=["linear_regression"],
    )

    prediction_data = data.drop(columns=["target"]).copy()

    prediction_data.loc[0, "feature_1"] = np.nan
    prediction_data.loc[1, "feature_2"] = np.nan

    predictions = automl.predict(prediction_data)

    assert predictions is not None
    assert len(predictions) == len(prediction_data)


def test_prediction_rejects_empty_dataframe():
    data = regression_data()

    automl = AutoML(cv=2)

    automl.fit(
        data,
        target="target",
        model_names=["linear_regression"],
    )

    empty_data = pd.DataFrame(
        columns=["feature_1", "feature_2", "feature_3"]
    )

    with pytest.raises((ValueError, RuntimeError)):
        automl.predict(empty_data)


def test_prediction_before_fit_fails_cleanly():
    automl = AutoML()

    prediction_data = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0],
            "feature_2": [3.0, 4.0],
        }
    )

    with pytest.raises((ValueError, RuntimeError, AttributeError)):
        automl.predict(prediction_data)


def test_prediction_handles_unseen_categorical_values():
    data = categorical_data()

    automl = AutoML(cv=2)

    automl.fit(
        data,
        target="target",
        model_names=["linear_regression"],
    )

    prediction_data = data.drop(columns=["target"]).copy()
    prediction_data["category"] = "UNSEEN_CATEGORY"

    predictions = automl.predict(prediction_data)

    assert predictions is not None
    assert len(predictions) == len(prediction_data)


def test_persistence_prediction_rejects_missing_feature(tmp_path):
    data = regression_data()

    automl = AutoML(cv=2)

    automl.fit(
        data,
        target="target",
        model_names=["linear_regression"],
    )

    pipeline = automl.best_pipeline

    model_path = tmp_path / "model.pkl"

    persistence = ModelPersistence()

    persistence.save(
        pipeline,
        str(model_path),
        overwrite=True,
    )

    loaded_pipeline = persistence.load(str(model_path))

    prediction_data = data.drop(
        columns=["feature_3", "target"]
    )

    with pytest.raises((ValueError, KeyError)):
        persistence.predict(
            loaded_pipeline,
            prediction_data,
        )


def test_persistence_prediction_rejects_extra_feature(tmp_path):
    data = regression_data()

    automl = AutoML(cv=2)

    automl.fit(
        data,
        target="target",
        model_names=["linear_regression"],
    )

    pipeline = automl.best_pipeline

    model_path = tmp_path / "model.pkl"

    persistence = ModelPersistence()

    persistence.save(
        pipeline,
        str(model_path),
        overwrite=True,
    )

    loaded_pipeline = persistence.load(str(model_path))

    prediction_data = data.drop(columns=["target"]).copy()
    prediction_data["unexpected_feature"] = 123.0

    with pytest.raises((ValueError, KeyError)):
        persistence.predict(
            loaded_pipeline,
            prediction_data,
        )


def test_persistence_prediction_accepts_valid_data(tmp_path):
    data = regression_data()

    automl = AutoML(cv=2)

    automl.fit(
        data,
        target="target",
        model_names=["linear_regression"],
    )

    pipeline = automl.best_pipeline

    model_path = tmp_path / "model.pkl"

    persistence = ModelPersistence()

    persistence.save(
        pipeline,
        str(model_path),
        overwrite=True,
    )

    loaded_pipeline = persistence.load(str(model_path))

    prediction_data = data.drop(columns=["target"])

    predictions = persistence.predict(
        loaded_pipeline,
        prediction_data,
    )

    assert predictions is not None
    assert len(predictions) == len(prediction_data)