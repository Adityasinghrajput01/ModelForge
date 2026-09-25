from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from modelforge.automl import AutoML


def make_regression_dataset(rows: int = 80) -> pd.DataFrame:
    rng = np.random.RandomState(42)

    feature_1 = rng.normal(size=rows)
    feature_2 = rng.normal(size=rows)
    feature_3 = rng.normal(size=rows)

    target = (
        5.0 * feature_1
        - 2.5 * feature_2
        + 1.5 * feature_3
        + rng.normal(scale=0.1, size=rows)
    )

    return pd.DataFrame(
        {
            "feature_1": feature_1,
            "feature_2": feature_2,
            "feature_3": feature_3,
            "target": target,
        }
    )


def make_classification_dataset(rows: int = 80) -> pd.DataFrame:
    rng = np.random.RandomState(42)

    feature_1 = rng.normal(size=rows)
    feature_2 = rng.normal(size=rows)
    feature_3 = rng.normal(size=rows)

    score = feature_1 + feature_2 - 0.5 * feature_3

    target = np.where(score > 0, "positive", "negative")

    return pd.DataFrame(
        {
            "feature_1": feature_1,
            "feature_2": feature_2,
            "feature_3": feature_3,
            "target": target,
        }
    )


# ---------------------------------------------------------------------
# Regression end-to-end workflow
# ---------------------------------------------------------------------


def test_end_to_end_regression_workflow(tmp_path):
    data = make_regression_dataset()

    automl = AutoML(
        cv=2,
        random_state=42,
        experiment_directory=tmp_path / "experiments",
    )

    result = automl.fit(
        data=data,
        target="target",
        task_type="regression",
        model_names=[
            "linear_regression",
            "ridge",
        ],
    )

    assert automl.is_fitted is True
    assert automl.best_pipeline is not None
    assert automl.best_model in {
        "linear_regression",
        "ridge",
    }

    assert result["target"]["target"] == "target"
    assert result["target"]["task_type"] == "regression"

    assert result["models_evaluated"] == 2

    assert not result["screening_results"].empty
    assert not result["cv_results"].empty
    assert not result["ranking"].empty

    assert result["best_pipeline"] is automl.best_pipeline

    assert result["run_id"].startswith("run_")
    assert result["experiment_id"].startswith("exp_")

    assert result["run_summary"]["status"] == "completed"


def test_end_to_end_regression_prediction(tmp_path):
    data = make_regression_dataset()

    train_data = data.iloc[:60].copy()
    test_data = data.iloc[60:].drop(columns=["target"])

    automl = AutoML(
        cv=2,
        experiment_directory=tmp_path / "experiments",
    )

    automl.fit(
        data=train_data,
        target="target",
        task_type="regression",
        model_names=["linear_regression"],
    )

    predictions = automl.predict(test_data)

    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(test_data)
    assert predictions.notna().all()


def test_end_to_end_regression_save_load_predict(tmp_path):
    data = make_regression_dataset()

    train_data = data.iloc[:60].copy()
    test_data = data.iloc[60:].drop(columns=["target"])

    experiment_directory = tmp_path / "experiments"
    model_path = tmp_path / "models" / "regression_model.joblib"

    automl = AutoML(
        cv=2,
        experiment_directory=experiment_directory,
    )

    automl.fit(
        data=train_data,
        target="target",
        task_type="regression",
        model_names=["linear_regression"],
    )

    original_predictions = automl.predict(test_data)

    saved_path = automl.save(str(model_path))

    assert model_path.exists()
    assert saved_path == str(model_path.resolve())

    loaded = AutoML(
        experiment_directory=experiment_directory,
    )

    loaded.load(str(model_path))

    assert loaded.is_fitted is True
    assert loaded.best_model == automl.best_model
    assert loaded.target == automl.target
    assert loaded.task_type == automl.task_type
    assert loaded.run_id == automl.run_id
    assert loaded.experiment_id == automl.experiment_id

    loaded_predictions = loaded.predict(test_data)

    np.testing.assert_allclose(
        original_predictions.to_numpy(),
        loaded_predictions.to_numpy(),
    )


def test_end_to_end_regression_csv_workflow(tmp_path):
    data = make_regression_dataset()

    csv_path = tmp_path / "train.csv"
    data.to_csv(csv_path, index=False)

    automl = AutoML(
        cv=2,
        experiment_directory=tmp_path / "experiments",
    )

    result = automl.fit(
        data=str(csv_path),
        target="target",
        task_type="regression",
        model_names=["linear_regression"],
    )

    assert automl.is_fitted is True
    assert result["target"]["target"] == "target"
    assert result["best_model"] == "linear_regression"


# ---------------------------------------------------------------------
# Classification end-to-end workflow
# ---------------------------------------------------------------------


def test_end_to_end_classification_workflow(tmp_path):
    data = make_classification_dataset()

    automl = AutoML(
        cv=2,
        random_state=42,
        experiment_directory=tmp_path / "experiments",
    )

    result = automl.fit(
        data=data,
        target="target",
        task_type="classification",
        model_names=[
            "logistic_regression",
            "decision_tree_classifier",
        ],
    )

    assert automl.is_fitted is True
    assert automl.best_pipeline is not None

    assert automl.best_model in {
        "logistic_regression",
        "decision_tree_classifier",
    }

    assert result["target"]["task_type"] == "classification"
    assert not result["screening_results"].empty
    assert not result["cv_results"].empty
    assert not result["ranking"].empty

    assert result["run_summary"]["status"] == "completed"


def test_end_to_end_classification_prediction_and_probability(
    tmp_path,
):
    data = make_classification_dataset()

    train_data = data.iloc[:60].copy()
    test_data = data.iloc[60:].drop(columns=["target"])

    automl = AutoML(
        cv=2,
        experiment_directory=tmp_path / "experiments",
    )

    automl.fit(
        data=train_data,
        target="target",
        task_type="classification",
        model_names=["logistic_regression"],
    )

    predictions = automl.predict(test_data)
    probabilities = automl.predict_proba(test_data)

    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(test_data)

    assert isinstance(probabilities, pd.DataFrame)
    assert len(probabilities) == len(test_data)

    assert probabilities.shape[1] == 2

    assert all(
        column.startswith("probability_")
        for column in probabilities.columns
    )

    np.testing.assert_allclose(
        probabilities.sum(axis=1).to_numpy(),
        np.ones(len(test_data)),
        atol=1e-6,
    )


# ---------------------------------------------------------------------
# Optimization end-to-end workflow
# ---------------------------------------------------------------------


def test_end_to_end_optimization_workflow(tmp_path):
    data = make_regression_dataset()

    automl = AutoML(
        cv=2,
        random_state=42,
        enable_optimization=True,
        optimization_models=1,
        optimization_max_trials=1,
        experiment_directory=tmp_path / "experiments",
    )

    result = automl.fit(
        data=data,
        target="target",
        task_type="regression",
        model_names=["ridge"],
    )

    assert automl.is_fitted is True
    assert automl.best_pipeline is not None

    assert result["optimization_enabled"] is True
    assert result["optimization_results"] is not None

    assert "ridge" in result["optimization_results"]

    optimization_result = result["optimization_results"]["ridge"]

    assert optimization_result["best_pipeline"] is not None
    assert optimization_result["successful_trials"] >= 1

    assert result["best_model"] == "ridge"


# ---------------------------------------------------------------------
# Experiment artifact validation
# ---------------------------------------------------------------------


def test_end_to_end_experiment_artifact(tmp_path):
    experiment_directory = tmp_path / "experiments"

    automl = AutoML(
        cv=2,
        experiment_directory=experiment_directory,
    )

    result = automl.fit(
        data=make_regression_dataset(),
        target="target",
        task_type="regression",
        model_names=["linear_regression"],
    )

    experiment_id = result["experiment_id"]

    experiment_path = (
        experiment_directory
        / f"{experiment_id}.json"
    )

    assert experiment_path.exists()

    with experiment_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved = json.load(file)

    assert saved["experiment_id"] == experiment_id
    assert saved["run_id"] == result["run_id"]
    assert saved["status"] == "completed"
    assert saved["best_model"] == automl.best_model

    assert isinstance(saved["target"], dict)
    assert saved["target"]["target"] == "target"
    assert saved["target"]["task_type"] == "regression"

    assert "configuration" in saved
    assert "run_summary" in saved


# ---------------------------------------------------------------------
# Failure-path validation
# ---------------------------------------------------------------------


def test_end_to_end_failure_is_tracked(tmp_path):
    experiment_directory = tmp_path / "experiments"

    automl = AutoML(
        cv=2,
        experiment_directory=experiment_directory,
    )

    with pytest.raises(ValueError):
        automl.fit(
            data=make_regression_dataset(),
            target="does_not_exist",
        )

    assert automl.result is not None
    assert automl.result["status"] == "failed"

    assert automl.run_id is not None
    assert automl.experiment_id is not None

    experiment = automl.get_experiment(
        automl.experiment_id
    )

    assert experiment["status"] == "failed"
    assert experiment["best_model"] is None
    assert experiment["run_id"] == automl.run_id