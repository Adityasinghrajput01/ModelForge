from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from modelforge import AutoML, automl


def make_regression_data() -> pd.DataFrame:
    rng = np.random.RandomState(42)

    return pd.DataFrame(
        {
            "feature_1": rng.normal(size=100),
            "feature_2": rng.normal(size=100),
            "feature_3": rng.normal(size=100),
            "target": rng.normal(size=100),
        }
    )


def make_classification_data() -> pd.DataFrame:
    rng = np.random.RandomState(42)

    return pd.DataFrame(
        {
            "feature_1": rng.normal(size=100),
            "feature_2": rng.normal(size=100),
            "feature_3": rng.normal(size=100),
            "target": rng.randint(0, 2, size=100),
        }
    )


def regression_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [
                1, 2, 3, 4, 5,
                6, 7, 8, 9, 10,
            ],
            "x2": [
                10, 9, 8, 7, 6,
                5, 4, 3, 2, 1,
            ],
            "target": [
                3, 5, 7, 9, 11,
                13, 15, 17, 19, 21,
            ],
        }
    )


def classification_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [
                1, 2, 3, 4, 5, 6,
                7, 8, 9, 10, 11, 12,
            ],
            "x2": [
                1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2,
            ],
            "target": [
                "A", "A", "A", "A", "A", "A",
                "B", "B", "B", "B", "B", "B",
            ],
        }
    )


# ---------------------------------------------------------------------
# Original AutoML coverage
# ---------------------------------------------------------------------


def test_automl_initial_state():
    automl = AutoML()

    assert automl.is_fitted is False
    assert automl.best_pipeline is None
    assert automl.best_model is None
    assert automl.result is None
    assert automl.target is None
    assert automl.task_type is None


def test_automl_convenience_function_prints_report(
    tmp_path,
    capsys,
):
    data_path = tmp_path / "training.csv"
    regression_data().to_csv(data_path, index=False)

    fitted = automl(
        data_path,
        "target",
        task_type="regression",
        model_names=["linear_regression"],
        cv=2,
        experiment_directory=tmp_path / "experiments",
    )

    output = capsys.readouterr().out
    assert isinstance(fitted, AutoML)
    assert fitted.is_fitted is True
    assert "AUTOFORGE REPORT" in output
    assert "Model Ranking" in output
    assert "Best Model & Settings" in output
    assert "Data Quality" in output
    assert "Run Information" in output
    assert "AUTOFORGE COMPLETE" in output


def test_automl_rejects_invalid_variance_threshold():
    with pytest.raises(ValueError):
        AutoML(
            variance_threshold=-0.1
        )


def test_automl_rejects_invalid_correlation_threshold():
    with pytest.raises(ValueError):
        AutoML(
            correlation_threshold=0
        )


def test_automl_invalid_objective():
    with pytest.raises(ValueError):
        AutoML(
            objective="invalid"
        )


def test_automl_rejects_invalid_cv():
    with pytest.raises(ValueError):
        AutoML(cv=1)


def test_automl_rejects_invalid_test_size():
    with pytest.raises(ValueError):
        AutoML(test_size=1)


def test_automl_fit_regression():
    data = make_regression_data()

    automl = AutoML(
        cv=3,
        objective="balanced",
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
    assert (
        result["target"]["task_type"]
        == "regression"
    )

    assert (
        result["models_evaluated"]
        == 2
    )


def test_automl_fit_classification():
    data = make_classification_data()

    automl = AutoML(
        cv=3,
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

    assert (
        result["target"]["task_type"]
        == "classification"
    )


def test_automl_predict():
    data = make_regression_data()

    train_data = data.iloc[:80]
    test_data = data.iloc[80:].drop(
        columns=["target"]
    )

    automl = AutoML(
        cv=3,
    )

    automl.fit(
        data=train_data,
        target="target",
        task_type="regression",
        model_names=[
            "linear_regression",
        ],
    )

    predictions = automl.predict(
        test_data
    )

    assert isinstance(
        predictions,
        pd.Series,
    )

    assert len(predictions) == len(
        test_data
    )


def test_automl_explain():
    data = make_regression_data()

    automl = AutoML(
        cv=3,
    )

    automl.fit(
        data=data,
        target="target",
        task_type="regression",
        model_names=[
            "linear_regression",
        ],
    )

    explanation = automl.explain(
        top_n=3
    )

    assert isinstance(
        explanation,
        pd.DataFrame,
    )

    assert len(explanation) <= 3


def test_automl_summary():
    data = make_regression_data()

    automl = AutoML(
        cv=3,
        objective="performance",
        variance_threshold=0.01,
        correlation_threshold=0.95,
    )

    automl.fit(
        data=data,
        target="target",
        task_type="regression",
        model_names=[
            "linear_regression",
        ],
    )

    summary = automl.summary()

    assert summary["target"] == "target"
    assert (
        summary["task_type"]
        == "regression"
    )
    assert (
        summary["best_model"]
        == "linear_regression"
    )
    assert (
        summary["objective"]
        == "performance"
    )
    assert (
        summary["variance_threshold"]
        == 0.01
    )
    assert (
        summary["correlation_threshold"]
        == 0.95
    )


def test_automl_result_contains_feature_selection():
    data = make_regression_data()

    automl = AutoML(
        cv=3,
        variance_threshold=0.01,
        correlation_threshold=0.95,
    )

    result = automl.fit(
        data=data,
        target="target",
        task_type="regression",
        model_names=[
            "linear_regression",
        ],
    )

    feature_selection = result[
        "feature_selection"
    ]

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
        == 0.95
    )


def test_automl_excluded_columns():
    data = make_regression_data()

    data["unwanted"] = (
        np.random.RandomState(42).normal(
            size=len(data)
        )
    )

    automl = AutoML(
        cv=3,
    )

    result = automl.fit(
        data=data,
        target="target",
        task_type="regression",
        model_names=[
            "linear_regression",
        ],
        excluded_columns=[
            "unwanted",
        ],
    )

    assert result["best_model"] == (
        "linear_regression"
    )

    assert automl.best_pipeline is not None


def test_automl_invalid_model():
    data = make_regression_data()

    automl = AutoML()

    with pytest.raises(
        ValueError
    ):
        automl.fit(
            data=data,
            target="target",
            task_type="regression",
            model_names=[
                "not_a_real_model",
            ],
        )


def test_automl_load_saved_model(
    tmp_path,
):
    data = make_regression_data()

    automl = AutoML(
        cv=3,
        variance_threshold=0.01,
        correlation_threshold=0.95,
    )

    automl.fit(
        data=data,
        target="target",
        task_type="regression",
        model_names=[
            "linear_regression",
        ],
    )

    model_path = (
        tmp_path / "model.joblib"
    )

    automl.save(
        str(model_path)
    )

    loaded = AutoML()

    loaded.load(
        str(model_path)
    )

    assert loaded.is_fitted is True
    assert (
        loaded.best_model
        == automl.best_model
    )
    assert (
        loaded.target
        == automl.target
    )
    assert (
        loaded.task_type
        == automl.task_type
    )


# ---------------------------------------------------------------------
# Run Manager + Experiment Tracker integration
# ---------------------------------------------------------------------


def test_automl_regression_fit(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    result = automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
            "ridge",
        ],
    )

    assert automl.is_fitted is True
    assert automl.best_pipeline is not None
    assert automl.best_model is not None

    assert result["run_id"].startswith(
        "run_"
    )

    assert result[
        "experiment_id"
    ].startswith("exp_")

    assert result[
        "run_summary"
    ]["status"] == "completed"


def test_automl_creates_experiment_file(
    tmp_path,
):
    experiment_directory = (
        tmp_path / "experiments"
    )

    automl = AutoML(
        cv=2,
        experiment_directory=(
            experiment_directory
        ),
    )

    result = automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    experiment_id = result[
        "experiment_id"
    ]

    path = (
        experiment_directory
        / f"{experiment_id}.json"
    )

    assert path.exists()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved = json.load(file)

    assert (
        saved["best_model"]
        == automl.best_model
    )


def test_automl_summary_contains_run_info(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    summary = automl.summary()

    assert summary["run_id"] is not None
    assert (
        summary["experiment_id"]
        is not None
    )


def test_automl_lists_experiments(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    experiments = (
        automl.list_experiments()
    )

    assert len(experiments) == 1

    assert (
        experiments[0]["best_model"]
        == automl.best_model
    )


def test_automl_get_experiment(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    result = automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    experiment = (
        automl.get_experiment(
            result["experiment_id"]
        )
    )

    assert (
        experiment["experiment_id"]
        == result["experiment_id"]
    )

    assert (
        experiment["best_model"]
        == automl.best_model
    )


def test_automl_classification_run_tracking(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    result = automl.fit(
        classification_data(),
        target="target",
        model_names=[
            "logistic_regression",
        ],
    )

    assert result["run_id"].startswith(
        "run_"
    )

    assert result[
        "experiment_id"
    ].startswith("exp_")

    assert (
        automl.task_type
        == "classification"
    )


def test_automl_failed_run_is_recorded(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    with pytest.raises(
        ValueError
    ):
        automl.fit(
            regression_data(),
            target="missing_target",
        )

    assert automl.run_id is not None

    assert automl.result[
        "status"
    ] == "failed"

    assert (
        automl.experiment_id
        is not None
    )

    experiment = (
        automl.get_experiment(
            automl.experiment_id
        )
    )

    assert (
        experiment["best_model"]
        is None
    )


def test_automl_save_contains_run_metadata(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    model_path = (
        tmp_path / "model.joblib"
    )

    automl.save(
        str(model_path)
    )

    loaded = AutoML(
        cv=2,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    loaded.load(
        str(model_path)
    )

    assert (
        loaded.run_id
        == automl.run_id
    )

    assert (
        loaded.experiment_id
        == automl.experiment_id
    )


def test_automl_experiment_configuration(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        objective="performance",
        enable_optimization=False,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    result = automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    experiment = (
        automl.get_experiment(
            result["experiment_id"]
        )
    )

    configuration = experiment[
        "configuration"
    ]

    assert configuration[
        "cv"
    ] == 2

    assert configuration[
        "objective"
    ] == "performance"

    assert configuration[
        "enable_optimization"
    ] is False


def test_automl_run_summary_contains_duration(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    result = automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    run_summary = result[
        "run_summary"
    ]

    assert (
        run_summary[
            "duration_seconds"
        ]
        is not None
    )

    assert (
        run_summary[
            "duration_seconds"
        ]
        >= 0
    )


def test_automl_multiple_experiments(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    first = automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    second = automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "ridge",
        ],
    )

    assert (
        first["run_id"]
        != second["run_id"]
    )

    assert (
        first["experiment_id"]
        != second["experiment_id"]
    )

    experiments = (
        automl.list_experiments()
    )

    assert len(experiments) == 2