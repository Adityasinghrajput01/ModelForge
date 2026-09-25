from __future__ import annotations

import json

import pandas as pd
import pytest

from modelforge.experiment_tracker import (
    ExperimentTracker,
)


def sample_result() -> dict:
    return {
        "target": {
            "target": "price",
            "task_type": "regression",
        },
        "profile": {
            "rows": 20,
            "columns": 7,
        },
        "column_intelligence": {
            "numeric_columns": [
                "area",
                "bedrooms",
            ],
        },
        "audit": {
            "leakage_detected": False,
        },
        "models_evaluated": 3,
        "screening_results": pd.DataFrame(
            [
                {
                    "model": "ridge",
                    "r2": 0.82,
                },
                {
                    "model": "random_forest_regressor",
                    "r2": 0.89,
                },
            ]
        ),
        "cv_results": pd.DataFrame(
            [
                {
                    "model": "ridge",
                    "mean_r2": 0.80,
                },
                {
                    "model": "random_forest_regressor",
                    "mean_r2": 0.87,
                },
            ]
        ),
        "initial_ranking": pd.DataFrame(
            [
                {
                    "model": "random_forest_regressor",
                    "rank": 1,
                    "status": "success",
                },
                {
                    "model": "ridge",
                    "rank": 2,
                    "status": "success",
                },
            ]
        ),
        "optimization_enabled": True,
        "optimization_results": {
            "random_forest_regressor": {
                "best_params": {
                    "model__n_estimators": 100,
                },
                "best_score": 0.91,
                "successful_trials": 5,
                "failed_trials": 0,
            }
        },
        "ranking": pd.DataFrame(
            [
                {
                    "model": "random_forest_regressor",
                    "rank": 1,
                    "status": "success",
                }
            ]
        ),
        "best_model": "random_forest_regressor",
        "feature_selection": {
            "variance_threshold": 0.0,
            "correlation_threshold": 0.95,
        },
    }


def test_tracker_creates_directory(tmp_path):
    tracker = ExperimentTracker(
        tmp_path / "experiments"
    )

    assert tracker.directory.exists()
    assert tracker.count() == 0


def test_create_experiment_id():
    tracker = ExperimentTracker()

    experiment_id = (
        tracker.create_experiment_id()
    )

    assert experiment_id.startswith(
        "exp_"
    )
    assert len(experiment_id) > 12


def test_record_and_get(tmp_path):
    tracker = ExperimentTracker(
        tmp_path / "experiments"
    )

    experiment_id = tracker.record(
        sample_result(),
        configuration={
            "cv": 5,
            "random_state": 42,
        },
    )

    experiment = tracker.get(
        experiment_id
    )

    assert (
        experiment["experiment_id"]
        == experiment_id
    )

    assert (
        experiment["best_model"]
        == "random_forest_regressor"
    )

    assert (
        experiment["optimization_enabled"]
        is True
    )

    assert (
        experiment["configuration"]["cv"]
        == 5
    )


def test_record_creates_json_file(tmp_path):
    tracker = ExperimentTracker(
        tmp_path / "experiments"
    )

    experiment_id = tracker.record(
        sample_result()
    )

    path = (
        tmp_path
        / "experiments"
        / f"{experiment_id}.json"
    )

    assert path.exists()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert data["best_model"] == (
        "random_forest_regressor"
    )


def test_list_experiments(tmp_path):
    tracker = ExperimentTracker(
        tmp_path / "experiments"
    )

    first = tracker.record(
        sample_result()
    )

    second = tracker.record(
        sample_result()
    )

    experiments = (
        tracker.list_experiments()
    )

    assert len(experiments) == 2

    ids = {
        item["experiment_id"]
        for item in experiments
    }

    assert first in ids
    assert second in ids

    assert (
        experiments[0]["task_type"]
        == "regression"
    )


def test_delete_experiment(tmp_path):
    tracker = ExperimentTracker(
        tmp_path / "experiments"
    )

    experiment_id = tracker.record(
        sample_result()
    )

    assert tracker.count() == 1

    tracker.delete(
        experiment_id
    )

    assert tracker.count() == 0

    with pytest.raises(
        FileNotFoundError
    ):
        tracker.get(
            experiment_id
        )


def test_clear_experiments(tmp_path):
    tracker = ExperimentTracker(
        tmp_path / "experiments"
    )

    tracker.record(sample_result())
    tracker.record(sample_result())
    tracker.record(sample_result())

    assert tracker.count() == 3

    deleted = tracker.clear()

    assert deleted == 3
    assert tracker.count() == 0


def test_invalid_experiment_id(tmp_path):
    tracker = ExperimentTracker(
        tmp_path / "experiments"
    )

    with pytest.raises(
        ValueError
    ):
        tracker.get("invalid")

    with pytest.raises(
        ValueError
    ):
        tracker.get(
            "../exp_test"
        )


def test_missing_experiment(tmp_path):
    tracker = ExperimentTracker(
        tmp_path / "experiments"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        tracker.get(
            "exp_does_not_exist"
        )


def test_invalid_record_input(tmp_path):
    tracker = ExperimentTracker(
        tmp_path / "experiments"
    )

    with pytest.raises(
        TypeError
    ):
        tracker.record(
            "invalid"
        )

    with pytest.raises(
        TypeError
    ):
        tracker.record(
            sample_result(),
            configuration="invalid",
        )