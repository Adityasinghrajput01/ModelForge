from __future__ import annotations

import json
import time

import pandas as pd
import pytest

from modelforge.experiment_tracker import ExperimentTracker


def make_result(
    *,
    target: str = "target",
    task_type: str = "regression",
    run_id: str = "run_test_001",
    status: str = "completed",
) -> dict:
    return {
        "target": {
            "target": target,
            "task_type": task_type,
            "dtype": "float64",
            "unique_values": 100,
        },
        "task_type": task_type,
        "run_id": run_id,
        "status": status,
        "profile": {
            "rows": 100,
            "columns": 4,
        },
        "column_intelligence": {
            "numeric": ["feature_1", "feature_2"],
            "categorical": [],
        },
        "audit": {
            "missing_values": {},
            "leakage": [],
        },
        "models_evaluated": 2,
        "screening_results": pd.DataFrame(
            {
                "model": ["linear_regression", "ridge"],
                "r2": [0.91, 0.90],
            }
        ),
        "cv_results": pd.DataFrame(
            {
                "model": ["linear_regression", "ridge"],
                "cv_r2_mean": [0.89, 0.88],
            }
        ),
        "initial_ranking": pd.DataFrame(
            {
                "model": ["linear_regression", "ridge"],
                "rank": [1, 2],
            }
        ),
        "optimization_enabled": False,
        "optimization_results": None,
        "ranking": pd.DataFrame(
            {
                "model": ["linear_regression", "ridge"],
                "rank": [1, 2],
            }
        ),
        "best_model": "linear_regression",
        "feature_selection": {
            "variance_threshold": None,
            "correlation_threshold": None,
        },
    }


# ---------------------------------------------------------------------
# Creation and persistence
# ---------------------------------------------------------------------


def test_artifact_creates_directory(tmp_path):
    directory = tmp_path / "experiments"

    assert not directory.exists()

    tracker = ExperimentTracker(directory)

    assert directory.exists()
    assert directory.is_dir()


def test_artifact_record_creates_json_file(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    experiment_id = tracker.record(
        result=make_result(),
        configuration={
            "cv": 5,
            "objective": "balanced",
        },
    )

    path = tmp_path / f"{experiment_id}.json"

    assert path.exists()
    assert path.is_file()


def test_artifact_contains_core_metadata(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    experiment_id = tracker.record(
        result=make_result(),
        configuration={
            "cv": 3,
            "objective": "performance",
        },
    )

    saved = tracker.get(experiment_id)

    assert saved["experiment_id"] == experiment_id
    assert saved["target"]["target"] == "target"
    assert saved["target"]["task_type"] == "regression"
    assert saved["run_id"] == "run_test_001"
    assert saved["status"] == "completed"
    assert saved["best_model"] == "linear_regression"

    assert saved["configuration"] == {
        "cv": 3,
        "objective": "performance",
    }


def test_artifact_serializes_dataframes(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    experiment_id = tracker.record(
        result=make_result(),
    )

    saved = tracker.get(experiment_id)

    assert isinstance(
        saved["screening_results"],
        list,
    )

    assert len(
        saved["screening_results"]
    ) == 2

    assert (
        saved["screening_results"][0]["model"]
        == "linear_regression"
    )

    assert isinstance(
        saved["cv_results"],
        list,
    )

    assert isinstance(
        saved["ranking"],
        list,
    )


def test_artifact_timestamp_exists(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    experiment_id = tracker.record(
        result=make_result(),
    )

    saved = tracker.get(experiment_id)

    assert "timestamp" in saved
    assert saved["timestamp"]


# ---------------------------------------------------------------------
# Experiment IDs
# ---------------------------------------------------------------------


def test_artifact_ids_are_unique(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    first = tracker.create_experiment_id()

    second = tracker.create_experiment_id()

    assert first.startswith("exp_")
    assert second.startswith("exp_")
    assert first != second


def test_record_accepts_explicit_experiment_id(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    experiment_id = "exp_custom_001"

    returned = tracker.record(
        result=make_result(),
        experiment_id=experiment_id,
    )

    assert returned == experiment_id

    saved = tracker.get(experiment_id)

    assert saved["experiment_id"] == experiment_id


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------


def test_artifact_get_returns_complete_experiment(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    experiment_id = tracker.record(
        result=make_result(),
    )

    experiment = tracker.get(experiment_id)

    assert isinstance(experiment, dict)
    assert experiment["experiment_id"] == experiment_id
    assert experiment["best_model"] == "linear_regression"


def test_artifact_get_missing_experiment(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    with pytest.raises(FileNotFoundError):
        tracker.get(
            "exp_does_not_exist"
        )


# ---------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------


def test_artifact_list_experiments(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    first = tracker.record(
        result=make_result(
            run_id="run_first",
        ),
    )

    second = tracker.record(
        result=make_result(
            run_id="run_second",
        ),
    )

    experiments = tracker.list_experiments()

    assert len(experiments) == 2

    ids = {
        experiment["experiment_id"]
        for experiment in experiments
    }

    assert first in ids
    assert second in ids


def test_artifact_list_is_newest_first(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    first = tracker.record(
        result=make_result(
            run_id="run_first",
        ),
    )

    time.sleep(0.01)

    second = tracker.record(
        result=make_result(
            run_id="run_second",
        ),
    )

    experiments = tracker.list_experiments()

    assert experiments[0]["experiment_id"] == second
    assert experiments[1]["experiment_id"] == first


def test_artifact_list_preserves_task_type(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    tracker.record(
        result=make_result(
            task_type="classification",
        ),
    )

    experiments = tracker.list_experiments()

    assert len(experiments) == 1
    assert (
        experiments[0]["task_type"]
        == "classification"
    )


def test_artifact_list_preserves_run_information(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    tracker.record(
        result=make_result(
            run_id="run_123",
            status="completed",
        ),
    )

    experiments = tracker.list_experiments()

    assert len(experiments) == 1
    assert experiments[0]["run_id"] == "run_123"
    assert experiments[0]["status"] == "completed"


# ---------------------------------------------------------------------
# Count / delete / clear
# ---------------------------------------------------------------------


def test_artifact_count(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    assert tracker.count() == 0

    tracker.record(
        result=make_result(),
    )

    assert tracker.count() == 1

    tracker.record(
        result=make_result(
            run_id="run_002",
        ),
    )

    assert tracker.count() == 2


def test_artifact_delete(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    experiment_id = tracker.record(
        result=make_result(),
    )

    assert tracker.count() == 1

    tracker.delete(experiment_id)

    assert tracker.count() == 0

    with pytest.raises(FileNotFoundError):
        tracker.get(experiment_id)


def test_artifact_delete_missing_experiment(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    with pytest.raises(FileNotFoundError):
        tracker.delete(
            "exp_does_not_exist"
        )


def test_artifact_clear(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    tracker.record(
        result=make_result(
            run_id="run_001",
        ),
    )

    tracker.record(
        result=make_result(
            run_id="run_002",
        ),
    )

    tracker.record(
        result=make_result(
            run_id="run_003",
        ),
    )

    assert tracker.count() == 3

    deleted = tracker.clear()

    assert deleted == 3
    assert tracker.count() == 0


def test_artifact_clear_empty_directory(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    deleted = tracker.clear()

    assert deleted == 0
    assert tracker.count() == 0


# ---------------------------------------------------------------------
# Validation and corruption handling
# ---------------------------------------------------------------------


def test_artifact_rejects_invalid_result(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    with pytest.raises(TypeError):
        tracker.record(
            result="not a dictionary",
        )


def test_artifact_rejects_invalid_configuration(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    with pytest.raises(TypeError):
        tracker.record(
            result=make_result(),
            configuration="invalid",
        )


def test_artifact_rejects_invalid_experiment_id_type(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    with pytest.raises(TypeError):
        tracker.get(123)


def test_artifact_rejects_empty_experiment_id(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    with pytest.raises(ValueError):
        tracker.get("")


def test_artifact_rejects_invalid_experiment_id(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    with pytest.raises(ValueError):
        tracker.get("invalid_id")


def test_artifact_rejects_path_traversal_id(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    with pytest.raises(ValueError):
        tracker.get(
            "exp_../outside"
        )


def test_artifact_skips_corrupted_json(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    valid_id = tracker.record(
        result=make_result(),
    )

    corrupted_path = (
        tmp_path / "exp_corrupted.json"
    )

    corrupted_path.write_text(
        "{ this is not valid json",
        encoding="utf-8",
    )

    experiments = tracker.list_experiments()

    assert len(experiments) == 1
    assert (
        experiments[0]["experiment_id"]
        == valid_id
    )


# ---------------------------------------------------------------------
# Failed experiment artifacts
# ---------------------------------------------------------------------


def test_artifact_persists_failed_run(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    experiment_id = tracker.record(
        result=make_result(
            run_id="run_failed_001",
            status="failed",
        ),
    )

    saved = tracker.get(experiment_id)

    assert saved["status"] == "failed"
    assert saved["run_id"] == "run_failed_001"


# ---------------------------------------------------------------------
# Multiple experiments remain isolated
# ---------------------------------------------------------------------


def test_artifacts_are_isolated(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    first = tracker.record(
        result=make_result(
            target="price",
            run_id="run_price",
        ),
    )

    second = tracker.record(
        result=make_result(
            target="churn",
            task_type="classification",
            run_id="run_churn",
        ),
    )

    first_data = tracker.get(first)
    second_data = tracker.get(second)

    assert first_data["target"]["target"] == "price"
    assert second_data["target"]["target"] == "churn"

    assert first_data["run_id"] == "run_price"
    assert second_data["run_id"] == "run_churn"

    assert (
        second_data["target"]["task_type"]
        == "classification"
    )