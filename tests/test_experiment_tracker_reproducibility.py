import json

import pandas as pd

from modelforge.experiment_tracker import ExperimentTracker


def test_tracker_persists_reproducibility(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    reproducibility = {
        "dataset": {
            "fingerprint": "dataset-sha256"
        },
        "configuration": {
            "fingerprint": "config-sha256"
        },
        "environment": {
            "python_version": "3.13.5",
            "random_state": 42,
        },
    }

    result = {
        "target": "target",
        "task_type": "regression",
        "best_model": "ridge",
        "models_evaluated": 3,
        "run_id": "run-001",
        "status": "completed",
        "run_summary": {
            "run_id": "run-001",
            "status": "completed",
        },
        "reproducibility": reproducibility,
    }

    experiment_id = tracker.record(
        result=result,
        configuration={
            "cv": 5,
            "random_state": 42,
        },
    )

    experiment = tracker.get(
        experiment_id
    )

    assert (
        experiment["reproducibility"]
        == reproducibility
    )


def test_tracker_without_reproducibility_remains_compatible(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    result = {
        "target": "target",
        "task_type": "regression",
        "best_model": "ridge",
        "models_evaluated": 1,
        "run_id": "run-002",
        "status": "completed",
        "run_summary": {
            "run_id": "run-002",
            "status": "completed",
        },
    }

    experiment_id = tracker.record(
        result=result,
    )

    experiment = tracker.get(
        experiment_id
    )

    assert experiment["reproducibility"] is None


def test_list_experiments_exposes_reproducibility(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    result = {
        "target": "target",
        "task_type": "regression",
        "best_model": "ridge",
        "models_evaluated": 1,
        "run_id": "run-003",
        "status": "completed",
        "run_summary": {
            "run_id": "run-003",
            "status": "completed",
        },
        "reproducibility": {
            "dataset": {
                "fingerprint": "abc123"
            }
        },
    }

    tracker.record(result=result)

    experiments = tracker.list_experiments()

    assert len(experiments) == 1

    experiment = experiments[0]

    assert experiment["reproducibility_available"] is True
    assert (
        experiment["reproducibility"]["dataset"][
            "fingerprint"
        ]
        == "abc123"
    )


def test_list_experiments_without_reproducibility(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    result = {
        "target": "target",
        "task_type": "regression",
        "best_model": "ridge",
        "models_evaluated": 1,
        "run_id": "run-004",
        "status": "completed",
        "run_summary": {
            "run_id": "run-004",
            "status": "completed",
        },
    }

    tracker.record(result=result)

    experiments = tracker.list_experiments()

    assert len(experiments) == 1
    assert (
        experiments[0]["reproducibility_available"]
        is False
    )
    assert (
        experiments[0]["reproducibility"] is None
    )


def test_reproducibility_is_json_serializable(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    result = {
        "target": "target",
        "task_type": "classification",
        "best_model": "random_forest_classifier",
        "models_evaluated": 2,
        "run_id": "run-005",
        "status": "completed",
        "run_summary": {
            "run_id": "run-005",
            "status": "completed",
        },
        "reproducibility": {
            "dataset": {
                "fingerprint": "dataset123",
                "rows": 100,
                "columns": 5,
            },
            "configuration": {
                "fingerprint": "config123",
                "values": {
                    "cv": 5,
                    "random_state": 42,
                },
            },
        },
    }

    experiment_id = tracker.record(
        result=result
    )

    path = (
        tmp_path
        / f"{experiment_id}.json"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        loaded = json.load(file)

    assert (
        loaded["reproducibility"]["configuration"][
            "values"
        ]["random_state"]
        == 42
    )


def test_reproducibility_dataframe_values_are_serialized(
    tmp_path,
):
    tracker = ExperimentTracker(tmp_path)

    reproducibility = {
        "dataset": {
            "fingerprint": "abc"
        }
    }

    result = {
        "target": "target",
        "task_type": "regression",
        "screening_results": pd.DataFrame(
            {
                "model": ["ridge"],
                "r2": [0.91],
            }
        ),
        "reproducibility": reproducibility,
    }

    experiment_id = tracker.record(
        result=result
    )

    experiment = tracker.get(
        experiment_id
    )

    assert (
        experiment["reproducibility"]
        == reproducibility
    )

    assert experiment[
        "screening_results"
    ][0]["model"] == "ridge"