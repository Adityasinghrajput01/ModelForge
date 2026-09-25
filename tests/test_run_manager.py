import time

import pytest

from modelforge.run_manager import RunManager


def test_initial_state():
    manager = RunManager()

    assert manager.run_id is None
    assert manager.status == "not_started"
    assert manager.started_at is None
    assert manager.finished_at is None
    assert manager.duration_seconds is None
    assert manager.metadata == {}


def test_start_run():
    manager = RunManager()

    run_id = manager.start(
        {
            "target": "price",
            "task_type": "regression",
        }
    )

    assert run_id.startswith("run_")
    assert manager.run_id == run_id
    assert manager.status == "running"
    assert manager.started_at is not None
    assert manager.is_running() is True
    assert manager.metadata["target"] == "price"


def test_start_running_run_fails():
    manager = RunManager()

    manager.start()

    with pytest.raises(RuntimeError):
        manager.start()


def test_update_metadata():
    manager = RunManager()

    manager.start(
        {
            "target": "price",
        }
    )

    manager.update(
        {
            "models_evaluated": 10,
            "best_model": "random_forest_regressor",
        }
    )

    assert (
        manager.metadata["models_evaluated"]
        == 10
    )

    assert (
        manager.metadata["best_model"]
        == "random_forest_regressor"
    )


def test_update_requires_dictionary():
    manager = RunManager()

    manager.start()

    with pytest.raises(TypeError):
        manager.update("invalid")


def test_complete_run():
    manager = RunManager()

    manager.start(
        {
            "target": "price",
        }
    )

    time.sleep(0.001)

    result = manager.complete(
        {
            "best_model": "ridge",
        }
    )

    assert result["status"] == "completed"
    assert result["run_id"] == manager.run_id
    assert result["finished_at"] is not None
    assert result["duration_seconds"] is not None
    assert result["duration_seconds"] >= 0
    assert (
        result["metadata"]["best_model"]
        == "ridge"
    )


def test_fail_run():
    manager = RunManager()

    manager.start(
        {
            "target": "price",
        }
    )

    result = manager.fail(
        ValueError("training failed")
    )

    assert result["status"] == "failed"
    assert result["finished_at"] is not None
    assert (
        result["metadata"]["error"]
        == "training failed"
    )


def test_fail_with_metadata():
    manager = RunManager()

    manager.start()

    result = manager.fail(
        "pipeline failed",
        {
            "stage": "optimization",
        },
    )

    assert result["status"] == "failed"
    assert (
        result["metadata"]["error"]
        == "pipeline failed"
    )
    assert (
        result["metadata"]["stage"]
        == "optimization"
    )


def test_complete_without_run_fails():
    manager = RunManager()

    with pytest.raises(RuntimeError):
        manager.complete()


def test_fail_without_run_fails():
    manager = RunManager()

    with pytest.raises(RuntimeError):
        manager.fail("error")


def test_update_after_completion_fails():
    manager = RunManager()

    manager.start()
    manager.complete()

    with pytest.raises(RuntimeError):
        manager.update(
            {
                "new": "value",
            }
        )


def test_summary_returns_copy_of_metadata():
    manager = RunManager()

    manager.start(
        {
            "target": "price",
        }
    )

    summary = manager.summary()

    summary["metadata"]["target"] = "changed"

    assert (
        manager.metadata["target"]
        == "price"
    )


def test_reset():
    manager = RunManager()

    manager.start(
        {
            "target": "price",
        }
    )

    manager.complete()

    manager.reset()

    assert manager.run_id is None
    assert manager.status == "not_started"
    assert manager.started_at is None
    assert manager.finished_at is None
    assert manager.duration_seconds is None
    assert manager.metadata == {}
    assert manager.is_running() is False


def test_run_ids_are_unique():
    first = RunManager().start()
    second = RunManager().start()

    assert first != second