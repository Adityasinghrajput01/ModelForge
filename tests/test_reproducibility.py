import json

import pandas as pd
import pytest

from modelforge.reproducibility import ReproducibilityManager


def sample_data():
    return pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4],
            "feature_2": [10.0, 20.0, 30.0, 40.0],
            "target": [0, 1, 0, 1],
        }
    )


def test_dataset_fingerprint_is_deterministic():
    manager = ReproducibilityManager()

    data = sample_data()

    first = manager.dataset_fingerprint(data)
    second = manager.dataset_fingerprint(data.copy())

    assert first == second
    assert len(first) == 64


def test_dataset_fingerprint_changes_when_data_changes():
    manager = ReproducibilityManager()

    data = sample_data()
    modified = data.copy()

    modified.loc[0, "feature_1"] = 999

    first = manager.dataset_fingerprint(data)
    second = manager.dataset_fingerprint(modified)

    assert first != second


def test_dataset_fingerprint_rejects_invalid_data():
    manager = ReproducibilityManager()

    with pytest.raises(TypeError):
        manager.dataset_fingerprint([1, 2, 3])


def test_dataset_fingerprint_rejects_empty_data():
    manager = ReproducibilityManager()

    with pytest.raises(ValueError):
        manager.dataset_fingerprint(pd.DataFrame())


def test_configuration_fingerprint_is_deterministic():
    manager = ReproducibilityManager()

    configuration = {
        "cv": 5,
        "random_state": 42,
        "models": ["ridge", "random_forest"],
    }

    first = manager.configuration_fingerprint(configuration)
    second = manager.configuration_fingerprint(
        {
            "models": ["ridge", "random_forest"],
            "random_state": 42,
            "cv": 5,
        }
    )

    assert first == second
    assert len(first) == 64


def test_environment_metadata_contains_required_information():
    manager = ReproducibilityManager(random_state=123)

    metadata = manager.environment_metadata()

    assert metadata["random_state"] == 123
    assert "python_version" in metadata
    assert "numpy_version" in metadata
    assert "pandas_version" in metadata
    assert "scikit_learn_version" in metadata


def test_create_snapshot_contains_reproducibility_information():
    manager = ReproducibilityManager(random_state=42)

    data = sample_data()

    snapshot = manager.create_snapshot(
        data=data,
        configuration={
            "cv": 5,
            "models": 3,
        },
        target="target",
        task_type="classification",
        extra_metadata={
            "run_id": "test-run-001",
        },
    )

    assert "dataset" in snapshot
    assert "configuration" in snapshot
    assert "environment" in snapshot

    assert snapshot["target"] == "target"
    assert snapshot["task_type"] == "classification"

    assert snapshot["dataset"]["rows"] == 4
    assert snapshot["dataset"]["columns"] == 3
    assert len(snapshot["dataset"]["fingerprint"]) == 64

    assert snapshot["configuration"]["values"]["cv"] == 5
    assert snapshot["extra_metadata"]["run_id"] == "test-run-001"


def test_snapshot_save_and_load(tmp_path):
    manager = ReproducibilityManager()

    data = sample_data()

    snapshot = manager.create_snapshot(
        data=data,
        configuration={"cv": 3},
        target="target",
        task_type="classification",
    )

    path = tmp_path / "reproducibility.json"

    saved_path = manager.save_snapshot(
        snapshot,
        path,
    )

    loaded = manager.load_snapshot(saved_path)

    assert loaded == snapshot


def test_snapshot_save_rejects_existing_file(tmp_path):
    manager = ReproducibilityManager()

    path = tmp_path / "reproducibility.json"

    path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        manager.save_snapshot(
            {},
            path,
        )


def test_snapshot_save_overwrite(tmp_path):
    manager = ReproducibilityManager()

    path = tmp_path / "reproducibility.json"

    manager.save_snapshot(
        {"version": 1},
        path,
    )

    manager.save_snapshot(
        {"version": 2},
        path,
        overwrite=True,
    )

    loaded = manager.load_snapshot(path)

    assert loaded["version"] == 2


def test_load_snapshot_missing_file(tmp_path):
    manager = ReproducibilityManager()

    with pytest.raises(FileNotFoundError):
        manager.load_snapshot(
            tmp_path / "missing.json"
        )


def test_artifact_fingerprint_and_verification(tmp_path):
    manager = ReproducibilityManager()

    artifact = tmp_path / "model.txt"

    artifact.write_text(
        "ModelForge artifact",
        encoding="utf-8",
    )

    fingerprint = manager.artifact_fingerprint(artifact)

    assert len(fingerprint) == 64
    assert manager.verify_artifact(
        artifact,
        fingerprint,
    )


def test_artifact_verification_detects_changes(tmp_path):
    manager = ReproducibilityManager()

    artifact = tmp_path / "model.txt"

    artifact.write_text(
        "original",
        encoding="utf-8",
    )

    fingerprint = manager.artifact_fingerprint(artifact)

    artifact.write_text(
        "modified",
        encoding="utf-8",
    )

    assert not manager.verify_artifact(
        artifact,
        fingerprint,
    )


def test_artifact_fingerprint_missing_file(tmp_path):
    manager = ReproducibilityManager()

    with pytest.raises(FileNotFoundError):
        manager.artifact_fingerprint(
            tmp_path / "missing.pkl"
        )


def test_verify_dataset():
    manager = ReproducibilityManager()

    data = sample_data()

    fingerprint = manager.dataset_fingerprint(data)

    assert manager.verify_dataset(
        data,
        fingerprint,
    )

    modified = data.copy()
    modified.loc[0, "target"] = 999

    assert not manager.verify_dataset(
        modified,
        fingerprint,
    )


def test_reproducibility_manager_rejects_invalid_random_state():
    with pytest.raises(TypeError):
        ReproducibilityManager(
            random_state="42"
        )


def test_snapshot_is_json_serializable(tmp_path):
    manager = ReproducibilityManager()

    snapshot = manager.create_snapshot(
        data=sample_data(),
        configuration={
            "cv": 5,
            "random_state": 42,
        },
        target="target",
        task_type="classification",
    )

    path = tmp_path / "snapshot.json"

    manager.save_snapshot(
        snapshot,
        path,
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        loaded = json.load(file)

    assert isinstance(loaded, dict)