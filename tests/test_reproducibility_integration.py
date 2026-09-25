from pathlib import Path

import pandas as pd
import pytest

from modelforge.reproducibility_integration import (
    ReproducibilityIntegration,
)


def sample_data():
    return pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4],
            "feature_2": [10.0, 20.0, 30.0, 40.0],
            "target": [0, 1, 0, 1],
        }
    )


def test_create_run_snapshot():
    integration = ReproducibilityIntegration(
        random_state=42
    )

    snapshot = integration.create_run_snapshot(
        data=sample_data(),
        configuration={
            "cv": 5,
            "objective": "balanced",
        },
        target="target",
        task_type="classification",
        run_id="run-001",
    )

    assert "dataset" in snapshot
    assert "configuration" in snapshot
    assert "environment" in snapshot

    assert snapshot["target"] == "target"
    assert snapshot["task_type"] == "classification"

    assert (
        snapshot["extra_metadata"]["run_id"]
        == "run-001"
    )


def test_attach_to_result():
    integration = ReproducibilityIntegration()

    snapshot = {
        "dataset": {
            "fingerprint": "abc",
        }
    }

    result = {
        "best_model": "ridge",
    }

    updated = integration.attach_to_result(
        result,
        snapshot,
    )

    assert updated["best_model"] == "ridge"
    assert updated["reproducibility"] == snapshot


def test_save_and_load_snapshot(tmp_path):
    integration = ReproducibilityIntegration()

    snapshot = integration.create_run_snapshot(
        data=sample_data(),
        configuration={
            "cv": 3,
        },
        target="target",
        task_type="classification",
        run_id="run-123",
    )

    saved = integration.save_snapshot(
        snapshot=snapshot,
        directory=tmp_path,
        run_id="run-123",
    )

    assert Path(saved).exists()

    loaded = integration.load_snapshot(
        directory=tmp_path,
        run_id="run-123",
    )

    assert loaded == snapshot


def test_save_snapshot_creates_run_directory(tmp_path):
    integration = ReproducibilityIntegration()

    snapshot = integration.create_run_snapshot(
        data=sample_data(),
        configuration={},
        target="target",
        task_type="classification",
        run_id="run-abc",
    )

    integration.save_snapshot(
        snapshot=snapshot,
        directory=tmp_path,
        run_id="run-abc",
    )

    assert (
        tmp_path
        / "run-abc"
        / "reproducibility.json"
    ).exists()


def test_save_snapshot_rejects_invalid_run_id(tmp_path):
    integration = ReproducibilityIntegration()

    with pytest.raises(ValueError):
        integration.save_snapshot(
            snapshot={},
            directory=tmp_path,
            run_id="",
        )


def test_load_snapshot_rejects_invalid_run_id(tmp_path):
    integration = ReproducibilityIntegration()

    with pytest.raises(ValueError):
        integration.load_snapshot(
            directory=tmp_path,
            run_id="",
        )


def test_attach_artifact_integrity(tmp_path):
    integration = ReproducibilityIntegration()

    artifact = tmp_path / "model.pkl"

    artifact.write_bytes(
        b"ModelForge test artifact"
    )

    metadata = {
        "best_model": "ridge",
    }

    updated = integration.attach_artifact_integrity(
        metadata=metadata,
        artifact_path=artifact,
    )

    assert updated["best_model"] == "ridge"
    assert "artifact" in updated
    assert updated["artifact"]["path"] == str(
        artifact
    )
    assert len(
        updated["artifact"]["sha256"]
    ) == 64


def test_verify_artifact(tmp_path):
    integration = ReproducibilityIntegration()

    artifact = tmp_path / "model.pkl"

    artifact.write_bytes(
        b"ModelForge artifact"
    )

    metadata = integration.attach_artifact_integrity(
        metadata={},
        artifact_path=artifact,
    )

    fingerprint = metadata["artifact"]["sha256"]

    assert integration.verify_artifact(
        artifact,
        fingerprint,
    )


def test_verify_artifact_detects_modification(tmp_path):
    integration = ReproducibilityIntegration()

    artifact = tmp_path / "model.pkl"

    artifact.write_bytes(
        b"original"
    )

    metadata = integration.attach_artifact_integrity(
        metadata={},
        artifact_path=artifact,
    )

    fingerprint = metadata["artifact"]["sha256"]

    artifact.write_bytes(
        b"modified"
    )

    assert not integration.verify_artifact(
        artifact,
        fingerprint,
    )


def test_verify_dataset():
    integration = ReproducibilityIntegration()

    data = sample_data()

    snapshot = integration.create_run_snapshot(
        data=data,
        configuration={},
        target="target",
        task_type="classification",
    )

    assert integration.verify_dataset(
        data,
        snapshot,
    )


def test_verify_dataset_detects_changes():
    integration = ReproducibilityIntegration()

    data = sample_data()

    snapshot = integration.create_run_snapshot(
        data=data,
        configuration={},
        target="target",
        task_type="classification",
    )

    modified = data.copy()

    modified.loc[0, "feature_1"] = 999

    assert not integration.verify_dataset(
        modified,
        snapshot,
    )


def test_attach_to_result_rejects_invalid_result():
    integration = ReproducibilityIntegration()

    with pytest.raises(TypeError):
        integration.attach_to_result(
            [],
            {},
        )


def test_attach_to_result_rejects_invalid_snapshot():
    integration = ReproducibilityIntegration()

    with pytest.raises(TypeError):
        integration.attach_to_result(
            {},
            [],
        )


def test_artifact_metadata_does_not_mutate_original(
    tmp_path,
):
    integration = ReproducibilityIntegration()

    artifact = tmp_path / "model.pkl"

    artifact.write_bytes(
        b"test"
    )

    original = {
        "best_model": "ridge",
    }

    updated = integration.attach_artifact_integrity(
        metadata=original,
        artifact_path=artifact,
    )

    assert "artifact" not in original
    assert "artifact" in updated