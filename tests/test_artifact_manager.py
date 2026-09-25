from pathlib import Path

import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from modelforge.artifact_manager import ArtifactManager


def make_pipeline():
    X, y = make_regression(
        n_samples=20,
        n_features=3,
        random_state=42,
    )

    pipeline = Pipeline(
        [
            (
                "model",
                LinearRegression(),
            )
        ]
    )

    pipeline.fit(X, y)

    return pipeline


def test_create_artifact_id():
    manager = ArtifactManager()

    artifact_id = manager.create_artifact_id()

    assert artifact_id.startswith(
        "artifact_"
    )


def test_save_and_load(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    pipeline = make_pipeline()

    artifact_id = manager.save(
        pipeline,
        metadata={
            "target": "price",
            "task_type": "regression",
        },
    )

    loaded = manager.load(
        artifact_id
    )

    assert loaded is not None
    assert manager.exists(
        artifact_id
    )


def test_metadata_persistence(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    pipeline = make_pipeline()

    metadata = {
        "target": "price",
        "task_type": "regression",
        "run_id": "run_001",
        "experiment_id": "exp_001",
    }

    artifact_id = manager.save(
        pipeline,
        metadata=metadata,
    )

    saved = manager.get_metadata(
        artifact_id
    )

    assert saved["artifact_id"] == artifact_id
    assert saved["artifact_type"] == (
        "model_pipeline"
    )

    assert saved["metadata"]["target"] == (
        "price"
    )

    assert saved["metadata"]["run_id"] == (
        "run_001"
    )


def test_custom_artifact_id(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    pipeline = make_pipeline()

    artifact_id = manager.save(
        pipeline,
        artifact_id="artifact_custom_001",
    )

    assert artifact_id == (
        "artifact_custom_001"
    )

    assert manager.exists(
        artifact_id
    )


def test_duplicate_artifact_rejected(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    pipeline = make_pipeline()

    manager.save(
        pipeline,
        artifact_id="artifact_duplicate_001",
    )

    with pytest.raises(
        FileExistsError
    ):
        manager.save(
            pipeline,
            artifact_id="artifact_duplicate_001",
        )


def test_overwrite_artifact(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    pipeline = make_pipeline()

    artifact_id = (
        "artifact_overwrite_001"
    )

    manager.save(
        pipeline,
        metadata={
            "version": 1,
        },
        artifact_id=artifact_id,
    )

    manager.save(
        pipeline,
        metadata={
            "version": 2,
        },
        artifact_id=artifact_id,
        overwrite=True,
    )

    metadata = manager.get_metadata(
        artifact_id
    )

    assert metadata["metadata"]["version"] == 2


def test_list_artifacts(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    pipeline = make_pipeline()

    manager.save(
        pipeline,
        metadata={
            "target": "price",
        },
        artifact_id="artifact_list_001",
    )

    manager.save(
        pipeline,
        metadata={
            "target": "salary",
        },
        artifact_id="artifact_list_002",
    )

    artifacts = manager.list_artifacts()

    assert len(artifacts) == 2

    ids = {
        item["artifact_id"]
        for item in artifacts
    }

    assert "artifact_list_001" in ids
    assert "artifact_list_002" in ids


def test_count(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    pipeline = make_pipeline()

    assert manager.count() == 0

    manager.save(
        pipeline,
        artifact_id="artifact_count_001",
    )

    manager.save(
        pipeline,
        artifact_id="artifact_count_002",
    )

    assert manager.count() == 2


def test_validate_valid_artifact(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    pipeline = make_pipeline()

    artifact_id = manager.save(
        pipeline
    )

    result = manager.validate(
        artifact_id
    )

    assert result["valid"] is True
    assert result["model_exists"] is True
    assert result["metadata_exists"] is True
    assert result["metadata_valid"] is True


def test_validate_missing_metadata(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    pipeline = make_pipeline()

    artifact_id = manager.save(
        pipeline
    )

    manager.metadata_path(
        artifact_id
    ).unlink()

    result = manager.validate(
        artifact_id
    )

    assert result["valid"] is False
    assert result["model_exists"] is True
    assert result["metadata_exists"] is False


def test_delete(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    pipeline = make_pipeline()

    artifact_id = manager.save(
        pipeline
    )

    assert manager.exists(
        artifact_id
    )

    manager.delete(
        artifact_id
    )

    assert not manager.exists(
        artifact_id
    )

    assert not manager.metadata_path(
        artifact_id
    ).exists()


def test_delete_missing_artifact(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    with pytest.raises(
        FileNotFoundError
    ):
        manager.delete(
            "artifact_missing_001"
        )


def test_load_missing_artifact(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    with pytest.raises(
        FileNotFoundError
    ):
        manager.load(
            "artifact_missing_001"
        )


def test_invalid_artifact_id():
    manager = ArtifactManager()

    with pytest.raises(
        ValueError
    ):
        manager.model_path(
            "invalid_id"
        )


def test_none_pipeline_rejected(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    with pytest.raises(
        ValueError
    ):
        manager.save(
            None
        )


def test_metadata_must_be_dictionary(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    pipeline = make_pipeline()

    with pytest.raises(
        TypeError
    ):
        manager.save(
            pipeline,
            metadata="invalid",
        )


def test_safe_nested_metadata(tmp_path):
    manager = ArtifactManager(
        tmp_path
    )

    pipeline = make_pipeline()

    artifact_id = manager.save(
        pipeline,
        metadata={
            "configuration": {
                "cv": 5,
                "models": [
                    "ridge",
                    "random_forest",
                ],
            }
        },
    )

    metadata = manager.get_metadata(
        artifact_id
    )

    assert (
        metadata["metadata"]["configuration"]["cv"]
        == 5
    )

    assert (
        metadata["metadata"]["configuration"]["models"]
        == [
            "ridge",
            "random_forest",
        ]
    )