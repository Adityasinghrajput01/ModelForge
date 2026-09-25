from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from modelforge.reproducibility import ReproducibilityManager


class ReproducibilityIntegration:
    """
    Integrates reproducibility metadata with ModelForge runs,
    experiments, and saved model artifacts.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.manager = ReproducibilityManager(
            random_state=random_state
        )

    def create_run_snapshot(
        self,
        data: pd.DataFrame,
        configuration: dict[str, Any],
        target: str | None = None,
        task_type: str | None = None,
        run_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create reproducibility information for a ModelForge run.
        """

        metadata = dict(extra_metadata or {})

        if run_id is not None:
            metadata["run_id"] = run_id

        return self.manager.create_snapshot(
            data=data,
            configuration=configuration,
            target=target,
            task_type=task_type,
            extra_metadata=metadata,
        )

    def attach_to_result(
        self,
        result: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Attach reproducibility information to a ModelForge result.
        """

        if not isinstance(result, dict):
            raise TypeError("result must be a dictionary.")

        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dictionary.")

        result["reproducibility"] = snapshot

        return result

    def save_snapshot(
        self,
        snapshot: dict[str, Any],
        directory: str | Path,
        run_id: str,
        overwrite: bool = False,
    ) -> str:
        """
        Save a run reproducibility snapshot inside an experiment directory.
        """

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(
                "run_id must be a non-empty string."
            )

        directory = Path(directory)

        path = (
            directory
            / run_id
            / "reproducibility.json"
        )

        return self.manager.save_snapshot(
            snapshot=snapshot,
            path=path,
            overwrite=overwrite,
        )

    def load_snapshot(
        self,
        directory: str | Path,
        run_id: str,
    ) -> dict[str, Any]:
        """
        Load a saved run reproducibility snapshot.
        """

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(
                "run_id must be a non-empty string."
            )

        path = (
            Path(directory)
            / run_id
            / "reproducibility.json"
        )

        return self.manager.load_snapshot(path)

    def attach_artifact_integrity(
        self,
        metadata: dict[str, Any],
        artifact_path: str | Path,
    ) -> dict[str, Any]:
        """
        Attach a SHA-256 fingerprint for a saved model artifact.
        """

        if not isinstance(metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

        fingerprint = self.manager.artifact_fingerprint(
            artifact_path
        )

        updated = dict(metadata)

        updated["artifact"] = {
            "path": str(artifact_path),
            "sha256": fingerprint,
        }

        return updated

    def verify_artifact(
        self,
        artifact_path: str | Path,
        expected_fingerprint: str,
    ) -> bool:
        """
        Verify the integrity of a saved artifact.
        """

        return self.manager.verify_artifact(
            artifact_path,
            expected_fingerprint,
        )

    def verify_dataset(
        self,
        data: pd.DataFrame,
        snapshot: dict[str, Any],
    ) -> bool:
        """
        Verify that a dataset matches the snapshot fingerprint.
        """

        if not isinstance(snapshot, dict):
            raise TypeError(
                "snapshot must be a dictionary."
            )

        dataset = snapshot.get("dataset")

        if not isinstance(dataset, dict):
            raise ValueError(
                "Snapshot does not contain dataset metadata."
            )

        expected = dataset.get("fingerprint")

        if not isinstance(expected, str):
            raise ValueError(
                "Snapshot does not contain a valid "
                "dataset fingerprint."
            )

        return self.manager.verify_dataset(
            data,
            expected,
        )