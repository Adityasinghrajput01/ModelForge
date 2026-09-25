from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn


class ReproducibilityManager:
    """
    Creates deterministic fingerprints and reproducibility metadata
    for ModelForge experiments.
    """

    def __init__(self, random_state: int = 42) -> None:
        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer.")

        self.random_state = random_state

    def dataset_fingerprint(
        self,
        data: pd.DataFrame,
    ) -> str:
        """
        Generate a deterministic SHA-256 fingerprint for a DataFrame.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        if data.empty:
            raise ValueError("data must not be empty.")

        normalized = data.copy()

        # Normalize column ordering only through explicit representation.
        # Original column order is preserved because it can affect a pipeline.
        normalized.columns = [str(column) for column in normalized.columns]

        payload = pd.util.hash_pandas_object(
            normalized,
            index=True,
        ).values.tobytes()

        metadata = {
            "columns": normalized.columns.tolist(),
            "dtypes": {
                column: str(dtype)
                for column, dtype in normalized.dtypes.items()
            },
            "shape": list(normalized.shape),
        }

        metadata_bytes = json.dumps(
            metadata,
            sort_keys=True,
            default=str,
        ).encode("utf-8")

        digest = hashlib.sha256()
        digest.update(payload)
        digest.update(metadata_bytes)

        return digest.hexdigest()

    def configuration_fingerprint(
        self,
        configuration: dict[str, Any],
    ) -> str:
        """
        Generate a deterministic SHA-256 fingerprint for configuration.
        """

        if not isinstance(configuration, dict):
            raise TypeError("configuration must be a dictionary.")

        serialized = json.dumps(
            configuration,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def artifact_fingerprint(
        self,
        path: str | Path,
    ) -> str:
        """
        Generate a SHA-256 fingerprint for a file artifact.
        """

        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Artifact does not exist: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Artifact path is not a file: {file_path}"
            )

        digest = hashlib.sha256()

        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        return digest.hexdigest()

    def environment_metadata(self) -> dict[str, Any]:
        """
        Capture environment information required to reproduce an experiment.
        """

        return {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "random_state": self.random_state,
        }

    def create_snapshot(
        self,
        data: pd.DataFrame,
        configuration: dict[str, Any] | None = None,
        target: str | None = None,
        task_type: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a complete reproducibility snapshot for an experiment.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        if data.empty:
            raise ValueError("data must not be empty.")

        configuration = configuration or {}
        dataset_fingerprint = self.dataset_fingerprint(data)
        configuration_fingerprint = (
            self.configuration_fingerprint(configuration)
        )

        snapshot: dict[str, Any] = {
            "dataset_fingerprint": dataset_fingerprint,
            "configuration_fingerprint": configuration_fingerprint,
            "dataset": {
                "fingerprint": dataset_fingerprint,
                "rows": int(data.shape[0]),
                "columns": int(data.shape[1]),
                "column_names": [
                    str(column)
                    for column in data.columns
                ],
            },
            "configuration": {
                "fingerprint": configuration_fingerprint,
                "values": configuration,
            },
            "target": target,
            "task_type": task_type,
            "environment": self.environment_metadata(),
        }

        if extra_metadata:
            snapshot["extra_metadata"] = extra_metadata

        return snapshot

    def save_snapshot(
        self,
        snapshot: dict[str, Any],
        path: str | Path,
        overwrite: bool = False,
    ) -> str:
        """
        Save a reproducibility snapshot as JSON.
        """

        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dictionary.")

        output_path = Path(path)

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Snapshot already exists: {output_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                snapshot,
                file,
                indent=2,
                sort_keys=True,
                default=str,
            )

        return str(output_path)

    def load_snapshot(
        self,
        path: str | Path,
    ) -> dict[str, Any]:
        """
        Load a reproducibility snapshot from JSON.
        """

        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Snapshot does not exist: {input_path}"
            )

        if not input_path.is_file():
            raise ValueError(
                f"Snapshot path is not a file: {input_path}"
            )

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            snapshot = json.load(file)

        if not isinstance(snapshot, dict):
            raise ValueError(
                "Invalid reproducibility snapshot."
            )

        return snapshot

    def verify_dataset(
        self,
        data: pd.DataFrame,
        expected_fingerprint: str,
    ) -> bool:
        """
        Verify that a DataFrame matches an expected fingerprint.
        """

        if not isinstance(expected_fingerprint, str):
            raise TypeError(
                "expected_fingerprint must be a string."
            )

        actual_fingerprint = self.dataset_fingerprint(data)

        return actual_fingerprint == expected_fingerprint

    def verify_artifact(
        self,
        path: str | Path,
        expected_fingerprint: str,
    ) -> bool:
        """
        Verify that a file matches an expected SHA-256 fingerprint.
        """

        if not isinstance(expected_fingerprint, str):
            raise TypeError(
                "expected_fingerprint must be a string."
            )

        actual_fingerprint = self.artifact_fingerprint(path)

        return actual_fingerprint == expected_fingerprint