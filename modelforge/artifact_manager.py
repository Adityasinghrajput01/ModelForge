from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


class ArtifactManager:
    """
    Manage persisted ModelForge model artifacts and metadata.

    Each artifact consists of:
        <artifact_id>.pkl
        <artifact_id>.json

    The JSON file stores metadata describing the trained model.
    """

    def __init__(
        self,
        directory: str | Path = ".modelforge/artifacts",
    ):
        self.directory = Path(directory)

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def create_artifact_id(
        self,
    ) -> str:
        """
        Generate a unique artifact identifier.
        """

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d%H%M%S")

        short_uuid = uuid.uuid4().hex[:8]

        return f"artifact_{timestamp}_{short_uuid}"

    def save(
        self,
        pipeline: Any,
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
        overwrite: bool = False,
    ) -> str:
        """
        Save a trained pipeline and its metadata.

        Returns:
            Artifact ID.
        """

        if pipeline is None:
            raise ValueError(
                "pipeline cannot be None."
            )

        if metadata is not None:
            if not isinstance(
                metadata,
                dict,
            ):
                raise TypeError(
                    "metadata must be a dictionary."
                )

        artifact_id = (
            artifact_id
            or self.create_artifact_id()
        )

        self._validate_artifact_id(
            artifact_id
        )

        model_path = self.model_path(
            artifact_id
        )

        metadata_path = self.metadata_path(
            artifact_id
        )

        if not overwrite:
            if model_path.exists():
                raise FileExistsError(
                    f"Artifact '{artifact_id}' "
                    "already exists."
                )

            if metadata_path.exists():
                raise FileExistsError(
                    f"Metadata for artifact "
                    f"'{artifact_id}' already exists."
                )

        artifact_metadata = {
            "artifact_id": artifact_id,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "artifact_type": "model_pipeline",
            "metadata": self._safe_value(
                metadata or {}
            ),
        }

        joblib.dump(
            pipeline,
            model_path,
        )

        self._write_json(
            metadata_path,
            artifact_metadata,
        )

        return artifact_id

    def load(
        self,
        artifact_id: str,
    ) -> Any:
        """
        Load a saved model pipeline.
        """

        self._validate_artifact_id(
            artifact_id
        )

        path = self.model_path(
            artifact_id
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Artifact '{artifact_id}' "
                "does not exist."
            )

        return joblib.load(path)

    def get_metadata(
        self,
        artifact_id: str,
    ) -> dict[str, Any]:
        """
        Load artifact metadata.
        """

        self._validate_artifact_id(
            artifact_id
        )

        path = self.metadata_path(
            artifact_id
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Metadata for artifact "
                f"'{artifact_id}' does not exist."
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def exists(
        self,
        artifact_id: str,
    ) -> bool:
        """
        Check whether an artifact exists.
        """

        self._validate_artifact_id(
            artifact_id
        )

        return self.model_path(
            artifact_id
        ).exists()

    def delete(
        self,
        artifact_id: str,
    ) -> None:
        """
        Delete an artifact and its metadata.
        """

        self._validate_artifact_id(
            artifact_id
        )

        model_path = self.model_path(
            artifact_id
        )

        metadata_path = self.metadata_path(
            artifact_id
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Artifact '{artifact_id}' "
                "does not exist."
            )

        model_path.unlink()

        if metadata_path.exists():
            metadata_path.unlink()

    def list_artifacts(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return metadata for all saved artifacts.

        Results are ordered newest first.
        """

        artifacts = []

        for path in self.directory.glob(
            "artifact_*.json"
        ):
            try:
                with path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    metadata = json.load(file)

                artifact_id = metadata.get(
                    "artifact_id"
                )

                if not artifact_id:
                    continue

                artifacts.append(
                    metadata
                )

            except (
                OSError,
                json.JSONDecodeError,
            ):
                continue

        artifacts.sort(
            key=lambda item: item.get(
                "created_at",
                "",
            ),
            reverse=True,
        )

        return artifacts

    def count(
        self,
    ) -> int:
        """
        Return the number of saved artifacts.
        """

        return len(
            list(
                self.directory.glob(
                    "artifact_*.pkl"
                )
            )
        )

    def model_path(
        self,
        artifact_id: str,
    ) -> Path:
        """
        Return the model path for an artifact.
        """

        self._validate_artifact_id(
            artifact_id
        )

        return (
            self.directory
            / f"{artifact_id}.pkl"
        )

    def metadata_path(
        self,
        artifact_id: str,
    ) -> Path:
        """
        Return the metadata path for an artifact.
        """

        self._validate_artifact_id(
            artifact_id
        )

        return (
            self.directory
            / f"{artifact_id}.json"
        )

    def validate(
        self,
        artifact_id: str,
    ) -> dict[str, Any]:
        """
        Validate that an artifact has both model
        and metadata files.

        Returns a validation report.
        """

        self._validate_artifact_id(
            artifact_id
        )

        model_path = self.model_path(
            artifact_id
        )

        metadata_path = self.metadata_path(
            artifact_id
        )

        model_exists = model_path.exists()
        metadata_exists = metadata_path.exists()

        metadata_valid = False
        metadata = None

        if metadata_exists:
            try:
                metadata = self.get_metadata(
                    artifact_id
                )

                metadata_valid = (
                    isinstance(
                        metadata,
                        dict,
                    )
                    and metadata.get(
                        "artifact_id"
                    )
                    == artifact_id
                )

            except (
                OSError,
                json.JSONDecodeError,
            ):
                metadata_valid = False

        valid = (
            model_exists
            and metadata_exists
            and metadata_valid
        )

        return {
            "artifact_id": artifact_id,
            "valid": valid,
            "model_exists": model_exists,
            "metadata_exists": metadata_exists,
            "metadata_valid": metadata_valid,
        }

    @staticmethod
    def _write_json(
        path: Path,
        data: dict[str, Any],
    ) -> None:
        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

    @staticmethod
    def _validate_artifact_id(
        artifact_id: str,
    ) -> None:
        if not isinstance(
            artifact_id,
            str,
        ):
            raise TypeError(
                "artifact_id must be a string."
            )

        if not artifact_id:
            raise ValueError(
                "artifact_id cannot be empty."
            )

        if Path(artifact_id).name != artifact_id:
            raise ValueError(
                "Invalid artifact_id."
            )

        if not artifact_id.startswith(
            "artifact_"
        ):
            raise ValueError(
                "Invalid artifact_id."
            )

    @staticmethod
    def _safe_value(
        value: Any,
    ) -> Any:
        if value is None:
            return None

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(
            value,
            Path,
        ):
            return str(value)

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): ArtifactManager._safe_value(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                ArtifactManager._safe_value(
                    item
                )
                for item in value
            ]

        try:
            json.dumps(value)
            return value
        except (
            TypeError,
            ValueError,
        ):
            return str(value)