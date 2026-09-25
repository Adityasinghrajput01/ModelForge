from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any


class RunManager:
    """
    Manage the lifecycle and metadata of a ModelForge run.

    A run represents one execution of an AutoML workflow.
    """

    def __init__(self) -> None:
        self.run_id: str | None = None
        self.status: str = "not_started"
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.duration_seconds: float | None = None
        self.metadata: dict[str, Any] = {}
        self._start_time: float | None = None

    def start(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Start a new run.

        Returns the generated run ID.
        """

        if self.status == "running":
            raise RuntimeError(
                "A run is already in progress."
            )

        self.run_id = self._create_run_id()
        self.status = "running"
        self.started_at = (
            datetime.now(timezone.utc).isoformat()
        )
        self.finished_at = None
        self.duration_seconds = None
        self._start_time = time.perf_counter()

        self.metadata = dict(
            metadata or {}
        )

        return self.run_id

    def complete(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Mark the current run as completed.
        """

        self._require_running()

        self._finish(
            status="completed",
            metadata=metadata,
        )

        return self.summary()

    def fail(
        self,
        error: Exception | str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Mark the current run as failed.
        """

        self._require_running()

        error_message = str(error)

        failure_metadata = dict(
            metadata or {}
        )

        failure_metadata["error"] = (
            error_message
        )

        self._finish(
            status="failed",
            metadata=failure_metadata,
        )

        return self.summary()

    def update(
        self,
        metadata: dict[str, Any],
    ) -> None:
        """
        Add or update run metadata.
        """

        if not isinstance(
            metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dictionary."
            )

        if self.status != "running":
            raise RuntimeError(
                "Run metadata can only be "
                "updated while a run is running."
            )

        self.metadata.update(
            metadata
        )

    def summary(self) -> dict[str, Any]:
        """
        Return the current run summary.
        """

        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": (
                self.duration_seconds
            ),
            "metadata": dict(
                self.metadata
            ),
        }

    def is_running(self) -> bool:
        """
        Return whether a run is currently active.
        """

        return self.status == "running"

    def reset(self) -> None:
        """
        Reset the manager to its initial state.
        """

        self.run_id = None
        self.status = "not_started"
        self.started_at = None
        self.finished_at = None
        self.duration_seconds = None
        self.metadata = {}
        self._start_time = None

    def _finish(
        self,
        status: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        self.status = status

        self.finished_at = (
            datetime.now(timezone.utc).isoformat()
        )

        if self._start_time is not None:
            self.duration_seconds = (
                time.perf_counter()
                - self._start_time
            )

        if metadata:
            self.metadata.update(
                metadata
            )

    def _require_running(self) -> None:
        if self.status != "running":
            raise RuntimeError(
                "No active run is currently running."
            )

    @staticmethod
    def _create_run_id() -> str:
        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d%H%M%S")

        short_uuid = uuid.uuid4().hex[:8]

        return f"run_{timestamp}_{short_uuid}"