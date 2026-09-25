from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class ExperimentTracker:
    """
    Local experiment tracking for ModelForge.

    Experiments are stored as individual JSON files so that
    users can inspect, copy, archive, or version them easily.
    """

    def __init__(
        self,
        directory: str | Path = ".modelforge/experiments",
        experiment_directory: str | Path | None = None,
    ):
        self.directory = Path(
            experiment_directory
            if experiment_directory is not None
            else directory
        )
        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def create_experiment_id(self) -> str:
        """
        Generate a unique experiment identifier.
        """

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d%H%M%S")

        short_uuid = uuid.uuid4().hex[:8]

        return f"exp_{timestamp}_{short_uuid}"

    def record(
        self,
        result: dict[str, Any],
        configuration: dict[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> str:
        """
        Save an AutoML result as an experiment.

        Returns the experiment ID.

        The reproducibility snapshot is copied exactly from
        the AutoML result before serialization.
        """

        if not isinstance(result, dict):
            raise TypeError(
                "result must be a dictionary."
            )

        if configuration is not None:
            if not isinstance(configuration, dict):
                raise TypeError(
                    "configuration must be a dictionary."
                )

        experiment_id = (
            experiment_id
            or self.create_experiment_id()
        )

        reproducibility = copy.deepcopy(
            result.get("reproducibility")
        )

        run_summary = result.get(
            "run_summary"
        )

        if not isinstance(run_summary, dict):
            run_summary = {}

        run_id = result.get(
            "run_id"
        ) or run_summary.get(
            "run_id"
        )

        status = result.get(
            "status"
        ) or run_summary.get(
            "status"
        )

        experiment = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "target": self._safe_value(
                result.get("target")
            ),
            "profile": self._safe_value(
                result.get("profile")
            ),
            "column_intelligence": self._safe_value(
                result.get("column_intelligence")
            ),
            "audit": self._safe_value(
                result.get("audit")
            ),
            "models_evaluated": self._safe_value(
                result.get("models_evaluated")
            ),
            "screening_results": self._dataframe_to_records(
                result.get("screening_results")
            ),
            "cv_results": self._dataframe_to_records(
                result.get("cv_results")
            ),
            "initial_ranking": self._dataframe_to_records(
                result.get("initial_ranking")
            ),
            "optimization_enabled": self._safe_value(
                result.get("optimization_enabled")
            ),
            "optimization_results": self._safe_value(
                result.get("optimization_results")
            ),
            "ranking": self._dataframe_to_records(
                result.get("ranking")
            ),
            "best_model": self._safe_value(
                result.get("best_model")
            ),
            "feature_selection": self._safe_value(
                result.get("feature_selection")
            ),
            "configuration": self._safe_value(
                configuration
            ),
            "reproducibility": self._safe_value(
                reproducibility
            ),
            "run_id": self._safe_value(
                run_id
            ),
            "run_summary": self._safe_value(
                run_summary
            ),
            "status": self._safe_value(
                status
            ),
            "error": self._safe_value(
                result.get("error")
            ),
        }

        path = self._experiment_path(
            experiment_id
        )

        self._write_json(
            path,
            experiment,
        )

        return experiment_id

    def get(
        self,
        experiment_id: str,
    ) -> dict[str, Any]:
        """
        Load one experiment by ID.
        """

        self._validate_experiment_id(
            experiment_id
        )

        path = self._experiment_path(
            experiment_id
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Experiment '{experiment_id}' "
                "does not exist."
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def list_experiments(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return a compact list of tracked experiments.

        Experiments are returned newest first.
        """

        experiments = []

        for path in self.directory.glob(
            "exp_*.json"
        ):
            try:
                with path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    experiment = json.load(file)

                reproducibility = experiment.get(
                    "reproducibility"
                )

                experiments.append(
                    {
                        "experiment_id": experiment.get(
                            "experiment_id"
                        ),
                        "timestamp": experiment.get(
                            "timestamp"
                        ),
                        "target": experiment.get(
                            "target"
                        ),
                        "task_type": self._extract_task_type(
                            experiment
                        ),
                        "best_model": experiment.get(
                            "best_model"
                        ),
                        "models_evaluated": experiment.get(
                            "models_evaluated"
                        ),
                        "optimization_enabled": experiment.get(
                            "optimization_enabled"
                        ),
                        "run_id": experiment.get(
                            "run_id"
                        ),
                        "status": experiment.get(
                            "status"
                        ),
                        "reproducibility": reproducibility,
                        "reproducibility_available": (
                            reproducibility is not None
                        ),
                    }
                )

            except (
                OSError,
                json.JSONDecodeError,
            ):
                continue

        experiments.sort(
            key=lambda item: item.get(
                "timestamp",
                "",
            ),
            reverse=True,
        )

        return experiments

    def delete(
        self,
        experiment_id: str,
    ) -> None:
        """
        Delete an experiment by ID.
        """

        self._validate_experiment_id(
            experiment_id
        )

        path = self._experiment_path(
            experiment_id
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Experiment '{experiment_id}' "
                "does not exist."
            )

        path.unlink()

    def clear(self) -> int:
        """
        Delete all tracked experiments.

        Returns the number of deleted experiments.
        """

        deleted = 0

        for path in self.directory.glob(
            "exp_*.json"
        ):
            try:
                path.unlink()
                deleted += 1
            except OSError:
                continue

        return deleted

    def count(self) -> int:
        """
        Return the number of tracked experiments.
        """

        return len(
            list(
                self.directory.glob(
                    "exp_*.json"
                )
            )
        )

    def _experiment_path(
        self,
        experiment_id: str,
    ) -> Path:
        return (
            self.directory
            / f"{experiment_id}.json"
        )

    @staticmethod
    def _validate_experiment_id(
        experiment_id: str,
    ) -> None:
        if not isinstance(
            experiment_id,
            str,
        ):
            raise TypeError(
                "experiment_id must be a string."
            )

        if not experiment_id:
            raise ValueError(
                "experiment_id cannot be empty."
            )

        if Path(experiment_id).name != experiment_id:
            raise ValueError(
                "Invalid experiment_id."
            )

        if not experiment_id.startswith(
            "exp_"
        ):
            raise ValueError(
                "Invalid experiment_id."
            )

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
    def _dataframe_to_records(
        value: Any,
    ) -> Any:
        if isinstance(
            value,
            pd.DataFrame,
        ):
            return value.to_dict(
                orient="records"
            )

        return ExperimentTracker._safe_value(
            value
        )

    @staticmethod
    def _extract_task_type(
        experiment: dict[str, Any],
    ) -> str | None:
        target = experiment.get(
            "target"
        )

        if isinstance(
            target,
            dict,
        ):
            return target.get(
                "task_type"
            )

        return None

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
            pd.DataFrame,
        ):
            return value.to_dict(
                orient="records"
            )

        if isinstance(
            value,
            pd.Series,
        ):
            return value.to_list()

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): ExperimentTracker._safe_value(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                ExperimentTracker._safe_value(
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