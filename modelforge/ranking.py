from typing import Any

import numpy as np
import pandas as pd


class RankingEngine:
    """
    Rank ModelForge models using multiple objectives.

    The engine does not train models.
    It only consumes model evaluation results.
    """

    DEFAULT_WEIGHTS = {
        "balanced": {
            "primary": 0.45,
            "error": 0.25,
            "stability": 0.20,
            "speed": 0.10,
        },
        "performance": {
            "primary": 0.70,
            "error": 0.15,
            "stability": 0.10,
            "speed": 0.05,
        },
        "error": {
            "primary": 0.20,
            "error": 0.60,
            "stability": 0.15,
            "speed": 0.05,
        },
        "speed": {
            "primary": 0.20,
            "error": 0.15,
            "stability": 0.15,
            "speed": 0.50,
        },
    }

    def rank(
        self,
        results: pd.DataFrame,
        task_type: str,
        objective: str = "balanced",
    ) -> pd.DataFrame:
        """
        Rank models using multiple objectives.
        """

        self._validate_inputs(
            results=results,
            task_type=task_type,
            objective=objective,
        )

        ranked = results.copy()

        weights = self.DEFAULT_WEIGHTS[
            objective
        ]

        primary_metric = (
            self._primary_metric(
                ranked,
                task_type,
            )
        )

        error_metric = (
            self._error_metric(
                ranked,
                task_type,
            )
        )

        stability_metric = (
            self._stability_metric(
                ranked,
                task_type,
            )
        )

        speed_metric = self._speed_metric(ranked)

        ranked["primary_score"] = (
            self._normalize_metric(
                ranked[primary_metric],
                maximize=True,
            )
        )

        ranked["error_score"] = (
            self._normalize_metric(
                ranked[error_metric],
                maximize=False,
            )
        )

        ranked["stability_score"] = (
            self._normalize_metric(
                ranked[stability_metric],
                maximize=False,
            )
        )

        if speed_metric is None:
            ranked["speed_score"] = 1.0
        else:
            ranked["speed_score"] = self._normalize_metric(
                ranked[speed_metric],
                maximize=False,
            )

        ranked["overall_score"] = (
            ranked["primary_score"]
            * weights["primary"]
            + ranked["error_score"]
            * weights["error"]
            + ranked["stability_score"]
            * weights["stability"]
            + ranked["speed_score"]
            * weights["speed"]
        )

        ranked["rank"] = (
            ranked["overall_score"]
            .rank(
                ascending=False,
                method="min",
            )
            .astype(int)
        )

        return (
            ranked.sort_values(
                by=[
                    "rank",
                    "overall_score",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

    @staticmethod
    def _primary_metric(
        results: pd.DataFrame,
        task_type: str,
    ) -> str:
        """Return the primary predictive metric."""

        if task_type == "regression":
            if "cv_mean_r2" in results.columns:
                return "cv_mean_r2"

            return "r2"

        if "cv_mean_f1" in results.columns:
            return "cv_mean_f1"

        return "f1"

    @staticmethod
    def _error_metric(
        results: pd.DataFrame,
        task_type: str,
    ) -> str:
        """Return the primary error metric."""

        if task_type == "regression":
            if "cv_mean_rmse" in results.columns:
                return "cv_mean_rmse"

            if "mae" in results.columns:
                return "mae"

            return "rmse"

        if "cv_mean_log_loss" in results.columns:
            return "cv_mean_log_loss"

        if "log_loss" in results.columns:
            return "log_loss"

        return "f1"

    @staticmethod
    def _stability_metric(
        results: pd.DataFrame,
        task_type: str,
    ) -> str:
        """Return the CV stability metric."""

        if task_type == "regression":
            if "cv_std_r2" in results.columns:
                return "cv_std_r2"

            return "r2"

        if "cv_std_f1" in results.columns:
            return "cv_std_f1"

        return "f1"

    @staticmethod
    def _speed_metric(
        results: pd.DataFrame,
    ) -> str | None:
        """Return the available training-time metric."""

        if (
            "total_time_seconds"
            in results.columns
        ):
            return "total_time_seconds"

        if (
            "training_time_seconds"
            in results.columns
        ):
            return "training_time_seconds"

        return None

    @staticmethod
    def _normalize_metric(
        values: pd.Series,
        maximize: bool,
    ) -> pd.Series:
        """
        Normalize values into [0, 1].

        Higher values are better when maximize=True.
        Lower values are better when maximize=False.
        """

        numeric = pd.to_numeric(
            values,
            errors="coerce",
        )

        if numeric.isna().all():
            return pd.Series(
                0.0,
                index=values.index,
            )

        minimum = numeric.min()
        maximum = numeric.max()

        if np.isclose(
            minimum,
            maximum,
        ):
            return pd.Series(
                1.0,
                index=values.index,
            )

        if maximize:
            normalized = (
                numeric - minimum
            ) / (
                maximum - minimum
            )
        else:
            normalized = (
                maximum - numeric
            ) / (
                maximum - minimum
            )

        return normalized.fillna(0.0)

    @classmethod
    def available_objectives(
        cls,
    ) -> list[str]:
        """Return supported ranking objectives."""

        return list(
            cls.DEFAULT_WEIGHTS.keys()
        )

    @classmethod
    def objective_weights(
        cls,
        objective: str,
    ) -> dict[str, float]:
        """Return weights for an objective."""

        if objective not in cls.DEFAULT_WEIGHTS:
            raise KeyError(
                f"Unknown objective: {objective}"
            )

        return cls.DEFAULT_WEIGHTS[
            objective
        ].copy()

    @staticmethod
    def _validate_inputs(
        results: pd.DataFrame,
        task_type: str,
        objective: str,
    ) -> None:
        """Validate ranking inputs."""

        if not isinstance(
            results,
            pd.DataFrame,
        ):
            raise TypeError(
                "results must be a pandas DataFrame."
            )

        if results.empty:
            raise ValueError(
                "Cannot rank an empty results dataset."
            )

        if task_type not in {
            "regression",
            "classification",
        }:
            raise ValueError(
                "task_type must be 'regression' "
                "or 'classification'."
            )

        if objective not in {
            "balanced",
            "performance",
            "error",
            "speed",
        }:
            raise ValueError(
                "objective must be one of: "
                "balanced, performance, "
                "error, speed."
            )

        if "model" not in results.columns:
            raise ValueError(
                "Results must contain a 'model' column."
            )