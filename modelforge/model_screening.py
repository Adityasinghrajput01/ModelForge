from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class ModelScreeningEngine:
    """
    Train and evaluate candidate ModelForge pipelines.

    This component performs fast model screening only.
    It does not rank or select the final best model.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.test_size = test_size
        self.random_state = random_state

        self._validate_configuration()

    def screen(
        self,
        data: pd.DataFrame,
        target: str,
        pipelines: dict[str, Pipeline],
        task_type: str,
    ) -> pd.DataFrame:
        """
        Train and evaluate multiple candidate pipelines.

        Parameters
        ----------
        data:
            Dataset containing features and target.

        target:
            Target column.

        pipelines:
            Candidate pipelines keyed by model name.

        task_type:
            'regression' or 'classification'.

        Returns
        -------
        pd.DataFrame
            One result row per candidate model.
        """

        self._validate_inputs(
            data=data,
            target=target,
            pipelines=pipelines,
            task_type=task_type,
        )

        X = data.drop(
            columns=[target]
        )

        y = data[target]

        stratify = (
            y
            if (
                task_type == "classification"
                and self._can_stratify(y)
            )
            else None
        )

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify,
            )
        )

        results: list[dict[str, Any]] = []

        for model_name, pipeline in pipelines.items():
            result = self._evaluate_pipeline(
                model_name=model_name,
                pipeline=pipeline,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                task_type=task_type,
            )

            results.append(result)

        return pd.DataFrame(results)

    def _evaluate_pipeline(
        self,
        model_name: str,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        task_type: str,
    ) -> dict[str, Any]:
        """Train and evaluate one pipeline safely."""

        start_time = time.perf_counter()

        try:
            fit_start = time.perf_counter()

            pipeline.fit(
                X_train,
                y_train,
            )

            training_time = (
                time.perf_counter()
                - fit_start
            )

            prediction_start = time.perf_counter()

            predictions = pipeline.predict(
                X_test
            )

            prediction_time = (
                time.perf_counter()
                - prediction_start
            )

            if task_type == "regression":
                metrics = self._regression_metrics(
                    y_test,
                    predictions,
                )
            else:
                metrics = self._classification_metrics(
                    pipeline,
                    X_test,
                    y_test,
                    predictions,
                )

            total_time = (
                time.perf_counter()
                - start_time
            )

            return {
                "model": model_name,
                "status": "success",
                "training_time_seconds": float(
                    training_time
                ),
                "prediction_time_seconds": float(
                    prediction_time
                ),
                "total_time_seconds": float(
                    total_time
                ),
                **metrics,
                "error": None,
            }

        except Exception as exc:
            total_time = (
                time.perf_counter()
                - start_time
            )

            return {
                "model": model_name,
                "status": "failed",
                "training_time_seconds": float(
                    total_time
                ),
                "prediction_time_seconds": None,
                "total_time_seconds": float(
                    total_time
                ),
                **self._empty_metrics(
                    task_type
                ),
                "error": str(exc),
            }

    @staticmethod
    def _regression_metrics(
        y_true,
        predictions,
    ) -> dict[str, float]:
        """Calculate regression metrics."""

        mse = mean_squared_error(
            y_true,
            predictions,
        )

        rmse = float(
            np.sqrt(mse)
        )

        return {
            "r2": float(
                r2_score(
                    y_true,
                    predictions,
                )
            ),
            "mae": float(
                mean_absolute_error(
                    y_true,
                    predictions,
                )
            ),
            "mse": float(mse),
            "rmse": rmse,
        }

    @staticmethod
    def _classification_metrics(
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        predictions,
    ) -> dict[str, float | None]:
        """Calculate classification metrics."""

        metrics: dict[str, float | None] = {
            "accuracy": float(
                accuracy_score(
                    y_test,
                    predictions,
                )
            ),
            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "roc_auc": None,
        }

        metrics["roc_auc"] = (
            ModelScreeningEngine._calculate_roc_auc(
                pipeline,
                X_test,
                y_test,
            )
        )

        return metrics

    @staticmethod
    def _calculate_roc_auc(
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> float | None:
        """Calculate ROC-AUC using probabilities or decision scores."""

        if hasattr(
            pipeline,
            "predict_proba",
        ):
            try:
                probabilities = pipeline.predict_proba(
                    X_test
                )

                probabilities = np.asarray(
                    probabilities
                )

                if probabilities.ndim != 2:
                    return None

                if probabilities.shape[1] == 2:
                    return float(
                        roc_auc_score(
                            y_test,
                            probabilities[:, 1],
                        )
                    )

                if probabilities.shape[1] > 2:
                    return float(
                        roc_auc_score(
                            y_test,
                            probabilities,
                            multi_class="ovr",
                            average="weighted",
                        )
                    )

            except (
                ValueError,
                TypeError,
                AttributeError,
            ):
                return None

        if hasattr(
            pipeline,
            "decision_function",
        ):
            try:
                decision_scores = (
                    pipeline.decision_function(
                        X_test
                    )
                )

                unique_classes = np.unique(
                    y_test
                )

                if len(unique_classes) == 2:
                    return float(
                        roc_auc_score(
                            y_test,
                            decision_scores,
                        )
                    )

                if (
                    np.asarray(
                        decision_scores
                    ).ndim == 2
                ):
                    return float(
                        roc_auc_score(
                            y_test,
                            decision_scores,
                            multi_class="ovr",
                            average="weighted",
                        )
                    )

            except (
                ValueError,
                TypeError,
                AttributeError,
            ):
                return None

        return None

    @staticmethod
    def _empty_metrics(
        task_type: str,
    ) -> dict[str, None]:
        """Return empty metrics for failed models."""

        if task_type == "regression":
            return {
                "r2": None,
                "mae": None,
                "mse": None,
                "rmse": None,
            }

        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
        }

    @staticmethod
    def _can_stratify(
        target: pd.Series,
    ) -> bool:
        """
        Determine whether classification data can safely use
        stratified splitting.
        """

        if target.empty:
            return False

        class_counts = target.value_counts()

        return bool(
            len(class_counts) >= 2
            and class_counts.min() >= 2
        )

    def _validate_configuration(self) -> None:
        """Validate engine configuration."""

        if not 0 < self.test_size < 1:
            raise ValueError(
                "test_size must be between 0 and 1."
            )

        if not isinstance(
            self.random_state,
            int,
        ):
            raise TypeError(
                "random_state must be an integer."
            )

    @staticmethod
    def _validate_inputs(
        data: pd.DataFrame,
        target: str,
        pipelines: dict[str, Pipeline],
        task_type: str,
    ) -> None:
        """Validate screening inputs."""

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Cannot screen models on an empty dataset."
            )

        if not isinstance(
            target,
            str,
        ):
            raise TypeError(
                "target must be a string."
            )

        if target not in data.columns:
            raise ValueError(
                f"Target column '{target}' "
                "does not exist."
            )

        if not isinstance(
            pipelines,
            dict,
        ):
            raise TypeError(
                "pipelines must be a dictionary."
            )

        if not pipelines:
            raise ValueError(
                "At least one pipeline is required."
            )

        if task_type not in {
            "regression",
            "classification",
        }:
            raise ValueError(
                "task_type must be 'regression' "
                "or 'classification'."
            )

        for name, pipeline in pipelines.items():
            if not isinstance(
                name,
                str,
            ):
                raise TypeError(
                    "Pipeline names must be strings."
                )

            if not name.strip():
                raise ValueError(
                    "Pipeline names cannot be empty."
                )

            if not isinstance(
                pipeline,
                Pipeline,
            ):
                raise TypeError(
                    f"Pipeline '{name}' must be "
                    "a sklearn Pipeline."
                )