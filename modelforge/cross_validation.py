from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone
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
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline


class CrossValidationEngine:
    """
    Evaluate ModelForge pipelines using cross-validation.

    Every fold receives a fresh cloned pipeline so that preprocessing
    and model fitting happen independently inside each fold.
    """

    def __init__(
        self,
        cv: int = 5,
        random_state: int = 42,
        shuffle: bool = True,
    ):
        self.cv = cv
        self.random_state = random_state
        self.shuffle = shuffle

        self._validate_configuration()

    def evaluate(
        self,
        data: pd.DataFrame,
        target: str,
        pipelines: dict[str, Pipeline],
        task_type: str,
    ) -> pd.DataFrame:
        """Evaluate pipelines using K-fold cross-validation."""

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

        splitter = self._create_splitter(
            y=y,
            task_type=task_type,
        )

        results: list[dict[str, Any]] = []

        for model_name, pipeline in pipelines.items():
            result = self._evaluate_pipeline(
                model_name=model_name,
                pipeline=pipeline,
                X=X,
                y=y,
                splitter=splitter,
                task_type=task_type,
            )

            results.append(result)

        return pd.DataFrame(results)

    def _evaluate_pipeline(
        self,
        model_name: str,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        splitter,
        task_type: str,
    ) -> dict[str, Any]:
        """Evaluate one pipeline across all folds."""

        fold_results: list[dict[str, Any]] = []

        total_start = time.perf_counter()

        for fold_number, (
            train_indices,
            validation_indices,
        ) in enumerate(
            splitter.split(
                X,
                y if task_type == "classification" else None,
            ),
            start=1,
        ):
            fold_start = time.perf_counter()

            try:
                X_train = X.iloc[
                    train_indices
                ]

                X_validation = X.iloc[
                    validation_indices
                ]

                y_train = y.iloc[
                    train_indices
                ]

                y_validation = y.iloc[
                    validation_indices
                ]

                fold_pipeline = clone(
                    pipeline
                )

                training_start = time.perf_counter()

                fold_pipeline.fit(
                    X_train,
                    y_train,
                )

                training_time = (
                    time.perf_counter()
                    - training_start
                )

                prediction_start = time.perf_counter()

                predictions = (
                    fold_pipeline.predict(
                        X_validation
                    )
                )

                prediction_time = (
                    time.perf_counter()
                    - prediction_start
                )

                if task_type == "regression":
                    metrics = self._regression_metrics(
                        y_validation,
                        predictions,
                    )
                else:
                    metrics = self._classification_metrics(
                        fold_pipeline,
                        X_validation,
                        y_validation,
                        predictions,
                    )

                fold_time = (
                    time.perf_counter()
                    - fold_start
                )

                fold_results.append(
                    {
                        "fold": fold_number,
                        "status": "success",
                        "training_time_seconds": float(
                            training_time
                        ),
                        "prediction_time_seconds": float(
                            prediction_time
                        ),
                        "total_time_seconds": float(
                            fold_time
                        ),
                        **metrics,
                        "error": None,
                    }
                )

            except Exception as exc:
                fold_time = (
                    time.perf_counter()
                    - fold_start
                )

                fold_results.append(
                    {
                        "fold": fold_number,
                        "status": "failed",
                        "training_time_seconds": float(
                            fold_time
                        ),
                        "prediction_time_seconds": None,
                        "total_time_seconds": float(
                            fold_time
                        ),
                        **self._empty_metrics(
                            task_type
                        ),
                        "error": str(exc),
                    }
                )

        total_time = (
            time.perf_counter()
            - total_start
        )

        return self._aggregate_results(
            model_name=model_name,
            fold_results=fold_results,
            task_type=task_type,
            total_time=total_time,
        )

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
            "rmse": float(
                np.sqrt(mse)
            ),
        }

    @staticmethod
    def _classification_metrics(
        pipeline: Pipeline,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        predictions,
    ) -> dict[str, float | None]:
        """Calculate classification metrics."""

        metrics: dict[str, float | None] = {
            "accuracy": float(
                accuracy_score(
                    y_validation,
                    predictions,
                )
            ),
            "precision": float(
                precision_score(
                    y_validation,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_validation,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_validation,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "roc_auc": None,
        }

        metrics["roc_auc"] = (
            CrossValidationEngine._calculate_roc_auc(
                pipeline,
                X_validation,
                y_validation,
            )
        )

        return metrics

    @staticmethod
    def _calculate_roc_auc(
        pipeline: Pipeline,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> float | None:
        """Calculate ROC-AUC using probabilities or decision scores."""

        if hasattr(
            pipeline,
            "predict_proba",
        ):
            try:
                probabilities = pipeline.predict_proba(
                    X_validation
                )

                probabilities = np.asarray(
                    probabilities
                )

                if probabilities.ndim != 2:
                    return None

                if probabilities.shape[1] == 2:
                    return float(
                        roc_auc_score(
                            y_validation,
                            probabilities[:, 1],
                        )
                    )

                if probabilities.shape[1] > 2:
                    return float(
                        roc_auc_score(
                            y_validation,
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
                        X_validation
                    )
                )

                unique_classes = np.unique(
                    y_validation
                )

                if len(unique_classes) == 2:
                    return float(
                        roc_auc_score(
                            y_validation,
                            decision_scores,
                        )
                    )

                decision_scores = np.asarray(
                    decision_scores
                )

                if decision_scores.ndim == 2:
                    return float(
                        roc_auc_score(
                            y_validation,
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
        """Return empty metrics for failed folds."""

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

    def _aggregate_results(
        self,
        model_name: str,
        fold_results: list[dict],
        task_type: str,
        total_time: float,
    ) -> dict[str, Any]:
        """Aggregate metrics across successful folds."""

        successful_folds = [
            result
            for result in fold_results
            if result["status"] == "success"
        ]

        failed_folds = [
            result
            for result in fold_results
            if result["status"] == "failed"
        ]

        result: dict[str, Any] = {
            "model": model_name,
            "status": (
                "success"
                if successful_folds
                else "failed"
            ),
            "cv_folds": len(
                fold_results
            ),
            "successful_folds": len(
                successful_folds
            ),
            "failed_folds": len(
                failed_folds
            ),
            "total_time_seconds": float(
                total_time
            ),
        }

        metric_names = self._metric_names(
            task_type
        )

        for metric_name in metric_names:
            values = [
                fold[metric_name]
                for fold in successful_folds
                if fold[metric_name] is not None
            ]

            if values:
                result[
                    f"cv_mean_{metric_name}"
                ] = float(
                    np.mean(values)
                )

                result[
                    f"cv_std_{metric_name}"
                ] = float(
                    np.std(
                        values,
                        ddof=1,
                    )
                    if len(values) > 1
                    else 0.0
                )
            else:
                result[
                    f"cv_mean_{metric_name}"
                ] = None

                result[
                    f"cv_std_{metric_name}"
                ] = None

        successful_training_times = [
            fold["training_time_seconds"]
            for fold in successful_folds
        ]

        successful_prediction_times = [
            fold["prediction_time_seconds"]
            for fold in successful_folds
            if fold["prediction_time_seconds"] is not None
        ]

        result["cv_mean_training_time_seconds"] = (
            float(
                np.mean(
                    successful_training_times
                )
            )
            if successful_training_times
            else None
        )

        result["cv_mean_prediction_time_seconds"] = (
            float(
                np.mean(
                    successful_prediction_times
                )
            )
            if successful_prediction_times
            else None
        )

        result["errors"] = [
            fold["error"]
            for fold in failed_folds
            if fold["error"] is not None
        ]

        return result

    @staticmethod
    def _metric_names(
        task_type: str,
    ) -> list[str]:
        """Return metrics for a task."""

        if task_type == "regression":
            return [
                "r2",
                "mae",
                "mse",
                "rmse",
            ]

        return [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]

    def _create_splitter(
        self,
        y: pd.Series,
        task_type: str,
    ):
        """Create the appropriate CV splitter."""

        if len(y) < self.cv:
            raise ValueError(
                f"Dataset must contain at least {self.cv} "
                "samples for cross-validation."
            )

        if task_type == "classification":
            class_counts = y.value_counts()

            if (
                len(class_counts) >= 2
                and class_counts.min() >= self.cv
            ):
                return StratifiedKFold(
                    n_splits=self.cv,
                    shuffle=self.shuffle,
                    random_state=(
                        self.random_state
                        if self.shuffle
                        else None
                    ),
                )

            raise ValueError(
                "Each classification class must "
                f"contain at least {self.cv} samples "
                "for stratified cross-validation."
            )

        return KFold(
            n_splits=self.cv,
            shuffle=self.shuffle,
            random_state=(
                self.random_state
                if self.shuffle
                else None
            ),
        )

    def _validate_configuration(self):
        """Validate engine configuration."""

        if not isinstance(
            self.cv,
            int,
        ):
            raise TypeError(
                "cv must be an integer."
            )

        if isinstance(
            self.cv,
            bool,
        ):
            raise TypeError(
                "cv must be an integer."
            )

        if self.cv < 2:
            raise ValueError(
                "cv must be at least 2."
            )

        if not isinstance(
            self.random_state,
            int,
        ):
            raise TypeError(
                "random_state must be an integer."
            )

        if isinstance(
            self.random_state,
            bool,
        ):
            raise TypeError(
                "random_state must be an integer."
            )

        if not isinstance(
            self.shuffle,
            bool,
        ):
            raise TypeError(
                "shuffle must be a boolean."
            )

    @staticmethod
    def _validate_inputs(
        data: pd.DataFrame,
        target: str,
        pipelines: dict[str, Pipeline],
        task_type: str,
    ):
        """Validate evaluation inputs."""

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Cannot evaluate an empty dataset."
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