from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


class EvaluationEngine:
    """
    Centralized evaluation engine for ModelForge.

    Provides standardized metrics for regression
    and classification tasks.
    """

    REGRESSION_METRICS = (
        "r2",
        "adjusted_r2",
        "mae",
        "mse",
        "rmse",
        "mape",
    )

    CLASSIFICATION_METRICS = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "log_loss",
    )

    METRIC_DIRECTIONS = {
        "r2": "maximize",
        "adjusted_r2": "maximize",
        "mae": "minimize",
        "mse": "minimize",
        "rmse": "minimize",
        "mape": "minimize",
        "accuracy": "maximize",
        "precision": "maximize",
        "recall": "maximize",
        "f1": "maximize",
        "roc_auc": "maximize",
        "log_loss": "minimize",
    }

    def evaluate_regression(
        self,
        y_true,
        predictions,
        feature_count: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate regression predictions.
        """

        self._validate_regression_inputs(
            y_true,
            predictions,
        )

        mse = mean_squared_error(
            y_true,
            predictions,
        )

        rmse = float(
            np.sqrt(mse)
        )

        r2 = float(
            r2_score(
                y_true,
                predictions,
            )
        )

        metrics = {
            "r2": r2,
            "mae": float(
                mean_absolute_error(
                    y_true,
                    predictions,
                )
            ),
            "mse": float(mse),
            "rmse": rmse,
            "mape": float(
                mean_absolute_percentage_error(
                    y_true,
                    predictions,
                )
            ),
        }

        if feature_count is not None:
            metrics["adjusted_r2"] = (
                self.adjusted_r2(
                    r2=r2,
                    sample_count=len(y_true),
                    feature_count=feature_count,
                )
            )
        else:
            metrics["adjusted_r2"] = np.nan

        return metrics

    def evaluate_classification(
        self,
        y_true,
        predictions,
        probabilities=None,
        decision_scores=None,
    ) -> dict[str, float | None]:
        """
        Evaluate classification predictions.
        """

        self._validate_classification_inputs(
            y_true,
            predictions,
        )

        metrics = {
            "accuracy": float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),
            "precision": float(
                precision_score(
                    y_true,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "roc_auc": self._calculate_roc_auc(
                y_true=y_true,
                probabilities=probabilities,
                decision_scores=decision_scores,
            ),
            "log_loss": self._calculate_log_loss(
                y_true=y_true,
                probabilities=probabilities,
            ),
        }

        return metrics

    @staticmethod
    def adjusted_r2(
        r2: float,
        sample_count: int,
        feature_count: int,
    ) -> float:
        """
        Calculate adjusted R².
        """

        if sample_count <= feature_count + 1:
            return float("nan")

        return float(
            1
            - (
                (1 - r2)
                * (
                    (sample_count - 1)
                    / (
                        sample_count
                        - feature_count
                        - 1
                    )
                )
            )
        )

    @classmethod
    def metric_direction(
        cls,
        metric: str,
    ) -> str:
        """
        Return whether a metric should be maximized
        or minimized.
        """

        if metric not in cls.METRIC_DIRECTIONS:
            raise KeyError(
                f"Unknown metric: {metric}"
            )

        return cls.METRIC_DIRECTIONS[
            metric
        ]

    @classmethod
    def available_metrics(
        cls,
        task_type: str,
    ) -> list[str]:
        """
        Return available metrics for a task.
        """

        if task_type == "regression":
            return list(
                cls.REGRESSION_METRICS
            )

        if task_type == "classification":
            return list(
                cls.CLASSIFICATION_METRICS
            )

        raise ValueError(
            "task_type must be 'regression' "
            "or 'classification'."
        )

    @staticmethod
    def _calculate_roc_auc(
        y_true,
        probabilities=None,
        decision_scores=None,
    ) -> float | None:
        """Calculate ROC-AUC when possible."""

        try:
            unique_classes = np.unique(
                y_true
            )

            if probabilities is not None:
                probabilities = np.asarray(
                    probabilities
                )

                if len(unique_classes) == 2:
                    if (
                        probabilities.ndim == 2
                        and probabilities.shape[1] >= 2
                    ):
                        return float(
                            roc_auc_score(
                                y_true,
                                probabilities[:, 1],
                            )
                        )

                    if probabilities.ndim == 1:
                        return float(
                            roc_auc_score(
                                y_true,
                                probabilities,
                            )
                        )

                if (
                    len(unique_classes) > 2
                    and probabilities.ndim == 2
                ):
                    return float(
                        roc_auc_score(
                            y_true,
                            probabilities,
                            multi_class="ovr",
                            average="weighted",
                        )
                    )

            if decision_scores is not None:
                decision_scores = np.asarray(
                    decision_scores
                )

                if len(unique_classes) == 2:
                    return float(
                        roc_auc_score(
                            y_true,
                            decision_scores,
                        )
                    )

                if (
                    len(unique_classes) > 2
                    and decision_scores.ndim == 2
                ):
                    return float(
                        roc_auc_score(
                            y_true,
                            decision_scores,
                            multi_class="ovr",
                            average="weighted",
                        )
                    )

        except (
            ValueError,
            TypeError,
        ):
            return None

        return None

    @staticmethod
    def _calculate_log_loss(
        y_true,
        probabilities=None,
    ) -> float | None:
        """Calculate log loss when probabilities are available."""

        if probabilities is None:
            return None

        try:
            return float(
                log_loss(
                    y_true,
                    probabilities,
                )
            )

        except (
            ValueError,
            TypeError,
        ):
            return None

    @staticmethod
    def _validate_regression_inputs(
        y_true,
        predictions,
    ) -> None:
        """Validate regression inputs."""

        if len(y_true) != len(predictions):
            raise ValueError(
                "y_true and predictions must "
                "have the same length."
            )

        if len(y_true) == 0:
            raise ValueError(
                "Cannot evaluate empty predictions."
            )

    @staticmethod
    def _validate_classification_inputs(
        y_true,
        predictions,
    ) -> None:
        """Validate classification inputs."""

        if len(y_true) != len(predictions):
            raise ValueError(
                "y_true and predictions must "
                "have the same length."
            )

        if len(y_true) == 0:
            raise ValueError(
                "Cannot evaluate empty predictions."
            )