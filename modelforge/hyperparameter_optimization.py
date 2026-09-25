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
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    ParameterGrid,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline


class HyperparameterOptimizationEngine:
    """
    Search model hyperparameters using cross-validation.

    The engine operates on complete sklearn pipelines so that
    preprocessing remains part of every optimization trial.
    """

    def __init__(
        self,
        cv: int = 5,
        random_state: int = 42,
        scoring: str | None = None,
        max_trials: int | None = None,
    ):
        self.cv = cv
        self.random_state = random_state
        self.scoring = scoring
        self.max_trials = max_trials

        self._validate_configuration()

    def optimize(
        self,
        data: pd.DataFrame,
        target: str,
        pipeline: Pipeline,
        parameter_space: dict[str, Any],
        task_type: str,
    ) -> dict[str, Any]:
        """
        Optimize a pipeline's hyperparameters.

        Returns a dictionary containing:
        - best_pipeline
        - best_params
        - best_score
        - trials
        - successful_trials
        - failed_trials
        - total_time_seconds
        """

        self._validate_inputs(
            data=data,
            target=target,
            pipeline=pipeline,
            parameter_space=parameter_space,
            task_type=task_type,
        )

        X = data.drop(
            columns=[target]
        )

        y = data[target]

        parameter_combinations = list(
            ParameterGrid(parameter_space)
        )

        if self.max_trials is not None:
            parameter_combinations = (
                parameter_combinations[
                    : self.max_trials
                ]
            )

        if not parameter_combinations:
            fitted_pipeline = clone(
                pipeline
            )

            start = time.perf_counter()

            fitted_pipeline.fit(
                X,
                y,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            return {
                "best_pipeline": fitted_pipeline,
                "best_params": {},
                "best_score": None,
                "trials": [],
                "successful_trials": 1,
                "failed_trials": 0,
                "total_time_seconds": float(
                    elapsed
                ),
            }

        splitter = self._create_splitter(
            y=y,
            task_type=task_type,
        )

        trials: list[dict[str, Any]] = []

        total_start = time.perf_counter()

        for trial_number, params in enumerate(
            parameter_combinations,
            start=1,
        ):
            trial = self._evaluate_trial(
                trial_number=trial_number,
                pipeline=pipeline,
                params=params,
                X=X,
                y=y,
                splitter=splitter,
                task_type=task_type,
            )

            trials.append(trial)

        successful_trials = [
            trial
            for trial in trials
            if trial["status"] == "success"
        ]

        failed_trials = [
            trial
            for trial in trials
            if trial["status"] == "failed"
        ]

        if not successful_trials:
            raise RuntimeError(
                "All hyperparameter optimization "
                "trials failed."
            )

        best_trial = self._select_best_trial(
            successful_trials,
            task_type=task_type,
        )

        best_pipeline = clone(
            pipeline
        )

        best_pipeline.set_params(
            **best_trial["params"]
        )

        total_time = (
            time.perf_counter()
            - total_start
        )

        return {
            "best_pipeline": best_pipeline,
            "best_params": best_trial["params"],
            "best_score": best_trial["score"],
            "trials": trials,
            "successful_trials": len(
                successful_trials
            ),
            "failed_trials": len(
                failed_trials
            ),
            "total_time_seconds": float(
                total_time
            ),
        }

    def _evaluate_trial(
        self,
        trial_number: int,
        pipeline: Pipeline,
        params: dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        splitter,
        task_type: str,
    ) -> dict[str, Any]:
        """Evaluate one parameter combination."""

        start = time.perf_counter()

        fold_scores: list[float] = []

        try:
            for train_indices, validation_indices in splitter.split(
                X,
                y if task_type == "classification" else None,
            ):
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

                fold_pipeline.set_params(
                    **params
                )

                fold_pipeline.fit(
                    X_train,
                    y_train,
                )

                predictions = (
                    fold_pipeline.predict(
                        X_validation
                    )
                )

                score = self._calculate_score(
                    y_true=y_validation,
                    predictions=predictions,
                    task_type=task_type,
                )

                fold_scores.append(
                    float(score)
                )

            mean_score = float(
                np.mean(fold_scores)
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            return {
                "trial": trial_number,
                "params": params.copy(),
                "score": mean_score,
                "fold_scores": fold_scores,
                "status": "success",
                "time_seconds": float(
                    elapsed
                ),
                "error": None,
            }

        except Exception as exc:
            elapsed = (
                time.perf_counter()
                - start
            )

            return {
                "trial": trial_number,
                "params": params.copy(),
                "score": None,
                "fold_scores": fold_scores,
                "status": "failed",
                "time_seconds": float(
                    elapsed
                ),
                "error": str(exc),
            }

    def _calculate_score(
        self,
        y_true,
        predictions,
        task_type: str,
    ) -> float:
        """Calculate the optimization score."""

        scoring = self.scoring

        if task_type == "regression":
            if scoring in {
                None,
                "r2",
            }:
                return float(
                    r2_score(
                        y_true,
                        predictions,
                    )
                )

            if scoring == "neg_mae":
                return float(
                    -mean_absolute_error(
                        y_true,
                        predictions,
                    )
                )

            if scoring == "neg_mse":
                return float(
                    -mean_squared_error(
                        y_true,
                        predictions,
                    )
                )

            if scoring == "neg_rmse":
                mse = mean_squared_error(
                    y_true,
                    predictions,
                )

                return float(
                    -np.sqrt(mse)
                )

            raise ValueError(
                "Unsupported regression scoring: "
                f"{scoring}"
            )

        if scoring in {
            None,
            "f1",
            "f1_weighted",
        }:
            return float(
                f1_score(
                    y_true,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            )

        if scoring == "accuracy":
            return float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            )

        if scoring == "precision":
            from sklearn.metrics import (
                precision_score,
            )

            return float(
                precision_score(
                    y_true,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            )

        if scoring == "recall":
            from sklearn.metrics import (
                recall_score,
            )

            return float(
                recall_score(
                    y_true,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            )

        raise ValueError(
            "Unsupported classification scoring: "
            f"{scoring}"
        )

    @staticmethod
    def _select_best_trial(
        trials: list[dict[str, Any]],
        task_type: str,
    ) -> dict[str, Any]:
        """Select the highest-scoring successful trial."""

        if not trials:
            raise ValueError(
                "No successful trials available."
            )

        return max(
            trials,
            key=lambda trial: trial["score"],
        )

    def _create_splitter(
        self,
        y: pd.Series,
        task_type: str,
    ):
        """Create the appropriate CV splitter."""

        if len(y) < self.cv:
            raise ValueError(
                f"Dataset must contain at least "
                f"{self.cv} samples."
            )

        if task_type == "classification":
            class_counts = y.value_counts()

            if (
                len(class_counts) >= 2
                and class_counts.min() >= self.cv
            ):
                return StratifiedKFold(
                    n_splits=self.cv,
                    shuffle=True,
                    random_state=self.random_state,
                )

            raise ValueError(
                "Each classification class must "
                f"contain at least {self.cv} "
                "samples for optimization."
            )

        return KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )

    def _validate_configuration(self):
        """Validate optimizer configuration."""

        if not isinstance(
            self.cv,
            int,
        ) or isinstance(
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
        ) or isinstance(
            self.random_state,
            bool,
        ):
            raise TypeError(
                "random_state must be an integer."
            )

        if self.scoring is not None:
            if not isinstance(
                self.scoring,
                str,
            ):
                raise TypeError(
                    "scoring must be a string or None."
                )

        if self.max_trials is not None:
            if not isinstance(
                self.max_trials,
                int,
            ) or isinstance(
                self.max_trials,
                bool,
            ):
                raise TypeError(
                    "max_trials must be an integer "
                    "or None."
                )

            if self.max_trials < 1:
                raise ValueError(
                    "max_trials must be at least 1."
                )

    @staticmethod
    def _validate_inputs(
        data: pd.DataFrame,
        target: str,
        pipeline: Pipeline,
        parameter_space: dict[str, Any],
        task_type: str,
    ):
        """Validate optimization inputs."""

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Cannot optimize on an empty dataset."
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
            pipeline,
            Pipeline,
        ):
            raise TypeError(
                "pipeline must be a sklearn Pipeline."
            )

        if not isinstance(
            parameter_space,
            dict,
        ):
            raise TypeError(
                "parameter_space must be a dictionary."
            )

        if task_type not in {
            "regression",
            "classification",
        }:
            raise ValueError(
                "task_type must be 'regression' "
                "or 'classification'."
            )

        for parameter_name in parameter_space:
            if not isinstance(
                parameter_name,
                str,
            ):
                raise TypeError(
                    "Parameter names must be strings."
                )