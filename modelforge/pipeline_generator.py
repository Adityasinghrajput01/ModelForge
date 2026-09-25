from typing import Any

import pandas as pd

from sklearn.pipeline import Pipeline

from modelforge.feature_selection import (
    CorrelationFeatureSelector,
    VarianceFeatureSelector,
)
from modelforge.model_registry import (
    ModelRegistry,
)
from modelforge.preprocessing import (
    PreprocessingEngine,
)


class PipelineGenerator:
    """
    Generate complete machine-learning pipelines
    for ModelForge.

    The generated pipeline connects:

        preprocessing
            ↓
        feature selection
            ↓
        model
    """

    def __init__(
        self,
        model_registry: ModelRegistry | None = None,
        preprocessing_engine: PreprocessingEngine | None = None,
    ):
        self.model_registry = (
            model_registry
            if model_registry is not None
            else ModelRegistry()
        )

        self.preprocessing_engine = (
            preprocessing_engine
            if preprocessing_engine is not None
            else PreprocessingEngine()
        )

    def build(
        self,
        data: pd.DataFrame,
        target: str,
        model_name: str,
        task_type: str,
        excluded_columns: list[str] | None = None,
        variance_threshold: float | None = None,
        correlation_threshold: float | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> Pipeline:
        """
        Build a complete ModelForge pipeline.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset.

        target : str
            Target column.

        model_name : str
            Registered model identifier.

        task_type : str
            'regression' or 'classification'.

        excluded_columns : list[str] | None
            Columns that should not be used as features.

        variance_threshold : float | None
            Optional variance filtering threshold.

        correlation_threshold : float | None
            Optional correlation filtering threshold.

        model_params : dict | None
            Parameters overriding model defaults.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Complete ML pipeline.
        """

        self._validate_inputs(
            data=data,
            target=target,
            task_type=task_type,
        )

        model_spec = self.model_registry.get(
            model_name
        )

        if model_spec.task_type != task_type:
            raise ValueError(
                f"Model '{model_name}' is a "
                f"{model_spec.task_type} model and "
                f"cannot be used for {task_type}."
            )

        excluded = list(
            excluded_columns or []
        )

        preprocessing = (
            self.preprocessing_engine.build(
                data=data,
                target=target,
                model_type=self._get_model_type(
                    model_spec
                ),
                excluded_columns=excluded,
            )
        )

        # Feature-selection transformers in ModelForge
        # explicitly require pandas DataFrames.
        #
        # ColumnTransformer normally produces a NumPy array.
        # When feature selection is enabled, configure the
        # preprocessing transformer to return pandas output.
        if (
            variance_threshold is not None
            or correlation_threshold is not None
        ):
            if not hasattr(
                preprocessing,
                "set_output",
            ):
                raise RuntimeError(
                    "The preprocessing transformer does not "
                    "support pandas output. A compatible "
                    "scikit-learn version is required."
                )

            preprocessing.set_output(
                transform="pandas"
            )

        steps = []

        steps.append(
            (
                "preprocessing",
                preprocessing,
            )
        )

        if variance_threshold is not None:
            steps.append(
                (
                    "variance_selection",
                    VarianceFeatureSelector(
                        threshold=variance_threshold
                    ),
                )
            )

        if correlation_threshold is not None:
            steps.append(
                (
                    "correlation_selection",
                    CorrelationFeatureSelector(
                        threshold=correlation_threshold
                    ),
                )
            )

        model = self.model_registry.create(
            model_name,
            **(model_params or {}),
        )

        steps.append(
            (
                "model",
                model,
            )
        )

        return Pipeline(
            steps=steps
        )

    @staticmethod
    def _validate_inputs(
        data: pd.DataFrame,
        target: str,
        task_type: str,
    ) -> None:
        """
        Validate pipeline construction inputs.
        """

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Cannot build a pipeline from "
                "an empty dataset."
            )

        if target not in data.columns:
            raise ValueError(
                f"Target column '{target}' "
                "does not exist."
            )

        if task_type not in {
            "regression",
            "classification",
        }:
            raise ValueError(
                "task_type must be 'regression' "
                "or 'classification'."
            )

    @staticmethod
    def _get_model_type(
        model_spec,
    ) -> str:
        """
        Convert model metadata into a preprocessing
        model category.
        """

        if model_spec.requires_scaling:
            if model_spec.category == "svm":
                return "svm"

            if model_spec.category == "distance_based":
                return "knn"

            if model_spec.category == "linear":
                if (
                    "classification"
                    == model_spec.task_type
                ):
                    return "logistic_regression"

                if model_spec.name.lower().startswith(
                    "ridge"
                ):
                    return "ridge"

                if model_spec.name.lower().startswith(
                    "lasso"
                ):
                    return "lasso"

                if model_spec.name.lower().startswith(
                    "elastic"
                ):
                    return "elasticnet"

                return "linear_regression"

            return "mlp"

        return "tree"