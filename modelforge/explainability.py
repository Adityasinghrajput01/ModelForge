from typing import Any

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


class ExplainabilityEngine:
    """
    Generate model explanations for ModelForge pipelines.

    Supports:
    - native feature importance
    - linear model coefficients
    - permutation importance
    - prediction-level summaries
    """

    def feature_importance(
        self,
        pipeline: Pipeline,
    ) -> pd.DataFrame:
        """
        Extract native feature importance or coefficients
        from a fitted pipeline.
        """

        self._validate_pipeline(pipeline)

        model = self._get_model(pipeline)

        feature_names = self._get_feature_names(
            pipeline
        )

        if hasattr(model, "feature_importances_"):
            importance = np.asarray(
                model.feature_importances_
            )

            return self._build_importance_dataframe(
                feature_names,
                importance,
                source="feature_importance",
            )

        if hasattr(model, "coef_"):
            coefficients = np.asarray(
                model.coef_
            )

            if coefficients.ndim == 1:
                importance = np.abs(
                    coefficients
                )

            elif coefficients.ndim == 2:
                importance = np.mean(
                    np.abs(coefficients),
                    axis=0,
                )

            else:
                raise ValueError(
                    "Unsupported coefficient shape."
                )

            return self._build_importance_dataframe(
                feature_names,
                importance,
                source="coefficient",
            )

        raise ValueError(
            f"Model '{type(model).__name__}' "
            "does not provide native feature importance "
            "or coefficients."
        )

    def permutation_importance(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y,
        scoring: str | None = None,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Calculate permutation importance on the complete pipeline.
        """

        self._validate_pipeline(pipeline)

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        if X.empty:
            raise ValueError(
                "X cannot be empty."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number "
                "of samples."
            )

        if n_repeats <= 0:
            raise ValueError(
                "n_repeats must be greater than 0."
            )

        result = permutation_importance(
            estimator=pipeline,
            X=X,
            y=y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        feature_names = X.columns.tolist()

        return self._build_importance_dataframe(
            feature_names,
            result.importances_mean,
            source="permutation",
            std=result.importances_std,
        )

    def explain_prediction(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Generate a simple prediction-level explanation.

        This does not attempt to calculate SHAP-style local
        contributions. Instead, it combines the prediction
        with global feature importance information.
        """

        self._validate_pipeline(pipeline)

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        if X.empty:
            raise ValueError(
                "X cannot be empty."
            )

        predictions = pipeline.predict(X)

        try:
            importance = self.feature_importance(
                pipeline
            )
        except ValueError:
            importance = None

        return {
            "predictions": predictions.tolist(),
            "prediction_count": int(
                len(predictions)
            ),
            "feature_importance": importance,
        }

    @staticmethod
    def top_features(
        importance: pd.DataFrame,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Return the top N most important features.
        """

        if not isinstance(
            importance,
            pd.DataFrame,
        ):
            raise TypeError(
                "importance must be a pandas DataFrame."
            )

        if importance.empty:
            raise ValueError(
                "importance cannot be empty."
            )

        if n <= 0:
            raise ValueError(
                "n must be greater than 0."
            )

        return (
            importance
            .sort_values(
                by="importance",
                ascending=False,
            )
            .head(n)
            .reset_index(drop=True)
        )

    @staticmethod
    def _get_model(
        pipeline: Pipeline,
    ):
        if not hasattr(
            pipeline,
            "named_steps",
        ):
            raise TypeError(
                "pipeline must contain named steps."
            )

        if "model" not in pipeline.named_steps:
            raise ValueError(
                "Pipeline must contain a 'model' step."
            )

        return pipeline.named_steps["model"]

    @staticmethod
    def _get_feature_names(
        pipeline: Pipeline,
    ) -> list[str]:
        """
        Extract transformed feature names from the
        preprocessing step.
        """

        if "preprocessing" not in pipeline.named_steps:
            raise ValueError(
                "Pipeline must contain a "
                "'preprocessing' step."
            )

        preprocessing = pipeline.named_steps[
            "preprocessing"
        ]

        try:
            feature_names = (
                preprocessing.get_feature_names_out()
            )

            return [
                str(name)
                for name in feature_names
            ]

        except (AttributeError, ValueError):
            pass

        if hasattr(
            preprocessing,
            "feature_names_in_",
        ):
            return [
                str(name)
                for name in preprocessing.feature_names_in_
            ]

        raise ValueError(
            "Unable to determine transformed "
            "feature names."
        )

    @staticmethod
    def _build_importance_dataframe(
        feature_names,
        importance,
        source: str,
        std=None,
    ) -> pd.DataFrame:
        importance = np.asarray(
            importance,
            dtype=float,
        )

        feature_names = list(
            feature_names
        )

        if len(feature_names) != len(
            importance
        ):
            raise ValueError(
                "Number of feature names does not "
                "match number of importance values."
            )

        result = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
                "source": source,
            }
        )

        if std is not None:
            result["std"] = np.asarray(
                std,
                dtype=float,
            )

        result["absolute_importance"] = (
            result["importance"].abs()
        )

        return (
            result
            .sort_values(
                by="absolute_importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    @staticmethod
    def _validate_pipeline(
        pipeline: Pipeline,
    ) -> None:
        if not isinstance(
            pipeline,
            Pipeline,
        ):
            raise TypeError(
                "pipeline must be a sklearn Pipeline."
            )

        if "model" not in pipeline.named_steps:
            raise ValueError(
                "Pipeline must contain a 'model' step."
            )