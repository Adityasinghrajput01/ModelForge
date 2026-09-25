from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


class PreprocessingEngine:
    """
    Build leakage-safe preprocessing pipelines for ModelForge.

    All preprocessing operations are placed inside a scikit-learn
    Pipeline / ColumnTransformer so that transformations are fitted
    only on training data during cross-validation.
    """

    SCALE_MODELS = {
        "knn",
        "svm",
        "svr",
        "linear_regression",
        "ridge",
        "lasso",
        "elasticnet",
        "logistic_regression",
        "mlp",
    }

    def build(
        self,
        data: pd.DataFrame,
        target: str,
        model_type: str = "tree",
        excluded_columns: Iterable[str] | None = None,
    ) -> ColumnTransformer:
        """
        Build a preprocessing transformer.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset.

        target : str
            Target column.

        model_type : str
            Model family. Determines whether numerical features
            should be scaled.

        excluded_columns : Iterable[str] | None
            Columns to exclude from preprocessing.

        Returns
        -------
        ColumnTransformer
            Leakage-safe preprocessing transformer.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        if data.empty:
            raise ValueError("Cannot preprocess an empty dataset.")

        if target not in data.columns:
            raise ValueError(
                f"Target column '{target}' does not exist."
            )

        excluded = set(excluded_columns or [])

        feature_data = data.drop(
            columns=[target],
            errors="ignore",
        )

        feature_data = feature_data.drop(
            columns=[
                column
                for column in excluded
                if column in feature_data.columns
            ],
            errors="ignore",
        )

        numerical_columns = feature_data.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = feature_data.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
                "string",
            ]
        ).columns.tolist()

        if not numerical_columns and not categorical_columns:
            raise ValueError(
                "No supported feature columns were found."
            )

        scale_numeric = self._should_scale(model_type)

        numeric_steps = [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]

        if scale_numeric:
            numeric_steps.append(
                (
                    "scaler",
                    StandardScaler(),
                )
            )

        numeric_pipeline = Pipeline(
            steps=numeric_steps
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                    ),
                ),
            ]
        )

        transformers = []

        if numerical_columns:
            transformers.append(
                (
                    "numerical",
                    numeric_pipeline,
                    numerical_columns,
                )
            )

        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns,
                )
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

    def _should_scale(self, model_type: str) -> bool:
        """Determine whether numerical features should be scaled."""

        normalized_model = model_type.lower().strip()

        return normalized_model in self.SCALE_MODELS