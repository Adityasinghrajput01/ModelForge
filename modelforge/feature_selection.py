from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
)


class VarianceFeatureSelector(
    BaseEstimator,
    TransformerMixin,
):
    """Remove features whose variance is below a configured threshold."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if self.threshold < 0:
            raise ValueError(
                "threshold must be greater than or equal to 0."
            )

        self.feature_names_in_ = X.columns.tolist()

        numeric = X.select_dtypes(include=np.number)

        self.numeric_columns_ = numeric.columns.tolist()

        if not self.numeric_columns_:
            self.selected_columns_ = self.feature_names_in_.copy()
            self.removed_columns_ = []
            self.variances_ = pd.Series(dtype=float)
            return self

        self.variances_ = numeric.var(
            ddof=0,
            numeric_only=True,
        )

        selected_numeric = self.variances_[
            self.variances_ > self.threshold
        ].index.tolist()

        self.selected_columns_ = [
            column
            for column in self.feature_names_in_
            if (
                column not in self.numeric_columns_
                or column in selected_numeric
            )
        ]

        self.removed_columns_ = [
            column
            for column in self.feature_names_in_
            if column not in self.selected_columns_
        ]

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if not hasattr(self, "selected_columns_"):
            raise RuntimeError(
                "VarianceFeatureSelector must be fitted before transform."
            )

        missing = [
            column
            for column in self.selected_columns_
            if column not in X.columns
        ]

        if missing:
            raise ValueError(
                "Input data is missing selected features: "
                f"{missing}"
            )

        return X.loc[:, self.selected_columns_].copy()

    def get_support(self) -> np.ndarray:
        if not hasattr(self, "selected_columns_"):
            raise RuntimeError(
                "VarianceFeatureSelector must be fitted before "
                "get_support()."
            )

        return np.array(
            [
                column in self.selected_columns_
                for column in self.feature_names_in_
            ],
            dtype=bool,
        )


class CorrelationFeatureSelector(
    BaseEstimator,
    TransformerMixin,
):
    """Remove highly correlated numerical features."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if not 0 < self.threshold <= 1:
            raise ValueError(
                "threshold must be greater than 0 and less than or equal "
                "to 1."
            )

        self.feature_names_in_ = X.columns.tolist()

        numeric = X.select_dtypes(include=np.number)

        self.numeric_columns_ = numeric.columns.tolist()

        if len(self.numeric_columns_) < 2:
            self.selected_columns_ = self.feature_names_in_.copy()
            self.removed_columns_ = []
            self.correlation_matrix_ = pd.DataFrame()
            return self

        self.correlation_matrix_ = numeric.corr(
            method="pearson"
        )

        upper = self.correlation_matrix_.where(
            np.triu(
                np.ones(
                    self.correlation_matrix_.shape,
                    dtype=bool,
                ),
                k=1,
            )
        )

        correlated_columns = [
            column
            for column in upper.columns
            if any(
                upper[column].abs() > self.threshold
            )
        ]

        self.removed_columns_ = correlated_columns

        self.selected_columns_ = [
            column
            for column in self.feature_names_in_
            if column not in self.removed_columns_
        ]

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if not hasattr(self, "selected_columns_"):
            raise RuntimeError(
                "CorrelationFeatureSelector must be fitted before "
                "transform."
            )

        missing = [
            column
            for column in self.selected_columns_
            if column not in X.columns
        ]

        if missing:
            raise ValueError(
                "Input data is missing selected features: "
                f"{missing}"
            )

        return X.loc[:, self.selected_columns_].copy()

    def get_support(self) -> np.ndarray:
        if not hasattr(self, "selected_columns_"):
            raise RuntimeError(
                "CorrelationFeatureSelector must be fitted before "
                "get_support()."
            )

        return np.array(
            [
                column in self.selected_columns_
                for column in self.feature_names_in_
            ],
            dtype=bool,
        )


class SelectKBestFeatureSelector(
    BaseEstimator,
    TransformerMixin,
):
    """
    Select the k statistically strongest features.

    For classification, ANOVA F-test is used.

    For regression, linear regression F-test is used.

    If k is larger than the number of available features, k is
    automatically capped to the feature count.
    """

    def __init__(
        self,
        k: int | str = 10,
        task_type: str = "classification",
    ):
        self.k = k
        self.task_type = task_type

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if self.task_type not in {
            "classification",
            "regression",
        }:
            raise ValueError(
                "task_type must be 'classification' or 'regression'."
            )

        if isinstance(self.k, int):
            if self.k <= 0:
                raise ValueError(
                    "k must be greater than 0."
                )
        elif self.k != "all":
            raise ValueError(
                "k must be a positive integer or 'all'."
            )

        self.feature_names_in_ = X.columns.tolist()

        numeric = X.select_dtypes(
            include=np.number
        )

        self.numeric_columns_ = numeric.columns.tolist()

        non_numeric = [
            column
            for column in self.feature_names_in_
            if column not in self.numeric_columns_
        ]

        self.non_numeric_columns_ = non_numeric

        if not self.numeric_columns_:
            self.selected_columns_ = self.feature_names_in_.copy()
            self.removed_columns_ = []
            self.scores_ = pd.Series(dtype=float)
            self.pvalues_ = pd.Series(dtype=float)
            self.effective_k_ = "all"
            return self

        if self.k == "all":
            effective_k = "all"
        else:
            effective_k = min(
                self.k,
                len(self.numeric_columns_),
            )

        self.effective_k_ = effective_k

        if self.task_type == "classification":
            score_func = f_classif
        else:
            score_func = f_regression

        numeric_data = numeric.copy()

        numeric_data = numeric_data.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        for column in numeric_data.columns:
            if numeric_data[column].isna().any():
                median = numeric_data[column].median()

                if pd.isna(median):
                    median = 0.0

                numeric_data[column] = (
                    numeric_data[column].fillna(median)
                )

        y_series = pd.Series(y).reset_index(drop=True)

        numeric_data = numeric_data.reset_index(
            drop=True
        )

        selector = SelectKBest(
            score_func=score_func,
            k=effective_k,
        )

        selector.fit(
            numeric_data,
            y_series,
        )

        self.selector_ = selector

        self.scores_ = pd.Series(
            selector.scores_,
            index=self.numeric_columns_,
            dtype=float,
        )

        self.pvalues_ = pd.Series(
            selector.pvalues_,
            index=self.numeric_columns_,
            dtype=float,
        )

        support = selector.get_support()

        selected_numeric = [
            column
            for column, selected in zip(
                self.numeric_columns_,
                support,
            )
            if selected
        ]

        self.selected_columns_ = (
            non_numeric + selected_numeric
        )

        self.selected_columns_ = [
            column
            for column in self.feature_names_in_
            if column in self.selected_columns_
        ]

        self.removed_columns_ = [
            column
            for column in self.feature_names_in_
            if column not in self.selected_columns_
        ]

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if not hasattr(self, "selected_columns_"):
            raise RuntimeError(
                "SelectKBestFeatureSelector must be fitted before "
                "transform."
            )

        missing = [
            column
            for column in self.selected_columns_
            if column not in X.columns
        ]

        if missing:
            raise ValueError(
                "Input data is missing selected features: "
                f"{missing}"
            )

        return X.loc[:, self.selected_columns_].copy()

    def get_support(self) -> np.ndarray:
        if not hasattr(self, "selected_columns_"):
            raise RuntimeError(
                "SelectKBestFeatureSelector must be fitted before "
                "get_support()."
            )

        return np.array(
            [
                column in self.selected_columns_
                for column in self.feature_names_in_
            ],
            dtype=bool,
        )


class FeatureSelectionEngine:
    """Coordinate ModelForge feature-selection strategies."""

    def variance_filter(
        self,
        data: pd.DataFrame,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        """Apply variance filtering."""

        selector = VarianceFeatureSelector(
            threshold=threshold
        )

        return selector.fit_transform(data)

    def correlation_filter(
        self,
        data: pd.DataFrame,
        threshold: float = 0.95,
    ) -> pd.DataFrame:
        """Apply correlation filtering."""

        selector = CorrelationFeatureSelector(
            threshold=threshold
        )

        return selector.fit_transform(data)

    def select_k_best(
        self,
        data: pd.DataFrame,
        target,
        k: int | str = 10,
        task_type: str = "classification",
    ) -> pd.DataFrame:
        """Apply SelectKBest feature selection."""

        selector = SelectKBestFeatureSelector(
            k=k,
            task_type=task_type,
        )

        return selector.fit_transform(
            data,
            target,
        )

    def select(
        self,
        data: pd.DataFrame,
        target=None,
        task_type: str | None = None,
        variance_threshold: float | None = None,
        correlation_threshold: float | None = None,
        k: int | str | None = None,
    ) -> pd.DataFrame:
        """
        Apply a configurable feature-selection sequence.

        Order:

        1. Variance filtering
        2. Correlation filtering
        3. SelectKBest
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        result = data.copy()

        if variance_threshold is not None:
            variance_selector = VarianceFeatureSelector(
                threshold=variance_threshold
            )

            result = variance_selector.fit_transform(
                result
            )

        if correlation_threshold is not None:
            correlation_selector = CorrelationFeatureSelector(
                threshold=correlation_threshold
            )

            result = correlation_selector.fit_transform(
                result
            )

        if k is not None:
            if target is None:
                raise ValueError(
                    "target is required when k is provided."
                )

            if task_type is None:
                raise ValueError(
                    "task_type is required when k is provided."
                )

            k_selector = SelectKBestFeatureSelector(
                k=k,
                task_type=task_type,
            )

            result = k_selector.fit_transform(
                result,
                target,
            )

        return result