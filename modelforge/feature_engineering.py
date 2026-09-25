from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class DateTimeFeatureExtractor(
    BaseEstimator,
    TransformerMixin,
):
    """
    Extract useful features from datetime columns.

    Datetime columns can be supplied explicitly. When they are not supplied,
    the transformer detects native datetime columns and datetime-like
    string/object columns during fit().

    The original datetime columns are removed after their derived features
    are created.
    """

    def __init__(self, datetime_columns=None):
        self.datetime_columns = datetime_columns

    def fit(self, X, y=None):
        """Fit the transformer and identify datetime columns."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        if self.datetime_columns is None:
            self.datetime_columns_ = []

            for column in X.columns:
                series = X[column]

                if pd.api.types.is_datetime64_any_dtype(series):
                    self.datetime_columns_.append(column)
                    continue

                if (
                    pd.api.types.is_object_dtype(series)
                    or pd.api.types.is_string_dtype(series)
                ):
                    if self._looks_like_datetime(series):
                        self.datetime_columns_.append(column)
        else:
            missing = [
                column
                for column in self.datetime_columns
                if column not in X.columns
            ]

            if missing:
                raise ValueError(
                    "The following datetime columns were not found: "
                    f"{missing}"
                )

            self.datetime_columns_ = list(
                self.datetime_columns
            )

        return self

    def transform(self, X):
        """Extract datetime features."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        if not hasattr(self, "datetime_columns_"):
            raise RuntimeError(
                "DateTimeFeatureExtractor must be fitted before transform."
            )

        result = X.copy()

        for column in self.datetime_columns_:
            if column not in result.columns:
                continue

            datetime_series = pd.to_datetime(
                result[column],
                errors="coerce",
                format="mixed",
            )

            prefix = str(column)

            result[f"{prefix}_year"] = (
                datetime_series.dt.year
            )

            result[f"{prefix}_month"] = (
                datetime_series.dt.month
            )

            result[f"{prefix}_day"] = (
                datetime_series.dt.day
            )

            result[f"{prefix}_weekday"] = (
                datetime_series.dt.weekday
            )

            result[f"{prefix}_quarter"] = (
                datetime_series.dt.quarter
            )

            result[f"{prefix}_hour"] = (
                datetime_series.dt.hour
            )

            result[f"{prefix}_is_weekend"] = (
                datetime_series.dt.weekday >= 5
            )

            result = result.drop(
                columns=[column]
            )

        return result

    def _looks_like_datetime(
        self,
        series: pd.Series,
    ) -> bool:
        """Determine whether an object/string column is datetime-like."""

        non_null = series.dropna()

        if non_null.empty:
            return False

        sample = non_null.astype(str).head(100)

        if sample.empty:
            return False

        try:
            parsed = pd.to_datetime(
                sample,
                errors="coerce",
                format="mixed",
            )
        except (TypeError, ValueError):
            return False

        return bool(
            parsed.notna().mean() >= 0.90
        )


class NumericalFeatureTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Generate optional mathematical features for numerical columns.

    Supported transformations:

    - log1p
    - square
    - square root

    Invalid mathematical operations are represented as NaN rather than
    infinity or complex values.
    """

    def __init__(
        self,
        numerical_columns=None,
        add_log=False,
        add_square=False,
        add_sqrt=False,
    ):
        self.numerical_columns = numerical_columns
        self.add_log = add_log
        self.add_square = add_square
        self.add_sqrt = add_sqrt

    def fit(self, X, y=None):
        """Fit the numerical feature transformer."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        if self.numerical_columns is None:
            self.numerical_columns_ = (
                X.select_dtypes(
                    include="number"
                ).columns.tolist()
            )
        else:
            missing = [
                column
                for column in self.numerical_columns
                if column not in X.columns
            ]

            if missing:
                raise ValueError(
                    "The following numerical columns were not found: "
                    f"{missing}"
                )

            self.numerical_columns_ = list(
                self.numerical_columns
            )

        return self

    def transform(self, X):
        """Generate numerical features."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        if not hasattr(self, "numerical_columns_"):
            raise RuntimeError(
                "NumericalFeatureTransformer must be fitted before transform."
            )

        result = X.copy()

        for column in self.numerical_columns_:
            if column not in result.columns:
                continue

            values = pd.to_numeric(
                result[column],
                errors="coerce",
            )

            safe_name = str(column)

            if self.add_log:
                log_values = pd.Series(
                    np.nan,
                    index=result.index,
                    dtype=float,
                )

                valid_log = values > -1

                log_values.loc[valid_log] = np.log1p(
                    values.loc[valid_log]
                )

                result[
                    f"{safe_name}_log1p"
                ] = log_values

            if self.add_square:
                result[
                    f"{safe_name}_squared"
                ] = values ** 2

            if self.add_sqrt:
                sqrt_values = pd.Series(
                    np.nan,
                    index=result.index,
                    dtype=float,
                )

                valid_sqrt = values >= 0

                sqrt_values.loc[valid_sqrt] = np.sqrt(
                    values.loc[valid_sqrt]
                )

                result[
                    f"{safe_name}_sqrt"
                ] = sqrt_values

        return result


class FeatureEngineeringEngine:
    """
    Coordinate ModelForge feature-engineering transformations.

    The engine can independently transform datetime and numerical features,
    or apply both transformations in sequence.
    """

    def fit_transform(
        self,
        data: pd.DataFrame,
        datetime_columns=None,
        numerical_columns=None,
        add_log=False,
        add_square=False,
        add_sqrt=False,
    ) -> pd.DataFrame:
        """Fit and apply the complete feature-engineering pipeline."""

        return self.transform(
            data=data,
            datetime_columns=datetime_columns,
            numerical_columns=numerical_columns,
            add_log=add_log,
            add_square=add_square,
            add_sqrt=add_sqrt,
        )

    def transform_datetime(
        self,
        data: pd.DataFrame,
        datetime_columns=None,
    ) -> pd.DataFrame:
        """Generate datetime features."""

        extractor = DateTimeFeatureExtractor(
            datetime_columns=datetime_columns
        )

        return extractor.fit_transform(data)

    def transform_numeric(
        self,
        data: pd.DataFrame,
        numerical_columns=None,
        add_log=False,
        add_square=False,
        add_sqrt=False,
    ) -> pd.DataFrame:
        """Generate numerical features."""

        transformer = NumericalFeatureTransformer(
            numerical_columns=numerical_columns,
            add_log=add_log,
            add_square=add_square,
            add_sqrt=add_sqrt,
        )

        return transformer.fit_transform(data)

    def transform(
        self,
        data: pd.DataFrame,
        datetime_columns=None,
        numerical_columns=None,
        add_log=False,
        add_square=False,
        add_sqrt=False,
    ) -> pd.DataFrame:
        """
        Apply the complete feature-engineering pipeline.

        Datetime transformations are applied first, followed by numerical
        transformations.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        result = data.copy()

        datetime_extractor = DateTimeFeatureExtractor(
            datetime_columns=datetime_columns
        )

        result = datetime_extractor.fit_transform(
            result
        )

        numerical_transformer = NumericalFeatureTransformer(
            numerical_columns=numerical_columns,
            add_log=add_log,
            add_square=add_square,
            add_sqrt=add_sqrt,
        )

        result = numerical_transformer.fit_transform(
            result
        )

        return result