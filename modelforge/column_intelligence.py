from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class ColumnIntelligence:
    """
    Analyze dataframe columns and infer useful semantic properties.

    The output preserves the public fields expected by ModelForge while
    exposing additional information for downstream AutoML components.
    """

    def __init__(
        self,
        high_cardinality_threshold: float = 0.90,
        id_like_threshold: float = 0.95,
        text_unique_threshold: float = 0.50,
    ) -> None:
        if not 0 < high_cardinality_threshold <= 1:
            raise ValueError(
                "high_cardinality_threshold must be between 0 and 1."
            )

        if not 0 < id_like_threshold <= 1:
            raise ValueError(
                "id_like_threshold must be between 0 and 1."
            )

        if not 0 < text_unique_threshold <= 1:
            raise ValueError(
                "text_unique_threshold must be between 0 and 1."
            )

        self.high_cardinality_threshold = high_cardinality_threshold
        self.id_like_threshold = id_like_threshold
        self.text_unique_threshold = text_unique_threshold

    def analyze(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze all columns in a dataframe."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        if data.empty:
            raise ValueError("data must not be empty.")

        columns: dict[str, dict[str, Any]] = {}

        for column in data.columns:
            columns[str(column)] = self._analyze_column(
                data[column],
                str(column),
            )

        return {
            "n_rows": int(len(data)),
            "n_columns": int(len(data.columns)),
            "columns": columns,
            "numeric_columns": [
                column
                for column, info in columns.items()
                if info["inferred_type"] == "numerical"
            ],
            "categorical_columns": [
                column
                for column, info in columns.items()
                if info["inferred_type"] == "categorical"
            ],
            "text_columns": [
                column
                for column, info in columns.items()
                if info["is_text_like"]
            ],
            "datetime_columns": [
                column
                for column, info in columns.items()
                if info["inferred_type"] == "datetime"
            ],
            "boolean_columns": [
                column
                for column, info in columns.items()
                if info["inferred_type"] == "boolean"
            ],
            "id_like_columns": [
                column
                for column, info in columns.items()
                if info["is_id_like"]
            ],
            "high_cardinality_columns": [
                column
                for column, info in columns.items()
                if info["is_high_cardinality"]
            ],
            "constant_columns": [
                column
                for column, info in columns.items()
                if info["inferred_type"] == "constant"
            ],
            "missing_columns": [
                column
                for column, info in columns.items()
                if info["missing_count"] > 0
            ],
        }

    def _analyze_column(
        self,
        series: pd.Series,
        column_name: str,
    ) -> dict[str, Any]:
        """Analyze one dataframe column."""

        total_count = int(len(series))
        missing_count = int(series.isna().sum())
        non_missing_count = total_count - missing_count

        unique_count = int(series.nunique(dropna=True))

        unique_ratio = (
            unique_count / non_missing_count
            if non_missing_count > 0
            else 0.0
        )

        dtype_name = str(series.dtype)

        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_boolean = pd.api.types.is_bool_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)

        datetime_like = False

        if not is_datetime and (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            datetime_like = self._looks_like_datetime(series)

        is_text_like = self._looks_like_text(
            series,
            unique_ratio,
        )

        is_constant = unique_count <= 1

        if is_constant:
            inferred_type = "constant"

        elif self._is_id_name(column_name):
            inferred_type = "id"

        elif is_boolean:
            inferred_type = "boolean"

        elif is_datetime or datetime_like:
            inferred_type = "datetime"

        elif is_numeric:
            inferred_type = "numerical"

        elif is_text_like:
            inferred_type = "text"

        else:
            inferred_type = "categorical"

        is_high_cardinality = (
            unique_ratio >= self.high_cardinality_threshold
            and unique_count >= 5
            and inferred_type in {"categorical", "text"}
        )

        is_id_like = self._is_id_like(
            series=series,
            column_name=column_name,
            unique_ratio=unique_ratio,
            inferred_type=inferred_type,
        )

        statistics = (
            self._numeric_statistics(series)
            if is_numeric
            else {}
        )

        return {
            # Existing/public contract.
            "name": column_name,
            "dtype": dtype_name,
            "inferred_type": inferred_type,
            "is_text_like": bool(is_text_like),
            "is_id_like": bool(is_id_like),
            "is_constant": bool(is_constant),

            # Additional intelligence.
            "role": self._role_from_type(inferred_type),
            "total_count": total_count,
            "non_missing_count": non_missing_count,
            "missing_count": missing_count,
            "missing_ratio": (
                missing_count / total_count
                if total_count > 0
                else 0.0
            ),
            "unique_count": unique_count,
            "unique_ratio": float(unique_ratio),
            "is_numeric": bool(is_numeric),
            "is_boolean": bool(is_boolean),
            "is_datetime": bool(is_datetime or datetime_like),
            "is_high_cardinality": bool(is_high_cardinality),
            "statistics": statistics,
        }

    def _role_from_type(self, inferred_type: str) -> str:
        """Map the public inferred type to a semantic role."""
        mapping = {
            "numerical": "numeric",
            "categorical": "categorical",
            "text": "text",
            "datetime": "datetime",
            "boolean": "boolean",
            "id": "id",
            "constant": "constant",
        }

        return mapping.get(inferred_type, inferred_type)

    def _is_id_name(self, column_name: str) -> bool:
        """Check whether a column name strongly indicates an identifier."""
        normalized_name = (
            column_name.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        explicit_id_names = {
            "id",
            "uuid",
            "guid",
            "identifier",
            "record_id",
            "row_id",
            "customer_id",
            "user_id",
            "employee_id",
            "transaction_id",
            "account_id",
            "order_id",
            "product_id",
            "patient_id",
            "student_id",
        }

        return (
            normalized_name in explicit_id_names
            or normalized_name.endswith("_id")
        )

    def _looks_like_text(
        self,
        series: pd.Series,
        unique_ratio: float,
    ) -> bool:
        """Detect likely free-form text columns."""
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            return False

        non_null = series.dropna()

        if non_null.empty:
            return False

        values = non_null.astype(str)

        average_length = float(values.str.len().mean())

        return bool(
            average_length >= 30
            and unique_ratio >= self.text_unique_threshold
        )

    def _looks_like_datetime(self, series: pd.Series) -> bool:
        """Detect object/string columns that appear to contain dates."""
        non_null = series.dropna()

        if non_null.empty:
            return False

        sample = non_null.astype(str).head(100)

        if sample.empty:
            return False

        parsed = pd.to_datetime(
            sample,
            errors="coerce",
            format="mixed",
        )

        parse_ratio = float(parsed.notna().mean())

        return parse_ratio >= 0.90

    def _is_id_like(
        self,
        series: pd.Series,
        column_name: str,
        unique_ratio: float,
        inferred_type: str,
    ) -> bool:
        """Detect columns that are likely identifiers."""

        if self._is_id_name(column_name):
            return True

        if inferred_type in {"categorical", "text"}:
            return bool(unique_ratio >= self.id_like_threshold)

        if inferred_type == "numerical":
            if unique_ratio >= self.id_like_threshold:
                values = series.dropna()

                if not values.empty:
                    try:
                        numeric_values = values.astype(float)

                        integer_like = np.all(
                            np.isclose(
                                numeric_values,
                                np.round(numeric_values),
                            )
                        )

                        if integer_like and self._looks_sequential(
                            values
                        ):
                            return True
                    except (TypeError, ValueError):
                        return False

        return False

    def _looks_sequential(self, series: pd.Series) -> bool:
        """Check whether numeric values resemble a sequential identifier."""
        if series.empty:
            return False

        values = np.sort(series.unique())

        if len(values) < 5:
            return False

        differences = np.diff(values)

        if len(differences) == 0:
            return False

        positive_differences = differences[differences > 0]

        if len(positive_differences) == 0:
            return False

        return bool(
            np.all(
                np.isclose(
                    positive_differences,
                    positive_differences[0],
                )
            )
        )

    def _numeric_statistics(
        self,
        series: pd.Series,
    ) -> dict[str, Any]:
        """Return safe summary statistics for numeric columns."""
        numeric = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        if numeric.empty:
            return {}

        return {
            "min": float(numeric.min()),
            "max": float(numeric.max()),
            "mean": float(numeric.mean()),
            "median": float(numeric.median()),
            "std": (
                float(numeric.std())
                if len(numeric) > 1
                else 0.0
            ),
            "zero_count": int((numeric == 0).sum()),
            "negative_count": int((numeric < 0).sum()),
        }