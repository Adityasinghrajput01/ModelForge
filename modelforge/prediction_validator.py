from __future__ import annotations

from typing import Any

import pandas as pd


class PredictionSchemaError(ValueError):
    """Raised when prediction data does not match the expected schema."""


class PredictionSchemaValidator:
    """
    Validate prediction data against a training feature schema.

    The validator checks:
        - required columns
        - unexpected columns
        - column order
        - empty datasets
        - basic data types
    """

    def __init__(
        self,
        allow_extra_columns: bool = True,
        enforce_column_order: bool = False,
        enforce_dtypes: bool = False,
    ):
        if not isinstance(
            allow_extra_columns,
            bool,
        ):
            raise TypeError(
                "allow_extra_columns must be a boolean."
            )

        if not isinstance(
            enforce_column_order,
            bool,
        ):
            raise TypeError(
                "enforce_column_order must be a boolean."
            )

        if not isinstance(
            enforce_dtypes,
            bool,
        ):
            raise TypeError(
                "enforce_dtypes must be a boolean."
            )

        self.allow_extra_columns = (
            allow_extra_columns
        )

        self.enforce_column_order = (
            enforce_column_order
        )

        self.enforce_dtypes = (
            enforce_dtypes
        )

    def validate(
        self,
        data: pd.DataFrame,
        expected_columns: list[str],
        expected_dtypes: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Validate prediction data.

        Returns a validation report.

        Raises:
            TypeError:
                If inputs have invalid types.

            PredictionSchemaError:
                If the prediction schema is invalid.
        """

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if not isinstance(
            expected_columns,
            list,
        ):
            raise TypeError(
                "expected_columns must be a list."
            )

        if not all(
            isinstance(column, str)
            for column in expected_columns
        ):
            raise TypeError(
                "expected_columns must contain strings."
            )

        if expected_dtypes is not None:
            if not isinstance(
                expected_dtypes,
                dict,
            ):
                raise TypeError(
                    "expected_dtypes must be a dictionary."
                )

        if data.empty:
            raise PredictionSchemaError(
                "Prediction data is empty."
            )

        actual_columns = list(
            data.columns
        )

        missing_columns = [
            column
            for column in expected_columns
            if column not in actual_columns
        ]

        unexpected_columns = [
            column
            for column in actual_columns
            if column not in expected_columns
        ]

        order_matches = (
            actual_columns
            == expected_columns
        )

        dtype_mismatches = {}

        if (
            self.enforce_dtypes
            and expected_dtypes is not None
        ):
            for column in expected_columns:
                if column not in data.columns:
                    continue

                expected_dtype = str(
                    expected_dtypes.get(
                        column,
                        "",
                    )
                )

                actual_dtype = str(
                    data[column].dtype
                )

                if (
                    expected_dtype
                    and actual_dtype
                    != expected_dtype
                ):
                    dtype_mismatches[column] = {
                        "expected": expected_dtype,
                        "actual": actual_dtype,
                    }

        errors = []

        if missing_columns:
            errors.append(
                "Missing required columns: "
                + ", ".join(
                    missing_columns
                )
            )

        if (
            unexpected_columns
            and not self.allow_extra_columns
        ):
            errors.append(
                "Unexpected columns: "
                + ", ".join(
                    unexpected_columns
                )
            )

        if (
            self.enforce_column_order
            and not order_matches
        ):
            errors.append(
                "Column order does not match "
                "the expected feature order."
            )

        if dtype_mismatches:
            errors.append(
                "Data type mismatch for columns: "
                + ", ".join(
                    dtype_mismatches.keys()
                )
            )

        valid = not errors

        report = {
            "valid": valid,
            "expected_columns": expected_columns,
            "actual_columns": actual_columns,
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
            "order_matches": order_matches,
            "dtype_mismatches": dtype_mismatches,
            "row_count": len(data),
        }

        if not valid:
            raise PredictionSchemaError(
                "Prediction data failed schema "
                "validation. "
                + " ".join(errors)
            )

        return report

    def validate_and_align(
        self,
        data: pd.DataFrame,
        expected_columns: list[str],
        expected_dtypes: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """
        Validate prediction data and return it aligned
        to the expected feature order.

        Extra columns are removed when they are allowed.
        """

        self.validate(
            data=data,
            expected_columns=expected_columns,
            expected_dtypes=expected_dtypes,
        )

        aligned = data.copy()

        aligned = aligned[
            expected_columns
        ]

        return aligned

    @staticmethod
    def infer_schema(
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Infer a prediction schema from a DataFrame.
        """

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise PredictionSchemaError(
                "Cannot infer schema from "
                "an empty DataFrame."
            )

        return {
            "columns": list(
                data.columns
            ),
            "dtypes": {
                column: str(
                    data[column].dtype
                )
                for column in data.columns
            },
            "n_features": len(
                data.columns
            ),
        }

    @staticmethod
    def format_error(
        error: Exception,
    ) -> str:
        """
        Return a clean user-facing error message.
        """

        if isinstance(
            error,
            PredictionSchemaError,
        ):
            return str(error)

        return (
            "Prediction validation failed: "
            f"{error}"
        )