from typing import Any

import pandas as pd


class DataQualityAuditor:
    """
    Detect common data-quality and leakage risks before model training.
    """

    def __init__(
        self,
        high_missing_threshold: float = 0.50,
        high_cardinality_threshold: float = 0.90,
        id_like_threshold: float = 0.95,
    ):
        self.high_missing_threshold = high_missing_threshold
        self.high_cardinality_threshold = (
            high_cardinality_threshold
        )
        self.id_like_threshold = id_like_threshold

        self._validate_configuration()

    def audit(
        self,
        data: pd.DataFrame,
        target: str | None = None,
        column_intelligence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Audit a dataset for common quality and leakage risks.
        """

        self._validate_data(data, target)

        duplicate_rows = int(data.duplicated().sum())

        constant_columns = [
            column
            for column in data.columns
            if data[column].nunique(dropna=False) <= 1
        ]

        high_missing_columns = [
            column
            for column in data.columns
            if data[column].isna().mean()
            >= self.high_missing_threshold
        ]

        high_cardinality_columns = (
            self._detect_high_cardinality(data, target)
        )

        id_like_columns = self._detect_id_like_columns(
            data,
            target,
            column_intelligence,
        )

        target_leakage_columns = (
            self._detect_target_leakage(data, target)
        )

        issues = []

        if duplicate_rows > 0:
            issues.append(
                {
                    "type": "duplicate_rows",
                    "severity": "low",
                    "message": "Dataset contains duplicate rows.",
                    "count": duplicate_rows,
                }
            )

        if constant_columns:
            issues.append(
                {
                    "type": "constant_columns",
                    "severity": "medium",
                    "message": "Columns contain only one unique value.",
                    "columns": constant_columns,
                }
            )

        if high_missing_columns:
            issues.append(
                {
                    "type": "high_missing",
                    "severity": "high",
                    "message": (
                        "Columns contain a high proportion "
                        "of missing values."
                    ),
                    "columns": high_missing_columns,
                }
            )

        if high_cardinality_columns:
            issues.append(
                {
                    "type": "high_cardinality",
                    "severity": "medium",
                    "message": (
                        "Categorical or text columns have "
                        "unusually high cardinality."
                    ),
                    "columns": high_cardinality_columns,
                }
            )

        if id_like_columns:
            issues.append(
                {
                    "type": "id_like_columns",
                    "severity": "medium",
                    "message": (
                        "Columns appear to contain identifiers."
                    ),
                    "columns": id_like_columns,
                }
            )

        if target_leakage_columns:
            issues.append(
                {
                    "type": "target_leakage",
                    "severity": "critical",
                    "message": (
                        "Columns may contain direct target leakage."
                    ),
                    "columns": target_leakage_columns,
                }
            )

        critical_issue_types = {
            "target_leakage",
        }

        has_critical_issues = any(
            issue["type"] in critical_issue_types
            for issue in issues
        )

        return {
            "target": target,
            "duplicate_rows": duplicate_rows,
            "constant_columns": constant_columns,
            "high_missing_columns": high_missing_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "id_like_columns": id_like_columns,
            "target_leakage_columns": target_leakage_columns,
            "issues": issues,
            "issue_count": len(issues),
            "has_critical_issues": has_critical_issues,
        }

    def _detect_high_cardinality(
        self,
        data: pd.DataFrame,
        target: str | None,
    ) -> list[str]:
        """
        Detect high-cardinality categorical/text columns.

        Continuous numerical features are intentionally excluded.
        """

        columns = []

        for column in data.columns:
            if column == target:
                continue

            series = data[column]

            is_text_or_categorical = (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or isinstance(
                    series.dtype,
                    pd.CategoricalDtype,
                )
            )

            if not is_text_or_categorical:
                continue

            non_missing = series.dropna()

            if non_missing.empty:
                continue

            unique_count = non_missing.nunique(
                dropna=True
            )

            unique_ratio = (
                unique_count / len(non_missing)
            )

            # High-cardinality categorical/text data is detected
            # when most values are unique. A minimum of 5 unique
            # values prevents tiny categorical columns from being
            # flagged unnecessarily.
            if (
                unique_ratio
                >= self.high_cardinality_threshold
                and unique_count >= 5
            ):
                columns.append(column)

        return columns

    def _detect_id_like_columns(
        self,
        data: pd.DataFrame,
        target: str | None,
        column_intelligence: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Detect columns that appear to be identifiers.
        """

        detected = []

        if column_intelligence:
            for key in (
                "id_like_columns",
                "identifier_columns",
                "id_columns",
            ):
                values = column_intelligence.get(key)

                if isinstance(values, list):
                    for column in values:
                        if (
                            column in data.columns
                            and column != target
                            and column not in detected
                        ):
                            detected.append(column)

        for column in data.columns:
            if column == target or column in detected:
                continue

            series = data[column]

            non_missing = series.dropna()

            if non_missing.empty:
                continue

            unique_ratio = (
                non_missing.nunique(dropna=True)
                / len(non_missing)
            )

            normalized_name = (
                str(column)
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            name_tokens = {
                token
                for token in normalized_name.split("_")
                if token
            }

            identifier_tokens = {
                "id",
                "identifier",
                "uuid",
                "guid",
                "customerid",
                "userid",
                "user_id",
                "recordid",
                "record_id",
            }

            name_suggests_id = bool(
                name_tokens.intersection(
                    identifier_tokens
                )
            ) or normalized_name.endswith("id")

            if (
                name_suggests_id
                and unique_ratio
                >= self.id_like_threshold
            ):
                detected.append(column)

        return detected

    def _detect_target_leakage(
        self,
        data: pd.DataFrame,
        target: str | None,
    ) -> list[str]:
        """
        Detect simple forms of direct target leakage.
        """

        if target is None or target not in data.columns:
            return []

        target_series = data[target]

        leaked_columns = []

        for column in data.columns:
            if column == target:
                continue

            series = data[column]

            if len(series) != len(target_series):
                continue

            try:
                comparison = series.eq(
                    target_series
                )

                comparable = (
                    series.notna()
                    & target_series.notna()
                )

                if comparable.any():
                    match_ratio = (
                        comparison[comparable].mean()
                    )

                    if match_ratio == 1.0:
                        leaked_columns.append(column)
                        continue

            except (TypeError, ValueError):
                continue

        return leaked_columns

    @staticmethod
    def _validate_data(
        data: pd.DataFrame,
        target: str | None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Cannot audit an empty dataset."
            )

        if target is not None and target not in data.columns:
            raise ValueError(
                f"Target column '{target}' does not exist."
            )

    def _validate_configuration(self) -> None:
        if not (
            0 < self.high_missing_threshold <= 1
        ):
            raise ValueError(
                "high_missing_threshold must be between 0 and 1."
            )

        if not (
            0 < self.high_cardinality_threshold <= 1
        ):
            raise ValueError(
                "high_cardinality_threshold must be between 0 and 1."
            )

        if not (
            0 < self.id_like_threshold <= 1
        ):
            raise ValueError(
                "id_like_threshold must be between 0 and 1."
            )