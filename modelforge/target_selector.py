import pandas as pd


class TargetSelector:
    """Validate and analyze the target column selected by the user."""

    def select(
        self,
        data: pd.DataFrame,
        target: str,
        task_type: str | None = None,
    ) -> dict:
        """
        Validate the selected target and determine the ML task type.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset.

        target : str
            Name of the target column selected by the user.

        task_type : str | None
            Optional explicit task type:
            'regression' or 'classification'.

        Returns
        -------
        dict
            Target information and task type.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        if data.empty:
            raise ValueError("Dataset is empty.")

        if not target:
            raise ValueError("Target column must be provided.")

        if target not in data.columns:
            raise ValueError(
                f"Target column '{target}' does not exist in the dataset."
            )

        target_series = data[target]

        if target_series.isna().all():
            raise ValueError(
                f"Target column '{target}' contains only missing values."
            )

        if task_type is not None:
            task_type = task_type.lower()

            if task_type not in {"regression", "classification"}:
                raise ValueError(
                    "task_type must be 'regression' or 'classification'."
                )
        else:
            task_type = self._detect_task_type(target_series)

        return {
            "target": target,
            "task_type": task_type,
            "dtype": str(target_series.dtype),
            "unique_values": int(target_series.nunique(dropna=True)),
            "missing_values": int(target_series.isna().sum()),
            "rows": int(len(target_series)),
        }

    @staticmethod
    def _detect_task_type(target: pd.Series) -> str:
        """Detect the likely ML task from the target column."""

        if pd.api.types.is_bool_dtype(target):
            return "classification"

        if pd.api.types.is_object_dtype(target):
            return "classification"

        if isinstance(target.dtype, pd.CategoricalDtype):
            return "classification"

        if pd.api.types.is_numeric_dtype(target):
            unique_values = target.nunique(dropna=True)
            total_values = target.notna().sum()

            if total_values == 0:
                raise ValueError("Target contains no valid values.")

            unique_ratio = unique_values / total_values

            # A small number of repeated integer values is likely
            # classification. High-cardinality numeric targets are
            # treated as regression.
            if (
                pd.api.types.is_integer_dtype(target)
                and unique_values <= 10
                and unique_ratio <= 0.20
            ):
                return "classification"

            return "regression"

        return "classification"