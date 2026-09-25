import pandas as pd


class DatasetProfiler:
    """Profile a dataset and generate structural and statistical information."""

    def profile(self, data: pd.DataFrame) -> dict:
        """
        Generate a complete dataset profile.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset to analyze.

        Returns
        -------
        dict
            Dataset profiling information.

        Raises
        ------
        TypeError
            If data is not a pandas DataFrame.

        ValueError
            If the dataset is empty.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        if data.empty:
            raise ValueError("Cannot profile an empty dataset.")

        column_info = {}

        for column in data.columns:
            series = data[column]

            column_info[column] = {
                "dtype": str(series.dtype),
                "missing_values": int(series.isna().sum()),
                "missing_percentage": float(
                    series.isna().mean() * 100
                ),
                "unique_values": int(
                    series.nunique(dropna=True)
                ),
                "is_numeric": bool(
                    pd.api.types.is_numeric_dtype(series)
                ),
                "is_categorical": bool(
                    pd.api.types.is_object_dtype(series)
                    or pd.api.types.is_string_dtype(series)
                    or isinstance(
                        series.dtype,
                        pd.CategoricalDtype,
                    )
                    or pd.api.types.is_bool_dtype(series)
                ),
            }

        numeric_columns = data.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = data.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist()

        return {
            "rows": int(data.shape[0]),
            "columns": int(data.shape[1]),
            "column_names": data.columns.tolist(),
            "duplicate_rows": int(
                data.duplicated().sum()
            ),
            "memory_usage_bytes": int(
                data.memory_usage(deep=True).sum()
            ),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "column_info": column_info,
        }