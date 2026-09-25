from pathlib import Path

import pandas as pd


class DatasetLoader:
    """Load and validate datasets for ModelForge."""

    SUPPORTED_FORMATS = {
        ".csv",
        ".xlsx",
        ".xls",
        ".parquet",
        ".json",
    }

    def load(self, file_path: str) -> pd.DataFrame:
        """
        Load a dataset from a local file.

        Parameters
        ----------
        file_path : str
            Path to the dataset.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset does not exist.

        ValueError
            If the path is not a file or the format is unsupported.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_FORMATS:
            supported = ", ".join(sorted(self.SUPPORTED_FORMATS))

            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {supported}"
            )

        if extension == ".csv":
            return pd.read_csv(path)

        if extension in {".xlsx", ".xls"}:
            return pd.read_excel(path)

        if extension == ".parquet":
            return pd.read_parquet(path)

        if extension == ".json":
            return pd.read_json(path)

        raise ValueError(
            f"Unable to load dataset: {path}"
        )