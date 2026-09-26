import pandas as pd
import pytest

from modelforge.data_loader import DatasetLoader


def test_load_csv(tmp_path):
    """Test loading a CSV dataset."""

    dataset = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "salary": [30000, 45000, 60000],
        }
    )

    file_path = tmp_path / "sample.csv"
    dataset.to_csv(file_path, index=False)

    loader = DatasetLoader()
    result = loader.load(str(file_path))

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 2)
    assert list(result.columns) == ["age", "salary"]


def test_load_csv_preserves_na_like_text_and_detects_blank_cells(tmp_path):
    file_path = tmp_path / "categories.csv"
    file_path.write_text(
        "category,notes\nNA,available\nN/A,\nNULL,unknown\n",
        encoding="utf-8",
    )

    result = DatasetLoader().load(str(file_path))

    assert result["category"].tolist() == ["NA", "N/A", "NULL"]
    assert result["notes"].iloc[1] is pd.NA or pd.isna(
        result["notes"].iloc[1]
    )


def test_missing_file():
    """Test that a missing dataset raises an error."""

    loader = DatasetLoader()

    with pytest.raises(FileNotFoundError):
        loader.load("does_not_exist.csv")


def test_unsupported_format(tmp_path):
    """Test unsupported file formats."""

    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello")

    loader = DatasetLoader()

    with pytest.raises(ValueError):
        loader.load(str(file_path))
