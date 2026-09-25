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