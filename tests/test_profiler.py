import pandas as pd
import pytest

from modelforge.profiler import DatasetProfiler


def test_profile_dataset():
    """Test basic dataset profiling."""

    data = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40],
            "salary": [30000, 40000, 50000, 60000, 70000],
            "city": ["Delhi", "Mumbai", "Delhi", "Pune", "Pune"],
        }
    )

    profiler = DatasetProfiler()
    result = profiler.profile(data)

    assert result["rows"] == 5
    assert result["columns"] == 3

    assert result["duplicate_rows"] == 0

    assert "age" in result["numeric_columns"]
    assert "salary" in result["numeric_columns"]

    assert "city" in result["categorical_columns"]


def test_missing_values():
    """Test missing-value detection."""

    data = pd.DataFrame(
        {
            "age": [20, None, 30, 40],
            "city": ["Delhi", "Mumbai", None, "Pune"],
        }
    )

    profiler = DatasetProfiler()
    result = profiler.profile(data)

    assert result["column_info"]["age"]["missing_values"] == 1
    assert result["column_info"]["city"]["missing_values"] == 1

    assert result["column_info"]["age"]["missing_percentage"] == 25.0


def test_duplicate_rows():
    """Test duplicate-row detection."""

    data = pd.DataFrame(
        {
            "age": [20, 25, 20],
            "salary": [30000, 40000, 30000],
        }
    )

    profiler = DatasetProfiler()
    result = profiler.profile(data)

    assert result["duplicate_rows"] == 1


def test_empty_dataset():
    """Test that empty datasets are rejected."""

    data = pd.DataFrame()

    profiler = DatasetProfiler()

    with pytest.raises(ValueError):
        profiler.profile(data)