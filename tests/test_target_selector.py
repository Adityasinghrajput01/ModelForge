import pandas as pd
import pytest

from modelforge.target_selector import TargetSelector


def test_regression_target():
    """Numeric target with many unique values should be regression."""

    data = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40],
            "salary": [25000, 35000, 45000, 55000, 65000],
        }
    )

    selector = TargetSelector()
    result = selector.select(data, "salary")

    assert result["target"] == "salary"
    assert result["task_type"] == "regression"
    assert result["missing_values"] == 0


def test_classification_target():
    """Categorical target should be classification."""

    data = pd.DataFrame(
        {
            "age": [20, 25, 30, 35],
            "purchased": ["yes", "no", "yes", "no"],
        }
    )

    selector = TargetSelector()
    result = selector.select(data, "purchased")

    assert result["target"] == "purchased"
    assert result["task_type"] == "classification"


def test_missing_target_column():
    """Invalid target column should raise an error."""

    data = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "salary": [30000, 40000, 50000],
        }
    )

    selector = TargetSelector()

    with pytest.raises(ValueError):
        selector.select(data, "price")


def test_empty_target():
    """Empty target name should raise an error."""

    data = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "salary": [30000, 40000, 50000],
        }
    )

    selector = TargetSelector()

    with pytest.raises(ValueError):
        selector.select(data, "")


def test_all_missing_target():
    """Target containing only missing values should raise an error."""

    data = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "salary": [None, None, None],
        }
    )

    selector = TargetSelector()

    with pytest.raises(ValueError):
        selector.select(data, "salary")