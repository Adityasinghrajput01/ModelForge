import pandas as pd
import pytest

from modelforge.feature_selection import (
    VarianceFeatureSelector,
    CorrelationFeatureSelector,
    SelectKBestFeatureSelector,
    FeatureSelectionEngine,
)


def test_variance_selector_removes_constant_column():
    data = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "constant": [1, 1, 1, 1],
        }
    )

    selector = VarianceFeatureSelector()

    result = selector.fit_transform(data)

    assert "age" in result.columns
    assert "constant" not in result.columns


def test_variance_selector_keeps_variable_columns():
    data = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [10, 20, 30, 40],
        }
    )

    selector = VarianceFeatureSelector()

    result = selector.fit_transform(data)

    assert list(result.columns) == ["a", "b"]


def test_correlation_selector_removes_redundant_feature():
    data = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24],
            "age_copy": [40, 42, 44, 46, 48],
            "salary": [31, 47, 36, 58, 42],
        }
    )

    selector = CorrelationFeatureSelector(
        threshold=0.95
    )

    result = selector.fit_transform(data)

    assert "age" in result.columns
    assert "age_copy" not in result.columns
    assert "salary" in result.columns


def test_correlation_selector_keeps_categorical_columns():
    data = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "city": [
                "Delhi",
                "Mumbai",
                "Delhi",
                "Pune",
            ],
        }
    )

    selector = CorrelationFeatureSelector(
        threshold=0.95
    )

    result = selector.fit_transform(data)

    assert "age" in result.columns
    assert "city" in result.columns


def test_select_k_best_regression():
    data = pd.DataFrame(
        {
            "strong_feature": [1, 2, 3, 4, 5],
            "weak_feature": [1, 4, 2, 5, 3],
            "another_feature": [8, 3, 9, 2, 7],
        }
    )

    target = pd.Series(
        [10, 20, 30, 40, 50]
    )

    selector = SelectKBestFeatureSelector(
        k=1,
        task_type="regression",
    )

    result = selector.fit_transform(
        data,
        target,
    )

    assert result.shape[1] == 1
    assert "strong_feature" in result.columns


def test_select_k_best_classification():
    data = pd.DataFrame(
        {
            "strong_feature": [0, 0, 0, 1, 1, 1],
            "weak_feature": [1, 2, 3, 2, 3, 1],
        }
    )

    target = pd.Series(
        [0, 0, 0, 1, 1, 1]
    )

    selector = SelectKBestFeatureSelector(
        k=1,
        task_type="classification",
    )

    result = selector.fit_transform(
        data,
        target,
    )

    assert result.shape[1] == 1
    assert "strong_feature" in result.columns


def test_select_k_best_caps_k_to_feature_count():
    data = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [4, 3, 2, 1],
        }
    )

    target = pd.Series(
        [1, 2, 3, 4]
    )

    selector = SelectKBestFeatureSelector(
        k=10,
        task_type="regression",
    )

    result = selector.fit_transform(
        data,
        target,
    )

    assert result.shape[1] == 2


def test_invalid_correlation_threshold():
    with pytest.raises(ValueError):
        CorrelationFeatureSelector(
            threshold=1.5
        ).fit(
            pd.DataFrame(
                {"a": [1, 2, 3]}
            )
        )


def test_invalid_task_type():
    data = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [3, 2, 1],
        }
    )

    target = pd.Series(
        [1, 2, 3]
    )

    with pytest.raises(ValueError):
        SelectKBestFeatureSelector(
            k=1,
            task_type="unknown",
        ).fit(
            data,
            target,
        )


def test_feature_selection_engine():
    data = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [2, 4, 6, 8],
            "constant": [1, 1, 1, 1],
        }
    )

    engine = FeatureSelectionEngine()

    variance_result = engine.variance_filter(
        data
    )

    assert "constant" not in variance_result.columns

    correlation_result = engine.correlation_filter(
        data,
        threshold=0.95,
    )

    assert "b" not in correlation_result.columns