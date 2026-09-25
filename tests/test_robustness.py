import numpy as np
import pandas as pd
import pytest

from modelforge.column_intelligence import ColumnIntelligence
from modelforge.data_audit import DataQualityAuditor


def _assert_dict_result(result):
    assert isinstance(result, dict)


def test_column_intelligence_rejects_empty_dataframe_cleanly():
    data = pd.DataFrame()

    with pytest.raises(ValueError, match="data must not be empty"):
        ColumnIntelligence().analyze(data)


def test_column_intelligence_handles_all_null_columns():
    data = pd.DataFrame(
        {
            "name": [None, None, None],
            "value": [np.nan, np.nan, np.nan],
        }
    )

    result = ColumnIntelligence().analyze(data)

    _assert_dict_result(result)


def test_column_intelligence_handles_constant_columns():
    data = pd.DataFrame(
        {
            "constant_numeric": [1, 1, 1, 1],
            "constant_text": ["A", "A", "A", "A"],
            "varying": [1, 2, 3, 4],
        }
    )

    result = ColumnIntelligence().analyze(data)

    _assert_dict_result(result)


def test_column_intelligence_handles_mixed_dtypes():
    data = pd.DataFrame(
        {
            "integer": [1, 2, 3, 4],
            "float": [1.1, 2.2, 3.3, 4.4],
            "category": ["A", "B", "A", "C"],
            "boolean": [True, False, True, False],
            "date": pd.date_range("2026-01-01", periods=4),
        }
    )

    result = ColumnIntelligence().analyze(data)

    _assert_dict_result(result)


def test_data_auditor_rejects_empty_dataframe_cleanly():
    data = pd.DataFrame()

    with pytest.raises(ValueError, match="Cannot audit an empty dataset"):
        DataQualityAuditor().audit(data)


def test_data_auditor_handles_all_null_columns():
    data = pd.DataFrame(
        {
            "feature_a": [None, None, None, None],
            "feature_b": [np.nan, np.nan, np.nan, np.nan],
        }
    )

    result = DataQualityAuditor().audit(data)

    _assert_dict_result(result)


def test_data_auditor_handles_constant_target():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [10, 10, 10, 10, 10],
        }
    )

    result = DataQualityAuditor().audit(data, target="target")

    _assert_dict_result(result)


def test_data_auditor_handles_infinite_values():
    data = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, np.inf, 4.0],
            "feature_b": [1.0, -np.inf, 3.0, 4.0],
            "target": [10.0, 20.0, 30.0, 40.0],
        }
    )

    result = DataQualityAuditor().audit(data, target="target")

    _assert_dict_result(result)


def test_data_auditor_handles_duplicate_rows():
    data = pd.DataFrame(
        {
            "feature": [1, 1, 2, 2, 3],
            "target": [10, 10, 20, 20, 30],
        }
    )

    result = DataQualityAuditor().audit(data, target="target")

    _assert_dict_result(result)


def test_data_auditor_rejects_missing_target_cleanly():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [10, 20, 30],
        }
    )

    with pytest.raises((ValueError, KeyError)):
        DataQualityAuditor().audit(data, target="missing_target")
