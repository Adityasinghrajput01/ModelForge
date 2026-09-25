import pandas as pd
import pytest

from modelforge.column_intelligence import ColumnIntelligence
from modelforge.data_audit import DataQualityAuditor


def test_duplicate_rows():

    data = pd.DataFrame(
        {
            "age": [20, 25, 20],
            "salary": [30000, 40000, 30000],
        }
    )

    auditor = DataQualityAuditor()
    result = auditor.audit(data)

    assert result["duplicate_rows"] == 1
    assert result["issue_count"] >= 1


def test_constant_columns():

    data = pd.DataFrame(
        {
            "country": ["India", "India", "India", "India"],
            "age": [20, 25, 30, 35],
        }
    )

    auditor = DataQualityAuditor()
    result = auditor.audit(data)

    assert "country" in result["constant_columns"]


def test_high_missing_values():

    data = pd.DataFrame(
        {
            "age": [20, None, None, None],
            "salary": [30000, 40000, 50000, 60000],
        }
    )

    auditor = DataQualityAuditor()
    result = auditor.audit(data)

    assert "age" in result["high_missing_columns"]


def test_id_like_columns():

    data = pd.DataFrame(
        {
            "customer_id": [
                1001,
                1002,
                1003,
                1004,
                1005,
            ],
            "age": [20, 25, 30, 35, 40],
        }
    )

    intelligence = ColumnIntelligence().analyze(data)

    auditor = DataQualityAuditor()

    result = auditor.audit(
        data,
        column_intelligence=intelligence,
    )

    assert "customer_id" in result["id_like_columns"]


def test_high_cardinality():

    data = pd.DataFrame(
        {
            "email": [
                "a@example.com",
                "b@example.com",
                "c@example.com",
                "d@example.com",
                "e@example.com",
                "f@example.com",
                "g@example.com",
                "h@example.com",
                "i@example.com",
                "j@example.com",
            ]
        }
    )

    auditor = DataQualityAuditor()
    result = auditor.audit(data)

    assert "email" in result["high_cardinality_columns"]


def test_target_leakage():

    data = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40],
            "salary": [30000, 40000, 50000, 60000, 70000],
            "salary_copy": [30000, 40000, 50000, 60000, 70000],
        }
    )

    auditor = DataQualityAuditor()

    result = auditor.audit(
        data,
        target="salary",
    )

    assert "salary_copy" in result["target_leakage_columns"]

    assert result["has_critical_issues"] is True


def test_invalid_target():

    data = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "salary": [30000, 40000, 50000],
        }
    )

    auditor = DataQualityAuditor()

    with pytest.raises(ValueError):
        auditor.audit(
            data,
            target="price",
        )


def test_empty_dataset():

    data = pd.DataFrame()

    auditor = DataQualityAuditor()

    with pytest.raises(ValueError):
        auditor.audit(data)