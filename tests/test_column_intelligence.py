import pandas as pd
import pytest

from modelforge.column_intelligence import ColumnIntelligence


def test_column_types():

    data = pd.DataFrame(
        {
            "age": [20, 25, 30, 35],
            "city": ["Delhi", "Mumbai", "Delhi", "Pune"],
            "is_active": [True, False, True, True],
        }
    )

    analyzer = ColumnIntelligence()
    result = analyzer.analyze(data)

    assert result["columns"]["age"]["inferred_type"] == "numerical"

    assert result["columns"]["city"]["inferred_type"] == "categorical"

    assert (
        result["columns"]["is_active"]["inferred_type"]
        == "boolean"
    )


def test_id_detection():

    data = pd.DataFrame(
        {
            "customer_id": [1001, 1002, 1003, 1004],
            "age": [20, 25, 30, 35],
        }
    )

    analyzer = ColumnIntelligence()
    result = analyzer.analyze(data)

    assert (
        result["columns"]["customer_id"]["inferred_type"]
        == "id"
    )

    assert (
        result["columns"]["customer_id"]["is_id_like"]
        is True
    )


def test_constant_column():

    data = pd.DataFrame(
        {
            "country": ["India", "India", "India", "India"],
            "age": [20, 25, 30, 35],
        }
    )

    analyzer = ColumnIntelligence()
    result = analyzer.analyze(data)

    assert (
        result["columns"]["country"]["inferred_type"]
        == "constant"
    )


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

    analyzer = ColumnIntelligence()
    result = analyzer.analyze(data)

    assert (
        result["columns"]["email"]["is_high_cardinality"]
        is True
    )


def test_text_detection():

    data = pd.DataFrame(
        {
            "description": [
                "This is a very long product description "
                "that contains enough text to be considered "
                "a free form textual feature.",
                "Another long description containing "
                "multiple words and useful information "
                "about the product.",
            ]
        }
    )

    analyzer = ColumnIntelligence()
    result = analyzer.analyze(data)

    assert (
        result["columns"]["description"]["is_text_like"]
        is True
    )

    assert (
        result["columns"]["description"]["inferred_type"]
        == "text"
    )


def test_empty_dataset():

    data = pd.DataFrame()

    analyzer = ColumnIntelligence()

    with pytest.raises(ValueError):
        analyzer.analyze(data)