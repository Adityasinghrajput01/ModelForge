import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from modelforge.preprocessing import PreprocessingEngine


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "age": [20, 25, None, 35],
            "salary": [30000, 40000, 50000, None],
            "city": ["Delhi", "Mumbai", "Delhi", None],
            "target": [1, 0, 1, 0],
        }
    )


def get_transformer(preprocessor, name):
    """Get a transformer by name from a ColumnTransformer."""

    for transformer_name, transformer, columns in (
        preprocessor.transformers
    ):
        if transformer_name == name:
            return transformer

    raise AssertionError(
        f"Transformer '{name}' was not found."
    )


def get_processed_columns(preprocessor):
    """Get all columns processed by the ColumnTransformer."""

    processed_columns = []

    for _, _, columns in preprocessor.transformers:
        processed_columns.extend(columns)

    return processed_columns


def test_build_preprocessor(sample_data):

    engine = PreprocessingEngine()

    preprocessor = engine.build(
        data=sample_data,
        target="target",
        model_type="tree",
    )

    assert isinstance(
        preprocessor,
        ColumnTransformer,
    )


def test_numerical_scaling(sample_data):

    engine = PreprocessingEngine()

    preprocessor = engine.build(
        data=sample_data,
        target="target",
        model_type="knn",
    )

    numeric_transformer = get_transformer(
        preprocessor,
        "numerical",
    )

    assert isinstance(
        numeric_transformer.named_steps["scaler"],
        StandardScaler,
    )


def test_tree_model_does_not_scale(sample_data):

    engine = PreprocessingEngine()

    preprocessor = engine.build(
        data=sample_data,
        target="target",
        model_type="random_forest",
    )

    numeric_transformer = get_transformer(
        preprocessor,
        "numerical",
    )

    assert "scaler" not in numeric_transformer.named_steps


def test_categorical_encoder(sample_data):

    engine = PreprocessingEngine()

    preprocessor = engine.build(
        data=sample_data,
        target="target",
        model_type="tree",
    )

    categorical_transformer = get_transformer(
        preprocessor,
        "categorical",
    )

    encoder = categorical_transformer.named_steps[
        "encoder"
    ]

    assert isinstance(
        encoder,
        OneHotEncoder,
    )

    assert encoder.handle_unknown == "ignore"


def test_target_is_not_processed(sample_data):

    engine = PreprocessingEngine()

    preprocessor = engine.build(
        data=sample_data,
        target="target",
    )

    all_columns = get_processed_columns(
        preprocessor
    )

    assert "target" not in all_columns


def test_excluded_columns(sample_data):

    engine = PreprocessingEngine()

    preprocessor = engine.build(
        data=sample_data,
        target="target",
        excluded_columns=["age"],
    )

    all_columns = get_processed_columns(
        preprocessor
    )

    assert "age" not in all_columns


def test_missing_target():

    data = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "salary": [30000, 40000, 50000],
        }
    )

    engine = PreprocessingEngine()

    with pytest.raises(ValueError):

        engine.build(
            data=data,
            target="price",
        )


def test_empty_dataset():

    data = pd.DataFrame()

    engine = PreprocessingEngine()

    with pytest.raises(ValueError):

        engine.build(
            data=data,
            target="target",
        )