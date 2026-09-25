import pandas as pd
import pytest

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.pipeline import Pipeline

from modelforge.persistence import (
    ModelPersistence,
)
from modelforge.preprocessing import (
    PreprocessingEngine,
)


@pytest.fixture
def regression_data():
    return pd.DataFrame(
        {
            "age": [
                20,
                25,
                30,
                35,
                40,
                45,
            ],
            "income": [
                20000,
                25000,
                30000,
                35000,
                40000,
                45000,
            ],
            "target": [
                200,
                250,
                300,
                350,
                400,
                450,
            ],
        }
    )


@pytest.fixture
def classification_data():
    return pd.DataFrame(
        {
            "age": [
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                55,
            ],
            "income": [
                20000,
                25000,
                30000,
                35000,
                40000,
                45000,
                50000,
                55000,
            ],
            "target": [
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
            ],
        }
    )


@pytest.fixture
def fitted_pipeline(
    regression_data,
):
    preprocessing = (
        PreprocessingEngine().build(
            data=regression_data,
            target="target",
            model_type="linear_regression",
        )
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    X = regression_data.drop(
        columns=["target"]
    )

    y = regression_data["target"]

    pipeline.fit(
        X,
        y,
    )

    return pipeline


@pytest.fixture
def fitted_classification_pipeline(
    classification_data,
):
    preprocessing = (
        PreprocessingEngine().build(
            data=classification_data,
            target="target",
            model_type="logistic_regression",
        )
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )

    X = classification_data.drop(
        columns=["target"]
    )

    y = classification_data["target"]

    pipeline.fit(
        X,
        y,
    )

    return pipeline


def test_save_model(
    fitted_pipeline,
    tmp_path,
):
    persistence = ModelPersistence()

    model_path = (
        tmp_path / "model.joblib"
    )

    saved_path = persistence.save(
        pipeline=fitted_pipeline,
        path=str(model_path),
    )

    assert model_path.exists()
    assert saved_path == str(
        model_path.resolve()
    )


def test_load_model(
    fitted_pipeline,
    tmp_path,
):
    persistence = ModelPersistence()

    model_path = (
        tmp_path / "model.joblib"
    )

    persistence.save(
        pipeline=fitted_pipeline,
        path=str(model_path),
    )

    loaded = persistence.load(
        str(model_path)
    )

    assert isinstance(
        loaded,
        Pipeline,
    )


def test_loaded_model_predictions_match(
    fitted_pipeline,
    regression_data,
    tmp_path,
):
    persistence = ModelPersistence()

    model_path = (
        tmp_path / "model.joblib"
    )

    persistence.save(
        fitted_pipeline,
        str(model_path),
    )

    loaded = persistence.load(
        str(model_path)
    )

    X = regression_data.drop(
        columns=["target"]
    )

    original_predictions = (
        fitted_pipeline.predict(X)
    )

    loaded_predictions = (
        loaded.predict(X)
    )

    assert (
        original_predictions.tolist()
        == loaded_predictions.tolist()
    )


def test_save_metadata(
    fitted_pipeline,
    tmp_path,
):
    persistence = ModelPersistence()

    model_path = (
        tmp_path / "model.joblib"
    )

    metadata = {
        "target": "target",
        "task_type": "regression",
        "model": "linear_regression",
        "version": "0.1.0",
    }

    persistence.save(
        fitted_pipeline,
        str(model_path),
        metadata=metadata,
    )

    loaded_metadata = (
        persistence.load_metadata(
            str(model_path)
        )
    )

    assert (
        loaded_metadata == metadata
    )


def test_missing_metadata_returns_empty_dict(
    fitted_pipeline,
    tmp_path,
):
    persistence = ModelPersistence()

    model_path = (
        tmp_path / "model.joblib"
    )

    persistence.save(
        fitted_pipeline,
        str(model_path),
    )

    metadata = (
        persistence.load_metadata(
            str(model_path)
        )
    )

    assert metadata == {}


def test_predict(
    fitted_pipeline,
    regression_data,
):
    persistence = ModelPersistence()

    X = regression_data.drop(
        columns=["target"]
    )

    predictions = (
        persistence.predict(
            fitted_pipeline,
            X,
        )
    )

    assert isinstance(
        predictions,
        pd.Series,
    )

    assert len(predictions) == len(X)

    assert predictions.name == "prediction"


def test_predict_preserves_index(
    fitted_pipeline,
    regression_data,
):
    persistence = ModelPersistence()

    X = regression_data.drop(
        columns=["target"]
    ).copy()

    X.index = [
        100,
        101,
        102,
        103,
        104,
        105,
    ]

    predictions = (
        persistence.predict(
            fitted_pipeline,
            X,
        )
    )

    assert predictions.index.tolist() == [
        100,
        101,
        102,
        103,
        104,
        105,
    ]


def test_predict_from_file(
    fitted_pipeline,
    regression_data,
    tmp_path,
):
    persistence = ModelPersistence()

    data_path = (
        tmp_path / "data.csv"
    )

    X = regression_data.drop(
        columns=["target"]
    )

    X.to_csv(
        data_path,
        index=False,
    )

    predictions = (
        persistence.predict_from_file(
            fitted_pipeline,
            str(data_path),
        )
    )

    assert isinstance(
        predictions,
        pd.Series,
    )

    assert len(predictions) == len(X)


def test_predict_proba_unsupported_for_regression(
    fitted_pipeline,
    regression_data,
):
    persistence = ModelPersistence()

    X = regression_data.drop(
        columns=["target"]
    )

    with pytest.raises(ValueError):
        persistence.predict_proba(
            fitted_pipeline,
            X,
        )


def test_invalid_model_path(
    tmp_path,
):
    persistence = ModelPersistence()

    missing_path = (
        tmp_path / "missing.joblib"
    )

    with pytest.raises(FileNotFoundError):
        persistence.load(
            str(missing_path)
        )


def test_save_existing_model_without_overwrite(
    fitted_pipeline,
    tmp_path,
):
    persistence = ModelPersistence()

    model_path = (
        tmp_path / "model.joblib"
    )

    persistence.save(
        fitted_pipeline,
        str(model_path),
    )

    with pytest.raises(
        FileExistsError
    ):
        persistence.save(
            fitted_pipeline,
            str(model_path),
        )


def test_save_existing_model_with_overwrite(
    fitted_pipeline,
    tmp_path,
):
    persistence = ModelPersistence()

    model_path = (
        tmp_path / "model.joblib"
    )

    persistence.save(
        fitted_pipeline,
        str(model_path),
    )

    saved_path = persistence.save(
        fitted_pipeline,
        str(model_path),
        overwrite=True,
    )

    assert model_path.exists()
    assert saved_path == str(
        model_path.resolve()
    )


def test_invalid_pipeline_save(
    tmp_path,
):
    persistence = ModelPersistence()

    model_path = (
        tmp_path / "model.joblib"
    )

    with pytest.raises(TypeError):
        persistence.save(
            pipeline="invalid",
            path=str(model_path),
        )


def test_invalid_prediction_data(
    fitted_pipeline,
):
    persistence = ModelPersistence()

    with pytest.raises(TypeError):
        persistence.predict(
            fitted_pipeline,
            "invalid",
        )


def test_empty_prediction_data(
    fitted_pipeline,
):
    persistence = ModelPersistence()

    with pytest.raises(ValueError):
        persistence.predict(
            fitted_pipeline,
            pd.DataFrame(),
        )


def test_invalid_prediction_file(
    fitted_pipeline,
    tmp_path,
):
    persistence = ModelPersistence()

    data_path = (
        tmp_path / "data.txt"
    )

    data_path.write_text(
        "invalid"
    )

    with pytest.raises(ValueError):
        persistence.predict_from_file(
            fitted_pipeline,
            str(data_path),
        )


def test_missing_prediction_file(
    fitted_pipeline,
    tmp_path,
):
    persistence = ModelPersistence()

    data_path = (
        tmp_path / "missing.csv"
    )

    with pytest.raises(FileNotFoundError):
        persistence.predict_from_file(
            fitted_pipeline,
            str(data_path),
        )


def test_predict_proba_invalid_data(
    fitted_pipeline,
):
    persistence = ModelPersistence()

    with pytest.raises(ValueError):
        persistence.predict_proba(
            fitted_pipeline,
            pd.DataFrame(),
        )


# ---------------------------------------------------------------------------
# Prediction schema validation integration tests
# ---------------------------------------------------------------------------


def test_predict_reorders_columns(
    fitted_pipeline,
    regression_data,
):
    persistence = ModelPersistence()

    X = regression_data.drop(
        columns=["target"]
    )

    reordered = X[
        [
            "income",
            "age",
        ]
    ]

    predictions = (
        persistence.predict(
            fitted_pipeline,
            reordered,
        )
    )

    expected = fitted_pipeline.predict(
        X
    )

    assert predictions.tolist() == (
        expected.tolist()
    )


def test_predict_missing_column(
    fitted_pipeline,
    regression_data,
):
    persistence = ModelPersistence()

    invalid_data = pd.DataFrame(
        {
            "age": [
                50,
                55,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        persistence.predict(
            fitted_pipeline,
            invalid_data,
        )


def test_predict_unexpected_column(
    fitted_pipeline,
    regression_data,
):
    persistence = ModelPersistence()

    X = regression_data.drop(
        columns=["target"]
    ).copy()

    X["unexpected"] = [
        1,
        2,
        3,
        4,
        5,
        6,
    ]

    with pytest.raises(
        ValueError,
        match="Unexpected columns",
    ):
        persistence.predict(
            fitted_pipeline,
            X,
        )


def test_predict_proba_reorders_columns(
    fitted_classification_pipeline,
    classification_data,
):
    persistence = ModelPersistence()

    X = classification_data.drop(
        columns=["target"]
    )

    reordered = X[
        [
            "income",
            "age",
        ]
    ]

    probabilities = (
        persistence.predict_proba(
            fitted_classification_pipeline,
            reordered,
        )
    )

    expected = (
        fitted_classification_pipeline.predict_proba(
            X
        )
    )

    assert probabilities.shape == (
        expected.shape
    )

    assert len(probabilities) == len(
        X
    )


def test_predict_proba_missing_column(
    fitted_classification_pipeline,
):
    persistence = ModelPersistence()

    invalid_data = pd.DataFrame(
        {
            "age": [
                50,
                55,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        persistence.predict_proba(
            fitted_classification_pipeline,
            invalid_data,
        )


def test_predict_proba_unexpected_column(
    fitted_classification_pipeline,
    classification_data,
):
    persistence = ModelPersistence()

    X = classification_data.drop(
        columns=["target"]
    ).copy()

    X["unexpected"] = [
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
    ]

    with pytest.raises(
        ValueError,
        match="Unexpected columns",
    ):
        persistence.predict_proba(
            fitted_classification_pipeline,
            X,
        )


def test_prediction_schema_matches_training_columns(
    fitted_pipeline,
):
    expected_columns = [
        "age",
        "income",
    ]

    assert list(
        fitted_pipeline.feature_names_in_
    ) == expected_columns