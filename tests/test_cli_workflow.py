from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from modelforge.cli import app


runner = CliRunner()


def create_regression_dataset(path):
    data = pd.DataFrame(
        {
            "feature_1": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
            "feature_2": [
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
            ],
            "target": [
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
            ],
        }
    )

    data.to_csv(path, index=False)


def create_prediction_dataset(path):
    data = pd.DataFrame(
        {
            "feature_1": [11, 12, 13],
            "feature_2": [0, 1, 2],
        }
    )

    data.to_csv(path, index=False)


def test_cli_help():
    result = runner.invoke(
        app,
        ["--help"],
    )

    assert result.exit_code == 0
    assert "ModelForge" in result.stdout


def test_cli_train_predict_workflow(tmp_path):
    train_path = tmp_path / "train.csv"
    prediction_path = tmp_path / "prediction.csv"
    model_path = tmp_path / "model.joblib"
    experiment_dir = tmp_path / "experiments"

    create_regression_dataset(train_path)
    create_prediction_dataset(prediction_path)

    train_result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(train_path),
            "--target",
            "target",
            "--models",
            "linear_regression",
            "--cv",
            "2",
            "--experiment-directory",
            str(experiment_dir),
            "--save",
            str(model_path),
        ],
    )

    assert train_result.exit_code == 0
    assert model_path.exists()
    assert experiment_dir.exists()


def test_cli_predict_workflow(tmp_path):
    train_path = tmp_path / "train.csv"
    prediction_path = tmp_path / "prediction.csv"
    model_path = tmp_path / "model.joblib"
    output_path = tmp_path / "predictions.csv"
    experiment_dir = tmp_path / "experiments"

    create_regression_dataset(train_path)
    create_prediction_dataset(prediction_path)

    train_result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(train_path),
            "--target",
            "target",
            "--models",
            "linear_regression",
            "--cv",
            "2",
            "--experiment-directory",
            str(experiment_dir),
            "--save",
            str(model_path),
        ],
    )

    assert train_result.exit_code == 0

    predict_result = runner.invoke(
        app,
        [
            "predict",
            "--model",
            str(model_path),
            "--data",
            str(prediction_path),
            "--output",
            str(output_path),
        ],
    )

    assert predict_result.exit_code == 0
    assert output_path.exists()

    predictions = pd.read_csv(output_path)

    assert "prediction" in predictions.columns
    assert len(predictions) == 3


def test_cli_experiments_list_after_training(tmp_path):
    train_path = tmp_path / "train.csv"
    model_path = tmp_path / "model.joblib"
    experiment_dir = tmp_path / "experiments"

    create_regression_dataset(train_path)

    train_result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(train_path),
            "--target",
            "target",
            "--models",
            "linear_regression",
            "--cv",
            "2",
            "--experiment-directory",
            str(experiment_dir),
            "--save",
            str(model_path),
        ],
    )

    assert train_result.exit_code == 0

    list_result = runner.invoke(
        app,
        [
            "experiments",
            "list",
            "--directory",
            str(experiment_dir),
        ],
    )

    assert list_result.exit_code == 0
    assert "target" in list_result.stdout
    assert "linear_regression" in list_result.stdout
    assert "regression" in list_result.stdout


def test_cli_explain_workflow(tmp_path):
    train_path = tmp_path / "train.csv"
    model_path = tmp_path / "model.joblib"
    experiment_dir = tmp_path / "experiments"

    create_regression_dataset(train_path)

    train_result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(train_path),
            "--target",
            "target",
            "--models",
            "linear_regression",
            "--cv",
            "2",
            "--experiment-directory",
            str(experiment_dir),
            "--save",
            str(model_path),
        ],
    )

    assert train_result.exit_code == 0

    explain_result = runner.invoke(
        app,
        [
            "explain",
            "--model",
            str(model_path),
        ],
    )

    assert explain_result.exit_code == 0


def test_cli_predict_missing_feature_fails(tmp_path):
    train_path = tmp_path / "train.csv"
    prediction_path = tmp_path / "bad_prediction.csv"
    model_path = tmp_path / "model.joblib"
    experiment_dir = tmp_path / "experiments"

    create_regression_dataset(train_path)

    bad_data = pd.DataFrame(
        {
            "feature_1": [11, 12],
        }
    )

    bad_data.to_csv(
        prediction_path,
        index=False,
    )

    train_result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(train_path),
            "--target",
            "target",
            "--models",
            "linear_regression",
            "--cv",
            "2",
            "--experiment-directory",
            str(experiment_dir),
            "--save",
            str(model_path),
        ],
    )

    assert train_result.exit_code == 0

    predict_result = runner.invoke(
        app,
        [
            "predict",
            "--model",
            str(model_path),
            "--data",
            str(prediction_path),
        ],
    )

    assert predict_result.exit_code != 0


def test_cli_predict_extra_feature_fails(tmp_path):
    train_path = tmp_path / "train.csv"
    prediction_path = tmp_path / "bad_prediction.csv"
    model_path = tmp_path / "model.joblib"
    experiment_dir = tmp_path / "experiments"

    create_regression_dataset(train_path)

    bad_data = pd.DataFrame(
        {
            "feature_1": [11, 12],
            "feature_2": [0, 1],
            "unexpected": [100, 200],
        }
    )

    bad_data.to_csv(
        prediction_path,
        index=False,
    )

    train_result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(train_path),
            "--target",
            "target",
            "--models",
            "linear_regression",
            "--cv",
            "2",
            "--experiment-directory",
            str(experiment_dir),
            "--save",
            str(model_path),
        ],
    )

    assert train_result.exit_code == 0

    predict_result = runner.invoke(
        app,
        [
            "predict",
            "--model",
            str(model_path),
            "--data",
            str(prediction_path),
        ],
    )

    assert predict_result.exit_code != 0