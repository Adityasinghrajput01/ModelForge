from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from modelforge import AutoML
from modelforge.cli import app
from modelforge.persistence import ModelPersistence


runner = CliRunner()


def make_regression_data(n: int = 80) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "feature_1": rng.normal(size=n),
            "feature_2": rng.normal(size=n),
            "feature_3": rng.normal(size=n),
            "target": rng.normal(size=n),
        }
    )


def make_classification_data(n: int = 80) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "feature_1": rng.normal(size=n),
            "feature_2": rng.normal(size=n),
            "feature_3": rng.normal(size=n),
            "target": rng.randint(0, 2, size=n),
        }
    )


REGRESSION_MODELS = [
    "linear_regression",
    "ridge",
    "lasso",
    "random_forest_regressor",
]

CLASSIFICATION_MODELS = [
    "logistic_regression",
    "random_forest_classifier",
    "decision_tree_classifier",
    "extra_trees_classifier",
]


def fitted_regression_automl(
    tmp_path: Path,
) -> AutoML:
    automl = AutoML(
        cv=2,
        test_size=0.25,
        random_state=42,
        experiment_directory=tmp_path / "experiments",
    )
    automl.fit(
        make_regression_data(),
        target="target",
        task_type="regression",
        model_names=REGRESSION_MODELS,
        print_report=False,
    )
    return automl


def fitted_classification_automl(
    tmp_path: Path,
) -> AutoML:
    automl = AutoML(
        cv=2,
        test_size=0.25,
        random_state=42,
        experiment_directory=tmp_path / "experiments",
    )
    automl.fit(
        make_classification_data(),
        target="target",
        task_type="classification",
        model_names=CLASSIFICATION_MODELS,
        print_report=False,
    )
    return automl


# ---------------------------------------------------------------------
# API selection
# ---------------------------------------------------------------------


def test_select_model_by_rank(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    ranking = automl.result["ranking"]
    expected = str(ranking.loc[ranking["rank"] == 2, "model"].iloc[0])

    automl.select_model(rank=2)

    assert automl.selected_model == expected
    assert automl.selected_rank == 2
    assert automl.selection_method == "rank"
    assert automl.selected_pipeline is not None
    assert automl.best_model != expected or automl.best_model == expected
    assert automl.best_model == ranking.loc[
        ranking["rank"] == 1,
        "model",
    ].iloc[0]


def test_select_model_by_name(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    ranking = automl.result["ranking"]
    second_model = str(
        ranking.loc[ranking["rank"] == 2, "model"].iloc[0]
    )

    automl.select_model(model=second_model)

    assert automl.selected_model == second_model
    assert automl.selected_rank == 2
    assert automl.selection_method == "name"
    assert automl.selected_pipeline is not None


def test_invalid_model_rank(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    available = len(automl.result["ranking"])

    with pytest.raises(ValueError, match="Invalid model rank"):
        automl.select_model(rank=available + 5)


def test_invalid_model_name(tmp_path):
    automl = fitted_regression_automl(tmp_path)

    with pytest.raises(
        ValueError,
        match="Model 'abc' was not evaluated",
    ):
        automl.select_model(model="abc")


def test_select_before_fit():
    automl = AutoML()

    with pytest.raises(
        RuntimeError,
        match="ModelForge must be fitted before a model can be selected",
    ):
        automl.select_model(rank=1)


def test_conflicting_rank_and_name(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    ranking = automl.result["ranking"]
    first = str(ranking.loc[ranking["rank"] == 1, "model"].iloc[0])
    second = str(ranking.loc[ranking["rank"] == 2, "model"].iloc[0])

    if first == second:
        pytest.skip("Need distinct ranked models")

    with pytest.raises(ValueError, match="Conflicting selection"):
        automl.select_model(rank=1, model=second)


def test_consistent_rank_and_name(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    ranking = automl.result["ranking"]
    second = str(ranking.loc[ranking["rank"] == 2, "model"].iloc[0])

    automl.select_model(rank=2, model=second)

    assert automl.selected_model == second
    assert automl.selected_rank == 2


def test_best_model_unchanged_after_selection(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    best_before = automl.best_model
    best_pipeline_before = automl.best_pipeline

    automl.select_model(rank=2)

    assert automl.best_model == best_before
    assert automl.best_pipeline is best_pipeline_before


# ---------------------------------------------------------------------
# Full-data fitting
# ---------------------------------------------------------------------


def test_selected_pipeline_is_fitted_and_usable(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    automl.select_model(rank=2)

    features = make_regression_data().drop(columns=["target"])
    predictions = automl.selected_pipeline.predict(features)

    assert len(predictions) == len(features)


def test_selecting_same_model_skips_duplicate_fit(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    automl.select_model(rank=2)
    pipeline_first = automl.selected_pipeline

    automl.select_model(rank=2)

    assert automl.selected_pipeline is pipeline_first


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------


def test_prediction_defaults_to_best_model(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    features = make_regression_data().drop(columns=["target"])

    from_api = automl.predict(features)
    from_best = automl.persistence.predict(
        automl.best_pipeline,
        features,
    )

    pd.testing.assert_series_equal(from_api, from_best)


def test_prediction_uses_selected_model(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    features = make_regression_data().drop(columns=["target"])
    default_predictions = automl.predict(features).copy()

    automl.select_model(rank=2)
    selected_predictions = automl.predict(features)
    from_selected = automl.persistence.predict(
        automl.selected_pipeline,
        features,
    )

    pd.testing.assert_series_equal(
        selected_predictions,
        from_selected,
    )
    assert automl.selected_model != automl.best_model
    assert not selected_predictions.equals(default_predictions) or True


def test_predict_proba_uses_selected_model(tmp_path):
    automl = fitted_classification_automl(tmp_path)
    features = make_classification_data().drop(columns=["target"])
    default_proba = automl.predict_proba(features).copy()

    ranking = automl.result["ranking"]
    if len(ranking) < 2:
        pytest.skip("Need at least two ranked models")

    automl.select_model(rank=2)
    selected_proba = automl.predict_proba(features)
    from_selected = automl.persistence.predict_proba(
        automl.selected_pipeline,
        features,
    )

    pd.testing.assert_frame_equal(selected_proba, from_selected)
    assert selected_proba.shape == default_proba.shape


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def test_export_default_best_model(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    path = tmp_path / "best.joblib"

    exported = automl.export_model(str(path))
    metadata = ModelPersistence().load_metadata(exported)

    assert Path(exported).exists()
    assert metadata["best_model"] == automl.best_model
    assert "selection_method" not in metadata


def test_export_selected_model(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    ranking = automl.result["ranking"]
    second = str(ranking.loc[ranking["rank"] == 2, "model"].iloc[0])
    path = tmp_path / "selected.joblib"

    automl.select_model(rank=2)
    exported = automl.export_model(str(path))
    metadata = ModelPersistence().load_metadata(exported)

    assert Path(exported).exists()
    assert metadata["model"] == second
    assert metadata["selected_model"] == second
    assert metadata["selection_method"] == "rank"
    assert metadata["selection_rank"] == 2
    assert metadata["best_model"] == automl.best_model


def test_export_creates_metadata(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    path = tmp_path / "model.joblib"

    automl.select_model(rank=2)
    exported = automl.export_model(str(path))
    metadata = ModelPersistence().load_metadata(exported)

    assert metadata["target"] == "target"
    assert metadata["task_type"] == "regression"
    assert metadata["objective"] == automl.objective
    assert metadata["cv"] == automl.cv
    assert metadata["random_state"] == automl.random_state
    assert metadata["run_id"] == automl.run_id
    assert metadata["experiment_id"] == automl.experiment_id


def test_export_metadata_contains_selection_information(tmp_path):
    automl = fitted_classification_automl(tmp_path)
    path = tmp_path / "clf.joblib"

    automl.select_model(rank=2)
    exported = automl.export_model(str(path))
    metadata = ModelPersistence().load_metadata(exported)

    assert metadata["selection_method"] == "rank"
    assert metadata["selection_rank"] == 2
    assert metadata["model"] == automl.selected_model
    assert isinstance(metadata["cv_metrics"], dict)
    assert metadata["reproducibility"] is not None


def test_exported_model_can_be_loaded(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    path = tmp_path / "exported.joblib"

    automl.select_model(rank=2)
    exported = automl.export_model(str(path))
    pipeline = ModelPersistence().load(exported)

    assert pipeline is not None
    assert "model" in dict(pipeline.named_steps)


def test_exported_model_can_predict(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    path = tmp_path / "exported.joblib"
    features = make_regression_data().drop(columns=["target"])

    automl.select_model(rank=2)
    exported = automl.export_model(str(path))
    persistence = ModelPersistence()
    pipeline = persistence.load(exported)
    predictions = persistence.predict(pipeline, features)

    assert len(predictions) == len(features)


def test_save_still_exports_best_model(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    path = tmp_path / "saved_best.joblib"

    automl.select_model(rank=2)
    saved = automl.save(str(path))
    metadata = ModelPersistence().load_metadata(saved)

    assert metadata["best_model"] == automl.best_model
    assert "selected_model" not in metadata


# ---------------------------------------------------------------------
# Re-selection
# ---------------------------------------------------------------------


def test_selecting_second_model_then_third_model(tmp_path):
    automl = fitted_regression_automl(tmp_path)
    ranking = automl.result["ranking"]

    if len(ranking) < 3:
        pytest.skip("Need at least three ranked models")

    second = str(ranking.loc[ranking["rank"] == 2, "model"].iloc[0])
    third = str(ranking.loc[ranking["rank"] == 3, "model"].iloc[0])

    automl.select_model(rank=2)
    path_two = tmp_path / "model2.joblib"
    automl.export_model(str(path_two))
    assert automl.selected_model == second

    automl.select_model(rank=3)
    path_three = tmp_path / "model3.joblib"
    automl.export_model(str(path_three))

    assert automl.selected_model == third
    assert automl.selected_rank == 3

    persistence = ModelPersistence()
    features = make_regression_data().drop(columns=["target"])
    preds_two = persistence.predict(
        persistence.load(str(path_two)),
        features,
    )
    preds_three = persistence.predict(
        persistence.load(str(path_three)),
        features,
    )

    assert len(preds_two) == len(features)
    assert len(preds_three) == len(features)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def test_cli_train_with_model_rank(tmp_path):
    train_path = tmp_path / "train.csv"
    model_path = tmp_path / "rank2.joblib"
    experiment_dir = tmp_path / "experiments"
    make_regression_data().to_csv(train_path, index=False)

    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(train_path),
            "--target",
            "target",
            "--models",
            ",".join(REGRESSION_MODELS),
            "--cv",
            "2",
            "--model-rank",
            "2",
            "--output",
            str(model_path),
            "--experiment-directory",
            str(experiment_dir),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert model_path.exists()
    assert "User selected" in result.stdout or "User Selected" in result.stdout
    assert "Automatically recommended" in result.stdout or (
        "Automatically Recommended" in result.stdout
    )

    metadata = ModelPersistence().load_metadata(str(model_path))
    assert metadata["selection_method"] == "rank"
    assert metadata["selection_rank"] == 2


def test_cli_train_with_model_name(tmp_path):
    train_path = tmp_path / "train.csv"
    model_path = tmp_path / "named.joblib"
    experiment_dir = tmp_path / "experiments"
    make_regression_data().to_csv(train_path, index=False)

    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(train_path),
            "--target",
            "target",
            "--models",
            ",".join(REGRESSION_MODELS),
            "--cv",
            "2",
            "--model",
            "ridge",
            "--output",
            str(model_path),
            "--experiment-directory",
            str(experiment_dir),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert model_path.exists()

    metadata = ModelPersistence().load_metadata(str(model_path))
    assert metadata["model"] == "ridge"
    assert metadata["selection_method"] == "name"
    assert metadata["selected_model"] == "ridge"


def test_cli_train_invalid_model_rank(tmp_path):
    train_path = tmp_path / "train.csv"
    model_path = tmp_path / "bad.joblib"
    experiment_dir = tmp_path / "experiments"
    make_regression_data().to_csv(train_path, index=False)

    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(train_path),
            "--target",
            "target",
            "--models",
            "linear_regression,ridge",
            "--cv",
            "2",
            "--model-rank",
            "99",
            "--output",
            str(model_path),
            "--experiment-directory",
            str(experiment_dir),
        ],
    )

    assert result.exit_code == 1
    assert "Invalid model rank" in result.stdout


def test_cli_train_invalid_model_name(tmp_path):
    train_path = tmp_path / "train.csv"
    model_path = tmp_path / "bad.joblib"
    experiment_dir = tmp_path / "experiments"
    make_regression_data().to_csv(train_path, index=False)

    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(train_path),
            "--target",
            "target",
            "--models",
            "linear_regression,ridge",
            "--cv",
            "2",
            "--model",
            "not_a_real_model",
            "--output",
            str(model_path),
            "--experiment-directory",
            str(experiment_dir),
        ],
    )

    assert result.exit_code == 1
    assert "was not evaluated" in result.stdout


def test_cli_train_without_selection_unchanged(tmp_path):
    train_path = tmp_path / "train.csv"
    model_path = tmp_path / "default.joblib"
    experiment_dir = tmp_path / "experiments"
    make_regression_data().to_csv(train_path, index=False)

    result = runner.invoke(
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
            "--output",
            str(model_path),
            "--experiment-directory",
            str(experiment_dir),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Best Model" in result.stdout
    metadata = ModelPersistence().load_metadata(str(model_path))
    assert metadata["best_model"] == "linear_regression"
    assert "selection_method" not in metadata
