from typer.testing import CliRunner

from modelforge.cli import app


runner = CliRunner()


def test_cli_help():
    result = runner.invoke(
        app,
        ["--help"],
    )

    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "predict" in result.stdout
    assert "explain" in result.stdout
    assert "models" in result.stdout
    assert "experiments" in result.stdout


def test_models_command():
    result = runner.invoke(
        app,
        ["models"],
    )

    assert result.exit_code == 0
    assert (
        "ModelForge Model Registry"
        in result.stdout
    )
    assert (
        "linear_regression"
        in result.stdout
    )


def test_models_regression():
    result = runner.invoke(
        app,
        [
            "models",
            "--task-type",
            "regression",
        ],
    )

    assert result.exit_code == 0
    assert (
        "linear_regression"
        in result.stdout
    )
    assert (
        "logistic_regression"
        not in result.stdout
    )


def test_models_classification():
    result = runner.invoke(
        app,
        [
            "models",
            "--task-type",
            "classification",
        ],
    )

    assert result.exit_code == 0
    assert (
        "logistic_regression"
        in result.stdout
    )
    assert (
        "linear_regression"
        not in result.stdout
    )


def test_models_invalid_task():
    result = runner.invoke(
        app,
        [
            "models",
            "--task-type",
            "invalid",
        ],
    )

    assert result.exit_code == 1
    assert "regression" in result.stdout


def test_predict_missing_model():
    result = runner.invoke(
        app,
        [
            "predict",
            "--model",
            "missing_model.joblib",
            "--data",
            "missing.csv",
        ],
    )

    assert result.exit_code == 1
    assert "Model not found" in result.stdout


def test_explain_missing_model():
    result = runner.invoke(
        app,
        [
            "explain",
            "--model",
            "missing_model.joblib",
        ],
    )

    assert result.exit_code == 1
    assert "Model not found" in result.stdout


def test_train_missing_dataset():
    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            "missing.csv",
            "--target",
            "target",
        ],
    )

    assert result.exit_code == 1
    assert "Dataset not found" in result.stdout


def test_train_missing_required_arguments():
    result = runner.invoke(
        app,
        ["train"],
    )

    assert result.exit_code != 0


def test_predict_missing_required_arguments():
    result = runner.invoke(
        app,
        ["predict"],
    )

    assert result.exit_code != 0


def test_explain_missing_required_arguments():
    result = runner.invoke(
        app,
        ["explain"],
    )

    assert result.exit_code != 0


def test_train_config_missing_target(
    tmp_path,
):
    config_path = (
        tmp_path / "config.yaml"
    )

    config_path.write_text(
        "objective: balanced\n"
        "cv: 3\n"
    )

    data_path = (
        tmp_path / "data.csv"
    )

    data_path.write_text(
        "feature,target\n"
        "1,10\n"
        "2,20\n"
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(data_path),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 1
    assert "Target is required" in result.stdout


def test_train_config_missing_file(
    tmp_path,
):
    data_path = (
        tmp_path / "data.csv"
    )

    data_path.write_text(
        "feature,target\n"
        "1,10\n"
        "2,20\n"
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(data_path),
            "--config",
            str(
                tmp_path
                / "missing.yaml"
            ),
        ],
    )

    assert result.exit_code == 1
    assert "Configuration file not found" in result.stdout


def test_train_config_help():
    result = runner.invoke(
        app,
        [
            "train",
            "--help",
        ],
    )

    assert result.exit_code == 0
    assert "--config" in result.stdout


def test_train_cli_target_overrides_config(
    tmp_path,
):
    config_path = (
        tmp_path / "config.yaml"
    )

    config_path.write_text(
        "target: wrong_target\n"
    )

    data_path = (
        tmp_path / "data.csv"
    )

    data_path.write_text(
        "feature,target\n"
        "1,10\n"
        "2,20\n"
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(data_path),
            "--target",
            "target",
            "--config",
            str(config_path),
        ],
    )

    assert "Target: target" in result.stdout


def test_train_config_invalid_configuration(
    tmp_path,
):
    config_path = (
        tmp_path / "config.yaml"
    )

    config_path.write_text(
        "objective: invalid_objective\n"
    )

    data_path = (
        tmp_path / "data.csv"
    )

    data_path.write_text(
        "feature,target\n"
        "1,10\n"
        "2,20\n"
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(data_path),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 1
    assert "objective" in result.stdout


# ---------------------------------------------------------------------------
# Experiment CLI tests
# ---------------------------------------------------------------------------


def test_experiments_help():
    result = runner.invoke(
        app,
        [
            "experiments",
            "--help",
        ],
    )

    assert result.exit_code == 0
    assert "list" in result.stdout
    assert "get" in result.stdout


def test_experiments_list_empty(
    tmp_path,
):
    result = runner.invoke(
        app,
        [
            "experiments",
            "list",
            "--directory",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "ModelForge Experiments" in result.stdout
    assert "Total experiments: 0" in result.stdout


def test_experiments_list(
    tmp_path,
):
    from modelforge.experiment_tracker import (
        ExperimentTracker,
    )

    tracker = ExperimentTracker(
        directory=tmp_path,
    )

    experiment_id = tracker.record(
        result={
            "target": {
                "target": "price",
                "task_type": "regression",
            },
            "task_type": "regression",
            "best_model": "ridge",
            "run_id": "run_test_001",
        },
        configuration={
            "cv": 5,
            "objective": "balanced",
        },
    )

    result = runner.invoke(
        app,
        [
            "experiments",
            "list",
            "--directory",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "ModelForge Experiments" in result.stdout
    assert experiment_id in result.stdout
    assert "price" in result.stdout
    assert "regression" in result.stdout
    assert "ridge" in result.stdout
    assert "Total experiments: 1" in result.stdout


def test_experiments_list_extracts_nested_target(
    tmp_path,
):
    from modelforge.experiment_tracker import (
        ExperimentTracker,
    )

    tracker = ExperimentTracker(
        directory=tmp_path,
    )

    tracker.record(
        result={
            "target": {
                "target": "sales",
                "task_type": "regression",
                "dtype": "float64",
                "unique_values": 100,
                "missing_values": 0,
                "rows": 100,
            },
            "best_model": "linear_regression",
        },
    )

    result = runner.invoke(
        app,
        [
            "experiments",
            "list",
            "--directory",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "sales" in result.stdout
    assert "linear_regression" in result.stdout

    # The complete profiler dictionary should not
    # be rendered as the Target column.
    assert "{'target':" not in result.stdout
    assert "'unique_values'" not in result.stdout


def test_experiments_get(
    tmp_path,
):
    from modelforge.experiment_tracker import (
        ExperimentTracker,
    )

    tracker = ExperimentTracker(
        directory=tmp_path,
    )

    experiment_id = tracker.record(
        result={
            "target": {
                "target": "price",
                "task_type": "regression",
            },
            "task_type": "regression",
            "best_model": "random_forest_regressor",
            "run_id": "run_test_002",
        },
        configuration={
            "cv": 5,
            "objective": "performance",
        },
    )

    result = runner.invoke(
        app,
        [
            "experiments",
            "get",
            experiment_id,
            "--directory",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert experiment_id in result.stdout
    assert "Experiment Summary" in result.stdout
    assert "price" in result.stdout
    assert "regression" in result.stdout
    assert "random_forest_regressor" in result.stdout
    assert "run_test_002" in result.stdout
    assert "Configuration" in result.stdout


def test_experiments_get_missing(
    tmp_path,
):
    result = runner.invoke(
        app,
        [
            "experiments",
            "get",
            "missing_experiment",
            "--directory",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "Failed to get experiment" in result.stdout


def test_experiments_list_run_id_from_summary(
    tmp_path,
):
    from modelforge.experiment_tracker import (
        ExperimentTracker,
    )

    tracker = ExperimentTracker(
        directory=tmp_path,
    )

    tracker.record(
        result={
            "target": {
                "target": "target",
                "task_type": "classification",
            },
            "best_model": "logistic_regression",
            "run_summary": {
                "run_id": "run_nested_001",
                "status": "completed",
            },
        },
    )

    result = runner.invoke(
        app,
        [
            "experiments",
            "list",
            "--directory",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "classification" in result.stdout
    assert "logistic_regression" in result.stdout
    assert "run_nested_001" in result.stdout
    assert "completed" in result.stdout


def test_experiments_list_failed_status(
    tmp_path,
):
    from modelforge.experiment_tracker import (
        ExperimentTracker,
    )

    tracker = ExperimentTracker(
        directory=tmp_path,
    )

    tracker.record(
        result={
            "target": {
                "target": "target",
                "task_type": "regression",
            },
            "status": "failed",
            "run_id": "run_failed_001",
        },
    )

    result = runner.invoke(
        app,
        [
            "experiments",
            "list",
            "--directory",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "failed" in result.stdout
    assert "run_failed_001" in result.stdout