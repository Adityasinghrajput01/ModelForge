from __future__ import annotations

import json

import pandas as pd

from modelforge.automl import AutoML


def regression_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [
                1, 2, 3, 4, 5,
                6, 7, 8, 9, 10,
            ],
            "x2": [
                10, 9, 8, 7, 6,
                5, 4, 3, 2, 1,
            ],
            "target": [
                3, 5, 7, 9, 11,
                13, 15, 17, 19, 21,
            ],
        }
    )


def test_automl_result_contains_reproducibility(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        random_state=42,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    result = automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    assert "reproducibility" in result

    reproducibility = result[
        "reproducibility"
    ]

    assert "dataset" in reproducibility
    assert "configuration" in reproducibility
    assert "environment" in reproducibility

    assert (
        reproducibility["target"]
        == "target"
    )

    assert (
        reproducibility["task_type"]
        == "regression"
    )


def test_automl_reproducibility_contains_dataset_fingerprint(
    tmp_path,
):
    data = regression_data()

    automl = AutoML(
        cv=2,
        random_state=42,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    result = automl.fit(
        data,
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    dataset = result[
        "reproducibility"
    ]["dataset"]

    assert len(
        dataset["fingerprint"]
    ) == 64

    assert dataset["rows"] == len(data)
    assert dataset["columns"] == len(data.columns)

    assert dataset["column_names"] == [
        "x1",
        "x2",
        "target",
    ]


def test_automl_reproducibility_contains_configuration(
    tmp_path,
):
    automl = AutoML(
        cv=3,
        random_state=123,
        objective="performance",
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    result = automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    configuration = result[
        "reproducibility"
    ]["configuration"]

    assert len(
        configuration["fingerprint"]
    ) == 64

    values = configuration["values"]

    assert values["cv"] == 3
    assert values["random_state"] == 123
    assert values["objective"] == "performance"


def test_automl_reproducibility_contains_environment(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        random_state=42,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    result = automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    environment = result[
        "reproducibility"
    ]["environment"]

    assert "python_version" in environment
    assert "numpy_version" in environment
    assert "pandas_version" in environment
    assert "scikit_learn_version" in environment

    assert (
        environment["random_state"]
        == 42
    )


def test_automl_reproducibility_contains_run_id(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    result = automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    reproducibility = result[
        "reproducibility"
    ]

    assert (
        reproducibility["extra_metadata"][
            "run_id"
        ]
        == result["run_id"]
    )


def test_automl_experiment_persists_reproducibility(
    tmp_path,
):
    experiment_directory = (
        tmp_path / "experiments"
    )

    automl = AutoML(
        cv=2,
        experiment_directory=(
            experiment_directory
        ),
    )

    result = automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    experiment = automl.get_experiment(
        result["experiment_id"]
    )

    assert (
        "reproducibility"
        in experiment
    )

    assert (
        experiment["reproducibility"]
        == result["reproducibility"]
    )


def test_automl_experiment_file_contains_reproducibility(
    tmp_path,
):
    experiment_directory = (
        tmp_path / "experiments"
    )

    automl = AutoML(
        cv=2,
        experiment_directory=(
            experiment_directory
        ),
    )

    result = automl.fit(
        regression_data(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    path = (
        experiment_directory
        / f"{result['experiment_id']}.json"
    )

    assert path.exists()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved = json.load(file)

    assert (
        "reproducibility"
        in saved
    )

    assert (
        saved["reproducibility"]
        == result["reproducibility"]
    )


def test_automl_reproducibility_is_deterministic_for_same_data(
    tmp_path,
):
    data = regression_data()

    first = AutoML(
        cv=2,
        random_state=42,
        experiment_directory=(
            tmp_path / "first"
        ),
    )

    second = AutoML(
        cv=2,
        random_state=42,
        experiment_directory=(
            tmp_path / "second"
        ),
    )

    first_result = first.fit(
        data,
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    second_result = second.fit(
        data.copy(),
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    first_fingerprint = (
        first_result[
            "reproducibility"
        ]["dataset"]["fingerprint"]
    )

    second_fingerprint = (
        second_result[
            "reproducibility"
        ]["dataset"]["fingerprint"]
    )

    assert (
        first_fingerprint
        == second_fingerprint
    )


def test_automl_reproducibility_changes_when_data_changes(
    tmp_path,
):
    original = regression_data()

    modified = original.copy()

    modified.loc[
        0,
        "x1",
    ] = 999

    first = AutoML(
        cv=2,
        random_state=42,
        experiment_directory=(
            tmp_path / "first"
        ),
    )

    second = AutoML(
        cv=2,
        random_state=42,
        experiment_directory=(
            tmp_path / "second"
        ),
    )

    first_result = first.fit(
        original,
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    second_result = second.fit(
        modified,
        target="target",
        model_names=[
            "linear_regression",
        ],
    )

    first_fingerprint = (
        first_result[
            "reproducibility"
        ]["dataset"]["fingerprint"]
    )

    second_fingerprint = (
        second_result[
            "reproducibility"
        ]["dataset"]["fingerprint"]
    )

    assert (
        first_fingerprint
        != second_fingerprint
    )


def test_automl_failed_run_contains_reproducibility(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    data = regression_data()

    try:
        automl.fit(
            data,
            target="missing_target",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError."
        )

    assert automl.result is not None

    assert (
        automl.result["status"]
        == "failed"
    )

    assert (
        "reproducibility"
        in automl.result
    )

    reproducibility = (
        automl.result[
            "reproducibility"
        ]
    )

    assert "dataset" in reproducibility
    assert "configuration" in reproducibility
    assert "environment" in reproducibility


def test_automl_failed_experiment_persists_reproducibility(
    tmp_path,
):
    automl = AutoML(
        cv=2,
        experiment_directory=(
            tmp_path / "experiments"
        ),
    )

    try:
        automl.fit(
            regression_data(),
            target="missing_target",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError."
        )

    experiment = (
        automl.get_experiment(
            automl.experiment_id
        )
    )

    assert (
        experiment["status"]
        == "failed"
    )

    assert (
        "reproducibility"
        in experiment
    )

    assert (
        experiment["reproducibility"]
        is not None
    )