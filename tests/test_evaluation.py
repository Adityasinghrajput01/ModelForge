import numpy as np
import pandas as pd
import pytest

from modelforge.evaluation import (
    EvaluationEngine,
)


def test_regression_metrics():
    engine = EvaluationEngine()

    y_true = np.array(
        [10, 20, 30, 40, 50]
    )

    predictions = np.array(
        [11, 19, 31, 39, 49]
    )

    metrics = engine.evaluate_regression(
        y_true,
        predictions,
        feature_count=2,
    )

    assert "r2" in metrics
    assert "adjusted_r2" in metrics
    assert "mae" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics


def test_regression_rmse():
    engine = EvaluationEngine()

    y_true = np.array(
        [1, 2, 3]
    )

    predictions = np.array(
        [1, 3, 5]
    )

    metrics = engine.evaluate_regression(
        y_true,
        predictions,
    )

    expected_rmse = np.sqrt(
        (0**2 + 1**2 + 2**2) / 3
    )

    assert np.isclose(
        metrics["rmse"],
        expected_rmse,
    )


def test_adjusted_r2():
    engine = EvaluationEngine()

    adjusted = engine.adjusted_r2(
        r2=0.8,
        sample_count=100,
        feature_count=5,
    )

    expected = (
        1
        - (
            (1 - 0.8)
            * (
                99
                / 94
            )
        )
    )

    assert np.isclose(
        adjusted,
        expected,
    )


def test_adjusted_r2_invalid_dimensions():
    engine = EvaluationEngine()

    result = engine.adjusted_r2(
        r2=0.8,
        sample_count=5,
        feature_count=5,
    )

    assert np.isnan(result)


def test_classification_metrics():
    engine = EvaluationEngine()

    y_true = np.array(
        [0, 0, 1, 1]
    )

    predictions = np.array(
        [0, 1, 1, 1]
    )

    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )

    metrics = (
        engine.evaluate_classification(
            y_true,
            predictions,
            probabilities=probabilities,
        )
    )

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "log_loss" in metrics


def test_classification_roc_auc():
    engine = EvaluationEngine()

    y_true = np.array(
        [0, 0, 1, 1]
    )

    predictions = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )

    metrics = (
        engine.evaluate_classification(
            y_true,
            predictions,
            probabilities=probabilities,
        )
    )

    assert np.isclose(
        metrics["roc_auc"],
        1.0,
    )


def test_classification_log_loss():
    engine = EvaluationEngine()

    y_true = np.array(
        [0, 1]
    )

    predictions = np.array(
        [0, 1]
    )

    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ]
    )

    metrics = (
        engine.evaluate_classification(
            y_true,
            predictions,
            probabilities=probabilities,
        )
    )

    assert metrics["log_loss"] is not None
    assert metrics["log_loss"] >= 0


def test_classification_without_probabilities():
    engine = EvaluationEngine()

    y_true = np.array(
        [0, 0, 1, 1]
    )

    predictions = np.array(
        [0, 0, 1, 1]
    )

    metrics = (
        engine.evaluate_classification(
            y_true,
            predictions,
        )
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] is None
    assert metrics["log_loss"] is None


def test_metric_direction_maximize():
    engine = EvaluationEngine()

    assert (
        engine.metric_direction("r2")
        == "maximize"
    )

    assert (
        engine.metric_direction("f1")
        == "maximize"
    )


def test_metric_direction_minimize():
    engine = EvaluationEngine()

    assert (
        engine.metric_direction("mae")
        == "minimize"
    )

    assert (
        engine.metric_direction("rmse")
        == "minimize"
    )

    assert (
        engine.metric_direction("log_loss")
        == "minimize"
    )


def test_unknown_metric_direction():
    engine = EvaluationEngine()

    with pytest.raises(KeyError):
        engine.metric_direction(
            "unknown_metric"
        )


def test_available_regression_metrics():
    engine = EvaluationEngine()

    metrics = engine.available_metrics(
        "regression"
    )

    assert "r2" in metrics
    assert "adjusted_r2" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics


def test_available_classification_metrics():
    engine = EvaluationEngine()

    metrics = engine.available_metrics(
        "classification"
    )

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "log_loss" in metrics


def test_invalid_task_type_for_available_metrics():
    engine = EvaluationEngine()

    with pytest.raises(ValueError):
        engine.available_metrics(
            "clustering"
        )


def test_mismatched_regression_lengths():
    engine = EvaluationEngine()

    with pytest.raises(ValueError):
        engine.evaluate_regression(
            [1, 2, 3],
            [1, 2],
        )


def test_empty_regression_inputs():
    engine = EvaluationEngine()

    with pytest.raises(ValueError):
        engine.evaluate_regression(
            [],
            [],
        )


def test_mismatched_classification_lengths():
    engine = EvaluationEngine()

    with pytest.raises(ValueError):
        engine.evaluate_classification(
            [0, 1, 1],
            [0, 1],
        )


def test_empty_classification_inputs():
    engine = EvaluationEngine()

    with pytest.raises(ValueError):
        engine.evaluate_classification(
            [],
            [],
        )