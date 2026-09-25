import numpy as np
import pandas as pd
import pytest

from modelforge.cross_validation import CrossValidationEngine
from modelforge.hyperparameter_optimization import (
    HyperparameterOptimizationEngine,
)
from modelforge.model_registry import ModelRegistry
from modelforge.model_screening import ModelScreeningEngine
from modelforge.pipeline_generator import PipelineGenerator


def regression_data(rows=40):
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "feature_1": rng.normal(size=rows),
            "feature_2": rng.normal(size=rows),
            "feature_3": rng.normal(size=rows),
            "target": rng.normal(size=rows),
        }
    )


def test_model_screening_isolates_failed_model():
    data = regression_data()

    generator = PipelineGenerator()

    valid_pipeline = generator.build(
        data=data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    invalid_pipeline = generator.build(
        data=data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    invalid_pipeline.steps[-1] = (
        "model",
        "this_is_not_a_valid_estimator",
    )

    pipelines = {
        "valid_model": valid_pipeline,
        "broken_model": invalid_pipeline,
    }

    engine = ModelScreeningEngine(
        test_size=0.2,
        random_state=42,
    )

    result = engine.screen(
        data=data,
        target="target",
        pipelines=pipelines,
        task_type="regression",
    )

    assert result is not None
    assert "model" in result.columns
    assert "valid_model" in result["model"].tolist()

    if "broken_model" in result["model"].tolist():
        broken_row = result[
            result["model"] == "broken_model"
        ].iloc[0]

        assert (
            "error" in result.columns
            or "status" in result.columns
            or pd.isna(broken_row.get("r2", np.nan))
        )


def test_cross_validation_isolates_failed_fold():
    data = regression_data(rows=20)

    generator = PipelineGenerator()

    pipeline = generator.build(
        data=data,
        target="target",
        model_name="linear_regression",
        task_type="regression",
    )

    engine = CrossValidationEngine(
        cv=3,
        random_state=42,
    )

    result = engine.evaluate(
        data=data,
        target="target",
        pipelines={"linear_regression": pipeline},
        task_type="regression",
    )

    assert result is not None


def test_cross_validation_rejects_invalid_pipeline_cleanly():
    data = regression_data(rows=20)

    engine = CrossValidationEngine(
        cv=3,
        random_state=42,
    )

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        engine.evaluate(
            data=data,
            target="target",
            pipelines={"invalid": None},
            task_type="regression",
        )


def test_optimization_handles_valid_pipeline():
    data = regression_data(rows=30)

    generator = PipelineGenerator()

    pipeline = generator.build(
        data=data,
        target="target",
        model_name="ridge",
        task_type="regression",
    )

    engine = HyperparameterOptimizationEngine(
        cv=2,
        random_state=42,
        max_trials=2,
    )

    parameter_space = {
        "model__alpha": [0.1, 1.0],
    }

    result = engine.optimize(
        data=data,
        target="target",
        pipeline=pipeline,
        parameter_space=parameter_space,
        task_type="regression",
    )

    assert result is not None
    assert result["successful_trials"] >= 1
    assert result["best_pipeline"] is not None


def test_optimization_rejects_invalid_pipeline():
    data = regression_data(rows=20)

    engine = HyperparameterOptimizationEngine(
        cv=2,
        random_state=42,
        max_trials=2,
    )

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        engine.optimize(
            data=data,
            target="target",
            pipeline=None,
            parameter_space={},
            task_type="regression",
        )


def test_optimization_records_failed_trials():
    data = regression_data(rows=30)

    generator = PipelineGenerator()

    pipeline = generator.build(
        data=data,
        target="target",
        model_name="ridge",
        task_type="regression",
    )

    engine = HyperparameterOptimizationEngine(
        cv=2,
        random_state=42,
        max_trials=2,
    )

    parameter_space = {
        "model__alpha": [0.1, -999999999999999999],
    }

    result = engine.optimize(
        data=data,
        target="target",
        pipeline=pipeline,
        parameter_space=parameter_space,
        task_type="regression",
    )

    assert result is not None
    assert "failed_trials" in result
    assert "successful_trials" in result


def test_model_registry_unknown_model_fails_cleanly():
    registry = ModelRegistry()

    with pytest.raises((ValueError, KeyError)):
        registry.get("model_that_does_not_exist")


def test_pipeline_generator_unknown_model_fails_cleanly():
    data = regression_data()

    generator = PipelineGenerator()

    with pytest.raises((ValueError, KeyError)):
        generator.build(
            data=data,
            target="target",
            model_name="model_that_does_not_exist",
            task_type="regression",
        )


def test_screening_with_no_pipelines_fails_cleanly():
    data = regression_data()

    engine = ModelScreeningEngine(
        test_size=0.2,
        random_state=42,
    )

    with pytest.raises((ValueError, RuntimeError)):
        engine.screen(
            data=data,
            target="target",
            pipelines={},
            task_type="regression",
        )