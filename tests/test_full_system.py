from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from modelforge import AutoML
from modelforge.column_intelligence import ColumnIntelligence
from modelforge.cross_validation import CrossValidationEngine
from modelforge.data_audit import DataQualityAuditor
from modelforge.experiment_tracker import ExperimentTracker
from modelforge.explainability import ExplainabilityEngine
from modelforge.feature_engineering import FeatureEngineeringEngine
from modelforge.feature_selection import FeatureSelectionEngine
from modelforge.model_registry import ModelRegistry
from modelforge.model_screening import ModelScreeningEngine
from modelforge.persistence import ModelPersistence
from modelforge.pipeline_generator import PipelineGenerator
from modelforge.prediction_validator import PredictionSchemaValidator
from modelforge.profiler import DatasetProfiler
from modelforge.ranking import RankingEngine
from modelforge.reproducibility import ReproducibilityManager
from modelforge.reproducibility_integration import ReproducibilityIntegration
from modelforge.run_manager import RunManager
from modelforge.target_selector import TargetSelector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_data() -> pd.DataFrame:
    X, y = make_regression(
        n_samples=160,
        n_features=5,
        n_informative=4,
        noise=10.0,
        random_state=42,
    )

    data = pd.DataFrame(
        X,
        columns=[
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
        ],
    )

    data["category"] = np.where(
        data["feature_1"] > data["feature_1"].median(),
        "A",
        "B",
    )

    data["target"] = y

    return data


@pytest.fixture
def classification_data() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=180,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )

    data = pd.DataFrame(
        X,
        columns=[
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
        ],
    )

    data["category"] = np.where(
        data["feature_1"] > data["feature_1"].median(),
        "A",
        "B",
    )

    data["target"] = y

    return data


# ---------------------------------------------------------------------------
# Core component tests
# ---------------------------------------------------------------------------


def test_dataset_profiler_works(regression_data):
    profiler = DatasetProfiler()

    result = profiler.profile(regression_data)

    assert isinstance(result, dict)
    assert result


def test_target_selector_works(regression_data):
    selector = TargetSelector()

    result = selector.select(
        regression_data,
        target="target",
    )

    assert result is not None


def test_column_intelligence_works(regression_data):
    engine = ColumnIntelligence()

    result = engine.analyze(regression_data)

    assert isinstance(result, dict)
    assert result


def test_data_quality_audit_works(regression_data):
    auditor = DataQualityAuditor()

    result = auditor.audit(
        regression_data,
        target="target",
    )

    assert isinstance(result, dict)
    assert result


def test_feature_engineering_works(regression_data):
    engine = FeatureEngineeringEngine()

    result = engine.fit_transform(
        regression_data.drop(columns=["target"]),
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(regression_data)


def test_feature_selection_works(regression_data):
    X = regression_data.drop(columns=["target"])
    y = regression_data["target"]

    engine = FeatureSelectionEngine()

    result = engine.select(
        X,
        y,
        task_type="regression",
    )

    assert result is not None


def test_model_registry_contains_models():
    registry = ModelRegistry()

    models = registry.list_models()

    assert isinstance(models, list)
    assert len(models) >= 10


def test_pipeline_generation_works(regression_data):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="random_forest_regressor",
        task_type="regression",
    )

    assert pipeline is not None
    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "predict")


def test_model_screening_works(regression_data):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="random_forest_regressor",
        task_type="regression",
    )

    engine = ModelScreeningEngine(
        test_size=0.2,
        random_state=42,
    )

    result = engine.screen(
        data=regression_data,
        target="target",
        pipelines={
            "random_forest_regressor": pipeline,
        },
        task_type="regression",
    )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_cross_validation_works(regression_data):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="ridge",
        task_type="regression",
    )

    engine = CrossValidationEngine(
        cv=3,
        random_state=42,
    )

    result = engine.evaluate(
        data=regression_data,
        target="target",
        pipelines={
            "ridge": pipeline,
        },
        task_type="regression",
    )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_ranking_works():
    results = pd.DataFrame(
        {
            "model": ["model_a", "model_b"],
            "r2": [0.80, 0.90],
            "mae": [20.0, 10.0],
        }
    )

    engine = RankingEngine()

    ranked = engine.rank(
        results,
        task_type="regression",
    )

    assert isinstance(ranked, pd.DataFrame)
    assert not ranked.empty


# ---------------------------------------------------------------------------
# Persistence / explainability
# ---------------------------------------------------------------------------


def test_persistence_round_trip(regression_data, tmp_path):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="ridge",
        task_type="regression",
    )

    X = regression_data.drop(columns=["target"])

    pipeline.fit(X, regression_data["target"])

    path = tmp_path / "model.pkl"

    persistence = ModelPersistence()

    saved_path = persistence.save(
        pipeline,
        path,
        metadata={
            "target": "target",
            "task_type": "regression",
        },
    )

    assert Path(saved_path).exists()

    loaded = persistence.load(path)

    predictions = persistence.predict(
        loaded,
        X.head(10),
    )

    assert len(predictions) == 10


def test_explainability_works(regression_data):
    generator = PipelineGenerator()

    pipeline = generator.build(
        data=regression_data,
        target="target",
        model_name="random_forest_regressor",
        task_type="regression",
    )

    X = regression_data.drop(columns=["target"])
    y = regression_data["target"]

    pipeline.fit(X, y)

    engine = ExplainabilityEngine()

    importance = engine.feature_importance(pipeline)

    assert isinstance(importance, pd.DataFrame)
    assert not importance.empty


def test_prediction_schema_validation_works():
    validator = PredictionSchemaValidator(
        allow_extra_columns=False,
    )

    data = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        }
    )

    result = validator.validate_and_align(
        data,
        expected_columns=["a", "b"],
    )

    assert list(result.columns) == ["a", "b"]


# ---------------------------------------------------------------------------
# Run management / experiments
# ---------------------------------------------------------------------------


def test_run_manager_lifecycle():
    manager = RunManager()

    run_id = manager.start(
        metadata={
            "test": True,
        }
    )

    assert run_id
    assert manager.is_running()

    summary = manager.complete()

    assert summary["status"] == "completed"
    assert not manager.is_running()


def test_experiment_tracker_persists_experiment(tmp_path):
    tracker = ExperimentTracker(
        experiment_directory=tmp_path,
    )

    result = {
        "run_id": "run_test",
        "status": "completed",
        "target": "target",
        "task_type": "regression",
        "best_model": {
            "model": "ridge",
            "metrics": {
                "r2": 0.85,
            },
        },
        "model_results": [],
        "rankings": [],
        "reproducibility": {
            "dataset_fingerprint": "abc",
            "configuration_fingerprint": "def",
        },
    }

    experiment_id = tracker.record(
        result=result,
        configuration={
            "random_state": 42,
        },
    )

    assert experiment_id

    files = list(tmp_path.glob("*.json"))

    assert files

    payload = json.loads(
        files[0].read_text(encoding="utf-8")
    )

    assert "reproducibility" in payload


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility_manager_works(regression_data, tmp_path):
    manager = ReproducibilityManager(
        random_state=42,
    )

    fingerprint = manager.dataset_fingerprint(
        regression_data,
    )

    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64

    snapshot = manager.create_snapshot(
        data=regression_data,
        configuration={
            "random_state": 42,
        },
        target="target",
        task_type="regression",
    )

    assert isinstance(snapshot, dict)
    assert snapshot["dataset_fingerprint"] == fingerprint

    path = tmp_path / "snapshot.json"

    manager.save_snapshot(
        snapshot,
        path,
    )

    loaded = manager.load_snapshot(path)

    assert loaded["dataset_fingerprint"] == fingerprint


def test_reproducibility_integration_works(
    regression_data,
    tmp_path,
):
    integration = ReproducibilityIntegration(
        random_state=42,
    )

    snapshot = integration.create_run_snapshot(
        data=regression_data,
        configuration={
            "random_state": 42,
        },
        target="target",
        task_type="regression",
    )

    assert isinstance(snapshot, dict)
    assert "dataset_fingerprint" in snapshot

    run_id = "integration_test"

    path = integration.save_snapshot(
        snapshot,
        tmp_path,
        run_id,
    )

    assert Path(path).exists()

    loaded = integration.load_snapshot(
        tmp_path,
        run_id,
    )

    assert loaded["dataset_fingerprint"] == snapshot[
        "dataset_fingerprint"
    ]


# ---------------------------------------------------------------------------
# Full AutoML regression workflow
# ---------------------------------------------------------------------------


def test_full_automl_regression_workflow(
    regression_data,
    tmp_path,
):
    data_path = tmp_path / "regression.csv"

    regression_data.to_csv(
        data_path,
        index=False,
    )

    automl = AutoML(
        random_state=42,
        experiment_directory=str(
            tmp_path / "experiments"
        ),
    )

    result = automl.fit(
        data=str(data_path),
        target="target",
    )

    assert isinstance(result, dict)

    assert result.get("task_type") == "regression"
    assert result["target"]["target"] == "target"

    assert result.get("model_results") is not None
    assert result.get("rankings") is not None
    assert result.get("best_model") is not None

    assert "reproducibility" in result

    assert result["reproducibility"] is not None

    experiments = list(
        (tmp_path / "experiments").glob("*.json")
    )

    assert experiments


# ---------------------------------------------------------------------------
# Full AutoML classification workflow
# ---------------------------------------------------------------------------


def test_full_automl_classification_workflow(
    classification_data,
    tmp_path,
):
    data_path = tmp_path / "classification.csv"

    classification_data.to_csv(
        data_path,
        index=False,
    )

    automl = AutoML(
        random_state=42,
        experiment_directory=str(
            tmp_path / "experiments"
        ),
    )

    result = automl.fit(
        data=str(data_path),
        target="target",
    )

    assert isinstance(result, dict)

    assert result.get("task_type") == "classification"
    assert result["target"]["target"] == "target"

    assert result.get("model_results") is not None
    assert result.get("rankings") is not None
    assert result.get("best_model") is not None

    assert "reproducibility" in result


# ---------------------------------------------------------------------------
# Prediction after AutoML
# ---------------------------------------------------------------------------


def test_prediction_after_full_automl(
    regression_data,
    tmp_path,
):
    train_path = tmp_path / "train.csv"
    regression_data.to_csv(
        train_path,
        index=False,
    )

    automl = AutoML(
        random_state=42,
        experiment_directory=str(
            tmp_path / "experiments"
        ),
    )

    result = automl.fit(
        data=str(train_path),
        target="target",
    )

    assert result.get("best_model") is not None

    model_path = tmp_path / "best_model.pkl"

    best_pipeline = result.get("best_pipeline")

    if best_pipeline is None:
        pytest.skip(
            "Current AutoML result does not expose best_pipeline."
        )

    persistence = ModelPersistence()

    persistence.save(
        best_pipeline,
        model_path,
        metadata={
            "target": "target",
            "task_type": "regression",
        },
    )

    loaded = persistence.load(model_path)

    prediction_data = regression_data.drop(
        columns=["target"]
    ).head(5)

    predictions = persistence.predict(
        loaded,
        prediction_data,
    )

    assert len(predictions) == 5
    assert np.isfinite(
        np.asarray(predictions, dtype=float)
    ).all()


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


def test_cli_help_works():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modelforge.cli",
            "--help",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    output = (
        result.stdout + result.stderr
    ).lower()

    assert "modelforge" in output


def test_cli_models_command_works():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modelforge.cli",
            "models",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    output = (
        result.stdout + result.stderr
    ).lower()

    assert "random" in output or "linear" in output


# ---------------------------------------------------------------------------
# Package import smoke test
# ---------------------------------------------------------------------------


def test_public_package_imports():
    import modelforge

    assert modelforge is not None
    assert hasattr(modelforge, "AutoML")


# ---------------------------------------------------------------------------
# Final system-level contract
# ---------------------------------------------------------------------------


def test_modelforge_system_contract():
    """
    High-level contract test.

    This intentionally checks that the main public building blocks
    remain importable together.
    """
    components = [
        AutoML,
        DatasetProfiler,
        TargetSelector,
        ColumnIntelligence,
        DataQualityAuditor,
        FeatureEngineeringEngine,
        FeatureSelectionEngine,
        ModelRegistry,
        PipelineGenerator,
        ModelScreeningEngine,
        CrossValidationEngine,
        RankingEngine,
        ExplainabilityEngine,
        ModelPersistence,
        PredictionSchemaValidator,
        ExperimentTracker,
        RunManager,
        ReproducibilityManager,
        ReproducibilityIntegration,
    ]

    assert all(component is not None for component in components)