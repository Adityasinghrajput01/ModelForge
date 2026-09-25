import numpy as np
import pandas as pd
import pytest

from modelforge.ranking import (
    RankingEngine,
)


@pytest.fixture
def regression_results():
    return pd.DataFrame(
        {
            "model": [
                "model_a",
                "model_b",
                "model_c",
            ],
            "cv_mean_r2": [
                0.90,
                0.85,
                0.80,
            ],
            "cv_mean_rmse": [
                10.0,
                15.0,
                20.0,
            ],
            "cv_std_r2": [
                0.01,
                0.05,
                0.10,
            ],
            "total_time_seconds": [
                10.0,
                5.0,
                2.0,
            ],
        }
    )


@pytest.fixture
def classification_results():
    return pd.DataFrame(
        {
            "model": [
                "model_a",
                "model_b",
                "model_c",
            ],
            "cv_mean_f1": [
                0.95,
                0.88,
                0.80,
            ],
            "cv_mean_log_loss": [
                0.10,
                0.20,
                0.35,
            ],
            "cv_std_f1": [
                0.01,
                0.04,
                0.08,
            ],
            "total_time_seconds": [
                10.0,
                5.0,
                2.0,
            ],
        }
    )


def test_regression_ranking(
    regression_results,
):
    engine = RankingEngine()

    result = engine.rank(
        regression_results,
        task_type="regression",
    )

    assert len(result) == 3

    assert "primary_score" in result.columns
    assert "error_score" in result.columns
    assert "stability_score" in result.columns
    assert "speed_score" in result.columns
    assert "overall_score" in result.columns
    assert "rank" in result.columns


def test_classification_ranking(
    classification_results,
):
    engine = RankingEngine()

    result = engine.rank(
        classification_results,
        task_type="classification",
    )

    assert len(result) == 3

    assert "overall_score" in result.columns
    assert "rank" in result.columns


def test_scores_are_between_zero_and_one(
    regression_results,
):
    engine = RankingEngine()

    result = engine.rank(
        regression_results,
        task_type="regression",
    )

    for column in [
        "primary_score",
        "error_score",
        "stability_score",
        "speed_score",
    ]:
        assert (
            result[column] >= 0
        ).all()

        assert (
            result[column] <= 1
        ).all()


def test_ranks_are_assigned(
    regression_results,
):
    engine = RankingEngine()

    result = engine.rank(
        regression_results,
        task_type="regression",
    )

    assert set(
        result["rank"]
    ) == {1, 2, 3}


def test_results_sorted_by_rank(
    regression_results,
):
    engine = RankingEngine()

    result = engine.rank(
        regression_results,
        task_type="regression",
    )

    assert result["rank"].tolist() == [
        1,
        2,
        3,
    ]


def test_available_objectives():
    engine = RankingEngine()

    objectives = (
        engine.available_objectives()
    )

    assert "balanced" in objectives
    assert "performance" in objectives
    assert "error" in objectives
    assert "speed" in objectives


def test_objective_weights():
    engine = RankingEngine()

    weights = (
        engine.objective_weights(
            "balanced"
        )
    )

    assert (
        sum(weights.values())
        == pytest.approx(1.0)
    )


def test_all_objective_weights_sum_to_one():
    engine = RankingEngine()

    for objective in (
        engine.available_objectives()
    ):
        weights = (
            engine.objective_weights(
                objective
            )
        )

        assert (
            sum(weights.values())
            == pytest.approx(1.0)
        )


def test_invalid_objective():
    engine = RankingEngine()

    results = pd.DataFrame(
        {
            "model": ["model_a"],
            "r2": [0.9],
            "rmse": [10.0],
            "cv_std_r2": [0.01],
            "training_time_seconds": [
                1.0
            ],
        }
    )

    with pytest.raises(ValueError):
        engine.rank(
            results,
            task_type="regression",
            objective="invalid",
        )


def test_invalid_task_type(
    regression_results,
):
    engine = RankingEngine()

    with pytest.raises(ValueError):
        engine.rank(
            regression_results,
            task_type="clustering",
        )


def test_empty_results():
    engine = RankingEngine()

    with pytest.raises(ValueError):
        engine.rank(
            pd.DataFrame(),
            task_type="regression",
        )


def test_missing_model_column():
    engine = RankingEngine()

    results = pd.DataFrame(
        {
            "r2": [0.9],
            "rmse": [10.0],
            "cv_std_r2": [0.01],
            "total_time_seconds": [1.0],
        }
    )

    with pytest.raises(ValueError):
        engine.rank(
            results,
            task_type="regression",
        )


def test_non_dataframe_results():
    engine = RankingEngine()

    with pytest.raises(TypeError):
        engine.rank(
            "invalid",
            task_type="regression",
        )


def test_speed_objective():
    engine = RankingEngine()

    results = pd.DataFrame(
        {
            "model": [
                "slow",
                "fast",
            ],
            "cv_mean_r2": [
                0.95,
                0.80,
            ],
            "cv_mean_rmse": [
                5.0,
                15.0,
            ],
            "cv_std_r2": [
                0.01,
                0.05,
            ],
            "total_time_seconds": [
                100.0,
                1.0,
            ],
        }
    )

    ranked = engine.rank(
        results,
        task_type="regression",
        objective="speed",
    )

    assert len(ranked) == 2

    assert (
        ranked.iloc[0]["model"]
        in {"slow", "fast"}
    )


def test_regression_fallback_metrics():
    engine = RankingEngine()

    results = pd.DataFrame(
        {
            "model": [
                "model_a",
                "model_b",
            ],
            "r2": [
                0.90,
                0.80,
            ],
            "rmse": [
                10.0,
                20.0,
            ],
            "training_time_seconds": [
                2.0,
                4.0,
            ],
            "cv_std_r2": [
                0.01,
                0.05,
            ],
        }
    )

    ranked = engine.rank(
        results,
        task_type="regression",
    )

    assert len(ranked) == 2


def test_classification_fallback_metrics():
    engine = RankingEngine()

    results = pd.DataFrame(
        {
            "model": [
                "model_a",
                "model_b",
            ],
            "f1": [
                0.90,
                0.80,
            ],
            "log_loss": [
                0.10,
                0.20,
            ],
            "training_time_seconds": [
                2.0,
                4.0,
            ],
            "cv_std_f1": [
                0.01,
                0.05,
            ],
        }
    )

    ranked = engine.rank(
        results,
        task_type="classification",
    )

    assert len(ranked) == 2


def test_equal_metric_values(
    regression_results,
):
    engine = RankingEngine()

    results = regression_results.copy()

    results["cv_mean_r2"] = 0.90
    results["cv_mean_rmse"] = 10.0
    results["cv_std_r2"] = 0.01
    results["total_time_seconds"] = 5.0

    ranked = engine.rank(
        results,
        task_type="regression",
    )

    assert np.isclose(
        ranked["overall_score"].to_numpy(),
        1.0,
    ).all()


def test_objective_weight_copy_is_independent():
    engine = RankingEngine()

    weights = (
        engine.objective_weights(
            "balanced"
        )
    )

    weights["primary"] = 999

    fresh_weights = (
        engine.objective_weights(
            "balanced"
        )
    )

    assert (
        fresh_weights["primary"]
        != 999
    )