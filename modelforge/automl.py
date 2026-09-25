from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from modelforge.column_intelligence import ColumnIntelligence
from modelforge.config import ModelForgeConfig
from modelforge.cross_validation import CrossValidationEngine
from modelforge.data_audit import DataQualityAuditor
from modelforge.data_loader import DatasetLoader
from modelforge.explainability import ExplainabilityEngine
from modelforge.experiment_tracker import ExperimentTracker
from modelforge.hyperparameter_optimization import (
    HyperparameterOptimizationEngine,
)
from modelforge.model_registry import ModelRegistry
from modelforge.model_screening import ModelScreeningEngine
from modelforge.persistence import ModelPersistence
from modelforge.pipeline_generator import PipelineGenerator
from modelforge.profiler import DatasetProfiler
from modelforge.ranking import RankingEngine
from modelforge.reproducibility_integration import (
    ReproducibilityIntegration,
)
from modelforge.run_manager import RunManager
from modelforge.target_selector import TargetSelector


class AutoML:
    """
    Main ModelForge AutoML orchestration engine.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        cv: int = 5,
        random_state: int = 42,
        objective: str = "balanced",
        variance_threshold: float | None = None,
        correlation_threshold: float | None = None,
        enable_optimization: bool = False,
        optimization_models: int = 3,
        optimization_max_trials: int = 10,
        experiment_directory: str | Path = (
            ".modelforge/experiments"
        ),
        config: ModelForgeConfig | None = None,
    ):
        if config is not None and not isinstance(
            config,
            ModelForgeConfig,
        ):
            raise TypeError(
                "config must be a ModelForgeConfig"
            )

        self._config_provided = config is not None

        self.config = (
            config
            if config is not None
            else ModelForgeConfig()
        )

        self._configured_target = self._config_value(
            "target",
            None,
        )

        self._configured_task_type = self._config_value(
            "task_type",
            None,
        )

        self._configured_models = self._config_value(
            "models",
            None,
        )

        self._configured_excluded_columns = (
            self._config_value(
                "excluded_columns",
                [],
            )
        )

        self.test_size = self._config_value(
            "test_size",
            test_size,
        )

        self.cv = self._config_value(
            "cv",
            cv,
        )

        self.random_state = self._config_value(
            "random_state",
            random_state,
        )

        self.objective = self._config_value(
            "objective",
            objective,
        )

        self.variance_threshold = self._config_value(
            "feature_selection.variance_threshold",
            variance_threshold,
        )

        self.correlation_threshold = self._config_value(
            "feature_selection.correlation_threshold",
            correlation_threshold,
        )

        configured_optimization_enabled = (
            self._config_value(
                "optimization.enabled",
                None,
            )
        )

        self.enable_optimization = (
            enable_optimization
            if configured_optimization_enabled is None
            else configured_optimization_enabled
        )

        configured_optimization_models = (
            self._config_value(
                "optimization.models",
                None,
            )
        )

        self.optimization_models = (
            optimization_models
            if configured_optimization_models is None
            else configured_optimization_models
        )

        configured_optimization_max_trials = (
            self._config_value(
                "optimization.max_trials",
                None,
            )
        )

        self.optimization_max_trials = (
            optimization_max_trials
            if configured_optimization_max_trials is None
            else configured_optimization_max_trials
        )

        self._configured_experiment_directory = (
            self._config_value(
                "experiment_directory",
                experiment_directory,
            )
        )

        self.experiment_directory = Path(
            self._configured_experiment_directory
        )

        self._validate_configuration()

        self.loader = DatasetLoader()
        self.target_selector = TargetSelector()
        self.profiler = DatasetProfiler()
        self.column_intelligence = ColumnIntelligence()
        self.data_audit = DataQualityAuditor()

        self.registry = ModelRegistry()
        self.pipeline_generator = PipelineGenerator()

        self.screening = ModelScreeningEngine(
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self.cross_validator = CrossValidationEngine(
            cv=self.cv,
            random_state=self.random_state,
        )

        self.optimizer = (
            HyperparameterOptimizationEngine(
                cv=self.cv,
                random_state=self.random_state,
                max_trials=self.optimization_max_trials,
            )
        )

        self.ranking = RankingEngine()
        self.explainability = ExplainabilityEngine()
        self.persistence = ModelPersistence()

        self.run_manager = RunManager()

        self.experiment_tracker = ExperimentTracker(
            self.experiment_directory
        )

        self.is_fitted = False
        self.best_pipeline = None
        self.best_model = None
        self.result = None
        self.target = None
        self.task_type = None
        self.run_id = None
        self.experiment_id = None

        self.reproducibility = None

        self.reproducibility_integration = (
            ReproducibilityIntegration(
                random_state=self.random_state
            )
        )

    def fit(
        self,
        data: Any,
        target: str | None = None,
        task_type: str | None = None,
        model_names: list[str] | None = None,
        excluded_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run the complete ModelForge AutoML workflow.

        A run is automatically started before training and
        recorded after successful or failed execution.

        Reproducibility metadata is captured for every run.
        """

        target = (
            target
            if target is not None
            else self._configured_target
        )

        task_type = (
            task_type
            if task_type is not None
            else self._configured_task_type
        )

        model_names = (
            model_names
            if model_names is not None
            else self._configured_models
        )

        excluded_columns = (
            excluded_columns
            if excluded_columns is not None
            else self._configured_excluded_columns
        )

        if target is None:
            raise ValueError(
                "target must be provided to fit() "
                "or defined in ModelForgeConfig."
            )

        self.run_id = self.run_manager.start(
            {
                "target": target,
                "requested_task_type": task_type,
                "objective": self.objective,
                "cv": self.cv,
                "test_size": self.test_size,
                "random_state": self.random_state,
                "enable_optimization": (
                    self.enable_optimization
                ),
            }
        )

        self.reproducibility = None

        try:
            reproducibility_data = self._load_data(data)

            result = self._fit_workflow(
                data=reproducibility_data,
                target=target,
                task_type=task_type,
                model_names=model_names,
                excluded_columns=excluded_columns,
            )

            effective_configuration = (
                self._configuration(
                    target=target,
                    task_type=self.task_type,
                    models=model_names,
                    excluded_columns=excluded_columns,
                )
            )

            self.reproducibility = (
                self.reproducibility_integration.create_run_snapshot(
                    data=reproducibility_data,
                    configuration=effective_configuration,
                    target=self.target,
                    task_type=self.task_type,
                    run_id=self.run_id,
                    extra_metadata={
                        "model_names": model_names,
                        "excluded_columns": excluded_columns,
                    },
                )
            )

            result["reproducibility"] = (
                self.reproducibility
            )

            self.run_manager.update(
                {
                    "best_model": self.best_model,
                    "task_type": self.task_type,
                    "models_evaluated": result.get(
                        "models_evaluated"
                    ),
                }
            )

            run_summary = self.run_manager.complete()

            self.run_id = run_summary["run_id"]

            result["run_id"] = self.run_id
            result["run_summary"] = run_summary

            self.experiment_id = (
                self.experiment_tracker.record(
                    result=result,
                    configuration=effective_configuration,
                )
            )

            result["experiment_id"] = (
                self.experiment_id
            )

            self.result = result

            return result

        except Exception as exc:
            failure_summary = self.run_manager.fail(exc)

            self.run_id = failure_summary["run_id"]

            if self.reproducibility is None:
                try:
                    reproducibility_data = (
                        self._load_data(data)
                    )

                    effective_configuration = (
                        self._configuration(
                            target=target,
                            task_type=self.task_type,
                            models=model_names,
                            excluded_columns=excluded_columns,
                        )
                    )

                    self.reproducibility = (
                        self.reproducibility_integration.create_run_snapshot(
                            data=reproducibility_data,
                            configuration=effective_configuration,
                            target=target,
                            task_type=self.task_type,
                            run_id=self.run_id,
                            extra_metadata={
                                "model_names": model_names,
                                "excluded_columns": (
                                    excluded_columns
                                ),
                            },
                        )
                    )
                except Exception:
                    self.reproducibility = None

            failure_configuration = (
                self._configuration(
                    target=target,
                    task_type=self.task_type,
                    models=model_names,
                    excluded_columns=excluded_columns,
                )
            )

            failure_result = {
                "run_id": self.run_id,
                "status": "failed",
                "error": str(exc),
                "run_summary": failure_summary,
                "target": self.target,
                "task_type": self.task_type,
                "configuration": failure_configuration,
                "reproducibility": (
                    self.reproducibility
                ),
            }

            self.experiment_id = (
                self.experiment_tracker.record(
                    result=failure_result,
                    configuration=failure_configuration,
                )
            )

            self.result = failure_result

            raise

    def _fit_workflow(
        self,
        data: Any,
        target: str,
        task_type: str | None,
        model_names: list[str] | None,
        excluded_columns: list[str] | None,
    ) -> dict[str, Any]:
        """
        Execute the core AutoML workflow.

        Run lifecycle and experiment persistence are handled
        by fit().
        """

        dataframe = self._load_data(data)

        target_info = self.target_selector.select(
            dataframe,
            target=target,
            task_type=task_type,
        )

        self.target = target_info["target"]
        self.task_type = target_info["task_type"]

        profile = self.profiler.profile(dataframe)

        column_info = self.column_intelligence.analyze(
            dataframe
        )

        audit = self.data_audit.audit(
            data=dataframe,
            target=self.target,
            column_intelligence=column_info,
        )

        selected_models = self._select_models(
            model_names=model_names,
            task_type=self.task_type,
        )

        pipelines = self._build_pipelines(
            dataframe=dataframe,
            target=self.target,
            task_type=self.task_type,
            model_names=selected_models,
            excluded_columns=excluded_columns,
        )

        screening_results = self.screening.screen(
            data=dataframe,
            target=self.target,
            pipelines=pipelines,
            task_type=self.task_type,
        )

        cv_results = self.cross_validator.evaluate(
            data=dataframe,
            target=self.target,
            pipelines=pipelines,
            task_type=self.task_type,
        )

        combined_results = self._combine_results(
            screening_results,
            cv_results,
        )

        initial_ranking = self.ranking.rank(
            results=combined_results,
            task_type=self.task_type,
            objective=self.objective,
        )

        optimization_results = None
        optimized_pipelines = {}

        if self.enable_optimization:
            optimization_results = (
                self._optimize_top_models(
                    data=dataframe,
                    target=self.target,
                    task_type=self.task_type,
                    ranking=initial_ranking,
                    pipelines=pipelines,
                )
            )

            optimized_pipelines = {
                name: result["best_pipeline"]
                for name, result in (
                    optimization_results.items()
                )
                if result.get("best_pipeline")
                is not None
            }

        if optimized_pipelines:
            final_candidates = optimized_pipelines

            final_cv_results = (
                self.cross_validator.evaluate(
                    data=dataframe,
                    target=self.target,
                    pipelines=final_candidates,
                    task_type=self.task_type,
                )
            )

            final_screening_results = (
                self.screening.screen(
                    data=dataframe,
                    target=self.target,
                    pipelines=final_candidates,
                    task_type=self.task_type,
                )
            )

            final_combined_results = (
                self._combine_results(
                    final_screening_results,
                    final_cv_results,
                )
            )

            final_ranking = self.ranking.rank(
                results=final_combined_results,
                task_type=self.task_type,
                objective=self.objective,
            )

            best_model = self._select_best_model(
                final_ranking
            )

            candidate_pipeline = (
                final_candidates.get(best_model)
            )

        else:
            final_ranking = initial_ranking

            best_model = self._select_best_model(
                final_ranking
            )

            candidate_pipeline = pipelines.get(
                best_model
            )

        if candidate_pipeline is None:
            raise RuntimeError(
                "Unable to locate the best pipeline."
            )

        final_pipeline = self._fit_final_pipeline(
            pipeline=candidate_pipeline,
            data=dataframe,
            target=self.target,
        )

        self.best_pipeline = final_pipeline
        self.best_model = best_model
        self.is_fitted = True

        return {
            "target": target_info,
            "task_type": self.task_type,
            "profile": profile,
            "column_intelligence": column_info,
            "audit": audit,
            "models_evaluated": len(selected_models),
            "screening_results": screening_results,
            "cv_results": cv_results,
            "model_results": cv_results,
            "initial_ranking": initial_ranking,
            "optimization_enabled": (
                self.enable_optimization
            ),
            "optimization_results": (
                optimization_results
            ),
            "ranking": final_ranking,
            "rankings": final_ranking,
            "best_model": best_model,
            "best_pipeline": final_pipeline,
            "feature_selection": {
                "variance_threshold": (
                    self.variance_threshold
                ),
                "correlation_threshold": (
                    self.correlation_threshold
                ),
            },
        }

    def predict(
        self,
        data: Any,
    ) -> pd.Series:
        """
        Generate predictions using the fitted pipeline.
        """

        self._require_fitted()

        dataframe = self._load_data(data)

        return self.persistence.predict(
            self.best_pipeline,
            dataframe,
        )

    def predict_proba(
        self,
        data: Any,
    ) -> pd.DataFrame:
        """
        Generate class probabilities.
        """

        self._require_fitted()

        if self.task_type != "classification":
            raise RuntimeError(
                "predict_proba is only available "
                "for classification tasks."
            )

        dataframe = self._load_data(data)

        return self.persistence.predict_proba(
            self.best_pipeline,
            dataframe,
        )

    def save(
        self,
        path: str,
        overwrite: bool = False,
    ) -> str:
        """
        Save the fitted pipeline and metadata.
        """

        self._require_fitted()

        metadata = {
            "target": self.target,
            "task_type": self.task_type,
            "best_model": self.best_model,
            "objective": self.objective,
            "test_size": self.test_size,
            "cv": self.cv,
            "random_state": self.random_state,
            "variance_threshold": (
                self.variance_threshold
            ),
            "correlation_threshold": (
                self.correlation_threshold
            ),
            "enable_optimization": (
                self.enable_optimization
            ),
            "optimization_models": (
                self.optimization_models
            ),
            "optimization_max_trials": (
                self.optimization_max_trials
            ),
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "reproducibility": self.reproducibility,
        }

        return self.persistence.save(
            pipeline=self.best_pipeline,
            path=path,
            metadata=metadata,
            overwrite=overwrite,
        )

    def load(
        self,
        path: str,
    ) -> "AutoML":
        """
        Load a previously saved ModelForge pipeline.
        """

        self.best_pipeline = self.persistence.load(path)

        metadata = self.persistence.load_metadata(path)

        self.best_model = metadata.get(
            "best_model"
        )

        self.target = metadata.get(
            "target"
        )

        self.task_type = metadata.get(
            "task_type"
        )

        self.objective = metadata.get(
            "objective",
            self.objective,
        )

        self.test_size = metadata.get(
            "test_size",
            self.test_size,
        )

        self.cv = metadata.get(
            "cv",
            self.cv,
        )

        self.random_state = metadata.get(
            "random_state",
            self.random_state,
        )

        self.variance_threshold = metadata.get(
            "variance_threshold",
            self.variance_threshold,
        )

        self.correlation_threshold = metadata.get(
            "correlation_threshold",
            self.correlation_threshold,
        )

        self.enable_optimization = metadata.get(
            "enable_optimization",
            self.enable_optimization,
        )

        self.optimization_models = metadata.get(
            "optimization_models",
            self.optimization_models,
        )

        self.optimization_max_trials = metadata.get(
            "optimization_max_trials",
            self.optimization_max_trials,
        )

        self.run_id = metadata.get(
            "run_id"
        )

        self.experiment_id = metadata.get(
            "experiment_id"
        )

        self.reproducibility = metadata.get(
            "reproducibility"
        )

        self.reproducibility_integration = (
            ReproducibilityIntegration(
                random_state=self.random_state
            )
        )

        self.is_fitted = True

        return self

    def explain(
        self,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Return the most important features.
        """

        self._require_fitted()

        importance = (
            self.explainability.feature_importance(
                self.best_pipeline
            )
        )

        return self.explainability.top_features(
            importance,
            n=top_n,
        )

    def summary(self) -> dict[str, Any]:
        """
        Return a compact AutoML summary.
        """

        self._require_fitted()

        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "target": self.target,
            "task_type": self.task_type,
            "best_model": self.best_model,
            "objective": self.objective,
            "test_size": self.test_size,
            "cv": self.cv,
            "random_state": self.random_state,
            "variance_threshold": (
                self.variance_threshold
            ),
            "correlation_threshold": (
                self.correlation_threshold
            ),
            "optimization_enabled": (
                self.enable_optimization
            ),
            "optimization_models": (
                self.optimization_models
            ),
            "optimization_max_trials": (
                self.optimization_max_trials
            ),
        }

    def list_experiments(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return all locally tracked experiments.
        """

        return self.experiment_tracker.list_experiments()

    def get_experiment(
        self,
        experiment_id: str,
    ) -> dict[str, Any]:
        """
        Retrieve one tracked experiment.
        """

        return self.experiment_tracker.get(
            experiment_id
        )

    def _optimize_top_models(
        self,
        data: pd.DataFrame,
        target: str,
        task_type: str,
        ranking: pd.DataFrame,
        pipelines: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """
        Optimize the top-ranked candidate models.
        """

        if ranking.empty:
            return {}

        if "model" not in ranking.columns:
            raise RuntimeError(
                "Ranking results must contain "
                "a 'model' column."
            )

        successful = ranking

        if "status" in ranking.columns:
            successful = ranking[
                ranking["status"] == "success"
            ]

        top_models = successful.head(
            self.optimization_models
        )

        results: dict[str, dict[str, Any]] = {}

        for model_name in top_models[
            "model"
        ].tolist():
            pipeline = pipelines.get(
                model_name
            )

            if pipeline is None:
                continue

            parameter_space = (
                self.registry.get_hyperparameter_space(
                    model_name
                )
            )

            if not parameter_space:
                continue

            parameter_space = (
                self._pipeline_parameter_space(
                    parameter_space
                )
            )

            try:
                results[model_name] = (
                    self.optimizer.optimize(
                        data=data,
                        target=target,
                        pipeline=pipeline,
                        parameter_space=parameter_space,
                        task_type=task_type,
                    )
                )
            except Exception as exc:
                results[model_name] = {
                    "best_pipeline": None,
                    "best_params": {},
                    "best_score": None,
                    "trials": [],
                    "successful_trials": 0,
                    "failed_trials": 0,
                    "total_time_seconds": 0.0,
                    "error": str(exc),
                }

        return results

    @staticmethod
    def _pipeline_parameter_space(
        parameter_space: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert model parameters into sklearn pipeline
        parameter names.
        """

        return {
            (
                parameter_name
                if "__" in parameter_name
                else f"model__{parameter_name}"
            ): values
            for parameter_name, values
            in parameter_space.items()
        }

    def _fit_final_pipeline(
        self,
        pipeline,
        data: pd.DataFrame,
        target: str,
    ):
        """
        Fit the winning pipeline on the complete dataset.
        """

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if target not in data.columns:
            raise ValueError(
                f"Target column '{target}' "
                "does not exist."
            )

        X = data.drop(
            columns=[target]
        )

        y = data[target]

        try:
            pipeline.fit(
                X,
                y,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to fit the final selected "
                f"pipeline: {exc}"
            ) from exc

        return pipeline

    def _build_pipelines(
        self,
        dataframe: pd.DataFrame,
        target: str,
        task_type: str,
        model_names: list[str],
        excluded_columns: list[str] | None,
    ) -> dict:
        """
        Generate candidate pipelines.
        """

        pipelines = {}

        for model_name in model_names:
            pipelines[model_name] = (
                self.pipeline_generator.build(
                    data=dataframe,
                    target=target,
                    model_name=model_name,
                    task_type=task_type,
                    excluded_columns=excluded_columns,
                    variance_threshold=(
                        self.variance_threshold
                    ),
                    correlation_threshold=(
                        self.correlation_threshold
                    ),
                )
            )

        return pipelines

    def _select_models(
        self,
        model_names: list[str] | None,
        task_type: str,
    ) -> list[str]:
        """
        Resolve and validate model names.
        """

        available = self.registry.list_models(
            task_type=task_type
        )

        if model_names is None:
            return available

        invalid = [
            name
            for name in model_names
            if name not in available
        ]

        if invalid:
            raise ValueError(
                "Unknown or incompatible models: "
                + ", ".join(invalid)
            )

        if not model_names:
            raise ValueError(
                "At least one model must be selected."
            )

        return model_names

    def _combine_results(
        self,
        screening_results: pd.DataFrame,
        cv_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine holdout and cross-validation results.
        """

        screening = pd.DataFrame(
            screening_results
        )

        cross_validation = pd.DataFrame(
            cv_results
        )

        if screening.empty:
            raise RuntimeError(
                "No model screening results available."
            )

        if cross_validation.empty:
            return screening

        if "model" not in screening.columns:
            raise RuntimeError(
                "Screening results must contain "
                "a 'model' column."
            )

        if "model" not in cross_validation.columns:
            raise RuntimeError(
                "Cross-validation results must "
                "contain a 'model' column."
            )

        return screening.merge(
            cross_validation,
            on="model",
            how="left",
            suffixes=(
                "",
                "_cv",
            ),
        )

    def _select_best_model(
        self,
        ranking: pd.DataFrame,
    ) -> str:
        """
        Select the top successful model from ranking.
        """

        if ranking.empty:
            raise RuntimeError(
                "Ranking produced no models."
            )

        if "status" in ranking.columns:
            successful = ranking[
                ranking["status"] == "success"
            ]
        else:
            successful = ranking

        if successful.empty:
            raise RuntimeError(
                "No successful model was found."
            )

        return str(
            successful.iloc[0]["model"]
        )

    def _load_data(
        self,
        data: Any,
    ) -> pd.DataFrame:
        """
        Load a dataset through DatasetLoader.
        """

        if isinstance(
            data,
            pd.DataFrame,
        ):
            return data.copy()

        if isinstance(
            data,
            (str, Path),
        ):
            return self.loader.load(
                str(data)
            )

        raise TypeError(
            "data must be a pandas DataFrame "
            "or a supported dataset path."
        )

    def _configuration(
        self,
        target: str | None = None,
        task_type: str | None = None,
        models: list[str] | None = None,
        excluded_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Return the effective configuration used by
        the current AutoML run.

        Explicit runtime values are included so the
        reproducibility snapshot and experiment tracker
        persist exactly the same configuration.
        """

        effective_target = (
            target
            if target is not None
            else self._configured_target
        )

        effective_task_type = (
            task_type
            if task_type is not None
            else self._configured_task_type
        )

        effective_models = (
            models
            if models is not None
            else self._configured_models
        )

        effective_excluded_columns = (
            excluded_columns
            if excluded_columns is not None
            else self._configured_excluded_columns
        )

        return {
            "target": effective_target,
            "task_type": effective_task_type,
            "models": effective_models,
            "excluded_columns": (
                effective_excluded_columns
            ),
            "test_size": self.test_size,
            "cv": self.cv,
            "random_state": self.random_state,
            "objective": self.objective,
            "variance_threshold": (
                self.variance_threshold
            ),
            "correlation_threshold": (
                self.correlation_threshold
            ),
            "enable_optimization": (
                self.enable_optimization
            ),
            "optimization_models": (
                self.optimization_models
            ),
            "optimization_max_trials": (
                self.optimization_max_trials
            ),
            "experiment_directory": (
                str(self._configured_experiment_directory)
            ),
        }

    def _config_value(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Read a configuration value, including nested keys.
        """

        if not self._config_provided:
            return default

        values = self.config.to_dict()
        current: Any = values

        for part in key.split("."):
            if not isinstance(
                current,
                dict,
            ):
                return default

            if part not in current:
                return default

            current = current[part]

        return current

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "AutoML has not been fitted yet."
            )

        if self.best_pipeline is None:
            raise RuntimeError(
                "AutoML is marked as fitted but "
                "no fitted pipeline is available."
            )

    def _validate_configuration(self) -> None:
        """
        Validate AutoML constructor configuration.
        """

        if not isinstance(
            self.test_size,
            (int, float),
        ):
            raise TypeError(
                "test_size must be numeric."
            )

        if isinstance(
            self.test_size,
            bool,
        ):
            raise TypeError(
                "test_size must be numeric."
            )

        if not 0 < self.test_size < 1:
            raise ValueError(
                "test_size must be between 0 and 1."
            )

        if not isinstance(
            self.cv,
            int,
        ):
            raise TypeError(
                "cv must be an integer."
            )

        if isinstance(
            self.cv,
            bool,
        ):
            raise TypeError(
                "cv must be an integer."
            )

        if self.cv < 2:
            raise ValueError(
                "cv must be at least 2."
            )

        if not isinstance(
            self.random_state,
            int,
        ):
            raise TypeError(
                "random_state must be an integer."
            )

        if isinstance(
            self.random_state,
            bool,
        ):
            raise TypeError(
                "random_state must be an integer."
            )

        if self.objective not in {
            "balanced",
            "performance",
            "error",
            "speed",
        }:
            raise ValueError(
                "Invalid ranking objective."
            )

        if self.variance_threshold is not None:
            if isinstance(
                self.variance_threshold,
                bool,
            ):
                raise TypeError(
                    "variance_threshold must be numeric."
                )

            if not isinstance(
                self.variance_threshold,
                (int, float),
            ):
                raise TypeError(
                    "variance_threshold must be numeric."
                )

            if self.variance_threshold < 0:
                raise ValueError(
                    "variance_threshold cannot "
                    "be negative."
                )

        if self.correlation_threshold is not None:
            if isinstance(
                self.correlation_threshold,
                bool,
            ):
                raise TypeError(
                    "correlation_threshold must "
                    "be numeric."
                )

            if not isinstance(
                self.correlation_threshold,
                (int, float),
            ):
                raise TypeError(
                    "correlation_threshold must "
                    "be numeric."
                )

            if not (
                0
                < self.correlation_threshold
                <= 1
            ):
                raise ValueError(
                    "correlation_threshold must "
                    "be between 0 and 1."
                )

        if not isinstance(
            self.enable_optimization,
            bool,
        ):
            raise TypeError(
                "enable_optimization must be a boolean."
            )

        if not isinstance(
            self.optimization_models,
            int,
        ) or isinstance(
            self.optimization_models,
            bool,
        ):
            raise TypeError(
                "optimization_models must be an integer."
            )

        if self.optimization_models < 1:
            raise ValueError(
                "optimization_models must be at least 1."
            )

        if not isinstance(
            self.optimization_max_trials,
            int,
        ) or isinstance(
            self.optimization_max_trials,
            bool,
        ):
            raise TypeError(
                "optimization_max_trials must be an integer."
            )

        if self.optimization_max_trials < 1:
            raise ValueError(
                "optimization_max_trials must be at least 1."
            )