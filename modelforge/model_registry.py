from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Type

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from sklearn.svm import (
    SVC,
    SVR,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


@dataclass(frozen=True)
class ModelSpec:
    """
    Metadata describing a machine-learning model.
    """

    name: str
    estimator: Type[BaseEstimator]
    task_type: str
    category: str
    requires_scaling: bool = False
    supports_probability: bool = False
    default_params: dict[str, Any] = field(
        default_factory=dict
    )
    hyperparameter_space: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        """Validate the model specification."""

        if not self.name.strip():
            raise ValueError(
                "Model name cannot be empty."
            )

        if self.task_type not in {
            "regression",
            "classification",
        }:
            raise ValueError(
                "task_type must be 'regression' or 'classification'."
            )

        if not self.category.strip():
            raise ValueError(
                "Model category cannot be empty."
            )

        if not isinstance(
            self.default_params,
            dict,
        ):
            raise TypeError(
                "default_params must be a dictionary."
            )

        if not isinstance(
            self.hyperparameter_space,
            dict,
        ):
            raise TypeError(
                "hyperparameter_space must be a dictionary."
            )


class ModelRegistry:
    """
    Central registry of models supported by ModelForge.

    The registry provides model metadata, estimator creation,
    task-based filtering, category filtering, and capability
    discovery for the AutoML pipeline.
    """

    def __init__(self):
        self._models = self._build_registry()

        for spec in self._models.values():
            spec.validate()

    @staticmethod
    def _build_registry() -> dict[str, ModelSpec]:
        """Build the default ModelForge model registry."""

        return {
            "linear_regression": ModelSpec(
                name="Linear Regression",
                estimator=LinearRegression,
                task_type="regression",
                category="linear",
                requires_scaling=False,
                default_params={},
                hyperparameter_space={},
            ),
            "ridge": ModelSpec(
                name="Ridge Regression",
                estimator=Ridge,
                task_type="regression",
                category="linear",
                requires_scaling=True,
                default_params={
                    "alpha": 1.0,
                },
                hyperparameter_space={
                    "alpha": [
                        0.01,
                        0.1,
                        1.0,
                        10.0,
                        100.0,
                    ],
                },
            ),
            "lasso": ModelSpec(
                name="Lasso Regression",
                estimator=Lasso,
                task_type="regression",
                category="linear",
                requires_scaling=True,
                default_params={
                    "alpha": 1.0,
                },
                hyperparameter_space={
                    "alpha": [
                        0.001,
                        0.01,
                        0.1,
                        1.0,
                        10.0,
                    ],
                },
            ),
            "elasticnet": ModelSpec(
                name="ElasticNet",
                estimator=ElasticNet,
                task_type="regression",
                category="linear",
                requires_scaling=True,
                default_params={
                    "alpha": 1.0,
                    "l1_ratio": 0.5,
                },
                hyperparameter_space={
                    "alpha": [
                        0.01,
                        0.1,
                        1.0,
                        10.0,
                    ],
                    "l1_ratio": [
                        0.1,
                        0.5,
                        0.9,
                    ],
                },
            ),
            "decision_tree_regressor": ModelSpec(
                name="Decision Tree Regressor",
                estimator=DecisionTreeRegressor,
                task_type="regression",
                category="tree",
                requires_scaling=False,
                default_params={
                    "random_state": 42,
                },
                hyperparameter_space={
                    "max_depth": [
                        None,
                        5,
                        10,
                        20,
                    ],
                    "min_samples_split": [
                        2,
                        5,
                        10,
                    ],
                },
            ),
            "random_forest_regressor": ModelSpec(
                name="Random Forest Regressor",
                estimator=RandomForestRegressor,
                task_type="regression",
                category="ensemble",
                requires_scaling=False,
                default_params={
                    "n_estimators": 100,
                    "random_state": 42,
                    "n_jobs": -1,
                },
                hyperparameter_space={
                    "n_estimators": [
                        100,
                        200,
                    ],
                    "max_depth": [
                        None,
                        10,
                        20,
                    ],
                    "min_samples_split": [
                        2,
                        5,
                    ],
                },
            ),
            "extra_trees_regressor": ModelSpec(
                name="Extra Trees Regressor",
                estimator=ExtraTreesRegressor,
                task_type="regression",
                category="ensemble",
                requires_scaling=False,
                default_params={
                    "n_estimators": 100,
                    "random_state": 42,
                    "n_jobs": -1,
                },
                hyperparameter_space={
                    "n_estimators": [
                        100,
                        200,
                    ],
                    "max_depth": [
                        None,
                        10,
                        20,
                    ],
                    "min_samples_split": [
                        2,
                        5,
                    ],
                },
            ),
            "gradient_boosting_regressor": ModelSpec(
                name="Gradient Boosting Regressor",
                estimator=GradientBoostingRegressor,
                task_type="regression",
                category="boosting",
                requires_scaling=False,
                default_params={
                    "random_state": 42,
                },
                hyperparameter_space={
                    "n_estimators": [
                        100,
                        200,
                    ],
                    "learning_rate": [
                        0.01,
                        0.05,
                        0.1,
                    ],
                    "max_depth": [
                        2,
                        3,
                        5,
                    ],
                },
            ),
            "knn_regressor": ModelSpec(
                name="K-Nearest Neighbors Regressor",
                estimator=KNeighborsRegressor,
                task_type="regression",
                category="distance_based",
                requires_scaling=True,
                default_params={
                    "n_neighbors": 5,
                },
                hyperparameter_space={
                    "n_neighbors": [
                        3,
                        5,
                        7,
                        11,
                    ],
                    "weights": [
                        "uniform",
                        "distance",
                    ],
                },
            ),
            "svr": ModelSpec(
                name="Support Vector Regressor",
                estimator=SVR,
                task_type="regression",
                category="svm",
                requires_scaling=True,
                default_params={
                    "kernel": "rbf",
                },
                hyperparameter_space={
                    "C": [
                        0.1,
                        1.0,
                        10.0,
                    ],
                    "gamma": [
                        "scale",
                        "auto",
                    ],
                    "epsilon": [
                        0.01,
                        0.1,
                        0.2,
                    ],
                },
            ),
            "logistic_regression": ModelSpec(
                name="Logistic Regression",
                estimator=LogisticRegression,
                task_type="classification",
                category="linear",
                requires_scaling=True,
                supports_probability=True,
                default_params={
                    "max_iter": 1000,
                },
                hyperparameter_space={
                    "C": [
                        0.01,
                        0.1,
                        1.0,
                        10.0,
                    ],
                },
            ),
            "decision_tree_classifier": ModelSpec(
                name="Decision Tree Classifier",
                estimator=DecisionTreeClassifier,
                task_type="classification",
                category="tree",
                requires_scaling=False,
                supports_probability=True,
                default_params={
                    "random_state": 42,
                },
                hyperparameter_space={
                    "max_depth": [
                        None,
                        5,
                        10,
                        20,
                    ],
                    "min_samples_split": [
                        2,
                        5,
                        10,
                    ],
                },
            ),
            "random_forest_classifier": ModelSpec(
                name="Random Forest Classifier",
                estimator=RandomForestClassifier,
                task_type="classification",
                category="ensemble",
                requires_scaling=False,
                supports_probability=True,
                default_params={
                    "n_estimators": 100,
                    "random_state": 42,
                    "n_jobs": -1,
                },
                hyperparameter_space={
                    "n_estimators": [
                        100,
                        200,
                    ],
                    "max_depth": [
                        None,
                        10,
                        20,
                    ],
                    "min_samples_split": [
                        2,
                        5,
                    ],
                },
            ),
            "extra_trees_classifier": ModelSpec(
                name="Extra Trees Classifier",
                estimator=ExtraTreesClassifier,
                task_type="classification",
                category="ensemble",
                requires_scaling=False,
                supports_probability=True,
                default_params={
                    "n_estimators": 100,
                    "random_state": 42,
                    "n_jobs": -1,
                },
                hyperparameter_space={
                    "n_estimators": [
                        100,
                        200,
                    ],
                    "max_depth": [
                        None,
                        10,
                        20,
                    ],
                    "min_samples_split": [
                        2,
                        5,
                    ],
                },
            ),
            "gradient_boosting_classifier": ModelSpec(
                name="Gradient Boosting Classifier",
                estimator=GradientBoostingClassifier,
                task_type="classification",
                category="boosting",
                requires_scaling=False,
                supports_probability=True,
                default_params={
                    "random_state": 42,
                },
                hyperparameter_space={
                    "n_estimators": [
                        100,
                        200,
                    ],
                    "learning_rate": [
                        0.01,
                        0.05,
                        0.1,
                    ],
                    "max_depth": [
                        2,
                        3,
                        5,
                    ],
                },
            ),
            "knn_classifier": ModelSpec(
                name="K-Nearest Neighbors Classifier",
                estimator=KNeighborsClassifier,
                task_type="classification",
                category="distance_based",
                requires_scaling=True,
                supports_probability=True,
                default_params={
                    "n_neighbors": 5,
                },
                hyperparameter_space={
                    "n_neighbors": [
                        3,
                        5,
                        7,
                        11,
                    ],
                    "weights": [
                        "uniform",
                        "distance",
                    ],
                },
            ),
            "svc": ModelSpec(
                name="Support Vector Classifier",
                estimator=SVC,
                task_type="classification",
                category="svm",
                requires_scaling=True,
                supports_probability=True,
                default_params={
                    "kernel": "rbf",
                    "probability": True,
                },
                hyperparameter_space={
                    "C": [
                        0.1,
                        1.0,
                        10.0,
                    ],
                    "gamma": [
                        "scale",
                        "auto",
                    ],
                },
            ),
        }

    def get(
        self,
        model_name: str,
    ) -> ModelSpec:
        """Retrieve a model specification."""

        if not isinstance(model_name, str):
            raise TypeError(
                "model_name must be a string."
            )

        if model_name not in self._models:
            raise KeyError(
                f"Unknown model: {model_name}"
            )

        return self._models[model_name]

    def get_models(
        self,
        task_type: str | None = None,
    ) -> dict[str, ModelSpec]:
        """Return all models or models for a specific task."""

        if task_type is None:
            return self._models.copy()

        if task_type not in {
            "regression",
            "classification",
        }:
            raise ValueError(
                "task_type must be 'regression' "
                "or 'classification'."
            )

        return {
            name: spec
            for name, spec in self._models.items()
            if spec.task_type == task_type
        }

    def list_models(
        self,
        task_type: str | None = None,
    ) -> list[str]:
        """Return model identifiers."""

        return list(
            self.get_models(task_type).keys()
        )

    def list_by_category(
        self,
        category: str,
        task_type: str | None = None,
    ) -> list[str]:
        """
        Return model identifiers belonging to a category.

        Examples of categories:
        linear, tree, ensemble, boosting, distance_based, svm
        """

        if not isinstance(category, str):
            raise TypeError(
                "category must be a string."
            )

        models = self.get_models(task_type)

        return [
            name
            for name, spec in models.items()
            if spec.category == category
        ]

    def models_requiring_scaling(
        self,
        task_type: str | None = None,
    ) -> list[str]:
        """Return models that require feature scaling."""

        models = self.get_models(task_type)

        return [
            name
            for name, spec in models.items()
            if spec.requires_scaling
        ]

    def models_supporting_probability(
        self,
        task_type: str | None = None,
    ) -> list[str]:
        """Return models supporting probability predictions."""

        models = self.get_models(task_type)

        return [
            name
            for name, spec in models.items()
            if spec.supports_probability
        ]

    def get_hyperparameter_space(
        self,
        model_name: str,
    ) -> dict[str, Any]:
        """Return the hyperparameter search space for a model."""

        return self.get(
            model_name
        ).hyperparameter_space.copy()

    def get_default_params(
        self,
        model_name: str,
    ) -> dict[str, Any]:
        """Return a copy of the model's default parameters."""

        return self.get(
            model_name
        ).default_params.copy()

    def create(
        self,
        model_name: str,
        **params,
    ) -> BaseEstimator:
        """Create an estimator instance."""

        spec = self.get(model_name)

        estimator_params = {
            **spec.default_params,
            **params,
        }

        return spec.estimator(
            **estimator_params
        )

    def supports_probability(
        self,
        model_name: str,
    ) -> bool:
        """Check whether a model supports probability predictions."""

        return bool(
            self.get(
                model_name
            ).supports_probability
        )

    def requires_scaling(
        self,
        model_name: str,
    ) -> bool:
        """Check whether a model requires feature scaling."""

        return bool(
            self.get(
                model_name
            ).requires_scaling
        )

    def count(
        self,
        task_type: str | None = None,
    ) -> int:
        """Return the number of registered models."""

        return len(
            self.get_models(task_type)
        )