from pathlib import Path
from typing import Any

import yaml


class ModelForgeConfig:
    """
    Central configuration manager for ModelForge.

    Supports:
    - Python dictionary configuration
    - YAML configuration files
    - Configuration validation
    - Configuration persistence
    - Nested feature-selection settings
    - Optimization settings
    - Experiment tracking settings
    """

    DEFAULTS = {
        "target": None,
        "task_type": None,
        "objective": "balanced",
        "test_size": 0.2,
        "cv": 5,
        "random_state": 42,
        "models": None,
        "excluded_columns": [],
        "feature_selection": {
            "variance_threshold": None,
            "correlation_threshold": None,
        },
        "optimization": {
            "enabled": False,
            "models": 3,
            "max_trials": 10,
        },
        "experiment_directory": ".modelforge/experiments",
    }

    VALID_TASK_TYPES = {
        "regression",
        "classification",
    }

    VALID_OBJECTIVES = {
        "balanced",
        "performance",
        "error",
        "speed",
    }

    def __init__(
        self,
        values: dict[str, Any] | None = None,
    ):
        if values is None:
            values = {}

        if not isinstance(values, dict):
            raise TypeError(
                "Configuration values must be a dictionary."
            )

        self.values = self._build_values(values)
        self.validate()

    @classmethod
    def from_file(
        cls,
        path: str,
    ) -> "ModelForgeConfig":
        """
        Load ModelForge configuration from YAML.
        """

        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        if not config_path.is_file():
            raise ValueError(
                f"Configuration path is not a file: {config_path}"
            )

        if config_path.suffix.lower() not in {
            ".yaml",
            ".yml",
        }:
            raise ValueError(
                "Configuration file must use "
                ".yaml or .yml extension."
            )

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            values = yaml.safe_load(file)

        if values is None:
            values = {}

        if not isinstance(values, dict):
            raise ValueError(
                "Configuration root must be a dictionary."
            )

        return cls(values)

    def save(
        self,
        path: str,
        overwrite: bool = False,
    ) -> str:
        """
        Save the configuration to YAML.
        """

        config_path = Path(path)

        if config_path.exists() and not overwrite:
            raise FileExistsError(
                f"Configuration already exists: {config_path}"
            )

        config_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with config_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(
                self.values,
                file,
                sort_keys=False,
            )

        return str(config_path.resolve())

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a top-level configuration value.
        """

        return self.values.get(
            key,
            default,
        )

    def get_nested(
        self,
        section: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a value from a nested configuration section.
        """

        section_value = self.values.get(
            section,
            {},
        )

        if not isinstance(section_value, dict):
            return default

        return section_value.get(
            key,
            default,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Return a deep copy of the configuration.
        """

        return self._deep_copy(
            self.values
        )

    def validate(self) -> None:
        """
        Validate all configuration values.
        """

        target = self.values.get(
            "target"
        )

        if target is not None:
            if not isinstance(
                target,
                str,
            ):
                raise TypeError(
                    "target must be a string."
                )

            if not target.strip():
                raise ValueError(
                    "target cannot be empty."
                )

        task_type = self.values.get(
            "task_type"
        )

        if task_type is not None:
            if not isinstance(
                task_type,
                str,
            ):
                raise TypeError(
                    "task_type must be a string."
                )

            if task_type not in self.VALID_TASK_TYPES:
                raise ValueError(
                    "task_type must be "
                    "'regression' or "
                    "'classification'."
                )

        objective = self.values.get(
            "objective"
        )

        if objective not in self.VALID_OBJECTIVES:
            raise ValueError(
                "objective must be one of: "
                "balanced, performance, "
                "error, speed."
            )

        test_size = self.values.get(
            "test_size"
        )

        if isinstance(test_size, bool):
            raise TypeError(
                "test_size must be numeric."
            )

        if not isinstance(
            test_size,
            (int, float),
        ):
            raise TypeError(
                "test_size must be numeric."
            )

        if not 0 < test_size < 1:
            raise ValueError(
                "test_size must be between 0 and 1."
            )

        cv = self.values.get(
            "cv"
        )

        if isinstance(cv, bool):
            raise TypeError(
                "cv must be an integer."
            )

        if not isinstance(
            cv,
            int,
        ):
            raise TypeError(
                "cv must be an integer."
            )

        if cv < 2:
            raise ValueError(
                "cv must be at least 2."
            )

        random_state = self.values.get(
            "random_state"
        )

        if isinstance(random_state, bool):
            raise TypeError(
                "random_state must be an integer."
            )

        if not isinstance(
            random_state,
            int,
        ):
            raise TypeError(
                "random_state must be an integer."
            )

        models = self.values.get(
            "models"
        )

        if models is not None:
            if not isinstance(
                models,
                list,
            ):
                raise TypeError(
                    "models must be a list."
                )

            if not models:
                raise ValueError(
                    "models cannot be empty."
                )

            if not all(
                isinstance(model, str)
                for model in models
            ):
                raise TypeError(
                    "Every model name must be a string."
                )

        excluded_columns = self.values.get(
            "excluded_columns"
        )

        if not isinstance(
            excluded_columns,
            list,
        ):
            raise TypeError(
                "excluded_columns must be a list."
            )

        if not all(
            isinstance(column, str)
            for column in excluded_columns
        ):
            raise TypeError(
                "Every excluded column must be a string."
            )

        feature_selection = self.values.get(
            "feature_selection"
        )

        if not isinstance(
            feature_selection,
            dict,
        ):
            raise TypeError(
                "feature_selection must be a dictionary."
            )

        variance_threshold = (
            feature_selection.get(
                "variance_threshold"
            )
        )

        if variance_threshold is not None:
            if isinstance(
                variance_threshold,
                bool,
            ):
                raise TypeError(
                    "variance_threshold must be numeric."
                )

            if not isinstance(
                variance_threshold,
                (int, float),
            ):
                raise TypeError(
                    "variance_threshold must be numeric."
                )

            if variance_threshold < 0:
                raise ValueError(
                    "variance_threshold cannot be negative."
                )

        correlation_threshold = (
            feature_selection.get(
                "correlation_threshold"
            )
        )

        if correlation_threshold is not None:
            if isinstance(
                correlation_threshold,
                bool,
            ):
                raise TypeError(
                    "correlation_threshold must be numeric."
                )

            if not isinstance(
                correlation_threshold,
                (int, float),
            ):
                raise TypeError(
                    "correlation_threshold must be numeric."
                )

            if not (
                0 < correlation_threshold <= 1
            ):
                raise ValueError(
                    "correlation_threshold must "
                    "be between 0 and 1."
                )

        optimization = self.values.get(
            "optimization"
        )

        if not isinstance(
            optimization,
            dict,
        ):
            raise TypeError(
                "optimization must be a dictionary."
            )

        enabled = optimization.get(
            "enabled"
        )

        if not isinstance(
            enabled,
            bool,
        ):
            raise TypeError(
                "optimization.enabled must be a boolean."
            )

        optimization_models = optimization.get(
            "models"
        )

        if isinstance(
            optimization_models,
            bool,
        ):
            raise TypeError(
                "optimization.models must be an integer."
            )

        if not isinstance(
            optimization_models,
            int,
        ):
            raise TypeError(
                "optimization.models must be an integer."
            )

        if optimization_models < 1:
            raise ValueError(
                "optimization.models must be at least 1."
            )

        max_trials = optimization.get(
            "max_trials"
        )

        if isinstance(
            max_trials,
            bool,
        ):
            raise TypeError(
                "optimization.max_trials must be an integer."
            )

        if not isinstance(
            max_trials,
            int,
        ):
            raise TypeError(
                "optimization.max_trials must be an integer."
            )

        if max_trials < 1:
            raise ValueError(
                "optimization.max_trials must be at least 1."
            )

        experiment_directory = self.values.get(
            "experiment_directory"
        )

        if not isinstance(
            experiment_directory,
            str,
        ):
            raise TypeError(
                "experiment_directory must be a string."
            )

        if not experiment_directory.strip():
            raise ValueError(
                "experiment_directory cannot be empty."
            )

    @classmethod
    def _build_values(
        cls,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Merge user configuration with defaults.
        """

        unknown_keys = (
            set(values) - set(cls.DEFAULTS)
        )

        if unknown_keys:
            raise ValueError(
                "Unknown configuration keys: "
                + ", ".join(
                    sorted(unknown_keys)
                )
            )

        result = cls._deep_copy(
            cls.DEFAULTS
        )

        for key, value in values.items():
            if (
                key in {
                    "feature_selection",
                    "optimization",
                }
                and isinstance(value, dict)
            ):
                result[key].update(value)
            else:
                result[key] = value

        return result

    @staticmethod
    def _deep_copy(value):
        if isinstance(
            value,
            dict,
        ):
            return {
                key: ModelForgeConfig._deep_copy(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            list,
        ):
            return [
                ModelForgeConfig._deep_copy(
                    item
                )
                for item in value
            ]

        return value