from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline

from modelforge.prediction_validator import (
    PredictionSchemaError,
    PredictionSchemaValidator,
)


class ModelPersistence:
    """
    Save, load, validate, and use trained ModelForge pipelines.

    ModelForge persists the complete sklearn pipeline so that
    preprocessing and model transformations remain identical
    during future prediction.

    Prediction schema validation is delegated to the dedicated
    PredictionSchemaValidator component.
    """

    MODEL_FILENAME = "model.joblib"
    METADATA_FILENAME = "metadata.joblib"

    def save(
        self,
        pipeline: Pipeline,
        path: str,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> str:
        """
        Save a trained pipeline to disk.

        Parameters
        ----------
        pipeline:
            Fitted sklearn Pipeline.

        path:
            Destination file path.

        metadata:
            Optional metadata associated with the model.

        overwrite:
            Whether an existing file may be replaced.

        Returns
        -------
        str
            Absolute path of the saved model.
        """

        self._validate_pipeline(
            pipeline
        )

        destination = Path(path)

        if (
            destination.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"Model already exists: {destination}. "
                "Set overwrite=True to replace it."
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            pipeline,
            destination,
        )

        if metadata is not None:
            metadata_path = self._metadata_path(
                destination
            )

            joblib.dump(
                metadata,
                metadata_path,
            )

        return str(
            destination.resolve()
        )

    def load(
        self,
        path: str,
    ) -> Pipeline:
        """
        Load a saved ModelForge pipeline.
        """

        model_path = Path(path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        if not model_path.is_file():
            raise ValueError(
                f"Model path is not a file: "
                f"{model_path}"
            )

        pipeline = joblib.load(
            model_path
        )

        self._validate_pipeline(
            pipeline
        )

        return pipeline

    def load_metadata(
        self,
        path: str,
    ) -> dict[str, Any]:
        """
        Load metadata associated with a saved model.
        """

        model_path = Path(path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        metadata_path = self._metadata_path(
            model_path
        )

        if not metadata_path.exists():
            return {}

        metadata = joblib.load(
            metadata_path
        )

        if not isinstance(
            metadata,
            dict,
        ):
            raise ValueError(
                "Saved metadata must be a dictionary."
            )

        return metadata

    def predict(
        self,
        pipeline: Pipeline,
        data: pd.DataFrame,
    ) -> pd.Series:
        """
        Generate predictions using a fitted pipeline.

        Prediction data is validated and aligned using
        PredictionSchemaValidator before inference.
        """

        self._validate_pipeline(
            pipeline
        )

        validated_data = (
            self._validate_prediction_data(
                pipeline,
                data,
            )
        )

        predictions = pipeline.predict(
            validated_data
        )

        return pd.Series(
            predictions,
            index=validated_data.index,
            name="prediction",
        )

    def predict_from_file(
        self,
        pipeline: Pipeline,
        data_path: str,
    ) -> pd.Series:
        """
        Load a CSV dataset and generate predictions.
        """

        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Prediction dataset not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Prediction dataset is not a file: "
                f"{path}"
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                "predict_from_file currently "
                "supports CSV files only."
            )

        data = pd.read_csv(
            path
        )

        return self.predict(
            pipeline,
            data,
        )

    @staticmethod
    def predict_proba(
        pipeline: Pipeline,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate class probabilities when supported.

        Prediction data is validated and aligned using
        PredictionSchemaValidator before inference.
        """

        if not isinstance(
            pipeline,
            Pipeline,
        ):
            raise TypeError(
                "pipeline must be a sklearn Pipeline."
            )

        if not hasattr(
            pipeline,
            "predict_proba",
        ):
            raise ValueError(
                "This pipeline does not support "
                "probability predictions."
            )

        validated_data = (
            ModelPersistence._validate_prediction_data(
                pipeline,
                data,
            )
        )

        probabilities = (
            pipeline.predict_proba(
                validated_data
            )
        )

        model = pipeline.named_steps.get(
            "model"
        )

        if (
            model is not None
            and hasattr(
                model,
                "classes_",
            )
        ):
            columns = [
                f"probability_{label}"
                for label in model.classes_
            ]
        else:
            columns = [
                f"probability_{index}"
                for index in range(
                    probabilities.shape[1]
                )
            ]

        return pd.DataFrame(
            probabilities,
            index=validated_data.index,
            columns=columns,
        )

    @staticmethod
    def predict_proba_from_file(
        pipeline: Pipeline,
        data_path: str,
    ) -> pd.DataFrame:
        """
        Load a CSV dataset and generate class probabilities.
        """

        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Prediction dataset not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Prediction dataset is not a file: "
                f"{path}"
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                "predict_proba_from_file currently "
                "supports CSV files only."
            )

        data = pd.read_csv(
            path
        )

        return ModelPersistence.predict_proba(
            pipeline,
            data,
        )

    @staticmethod
    def _validate_prediction_data(
        pipeline: Pipeline,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate prediction data through the dedicated
        PredictionSchemaValidator.

        The fitted sklearn pipeline provides the expected
        training feature schema.
        """

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Cannot predict on an empty dataset."
            )

        expected_columns = (
            ModelPersistence._get_expected_columns(
                pipeline
            )
        )

        if expected_columns is None:
            return data

        validator = (
            PredictionSchemaValidator(
                allow_extra_columns=False,
                enforce_column_order=False,
                enforce_dtypes=False,
            )
        )

        try:
            return validator.validate_and_align(
                data=data,
                expected_columns=expected_columns,
            )

        except PredictionSchemaError as exc:
            raise ValueError(
                str(exc)
            ) from exc

    @staticmethod
    def _get_expected_columns(
        pipeline: Pipeline,
    ) -> list[str] | None:
        """
        Extract the feature schema recorded by sklearn.

        sklearn pipelines fitted with a pandas DataFrame
        expose feature names through feature_names_in_.
        """

        feature_names = getattr(
            pipeline,
            "feature_names_in_",
            None,
        )

        if feature_names is None:
            return None

        return [
            str(column)
            for column in feature_names
        ]

    @staticmethod
    def _metadata_path(
        model_path: Path,
    ) -> Path:
        """
        Generate the metadata path corresponding
        to a model file.
        """

        return model_path.with_name(
            f"{model_path.stem}_metadata.joblib"
        )

    @staticmethod
    def _validate_pipeline(
        pipeline: Pipeline,
    ) -> None:
        """
        Validate that the supplied object is a
        usable sklearn Pipeline.
        """

        if not isinstance(
            pipeline,
            Pipeline,
        ):
            raise TypeError(
                "pipeline must be a sklearn Pipeline."
            )

        if "model" not in pipeline.named_steps:
            raise ValueError(
                "Pipeline must contain a 'model' step."
            )