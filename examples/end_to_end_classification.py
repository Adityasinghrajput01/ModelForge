"""Run a complete local ModelForge classification workflow."""

from pathlib import Path

import pandas as pd

from modelforge import AutoML

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "iris_classification.csv"


def main() -> None:
    data = pd.read_csv(DATA)
    automl = AutoML(cv=3, random_state=42)
    result = automl.fit(
        data,
        target="target",
        task_type="classification",
        model_names=["logistic_regression", "random_forest_classifier"],
    )
    predictions = automl.predict(data.drop(columns=["target"]))
    print(f"best model: {result['best_model']}")
    print(f"predictions generated: {len(predictions)}")


if __name__ == "__main__":
    main()