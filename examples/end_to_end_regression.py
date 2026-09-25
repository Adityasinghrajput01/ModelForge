"""Run a complete local ModelForge regression workflow."""

from pathlib import Path

import pandas as pd

from modelforge import AutoML

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "house_prices.csv"


def main() -> None:
    data = pd.read_csv(DATA)
    automl = AutoML(
        cv=3,
        random_state=42,
        experiment_directory=ROOT / ".modelforge" / "experiments",
    )
    result = automl.fit(
        data,
        target="price",
        task_type="regression",
        model_names=["linear_regression", "random_forest_regressor"],
    )
    output = ROOT / ".modelforge" / "artifacts" / "house_prices.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    automl.save(str(output), overwrite=True)
    print(f"best model: {result['best_model']}")
    print(f"experiment: {result['experiment_id']}")
    print(f"artifact: {output}")


if __name__ == "__main__":
    main()