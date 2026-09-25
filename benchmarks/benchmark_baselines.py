"""Compare ModelForge with simple sklearn baselines."""

from time import perf_counter

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from modelforge import AutoML


def main() -> None:
    dataset = load_diabetes(as_frame=True)
    data = dataset.frame.rename(columns={"target": "label"})
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    baseline = LinearRegression()
    started = perf_counter()
    baseline.fit(train.drop(columns=["label"]), train["label"])
    baseline_time = perf_counter() - started
    baseline_error = mean_absolute_error(
        test["label"], baseline.predict(test.drop(columns=["label"]))
    )

    automl = AutoML(cv=3, random_state=42)
    started = perf_counter()
    automl.fit(
        train,
        target="label",
        task_type="regression",
        model_names=["linear_regression", "random_forest_regressor"],
    )
    automl_time = perf_counter() - started
    automl_error = mean_absolute_error(
        test["label"], automl.predict(test.drop(columns=["label"]))
    )

    print("ModelForge baseline benchmark")
    print(f"sklearn linear regression: MAE={baseline_error:.4f}, seconds={baseline_time:.3f}")
    print(f"ModelForge ({automl.best_model}): MAE={automl_error:.4f}, seconds={automl_time:.3f}")


if __name__ == "__main__":
    main()