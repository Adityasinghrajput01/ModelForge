# ModelForge

> **Transparent, local-first AutoML experimentation for reproducible and inspectable machine learning.**

ModelForge is a Python AutoML library for preparing data, screening and ranking
scikit-learn pipelines, evaluating models, and saving models for later
predictions. Training runs also record experiment and reproducibility
information locally.

A typical ModelForge workflow profiles a dataset, selects a target, audits and
preprocesses its features, generates candidate pipelines, screens and ranks
models, and saves the chosen pipeline for prediction. The workflow stays
inspectable rather than hiding every stage behind a black-box call.

The package is published as **`autoforge-engine`** and imported in Python as
**`modelforge`**.

## Features

- Regression and classification workflows
- Data profiling, column intelligence, and data-quality auditing
- Preprocessing, feature engineering, and feature selection
- Model screening, cross-validation, and ranking
- Saved model pipelines and predictions from the command line or Python
- Local experiment tracking, run metadata, and reproducibility information
- Optional boosting models and hyperparameter optimization

## Requirements

- Python 3.11 or newer
- A local dataset in a supported format

The default installation includes NumPy, pandas, scikit-learn, Rich, Typer,
and PyYAML.

## Install

### Install from PyPI

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install autoforge-engine
```

If PowerShell does not allow virtual-environment activation, you can call its
Python executable directly:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install autoforge-engine
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install autoforge-engine
```

Optional integrations can be installed with extras:

```bash
python -m pip install "autoforge-engine[boosting,optimization]"
```

The `boosting` extra installs XGBoost, LightGBM, and CatBoost. The
`optimization` extra installs Optuna.

### Install from source

Clone the repository, enter its directory, and install the development extras.
This makes the `modelforge` command and test tools available in the active
Python environment.

```bash
git clone https://github.com/Adityasinghrajput01/ModelForge.git
cd ModelForge
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

To install optional integrations from a source checkout, use
`python -m pip install -e ".[dev,boosting,optimization]"`.

## Dataset and file paths

ModelForge reads local files. Training through the Python API supports CSV,
Excel (`.xlsx` or `.xls`), Parquet, and JSON files. The CLI prediction command
expects a CSV file.

The training data must have a header row. Choose the column you want the model
to predict as the **target**; all other usable columns become input features.
For example, a regression CSV might look like this:

```csv
area,bedrooms,age,price
1200,2,15,250000
1850,3,8,385000
900,1,30,190000
```

Here, `price` is the target. A prediction CSV should contain the feature
columns (`area`, `bedrooms`, and `age`) with compatible names, but should not
contain the target column.

Paths are interpreted from the directory where you run the command or Python
script:

- A **relative path** such as `data/train.csv` starts from the current working
	directory. Run commands from the repository root when using paths such as
	`examples/house_prices.csv`.
- An **absolute path** works from any directory. On Windows, quote paths that
	contain spaces, for example `"C:\Users\Aditya Kumar\Datasets\train.csv"`.
- In Python, use a raw string for a Windows path, such as
	`r"C:\Users\Aditya Kumar\Datasets\train.csv"`, or use forward slashes:
	`"C:/Users/Aditya Kumar/Datasets/train.csv"`.

## Train and predict with the CLI

After installation, check that the command is available:

```bash
modelforge --help
```

From the repository root, train a regression model using the included housing
dataset. The target column is `price`:

```bash
modelforge train \
	--data "examples/house_prices.csv" \
	--target price \
	--task-type regression \
	--output "house_prices_model.joblib"
```

Run the saved model on new rows. The example prediction file contains features
without the `price` target:

```bash
modelforge predict \
	--model "house_prices_model.joblib" \
	--data "examples/house_prices_new.csv" \
	--output "predictions.csv"
```

On Windows PowerShell, the same commands can be written on one line, or use a
backtick at the end of each continued line. For a dataset outside the
repository, supply its absolute path to `--data`.

For classification, use a classification dataset, provide its label column,
and set `--task-type classification`. For example, the included Iris dataset
uses `species` as its label:

```bash
modelforge train --data "examples/iris_classification.csv" --target species --task-type classification --output "iris_model.joblib"
```

To return class probabilities instead of predicted class labels, add
`--proba` to `modelforge predict`. This option is only for classification
models.

Useful CLI commands:

```bash
modelforge train --help
modelforge predict --help
modelforge models --task-type classification
modelforge experiments list --directory .modelforge/experiments
modelforge experiments get EXPERIMENT_ID
modelforge experiments compare ID_ONE ID_TWO
```

By default, experiment records are written to `.modelforge/experiments` in
the current working directory. The model is saved to the path passed to
`--output` (default: `model.joblib`). Use `--overwrite` to replace an existing
model file.

## Use ModelForge from Python

For a one-call workflow that prints a dataset, model-ranking, data-quality,
and run report, use the lowercase `automl` helper. It accepts a file path or a
pandas DataFrame and returns the fitted `AutoML` instance:

```python
from modelforge import automl

run = automl("data/heart_failure.csv", "DEATH_EVENT")
predictions = run.predict("data/new_patients.csv")
```

Pass options such as `task_type="classification"`, `cv=5`, or
`model_names=["logistic_regression"]` as keyword arguments when needed.

Pass a pandas DataFrame to `AutoML.fit`, name the target column, then save the
fitted pipeline. Replace the example path with the path to your own dataset.

```python
from pathlib import Path

import pandas as pd
from modelforge import AutoML

data_path = Path("examples/house_prices.csv")
data = pd.read_csv(data_path)

automl = AutoML(cv=5, random_state=42)
result = automl.fit(
		data,
		target="price",
		task_type="regression",
)

model_path = Path("house_prices_model.joblib")
automl.save(str(model_path), overwrite=True)
print("Best model:", result["best_model"])

new_data = pd.read_csv("examples/house_prices_new.csv")
predictions = automl.predict(new_data)
print(predictions.head())
```

For a file path outside the working directory, set `data_path` to an absolute
path, for example `Path(r"C:\Users\Aditya Kumar\Datasets\train.csv")`.

## Configuration

You can put training settings in a YAML file and pass it to the CLI with
`--config`. For example:

```yaml
target: price
task_type: regression
objective: balanced
test_size: 0.2
cv: 5
random_state: 42
experiment_directory: .modelforge/experiments
```

Save this as `modelforge.yaml`, then run:

```bash
modelforge train --data "examples/house_prices.csv" --config "modelforge.yaml" --output "house_prices_model.joblib"
```

The CLI options `--target`, `--task-type`, `--objective`, `--cv`, and
`--test-size` can also be set directly on the command line. See
`examples/modelforge_config.yaml` for a configuration that includes feature
selection settings.

## Run tests and checks

Install the source checkout with the `dev` extra first, then run these from
the repository root with the virtual environment activated:

```bash
python -m pytest -q
ruff check .
```

To run a single test module while developing:

```bash
python -m pytest tests/test_cli_workflow.py -q
```

To exercise the included Python workflows and benchmark:

```bash
python examples/end_to_end_regression.py
python benchmarks/benchmark_baselines.py
```

To verify a classification train-and-predict workflow with the included Iris
files:

```bash
modelforge train --data "examples/iris_classification.csv" --target species --task-type classification --output "iris_model.joblib"
modelforge predict --model "iris_model.joblib" --data "examples/iris_new.csv" --output "iris_predictions.csv"
```

To check the installed CLI and the available model names:

```bash
modelforge --help
modelforge models --task-type regression
modelforge models --task-type classification
```

## Project contents

- `examples/` contains sample datasets, YAML configuration, and end-to-end
	workflows.
- `tests/` contains the automated test suite.
- `benchmarks/` contains a baseline comparison script.
- `docs/ROADMAP.md` describes planned project work.

## License

ModelForge is distributed under the MIT License. See [LICENSE](LICENSE).
