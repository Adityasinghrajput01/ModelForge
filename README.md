<div align="center">

# 🔨 ModelForge

**A transparent, local-first AutoML framework for Python**

[![PyPI version](https://img.shields.io/pypi/v/autoforge-engine.svg)](https://pypi.org/project/autoforge-engine/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://pypi.org/project/autoforge-engine/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-491%20passing-brightgreen.svg)](#testing--validation)
[![GitHub](https://img.shields.io/badge/GitHub-ModelForge-181717?logo=github)](https://github.com/Adityasinghrajput01/ModelForge)

*Give it a dataset and a target column. It handles the rest — transparently.*

[Installation](#installation) •
[Quick Start](#quick-start) •
[Configuration](#configuration) •
[Architecture](#architecture) •
[Stress Test Results](#stress-test-results) •
[Roadmap](#roadmap)

</div>

---

## What is ModelForge?

**ModelForge** is a local-first AutoML framework that automates the repetitive parts of a machine learning workflow — data profiling, quality auditing, preprocessing, pipeline construction, model evaluation, cross-validation, ranking, and reporting — while staying inspectable at every step.

Unlike black-box AutoML tools, ModelForge is built around one core principle:

> **You should always be able to see what it did and why.**

Give it a dataset and a target column:

```python
import pandas as pd
from modelforge.automl import AutoML

df = pd.read_csv("dataset.csv")

automl = AutoML()
automl.fit(df, target="target_column")
```

And it walks through the full pipeline automatically:

```
Dataset → Loading → Profiling → Data Quality Audit → Target/Task Detection
        → Column Intelligence → Feature Selection/Engineering → Preprocessing
        → Pipeline Generation → Multi-Model Evaluation → Cross-Validation
        → Metric Calculation → Model Ranking → Best Model Selection
        → (Optional) Hyperparameter Optimization → Explainability
        → Persistence/Artifacts → Experiment Tracking → Human-Readable Report
```

---

## Installation

```bash
pip install autoforge-engine
```

Install a specific version:

```bash
pip install autoforge-engine==0.1.4
```

Verify the install:

```bash
pip show autoforge-engine
python -c "import modelforge; print(modelforge.__file__)"
```

> **Note:** the PyPI distribution name is `autoforge-engine`, but the importable package is `modelforge`.

---

## Quick Start

### Using the `AutoML` class

```python
import pandas as pd
from modelforge.automl import AutoML

df = pd.read_csv("autoforge_10000_test_dataset.csv")

automl = AutoML()
automl.fit(df, target="loan_default")
```

### Using the convenience function

```python
from modelforge.automl import automl

result = automl(
    data=df,
    target="loan_default"
)
```

### Command line

```bash
modelforge --help
```

---

## Configuration

The `AutoML` constructor supports the following parameters:

```python
AutoML(
    test_size=0.2,
    cv=5,
    random_state=42,
    objective="balanced",
    variance_threshold=None,
    correlation_threshold=None,
    enable_optimization=False,
    optimization_models=3,
    optimization_max_trials=10,
    experiment_directory=".modelforge/experiments",
    config=None
)
```

| Parameter | Description |
|---|---|
| `test_size` | Holdout test fraction |
| `cv` | Number of cross-validation folds |
| `random_state` | Reproducibility seed |
| `objective` | Model ranking objective (default: `"balanced"`) |
| `variance_threshold` | Optional low-variance feature filtering |
| `correlation_threshold` | Optional high-correlation feature filtering |
| `enable_optimization` | Enables/disables hyperparameter optimization |
| `optimization_models` | Number of models considered for optimization |
| `optimization_max_trials` | Maximum optimization trials |
| `experiment_directory` | Where experiment metadata/artifacts are stored |

The automatic post-fit report can be suppressed with `print_report=False` (if supported by the installed `fit()` signature).

---

## What ModelForge Does

### 🧹 Data Quality Auditing
Automatically flags:
- Missing values
- Duplicate rows
- Constant columns
- Possible identifier columns
- Other dataset-quality findings

### 🎯 Target & Task Detection
Given a target column, ModelForge determines whether the problem is **classification** or **regression**.

### ⚙️ Preprocessing
Builds a leakage-safe `scikit-learn` `Pipeline` / `ColumnTransformer`:

| Data type | Steps |
|---|---|
| Numerical | `SimpleImputer` → `StandardScaler` |
| Categorical | `SimpleImputer` → `OneHotEncoder` |

Preprocessing is fit independently inside each cross-validation fold to prevent data leakage.

### 🤖 Model Evaluation
Evaluates multiple candidate models (verified against source for the current release), including — for classification tasks — Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, K-Nearest Neighbors, Support Vector Classifier, and Decision Tree.

### 🔁 Cross-Validation
Uses `StratifiedKFold` (`n_splits=5`, `shuffle=True`, `random_state=42`) for classification tasks where class counts permit. Each fold clones the pipeline fresh, fits on the training split, and evaluates on the validation split.

### 📊 Metrics

**Classification:** Accuracy, Precision, Recall, F1 (`average="weighted"`, `zero_division=0`), ROC-AUC (via `predict_proba` or `decision_function`), Log Loss.

**Regression:** R², Adjusted R², MAE, MSE, RMSE, MAPE.

### 🏆 Model Ranking
Candidate models are ranked according to the configured `objective` (default: `"balanced"`) rather than by a single metric alone.

### 📄 Automatic Reporting
After `fit()`, ModelForge generates a human-readable report covering dataset shape, target/task, data-quality findings, evaluated models and metrics, the selected model, pipeline steps, run/experiment IDs, and run duration.

---

## Architecture

ModelForge is organized into focused modules, each with a single responsibility:

| Module | Responsibility |
|---|---|
| `data_loader.py` | Dataset loading |
| `profiler.py` | Dataset profiling |
| `data_audit.py` | Data quality / auditing |
| `target_selector.py` | Target and task detection |
| `column_intelligence.py` | Column-level analysis |
| `feature_engineering.py` | Feature engineering |
| `feature_selection.py` | Feature selection |
| `preprocessing.py` | Imputation, scaling, encoding |
| `pipeline_generator.py` | sklearn pipeline construction |
| `model_registry.py` | Candidate model catalog |
| `model_screening.py` | Fast holdout evaluation |
| `cross_validation.py` | K-fold cross-validation |
| `evaluation.py` | Metric computation |
| `ranking.py` | Model ranking logic |
| `hyperparameter_optimization.py` | Optional HPO |
| `explainability.py` | Model explainability |
| `persistence.py` | Model save/load |
| `artifact_manager.py` | Artifact management |
| `experiment_tracker.py` | Experiment tracking |
| `reproducibility.py` / `reproducibility_integration.py` | Reproducibility guarantees |
| `prediction_validator.py` | Prediction-time validation |
| `run_manager.py` | Run metadata |
| `cli.py` | Typer-based CLI |
| `automl.py` | Orchestration layer |
| `config.py` | Configuration handling |

> **Note on Model Screening vs. Cross-Validation:** these are distinct evaluation paths. `ModelScreeningEngine` performs a fast `train_test_split` (80/20, stratified where applicable) holdout evaluation, while `CrossValidationEngine` performs full K-fold evaluation. Holdout screening metrics and CV metrics should not be conflated.

---

## Stress Test Results

ModelForge 0.1.4 was validated against a synthetic 10,020-row, 18-column dataset (target: `loan_default`) intentionally containing missing values, duplicates, a constant feature, and an identifier-like column.

**Data quality detected:** 1,000 missing cells · 20 duplicate rows · 3 quality findings (all correctly identified)

**Cross-validation accuracy:**

| Model | CV Accuracy |
|---|---|
| Logistic Regression | 69.51% |
| Gradient Boosting | 69.51% |
| SVC | 69.12% |
| Random Forest | 69.02% |
| Extra Trees | 68.26% |
| KNN | 64.52% |
| Decision Tree | 58.59% |

**Selected model:** Logistic Regression · **Run time:** ~33.25 seconds

> These figures are a functional stress-test snapshot, not a claim that any single algorithm is universally best.

### Independent Validation

Results were cross-checked against raw `scikit-learn` implementations. Six of seven models matched almost exactly; Decision Tree showed a ~0.13 percentage-point difference (58.59% vs. 58.72%), currently logged as an open, low-priority investigation rather than a confirmed defect.

---

## Testing & Validation

- ✅ 491 tests passing locally
- ✅ GitHub Actions CI passing
- ✅ Package built (`.tar.gz` + `.whl`) and validated with `twine check`
- ✅ Published to PyPI: [`autoforge-engine`](https://pypi.org/project/autoforge-engine/0.1.4/)

Test coverage includes unit, integration, CLI, preprocessing, model evaluation, cross-validation, ranking, persistence, experiment tracking, edge cases, and large mixed-type stress tests.

---

## Development Workflow

```bash
# Run tests
pytest

# Build the package
python -m build

# Validate the build
python -m twine check dist/*

# Upload to PyPI
python -m twine upload dist/*
```

---

## Roadmap

- [ ] Investigate the minor Decision Tree CV discrepancy (`model_registry` / `pipeline_generator` / `preprocessing`)
- [ ] Expand hyperparameter optimization coverage
- [ ] Broaden regression model support
- [ ] Deepen explainability outputs
- [ ] Continued CLI and documentation improvements

---

## Philosophy

ModelForge is built to demonstrate serious ML engineering practice — not to replace a data scientist's judgment. It automates repeatable, mechanical steps of an ML workflow (preprocessing, evaluation, cross-validation, ranking, reporting) while keeping every step reproducible and inspectable, so the framework never becomes a black box.

---

## Author

**Aditya Kumar Singh**

## License

[MIT](LICENSE)

## Links

- 📦 PyPI: [autoforge-engine](https://pypi.org/project/autoforge-engine/)
- 💻 GitHub: [ModelForge](https://github.com/Adityasinghrajput01/ModelForge)
