 # ModelForge

ModelForge is a transparent, local-first AutoML framework for discovering,
evaluating, explaining, and persisting scikit-learn pipelines. It keeps the
workflow inspectable: data audits, preprocessing, model screening,
cross-validation, ranking, reproducibility metadata, and artifacts are all
available as Python objects and local files.

## Quick start

```bash
python -m pip install -e ".[dev]"
modelforge train --data examples/house_prices.csv --target price \
	--task-type regression --output model.joblib
modelforge predict --model model.joblib --data examples/house_prices_new.csv
```

The same workflow is available from Python:

```python
from modelforge import AutoML

automl = AutoML(cv=5, random_state=42)
result = automl.fit(data, target="price", task_type="regression")
automl.save("artifacts/house-prices.joblib", overwrite=True)
print(result["best_model"])
```

## What is included

ModelForge currently provides ingestion, target detection, profiling, column
intelligence, data-quality and leakage auditing, preprocessing, feature
engineering and selection, a model registry, screening, cross-validation,
ranking, explainability, persistence, prediction validation, optimization,
experiment tracking, run management, configuration, CLI workflows, failure
isolation, robustness checks, and reproducibility snapshots.

Optional model and optimization integrations are declared in `pyproject.toml`:

```bash
python -m pip install -e ".[boosting,optimization]"
```

## CLI

```bash
modelforge --help
modelforge models --task-type classification
modelforge experiments list --directory .modelforge/experiments
modelforge experiments get EXPERIMENT_ID
modelforge experiments compare ID_ONE ID_TWO
```

The CLI uses Rich tables and keeps generated models, experiment JSON, and
metadata local by default.

## Examples and benchmarks

- `examples/end_to_end_regression.py` demonstrates a complete Python workflow.
- `examples/end_to_end_classification.py` demonstrates classification and
	probability prediction.
- `benchmarks/benchmark_baselines.py` compares ModelForge with simple baseline
	estimators and records runtime and quality metrics.

Run an example with `python examples/end_to_end_regression.py` or run the
benchmark with `python benchmarks/benchmark_baselines.py`.

## Development

```bash
.venv\\Scripts\\python.exe -m pytest -q
ruff check .
```

GitHub Actions runs the test suite on supported Python versions. The roadmap
for the remaining product work lives in `docs/ROADMAP.md`; it separates
implemented foundations from integrations that require optional dependencies
or additional design work.
