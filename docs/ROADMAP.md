# ModelForge Roadmap

## Next release: usability and release quality

- Expand the README with API, configuration, troubleshooting, and artifact
  lifecycle documentation.
- Keep the end-to-end regression and classification examples runnable in CI.
- Continue improving CLI summaries with consistent tables, status, and timing.
- Publish benchmark results for representative small, medium, and wide data.
- Add package metadata, release checks, and generated documentation before the
  first PyPI release.

## Advanced capabilities

| Area | Direction | Dependency or design note |
| --- | --- | --- |
| Advanced feature engineering | Date parts, ratios, interactions, skew transforms, and cardinality-aware encoding | Must remain leakage-safe inside pipelines |
| Advanced model support | XGBoost, LightGBM, and CatBoost registry adapters | Optional extras; unavailable integrations should fail clearly |
| Better HPO | Optuna sampler and pruner backend | Preserve the current deterministic grid-search fallback |
| Ensembling | Voting, stacking, and blending candidates | Needs task-aware validation and explainability rules |
| Time-series AutoML | Temporal splits, lag features, rolling features, and horizon-aware metrics | Never use random CV for temporal data |
| Experiment comparison | CLI and Python APIs for comparing metrics, configuration, and fingerprints | Current CLI supports pairwise summary comparison |
| Artifact management | Versioned model, report, metadata, and dataset bundles | Extend `ArtifactManager` without breaking existing artifacts |
| Production packaging | Build, metadata validation, supported-Python matrix, and PyPI publishing | Release only after reproducible CI builds |
| CI/CD | Tests, lint, type checks, build, and optional dependency jobs | Keep the default job lightweight |
| Performance benchmarking | Runtime, memory, scaling, and dataset-shape reports | Store machine and dependency metadata with results |

## Completion criteria

Each advanced area is complete when it has a public API, focused tests,
documentation, failure behavior for missing optional dependencies, and a
reproducible example or benchmark.