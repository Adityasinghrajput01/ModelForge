from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from modelforge import AutoML
from modelforge.config import ModelForgeConfig
from modelforge.model_registry import ModelRegistry
from modelforge.persistence import ModelPersistence


app = typer.Typer(
    name="modelforge",
    help=(
        "ModelForge - Transparent, "
        "local-first AutoML framework."
    ),
    no_args_is_help=True,
)

experiments_app = typer.Typer(
    name="experiments",
    help="Manage ModelForge experiments.",
    no_args_is_help=True,
)

app.add_typer(
    experiments_app,
    name="experiments",
)

console = Console()


def _safe_value(value, default=""):
    """Convert experiment values into clean CLI strings."""
    if value is None:
        return default

    if isinstance(value, (dict, list, tuple)):
        return default

    return str(value)


def _experiment_target(experiment):
    """Extract the target name from an experiment record."""
    target = experiment.get("target")

    if isinstance(target, dict):
        return target.get("target", "")

    return target or ""


def _experiment_task_type(experiment):
    """Extract task type from an experiment record."""
    task_type = experiment.get("task_type")

    if task_type:
        return task_type

    target = experiment.get("target")

    if isinstance(target, dict):
        return target.get("task_type", "")

    return ""


def _experiment_best_model(experiment):
    """Extract best model from an experiment record."""
    return experiment.get("best_model", "")


def _experiment_run_id(experiment):
    """
    Extract run ID from an experiment record.

    New records store run_id directly.
    Older records may store it inside run_summary.
    """
    run_id = experiment.get("run_id")

    if run_id:
        return run_id

    run_summary = experiment.get("run_summary")

    if isinstance(run_summary, dict):
        return run_summary.get("run_id", "")

    return ""


def _experiment_status(experiment):
    """
    Extract experiment/run status.

    New records may contain status directly.
    Failed records can also expose status through run_summary.
    """
    status = experiment.get("status")

    if status:
        return status

    run_summary = experiment.get("run_summary")

    if isinstance(run_summary, dict):
        return run_summary.get("status", "")

    return ""


@app.command()
def train(
    data: str = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to the training dataset.",
    ),
    target: str | None = typer.Option(
        None,
        "--target",
        "-t",
        help=(
            "Target column. "
            "Can also be supplied through --config."
        ),
    ),
    output: str = typer.Option(
        "model.joblib",
        "--output",
        "--save",
        "-o",
        help=(
            "Path where the best model will be saved. "
            "--save is an alias for --output."
        ),
    ),
    experiment_directory: str = typer.Option(
        ".modelforge/experiments",
        "--experiment-directory",
        help="Directory where experiment artifacts are stored.",
    ),
    task_type: str | None = typer.Option(
        None,
        "--task-type",
        help=(
            "Optional: regression or classification. "
            "Can also be supplied through --config."
        ),
    ),
    objective: str | None = typer.Option(
        None,
        "--objective",
        help=(
            "Ranking objective: balanced, "
            "performance, error, or speed."
        ),
    ),
    cv: int | None = typer.Option(
        None,
        "--cv",
        help="Number of cross-validation folds.",
    ),
    test_size: float | None = typer.Option(
        None,
        "--test-size",
        help="Holdout test-set proportion.",
    ),
    models: str | None = typer.Option(
        None,
        "--models",
        help=(
            "Comma-separated model names. "
            "Overrides models from --config."
        ),
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a ModelForge YAML configuration file.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite an existing model file.",
    ),
):
    """
    Train ModelForge AutoML and save the best pipeline.
    """

    try:
        if not Path(data).exists():
            raise FileNotFoundError(
                f"Dataset not found: {data}"
            )

        if config:
            configuration = ModelForgeConfig.from_file(config)
            config_values = configuration.to_dict()
        else:
            config_values = {}

        resolved_target = (
            target
            if target is not None
            else config_values.get("target")
        )

        resolved_task_type = (
            task_type
            if task_type is not None
            else config_values.get("task_type")
        )

        resolved_objective = (
            objective
            if objective is not None
            else config_values.get(
                "objective",
                "balanced",
            )
        )

        resolved_cv = (
            cv
            if cv is not None
            else config_values.get(
                "cv",
                5,
            )
        )

        resolved_test_size = (
            test_size
            if test_size is not None
            else config_values.get(
                "test_size",
                0.2,
            )
        )

        resolved_experiment_directory = (
            experiment_directory
        )

        if (
            experiment_directory
            == ".modelforge/experiments"
            and config_values.get(
                "experiment_directory"
            )
        ):
            resolved_experiment_directory = (
                config_values["experiment_directory"]
            )

        if not resolved_target:
            raise ValueError(
                "Target is required. "
                "Provide --target or define "
                "'target' in the configuration file."
            )

        if models:
            selected_models = [
                model.strip()
                for model in models.split(",")
                if model.strip()
            ]
        else:
            selected_models = config_values.get(
                "models"
            )

        excluded_columns = config_values.get(
            "excluded_columns",
            [],
        )

        feature_selection = config_values.get(
            "feature_selection",
            {},
        )

        variance_threshold = feature_selection.get(
            "variance_threshold"
        )

        correlation_threshold = feature_selection.get(
            "correlation_threshold"
        )

        console.print()

        console.print(
            "[bold cyan]ModelForge[/bold cyan]"
        )

        console.print(
            "[dim]Automated ML Pipeline Discovery[/dim]"
        )

        console.print()

        console.print(
            f"[cyan]Dataset:[/cyan] {data}"
        )

        console.print(
            f"[cyan]Target:[/cyan] "
            f"{resolved_target}"
        )

        console.print(
            f"[cyan]Objective:[/cyan] "
            f"{resolved_objective}"
        )

        console.print(
            f"[cyan]CV folds:[/cyan] "
            f"{resolved_cv}"
        )

        console.print(
            f"[cyan]Test size:[/cyan] "
            f"{resolved_test_size}"
        )

        console.print(
            f"[cyan]Variance threshold:[/cyan] "
            f"{variance_threshold}"
        )

        console.print(
            f"[cyan]Correlation threshold:[/cyan] "
            f"{correlation_threshold}"
        )

        console.print(
            f"[cyan]Experiment directory:[/cyan] "
            f"{resolved_experiment_directory}"
        )

        if config:
            console.print(
                f"[cyan]Config:[/cyan] "
                f"{config}"
            )

        console.print()

        automl = AutoML(
            test_size=resolved_test_size,
            cv=resolved_cv,
            objective=resolved_objective,
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
            experiment_directory=(
                resolved_experiment_directory
            ),
        )

        with console.status(
            "[bold green]Running AutoML...[/bold green]"
        ):
            result = automl.fit(
                data=data,
                target=resolved_target,
                task_type=resolved_task_type,
                model_names=selected_models,
                excluded_columns=excluded_columns,
                print_report=False,
            )

        model_path = automl.save(
            output,
            overwrite=overwrite,
        )

        console.print(
            "[bold green]✓ Training completed[/bold green]"
        )

        console.print()

        summary_table = Table(
            title="ModelForge Training Summary"
        )

        summary_table.add_column(
            "Property",
            style="cyan",
        )

        summary_table.add_column(
            "Value",
            style="green",
        )

        summary_table.add_row(
            "Target",
            result["target"]["target"],
        )

        summary_table.add_row(
            "Task Type",
            result["target"]["task_type"],
        )

        summary_table.add_row(
            "Models Evaluated",
            str(result["models_evaluated"]),
        )

        summary_table.add_row(
            "Best Model",
            result["best_model"],
        )

        summary_table.add_row(
            "Ranking Objective",
            resolved_objective,
        )

        summary_table.add_row(
            "Variance Threshold",
            str(variance_threshold),
        )

        summary_table.add_row(
            "Correlation Threshold",
            str(correlation_threshold),
        )

        summary_table.add_row(
            "Run ID",
            _safe_value(
                result.get("run_id"),
                "N/A",
            ),
        )

        summary_table.add_row(
            "Experiment ID",
            _safe_value(
                result.get("experiment_id"),
                "N/A",
            ),
        )

        summary_table.add_row(
            "Saved Model",
            model_path,
        )

        console.print(summary_table)

        console.print()

        ranking = result["ranking"]

        display_columns = [
            column
            for column in [
                "rank",
                "model",
                "overall_score",
                "status",
            ]
            if column in ranking.columns
        ]

        ranking_table = Table(
            title="Model Ranking"
        )

        for column in display_columns:
            ranking_table.add_column(
                column.replace(
                    "_",
                    " ",
                ).title()
            )

        for _, row in ranking[
            display_columns
        ].head(10).iterrows():

            values = []

            for column in display_columns:
                value = row[column]

                if isinstance(
                    value,
                    float,
                ):
                    values.append(
                        f"{value:.4f}"
                    )
                else:
                    values.append(
                        str(value)
                    )

            ranking_table.add_row(
                *values
            )

        console.print(
            ranking_table
        )

    except Exception as exc:
        console.print(
            f"[bold red]✗ Training failed:[/bold red] "
            f"{exc}"
        )

        raise typer.Exit(
            code=1
        )


@app.command()
def predict(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to a saved ModelForge model.",
    ),
    data: str = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to the prediction CSV.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional CSV output path.",
    ),
    proba: bool = typer.Option(
        False,
        "--proba",
        help=(
            "Generate class probabilities instead of "
            "class predictions. Only supported for "
            "classification models."
        ),
    ),
):
    """
    Generate predictions using a saved ModelForge model.
    """

    try:
        if not Path(model).exists():
            raise FileNotFoundError(
                f"Model not found: {model}"
            )

        if not Path(data).exists():
            raise FileNotFoundError(
                f"Dataset not found: {data}"
            )

        persistence = ModelPersistence()

        pipeline = persistence.load(
            model
        )

        if proba:
            probabilities = (
                persistence.predict_proba_from_file(
                    pipeline,
                    data,
                )
            )

            if output:
                probabilities.to_csv(
                    output,
                    index=False,
                )

                console.print(
                    "[bold green]✓ Prediction "
                    "probabilities saved:[/bold green] "
                    f"{output}"
                )
            else:
                console.print()

                console.print(
                    "[bold cyan]Prediction "
                    "Probabilities[/bold cyan]"
                )

                console.print()

                for index, row in probabilities.iterrows():
                    values = ", ".join(
                        f"{column}={row[column]:.6f}"
                        for column in probabilities.columns
                    )

                    console.print(
                        f"{index}: {values}"
                    )

            console.print()

            console.print(
                f"[green]✓ Generated probabilities "
                f"for {len(probabilities)} samples.[/green]"
            )

        else:
            predictions = (
                persistence.predict_from_file(
                    pipeline,
                    data,
                )
            )

            if output:
                predictions.to_frame().to_csv(
                    output,
                    index=False,
                )

                console.print(
                    "[bold green]✓ Predictions saved:[/bold green] "
                    f"{output}"
                )
            else:
                console.print()

                console.print(
                    "[bold cyan]Predictions[/bold cyan]"
                )

                console.print()

                for (
                    index,
                    prediction,
                ) in enumerate(
                    predictions.tolist()
                ):
                    console.print(
                        f"{index}: {prediction}"
                    )

            console.print()

            console.print(
                f"[green]✓ Generated "
                f"{len(predictions)} predictions.[/green]"
            )

    except Exception as exc:
        console.print(
            f"[bold red]✗ Prediction failed:[/bold red] "
            f"{exc}"
        )

        raise typer.Exit(
            code=1
        )


@app.command()
def explain(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to a saved ModelForge model.",
    ),
    top: int = typer.Option(
        10,
        "--top",
        "-n",
        help="Number of important features to display.",
    ),
):
    """
    Display feature importance for a saved model.
    """

    try:
        if not Path(model).exists():
            raise FileNotFoundError(
                f"Model not found: {model}"
            )

        persistence = ModelPersistence()

        pipeline = persistence.load(
            model
        )

        from modelforge.explainability import (
            ExplainabilityEngine,
        )

        engine = ExplainabilityEngine()

        importance = (
            engine.feature_importance(
                pipeline
            )
        )

        top_features = (
            engine.top_features(
                importance,
                n=top,
            )
        )

        table = Table(
            title="Model Explainability"
        )

        table.add_column(
            "Rank",
            style="cyan",
        )

        table.add_column(
            "Feature",
            style="green",
        )

        table.add_column(
            "Importance",
            style="yellow",
        )

        table.add_column(
            "Source",
            style="magenta",
        )

        for (
            rank,
            (_, row),
        ) in enumerate(
            top_features.iterrows(),
            start=1,
        ):
            table.add_row(
                str(rank),
                str(row["feature"]),
                f"{row['importance']:.6f}",
                str(row["source"]),
            )

        console.print()

        console.print(table)

    except Exception as exc:
        console.print(
            f"[bold red]✗ Explanation failed:[/bold red] "
            f"{exc}"
        )

        raise typer.Exit(
            code=1
        )


@app.command(name="models")
def list_models(
    task_type: str | None = typer.Option(
        None,
        "--task-type",
        help="Optional: regression or classification.",
    ),
):
    """
    List models available in ModelForge.
    """

    try:
        registry = ModelRegistry()

        if task_type is not None:
            task_type = (
                task_type
                .lower()
                .strip()
            )

            if task_type not in {
                "regression",
                "classification",
            }:
                raise ValueError(
                    "task_type must be "
                    "'regression' or "
                    "'classification'."
                )

        model_names = registry.list_models(
            task_type=task_type
        )

        models = [
            registry.get(name)
            for name in model_names
        ]

        table = Table(
            title="ModelForge Model Registry",
            expand=True,
        )

        table.add_column(
            "Name",
            style="cyan",
            no_wrap=True,
        )

        table.add_column(
            "Task",
            style="green",
            no_wrap=True,
        )

        table.add_column(
            "Category",
            style="yellow",
            no_wrap=True,
        )

        table.add_column(
            "Scaling",
            style="magenta",
            no_wrap=True,
        )

        table.add_column(
            "Probability",
            style="blue",
            no_wrap=True,
        )

        for spec in models:
            table.add_row(
                spec.name,
                spec.task_type,
                spec.category,
                (
                    "Yes"
                    if spec.requires_scaling
                    else "No"
                ),
                (
                    "Yes"
                    if spec.supports_probability
                    else "No"
                ),
            )

        console.print()

        console.print(table)

        console.print()

        console.print(
            "[bold cyan]Available models:[/bold cyan]"
        )

        for name in model_names:
            console.print(
                f"  {name}"
            )

        console.print()

        console.print(
            f"[green]Total models: "
            f"{len(models)}[/green]"
        )

    except Exception as exc:
        console.print(
            f"[bold red]✗ Failed to list models:[/bold red] "
            f"{exc}"
        )

        raise typer.Exit(
            code=1
        )


@experiments_app.command(name="list")
def list_experiments(
    directory: str = typer.Option(
        ".modelforge/experiments",
        "--directory",
        "-d",
        help="Experiment storage directory.",
    ),
):
    """
    List recorded ModelForge experiments.
    """

    try:
        automl = AutoML(
            experiment_directory=directory
        )

        experiments = (
            automl.list_experiments()
        )

        console.print()
        console.print(
            "[bold cyan]ModelForge Experiments[/bold cyan]"
        )
        console.print()

        if not experiments:
            console.print(
                "[dim]No experiments found.[/dim]"
            )
            console.print()

        for experiment in experiments:
            experiment_id = (
                experiment.get(
                    "experiment_id",
                    "",
                )
            )

            status = _experiment_status(
                experiment
            )

            target = _experiment_target(
                experiment
            )

            task_type = _experiment_task_type(
                experiment
            )

            best_model = _experiment_best_model(
                experiment
            )

            run_id = _experiment_run_id(
                experiment
            )

            console.print(
                f"Experiment ID: "
                f"{_safe_value(experiment_id, 'N/A')}"
            )

            console.print(
                f"  Status: "
                f"{_safe_value(status, 'N/A')}"
            )

            console.print(
                f"  Target: "
                f"{_safe_value(target, 'N/A')}"
            )

            console.print(
                f"  Task: "
                f"{_safe_value(task_type, 'N/A')}"
            )

            console.print(
                f"  Best Model: "
                f"{_safe_value(best_model, 'N/A')}"
            )

            console.print(
                f"  Run ID: "
                f"{_safe_value(run_id, 'N/A')}"
            )

            console.print()

        console.print(
            f"[green]Total experiments: "
            f"{len(experiments)}[/green]"
        )

    except Exception as exc:
        console.print(
            f"[bold red]✗ Failed to list "
            f"experiments:[/bold red] "
            f"{exc}"
        )

        raise typer.Exit(
            code=1
        )


@experiments_app.command(name="get")
def get_experiment(
    experiment_id: str = typer.Argument(
        ...,
        help="Experiment ID to inspect.",
    ),
    directory: str = typer.Option(
        ".modelforge/experiments",
        "--directory",
        "-d",
        help="Experiment storage directory.",
    ),
):
    """
    Display details for a ModelForge experiment.
    """

    try:
        automl = AutoML(
            experiment_directory=directory
        )

        experiment = automl.get_experiment(
            experiment_id
        )

        console.print()

        console.print(
            f"[bold cyan]Experiment:[/bold cyan] "
            f"{experiment_id}"
        )

        console.print()

        summary_table = Table(
            title="Experiment Summary"
        )

        summary_table.add_column(
            "Property",
            style="cyan",
        )

        summary_table.add_column(
            "Value",
            style="green",
        )

        summary_table.add_row(
            "Experiment ID",
            _safe_value(
                experiment.get(
                    "experiment_id"
                ),
                experiment_id,
            ),
        )

        summary_table.add_row(
            "Status",
            _safe_value(
                _experiment_status(
                    experiment
                ),
                "N/A",
            ),
        )

        summary_table.add_row(
            "Target",
            _safe_value(
                _experiment_target(
                    experiment
                ),
                "N/A",
            ),
        )

        summary_table.add_row(
            "Task Type",
            _safe_value(
                _experiment_task_type(
                    experiment
                ),
                "N/A",
            ),
        )

        summary_table.add_row(
            "Best Model",
            _safe_value(
                _experiment_best_model(
                    experiment
                ),
                "N/A",
            ),
        )

        summary_table.add_row(
            "Run ID",
            _safe_value(
                _experiment_run_id(
                    experiment
                ),
                "N/A",
            ),
        )

        if "configuration" in experiment:
            summary_table.add_row(
                "Configuration",
                str(
                    experiment["configuration"]
                ),
            )

        console.print(
            summary_table
        )

        console.print()

        console.print(
            "[bold cyan]Experiment Data[/bold cyan]"
        )

        console.print()

        details = Table(
            show_header=True,
            header_style="bold cyan",
        )

        details.add_column(
            "Key"
        )

        details.add_column(
            "Value"
        )

        for key, value in experiment.items():
            if key in {
                "experiment_id",
                "status",
                "target",
                "task_type",
                "best_model",
                "run_id",
                "configuration",
            }:
                continue

            details.add_row(
                str(key),
                str(value),
            )

        if details.row_count > 0:
            console.print(details)

    except Exception as exc:
        console.print(
            f"[bold red]✗ Failed to get "
            f"experiment:[/bold red] "
            f"{exc}"
        )

        raise typer.Exit(
            code=1
        )


@experiments_app.command(name="compare")
def compare_experiments(
    experiment_ids: list[str] = typer.Argument(
        ...,
        help="Two or more experiment IDs to compare.",
    ),
    directory: str = typer.Option(
        ".modelforge/experiments",
        "--directory",
        "-d",
        help="Experiment storage directory.",
    ),
):
    """Compare the key outcomes and configuration of experiments."""

    if len(experiment_ids) < 2:
        console.print(
            "[bold red]✗ Provide at least two experiment IDs.[/bold red]"
        )
        raise typer.Exit(code=1)

    try:
        automl = AutoML(experiment_directory=directory)
        table = Table(title="ModelForge Experiment Comparison")
        table.add_column("Experiment", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Task", style="yellow")
        table.add_column("Target", style="magenta")
        table.add_column("Best Model", style="blue")
        table.add_column("Run ID", style="dim")

        for experiment_id in experiment_ids:
            experiment = automl.get_experiment(experiment_id)
            table.add_row(
                experiment_id,
                _safe_value(_experiment_status(experiment), "N/A"),
                _safe_value(_experiment_task_type(experiment), "N/A"),
                _safe_value(_experiment_target(experiment), "N/A"),
                _safe_value(_experiment_best_model(experiment), "N/A"),
                _safe_value(_experiment_run_id(experiment), "N/A"),
            )

        console.print(table)

    except Exception as exc:
        console.print(
            f"[bold red]✗ Failed to compare experiments:[/bold red] {exc}"
        )
        raise typer.Exit(code=1)


@app.callback()
def main():
    """
    ModelForge CLI entry point.
    """
    pass


if __name__ == "__main__":
    app()