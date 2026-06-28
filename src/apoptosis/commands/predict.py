from pathlib import Path

import typer

from apoptosis.app import app
from apoptosis.core.session import DEFAULT_DATA_DIR, PROJECT_ROOT
from apoptosis.services.inference import (
    _default_checkpoint,
    infer_all_cells,
    save_inference,
)
from apoptosis.services.plot_validation import plot_validation


@app.command()
def predict(
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR,
        "--data-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    checkpoint: Path = typer.Option(
        _default_checkpoint(PROJECT_ROOT),
        "--checkpoint",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output_json: Path = typer.Option(
        PROJECT_ROOT / "runs" / "viability" / "inference.json",
        "--output-json",
        resolve_path=True,
    ),
    output_plot: Path = typer.Option(
        PROJECT_ROOT / "runs" / "viability" / "validation_plot.png",
        "--output-plot",
        resolve_path=True,
    ),
    batch_size: int = typer.Option(64, min=1),
    accelerator: str = typer.Option("auto", help="auto, gpu, or cpu"),
) -> None:
    """Run viability inference on all cells and plot toto vs brightfield timing."""
    typer.echo(f"Inferring {data_dir} with {checkpoint.name}")
    results = infer_all_cells(
        data_dir=data_dir,
        checkpoint_path=checkpoint,
        batch_size=batch_size,
        accelerator=accelerator,
    )
    save_inference(results, output_json)
    plot_validation(results, output_plot)
    typer.echo(f"Wrote {len(results)} cell predictions to {output_json}")
    typer.echo(f"Wrote validation figure to {output_plot}")
