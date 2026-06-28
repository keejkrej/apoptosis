from pathlib import Path

import typer

from apoptosis.app import app
from apoptosis.core.session import DEFAULT_DATA_DIR, DEFAULT_LABELS_PATH, PROJECT_ROOT
from apoptosis.services.dataset_build import build_dataset_manifest, default_dataset_dir


@app.command("dataset-build")
def dataset_build(
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR,
        "--data-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    labels_path: Path = typer.Option(
        DEFAULT_LABELS_PATH,
        "--labels-path",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        default_dataset_dir(PROJECT_ROOT),
        "--output-dir",
        resolve_path=True,
    ),
    val_fraction: float = typer.Option(0.2, min=0.05, max=0.5),
    seed: int = typer.Option(42, help="Shuffle seed for train/val cell split."),
) -> None:
    """Build a per-frame viability dataset manifest from manual labels."""
    manifest = build_dataset_manifest(
        data_dir=data_dir,
        labels_path=labels_path,
        output_dir=output_dir,
        val_fraction=val_fraction,
        seed=seed,
    )
    typer.echo(f"Wrote {output_dir / 'manifest.json'}")
    typer.echo(
        f"Cells: {manifest.train_cells} train / {manifest.val_cells} val | "
        f"Frames: {manifest.train_samples} train / {manifest.val_samples} val"
    )
    typer.echo(
        f"Train class balance: {manifest.viable_train} viable / "
        f"{manifest.dead_train} dead"
    )
