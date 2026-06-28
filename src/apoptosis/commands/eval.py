from pathlib import Path

import typer

from apoptosis.app import app
from apoptosis.core.session import PROJECT_ROOT
from apoptosis.services.dataset_build import default_dataset_dir
from apoptosis.services.evaluate import evaluate_checkpoint


def _default_checkpoint() -> Path:
    root = PROJECT_ROOT / "runs" / "viability" / "lightning_logs"
    candidates = sorted(
        root.rglob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if "best-" in path.name:
            return path
    if candidates:
        return candidates[0]
    msg = f"No checkpoints found under {root}"
    raise FileNotFoundError(msg)


@app.command()
def eval(
    checkpoint: Path = typer.Option(
        _default_checkpoint(),
        "--checkpoint",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    manifest_path: Path = typer.Option(
        default_dataset_dir(PROJECT_ROOT) / "manifest.json",
        "--manifest",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    batch_size: int = typer.Option(64, min=1),
    num_workers: int = typer.Option(4, min=0),
    accelerator: str = typer.Option("auto", help="auto, gpu, or cpu"),
) -> None:
    """Evaluate a trained viability model on train and val splits."""
    typer.echo(f"Checkpoint: {checkpoint}")
    results = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        manifest_path=manifest_path,
        batch_size=batch_size,
        num_workers=num_workers,
        accelerator=accelerator,
    )
    for metrics in results:
        typer.echo("")
        typer.echo(f"[{metrics.split}] {metrics.samples} frames")
        typer.echo(
            f"  accuracy={metrics.accuracy:.3f}  "
            f"f1={metrics.f1:.3f}  "
            f"precision={metrics.precision:.3f}  "
            f"recall={metrics.recall:.3f}"
        )
        typer.echo(
            f"  ground truth: {metrics.true_viable} viable / {metrics.true_dead} dead"
        )
        typer.echo(
            f"  predicted:    {metrics.pred_viable} viable / {metrics.pred_dead} dead"
        )
        typer.echo(
            "  confusion [[TN, FP], [FN, TP]] "
            f"(label 0=dead, 1=viable): {metrics.confusion}"
        )
