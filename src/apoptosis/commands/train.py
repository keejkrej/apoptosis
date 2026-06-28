from pathlib import Path

import typer
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from apoptosis.app import app
from apoptosis.core.session import PROJECT_ROOT
from apoptosis.ml.viability_datamodule import ViabilityDataModule
from apoptosis.ml.viability_module import ViabilityModule
from apoptosis.services.dataset_build import default_dataset_dir


@app.command()
def train(
    manifest_path: Path = typer.Option(
        default_dataset_dir(PROJECT_ROOT) / "manifest.json",
        "--manifest",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        PROJECT_ROOT / "runs" / "viability",
        "--output-dir",
        resolve_path=True,
    ),
    epochs: int = typer.Option(20, min=1),
    batch_size: int = typer.Option(32, min=1),
    lr: float = typer.Option(1e-4, min=1e-6),
    num_workers: int = typer.Option(4, min=0),
    accelerator: str = typer.Option("auto", help="auto, gpu, or cpu"),
) -> None:
    """Train a ResNet viability classifier with PyTorch Lightning."""
    datamodule = ViabilityDataModule(
        manifest_path=manifest_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    datamodule.setup()
    model = ViabilityModule(lr=lr, class_weights=datamodule.class_weights())

    checkpoint = ModelCheckpoint(
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        filename="best-epoch={epoch:02d}-val-acc={val/acc:.3f}",
    )
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        default_root_dir=str(output_dir),
        callbacks=[checkpoint],
        log_every_n_steps=10,
    )
    typer.echo(f"Training on {len(datamodule.train_dataset)} train frames")
    typer.echo(f"Validating on {len(datamodule.val_dataset)} val frames")
    typer.echo(f"Class weights: {datamodule.class_weights()}")
    trainer.fit(model, datamodule=datamodule)
    typer.echo(f"Best checkpoint: {checkpoint.best_model_path}")
