from pathlib import Path

import typer
import uvicorn

from apoptosis.app import app
from apoptosis.core.session import DEFAULT_DATA_DIR, configure_session


@app.command()
def label(
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR,
        "--data-dir",
        help="Experiment root containing roi/, Pos*/ folders.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    labels_path: Path | None = typer.Option(
        None,
        "--labels-path",
        help="Where to store labels JSON (default: <data-dir>/labels.json).",
        resolve_path=True,
    ),
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, help="Enable auto-reload."),
) -> None:
    """Launch the ROI viability labeling webapp."""
    resolved_labels = labels_path or (data_dir / "labels.json")
    configure_session(data_dir=data_dir, labels_path=resolved_labels)
    typer.echo(f"Labeling {data_dir}")
    typer.echo(f"Labels -> {resolved_labels}")
    typer.echo(f"Open http://{host}:{port}")
    uvicorn.run(
        "apoptosis.api:api",
        host=host,
        port=port,
        reload=reload,
        factory=False,
    )
