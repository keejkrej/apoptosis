import typer

from apoptosis.app import app

__version__ = "0.1.0"


@app.command()
def version() -> None:
    """Show the installed version."""
    typer.echo(__version__)
