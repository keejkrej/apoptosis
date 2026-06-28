import typer


def greet(name: str) -> None:
    typer.echo(f"Hello, {name}!")
