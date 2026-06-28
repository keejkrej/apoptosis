from apoptosis.app import app
from apoptosis.services.hello import run_hello


@app.command()
def hello(name: str = "world") -> None:
    """Greet someone."""
    run_hello(name)
