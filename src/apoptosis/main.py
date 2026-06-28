from apoptosis import commands  # noqa: F401
from apoptosis.app import app


def run() -> None:
    app()


if __name__ == "__main__":
    run()
