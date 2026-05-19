"""Command-line interface for scifield."""

import typer

app = typer.Typer(
    name="scifield",
    help="SciField — multi-axis framework for monitoring scientific field health.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """SciField — multi-axis framework for monitoring scientific field health."""


@app.command()
def demo() -> None:
    """Run the end-to-end demo on a toy corpus (placeholder; implemented in V1-S02)."""
    typer.echo("demo not yet implemented")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
