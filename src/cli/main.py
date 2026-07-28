"""
Petrel ↔ JewelSuite Bridge — CLI entry point.

Commands
--------
  convert   Convert between Petrel and JewelSuite formats
  validate  Validate a grid file without converting
  info      Inspect a grid file and print metadata
  serve     Launch the web UI
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cli.commands.convert import convert_app
from cli.commands.validate import validate_app
from cli.commands.info import info_app

console = Console()

app = typer.Typer(
    name="petrel-jewel",
    help="Bi-directional data bridge between Petrel and JewelSuite.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_enable=True,
)

app.add_typer(convert_app,  name="convert",  help="Convert grid files between formats")
app.add_typer(validate_app, name="validate", help="Validate a grid file")
app.add_typer(info_app,     name="info",     help="Inspect and describe a grid file")


def _banner() -> None:
    text = Text()
    text.append("Petrel ", style="bold #48CAE4")
    text.append("↔ ", style="bold white")
    text.append("JewelSuite ", style="bold #FF9F1C")
    text.append("Bi-Directional Bridge", style="bold white")
    console.print(Panel(text, subtitle="v0.1.0  |  G&G Workflow Automation", border_style="dim"))


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    _banner()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind the web UI to"),
    port: int = typer.Option(8000,        help="Port for the web UI"),
    reload: bool = typer.Option(False,    help="Enable auto-reload for development"),
) -> None:
    """Launch the web UI (FastAPI + Uvicorn)."""
    try:
        import uvicorn  # type: ignore
    except ImportError:
        console.print("[red]uvicorn not installed.  Run: pip install uvicorn[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Starting web UI at http://{host}:{port}[/green]")
    uvicorn.run(
        "web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    app()
