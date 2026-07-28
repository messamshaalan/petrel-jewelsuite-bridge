"""
CLI validate command — checks a grid file for structural integrity and
reports property statistics without performing any conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

import numpy as np

console = Console()
validate_app = typer.Typer(no_args_is_help=True)


@validate_app.command("grid")
def validate_grid(
    input_file: Path = typer.Argument(..., help="GRDECL, EGRID, or RESQML .epc file"),
    check_properties: bool = typer.Option(True,  help="Validate property ranges"),
    check_volumes: bool    = typer.Option(True,  help="Check for zero/negative volumes"),
    check_actnum: bool     = typer.Option(True,  help="Report active cell fraction"),
    strict: bool           = typer.Option(False, help="Exit non-zero if any warning found"),
) -> None:
    """Validate a reservoir grid file and report QC metrics."""

    from formats.grdecl_reader import read_grdecl
    from formats.egrid_reader import read_egrid
    from formats.resqml_io import read_resqml

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
        task = prog.add_task("Reading file…", total=None)
        suffix = input_file.suffix.lower()
        try:
            if suffix in (".grdecl", ".dat"):
                model = read_grdecl(input_file)
            elif suffix in (".egrid", ".grid"):
                model = read_egrid(input_file)
            elif suffix in (".epc",):
                model = read_resqml(input_file)
            else:
                console.print(f"[red]Unsupported format: {suffix}[/red]")
                raise typer.Exit(1)
        except Exception as exc:
            prog.stop()
            console.print(f"[red]Failed to read file: {exc}[/red]")
            raise typer.Exit(1)
        prog.update(task, description="File read ✓")

    warnings: list[str] = []
    errors: list[str] = []

    grid = model.active_grid()
    if grid is None:
        errors.append("No grid object found in model")
        _report([], warnings, errors, strict)
        return

    # --- Basic dimensions ---
    d = grid.dims if hasattr(grid, "dims") and grid.dims else None
    rows: list[tuple[str, str, str]] = []

    if d:
        rows.append(("Grid dimensions",
                     f"{d.ni} × {d.nj} × {d.nk}",
                     f"Total: {d.ncells:,}"))

    # --- ACTNUM check ---
    if check_actnum and hasattr(grid, "actnum") and grid.actnum is not None:
        actnum = grid.actnum
        n_active = int(actnum.sum())
        n_total = actnum.size
        frac = n_active / n_total if n_total else 0.0
        status = "✓" if frac > 0 else "✗ No active cells!"
        rows.append(("Active cells", f"{n_active:,} / {n_total:,}", f"{frac:.1%}  {status}"))
        if frac == 0:
            errors.append("Grid has zero active cells")
        elif frac < 0.01:
            warnings.append(f"Very low active fraction: {frac:.1%}")

    # --- COORD / ZCORN checks (CornerPointGrid) ---
    if check_volumes and hasattr(grid, "coord") and grid.coord is not None:
        coord = grid.coord
        zcorn = grid.zcorn
        if np.any(np.isnan(coord)):
            errors.append("NaN values found in COORD array")
        if np.any(np.isnan(zcorn)):
            errors.append("NaN values found in ZCORN array")
        else:
            z_min, z_max = float(zcorn.min()), float(zcorn.max())
            rows.append(("Depth range (ZCORN)", f"{z_min:.1f} – {z_max:.1f} m", ""))
            if z_min < 0:
                warnings.append("ZCORN contains negative depths (above sea level?)")

    # --- Property checks ---
    if check_properties and model.properties:
        for prop in model.properties:
            v = prop.values.astype(np.float64)
            has_nan = bool(np.any(np.isnan(v)))
            has_inf = bool(np.any(np.isinf(v)))
            p_min, p_max = float(np.nanmin(v)), float(np.nanmax(v))
            issues = []
            if has_nan:
                issues.append("NaN")
            if has_inf:
                issues.append("Inf")
            status_icon = "✓" if not issues else f"[red]✗ {','.join(issues)}[/red]"
            rows.append((
                prop.name,
                f"{p_min:.4g} – {p_max:.4g}  ({prop.unit or '–'})",
                status_icon,
            ))
            if has_nan:
                warnings.append(f"{prop.name}: contains NaN values")
            if has_inf:
                errors.append(f"{prop.name}: contains Inf values")

    _report(rows, warnings, errors, strict)


def _report(
    rows: list[tuple[str, str, str]],
    warnings: list[str],
    errors: list[str],
    strict: bool,
) -> None:
    t = Table(title="Grid Validation Report", border_style="dim")
    t.add_column("Item",    style="cyan", min_width=22)
    t.add_column("Value",   style="white")
    t.add_column("Status",  style="green")
    for r in rows:
        t.add_row(*r)
    console.print(t)

    for w in warnings:
        console.print(f"  [yellow]⚠ {w}[/yellow]")
    for e in errors:
        console.print(f"  [red]✗ {e}[/red]")

    if errors:
        console.print("\n[red bold]VALIDATION FAILED[/red bold]")
        raise typer.Exit(1)
    elif warnings and strict:
        console.print("\n[yellow bold]VALIDATION WARNINGS (strict mode)[/yellow bold]")
        raise typer.Exit(2)
    else:
        console.print("\n[green bold]✓ Validation passed[/green bold]")
