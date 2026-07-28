"""
CLI info command — inspect and summarise a reservoir grid file.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
info_app = typer.Typer(no_args_is_help=True)


@info_app.command("show")
def show_info(
    input_file: Path = typer.Argument(..., help="Grid file to inspect"),
    show_properties: bool = typer.Option(True,  help="List all property arrays"),
    show_framework:  bool = typer.Option(True,  help="List structural elements"),
) -> None:
    """Print metadata and statistics for a reservoir grid file."""
    from formats.grdecl_reader import read_grdecl
    from formats.egrid_reader import read_egrid
    from formats.resqml_io import read_resqml

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
        console.print(f"[red]Error reading {input_file}: {exc}[/red]")
        raise typer.Exit(1)

    # --- Header panel ---
    console.print(
        Panel(
            f"[cyan]{input_file.name}[/cyan]  "
            f"[dim]{input_file.stat().st_size / 1024:.1f} KB[/dim]\n"
            f"Grid type: [bold]{model.grid_type.name}[/bold]  |  "
            f"Source: [bold]{model.source_software or '–'}[/bold]  |  "
            f"Project: [bold]{model.project_name or '–'}[/bold]",
            title="File Info",
            border_style="blue",
        )
    )

    # --- Grid dimensions ---
    grid = model.active_grid()
    if grid is not None and hasattr(grid, "dims") and grid.dims:
        d = grid.dims
        gt = Table(title="Grid Geometry", border_style="dim")
        gt.add_column("Parameter", style="cyan")
        gt.add_column("Value",     style="white")
        gt.add_row("NI × NJ × NK",  f"{d.ni} × {d.nj} × {d.nk}")
        gt.add_row("Total cells",    f"{d.ncells:,}")

        if hasattr(grid, "actnum") and grid.actnum is not None:
            n_act = int(grid.actnum.sum())
            gt.add_row("Active cells", f"{n_act:,}  ({n_act/d.ncells:.1%})")

        if hasattr(grid, "coord") and grid.coord is not None:
            c = grid.coord
            gt.add_row("X range", f"{c[..., 0].min():.1f} – {c[..., 0].max():.1f} m")
            gt.add_row("Y range", f"{c[..., 1].min():.1f} – {c[..., 1].max():.1f} m")
        if hasattr(grid, "zcorn") and grid.zcorn is not None:
            z = grid.zcorn
            gt.add_row("Z range", f"{z.min():.1f} – {z.max():.1f} m")

        if hasattr(grid, "origin"):
            origin = grid.origin
            gt.add_row("Origin", f"({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f})")

        if hasattr(grid, "nodes"):
            gt.add_row("Nodes",  f"{len(grid.nodes):,}")
        if hasattr(grid, "cells"):
            gt.add_row("Cells",  f"{len(grid.cells):,}")
        if hasattr(grid, "tets"):
            gt.add_row("Tetrahedra", f"{len(grid.tets):,}")

        console.print(gt)

    # --- Properties ---
    if show_properties and model.properties:
        import numpy as np
        pt = Table(title="Property Arrays", border_style="dim")
        pt.add_column("Name",    style="cyan",  min_width=12)
        pt.add_column("Type",    style="dim",   min_width=14)
        pt.add_column("Min",     style="white", justify="right")
        pt.add_column("Max",     style="white", justify="right")
        pt.add_column("Mean",    style="white", justify="right")
        pt.add_column("Unit",    style="dim")
        pt.add_column("Step",    style="dim")

        for prop in model.properties:
            v = prop.values.astype(np.float64)
            step_str = (
                f"t={prop.timestep_days:.0f}d"
                if prop.timestep_days is not None else "static"
            )
            pt.add_row(
                prop.name,
                prop.prop_type.value,
                f"{v.min():.4g}",
                f"{v.max():.4g}",
                f"{v.mean():.4g}",
                prop.unit or "–",
                step_str,
            )
        console.print(pt)

    # --- Structural framework ---
    if show_framework and model.framework:
        fw = model.framework
        ft = Table(title="Structural Framework", border_style="dim")
        ft.add_column("Type",    style="cyan")
        ft.add_column("Name",    style="white")
        ft.add_column("Details", style="dim")
        for f_ in fw.faults:
            ft.add_row("Fault", f_.name,
                       f"{len(f_.vertices):,} verts  "
                       f"| throw≈{f_.throw_avg_m:.0f}m" if f_.throw_avg_m else "")
        for h in fw.horizons:
            ft.add_row("Horizon", h.name,
                       f"{len(h.vertices):,} verts  "
                       f"| depth≈{h.depth_avg_m:.0f}m" if h.depth_avg_m else "")
        for z in fw.zones:
            ft.add_row("Zone", z.name,
                       f"{z.top_horizon} → {z.base_horizon}  ({z.layer_count} layers)")
        console.print(ft)
    elif show_framework:
        console.print("[dim]No structural framework data.[/dim]")
