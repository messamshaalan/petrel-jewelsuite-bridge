"""
CLI convert command group.

Usage examples
--------------
  petrel-jewel convert petrel-to-jewel input.grdecl output.epc
  petrel-jewel convert jewel-to-petrel input.epc    output.grdecl
  petrel-jewel convert tet-to-voxel    mesh.vtk     output.grdecl --resolution 80
  petrel-jewel convert to-resqml       input.grdecl output.epc
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()
convert_app = typer.Typer(no_args_is_help=True)


class OutputFormat(str, Enum):
    grdecl = "grdecl"
    egrid  = "egrid"
    resqml = "resqml"
    vtk    = "vtk"
    hdf5   = "hdf5"


@convert_app.command("petrel-to-jewel")
def petrel_to_jewel(
    input_file: Path  = typer.Argument(..., help="GRDECL or EGRID input file"),
    output_file: Path = typer.Argument(..., help="Output .epc (RESQML) or .grdecl path"),
    init_file: Optional[Path] = typer.Option(None, help="Eclipse .INIT for static properties"),
    unrst_file: Optional[Path] = typer.Option(None, help="Eclipse .UNRST for dynamic results"),
    output_format: OutputFormat = typer.Option(OutputFormat.resqml, help="Output format"),
    project_name: str = typer.Option("", help="Optional project name tag"),
    qc_report: bool = typer.Option(True, help="Print QC metrics after conversion"),
) -> None:
    """Convert a Petrel corner-point grid to JewelGrid (via RESQML or GRDECL)."""
    from formats.grdecl_reader import read_grdecl
    from formats.egrid_reader import read_egrid
    from formats.eclipse_results import read_init, read_unrst
    from formats.resqml_io import write_resqml
    from converters.petrel_to_jewel.corner_point_to_jewel import corner_point_to_jewel
    from converters.petrel_to_jewel.property_mapper import map_properties

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(),
        console=console,
    ) as prog:
        # --- Read grid ---
        task = prog.add_task("Reading grid…", total=None)
        suffix = input_file.suffix.lower()
        if suffix in (".grdecl", ".dat", ".txt"):
            model = read_grdecl(input_file)
        elif suffix in (".egrid", ".grid"):
            model = read_egrid(input_file)
        else:
            console.print(f"[red]Unknown grid format: {suffix}[/red]")
            raise typer.Exit(1)
        prog.update(task, description="Grid read ✓")

        # --- Append INIT properties ---
        if init_file:
            task2 = prog.add_task("Reading INIT…", total=None)
            static = read_init(init_file, model.corner_point.dims.ncells)
            model.properties.extend(static)
            prog.update(task2, description=f"INIT read ✓  ({len(static)} properties)")

        # --- Append UNRST properties ---
        if unrst_file:
            task3 = prog.add_task("Reading UNRST…", total=None)
            steps = read_unrst(unrst_file, model.corner_point.dims.ncells)
            for step in steps:
                model.properties.extend(step.properties)
            prog.update(task3, description=f"UNRST read ✓  ({len(steps)} steps)")

        # --- Map properties ---
        task4 = prog.add_task("Mapping properties…", total=None)
        model.properties, map_report = map_properties(
            model.properties,
            model.corner_point.dims,
            model.corner_point.actnum,
        )
        prog.update(task4, description="Properties mapped ✓")

        # --- Convert ---
        task5 = prog.add_task("Converting to JewelGrid…", total=None)
        if project_name:
            model.project_name = project_name
        jewel_model, qc = corner_point_to_jewel(model)
        prog.update(task5, description="Converted ✓")

        # --- Write ---
        task6 = prog.add_task("Writing output…", total=None)
        if output_format == OutputFormat.resqml or output_file.suffix == ".epc":
            write_resqml(jewel_model, output_file)
        else:
            from formats.grdecl_reader import write_grdecl
            from converters.jewel_to_petrel.jewel_to_corner_point import jewel_to_corner_point
            final, _ = jewel_to_corner_point(jewel_model)
            write_grdecl(final, output_file)
        prog.update(task6, description="Written ✓")

    console.print(f"\n[green bold]✓ Written to {output_file}[/green bold]")

    if qc_report:
        _print_qc(qc, map_report)


@convert_app.command("jewel-to-petrel")
def jewel_to_petrel(
    input_file: Path  = typer.Argument(..., help="RESQML .epc or GRDECL input"),
    output_file: Path = typer.Argument(..., help="Output GRDECL or EGRID path"),
    output_format: OutputFormat = typer.Option(OutputFormat.grdecl, help="Output format"),
    qc_report: bool = typer.Option(True, help="Print QC metrics"),
) -> None:
    """Convert a JewelGrid (RESQML) to Petrel corner-point grid (GRDECL)."""
    from formats.resqml_io import read_resqml
    from formats.grdecl_reader import write_grdecl
    from converters.jewel_to_petrel.jewel_to_corner_point import jewel_to_corner_point

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Reading RESQML…", total=None)
        model = read_resqml(input_file)
        prog.update(task, description="RESQML read ✓")

        task2 = prog.add_task("Converting to corner-point…", total=None)
        cp_model, qc = jewel_to_corner_point(model)
        prog.update(task2, description="Converted ✓")

        task3 = prog.add_task("Writing GRDECL…", total=None)
        write_grdecl(cp_model, output_file)
        prog.update(task3, description="Written ✓")

    console.print(f"\n[green bold]✓ Written to {output_file}[/green bold]")
    if qc_report:
        _print_jewel_qc(qc)


@convert_app.command("tet-to-voxel")
def tet_to_voxel_cmd(
    input_file: Path  = typer.Argument(..., help="Mesh file (VTK, Abaqus, Gmsh, etc.)"),
    output_file: Path = typer.Argument(..., help="Output GRDECL path"),
    resolution: int   = typer.Option(50,   help="Voxels along longest axis"),
    dx: Optional[float] = typer.Option(None, help="Override voxel X size (m)"),
    dy: Optional[float] = typer.Option(None, help="Override voxel Y size (m)"),
    dz: Optional[float] = typer.Option(None, help="Override voxel Z size (m)"),
    pointset: Optional[Path] = typer.Option(None, help="Also export point-set CSV"),
) -> None:
    """Convert a tetrahedral mesh to a Petrel-compatible voxel grid."""
    try:
        import meshio  # type: ignore
    except ImportError:
        console.print("[red]meshio required: pip install meshio[/red]")
        raise typer.Exit(1)

    from core.grid_models import BridgeModel, TetrahedralMesh, GridType
    from converters.jewel_to_petrel.tet_to_voxel import tet_to_voxel, tet_to_pointset_csv
    from formats.grdecl_reader import write_grdecl

    import numpy as np

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Reading mesh…", total=None)
        mio = meshio.read(str(input_file))
        nodes = mio.points.astype(np.float64)
        tets = None
        for cell_block in mio.cells:
            if cell_block.type == "tetra":
                tets = cell_block.data.astype(np.int32)
                break
        if tets is None:
            console.print("[red]No tetrahedral cells found in mesh.[/red]")
            raise typer.Exit(1)
        tet_mesh = TetrahedralMesh(nodes=nodes, tets=tets, source_file=input_file)
        prog.update(task, description=f"Mesh read ✓  ({len(tets):,} tets)")

        task2 = prog.add_task("Voxelizing…", total=None)
        voxel_size = (dx, dy, dz) if (dx and dy and dz) else None
        tet_model = BridgeModel(
            grid_type=GridType.TETRAHEDRAL,
            tet_mesh=tet_mesh,
            properties=[],
        )
        vox_model, qc = tet_to_voxel(
            tet_model, resolution=resolution, voxel_size=voxel_size
        )
        prog.update(task2, description="Voxelized ✓")

        task3 = prog.add_task("Writing GRDECL…", total=None)
        write_grdecl(vox_model, output_file)
        prog.update(task3, description="Written ✓")

        if pointset:
            task4 = prog.add_task("Writing point-set CSV…", total=None)
            tet_to_pointset_csv(tet_model, pointset)
            prog.update(task4, description="Point-set written ✓")

    console.print(f"\n[green bold]✓ Voxelized to {output_file}[/green bold]")
    _print_voxel_qc(qc)


def _print_qc(qc, map_report) -> None:
    t = Table(title="Conversion QC", border_style="dim")
    t.add_column("Metric", style="cyan")
    t.add_column("Value",  style="white")
    t.add_row("Total cells",      f"{qc.n_cells_total:,}")
    t.add_row("Active cells",     f"{qc.n_cells_active:,}")
    t.add_row("Degenerate cells", f"{qc.n_degenerate:,}")
    t.add_row("Negative-volume",  f"{qc.n_negative_volume:,}")
    t.add_row("Volume loss",      f"{qc.volume_loss_fraction:.2%}")
    t.add_row("Max aspect ratio", f"{qc.max_aspect_ratio:.1f}")
    t.add_row("Properties",       str(map_report.n_properties))
    console.print(t)
    for w in qc.warnings + map_report.warnings:
        console.print(f"  [yellow]⚠ {w}[/yellow]")


def _print_jewel_qc(qc) -> None:
    t = Table(title="JewelGrid → Corner-Point QC", border_style="dim")
    t.add_column("Metric", style="cyan")
    t.add_column("Value",  style="white")
    t.add_row("Pillars",           f"{qc.n_pillars:,}")
    t.add_row("Max pillar tilt",   f"{qc.max_pillar_tilt_deg:.1f}°")
    t.add_row("Mean XY deviation", f"{qc.mean_xy_deviation_m:.3f} m")
    t.add_row("Max XY deviation",  f"{qc.max_xy_deviation_m:.3f} m")
    console.print(t)
    for w in qc.warnings:
        console.print(f"  [yellow]⚠ {w}[/yellow]")


def _print_voxel_qc(qc) -> None:
    t = Table(title="Tet → Voxel QC", border_style="dim")
    t.add_column("Metric", style="cyan")
    t.add_column("Value",  style="white")
    t.add_row("Source tet cells",  f"{qc.n_tet_cells:,}")
    t.add_row("Voxels total",      f"{qc.n_voxels_total:,}")
    t.add_row("Voxels active",     f"{qc.n_voxels_active:,}")
    dx, dy, dz = qc.voxel_size_m
    t.add_row("Voxel size (m)",    f"{dx:.2f} × {dy:.2f} × {dz:.2f}")
    console.print(t)
    for w in qc.warnings:
        console.print(f"  [yellow]⚠ {w}[/yellow]")
