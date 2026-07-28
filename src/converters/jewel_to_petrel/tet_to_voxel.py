"""
Tetrahedral Mesh → Petrel-Compatible Voxel/Stair-Step Grid converter.

Addresses Gap 3: Petrel cannot import tet meshes natively.

Strategy
--------
1. Load the tetrahedral mesh via meshio (supports VTK, Abaqus, Gmsh, etc.)
2. Voxelize to a regular Cartesian grid using PyVista's voxelize()
3. Map cell properties from tet cells to voxels via cell-centre proximity
4. Return a BridgeModel with a StairStepGrid (Petrel-compatible)

Alternative output: point-set CSV for importing as a Petrel point attribute.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.grid_models import (
    BridgeModel,
    GridDimensions,
    GridProperty,
    GridType,
    StairStepGrid,
    TetrahedralMesh,
)


@dataclass
class TetVoxelQC:
    n_tet_cells: int
    n_voxels_total: int
    n_voxels_active: int
    voxel_size_m: tuple[float, float, float]
    property_coverage: dict[str, float]   # prop_name → fraction of voxels covered
    warnings: list[str]


def tet_to_voxel(
    model: BridgeModel,
    *,
    resolution: int = 50,
    voxel_size: tuple[float, float, float] | None = None,
    fill_value: float = 0.0,
) -> tuple[BridgeModel, TetVoxelQC]:
    """
    Convert a tetrahedral mesh BridgeModel to a voxel StairStepGrid.

    Parameters
    ----------
    model       : BridgeModel with tet_mesh populated
    resolution  : number of voxels along the longest axis (used when
                  voxel_size is None)
    voxel_size  : explicit (dx, dy, dz) voxel dimensions in metres
    fill_value  : property value for voxels outside the mesh

    Returns
    -------
    (BridgeModel with StairStepGrid, TetVoxelQC)
    """
    try:
        import pyvista as pv  # type: ignore
    except ImportError as exc:
        raise ImportError("pyvista required: pip install pyvista") from exc

    if model.tet_mesh is None:
        raise ValueError("Input BridgeModel has no TetrahedralMesh")

    tet = model.tet_mesh
    pv_mesh = _tet_to_pyvista(tet)

    # Attach property arrays as cell data
    for prop in model.properties:
        if prop.values.size == pv_mesh.n_cells:
            pv_mesh.cell_data[prop.name] = prop.values.astype(np.float32)

    # Compute voxel size from bounding box if not specified
    bounds = pv_mesh.bounds   # (xmin, xmax, ymin, ymax, zmin, zmax)
    lx = bounds[1] - bounds[0]
    ly = bounds[3] - bounds[2]
    lz = bounds[5] - bounds[4]

    if voxel_size is None:
        longest = max(lx, ly, lz)
        cell_size = longest / resolution
        voxel_size = (cell_size, cell_size, cell_size)

    dx, dy, dz_size = voxel_size

    # Voxelize
    vox = pv.voxelize(pv_mesh, density=min(dx, dy, dz_size), check_surface=False)

    if vox.n_cells == 0:
        raise RuntimeError(
            "Voxelization produced 0 cells.  "
            "Check that the tet mesh is watertight and not degenerate."
        )

    # Determine IJK dimensions from voxel grid
    ni = max(1, int(round(lx / dx)))
    nj = max(1, int(round(ly / dy)))
    nk = max(1, int(round(lz / dz_size)))

    # Sample voxelized mesh back onto a regular ImageData for clean IJK ordering
    image = pv.ImageData(
        dimensions=(ni + 1, nj + 1, nk + 1),
        spacing=(dx, dy, dz_size),
        origin=(bounds[0], bounds[2], bounds[4]),
    )
    sampled = image.sample(vox)

    # Build actnum from valid samples
    actnum_mask = _build_actnum(sampled, ni, nj, nk)
    actnum = actnum_mask.astype(np.int32).ravel()

    # Build stair-step geometry
    dims = GridDimensions(ni=ni, nj=nj, nk=nk)
    dx_arr = np.full(ni, dx, dtype=np.float64)
    dy_arr = np.full(nj, dy, dtype=np.float64)
    dz_arr = np.full((ni, nj, nk), dz_size, dtype=np.float64)

    # Tops array: z increases downward (depth)
    tops = np.zeros((ni, nj, nk), dtype=np.float64)
    for k in range(nk):
        tops[:, :, k] = bounds[4] + k * dz_size

    grid = StairStepGrid(
        dims=dims,
        origin=(bounds[0], bounds[2], bounds[4]),
        dx=dx_arr,
        dy=dy_arr,
        dz=dz_arr,
        tops=tops,
        actnum=actnum,
    )

    # Map sampled properties
    out_props: list[GridProperty] = []
    coverage: dict[str, float] = {}
    warnings: list[str] = []

    for prop in model.properties:
        cd = sampled.cell_data.get(prop.name)
        if cd is None:
            warnings.append(f"{prop.name}: not found in voxelized result")
            continue
        arr = np.asarray(cd, dtype=np.float32).ravel()
        if arr.size != dims.ncells:
            arr = np.resize(arr, dims.ncells)
        # Fill inactive voxels
        arr[~actnum_mask.ravel()] = fill_value
        covered = float(np.sum(actnum_mask.ravel() & (arr != fill_value))) / dims.ncells
        coverage[prop.name] = covered

        out_props.append(
            GridProperty(
                name=prop.name,
                prop_type=prop.prop_type,
                values=arr,
                unit=prop.unit,
                timestep_idx=prop.timestep_idx,
                timestep_days=prop.timestep_days,
            )
        )

    qc = TetVoxelQC(
        n_tet_cells=pv_mesh.n_cells,
        n_voxels_total=dims.ncells,
        n_voxels_active=int(actnum.sum()),
        voxel_size_m=voxel_size,
        property_coverage=coverage,
        warnings=warnings,
    )

    out_model = BridgeModel(
        grid_type=GridType.VOXEL,
        stair_step=grid,
        properties=out_props,
        framework=model.framework,
        project_name=model.project_name,
        source_software="Petrel",
    )
    return out_model, qc


def _tet_to_pyvista(tet: TetrahedralMesh):
    import pyvista as pv  # type: ignore

    n_tets = len(tet.tets)
    # PyVista connectivity: [4, n0, n1, n2, n3, 4, ...]
    cells = np.hstack([
        np.full((n_tets, 1), 4, dtype=np.int64),
        tet.tets.astype(np.int64),
    ]).ravel()
    cell_types = np.full(n_tets, pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, cell_types, tet.nodes.astype(np.float64))


def _build_actnum(sampled, ni: int, nj: int, nk: int) -> np.ndarray:
    """Return bool (ni, nj, nk) mask: True where the voxel is inside the mesh."""
    # PyVista marks cells sampled outside the source with NaN in all arrays
    total = sampled.n_cells
    if total == 0:
        return np.zeros((ni, nj, nk), dtype=bool)

    mask = np.ones(total, dtype=bool)
    for arr in sampled.cell_data.values():
        a = np.asarray(arr)
        if a.ndim == 1:
            mask &= ~np.isnan(a.astype(np.float64))

    # Reshape to IJK — pyvista ImageData uses (k, j, i) Fortran ordering
    n_expected = ni * nj * nk
    if mask.size != n_expected:
        mask = np.resize(mask, n_expected)
    return mask.reshape(ni, nj, nk)


def tet_to_pointset_csv(model: BridgeModel, path: str | Path) -> None:
    """
    Export tet mesh cell centres + properties as a CSV point set.
    Importable into Petrel as a point attribute dataset.
    """
    import csv

    if model.tet_mesh is None:
        raise ValueError("BridgeModel has no TetrahedralMesh")

    tet = model.tet_mesh
    centres = tet.nodes[tet.tets].mean(axis=1)   # (n_tets, 3)

    prop_names = [p.name for p in model.properties if p.values.size == len(tet.tets)]
    prop_data = {p.name: p.values for p in model.properties if p.values.size == len(tet.tets)}

    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["X", "Y", "Z"] + prop_names)
        for row_idx, (x, y, z) in enumerate(centres):
            row = [f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"]
            for name in prop_names:
                row.append(f"{prop_data[name][row_idx]:g}")
            writer.writerow(row)
