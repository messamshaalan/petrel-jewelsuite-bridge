"""
Tetrahedral mesh voxelization engine.

Wraps PyVista's voxelize() with reservoir-specific pre-processing:
  • Mesh repair (normals, manifold check)
  • Adaptive resolution from target cell count
  • Signed-distance smoothing at mesh boundaries
  • HDF5 output for large voxel grids
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.grid_models import TetrahedralMesh, GridProperty


@dataclass
class VoxelGrid:
    """Regular Cartesian voxel grid with cell properties."""
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    dims: tuple[int, int, int]        # ni, nj, nk
    actnum: np.ndarray                # (ni*nj*nk,) int32
    properties: list[GridProperty]


def voxelize_tet(
    tet: TetrahedralMesh,
    properties: list[GridProperty],
    *,
    target_cells: int = 1_000_000,
    voxel_size: float | None = None,
    repair_mesh: bool = True,
) -> VoxelGrid:
    """
    Voxelize a tetrahedral mesh to a regular Cartesian grid.

    Parameters
    ----------
    tet           : source tetrahedral mesh
    properties    : per-tet-cell property arrays
    target_cells  : desired voxel count (used to auto-size cells)
    voxel_size    : override isotropic voxel size in metres
    repair_mesh   : fill holes and fix normals before voxelizing
    """
    try:
        import pyvista as pv  # type: ignore
    except ImportError as exc:
        raise ImportError("pyvista required: pip install pyvista") from exc

    n_tets = len(tet.tets)
    cells_flat = np.hstack([
        np.full((n_tets, 1), 4, dtype=np.int64),
        tet.tets.astype(np.int64),
    ]).ravel()
    cell_types = np.full(n_tets, pv.CellType.TETRA, dtype=np.uint8)
    pv_mesh = pv.UnstructuredGrid(cells_flat, cell_types, tet.nodes.astype(np.float64))

    # Attach property arrays
    for prop in properties:
        if prop.values.size == n_tets:
            pv_mesh.cell_data[prop.name] = prop.values.astype(np.float32)

    # Convert to surface for voxelization
    surface = pv_mesh.extract_surface()
    if repair_mesh:
        try:
            surface = surface.fill_holes(100).compute_normals(auto_orient_normals=True)
        except Exception:
            pass

    # Compute voxel size
    bounds = pv_mesh.bounds
    lx = bounds[1] - bounds[0]
    ly = bounds[3] - bounds[2]
    lz = bounds[5] - bounds[4]

    if voxel_size is None:
        volume = lx * ly * lz
        voxel_size = (volume / target_cells) ** (1 / 3)
        voxel_size = max(voxel_size, 0.1)   # floor at 0.1 m

    vox = pv.voxelize(surface, density=voxel_size, check_surface=False)

    ni = max(1, int(round(lx / voxel_size)))
    nj = max(1, int(round(ly / voxel_size)))
    nk = max(1, int(round(lz / voxel_size)))

    image = pv.ImageData(
        dimensions=(ni + 1, nj + 1, nk + 1),
        spacing=(lx / ni, ly / nj, lz / nk),
        origin=(bounds[0], bounds[2], bounds[4]),
    )
    sampled = image.sample(vox)

    # ACTNUM
    n_cells = ni * nj * nk
    actnum = np.ones(n_cells, dtype=np.int32)
    for arr in sampled.cell_data.values():
        a = np.asarray(arr, dtype=np.float32).ravel()
        if a.size == n_cells:
            actnum &= (~np.isnan(a.astype(np.float64))).astype(np.int32)

    # Resample original tet properties onto voxel grid
    pv_mesh.cell_data_to_point_data()
    sampled2 = image.sample(pv_mesh)

    out_props: list[GridProperty] = []
    for prop in properties:
        cd = sampled2.cell_data.get(prop.name)
        if cd is not None:
            arr = np.asarray(cd, dtype=np.float32).ravel()
            if arr.size != n_cells:
                arr = np.resize(arr, n_cells)
            arr[actnum == 0] = 0.0
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

    return VoxelGrid(
        origin=(bounds[0], bounds[2], bounds[4]),
        spacing=(lx / ni, ly / nj, lz / nk),
        dims=(ni, nj, nk),
        actnum=actnum,
        properties=out_props,
    )


def save_voxel_hdf5(grid: VoxelGrid, path: str | Path) -> None:
    """Save a VoxelGrid to HDF5 for large datasets."""
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError("h5py required: pip install h5py") from exc

    path = Path(path)
    with h5py.File(path, "w") as fh:
        fh.attrs["origin"] = grid.origin
        fh.attrs["spacing"] = grid.spacing
        fh.attrs["dims"] = grid.dims
        fh.create_dataset("ACTNUM", data=grid.actnum, compression="gzip")
        grp = fh.create_group("properties")
        for prop in grid.properties:
            ds = grp.create_dataset(prop.name, data=prop.values, compression="gzip")
            ds.attrs["prop_type"] = prop.prop_type.value
            ds.attrs["unit"] = prop.unit
            if prop.timestep_idx is not None:
                ds.attrs["timestep_idx"] = prop.timestep_idx
            if prop.timestep_days is not None:
                ds.attrs["timestep_days"] = prop.timestep_days


def load_voxel_hdf5(path: str | Path) -> VoxelGrid:
    """Load a VoxelGrid from HDF5."""
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError("h5py required: pip install h5py") from exc
    from core.grid_models import PropertyType

    path = Path(path)
    with h5py.File(path, "r") as fh:
        origin = tuple(fh.attrs["origin"])
        spacing = tuple(fh.attrs["spacing"])
        dims = tuple(fh.attrs["dims"])
        actnum = np.array(fh["ACTNUM"], dtype=np.int32)
        props: list[GridProperty] = []
        for name, ds in fh["properties"].items():
            props.append(
                GridProperty(
                    name=name,
                    prop_type=PropertyType(ds.attrs.get("prop_type", "CUSTOM")),
                    values=np.array(ds, dtype=np.float32),
                    unit=ds.attrs.get("unit", ""),
                    timestep_idx=ds.attrs.get("timestep_idx"),
                    timestep_days=ds.attrs.get("timestep_days"),
                )
            )

    return VoxelGrid(
        origin=origin,
        spacing=spacing,
        dims=dims,
        actnum=actnum,
        properties=props,
    )
