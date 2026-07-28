"""
JewelGrid → Petrel Corner-Point Grid converter.

Addresses Gap 2: JewelSuite curvilinear JewelGrid → Petrel pillar grid.

Strategy
--------
1. Extract 8-corner XYZ positions from the hex cell array.
2. Fit vertical (or tilted) pillars through the J-I node columns by averaging
   all cell-corner positions that share the same pillar index.
3. Project Z-coordinates back through the fitted pillars to build ZCORN.
4. Report geometric fidelity (pillar tilt, max deviation from pillars).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.grid_models import (
    BridgeModel,
    CornerPointGrid,
    GridDimensions,
    GridProperty,
    GridType,
    JewelGrid,
)


@dataclass
class JewelToCornerQC:
    n_pillars: int
    max_pillar_tilt_deg: float
    mean_xy_deviation_m: float
    max_xy_deviation_m: float
    warnings: list[str]


def jewel_to_corner_point(model: BridgeModel) -> tuple[BridgeModel, JewelToCornerQC]:
    """
    Convert a JewelGrid BridgeModel to a CornerPointGrid BridgeModel.

    The JewelGrid must have structured dimensions set (dims attribute).
    For unstructured JewelGrids, call `resample_to_structured` first.
    """
    if model.jewel_grid is None:
        raise ValueError("Input BridgeModel has no JewelGrid")

    jg: JewelGrid = model.jewel_grid

    if jg.dims is None:
        raise ValueError(
            "JewelGrid has no structured dimensions.  "
            "Set dims or call resample_to_structured() first."
        )

    d = jg.dims
    ni, nj, nk = d.ni, d.nj, d.nk

    # ------------------------------------------------------------------ #
    # Step 1: Unpack hex cell corners into (ni, nj, nk, 2, 2, 2, 3)     #
    # ------------------------------------------------------------------ #
    nodes_raw = _unpack_cells(jg, d)

    # ------------------------------------------------------------------ #
    # Step 2: Fit pillars at each (i+1, j+1) grid node                  #
    # ------------------------------------------------------------------ #
    # pillar_pts[pi, pj, :, xyz] — top and bottom XYZ of fitted pillar
    coord, pillar_qc = _fit_pillars(nodes_raw, ni, nj, nk)

    # ------------------------------------------------------------------ #
    # Step 3: Build ZCORN by projecting cell corner Z values             #
    # ------------------------------------------------------------------ #
    zcorn = _build_zcorn(nodes_raw, ni, nj, nk)

    actnum = jg.actnum.copy().astype(np.int32)
    if actnum.size != d.ncells:
        actnum = np.ones(d.ncells, dtype=np.int32)

    grid = CornerPointGrid(
        dims=d,
        coord=coord,
        zcorn=zcorn,
        actnum=actnum,
        source_file=jg.source_file,
    )

    qc = JewelToCornerQC(
        n_pillars=(ni + 1) * (nj + 1),
        max_pillar_tilt_deg=pillar_qc["max_tilt"],
        mean_xy_deviation_m=pillar_qc["mean_dev"],
        max_xy_deviation_m=pillar_qc["max_dev"],
        warnings=pillar_qc["warnings"],
    )

    out_model = BridgeModel(
        grid_type=GridType.CORNER_POINT,
        corner_point=grid,
        properties=list(model.properties),
        framework=model.framework,
        project_name=model.project_name,
        source_software="Petrel",
    )
    return out_model, qc


def _unpack_cells(jg: JewelGrid, d: GridDimensions) -> np.ndarray:
    """
    Unpack the (n_cells, 8) connectivity + (n_nodes, 3) node array into
    (ni, nj, nk, 2, 2, 2, 3) corner XYZ array.

    JewelGrid hex ordering (VTK-style bottom-first, CCW):
        0:(i,j,k+1) 1:(i+1,j,k+1) 2:(i+1,j+1,k+1) 3:(i,j+1,k+1)  — bottom
        4:(i,j,k  ) 5:(i+1,j,k  ) 6:(i+1,j+1,k  ) 7:(i,j+1,k  )  — top
    """
    ni, nj, nk = d.ni, d.nj, d.nk
    nodes = jg.nodes        # (n_nodes, 3)
    cells = jg.cells        # (n_cells, 8)  int32 node indices

    out = np.zeros((ni, nj, nk, 2, 2, 2, 3), dtype=np.float64)

    # VTK hex node → (dk, dj, di) in GRDECL convention (k down)
    # VTK bottom (dk=1 in GRDECL), top (dk=0 in GRDECL)
    vtk_to_grdecl = {
        0: (1, 0, 0), 1: (1, 0, 1), 2: (1, 1, 1), 3: (1, 1, 0),
        4: (0, 0, 0), 5: (0, 0, 1), 6: (0, 1, 1), 7: (0, 1, 0),
    }

    cell_idx = 0
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                if cell_idx >= len(cells):
                    break
                for vtk_n, (dk, dj, di) in vtk_to_grdecl.items():
                    nidx = cells[cell_idx, vtk_n]
                    out[i, j, k, dk, dj, di] = nodes[nidx]
                cell_idx += 1

    return out


def _fit_pillars(
    nodes_raw: np.ndarray, ni: int, nj: int, nk: int
) -> tuple[np.ndarray, dict]:
    """
    Fit pillars at each grid node by collecting all cell-corner positions that
    share the same (i, j) pillar index and computing top/bottom endpoints.

    Returns COORD array (ni+1, nj+1, 6) and QC dict.
    """
    coord = np.zeros((ni + 1, nj + 1, 6), dtype=np.float64)
    tilt_angles: list[float] = []
    deviations: list[float] = []
    warnings: list[str] = []

    for j in range(nj + 1):
        jlo = max(j - 1, 0)
        jhi = min(j, nj - 1)
        dj_range = list({jlo, jhi})

        for i in range(ni + 1):
            ilo = max(i - 1, 0)
            ihi = min(i, ni - 1)
            di_range = list({ilo, ihi})

            # Collect all z values and corresponding XY positions at this pillar
            xyz_top_list: list[np.ndarray] = []
            xyz_bot_list: list[np.ndarray] = []

            for ii in di_range:
                for jj in dj_range:
                    di = i - ii   # 0 or 1
                    dj = j - jj   # 0 or 1

                    # Top of pillar = shallowest z (dk=0)
                    xyz_top_list.append(nodes_raw[ii, jj, 0, 0, dj, di])
                    # Bottom of pillar = deepest z (dk=1, k=nk-1)
                    xyz_bot_list.append(nodes_raw[ii, jj, nk - 1, 1, dj, di])

            top_xyz = np.mean(xyz_top_list, axis=0)
            bot_xyz = np.mean(xyz_bot_list, axis=0)

            coord[i, j, :3] = top_xyz
            coord[i, j, 3:] = bot_xyz

            # QC: pillar tilt angle
            dv = bot_xyz - top_xyz
            horiz = np.sqrt(dv[0] ** 2 + dv[1] ** 2)
            vert = abs(dv[2])
            if vert > 1e-6:
                tilt = np.degrees(np.arctan2(horiz, vert))
            else:
                tilt = 90.0
            tilt_angles.append(tilt)

            # QC: deviation of contributing nodes from the fitted pillar line
            for xyz in xyz_top_list + xyz_bot_list:
                dev = _point_line_distance(xyz, top_xyz, bot_xyz)
                deviations.append(dev)

    max_tilt = float(max(tilt_angles)) if tilt_angles else 0.0
    if max_tilt > 10.0:
        warnings.append(
            f"Max pillar tilt {max_tilt:.1f}° > 10° — Petrel may reject steep pillars"
        )

    mean_dev = float(np.mean(deviations)) if deviations else 0.0
    max_dev = float(max(deviations)) if deviations else 0.0
    if max_dev > 1.0:
        warnings.append(
            f"Max XY deviation from pillar line: {max_dev:.2f} m — curved JewelGrid "
            "geometry cannot be exactly represented as straight pillars"
        )

    return coord, {
        "max_tilt": max_tilt,
        "mean_dev": mean_dev,
        "max_dev": max_dev,
        "warnings": warnings,
    }


def _build_zcorn(
    nodes_raw: np.ndarray, ni: int, nj: int, nk: int
) -> np.ndarray:
    """Build ZCORN from the Z-components of cell corners."""
    zcorn = np.zeros((2 * ni, 2 * nj, 2 * nk), dtype=np.float64)
    for dk in range(2):
        for dj in range(2):
            for di in range(2):
                zcorn[di::2, dj::2, dk::2] = nodes_raw[:, :, :, dk, dj, di, 2]
    return zcorn


def _point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to the infinite line through a and b (XY only)."""
    ab = b - a
    ab_xy = ab[:2]
    ap_xy = (p - a)[:2]
    len_ab = np.linalg.norm(ab_xy)
    if len_ab < 1e-10:
        return float(np.linalg.norm(ap_xy))
    return float(abs(np.cross(ab_xy, ap_xy)) / len_ab)


def resample_to_structured(
    model: BridgeModel, target_ni: int, target_nj: int, target_nk: int
) -> BridgeModel:
    """
    Re-sample an unstructured JewelGrid onto a regular IJK grid using
    nearest-neighbour mapping.  Use this before jewel_to_corner_point() when
    the JewelGrid has no structured dims.

    Requires pyvista.
    """
    if model.jewel_grid is None:
        raise ValueError("Input BridgeModel has no JewelGrid")

    try:
        import pyvista as pv  # type: ignore
    except ImportError as exc:
        raise ImportError("pyvista required: pip install pyvista") from exc

    jg = model.jewel_grid
    mesh = pv.UnstructuredGrid(
        {pv.CellType.HEXAHEDRON: jg.cells},
        jg.nodes,
    )

    # Attach properties as cell data
    for prop in model.properties:
        if prop.values.size == mesh.n_cells:
            mesh.cell_data[prop.name] = prop.values

    # Create regular IJK sampling grid
    bounds = mesh.bounds
    sample_grid = pv.ImageData(
        dimensions=(target_ni + 1, target_nj + 1, target_nk + 1),
        spacing=(
            (bounds[1] - bounds[0]) / target_ni,
            (bounds[3] - bounds[2]) / target_nj,
            (bounds[5] - bounds[4]) / target_nk,
        ),
        origin=(bounds[0], bounds[2], bounds[4]),
    )
    result = sample_grid.sample(mesh)

    # Extract resampled arrays
    new_nodes = np.array(result.points, dtype=np.float64)
    new_actnum = (~np.isnan(result.cell_data.get("vtkValidPointMask",
                   np.ones(result.n_cells)))).astype(np.int32)

    # Build regular hex connectivity
    ni, nj, nk = target_ni, target_nj, target_nk
    dims = GridDimensions(ni=ni, nj=nj, nk=nk)
    new_cells = _regular_hex_cells(ni, nj, nk)

    new_props = []
    for prop in model.properties:
        cd = result.cell_data.get(prop.name)
        if cd is not None:
            new_props.append(
                GridProperty(
                    name=prop.name,
                    prop_type=prop.prop_type,
                    values=np.asarray(cd, dtype=np.float32),
                    unit=prop.unit,
                    is_discrete=prop.is_discrete,
                )
            )

    new_jg = JewelGrid(
        nodes=new_nodes,
        cells=new_cells,
        actnum=new_actnum,
        dims=dims,
    )
    return BridgeModel(
        grid_type=GridType.JEWEL_GRID,
        jewel_grid=new_jg,
        properties=new_props,
        project_name=model.project_name,
        source_software=model.source_software,
    )


def _regular_hex_cells(ni: int, nj: int, nk: int) -> np.ndarray:
    stride_i = (nj + 1) * (nk + 1)
    stride_j = nk + 1
    I = np.arange(ni)
    J = np.arange(nj)
    K = np.arange(nk)
    ii, jj, kk = np.meshgrid(I, J, K, indexing="ij")
    base = ii * stride_i + jj * stride_j + kk
    cells = np.stack([
        base, base + stride_i, base + stride_i + stride_j, base + stride_j,
        base + 1, base + stride_i + 1, base + stride_i + stride_j + 1, base + stride_j + 1,
    ], axis=-1).reshape(-1, 8).astype(np.int32)
    return cells
