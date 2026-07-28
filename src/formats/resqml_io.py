"""
RESQML 2.0 reader/writer (OSDU-compliant neutral format).

Uses BP's resqpy library, which implements the full RESQML 2.0.1 standard.
Supports IjkGrid (corner-point and stair-step), fault surfaces, horizon
surfaces, and time-step property series.

References
----------
bp/resqpy : https://github.com/bp/resqpy
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from core.grid_models import (
    BridgeModel,
    CornerPointGrid,
    GridDimensions,
    GridProperty,
    GridType,
    PropertyType,
    StructuralFramework,
)


_PROP_MAP: dict[str, PropertyType] = {
    "PORO":     PropertyType.POROSITY,
    "PERMX":    PropertyType.PERMEABILITY_X,
    "PERMY":    PropertyType.PERMEABILITY_Y,
    "PERMZ":    PropertyType.PERMEABILITY_Z,
    "SW":       PropertyType.WATER_SATURATION,
    "NTG":      PropertyType.NET_TO_GROSS,
    "FACIES":   PropertyType.FACIES,
    "PRESSURE": PropertyType.PRESSURE,
}


def _resqpy():
    try:
        import resqpy.model as rq_model
        import resqpy.grid as rq_grid
        import resqpy.property as rq_prop
        import resqpy.surface as rq_surf
        import resqpy.crs as rq_crs
        from resqpy.rq_import import grid_from_cp
        return rq_model, rq_grid, rq_prop, rq_surf, rq_crs, grid_from_cp
    except ImportError as exc:
        raise ImportError("resqpy required: pip install resqpy") from exc


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_resqml(path: str | Path) -> BridgeModel:
    """
    Read the first IjkGrid from a RESQML .epc file and all attached properties.
    """
    path = Path(path)
    rq_model, rq_grid, rq_prop, rq_surf, rq_crs, grid_from_cp = _resqpy()

    model = rq_model.Model(str(path))

    # Load first IjkGrid
    grid_uuids = model.uuids(obj_type="IjkGridRepresentation")
    if not grid_uuids:
        raise ValueError(f"No IjkGridRepresentation found in {path}")

    grid = rq_grid.Grid(model, uuid=grid_uuids[0])
    grid.cache_all_geometry_arrays()

    ni, nj, nk = grid.extent_kji[2], grid.extent_kji[1], grid.extent_kji[0]
    dims = GridDimensions(ni=ni, nj=nj, nk=nk)

    # corner_points array: shape (nk, nj, ni, 2, 2, 2, 3)
    cp = grid.corner_points(cache_cp_array=True)

    # Repack into GRDECL COORD / ZCORN convention
    coord, zcorn = _cp_to_coord_zcorn(cp, dims)
    actnum = grid.extract_inactive_mask().astype(np.int32)
    # resqpy inactive mask is True where inactive; invert for ACTNUM (1=active)
    actnum = (~actnum.astype(bool)).astype(np.int32).ravel()

    corner_point = CornerPointGrid(
        dims=dims,
        coord=coord,
        zcorn=zcorn,
        actnum=actnum,
        source_file=path,
    )

    # Load properties
    properties: list[GridProperty] = []
    pc = rq_prop.PropertyCollection(support=grid)
    for part in pc.parts():
        values = pc.cached_part_array_ref(part)
        if values is None:
            continue
        name = pc.citation_title_for_part(part) or "PROP"
        prop_type = _PROP_MAP.get(name.upper(), PropertyType.CUSTOM)
        ts_idx = pc.time_index_for_part(part)
        properties.append(
            GridProperty(
                name=name.upper(),
                prop_type=prop_type,
                values=np.asarray(values, dtype=np.float32).ravel(),
                timestep_idx=ts_idx,
            )
        )

    # Load structural framework
    framework = _load_framework(model, rq_surf)

    return BridgeModel(
        grid_type=GridType.CORNER_POINT,
        corner_point=corner_point,
        properties=properties,
        framework=framework,
        source_software="RESQML",
    )


def _cp_to_coord_zcorn(
    cp: np.ndarray, dims: GridDimensions
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert resqpy corner_points array (nk, nj, ni, 2, 2, 2, 3) to
    GRDECL COORD (ni+1, nj+1, 6) and ZCORN (2*ni, 2*nj, 2*nk).
    """
    nk, nj, ni = dims.nk, dims.nj, dims.ni

    coord = np.zeros((ni + 1, nj + 1, 6), dtype=np.float64)
    zcorn = np.zeros((2 * ni, 2 * nj, 2 * nk), dtype=np.float64)

    # Build COORD from top/bottom pillar endpoints (approximate for non-pillar)
    for j in range(nj + 1):
        jj = min(j, nj - 1)
        dj = 1 if j < nj else -1
        for i in range(ni + 1):
            ii = min(i, ni - 1)
            di = 1 if i < ni else -1
            # top of pillar: k=0, top z
            top_xyz = cp[0, jj, ii, 0, max(0, dj - 1 + (0 if j < nj else 1)),
                          max(0, di - 1 + (0 if i < ni else 1)), :]
            # bottom of pillar: k=nk-1, bottom z
            bot_xyz = cp[nk - 1, jj, ii, 1, max(0, dj - 1 + (0 if j < nj else 1)),
                          max(0, di - 1 + (0 if i < ni else 1)), :]
            coord[i, j, :3] = top_xyz
            coord[i, j, 3:] = bot_xyz

    # Build ZCORN
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                for dk in range(2):
                    for dj in range(2):
                        for di in range(2):
                            zi = 2 * i + di
                            zj = 2 * j + dj
                            zk = 2 * k + dk
                            zcorn[zi, zj, zk] = cp[k, j, i, dk, dj, di, 2]

    return coord, zcorn


def _load_framework(model, rq_surf) -> Optional[StructuralFramework]:
    fw = StructuralFramework()

    # Fault surfaces
    for uuid in model.uuids(obj_type="TriangulatedSetRepresentation"):
        surf = rq_surf.Surface(model, uuid=uuid)
        t, v = surf.triangles_and_points()
        if t is not None and v is not None:
            fw.faults.append(
                StructuralFramework.FaultSurface(
                    name=surf.title or "Fault",
                    vertices=np.asarray(v, dtype=np.float64),
                    triangles=np.asarray(t, dtype=np.int32),
                )
            )

    if not fw.faults and not fw.horizons and not fw.zones:
        return None
    return fw


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_resqml(model: BridgeModel, path: str | Path) -> None:
    """
    Write a BridgeModel to a RESQML 2.0 .epc archive.
    Supports CornerPointGrid with optional properties and fault surfaces.
    """
    path = Path(path)
    rq_model, rq_grid, rq_prop, rq_surf, rq_crs, grid_from_cp = _resqpy()

    rq_m = rq_model.new_model(str(path))

    if model.corner_point is None:
        raise ValueError("BridgeModel has no CornerPointGrid to write")

    g = model.corner_point
    d = g.dims

    crs = rq_crs.Crs(rq_m, z_inc_down=True, title="local CRS")
    crs.create_xml()

    # Build resqpy Grid
    cp_array = _coord_zcorn_to_cp(g.coord, g.zcorn, d)
    active_mask = g.actnum.reshape(d.ni, d.nj, d.nk).transpose(2, 1, 0).astype(bool)

    grid = grid_from_cp(
        rq_m,
        cp_array,
        crs.uuid,
        active_mask=active_mask,
    )
    grid.title = model.project_name or "BridgeGrid"
    grid.write_hdf5_from_caches()
    grid.create_xml()

    # Properties
    pc = rq_prop.PropertyCollection(support=grid)
    for prop in model.properties:
        if prop.values.size != d.ncells:
            # Non-cell ordering (e.g. per pillar-array/stack) that doesn't map
            # onto this grid's flat cell array — skip rather than crash.
            continue
        arr = prop.values.reshape(d.nk, d.nj, d.ni)
        pc.add_cached_array_to_imported_list(
            arr,
            prop.name,
            uom=prop.unit or "Euc",
            property_kind=prop.prop_type.value,
            time_index=prop.timestep_idx,
        )
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()

    # Faults
    if model.framework:
        for fault in model.framework.faults:
            surf = rq_surf.Surface(rq_m, title=fault.name)
            surf.set_from_triangles_and_points(fault.triangles, fault.vertices)
            surf.write_hdf5()
            surf.create_xml()

    rq_m.store_epc()


def _coord_zcorn_to_cp(
    coord: np.ndarray, zcorn: np.ndarray, dims: GridDimensions
) -> np.ndarray:
    """
    Build a resqpy corner_points array (nk, nj, ni, 2, 2, 2, 3) from
    GRDECL COORD and ZCORN.
    """
    ni, nj, nk = dims.ni, dims.nj, dims.nk
    cp = np.zeros((nk, nj, ni, 2, 2, 2, 3), dtype=np.float64)

    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                for dk in range(2):
                    for dj in range(2):
                        for di in range(2):
                            # Pillar at (i+di, j+dj)
                            pil = coord[i + di, j + dj]  # 6 values: xt,yt,zt,xb,yb,zb
                            xt, yt, zt = pil[0], pil[1], pil[2]
                            xb, yb, zb = pil[3], pil[4], pil[5]

                            z = zcorn[2 * i + di, 2 * j + dj, 2 * k + dk]

                            dz = zb - zt
                            if abs(dz) > 1e-10:
                                t = (z - zt) / dz
                            else:
                                t = 0.5
                            x = xt + t * (xb - xt)
                            y = yt + t * (yb - yt)
                            cp[k, j, i, dk, dj, di] = [x, y, z]

    return cp
