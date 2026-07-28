"""
Eclipse EGRID / GRID binary reader.

Uses resfo (pure-Python, Windows-compatible) to read the Fortran-unformatted
binary records produced by Eclipse, OPM Flow, and Petrel export.

References
----------
equinor/resfo : https://github.com/equinor/resfo
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.grid_models import (
    BridgeModel,
    CornerPointGrid,
    GridDimensions,
    GridType,
)


def _load_resfo(path: Path) -> dict[str, np.ndarray]:
    """Return keyword → array mapping from an EGRID binary file via resfo."""
    try:
        import resfo  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "resfo is required for EGRID reading.  Install with: pip install resfo"
        ) from exc

    kw_map: dict[str, np.ndarray] = {}
    for kw, arr in resfo.lazy_read(path):
        kw_map[kw.strip()] = np.asarray(arr)
    return kw_map


def read_egrid(path: str | Path) -> BridgeModel:
    """
    Read an Eclipse EGRID binary file and return a BridgeModel.

    The EGRID format stores COORD and ZCORN (same semantics as GRDECL) along
    with GRIDHEAD (contains NI, NJ, NK) and ACTNUM.
    """
    path = Path(path)
    kw = _load_resfo(path)

    # GRIDHEAD: [type_flag, NI, NJ, NK, ...]
    gridhead = kw.get("GRIDHEAD")
    if gridhead is None:
        raise ValueError(f"No GRIDHEAD record in {path}")
    ni, nj, nk = int(gridhead[1]), int(gridhead[2]), int(gridhead[3])
    dims = GridDimensions(ni=ni, nj=nj, nk=nk)

    coord_flat = kw.get("COORD")
    zcorn_flat = kw.get("ZCORN")
    if coord_flat is None or zcorn_flat is None:
        raise ValueError(f"COORD or ZCORN missing in {path}")

    coord = coord_flat.astype(np.float64).reshape(ni + 1, nj + 1, 6)
    zcorn = zcorn_flat.astype(np.float64).reshape(2 * ni, 2 * nj, 2 * nk)

    actnum_raw = kw.get("ACTNUM")
    if actnum_raw is not None:
        actnum = actnum_raw.astype(np.int32)
    else:
        actnum = np.ones(dims.ncells, dtype=np.int32)

    grid = CornerPointGrid(
        dims=dims,
        coord=coord,
        zcorn=zcorn,
        actnum=actnum,
        source_file=path,
    )
    return BridgeModel(
        grid_type=GridType.CORNER_POINT,
        corner_point=grid,
        properties=[],
        source_software="Petrel",
    )


def read_egrid_xtgeo(path: str | Path) -> BridgeModel:
    """
    Alternative reader using xtgeo for EGRID files.  Provides richer metadata
    (CRS, unit system) and handles non-standard corner-point variants.
    """
    path = Path(path)
    try:
        import xtgeo  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "xtgeo is required for this reader.  Install with: pip install xtgeo"
        ) from exc

    grd = xtgeo.grid_from_file(str(path))
    ni, nj, nk = grd.dimensions

    # Export internal arrays to numpy via xtgeo's accessor
    coord = np.asarray(grd._coordsv, dtype=np.float64).reshape(ni + 1, nj + 1, 6)
    zcorn = np.asarray(grd._zcornsv, dtype=np.float64).reshape(2 * ni, 2 * nj, 2 * nk)
    actnum = np.asarray(grd._actnumsv, dtype=np.int32).ravel()

    dims = GridDimensions(ni=ni, nj=nj, nk=nk)
    grid = CornerPointGrid(
        dims=dims,
        coord=coord,
        zcorn=zcorn,
        actnum=actnum,
        source_file=path,
    )
    return BridgeModel(
        grid_type=GridType.CORNER_POINT,
        corner_point=grid,
        properties=[],
        source_software="Petrel",
    )
