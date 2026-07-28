"""
GRDECL ASCII format reader/writer.

Handles the full Eclipse/Petrel corner-point grid description file, including
the repeat-count notation (N*value) and inline comments (-- …).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator  # noqa: F401 (used in write helpers)

import numpy as np

from core.grid_models import (
    BridgeModel,
    CornerPointGrid,
    GridDimensions,
    GridProperty,
    GridType,
    PropertyType,
)

_PROP_MAP: dict[str, PropertyType] = {
    "PORO": PropertyType.POROSITY,
    "PERMX": PropertyType.PERMEABILITY_X,
    "PERMY": PropertyType.PERMEABILITY_Y,
    "PERMZ": PropertyType.PERMEABILITY_Z,
    "SW": PropertyType.WATER_SATURATION,
    "NTG": PropertyType.NET_TO_GROSS,
    "FACIES": PropertyType.FACIES,
    "PRESSURE": PropertyType.PRESSURE,
    "STRESS_XX": PropertyType.STRESS_XX,
    "STRESS_YY": PropertyType.STRESS_YY,
    "STRESS_ZZ": PropertyType.STRESS_ZZ,
}

_SKIP_KEYWORDS = {
    "MAPAXES", "GRIDUNIT", "RADIAL", "NOECHO", "ECHO",
    "MAPUNITS", "METRIC", "FIELD", "LAB",
}

# Keywords whose data is a *block* of one or more "/"-terminated records
# (region-modifier syntax: KEYWORD value i1 i2 j1 j2 k1 k2 /  ... /), rather
# than a single flat array.  These must be skipped record-by-record until an
# empty record (a lone "/") closes the block, otherwise the trailing records
# are left as loose tokens that get misread as the next flat-array keyword.
_BLOCK_KEYWORDS = {"EQUALS", "ADD", "MULTIPLY", "COPY", "FAULTS"}

# Keywords with no data at all (no trailing "/").
_BARE_KEYWORDS = {"ENDBOX"}


def _tokenize(path: Path) -> list[str]:
    """Return all non-comment tokens from a GRDECL file as a plain list.

    Using a list (not a generator) ensures the file handle is closed before
    any parsing begins, preventing PermissionError during cleanup on Windows.
    """
    comment_re = re.compile(r"--.*")
    tokens: list[str] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            line = comment_re.sub("", line).strip()
            if line:
                tokens.extend(line.split())
    return tokens


def _consume_numbers(tokens: list[str], pos: int, count: int) -> tuple[np.ndarray, int]:
    """
    Read numbers starting at `pos` in the token list, expanding N*value repeat
    syntax.  Always consumes up to and including the trailing '/'.

    Returns (values_array, new_pos).
    """
    result: list[float] = []
    i = pos
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if tok == "/":
            break
        if "*" in tok:
            n_str, v_str = tok.split("*", 1)
            result.extend([float(v_str)] * int(n_str))
        else:
            try:
                result.append(float(tok))
            except ValueError:
                i -= 1  # put non-numeric token back
                break
    return np.array(result[:count], dtype=np.float64), i


def _check_count(
    flat: np.ndarray, expected: int, keyword: str, dims: GridDimensions, path: Path
) -> None:
    """Raise a clear, actionable error instead of letting a bad count fall
    through to an opaque numpy reshape crash."""
    if flat.size != expected:
        raise ValueError(
            f"{keyword} in {path.name} has {flat.size} values but the grid "
            f"(ni={dims.ni}, nj={dims.nj}, nk={dims.nk}) expects {expected}. "
            "The section is likely truncated, or uses a region-modifier syntax "
            "(EQUALS/BOX/ADD/MULTIPLY) this reader doesn't fully expand - "
            "check the source file's SPECGRID dimensions and the raw "
            f"{keyword} block."
        )


def _skip_block(tokens: list[str], pos: int) -> int:
    """
    Skip a multi-record region-modifier block (EQUALS/ADD/MULTIPLY/COPY),
    where each record is terminated by "/" and the whole block is closed by
    one additional, empty "/" record.
    """
    while pos < len(tokens):
        record_start = pos
        while pos < len(tokens):
            t = tokens[pos]; pos += 1
            if t == "/":
                break
        if pos == record_start + 1:
            # Empty record (the "/" was the very first token) — block closed.
            break
    return pos


def _point_in_triangle_2d(p, a, b, c) -> bool:
    """Barycentric-sign point-in-triangle test in the XY plane."""
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _segments_intersect_2d(p1, p2, p3, p4) -> bool:
    """True if segment p1-p2 crosses segment p3-p4 in the XY plane."""
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _segment_intersects_triangle_2d(p1, p2, a, b, c) -> bool:
    """True if segment p1-p2 crosses triangle (a,b,c), or either endpoint
    lies inside it, when all points are projected onto the XY plane."""
    if _point_in_triangle_2d(p1, a, b, c) or _point_in_triangle_2d(p2, a, b, c):
        return True
    for e1, e2 in ((a, b), (b, c), (c, a)):
        if _segments_intersect_2d(p1, p2, e1, e2):
            return True
    return False


def _fault_keyword_entries(
    framework, coord: np.ndarray, zcorn: np.ndarray, dims: GridDimensions
) -> list[tuple[str, int, int, int, int, int, int, str]]:
    """
    Approximate which grid cell faces sit on each fault surface, by
    intersecting the fault's triangulated geometry (XY projection) against
    the internal I- and J-direction pillar boundaries of the reconstructed
    grid, then locating the affected K-layers from the local Z-range of the
    intersecting triangles.

    Returns rows already converted to Eclipse's 1-based, J-flipped output
    convention (matching how COORD/ZCORN/ACTNUM are written), ready to write
    directly into a FAULTS keyword.

    This does NOT alter grid geometry — it only labels existing cell faces
    for fault naming / MULTFLT transmissibility-multiplier assignment.  Any
    real geometric throw already present in ZCORN is preserved as-is; this
    reader's own JewelGrid reconstruction currently smooths pillar Z-values,
    so exported grids typically have no throw to begin with.
    """
    ni, nj, nk = dims.ni, dims.nj, dims.nk
    if not framework or not framework.faults:
        return []

    # Representative XY per pillar (midpoint of top/bottom — exact for the
    # vertical pillars this bridge currently produces, approximate for tilt).
    pillar_xy = 0.5 * (coord[:, :, 0:2] + coord[:, :, 3:5])

    # Column-average top/bottom Z per (i, j, k) layer for K-range testing.
    layer_z = 0.25 * (
        zcorn[0::2, 0::2, :] + zcorn[1::2, 0::2, :]
        + zcorn[0::2, 1::2, :] + zcorn[1::2, 1::2, :]
    )  # shape (ni, nj, 2*nk)

    entries: list[tuple[str, int, int, int, int, int, int, str]] = []

    for fault in framework.faults:
        verts_xy = fault.vertices[:, :2]
        verts_z = fault.vertices[:, 2]
        tris = fault.triangles

        # I-direction boundaries: pillar line i=b separates cell i=b-1 (I+) from cell i=b.
        for b in range(1, ni):
            for j in range(nj):
                p1, p2 = pillar_xy[b, j], pillar_xy[b, j + 1]
                z_hits: list[float] = []
                for t in tris:
                    a, bb, c = verts_xy[t[0]], verts_xy[t[1]], verts_xy[t[2]]
                    if _segment_intersects_triangle_2d(p1, p2, a, bb, c):
                        z_hits.extend(verts_z[t].tolist())
                if not z_hits:
                    continue
                z_lo, z_hi = min(z_hits), max(z_hits)
                k_matches = [
                    k for k in range(nk)
                    if not (layer_z[b - 1, j, 2 * k + 1] < z_lo or layer_z[b - 1, j, 2 * k] > z_hi)
                ]
                if not k_matches:
                    continue
                k_lo, k_hi = min(k_matches), max(k_matches)
                entries.append((
                    fault.name,
                    b, b,                      # I1, I2 (1-based; cell b's +I face)
                    nj - j, nj - j,             # J1, J2 (flipped to match output J order)
                    k_lo + 1, k_hi + 1,         # K1, K2
                    "I+",
                ))

        # J-direction boundaries: pillar line j=b separates cell j=b-1 from cell j=b.
        for b in range(1, nj):
            for i in range(ni):
                p1, p2 = pillar_xy[i, b], pillar_xy[i + 1, b]
                z_hits = []
                for t in tris:
                    a, bb, c = verts_xy[t[0]], verts_xy[t[1]], verts_xy[t[2]]
                    if _segment_intersects_triangle_2d(p1, p2, a, bb, c):
                        z_hits.extend(verts_z[t].tolist())
                if not z_hits:
                    continue
                z_lo, z_hi = min(z_hits), max(z_hits)
                k_matches = [
                    k for k in range(nk)
                    if not (layer_z[i, b, 2 * k + 1] < z_lo or layer_z[i, b, 2 * k] > z_hi)
                ]
                if not k_matches:
                    continue
                k_lo, k_hi = min(k_matches), max(k_matches)
                # Output J is flipped: internal cell j=b has the SMALLER
                # output J index and sits on the +J side of internal cell
                # j=b-1's neighbour — so label it on the b-side cell as 'J+'.
                entries.append((
                    fault.name,
                    i + 1, i + 1,               # I1, I2
                    nj - b, nj - b,             # J1, J2 (cell j=b, flipped)
                    k_lo + 1, k_hi + 1,         # K1, K2
                    "J+",
                ))

    return entries


def read_grdecl(path: str | Path) -> BridgeModel:
    """Parse a GRDECL file and return a BridgeModel containing a CornerPointGrid."""
    path = Path(path)
    tokens: list[str] = _tokenize(path)   # file is fully read and closed here

    dims: GridDimensions | None = None
    coord: np.ndarray | None = None
    zcorn: np.ndarray | None = None
    actnum: np.ndarray | None = None
    properties: list[GridProperty] = []

    pos = 0
    while pos < len(tokens):
        kw = tokens[pos].upper()
        pos += 1

        if kw == "SPECGRID":
            vals: list[int] = []
            while pos < len(tokens):
                t = tokens[pos]; pos += 1
                if t == "/":
                    break
                upper = t.upper()
                if upper in ("F", "T", "1*", "CYLINDRICAL", "IRREGULAR"):
                    continue
                try:
                    vals.append(int(t))
                except ValueError:
                    pass
            dims = GridDimensions(ni=vals[0], nj=vals[1], nk=vals[2])

        elif kw == "COORD":
            if dims is None:
                raise ValueError("COORD encountered before SPECGRID")
            n_pillars = (dims.ni + 1) * (dims.nj + 1)
            flat, pos = _consume_numbers(tokens, pos, n_pillars * 6)
            _check_count(flat, n_pillars * 6, "COORD", dims, path)
            # Eclipse: J outer (slowest), I inner (fastest) → internal: I outer, J inner
            coord = flat.reshape(dims.nj + 1, dims.ni + 1, 6).transpose(1, 0, 2).copy()

        elif kw == "ZCORN":
            if dims is None:
                raise ValueError("ZCORN encountered before SPECGRID")
            n_expected = 2 * dims.ni * 2 * dims.nj * 2 * dims.nk
            flat, pos = _consume_numbers(tokens, pos, n_expected)
            _check_count(flat, n_expected, "ZCORN", dims, path)
            # Eclipse: K outer (slowest), J middle, I inner (fastest) → internal: I outer, K inner
            zcorn = flat.reshape(2 * dims.nk, 2 * dims.nj, 2 * dims.ni).transpose(2, 1, 0).copy()

        elif kw == "ACTNUM":
            if dims is None:
                raise ValueError("ACTNUM encountered before SPECGRID")
            flat, pos = _consume_numbers(tokens, pos, dims.ncells)
            _check_count(flat, dims.ncells, "ACTNUM", dims, path)
            # Eclipse: K outer, J middle, I inner → internal: I outer, J middle, K inner
            actnum = flat.reshape(dims.nk, dims.nj, dims.ni).transpose(2, 1, 0).ravel().astype(np.int32)

        elif kw in _BARE_KEYWORDS:
            continue  # no data, no trailing "/"

        elif kw in _BLOCK_KEYWORDS:
            # Multi-record region-modifier block: keep consuming "/"-terminated
            # records until an empty record (a lone "/") closes the block.
            pos = _skip_block(tokens, pos)

        elif kw in _SKIP_KEYWORDS or kw == "/":
            # consume until slash (or skip bare slash)
            while pos < len(tokens):
                t = tokens[pos]; pos += 1
                if t == "/":
                    break

        else:
            # Attempt to parse as a property keyword
            if dims is not None and kw not in ("INCLUDE", "END"):
                try:
                    vals_f, pos = _consume_numbers(tokens, pos, dims.ncells)
                    if len(vals_f) == dims.ncells:
                        prop_type = _PROP_MAP.get(kw, PropertyType.CUSTOM)
                        is_disc = kw in ("FACIES", "SATNUM", "PVTNUM", "EQLNUM")
                        dtype = np.int32 if is_disc else np.float32
                        # Eclipse: K outer, J middle, I inner → internal: I outer, J middle, K inner
                        prop_ijk = vals_f.reshape(dims.nk, dims.nj, dims.ni).transpose(2, 1, 0).ravel()
                        properties.append(
                            GridProperty(
                                name=kw,
                                prop_type=prop_type,
                                values=prop_ijk.astype(dtype),
                                is_discrete=is_disc,
                            )
                        )
                    elif len(vals_f) > 0:
                        # Looked numeric but wrong length — most likely an
                        # unrecognized block/region keyword whose records got
                        # misread as a flat array.  Skip rather than crash
                        # downstream with an opaque reshape error.
                        pass
                except (ValueError, IndexError):
                    pass

    if dims is None:
        raise ValueError(f"No SPECGRID keyword found in {path}")
    if coord is None or zcorn is None:
        raise ValueError(f"COORD or ZCORN missing in {path}")
    if actnum is None:
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
        properties=properties,
        source_software="Petrel",
    )


def write_grdecl(model: BridgeModel, path: str | Path) -> list[str]:
    """
    Write a CornerPointGrid BridgeModel to GRDECL ASCII format.

    Returns a list of warning strings (empty if none) — currently used to
    flag the approximate nature of any FAULTS keyword written.
    """
    path = Path(path)
    if model.corner_point is None:
        raise ValueError("BridgeModel has no CornerPointGrid to write")

    g = model.corner_point
    d = g.dims

    def _write_section(arr: np.ndarray, keyword: str, fh) -> None:
        fh.write(f"\n{keyword}\n")
        flat = arr.ravel()
        buf: list[str] = []
        i = 0
        while i < len(flat):
            v = flat[i]
            run = 1
            while i + run < len(flat) and flat[i + run] == v and run < 9999:
                run += 1
            buf.append(f"{run}*{v:g}" if run > 1 else f"{v:g}")
            i += run
            if len(buf) >= 8:
                fh.write("  " + "  ".join(buf) + "\n")
                buf = []
        if buf:
            fh.write("  " + "  ".join(buf) + "\n")
        fh.write("/\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "-- GRDECL file written by Petrel-JewelSuite Bridge\n"
            f"-- Grid: {d.ni} x {d.nj} x {d.nk}"
            f"  ({g.active_count()} active cells)\n\n"
        )
        fh.write(f"SPECGRID\n  {d.ni}  {d.nj}  {d.nk}  1  F\n/\n")

        # MAPAXES: declares the map (geographic) coordinate system used in COORD.
        # Format: point_on_X_axis  origin  point_on_Y_axis
        # Our COORD uses standard East–North map coordinates (no rotation).
        x_orig = float(g.coord[:, :, 0].min())
        y_orig = float(g.coord[:, :, 1].min())
        fh.write(
            f"MAPAXES\n"
            f"  {x_orig + 1.0:.4f}  {y_orig:.4f}"   # point East of origin (X-axis = East)
            f"  {x_orig:.4f}  {y_orig:.4f}"           # origin (SW corner in map coords)
            f"  {x_orig:.4f}  {y_orig + 1.0:.4f} /\n\n"  # point North of origin (Y-axis = North)
        )

        # COORD: I outer → Eclipse J outer, I inner; J flipped for right-handed system
        # (J=0 in file = northernmost pillar, J increases southward = right-handed with K down)
        _write_section(g.coord.transpose(1, 0, 2)[::-1, :, :], "COORD", fh)
        # ZCORN: I outer, K inner → Eclipse K outer, J middle, I inner; J flipped
        _write_section(g.zcorn.transpose(2, 1, 0)[:, ::-1, :], "ZCORN", fh)
        # ACTNUM: same K-outer + J-flipped ordering
        act_3d = g.actnum.reshape(d.ni, d.nj, d.nk)
        _write_section(act_3d.transpose(2, 1, 0)[:, ::-1, :], "ACTNUM", fh)
        for prop in model.properties:
            if prop.values.size != d.ncells:
                # Property is in a non-cell ordering (e.g. per pillar-array or
                # per stack, common in JewelSuite exports) and can't be
                # reshaped onto this grid's flat cell array — GRDECL has no
                # way to represent it, so skip rather than crash the export.
                continue
            pv = prop.values.reshape(d.ni, d.nj, d.nk)
            _write_section(pv.transpose(2, 1, 0)[:, ::-1, :], prop.name, fh)

        warnings: list[str] = []
        if model.framework and model.framework.faults:
            fault_rows = _fault_keyword_entries(model.framework, g.coord, g.zcorn, d)
            if fault_rows:
                fh.write(
                    "\n-- FAULTS: cell faces labelled from fault-surface geometry "
                    "projected onto the reconstructed grid (approximate - see "
                    "conversion QC warnings). Does not itself alter ZCORN geometry.\n"
                )
                fh.write("FAULTS\n")
                for name, i1, i2, j1, j2, k1, k2, face in fault_rows:
                    fh.write(f"  '{name}'  {i1} {i2}  {j1} {j2}  {k1} {k2}  '{face}' /\n")
                fh.write("/\n")
                n_faults = len({row[0] for row in fault_rows})
                warnings.append(
                    f"FAULTS keyword written for {n_faults} fault(s) ({len(fault_rows)} face "
                    "entries) - geometrically approximate (projected from fault-surface "
                    "triangles onto the reconstructed grid), for fault naming/MULTFLT use; "
                    "does not add geometric throw to ZCORN."
                )
            else:
                warnings.append(
                    f"{len(model.framework.faults)} fault surface(s) present in source model "
                    "but none intersected the reconstructed grid - no FAULTS keyword written."
                )

    return warnings
