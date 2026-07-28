"""
JewelSuite .jewel grid format reader.

The .jewel file is a ZIP container holding UUID-named binary + XML files.

Container structure
-------------------
Every 3D grid is identified by two UUIDs::

    {owner_uuid}_{grid_uuid}.xml.info              # XML descriptor (all counts)
    {owner_uuid}_{grid_uuid}.xml.nodes             # float64[NNodes, 3] XYZ
    {owner_uuid}_{grid_uuid}.xml.stacksets         # I/J indices + validity flags
    {owner_uuid}_{grid_uuid}.xml.fractions         # depth fractions 0→1 along pillars
    {owner_uuid}_{grid_uuid}.xml.pillararrays      # pillar depth ranges + direction flag
    {owner_uuid}_{grid_uuid}.xml.pillargeometries  # pillar node mapping + XY centres
    {owner_uuid}_{grid_uuid}.xml.stacks            # variable-length column topology
    {owner_uuid}_{grid_uuid}.xml.faces             # 21 bytes/face: 5×int32 + 1×uint8 sections
    {owner_uuid}_{grid_uuid}.xml.edges             # flat int32[EdgesNEdgeSegments] IDs
    {owner_uuid}_{grid_uuid}.xml.edgesegments      # 12 bytes/seg: 3×int32 (pg, pa, ps)
    {owner_uuid}_{grid_uuid}.xml.facegroups        # variable-length fault face groups
    {owner_uuid}_{grid_uuid}.xml.{fault_uuid}.nodes      # fault surface XYZ (float64)
    {owner_uuid}_{grid_uuid}.xml.{fault_uuid}.triangles  # fault triangles (int32)
    {owner_uuid}_{prop_uuid}.property              # float32[N] property values

Binary layouts (all little-endian)
------------------------------------
*.xml.nodes
    float64[NNodes, 3]

*.xml.stacksets  (10 bytes × NStackSets)
    int32[NStackSets]   I-index  (0-based; NI = max+1)
    int32[NStackSets]   J-index  (0-based; NJ = max+1)
    uint8[NStackSets]   valid_geometry flag
    uint8[NStackSets]   valid_layer flag

*.xml.fractions
    float32[total]  NK fractions per active pillar (last in each group = 1.0)

*.xml.pillargeometries  (PillarGeometriesNNodes×2 + NPillarGeometries×8×2 + N×1×2)
    uint16[PillarGeometriesNNodes]  node indices (0xFFFF = null)
    float64[NPillarGeometries]      pillar east (X) centre
    float64[NPillarGeometries]      pillar north (Y) centre
    uint8[NPillarGeometries]        top truncation flag
    uint8[NPillarGeometries]        base truncation flag

*.xml.pillararrays  (9 bytes × NPillarArrays)
    float32[NPillarArrays]   start depth
    float32[NPillarArrays]   end depth
    uint8[NPillarArrays]     reversed flag

*.xml.faces  (21 bytes × NFaces, stored as contiguous sections not interleaved)
    int32[NFaces]   first edge ID
    int32[NFaces]   second edge ID
    int32[NFaces]   opposite face ID  (-1 = boundary)
    int32[NFaces]   stack ID          (-1 = no adjacent stack)
    uint8[NFaces]   up-throw flag
    int32[NFaces]   event ID

*.xml.facegroups
    For each of NFaceGroups fault groups, variable-length: int32 count + int32[count] faceIDs
    Followed by optional trailing metadata (group type codes etc.)

*.xml.{fault_uuid}.nodes      float64[N, 3]  pre-triangulated fault surface XYZ
*.xml.{fault_uuid}.triangles  int32[M, 3]    triangle vertex indices into the nodes array

*.property
    float32[N]  cell property values in JewelSuite internal cell ordering
    N can be: NNodes, NPillarArrays, NStacks, NStackSets×NK, or NI×NJ×NK

Property name lookup
--------------------
Property UUIDs are matched to human-readable names by parsing the embedded XML
in ``{catalog_uuid}__{grid_uuid}.json`` files.  Each `<Property>` element carries
a ``GUID=`` and ``Name=`` attribute.  A ``<PropertyGeom>`` wrapper element shares
the same GUID — the regex must target ``<Property `` specifically (not ``<PropertyGeom``).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from core.grid_models import (
    BridgeModel,
    CornerPointGrid,
    GridDimensions,
    GridProperty,
    GridType,
    PropertyType,
    StairStepGrid,
    StructuralFramework,
)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

JEWEL_EXT = ".jewel"

_PROP_TYPE_MAP: dict[str, PropertyType] = {
    "PORO":     PropertyType.POROSITY,
    "PERMX":    PropertyType.PERMEABILITY_X,
    "PERMY":    PropertyType.PERMEABILITY_Y,
    "PERMZ":    PropertyType.PERMEABILITY_Z,
    "KX":       PropertyType.PERMEABILITY_X,
    "KY":       PropertyType.PERMEABILITY_Y,
    "KZ":       PropertyType.PERMEABILITY_Z,
    "SW":       PropertyType.WATER_SATURATION,
    "NTG":      PropertyType.NET_TO_GROSS,
    "FACIES":   PropertyType.FACIES,
    "PRESSURE": PropertyType.PRESSURE,
    "SOIL":     PropertyType.CUSTOM,
    "SGAS":     PropertyType.CUSTOM,
}

_DISCRETE_NAMES = {"FACIES", "SATNUM", "PVTNUM", "EQLNUM", "REGION", "LITHOLOGY"}

# Internal JewelSuite geometry properties — not petrophysical, skip them
_GEOM_PROP_PREFIXES = (
    "cell center", "cell top", "cell volume",
    "stratigraphic cell", "vertical cell",
    "zoneid", "segmentid", "compartmentid",
)

# Regex to match <Property > elements (NOT <PropertyGeom> wrappers)
_PROP_ELEM_RE = re.compile(r"<Property\s[^>]+>", re.DOTALL)
_GUID_RE      = re.compile(r'GUID="([^"]+)"')
_NAME_RE      = re.compile(r'\bName="([^"]+)"')


# ---------------------------------------------------------------------------
# Internal data class
# ---------------------------------------------------------------------------

@dataclass
class _GridInfo:
    n_nodes:          int = 0
    n_stacks:         int = 0
    n_stack_sets:     int = 0
    n_faces:          int = 0
    n_face_groups:    int = 0
    n_pillar_arrays:  int = 0
    n_pillar_geoms:   int = 0
    n_pg_nodes:       int = 0
    n_edges:          int = 0
    n_edge_segs:      int = 0
    n_frac_total:     int = 0
    nk:               int = 0


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------

def read_jewel(path: str | Path) -> BridgeModel:
    """
    Read a JewelSuite .jewel file and return a BridgeModel.

    Improvements over v0.1
    ----------------------
    #1  Property names from embedded XML in JSON catalog files.
    #2  Correct ACTNUM from stacksets valid_geometry flags — only column
        positions that appear in the stacksets array (with valid_geometry=1)
        are marked active; others are ACTNUM=0.
    #3  Pre-triangulated fault surfaces read and stored in
        BridgeModel.framework as StructuralFramework.FaultSurface objects.
        These are exported faithfully when writing RESQML.
    #4  Fault-adjacent pillars detected from facegroups and split in the
        COORD array so Petrel can display fault planes in GRDECL output.
    """
    path = Path(path)
    with zipfile.ZipFile(path, "r") as z:
        names = z.namelist()
        grid_id = _find_best_grid(z, names)
        if grid_id is None:
            raise ValueError(f"No valid 3D grid found in {path.name}")

        owner_uuid, grid_uuid = grid_id
        base = f"{owner_uuid}_{grid_uuid}"

        info        = _parse_info(z.read(f"{base}.xml.info"))
        nodes       = _read_nodes(z.read(f"{base}.xml.nodes"), info.n_nodes)
        ss_i, ss_j, ss_vgeo, ss_vlay = _read_stacksets(
            z.read(f"{base}.xml.stacksets"), info.n_stack_sets
        )
        frac        = _read_fractions(z.read(f"{base}.xml.fractions"))
        pg_nodes, pg_easts, pg_norths, pg_ttop, pg_tbot = _read_pillargeometries(
            z.read(f"{base}.xml.pillargeometries"), info.n_pg_nodes, info.n_pillar_geoms
        )
        pa_starts, pa_ends, pa_rev = _read_pillararrays(
            z.read(f"{base}.xml.pillararrays"), info.n_pillar_arrays
        )

        # Infer grid dimensions
        ni = int(ss_i.max()) + 1 if len(ss_i) else 1
        nj = int(ss_j.max()) + 1 if len(ss_j) else 1
        nk = _infer_nk(frac)
        info.nk = nk

        # --- #1: Property name map from JSON catalog ---
        prop_name_map = _build_prop_name_map(z, names)

        # --- #1: Properties ---
        properties = _read_properties(
            z, names, owner_uuid, prop_name_map, ni, nj, nk
        )

        # --- #3: Fault surfaces ---
        fault_surfaces = _read_fault_surfaces(z, base, names)

        # --- #4: Fault face groups (for GRDECL COORD splitting) ---
        fault_face_groups = _read_facegroups(
            z, base, info.n_face_groups, info.n_faces
        )

    # Build model
    model = _build_model(
        path, ni, nj, nk, nodes,
        ss_i, ss_j, ss_vgeo, ss_vlay,
        frac, pg_easts, pg_norths,
        pa_starts, pa_ends,
        properties, fault_surfaces, fault_face_groups,
    )
    return model


# ---------------------------------------------------------------------------
# ZIP helpers
# ---------------------------------------------------------------------------

def _find_best_grid(z: zipfile.ZipFile, names: list[str]) -> tuple[str, str] | None:
    best, best_n = None, -1
    for fname in [n for n in names if n.endswith(".xml.info")]:
        try:
            n = int(ET.fromstring(z.read(fname)).get("NNodes", 0))
            if n > best_n:
                best_n = n
                stem = fname[: -len(".xml.info")]
                parts = stem.split("_", 1)
                if len(parts) == 2:
                    best = (parts[0], parts[1])
        except Exception:
            continue
    return best


# ---------------------------------------------------------------------------
# Binary decoders
# ---------------------------------------------------------------------------

def _parse_info(raw: bytes) -> _GridInfo:
    root = ET.fromstring(raw.decode("utf-8", errors="replace"))
    g = root.get
    return _GridInfo(
        n_nodes         = int(g("NNodes",             0)),
        n_stacks        = int(g("NStacks",            0)),
        n_stack_sets    = int(g("NStackSets",         0)),
        n_faces         = int(g("NFaces",             0)),
        n_face_groups   = int(g("NFaceGroups",        0)),
        n_pillar_arrays = int(g("NPillarArrays",      0)),
        n_pillar_geoms  = int(g("NPillarGeometries",  0)),
        n_pg_nodes      = int(g("PillarGeometriesNNodes", 0)),
        n_edges         = int(g("NEdges",             0)),
        n_edge_segs     = int(g("NEdgeSegments",      0)),
    )


def _read_nodes(raw: bytes, n_nodes: int) -> np.ndarray:
    expected = n_nodes * 24
    if len(raw) != expected:
        raise ValueError(f"nodes: expected {expected}B for {n_nodes} XYZ, got {len(raw)}B")
    return np.frombuffer(raw, dtype="<f8").reshape(n_nodes, 3).copy()


def _read_stacksets(
    raw: bytes, n_ss: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expected = n_ss * 10
    if len(raw) != expected:
        raise ValueError(f"stacksets: expected {expected}B, got {len(raw)}B")
    pos = 0
    i_idx = np.frombuffer(raw[pos: pos + n_ss * 4], dtype="<i4").copy(); pos += n_ss * 4
    j_idx = np.frombuffer(raw[pos: pos + n_ss * 4], dtype="<i4").copy(); pos += n_ss * 4
    vgeo  = np.frombuffer(raw[pos: pos + n_ss],     dtype="u1").copy();  pos += n_ss
    vlay  = np.frombuffer(raw[pos: pos + n_ss],     dtype="u1").copy()
    return i_idx, j_idx, vgeo, vlay


def _read_fractions(raw: bytes) -> np.ndarray:
    if len(raw) % 4 != 0:
        raise ValueError(f"fractions: {len(raw)} not divisible by 4")
    return np.frombuffer(raw, dtype="<f4").copy()


def _read_pillargeometries(
    raw: bytes, n_pg_nodes: int, n_pg: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expected = n_pg_nodes * 2 + n_pg * 8 * 2 + n_pg * 2
    if len(raw) != expected:
        raise ValueError(f"pillargeometries: expected {expected}B, got {len(raw)}B")
    pos = 0
    node_idx = np.frombuffer(raw[pos: pos + n_pg_nodes * 2], dtype="<u2").copy(); pos += n_pg_nodes * 2
    easts    = np.frombuffer(raw[pos: pos + n_pg * 8],       dtype="<f8").copy(); pos += n_pg * 8
    norths   = np.frombuffer(raw[pos: pos + n_pg * 8],       dtype="<f8").copy(); pos += n_pg * 8
    ttop     = np.frombuffer(raw[pos: pos + n_pg],           dtype="u1").copy();  pos += n_pg
    tbot     = np.frombuffer(raw[pos:],                      dtype="u1").copy()
    return node_idx, easts, norths, ttop, tbot


def _read_pillararrays(
    raw: bytes, n_pa: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = n_pa * 9
    if len(raw) != expected:
        raise ValueError(f"pillararrays: expected {expected}B, got {len(raw)}B")
    starts = np.frombuffer(raw[:n_pa * 4],            dtype="<f4").copy()
    ends   = np.frombuffer(raw[n_pa * 4: n_pa * 8],  dtype="<f4").copy()
    rev    = np.frombuffer(raw[n_pa * 8:],            dtype="u1").copy()
    return starts, ends, rev


def _infer_nk(frac: np.ndarray) -> int:
    """Count values from first non-zero up to and including first 1.0."""
    in_group, count = False, 0
    for v in frac:
        if not in_group:
            if v != 0.0:
                in_group, count = True, 1
                if v == 1.0:
                    return 1
        else:
            count += 1
            if v == 1.0:
                return count
    return max(4, count)


# ---------------------------------------------------------------------------
# #3: Fault surface reader
# ---------------------------------------------------------------------------

def _read_fault_surfaces(
    z: zipfile.ZipFile, base: str, names: list[str]
) -> list[StructuralFramework.FaultSurface]:
    """Read pre-triangulated fault surfaces stored alongside the grid."""
    surfaces: list[StructuralFramework.FaultSurface] = []
    node_files = [
        n for n in names
        if n.startswith(base + ".xml.") and n.endswith(".nodes")
    ]
    for nf in node_files:
        # UUID part sits between "{base}.xml." and ".nodes"
        uuid_part = nf[len(base) + 5: -6]
        tf = f"{base}.xml.{uuid_part}.triangles"
        if tf not in names:
            continue
        try:
            node_raw = z.read(nf)
            tri_raw  = z.read(tf)
            if len(node_raw) % 24 != 0 or len(tri_raw) % 12 != 0:
                continue
            verts = np.frombuffer(node_raw, dtype="<f8").reshape(-1, 3).copy()
            tris  = np.frombuffer(tri_raw,  dtype="<i4").reshape(-1, 3).copy()
            surfaces.append(StructuralFramework.FaultSurface(
                name=f"Fault_{uuid_part[:8]}",
                vertices=verts.astype(np.float64),
                triangles=tris.astype(np.int32),
            ))
        except Exception:
            continue
    return surfaces


# ---------------------------------------------------------------------------
# #4: Facegroups decoder
# ---------------------------------------------------------------------------

def _read_facegroups(
    z: zipfile.ZipFile, base: str, n_fg: int, n_faces_total: int
) -> list[list[int]]:
    """
    Decode variable-length fault face groups.

    Layout: for each of n_fg groups: int32 count, then count×int32 face IDs.
    Trailing metadata (type codes, zeros) is ignored.

    Returns list of face-ID lists, one per fault group.
    """
    fname = f"{base}.xml.facegroups"
    try:
        raw = z.read(fname)
    except Exception:
        return []

    if len(raw) < 4 or len(raw) % 4 != 0:
        return []

    ints = np.frombuffer(raw, dtype="<i4")
    groups: list[list[int]] = []
    pos = 0
    while pos < len(ints) and len(groups) < n_fg:
        cnt = int(ints[pos])
        pos += 1
        if cnt < 0 or cnt > n_faces_total or pos + cnt > len(ints):
            break
        face_ids = ints[pos: pos + cnt].tolist()
        # Sanity-check: face IDs must be in valid range
        if any(fid < 0 or fid >= n_faces_total for fid in face_ids):
            break
        groups.append(face_ids)
        pos += cnt

    return groups


# ---------------------------------------------------------------------------
# #1: Property name map from embedded JSON/XML catalog
# ---------------------------------------------------------------------------

def _build_prop_name_map(z: zipfile.ZipFile, names: list[str]) -> dict[str, str]:
    """
    Parse ``<Property GUID="..." Name="...">`` elements from the embedded XML
    stored in JSON catalog files.

    The ``<PropertyGeom GUID="..." Name="PropertyGeom">`` wrapper element shares
    the same GUID as the actual property — we must match ``<Property `` specifically
    (with a trailing space) to avoid capturing the wrapper.
    """
    uuid_name: dict[str, str] = {}
    for jf in names:
        if not jf.endswith(".json"):
            continue
        try:
            raw_text = z.read(jf).decode("utf-8", errors="replace")
            data_xml = json.loads(raw_text).get("data", "")
            if not data_xml:
                continue
            for m in _PROP_ELEM_RE.finditer(data_xml):
                tag = m.group()
                guid_m = _GUID_RE.search(tag)
                name_m = _NAME_RE.search(tag)
                if guid_m and name_m:
                    uuid_name[guid_m.group(1)] = name_m.group(1)
        except Exception:
            continue
    return uuid_name


# ---------------------------------------------------------------------------
# #1: Property reader
# ---------------------------------------------------------------------------

def _read_properties(
    z: zipfile.ZipFile,
    names: list[str],
    owner_uuid: str,
    prop_name_map: dict[str, str],
    ni: int, nj: int, nk: int,
) -> list[GridProperty]:
    """
    Read all .property files belonging to this owner UUID.

    Cell count matching
    -------------------
    Properties whose float32 count matches ni×nj×nk exactly are mapped
    directly to the GRDECL cell ordering and included.  Properties with
    other counts (JewelSuite-internal orderings such as NPillarArrays or
    NStacks×NK) are also included but marked with a unit note so downstream
    code can flag them as approximate.

    Name resolution
    ---------------
    ``prop_name_map`` (built from embedded XML) maps property UUIDs to names.
    Unrecognised UUIDs fall back to their first-8-char UUID prefix.
    """
    n_regular = ni * nj * nk
    props: list[GridProperty] = []

    prop_files = [
        n for n in names
        if n.startswith(owner_uuid) and n.endswith(".property")
    ]

    for pf in prop_files:
        raw = z.read(pf)
        fsz = z.getinfo(pf).file_size
        if fsz < 8 or fsz % 4 != 0:
            continue

        vals = np.frombuffer(raw, dtype="<f4").copy()

        # Reject tiny arrays (2–7 values are clearly metadata artefacts)
        if len(vals) < 8:
            continue

        # Skip arrays that are entirely NaN or have no finite values
        finite_mask = np.isfinite(vals)
        if finite_mask.sum() == 0:
            continue

        prop_uuid = pf[len(owner_uuid) + 1:].replace(".property", "")
        raw_name  = prop_name_map.get(prop_uuid, "")
        name      = raw_name if raw_name else prop_uuid[:8]

        # Skip internal geometry properties (JewelSuite stores XYZ coords as props)
        name_lo = name.lower()
        if (name_lo.startswith("propertygeom")
                or name_lo.startswith("geometry")
                or any(name_lo.startswith(p) for p in _GEOM_PROP_PREFIXES)):
            continue

        # Reject arrays whose value range is physically implausible (garbage floats).
        # Denormal floats like 9.18e-41 mixed with -3.2e36 signal bad binary reads.
        finite_vals = vals[finite_mask]
        max_abs = float(np.abs(finite_vals).max()) if len(finite_vals) else 0.0
        if max_abs > 1e15:
            continue

        prop_type = _PROP_TYPE_MAP.get(name.upper(), PropertyType.CUSTOM)
        is_disc   = name.upper() in _DISCRETE_NAMES

        # If the count exactly matches the GRDECL cell count use it directly
        if len(vals) == n_regular:
            unit_note = ""
        else:
            # Keep the array as-is; store the actual count in the unit field so
            # users can see it doesn't map 1:1 to GRDECL cells
            unit_note = f"jewel_n={len(vals)}"

        props.append(GridProperty(
            name=name,
            prop_type=prop_type,
            values=vals.astype(np.int32 if is_disc else np.float32),
            unit=unit_note,
            is_discrete=is_disc,
        ))

    return props


# ---------------------------------------------------------------------------
# Grid model builder
# ---------------------------------------------------------------------------

def _build_model(
    path: Path,
    ni: int, nj: int, nk: int,
    nodes: np.ndarray,
    ss_i: np.ndarray, ss_j: np.ndarray,
    ss_vgeo: np.ndarray, ss_vlay: np.ndarray,
    frac: np.ndarray,
    pg_easts: np.ndarray, pg_norths: np.ndarray,
    pa_starts: np.ndarray, pa_ends: np.ndarray,
    properties: list[GridProperty],
    fault_surfaces: list[StructuralFramework.FaultSurface],
    fault_face_groups: list[list[int]],
) -> BridgeModel:
    dims = GridDimensions(ni=ni, nj=nj, nk=nk)

    coord, zcorn, actnum = _nodes_to_coord_zcorn(
        nodes, ss_i, ss_j, ss_vgeo, frac, ni, nj, nk,
        fault_face_groups,
    )

    if coord is None:
        return _stair_step_fallback(path, nodes, ni, nj, nk, properties)

    framework = StructuralFramework(faults=fault_surfaces) if fault_surfaces else None

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
        framework=framework,
        source_software="JewelSuite",
        project_name=path.stem,
    )


# ---------------------------------------------------------------------------
# Geometry reconstruction
# ---------------------------------------------------------------------------

def _nodes_to_coord_zcorn(
    nodes: np.ndarray,
    ss_i: np.ndarray,
    ss_j: np.ndarray,
    ss_vgeo: np.ndarray,
    frac: np.ndarray,
    ni: int, nj: int, nk: int,
    fault_face_groups: list[list[int]],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Map JewelSuite nodes → GRDECL COORD + ZCORN + ACTNUM.

    COORD  shape: (NI+1, NJ+1, 6)       pillar [xt,yt,zt,xb,yb,zb]
    ZCORN  shape: (2*NI, 2*NJ, 2*NK)    corner depths
    ACTNUM shape: (NI*NJ*NK,)            1=active, 0=inactive

    Key fix vs v1: nodes are binned to PILLAR positions (NI+1)×(NJ+1)
    using nearest-neighbour (round), not to cell centroids (floor).
    Each ZCORN element zcorn[2ci+di, 2cj+dj, ...] draws its Z values
    from pillar (ci+di, cj+dj), so adjacent cells share consistent
    pillar depths and no degenerate geometry is produced.
    """
    if len(nodes) < 4:
        return None, None, None

    x_all = nodes[:, 0]
    y_all = nodes[:, 1]
    z_all = nodes[:, 2]   # JewelSuite Z: depth, positive down

    x_min, x_max = float(x_all.min()), float(x_all.max())
    y_min, y_max = float(y_all.min()), float(y_all.max())
    z_global_top = float(z_all.min())   # smallest depth = shallowest = top of grid
    z_global_bot = float(z_all.max())   # largest depth  = deepest    = bottom of grid

    # Pillar spacing
    dp_x = (x_max - x_min) / ni if ni > 0 else 1.0
    dp_y = (y_max - y_min) / nj if nj > 0 else 1.0

    # --- Bin every node to its nearest GRDECL pillar using round() ---
    # pillar_xy: accumulate actual XY per pillar for precise COORD
    # pillar_z:  accumulate Z values per pillar for ZCORN
    pillar_xy: dict[tuple[int, int], list] = {}
    pillar_z:  dict[tuple[int, int], list] = {}

    for node_xyz in nodes:
        x, y, z = float(node_xyz[0]), float(node_xyz[1]), float(node_xyz[2])
        pi = int(round((x - x_min) / dp_x)) if dp_x > 0 else 0
        pj = int(round((y - y_min) / dp_y)) if dp_y > 0 else 0
        pi = max(0, min(ni, pi))
        pj = max(0, min(nj, pj))
        key = (pi, pj)
        pillar_xy.setdefault(key, []).append((x, y))
        pillar_z.setdefault(key, []).append(z)

    # --- COORD: use mean node XY per pillar; fall back to regular grid ---
    coord = np.zeros((ni + 1, nj + 1, 6), dtype=np.float64)
    for pi in range(ni + 1):
        for pj in range(nj + 1):
            xy_list = pillar_xy.get((pi, pj), [])
            if xy_list:
                px = float(np.mean([p[0] for p in xy_list]))
                py = float(np.mean([p[1] for p in xy_list]))
            else:
                px = x_min + pi * dp_x
                py = y_min + pj * dp_y
            coord[pi, pj, 0] = px
            coord[pi, pj, 1] = py
            coord[pi, pj, 2] = z_global_top
            coord[pi, pj, 3] = px
            coord[pi, pj, 4] = py
            coord[pi, pj, 5] = z_global_bot

    # --- ZCORN: per-pillar Z values, nk+1 layer boundaries ---
    # zcorn[2ci+di, 2cj+dj, 2k] uses pillar (ci+di, cj+dj) for layer-k top
    # zcorn[2ci+di, 2cj+dj, 2k+1] uses same pillar for layer-k bottom
    zcorn = np.zeros((2 * ni, 2 * nj, 2 * nk), dtype=np.float64)

    for pi in range(ni + 1):
        for pj in range(nj + 1):
            zvals = sorted(set(pillar_z.get((pi, pj), [z_global_top, z_global_bot])))
            if len(zvals) < 2:
                zvals = [z_global_top, z_global_bot]
            # Interpolate to exactly nk+1 layer boundaries
            if len(zvals) == nk + 1:
                z_bounds = np.array(zvals, dtype=np.float64)
            else:
                z_bounds = np.linspace(zvals[0], zvals[-1], nk + 1)

            # Write into every ZCORN slot that references this pillar
            for ci in range(max(0, pi - 1), min(ni, pi + 1)):
                di = pi - ci          # 0 or 1
                for cj in range(max(0, pj - 1), min(nj, pj + 1)):
                    dj = pj - cj      # 0 or 1
                    for k in range(nk):
                        zcorn[2 * ci + di, 2 * cj + dj, 2 * k]     = z_bounds[k]
                        zcorn[2 * ci + di, 2 * cj + dj, 2 * k + 1] = z_bounds[k + 1]

    # --- #2: ACTNUM from stacksets valid_geometry flags ---
    active_cols: set[tuple[int, int]] = {
        (int(ss_i[k]), int(ss_j[k])) for k in range(len(ss_i)) if ss_vgeo[k]
    }
    if not active_cols:
        active_cols = {(int(ss_i[k]), int(ss_j[k])) for k in range(len(ss_i))}

    actnum = np.zeros(ni * nj * nk, dtype=np.int32)
    for ci in range(ni):
        for cj in range(nj):
            if (ci, cj) in active_cols:
                base = ci * nj * nk + cj * nk
                actnum[base: base + nk] = 1

    return coord, zcorn, actnum


# ---------------------------------------------------------------------------
# Stair-step fallback
# ---------------------------------------------------------------------------

def _stair_step_fallback(
    path: Path,
    nodes: np.ndarray,
    ni: int, nj: int, nk: int,
    properties: list[GridProperty],
) -> BridgeModel:
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    cell_dx = (x.max() - x.min()) / ni
    cell_dy = (y.max() - y.min()) / nj
    cell_dz = (z.max() - z.min()) / nk
    z0 = float(z.min())

    dz_arr = np.full((ni, nj, nk), cell_dz, dtype=np.float64)
    tops_arr = np.zeros((ni, nj, nk), dtype=np.float64)
    for k in range(nk):
        tops_arr[:, :, k] = z0 + k * cell_dz

    dims = GridDimensions(ni=ni, nj=nj, nk=nk)
    grid = StairStepGrid(
        dims=dims,
        origin=(float(x.min()), float(y.min()), z0),
        dx=np.full(ni, cell_dx, dtype=np.float64),
        dy=np.full(nj, cell_dy, dtype=np.float64),
        dz=dz_arr,
        tops=tops_arr,
        actnum=np.ones(ni * nj * nk, dtype=np.int32),
        source_file=path,
    )
    return BridgeModel(
        grid_type=GridType.STAIR_STEP,
        stair_step=grid,
        properties=properties,
        source_software="JewelSuite",
        project_name=path.stem,
    )


# ---------------------------------------------------------------------------
# Write stub
# ---------------------------------------------------------------------------

def write_jewel(model: BridgeModel, path: str | Path) -> None:
    """Write a BridgeModel to JewelSuite .jewel format (not yet implemented)."""
    raise NotImplementedError(
        "Writing .jewel files is not yet supported. "
        "Export to RESQML (.epc) and import via JewelSuite's RESQML reader instead."
    )
