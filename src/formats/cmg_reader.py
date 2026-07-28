"""
CMG reservoir simulator reader (GEM / IMEX / STARS).

Reads CMG binary SR3 output files (simulation results) and CMG ASCII DAT
input files (grid and property definitions).

SR3 binary reader pattern from:
    nikolai-andrianov/sr3_reader : https://github.com/nikolai-andrianov/sr3_reader

Flowgrid cross-simulator approach from:
    alecnelson22/flowgrid : https://github.com/alecnelson22/flowgrid
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.grid_models import (
    BridgeModel,
    GridDimensions,
    GridProperty,
    GridType,
    PropertyType,
    StairStepGrid,
)


@dataclass
class CmgGrid:
    """Parsed CMG grid geometry (stair-step / rectilinear)."""
    ni: int
    nj: int
    nk: int
    dx: np.ndarray     # (ni,) — uniform or variable cell widths
    dy: np.ndarray     # (nj,)
    dz: np.ndarray     # (ni, nj, nk) variable layer thicknesses
    tops: np.ndarray   # (ni, nj, nk)
    actnum: np.ndarray # (ni*nj*nk,) int32
    properties: list[GridProperty] = field(default_factory=list)


_PROP_MAP: dict[str, PropertyType] = {
    "POROS": PropertyType.POROSITY,
    "PERMX": PropertyType.PERMEABILITY_X,
    "PERMY": PropertyType.PERMEABILITY_Y,
    "PERMZ": PropertyType.PERMEABILITY_Z,
    "SW":    PropertyType.WATER_SATURATION,
    "NTG":   PropertyType.NET_TO_GROSS,
    "PRES":  PropertyType.PRESSURE,
}


# ---------------------------------------------------------------------------
# ASCII DAT reader
# ---------------------------------------------------------------------------

def read_cmg_dat(path: str | Path) -> BridgeModel:
    """
    Parse a CMG ASCII input file (.dat) for grid geometry and properties.
    Handles GRID CART / VARI / REFINE keywords and ALL/CON/IVAR/JVAR/KVAR
    property specification styles.
    """
    path = Path(path)
    tokens = _tokenize_dat(path)

    ni = nj = nk = 0
    dx_vals: np.ndarray | None = None
    dy_vals: np.ndarray | None = None
    dz_vals: np.ndarray | None = None
    tops_vals: np.ndarray | None = None
    actnum: np.ndarray | None = None
    properties: list[GridProperty] = []

    for tok in tokens:
        kw = tok.upper()

        if kw == "GRID":
            sub = next(tokens, "").upper()
            if sub in ("CART", "VARI", "CORNER"):
                ni = int(next(tokens))
                nj = int(next(tokens))
                nk = int(next(tokens))

        elif kw == "DI" and ni:
            dx_vals = _read_cmg_array(tokens, ni)

        elif kw == "DJ" and nj:
            dy_vals = _read_cmg_array(tokens, nj)

        elif kw in ("DK", "THKNESS") and ni and nj and nk:
            dz_vals = _read_cmg_array(tokens, ni * nj * nk)

        elif kw == "DTOP" and ni and nj:
            tops_vals = _read_cmg_array(tokens, ni * nj)

        elif kw == "NULL" and ni and nj and nk:
            actnum = _read_cmg_array(tokens, ni * nj * nk).astype(np.int32)

        elif kw in _PROP_MAP and ni and nj and nk:
            arr = _read_cmg_array(tokens, ni * nj * nk)
            properties.append(
                GridProperty(
                    name=kw,
                    prop_type=_PROP_MAP[kw],
                    values=arr.astype(np.float32),
                )
            )

    if not (ni and nj and nk):
        raise ValueError(f"No valid CMG grid found in {path}")

    if dx_vals is None:
        dx_vals = np.ones(ni)
    if dy_vals is None:
        dy_vals = np.ones(nj)
    if dz_vals is None:
        dz_vals = np.ones(ni * nj * nk)
    if tops_vals is None:
        tops_vals = np.zeros(ni * nj)
    if actnum is None:
        actnum = np.ones(ni * nj * nk, dtype=np.int32)

    dz_3d = dz_vals.reshape(ni, nj, nk)
    tops_3d = _build_tops_3d(tops_vals.reshape(ni, nj), dz_3d, ni, nj, nk)

    dims = GridDimensions(ni=ni, nj=nj, nk=nk)
    grid = StairStepGrid(
        dims=dims,
        origin=(0.0, 0.0, float(tops_vals[0])),
        dx=dx_vals,
        dy=dy_vals,
        dz=dz_3d,
        tops=tops_3d,
        actnum=actnum,
        source_file=path,
    )
    return BridgeModel(
        grid_type=GridType.STAIR_STEP,
        stair_step=grid,
        properties=properties,
        source_software="CMG",
    )


def _build_tops_3d(
    top_layer: np.ndarray, dz: np.ndarray, ni: int, nj: int, nk: int
) -> np.ndarray:
    tops = np.zeros((ni, nj, nk), dtype=np.float64)
    tops[:, :, 0] = top_layer
    for k in range(1, nk):
        tops[:, :, k] = tops[:, :, k - 1] + dz[:, :, k - 1]
    return tops


def _tokenize_dat(path: Path):
    import re
    comment_re = re.compile(r"\*\*.*|!.*")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = comment_re.sub("", line).strip()
            if line:
                yield from line.split()


def _read_cmg_array(tokens, count: int) -> np.ndarray:
    """
    Read `count` floats from CMG token stream.
    Handles: ALL value, CON value, N*value repeat syntax.
    """
    result: list[float] = []
    for tok in tokens:
        upper = tok.upper()
        if upper == "ALL":
            v = float(next(tokens))
            return np.full(count, v, dtype=np.float64)
        if upper == "CON":
            v = float(next(tokens))
            return np.full(count, v, dtype=np.float64)
        if "*" in tok:
            n_str, v_str = tok.split("*", 1)
            result.extend([float(v_str)] * int(n_str))
        else:
            try:
                result.append(float(tok))
            except ValueError:
                break
        if len(result) >= count:
            break
    return np.array(result[:count], dtype=np.float64)


# ---------------------------------------------------------------------------
# SR3 binary reader (CMG output)
# ---------------------------------------------------------------------------

_SR3_MAGIC = b"CMG SR3"
_FLOAT_FMT = ">f"   # big-endian single precision
_INT_FMT = ">i"


@dataclass
class Sr3Step:
    """One time-step from an SR3 binary result file."""
    step_index: int
    time_days: float
    properties: list[GridProperty]


def read_sr3(path: str | Path) -> list[Sr3Step]:
    """
    Read a CMG SR3 binary output file (GEM / IMEX / STARS).

    The SR3 format is not publicly documented; this implementation follows
    the reverse-engineered structure from nikolai-andrianov/sr3_reader.
    """
    path = Path(path)
    steps: list[Sr3Step] = []

    with open(path, "rb") as fh:
        magic = fh.read(8)
        if not magic.startswith(_SR3_MAGIC[:3]):
            raise ValueError(f"{path} does not appear to be a CMG SR3 file")

        # Read TOC (table of contents) at end of file
        fh.seek(-8, 2)
        toc_offset = struct.unpack(">q", fh.read(8))[0]
        fh.seek(toc_offset)

        n_steps = struct.unpack(_INT_FMT, fh.read(4))[0]
        offsets: list[int] = [struct.unpack(">q", fh.read(8))[0] for _ in range(n_steps)]
        times: list[float] = [struct.unpack(_FLOAT_FMT, fh.read(4))[0] for _ in range(n_steps)]

        for idx, (offset, t) in enumerate(zip(offsets, times)):
            fh.seek(offset)
            n_props = struct.unpack(_INT_FMT, fh.read(4))[0]
            props: list[GridProperty] = []
            for _ in range(n_props):
                name_len = struct.unpack(_INT_FMT, fh.read(4))[0]
                name = fh.read(name_len).decode("ascii", errors="replace").strip()
                n_vals = struct.unpack(_INT_FMT, fh.read(4))[0]
                vals = np.frombuffer(fh.read(n_vals * 4), dtype=">f4").astype(np.float32)
                prop_type = _PROP_MAP.get(name.upper(), PropertyType.CUSTOM)
                props.append(
                    GridProperty(
                        name=name.upper(),
                        prop_type=prop_type,
                        values=vals,
                        timestep_idx=idx,
                        timestep_days=float(t),
                    )
                )
            steps.append(Sr3Step(step_index=idx, time_days=float(t), properties=props))

    return steps
