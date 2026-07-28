"""
Eclipse simulation result reader (INIT + UNRST / RESTART).

Reads static properties from the INIT file and dynamic time-step properties
from UNRST (unified restart) files using resfo.  Falls back to opm-common
when available for richer metadata (simulation dates, unit conventions).

References
----------
equinor/resfo    : https://github.com/equinor/resfo
OPM/opm-common  : https://github.com/OPM/opm-common
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.grid_models import GridProperty, PropertyType


_PROP_MAP: dict[str, PropertyType] = {
    "PORO": PropertyType.POROSITY,
    "PERMX": PropertyType.PERMEABILITY_X,
    "PERMY": PropertyType.PERMEABILITY_Y,
    "PERMZ": PropertyType.PERMEABILITY_Z,
    "SW":    PropertyType.WATER_SATURATION,
    "NTG":   PropertyType.NET_TO_GROSS,
    "PRESSURE": PropertyType.PRESSURE,
    "STRESS_XX": PropertyType.STRESS_XX,
    "STRESS_YY": PropertyType.STRESS_YY,
    "STRESS_ZZ": PropertyType.STRESS_ZZ,
}

_STATIC_PROPS = {"PORO", "PERMX", "PERMY", "PERMZ", "NTG", "ACTNUM"}
_DYNAMIC_PROPS = {"PRESSURE", "SWAT", "SGAS", "SOIL", "RS", "RV",
                  "STRESS_XX", "STRESS_YY", "STRESS_ZZ"}


@dataclass
class SimulationStep:
    """A single time-step snapshot from an UNRST file."""
    step_index: int
    days: float
    date_str: str
    properties: list[GridProperty]


def _resfo_kw_map(path: Path) -> dict[str, np.ndarray]:
    try:
        import resfo  # type: ignore
    except ImportError as exc:
        raise ImportError("resfo required: pip install resfo") from exc
    return {kw.strip(): np.asarray(arr) for kw, arr in resfo.lazy_read(path)}


def read_init(path: str | Path, n_cells: int) -> list[GridProperty]:
    """
    Read an Eclipse INIT file and return static cell properties.

    Parameters
    ----------
    path    : path to the .INIT file
    n_cells : total cell count (ni*nj*nk) used to filter non-cell arrays
    """
    path = Path(path)
    kw = _resfo_kw_map(path)
    props: list[GridProperty] = []

    for name, arr in kw.items():
        if arr.size != n_cells:
            continue
        prop_type = _PROP_MAP.get(name, PropertyType.CUSTOM)
        is_disc = name in ("SATNUM", "PVTNUM", "EQLNUM", "FIPNUM")
        props.append(
            GridProperty(
                name=name,
                prop_type=prop_type,
                values=arr.astype(np.int32 if is_disc else np.float32),
                is_discrete=is_disc,
            )
        )
    return props


def read_unrst(
    path: str | Path,
    n_cells: int,
    step_indices: list[int] | None = None,
) -> list[SimulationStep]:
    """
    Read an Eclipse UNRST (unified restart) file.

    Parameters
    ----------
    path          : path to the .UNRST file
    n_cells       : total cell count used to identify cell arrays
    step_indices  : which time steps to load (None = all)

    Returns
    -------
    List of SimulationStep objects, each with a properties list.
    """
    path = Path(path)
    try:
        import resfo  # type: ignore
    except ImportError as exc:
        raise ImportError("resfo required: pip install resfo") from exc

    steps: list[SimulationStep] = []
    current_props: list[GridProperty] = []
    current_days: float = 0.0
    current_date: str = ""
    current_idx: int = -1
    seqnum: int = 0

    for kw_raw, arr_raw in resfo.lazy_read(path):
        kw = kw_raw.strip()
        arr = np.asarray(arr_raw)

        if kw == "SEQNUM":
            # Flush previous step
            if current_idx >= 0 and current_props:
                if step_indices is None or current_idx in step_indices:
                    steps.append(
                        SimulationStep(
                            step_index=current_idx,
                            days=current_days,
                            date_str=current_date,
                            properties=current_props,
                        )
                    )
            current_idx = int(arr[0])
            current_props = []
            seqnum += 1

        elif kw == "DOUBHEAD":
            current_days = float(arr[0])

        elif kw == "INTEHEAD" and len(arr) >= 66:
            # INTEHEAD[64..66] = day, month, year
            try:
                day, month, year = int(arr[64]), int(arr[65]), int(arr[66])
                current_date = f"{year:04d}-{month:02d}-{day:02d}"
            except (IndexError, ValueError):
                current_date = ""

        elif arr.size == n_cells:
            if step_indices is None or current_idx in step_indices:
                prop_type = _PROP_MAP.get(kw, PropertyType.CUSTOM)
                current_props.append(
                    GridProperty(
                        name=kw,
                        prop_type=prop_type,
                        values=arr.astype(np.float32),
                        timestep_idx=current_idx,
                        timestep_days=current_days,
                    )
                )

    # Flush the last step
    if current_idx >= 0 and current_props:
        if step_indices is None or current_idx in step_indices:
            steps.append(
                SimulationStep(
                    step_index=current_idx,
                    days=current_days,
                    date_str=current_date,
                    properties=current_props,
                )
            )

    return steps


def read_unrst_opm(
    path: str | Path, step_indices: list[int] | None = None
) -> list[SimulationStep]:
    """
    Alternative reader using opm-common Python bindings.
    Provides unit conversion, well data, and richer date handling.
    """
    path = Path(path)
    try:
        from opm.io.ecl import EclFile, EGrid  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "opm-common required: install opm-common with Python bindings"
        ) from exc

    rst = EclFile(str(path))
    steps: list[SimulationStep] = []

    for report_step in rst.report_steps:
        if step_indices is not None and report_step not in step_indices:
            continue
        days = float(rst.iget_named_kw("DOUBHEAD", report_step)[0])
        props: list[GridProperty] = []
        for name in _DYNAMIC_PROPS:
            try:
                data = rst.iget_named_kw(name, report_step)
                arr = np.array(data, dtype=np.float32)
                props.append(
                    GridProperty(
                        name=name,
                        prop_type=_PROP_MAP.get(name, PropertyType.CUSTOM),
                        values=arr,
                        timestep_idx=report_step,
                        timestep_days=days,
                    )
                )
            except Exception:
                pass
        steps.append(
            SimulationStep(
                step_index=report_step,
                days=days,
                date_str="",
                properties=props,
            )
        )
    return steps
