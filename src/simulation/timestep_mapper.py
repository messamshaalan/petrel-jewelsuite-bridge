"""
Time-step aware property mapper for dynamic simulation results.

Handles:
  • Linear interpolation between time steps (for in-between dates)
  • Geomechanical stress/strain tensor transfer (Visage output → standard grids)
  • PRESSURE, SWAT, SGAS, SOIL series mapping
  • Unit normalisation (field → SI, SI → field)

Visage output parser follows the pattern described in the gap analysis:
  Visage outputs .vgr / .vre files — this parser handles ASCII Visage output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.grid_models import GridProperty, PropertyType


_STRESS_PROPS = {
    "STRESS_XX", "STRESS_YY", "STRESS_ZZ",
    "STRESS_XY", "STRESS_XZ", "STRESS_YZ",
    "STRAIN_XX", "STRAIN_YY", "STRAIN_ZZ",
    "DISP_X", "DISP_Y", "DISP_Z",
}


@dataclass
class TimeSeries:
    """Collection of snapshots for a single property over multiple time steps."""
    prop_name: str
    prop_type: PropertyType
    unit: str
    days: list[float]
    snapshots: list[np.ndarray]   # each (n_cells,)

    def at(self, target_days: float) -> np.ndarray:
        """
        Return the property array interpolated to `target_days`.
        Clamps to first/last available step outside the range.
        """
        if len(self.days) == 0:
            raise ValueError(f"TimeSeries '{self.prop_name}' has no data")
        if target_days <= self.days[0]:
            return self.snapshots[0].copy()
        if target_days >= self.days[-1]:
            return self.snapshots[-1].copy()

        for i in range(len(self.days) - 1):
            t0, t1 = self.days[i], self.days[i + 1]
            if t0 <= target_days <= t1:
                alpha = (target_days - t0) / (t1 - t0)
                return ((1 - alpha) * self.snapshots[i].astype(np.float64)
                        + alpha * self.snapshots[i + 1].astype(np.float64)).astype(np.float32)

        return self.snapshots[-1].copy()

    def as_property_list(self) -> list[GridProperty]:
        props = []
        for days, snap in zip(self.days, self.snapshots):
            props.append(
                GridProperty(
                    name=self.prop_name,
                    prop_type=self.prop_type,
                    values=snap,
                    unit=self.unit,
                    timestep_days=days,
                )
            )
        return props


def build_time_series(properties: list[GridProperty]) -> dict[str, TimeSeries]:
    """
    Group a flat list of GridProperty (possibly multi-step) into TimeSeries
    objects keyed by property name.
    """
    groups: dict[str, list[tuple[float, np.ndarray]]] = {}
    types: dict[str, PropertyType] = {}
    units: dict[str, str] = {}

    for prop in properties:
        days = prop.timestep_days if prop.timestep_days is not None else 0.0
        groups.setdefault(prop.name, []).append((days, prop.values))
        types[prop.name] = prop.prop_type
        units[prop.name] = prop.unit

    series: dict[str, TimeSeries] = {}
    for name, entries in groups.items():
        entries.sort(key=lambda x: x[0])
        series[name] = TimeSeries(
            prop_name=name,
            prop_type=types[name],
            unit=units[name],
            days=[e[0] for e in entries],
            snapshots=[e[1] for e in entries],
        )
    return series


def extract_at_days(
    series_map: dict[str, TimeSeries],
    target_days: float,
    prop_names: list[str] | None = None,
) -> list[GridProperty]:
    """
    Interpolate all (or selected) time series to `target_days`.

    Parameters
    ----------
    series_map  : from build_time_series()
    target_days : simulation time in days from start
    prop_names  : which properties to extract (None = all)
    """
    names = prop_names if prop_names is not None else list(series_map)
    out: list[GridProperty] = []
    for name in names:
        ts = series_map.get(name)
        if ts is None:
            continue
        arr = ts.at(target_days)
        out.append(
            GridProperty(
                name=name,
                prop_type=ts.prop_type,
                values=arr,
                unit=ts.unit,
                timestep_days=target_days,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Visage geomechanical output reader
# ---------------------------------------------------------------------------

def read_visage_output(path: str | Path, n_cells: int) -> list[GridProperty]:
    """
    Parse an ASCII Visage output file (.vre or .vgr) and return stress/strain
    properties.

    Visage ASCII format:
        KEYWORD
        val1  val2  val3  ...
        /
    """
    path = Path(path)
    results: list[GridProperty] = []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("--") or line.startswith("!"):
            i += 1
            continue

        kw = line.split()[0].upper()
        if kw in _STRESS_PROPS or kw.startswith("STRESS") or kw.startswith("STRAIN"):
            # Collect values until '/'
            vals: list[float] = []
            i += 1
            while i < len(lines):
                ln = lines[i].strip()
                if ln == "/":
                    i += 1
                    break
                for tok in ln.split():
                    if tok == "/":
                        break
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        pass
                i += 1

            if len(vals) == n_cells:
                prop_type = (
                    PropertyType.STRESS_XX if "XX" in kw else
                    PropertyType.STRESS_YY if "YY" in kw else
                    PropertyType.STRESS_ZZ if "ZZ" in kw else
                    PropertyType.CUSTOM
                )
                results.append(
                    GridProperty(
                        name=kw,
                        prop_type=prop_type,
                        values=np.array(vals, dtype=np.float32),
                        unit="Pa",
                    )
                )
        else:
            i += 1

    return results


def normalise_stress(
    props: list[GridProperty],
    *,
    from_unit: str = "Pa",
    to_unit: str = "psi",
) -> list[GridProperty]:
    """Convert stress/strain properties between unit systems."""
    factors = {
        ("Pa", "psi"):  1.0 / 6894.757,
        ("psi", "Pa"):  6894.757,
        ("Pa", "MPa"):  1e-6,
        ("MPa", "Pa"):  1e6,
        ("Pa", "bar"):  1e-5,
        ("bar", "Pa"):  1e5,
    }
    factor = factors.get((from_unit, to_unit), 1.0)
    out = []
    for prop in props:
        if prop.name in _STRESS_PROPS or prop.prop_type in (
            PropertyType.STRESS_XX, PropertyType.STRESS_YY, PropertyType.STRESS_ZZ
        ):
            out.append(
                GridProperty(
                    name=prop.name,
                    prop_type=prop.prop_type,
                    values=(prop.values.astype(np.float64) * factor).astype(np.float32),
                    unit=to_unit,
                    timestep_idx=prop.timestep_idx,
                    timestep_days=prop.timestep_days,
                )
            )
        else:
            out.append(prop)
    return out
