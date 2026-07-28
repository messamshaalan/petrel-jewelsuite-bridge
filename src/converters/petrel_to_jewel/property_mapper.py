"""
Property array mapper: corner-point cell ordering → JewelGrid cell ordering.

For structured grids (ni×nj×nk) the cell ordering is identical in both
platforms (I fastest, then J, then K).  This module handles the edge cases:

  • Inactive-cell masking (JewelSuite may use a compact active-only list)
  • Unit conversion (mD, fraction, psi → SI or field units)
  • Discrete/continuous type detection
  • Time-step series mapping
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.grid_models import GridProperty, GridDimensions, PropertyType


# ------------------------------------------------------------------ #
# Unit conversion factors (multiply by factor to get target unit)     #
# ------------------------------------------------------------------ #

UNIT_FACTORS: dict[tuple[str, str], float] = {
    ("mD",  "m2"):     9.869233e-16,
    ("m2",  "mD"):     1.0 / 9.869233e-16,
    ("psi", "Pa"):     6894.757,
    ("Pa",  "psi"):    1.0 / 6894.757,
    ("bar", "Pa"):     1e5,
    ("Pa",  "bar"):    1e-5,
    ("ft",  "m"):      0.3048,
    ("m",   "ft"):     1.0 / 0.3048,
}


@dataclass
class MappingReport:
    """Result of a property mapping operation."""
    n_properties: int
    n_active_cells: int
    unit_conversions: list[str]
    clipped_properties: list[str]
    warnings: list[str]


def map_properties(
    properties: list[GridProperty],
    dims: GridDimensions,
    actnum: np.ndarray,
    *,
    compact: bool = False,
    unit_overrides: dict[str, tuple[str, str]] | None = None,
    value_clamps: dict[str, tuple[float, float]] | None = None,
) -> tuple[list[GridProperty], MappingReport]:
    """
    Remap property arrays from full-grid (ni×nj×nk) ordering to JewelGrid
    ordering.

    Parameters
    ----------
    properties     : source GridProperty list (full-grid cell ordering)
    dims           : grid dimensions
    actnum         : (ncells,) int32 — 1=active, 0=inactive
    compact        : if True, output arrays contain only active cells
    unit_overrides : {prop_name: (from_unit, to_unit)} — apply unit conversion
    value_clamps   : {prop_name: (min, max)} — clamp property values

    Returns
    -------
    Tuple of (remapped properties list, MappingReport)
    """
    unit_overrides = unit_overrides or {}
    value_clamps = value_clamps or {}

    report_units: list[str] = []
    report_clips: list[str] = []
    report_warns: list[str] = []

    active_mask = actnum.astype(bool)
    n_active = int(active_mask.sum())

    out_props: list[GridProperty] = []

    for prop in properties:
        arr = prop.values.copy()

        # Validate size
        if arr.size != dims.ncells:
            report_warns.append(
                f"{prop.name}: size mismatch ({arr.size} vs {dims.ncells}), skipped"
            )
            continue

        # Unit conversion
        if prop.name in unit_overrides:
            from_u, to_u = unit_overrides[prop.name]
            factor = UNIT_FACTORS.get((from_u, to_u))
            if factor is not None:
                arr = arr * factor
                unit = to_u
                report_units.append(f"{prop.name}: {from_u} → {to_u} (×{factor:.3e})")
            else:
                unit = prop.unit
                report_warns.append(
                    f"{prop.name}: unknown unit pair ({from_u}, {to_u}), no conversion applied"
                )
        else:
            unit = prop.unit

        # Value clamping
        if prop.name in value_clamps:
            lo, hi = value_clamps[prop.name]
            n_clipped = int(np.sum((arr < lo) | (arr > hi)))
            if n_clipped:
                arr = np.clip(arr, lo, hi)
                report_clips.append(f"{prop.name}: {n_clipped} values clamped to [{lo}, {hi}]")

        # Property-type-specific sanity bounds
        _apply_physical_bounds(prop.prop_type, arr, prop.name, report_warns)

        # Apply compact masking (active-cells only)
        if compact:
            arr = arr[active_mask]

        out_props.append(
            GridProperty(
                name=prop.name,
                prop_type=prop.prop_type,
                values=arr,
                unit=unit,
                timestep_idx=prop.timestep_idx,
                timestep_days=prop.timestep_days,
                is_discrete=prop.is_discrete,
            )
        )

    report = MappingReport(
        n_properties=len(out_props),
        n_active_cells=n_active,
        unit_conversions=report_units,
        clipped_properties=report_clips,
        warnings=report_warns,
    )
    return out_props, report


def _apply_physical_bounds(
    prop_type: PropertyType,
    arr: np.ndarray,
    name: str,
    warnings: list[str],
) -> None:
    """Warn (do not clamp) if values lie outside physical domain."""
    bounds: dict[PropertyType, tuple[float, float]] = {
        PropertyType.POROSITY:        (0.0, 1.0),
        PropertyType.WATER_SATURATION:(0.0, 1.0),
        PropertyType.NET_TO_GROSS:    (0.0, 1.0),
        PropertyType.PERMEABILITY_X:  (0.0, 1e6),
        PropertyType.PERMEABILITY_Y:  (0.0, 1e6),
        PropertyType.PERMEABILITY_Z:  (0.0, 1e6),
    }
    if prop_type in bounds:
        lo, hi = bounds[prop_type]
        n_out = int(np.sum((arr < lo) | (arr > hi)))
        if n_out:
            warnings.append(
                f"{name}: {n_out} values outside physical range [{lo}, {hi}]"
            )


def interpolate_timestep(
    prop_a: GridProperty,
    prop_b: GridProperty,
    target_days: float,
) -> GridProperty:
    """
    Linearly interpolate between two time-step snapshots of the same property.

    Used for time-step aware property mapping when the target time does not
    exactly coincide with a simulator output step.
    """
    if prop_a.name != prop_b.name:
        raise ValueError("Cannot interpolate properties with different names")
    t_a = prop_a.timestep_days or 0.0
    t_b = prop_b.timestep_days or 0.0
    span = t_b - t_a
    if abs(span) < 1e-10:
        return prop_a

    alpha = (target_days - t_a) / span
    alpha = float(np.clip(alpha, 0.0, 1.0))
    values = (1.0 - alpha) * prop_a.values.astype(np.float64) + alpha * prop_b.values.astype(np.float64)

    return GridProperty(
        name=prop_a.name,
        prop_type=prop_a.prop_type,
        values=values.astype(np.float32),
        unit=prop_a.unit,
        timestep_idx=None,
        timestep_days=target_days,
    )
