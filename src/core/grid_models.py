"""
Core grid data models shared across all converters.

These dataclasses are the canonical in-memory representation.  Every reader
produces one of these; every writer consumes one.  Keeping a single neutral
model avoids pairwise converter explosion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np


class GridType(Enum):
    CORNER_POINT = auto()    # Petrel pillar / GRDECL corner-point
    STAIR_STEP = auto()      # JewelSuite stair-step (IJK orthogonal)
    JEWEL_GRID = auto()      # JewelGrid (curvilinear, non-pillar)
    TETRAHEDRAL = auto()     # 3-D unstructured tet mesh (Visage / JewelSuite)
    VOXEL = auto()           # Regular Cartesian voxel (Petrel compatibility fallback)


class PropertyType(Enum):
    POROSITY = "PORO"
    PERMEABILITY_X = "PERMX"
    PERMEABILITY_Y = "PERMY"
    PERMEABILITY_Z = "PERMZ"
    WATER_SATURATION = "SW"
    NET_TO_GROSS = "NTG"
    FACIES = "FACIES"
    PRESSURE = "PRESSURE"
    STRESS_XX = "STRESS_XX"
    STRESS_YY = "STRESS_YY"
    STRESS_ZZ = "STRESS_ZZ"
    CUSTOM = "CUSTOM"


@dataclass
class GridDimensions:
    ni: int
    nj: int
    nk: int

    @property
    def ncells(self) -> int:
        return self.ni * self.nj * self.nk


@dataclass
class CornerPointGrid:
    """
    Petrel-style corner-point (pillar) grid.

    coord  : (ni+1, nj+1, 6) float64 — pillar top/bottom XYZ pairs
    zcorn  : (2*ni, 2*nj, 2*nk) float64 — cell corner depths
    actnum : (ni*nj*nk,) int32  — 1=active, 0=inactive
    """
    dims: GridDimensions
    coord: np.ndarray
    zcorn: np.ndarray
    actnum: np.ndarray
    source_file: Optional[Path] = None
    crs: Optional[str] = None  # WKT or EPSG string

    def active_count(self) -> int:
        return int(self.actnum.sum())


@dataclass
class StairStepGrid:
    """
    Regular IJK grid where all cell faces are orthogonal to axes.

    origin  : (x0, y0, z0)
    dx, dy  : (ni,) and (nj,) cell sizes
    dz      : (ni, nj, nk) — variable layer thickness per column
    tops    : (ni, nj, nk) — top depth of each cell
    actnum  : (ni*nj*nk,) int32
    """
    dims: GridDimensions
    origin: tuple[float, float, float]
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    tops: np.ndarray
    actnum: np.ndarray
    source_file: Optional[Path] = None


@dataclass
class JewelGrid:
    """
    JewelSuite curvilinear JewelGrid.

    nodes   : (n_nodes, 3) float64 — XYZ node coordinates
    cells   : (n_cells, 8) int32   — 8 node indices per hex cell
    actnum  : (n_cells,) int32
    """
    nodes: np.ndarray
    cells: np.ndarray
    actnum: np.ndarray
    dims: Optional[GridDimensions] = None  # set when structured
    source_file: Optional[Path] = None


@dataclass
class TetrahedralMesh:
    """
    Unstructured tetrahedral mesh (Visage / JewelSuite mesh solver).

    nodes   : (n_nodes, 3) float64
    tets    : (n_tets, 4) int32 — node indices per tetrahedron
    """
    nodes: np.ndarray
    tets: np.ndarray
    source_file: Optional[Path] = None


@dataclass
class GridProperty:
    """
    A single cell property array attached to any grid type.

    values       : 1-D float/int array aligned to cell ordering of host grid
    timestep_idx : None for static; integer index for dynamic results
    timestep_days: simulation time in days from start
    """
    name: str
    prop_type: PropertyType
    values: np.ndarray
    unit: str = ""
    timestep_idx: Optional[int] = None
    timestep_days: Optional[float] = None
    is_discrete: bool = False


@dataclass
class StructuralFramework:
    """
    Faults, horizons, and zone boundaries extracted from either platform.
    """

    @dataclass
    class FaultSurface:
        name: str
        vertices: np.ndarray    # (n, 3) float64
        triangles: np.ndarray   # (m, 3) int32
        throw_avg_m: Optional[float] = None

    @dataclass
    class HorizonSurface:
        name: str
        vertices: np.ndarray
        triangles: np.ndarray
        depth_avg_m: Optional[float] = None

    @dataclass
    class Zone:
        name: str
        top_horizon: str
        base_horizon: str
        layer_count: int

    faults: list[FaultSurface] = field(default_factory=list)
    horizons: list[HorizonSurface] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)


@dataclass
class BridgeModel:
    """
    Top-level container passed between all converters.

    Exactly one of the grid variants is populated; the rest are None.
    Properties and structural framework are always optional addons.
    """
    grid_type: GridType
    corner_point: Optional[CornerPointGrid] = None
    stair_step: Optional[StairStepGrid] = None
    jewel_grid: Optional[JewelGrid] = None
    tet_mesh: Optional[TetrahedralMesh] = None

    properties: list[GridProperty] = field(default_factory=list)
    framework: Optional[StructuralFramework] = None

    project_name: str = ""
    source_software: str = ""  # "Petrel" | "JewelSuite" | "Eclipse" | ...

    def active_grid(self):
        """Return whichever grid object is populated."""
        return (
            self.corner_point
            or self.stair_step
            or self.jewel_grid
            or self.tet_mesh
        )
