"""
Structural framework converter: faults, horizons, and zone boundaries.

Handles bi-directional transfer of:
  • Fault surfaces (triangulated)        — Petrel ↔ JewelSuite
  • Horizon surfaces (gridded or TIN)    — Petrel ↔ JewelSuite
  • Zone boundaries                      — purely metadata

Algorithms
----------
• Fault triangulation: Delaunay 2.5D triangulation of fault plane point clouds
• Horizon resampling: RBF interpolation onto target IJK grid nodes
• Surface smoothing: Laplacian smoothing via PyVista or trimesh

References
----------
softwareunderground/subsurface: https://github.com/softwareunderground/subsurface
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.grid_models import StructuralFramework


@dataclass
class FrameworkQC:
    n_faults: int
    n_horizons: int
    n_zones: int
    degenerate_triangles: int
    max_fault_throw_deviation_m: float
    warnings: list[str]


def convert_framework(
    framework: StructuralFramework,
    *,
    smooth_faults: bool = True,
    resample_resolution_m: float | None = None,
) -> tuple[StructuralFramework, FrameworkQC]:
    """
    Validate, repair, and optionally smooth/resample a StructuralFramework.

    Returns a new framework and a QC report.
    """
    warnings: list[str] = []
    total_degen = 0

    new_faults: list[StructuralFramework.FaultSurface] = []
    for fault in framework.faults:
        verts, tris, degen, warn = _process_surface(
            fault.vertices, fault.triangles,
            name=fault.name,
            smooth=smooth_faults,
        )
        total_degen += degen
        warnings.extend(warn)
        throw = _estimate_fault_throw(verts, tris)
        new_faults.append(
            StructuralFramework.FaultSurface(
                name=fault.name,
                vertices=verts,
                triangles=tris,
                throw_avg_m=throw,
            )
        )

    new_horizons: list[StructuralFramework.HorizonSurface] = []
    for hor in framework.horizons:
        verts, tris, degen, warn = _process_surface(
            hor.vertices, hor.triangles,
            name=hor.name,
            smooth=False,
            target_spacing=resample_resolution_m,
        )
        total_degen += degen
        warnings.extend(warn)
        depth_avg = float(verts[:, 2].mean()) if len(verts) else None
        new_horizons.append(
            StructuralFramework.HorizonSurface(
                name=hor.name,
                vertices=verts,
                triangles=tris,
                depth_avg_m=depth_avg,
            )
        )

    max_throw_dev = _max_throw_deviation(new_faults, framework.faults)

    qc = FrameworkQC(
        n_faults=len(new_faults),
        n_horizons=len(new_horizons),
        n_zones=len(framework.zones),
        degenerate_triangles=total_degen,
        max_fault_throw_deviation_m=max_throw_dev,
        warnings=warnings,
    )

    out = StructuralFramework(
        faults=new_faults,
        horizons=new_horizons,
        zones=list(framework.zones),
    )
    return out, qc


def _process_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    name: str = "",
    smooth: bool = False,
    target_spacing: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    """Repair, optionally smooth, and optionally resample a triangulated surface."""
    warnings: list[str] = []
    verts = vertices.astype(np.float64)
    tris = triangles.astype(np.int32)

    if len(verts) == 0 or len(tris) == 0:
        return verts, tris, 0, [f"{name}: empty surface"]

    # Remove degenerate triangles (zero area)
    degen_mask = _degenerate_mask(verts, tris)
    n_degen = int(degen_mask.sum())
    if n_degen:
        tris = tris[~degen_mask]
        warnings.append(f"{name}: removed {n_degen} degenerate triangles")

    # Smooth via trimesh
    if smooth and len(tris):
        try:
            import trimesh  # type: ignore
            mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
            mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=5)
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            tris = np.asarray(mesh.faces, dtype=np.int32)
        except ImportError:
            warnings.append(f"{name}: trimesh not installed, skipping Laplacian smooth")
        except Exception as exc:
            warnings.append(f"{name}: smoothing failed ({exc})")

    # Resample onto regular grid if spacing provided
    if target_spacing is not None and len(tris):
        verts, tris, rs_warn = _resample_surface(verts, tris, target_spacing, name)
        warnings.extend(rs_warn)

    return verts, tris, n_degen, warnings


def _degenerate_mask(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Return bool mask of degenerate (zero-area) triangles."""
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return areas < 1e-10


def _estimate_fault_throw(verts: np.ndarray, tris: np.ndarray) -> float | None:
    """Estimate average fault throw as the z-range of the surface."""
    if len(verts) == 0:
        return None
    return float(verts[:, 2].max() - verts[:, 2].min())


def _max_throw_deviation(
    new_faults: list[StructuralFramework.FaultSurface],
    old_faults: list[StructuralFramework.FaultSurface],
) -> float:
    """Max absolute difference in estimated throw before/after conversion."""
    devs: list[float] = []
    for nf, of in zip(new_faults, old_faults):
        if nf.throw_avg_m is not None and of.throw_avg_m is not None:
            devs.append(abs(nf.throw_avg_m - of.throw_avg_m))
    return float(max(devs)) if devs else 0.0


def _resample_surface(
    verts: np.ndarray,
    tris: np.ndarray,
    spacing: float,
    name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Resample a surface onto a uniform XY grid at `spacing` metres."""
    warnings: list[str] = []
    try:
        from scipy.interpolate import RBFInterpolator  # type: ignore
    except ImportError:
        return verts, tris, [f"{name}: scipy not installed, skipping resample"]

    xs, ys, zs = verts[:, 0], verts[:, 1], verts[:, 2]
    x_grid = np.arange(xs.min(), xs.max(), spacing)
    y_grid = np.arange(ys.min(), ys.max(), spacing)
    xg, yg = np.meshgrid(x_grid, y_grid)
    xy_query = np.column_stack([xg.ravel(), yg.ravel()])

    try:
        rbf = RBFInterpolator(
            np.column_stack([xs, ys]), zs,
            kernel="thin_plate_spline",
            smoothing=1.0,
        )
        z_interp = rbf(xy_query)
    except Exception as exc:
        return verts, tris, [f"{name}: RBF interpolation failed ({exc})"]

    new_verts = np.column_stack([xy_query[:, 0], xy_query[:, 1], z_interp])

    # Delaunay triangulation on the regular grid
    try:
        from scipy.spatial import Delaunay  # type: ignore
        tri = Delaunay(xy_query)
        new_tris = tri.simplices.astype(np.int32)
    except Exception as exc:
        return verts, tris, [f"{name}: Delaunay failed after resample ({exc})"]

    return new_verts, new_tris, warnings


def read_fault_pts(path: str | Path) -> StructuralFramework.FaultSurface:
    """
    Read a Petrel fault point file (XYZ text, space-delimited) and
    triangulate via Delaunay 2.5D.
    """
    path = Path(path)
    pts = np.loadtxt(path, comments="--")
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    from scipy.spatial import Delaunay  # type: ignore
    tri = Delaunay(pts[:, :2])
    tris = tri.simplices.astype(np.int32)

    return StructuralFramework.FaultSurface(
        name=path.stem,
        vertices=pts[:, :3].astype(np.float64),
        triangles=tris,
    )


def write_irap_classic(
    hor: StructuralFramework.HorizonSurface, path: str | Path
) -> None:
    """
    Write a horizon surface in Irap Classic (RMS) format — readable by both
    Petrel and RMS.
    """
    path = Path(path)
    verts = hor.vertices

    xs, ys, zs = verts[:, 0], verts[:, 1], verts[:, 2]
    xinc = float((xs.max() - xs.min()) / max(len(np.unique(xs)) - 1, 1))
    yinc = float((ys.max() - ys.min()) / max(len(np.unique(ys)) - 1, 1))

    with open(path, "w") as fh:
        fh.write(
            f"-996 {len(zs)} {xinc:.4f} {yinc:.4f}\n"
            f"{xs.min():.4f} {xs.max():.4f} {ys.min():.4f} {ys.max():.4f}\n"
            f"{len(np.unique(xs))} {len(np.unique(ys))} {xs.min():.4f} {ys.min():.4f}\n"
            "0 0 0 0 0 0 0\n"
        )
        for z in zs:
            fh.write(f"{z:.4f}\n")
