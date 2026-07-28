"""
FastAPI web backend for the Petrel ↔ JewelSuite Bridge.

Endpoints
---------
GET  /                       — Serve the main UI
POST /api/convert            — Upload + convert a grid file
GET  /api/job/{job_id}       — Poll job status
GET  /api/download/{job_id}  — Download converted file
POST /api/validate           — Validate a grid file
POST /api/info               — Get grid metadata as JSON
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

# Add src to Python path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Petrel ↔ JewelSuite Bridge",
    description="Bi-directional 3D grid format converter",
    version="0.1.0",
)

_HERE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")
_INDEX_HTML = (_HERE / "templates" / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED    = "queued"
    RUNNING   = "running"
    DONE      = "done"
    ERROR     = "error"


@dataclass
class ConversionJob:
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress_pct: int = 0
    message: str = ""
    output_file: Path | None = None
    qc: dict = field(default_factory=dict)
    error: str = ""
    fault_data: list = field(default_factory=list)  # fault surfaces from source model


_jobs: dict[str, ConversionJob] = {}
_WORK_DIR = Path(tempfile.gettempdir()) / "petrel_jewel_bridge"
_WORK_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index():
    return HTMLResponse(content=_INDEX_HTML)


@app.post("/api/convert")
async def convert_file(
    file:          UploadFile = File(..., description="Input grid file"),
    direction:     str        = Form(..., description="petrel_to_jewel | jewel_to_petrel | tet_to_voxel"),
    output_format: str        = Form("resqml", description="resqml | grdecl | vtk"),
    resolution:    int        = Form(50,       description="Voxel resolution (tet_to_voxel only)"),
    project_name:  str        = Form("",       description="Optional project name"),
):
    """Upload a grid file and start an async conversion job."""
    job_id = str(uuid.uuid4())
    job = ConversionJob(job_id=job_id)
    _jobs[job_id] = job

    # Save upload
    upload_path = _WORK_DIR / f"{job_id}_{file.filename}"
    with open(upload_path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    # Launch conversion in background
    asyncio.create_task(
        _run_conversion(job, upload_path, direction, output_format, resolution, project_name)
    )

    return {"job_id": job_id, "status": JobStatus.QUEUED}


@app.get("/api/job/{job_id}")
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result = {
        "job_id": job_id,
        "status": job.status,
        "progress_pct": job.progress_pct,
        "message": job.message,
        "qc": job.qc,
        "error": job.error,
    }
    if job.output_file:
        result["download_url"] = f"/api/download/{job_id}"
        result["output_filename"] = job.output_file.name
    return result


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    job = _jobs.get(job_id)
    if not job or job.output_file is None:
        raise HTTPException(status_code=404, detail="Result not ready")
    return FileResponse(
        path=str(job.output_file),
        filename=job.output_file.name,
        media_type="application/octet-stream",
    )


@app.post("/api/geometry")
async def geometry_from_file(file: UploadFile = File(...)):
    """Return Three.js outer-surface geometry JSON for a dropped grid file."""
    tmp = _WORK_DIR / f"geo_{uuid.uuid4()}_{file.filename}"
    with open(tmp, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        model = _read_any(tmp)
        return JSONResponse(_model_geometry(model))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        tmp.unlink(missing_ok=True)


@app.get("/api/geometry/{job_id}")
async def geometry_from_job(job_id: str):
    """Return Three.js outer-surface geometry JSON for a completed conversion job."""
    job = _jobs.get(job_id)
    if not job or job.output_file is None or job.status.value != "done":
        raise HTTPException(status_code=404, detail="Job not ready or not found")
    try:
        from formats.grdecl_reader import read_grdecl
        model = read_grdecl(job.output_file)
        data = _model_geometry(model)
        # GRDECL has no framework section; restore fault surfaces from the source model
        if job.fault_data:
            data["faults"] = job.fault_data
        return JSONResponse(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/validate")
async def validate_file(file: UploadFile = File(...)):
    """Validate an uploaded grid file and return QC report."""
    tmp = _WORK_DIR / f"val_{uuid.uuid4()}_{file.filename}"
    with open(tmp, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        model = _read_any(tmp)
        model.project_name = Path(file.filename).stem
        report = _build_info_dict(model)
        report["valid"] = True
        report["warnings"] = []
        return JSONResponse(report)
    except Exception as exc:
        return JSONResponse({"valid": False, "error": str(exc)}, status_code=400)
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/api/info")
async def file_info(file: UploadFile = File(...)):
    """Return metadata JSON for an uploaded grid file."""
    tmp = _WORK_DIR / f"info_{uuid.uuid4()}_{file.filename}"
    with open(tmp, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        model = _read_any(tmp)
        # Use the original uploaded filename as the project name, not the temp path
        model.project_name = Path(file.filename).stem
        return JSONResponse(_build_info_dict(model))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Conversion worker
# ---------------------------------------------------------------------------

async def _run_conversion(
    job: ConversionJob,
    upload_path: Path,
    direction: str,
    output_format: str,
    resolution: int,
    project_name: str,
) -> None:
    job.status = JobStatus.RUNNING
    try:
        job.message = "Reading input file…"
        job.progress_pct = 10
        model = await asyncio.to_thread(_read_any, upload_path)

        # Stash fault surfaces so the converted-grid viewer can show them
        if model.framework:
            job.fault_data = [
                {
                    "name": f.name,
                    "vertices": f.vertices.flatten().tolist(),
                    "indices": f.triangles.flatten().tolist(),
                }
                for f in model.framework.faults
            ]

        job.message = "Converting…"
        job.progress_pct = 40

        out_path, qc = await asyncio.to_thread(
            _do_convert, model, direction, output_format, resolution, project_name, job
        )

        job.output_file = out_path
        job.qc = qc
        job.progress_pct = 100
        job.status = JobStatus.DONE
        job.message = "Conversion complete"

    except Exception as exc:
        job.status = JobStatus.ERROR
        job.error = f"{type(exc).__name__}: {exc}"
        job.message = "Conversion failed"
    finally:
        upload_path.unlink(missing_ok=True)


def _do_convert(model, direction, output_format, resolution, project_name, job) -> tuple[Path, dict]:
    from formats.grdecl_reader import write_grdecl

    if project_name:
        model.project_name = project_name

    qc_dict: dict[str, Any] = {}
    out_suffix = ".epc" if output_format == "resqml" else ".grdecl"
    out_path = _WORK_DIR / f"{job.job_id}_result{out_suffix}"

    if direction == "petrel_to_jewel":
        from converters.petrel_to_jewel.corner_point_to_jewel import corner_point_to_jewel
        from converters.petrel_to_jewel.property_mapper import map_properties

        job.message = "Mapping properties…"
        job.progress_pct = 50
        if model.corner_point:
            model.properties, _ = map_properties(
                model.properties,
                model.corner_point.dims,
                model.corner_point.actnum,
            )

        job.message = "Converting corner-point → JewelGrid…"
        job.progress_pct = 65
        jewel_model, qc = corner_point_to_jewel(model)
        qc_dict = {
            "total_cells": qc.n_cells_total,
            "active_cells": qc.n_cells_active,
            "degenerate_cells": qc.n_degenerate,
            "volume_loss_pct": round(qc.volume_loss_fraction * 100, 2),
            "max_aspect_ratio": round(qc.max_aspect_ratio, 1),
            "warnings": qc.warnings,
        }

        job.message = "Writing output…"
        job.progress_pct = 85
        if output_format == "resqml":
            from formats.resqml_io import write_resqml
            write_resqml(jewel_model, out_path)
        else:
            from converters.jewel_to_petrel.jewel_to_corner_point import jewel_to_corner_point
            cp_model, _ = jewel_to_corner_point(jewel_model)
            write_grdecl(cp_model, out_path)

    elif direction == "jewel_to_petrel":
        job.message = "Converting JewelSuite → corner-point…"
        job.progress_pct = 65

        # read_jewel already reconstructs a CornerPointGrid, so skip the
        # JewelGrid→CornerPoint converter when the grid is already corner-point.
        from core.grid_models import GridType as _GT
        if model.grid_type == _GT.CORNER_POINT and model.corner_point is not None:
            cp_model = model
            cp = model.corner_point
            qc_dict = {
                "pillars": (cp.dims.ni + 1) * (cp.dims.nj + 1),
                "max_tilt_deg": 0.0,
                "mean_dev_m": 0.0,
                "warnings": [],
            }
        else:
            from converters.jewel_to_petrel.jewel_to_corner_point import jewel_to_corner_point
            cp_model, qc = jewel_to_corner_point(model)
            qc_dict = {
                "pillars": qc.n_pillars,
                "max_tilt_deg": round(qc.max_pillar_tilt_deg, 1),
                "mean_dev_m": round(qc.mean_xy_deviation_m, 3),
                "warnings": qc.warnings,
            }

        job.message = "Writing GRDECL…"
        job.progress_pct = 85
        write_grdecl(cp_model, out_path)

    elif direction == "tet_to_voxel":
        from converters.jewel_to_petrel.tet_to_voxel import tet_to_voxel

        job.message = "Voxelizing tetrahedral mesh…"
        job.progress_pct = 60
        vox_model, qc = tet_to_voxel(model, resolution=resolution)
        qc_dict = {
            "tet_cells": qc.n_tet_cells,
            "voxels_total": qc.n_voxels_total,
            "voxels_active": qc.n_voxels_active,
            "warnings": qc.warnings,
        }
        job.message = "Writing GRDECL…"
        job.progress_pct = 85
        out_path = out_path.with_suffix(".grdecl")
        write_grdecl(vox_model, out_path)

    else:
        raise ValueError(f"Unknown conversion direction: {direction}")

    return out_path, qc_dict


def _read_any(path: Path):
    suffix = path.suffix.lower()
    # Strip job-id prefix from filename to get true suffix
    original_name = "_".join(path.stem.split("_")[1:]) + path.suffix
    true_suffix = Path(original_name).suffix.lower()

    from formats.grdecl_reader import read_grdecl
    from formats.egrid_reader import read_egrid
    from formats.resqml_io import read_resqml
    from formats.jewel_reader import read_jewel

    if true_suffix in (".grdecl", ".dat", ".txt"):
        return read_grdecl(path)
    elif true_suffix in (".egrid", ".grid"):
        return read_egrid(path)
    elif true_suffix in (".epc",):
        return read_resqml(path)
    elif true_suffix in (".jewel",):
        return read_jewel(path)
    else:
        # Fallback: try GRDECL
        return read_grdecl(path)


def _model_geometry(model) -> dict:
    """Extract Three.js-ready outer-surface triangles from a BridgeModel."""
    import numpy as np

    grid = model.active_grid()
    result: dict[str, Any] = {"type": model.grid_type.name.lower(), "faults": []}

    if grid is None or not hasattr(grid, "coord") or grid.coord is None:
        result.update({"vertices": [], "indices": [], "z_min": 0.0, "z_max": 0.0})
        return result

    verts, idxs = _cp_outer_surface(grid)
    result["vertices"] = verts
    result["indices"] = idxs
    result["z_min"] = float(min(verts[2::3])) if verts else 0.0
    result["z_max"] = float(max(verts[2::3])) if verts else 0.0

    # Per-cell geometry for interactive slice viewer
    cell_positions, cell_ijk, n_cells = _cp_cells_geometry(grid)
    result["cell_positions"] = cell_positions
    result["cell_ijk"]       = cell_ijk
    result["n_cells"]        = n_cells
    result["ni"] = int(grid.dims.ni)
    result["nj"] = int(grid.dims.nj)
    result["nk"] = int(grid.dims.nk)

    try:
        result["diagnostics"] = _grid_diagnostics(grid)
    except Exception:
        pass

    if model.framework:
        for fault in model.framework.faults:
            result["faults"].append({
                "name": fault.name,
                "vertices": fault.vertices.flatten().tolist(),
                "indices": fault.triangles.flatten().tolist(),
            })
    return result


def _cp_cells_geometry(grid) -> tuple[list, list, int]:
    """Return (flat_positions, flat_ijk, n_active_cells) for the cell-slice viewer.

    flat_positions: 8 corners × 3 floats per active cell.
    Corner vertex order (matches CELL_TRI_IDX / CELL_EDGE_IDX in viewer.js):
      v0=(0,0,0) v1=(1,0,0) v2=(0,1,0) v3=(1,1,0)
      v4=(0,0,1) v5=(1,0,1) v6=(0,1,1) v7=(1,1,1)
    flat_ijk: (ci, cj, ck) per active cell.
    """
    import numpy as np

    ni, nj, nk = grid.dims.ni, grid.dims.nj, grid.dims.nk
    coord = grid.coord
    zcorn = grid.zcorn
    act = grid.actnum.reshape(ni, nj, nk) if grid.actnum is not None else np.ones((ni, nj, nk), dtype=np.int32)

    positions: list[float] = []
    ijk_list:  list[int]   = []

    # loop order chosen so vertex index = dk*4 + dj*2 + di
    for ci in range(ni):
        for cj in range(nj):
            for ck in range(nk):
                if not act[ci, cj, ck]:
                    continue
                for dk in range(2):
                    for dj in range(2):
                        for di in range(2):
                            pi, pj = ci + di, cj + dj
                            positions.append(float(coord[pi, pj, 0]))
                            positions.append(float(coord[pi, pj, 1]))
                            positions.append(float(zcorn[2*ci+di, 2*cj+dj, 2*ck+dk]))
                ijk_list.extend([ci, cj, ck])

    return positions, ijk_list, len(ijk_list) // 3


def _cp_outer_surface(grid) -> tuple[list, list]:
    """Return (flat_vertices, flat_indices) for the outer shell of a CornerPointGrid."""
    import numpy as np

    ni, nj, nk = grid.dims.ni, grid.dims.nj, grid.dims.nk
    coord = grid.coord   # (ni+1, nj+1, 6)
    zcorn = grid.zcorn   # (2*ni, 2*nj, 2*nk)

    act = grid.actnum.reshape(ni, nj, nk) if grid.actnum is not None else np.ones((ni, nj, nk), dtype=np.int32)

    def cxyz(ci, cj, ck, di, dj, dk):
        pi, pj = ci + di, cj + dj
        return float(coord[pi, pj, 0]), float(coord[pi, pj, 1]), float(zcorn[2*ci+di, 2*cj+dj, 2*ck+dk])

    verts: list[float] = []
    idxs:  list[int]   = []

    def quad(a, b, c, d):
        base = len(verts) // 3
        for pt in (a, b, c, d):
            verts.extend(pt)
        idxs.extend([base, base+1, base+2, base, base+2, base+3])

    for ci in range(ni):
        for cj in range(nj):
            for ck in range(nk):
                if not act[ci, cj, ck]:
                    continue

                if ck == 0 or not act[ci, cj, ck-1]:
                    quad(cxyz(ci,cj,ck,0,0,0), cxyz(ci,cj,ck,1,0,0), cxyz(ci,cj,ck,1,1,0), cxyz(ci,cj,ck,0,1,0))

                if ck == nk-1 or not act[ci, cj, ck+1]:
                    quad(cxyz(ci,cj,ck,0,0,1), cxyz(ci,cj,ck,0,1,1), cxyz(ci,cj,ck,1,1,1), cxyz(ci,cj,ck,1,0,1))

                if ci == 0 or not act[ci-1, cj, ck]:
                    quad(cxyz(ci,cj,ck,0,0,0), cxyz(ci,cj,ck,0,1,0), cxyz(ci,cj,ck,0,1,1), cxyz(ci,cj,ck,0,0,1))

                if ci == ni-1 or not act[ci+1, cj, ck]:
                    quad(cxyz(ci,cj,ck,1,0,0), cxyz(ci,cj,ck,1,0,1), cxyz(ci,cj,ck,1,1,1), cxyz(ci,cj,ck,1,1,0))

                if cj == 0 or not act[ci, cj-1, ck]:
                    quad(cxyz(ci,cj,ck,0,0,0), cxyz(ci,cj,ck,0,0,1), cxyz(ci,cj,ck,1,0,1), cxyz(ci,cj,ck,1,0,0))

                if cj == nj-1 or not act[ci, cj+1, ck]:
                    quad(cxyz(ci,cj,ck,0,1,0), cxyz(ci,cj,ck,1,1,0), cxyz(ci,cj,ck,1,1,1), cxyz(ci,cj,ck,0,1,1))

    return verts, idxs


def _grid_diagnostics(grid) -> dict:
    """Return a diagnostic table describing a CornerPointGrid's orientation.

    Derives all values from the actual coord/zcorn geometry so it works for both
    JewelSuite-derived grids and imported GRDECL files.
    """
    coord = grid.coord   # (NI+1, NJ+1, 6)

    origin_x = float(coord[0, 0, 0])
    origin_y = float(coord[0, 0, 1])

    # I direction: vector from pillar (0,0) to (1,0)
    di_x = float(coord[1, 0, 0]) - float(coord[0, 0, 0])
    di_y = float(coord[1, 0, 1]) - float(coord[0, 0, 1])
    i_dir = "East (+X)" if di_x > 0 else "West (−X)"

    # J direction: vector from pillar (0,0) to (0,1)
    dj_x = float(coord[0, 1, 0]) - float(coord[0, 0, 0])
    dj_y = float(coord[0, 1, 1]) - float(coord[0, 0, 1])
    j_dir = "North (+Y)" if dj_y > 0 else "South (−Y)"

    # Corner that pillar (I=0, J=0) sits on
    if di_x > 0 and dj_y > 0:
        cell_origin = "SW corner (I=0, J=0)"
    elif di_x > 0 and dj_y < 0:
        cell_origin = "NW corner (I=0, J=0)"
    elif di_x < 0 and dj_y > 0:
        cell_origin = "SE corner (I=0, J=0)"
    else:
        cell_origin = "NE corner (I=0, J=0)"

    # Handedness: cross(I, J) z-component; negative → thumb points down = right-handed
    cross_z = di_x * dj_y - di_y * dj_x
    handedness = "Right-handed ✓" if cross_z < 0 else "Left-handed (J goes North; flipped to RH on GRDECL export)"

    x_min = float(coord[:, :, 0].min())
    x_max = float(coord[:, :, 0].max())
    y_min = float(coord[:, :, 1].min())
    y_max = float(coord[:, :, 1].max())
    z_min = float(grid.zcorn.min())
    z_max = float(grid.zcorn.max())

    return {
        "cell_origin": cell_origin,
        "origin_x":    round(origin_x, 2),
        "origin_y":    round(origin_y, 2),
        "i_direction": i_dir,
        "j_direction": j_dir,
        "x_range":     [round(x_min, 2), round(x_max, 2)],
        "y_range":     [round(y_min, 2), round(y_max, 2)],
        "z_range":     [round(z_min, 2), round(z_max, 2)],
        "handedness":  handedness,
    }


def _build_info_dict(model) -> dict:
    import numpy as np
    grid = model.active_grid()
    info: dict[str, Any] = {
        "grid_type": model.grid_type.name,
        "source_software": model.source_software,
        "project_name": model.project_name,
    }
    if grid and hasattr(grid, "dims") and grid.dims:
        d = grid.dims
        info["ni"] = d.ni
        info["nj"] = d.nj
        info["nk"] = d.nk
        info["total_cells"] = d.ncells
    if grid and hasattr(grid, "actnum") and grid.actnum is not None:
        info["active_cells"] = int(grid.actnum.sum())
    if grid and hasattr(grid, "coord") and grid.coord is not None:
        c = grid.coord
        info["x_range"] = [float(c[..., 0].min()), float(c[..., 0].max())]
        info["y_range"] = [float(c[..., 1].min()), float(c[..., 1].max())]
    if grid and hasattr(grid, "zcorn") and grid.zcorn is not None:
        z = grid.zcorn
        info["z_range"] = [float(z.min()), float(z.max())]
    info["n_properties"] = len(model.properties)
    info["properties"] = [
        {
            "name": p.name,
            "type": p.prop_type.value,
            "min": float(np.nanmin(p.values)),
            "max": float(np.nanmax(p.values)),
            "unit": p.unit,
            "timestep_days": p.timestep_days,
        }
        for p in model.properties
    ]
    if model.framework:
        fw = model.framework
        info["framework"] = {
            "faults":   [f.name for f in fw.faults],
            "horizons": [h.name for h in fw.horizons],
            "zones":    [z.name for z in fw.zones],
        }
    return info
