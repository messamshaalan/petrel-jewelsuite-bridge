/**
 * viewer.js — Three.js ES module for side-by-side 3D grid preview.
 *
 * Display modes:  surface  – outer shell only (fast)
 *                 cells    – every active cell as a solid hexahedron
 *
 * Slice player:  I-column and J-row sliders with auto-play.
 *
 * Exposed on window.viewerModule:
 *   loadOriginalGeometry(file)
 *   loadConvertedGeometry(jobId)
 *   clearAll()
 *   setDisplayMode(mode, btn)    'surface' | 'cells'
 *   setSlice(axis, idx)          axis 'i'|'j', idx -1=all
 *   playAxis(axis, btn)
 *   toggleSync(btn)
 *   toggleWireframe(btn)
 *   toggleFaults(btn)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Connectivity tables for one hexahedral cell (8 vertices v0..v7) ─────────
// Vertex layout:  v(dk*4 + dj*2 + di),  di/dj/dk ∈ {0,1}
//   v0=(0,0,0) v1=(1,0,0) v2=(0,1,0) v3=(1,1,0)
//   v4=(0,0,1) v5=(1,0,1) v6=(0,1,1) v7=(1,1,1)
const CELL_TRI_IDX = [
  0,2,3, 0,3,1,   // top    face (dk=0)
  4,5,7, 4,7,6,   // bottom face (dk=1)
  0,1,5, 0,5,4,   // front  face (dj=0)
  2,6,7, 2,7,3,   // back   face (dj=1)
  0,4,6, 0,6,2,   // left   face (di=0)
  1,3,7, 1,7,5,   // right  face (di=1)
];  // 36 indices per cell
const CELL_EDGE_IDX = [
  0,1, 1,3, 3,2, 2,0,   // top    ring
  4,5, 5,7, 7,6, 6,4,   // bottom ring
  0,4, 1,5, 2,6, 3,7,   // vertical pillars
];  // 24 indices per cell

// ── Viridis colormap ────────────────────────────────────────────────────────
const _VIR = [
  [0.267,0.005,0.329],[0.283,0.141,0.458],[0.254,0.265,0.530],
  [0.207,0.372,0.553],[0.164,0.471,0.558],[0.128,0.567,0.551],
  [0.135,0.659,0.518],[0.267,0.749,0.441],[0.478,0.821,0.318],
  [0.741,0.873,0.150],[0.993,0.906,0.144],
];
function viridisRGB(t) {
  const n = _VIR.length - 1;
  const i = Math.min(Math.floor(t * n), n - 1);
  const f = t * n - i;
  const a = _VIR[i], b = _VIR[i+1] || a;
  return [a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1]), a[2]+f*(b[2]-a[2])];
}

// ── Canvas-texture sprite helper ─────────────────────────────────────────────
function _makeTextSprite(text, cssColor = '#ffffff') {
  const canvas = document.createElement('canvas');
  canvas.width = 320; canvas.height = 72;
  const ctx = canvas.getContext('2d');
  ctx.font = 'bold 42px Arial';
  ctx.fillStyle = cssColor;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 6, 36);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  return sprite;
}

// ── K-layer (zone) colormap — categorical HSL rainbow ───────────────────────
function zoneRGB(k, nk) {
  const hue = nk <= 1 ? 0 : (k / (nk - 1)) * 280;   // 0–280° avoids red wrap
  // HSL to RGB
  const h = hue / 360, s = 0.80, l = 0.52;
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  function hue2rgb(t) {
    if (t < 0) t += 1; if (t > 1) t -= 1;
    if (t < 1/6) return p + (q - p) * 6 * t;
    if (t < 1/2) return q;
    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
    return p;
  }
  return [hue2rgb(h + 1/3), hue2rgb(h), hue2rgb(h - 1/3)];
}

// ── GridViewer ────────────────────────────────────────────────────────────────
class GridViewer {
  constructor(canvasId, emptyId) {
    this.canvas  = document.getElementById(canvasId);
    this.emptyEl = document.getElementById(emptyId);

    // Geometry state
    this.surfaceMesh  = null;   // outer-shell mesh
    this.wireMesh     = null;   // overlay wireframe
    this.cellMesh     = null;   // per-cell solid mesh
    this.cellEdgeMesh = null;   // per-cell edge mesh
    this.faultMeshes  = [];

    // Data
    this.cellData = null;  // { positions, ijk, n_cells, ni, nj, nk }
    this.zMin = 0; this.zMax = 1;
    this.colorMode = 'depth';   // 'depth' | 'zone'

    // UI state
    this.displayMode = 'surface';   // 'surface' | 'cells'
    this.sliceI = -1;
    this.sliceJ = -1;
    this._wireframe  = false;
    this._showFaults = true;
    this._showBox    = true;
    this.boxGroup    = null;

    // Three.js setup
    this.scene    = new THREE.Scene();
    this.scene.background = new THREE.Color(0x07091a);
    this.camera   = new THREE.PerspectiveCamera(45, 1, 0.1, 1e8);
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const amb = new THREE.AmbientLight(0xffffff, 0.55);
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(1, 2, 1.5);
    this.scene.add(amb, dir);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    this._animate = () => {
      requestAnimationFrame(this._animate);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    this._animate();

    new ResizeObserver(() => this._resize()).observe(this.canvas.parentElement);
    this._resize();
  }

  _resize() {
    const el = this.canvas.parentElement;
    const w = el.clientWidth || 400, h = el.clientHeight || 360;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // ── clear ────────────────────────────────────────────────────────────────
  clear() {
    const meshes = [this.surfaceMesh, this.wireMesh, this.cellMesh, this.cellEdgeMesh, ...this.faultMeshes];
    meshes.forEach(m => {
      if (!m) return;
      this.scene.remove(m);
      m.geometry.dispose();
      (Array.isArray(m.material) ? m.material : [m.material]).forEach(x => x.dispose());
    });
    this.surfaceMesh = this.wireMesh = this.cellMesh = this.cellEdgeMesh = null;
    this.faultMeshes = [];
    this.cellData = null;
    if (this.boxGroup) {
      this.scene.remove(this.boxGroup);
      this.boxGroup.traverse(obj => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (obj.material.map) obj.material.map.dispose();
          obj.material.dispose();
        }
      });
      this.boxGroup = null;
    }
    if (this.emptyEl) this.emptyEl.style.display = 'flex';
  }

  // ── load ─────────────────────────────────────────────────────────────────
  load(data) {
    this.clear();
    if (!data.vertices?.length) return;

    this.zMin = data.z_min ?? 0;
    this.zMax = data.z_max ?? 1;

    // --- surface mesh ---
    const posArr = new Float32Array(data.vertices);
    const idxArr = new Uint32Array(data.indices);
    const colArr = this._makeColors(posArr);

    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    sg.setAttribute('color',    new THREE.BufferAttribute(colArr, 3));
    sg.setIndex(new THREE.BufferAttribute(idxArr, 1));
    sg.computeVertexNormals();

    this.surfaceMesh = new THREE.Mesh(sg,
      new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
    // Remap Geo(X,Y,Z_depth) → Three.js(X, -Z_depth, Y): depth becomes vertical (-Y = deep)
    this.surfaceMesh.rotation.x = Math.PI / 2;
    this.scene.add(this.surfaceMesh);

    // wireframe overlay
    const wg = sg.clone();
    this.wireMesh = new THREE.Mesh(wg,
      new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, opacity: 0.08, transparent: true }));
    this.wireMesh.rotation.x = Math.PI / 2;
    this.wireMesh.visible = this._wireframe;
    this.scene.add(this.wireMesh);

    // --- store cell data ---
    if (data.n_cells > 0) {
      this.cellData = {
        positions: data.cell_positions,
        ijk:       data.cell_ijk,
        n_cells:   data.n_cells,
        ni: data.ni, nj: data.nj, nk: data.nk,
      };
    }
    this._nk = data.nk || 1;

    // --- fault surfaces ---
    const faultColors = [0xFF6B35, 0xA855F7, 0xF59E0B, 0x10B981, 0xEC4899];
    (data.faults || []).forEach((f, idx) => {
      if (!f.vertices?.length) return;
      const fg = new THREE.BufferGeometry();
      fg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(f.vertices), 3));
      fg.setIndex(new THREE.BufferAttribute(new Uint32Array(f.indices), 1));
      fg.computeVertexNormals();
      const fm = new THREE.Mesh(fg, new THREE.MeshBasicMaterial({
        color: faultColors[idx % faultColors.length], side: THREE.DoubleSide,
        transparent: true, opacity: 0.72,
      }));
      fm.rotation.x = Math.PI / 2;
      fm.visible = this._showFaults;
      this.faultMeshes.push(fm);
      this.scene.add(fm);
    });

    // auto-fit camera — after Rx(π/2) world coords are (geoX, -geoZ_depth, geoY)
    // so higher world-Y = shallower; camera above and to the south gives geological oblique view
    const box = new THREE.Box3().setFromObject(this.surfaceMesh);
    const center = box.getCenter(new THREE.Vector3());
    const size   = box.getSize(new THREE.Vector3());
    const dist   = Math.max(size.x, size.y, size.z) * 1.6;
    this.camera.position.set(center.x + dist * 0.6, center.y + dist * 0.7, center.z + dist * 1.1);
    this.controls.target.copy(center);
    this.camera.up.set(0, 1, 0);
    this.controls.update();

    if (this.emptyEl) this.emptyEl.style.display = 'none';

    // apply current display mode
    if (this.displayMode === 'cells') this._rebuildCellMesh();

    // bounding box with geo-axis labels
    const d = data.diagnostics;
    if (d) {
      this._buildBoundingBox(
        d.x_range[0], d.x_range[1],
        d.y_range[0], d.y_range[1],
        d.z_range[0], d.z_range[1],
      );
    }

    return { center, dist };
  }

  // ── display mode ─────────────────────────────────────────────────────────
  setDisplayMode(mode) {
    this.displayMode = mode;
    const inCells = mode === 'cells';

    if (this.surfaceMesh) this.surfaceMesh.visible = !inCells;
    if (this.wireMesh)    this.wireMesh.visible    = !inCells && this._wireframe;

    if (inCells) {
      this._rebuildCellMesh();
    } else {
      if (this.cellMesh)     this.cellMesh.visible     = false;
      if (this.cellEdgeMesh) this.cellEdgeMesh.visible = false;
    }
  }

  // ── slice ─────────────────────────────────────────────────────────────────
  setSlice(axis, idx) {
    if (axis === 'i') this.sliceI = idx;
    else              this.sliceJ = idx;
    if (this.displayMode === 'cells') this._rebuildCellMesh();
  }

  // ── rebuild cell geometry ─────────────────────────────────────────────────
  _rebuildCellMesh() {
    // remove old
    [this.cellMesh, this.cellEdgeMesh].forEach(m => {
      if (!m) return;
      this.scene.remove(m);
      m.geometry.dispose();
      (Array.isArray(m.material) ? m.material : [m.material]).forEach(x => x.dispose());
    });
    this.cellMesh = this.cellEdgeMesh = null;

    if (!this.cellData) return;

    const { positions, ijk, n_cells, nk } = this.cellData;
    const fi = this.sliceI, fj = this.sliceJ;
    const zRng = (this.zMax - this.zMin) || 1;
    const useZone = this.colorMode === 'zone';

    const verts  = [];
    const colors = [];
    const tris   = [];
    const edges  = [];
    let nAdded   = 0;

    for (let c = 0; c < n_cells; c++) {
      const ci = ijk[c*3], cj = ijk[c*3+1], ck = ijk[c*3+2];
      if (fi >= 0 && ci !== fi) continue;
      if (fj >= 0 && cj !== fj) continue;

      const vBase    = c * 24;  // 8 verts × 3 floats
      const newBase  = nAdded * 8;

      for (let v = 0; v < 24; v++) verts.push(positions[vBase + v]);

      // All 8 corners of this cell get the same color
      let cellColor;
      if (useZone) {
        cellColor = zoneRGB(ck, nk);
      } else {
        const z = positions[vBase + 2];  // use first vertex Z as representative
        const t = Math.max(0, Math.min(1, (z - this.zMin) / zRng));
        cellColor = viridisRGB(t);
      }
      for (let v = 0; v < 8; v++) colors.push(...cellColor);

      for (const idx of CELL_TRI_IDX)  tris.push(newBase + idx);
      for (const idx of CELL_EDGE_IDX) edges.push(newBase + idx);
      nAdded++;
    }

    if (!nAdded) return;

    const posF = new Float32Array(verts);
    const colF = new Float32Array(colors);

    // solid faces
    const cg = new THREE.BufferGeometry();
    cg.setAttribute('position', new THREE.BufferAttribute(posF, 3));
    cg.setAttribute('color',    new THREE.BufferAttribute(colF, 3));
    cg.setIndex(tris);
    cg.computeVertexNormals();

    this.cellMesh = new THREE.Mesh(cg,
      new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
    this.cellMesh.rotation.x = Math.PI / 2;
    this.scene.add(this.cellMesh);

    // cell edges
    const eg = new THREE.BufferGeometry();
    eg.setAttribute('position', new THREE.BufferAttribute(posF.slice(), 3));
    eg.setIndex(new THREE.BufferAttribute(new Uint32Array(edges), 1));
    this.cellEdgeMesh = new THREE.LineSegments(eg,
      new THREE.LineBasicMaterial({ color: 0x000000, opacity: 0.35, transparent: true }));
    this.cellEdgeMesh.rotation.x = Math.PI / 2;
    this.scene.add(this.cellEdgeMesh);
  }

  // ── helpers ───────────────────────────────────────────────────────────────
  _makeColors(posArr) {
    const n = posArr.length / 3;
    const col = new Float32Array(posArr.length);
    const zRng = (this.zMax - this.zMin) || 1;
    const nk = this._nk || 1;
    for (let i = 0; i < n; i++) {
      let r, g, b;
      if (this.colorMode === 'zone') {
        // estimate K index from Z value (linear mapping)
        const t = Math.max(0, Math.min(1, (posArr[i*3+2] - this.zMin) / zRng));
        const k = Math.round(t * (nk - 1));
        [r, g, b] = zoneRGB(k, nk);
      } else {
        const t = Math.max(0, Math.min(1, (posArr[i*3+2] - this.zMin) / zRng));
        [r, g, b] = viridisRGB(t);
      }
      col[i*3] = r; col[i*3+1] = g; col[i*3+2] = b;
    }
    return col;
  }

  // ── bounding box + geo-axis labels ───────────────────────────────────────
  _buildBoundingBox(xMin, xMax, yMin, yMax, zMin, zMax) {
    // After mesh Rx(π/2): world_x=geoX(East), world_y=-geoZ(depth), world_z=geoY(North)
    const wx0 = xMin, wx1 = xMax;
    const wy0 = -zMax, wy1 = -zMin;   // wy0 = deepest (bottom),  wy1 = shallowest (top)
    const wz0 = yMin,  wz1 = yMax;    // wz0 = south,              wz1 = north

    const group = new THREE.Group();

    // 12-edge wireframe box
    const v = [wx0,wy0,wz0, wx1,wy0,wz0, wx1,wy0,wz1, wx0,wy0,wz1,
               wx0,wy1,wz0, wx1,wy1,wz0, wx1,wy1,wz1, wx0,wy1,wz1];
    const ei = [0,1, 1,2, 2,3, 3,0,   4,5, 5,6, 6,7, 7,4,   0,4, 1,5, 2,6, 3,7];
    const pts = new Float32Array(ei.length * 3);
    ei.forEach((vi, ii) => { pts[ii*3]=v[vi*3]; pts[ii*3+1]=v[vi*3+1]; pts[ii*3+2]=v[vi*3+2]; });
    const bGeo = new THREE.BufferGeometry();
    bGeo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
    group.add(new THREE.LineSegments(bGeo, new THREE.LineBasicMaterial({
      color: 0x5588aa, opacity: 0.55, transparent: true
    })));

    // Axes from the NW-shallow-south corner (wx0, wy1, wz0) = west, shallowest, south.
    // This is the top-left-back corner from the NE-above camera view.
    // X →  East  (+world X),  Y →  North (+world Z),  Z ↓  Depth (+depth = -world Y)
    const axLen = Math.max(wx1-wx0, wy1-wy0, wz1-wz0) * 0.20;
    const cx = wx0, cy = wy1, cz = wz0;   // top-left-back corner

    const addArrow = (origin, dir, color, label) => {
      const [ox, oy, oz] = origin;
      const [dx, dy, dz] = dir;
      const arrowDir = new THREE.Vector3(dx, dy, dz).normalize();
      const tipX = ox + dx*axLen, tipY = oy + dy*axLen, tipZ = oz + dz*axLen;
      group.add(new THREE.ArrowHelper(arrowDir, new THREE.Vector3(ox,oy,oz), axLen,
                                      color, axLen*0.18, axLen*0.09));
      const sprite = _makeTextSprite(label, `#${color.toString(16).padStart(6,'0')}`);
      sprite.position.set(tipX + dx*axLen*0.18, tipY + dy*axLen*0.18, tipZ + dz*axLen*0.18);
      sprite.scale.set(axLen * 1.9, axLen * 0.54, 1);
      group.add(sprite);
    };

    // X = East:  world +X from the top-left corner
    addArrow([cx, cy, cz], [1, 0, 0], 0xff5555, 'X  East →');
    // Y = North: world +Z from the top-left corner
    addArrow([cx, cy, cz], [0, 0, 1], 0x55dd55, 'Y  North →');
    // Z = Depth: world -Y from the top (shallow) corner → arrow points down = depth increasing ↓
    addArrow([cx, cy, cz], [0, -1, 0], 0x55aaff, 'Z  Depth ↓');

    this.boxGroup = group;
    this.boxGroup.visible = this._showBox;
    this.scene.add(group);
  }

  setBoxVisible(on) {
    this._showBox = on;
    if (this.boxGroup) this.boxGroup.visible = on;
  }

  setColorMode(mode) {
    this.colorMode = mode;
    // Rebuild surface colors
    if (this.surfaceMesh) {
      const posArr = this.surfaceMesh.geometry.attributes.position.array;
      const colArr = this._makeColors(posArr);
      this.surfaceMesh.geometry.attributes.color.array.set(colArr);
      this.surfaceMesh.geometry.attributes.color.needsUpdate = true;
    }
    // Rebuild cell colors
    if (this.displayMode === 'cells') this._rebuildCellMesh();
  }

  setWireframe(on) {
    this._wireframe = on;
    if (this.wireMesh && this.displayMode === 'surface') this.wireMesh.visible = on;
  }

  setFaultsVisible(on) {
    this._showFaults = on;
    this.faultMeshes.forEach(m => { m.visible = on; });
  }

  getCameraState() {
    return { pos: this.camera.position.clone(), target: this.controls.target.clone(), quat: this.camera.quaternion.clone() };
  }

  setCameraState(s) {
    this.camera.position.copy(s.pos);
    this.camera.quaternion.copy(s.quat);
    this.controls.target.copy(s.target);
    this.controls.update();
  }
}

// ── Module state ─────────────────────────────────────────────────────────────
let left  = null;
let right = null;
let syncEnabled  = false;
let syncing      = false;
const _playTimers = { i: null, j: null };
const _playIdx    = { i: 0,    j: 0    };

// ── Diagnostics helper ────────────────────────────────────────────────────────
function _renderDiagnostics(data, panelId, tableId) {
  const d = data?.diagnostics;
  const panel = document.getElementById(panelId);
  const table = document.getElementById(tableId);
  if (!panel || !table || !d) return;

  const rows = [
    ['Cell origin (I=0, J=0)', d.cell_origin],
    ['Origin X', d.origin_x + ' m'],
    ['Origin Y', d.origin_y + ' m'],
    ['X range (East)', `${d.x_range?.[0]} m  →  ${d.x_range?.[1]} m`],
    ['Y range (North)', `${d.y_range?.[0]} m  →  ${d.y_range?.[1]} m`],
    ['Z range (Depth +down)', `${d.z_range?.[0]} m  →  ${d.z_range?.[1]} m`],
    ['I direction', d.i_direction],
    ['J direction', d.j_direction],
    ['Coordinate system', d.handedness],
  ];

  table.innerHTML = rows.map(([k, v]) =>
    `<tr><td class="diag-key">${k}</td><td class="diag-val">${v ?? '–'}</td></tr>`
  ).join('');
  panel.style.display = 'block';
}

function _ensureViewers() {
  if (!left)  left  = new GridViewer('canvas-original',  'viewer-original-empty');
  if (!right) right = new GridViewer('canvas-converted', 'viewer-converted-empty');

  left.controls.addEventListener('change', () => {
    if (syncEnabled && !syncing && right) { syncing = true; right.setCameraState(left.getCameraState()); syncing = false; }
  });
  right.controls.addEventListener('change', () => {
    if (syncEnabled && !syncing && left) { syncing = true; left.setCameraState(right.getCameraState()); syncing = false; }
  });
}

async function _fetchJSON(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

function _showSection() {
  const s = document.getElementById('viewerSection');
  if (s) s.style.display = 'block';
}

function _updateLegend(viewer) {
  const minEl = document.getElementById('legend-min');
  const maxEl = document.getElementById('legend-max');
  if (minEl) minEl.textContent = viewer.zMin.toFixed(0) + ' m';
  if (maxEl) maxEl.textContent = viewer.zMax.toFixed(0) + ' m';
}

function _updateSliderRanges(viewer) {
  if (!viewer.cellData) return;
  const { ni, nj } = viewer.cellData;
  ['I','J'].forEach(ax => {
    const el = document.getElementById(`slice${ax}`);
    if (!el) return;
    el.max  = (ax === 'I' ? ni : nj) - 1;
    el.min  = -1;
    el.value = -1;
    const lbl = document.getElementById(`slice${ax}Val`);
    if (lbl) lbl.textContent = 'All';
  });
}

// ── Public API ────────────────────────────────────────────────────────────────
async function loadOriginalGeometry(file) {
  _showSection();
  _ensureViewers();
  try {
    const fd = new FormData();
    fd.append('file', file);
    const data = await _fetchJSON('/api/geometry', { method: 'POST', body: fd });
    left.load(data);
    _updateLegend(left);
    _updateSliderRanges(left);
    _renderDiagnostics(data, 'diag-original', 'diag-table-original');
  } catch (e) { console.warn('[viewer] loadOriginalGeometry:', e); }
}

async function loadConvertedGeometry(jobId) {
  _showSection();
  _ensureViewers();
  try {
    const data = await _fetchJSON(`/api/geometry/${jobId}`);
    right.load(data);
    _updateLegend(right);
    _updateSliderRanges(right);
    _renderDiagnostics(data, 'diag-converted', 'diag-table-converted');
  } catch (e) { console.warn('[viewer] loadConvertedGeometry:', e); }
}

function clearAll() {
  if (left)  left.clear();
  if (right) right.clear();
  const s = document.getElementById('viewerSection');
  if (s) s.style.display = 'none';
}

function setDisplayMode(mode, btn) {
  // update buttons
  document.querySelectorAll('.viewer-mode-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');

  const sliders = document.getElementById('viewerSliders');
  if (sliders) sliders.style.display = mode === 'cells' ? 'flex' : 'none';

  if (left)  left.setDisplayMode(mode);
  if (right) right.setDisplayMode(mode);
}

function setSlice(axis, idx) {
  const lbl = document.getElementById(`slice${axis.toUpperCase()}Val`);
  if (lbl) lbl.textContent = idx < 0 ? 'All' : (axis === 'i' ? 'Col ' : 'Row ') + idx;
  if (left)  left.setSlice(axis, idx);
  if (right) right.setSlice(axis, idx);
}

function playAxis(axis, btn) {
  // toggle
  if (_playTimers[axis]) {
    clearInterval(_playTimers[axis]);
    _playTimers[axis] = null;
    if (btn) { btn.textContent = '▶'; btn.classList.remove('playing'); }
    return;
  }

  const viewer = (left?.cellData ? left : right);
  if (!viewer?.cellData) return;

  const maxIdx = axis === 'i' ? viewer.cellData.ni - 1 : viewer.cellData.nj - 1;
  _playIdx[axis] = 0;
  if (btn) { btn.textContent = '⏸'; btn.classList.add('playing'); }

  _playTimers[axis] = setInterval(() => {
    const idx = _playIdx[axis];
    const sliderEl = document.getElementById(`slice${axis.toUpperCase()}`);
    if (sliderEl) sliderEl.value = idx;
    setSlice(axis, idx);

    _playIdx[axis]++;
    if (_playIdx[axis] > maxIdx) {
      clearInterval(_playTimers[axis]);
      _playTimers[axis] = null;
      if (btn) { btn.textContent = '▶'; btn.classList.remove('playing'); }
      // reset to all
      setTimeout(() => {
        if (sliderEl) sliderEl.value = -1;
        setSlice(axis, -1);
      }, 600);
    }
  }, 450);
}

function toggleSync(btn) {
  syncEnabled = !syncEnabled;
  if (btn) btn.classList.toggle('active', syncEnabled);
}

function toggleWireframe(btn) {
  const on = btn ? btn.classList.toggle('active') : false;
  if (left)  left.setWireframe(on);
  if (right) right.setWireframe(on);
}

function toggleFaults(btn) {
  if (!btn) return;
  const on = btn.classList.toggle('active');
  if (left)  left.setFaultsVisible(on);
  if (right) right.setFaultsVisible(on);
}

function setColorMode(mode) {
  const legend = document.getElementById('depthLegend');
  if (legend) legend.style.opacity = mode === 'zone' ? '0.4' : '1';
  if (left)  left.setColorMode(mode);
  if (right) right.setColorMode(mode);
}

function toggleBox(btn) {
  if (!btn) return;
  const on = btn.classList.toggle('active');
  if (left)  left.setBoxVisible(on);
  if (right) right.setBoxVisible(on);
}

window.viewerModule = {
  loadOriginalGeometry, loadConvertedGeometry, clearAll,
  setDisplayMode, setSlice, playAxis,
  toggleSync, toggleWireframe, toggleFaults, setColorMode, toggleBox,
};
