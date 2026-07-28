/* ===================================================================
   Petrel ↔ JewelSuite Bridge — Frontend Logic
   =================================================================== */

let selectedFile = null;
let currentDirection = 'petrel_to_jewel';
let pollTimer = null;

// ─────────────────────────────────────────────
// DIRECTION
// ─────────────────────────────────────────────
function setDirection(btn) {
  document.querySelectorAll('.dir-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentDirection = btn.dataset.dir;

  const resGroup = document.getElementById('resGroup');
  const simGroup = document.getElementById('simGroup');
  const outFormat = document.getElementById('outFormat');

  if (currentDirection === 'tet_to_voxel') {
    resGroup.style.display = 'block';
    simGroup.style.display = 'none';
    // Default output to GRDECL for voxel
    outFormat.value = 'grdecl';
  } else if (currentDirection === 'jewel_to_petrel') {
    resGroup.style.display = 'none';
    simGroup.style.display = 'none';
    outFormat.value = 'grdecl';
  } else {
    resGroup.style.display = 'none';
    simGroup.style.display = 'block';
    outFormat.value = 'resqml';
  }

  // Update arrow animation direction visual cue
  const arrows = document.getElementById('bridgeArrows');
  arrows.classList.remove('fwd', 'rev');
  arrows.classList.add(currentDirection === 'jewel_to_petrel' ? 'rev' : 'fwd');
}

// ─────────────────────────────────────────────
// FILE SELECTION
// ─────────────────────────────────────────────
function handleDragOver(evt) {
  evt.preventDefault();
  document.getElementById('dropZone').classList.add('drag-over');
}
function handleDragLeave(evt) {
  document.getElementById('dropZone').classList.remove('drag-over');
}
function handleDrop(evt) {
  evt.preventDefault();
  document.getElementById('dropZone').classList.remove('drag-over');
  const file = evt.dataTransfer.files[0];
  if (file) acceptFile(file);
}
function handleFileSelect(input) {
  if (input.files[0]) acceptFile(input.files[0]);
}

function acceptFile(file) {
  selectedFile = file;
  document.getElementById('dropInner').style.display = 'none';
  const sel = document.getElementById('fileSelected');
  sel.style.display = 'flex';
  document.getElementById('selectedFileName').textContent = file.name;
  document.getElementById('selectedFileSize').textContent = formatBytes(file.size);
  document.getElementById('convertBtn').disabled = false;

  // Auto-fetch file info + 3D preview
  fetchFileInfo(file);
  window.viewerModule?.loadOriginalGeometry(file);
}

function clearFile() {
  selectedFile = null;
  document.getElementById('dropInner').style.display = 'block';
  document.getElementById('fileSelected').style.display = 'none';
  document.getElementById('convertBtn').disabled = true;
  resetInfoPanel();
  window.viewerModule?.clearAll();
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

// ─────────────────────────────────────────────
// FILE INFO PANEL
// ─────────────────────────────────────────────
async function fetchFileInfo(file) {
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/api/info', { method: 'POST', body: fd });
    if (!res.ok) return;
    const info = await res.json();
    renderInfoPanel(info);
  } catch (_) {}
}

function renderInfoPanel(info) {
  document.getElementById('infoPlaceholder').style.display = 'none';
  document.getElementById('infoContent').style.display = 'block';

  const rows = [
    ['Grid type',    info.grid_type    || '–'],
    ['Source',       info.source_software || '–'],
    ['NI × NJ × NK', info.ni ? `${info.ni} × ${info.nj} × ${info.nk}` : '–'],
    ['Total cells',  info.total_cells  ? info.total_cells.toLocaleString() : '–'],
    ['Active cells', info.active_cells ? info.active_cells.toLocaleString() : '–'],
    ['Properties',   info.n_properties ?? '–'],
  ];
  if (info.z_range) rows.push(['Depth range', `${info.z_range[0].toFixed(0)} – ${info.z_range[1].toFixed(0)} m`]);

  const table = document.getElementById('infoTable');
  table.innerHTML = rows.map(([k, v]) =>
    `<tr><td>${k}</td><td>${v}</td></tr>`
  ).join('');

  // Properties
  const ps = document.getElementById('propsSection');
  if (info.properties && info.properties.length) {
    ps.innerHTML = `<div class="info-props-title">Properties</div>` +
      info.properties.map(p =>
        `<span class="prop-chip" title="${p.min?.toFixed(3)} – ${p.max?.toFixed(3)} ${p.unit || ''}">${p.name}</span>`
      ).join('');
  } else {
    ps.innerHTML = '';
  }

  // Framework
  const fw = document.getElementById('fwSection');
  if (info.framework) {
    const f = info.framework;
    const parts = [];
    if (f.faults.length)   parts.push(`${f.faults.length} fault(s)`);
    if (f.horizons.length) parts.push(`${f.horizons.length} horizon(s)`);
    if (f.zones.length)    parts.push(`${f.zones.length} zone(s)`);
    fw.innerHTML = parts.length
      ? `<div class="info-props-title">Framework</div><div style="font-size:0.78rem;color:var(--text-dim)">${parts.join('  ·  ')}</div>`
      : '';
  } else {
    fw.innerHTML = '';
  }
}

function resetInfoPanel() {
  document.getElementById('infoPlaceholder').style.display = 'block';
  document.getElementById('infoContent').style.display = 'none';
}

// ─────────────────────────────────────────────
// CONVERSION
// ─────────────────────────────────────────────
async function startConversion() {
  if (!selectedFile) return;

  // Hide old panels
  hidePanel('resultPanel');
  hidePanel('errorPanel');
  showPanel('progressPanel');
  document.getElementById('convertBtn').disabled = true;

  const fd = new FormData();
  fd.append('file', selectedFile);
  fd.append('direction', currentDirection);
  fd.append('output_format', document.getElementById('outFormat').value);
  fd.append('resolution', document.getElementById('resolution').value);
  fd.append('project_name', document.getElementById('projName').value);

  let jobId;
  try {
    const res = await fetch('/api/convert', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json();
      showError(err.detail || 'Upload failed');
      return;
    }
    const data = await res.json();
    jobId = data.job_id;
  } catch (err) {
    showError('Network error: ' + err.message);
    return;
  }

  pollJob(jobId);
}

function pollJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/job/${jobId}`);
      const job = await res.json();
      updateProgress(job);
      if (job.status === 'done' || job.status === 'error') {
        clearInterval(pollTimer);
        if (job.status === 'done') showResult(job);
        else showError(job.error || 'Conversion failed');
      }
    } catch (_) {}
  }, 800);
}

function updateProgress(job) {
  const pct = job.progress_pct || 0;
  document.getElementById('progressPct').textContent = pct + '%';
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressMsg').textContent = job.message || '';
  document.getElementById('progressTitle').textContent =
    job.status === 'running' ? 'Converting…' :
    job.status === 'queued'  ? 'Queued…' : 'Processing…';
}

function showResult(job) {
  hidePanel('progressPanel');
  showPanel('resultPanel');

  // QC summary
  const qc = job.qc || {};
  const qcEl = document.getElementById('qcSummary');
  if (Object.keys(qc).length) {
    const rows = Object.entries(qc)
      .filter(([k]) => k !== 'warnings')
      .map(([k, v]) => `<div class="qc-row"><span class="qc-key">${formatKey(k)}</span><span class="qc-val">${v}</span></div>`)
      .join('');
    const warns = (qc.warnings || []).map(w =>
      `<div style="font-size:0.72rem;color:#FBBF24;margin-top:4px">⚠ ${w}</div>`
    ).join('');
    qcEl.innerHTML = rows + warns;
  } else {
    qcEl.innerHTML = '<div style="color:var(--text-dim);font-size:0.82rem">No QC data available</div>';
  }

  // Load converted geometry into 3D viewer
  if (job.job_id) window.viewerModule?.loadConvertedGeometry(job.job_id);

  // Download link
  if (job.download_url) {
    const dl = document.getElementById('downloadBtn');
    dl.href = job.download_url;
    dl.download = job.output_filename || 'result';
  }
}

function showError(msg) {
  hidePanel('progressPanel');
  showPanel('errorPanel');
  document.getElementById('errorMsg').textContent = msg;
  document.getElementById('convertBtn').disabled = false;
}

function resetUI() {
  hidePanel('progressPanel');
  hidePanel('resultPanel');
  hidePanel('errorPanel');
  clearFile();
  document.getElementById('convertBtn').disabled = true;
}

function showPanel(id) { document.getElementById(id).style.display = 'block'; }
function hidePanel(id) { document.getElementById(id).style.display = 'none'; }

function formatKey(k) {
  return k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ─────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────
document.getElementById('initFile').addEventListener('change', e => {
  const f = e.target.files[0];
  document.getElementById('initName').textContent = f ? f.name : 'no file';
});
document.getElementById('unrstFile').addEventListener('change', e => {
  const f = e.target.files[0];
  document.getElementById('unrstName').textContent = f ? f.name : 'no file';
});

// Logo hover 3D tilt effect
['logo-petrel', 'logo-jewel'].forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener('mousemove', e => {
    const rect = el.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top  + rect.height / 2;
    const rx = ((e.clientY - cy) / (rect.height / 2)) * 12;
    const ry = ((e.clientX - cx) / (rect.width  / 2)) * -12;
    el.style.transform = `perspective(300px) rotateX(${rx}deg) rotateY(${ry}deg) scale(1.08)`;
  });
  el.addEventListener('mouseleave', () => {
    el.style.transform = '';
  });
});
