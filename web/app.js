/* NET-INSPECT // ROV CONSOLE — client logic */
const $ = (id) => document.getElementById(id);
const METHOD_META = {
  classical: { label: "CLASSICAL", sub: "OpenCV heuristic",
    info: "Explainable OpenCV baseline (darkness + low-edge-density + texture gate). No training; a fast difficulty probe. Tops out ~F1 0.50 — fires on shadows/fouling." },
  anomaly:   { label: "ANOMALY",   sub: "Mahalanobis",
    info: "Hand-crafted patch-Mahalanobis 'normal-net' model. Label-free but a weak localiser (F1 0.12); flags deviation, not confirmed damage." },
  patchcore: { label: "PATCHCORE", sub: "foundation",
    info: "PatchCore on pretrained-CNN features — label-free, F1 ~0.78. Strong image-level screen (AUROC ~1.0); see the DINOv2-vs-ResNet ablation in the report." },
  yolo:      { label: "YOLO",      sub: "supervised",
    info: "YOLOv8 detector (det v1) trained on synthetic-damage-on-real frames. The most robust model: 1% false alarms on a different day, F1 ≈ 0.97." },
  ensemble:  { label: "ENSEMBLE",  sub: "det∧seg agree",
    info: "det v1 proposes, the segmenter confirms (box agreement). Keeps the detector's 1% different-day false-alarm rate AND its recall, while adding masks — no retraining." },
};
const METHOD_ORDER = ["classical", "anomaly", "patchcore", "yolo", "ensemble"];

const state = { sources: [], methods: [], method: "yolo", source: null,
                frames: [], idx: 0, conf: 0.25, busy: false,
                sourceInfo: {}, ood: false, oodAvailable: false };

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function setStatus(text, cls) {
  $("statusText").textContent = text;
  $("statusDot").className = "dot " + (cls || "");
}

async function boot() {
  try {
    const h = await api("/api/health");
    state.methods = h.methods; state.sources = h.sources;
    state.sourceInfo = h.source_info || {};
    state.oodAvailable = !!h.ood_gate;
    $("methodCount").textContent = h.methods.length;
    setStatus("ONLINE", "live");

    // OOD gate toggle (only shown if the server has an anomaly/patchcore model)
    if (state.oodAvailable) {
      $("oodToggleRow").hidden = false;
      $("oodToggle").onchange = (e) => { state.ood = e.target.checked; infer(); };
    }

    // deep-link params: ?source=&frame=&method=&conf=
    const params = new URLSearchParams(location.search);
    const wantSource = h.sources.find((s) => s === params.get("source"));
    const wantMethod = params.get("method");
    const wantConf = parseFloat(params.get("conf"));
    state._wantFrame = parseInt(params.get("frame"));
    if (!Number.isNaN(wantConf)) { state.conf = wantConf; $("confSlider").value = wantConf; $("confVal").textContent = wantConf.toFixed(2); }

    // sources dropdown
    const sel = $("sourceSelect");
    sel.innerHTML = h.sources.map((s) => `<option ${s === wantSource ? "selected" : ""}>${s}</option>`).join("");
    sel.onchange = () => loadSource(sel.value);

    // method buttons
    state.method = (wantMethod && h.methods.includes(wantMethod)) ? wantMethod
      : (h.methods.includes("yolo") ? "yolo" : h.methods[0]);
    renderMethods();

    // controls
    $("frameSlider").oninput = (e) => { state.idx = +e.target.value; infer(); };
    $("prevBtn").onclick = () => step(-1);
    $("nextBtn").onclick = () => step(1);
    $("confSlider").oninput = debounce((e) => { state.conf = +e.target.value; $("confVal").textContent = state.conf.toFixed(2); infer(); }, 220);
    $("compareBtn").onclick = compareAll;

    await loadSource(wantSource || h.sources[0]);
  } catch (e) {
    setStatus("LINK FAILED", "err");
    $("vpReadout").textContent = "BACKEND UNREACHABLE — start scripts/serve.py";
    console.error(e);
  }
}

function renderMethods() {
  const grid = $("methodGrid");
  grid.innerHTML = "";
  for (const m of METHOD_ORDER) {
    const meta = METHOD_META[m];
    const on = state.methods.includes(m);
    const btn = document.createElement("button");
    btn.className = "method-btn" + (m === state.method ? " active" : "") + (on ? "" : " disabled");
    btn.innerHTML = `${meta.label}<span class="m-sub">${on ? meta.sub : "unavailable"}</span>`;
    if (on) btn.onclick = () => { state.method = m; renderMethods(); infer(); };
    grid.appendChild(btn);
  }
  const ex = $("methodExplain");
  if (ex) ex.textContent = (METHOD_META[state.method] || {}).info || "";
}

async function loadSource(name) {
  state.source = name;
  $("sourceInfo").textContent = state.sourceInfo[name] || "";
  const d = await api(`/api/frames?source=${encodeURIComponent(name)}`);
  state.frames = d.frames;
  state.idx = (Number.isInteger(state._wantFrame) && state._wantFrame >= 0
    && state._wantFrame < d.frames.length) ? state._wantFrame : 0;
  state._wantFrame = undefined;
  $("frameSlider").max = Math.max(0, d.frames.length - 1);
  $("frameSlider").value = state.idx;
  $("frameTotal").textContent = String(d.frames.length).padStart(3, "0");
  $("compareOut").hidden = true;
  await infer();
}

function step(n) {
  if (!state.frames.length) return;
  state.idx = (state.idx + n + state.frames.length) % state.frames.length;
  $("frameSlider").value = state.idx;
  infer();
}

async function infer() {
  if (state.busy || !state.frames.length) return;
  state.busy = true;
  const name = state.frames[state.idx];
  $("frameIdx").textContent = String(state.idx).padStart(3, "0");
  $("frameName").textContent = name;
  $("viewport").classList.add("loading");
  $("compareOut").hidden = true;
  try {
    const q = `source=${encodeURIComponent(state.source)}&name=${encodeURIComponent(name)}&method=${state.method}&conf=${state.conf}&ood=${state.ood ? 1 : 0}`;
    const r = await api(`/api/infer?${q}`);
    renderOOD(r.ood);
    $("frameImg").src = r.overlay;
    $("frameDims").textContent = `${r.image_size.width}×${r.image_size.height}`;
    $("latencyHud").textContent = r.latency_ms;
    $("statCount").textContent = r.count;
    $("statLat").textContent = r.latency_ms;
    $("statMethod").textContent = METHOD_META[r.method].label;
    $("vpMethod").textContent = r.is_heatmap ? r.method + " · heatmap" : r.method;
    $("vpMethod").className = "vp-tag" + (r.is_heatmap ? " heat" : "");
    $("vpReadout").textContent = `${r.count} REGION(S) · ${r.latency_ms} ms · conf≥${r.conf}`;
    renderDetections(r.detections, r.is_heatmap);
  } catch (e) {
    $("vpReadout").textContent = "INFERENCE ERROR";
    console.error(e);
  } finally {
    $("viewport").classList.remove("loading");
    state.busy = false;
  }
}

function renderOOD(ood) {
  const el = $("oodBadge");
  if (!ood) { el.hidden = true; return; }
  el.hidden = false;
  if (ood.flagged) {
    el.textContent = "⚠ OOD · review";
    el.className = "vp-tag ood-flag";
    el.title = `Out-of-distribution (score ${ood.score} ≥ ${ood.threshold}, via ${ood.via}) — route to a human`;
  } else {
    el.textContent = "✓ in-distribution";
    el.className = "vp-tag ood-ok";
    el.title = `In-distribution (score ${ood.score} < ${ood.threshold}, via ${ood.via})`;
  }
}

function renderDetections(dets, heat) {
  const list = $("detList");
  if (!dets.length) { list.innerHTML = '<div class="det-empty">NO DETECTIONS</div>'; return; }
  list.innerHTML = "";
  dets.slice(0, 40).forEach((d, i) => {
    const row = document.createElement("div");
    row.className = "det-row" + (heat ? " heat" : "");
    row.style.animationDelay = (i * 28) + "ms";
    row.innerHTML =
      `<div><div class="d-class">${d.class}</div>` +
      `<div class="d-bbox">[${d.bbox.join(", ")}]</div></div>` +
      `<div class="d-score">${d.score.toFixed(2)}</div>`;
    list.appendChild(row);
  });
}

async function compareAll() {
  if (!state.frames.length || state.busy) return;
  state.busy = true;
  const name = state.frames[state.idx];
  $("compareBtn").textContent = "⟳ RUNNING…";
  const rows = [];
  for (const m of state.methods) {
    try {
      const q = `source=${encodeURIComponent(state.source)}&name=${encodeURIComponent(name)}&method=${m}&conf=${state.conf}`;
      const r = await api(`/api/infer?${q}`);
      rows.push({ method: m, count: r.count, lat: r.latency_ms });
    } catch (e) { rows.push({ method: m, count: "—", lat: "—" }); }
  }
  const maxc = Math.max(1, ...rows.map((r) => +r.count || 0));
  $("compareRows").innerHTML = rows.map((r) =>
    `<div class="cmp-row" data-m="${r.method}">
       <span class="c-name">${METHOD_META[r.method].label}</span>
       <span class="cmp-bar"><i style="width:${(100 * (+r.count || 0) / maxc)}%"></i></span>
       <span class="c-meta">${r.count} · ${r.lat}ms</span>
     </div>`).join("");
  $("compareOut").hidden = false;
  $("compareRows").querySelectorAll(".cmp-row").forEach((el) => {
    el.onclick = () => { state.method = el.dataset.m; renderMethods(); infer(); };
  });
  $("compareBtn").textContent = "⟳ RUN ALL METHODS";
  state.busy = false;
}

function debounce(fn, ms) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

boot();

/* ---------------------------------------------------------------------------
 * Input modes: Browse (server-side frames), Drop (your own image), Live (camera).
 *
 * All three render through the same renderOOD/renderDetections path, so a
 * dropped frame and a browsed one are presented identically — a separate
 * display path for live would be a place for the two to silently disagree.
 * ------------------------------------------------------------------------- */
const live = { timer: null, running: false };

function setMode(mode) {
  state.mode = mode;
  for (const [id, m] of [["modeBrowse", "browse"], ["modeDrop", "drop"], ["modeLive", "live"]]) {
    const b = $(id);
    if (!b) continue;
    b.classList.toggle("active", m === mode);
    b.setAttribute("aria-selected", String(m === mode));
  }
  const show = (id, on) => { const el = $(id); if (el) el.hidden = !on; };
  show("browsePanel", mode === "browse");
  show("dropPanel", mode === "drop");
  show("livePanel", mode === "live");
  if (mode !== "live" && live.running) stopLive();
  if (mode === "browse" && state.frames && state.frames.length) infer();
}

/* ---- Drop ---------------------------------------------------------------- */
async function analyzeFile(file) {
  if (!file || state.busy) return;
  if (!/^image\/(png|jpeg)$/.test(file.type)) {
    $("vpReadout").textContent = "UNSUPPORTED FILE — PNG or JPEG only";
    return;
  }
  state.busy = true;
  $("viewport").classList.add("loading");
  $("compareOut").hidden = true;
  $("frameName").textContent = file.name;
  try {
    const body = new FormData();
    body.append("file", file);
    const q = `method=${state.method}&conf=${state.conf}&ood=${state.oodAvailable ? 1 : 0}`;
    const res = await fetch(`/api/analyze?${q}`, { method: "POST", body });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const r = await res.json();
    renderOOD(r.ood);
    $("frameImg").src = r.overlay;
    $("frameDims").textContent = `${r.image_size.width}×${r.image_size.height}`;
    $("latencyHud").textContent = r.latency_ms;
    $("statCount").textContent = r.count;
    $("statLat").textContent = r.latency_ms;
    $("statMethod").textContent = (METHOD_META[r.method] || {}).label || r.method;
    $("vpMethod").textContent = r.is_heatmap ? r.method + " · heatmap" : r.method;
    $("vpMethod").className = "vp-tag" + (r.is_heatmap ? " heat" : "");
    $("vpReadout").textContent = `${r.count} REGION(S) · ${r.latency_ms} ms · your image`;
    renderDetections(r.detections, r.is_heatmap);
  } catch (e) {
    $("vpReadout").textContent = "ANALYSIS FAILED";
    console.error(e);
  } finally {
    $("viewport").classList.remove("loading");
    state.busy = false;
  }
}

function wireDrop() {
  const zone = $("dropZone"), input = $("fileInput"), overlay = $("dropOverlay");
  if (input) input.onchange = (e) => { if (e.target.files[0]) analyzeFile(e.target.files[0]); };
  if (zone) {
    zone.onclick = () => input && input.click();
    zone.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input && input.click(); } };
  }
  // Dropping anywhere in the window works — hunting for a small target is friction.
  let depth = 0;
  window.addEventListener("dragenter", (e) => {
    if (!e.dataTransfer || ![...e.dataTransfer.types].includes("Files")) return;
    e.preventDefault(); depth++; if (overlay) overlay.hidden = false;
  });
  window.addEventListener("dragover", (e) => { e.preventDefault(); });
  window.addEventListener("dragleave", (e) => {
    e.preventDefault(); depth = Math.max(0, depth - 1);
    if (!depth && overlay) overlay.hidden = true;
  });
  window.addEventListener("drop", (e) => {
    e.preventDefault(); depth = 0; if (overlay) overlay.hidden = true;
    const f = e.dataTransfer && e.dataTransfer.files[0];
    if (!f) return;
    setMode("drop");
    analyzeFile(f);
  });
}

/* ---- Live ---------------------------------------------------------------- */
async function startLive() {
  const source = ($("liveSource").value || "0").trim();
  const minHits = $("minHits") ? $("minHits").value : 3;
  $("liveStart").disabled = true;
  $("liveNote").hidden = true;
  try {
    const q = `source=${encodeURIComponent(source)}&method=${state.method}` +
              `&conf=${state.conf}&min_hits=${minHits}&ood=${state.oodAvailable ? 1 : 0}`;
    const res = await fetch(`/api/live/start?${q}`, { method: "POST" });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${res.status}`);
    }
    live.running = true;
    $("liveStop").disabled = false;
    $("liveStats").hidden = false;
    // Cache-bust so a restart is not served the previous stream.
    $("frameImg").src = `/api/live/stream?fps=12&t=${Date.now()}`;
    $("frameName").textContent = source;
    live.timer = setInterval(pollLive, 1000);
    pollLive();
  } catch (e) {
    $("liveStart").disabled = false;
    const note = $("liveNote");
    note.hidden = false;
    note.textContent = "Could not open that source: " + e.message;
    console.error(e);
  }
}

async function stopLive() {
  clearInterval(live.timer); live.timer = null; live.running = false;
  try { await fetch("/api/live/stop", { method: "POST" }); } catch (e) { /* already gone */ }
  $("liveStart").disabled = false;
  $("liveStop").disabled = true;
  $("frameImg").src = "";
  $("vpReadout").textContent = "LIVE STOPPED";
}

async function pollLive() {
  try {
    const s = await api("/api/live/status");
    if (!s.running) {
      $("vpReadout").textContent = s.error ? `LIVE ENDED — ${s.error}` : "LIVE ENDED";
      if (live.running) stopLive();
      return;
    }
    $("liveFps").textContent = s.inference_fps;
    $("liveDropped").textContent = s.frames_dropped;
    $("liveEvents").textContent = s.events;
    $("latencyHud").textContent = s.latency_ms ?? "—";
    $("statCount").textContent = s.confirmed_now;
    $("statLat").textContent = s.latency_ms ?? 0;
    $("statMethod").textContent = (METHOD_META[s.method] || {}).label || s.method;
    $("vpMethod").textContent = s.method + " · live";
    $("vpMethod").className = "vp-tag";
    $("vpReadout").textContent =
      `${s.confirmed_now} CONFIRMED · ${s.inference_fps} fps · ${s.frames_dropped} dropped · ${s.source_kind}`;
    if (s.ood !== null && s.ood !== undefined) {
      renderOOD({ flagged: s.ood, score: 0, threshold: 0, via: "patchcore" });
    }
    const evs = (s.recent_events || []).slice().reverse();
    const list = $("detList");
    list.innerHTML = evs.length
      ? evs.map((e) =>
          `<div class="det-row"><div><div class="d-class">event · track ${e.track_id}</div>` +
          `<div class="d-bbox">frame ${e.frame} · [${(e.bbox || []).join(", ")}]</div></div>` +
          `<div class="d-score">${(e.score ?? 0).toFixed ? e.score.toFixed(2) : e.score}</div></div>`).join("")
      : '<div class="det-empty">NO CONFIRMED EVENTS YET</div>';
  } catch (e) { console.error(e); }
}

function wireModes() {
  const b = $("modeBrowse"), d = $("modeDrop"), l = $("modeLive");
  if (b) b.onclick = () => setMode("browse");
  if (d) d.onclick = () => setMode("drop");
  if (l) l.onclick = () => setMode("live");
  if ($("liveStart")) $("liveStart").onclick = startLive;
  if ($("liveStop")) $("liveStop").onclick = stopLive;
  if ($("minHits")) $("minHits").oninput = (e) => { $("minHitsVal").textContent = e.target.value; };
  window.addEventListener("beforeunload", () => {
    if (live.running) navigator.sendBeacon("/api/live/stop");
  });
}

wireModes();
wireDrop();
setMode("browse");
