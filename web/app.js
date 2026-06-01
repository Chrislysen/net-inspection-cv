/* NET-INSPECT // ROV CONSOLE — client logic */
const $ = (id) => document.getElementById(id);
const METHOD_META = {
  classical: { label: "CLASSICAL", sub: "OpenCV heuristic" },
  anomaly:   { label: "ANOMALY",   sub: "Mahalanobis" },
  patchcore: { label: "PATCHCORE", sub: "foundation" },
  yolo:      { label: "YOLO",      sub: "supervised" },
};

const state = { sources: [], methods: [], method: "yolo", source: null,
                frames: [], idx: 0, conf: 0.25, busy: false };

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
    $("methodCount").textContent = h.methods.length;
    setStatus("ONLINE", "live");

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
  for (const m of ["classical", "anomaly", "patchcore", "yolo"]) {
    const meta = METHOD_META[m];
    const on = state.methods.includes(m);
    const btn = document.createElement("button");
    btn.className = "method-btn" + (m === state.method ? " active" : "") + (on ? "" : " disabled");
    btn.innerHTML = `${meta.label}<span class="m-sub">${on ? meta.sub : "unavailable"}</span>`;
    if (on) btn.onclick = () => { state.method = m; renderMethods(); infer(); };
    grid.appendChild(btn);
  }
}

async function loadSource(name) {
  state.source = name;
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
    const q = `source=${encodeURIComponent(state.source)}&name=${encodeURIComponent(name)}&method=${state.method}&conf=${state.conf}`;
    const r = await api(`/api/infer?${q}`);
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
