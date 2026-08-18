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
                sourceInfo: {}, ood: false, oodAvailable: false,
                mode: "browse",
                // Kept so a threshold change can re-run the SAME image rather
                // than silently falling back to a server-side frame, plus the
                // settings it was last analysed with, so re-entering Drop after
                // changing the detector elsewhere does not show a stale result.
                lastDrop: null, lastDropSettings: null };

// Set when a frame request arrives while one is already running, so the final
// slider position is honoured instead of dropped.
let pendingFrame = null;

/* ---- Authentication -------------------------------------------------------
 * The service requires an API key on every /api route once NETINSPECT_API_KEY
 * is set — which it must be for any bind other than loopback. Without the code
 * below the console 401s on its first real request and reports "BACKEND
 * UNREACHABLE", so the product would be either unauthenticated or unusable,
 * with no configuration that is both secure and working.
 *
 * The key arrives once as ?key=… , is kept in sessionStorage, and is stripped
 * from the address bar so it does not sit in browser history or get copied out
 * of the URL bar into a chat message.
 * ------------------------------------------------------------------------- */
const auth = {
  key: null,

  load() {
    const params = new URLSearchParams(location.search);
    const fromUrl = params.get("key");
    if (fromUrl) {
      try { sessionStorage.setItem("netinspect_key", fromUrl); } catch (_) { /* private mode */ }
      params.delete("key");
      const rest = params.toString();
      history.replaceState({}, "", location.pathname + (rest ? `?${rest}` : ""));
      auth.key = fromUrl;
      return;
    }
    try { auth.key = sessionStorage.getItem("netinspect_key"); } catch (_) { auth.key = null; }
  },

  headers(extra) {
    // A custom header also lifts these out of the CORS "simple request" class,
    // so a page on another origin cannot silently drive this API.
    return auth.key ? { ...(extra || {}), "X-API-Key": auth.key } : (extra || {});
  },

  /* For <img src> and sendBeacon, where no header can be attached. */
  url(path) {
    if (!auth.key) return path;
    return path + (path.includes("?") ? "&" : "?") + "key=" + encodeURIComponent(auth.key);
  },
};
auth.load();

class Unauthorized extends Error {}

async function authFetch(path, opts) {
  const r = await fetch(path, { ...(opts || {}), headers: auth.headers((opts || {}).headers) });
  if (r.status === 401) throw new Unauthorized("401");
  return r;
}

async function api(path) {
  const r = await authFetch(path);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

/* Every async request here ends by writing the same handful of DOM nodes — the
 * viewport image, the readout, the stats, the detection list. Without a guard,
 * a slow response lands after the user has moved on and paints the old mode's
 * result over the new one: a dropped photo appearing in Browse under a frame
 * counter describing something else, a live stream overwritten by a stale
 * inference. Each writer takes a ticket before awaiting and checks it after; a
 * mode change invalidates every ticket outstanding.
 *
 * A guard on entry is not enough. The mode is re-checked *after* every await,
 * because that is where the user's click lands. */
let viewGen = 0;
const takeTicket = () => ++viewGen;
const ticketValid = (t) => t === viewGen;

/* Wipe everything that describes the outgoing view, so a mode never inherits a
 * picture, a verdict or a count from the one before it. */
function resetViewport() {
  const img = $("frameImg");
  if (img) img.removeAttribute("src");
  $("oodBadge").hidden = true;
  $("frameName").textContent = "—";
  $("frameDims").textContent = "—";
  $("vpMethod").textContent = "—";
  $("vpMethod").className = "vp-tag";
  $("statCount").textContent = "0";
  $("statLat").textContent = "0";
  $("statMethod").textContent = "—";
  $("detList").innerHTML = '<div class="det-empty">No detections</div>';
  $("compareOut").hidden = true;
  $("vpReadout").textContent = "—";
  $("viewport").classList.remove("loading");
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
      $("oodToggle").onchange = (e) => { state.ood = e.target.checked; applySettings(); };
    }
    applyModeCaps(state.mode);

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
    $("frameSlider").oninput = (e) => {
      state.idx = +e.target.value;
      if (state.busy) { pendingFrame = state.idx; return; }
      infer();
    };
    $("prevBtn").onclick = () => step(-1);
    $("nextBtn").onclick = () => step(1);
    $("confSlider").oninput = (e) => {
      state.conf = +e.target.value;
      $("confVal").textContent = state.conf.toFixed(2);
      applySettings();
    };
    $("compareBtn").onclick = compareAll;

    await loadSource(wantSource || h.sources[0]);
  } catch (e) {
    if (e instanceof Unauthorized) {
      setStatus("UNAUTHORIZED", "err");
      $("vpReadout").textContent =
        "AUTHENTICATION REQUIRED — open this console with ?key=YOUR_API_KEY";
    } else {
      setStatus("LINK FAILED", "err");
      $("vpReadout").textContent = "BACKEND UNREACHABLE — start scripts/serve.py";
    }
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
    if (on) btn.onclick = () => { state.method = m; renderMethods(); applySettings(); };
    grid.appendChild(btn);
  }
  const ex = $("methodExplain");
  if (ex) ex.textContent = (METHOD_META[state.method] || {}).info || "";
}

async function loadSource(name) {
  // Ticketed like the rest: switching source twice quickly could otherwise leave
  // state.source naming one directory and state.frames listing another's files.
  const ticket = takeTicket();
  $("sourceInfo").textContent = state.sourceInfo[name] || "";
  const d = await api(`/api/frames?source=${encodeURIComponent(name)}`);
  if (!ticketValid(ticket)) return;
  // Name and file list are assigned together, after the await. Setting the name
  // first would leave state.source and state.frames describing different
  // directories whenever the request is superseded.
  state.source = name;
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
  // Guarded so a stray call cannot paint a browse frame over a dropped image
  // or a live stream.
  if (state.mode !== "browse" || state.busy || !state.frames.length) return;
  const ticket = takeTicket();
  state.busy = true;
  const name = state.frames[state.idx];
  $("frameIdx").textContent = String(state.idx).padStart(3, "0");
  $("frameName").textContent = name;
  $("viewport").classList.add("loading");
  $("compareOut").hidden = true;
  try {
    const q = `source=${encodeURIComponent(state.source)}&name=${encodeURIComponent(name)}&method=${state.method}&conf=${state.conf}&ood=${state.ood ? 1 : 0}`;
    const r = await api(`/api/infer?${q}`);
    if (!ticketValid(ticket)) return;      // the user moved on while this ran
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
    if (ticketValid(ticket)) $("vpReadout").textContent = "INFERENCE ERROR";
    console.error(e);
  } finally {
    $("viewport").classList.remove("loading");
    state.busy = false;
    // A slider dragged while this was running had its input dropped; honour the
    // final position now instead of leaving the view a frame behind.
    if (pendingFrame !== null) { pendingFrame = null; infer(); }
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
  // Browse-only: it re-runs one server-side frame, which is not what a dropped
  // image or a live stream is showing.
  if (state.mode !== "browse" || !state.frames.length || state.busy) return;
  const ticket = takeTicket();
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
  // Leaving Browse mid-run must not pop the comparison panel open somewhere it
  // does not belong.
  if (!ticketValid(ticket)) { $("compareBtn").textContent = "⟳ RUN ALL METHODS"; state.busy = false; return; }
  const maxc = Math.max(1, ...rows.map((r) => +r.count || 0));
  $("compareRows").innerHTML = rows.map((r) =>
    `<div class="cmp-row" data-m="${r.method}">
       <span class="c-name">${METHOD_META[r.method].label}</span>
       <span class="cmp-bar"><i style="width:${(100 * (+r.count || 0) / maxc)}%"></i></span>
       <span class="c-meta">${r.count} · ${r.lat}ms</span>
     </div>`).join("");
  $("compareOut").hidden = false;
  $("compareRows").querySelectorAll(".cmp-row").forEach((el) => {
    el.onclick = () => { state.method = el.dataset.m; renderMethods(); applySettings(); };
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

/* Which shared controls each mode actually uses.
 *
 * The detector and threshold drive Browse, Drop and Live, but a Net 3-D scene
 * is built from a map that was computed offline — leaving those controls live
 * there would offer a knob that changes nothing. "Compare all methods" is
 * Browse-only because it re-runs one server-side frame, so in any other mode it
 * would quietly analyse something the viewer is not looking at. */
const MODE_CAPS = {
  browse: { method: true, conf: true, compare: true, ood: true, frameResults: true },
  drop:   { method: true, conf: true, compare: false, ood: true, frameResults: true },
  live:   { method: true, conf: true, compare: false, ood: true, frameResults: true },
  net:    { method: false, conf: false, compare: false, ood: false, frameResults: false },
};

const MODE_NOTE = {
  browse: "Frames are real SOLAQUA ROV footage (undamaged net) or synthetic damage " +
          "composited onto real net. Boxes are model output, not ground truth.",
  drop:   "Your image is analysed with the detector and threshold above; changing " +
          "either re-runs it on the same image.",
  live:   "Detector and threshold apply to the running session — changing them " +
          "restarts it so what you see always matches the controls.",
  net:    "Positions come from a map built offline by scripts/map_inspection.py, so " +
          "the detector and threshold above do not apply here.",
};

/* Visibility only, no side effects — boot() re-applies it once the server has
 * said whether an OOD model exists, which lands after the first setMode. */
function applyModeCaps(mode) {
  const caps = MODE_CAPS[mode] || MODE_CAPS.browse;
  const show = (id, on) => { const el = $(id); if (el) el.hidden = !on; };
  show("methodField", caps.method);
  show("confField", caps.conf);
  show("compareBtn", caps.compare);
  show("oodToggleRow", caps.ood && state.oodAvailable);
  show("frameResults", caps.frameResults);
  if (!caps.compare) show("compareOut", false);
  if ($("controlsNote")) $("controlsNote").textContent = MODE_NOTE[mode] || "";
}

function setMode(mode) {
  const previous = state.mode;
  if (previous !== mode) {
    takeTicket();          // invalidate anything still in flight for the old view
    resetViewport();
  }
  state.mode = mode;
  for (const [id, m] of [["modeBrowse", "browse"], ["modeDrop", "drop"],
                         ["modeLive", "live"], ["modeNet", "net"]]) {
    const b = $(id);
    if (!b) continue;
    b.classList.toggle("active", m === mode);
    b.setAttribute("aria-selected", String(m === mode));
  }
  const show = (id, on) => { const el = $(id); if (el) el.hidden = !on; };
  show("browsePanel", mode === "browse");
  show("dropPanel", mode === "drop");
  show("livePanel", mode === "live");
  show("netPanel", mode === "net");

  // A drag that ends outside the window never fires a matching dragleave, so the
  // overlay can be left up. Switching mode is a definite "not dragging".
  hideDropOverlay();

  applyModeCaps(mode);

  // The 3-D stage replaces the frame image rather than sitting beside it, so
  // the viewport always shows exactly one thing.
  show("netCanvas", mode === "net");
  const img = $("frameImg");
  if (img) img.hidden = mode === "net";
  show("siteOut", mode === "net" && !!net.scene);

  // Live keeps running ONLY into the 3-D view, which consumes its positions.
  // Any other mode owns the viewport, so a stream left running would fight it
  // for the same <img> and keep the camera open for nothing.
  const keepLive = mode === "net" && $("netLiveWire") && $("netLiveWire").checked;
  if (mode !== "live" && live.running && !keepLive) stopLive();

  // Each mode states what it is waiting for, so the readout never carries the
  // previous mode's sentence.
  if (mode === "live" && !live.running) {
    $("vpReadout").textContent = "PRESS START TO OPEN THE SOURCE";
  }
  if (mode === "net") $("vpReadout").textContent = "LOADING THE CAGE…";
  if (mode === "browse" && state.frames && state.frames.length) infer();
  if (mode === "drop" && previous !== "drop") {
    if (!state.lastDrop) {
      $("vpReadout").textContent = "DROP AN IMAGE TO ANALYSE";
    } else {
      // Always re-run: entering the mode cleared the viewport, so there is
      // nothing to preserve, and the detector or threshold may have been changed
      // in another mode since this image was last analysed.
      analyzeFile(state.lastDrop);
    }
  }
  if (mode === "net" && net.view) {
    // refit: the canvas had no size while the tab was hidden.
    net.view.resize(true);
    if (!net.scene) loadNetClips();
  }
}

/* One entry point for "the detector or threshold changed", because what that
 * should do depends entirely on what is on screen. Calling infer() regardless —
 * as this used to — replaced a dropped image with a server-side browse frame. */
const applySettings = debounce(() => {
  if (state.mode === "browse") { infer(); return; }
  if (state.mode === "drop") {
    if (state.lastDrop) analyzeFile(state.lastDrop);
    return;
  }
  if (state.mode === "live" && live.running) { restartLive(); }
  // net: nothing to re-run — the map is precomputed.
}, 320);

/* ---- Drop ---------------------------------------------------------------- */
async function analyzeFile(file) {
  // Deliberately does NOT bail on state.busy. A dropped file is an explicit
  // user action; discarding it silently because a browse inference happened to
  // be in flight looks exactly like a broken drop target.
  if (!file) return;
  if (!/^image\/(png|jpeg)$/.test(file.type)) {
    $("vpReadout").textContent = "UNSUPPORTED FILE — PNG or JPEG only";
    return;
  }
  state.lastDrop = file;
  state.lastDropSettings = `${state.method}|${state.conf}|${state.ood}`;
  const ticket = takeTicket();
  state.busy = true;
  $("viewport").classList.add("loading");
  $("compareOut").hidden = true;
  $("frameName").textContent = file.name;
  try {
    const body = new FormData();
    body.append("file", file);
    // Honour the checkbox rather than "on whenever a model exists" — a control
    // that is shown and ignored is worse than no control.
    const q = `method=${state.method}&conf=${state.conf}&ood=${state.ood ? 1 : 0}`;
    const res = await authFetch(`/api/analyze?${q}`, { method: "POST", body });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const r = await res.json();
    if (!ticketValid(ticket)) return;
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
    if (ticketValid(ticket)) $("vpReadout").textContent = "ANALYSIS FAILED";
    console.error(e);
  } finally {
    if (ticketValid(ticket)) $("viewport").classList.remove("loading");
    state.busy = false;
  }
}

let dropOverlayTimer = null;

function hideDropOverlay() {
  clearTimeout(dropOverlayTimer);
  dropOverlayTimer = null;
  const overlay = $("dropOverlay");
  if (overlay) overlay.hidden = true;
}

function wireDrop() {
  const zone = $("dropZone"), input = $("fileInput"), overlay = $("dropOverlay");
  if (input) input.onchange = (e) => { if (e.target.files[0]) analyzeFile(e.target.files[0]); };
  if (zone) {
    zone.onclick = () => input && input.click();
    zone.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input && input.click(); } };
  }
  // Dropping anywhere in the window works — hunting for a small target is friction.
  //
  // Shown on a heartbeat rather than an enter/leave counter. Counting looks
  // right and is not: dragenter and dragleave only balance while the pointer
  // stays inside the window, so a drag that ends outside it, is cancelled with
  // Escape, or drops on another app never delivers the closing event and the
  // overlay stays up over the viewport for the rest of the session. dragover
  // fires continuously while a drag is live, so "no dragover recently" is the
  // one reliable signal that the drag is over.
  const isFileDrag = (e) => !!e.dataTransfer &&
    Array.from(e.dataTransfer.types || []).includes("Files");

  const keepOverlayAlive = () => {
    if (overlay) overlay.hidden = false;
    clearTimeout(dropOverlayTimer);
    dropOverlayTimer = setTimeout(hideDropOverlay, 180);
  };

  window.addEventListener("dragenter", (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault(); keepOverlayAlive();
  });
  window.addEventListener("dragover", (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault(); keepOverlayAlive();
  });
  for (const ev of ["dragleave", "dragend", "drop", "blur"]) {
    window.addEventListener(ev, () => hideDropOverlay());
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") hideDropOverlay();
  });
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    hideDropOverlay();
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
  // Track ids are per-session and restart at 1, so a remembered set from the
  // previous session would suppress every defect the new one finds.
  net.liveSeen.clear();
  net.liveSites = [];
  if (net.view) net.view.setLive([]);
  $("liveStart").disabled = true;
  $("liveNote").hidden = true;
  try {
    // Odometry only runs when the 3-D view actually wants positions; it costs a
    // feature match per frame and buys nothing if nobody is placing the result.
    const odo = $("netLiveWire") && $("netLiveWire").checked;
    const q = `source=${encodeURIComponent(source)}&method=${state.method}` +
              `&conf=${state.conf}&min_hits=${minHits}&ood=${state.ood ? 1 : 0}` +
              `&odometry=${odo ? 1 : 0}&standoff_m=0.6`;
    const res = await authFetch(`/api/live/start?${q}`, { method: "POST" });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${res.status}`);
    }
    live.running = true;
    $("liveStop").disabled = false;
    $("liveStats").hidden = false;
    // Cache-bust so a restart is not served the previous stream.
    $("frameImg").src = auth.url(`/api/live/stream?fps=12&t=${Date.now()}`);
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

/* Changing the detector or threshold while a session is open has to restart it —
 * the server built the pipeline at start time, so without this the controls
 * would claim to be doing something the running session never sees. */
async function restartLive() {
  if (!live.running) return;
  await stopLive();
  await startLive();
}

async function stopLive() {
  clearInterval(live.timer); live.timer = null; live.running = false;
  try { await authFetch("/api/live/stop", { method: "POST" }); } catch (e) { /* already gone */ }
  $("liveStart").disabled = false;
  $("liveStop").disabled = true;
  // Only blank the stage if Live still owns it; after a mode switch the new mode
  // has already painted and these writes would wipe it.
  if (state.mode === "live") {
    $("frameImg").removeAttribute("src");
    $("vpReadout").textContent = "LIVE STOPPED";
  }
}

async function pollLive() {
  try {
    const s = await api("/api/live/status");
    // In Net 3-D the session runs only to supply positions; the viewport belongs
    // to the cage, so writing the stream's readout there would overwrite the
    // coverage headline once a second.
    const ownsViewport = state.mode === "live";
    if (!s.running) {
      if (ownsViewport) {
        $("vpReadout").textContent = s.error ? `LIVE ENDED — ${s.error}` : "LIVE ENDED";
      }
      if (live.running) stopLive();
      return;
    }
    $("liveFps").textContent = s.inference_fps;
    $("liveDropped").textContent = s.frames_dropped;
    $("liveEvents").textContent = s.events;
    $("latencyHud").textContent = s.latency_ms ?? "—";
    if (!ownsViewport) { pushLiveSites(s); return; }
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
    pushLiveSites(s);
  } catch (e) { console.error(e); }
}

/* ---- Net 3D --------------------------------------------------------------
 * An inspection map is a table of metres from wherever the pass began, which
 * is precise and unusable. This puts it on the cage next to the feed barge, so
 * "3.1 m along" becomes "just clockwise of the barge, 1.7 m down" — and one
 * click shows the picture, which is what decides whether it is worth a trip.
 *
 * The renderer draws the declared shell and the measured data differently on
 * purpose (see net3d.js); this module's job is to keep that distinction true in
 * the numbers it displays alongside.
 * ------------------------------------------------------------------------- */
const net = { view: null, scene: null, clip: null, liveSites: [], liveSeen: new Set() };

function netParams() {
  const v = (id, dflt) => {
    const el = $(id);
    const n = el ? parseFloat(el.value) : NaN;
    return Number.isFinite(n) ? n : dflt;
  };
  return new URLSearchParams({
    clip: net.clip || "",
    circumference_m: v("penCirc", 160),
    cylinder_depth_m: v("penCyl", 15),
    cone_depth_m: v("penCone", 10),
    barge_bearing_deg: v("penBarge", 0),
    start_bearing_deg: v("penStart", 0),
    clockwise: ($("penDir") || {}).value === "0" ? "false" : "true",
  });
}

async function loadScene() {
  if (!net.clip) return;
  const ticket = takeTicket();
  // A previous selection's crop and placement text describe a cage that is about
  // to be replaced.
  $("siteDetail").hidden = true;
  try {
    const s = await api(`/api/scene?${netParams()}`);
    if (!ticketValid(ticket)) return;
    net.scene = s;
    net.view.setScene(s);
    renderSites(s);
    renderNetCoverage(s);
    $("vpReadout").textContent =
      `${s.sites.length} SITE(S) · ${s.coverage.ring_percent}% OF THE RING · ` +
      `${s.coverage.area_percent}% OF THE NET`;
    $("frameName").textContent = `${s.clip} on a ${s.pen.circumference_m} m cage`;
    $("frameDims").textContent = `${s.pen.net_area_m2} m² net`;
    $("vpMethod").textContent = `${s.method || "map"} · 3D`;
    $("vpMethod").className = "vp-tag";
  } catch (e) {
    if (!ticketValid(ticket)) return;
    // Clear the cage rather than leaving the previous one drawn under an error
    // message — a stale cage with new dimensions in the panel is a lie.
    net.scene = null;
    if (net.view) net.view.setScene(null);
    $("siteOut").hidden = true;
    $("netCoverage").hidden = true;
    const msg = String(e.message || e);
    $("vpReadout").textContent = /400/.test(msg)
      ? "CAGE DIMENSIONS REJECTED — check the numbers above"
      : "NO MAP FOR THIS PASS — run scripts/map_inspection.py";
    console.error(e);
  }
}

function renderNetCoverage(s) {
  const el = $("netCoverage");
  if (!el) return;
  el.hidden = false;
  // The honest headline: a pass that sounds thorough in metres is a rounding
  // error of a real cage. Leading with it is the point, not a footnote.
  el.innerHTML =
    `<b>${s.coverage.area_percent}%</b> of this cage was looked at` +
    `<span>${s.coverage.swept_area_m2} m² swept of ${s.coverage.net_area_m2} m² of netting · ` +
    `${s.coverage.ring_percent}% of the ring · ~${s.coverage.passes_to_cover_ring} passes ` +
    `to circle it once, at this depth alone.</span>`;
}

function renderSites(s) {
  $("siteOut").hidden = false;
  const list = $("siteList");
  if (!s.sites.length) { list.innerHTML = '<div class="det-empty">NO SITES</div>'; return; }
  list.innerHTML = s.sites.map((k) =>
    `<div class="det-row site-row" data-site="${k.site_id}">` +
    `<div><div class="d-class">site ${k.site_id} · ${k.sightings}× · ${k.evidence || ""}</div>` +
    `<div class="d-bbox">${k.placed.description}</div></div>` +
    `<div class="d-score">${Math.round(k.median_width_mm)}×${Math.round(k.median_height_mm)} mm</div></div>`
  ).join("");
  for (const row of list.querySelectorAll(".site-row")) {
    row.onclick = () => selectSite(parseInt(row.dataset.site, 10));
  }
}

function selectSite(id) {
  const s = net.scene && net.scene.sites.find((k) => k.site_id === id);
  if (!s) {
    // Live markers are drawn on the cage but have no mapped site behind them —
    // no crop, no evidence count. Say that instead of ignoring the click.
    if (typeof id === "string" && id.startsWith("live-")) {
      const detail = $("siteDetail");
      detail.hidden = false;
      $("siteCrop").hidden = true;
      $("siteWhere").innerHTML =
        `<b>Live detection</b><br>Confirmed during the running session and placed ` +
        `by live odometry.<span class="warn-line">Not part of the mapped pass: no ` +
        `crop and no sighting count, and its position rests on a declared standoff ` +
        `rather than telemetry. Re-run scripts/map_inspection.py to map it properly.` +
        `</span>`;
    }
    return;
  }
  net.view.flyTo(id);
  for (const row of document.querySelectorAll(".site-row")) {
    row.classList.toggle("sel", parseInt(row.dataset.site, 10) === id);
  }
  const detail = $("siteDetail");
  detail.hidden = false;
  const hasCrop = net.scene.crops && net.scene.crops[String(id)];
  const img = $("siteCrop");
  img.hidden = !hasCrop;
  if (hasCrop) img.src = auth.url(`/api/scene/crop?clip=${encodeURIComponent(net.clip)}&site=${id}`);
  $("siteWhere").innerHTML =
    `<b>Site ${id}</b> — ${s.sightings} sighting(s), ${s.evidence || ""}<br>` +
    `${s.placed.description}<br>` +
    `<span class="muted">~${Math.round(s.median_width_mm)}×${Math.round(s.median_height_mm)} mm · ` +
    `bearing ${s.placed.bearing_deg}° · ${s.placed.section}</span>` +
    `<span class="warn-line">Sightings evidence a distinct object, not damage. ` +
    `On SOLAQUA the net is undamaged, so this is a false positive.</span>`;
}

async function loadNetClips() {
  try {
    const r = await api("/api/maps");
    const sel = $("netClip");
    sel.innerHTML = r.maps.length
      ? r.maps.map((m) => `<option value="${m}">${m}</option>`).join("")
      : '<option value="">— no maps found —</option>';
    net.clip = r.maps[0] || null;
    if (net.clip) await loadScene();
  } catch (e) { console.error(e); }
}

/* Live wiring: a confirmed defect gets an along-track position from the live
 * session's own odometry, which is placed on the cage exactly like a mapped
 * site. Positions are dashed in the view because live scale rests on a
 * declared standoff rather than telemetry. */
function pushLiveSites(status) {
  if (!net.view || !$("netLiveWire").checked || !net.scene) return;
  const geom = net.scene.pen;
  for (const e of (status.recent_events || [])) {
    if (net.liveSeen.has(e.track_id)) continue;
    if (e.along_m === null || e.along_m === undefined) continue;
    net.liveSeen.add(e.track_id);
    const bearing = (geom.start_bearing_deg +
      (geom.clockwise ? 1 : -1) * 360 * e.along_m / geom.circumference_m) * Math.PI / 180;
    const r = geom.radius_m;
    const depth = e.depth_m || 0;
    net.liveSites.push({ id: `live-${e.track_id}`, hits: e.hits || 1,
                         p: { x: r * Math.sin(bearing), y: r * Math.cos(bearing), z: -depth } });
  }
  if (net.liveSites.length) net.view.setLive(net.liveSites);
  const note = $("netLiveNote");
  note.hidden = false;
  note.textContent = net.liveSites.length
    ? `${net.liveSites.length} live defect(s) placed. Live positions use a declared ` +
      `standoff for scale, not telemetry — treat them as approximate.`
    : "Live wiring on. Confirmed defects will appear once the session reports positions.";
}

function wireNet() {
  const canvas = $("netCanvas");
  if (!canvas || !window.NET3D) return;
  net.view = NET3D.create(canvas, { onSelect: selectSite });
  window.addEventListener("resize", () => { if (state.mode === "net") net.view.resize(); });
  const reload = debounce(loadScene, 250);
  for (const id of ["penCirc", "penCyl", "penCone", "penBarge", "penStart"]) {
    if ($(id)) $(id).oninput = reload;
  }
  if ($("penDir")) $("penDir").onchange = loadScene;
  if ($("netClip")) $("netClip").onchange = (e) => { net.clip = e.target.value; loadScene(); };
}

function wireModes() {
  const b = $("modeBrowse"), d = $("modeDrop"), l = $("modeLive"), n = $("modeNet");
  if (b) b.onclick = () => setMode("browse");
  if (d) d.onclick = () => setMode("drop");
  if (l) l.onclick = () => setMode("live");
  if (n) n.onclick = () => setMode("net");
  if ($("liveStart")) $("liveStart").onclick = startLive;
  if ($("liveStop")) $("liveStop").onclick = stopLive;
  if ($("minHits")) {
    $("minHits").oninput = (e) => { $("minHitsVal").textContent = e.target.value; };
    // The server fixes min_hits when the session opens, so a running session has
    // to be restarted for the new value to mean anything. onchange, not oninput,
    // so dragging the slider does not restart the camera on every step.
    $("minHits").onchange = () => { if (live.running) restartLive(); };
  }
  if ($("netLiveWire")) {
    $("netLiveWire").onchange = (e) => {
      if (!e.target.checked) {
        // Stale markers would keep sitting on the cage claiming to be live.
        net.liveSites = []; net.liveSeen.clear();
        if (net.view) net.view.setLive([]);
        if ($("netLiveNote")) $("netLiveNote").hidden = true;
      }
      // Odometry is decided at session start, so the toggle only takes effect
      // via a restart.
      if (live.running) restartLive();
    };
  }
  window.addEventListener("beforeunload", () => {
    if (live.running) navigator.sendBeacon(auth.url("/api/live/stop"));
  });
}

wireModes();
wireDrop();
wireNet();
setMode("browse");
