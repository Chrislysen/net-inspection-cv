/* Interactive 3-D cage viewer — where the damage is, and what it is next to.
 *
 * Hand-rolled on a 2-D canvas rather than pulled from a 3-D library. The scene
 * is a few thousand line segments; a renderer for that is ~200 lines, and the
 * console stays dependency-free and offline — which matters for something meant
 * to run on a boat.
 *
 * The visual grammar carries the honesty of the data, and it is the reason this
 * file is not just "draw a net":
 *   DECLARED  (cage shell, collar, cone, feed barge) -> thin, muted, dashed.
 *             The operator told us these dimensions; we did not measure them.
 *   MEASURED  (the swept band, the defect sites)     -> solid, saturated, bold.
 *             These came from the footage.
 * A viewer must never be able to mistake the reference frame for a
 * reconstruction, so the two never share a colour or a line weight.
 */
'use strict';

const NET3D = (() => {
  const INK = '#0b0b0b', SOFT = '#52514e', MUTED = '#b9b8ae';
  const DECLARED = '#9aa79c';            // the shell we were told about
  const MEASURED = '#2a78d6';            // the strip we actually saw
  const ALARM = '#eb6834';               // sites
  const SURFACE_SEA = '#cfe0ea';
  const BARGE = '#7a6a55';

  // ---------------------------------------------------------------- camera
  function makeCamera() {
    return { az: 0.6, el: 0.42, dist: 90, tx: 0, ty: 0, tz: -6, fov: 900 };
  }

  function basis(cam) {
    const ce = Math.cos(cam.el), se = Math.sin(cam.el);
    const eye = {
      x: cam.tx + cam.dist * ce * Math.sin(cam.az),
      y: cam.ty + cam.dist * ce * Math.cos(cam.az),
      z: cam.tz + cam.dist * se,
    };
    const f = norm({ x: cam.tx - eye.x, y: cam.ty - eye.y, z: cam.tz - eye.z });
    const r = norm(cross(f, { x: 0, y: 0, z: 1 }));
    const u = cross(r, f);
    return { eye, f, r, u };
  }

  const cross = (a, b) => ({ x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x });
  const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
  function norm(v) { const l = Math.hypot(v.x, v.y, v.z) || 1; return { x: v.x / l, y: v.y / l, z: v.z / l }; }

  function toCam(p, B) {
    const v = { x: p.x - B.eye.x, y: p.y - B.eye.y, z: p.z - B.eye.z };
    return { x: dot(v, B.r), y: dot(v, B.u), z: dot(v, B.f) };
  }

  function toScreen(c, cam, w, h) {
    const s = cam.fov / c.z;
    return { x: w / 2 + c.x * s, y: h / 2 - c.y * s, z: c.z };
  }

  const NEAR = 0.35;

  /* Clip a segment to the near plane so geometry behind the camera does not
   * wrap around and smear across the view. */
  function clipNear(a, b) {
    if (a.z > NEAR && b.z > NEAR) return [a, b];
    if (a.z <= NEAR && b.z <= NEAR) return null;
    const t = (NEAR - a.z) / (b.z - a.z);
    const m = { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: NEAR };
    return a.z > NEAR ? [a, m] : [m, b];
  }

  // ---------------------------------------------------------------- geometry
  const RAD = Math.PI / 180;

  /* Cage shell from declared dimensions: collar ring, cylindrical wall, and the
   * cone that tapers to the bottom ring. The cone is not decoration — a hole in
   * it is a different repair and a different escape risk from a wall hole. */
  function shellSegments(pen) {
    const R = pen.radius_m, cyl = pen.cylinder_depth_m, cone = pen.cone_depth_m;
    const segs = [];
    const ring = (r, z, step = 6) => {
      for (let a = 0; a < 360; a += step) {
        const b = a + step;
        segs.push([{ x: r * Math.sin(a * RAD), y: r * Math.cos(a * RAD), z },
                   { x: r * Math.sin(b * RAD), y: r * Math.cos(b * RAD), z }]);
      }
    };
    ring(R, 0, 4);                                   // floating collar
    const depths = [];
    for (let d = 5; d <= cyl; d += 5) depths.push(d);
    depths.forEach((d) => ring(R, -d, 8));
    for (let d = cyl + 2.5; d < cyl + cone; d += 2.5) {
      ring(R * (1 - (d - cyl) / cone), -d, 8);
    }
    // Vertical netting lines, wall then cone, so the taper is visible.
    for (let a = 0; a < 360; a += 7.5) {
      const sx = Math.sin(a * RAD), cy = Math.cos(a * RAD);
      segs.push([{ x: R * sx, y: R * cy, z: 0 }, { x: R * sx, y: R * cy, z: -cyl }]);
      if (cone > 0) {
        segs.push([{ x: R * sx, y: R * cy, z: -cyl }, { x: 0, y: 0, z: -(cyl + cone) }]);
      }
    }
    return segs;
  }

  /* Sea surface as a sparse grid, purely to give the eye a horizon. */
  function seaSegments(R) {
    const segs = [], e = R * 2.6, step = R / 2.5;
    for (let i = -e; i <= e; i += step) {
      segs.push([{ x: i, y: -e, z: 0 }, { x: i, y: e, z: 0 }]);
      segs.push([{ x: -e, y: i, z: 0 }, { x: e, y: i, z: 0 }]);
    }
    return segs;
  }

  /* The feed barge: the one landmark on a farm nobody mistakes for anything
   * else, and therefore the thing every position should be described against. */
  function bargeSegments(barge, pen) {
    const bx = barge.x_m, by = barge.y_m;
    const ang = Math.atan2(bx, by);
    const ux = Math.cos(ang), uy = -Math.sin(ang);      // along the barge beam
    const vx = Math.sin(ang), vy = Math.cos(ang);       // toward the pen
    const L = 11, W = 7, H = 4.2;
    const corner = (s, t, z) => ({ x: bx + ux * s + vx * t, y: by + uy * s + vy * t, z });
    const segs = [];
    const deck = [corner(-L, -W, 0), corner(L, -W, 0), corner(L, W, 0), corner(-L, W, 0)];
    const roof = deck.map((p) => ({ x: p.x, y: p.y, z: H }));
    for (let i = 0; i < 4; i++) {
      segs.push([deck[i], deck[(i + 1) % 4]]);
      segs.push([roof[i], roof[(i + 1) % 4]]);
      segs.push([deck[i], roof[i]]);
    }
    // Feed silos on deck — the silhouette that makes it read as a feed barge.
    for (const s of [-5.5, 0, 5.5]) {
      const b = corner(s, 1.5, 0), t = corner(s, 1.5, H * 1.5);
      segs.push([b, t]);
      const r = 1.6;
      for (let a = 0; a < 360; a += 45) {
        const a2 = a + 45;
        segs.push([{ x: t.x + r * Math.sin(a * RAD), y: t.y + r * Math.cos(a * RAD), z: t.z },
                   { x: t.x + r * Math.sin(a2 * RAD), y: t.y + r * Math.cos(a2 * RAD), z: t.z }]);
      }
    }
    // Walkway from the barge to the collar: the route a person actually takes.
    const onRing = { x: pen.radius_m * Math.sin(ang), y: pen.radius_m * Math.cos(ang), z: 0 };
    segs.push([corner(-1.4, W, 0), { x: onRing.x, y: onRing.y, z: 0 }]);
    segs.push([corner(1.4, W, 0), { x: onRing.x, y: onRing.y, z: 0 }]);
    return segs;
  }

  function bandRibbon(band, pen) {
    // The measured strip as ONE polygon — upper edge forward, lower edge back —
    // rather than a quad per step. Overlapping translucent quads accumulate
    // alpha at every seam, which draws a stripe per frame and makes an evenly
    // swept band look like it was inspected in ragged bursts.
    if (band.length < 2) return [];
    const r = pen.radius_m * 1.002;                  // a hair proud of the netting
    const at = (p, sgn) => {
      const t = p.bearing_deg * RAD;
      const z = -(p.depth_m || 0) + sgn * (p.footprint_m || 0.8) / 2;
      return { x: r * Math.sin(t), y: r * Math.cos(t), z };
    };
    const top = band.map((p) => at(p, 1));
    const bottom = band.map((p) => at(p, -1)).reverse();
    return [top.concat(bottom)];
  }

  // ---------------------------------------------------------------- viewer
  function create(canvas, opts = {}) {
    const cam = makeCamera();
    let scene = null, shell = [], sea = [], barge = [], ribbon = [];
    let sitePx = [];                 // screen positions, for hit-testing
    let selected = null, live = [];
    let dragging = false, lx = 0, ly = 0;
    const onSelect = opts.onSelect || (() => {});

    function setScene(s) {
      scene = s;
      if (!s) {
        // Redraw rather than return: leaving the previous cage on screen would
        // show a pass that is no longer loaded.
        shell = []; sea = []; barge = []; ribbon = []; live = []; selected = null;
        draw();
        return;
      }
      shell = shellSegments(s.pen);
      sea = seaSegments(s.pen.radius_m);
      barge = bargeSegments(s.barge, s.pen);
      ribbon = bandRibbon(s.band || [], s.pen);
      cam.dist = s.pen.radius_m * 2.9;
      cam.tz = -s.pen.total_depth_m / 2.4;
      // Open on the inspected band rather than an arbitrary heading: the first
      // thing a viewer should see is the thing that was measured.
      if (s.band && s.band.length) cam.az = (s.band[0].bearing_deg + 180) * RAD;
      draw();
    }

    function setLive(sites) { live = sites || []; draw(); }
    function select(id) { selected = id; draw(); }

    function flyTo(id) {
      const s = (scene && scene.sites || []).find((k) => k.site_id === id);
      if (!s) return;
      // Move the whole target, not just its depth: leaving tx/ty at the pen
      // centre orbits the cage instead of framing the site.
      cam.tx = s.placed.x_m;
      cam.ty = s.placed.y_m;
      cam.tz = s.placed.z_m;
      // Stand inside the pen looking out at the wall — the far netting is then
      // behind the camera rather than cluttering the panel being examined.
      cam.az = (s.placed.bearing_deg + 180) * RAD;
      cam.el = 0.16;
      cam.dist = 9;
      selected = id;
      draw();
    }

    function resize() {
      const r = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(r.width * dpr));
      canvas.height = Math.max(1, Math.round(r.height * dpr));
      draw();
    }

    function draw() {
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.width / dpr, h = canvas.height / dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (!scene) {
        ctx.fillStyle = SOFT; ctx.font = '13px IBM Plex Sans, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No inspection map loaded.', w / 2, h / 2);
        return;
      }
      const B = basis(cam);
      const P = (p) => toScreen(toCam(p, B), cam, w, h);

      const strokeSegs = (segs, color, width, alpha, dash) => {
        ctx.save();
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.globalAlpha = alpha;
        if (dash) ctx.setLineDash(dash);
        ctx.beginPath();
        for (const [a, b] of segs) {
          const ca = toCam(a, B), cb = toCam(b, B);
          const cl = clipNear(ca, cb);
          if (!cl) continue;
          const sa = toScreen(cl[0], cam, w, h), sb = toScreen(cl[1], cam, w, h);
          ctx.moveTo(sa.x, sa.y); ctx.lineTo(sb.x, sb.y);
        }
        ctx.stroke();
        ctx.restore();
      };

      // Declared reference frame — thin, muted, dashed. Never mistakable for data.
      strokeSegs(sea, SURFACE_SEA, 1, 0.5);
      strokeSegs(shell, DECLARED, 1, 0.55, [3, 3]);
      strokeSegs(barge, BARGE, 1.4, 0.85);

      // Measured strip — solid and saturated.
      ctx.save();
      ctx.fillStyle = MEASURED; ctx.globalAlpha = 0.5;
      for (const q of ribbon) {
        const cs = q.map((p) => toCam(p, B));
        if (cs.some((c) => c.z <= NEAR)) continue;
        const ss = cs.map((c) => toScreen(c, cam, w, h));
        ctx.beginPath();
        ctx.moveTo(ss[0].x, ss[0].y);
        for (let i = 1; i < ss.length; i++) ctx.lineTo(ss[i].x, ss[i].y);
        ctx.closePath(); ctx.fill();
      }
      ctx.restore();

      // Sites, painted back to front so near markers sit on top.
      sitePx = [];
      const marks = [];
      for (const s of (scene.sites || [])) {
        marks.push({ id: s.site_id, n: s.sightings, ev: s.evidence || '',
                     p: { x: s.placed.x_m, y: s.placed.y_m, z: s.placed.z_m }, live: false });
      }
      for (const s of live) {
        marks.push({ id: s.id, n: s.hits || 1, ev: 'live', p: s.p, live: true });
      }
      const drawn = [];
      for (const m of marks) {
        const c = toCam(m.p, B);
        if (c.z <= NEAR) continue;
        drawn.push({ m, s: toScreen(c, cam, w, h) });
      }
      drawn.sort((a, b) => b.s.z - a.s.z);
      for (const { m, s } of drawn) {
        const r = Math.max(5, Math.min(26, (6 + Math.sqrt(m.n) * 2.4) * (cam.fov / s.z) / 12));
        ctx.save();
        ctx.strokeStyle = ALARM; ctx.lineWidth = m.id === selected ? 3.4 : 2.1;
        if (m.live) ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
        ctx.stroke();
        if (m.id === selected) { ctx.globalAlpha = 0.18; ctx.fillStyle = ALARM; ctx.fill(); }
        ctx.restore();
        if (m.n >= 3 || m.id === selected) {
          ctx.save();
          ctx.fillStyle = ALARM; ctx.font = '600 11px IBM Plex Sans, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(`${m.n}×`, s.x, s.y - r - 5);
          ctx.restore();
        }
        sitePx.push({ id: m.id, x: s.x, y: s.y, r: r + 6 });
      }

      // Labels for the fixed things, so orientation never depends on memory.
      const label = (p, text, color) => {
        const c = toCam(p, B);
        if (c.z <= NEAR) return;
        const s = toScreen(c, cam, w, h);
        ctx.save();
        ctx.fillStyle = color; ctx.font = '600 11px IBM Plex Sans, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(text, s.x, s.y);
        ctx.restore();
      };
      label({ x: scene.barge.x_m, y: scene.barge.y_m, z: 7.5 }, 'FEED BARGE', BARGE);
      label({ x: 0, y: 0, z: -scene.pen.total_depth_m - 1.6 }, 'centre weight', MUTED);

      drawInspectedSector(ctx, w, h, B);
      drawLegend(ctx, w, h);
      drawCompass(ctx, w, h, B);
    }

    /* The inspected band is 5.5 m of a 160 m ring — genuinely a sliver, and at
     * a whole-cage zoom it is smaller than a pixel. Marking the sector on the
     * collar keeps it findable without exaggerating its size, which a fattened
     * band would do. The sliver is the honest headline, not a rendering problem
     * to design away. */
    function drawInspectedSector(ctx, w, h, B) {
      const band = scene.band || [];
      if (band.length < 2) return;
      const R = scene.pen.radius_m;
      const b0 = band[0].bearing_deg, b1 = band[band.length - 1].bearing_deg;
      const pt = (deg, z) => ({ x: R * Math.sin(deg * RAD), y: R * Math.cos(deg * RAD), z });
      const arc = [];
      const steps = 12;
      for (let i = 0; i <= steps; i++) arc.push(pt(b0 + (b1 - b0) * i / steps, 0));

      ctx.save();
      ctx.strokeStyle = MEASURED; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.beginPath();
      let started = false;
      for (const p of arc) {
        const c = toCam(p, B);
        if (c.z <= NEAR) { started = false; continue; }
        const s = toScreen(c, cam, w, h);
        if (!started) { ctx.moveTo(s.x, s.y); started = true; } else { ctx.lineTo(s.x, s.y); }
      }
      ctx.stroke();

      // A leader from the collar down to the band's depth, so the eye can find
      // it when the band itself is too small to see.
      const mid = (b0 + b1) / 2;
      const depth = band[Math.floor(band.length / 2)].depth_m || 0;
      const a = toCam(pt(mid, 0), B), bb = toCam(pt(mid, -depth - 1.2), B);
      const cl = clipNear(a, bb);
      if (cl) {
        const sa = toScreen(cl[0], cam, w, h), sb = toScreen(cl[1], cam, w, h);
        // Solid, not dashed: the leader points at measured geometry, and dash
        // is reserved for things this footage did not measure.
        ctx.lineWidth = 1.4; ctx.globalAlpha = 0.85;
        ctx.beginPath(); ctx.moveTo(sa.x, sa.y); ctx.lineTo(sb.x, sb.y); ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = MEASURED; ctx.font = '600 10px IBM Plex Sans, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('inspected band', sa.x, sa.y - 8);
      }
      ctx.restore();
    }

    /* A compass fixed to the screen rather than a label in the world: the barge
     * defaults to due north, so an in-world "N" sits on top of it. */
    function drawCompass(ctx, w, h, B) {
      const cx = w - 46, cy = 46, r = 22;
      ctx.save();
      ctx.strokeStyle = MUTED; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
      ctx.font = '600 10px IBM Plex Sans, sans-serif';
      ctx.textAlign = 'center';
      for (const [deg, name] of [[0, 'N'], [90, 'E'], [180, 'S'], [270, 'W']]) {
        // Project the world direction through the camera basis rather than
        // reconstructing the screen angle by hand — the same transform the rest
        // of the scene uses, so the compass cannot disagree with the view.
        const d = { x: Math.sin(deg * RAD), y: Math.cos(deg * RAD), z: 0 };
        const sx = dot(d, B.r), sy = dot(d, B.u);
        const len = Math.hypot(sx, sy) || 1;
        ctx.fillStyle = name === 'N' ? ALARM : SOFT;
        ctx.fillText(name, cx + (sx / len) * (r - 7), cy - (sy / len) * (r - 7) + 3.5);
      }
      ctx.restore();
    }

    function drawLegend(ctx, w, h) {
      const rows = [
        ['measured — inspected band', MEASURED, false],
        ['measured — defect sites', ALARM, false],
        ['declared — cage shell', DECLARED, true],
        ['declared — feed barge', BARGE, false],
      ];
      ctx.save();
      ctx.font = '11px IBM Plex Sans, sans-serif';
      ctx.textAlign = 'left';
      let y = h - 12 - rows.length * 15;
      for (const [text, color, dash] of rows) {
        ctx.strokeStyle = color; ctx.lineWidth = 2.2;
        ctx.setLineDash(dash ? [3, 3] : []);
        ctx.beginPath(); ctx.moveTo(12, y); ctx.lineTo(30, y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = SOFT; ctx.fillText(text, 36, y + 4);
        y += 15;
      }
      ctx.restore();
    }

    // ------------------------------------------------------------ input
    canvas.addEventListener('pointerdown', (e) => {
      dragging = true; lx = e.clientX; ly = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      cam.az -= (e.clientX - lx) * 0.008;
      cam.el = Math.max(-0.25, Math.min(1.45, cam.el + (e.clientY - ly) * 0.006));
      lx = e.clientX; ly = e.clientY;
      draw();
    });
    const stop = (e) => {
      if (!dragging) return;
      dragging = false;
      try { canvas.releasePointerCapture(e.pointerId); } catch (_) { /* already gone */ }
    };
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      cam.dist = Math.max(6, Math.min(600, cam.dist * (1 + Math.sign(e.deltaY) * 0.12)));
      draw();
    }, { passive: false });
    canvas.addEventListener('click', (e) => {
      const r = canvas.getBoundingClientRect();
      const x = e.clientX - r.left, y = e.clientY - r.top;
      let best = null, bestD = 1e9;
      for (const s of sitePx) {
        const d = Math.hypot(s.x - x, s.y - y);
        if (d < s.r && d < bestD) { best = s.id; bestD = d; }
      }
      if (best !== null) { selected = best; draw(); onSelect(best); }
    });

    return { setScene, setLive, select, flyTo, resize, draw,
             get camera() { return cam; } };
  }

  return { create };
})();

window.NET3D = NET3D;
