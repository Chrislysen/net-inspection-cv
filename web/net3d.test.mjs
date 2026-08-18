/* Headless checks for the 3-D cage renderer.
 *
 * A renderer that is only ever eyeballed is a renderer whose maths is untested,
 * and projection bugs are the kind that look plausible: a scene that is mirrored,
 * inside-out, or quietly drawing geometry behind the camera still produces a
 * picture. So the canvas context is stubbed, every draw call recorded, and the
 * results checked against things that must be true.
 *
 *     node web/net3d.test.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const HERE = dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------- stubs
function makeCtx(rec) {
  // save/restore are implemented for real. Stubbing them as no-ops lets the
  // cage shell's dash leak into every later stroke, so the measured data gets
  // recorded in the declared style and the provenance assertions below pass for
  // the wrong reason.
  const stack = [];
  const snapshot = () => ({
    strokeStyle: ctx.strokeStyle, fillStyle: ctx.fillStyle, lineWidth: ctx.lineWidth,
    globalAlpha: ctx.globalAlpha, font: ctx.font, textAlign: ctx.textAlign,
    dash: ctx._dash,
  });
  const ctx = {
    canvas: null, _p: { x: 0, y: 0 }, _dash: [],
    setTransform() {}, clearRect() {},
    save() { stack.push(snapshot()); },
    restore() { const s = stack.pop(); if (s) Object.assign(ctx, s, { _dash: s.dash }); },
    // A path becomes ink on stroke()/fill(), not on lineTo — otherwise the
    // filled band ribbon is counted as hundreds of colourless line segments.
    beginPath() { ctx._path = []; }, closePath() {},
    fill() { rec.fills++; },
    stroke() {
      rec.strokes++;
      for (const sub of (ctx._path || [])) {
        for (let i = 1; i < sub.length; i++) {
          rec.segments.push({ a: sub[i - 1], b: sub[i], color: ctx.strokeStyle,
                              dash: !!(ctx._dash || []).length });
        }
      }
    },
    setLineDash(d) { ctx._dash = d || []; },
    moveTo(x, y) { ctx._p = { x, y }; (ctx._path = ctx._path || []).push([{ x, y }]); },
    lineTo(x, y) {
      ctx._path = ctx._path || [];
      if (!ctx._path.length) ctx._path.push([{ x: ctx._p.x, y: ctx._p.y }]);
      ctx._path[ctx._path.length - 1].push({ x, y });
      ctx._p = { x, y };
    },
    arc(x, y, r) { rec.arcs.push({ x, y, r, color: ctx.strokeStyle }); },
    fillText(t, x, y) { rec.texts.push({ t, x, y }); },
    strokeStyle: '', fillStyle: '', lineWidth: 1, globalAlpha: 1, font: '', textAlign: '',
  };
  return ctx;
}

function makeCanvas(w = 900, h = 600) {
  const listeners = {};
  const rec = { segments: [], arcs: [], texts: [], points: [], strokes: 0, fills: 0 };
  const ctx = makeCtx(rec);
  const canvas = {
    width: w, height: h, _rec: rec,
    getContext: () => ctx,
    getBoundingClientRect: () => ({ width: w, height: h, left: 0, top: 0 }),
    addEventListener: (k, fn) => { (listeners[k] = listeners[k] || []).push(fn); },
    setPointerCapture() {}, releasePointerCapture() {},
    _fire: (k, e) => (listeners[k] || []).forEach((fn) => fn(e)),
  };
  ctx.canvas = canvas;
  return canvas;
}

// ---------------------------------------------------------------- load module
globalThis.window = {};
globalThis.document = { querySelectorAll: () => [] };
const src = readFileSync(join(HERE, 'net3d.js'), 'utf8');
new Function(src)();
const NET3D = globalThis.window.NET3D;
assert.ok(NET3D, 'net3d.js must expose window.NET3D');

// ---------------------------------------------------------------- scene
const scene = {
  clip: 'test',
  pen: { circumference_m: 160, radius_m: 25.465, cylinder_depth_m: 15,
         cone_depth_m: 10, total_depth_m: 25, net_area_m2: 4588.6, declared: true,
         start_bearing_deg: 0, clockwise: true },
  barge: { x_m: 0, y_m: 43.46, bearing_deg: 0, label: 'feed barge' },
  band: Array.from({ length: 40 }, (_, i) => ({
    along_m: i * 0.14, depth_m: 1.7, bearing_deg: i * 0.14 / 160 * 360,
    standoff_m: 0.6, footprint_m: 1.15,
  })),
  sites: [
    { site_id: 4, sightings: 72, evidence: 'strong', median_width_mm: 53, median_height_mm: 50,
      placed: { x_m: 25.4, y_m: 1.4, z_m: -1.7, bearing_deg: 3.1, description: 'x' } },
    { site_id: 5, sightings: 27, evidence: 'strong', median_width_mm: 65, median_height_mm: 50,
      placed: { x_m: 25.3, y_m: 2.8, z_m: -1.7, bearing_deg: 6.3, description: 'x' } },
  ],
  coverage: { area_percent: 0.137, ring_percent: 3.43 },
};

let tests = 0, failed = 0;
function test(name, fn) {
  tests++;
  try { fn(); console.log(`  ok   ${name}`); }
  catch (e) { failed++; console.error(`  FAIL ${name}\n       ${e.message}`); }
}

const inView = (p, c) => p.x >= -c.width && p.x <= c.width * 2 &&
                         p.y >= -c.height && p.y <= c.height * 2;

console.log('net3d renderer');

test('draws a scene without throwing, and draws a lot of it', () => {
  const c = makeCanvas();
  NET3D.create(c).setScene(scene);
  assert.ok(c._rec.segments.length > 300,
    `expected a full cage of segments, got ${c._rec.segments.length}`);
});

test('no segment is flung to infinity by the near plane', () => {
  // The classic near-plane bug: geometry behind the camera projects to enormous
  // coordinates and smears across the view instead of being clipped away.
  // Merely leaving the viewport is normal — the sea grid is meant to run past
  // the edge — so the check is on magnitude, not on being on-screen.
  const c = makeCanvas();
  NET3D.create(c).setScene(scene);
  const limit = Math.max(c.width, c.height) * 50;
  const wild = c._rec.segments.filter((s) =>
    [s.a, s.b].some((p) => !Number.isFinite(p.x) || !Number.isFinite(p.y) ||
                           Math.abs(p.x) > limit || Math.abs(p.y) > limit));
  assert.equal(wild.length, 0, `${wild.length} segments projected past ±${limit}px`);
});

test('most of the scene is actually inside the viewport', () => {
  const c = makeCanvas();
  NET3D.create(c).setScene(scene);
  const on = c._rec.segments.filter((s) => inView(s.a, c) && inView(s.b, c));
  assert.ok(on.length / c._rec.segments.length > 0.85,
    `only ${on.length}/${c._rec.segments.length} segments framed — camera is off`);
});

test('both sites are drawn as markers', () => {
  const c = makeCanvas();
  NET3D.create(c).setScene(scene);
  assert.ok(c._rec.arcs.length >= 2, `expected >= 2 site markers, got ${c._rec.arcs.length}`);
});

test('a site seen 72 times is drawn bigger than one seen 27 times', () => {
  const c = makeCanvas();
  NET3D.create(c).setScene(scene);
  const [a, b] = c._rec.arcs;
  assert.ok(Math.abs(a.r - b.r) > 0.2, 'marker size must carry the evidence count');
});

test('the declared shell is dashed and the measured band is not', () => {
  // This is the honesty rule made testable: if the shell ever stops being
  // visually distinct from the data, the picture starts implying we measured
  // a pen we did not.
  const c = makeCanvas();
  NET3D.create(c).setScene(scene);
  const dashed = c._rec.segments.filter((s) => s.dash);
  const solid = c._rec.segments.filter((s) => !s.dash);
  // Assert on colours, not counts: off-screen geometry is clipped away, so how
  // many segments survive depends on framing and is not the contract.
  assert.ok(dashed.length > 100, `cage shell must be dashed, got ${dashed.length}`);
  assert.ok(solid.some((s) => s.color === '#2a78d6'), 'measured band must be solid');
  assert.ok(solid.some((s) => s.color === '#7a6a55'), 'the barge must be solid');
  // The shell is the declared colour and nothing solid may borrow it.
  assert.ok(dashed.every((s) => s.color === '#9aa79c'),
    'only the declared shell may be dashed');
  assert.ok(!solid.some((s) => s.color === '#9aa79c'),
    'declared geometry must never be drawn solid');
  assert.ok(c._rec.fills > 0, 'the measured band must be a filled ribbon');
});

test('the feed barge is labelled so orientation never relies on memory', () => {
  const c = makeCanvas();
  NET3D.create(c).setScene(scene);
  assert.ok(c._rec.texts.some((t) => /FEED BARGE/i.test(t.t)), 'barge label missing');
  assert.ok(c._rec.texts.some((t) => t.t === 'N'), 'north marker missing');
});

test('the legend names both provenance classes', () => {
  const c = makeCanvas();
  NET3D.create(c).setScene(scene);
  const all = c._rec.texts.map((t) => t.t).join(' | ');
  assert.ok(/measured/.test(all) && /declared/.test(all),
    `legend must distinguish measured from declared, got: ${all}`);
});

test('orbiting changes the projection rather than redrawing the same picture', () => {
  const c = makeCanvas();
  const v = NET3D.create(c);
  v.setScene(scene);
  const before = c._rec.segments.length;
  c._fire('pointerdown', { clientX: 0, clientY: 0, pointerId: 1 });
  c._fire('pointermove', { clientX: 120, clientY: 10, pointerId: 1 });
  c._fire('pointerup', { clientX: 120, clientY: 10, pointerId: 1 });
  assert.ok(c._rec.segments.length > before, 'a drag must trigger a redraw');
  assert.ok(Math.abs(v.camera.az - 0.6) > 0.1, 'azimuth must change on drag');
});

test('dragging right turns the model right — grab-and-drag, not inverted', () => {
  // The sign here is genuinely counter-intuitive (increasing azimuth walks the
  // eye clockwise, and at azimuth 0 screen-right is west), so this asserts the
  // observable behaviour rather than the sign: a landmark under the pointer must
  // travel the same way the pointer does.
  const c = makeCanvas();
  const v = NET3D.create(c);
  v.setScene(scene);
  // Judge it on the NEAR side. Grab-and-drag is about the surface facing you;
  // anything beyond the pivot swings the other way, so testing against a far-side
  // landmark would assert the exact inversion this guards against. Azimuth 0 puts
  // the eye north of the pen, which is the side the barge is moored on.
  v.camera.az = 0;

  const bargeX = () => {
    c._rec.texts.length = 0;
    v.draw();
    const t = c._rec.texts.find((k) => /FEED BARGE/i.test(k.t));
    assert.ok(t, 'barge label must be on screen for this check');
    return t.x;
  };

  const before = bargeX();
  c._fire('pointerdown', { clientX: 400, clientY: 300, pointerId: 1 });
  c._fire('pointermove', { clientX: 460, clientY: 300, pointerId: 1 });
  c._fire('pointerup', { clientX: 460, clientY: 300, pointerId: 1 });
  const after = bargeX();

  assert.ok(after > before,
    `dragging right must move the scene right (barge ${before} -> ${after})`);
});

test('dragging down tips the cage so its top comes toward the viewer', () => {
  const c = makeCanvas();
  const v = NET3D.create(c);
  v.setScene(scene);
  const el0 = v.camera.el;
  c._fire('pointerdown', { clientX: 400, clientY: 300, pointerId: 2 });
  c._fire('pointermove', { clientX: 400, clientY: 360, pointerId: 2 });
  c._fire('pointerup', { clientX: 400, clientY: 360, pointerId: 2 });
  assert.ok(v.camera.el > el0, 'dragging down must raise the eye, not lower it');
});

test('elevation is clamped so the camera cannot flip through the poles', () => {
  const c = makeCanvas();
  const v = NET3D.create(c);
  v.setScene(scene);
  c._fire('pointerdown', { clientX: 0, clientY: 0, pointerId: 3 });
  for (let i = 1; i <= 200; i++) c._fire('pointermove', { clientX: 0, clientY: i * 40, pointerId: 3 });
  assert.ok(v.camera.el <= 1.45 && v.camera.el >= -0.25, `el escaped: ${v.camera.el}`);
});

test('zoom is clamped so the scene cannot be lost', () => {
  const c = makeCanvas();
  const v = NET3D.create(c);
  v.setScene(scene);
  for (let i = 0; i < 200; i++) c._fire('wheel', { deltaY: 100, preventDefault() {} });
  assert.ok(v.camera.dist <= 4000, 'zoom-out must clamp');
  for (let i = 0; i < 400; i++) c._fire('wheel', { deltaY: -100, preventDefault() {} });
  assert.ok(v.camera.dist >= 6, 'zoom-in must clamp');
});

test('clicking a marker selects that site', () => {
  const c = makeCanvas();
  let picked = null;
  const v = NET3D.create(c, { onSelect: (id) => { picked = id; } });
  v.setScene(scene);
  const arc = c._rec.arcs[0];
  c._fire('click', { clientX: arc.x, clientY: arc.y });
  assert.ok(picked === 4 || picked === 5, `click should select a site, got ${picked}`);
});

test('flyTo aims the camera at the site bearing', () => {
  const c = makeCanvas();
  const v = NET3D.create(c);
  v.setScene(scene);
  v.flyTo(5);
  const want = (scene.sites[1].placed.bearing_deg + 180) * Math.PI / 180;
  assert.ok(Math.abs(v.camera.az - want) < 1e-6, 'flyTo must face the site');
});

test('an empty scene renders a message instead of crashing', () => {
  const c = makeCanvas();
  NET3D.create(c).setScene(null);
  assert.ok(c._rec.texts.some((t) => /No inspection map/i.test(t.t)));
});

test('a cage with no cone still draws', () => {
  const c = makeCanvas();
  const flat = JSON.parse(JSON.stringify(scene));
  flat.pen.cone_depth_m = 0;
  flat.pen.total_depth_m = 15;
  NET3D.create(c).setScene(flat);
  assert.ok(c._rec.segments.length > 200, 'a cone-less cage must still render');
});

test('live sites are drawn alongside mapped ones', () => {
  const c = makeCanvas();
  const v = NET3D.create(c);
  v.setScene(scene);
  const before = c._rec.arcs.length;
  v.setLive([{ id: 'live-1', hits: 4, p: { x: 25.0, y: -4.0, z: -1.5 } }]);
  assert.ok(c._rec.arcs.length > before, 'a live site must add a marker');
});


test('orbiting past a marker does not select it', () => {
  // A drag ends with a click on the same element. Without a movement threshold,
  // letting go of the mouse near a site teleports the camera into it.
  const c = makeCanvas();
  let picked = null;
  const v = NET3D.create(c, { onSelect: (id) => { picked = id; } });
  v.setScene(scene);
  const arc = c._rec.arcs[0];
  c._fire('pointerdown', { clientX: arc.x - 80, clientY: arc.y, pointerId: 9 });
  c._fire('pointermove', { clientX: arc.x, clientY: arc.y, pointerId: 9 });
  c._fire('pointerup', { clientX: arc.x, clientY: arc.y, pointerId: 9 });
  c._fire('click', { clientX: arc.x, clientY: arc.y });
  assert.equal(picked, null, 'a drag that ends on a marker must not select it');
});

test('a genuine click still selects after a previous drag', () => {
  const c = makeCanvas();
  let picked = null;
  const v = NET3D.create(c, { onSelect: (id) => { picked = id; } });
  v.setScene(scene);
  c._fire('pointerdown', { clientX: 10, clientY: 10, pointerId: 8 });
  c._fire('pointermove', { clientX: 200, clientY: 10, pointerId: 8 });
  c._fire('pointerup', { clientX: 200, clientY: 10, pointerId: 8 });
  c._fire('click', { clientX: 200, clientY: 10 });          // swallowed by the drag
  c._rec.arcs.length = 0;
  v.draw();
  const arc = c._rec.arcs[0];
  c._fire('pointerdown', { clientX: arc.x, clientY: arc.y, pointerId: 7 });
  c._fire('pointerup', { clientX: arc.x, clientY: arc.y, pointerId: 7 });
  c._fire('click', { clientX: arc.x, clientY: arc.y });
  assert.ok(picked !== null, 'a still click must still select');
});

console.log(`\n${tests - failed}/${tests} passed`);
process.exit(failed ? 1 : 0);
