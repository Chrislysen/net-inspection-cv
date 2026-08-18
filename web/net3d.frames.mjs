/* Render an orbit of the 3-D cage viewer to a JSON frame dump.
 *
 * Feeds scripts/make_cage_gif.py, which rasterises the frames into an animated
 * GIF for the README. The orbit is produced by the real renderer rather than a
 * screen recording, so the animation cannot drift from the code and regenerates
 * on any machine with node.
 *
 *     node web/net3d.frames.mjs <scene.json> <out.json> [frames]
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const W = 900, H = 480;

function recorder() {
  const rec = { segs: [], arcs: [], texts: [], polys: [] };
  let p = { x: 0, y: 0 }, dash = [], path = [];
  const stack = [];
  const snapshot = () => ({
    strokeStyle: c.strokeStyle, fillStyle: c.fillStyle, lineWidth: c.lineWidth,
    globalAlpha: c.globalAlpha, font: c.font, textAlign: c.textAlign, dash,
  });
  const c = {
    canvas: null,
    setTransform() {}, clearRect() {},
    save() { stack.push(snapshot()); },
    restore() {
      const s = stack.pop();
      if (!s) return;
      c.strokeStyle = s.strokeStyle; c.fillStyle = s.fillStyle;
      c.lineWidth = s.lineWidth; c.globalAlpha = s.globalAlpha;
      c.font = s.font; c.textAlign = s.textAlign; dash = s.dash;
    },
    beginPath() { path = []; }, closePath() {},
    fill() {
      for (const sub of path) {
        if (sub.length > 2) rec.polys.push({ pts: sub.slice(), fill: c.fillStyle, a: c.globalAlpha });
      }
    },
    stroke() {
      for (const sub of path) {
        for (let i = 1; i < sub.length; i++) {
          rec.segs.push({ x1: sub[i - 1].x, y1: sub[i - 1].y, x2: sub[i].x, y2: sub[i].y,
                          s: c.strokeStyle, w: c.lineWidth, a: c.globalAlpha, d: dash.length > 0 });
        }
      }
    },
    setLineDash(d) { dash = d || []; },
    moveTo(x, y) { p = { x, y }; path.push([{ x, y }]); },
    lineTo(x, y) {
      if (!path.length) path.push([{ x: p.x, y: p.y }]);
      path[path.length - 1].push({ x, y });
      p = { x, y };
    },
    arc(x, y, r) { rec.arcs.push({ x, y, r, s: c.strokeStyle, w: c.lineWidth }); },
    fillText(t, x, y) { rec.texts.push({ t, x, y, f: c.fillStyle, al: c.textAlign, fo: c.font }); },
    strokeStyle: '', fillStyle: '', lineWidth: 1, globalAlpha: 1, font: '', textAlign: 'start',
  };
  const canvas = {
    width: W, height: H,
    getContext: () => c,
    getBoundingClientRect: () => ({ width: W, height: H, left: 0, top: 0 }),
    addEventListener() {}, setPointerCapture() {}, releasePointerCapture() {},
  };
  c.canvas = canvas;
  return { rec, canvas };
}

const scene = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const outPath = process.argv[3];
const nFrames = parseInt(process.argv[4] || '48', 10);

globalThis.window = {};
globalThis.document = { querySelectorAll: () => [] };
new Function(readFileSync(join(HERE, 'net3d.js'), 'utf8'))();
const NET3D = globalThis.window.NET3D;

const frames = [];
for (let i = 0; i < nFrames; i++) {
  const { rec, canvas } = recorder();
  const view = NET3D.create(canvas, {});
  view.setScene(scene);
  // Centre on the pen, not on the midpoint between pen and barge: with an
  // offset target the cage swings across the frame as the camera orbits, which
  // reads as the model sliding around rather than turning. Centred, the cage
  // stays put and the barge orbits it — which is also the truth.
  view.camera.tx = 0;
  view.camera.ty = 0;
  view.camera.tz = -scene.pen.total_depth_m / 2.5;
  view.camera.el = 0.34;
  view.fit(0.74);            // headroom so the barge stays in frame all the way round
  // A full turn, so the loop is seamless.
  view.camera.az = (i / nFrames) * Math.PI * 2;
  rec.segs.length = 0; rec.arcs.length = 0; rec.texts.length = 0; rec.polys.length = 0;
  view.draw();
  frames.push(rec);
}

writeFileSync(outPath, JSON.stringify({ W, H, frames }));
console.log(`wrote ${outPath}: ${nFrames} frames, ${frames[0].segs.length} segments each`);
