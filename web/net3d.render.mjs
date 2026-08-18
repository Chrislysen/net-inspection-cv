/* Render the 3-D cage viewer to a static SVG, headlessly.
 *
 * The README needs a picture of the interactive view, and a screenshot would be
 * a photograph of a thing nobody can regenerate. This drives the *actual*
 * renderer through a stubbed canvas and writes what it drew, so the figure in
 * the docs is produced by the code it documents and cannot drift from it.
 *
 *     node web/net3d.render.mjs <scene.json> <out.svg>
 *
 * A scene comes from the running server:
 *     curl "localhost:8000/api/scene?clip=2024-08-22_14-29-05" > scene.json
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const W = 1240, H = 520;                       // one panel

function recorder() {
  const rec = { segs: [], arcs: [], texts: [], polys: [] };
  let p = { x: 0, y: 0 }, dash = [], path = [];
  const stack = [];
  // save/restore must really save and restore. A no-op pair lets the cage
  // shell's dash leak into everything drawn afterwards, which silently paints
  // the measured data in the declared style — the exact confusion this whole
  // renderer exists to prevent.
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
    // Canvas semantics: a path becomes ink only when stroke() or fill() is
    // called. Emitting on lineTo instead turned the filled band ribbon into 371
    // phantom stroke="" lines in the output.
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
  return { rec, ctx: c };
}

function stubCanvas(ctx) {
  const canvas = {
    width: W, height: H,
    getContext: () => ctx,
    getBoundingClientRect: () => ({ width: W, height: H, left: 0, top: 0 }),
    addEventListener() {}, setPointerCapture() {}, releasePointerCapture() {},
  };
  ctx.canvas = canvas;
  return canvas;
}

const esc = (s) => String(s).replace(/[&<>]/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[m]));
const n = (v) => Math.round(v * 10) / 10;

function toSvg2(rec, dy, id, title, subtitle) {
  // Each panel is clipped to its own band. Without this the second panel's
  // geometry — the barge is overhead in the close-up — paints up into the
  // first, because it is drawn later in document order.
  const out = [`<clipPath id="${id}"><rect x="0" y="${dy}" width="${W}" height="${H}"/></clipPath>`,
               `<g clip-path="url(#${id})">`];
  // Flat fill rather than a gradient: this SVG is rendered by GitHub, and the
  // fewer features it leans on, the fewer ways it has to come out wrong.
  out.push(`<rect x="0" y="${dy}" width="${W}" height="${H}" fill="#eef4f7"/>`);
  for (const q of rec.polys) {
    const pts = q.pts.map((p) => `${n(p.x)},${n(p.y + dy)}`).join(' ');
    out.push(`<polygon points="${pts}" fill="${q.fill}" fill-opacity="${n(q.a)}"/>`);
  }
  for (const s of rec.segs) {
    out.push(`<line x1="${n(s.x1)}" y1="${n(s.y1 + dy)}" x2="${n(s.x2)}" y2="${n(s.y2 + dy)}" ` +
             `stroke="${s.s}" stroke-width="${n(s.w)}" stroke-opacity="${n(s.a)}"` +
             `${s.d ? ' stroke-dasharray="3 3"' : ''}/>`);
  }
  for (const a of rec.arcs) {
    out.push(`<circle cx="${n(a.x)}" cy="${n(a.y + dy)}" r="${n(a.r)}" fill="none" ` +
             `stroke="${a.s}" stroke-width="${n(a.w)}"/>`);
  }
  for (const t of rec.texts) {
    const anchor = t.al === 'center' ? 'middle' : 'start';
    const bold = /600|bold/.test(t.fo || '') ? ' font-weight="600"' : '';
    const size = (t.fo || '').match(/(\d+)px/);
    out.push(`<text x="${n(t.x)}" y="${n(t.y + dy)}" fill="${t.f}" text-anchor="${anchor}"` +
             `${bold} font-size="${size ? size[1] : 11}">${esc(t.t)}</text>`);
  }
  out.push(`<text x="18" y="${dy + 26}" font-size="14" font-weight="700" fill="#0b0b0b">${esc(title)}</text>`);
  out.push(`<text x="18" y="${dy + 44}" font-size="11.5" fill="#52514e">${esc(subtitle)}</text>`);
  out.push('</g>');
  return out.join('\n');
}

// ---------------------------------------------------------------- run
const scene = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const outPath = process.argv[3] || join(HERE, '..', 'docs', 'images', 'net3d_console.svg');

globalThis.window = {};
globalThis.document = { querySelectorAll: () => [] };
new Function(readFileSync(join(HERE, 'net3d.js'), 'utf8'))();
const NET3D = globalThis.window.NET3D;

function panel(fn) {
  const { rec, ctx } = recorder();
  const view = NET3D.create(stubCanvas(ctx), {});
  view.setScene(scene);
  // Clear what setScene drew so a panel records exactly one frame.
  rec.segs.length = 0; rec.arcs.length = 0; rec.texts.length = 0; rec.polys.length = 0;
  fn(view);
  return rec;
}

// The viewer's default framing assumes a tall browser stage; these panels are
// wide and short, so pull back and look down a little more to fit the cage.
// These panels are wide and short, so look down a little more than the app's
// default and let the viewer's own fit() size the scene to the panel.
const wide = panel((v) => {
  v.camera.el = 0.62;
  v.fit(0.82);   // leave room for the panel title
  v.draw();
});
const best = (scene.sites || []).slice().sort((a, b) => b.sightings - a.sightings)[0];
const close = panel((v) => { if (best) v.flyTo(best.site_id); else v.draw(); });

const cov = scene.coverage || {};
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H * 2}" ` +
  `viewBox="0 0 ${W} ${H * 2}" font-family="IBM Plex Sans, Segoe UI, sans-serif">
${toSvg2(wide, 0, 'p0', `The cage — ${scene.clip || 'inspection pass'}`,
  `${cov.area_percent}% of ${cov.net_area_m2} m² of netting inspected · ${cov.ring_percent}% of the ring · ` +
  `dashed = declared cage, solid = measured data`)}
<line x1="0" y1="${H}" x2="${W}" y2="${H}" stroke="#c7d0d6" stroke-width="1"/>
${toSvg2(close, H, 'p1', best ? `Flown to site ${best.site_id} — ${best.sightings} sightings` : 'Inspected band',
  best ? best.placed.description : 'the measured band on the cage wall')}
</svg>
`;
writeFileSync(outPath, svg);
console.log(`wrote ${outPath} (${wide.segs.length + close.segs.length} segments)`);
