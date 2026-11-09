import { useEffect, useRef, useState } from 'react';
import { panScene, useScenePointer } from '../scenePointer.js';

import { ASSOC_VIEWS, assocHit, assocProject } from './assocLayerGeometry.js';

const ZONES = {
  heads: { x0: -8, x1: 143, z0: -147, z1: -34 },
  hs: { x0: 181, x1: 491, z0: -147, z1: -34 },
  mlp: { x0: -8, x1: 491, z0: 31, z1: 203 },
};

export default function AssocLayerScene({ dims, objs, layer, view, tex: TX, marks, links = [], onPick, overlays, live }) {
  const cv = useRef(null), st = useRef({ yaw: ASSOC_VIEWS[view][0], pitch: ASSOC_VIEWS[view][1], view, zoom: 1, hits: [], linkHits: [], W: 0, H: 0, dpr: 1 });
  const [tick, setTick] = useState(0), [tip, setTip] = useState(null);
  useEffect(() => {
    const el = cv.current, ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect(), s = st.current;
      s.dpr = window.devicePixelRatio || 1; s.W = r.width; s.H = r.height;
      el.width = r.width * s.dpr; el.height = r.height * s.dpr; setTick(t => t + 1);
    });
    ro.observe(el);
    const wheel = e => { e.preventDefault(); st.current.zoom = Math.max(0.35, Math.min(12, st.current.zoom * (e.deltaY < 0 ? 1.1 : 0.9))); setTick(t => t + 1); };
    el.addEventListener('wheel', wheel, { passive: false });
    return () => { ro.disconnect(); el.removeEventListener('wheel', wheel); };
  }, []);
  const NL = dims.NL, LY = l => (l - (NL - 1) / 2) * (331.2 / Math.max(1, NL - 1)) + 20;
  const proj = (x, y, z) => assocProject(st.current, x, y, z);
  useEffect(() => {
    const ctx = cv.current.getContext('2d'), s = st.current;
    if (s.view !== view) { s.view = view; s.yaw = ASSOC_VIEWS[view][0]; s.pitch = ASSOC_VIEWS[view][1]; }
    if (!s.W) return;
    ctx.setTransform(s.dpr, 0, 0, s.dpr, 0, 0); ctx.clearRect(0, 0, s.W, s.H); s.hits = []; s.linkHits = [];
    const fontScale = Math.max(0.68, Math.min(1, s.W / 760));
    const label = (text, p, color = '#3e4b60', size = 12) => { ctx.fillStyle = color; ctx.font = `${size * fontScale}px sans-serif`; ctx.fillText(text, p[0], p[1]); };
    const quad = (h, fill, stroke, lw = 1) => {
      ctx.beginPath(); [[h.x0, h.z0], [h.x1, h.z0], [h.x1, h.z1], [h.x0, h.z1]].forEach(([x, z], i) => { const p = proj(x, h.y, z); i ? ctx.lineTo(...p) : ctx.moveTo(...p); });
      ctx.closePath(); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
    };
    const drawTex = (img, h) => {
      const p = proj(h.x0, h.y, h.z0), a = proj(h.x1, h.y, h.z0), b = proj(h.x0, h.y, h.z1);
      ctx.save(); ctx.imageSmoothingEnabled = false;
      ctx.setTransform(s.dpr * (a[0] - p[0]) / img.width, s.dpr * (a[1] - p[1]) / img.width,
        s.dpr * (b[0] - p[0]) / img.height, s.dpr * (b[1] - p[1]) / img.height, s.dpr * p[0], s.dpr * p[1]);
      ctx.drawImage(img, 0, 0); ctx.restore();
    };
    const alpha = l => l === layer ? 1 : l > layer ? 0.18 : 0.8;
    const llm = { ...objs[0], x: -283, w: 202, d: 91 };
    for (let l = 0; l < NL; l++) {
      const h = { o: llm, l, y: LY(l), x0: llm.x - 101, x1: llm.x + 101, z0: -45.5, z1: 45.5 };
      ctx.globalAlpha = alpha(l); quad(h, l ? '#f7f9fc' : '#e8ecf2', l === layer ? '#d29a00' : '#97a6bb', l === layer ? 2.2 : 0.8);
      if (l) {
        quad({ ...h, x0: h.x0 + 9, x1: llm.x - 16, z0: -35, z1: 35 }, '#e5def3');
        quad({ ...h, x0: llm.x + 16, x1: h.x1 - 9, z0: -35, z1: 35 }, '#dcefe4');
      }
      s.hits.push(h);
    }
    ctx.globalAlpha = 1;
    ctx.beginPath(); for (let l = 0; l < NL; l++) { const p = proj(llm.x, LY(l), 0); l ? ctx.lineTo(...p) : ctx.moveTo(...p); }
    ctx.strokeStyle = '#172033'; ctx.lineWidth = 2.5; ctx.stroke();
    for (let l = 0; l < NL; l += 6) { const p = proj(llm.x - 108, LY(l), 45.5); ctx.textAlign = 'right'; label(l ? `L${l}` : '词嵌入', [p[0], p[1] + 4], '#68748a', 11); }
    ctx.textAlign = 'left';
    const title = proj(llm.x - 105, LY(NL - 1) + 28, -45.5); label(llm.name, title);
    const start = proj(llm.x + llm.w / 2, LY(layer), 0), end = proj(-16, 20, 0);
    ctx.strokeStyle = '#d29a00'; ctx.lineWidth = 1.3; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(...start); ctx.lineTo(...end); ctx.stroke(); ctx.setLineDash([]);
    label('局部放大', [(start[0] + end[0]) / 2 - 18, (start[1] + end[1]) / 2 - 8], '#986e00', 10);
    for (const o of objs.slice(1)) {
      if (layer === 0 && o.id !== 'hs') continue;
      const h = { o, l: layer, y: 20, ...ZONES[o.id] };
      quad(h, '#e4eaf1', '#b4bfce');
      const t = TX[o.id]?.[o.id === 'hs' ? layer : layer - 1];
      if (t) drawTex(t, h);
      quad(h, null, '#9cadc4'); s.hits.push(h);
      const p = proj(h.x0, 20, h.z0); label(o.name, [p[0], p[1] - 10]);
    }
    if (s.W > 640) label(layer ? `第 ${layer} 层 · 块 B${layer - 1} 的内部` : '输入词嵌入 · 全部坐标', proj(0, 20, -220), '#183f77', 14);
    overlays?.({ ctx, proj, LY, llm, alpha });
    if (live?.curve) {
      const mx = Math.max(...live.curve.slice(1)) || 1;
      for (let l = 1; l < Math.min(NL, live.curve.length); l++) { const [x, y] = proj(llm.x, LY(l), 0); ctx.fillStyle = 'rgba(210,154,0,.55)'; ctx.beginPath(); ctx.arc(x, y, 2 + 9 * live.curve[l] / mx, 0, 7); ctx.fill(); }
    }
    const point = m => {
      const h = s.hits.find(t => t.o.id === m.id && t.l === layer);
      if (!h?.o.grid || m.idx >= h.o.count) return null;
      const [gw, gh] = h.o.grid;
      return proj(h.x0 + (h.x1 - h.x0) * ((m.idx % gw) + 0.5) / gw, h.y,
        h.z0 + (h.z1 - h.z0) * (Math.floor(m.idx / gw) + 0.5) / gh);
    };
    for (const m of marks) { const p = point(m); if (!p) continue; ctx.strokeStyle = '#a33e43'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(...p, 6, 0, 7); ctx.stroke(); }
    const origin = marks[0] && point(marks[0]);
    if (origin) {
      // Endpoints are layer anchors with explicit native IDs, not a geometric neuron embedding.
      let lastLabelY = -Infinity;
      links.slice(0, 6).map(link => ({ link, p: proj(llm.x - 105, LY(link.l), 45.5) })).sort((a, b) => a.p[1] - b.p[1]).forEach(({ link, p }) => {
        const y = Math.max(p[1] - 24, lastLabelY + 15); lastLabelY = y;
        ctx.strokeStyle = link.kind === 'observation' ? '#71549a' : '#54789a'; ctx.lineWidth = 1;
        ctx.setLineDash(link.kind === 'observation' ? [3, 4] : []); ctx.beginPath(); ctx.moveTo(...origin);
        ctx.bezierCurveTo(origin[0] - 70, origin[1], p[0] + 70, y, p[0], y); ctx.stroke(); ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(p[0], y); ctx.lineTo(...p); ctx.stroke(); ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(...p, 3, 0, 7); ctx.fill();
        const text = `B${link.l - 1} ${link.o === 'mlp' ? 'U' : 'H'}${link.idx}`;
        ctx.font = `${10 * fontScale}px sans-serif`; const w = ctx.measureText(text).width, x = Math.max(4, p[0] - 5 - w);
        label(text, [x, y + 3], ctx.strokeStyle, 10);
        s.linkHits.push({ x, y: y - 9, w: w + 6, h: 14, link });
      });
    }
  });
  const pickAt = e => {
    const r = e.currentTarget.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, s = st.current;
    const link = [...s.linkHits].reverse().find(h => x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h);
    if (link) return { h: link.link, x, y };
    for (const h of [...s.hits].reverse()) { const hit = assocHit(h, proj, x, y); if (hit) return { h: hit, x, y }; }
    return { h: null, x, y };
  };
  const pointer = useScenePointer({
    onPan: (dx, dy) => { st.current = panScene(st.current, dx, dy); setTick(t => t + 1); },
    onRotate: (dx, dy) => { st.current.yaw += dx * 0.006; st.current.pitch = Math.max(0.05, Math.min(1.4, st.current.pitch + dy * 0.005)); setTick(t => t + 1); },
    onHover: e => { const hit = pickAt(e); setTip(hit.h ? { ...hit, left: Math.max(4, Math.min(hit.x + 12, st.current.W - 280)), top: Math.min(hit.y + 12, st.current.H - 60) } : null); }, onLeave: () => setTip(null),
    onPick: e => { const { h, x, y } = pickAt(e); onPick(h, x, y); },
  });
  return <div className="aw-canvas">
    <canvas ref={cv} data-tick={tick} aria-label={`LLM 全部 ${NL - 1} 层；当前第 ${layer} 层的 Head、HiddenState 与 MLP 原生单元；可用精确定位选择`} {...pointer}/>
    {tip && <div className="aw-tip" style={{ left: tip.left, top: tip.top }}>{dims.describe(tip.h)}</div>}
  </div>;
}
