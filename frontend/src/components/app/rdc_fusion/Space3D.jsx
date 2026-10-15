/* 3D 响应空间（analysis.v1 → coords / stack）。每个点 = 一条测试样本在某层目标位置的 HiddenState 投影。
   坐标轴：LDA（白化空间中该因子组质心的判别方向，discovery 拟合）或原始 PCA（discovery 前三主成分）；
   所有划分的样本都投影到同一组由 discovery 拟合的轴上，所以 lockbox / lockbox_frame 的点落在哪里，
   就是「未见材料能否被发现期的结构解释」的直观检查。层堆叠模式：每层 2D LDA，相邻层 Procrustes 对齐，z = 层序。
   零依赖 canvas 投影（与结构示意视图同一实现方式）；左键拖动旋转、滚轮缩放、悬停查看样本。 */
import { useEffect, useMemo, useRef, useState } from 'react';

export const CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
const OTHER = '#94a3b8';
const SPLIT_ORDER = ['discovery', 'confirm', 'lockbox', 'lockbox_frame'];
const SPLIT_LABEL = { discovery: 'discovery（发现）', confirm: 'confirm（留出）', lockbox: 'lockbox（未见词表）', lockbox_frame: 'lockbox_frame（未见句式）' };
const SHAPE = { discovery: 'dot', confirm: 'dot', lockbox: 'ring', lockbox_frame: 'diamond' };

function colorKeyOf(point, colorBy) {
  if (colorBy === 'split') return point.split;
  if (colorBy === 'correct') return point.correct === 1 ? '作答正确' : point.correct === 0 ? '作答错误' : '非作答项';
  return String(point.cond[colorBy.slice(7)]);
}

export function paletteFor(keys, colorBy) {
  if (colorBy === 'correct') return { 作答正确: '#0ca30c', 作答错误: '#d03b3b', 非作答项: OTHER };
  const ordered = colorBy === 'split' ? SPLIT_ORDER.filter(k => keys.includes(k)) : [...keys].sort();
  return Object.fromEntries(ordered.map((k, i) => [k, i < CATEGORICAL.length ? CATEGORICAL[i] : OTHER]));
}

function drawShape(ctx, shape, x, y, r, color) {
  ctx.beginPath();
  if (shape === 'diamond') {
    ctx.moveTo(x, y - r * 1.25); ctx.lineTo(x + r * 1.25, y); ctx.lineTo(x, y + r * 1.25); ctx.lineTo(x - r * 1.25, y); ctx.closePath();
    ctx.fillStyle = color; ctx.fill();
    ctx.lineWidth = 1.5; ctx.strokeStyle = '#ffffff'; ctx.stroke();
  } else if (shape === 'ring') {
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff'; ctx.fill();
    ctx.lineWidth = 2.2; ctx.strokeStyle = color; ctx.stroke();
  } else {
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
    ctx.lineWidth = 1.5; ctx.strokeStyle = '#ffffff'; ctx.stroke();
  }
}

export default function Space3D({ analysis, layerTag, method, colorBy, visibleSplits, height = 520 }) {
  const wrapRef = useRef(null);
  const cvRef = useRef(null);
  const cam = useRef({ yaw: 0.7, pitch: 0.35, zoom: 1, drag: null });
  const projected = useRef([]);
  const [hover, setHover] = useState(null);
  const [tick, setTick] = useState(0);

  const scene = useMemo(() => {
    const pts = analysis.points;
    const keys = [...new Set(pts.map(p => colorKeyOf(p, colorBy)))];
    const pal = paletteFor(keys, colorBy);
    if (method === 'stack') {
      const st = analysis.stack;
      if (!st) return null;
      const n = st.layers.length;
      const zOf = i => (n === 1 ? 0 : (i / (n - 1)) * 2 - 1) * 1.6;
      const items = [];
      st.layers.forEach((layer, li) => {
        layer.points.forEach((xy, k) => {
          const p = pts[k];
          if (!visibleSplits.has(p.split)) return;
          items.push({ x: xy[0], y: xy[1], z: zOf(li), p, color: pal[colorKeyOf(p, colorBy)] || OTHER, shape: SHAPE[p.split], layer: layer.tag });
        });
      });
      const trails = st.levels.map((lv, k) => ({
        label: lv, color: colorBy === `factor:${st.factor}` ? pal[lv] : '#334155',
        path: st.layers.map((layer, li) => [layer.centroids[k][0], layer.centroids[k][1], zOf(li)]),
      }));
      const planes = st.layers.map((layer, li) => ({ tag: layer.tag, z: zOf(li) }));
      return { items, trails, planes, pal, axes: ['LDA1', 'LDA2', '层序'], centroids: [] };
    }
    const coords = analysis.coords?.[layerTag]?.[method];
    if (!coords) return null;
    const items = [];
    coords.forEach((c, k) => {
      const p = pts[k];
      if (!visibleSplits.has(p.split)) return;
      items.push({ x: c[0], y: c[1], z: c[2], p, color: pal[colorKeyOf(p, colorBy)] || OTHER, shape: SHAPE[p.split] });
    });
    // discovery 质心（按当前着色键）
    const groups = {};
    coords.forEach((c, k) => {
      const p = pts[k];
      if (p.split !== 'discovery') return;
      const key = colorKeyOf(p, colorBy);
      (groups[key] = groups[key] || []).push(c);
    });
    const centroids = colorBy === 'correct' || colorBy === 'split' ? [] : Object.entries(groups).map(([key, cs]) => ({
      label: key, color: pal[key] || OTHER,
      x: cs.reduce((s, c) => s + c[0], 0) / cs.length, y: cs.reduce((s, c) => s + c[1], 0) / cs.length, z: cs.reduce((s, c) => s + c[2], 0) / cs.length,
    }));
    const axisNames = method === 'pca' ? ['PC1', 'PC2', 'PC3'] : [1, 2, 3].map(i => `LDA${i}（${method.slice(4)}）`);
    return { items, trails: [], planes: [], pal, axes: axisNames, centroids };
  }, [analysis, layerTag, method, colorBy, visibleSplits]);

  useEffect(() => {
    const cv = cvRef.current;
    const wrap = wrapRef.current;
    if (!cv || !wrap || !scene) return undefined;
    const dpr = window.devicePixelRatio || 1;
    const W = wrap.clientWidth;
    const H = height;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { yaw, pitch, zoom } = cam.current;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const scale = Math.min(W, H) * 0.26 * zoom;
    const proj = (x, y, z) => {
      const x1 = x * cy - z * sy;
      const z1 = x * sy + z * cy;
      const y1 = y * cp - z1 * sp;
      const z2 = y * sp + z1 * cp;
      const f = 6 / (6 + z2);
      return [W / 2 + x1 * scale * f, H / 2 - y1 * scale * f, z2, f];
    };
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, W, H);
    // 轴
    const axLen = 2.2;
    [[axLen, 0, 0], [0, axLen, 0], [0, 0, axLen]].forEach((v, i) => {
      const a = proj(-v[0], -v[1], -v[2]);
      const b = proj(v[0], v[1], v[2]);
      ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      ctx.fillStyle = '#64748b'; ctx.font = '11px system-ui, sans-serif';
      ctx.fillText(scene.axes[i], b[0] + 4, b[1] - 4);
    });
    // 层平面（堆叠模式）
    scene.planes.forEach(pl => {
      const c = [[-1.8, -1.8], [1.8, -1.8], [1.8, 1.8], [-1.8, 1.8]].map(([x, y]) => proj(x, y, pl.z));
      ctx.beginPath(); c.forEach((q, i) => (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]))); ctx.closePath();
      ctx.fillStyle = 'rgba(148,163,184,0.06)'; ctx.fill();
      ctx.strokeStyle = 'rgba(100,116,139,0.35)'; ctx.stroke();
      ctx.fillStyle = '#334155'; ctx.font = '11px system-ui, sans-serif';
      ctx.fillText(pl.tag, c[3][0] - 26, c[3][1]);
    });
    // 点（按深度排序，远处淡化）
    const drawn = scene.items.map(it => ({ it, q: proj(it.x, it.y, it.z) })).sort((a, b) => b.q[2] - a.q[2]);
    drawn.forEach(({ it, q }) => {
      ctx.globalAlpha = Math.max(0.35, Math.min(1, 0.55 + q[3] * 0.4));
      drawShape(ctx, it.shape, q[0], q[1], 4, it.color);
    });
    ctx.globalAlpha = 1;
    projected.current = drawn.map(({ it, q }) => ({ it, sx: q[0], sy: q[1] }));
    // 质心轨迹（堆叠模式）
    scene.trails.forEach(tr => {
      ctx.strokeStyle = tr.color; ctx.lineWidth = 2;
      ctx.beginPath();
      tr.path.forEach((p, i) => { const q = proj(p[0], p[1], p[2]); if (i) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); });
      ctx.stroke();
      const last = tr.path[tr.path.length - 1];
      const q = proj(last[0], last[1], last[2]);
      ctx.fillStyle = '#0f172a'; ctx.font = '600 11px system-ui, sans-serif';
      ctx.fillText(tr.label, q[0] + 7, q[1] - 6);
    });
    // 质心（单层模式）
    scene.centroids.forEach(c => {
      const q = proj(c.x, c.y, c.z);
      ctx.beginPath(); ctx.arc(q[0], q[1], 8, 0, Math.PI * 2);
      ctx.fillStyle = c.color; ctx.fill(); ctx.lineWidth = 2.5; ctx.strokeStyle = '#0f172a'; ctx.stroke();
      ctx.fillStyle = '#0f172a'; ctx.font = '600 11px system-ui, sans-serif';
      ctx.fillText(c.label, q[0] + 11, q[1] + 4);
    });
    return undefined;
  }, [scene, height, tick]);

  useEffect(() => {
    const onResize = () => setTick(t => t + 1);
    window.addEventListener('resize', onResize);
    const cv = cvRef.current;
    // 滚轮缩放需要非被动监听才能阻止页面滚动
    const onWheel = e => {
      e.preventDefault();
      cam.current.zoom = Math.max(0.3, Math.min(5, cam.current.zoom * (e.deltaY > 0 ? 0.9 : 1.1)));
      setTick(t => t + 1);
    };
    if (cv) cv.addEventListener('wheel', onWheel, { passive: false });
    return () => { window.removeEventListener('resize', onResize); if (cv) cv.removeEventListener('wheel', onWheel); };
  }, [scene]);

  if (!scene) return <div className="v6-empty">该层没有 3D 坐标（样本不足或该视图不可用）。</div>;

  const onDown = e => { cam.current.drag = { x: e.clientX, y: e.clientY, yaw: cam.current.yaw, pitch: cam.current.pitch }; };
  const onUp = () => { cam.current.drag = null; };
  const onMove = e => {
    const d = cam.current.drag;
    if (d) {
      cam.current.yaw = d.yaw + (e.clientX - d.x) * 0.01;
      cam.current.pitch = Math.max(-1.4, Math.min(1.4, d.pitch + (e.clientY - d.y) * 0.01));
      setTick(t => t + 1);
      return;
    }
    const box = cvRef.current.getBoundingClientRect();
    const mx = e.clientX - box.left, my = e.clientY - box.top;
    let best = null, bd = 100;
    for (const p of projected.current) {
      const dd = (p.sx - mx) ** 2 + (p.sy - my) ** 2;
      if (dd < bd) { bd = dd; best = p; }
    }
    setHover(best ? { x: mx, y: my, p: best.it.p, layer: best.it.layer, scene } : null);
  };

  return (
    <div className="v6-space3d" ref={wrapRef}>
      <canvas ref={cvRef} onMouseDown={onDown} onMouseUp={onUp} onMouseLeave={() => { onUp(); setHover(null); }}
        onMouseMove={onMove} role="img" aria-label="3D 响应空间：每个点是一条样本在该层的投影" />
      {hover && hover.scene === scene && (   /* 场景变化后旧的悬停坐标失效 */
        <div className="v6-tip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          {hover.p.text || hover.p.id}{'\n'}
          {Object.entries(hover.p.cond).map(([k, v]) => `${k}=${v}`).join(' · ')}{'\n'}
          {SPLIT_LABEL[hover.p.split] || hover.p.split} · 填充 {hover.p.filler}{hover.layer ? ` · ${hover.layer}` : ''}{'\n'}
          {hover.p.correct === 1 ? '模型作答正确' : hover.p.correct === 0 ? '模型作答错误' : '非作答项'} · {hover.p.n_tok} tokens
        </div>
      )}
      <div className="v6-space3d-legend">
        <div>
          {Object.entries(scene.pal).map(([k, c]) => <span key={k}><i style={{ background: c }} />{colorBy === 'split' ? (SPLIT_LABEL[k] || k) : k}</span>)}
        </div>
        <div>
          <span><i className="shape dot" />discovery / confirm</span>
          <span><i className="shape ring" />lockbox（未见词表）</span>
          <span><i className="shape diamond" />lockbox_frame（未见句式）</span>
          {scene.centroids.length > 0 && <span><i className="shape centroid" />discovery 质心</span>}
        </div>
      </div>
    </div>
  );
}
