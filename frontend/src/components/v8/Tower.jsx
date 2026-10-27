/* 3D 层塔：一个模型的全部层。层板按 HiddenState 序号排列：最下面 hs[0] = 嵌入层，块 b 的层板 = 该块的输出 hs[b+1]，
   最上面是输出头（只作标识，不可选）。层数来自模型骨架。
   各 tab 自己决定显示什么（研究工作台 · 空间的 SpaceLens 按 tab 组装）：
     values  按 hs 序号的叠加量（长度 nLayers + 1，null = 无数据），颜色单一色相由浅到深、按最大值归一化；
     marks   按 hs 序号的边缘标记（如「未见词表上可读出」），peak 为峰值层；
     onPick  点击某层（参数为 hs 序号；allowEmbedding 为 false 时嵌入层不可选）。点击从不切换 tab。
   零依赖 canvas 投影：左键拖动旋转、滚轮缩放。层板只是排列，不代表模型几何。 */
import { useEffect, useRef, useState } from 'react';

import { hsLabel, seq } from './v8Util.js';

const DEG = Math.PI / 180;
const MARK = '#1baf7a';

function inside(pt, pts) {
  let c = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}

export default function Tower({ nLayers, selected, onPick, values, marks, peak, valueLabel = '叠加量', markLabel = '', allowEmbedding = true, caption }) {
  const cv = useRef(null), hits = useRef([]), drag = useRef(null);
  const [view, setView] = useState({ yaw: -30 * DEG, pitch: 16 * DEG, zoom: 1 });
  const [hover, setHover] = useState(null);
  const [w, setW] = useState(340);
  const H = 600;
  useEffect(() => {
    const el = cv.current?.parentElement;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const c = cv.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = w * dpr; c.height = H * dpr;
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, H);
    const n = nLayers + 2, span = 4.2, s = Math.min(w * 0.3, H * 0.2) * view.zoom;
    const yOf = i => -span / 2 + (span * i) / (n - 1);
    const P = (x, y, z) => {
      const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw), cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
      const x1 = x * cy - z * sy, z1 = x * sy + z * cy;
      return [w / 2 + x1 * s, H / 2 - (y * cp - z1 * sp) * s];
    };
    const nums = (values || []).filter(v => v != null);
    const max = nums.length ? Math.max(1e-9, ...nums) : 1;
    const markSet = new Set(marks || []);
    const out = [];
    for (let i = 0; i < n; i += 1) {
      const y = yOf(i), D = 0.6, isHead = i === n - 1, hs = isHead ? null : i;
      const pts = [P(-1, y, -D), P(1, y, -D), P(1, y, D), P(-1, y, D)];
      ctx.beginPath(); pts.forEach(([a, c2], k) => (k ? ctx.lineTo(a, c2) : ctx.moveTo(a, c2))); ctx.closePath();
      const v = hs != null && values ? values[hs] : null, sel = hs != null && hs === selected;
      ctx.fillStyle = isHead ? 'rgba(148,163,184,0.28)' : v == null ? (hs === 0 ? 'rgba(148,163,184,0.22)' : 'rgba(241,245,249,0.9)') : seq(0.06 + (0.94 * v) / max);
      ctx.globalAlpha = 0.93; ctx.fill(); ctx.globalAlpha = 1;
      ctx.lineWidth = sel ? 2.4 : 0.8; ctx.strokeStyle = sel ? '#e34948' : '#94a3b8'; ctx.stroke();
      if (hs != null && markSet.has(hs)) {                  // 边缘标记：层板右前沿一段粗线
        const [a, b] = [P(0.55, y, D), P(1, y, D)];
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
        ctx.lineWidth = hs === peak ? 5 : 3.2; ctx.strokeStyle = MARK; ctx.stroke();
      }
      const clickable = hs != null && (hs > 0 || allowEmbedding);
      out.push({ hs: clickable ? hs : null, pts, v, label: isHead ? '输出头（unembedding）' : hsLabel(hs), mark: hs != null && markSet.has(hs), peak: hs === peak });
      if (isHead || hs === 0 || (hs - 1) % 5 === 0 || sel || hs === peak) {
        const lc = pts.slice().sort((a2, c2) => a2[0] - c2[0])[0];
        ctx.font = `${sel || hs === peak ? 600 : 400} 10.5px ui-sans-serif, system-ui, sans-serif`;
        ctx.fillStyle = sel ? '#e34948' : hs === peak ? MARK : '#475569'; ctx.textAlign = 'right';
        ctx.fillText(isHead ? '输出头' : hs === 0 ? '嵌入' : `块 ${hs - 1}${hs === peak ? ' 峰' : ''}`, lc[0] - 6, lc[1] + 3);
      }
    }
    hits.current = out.reverse();
  }, [nLayers, selected, values, marks, peak, allowEmbedding, view, w]);
  const locate = e => {
    const r = cv.current.getBoundingClientRect();
    const pt = [e.clientX - r.left, e.clientY - r.top];
    return { pt, hit: hits.current.find(h => inside(pt, h.pts)) };
  };
  useEffect(() => {
    const c = cv.current;
    const wheel = e => { e.preventDefault(); setView(v => ({ ...v, zoom: Math.max(0.5, Math.min(2.5, v.zoom * (e.deltaY < 0 ? 1.1 : 0.91))) })); };
    c?.addEventListener('wheel', wheel, { passive: false });
    return () => c?.removeEventListener('wheel', wheel);
  }, []);
  return (
    <div className="v8-tower-canvas">
      <canvas ref={cv} style={{ width: w, height: H }} role="img" aria-label={caption || `${nLayers} 个块的层塔`}
        onMouseDown={e => { drag.current = { x: e.clientX, y: e.clientY, view, moved: false }; }}
        onMouseMove={e => {
          if (drag.current) {
            const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y;
            if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true;
            setView({ ...drag.current.view, yaw: drag.current.view.yaw + dx * 0.008, pitch: Math.max(-1.2, Math.min(1.2, drag.current.view.pitch + dy * 0.008)) });
            return;
          }
          const { pt, hit } = locate(e);
          setHover(hit ? { ...hit, x: pt[0], y: pt[1] } : null);
        }}
        onMouseUp={e => { const d = drag.current; drag.current = null; if (d && !d.moved) { const { hit } = locate(e); if (hit?.hs != null) onPick?.(hit.hs); } }}
        onMouseLeave={() => { drag.current = null; setHover(null); }} />
      {hover && (
        <div className="v8-tip" style={{ left: Math.min(hover.x + 12, w - 190), top: hover.y + 10 }}>
          {hover.label}
          {hover.v != null && <small>{valueLabel} {Number(hover.v).toLocaleString()}</small>}
          {hover.mark && markLabel && <small>● {markLabel}{hover.peak ? '（峰值层）' : ''}</small>}
        </div>
      )}
      <button type="button" className="fw-tbtn ghost v8-reset" onClick={() => setView({ yaw: -30 * DEG, pitch: 16 * DEG, zoom: 1 })}>重置视角</button>
    </div>
  );
}
