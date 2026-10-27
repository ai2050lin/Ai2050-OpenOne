/* 3D 层结构：一个模型的多层结构（层板按层序自下而上，hs[0] 在最下面），叠加所选语言项的逐层编码（路线 T）。
   层序与层数来自模型骨架（真实结构参数）；层板内的位置只是索引排布与通道分区，不代表模型几何。
   每个语言项占层板上的一条通道：条长 = 该层留出投影间隔 ÷ 该语言项各层最大值（各自归一化），
   实心 = 该层在未见词表上达到可读出判据，空心 = 未达到；圆点 = 峰值层。零依赖 canvas 投影，左键拖动旋转、滚轮缩放、悬停查看数值。 */
import { useEffect, useRef, useState } from 'react';

const DEG = Math.PI / 180;

function project(p, view, w, h) {
  const { yaw, pitch, zoom } = view;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = p[0] * cy - p[2] * sy, z1 = p[0] * sy + p[2] * cy;
  const y2 = p[1] * cp - z1 * sp, z2 = p[1] * sp + z1 * cp;
  const s = Math.min(w, h) * 0.25 * zoom;
  return [w / 2 + x1 * s, h / 2 - y2 * s, z2];
}

function poly(ctx, pts) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
}

function inside(pt, pts) {
  let c = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}

export default function LayerStack3D({ nLayers, rows, selectedLayer, onPickLayer }) {
  const wrap = useRef(null);
  const canvas = useRef(null);
  const hits = useRef([]);
  const drag = useRef(null);
  const [size, setSize] = useState({ w: 600, h: 520 });
  const [view, setView] = useState({ yaw: -28 * DEG, pitch: 14 * DEG, zoom: 1 });
  const [hover, setHover] = useState(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(([e]) => setSize({ w: Math.max(320, e.contentRect.width), h: Math.max(380, Math.min(640, e.contentRect.width * 0.8)) }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const cv = canvas.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = size.w * dpr; cv.height = size.h * dpr;
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);
    const n = nLayers || 0;
    const H = 3.6;                                         // 层堆叠总高度（世界坐标）
    const yOf = i => (n ? -H / 2 + (H * i) / n : 0);
    const P = (x, y, z) => project([x, y, z], view, size.w, size.h);
    const K = Math.max(1, rows.length);
    const D = 0.62;                                       // 层板半深度
    const lane = k => [-D + 0.04 + (2 * (D - 0.04) * k) / K + 0.02, -D + 0.04 + (2 * (D - 0.04) * (k + 1)) / K - 0.02];
    const norms = rows.map(r => Math.max(1e-9, ...r.sep.filter(v => v != null)));
    const out = [];
    for (let i = 0; i <= n; i += 1) {
      const y = yOf(i);
      const corners = [P(-1, y, -D), P(1, y, -D), P(1, y, D), P(-1, y, D)];
      const sel = i === selectedLayer;
      poly(ctx, corners);
      ctx.fillStyle = sel ? 'rgba(2,132,199,0.10)' : i === 0 ? 'rgba(148,163,184,0.14)' : 'rgba(241,245,249,0.55)';
      ctx.fill();
      ctx.lineWidth = sel ? 1.6 : 0.8;
      ctx.strokeStyle = sel ? '#0284c7' : '#cbd5e1';
      ctx.stroke();
      out.push({ kind: 'layer', layer: i, pts: corners });
      rows.forEach((r, k) => {
        const idx = r.layers.indexOf(i);
        if (idx < 0) return;
        const v = r.sep[idx];
        if (v == null) return;
        const [z0, z1] = lane(k);
        const len = 1.9 * Math.max(0, v) / norms[k];
        const quad = [P(-0.95, y, z0), P(-0.95 + len, y, z0), P(-0.95 + len, y, z1), P(-0.95, y, z1)];
        poly(ctx, quad);
        if (r.readable[idx]) {
          ctx.globalAlpha = 0.88; ctx.fillStyle = r.color; ctx.fill(); ctx.globalAlpha = 1;
        } else {
          ctx.lineWidth = 1; ctx.strokeStyle = r.color; ctx.stroke();
        }
        out.push({ kind: 'bar', layer: i, row: r, value: v, readable: r.readable[idx], pts: quad });
        if (r.peak && r.peak.layer === i) {
          const [px, py] = P(-0.95 + len, y + 0.02, (z0 + z1) / 2);
          ctx.beginPath(); ctx.arc(px, py, 4.5, 0, Math.PI * 2);
          ctx.fillStyle = r.color; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#ffffff'; ctx.stroke();
        }
      });
      if (i === 0 || i === n || i % 4 === 0 || sel) {
        const lc = [[-1, -D], [1, -D], [1, D], [-1, D]].map(([x, z]) => P(x, y, z)).sort((a, b) => a[0] - b[0])[0];
        const [lx, ly] = [lc[0] - 6, lc[1]];
        ctx.font = `${sel ? 600 : 400} 10.5px ui-sans-serif, system-ui, sans-serif`;
        ctx.fillStyle = sel ? '#0284c7' : '#475569';
        ctx.textAlign = 'right';
        ctx.fillText(i === 0 ? 'hs[0] 嵌入' : `L${i}`, lx, ly + 3);
      }
    }
    const top = [[-1, -D], [1, -D], [1, D], [-1, D]].map(([x, z]) => P(x, yOf(n) + 0.25, z)).sort((a, b) => a[0] - b[0])[0];
    const [ax, ay] = [top[0] - 6, top[1]];
    ctx.font = '10.5px ui-sans-serif, system-ui, sans-serif';
    ctx.fillStyle = '#94a3b8'; ctx.textAlign = 'right';
    ctx.fillText('层序 ↑', ax, ay);
    hits.current = out.reverse();                        // 后画的在上面：命中检测从上往下
  }, [nLayers, rows, view, size, selectedLayer]);

  const locate = e => {
    const r = canvas.current.getBoundingClientRect();
    const pt = [e.clientX - r.left, e.clientY - r.top];
    return { pt, hit: hits.current.find(h => h.kind === 'bar' && inside(pt, h.pts)) || hits.current.find(h => inside(pt, h.pts)) };
  };
  const onDown = e => { drag.current = { x: e.clientX, y: e.clientY, view, moved: false }; };
  const onMove = e => {
    if (drag.current) {
      const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true;
      setView({ ...drag.current.view, yaw: drag.current.view.yaw + dx * 0.008,
                pitch: Math.max(-80 * DEG, Math.min(80 * DEG, drag.current.view.pitch + dy * 0.008)) });
      return;
    }
    const { pt, hit } = locate(e);
    setHover(hit ? { ...hit, x: pt[0], y: pt[1] } : null);
  };
  const onUp = e => {
    const d = drag.current;
    drag.current = null;
    if (d && !d.moved) {
      const { hit } = locate(e);
      if (hit) onPickLayer?.(hit.layer);
    }
  };
  const onWheel = e => { e.preventDefault(); setView(v => ({ ...v, zoom: Math.max(0.5, Math.min(3, v.zoom * (e.deltaY < 0 ? 1.1 : 0.91))) })); };
  useEffect(() => {
    const cv = canvas.current;
    if (!cv) return undefined;
    cv.addEventListener('wheel', onWheel, { passive: false });
    return () => cv.removeEventListener('wheel', onWheel);
  }, []);

  return (
    <div className="v7-stack" ref={wrap}>
      <canvas ref={canvas} style={{ width: size.w, height: size.h }} role="img"
              aria-label={`${nLayers} 层模型结构，叠加 ${rows.length} 个语言项的逐层编码；数值见下方热力表`}
              onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={() => { drag.current = null; setHover(null); }} />
      {hover && (
        <div className="v7-tip" style={{ left: Math.min(hover.x + 14, size.w - 220), top: hover.y + 12 }}>
          {hover.kind === 'bar' ? (
            <>
              <b>{hover.row.name}</b> · {hover.layer === 0 ? 'hs[0] 嵌入层' : `L${hover.layer}`}
              <div>留出投影间隔 <span className="mono">{hover.value.toFixed(3)}</span></div>
              <div>{hover.readable ? '● 未见词表上可读出' : '○ 未达到可读出判据'}</div>
              <small>{hover.row.family_id}.{hover.row.factor} · {hover.row.readout} 读出位置</small>
            </>
          ) : <>{hover.layer === 0 ? 'hs[0] 嵌入层' : `第 ${hover.layer} 层之后的残差流`}<small>点击选中该层</small></>}
        </div>
      )}
      <button type="button" className="fw-tbtn ghost v7-reset" onClick={() => setView({ yaw: -28 * DEG, pitch: 14 * DEG, zoom: 1 })}>重置视角</button>
    </div>
  );
}
