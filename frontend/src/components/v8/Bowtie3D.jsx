/* 3D「前后锥」：一个 MLP 单元的上下游一致性的三维视图（Goodfire 口径）。
   判据与几何见 server/neuron_bowtie_service.py：
     读方向 r̂（gate 行，它检测什么）、写方向 ŵ（down 列，它写出什么）张成一个精确的二维平面，
     写轴在该平面内的坐标就是 (cosθ, sinθ)，θ = 自洽角。
     前一层写入者的写出方向、后一层读取者的读方向都投影到同一平面：
       上游落在读轴 (1,0) 附近  ⇔ 它写的正是目标读的
       下游落在写轴 (cosθ,sinθ) 附近 ⇔ 它读的正是目标写的
     所以「前后是否一致」= 这个蝴蝶结是否收拢：三块层板（前一层 / 本层 / 后一层）上节点的横向偏离。
   层板只是排列，平面坐标才是数据；点击任一节点沿电路走过去（换目标单元重算）。
   零依赖 canvas 投影：拖动旋转、滚轮缩放。 */
import { useEffect, useMemo, useRef, useState } from 'react';

import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { TYPE_COLOR } from './v8Util.js';

const DEG = Math.PI / 180;
const UP_C = '#d85a30';      // 上游 / 读轴
const DOWN_C = '#378add';    // 下游 / 写轴
const TGT_C = '#e34948';
const LAYER_GAP = 1.3;       // 层板间距（沿 y）
const HALF = 1.05;           // 层板半宽（u、v 方向）

const proj = (view, w, h, s) => (x, y, z) => {
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
  const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  const x1 = x * cy - z * sy;
  const z1 = x * sy + z * cy;
  return [w / 2 + x1 * s, h / 2 - (y * cp - z1 * sp) * s];
};

function quad(ctx, P, y) {
  const pts = [P(-HALF, y, -HALF), P(HALF, y, -HALF), P(HALF, y, HALF), P(-HALF, y, HALF)];
  ctx.beginPath();
  pts.forEach(([a, b], k) => (k ? ctx.lineTo(a, b) : ctx.moveTo(a, b)));
  ctx.closePath();
  return pts;
}

function arrow(ctx, a, b, color) {
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
  const ang = Math.atan2(b[1] - a[1], b[0] - a[0]), ah = 8;
  ctx.beginPath();
  ctx.moveTo(b[0], b[1]);
  ctx.lineTo(b[0] - ah * Math.cos(ang - 0.4), b[1] - ah * Math.sin(ang - 0.4));
  ctx.lineTo(b[0] - ah * Math.cos(ang + 0.4), b[1] - ah * Math.sin(ang + 0.4));
  ctx.closePath(); ctx.fillStyle = color; ctx.fill();
}

const label = (tokens, n = 2) => (tokens || []).slice(0, n).map(t => String(t).trim() || '·').join('·');

/* 画布下按编号列出两侧节点：3D 里只画徽标，文字放这里，避免标签互相压住。 */
function SideList({ title, color, nodes, axis, unit, onPick }) {
  return (
    <div className="v8-bowtie-side">
      <h4><i style={{ background: color }} />{title} <small>应对齐：{axis}</small></h4>
      {nodes.length ? nodes.map((n, k) => (
        <button key={`${n.block}-${n.unit}`} type="button" className="v8-bowtie-row" onClick={() => onPick?.(n.block, n.unit)}
                title={`B${n.block}·U${n.unit} 的读出：${(n.tokens || []).join(' / ')}${n.census_ro ? ` · 普查首位读出 ${n.census_ro}` : ''}`}>
          <b style={{ borderColor: color }}>{k + 1}</b>
          <span>B{n.block}·U{n.unit}</span>
          <em>{label(n.tokens, 2)}</em>
          <small>强度 {Number(n.score).toFixed(2)} · 对齐 {Number(n.align).toFixed(2)} · 偏离 {Number(n.off).toFixed(2)}</small>
        </button>
      )) : <p className="v8-note">{unit === 0 ? '更前面没有层。' : '这一侧没有正分的写入者 / 读取者。'}</p>}
    </div>
  );
}

export default function Bowtie3D({ census, run, block, unit, onPick, height = 460 }) {
  const api = useApi(block != null && unit != null
    ? `/api/neurons/bowtie?block=${block}&unit=${unit}&k=6&model_key=${encodeURIComponent(census || '')}&run=${encodeURIComponent(run || '')}`
    : null);
  const d = api.data && api.data.available ? api.data : null;
  const cv = useRef(null);
  const hits = useRef([]);
  const drag = useRef(null);
  const [view, setView] = useState({ yaw: -28 * DEG, pitch: 16 * DEG, zoom: 1 });
  const [w, setW] = useState(560);
  const [hover, setHover] = useState(null);
  const H = height;

  useEffect(() => {
    const el = cv.current?.parentElement;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(([e]) => setW(Math.max(320, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  /* 换单元时不重置视角：沿电路走下去要能对比同一个角度下的形状 */
  const geom = useMemo(() => {
    if (!d) return null;
    const th = d.target.angle_deg * DEG;
    // 上游：沿读轴 (1,0)；下游：沿写轴 (cosθ, sinθ)
    /* 径向按 r^0.6 展开：实测投影半径多在 0.1–0.4，直接用会挤成一团。
       只在径向放大，角度与谁更靠轴的关系不变；精确值见下方编号清单。 */
    const place = n => {
      const r = Math.hypot(n.u, n.v);
      const k = r < 1e-6 ? 0 : 0.95 * Math.pow(r, 0.6) / r;
      return { px: n.u * k, pz: n.v * k };
    };
    const up = d.upstream.map((n, k) => ({ ...n, ...place(n), side: 'up', no: k + 1, y: -LAYER_GAP }));
    const dn = d.downstream.map((n, k) => ({ ...n, ...place(n), side: 'dn', no: k + 1, y: LAYER_GAP }));
    return { th, up, dn, axis: { read: [1, 0], write: [Math.cos(th), Math.sin(th)] } };
  }, [d]);

  useEffect(() => {
    const c = cv.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = w * dpr; c.height = H * dpr;
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, H);
    if (!d || !geom) return;
    const s = Math.min(w * 0.40, H * 0.36) * view.zoom;
    const P = proj(view, w, H, s);
    const out = [];

    /* 三块层板：前一层（上）—本层（中）—后一层（下）。按投影高度从远到近画。 */
    const planes = [
      { y: -LAYER_GAP, text: `块 ${d.block - 1} · 写入者`, ink: UP_C },
      { y: 0, text: `块 ${d.block} · 目标 U${d.unit}`, ink: TGT_C },
      { y: LAYER_GAP, text: `块 ${d.block + 1} · 读取者`, ink: DOWN_C },
    ].map(p => ({ ...p, at: P(0, p.y, 0) })).sort((a, b) => a.at[1] - b.at[1]);
    for (const p of planes) {
      quad(ctx, P, p.y);
      ctx.fillStyle = 'rgba(241,245,249,0.82)'; ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = '#cbd5e1'; ctx.stroke();
      const lc = P(-HALF, p.y, -HALF);
      ctx.font = '600 10.5px ui-sans-serif, system-ui, sans-serif';
      ctx.fillStyle = p.ink; ctx.textAlign = 'left';
      ctx.fillText(p.text, lc[0] + 4, lc[1] - 5);
    }

    /* 锥：本层目标到两侧节点的扇形，收拢 = 前后一致 */
    const cone = (nodes, color) => {
      if (nodes.length < 2) return;
      const o = P(0, 0, 0);
      const pts = nodes.map(n => P(n.px, n.y, n.pz)).sort((a, b) => Math.atan2(a[1] - o[1], a[0] - o[0]) - Math.atan2(b[1] - o[1], b[0] - o[0]));
      ctx.beginPath(); ctx.moveTo(o[0], o[1]);
      pts.forEach(([a, b]) => ctx.lineTo(a, b));
      ctx.closePath(); ctx.fillStyle = color; ctx.globalAlpha = 0.10; ctx.fill(); ctx.globalAlpha = 1;
    };
    cone(geom.up, UP_C); cone(geom.dn, DOWN_C);

    /* 本层的读轴 / 写轴：它们就是自洽角的两个边 */
    const o = P(0, 0, 0);
    const rd = P(geom.axis.read[0] * 0.98, 0, geom.axis.read[1] * 0.98);
    const wr = P(geom.axis.write[0] * 0.98, 0, geom.axis.write[1] * 0.98);
    arrow(ctx, o, rd, UP_C);
    arrow(ctx, o, wr, DOWN_C);
    ctx.font = '600 10px ui-sans-serif, system-ui, sans-serif';
    ctx.fillStyle = UP_C; ctx.textAlign = 'center';
    ctx.fillText('读轴', rd[0], rd[1] - 7);
    ctx.fillStyle = DOWN_C;
    ctx.fillText('写轴', wr[0], wr[1] - 7);
    /* 自洽角圆弧 */
    ctx.beginPath();
    for (let k = 0; k <= 12; k += 1) {
      const a = (geom.th * k) / 12, [x, y] = P(Math.cos(a) * 0.34, 0, Math.sin(a) * 0.34);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);
    const [ax, ay] = P(Math.cos(geom.th / 2) * 0.42, 0, Math.sin(geom.th / 2) * 0.42);
    ctx.fillStyle = '#475569'; ctx.font = '600 10px ui-sans-serif, system-ui, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(`θ ${d.target.angle_deg}°`, ax, ay + 3);

    /* 节点：大小 = 权重路径强度，填充 = 普查分型，描边 = 哪一侧，描边粗细 = token 与目标标签的重叠 */
    const nodes = [...geom.dn, ...geom.up];
    const order = nodes.map(n => ({ n, p: P(n.px, n.y, n.pz) })).sort((a, b) => a.p[1] - b.p[1]);
    for (const { n, p } of order) {
      const [a, b] = [o[0], o[1]];
      ctx.beginPath(); ctx.moveTo(a, b);
      ctx.quadraticCurveTo((a + p[0]) / 2, (b + p[1]) / 2 - 10, p[0], p[1]);
      ctx.strokeStyle = n.side === 'up' ? UP_C : DOWN_C;
      ctx.globalAlpha = 0.42; ctx.lineWidth = 1.1;
      ctx.setLineDash(n.side === 'up' ? [] : [4, 3]);
      ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    for (const { n, p } of order) {
      const r = 5.5 + 8 * Math.min(1, Math.abs(Number(n.align) || 0));   // 大小 = 沿轴对齐，位置与大小同源，避免「强度大但位置偏」的自相矛盾
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      ctx.fillStyle = TYPE_COLOR[n.type] ?? '#64748b';
      ctx.globalAlpha = 0.92; ctx.fill(); ctx.globalAlpha = 1;
      ctx.lineWidth = 1.4 + 2.4 * (n.overlap || 0);
      ctx.strokeStyle = n.side === 'up' ? UP_C : DOWN_C;
      ctx.stroke();
      ctx.font = '600 9px ui-sans-serif, system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.lineWidth = 2.4; ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.strokeText(String(n.no), p[0], p[1] + 0.5);
      ctx.fillStyle = '#fff';
      ctx.fillText(String(n.no), p[0], p[1] + 0.5);
      ctx.textBaseline = 'alphabetic';
      out.push({ node: n, x: p[0], y: p[1], r: r + 4 });
    }
    /* 目标 */
    ctx.beginPath(); ctx.arc(o[0], o[1], 7.5, 0, Math.PI * 2);
    ctx.fillStyle = TGT_C; ctx.fill();
    ctx.beginPath(); ctx.arc(o[0], o[1], 12, 0, Math.PI * 2);
    ctx.strokeStyle = TGT_C; ctx.lineWidth = 1.6; ctx.stroke();
    ctx.font = '600 10.5px ui-sans-serif, system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = '#7f1d1d';
    ctx.fillText(`U${d.unit}`, o[0], o[1] - 18);
    hits.current = out;
  }, [d, geom, view, w, H]);

  useEffect(() => {
    const c = cv.current;
    const wheel = e => { e.preventDefault(); setView(v => ({ ...v, zoom: Math.max(0.5, Math.min(3.5, v.zoom * (e.deltaY < 0 ? 1.09 : 0.92))) })); };
    c?.addEventListener('wheel', wheel, { passive: false });
    return () => c?.removeEventListener('wheel', wheel);
  }, []);

  if (block == null || unit == null) return <Empty>先在左侧层塔或单元阵列里选一个 MLP 单元，这里显示它的前后锥。</Empty>;
  const retry = <button type="button" className="v8-retry" onClick={api.reload}>重新计算</button>;
  if (api.data && api.data.available === false) {
    return <Empty action={retry}>{api.data.reason || '前后锥暂不可用'}</Empty>;
  }
  if (!d) return <Empty action={api.error ? retry : null}>{api.error ? `读取前后锥失败：${api.error}` : '沿权重路径计算上下游（首次需加载本机模型）…'}</Empty>;

  const c = d.consistency;
  const pct = v => (v == null ? '—' : `${Math.round(v * 100)}%`);
  const locate = e => {
    const r = cv.current.getBoundingClientRect();
    const pt = [e.clientX - r.left, e.clientY - r.top];
    return { pt, hit: hits.current.find(h => Math.hypot(pt[0] - h.x, pt[1] - h.y) <= h.r)?.node || null };
  };
  return (
    <div className="v8-bowtie">
      <div className="v8-bowtie-hd">
        <div className="v8-bowtie-metrics">
          <span title="读方向（它检测什么）与写方向（它写出什么）的夹角。小 = 检测与写出同一个概念；大 = 像在做转换。">
            自洽角 <b>{d.target.angle_deg}°</b>
            <i className="v8-bowtie-bar"><em style={{ width: `${Math.min(100, (d.target.angle_deg / 90) * 100)}%` }} /></i>
          </span>
          <span title="上游写入者的写出方向沿读轴的平均投影（1 = 完全对齐）。">上游对齐 <b>{pct(c.in_align)}</b><small>标签重叠 {pct(c.in_overlap)}</small></span>
          <span title="下游读取者的读方向沿写轴的平均投影。">下游对齐 <b>{pct(c.out_align)}</b><small>标签重叠 {pct(c.out_overlap)}</small></span>
        </div>
        <p className="v8-bowtie-say">{c.verdict}</p>
        <p className="v8-bowtie-tags">
          读（gate 行）<b>{label(d.target.read.tokens, 3)}</b>
          · 写（down 列）<b>{label(d.target.write.tokens, 3)}</b>
          {d.target.census_ro ? <> · 普查首位读出 <b>{d.target.census_ro}</b></> : null}
          {d.target.type_name ? <> · {d.target.type_name}</> : null}
        </p>
      </div>
      <div className="v8-bowtie-canvas">
        <canvas ref={cv} style={{ width: w, height: H }} role="img"
          aria-label={`块 ${d.block} 单元 ${d.unit} 的上下游前后锥：自洽角 ${d.target.angle_deg} 度，上游 ${d.upstream.length} 个、下游 ${d.downstream.length} 个`}
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
          onMouseUp={e => {
            const dd = drag.current; drag.current = null;
            if (dd && !dd.moved) { const { hit } = locate(e); if (hit) onPick?.(hit.block, hit.unit); }
          }}
          onMouseLeave={() => { drag.current = null; setHover(null); }} />
        {hover && (
          <div className="v8-tip" style={{ left: Math.min(hover.x + 12, w - 230), top: hover.y + 10 }}>
            <b>B{hover.block}·U{hover.unit}</b>
            <small>{hover.side === 'up' ? '上游 · 它写出' : '下游 · 它读取'} {label(hover.tokens, 4)}</small>
            <small>路径强度 {Number(hover.score).toFixed(3)} · 沿轴对齐 {Number(hover.align).toFixed(3)} · 横向偏离 {Number(hover.off).toFixed(3)}</small>
            <small>与目标标签重叠 {pct(hover.overlap)}{hover.type_name ? ` · ${hover.type_name}` : ''}{hover.explained ? ` · ${hover.explained}` : ''}</small>
            <small>点一下 = 以它为新的目标继续走</small>
          </div>
        )}
        <div className="v8-bowtie-btns">
          <button type="button" className="fw-tbtn ghost" onClick={() => setView(v => ({ ...v, zoom: Math.min(3.5, v.zoom * 1.25) }))}>放大</button>
          <button type="button" className="fw-tbtn ghost" onClick={() => setView(v => ({ ...v, zoom: Math.max(0.5, v.zoom / 1.25) }))}>缩小</button>
          <button type="button" className="fw-tbtn ghost v8-reset" onClick={() => setView({ yaw: -28 * DEG, pitch: 16 * DEG, zoom: 1 })}>重置视角</button>
        </div>
      </div>
      <div className="v8-bowtie-lists">
        <SideList title="上游 · 前一层写入者" color={UP_C} nodes={d.upstream} axis="读轴" unit={d.unit} onPick={onPick} />
        <SideList title="下游 · 后一层读取者" color={DOWN_C} nodes={d.downstream} axis="写轴" unit={d.unit} onPick={onPick} />
      </div>
      <div className="v8-bowtie-lg">
        <span><i style={{ background: UP_C }} />读轴：它检测什么（上游节点应向这里收拢）</span>
        <span><i style={{ background: DOWN_C }} />写轴：它写出什么（下游节点应向这里收拢）</span>
        <span><i className="ring" />描边越粗 = 标签与目标重叠越多</span>
        <span>节点大小 = 沿轴对齐 |投影|（不是路径强度，强度含经注意力头的路径，见清单）；径向按 r^0.6 展开只为分辨；层板只是排列</span>
      </div>
      <p className="v8-note">
        读 = gate 行、写 = down 列；标签来自 logit lens（{`W_U(γ ⊙ d̂)`} 的 top-4），在中间层噪声很大、常出现碎 token，只作辅助，判断以几何（角度 / 对齐 / 偏离）为主。分数是权重路径强度。
        读得出 ≠ 控得住：这里显示的是电路假设，确认需要干预实验。
      </p>
    </div>
  );
}
