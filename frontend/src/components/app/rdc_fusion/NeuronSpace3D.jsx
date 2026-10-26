/* 单层神经元 3D 视图（layer_space.v1，GET /api/neurons/space）。
   属性空间：x = 因果作用 ΔL̂（symlog），y = 激活密度（log），z = 直接读出占比 —— 三个轴都是实测量。
   读出语义空间：读出向量（W_U(γ⊙W_down[:,j])）单位化后的 PCA 前三维，轴端标注该主方向最推高的 token；降维会失真。
   点 = 一个 MLP 神经元；颜色 = 类型；大小 = 因果作用排序；带环并标注的点 = 触发点消融已确认（L30-v3）。
   拖动旋转、滚轮缩放、悬停查看、点击打开单元卡。布局只是展示，距离不代表模型内部的计算关系。 */
import { useEffect, useMemo, useRef, useState } from 'react';

import { SPACE_TOPN, SPACE_TYPES, SPACE_TYPE_COLOR, fmtNum } from './neuronUtil.js';

export default function NeuronSpace3D({ data, view, selected, onSelect, height = 560 }) {
  const wrapRef = useRef(null);
  const cvRef = useRef(null);
  const cam = useRef({ yaw: 0.7, pitch: 0.35, zoom: 1, drag: null, moved: false });
  const proj = useRef([]);
  const [hover, setHover] = useState(null);
  const [tick, setTick] = useState(0);
  const [hidden, setHidden] = useState(() => new Set());
  const [topN, setTopN] = useState(0);
  const [labels, setLabels] = useState(true);

  const pts = useMemo(() => {
    const key = view === 'sem' ? 's' : 'p';
    const F = data.n_units;
    return data.units
      .filter(u => !hidden.has(u.t) && (!topN || u.rk <= topN))
      .map(u => ({ u, x: u[key][0] * 1.6, y: u[key][1] * 1.6, z: u[key][2] * 1.6,
        r: 1.4 + 4.6 * Math.pow(1 - (u.rk - 1) / F, 6), c: SPACE_TYPE_COLOR[u.t] || '#94a3b8',
        big: u.t === 'T1' || u.t === 'T2' || u.ok }));
  }, [data, view, hidden, topN]);
  const axes = view === 'sem' ? data.semantic_axes : data.property_axes;

  useEffect(() => {
    const cv = cvRef.current, wrap = wrapRef.current;
    if (!cv || !wrap) return;
    const dpr = window.devicePixelRatio || 1;
    const W = wrap.clientWidth, H = height;
    cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { yaw, pitch, zoom } = cam.current;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const scale = Math.min(W, H) * 0.26 * zoom;
    const P = (x, y, z) => {
      const x1 = x * cy - z * sy, z1 = x * sy + z * cy;
      const y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      const f = 6 / (6 + z2);
      return [W / 2 + x1 * scale * f, H / 2 - y1 * scale * f, z2];
    };
    ctx.fillStyle = '#fbfcfd'; ctx.fillRect(0, 0, W, H);
    // 轴：-1.6 … +1.6，轴端写名称（属性空间带刻度）
    ctx.font = '11px system-ui, sans-serif';
    axes.forEach((ax, i) => {
      const v = [0, 0, 0];
      v[i] = 1;
      const a = P(-1.6 * v[0], -1.6 * v[1], -1.6 * v[2]), b = P(1.6 * v[0], 1.6 * v[1], 1.6 * v[2]);
      ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      ctx.fillStyle = '#475569';
      const short = view === 'sem' ? `PC${i + 1} +：${(ax.pos_tokens || []).slice(0, 3).map(t => JSON.stringify(t)).join(' ')}` : ax.name.split('（')[0];
      ctx.fillText(`${'xyz'[i]} · ${short}`, b[0] + 4, b[1] - 2);
      if (view === 'sem') {
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(`−：${(ax.neg_tokens || []).slice(0, 3).map(t => JSON.stringify(t)).join(' ')}`, a[0] + 4, a[1] + 12);
      } else {
        (ax.ticks || []).forEach(tk => {
          const q = P(1.6 * v[0] * tk.pos, 1.6 * v[1] * tk.pos, 1.6 * v[2] * tk.pos);
          ctx.fillStyle = '#94a3b8'; ctx.fillRect(q[0] - 1.5, q[1] - 1.5, 3, 3);
          ctx.fillText(tk.label, q[0] + 3, q[1] + 11);
        });
      }
    });
    const out = pts.map(p => { const q = P(p.x, p.y, p.z); return { p, sx: q[0], sy: q[1], d: q[2] }; });
    out.sort((a, b) => b.d - a.d);
    for (const o of out) {
      if (o.p.big) continue;
      ctx.globalAlpha = 0.45; ctx.fillStyle = o.p.c;
      const r = o.p.r;
      ctx.fillRect(o.sx - r / 2, o.sy - r / 2, r, r);
    }
    ctx.globalAlpha = 1;
    for (const o of out) {
      if (!o.p.big) continue;
      ctx.beginPath(); ctx.arc(o.sx, o.sy, Math.max(2.6, o.p.r), 0, Math.PI * 2);
      ctx.fillStyle = o.p.c; ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = '#ffffff'; ctx.stroke();
      if (o.p.u.ok) {
        ctx.beginPath(); ctx.arc(o.sx, o.sy, Math.max(2.6, o.p.r) + 3, 0, Math.PI * 2);
        ctx.lineWidth = 1.6; ctx.strokeStyle = '#0f172a'; ctx.stroke();
        if (labels) {
          ctx.fillStyle = '#0f172a';
          ctx.fillText(`U${o.p.u.u} ${o.p.u.mt ?? ''}→${o.p.u.ro[0] ?? ''}`, o.sx + 7, o.sy - 6);
        }
      }
    }
    const sel = out.find(o => o.p.u.u === selected);
    if (sel) {
      ctx.beginPath(); ctx.arc(sel.sx, sel.sy, 9, 0, Math.PI * 2);
      ctx.lineWidth = 2.2; ctx.strokeStyle = '#c2410c'; ctx.stroke();
    }
    proj.current = out;
  }, [pts, axes, view, selected, labels, height, tick]);

  useEffect(() => {          // 滚轮缩放：需要非 passive 监听才能阻止页面滚动
    const cv = cvRef.current;
    if (!cv) return undefined;
    const wheel = e => {
      e.preventDefault();
      cam.current.zoom = Math.max(0.4, Math.min(8, cam.current.zoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
      setTick(t => t + 1);
    };
    cv.addEventListener('wheel', wheel, { passive: false });
    return () => cv.removeEventListener('wheel', wheel);
  }, []);

  useEffect(() => {
    const onResize = () => setTick(t => t + 1);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const nearest = (e) => {
    const rect = cvRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = null, bd = 49;
    for (const o of proj.current) {
      const d = (o.sx - mx) ** 2 + (o.sy - my) ** 2;
      if (d < bd || (best && d === bd && o.d < best.d)) { bd = d; best = o; }
    }
    return best ? { o: best, mx, my } : null;
  };
  const onDown = e => { cam.current.drag = [e.clientX, e.clientY]; cam.current.moved = false; };
  const onMove = e => {
    const c = cam.current;
    if (c.drag) {
      const dx = e.clientX - c.drag[0], dy = e.clientY - c.drag[1];
      if (Math.abs(dx) + Math.abs(dy) > 2) c.moved = true;
      c.yaw += dx * 0.008; c.pitch = Math.max(-1.4, Math.min(1.4, c.pitch + dy * 0.008));
      c.drag = [e.clientX, e.clientY];
      setTick(t => t + 1);
      return;
    }
    const h = nearest(e);
    setHover(h ? { u: h.o.p.u, x: h.mx, y: h.my } : null);
  };
  const onUp = e => {
    const c = cam.current;
    if (!c.moved) {
      const h = nearest(e);
      if (h) onSelect(h.o.p.u.u);
    }
    c.drag = null;
  };
  const counts = useMemo(() => Object.fromEntries(SPACE_TYPES.map(t => [t, data.units.filter(u => u.t === t).length])), [data]);
  const nOk = data.units.filter(u => u.ok).length;

  return (
    <div className="ns3-wrap">
      <div className="ns3-bar">
        <div className="ns3-legend" role="group" aria-label="神经元类型（点击显示 / 隐藏）">
          {SPACE_TYPES.map(t => (
            <button key={t} type="button" className={hidden.has(t) ? 'off' : ''} aria-pressed={!hidden.has(t)}
                    onClick={() => setHidden(h => { const n = new Set(h); if (n.has(t)) n.delete(t); else n.add(t); return n; })}>
              <i style={{ background: SPACE_TYPE_COLOR[t] }} />{t} {data.type_names[t]} <small>{counts[t]}</small>
            </button>
          ))}
          <span className="ns3-ok"><i />已确认（触发点消融） <small>{nOk}</small></span>
        </div>
        <div className="ns3-ctl">
          <select value={topN} onChange={e => setTopN(Number(e.target.value))} aria-label="按因果作用筛选">
            {SPACE_TOPN.map(o => <option key={o.v} value={o.v}>{o.label}</option>)}
          </select>
          <label className="nu-note"><input type="checkbox" checked={labels} onChange={e => setLabels(e.target.checked)} /> 标注已确认单元</label>
          <button type="button" className="ns3-reset" onClick={() => { cam.current = { yaw: 0.7, pitch: 0.35, zoom: 1, drag: null, moved: false }; setTick(t => t + 1); }}>重置视角</button>
        </div>
      </div>
      <div ref={wrapRef} className="ns3-canvas">
        <canvas ref={cvRef} onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp}
                onMouseLeave={() => { cam.current.drag = null; setHover(null); }}
                role="img" aria-label={`第 ${data.layer} 层 ${pts.length} 个神经元的 3D 散点`} />
        {hover && (
          <div className="ns3-tip" style={{ left: hover.x + 14, top: hover.y + 10 }}>
            <b>L{data.layer}.U{hover.u.u}</b> · {hover.u.t} {data.type_names[hover.u.t]}
            <div>推高：{hover.u.ro.map(t => JSON.stringify(t)).join(' ')}</div>
            <div>因果作用 ΔL̂ {fmtNum(hover.u.tot * 1e4)}×10⁻⁴（排序 #{hover.u.rk}）· 强激活率 {fmtNum(hover.u.sr * 100, 1)}% · 直接读出 {fmtNum(hover.u.ds, 2)}</div>
            {hover.u.ok && <div>触发点消融已确认：{hover.u.mt} → {hover.u.ro[0]}，损失 +{fmtNum(hover.u.dR, 3)} nats</div>}
          </div>
        )}
      </div>
      <div className="nu-note ns3-axes">
        {axes.map((ax, i) => (
          <div key={i}><b>{'xyz'[i]}</b> = {ax.name}{ax.neg ? `（左：${ax.neg}；右：${ax.pos}）` : ''}{ax.source ? ` · 来源 ${ax.source}` : ''}</div>
        ))}
        {view === 'sem' && <div>前三个主成分共解释 {fmtNum(100 * data.semantic_var_ratio_top10.slice(0, 3).reduce((a, b) => a + b, 0), 1)}% 的方差：读出方向几乎不成簇，大多数单元在这个投影里重叠。</div>}
        <div>{data.boundary} 显示 {pts.length} / {data.n_units} 个单元。</div>
      </div>
    </div>
  );
}
