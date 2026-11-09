/* 关联机制 · 全部层的 3D 空间与 HiddenState 流动画（MA-H2 真实汇总，Phase 332）。
   层板 = 词嵌入 hs[0] + 36 块；每块左半是 32 个注意力头（颜色 = 答案位置对记录颜色 token 的注意力），右半是 MLP（本案例未分析，灰显）。
   中线 = HiddenState；彩线 = 各因素在“问题中的目标名”位置的差异幅度（std_rms），离中线越远越大。
   都是 discovery 上的候选观察：注意力去向不等于计算使用，逐层读出是近似。 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { panScene, useScenePointer } from '../scenePointer.js';

import './assoc_space.css';

const L = 37, NH = 32, PW = 1.0, PD = 0.55;
const LY = l => (l - 18) * 0.06;
const VIEWS = { iso: [-0.65, 0.42, '斜视'], side: [0, 0.05, '侧视'], top: [-0.4, 1.2, '俯视当前层'] };
const VORDER = ['iso', 'side', 'top'];
const SCOPES = [['doc', '按资料'], ['real', '按现实'], ['diff', '差值']];
const sign = v => `${v > 0 ? '+' : ''}${v.toFixed(1)}`;

function inPoly(x, y, poly) {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
  }
  return c;
}

/* data: { lensDoc[37], lensReal[37], heads: { doc[36][32], real[36][32] }, factors: [[key, label, color, values[37]]] } */
/* band：生命周期环节在本主张里有观察依据的层段 [lo, hi]（层板序号），高亮显示；claim / toolbar 由外层传入（Phase 340） */
export default function AssocFlow3D({ data, cur, onCur, facOn, selHead, live, band = null, claim = '', toolbar = null }) {
  const canvasRef = useRef(null);
  const viewRef = useRef({ yaw: -0.65, pitch: 0.42 });
  const platesRef = useRef([]);
  const sizeRef = useRef({ W: 0, H: 0 });
  const [scope, setScope] = useState('doc');
  const [view, setView] = useState('iso');
  const [card, setCard] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(8);
  const [tick, setTick] = useState(0);
  const hmax = useMemo(() => Math.max(...data.heads.doc.flat()), [data]);
  const ci = Math.round(cur);

  /* 播放：沿层上升，到顶停 */
  useEffect(() => {
    if (!playing) return undefined;
    let raf = 0, last = 0, pos = cur >= 36 ? 0 : cur;
    const step = t => {
      const dt = last ? (t - last) / 1000 : 0; last = t; pos = Math.min(36, pos + dt * speed); onCur(pos);
      if (pos >= 36) { setPlaying(false); return; }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const cv = canvasRef.current;
    const resize = () => { const r = cv.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
      sizeRef.current = { W: r.width, H: r.height }; cv.width = r.width * dpr; cv.height = r.height * dpr;
      cv.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0); setTick(t => t + 1); };
    resize();
    const ro = new ResizeObserver(resize); ro.observe(cv);
    return () => ro.disconnect();
  }, []);

  /* 绘制 */
  useEffect(() => {
    const cv = canvasRef.current, ctx = cv.getContext('2d'), { W, H } = sizeRef.current;
    if (!W) return;
    const { yaw, pitch, panX = 0, panY = 0 } = viewRef.current;
    const proj = (x, y, z) => {
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = x * cy - z * sy, z1 = x * sy + z * cy, y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      const s = Math.min(W * 0.95, H * 1.15) / 2.8, f = 7 / (7 + z2);
      return [W / 2 + panX + x1 * s * f, H * 0.5 + panY - y1 * s * f];
    };
    const quad = (pts, fill, stroke, lw = 1) => {
      ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
    };
    const headVal = (b, h) => (scope === 'doc' ? data.heads.doc[b][h] : scope === 'real' ? data.heads.real[b][h] : data.heads.doc[b][h] - data.heads.real[b][h]);
    const mix = (c, e) => `rgb(${203 + (c[0] - 203) * e | 0},${215 + (c[1] - 215) * e | 0},${232 + (c[2] - 232) * e | 0})`;  // 低值也有可见底色
    const colorFor = v => {
      if (scope === 'diff') { const t = Math.max(-1, Math.min(1, v / hmax)); return mix(t >= 0 ? [31, 95, 191] : [235, 104, 52], Math.abs(t) ** 0.5); }
      return mix([22, 72, 160], Math.max(0, Math.min(1, v / hmax)) ** 0.5);
    };
    const headXZ = (h, dx = 0, dz = 0) => [-PW + 0.04 + (h & 7) * 0.115 + dx, -PD + 0.05 + (h >> 3) * 0.25 + dz];
    ctx.clearRect(0, 0, W, H);
    const plates = [];
    for (let l = 0; l < L; l++) {
      const y = LY(l), isCur = l === ci, active = l <= cur + 0.001, inBand = band && l >= band[0] && l <= band[1];
      ctx.globalAlpha = view === 'top' ? (isCur ? 1 : 0.1) : isCur ? 1 : active ? 0.92 : 0.6;
      const c = [[-PW, -PD], [PW, -PD], [PW, PD], [-PW, PD]].map(([x, z]) => proj(x, y, z));
      plates.push({ l, poly: c });
      quad(c, inBand ? '#d6e4f7' : l === 0 ? '#dbe3ee' : '#eef2f7', isCur ? '#d29a00' : inBand ? '#2a5fae' : '#7b8aa3', isCur ? 2.5 : inBand ? 2 : 1.2);
      if (l === 0) { ctx.fillStyle = '#334155'; ctx.font = '12px sans-serif'; const p = proj(-PW, y, PD); ctx.fillText('词嵌入', p[0] - 44, p[1] + 4); }
      else {
        for (let h = 0; h < NH; h++) {
          const [x0, z0] = headXZ(h), hi = selHead && selHead.l === l - 1 && selHead.h === h;
          quad([[x0, z0], [x0 + 0.1, z0], [x0 + 0.1, z0 + 0.22], [x0, z0 + 0.22]].map(([a, b]) => proj(a, y, b)), colorFor(headVal(l - 1, h)), hi ? '#d29a00' : null, 2);
        }
        quad([[0.04, -PD + 0.05], [PW - 0.04, -PD + 0.05], [PW - 0.04, PD - 0.05], [0.04, PD - 0.05]].map(([a, b]) => proj(a, y, b)), '#b8c2d1', '#94a3b8', 1);
      }
      if (l % 6 === 0 || isCur) { ctx.fillStyle = isCur ? '#8a5a00' : '#475569'; ctx.font = `${isCur ? 'bold ' : ''}12px sans-serif`; const p = proj(PW, y, -PD); ctx.fillText(`L${l}`, p[0] + 6, p[1] + 4); }
    }
    platesRef.current = plates;
    ctx.globalAlpha = 1;
    if (view === 'top') {
      if (ci > 0) for (let h = 0; h < NH; h++) { const v = headVal(ci - 1, h); if (Math.abs(v) < 0.1) continue;
        const [x, z] = headXZ(h, 0.05, 0.12), q = proj(x, LY(ci), z);
        ctx.fillStyle = Math.abs(v) > 0.3 ? '#fff' : '#172033'; ctx.font = 'bold 10px sans-serif'; ctx.fillText(v.toFixed(2), q[0] - 11, q[1] + 3); }
      return;
    }
    const upto = Math.floor(cur), frac = cur - upto;
    const yAt = l => LY(Math.floor(l)) + (LY(Math.min(36, Math.floor(l) + 1)) - LY(Math.floor(l))) * (l % 1);
    const path = fn => { ctx.beginPath(); for (let l = 0; l <= upto; l++) { const [x, y] = fn(l); l ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      if (upto < 36) { const [x, y] = fn(upto + frac); ctx.lineTo(x, y); } ctx.stroke(); };
    data.factors.forEach(([key, , color, values], fi) => {
      if (!facOn[key]) return;
      const th = (fi / data.factors.length) * Math.PI * 2;
      const rad = l => { const a = values[Math.floor(l)], b = values[Math.min(36, Math.ceil(l))]; return 0.03 + 0.55 * (a * (1 - (l % 1)) + b * (l % 1)); };
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.globalAlpha = 0.9;
      path(l => proj(Math.cos(th) * rad(l), yAt(l) + 0.004, Math.sin(th) * rad(l) * 0.6));
    });
    ctx.globalAlpha = 1; ctx.strokeStyle = '#172033'; ctx.lineWidth = 3; path(l => proj(0, yAt(l), 0));
    /* 本次输入：每层改写幅度画成主干上的金色圆点（只画到当前层） */
    if (live?.curve) {
      const mx = Math.max(...live.curve.slice(1)) || 1;
      for (let l = 1; l <= Math.min(upto, live.curve.length - 1); l++) {
        const [x, y] = proj(0, LY(l), 0), r = 2 + 10 * (live.curve[l] / mx);
        ctx.fillStyle = 'rgba(210,154,0,.55)'; ctx.strokeStyle = '#b07d00'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
    }
    const [px, py] = proj(0, yAt(cur), 0);
    const g = ctx.createRadialGradient(px, py, 0, px, py, 18); g.addColorStop(0, 'rgba(210,154,0,.9)'); g.addColorStop(1, 'rgba(210,154,0,0)');
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(px, py, 18, 0, Math.PI * 2); ctx.fill();
    if (ci > 0) {
      data.heads.doc[ci - 1].map((_, h) => ({ h, v: headVal(ci - 1, h) })).sort((a, b) => b.v - a.v).slice(0, 3).forEach(({ h, v }) => {
        if (Math.abs(v) < 0.12) return;
        const [x, z] = headXZ(h, 0.05, 0.11), [hx, hy] = proj(x, LY(ci), z);
        ctx.strokeStyle = 'rgba(176,125,0,.85)'; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(px, py); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = '#8a5010'; ctx.font = '11px sans-serif'; ctx.fillText(`L${ci - 1}H${h}`, hx - 18, hy - 6);
      });
    }
  }, [cur, ci, scope, view, facOn, selHead, tick, data, hmax, live, band]);

  const pointer = useScenePointer({
    onPan: (dx, dy) => { viewRef.current = panScene(viewRef.current, dx, dy); setTick(t => t + 1); },
    onRotate: (dx, dy) => {
      const v = viewRef.current;
      viewRef.current = { ...v, yaw: v.yaw + dx * 0.008, pitch: Math.max(-0.2, Math.min(1.45, v.pitch + dy * 0.006)) };
      setTick(t => t + 1);
    },
    onPick: e => {
      const r = e.currentTarget.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, plates = platesRef.current;
      for (let i = plates.length - 1; i >= 0; i--) if (inPoly(x, y, plates[i].poly)) { setPlaying(false); onCur(plates[i].l); setCard(true); break; }
    },
  });
  const onKey = e => {
    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') { e.preventDefault(); setPlaying(false); onCur(Math.min(36, ci + 1)); }
    if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') { e.preventDefault(); setPlaying(false); onCur(Math.max(0, ci - 1)); }
    if (e.key === 'Enter') setCard(true);
  };

  const meter = (label, value, color) => { const v = Math.max(-14, Math.min(14, value)), w = (Math.abs(v) / 14) * 50;
    return <div className="af-meter"><span>{label}</span><i><b style={{ left: `${v >= 0 ? 50 : 50 - w}%`, width: `${w}%`, background: color }} /></i><span>{sign(value)}</span></div>; };
  const topHeads = ci > 0 ? data.heads.doc[ci - 1].map((d, h) => ({ h, d })).sort((a, b) => b.d - a.d).slice(0, 3) : [];

  return <section className="af-stage" aria-label="全部层的 3D 空间">
    <div className="af-head">
      <div><h3>一段文字怎样决定下一个词</h3><span className="v8-note">看 HiddenState 从词嵌入往上流过 36 层 · MA-H2 真实汇总（Qwen3-4B bf16，{data.n.toLocaleString()} 条）{claim ? ` · 主张 ${claim}` : ''}{live ? ' · 金色圆点为本次输入' : ''}</span></div>
      <div className="af-seg" role="group" aria-label="问法范围">{SCOPES.map(([id, label]) =>
        <button type="button" key={id} aria-pressed={scope === id} onClick={() => setScope(id)}>{label}</button>)}</div>
      <button type="button" className="fw-tbtn" onClick={() => { const next = VORDER[(VORDER.indexOf(view) + 1) % 3]; viewRef.current = { yaw: VIEWS[next][0], pitch: VIEWS[next][1] }; setView(next); }}>视角：{VIEWS[view][2]}</button>
    </div>
    {toolbar}
    <div className="af-body">
      <canvas ref={canvasRef} tabIndex={0} {...pointer} onKeyDown={onKey}
        aria-label={`37 块层板自下而上排列，当前第 ${ci} 层。方向键换层，回车看详情。`} />
      {!card && <div className="af-overlay" aria-live="polite"><b>{ci === 0 ? '词嵌入' : `第 ${ci} 层`}</b>
        <span className="v8-note">按资料 {sign(data.lensDoc[ci])} · 按现实 {sign(data.lensReal[ci])}</span></div>}
      {!card && <div className="af-hint v8-note">点任一层板看详情 · 左键拖动平移 · 右键拖动旋转</div>}
      {card && <aside className="af-card" aria-label="当前层详情">
        <button type="button" className="af-x" aria-label="关闭详情" onClick={() => setCard(false)}>×</button>
        <div><b className="af-big">{ci === 0 ? '词嵌入' : `第 ${ci} 层`}</b><div className="v8-note">{ci === 0 ? '只有词身份，作基线' : `第 ${ci - 1} 块输出 · hs[${ci}]`}</div></div>
        <div><h4>逐层读出：资料色 − 常识色</h4>{meter('按资料', data.lensDoc[ci], '#2a78d6')}{meter('按现实', data.lensReal[ci], '#eb6834')}</div>
        <div><h4>各因素的差异（问题中的目标名处）</h4>{data.factors.map(([key, label, color, values]) =>
          <div className={`af-row${facOn[key] ? '' : ' off'}`} key={key}><s style={{ background: color }} /><span>{label}</span><span>{values[ci].toFixed(2)}</span></div>)}</div>
        <div><h4>最看资料颜色的头</h4>{ci === 0 ? <span className="v8-note">词嵌入层没有注意力头</span>
          : topHeads.map(x => <div className="af-row" key={x.h}><s style={{ background: '#245ca8' }} /><span>L{ci - 1}H{x.h}</span><span>{x.d.toFixed(2)}</span></div>)}</div>
      </aside>}
    </div>
    <div className="af-player">
      <button type="button" className="af-play" onClick={() => setPlaying(p => !p)}>{playing ? '❚❚ 暂停' : '▶ 播放'}</button>
      <input type="range" min="0" max="36" step="0.01" value={cur} aria-label="层位" onChange={e => { setPlaying(false); onCur(Number(e.target.value)); }} />
      <select aria-label="播放速度" value={speed} onChange={e => setSpeed(Number(e.target.value))}><option value={4}>慢</option><option value={8}>中</option><option value={16}>快</option></select>
    </div>
    <div className="af-legend v8-note">
      <span><i style={{ background: 'linear-gradient(90deg,#cbd7e8,#6f95cf,#1648a0)' }} />头越蓝，越看资料里的颜色</span>
      <span><i style={{ background: '#b8c2d1' }} />MLP（本案例未分析）</span>
      {band && <span><i style={{ background: '#d6e4f7', border: '1px solid #2a5fae' }} />所选环节有观察依据的层段</span>}
      <span><i style={{ background: '#172033', height: 3 }} />HiddenState</span>
      {live?.curve && <span><i style={{ background: '#d29a00', borderRadius: '50%', width: 10, height: 10 }} />本次输入每层改写幅度</span>}
      <span><i style={{ background: 'linear-gradient(90deg,#7050a0,#087b70,#b07d00,#c0392b,#4a7fd0)', height: 3 }} />各因素（离中线越远差异越大）</span>
    </div>
  </section>;
}
