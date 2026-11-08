/* 空间 · 关联机制工作台（Phase 346）。
   上：模型列表；中：3D 里的基础 LLM（每层 = 注意力 + MLP，中间是残差主干），打开插件后旁边出现单独的 HiddenState / MLP / 注意力模型，
   点格子看具体数值；可全屏。下：控制台——输入文本运行、token 条、操作面板（全屏时变成浮窗，同一个面板）、插件面板。
   原来的四个页签（运行与路径 / 响应对照 / 语言项实测 / 共享关系）是“分析视角”插件，面板里是原来的页面。
   数据：本机后端可读时，从 /api/traces/mainline/ma-h2/ 读 MA-H2 discovery 冻结样本的原生值（float16）；
   网页版与没有原生数据时，只用案例汇总（FLOW_DATA），需要原生数据的插件置灰并说明原因。 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { useApi } from '../app/rdc_fusion/api.js';
import { TestsPanel, TopTokens } from './AssocConsole.jsx';
import { liveSampler, runOnce, useLiveBase, useLiveModel, useLocalBase } from './liveRun.js';
import { request } from './MechanismTraceExplorer.jsx';
import { FLOW_DATA } from './data/assocFlow.js';
import { PageHead } from './CoreKit.jsx';
import { shortModel } from './v8Util.js';
import './assoc_workbench.css';

const POSN = { ans: '答案位置', qname: '问题中的实体名', scope: '范围词', tname: '资料里的实体名', tcol: '资料里的颜色', lastcol: '最后一条记录的颜色' };
const CN = { red: '红', yellow: '黄', green: '绿', blue: '蓝', white: '白', black: '黑', brown: '棕', orange: '橙', purple: '紫', pink: '粉', gray: '灰' };
const MODE_PLUGINS = [
  { id: 'live', name: '运行与路径', tag: 'have', desc: '案例汇总的逐层信息流、因素彩线与原控制台（原“运行与路径”页签）' },
  { id: 'response', name: '响应对照', tag: 'have', desc: '只改句子里的一处，看模型内部哪里跟着变（原页签）' },
  { id: 'langmeasure', name: '语言项实测', tag: 'have', desc: '一项语言能力在哪一层读得出（原页签）' },
  { id: 'relations', name: '共享关系', tag: 'have', desc: '不同语言能力是否共用同一个读出方向（原页签）' },
];
const PLUGINS = [
  ['分析视角', MODE_PLUGINS],
  ['观察', [
    { id: 'hs', name: 'HiddenState', tag: 'on', native: true, desc: '在 LLM 旁边显示单独的 HiddenState 模型（每层全部坐标），点格子看具体数值' },
    { id: 'mlp', name: 'MLP 神经元', tag: 'on', native: true, desc: '单独的 MLP 单元模型（每块全部中间单元）；MA-H2 只在答案位置存了 MLP' },
    { id: 'heads', name: '注意力头', tag: 'on', desc: '单独的注意力模型：头的颜色 = 答案位置对所选 token 的注意力；没有原生数据时用案例汇总' },
    { id: 'delta', name: '层间变化', tag: 'on', native: true, desc: '每层 HiddenState 的总变化 ‖Δh‖ 与方向变化；注意力 / MLP 拆分需要原生权重（待接入）' },
    { id: 'diff', name: '条件对照', tag: 'on', native: true, desc: '按资料与按现实两条运行的差值（先查对齐）' },
  ]],
  ['行业工具', [
    { id: 'lens', name: 'Logit Lens', tag: 'on', desc: '逐层读出：资料色与常识色的 logit（每层先做最终归一化再接输出头）' },
    { id: 'dla', name: '直接 Logit 归因', tag: 'todo', desc: '各头 / 单元写入在答案方向上的投影；需要本机权重' },
    { id: 'patch', name: '激活修补', tag: 'have', desc: 'Activation patching：项目 mech_bench exact' },
    { id: 'atp', name: '归因修补', tag: 'have', desc: 'AtP / AtP+IG：项目 mech_bench；atp_ig 在 Qwen3 上偏重前段（Phase 345）' },
    { id: 'tuned', name: 'Tuned Lens', tag: 'todo', desc: '每层训练一个仿射读出' },
    { id: 'attnviz', name: '注意力可视化', tag: 'todo', desc: '逐头的 token 对 token 注意力' },
    { id: 'sae', name: 'SAE 特征', tag: 'todo', desc: '稀疏自编码器特征与特征卡' },
    { id: 'tc', name: 'Transcoder 归因图', tag: 'have', desc: '特征到 logit 的归因图；项目 transcoder.v1（第 30 块）' },
    { id: 'probe', name: '线性探针', tag: 'have', desc: '逐层可读出检验；项目 analysis.v1' },
    { id: 'steer', name: '引导向量', tag: 'todo', desc: '在某层加方向，正负双向' },
  ]],
  ['检验', [{ id: 'tests', name: '三类检验', tag: 'have', desc: '响应预测 / 输出预测 / 原生机制检验（Phase 340）' }]],
];
const ALL = PLUGINS.flatMap(([, ps]) => ps);
const TAGN = { on: '可用', have: '项目已有', todo: '待接入' };
const MODE_IDS = new Set(MODE_PLUGINS.map(p => p.id));

/* ---------- 数据 ---------- */
function f16(o) {      // float16 小端 → Float32Array
  const bin = atob(o.b64), n = bin.length / 2, out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8), s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return { a: out, shape: o.shape };
}
/* 实时运行：一次取一个 token 的全部层（/api/traces/{id}/bundle），查询位置固定为最后一个 token */
function useBundle(base, id, position) {
  const [s, set] = useState({});
  useEffect(() => {
    if (!base || !id || position == null) return undefined;
    let dead = false;
    request(base, `/api/traces/${id}/bundle?position=${position}`).then(d => { if (!dead) set({ key: `${id}:${position}`, data: d }); })
      .catch(e => { if (!dead) set({ key: `${id}:${position}`, error: e.message }); });
    return () => { dead = true; };
  }, [base, id, position]);
  return s.key === `${id}:${position}` ? s : {};
}
function useSample(base, i) {
  const [s, set] = useState({});
  useEffect(() => {
    if (base == null || i == null) return undefined;
    let dead = false;
    request(base, `/api/traces/mainline/ma-h2/sample/${i}`).then(d => {
      if (dead) return;
      set({ i, data: { ...d, hs: f16(d.hs), mlp: f16(d.mlp), attn: f16(d.attn) } });
    }).catch(e => { if (!dead) set({ i, error: e.message }); });
    return () => { dead = true; };
  }, [base, i]);
  return s.i === i ? s : {};
}
const norm = v => { let s = 0; for (const x of v) s += x * x; return Math.sqrt(s); };
const cosv = (u, v) => { let a = 0, b = 0, c = 0; for (let i = 0; i < u.length; i++) { a += u[i] * v[i]; b += u[i] * u[i]; c += v[i] * v[i]; } return a / Math.sqrt(b * c); };
function pct(arrs, q) { const a = []; for (const v of arrs) for (const x of v) a.push(Math.abs(x)); a.sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(a.length * q))] || 1; }
const absMax = v => { let m = 0; for (const x of v) m = Math.max(m, Math.abs(x)); return m || 1; };
function divc(v, m) { const t = Math.max(-1, Math.min(1, v / (m || 1))), e = Math.abs(t) ** 0.6, c = t >= 0 ? [31, 95, 191] : [212, 98, 43]; return [255 + (c[0] - 255) * e, 255 + (c[1] - 255) * e, 255 + (c[2] - 255) * e]; }
function seqc(v, m) { const e = Math.max(0, Math.min(1, v / (m || 1))) ** 0.5; return [255 - 225 * e, 255 - 160 * e, 255 - 64 * e]; }
function tex(w, h, fill) { const c = document.createElement('canvas'); c.width = w; c.height = h; const x = c.getContext('2d'), img = x.createImageData(w, h); fill(img.data); x.putImageData(img, 0, 0); return c; }
const put = (d, i, rgb) => { d[i * 4] = rgb[0]; d[i * 4 + 1] = rgb[1]; d[i * 4 + 2] = rgb[2]; d[i * 4 + 3] = 255; };
const grid = n => { const w = Math.ceil(Math.sqrt(n * 1.6)); return [w, Math.ceil(n / w)]; };

/* 一条样本的取值器（层 l、位置 p） */
function sampler(d) {
  if (!d) return null;
  const [P, NL, DM] = d.hs.shape, [NB, NF] = d.mlp.shape, NH = d.attn.shape[1], T = d.attn.shape[2];
  return {
    P, NL, DM, NB, NF, NH, T, d,
    hs: (p, l) => d.hs.a.subarray((p * NL + l) * DM, (p * NL + l + 1) * DM),
    mlp: b => d.mlp.a.subarray(b * NF, (b + 1) * NF),
    att: (b, h) => d.attn.a.subarray((b * NH + h) * T, (b * NH + h + 1) * T),
  };
}

/* ---------- 3D ---------- */
const VIEWS = { iso: [-0.5, 0.45, '斜视'], side: [0, 0.05, '侧视'], top: [-0.25, 1.2, '俯视'] };
const VORDER = ['iso', 'side', 'top'];
function Scene({ dims, objs, layer, view, tex: TX, marks, onPick, overlays, live }) {
  const cv = useRef(null), st = useRef({ yaw: VIEWS[view][0], pitch: VIEWS[view][1], view, zoom: 1, hits: [], W: 0, H: 0, dpr: 1 });
  const [tick, setTick] = useState(0);
  const [tip, setTip] = useState(null);
  useEffect(() => {
    const el = cv.current, ro = new ResizeObserver(() => { const r = el.getBoundingClientRect(), s = st.current; s.dpr = window.devicePixelRatio || 1; s.W = r.width; s.H = r.height; el.width = r.width * s.dpr; el.height = r.height * s.dpr; setTick(t => t + 1); });
    ro.observe(el);
    const wheel = e => { e.preventDefault(); st.current.zoom = Math.max(0.5, Math.min(3, st.current.zoom * (e.deltaY < 0 ? 1.1 : 0.9))); setTick(t => t + 1); };
    el.addEventListener('wheel', wheel, { passive: false });
    return () => { ro.disconnect(); el.removeEventListener('wheel', wheel); };
  }, []);
  const NL = dims.NL, LY = l => (l - (NL - 1) / 2) * (2.7 / NL);
  const span = Math.max(...objs.map(o => o.x + o.w / 2)) - Math.min(...objs.map(o => o.x - o.w / 2)) + 0.9;
  const proj = (x, y, z) => { const s = st.current, cy = Math.cos(s.yaw), sy = Math.sin(s.yaw), cp = Math.cos(s.pitch), sp = Math.sin(s.pitch);
    const x1 = x * cy - z * sy, z1 = x * sy + z * cy, y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp, k = Math.min(s.W / (span * 1.12 + 0.5), s.H / 3.6) * s.zoom;
    return [s.W * 0.5 + x1 * k, s.H * 0.53 - y1 * k, z2]; };
  useEffect(() => {
    const el = cv.current, ctx = el.getContext('2d'), s = st.current;
    if (s.view !== view) { s.view = view; s.yaw = VIEWS[view][0]; s.pitch = VIEWS[view][1]; }   // 换视角
    if (!s.W) return;
    ctx.setTransform(s.dpr, 0, 0, s.dpr, 0, 0); ctx.clearRect(0, 0, s.W, s.H); s.hits = [];
    const quad = (pts, fill, stroke, lw = 1) => { ctx.beginPath(); pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath(); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); } };
    const drawTex = (img, x0, x1, z0, z1, y, a) => { const p00 = proj(x0, y, z0), p10 = proj(x1, y, z0), p01 = proj(x0, y, z1);
      ctx.save(); ctx.globalAlpha = a; ctx.imageSmoothingEnabled = false;
      ctx.setTransform(s.dpr * (p10[0] - p00[0]) / img.width, s.dpr * (p10[1] - p00[1]) / img.width, s.dpr * (p01[0] - p00[0]) / img.height, s.dpr * (p01[1] - p00[1]) / img.height, s.dpr * p00[0], s.dpr * p00[1]); ctx.drawImage(img, 0, 0); ctx.restore(); };
    const alpha = l => view === 'top' ? (l === layer ? 1 : 0.07) : l === layer ? 1 : l > layer ? 0.13 : 0.85;
    const llm = objs[0];
    for (const o of objs) {
      const x0 = o.x - o.w / 2, x1 = o.x + o.w / 2, z0 = -o.d / 2, z1 = o.d / 2;
      for (let l = 0; l < NL; l++) {
        if (o.id !== 'llm' && o.id !== 'hs' && l === 0) continue;
        const y = LY(l), a = alpha(l), cur = l === layer, c = [[x0, z0], [x1, z0], [x1, z1], [x0, z1]].map(([x, z]) => proj(x, y, z));
        ctx.globalAlpha = a;
        if (o.id === 'llm') {
          quad(c, '#f7f9fc', cur ? '#d29a00' : '#97a6bb', cur ? 2.2 : 0.8);
          if (l > 0) { quad([[x0 + 0.05, z0 + 0.06], [o.x - 0.12, z0 + 0.06], [o.x - 0.12, z1 - 0.06], [x0 + 0.05, z1 - 0.06]].map(([x, z]) => proj(x, y, z)), '#e5def3');
            quad([[o.x + 0.12, z0 + 0.06], [x1 - 0.05, z0 + 0.06], [x1 - 0.05, z1 - 0.06], [o.x + 0.12, z1 - 0.06]].map(([x, z]) => proj(x, y, z)), '#dcefe4'); }
          else quad(c, '#e8ecf2');
        } else {
          quad(c, '#fff', cur ? '#d29a00' : '#b4bfce', cur ? 2 : 0.6);
          const t = TX[o.id]?.[o.id === 'hs' ? l : l - 1]; if (t) drawTex(t, x0, x1, z0, z1, y, a);
        }
        ctx.globalAlpha = 1; s.hits.push({ o, l, y, x0, x1, z0, z1 });
      }
      const pt = proj(o.x, LY(NL - 1), z0); ctx.fillStyle = '#3e4b60'; ctx.font = `${o.id === 'llm' ? 'bold 12px' : '12px'} sans-serif`; ctx.textAlign = 'center'; ctx.fillText(o.name, pt[0], pt[1] - 12); ctx.textAlign = 'left';
      if (o.id !== 'llm' && !(o.id !== 'hs' && layer === 0)) { const a = proj(o.x > llm.x ? llm.x + llm.w / 2 : llm.x - llm.w / 2, LY(layer), 0), b = proj(o.x > llm.x ? x0 : x1, LY(layer), 0);
        ctx.setLineDash([4, 4]); ctx.strokeStyle = '#d29a00'; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.setLineDash([]); }
    }
    ctx.strokeStyle = '#172033'; ctx.lineWidth = 2.5; ctx.beginPath(); for (let l = 0; l < NL; l++) { const p = proj(llm.x, LY(l), 0); l ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]); } ctx.stroke();
    for (let l = 0; l < NL; l += 6) { const p = proj(llm.x - llm.w / 2, LY(l), llm.d / 2); ctx.fillStyle = '#68748a'; ctx.font = '10.5px sans-serif'; ctx.textAlign = 'right'; ctx.fillText(l ? `L${l}` : '词嵌入', p[0] - 6, p[1] + 4); ctx.textAlign = 'left'; }
    overlays?.({ ctx, proj, LY, llm, alpha });
    if (live?.curve) { const mx = Math.max(...live.curve.slice(1)) || 1; for (let l = 1; l < Math.min(NL, live.curve.length); l++) { const [x, y] = proj(llm.x, LY(l), 0), r = 2 + 9 * (live.curve[l] / mx); ctx.fillStyle = 'rgba(210,154,0,.55)'; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); } }
    const pc = proj(llm.x, LY(layer), 0), g = ctx.createRadialGradient(pc[0], pc[1], 0, pc[0], pc[1], 15); g.addColorStop(0, 'rgba(210,154,0,.9)'); g.addColorStop(1, 'rgba(210,154,0,0)'); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(pc[0], pc[1], 15, 0, 7); ctx.fill();
    for (const m of marks) { const h = s.hits.find(t => t.o.id === m.id && t.l === layer); if (!h || !h.o.grid) continue; const [gw, gh] = h.o.grid;
      const x = h.x0 + (h.x1 - h.x0) * ((m.idx % gw) + 0.5) / gw, z = h.z0 + (h.z1 - h.z0) * (Math.floor(m.idx / gw) + 0.5) / gh, p = proj(x, LY(layer), z);
      ctx.strokeStyle = '#a33e43'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, 7); ctx.stroke(); }
  });   // 每次渲染都重画（纹理、对象和视角都可能变）
  const pick = (mx, my) => {
    const s = st.current, order = view === 'top' ? s.hits.filter(h => h.l === layer) : [...s.hits].sort((a, b) => b.l - a.l);
    for (const h of order) { if (view !== 'top' && h.l > layer) continue;
      const p00 = proj(h.x0, h.y, h.z0), p10 = proj(h.x1, h.y, h.z0), p01 = proj(h.x0, h.y, h.z1), ax = p10[0] - p00[0], ay = p10[1] - p00[1], bx = p01[0] - p00[0], by = p01[1] - p00[1], det = ax * by - ay * bx, dx = mx - p00[0], dy = my - p00[1];
      const u = (dx * by - dy * bx) / det, v = (ax * dy - ay * dx) / det; if (u < 0 || u > 1 || v < 0 || v > 1) continue;
      if (!h.o.grid) return { o: h.o.id, l: h.l, part: u < 0.5 ? '注意力' : 'MLP' };
      const [gw, gh] = h.o.grid; return { o: h.o.id, l: h.l, idx: Math.floor(v * gh) * gw + Math.floor(u * gw) };
    }
    return null;
  };
  const drag = useRef(null);
  return <div className="aw-canvas">
    <canvas ref={cv} data-tick={tick} aria-label={`LLM 的 3D 结构：${NL - 1} 层，从下到上；当前第 ${layer} 层`}
      onPointerDown={e => { drag.current = { x: e.clientX, y: e.clientY, yaw: st.current.yaw, pitch: st.current.pitch, moved: false }; e.currentTarget.setPointerCapture(e.pointerId); }}
      onPointerMove={e => { const d = drag.current, r = e.currentTarget.getBoundingClientRect();
        if (d) { const dx = e.clientX - d.x, dy = e.clientY - d.y; if (Math.abs(dx) + Math.abs(dy) > 4) d.moved = true; st.current.yaw = d.yaw + dx * 0.006; st.current.pitch = Math.max(-0.1, Math.min(1.4, d.pitch + dy * 0.005)); setTick(t => t + 1); return; }
        const h = pick(e.clientX - r.left, e.clientY - r.top); setTip(h ? { x: e.clientX - r.left, y: e.clientY - r.top, h } : null); }}
      onPointerLeave={() => setTip(null)}
      onPointerUp={e => { const d = drag.current; drag.current = null; if (!d || d.moved) return; const r = e.currentTarget.getBoundingClientRect(), h = pick(e.clientX - r.left, e.clientY - r.top); onPick(h, e.clientX - r.left, e.clientY - r.top); }} />
    {tip && <div className="aw-tip" style={{ left: tip.x + 12, top: tip.y + 12 }}>{dims.describe(tip.h)}</div>}
  </div>;
}

/* ---------- 主组件 ---------- */
export default function AssocWorkbench({ mode, onMode, contextBar, children }) {
  const live = useLiveBase();                                   // 实时生成
  const lm = useLiveModel(live.base);                           // 本机模型的加载 / 卸载
  const nat = useLocalBase('/api/traces/mainline/ma-h2/samples'); // 冻结样本的原生数据（只在本机可读）
  const base = nat.base;
  const [samples, setSamples] = useState(null);
  const [run, setRun] = useState({ i: 0, other: null });
  const [fpos, setFpos] = useState(0);           // 冻结样本里的读出位置
  const [livePos, setLivePos] = useState(0);     // 实时运行里的 token 序号
  const [src, setSrc] = useState('frozen');      // frozen = 冻结样本；live = 本次实时运行
  const [gen, setGen] = useState({ busy: false, res: null, err: '' });
  const [layer, setLayer] = useState(24);
  const [on, setOn] = useState(() => new Set(['hs', 'heads']));
  const [tab, setTab] = useState('hs');
  const [sel, setSel] = useState({ coord: null, unit: null, head: null });
  const [view, setView] = useState('iso');
  const [max, setMax] = useState(false);
  const [pop, setPop] = useState(null);
  const [opt, setOpt] = useState({ hsNorm: 'global', hsClip: 0.99, mlpClip: 0.99, headTarget: 'follow', headMax: 0.5, diffAlign: 'named' });
  const [hide, setHide] = useState(() => new Set());
  const [listed, setListed] = useState(() => new Set(['hs', 'mlp', 'heads', 'delta', 'diff', 'lens']));
  const [openOpt, setOpenOpt] = useState(() => new Set(['hs']));
  const [cfg, setCfg] = useState(false);
  const [mini, setMini] = useState(false);
  const [float, setFloat] = useState({ x: 16, y: 96 });
  const [text, setText] = useState('');
  const skel = useApi('/api/model/skeleton');
  const models = (skel.data?.models || []).filter(m => Number.isInteger(m.n_layers));

  useEffect(() => {
    if (base == null) return undefined;
    let dead = false;
    Promise.resolve(nat.status).then(d => { if (dead || !d?.items) return; setSamples(d);
      const it = d.items[0], twin = d.items.find(x => x.entity === it.entity && x.cell[0] !== it.cell[0] && x.cell.slice(1).join() === it.cell.slice(1).join());
      setRun({ i: 0, other: twin?.i ?? null }); setFpos(d.positions.indexOf('ans')); })
      .catch(e => { if (!dead) setSamples({ error: e.message }); });
    return () => { dead = true; };
  }, [base, nat.status]);
  const A = useSample(samples?.items ? base : null, samples?.items ? run.i : null);
  const B = useSample(samples?.items && run.other != null ? base : null, run.other);
  const saF = useMemo(() => sampler(A.data), [A.data]), sbF = useMemo(() => sampler(B.data), [B.data]);
  const isLive = src === 'live' && Boolean(gen.res);
  const LB = useBundle(live.base, isLive ? gen.res.id : null, isLive ? livePos : null);
  const saL = useMemo(() => liveSampler(LB.data, gen.res), [LB.data, gen.res]);
  const sa = isLive ? saL : saF, sb = isLive ? null : sbF;
  const native = Boolean(sa);
  const reason = isLive ? (LB.error || (!sa ? '读取实时运行…' : ''))
    : !nat.checked ? '正在查找本机数据…' : samples?.error || A.error || (base == null ? '网页版不公开原生全场：原生数据保存在采集节点，桌面客户端或本机后端可用' : !sa ? '读取冻结样本…' : '');
  const positions = isLive ? gen.res.tokens : (samples?.positions || ['ans']);
  const pos = isLive ? livePos : fpos;
  const setPos = isLive ? setLivePos : setFpos;
  const ANS = isLive ? gen.res.tokens.length - 1 : positions.indexOf('ans');
  const mlpOk = isLive ? Boolean(sa?.hasMlp) : pos === ANS;     // 实时运行只在模块读出位置（首 / 中 / 末）采了 MLP
  const tokText = t => String(t ?? '').replace(/\n/g, '⏎') || '␣';
  const posName = i => (isLive ? `token ${i}「${tokText(positions[i])}」` : (POSN[positions[i]] || positions[i] || ''));
  const NL = sa?.NL || 37, NB = NL - 1, NH = sa?.NH || 32, DM = sa?.DM || 2560, NF = sa?.NF || 9728;
  const [hw, hh] = grid(DM), [mw, mh] = grid(NF);
  const aligned = sa && sb && sa.T === sb.T && sa.d.tokpos.join() === sb.d.tokpos.join();
  const diffOn = on.has('diff') && sb && (aligned || opt.diffAlign === 'named');
  const shown = id => on.has(id) && !hide.has(id) && (id === 'heads' || native);

  /* 纹理 */
  const TX = useMemo(() => {
    const out = {};
    if (shown('hs') && sa) { const v = l => { const a = sa.hs(pos, l); if (!diffOn) return a; const b = sb.hs(pos, l); return a.map((x, i) => x - b[i]); };
      const vals = Array.from({ length: NL }, (_, l) => v(l)), sc = pct(vals.slice(0, NL - 1), opt.hsClip);
      out.hs = vals.map(x => { const m = opt.hsNorm === 'layer' ? absMax(x) : sc; return tex(hw, hh, d => { for (let i = 0; i < hw * hh; i++) put(d, i, i < DM ? divc(x[i], m) : [246, 248, 251]); }); }); }
    if (shown('mlp') && sa) { const ok = mlpOk, v = b => { const a = sa.mlp(b); if (!diffOn) return a; const c = sb.mlp(b); return a.map((x, i) => x - c[i]); };
      const vals = ok ? Array.from({ length: NB }, (_, b) => v(b)) : null, sc = ok ? pct(vals, opt.mlpClip) : 1;
      out.mlp = Array.from({ length: NB }, (_, b) => tex(mw, mh, d => { for (let i = 0; i < mw * mh; i++) put(d, i, i >= NF ? [246, 248, 251] : vals ? divc(vals[b][i], sc) : [214, 222, 232]); })); }
    if (shown('heads')) {
      const target = opt.headTarget === 'follow' || isLive ? pos : positions.indexOf(opt.headTarget);
      out.heads = Array.from({ length: NB }, (_, b) => tex(8, Math.ceil(NH / 8), d => { for (let h = 0; h < NH; h++) {
        let val, rgb;
        if (sa) { val = sa.att(b, h)[sa.d.tokpos[target]]; rgb = diffOn ? divc(val - sb.att(b, h)[sb.d.tokpos[target]], opt.headMax * 0.8) : seqc(val, opt.headMax); }
        else { val = FLOW_DATA.heads.doc[b]?.[h] ?? 0; rgb = seqc(val, opt.headMax); }       // 案例汇总：答案位置对资料颜色的注意力
        put(d, h, rgb); } }));
    }
    return out;
  }, [sa, sb, pos, on, hide, opt, diffOn]);   // eslint-disable-line react-hooks/exhaustive-deps

  const objs = useMemo(() => {
    const o = [{ id: 'llm', x: 0, w: 1.3, d: 0.6, name: `${sa ? 'Qwen3-4B' : '模型'} · ${NB} 层` }];
    let r = 1.1, l = -1.1;
    if (shown('hs')) { o.push({ id: 'hs', x: r + 0.55, w: 1.1, d: 1.1 * hh / hw, name: `HiddenState · ${DM} 坐标`, grid: [hw, hh] }); r += 1.55; }
    if (shown('mlp')) { o.push({ id: 'mlp', x: r + 0.65, w: 1.3, d: 1.3 * mh / mw, name: `MLP 单元 · ${NF}`, grid: [mw, mh] }); r += 1.75; }
    if (shown('heads')) { o.push({ id: 'heads', x: l - 0.45, w: 0.9, d: 0.9 * Math.ceil(NH / 8) / 8, name: `注意力 · ${NH} 头${sa ? '' : '（案例汇总）'}`, grid: [8, Math.ceil(NH / 8)] }); l -= 1.35; }
    const cx = (r + l) / 2; o.forEach(x => { x.x -= cx; }); return o;
  }, [sa, on, hide, hw, hh, mw, mh]);   // eslint-disable-line react-hooks/exhaustive-deps

  const describe = h => {
    if (h.o === 'llm') return `第 ${h.l} 层 · ${h.l ? h.part : '词嵌入'}`;
    if (h.o === 'hs') return sa && h.idx < DM ? `HiddenState · 第 ${h.l} 层 · 坐标 ${h.idx} · ${sa.hs(pos, h.l)[h.idx].toFixed(4)}` : '';
    if (h.o === 'mlp') return sa && h.idx < NF ? `MLP · 块 ${h.l - 1} · 单元 ${h.idx}${mlpOk ? ' · A ' + sa.mlp(h.l - 1)[h.idx].toFixed(4) : ''}` : '';
    if (h.o === 'heads') return h.idx < NH ? `注意力 · L${h.l - 1}H${h.idx} · ${sa ? sa.att(h.l - 1, h.idx)[sa.d.tokpos[pos]].toFixed(4) : (FLOW_DATA.heads.doc[h.l - 1]?.[h.idx] ?? 0).toFixed(3) + '（案例汇总）'}` : '';
    return '';
  };
  const onPick = (h, x, y) => {
    if (!h) { setPop(null); return; }
    setLayer(h.l);
    if (h.o === 'hs' && h.idx < DM) { setSel(s => ({ ...s, coord: h.idx })); setTab('hs'); }
    if (h.o === 'mlp' && h.idx < NF) { setSel(s => ({ ...s, unit: h.idx })); setTab('mlp'); }
    if (h.o === 'heads' && h.idx < NH) { setSel(s => ({ ...s, head: h.idx })); setTab('heads'); }
    setPop({ h, x, y });
  };
  const toggle = id => setOn(s => { const n = new Set(s); if (n.has(id)) n.delete(id); else { n.add(id); setTab(id); if (MODE_IDS.has(id)) onMode(id); } return n; });

  /* 逐层播放 */
  useEffect(() => { const k = e => { if (e.key === 'Escape') setMax(false); }; window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k); }, []);

  async function generate() {
    setGen({ busy: true, res: null, err: '' });
    try { const r = await runOnce(live.base, text); setGen({ busy: false, res: r, err: '' }); setTab('result'); setSrc('live'); setLivePos(r.tokens.length - 1); setLayer(36); }
    catch (e) { setGen({ busy: false, res: null, err: e.message }); }
  }

  /* 叠加：层间变化（LLM 两侧的条）与逐层读出（右缘） */
  const overlays = useCallback(({ ctx, proj, LY, llm, alpha }) => {
    if (shown('delta') && sa) { const v = []; for (let l = 1; l < NL - 1; l++) v.push(norm(sa.hs(pos, l).map((x, i) => x - sa.hs(pos, l - 1)[i])));
      const mx = Math.max(...v) || 1; v.forEach((x, k) => { const l = k + 1, p = proj(llm.x - llm.w / 2, LY(l), llm.d / 2); ctx.globalAlpha = alpha(l); ctx.fillStyle = '#7050a0'; ctx.fillRect(p[0] - 40 - 60 * x / mx, p[1] - 2, 60 * x / mx, 3); ctx.globalAlpha = 1; }); }
    if (on.has('lens') && !isLive) { const L = sa ? [sa.d.lens.map(r => r[0] - r[1]), sb ? sb.d.lens.map(r => r[0] - r[1]) : null] : [FLOW_DATA.lensDoc, FLOW_DATA.lensReal];
      for (let l = 0; l < NL; l++) { const p = proj(llm.x + llm.w / 2, LY(l), -llm.d / 2); ctx.globalAlpha = alpha(l);
        ctx.fillStyle = '#2a78d6'; ctx.fillRect(p[0] + 8, p[1] - 4, Math.max(-60, Math.min(60, (L[0][l] || 0) * 4)), 3);
        if (L[1]) { ctx.fillStyle = '#d4622b'; ctx.fillRect(p[0] + 8, p[1], Math.max(-60, Math.min(60, (L[1][l] || 0) * 4)), 3); } ctx.globalAlpha = 1; } }
  }, [sa, sb, pos, on, hide, NL]);   // eslint-disable-line react-hooks/exhaustive-deps

  const marks = [sel.coord != null && { id: 'hs', idx: sel.coord }, sel.unit != null && { id: 'mlp', idx: sel.unit }, sel.head != null && { id: 'heads', idx: sel.head }].filter(Boolean);
  const k = sa?.d, Bk = layer - 1;

  /* ---------- 操作面板（非全屏停靠在控制台，全屏时浮在 3D 上：同一个组件） ---------- */
  const Seg = ({ o, items }) => <span className="aw-seg">{items.map(([v, n]) => <button type="button" key={v} aria-pressed={String(opt[o]) === v} onClick={() => setOpt(x => ({ ...x, [o]: v }))}>{n}</button>)}</span>;
  const Sel = ({ o, items, num }) => <select value={String(opt[o])} onChange={e => setOpt(x => ({ ...x, [o]: num ? +e.target.value : e.target.value }))}>{items.map(([v, n]) => <option key={v} value={v}>{n}</option>)}</select>;
  const CLIP = [['0.95', '95% 分位'], ['0.99', '99% 分位'], ['0.999', '99.9% 分位'], ['1', '最大值']];
  const OPTS = {
    hs: () => <><div className="aw-r"><span>着色口径</span><Seg o="hsNorm" items={[['global', '全图'], ['layer', '逐层']]} /></div><div className="aw-r"><span>色标截断</span><Sel o="hsClip" items={CLIP} num /></div><div className="aw-r"><span>排列</span><span className="v8-note">原生编号 · {hw} × {hh}</span></div></>,
    mlp: () => <><div className="aw-r"><span>色标截断</span><Sel o="mlpClip" items={CLIP} num /></div>{!mlpOk && <p className="v8-note">{isLive ? '实时运行只在模块读出位置（首 / 中 / 末 token）采集 MLP，这个 token 显示为灰。' : '当前 token 不是答案位置：MA-H2 只在答案位置存 MLP，模型显示为灰。'}</p>}</>,
    heads: () => <><div className="aw-r"><span>看向</span><Sel o="headTarget" items={[['follow', '跟随所选 token'], ...positions.map(p => [p, POSN[p] || p])]} /></div><div className="aw-r"><span>色标上限</span><Sel o="headMax" items={[['0.25', '0.25'], ['0.5', '0.5'], ['1', '1.0']]} num /></div></>,
    diff: () => <><div className="aw-r"><span>对照运行</span><span>{run.other != null ? `#${run.other}` : '—'}</span></div><div className="aw-r"><span>对齐</span><Seg o="diffAlign" items={[['strict', '严格'], ['named', '命名位置']]} /></div>
      {sb && !aligned && <p className="v8-note">两句 token 数不同（{sa?.T} / {sb.T}）：{opt.diffAlign === 'strict' ? '严格对齐时不显示差值。' : '按命名读出位置比较，差值含前缀差异。'}</p>}</>,
  };
  const ops = <aside className={`aw-ops${max ? ' float' : ' docked'}${mini ? ' min' : ''}`} style={max ? { left: float.x, top: float.y } : undefined} aria-label="操作面板">
    <div className="aw-bar" onPointerDown={e => {     // 全屏时拖动标题栏移动浮窗
      if (!max || e.target.tagName === 'BUTTON') return;
      const s = { x0: e.clientX, y0: e.clientY, x: float.x, y: float.y };
      const move = ev => setFloat({ x: Math.max(0, s.x + ev.clientX - s.x0), y: Math.max(0, s.y + ev.clientY - s.y0) });
      const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
      window.addEventListener('pointermove', move); window.addEventListener('pointerup', up); }}>
      <b>操作</b>{max && <span className="v8-note">拖动标题栏移动</span>}
      <button type="button" className={cfg ? 'on' : ''} title="选择在这里显示哪些插件" onClick={() => { setCfg(c => !c); setMini(false); }}>⚙</button>
      {max && <button type="button" title="收起 / 展开" onClick={() => setMini(m => !m)}>{mini ? '+' : '—'}</button>}
    </div>
    {cfg && <div className="aw-cfg">{PLUGINS.map(([g, ps]) => <div key={g}><h5>{g}</h5>{ps.map(p => <label key={p.id}><input type="checkbox" checked={listed.has(p.id)} onChange={() => setListed(s => { const n = new Set(s); if (n.has(p.id)) n.delete(p.id); else n.add(p.id); return n; })} />{p.name}<small>{TAGN[p.tag]}{on.has(p.id) ? ' · 已启用' : ''}</small></label>)}</div>)}
      <div className="aw-acts"><button type="button" onClick={() => setListed(new Set(ALL.map(p => p.id)))}>全选</button><button type="button" onClick={() => setListed(new Set(on))}>只留已启用</button><button type="button" onClick={() => setListed(new Set())}>全不选</button><button type="button" onClick={() => setCfg(false)}>完成</button></div></div>}
    <div className="aw-in">
      <h4>运行参数</h4>
      <div className="aw-r"><span>模型</span><span>{k ? `Qwen3-4B · bf16 · ${k.model_snapshot.slice(0, 8)}` : '—'}</span></div>
      <div className="aw-r"><span>来源</span><span className="aw-seg"><button type="button" aria-pressed={!isLive} onClick={() => setSrc('frozen')}>冻结样本</button><button type="button" aria-pressed={isLive} disabled={!gen.res} onClick={() => setSrc('live')}>实时运行</button></span></div>
      <div className="aw-r"><span>输入</span>{isLive ? <span className="v8-note">本次输入的文字（{gen.res.tokens.length} token）</span> : samples?.items ? <select value={run.i} onChange={e => { const i = +e.target.value, it = samples.items[i], twin = samples.items.find(x => x.entity === it.entity && x.cell[0] !== it.cell[0] && x.cell.slice(1).join() === it.cell.slice(1).join()); setRun({ i, other: twin?.i ?? null }); }}>
        {samples.items.slice(0, 200).map(x => <option key={x.i} value={x.i}>#{x.i} · {x.entity} · {x.cell[0] === 'doc' ? '按资料' : '按现实'} · {x.cell.slice(1).join('/')}</option>)}</select> : <span className="v8-note">{reason}</span>}</div>
      <div className="aw-r"><span>token</span><select value={pos} onChange={e => setPos(+e.target.value)}>{positions.map((p, i) => <option key={i} value={i}>{isLive ? `${i} · ${tokText(p)}` : `${POSN[p] || p}${k ? `（${k.tokpos[i]}）` : ''}`}</option>)}</select></div>
      <div className="aw-r"><span>层</span><span className="aw-row"><input type="range" min="0" max={NL - 1} value={layer} onChange={e => setLayer(+e.target.value)} /><b>{layer}</b></span></div>
      <div className="aw-r"><span>视角</span><span className="aw-seg">{VORDER.map(v => <button type="button" key={v} aria-pressed={view === v} onClick={() => setView(v)}>{VIEWS[v][2]}</button>)}</span></div>
      <div className="aw-r"><span>数据</span><span className="v8-note">{isLive ? `实时运行 · ${gen.res.tokens.length} token · 原生 bf16 · 未登记观察` : native ? 'MA-H2 discovery 冻结样本 · 原生 float16 · 节点本地' : '案例汇总（MA-H2，2,520 条）'}</span></div>
      <h4>插件</h4>
      {ALL.filter(p => listed.has(p.id)).map(p => { const en = on.has(p.id), dis = p.native && !native, vis = en && !hide.has(p.id);
        return <div key={p.id} className={`aw-pl${en ? '' : ' off'}`}>
          <div onClick={() => OPTS[p.id] && setOpenOpt(s => { const n = new Set(s); if (n.has(p.id)) n.delete(p.id); else n.add(p.id); return n; })}>
            <input type="checkbox" checked={en} disabled={dis} title={dis ? reason : '启用插件'} onClick={e => e.stopPropagation()} onChange={() => toggle(p.id)} /><b>{p.name}</b>
            {en && ['hs', 'mlp', 'heads', 'delta'].includes(p.id) && <button type="button" className="aw-eye" aria-pressed={vis} onClick={e => { e.stopPropagation(); setHide(s => { const n = new Set(s); if (n.has(p.id)) n.delete(p.id); else n.add(p.id); return n; }); }}>{vis ? '显示' : '隐藏'}</button>}
            <span className="v8-note">{OPTS[p.id] ? (openOpt.has(p.id) ? '▾' : '▸') : ''}</span></div>
          {dis && <p className="v8-note aw-why">{reason}</p>}
          {OPTS[p.id] && openOpt.has(p.id) && en && <div className="aw-opt">{OPTS[p.id]()}</div>}
        </div>; })}
      <p className="v8-note">⚙ 选择在这里显示哪些插件（{listed.size} / {ALL.length}）。</p>
      <h4>当前对象</h4>
      <div className="aw-kv"><span>层</span><span>{layer ? `第 ${layer} 层（块 ${layer - 1} 输出）` : '词嵌入'}</span><span>token</span><span>{POSN[positions[pos]] || positions[pos]}</span>
        {sa && sel.coord != null && <><span>坐标</span><span>{sel.coord} = {sa.hs(pos, layer)[sel.coord].toFixed(4)}</span></>}
        {sa && sel.unit != null && layer > 0 && mlpOk && <><span>MLP 单元</span><span>B{Bk}·U{sel.unit} = {sa.mlp(Bk)[sel.unit].toFixed(4)}</span></>}
        {sel.head != null && layer > 0 && <><span>注意力头</span><span>L{Bk}H{sel.head}</span></>}</div>
      {max && <p className="v8-note">Esc 或右上角按钮退出全屏。</p>}
    </div>
  </aside>;

  /* ---------- 插件面板 ---------- */
  const line = (series, mark, dp = 1) => { const W = 420, H = 130, all = series.flatMap(s => s.v.filter(Number.isFinite)); let lo = Math.min(0, ...all), hi = Math.max(...all); if (hi === lo) hi = lo + 1;
    const n = Math.max(...series.map(s => s.v.length)), X = i => 30 + (W - 38) * i / (n - 1), Y = v => 6 + (H - 22) * (1 - (v - lo) / (hi - lo));
    return <><svg viewBox={`0 0 ${W} ${H}`} className="aw-line" role="img" aria-label={series.map(s => s.name).join('、')}><line x1="30" x2={W - 8} y1={Y(0)} y2={Y(0)} stroke="#e2e7ef" />
      {mark != null && <line x1={X(mark)} x2={X(mark)} y1="4" y2={H - 16} stroke="#d29a00" strokeDasharray="3 3" />}
      {series.map(s => <path key={s.name} d={s.v.map((v, i) => Number.isFinite(v) ? `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}` : '').join('')} fill="none" stroke={s.color} strokeWidth="1.7" strokeDasharray={s.dash ? '4 3' : undefined} />)}
      <text x="0" y={Y(hi) + 4} fontSize="10" fill="#68748a">{hi.toFixed(dp)}</text><text x="0" y={Y(lo)} fontSize="10" fill="#68748a">{lo.toFixed(dp)}</text></svg>
      <div className="v8-note">{series.map(s => <span key={s.name} className="aw-key"><i style={{ background: s.color }} />{s.name}</span>)}</div></>; };
  const nameK = k ? `${CN[k.K] || k.K}` : '资料色', nameP = k ? `${CN[k.P] || k.P}` : '常识色';
  const needNative = <p className="v8-note">{reason}</p>;
  const P = {
    hs: () => !sa ? [needNative] : (() => { const v = sa.hs(pos, layer), top = [...v].map((x, i) => [i, x]).sort((p, q) => Math.abs(q[1]) - Math.abs(p[1])).slice(0, 8);
      return [<><h4>第 {layer} 层 · {posName(pos)} · ‖h‖ {norm(v).toFixed(1)}</h4><table className="v6-table"><tbody>{top.map(([i, x]) => <tr key={i} className={sel.coord === i ? 'on' : ''} onClick={() => setSel(s => ({ ...s, coord: i }))}><td>坐标 {i}</td><td className="num">{x.toFixed(4)}</td></tr>)}</tbody></table></>,
        sel.coord != null ? <><h4>坐标 {sel.coord} 跨层</h4>{line([{ v: Array.from({ length: NL }, (_, l) => sa.hs(pos, l)[sel.coord]), color: '#2a78d6', name: '本运行' }, ...(sb ? [{ v: Array.from({ length: NL }, (_, l) => sb.hs(pos, l)[sel.coord]), color: '#d4622b', name: '对照运行', dash: true }] : [])], layer, 2)}</>
          : <><h4>每层 ‖h‖</h4>{line([{ v: Array.from({ length: NL }, (_, l) => norm(sa.hs(pos, l))), color: '#2a78d6', name: '本运行' }], layer)}<p className="v8-note">点 3D 里的 HiddenState 模型选一个坐标。</p></>,
        '坐标按原生编号排成网格，网格位置没有含义；颜色按全图 |值| 的分位截断，悬停或点击看原值（float16）。']; })(),
    mlp: () => !sa ? [needNative] : !mlpOk ? [<p key="m" className="v8-note">{isLive ? '实时运行只在模块读出位置（首 / 中 / 末 token）采集 MLP；在 token 条里选这几个位置。' : 'MA-H2 只在答案位置存了 MLP 激活；把 token 切到答案位置。'}</p>] : layer === 0 ? [<p key="m" className="v8-note">词嵌入层没有 MLP。</p>] : (() => { const a = sa.mlp(Bk), top = [...a.keys()].sort((x, y) => Math.abs(a[y]) - Math.abs(a[x])).slice(0, 8); if (sel.unit != null && !top.includes(sel.unit)) top.push(sel.unit);
      return [<><h4>块 {Bk} · 最活跃的单元（A = silu(G)⊙V）</h4><table className="v6-table"><thead><tr><th>单元</th><th className="num">本运行</th><th className="num">对照</th></tr></thead><tbody>{top.map(u => <tr key={u} className={sel.unit === u ? 'on' : ''} onClick={() => setSel(s => ({ ...s, unit: u }))}><td>U{u}</td><td className="num">{a[u].toFixed(3)}</td><td className="num">{sb ? sb.mlp(Bk)[u].toFixed(3) : '—'}</td></tr>)}</tbody></table></>,
        <p className="v8-note">正式版在这里接普查单元卡、gate 输入的来源分解与四级状态。</p>, '单元按原生编号排成网格，网格位置没有含义；激活大小不说明它被使用。']; })(),
    heads: () => layer === 0 ? [<p key="h" className="v8-note">词嵌入层没有注意力头。</p>] : !sa ? [<><h4>块 {Bk} · 案例汇总：答案位置对资料颜色的注意力</h4><table className="v6-table"><thead><tr><th>头</th><th className="num">按资料</th><th className="num">按现实</th></tr></thead><tbody>{[...Array(NH).keys()].sort((x, y) => FLOW_DATA.heads.doc[Bk][y] - FLOW_DATA.heads.doc[Bk][x]).slice(0, 8).map(h => <tr key={h}><td>L{Bk}H{h}</td><td className="num">{FLOW_DATA.heads.doc[Bk][h].toFixed(3)}</td><td className="num">{FLOW_DATA.heads.real[Bk][h].toFixed(3)}</td></tr>)}</tbody></table></>, needNative, '注意力去向不等于计算使用。']
      : (() => { const tp = sa.d.tokpos[pos], rank = [...Array(NH).keys()].sort((x, y) => sa.att(Bk, y)[tp] - sa.att(Bk, x)[tp]).slice(0, 8);
        return [<><h4>块 {Bk} · 答案位置看“{posName(pos)}”最多的头</h4><table className="v6-table"><thead><tr><th>头</th><th className="num">本运行</th><th className="num">对照</th></tr></thead><tbody>{rank.map(h => <tr key={h} className={sel.head === h ? 'on' : ''} onClick={() => setSel(s => ({ ...s, head: h }))}><td>L{Bk}H{h}</td><td className="num">{sa.att(Bk, h)[tp].toFixed(3)}</td><td className="num">{sb ? sb.att(Bk, h)[sb.d.tokpos[pos]].toFixed(3) : '—'}</td></tr>)}</tbody></table></>,
          sel.head != null ? <><h4>L{Bk}H{sel.head} 对全部 {sa.T} 个 token</h4>{line([{ v: Array.from(sa.att(Bk, sel.head)), color: '#b07d00', name: '虚线 = 所选位置' }], tp, 2)}</> : <p className="v8-note">点 3D 里的注意力模型或表中一行。</p>, '注意力去向不等于计算使用。']; })(),
    delta: () => !sa ? [needNative] : (() => { const dn = [], cs = []; for (let l = 1; l < NL; l++) { const a = sa.hs(pos, l), b = sa.hs(pos, l - 1); dn.push(l === NL - 1 ? NaN : norm(a.map((x, i) => x - b[i]))); cs.push(l === NL - 1 ? NaN : cosv(a, b)); }
      const split = sa.hasSplit ? [{ v: Array.from({ length: NB }, (_, b) => norm(sa.ao(b))), color: '#7050a0', name: '注意力写入', dash: true }, { v: Array.from({ length: NB }, (_, b) => norm(sa.mo(b))), color: '#2f8f5b', name: 'MLP 写入' }] : [];
      return [<><h4>每层总变化 ‖Δh‖{sa.hasSplit ? ' 与注意力 / MLP 写入（本次实时运行，真实采集）' : '（3D：LLM 左侧紫条）'}</h4>{line([{ v: dn, color: '#183f77', name: '‖h_l − h_(l−1)‖' }, ...split], layer - 1)}</>, <><h4>相邻层方向 cos(h_l, h_(l−1))</h4>{line([{ v: cs, color: '#183f77', name: '越接近 1 改写越少' }], layer - 1, 2)}</>,
        sa.hasSplit ? '注意力写入与 MLP 写入是 o_proj 与 down_proj 的真实输出（本次运行在模块读出位置采集）；最后一层的 hs 已过最终归一化。' : isLive ? '这个 token 不是模块读出位置，没有写入拆分；在 token 条里选首 / 中 / 末位置。' : '最后一层的 hs 已过最终归一化，不与前层比较。冻结样本的注意力 / MLP 写入拆分需要原生权重（待接入）。']; })(),
    diff: () => isLive ? [<p key="d" className="v8-note">实时运行没有对照运行：资料 / 现实成对的比较只在冻结样本里。切回“冻结样本”使用。</p>, '', ''] : !sa || !sb ? [needNative] : (() => { if (!aligned && opt.diffAlign === 'strict') return [<p key="d">对齐检查：token 数 {sa.T} / {sb.T} ✗。严格对齐时不显示差值。</p>, '', '差值是观察，原因要靠干预。'];
      const rel = Array.from({ length: NL }, (_, l) => { const a = sa.hs(pos, l), b = sb.hs(pos, l); return norm(a.map((x, i) => x - b[i])) / norm(a); });
      return [<><p>对齐检查：同一快照 ✓ · token 数 {sa.T} / {sb.T} {aligned ? '✓' : '✗（按命名读出位置比较）'}</p><p className="v8-note">3D 里各插件模型显示本运行 − 对照运行的差值。</p></>, <><h4>每层相对差异</h4>{line([{ v: rel, color: '#a33e43', name: POSN[positions[pos]] }], layer, 2)}</>, '差值是观察，原因要靠干预；不对齐时差值含前缀差异。']; })(),
    lens: () => isLive ? [<p key="l" className="v8-note">实时运行的逐层读出待接入（需要本机权重）；本次的最终输出见“运行结果”。</p>, '', ''] : (() => { const L = sa ? [{ v: sa.d.lens.map(r => r[0] - r[1]), color: '#2a78d6', name: '本运行' }, ...(sb ? [{ v: sb.d.lens.map(r => r[0] - r[1]), color: '#d4622b', name: '对照运行' }] : [])] : [{ v: FLOW_DATA.lensDoc, color: '#2a78d6', name: '按资料（案例汇总）' }, { v: FLOW_DATA.lensReal, color: '#d4622b', name: '按现实（案例汇总）' }];
      return [<><h4>Logit Lens · 答案位置 · {nameK}色 − {nameP}色</h4>{line(L, layer)}</>, <p className="v8-note">3D：LLM 右缘的条。正式版另给每层前 5 个 token。</p>, '读出近似，早层不可靠。']; })(),
    tests: () => [<TestsPanel key="t" />, '', ''],
    result: () => gen.res ? [<><h4>下一个词的预测 · {gen.res.tokens.length} 个 token · {gen.res.elapsed} ms</h4><TopTokens top={gen.res.top} /></>, <><h4>每层改写幅度（3D 主干上的金色圆点）</h4>{line([{ v: gen.res.curve, color: '#d29a00', name: '1 − cos(h_l, h_(l−1))，最后一个位置' }], layer, 2)}</>, '实时生成是一次前向的未登记观察，不产生研究结论。'] : [<p key="r" className="v8-note">还没有运行。</p>],
  };
  const tabs = [...ALL.filter(p => on.has(p.id)), ...(gen.res ? [{ id: 'result', name: '运行结果' }] : [])];
  const cur = tabs.find(t => t.id === tab) ? tab : tabs[0]?.id;
  const panel = !cur ? <p className="v8-note">从操作面板启用一个插件。</p>
    : MODE_IDS.has(cur) ? (mode === cur ? <div className="aw-mode">{contextBar}{children}</div> : <p className="v8-note">正在切换到“{MODE_PLUGINS.find(p => p.id === cur).name}”…</p>)
      : P[cur] ? (() => { const [a, b, bound] = P[cur](); return <div className="aw-pbody"><div>{a}</div><div>{b}</div>{bound && <div className="aw-bound">{bound}</div>}</div>; })()
        : (() => { const p = ALL.find(x => x.id === cur); return <div className="aw-pbody"><div><h4>{p.name}</h4><p>{p.desc}</p></div><div><p className="v8-note">{p.tag === 'have' ? '项目里已有实现，正式接成插件后读同一个观察对象，并在 3D 上画自己的图层。' : '待接入：按插件规范声明需要的数据、3D 图层与面板。'}</p></div></div>; })();

  const popBody = pop && (() => { const { h } = pop;
    if (h.o === 'hs' && sa && h.idx < DM) { const vals = Array.from({ length: NL }, (_, l) => sa.hs(pos, l)[h.idx]);
      return <><h4>HiddenState · 第 {h.l} 层 · 坐标 {h.idx}</h4><div className="aw-kv"><span>值</span><span><b>{vals[h.l].toFixed(4)}</b></span>{sb && <><span>对照运行</span><span>{sb.hs(pos, h.l)[h.idx].toFixed(4)}</span></>}<span>本层 ‖h‖</span><span>{norm(sa.hs(pos, h.l)).toFixed(1)}</span></div>
        {line([{ v: vals, color: '#2a78d6', name: '这个坐标跨层' }], h.l, 2)}<p className="v8-note">相邻坐标：{[-2, -1, 1, 2].map(d => h.idx + d).filter(i => i >= 0 && i < DM).map(i => `${i}: ${sa.hs(pos, h.l)[i].toFixed(2)}`).join('　')}</p></>; }
    if (h.o === 'mlp' && sa && h.idx < NF) return !mlpOk ? <p className="v8-note">这个位置没有采集 MLP 激活。</p> : <><h4>MLP · 块 {h.l - 1} · 单元 {h.idx}</h4><div className="aw-kv"><span>A</span><span><b>{sa.mlp(h.l - 1)[h.idx].toFixed(4)}</b></span>{sb && <><span>对照运行</span><span>{sb.mlp(h.l - 1)[h.idx].toFixed(4)}</span></>}</div></>;
    if (h.o === 'heads' && h.idx < NH) return <><h4>注意力头 L{h.l - 1}H{h.idx}</h4><p>{describe(h)}</p></>;
    return <><h4>第 {h.l} 层</h4><p className="v8-note">{h.l ? h.part : '词嵌入'}。打开 HiddenState、MLP 神经元或注意力头插件，在旁边看这一层的具体数值。</p></>; })();

  return <div className="v8-route aw" aria-label="关联机制">
    <PageHead title="关联机制" sub="以 3D 里的模型结构为核心，用插件观察和研究：HiddenState、神经元、注意力、读出、对照，以及行业常用的归因与干预工具。" />
    <div className="aw-models" role="group" aria-label="模型">{(models.length ? models : [{ model: 'qwen3-4b', n_layers: 36 }]).map(m => {
      const isMain = /qwen3-4b/i.test(m.model);
      return <button type="button" key={m.model} aria-pressed={isMain} title={isMain ? '' : '这个模型还没有工作台的冻结样本'} disabled={!isMain}>
        <b>{shortModel(m.model)}</b><small>{m.n_layers} 层{m.d_ff ? ` · MLP ${m.d_ff}` : ''}</small><span className={isMain && native ? 'ok' : 'warn'}>{isMain ? (native ? '冻结样本 · 原生数据' : '案例汇总') : '暂无冻结样本'}</span></button>; })}
      <div className="aw-load" aria-label="本机模型">{(() => {
        const s = lm.status, state = !live.checked ? 'checking' : !live.base ? 'none' : s?.status || 'checking';
        const label = { checking: '查找本机模型…', none: '没有本机模型服务', unavailable: '本机没有模型权重', idle: '未加载', loading: `加载中${s?.progress ? ` · ${s.progress}` : ''}…`, ready: `已加载${s?.vram_gb != null ? ` · 显存 ${s.vram_gb} GB` : ''}`, error: '加载出错' }[state] || state;
        return <>
          <span className="v8-note">本机模型</span>
          <b className={`aw-ls ${state}`}>{s?.model_name ? `${s.model_name} · ` : ''}{label}</b>
          <span className="aw-row">
            <button type="button" className="fw-tbtn primary" disabled={!live.base || lm.busy || !['idle', 'error'].includes(state)} title={live.base ? '把模型加载到 GPU（占用显存）' : '需要桌面节点或本机后端（:5001 / :5002）'} onClick={lm.load}>加载</button>
            <button type="button" className="fw-tbtn" disabled={!live.base || lm.busy || !['ready', 'loading'].includes(state)} title="卸载模型，释放 GPU 显存" onClick={lm.unload}>卸载</button>
          </span>
          {(lm.msg || s?.error || (state === 'unavailable' && s?.reason)) && <small className="v8-note">{lm.msg || s?.error || s?.reason}</small>}
        </>; })()}</div></div>
    <section className={`aw-space${max ? ' max' : ''}`}>
      <Scene dims={{ NL, describe }} objs={objs} layer={layer} view={view} tex={TX} marks={marks} onPick={onPick} overlays={overlays} live={gen.res} />
      <div className="aw-ov tr"><button type="button" onClick={() => { setMax(m => !m); setPop(null); }}>{max ? '退出全屏（Esc）' : '⛶ 全屏'}</button></div>
      <div className="aw-big"><b>{layer ? `第 ${layer} 层` : '词嵌入'}</b><span className="v8-note">{isLive ? '实时运行 · ' : '冻结样本 · '}{posName(pos)}{k ? `（token ${k.tokpos[pos]}）· ‖h‖ ${norm(sa.hs(pos, layer)).toFixed(1)}` : ''}</span></div>
      <div className="aw-legend v8-note"><span><i style={{ background: '#e5def3' }} />注意力</span><span><i style={{ background: '#dcefe4' }} />MLP</span><span><i style={{ background: '#172033' }} />残差主干</span>
        {(shown('hs') || shown('mlp')) && <span><i style={{ background: 'linear-gradient(90deg,#d4622b,#fff,#1f5fbf)' }} />{diffOn ? '差值：蓝 = 本运行大' : '蓝正橙负'}</span>}
        {shown('heads') && <span><i style={{ background: 'linear-gradient(90deg,#fff,#1e5fbf)' }} />头越蓝，越看所选 token</span>}
        <span>拖动旋转 · 滚轮缩放 · 点格子看数值 · 当前层以上半透明</span></div>
      {pop && <div className="aw-pop" style={{ left: Math.min(pop.x + 16, 9999), top: pop.y + 10 }}><button type="button" className="x" aria-label="关闭" onClick={() => setPop(null)}>×</button>{popBody}</div>}
      {max && ops}
    </section>
    <section className="v8-card aw-cons" aria-label="控制台">
      <div className="aw-chead"><h3>控制台</h3><span className="v8-note">输入一段文本，看模型逐层怎样运行；用插件观察和研究</span></div>
      <div className="aw-cin"><div><textarea value={text} onChange={e => setText(e.target.value)} placeholder="输入一段文字，点“执行”后模型做一次前向，3D 主干上叠加本次每层的改写幅度（需要桌面节点或本机后端）；不输入时“执行”显示当前样本的完整运行，“下一步”逐层前进" />
        <p className="v8-note">{isLive ? `现在显示的是本次实时运行（Qwen3-4B 真实前向，${gen.res.tokens.length} 个 token，用时 ${gen.res.elapsed} ms）：3D 与插件都读这次运行的原生值。` : native ? `现在显示的是冻结样本 #${run.i}：实体 ${k.entity}，资料里写的颜色 = ${nameK}，常识颜色 = ${nameP}。` : reason}</p>{gen.err && <p className="ac-err" role="alert">{gen.err}</p>}</div>
        <div className="aw-run">
          <button type="button" className="fw-tbtn primary" disabled={gen.busy || (text.trim() && (!live.base || live.status?.available === false))}
            title={text.trim() ? (live.base ? '用本机模型运行这段文字' : '没有连接到可用的本机模型：运行需要桌面节点或本机后端') : '显示当前样本的完整运行（到最后一层）'}
            onClick={() => { if (text.trim()) generate(); else setLayer(NL - 1); }}>{gen.busy ? '执行中…' : '执行'}</button>
          <button type="button" className="fw-tbtn" title="前进一层；到最后一层后从词嵌入重新开始" onClick={() => setLayer(l => (l >= NL - 1 ? 0 : l + 1))}>下一步</button>
          <span className="v8-note">{layer ? `第 ${layer} / ${NL - 1} 层` : `词嵌入 / ${NL - 1} 层`}</span>
          <span className="aw-seg aw-src" role="group" aria-label="数据来源"><button type="button" aria-pressed={!isLive} onClick={() => setSrc('frozen')}>冻结样本</button>
            <button type="button" aria-pressed={isLive} disabled={!gen.res} title={gen.res ? '本次实时运行' : '先用“执行”运行一段文字'} onClick={() => setSrc('live')}>实时运行</button></span>
        </div></div>
      {isLive ? <div className="aw-toks live"><span className="v8-note">本次输入的 {positions.length} 个 token（点一个看它的各层；带下划线的是模块读出位置，有 MLP 与写入拆分）</span>
        {positions.map((t, i) => <button type="button" key={i} className={`${i === pos ? 'on' : ''}${gen.res.readout?.includes(i) ? ' ro' : ''}`} title={`token ${i}`} onClick={() => setLivePos(i)}>{tokText(t)}</button>)}</div>
        : k && <div className="aw-toks"><span className="v8-note">token（{k.ntok} 个；有名字的是读出位置）</span>{Array.from({ length: k.ntok }, (_, t) => { const i = k.tokpos.indexOf(t);
        return i >= 0 ? <button type="button" key={t} className={i === pos ? 'on' : ''} title={POSN[positions[i]]} onClick={() => setPos(i)}>{t}</button> : <span key={t} title={`token ${t}`}>·</span>; })}</div>}
      <div className="aw-body">{!max && ops}<div className="aw-right"><div className="aw-tabs" role="tablist">{tabs.map(t => <button type="button" role="tab" key={t.id} aria-selected={cur === t.id} onClick={() => { setTab(t.id); if (MODE_IDS.has(t.id)) onMode(t.id); }}>{t.name}</button>)}</div>{panel}</div></div>
      <p className="v8-note aw-foot">实时生成是一次前向的未登记观察，不产生研究结论。操作面板与全屏时的浮窗是同一个面板。</p>
    </section>
  </div>;
}
