/* LiveStage · 空间 · 实时分析 3D 舞台（live_stage.v1 / 「参数星河」）
   ────────────────────────────────────────────────────────────────
   与 StackMode（36 层平铺玻璃盒）的分工：
     StackMode = 结构视角（层盒 + 形状/参数量，静态总览）
     LiveStage = 运行视角（数值视角 + 逐层推进 + 真实权重读数）
   本组件只服务「实时分析」，不接管数据分析模式。

   五种显示效果（底部「视图」下拉切换，同一份数据换摆法）：
     ① 光塔   层盘环（默认）—— 每层一圈盘，格=tensor
     ② 螺旋梯 层盘逐层旋进 + 螺旋脊光带
     ③ 地层矩阵 行=层、列=tensor 的 3D 柱阵（柱高=数值）
     ④ 星核   同心陀螺球壳，由内向外展开
     ⑤ 粒子星河 每格一颗粒子，未取数=暗尘
   切换视图只改呈现，不改数值口径：未取数的格一律冷灰待取。

   参考的公开项目（设计取向，非代码依赖）：
     · 09Catho/axon —— 逐 token 流式推送 + 节点亮度=激活强度 + 参数滑块
     · norika1207-lab/llm-neuron-atlas —— neuron-as-city，全层一次渲染，空间导航
     · fbkaragoz/synapse —— 确定性层网格（非随机散布）+ bloom 辉光 + 阈值控制
     · bbycroft/llm-viz —— 单步推进（下一步）与可暂停的节奏控制
   本组件采纳「确定性布局 + 单步推进 + 强度着色」，但数据落在**参数级**（真实权重），
   不是激活级：激活需要 GPU 前向（属 P2，不在本轮伪造）。

   数据（全部真实，零 demo 播种值）：
     conn   = POST /api/live/connect 的载荷（tensors/groups/config，权重头扫描，不加载权重）
     数值   = POST /api/live/weights/summary { names } → {rms, mean_abs, max_abs, sampled}
              （safetensors 按 32×64 行块切片读；行数超限时行域均匀抽样）
   纪律：
     ① 未连接 → 只画结构轮廓（虚线，显式标注「未连接」），绝不画数值着色；
     ② 每格只显示已真实取回的数值，未取回的格保持「待采」暗态；
     ③ 面板固定标注抽样口径（32×64 行块 / 全量遍历），不把采样值说成全量。
   交互：左键旋转 / 右键平移 / 滚轮缩放；点层盘选中该层 → 柱阵展开 + 右侧参数表。
*/
import { API_BASE } from '../../../config.js';
import { useCallback, useEffect, useRef, useState } from 'react';


/* ── 数值格式化 ───────────────────────────────────────────── */
const fmtP = n => n >= 1e9 ? (n / 1e9).toFixed(2) + 'B'
  : n >= 1e6 ? (n / 1e6).toFixed(2) + 'M'
  : n >= 1e3 ? (n / 1e3).toFixed(1) + 'K' : String(n ?? 0);
const fmtV = v => v == null ? '—'
  : Math.abs(v) >= 1e-3 ? v.toFixed(5) : v.toExponential(3);

/* ── 颜色：v∈[0,1] → 冷灰 → 青 → 品红 → 琥珀（浅底上的高辨识色带）── */
const RAMP = [
  [148, 163, 184],   // 0.00 石板灰
  [ 14, 165, 233],   // 0.34 天青
  [ 99, 102, 241],   // 0.58 靛蓝
  [217,  70, 239],   // 0.80 品红
  [245, 158,  11],   // 1.00 琥珀
];
function rampRGB(v) {
  const t = Math.max(0, Math.min(1, v)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(t));
  const f = t - i, a = RAMP[i], b = RAMP[i + 1];
  return a.map((x, k) => Math.round(x + (b[k] - x) * f));
}
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

/* ── 层号解析：兼容 model.layers.N / transformer.h.N / blocks.N ── */
const LAYER_RE = /(?:^|\.)(?:layers|h|blocks)\.(\d+)\./;

/** conn → [{layer, tensors(按参数量降序)}]，层号升序 */
function buildLayerIndex(conn) {
  if (!conn || !Array.isArray(conn.tensors)) return [];
  const map = new Map();
  for (const t of conn.tensors) {
    if (!t || typeof t.name !== 'string') continue;
    const m = LAYER_RE.exec('.' + t.name);
    if (!m) continue;
    const l = Number(m[1]);
    if (!Number.isFinite(l)) continue;
    if (!map.has(l)) map.set(l, []);
    map.get(l).push(t);
  }
  return [...map.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([layer, ts]) => ({ layer, tensors: ts.sort((a, b) => (b.params || 0) - (a.params || 0)) }));
}

/* ── 显示效果（视图）注册表 ────────────────────────────────────
   下拉切换只改「怎么摆」，不改「数值从哪来」：
   所有视图共用同一份已取回的真实 rms，未取数的格一律画成冷灰待取，
   绝不为了好看而填示意值。cam = 切到该视图时的相机预设。        */
const VIEWS = [
  { id: 'tower', label: '光塔 · 层盘环', hint: '每层一圈盘，格 = tensor，亮度 = 真实 rms',
    cam: { yaw: 0.62, pitch: 0.30, dist: 30 } },
  { id: 'helix', label: '螺旋梯 · 光带', hint: '层盘逐层旋进，螺旋脊把各层串成一条光带',
    cam: { yaw: 0.95, pitch: 0.24, dist: 32 } },
  { id: 'matrix', label: '地层矩阵 · 柱阵', hint: '行 = 层，列 = tensor，柱高 = 真实 rms（精细取数可填满列）',
    cam: { yaw: 0.70, pitch: 0.60, dist: 34 } },
  { id: 'orb', label: '星核 · 同心球壳', hint: '层 = 同心陀螺壳，由内向外逐层展开',
    cam: { yaw: 1.02, pitch: 0.34, dist: 30 } },
  { id: 'nebula', label: '粒子星河 · 旋涡', hint: '每个已取数格 = 一颗粒子，未取数 = 暗尘',
    cam: { yaw: 0.42, pitch: 0.26, dist: 32 } },
];

/* 图例按视图换文案，避免「图例说 A、画面画 B」的错配 */
const LEGEND = {
  tower: [['ramp', '权重 rms 低 → 高'], ['flat', '未取数（待运行）'], ['ring', '当前步进层（扫描环）']],
  helix: [['ramp', '权重 rms 低 → 高'], ['line', '螺旋脊：层间推进方向'], ['ring', '当前步进层']],
  matrix: [['ramp', '柱高 / 色 = 真实 rms'], ['flat', '未取数的格（待运行）'], ['ring', '当前层行（高亮带）']],
  orb: [['ramp', '弧段色 = 真实 rms'], ['flat', '未取数弧段'], ['ring', '当前层球壳（加粗）']],
  nebula: [['ramp', '粒子色 / 大小 = 真实 rms'], ['dot', '未取数灰尘（结构占位）'], ['ring', '当前层轨道']],
};

/** 该层参与「取数」的 tensor：优先 Q / gate / down 三大承重矩阵（跨模型零字面量兜底=按参数量取前 N） */
function pickProbes(tensors, fine) {
  const big = tensors.filter(t =>
    (t.params || 0) >= 4096 && !/bias$/i.test(t.name) && !/norm/i.test(t.name));
  if (!big.length) return [];
  if (fine) return big.slice(0, 24).map(t => t.name);
  const prefer = ['q_proj', 'gate_proj', 'down_proj'];
  const out = [];
  for (const k of prefer) {
    const hit = big.find(t => t.name.includes(k + '.'));
    if (hit && !out.includes(hit.name)) out.push(hit.name);
  }
  for (const t of big) {
    if (out.length >= 3) break;
    if (!out.includes(t.name)) out.push(t.name);
  }
  return out;
}


export default function LiveStage({ conn, scan, scanState, modelSlot, onOpenNeuron, onOpenParam }) {
  const cvRef = useRef(null);
  const uiRef = useRef(null);

  const [layers, setLayers] = useState([]);       // [{layer, tensors}]
  const [vals, setVals] = useState({});            // layer → {name: stats}
  const [cursor, setCursor] = useState(-1);        // 当前层（-1 未开始）
  const [running, setRunning] = useState(false);
  const [loadingL, setLoadingL] = useState(-1);
  const [err, setErr] = useState('');
  const [sel, setSel] = useState(-1);              // 点选层
  const [fine, setFine] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [autoRot, setAutoRot] = useState(true);
  const [view, setView] = useState('tower');       // 显示效果（VIEWS.id）

  const runRef = useRef(false);                    // 运行循环开关（避免闭包读到旧 state）
  const abortRef = useRef(null);
  const cvState = useRef({ yaw: 0.62, pitch: 0.30, dist: 30, panX: 0, panY: 0, spin: 0, t: 0 });
  const dragRef = useRef(null);
  const hitRef = useRef([]);                       // 每帧层盘命中区 {layer, sx, sy, r}

  /* conn 变化 → 层索引（层数/格数全部来自真实权重头，不硬编码 36） */
  useEffect(() => {
    setLayers(buildLayerIndex(conn));
    setVals({}); setCursor(-1); setSel(-1); setErr('');
    runRef.current = false; setRunning(false);
  }, [conn]);

  const layerNos = layers.map(x => x.layer);
  const total = layerNos.length;
  const maxL = total ? layerNos[total - 1] : -1;

  /* 全局归一化区间（只用真实取回的值；无值时不归一） */
  let vMin = Infinity, vMax = -Infinity;
  Object.values(vals).forEach(bag => Object.values(bag).forEach(s => {
    if (typeof s.rms === 'number') { vMin = Math.min(vMin, s.rms); vMax = Math.max(vMax, s.rms); }
  }));
  const hasScale = Number.isFinite(vMin) && Number.isFinite(vMax) && vMax > vMin;
  const norm = v => hasScale ? (v - vMin) / (vMax - vMin) : 0.5;

  /* 绘制所需快照进 ref（渲染循环不随 state 重建） */
  uiRef.current = {
    layers, vals, cursor, sel, total, maxL, hasScale, norm, autoRot, loadingL, view,
    scan: scan && scan.layers ? scan.layers : null, scanState,
  };

  /* 切换视图：同时套用该视图的相机预设（旋转/缩放仍可手动调） */
  const changeView = useCallback(id => {
    setView(id);
    const v = VIEWS.find(x => x.id === id);
    if (v && v.cam) {
      const M = cvState.current;
      M.yaw = v.cam.yaw; M.pitch = v.cam.pitch; M.dist = v.cam.dist;
    }
  }, []);

  /* ── 取一层真实数值 ─────────────────────────────────────── */
  const fetchLayer = useCallback(async (l) => {
    const li = layers.find(x => x.layer === l);
    if (!li) return false;
    const names = pickProbes(li.tensors, fine);
    if (!names.length) return false;
    setLoadingL(l);
    try {
      const ab = new AbortController();
      abortRef.current = ab;
      const r = await fetch(`${API_BASE}/api/live/weights/summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ names }),
        signal: ab.signal,
      });
      if (!r.ok) {
        const t = await r.json().catch(() => ({}));
        throw new Error(t.detail || ('HTTP ' + r.status));
      }
      const d = await r.json();
      setVals(v => ({ ...v, [l]: d.results || {} }));
      return true;
    } catch (e) {
      if (String(e.name) !== 'AbortError') {
        setErr(String(e.message || e).includes('Failed to fetch')
          ? '中心节点不可达（:5001）' : String(e.message || e));
      }
      return false;
    } finally {
      setLoadingL(-1);
      abortRef.current = null;
    }
  }, [layers, fine]);

  /* ── 运行 / 暂停 ────────────────────────────────────────── */
  const start = useCallback(async () => {
    if (!total) return;
    runRef.current = true; setRunning(true); setErr('');
    let i = cursor < 0 ? 0 : cursor;
    while (runRef.current && i < total) {
      const l = layerNos[i];
      if (!vals[l]) {
        const ok = await fetchLayer(l);
        if (!ok && !runRef.current) break;
      }
      if (!runRef.current) break;
      setCursor(l);
      i += 1;
      await new Promise(res => setTimeout(res, Math.max(90, 460 / Math.max(0.25, speed))));
    }
    if (i >= total) runRef.current = false;
    setRunning(!!runRef.current);
  }, [total, cursor, layerNos, vals, fetchLayer, speed]);

  const pause = useCallback(() => {
    runRef.current = false; setRunning(false);
    if (abortRef.current) abortRef.current.abort();
  }, []);

  const stepOnce = useCallback(async () => {
    if (!total) return;
    runRef.current = false; setRunning(false);
    const i = cursor < 0 ? 0 : Math.min(total - 1, layerNos.indexOf(cursor) + 1);
    const l = layerNos[i];
    if (!vals[l]) await fetchLayer(l);
    setCursor(l);
  }, [total, cursor, layerNos, vals, fetchLayer]);

  const reset = useCallback(() => {
    runRef.current = false; setRunning(false); setCursor(-1); setSel(-1);
    setVals({}); setErr('');
  }, []);

  /* 卸载时打断在途请求 */
  useEffect(() => () => { runRef.current = false; if (abortRef.current) abortRef.current.abort(); }, []);

  /* ── 渲染循环 ───────────────────────────────────────────── */
  useEffect(() => {
    const cv = cvRef.current;
    if (!cv) return undefined;
    const ctx = cv.getContext('2d');
    const M = cvState.current;
    let raf = 0;

    const dpr = () => window.devicePixelRatio || 1;
    const resize = () => {
      const d = dpr();
      cv.width = Math.max(1, cv.clientWidth * d);
      cv.height = Math.max(1, cv.clientHeight * d);
    };
    resize();

    /* 投影：yaw 绕 Y、pitch 绕 X、透视除法
       焦距标定：dist=30 时约 2.7×画布高（device px）/ 世界单位，
       即 28 层塔（世界高 ~8.1）约占 65% 画面；滚轮改 dist 即等比缩放。
       注：cv.height 已是设备像素，不能再乘 dpr()。                      */
    function proj(x, y, z) {
      const cy = Math.cos(M.yaw), sy = Math.sin(M.yaw);
      const cp = Math.cos(M.pitch), sp = Math.sin(M.pitch);
      const x1 = x * cy - z * sy;
      const z1 = x * sy + z * cy;
      const y1 = y * cp - z1 * sp;
      const z2 = y * sp + z1 * cp;
      const d = M.dist + z2;
      const f = (cv.height * 2.7) / Math.max(2, d);
      return {
        sx: cv.width / 2 + M.panX * dpr() + x1 * f,
        sy: cv.height * 0.5 + M.panY * dpr() - y1 * f,
        depth: z2, f,
      };
    }

    function poly(pts, fill, stroke, lw) {
      ctx.beginPath();
      pts.forEach((q, i) => (i ? ctx.lineTo(q.sx, q.sy) : ctx.moveTo(q.sx, q.sy)));
      ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = (lw || 1) * dpr(); ctx.stroke(); }
    }

    function glowDot(p, r, col, a) {
      ctx.fillStyle = rgba(col, 0.16 * a);
      ctx.beginPath(); ctx.arc(p.sx, p.sy, r * 2.9, 0, 6.2832); ctx.fill();
      ctx.fillStyle = rgba(col, 0.42 * a);
      ctx.beginPath(); ctx.arc(p.sx, p.sy, r * 1.65, 0, 6.2832); ctx.fill();
      ctx.fillStyle = rgba(col, 0.95 * a);
      ctx.beginPath(); ctx.arc(p.sx, p.sy, r, 0, 6.2832); ctx.fill();
    }

    /* ── 共用：几何与强度（未取数一律 null，绝不填假值） ── */
    function geo(S) {
      const n = S.total;
      const H = n ? Math.min(0.30, 9.6 / n) : 0.30;   // 层距自适应（36 层→0.267）
      return { n, H, top: n ? ((n - 1) / 2) * H : 0 };
    }
    function lvOf(S, l) {
      const bag = S.vals[l];
      if (!bag) return null;
      let s = 0, c = 0;
      Object.values(bag).forEach(x => {
        if (typeof x.rms === 'number') { s += S.norm(x.rms); c += 1; }
      });
      return c ? s / c : null;
    }
    function cellV(S, l, nm) {
      const bag = S.vals[l];
      const st = bag && nm ? bag[nm] : null;
      return st && typeof st.rms === 'number' ? S.norm(st.rms) : null;
    }
    /* 层号标签：统一排版口径（fs = 设备像素字号） */
    function layerLabel(l, x, y, fs, strong) {
      ctx.globalAlpha = strong ? 0.98 : 0.7;
      ctx.fillStyle = strong ? '#4338ca' : '#64748b';
      ctx.font = `600 ${fs}px ui-monospace,Consolas,monospace`;
      ctx.textAlign = 'center';
      ctx.fillText('L' + l, x, y);
      ctx.textAlign = 'left';
      ctx.globalAlpha = 1;
    }

    /* ═══ 视图 ①：光塔 · 层盘环 ═════════════════════════════ */
    function drawTower(S, d) {
      const { n, H, top: TOP } = geo(S);
      const yOf = l => {
        if (!n) return 0;
        const i = S.layers.findIndex(x => x.layer === l);
        return (i - (n - 1) / 2) * H;
      };

      /* ① 背景：极淡的同心环地面，给空间参照（不承载数据） */
      for (let k = 1; k <= 3; k += 1) {
        const r = k * 1.35;
        const pts = [];
        for (let a = 0; a < 48; a += 1) {
          const th = (a / 48) * 6.2832;
          pts.push(proj(Math.cos(th) * r, -TOP - H * 1.1, Math.sin(th) * r));
        }
        poly(pts, null, 'rgba(148,163,184,.10)', 1);
      }

      /* ② 中心残差流立柱：自底向上，越靠上越亮（信息流方向） */
      const cBot = proj(0, -TOP - H * 0.9, 0);
      const cTop = proj(0, TOP + H * 0.9, 0);
      const colGrad = ctx.createLinearGradient(cBot.sx, cBot.sy, cTop.sx, cTop.sy);
      colGrad.addColorStop(0, 'rgba(148,163,184,.16)');
      colGrad.addColorStop(0.55, 'rgba(14,165,233,.44)');
      colGrad.addColorStop(1, 'rgba(217,70,239,.62)');
      ctx.strokeStyle = colGrad;
      ctx.lineWidth = 5.2 * d;
      ctx.beginPath(); ctx.moveTo(cBot.sx, cBot.sy); ctx.lineTo(cTop.sx, cTop.sy); ctx.stroke();
      ctx.strokeStyle = 'rgba(255,255,255,.85)';
      ctx.lineWidth = 1.3 * d;
      ctx.beginPath(); ctx.moveTo(cBot.sx, cBot.sy); ctx.lineTo(cTop.sx, cTop.sy); ctx.stroke();

      /* ③ 层盘（远→近排序） */
      const sorted = [...S.layers].map(x => {
        const y = yOf(x.layer);
        return { ...x, y, q: proj(0, y, 0) };
      }).sort((a, b) => a.q.depth - b.q.depth);

      const valsAll = S.vals;

      sorted.forEach(({ layer: l, y, tensors }) => {
        const bag = valsAll[l];
        const isCur = l === S.cursor;
        const isSel = l === S.sel;
        const rings = Math.max(1, tensors.length);
        const Ri = 1.34, Ro = 2.02 + (isSel ? 0.10 : 0);
        const TH = 0.055;

        /* 该层盘的强度：已取数则用各 probe rms 的均值归一；未取数 = null（暗态） */
        const lvN = lvOf(S, l);
        const known = lvN != null;
        const lv = lvN || 0;

        /* 已扫过的层保留余晖；未扫过的层是冷暗轮廓 */
        const baseA = isSel ? 0.96 : isCur ? 0.90 : known ? 0.58 : 0.20;

        for (let i = 0; i < rings; i += 1) {
          const a0 = (i / rings) * 6.2832 + 0.035;
          const a1 = ((i + 1) / rings) * 6.2832 - 0.035;
          const tName = tensors[i] && tensors[i].name;
          const sv = cellV(S, l, tName);
          const segV = sv != null ? sv : (known ? lv * 0.55 : 0);

          const col = sv != null ? rampRGB(segV) : (known ? [148, 163, 184] : [203, 213, 225]);
          const alphaTop = sv != null ? 0.20 + segV * 0.52 : (known ? 0.14 : 0.09);

          /* 顶面四点 */
          const yi = y + TH;
          const pIn0 = proj(Math.cos(a0) * Ri, yi, Math.sin(a0) * Ri);
          const pOut0 = proj(Math.cos(a0) * Ro, yi, Math.sin(a0) * Ro);
          const pOut1 = proj(Math.cos(a1) * Ro, yi, Math.sin(a1) * Ro);
          const pIn1 = proj(Math.cos(a1) * Ri, yi, Math.sin(a1) * Ri);
          const topPts = [pIn0, pOut0, pOut1, pIn1];

          /* 外侧面 */
          const qOut0 = proj(Math.cos(a0) * Ro, y - TH, Math.sin(a0) * Ro);
          const qOut1 = proj(Math.cos(a1) * Ro, y - TH, Math.sin(a1) * Ro);
          const sidePts = [pOut0, pOut1, qOut1, qOut0];

          poly(sidePts, rgba(col, alphaTop * 0.62), null, 0);
          poly(topPts, rgba(col, alphaTop * baseA), rgba(col, Math.min(0.95, baseA + 0.1)), 0.8);

          /* 已取数的格：格心画一颗强度点（亮度∝真实值） */
          if (sv != null) {
            const am = (a0 + a1) / 2;
            const rr = (Ri + Ro) / 2;
            const pc = proj(Math.cos(am) * rr, yi + 0.012, Math.sin(am) * rr);
            glowDot(pc, Math.max(2.0, 3.2 * (0.4 + segV)) * d, col, 0.55 + segV * 0.45);
          }
        }

        /* 选中层：柱阵展开（每格一根柱，高度∝真实 rms） */
        if (isSel && bag) {
          tensors.forEach((t, i) => {
            const st = bag[t.name];
            if (!st || typeof st.rms !== 'number') return;
            const am = ((i + 0.5) / rings) * 6.2832;
            const rr = (Ri + Ro) / 2;
            const hv = 0.10 + S.norm(st.rms) * 1.15;
            const x = Math.cos(am) * rr, z = Math.sin(am) * rr;
            const b0 = proj(x, y + TH, z);
            const b1 = proj(x, y + TH + hv, z);
            const col = rampRGB(S.norm(st.rms));
            const g2 = ctx.createLinearGradient(b0.sx, b0.sy, b1.sx, b1.sy);
            g2.addColorStop(0, rgba(col, 0.30));
            g2.addColorStop(1, rgba(col, 0.95));
            ctx.strokeStyle = g2;
            ctx.lineWidth = 4.2 * d;
            ctx.beginPath(); ctx.moveTo(b0.sx, b0.sy); ctx.lineTo(b1.sx, b1.sy); ctx.stroke();
            glowDot(b1, 3.2 * d, col, 0.95);
          });
        }

        /* 当前层：旋转扫描环 + 层号数值 */
        if (isCur) {
          const rr = Ro + 0.30;
          const segs = 40;
          ctx.lineWidth = 2.0 * d;
          for (let s = 0; s < segs; s += 1) {
            const a0 = (s / segs) * 6.2832 + M.spin * 0.9;
            const a1 = ((s + 1) / segs) * 6.2832 + M.spin * 0.9;
            const fade = 0.10 + 0.75 * ((s / segs + (M.spin * 0.12) % 1) % 1) ** 2;
            ctx.strokeStyle = rgba([14, 165, 233], fade);
            const p0 = proj(Math.cos(a0) * rr, y + TH, Math.sin(a0) * rr);
            const p1 = proj(Math.cos(a1) * rr, y + TH, Math.sin(a1) * rr);
            ctx.beginPath(); ctx.moveTo(p0.sx, p0.sy); ctx.lineTo(p1.sx, p1.sy); ctx.stroke();
          }
          /* 向上的光锥（正在"运行"的层） */
          const apex = proj(0, y + TH + 0.55, 0);
          const halo = ctx.createRadialGradient(apex.sx, apex.sy, 0, apex.sx, apex.sy, 34 * d);
          halo.addColorStop(0, 'rgba(14,165,233,.34)');
          halo.addColorStop(1, 'rgba(14,165,233,0)');
          ctx.fillStyle = halo;
          ctx.beginPath(); ctx.arc(apex.sx, apex.sy, 34 * d, 0, 6.2832); ctx.fill();
        }

        /* 层号：盘中心下方 */
        const qc = proj(0, y - TH, 0);
        if (qc.f > 10) {
          ctx.globalAlpha = isCur || isSel ? 0.98 : known ? 0.72 : 0.42;
          ctx.fillStyle = isSel ? '#4338ca' : isCur ? '#0369a1' : '#64748b';
          ctx.font = `600 ${Math.max(8, Math.min(12, qc.f * 0.16)) * d}px ui-monospace,Consolas,monospace`;
          ctx.textAlign = 'center';
          ctx.fillText('L' + l, qc.sx, qc.sy + 3.4 * d);
          ctx.textAlign = 'left';
          ctx.globalAlpha = 1;
        }

        /* 命中区（屏幕近似半径） */
        const rpx = Math.abs(proj(Ro, y, 0).sx - proj(-Ro, y, 0).sx) / 2;
        hitRef.current.push({ layer: l, sx: qc.sx, sy: qc.sy, r: Math.max(10 * d, rpx) });
      });

      /* ④ 粒子流：沿塔轴上升，速度∝当前层强度（真实值的可视化，非随机装饰） */
      const P = 46;
      for (let k = 0; k < P; k += 1) {
        const seed = (k * 0.6180339887) % 1;
        const ph = ((M.t * 0.0022 * (0.55 + seed) + seed) % 1);
        const yy = -TOP - H * 0.9 + ph * (TOP * 2 + H * 1.8);
        const rad = 0.18 + seed * 0.95;
        const ang = seed * 6.2832 + M.t * 0.004 + ph * 2.2;
        const p = proj(Math.cos(ang) * rad * 0.35, yy, Math.sin(ang) * rad * 0.35);
        const fade = Math.sin(ph * Math.PI);
        const col = ph < 0.5 ? [14, 165, 233] : [217, 70, 239];
        glowDot(p, Math.max(1.4, 2.6 * fade) * d, col, 0.55 * fade);
      }


    }

    /* ═══ 视图 ②：螺旋梯 · 螺旋光带 ═══════════════════════════
       同样的层与格，改成「逐层旋进」的排布：每层盘整体转过 STEP，
       相邻层外缘用光带相连，形成一条贯穿全模型的螺旋脊。       */
    function drawHelix(S, d) {
      const { n, H, top: TOP } = geo(S);
      if (!n) return;
      const Ri = 1.62, Ro = 2.16, TH = 0.048, STEP = 0.44;
      const yOf = i => (i - (n - 1) / 2) * H;

      /* 地面参照环 */
      for (let k = 1; k <= 2; k += 1) {
        const rr = k * 1.55, pts = [];
        for (let a = 0; a < 48; a += 1) {
          const th = (a / 48) * 6.2832;
          pts.push(proj(Math.cos(th) * rr, -TOP - H, Math.sin(th) * rr));
        }
        poly(pts, null, 'rgba(148,163,184,.09)', 1);
      }

      /* 螺旋脊：相邻层外缘连线（28 段弦逼近螺旋曲线） */
      const spine = S.layers.map((x, i) => proj(
        Math.cos(i * STEP) * Ro, yOf(i), Math.sin(i * STEP) * Ro));
      for (let i = 0; i < spine.length - 1; i += 1) {
        const a = spine[i], b = spine[i + 1];
        const hot = S.layers[i].layer === S.cursor;
        const g = ctx.createLinearGradient(a.sx, a.sy, b.sx, b.sy);
        g.addColorStop(0, hot ? 'rgba(14,165,233,.88)' : 'rgba(14,165,233,.14)');
        g.addColorStop(1, hot ? 'rgba(217,70,239,.88)' : 'rgba(217,70,239,.32)');
        ctx.strokeStyle = g;
        ctx.lineWidth = (hot ? 2.8 : 1.5) * d;
        ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
      }

      /* 层环（远→近） */
      const rows = S.layers.map((x, i) => ({ ...x, y: yOf(i), off: i * STEP }))
        .map(o => ({ ...o, q: proj(0, o.y, 0) }))
        .sort((a, b) => a.q.depth - b.q.depth);

      rows.forEach(({ layer: l, tensors, y, off }) => {
        const known = lvOf(S, l) != null;
        const isCur = l === S.cursor, isSel = l === S.sel;
        const rings = Math.max(1, tensors.length);
        const baseA = isSel ? 0.98 : isCur ? 0.94 : known ? 0.62 : 0.22;

        for (let j = 0; j < rings; j += 1) {
          const nm = tensors[j] && tensors[j].name;
          const sv = cellV(S, l, nm);
          const col = sv != null ? rampRGB(sv) : [148, 163, 184];
          const alpha = (sv != null ? 0.16 + sv * 0.58 : known ? 0.12 : 0.07) * baseA;
          const a0 = off + (j / rings) * 6.2832 + 0.03;
          const a1 = off + ((j + 1) / rings) * 6.2832 - 0.03;
          const pIn0 = proj(Math.cos(a0) * Ri, y + TH, Math.sin(a0) * Ri);
          const pOut0 = proj(Math.cos(a0) * Ro, y + TH, Math.sin(a0) * Ro);
          const pOut1 = proj(Math.cos(a1) * Ro, y + TH, Math.sin(a1) * Ro);
          const pIn1 = proj(Math.cos(a1) * Ri, y + TH, Math.sin(a1) * Ri);
          const qOut0 = proj(Math.cos(a0) * Ro, y - TH, Math.sin(a0) * Ro);
          const qOut1 = proj(Math.cos(a1) * Ro, y - TH, Math.sin(a1) * Ro);

          poly([pOut0, pOut1, qOut1, qOut0], rgba(col, alpha * 0.6), null, 0);
          poly([pIn0, pOut0, pOut1, pIn1], rgba(col, alpha),
            sv != null ? rgba(col, Math.min(0.95, baseA + 0.08)) : 'rgba(148,163,184,.26)', 0.7);

          if (sv != null) {
            const am = (a0 + a1) / 2, rr = (Ri + Ro) / 2;
            const pc = proj(Math.cos(am) * rr, y + TH + 0.01, Math.sin(am) * rr);
            glowDot(pc, Math.max(1.8, 2.8 * (0.35 + sv)) * d, col, 0.5 + sv * 0.5);
          }
        }

        /* 选中层：从环上向外斜升光柱（柱高 = 真实 rms） */
        if (isSel) {
          tensors.forEach((t, j) => {
            const sv = cellV(S, l, t.name);
            if (sv == null) return;
            const am = off + ((j + 0.5) / rings) * 6.2832;
            const r0 = Ro + 0.05, hv = 0.28 + sv * 1.75;
            const b0 = proj(Math.cos(am) * r0, y + TH, Math.sin(am) * r0);
            const b1 = proj(Math.cos(am) * (r0 + hv * 0.42), y + TH + hv * 0.9,
              Math.sin(am) * (r0 + hv * 0.42));
            const col = rampRGB(sv);
            const g2 = ctx.createLinearGradient(b0.sx, b0.sy, b1.sx, b1.sy);
            g2.addColorStop(0, rgba(col, 0.24));
            g2.addColorStop(1, rgba(col, 0.95));
            ctx.strokeStyle = g2;
            ctx.lineWidth = 4.0 * d;
            ctx.beginPath(); ctx.moveTo(b0.sx, b0.sy); ctx.lineTo(b1.sx, b1.sy); ctx.stroke();
            glowDot(b1, 3.2 * d, col, 0.95);
          });
        }

        /* 当前层：扫描环 */
        if (isCur) {
          const rr = Ro + 0.26, segs = 36;
          for (let s = 0; s < segs; s += 1) {
            const a0 = (s / segs) * 6.2832 + M.spin;
            const a1 = ((s + 1) / segs) * 6.2832 + M.spin;
            const fd = 0.08 + 0.8 * ((s / segs + (M.spin * 0.14) % 1) % 1) ** 2;
            ctx.strokeStyle = rgba([14, 165, 233], fd);
            ctx.lineWidth = 2.0 * d;
            const p0 = proj(Math.cos(a0) * rr, y + TH, Math.sin(a0) * rr);
            const p1 = proj(Math.cos(a1) * rr, y + TH, Math.sin(a1) * rr);
            ctx.beginPath(); ctx.moveTo(p0.sx, p0.sy); ctx.lineTo(p1.sx, p1.sy); ctx.stroke();
          }
        }

        /* 层号：螺旋外侧 */
        const ql = proj(Math.cos(off) * (Ro + 0.44), y, Math.sin(off) * (Ro + 0.44));
        if (ql.f > 8) {
          layerLabel(l, ql.sx, ql.sy, Math.max(8, Math.min(11.5, ql.f * 0.15)) * d, isCur || isSel);
        }

        const rc = Math.abs(proj(Ro, y, 0).sx - proj(-Ro, y, 0).sx) / 2;
        hitRef.current.push({ layer: l, sx: ql.sx, sy: ql.sy, r: Math.max(10 * d, rc) });
      });
    }

    /* ═══ 视图 ③：地层矩阵 · 3D 柱阵 ═══════════════════════════
       行 = 层（纵深方向），列 = 层内 tensor，柱高 = 归一化真实 rms。
       这是唯一「数值高度即几何高度」的视图，一眼看出层内分布。   */
    function drawMatrix(S, d) {
      const { n } = geo(S);
      if (!n) return;
      const cols = Math.max(1, ...S.layers.map(x => x.tensors.length));
      const CW = Math.min(0.86, 8.2 / cols);          // 列距
      const DW = Math.min(0.60, 11.0 / n);            // 行距（纵深）
      const w = CW * 0.36, dz = Math.max(0.05, DW * 0.30);
      const hv = v => 0.10 + v * 1.55;

      /* 基准面网格（只做空间参照） */
      const xL = -(cols * CW) / 2 - CW * 0.25, xR = -xL;
      const zB = -(n * DW) / 2 - DW * 0.2, zF = -zB;
      const rowStep = n > 16 ? 2 : 1;
      for (let r = 0; r <= n; r += rowStep) {
        const z = -(n * DW) / 2 + r * DW;
        const p0 = proj(xL, 0, z), p1 = proj(xR, 0, z);
        ctx.strokeStyle = 'rgba(148,163,184,.13)';
        ctx.lineWidth = 1 * d;
        ctx.beginPath(); ctx.moveTo(p0.sx, p0.sy); ctx.lineTo(p1.sx, p1.sy); ctx.stroke();
      }
      for (let c = 0; c <= cols; c += 1) {
        const x = -(cols * CW) / 2 + c * CW;
        const p0 = proj(x, 0, zB), p1 = proj(x, 0, zF);
        ctx.strokeStyle = 'rgba(148,163,184,.08)';
        ctx.lineWidth = 1 * d;
        ctx.beginPath(); ctx.moveTo(p0.sx, p0.sy); ctx.lineTo(p1.sx, p1.sy); ctx.stroke();
      }

      /* 行 = 层：由远及近绘制 */
      const rows = S.layers.map((x, i) => ({ ...x, z: (i - (n - 1) / 2) * DW }))
        .map(o => ({ ...o, q: proj(0, 0, o.z) }))
        .sort((a, b) => a.q.depth - b.q.depth);

      rows.forEach(({ layer: l, tensors, z }) => {
        const known = lvOf(S, l) != null;
        const isCur = l === S.cursor, isSel = l === S.sel;
        const baseA = isSel ? 1 : isCur ? 0.96 : known ? 0.9 : 0.55;

        /* 当前 / 选中行：底面高亮带 */
        if (isCur || isSel) {
          const p0 = proj(xL, 0.012, z - dz * 1.7), p1 = proj(xL, 0.012, z + dz * 1.7);
          const p2 = proj(xR, 0.012, z + dz * 1.7), p3 = proj(xR, 0.012, z - dz * 1.7);
          poly([p0, p1, p2, p3], isSel ? 'rgba(99,102,241,.10)' : 'rgba(14,165,233,.09)',
            isSel ? 'rgba(99,102,241,.45)' : 'rgba(14,165,233,.42)', 0.8);
        }

        tensors.forEach((t, j) => {
          const sv = cellV(S, l, t.name);
          const x = (j - (cols - 1) / 2) * CW;
          const h = sv == null ? 0.015 : hv(sv);
          const c0 = proj(x - w, h, z - dz), c1 = proj(x + w, h, z - dz);
          const c2 = proj(x + w, h, z + dz), c3 = proj(x - w, h, z + dz);

          if (sv != null) {
            const col = rampRGB(sv);
            const b0 = proj(x, 0, z);
            const mid = { sx: (c0.sx + c2.sx) / 2, sy: (c0.sy + c2.sy) / 2 };
            const g = ctx.createLinearGradient(b0.sx, b0.sy, mid.sx, mid.sy);
            g.addColorStop(0, rgba(col, 0.08 * baseA));
            g.addColorStop(1, rgba(col, 0.66 * baseA));
            ctx.strokeStyle = g;
            ctx.lineWidth = Math.max(1.1 * d, Math.abs(c1.sx - c0.sx) * 0.40);
            ctx.beginPath(); ctx.moveTo(b0.sx, b0.sy); ctx.lineTo(mid.sx, mid.sy); ctx.stroke();
            poly([c0, c1, c2, c3], rgba(col, 0.26 + 0.5 * baseA),
              rgba(col, 0.70 + 0.22 * baseA), 0.7);
          } else {
            poly([c0, c1, c2, c3], 'rgba(203,213,225,.20)', 'rgba(148,163,184,.26)', 0.6);
          }
        });

        /* 层号：行左端 */
        const ql = proj(xL - CW * 0.5, 0, z);
        if (ql.f > 7) {
          layerLabel(l, ql.sx, ql.sy, Math.max(7.5, Math.min(11, ql.f * 0.13)) * d, isCur || isSel);
        }

        const qc = proj(0, 0, z);
        const halfW = Math.abs(proj((cols * CW) / 2, 0, z).sx
          - proj(-(cols * CW) / 2, 0, z).sx) / 2;
        hitRef.current.push({ layer: l, sx: qc.sx, sy: qc.sy, r: Math.max(10 * d, halfW) });
      });
    }

    /* ═══ 视图 ④：星核 · 同心陀螺球 ═══════════════════════════
       层 = 同心球壳（半径随层号递增），每壳用 2–3 个不同倾角的大圆，
       弧段色 = 该格真实 rms。内层更实、外层更透，读起来像洋葱剖面。 */
    function drawOrb(S, d) {
      const { n } = geo(S);
      if (!n) return;
      const R0 = 0.85, R1 = 3.45;
      const shells = n > 22 ? 2 : 3;

      /* 外圈参照 */
      for (let k = 1; k <= 2; k += 1) {
        const rr = R1 + 0.35 * k, pts = [];
        for (let a = 0; a < 56; a += 1) {
          const th = (a / 56) * 6.2832;
          pts.push(proj(Math.cos(th) * rr, 0, Math.sin(th) * rr));
        }
        poly(pts, null, 'rgba(148,163,184,.07)', 1);
      }

      const pointAt = (a, rr, tilt, spin) => {
        const px = Math.cos(a) * rr, pz = Math.sin(a) * rr;
        const yy = pz * Math.sin(tilt), zz = pz * Math.cos(tilt);
        return proj(px * Math.cos(spin) - zz * Math.sin(spin), yy,
          px * Math.sin(spin) + zz * Math.cos(spin));
      };

      const list = S.layers.map((x, i) => ({ ...x, i, r: R0 + (i / Math.max(1, n - 1)) * (R1 - R0) }))
        .sort((a, b) => b.r - a.r);                    // 外层先画（更淡）

      list.forEach(({ layer: l, tensors, i, r }) => {
        const known = lvOf(S, l) != null;
        const isCur = l === S.cursor, isSel = l === S.sel;
        const rings = Math.max(6, tensors.length);
        const depthK = 0.35 + 0.65 * (1 - i / Math.max(1, n));   // 内层更实
        const baseA = (isSel ? 1 : isCur ? 0.96 : known ? 0.72 : 0.3) * depthK;

        for (let k = 0; k < shells; k += 1) {
          const tilt = 0.55 + k * 0.95 + i * 0.10;
          const spin = i * 0.19 + k * 1.02;
          const rot = M.spin * (0.10 + 0.06 * k);
          for (let j = 0; j < rings; j += 1) {
            const nm = tensors[j] && tensors[j].name;
            const sv = cellV(S, l, nm);
            const a0 = (j / rings) * 6.2832 + rot;
            const a1 = ((j + 1) / rings) * 6.2832 + rot;
            const p0 = pointAt(a0, r, tilt, spin), p1 = pointAt(a1, r, tilt, spin);
            const col = sv != null ? rampRGB(sv) : [148, 163, 184];
            ctx.strokeStyle = rgba(col, sv != null ? (0.16 + sv * 0.6) * baseA : 0.1 * baseA);
            ctx.lineWidth = (isSel ? 3.0 : isCur ? 2.6 : 1.5 * (0.7 + depthK * 0.5)) * d;
            ctx.beginPath(); ctx.moveTo(p0.sx, p0.sy); ctx.lineTo(p1.sx, p1.sy); ctx.stroke();
            if (sv != null && (isSel || isCur || sv > 0.8)) {
              glowDot(p0, Math.max(1.6, 2.6 * (0.3 + sv)) * d, col, 0.42 + sv * 0.4);
            }
          }
        }

        /* 选中层：每格沿壳法向拉出一根光针（长度 = 真实 rms） */
        if (isSel) {
          const tilt = 0.55, spin = i * 0.19;
          tensors.forEach((t, j) => {
            const sv = cellV(S, l, t.name);
            if (sv == null) return;
            const a = ((j + 0.5) / rings) * 6.2832 + M.spin * 0.1;
            const b0 = pointAt(a, r, tilt, spin);
            const b1 = pointAt(a, r + 0.22 + sv * 1.05, tilt, spin);
            const col = rampRGB(sv);
            const g2 = ctx.createLinearGradient(b0.sx, b0.sy, b1.sx, b1.sy);
            g2.addColorStop(0, rgba(col, 0.2));
            g2.addColorStop(1, rgba(col, 0.95));
            ctx.strokeStyle = g2;
            ctx.lineWidth = 3.6 * d;
            ctx.beginPath(); ctx.moveTo(b0.sx, b0.sy); ctx.lineTo(b1.sx, b1.sy); ctx.stroke();
            glowDot(b1, 3.0 * d, col, 0.95);
          });
        }

        /* 层号：球壳赤道起点 */
        const ql = proj(Math.cos(i * 0.19) * r, 0, Math.sin(i * 0.19) * r);
        if (ql.f > 7) {
          layerLabel(l, ql.sx, ql.sy, Math.max(7.5, Math.min(11, ql.f * 0.13)) * d, isCur || isSel);
        }
        hitRef.current.push({ layer: l, sx: ql.sx, sy: ql.sy, r: 14 * d });
      });
    }

    /* ═══ 视图 ⑤：粒子星河 · 旋涡 ═══════════════════════════════
       每个已取数格 = 一颗彩色粒子（大小/亮度 = 真实 rms）；
       未取数格 = 极暗的灰尘，只表示「这里有一个 tensor」，不表示数值。 */
    function drawNebula(S, d) {
      const { n, H } = geo(S);
      if (!n) return;
      const yOf = i => (i - (n - 1) / 2) * H * 0.66;
      const radOf = i => 0.75 + (i / Math.max(1, n - 1)) * 2.5;
      const swirl = M.t * 0.0016;

      /* 旋臂（纯空间参照，不承载数值） */
      for (let arm = 0; arm < 3; arm += 1) {
        const pts = [];
        for (let s = 0; s <= 40; s += 1) {
          const t = s / 40;
          const a = arm * 2.0944 + t * 3.1 + swirl;
          const rr = 0.6 + t * 3.4;
          pts.push(proj(Math.cos(a) * rr, (t - 0.5) * 2.2, Math.sin(a) * rr));
        }
        poly(pts, null, 'rgba(148,163,184,.10)', 1);
      }

      S.layers.forEach((x, i) => {
        const l = x.layer;
        const y = yOf(i), rad = radOf(i);
        const known = lvOf(S, l) != null;
        const isCur = l === S.cursor, isSel = l === S.sel;
        const rings = Math.max(1, x.tensors.length);
        const spin = i * 0.30 + swirl * (0.5 + i / Math.max(1, n));

        x.tensors.forEach((t, j) => {
          const sv = cellV(S, l, t.name);
          const a = spin + (j / rings) * 6.2832;
          const p = proj(Math.cos(a) * rad, y, Math.sin(a) * rad);
          if (sv != null) {
            glowDot(p, Math.max(1.8, (1.6 + sv * 3.2) * (isSel ? 1.45 : 1)) * d,
              rampRGB(sv), 0.55 + sv * 0.45);
          } else {
            ctx.fillStyle = known ? 'rgba(148,163,184,.30)' : 'rgba(148,163,184,.14)';
            ctx.beginPath(); ctx.arc(p.sx, p.sy, 0.9 * d, 0, 6.2832); ctx.fill();
          }
        });

        /* 当前 / 选中层：发光轨道 */
        if (isCur || isSel) {
          const segs = 48, rr = rad + 0.12;
          for (let s = 0; s < segs; s += 1) {
            const a0 = (s / segs) * 6.2832 + M.spin * 0.6;
            const a1 = ((s + 1) / segs) * 6.2832 + M.spin * 0.6;
            const fd = 0.06 + 0.75 * ((s / segs + (M.spin * 0.1) % 1) % 1) ** 2;
            ctx.strokeStyle = isSel ? rgba([99, 102, 241], fd) : rgba([14, 165, 233], fd);
            ctx.lineWidth = 1.8 * d;
            const p0 = proj(Math.cos(a0) * rr, y, Math.sin(a0) * rr);
            const p1 = proj(Math.cos(a1) * rr, y, Math.sin(a1) * rr);
            ctx.beginPath(); ctx.moveTo(p0.sx, p0.sy); ctx.lineTo(p1.sx, p1.sy); ctx.stroke();
          }
        }

        const ql = proj(Math.cos(spin) * (rad + 0.34), y, Math.sin(spin) * (rad + 0.34));
        if (ql.f > 7) {
          layerLabel(l, ql.sx, ql.sy, Math.max(7.5, Math.min(11, ql.f * 0.13)) * d, isCur || isSel);
        }
        const rpx = Math.abs(proj(rad, 0, 0).sx - proj(-rad, 0, 0).sx) / 2;
        hitRef.current.push({ layer: l, sx: ql.sx, sy: ql.sy, r: 12 * d, ring: rpx });
      });
    }

    /* ═══ 视图注册表 + 渲染循环 ═══ */
    const RENDER = {
      tower: drawTower, helix: drawHelix, matrix: drawMatrix, orb: drawOrb, nebula: drawNebula,
    };

    function draw() {
      const S = uiRef.current;
      M.t += 1;
      M.spin += 0.0225;
      if (S.autoRot && !dragRef.current) M.yaw += 0.0016 * dpr();
      ctx.clearRect(0, 0, cv.width, cv.height);
      hitRef.current = [];
      (RENDER[S.view] || drawTower)(S, dpr());
      raf = requestAnimationFrame(draw);
    }
    raf = requestAnimationFrame(draw);

    /* ── 交互 ── */
    function pick(mx, my) {
      /* 第一优先：点中标签 / 盘面内的命中圆 */
      let best = null;
      for (const h of hitRef.current) {
        const dist = Math.hypot(mx - h.sx, my - h.sy);
        if (dist <= h.r) {
          if (!best || dist < best.dist) best = { layer: h.layer, dist };
        }
      }
      if (best) return best.layer;
      /* 第二优先：点中环带（星核 / 粒子星河的轨道环） */
      let ring = null;
      for (const h of hitRef.current) {
        if (!h.ring) continue;
        const dist = Math.abs(Math.hypot(mx - h.sx, my - h.sy) - h.ring);
        if (dist <= 13 * dpr() && (!ring || dist < ring.dist)) ring = { layer: h.layer, dist };
      }
      return ring ? ring.layer : -1;
    }

    const onMove = e => {
      const r = cv.getBoundingClientRect();
      const mx = (e.clientX - r.left) * (cv.width / r.width);
      const my = (e.clientY - r.top) * (cv.height / r.height);
      if (dragRef.current) {
        const dx = e.clientX - dragRef.current.x, dy = e.clientY - dragRef.current.y;
        if (dragRef.current.btn === 2) {
          M.panX = Math.max(-900, Math.min(900, M.panX + dx));
          M.panY = Math.max(-900, Math.min(900, M.panY + dy));
        } else {
          M.yaw += dx * 0.0062;
          M.pitch = Math.max(-0.28, Math.min(1.22, M.pitch + dy * 0.0052));
        }
        dragRef.current = { x: e.clientX, y: e.clientY, btn: dragRef.current.btn };
        return;
      }
      const l = pick(mx, my);
      cv.style.cursor = l >= 0 ? 'pointer' : 'grab';
    };
    const onDown = e => {
      if (e.button !== 0 && e.button !== 2) return;
      dragRef.current = { x: e.clientX, y: e.clientY, btn: e.button };
    };
    const onUp = e => {
      if (dragRef.current && dragRef.current.btn === 0) {
        const moved = Math.abs(e.clientX - dragRef.current.x) + Math.abs(e.clientY - dragRef.current.y);
        if (moved < 5) {
          const r = cv.getBoundingClientRect();
          const mx = (e.clientX - r.left) * (cv.width / r.width);
          const my = (e.clientY - r.top) * (cv.height / r.height);
          const l = pick(mx, my);
          setSel(prev => (prev === l ? -1 : l));
        }
      }
      dragRef.current = null;
    };
    const onWheel = e => {
      e.preventDefault();
      M.dist = Math.max(10, Math.min(90, M.dist + e.deltaY * 0.028));
    };
    const onCtx = e => e.preventDefault();
    const onResize = () => resize();

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    window.addEventListener('resize', onResize);
    cv.addEventListener('mousedown', onDown);
    cv.addEventListener('wheel', onWheel, { passive: false });
    cv.addEventListener('contextmenu', onCtx);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('resize', onResize);
      cv.removeEventListener('mousedown', onDown);
      cv.removeEventListener('wheel', onWheel);
      cv.removeEventListener('contextmenu', onCtx);
    };
  }, []);

  /* ── HUD 数据 ──────────────────────────────────────────── */
  const progress = total ? (cursor < 0 ? 0 : (layerNos.indexOf(cursor) + 1) / total) : 0;
  const curBag = cursor >= 0 ? vals[cursor] : null;
  const selBag = sel >= 0 ? vals[sel] : null;
  const selLayer = sel >= 0 ? layers.find(x => x.layer === sel) : null;
  const cfg = (conn && conn.config) || {};
  const curView = VIEWS.find(v => v.id === view) || VIEWS[0];

  /* 选中层的参数表（真实统计；未取数的行标「待取」） */
  const selRows = selLayer ? selLayer.tensors.slice(0, 16).map(t => {
    const st = selBag && selBag[t.name];
    return { name: t.name, params: t.params, shape: t.shape, dtype: t.dtype, st };
  }) : [];

  return (
    <div className="lvs-root">
      {/* ── 舞台 ── */}
      <div className="lvs-stage">
        <canvas ref={cvRef} className="lvs-cv"/>

        {/* HUD 左上：模型与进度 */}
        <div className="lvs-hud lvs-hud-tl">
          <div className="lvs-hud-title">
            参数星河 · 实时权重
            {!conn && <span className="lvs-tag warn">未连接</span>}
            {conn && <span className="lvs-tag live">LIVE</span>}
          </div>
          <div className="lvs-hud-row">
            <span className="k">模型</span>
            <span className="v mono" title={conn ? conn.model_path : ''}>
              {conn ? String(conn.model_path).split(/[\\/]/).pop() : '—'}
            </span>
          </div>
          <div className="lvs-hud-row">
            <span className="k">结构</span>
            <span className="v mono">
              {total ? `${total} 层 · d_model ${cfg.hidden ?? '—'} · ${fmtP(conn.total_params || 0)}` : '—'}
            </span>
          </div>
          <div className="lvs-hud-row">
            <span className="k">步进</span>
            <span className="v mono">{cursor < 0 ? '待运行' : `L${cursor} / L${maxL}`}</span>
          </div>
          <div className="lvs-progress"><i style={{ width: (progress * 100).toFixed(1) + '%' }}/></div>
          <div className="lvs-hud-note">
            {hasScale
              ? `色带区间 rms ${fmtV(vMin)} – ${fmtV(vMax)}（已取 ${Object.keys(vals).length} 层真实值）`
              : '尚未取回数值——运行后按真实权重 rms 着色'}
          </div>
          <div className="lvs-hud-note lvs-hud-view">视图 · {curView.hint}</div>
        </div>

        {/* HUD 右上：当前层读数 */}
        {curBag && (
          <div className="lvs-hud lvs-hud-tr">
            <div className="lvs-hud-title small">L{cursor} 当前读数</div>
            {Object.entries(curBag).map(([nm, st]) => (
              <div className="lvs-read" key={nm}>
                <b className="mono">{nm.split('.').slice(-3, -1).join('.')}</b>
                <span className="rms mono">{fmtV(st.rms)}</span>
                <span className="dim2 mono">|W|̄ {fmtV(st.mean_abs)} · max {fmtV(st.max_abs)}</span>
                {st.sampled && <span className="lvs-tag tiny">抽样</span>}
              </div>
            ))}
          </div>
        )}

        {/* 底部控制条 */}
        <div className="lvs-ctrl">
          <button type="button" className={'lvs-btn primary' + (running ? ' on' : '')}
                  disabled={!conn || !total}
                  onClick={() => (running ? pause() : start())}
                  title={running ? '暂停逐层推进' : '自底向上逐层读取真实权重数值并点亮'}>
            {running ? '⏸ 暂停' : '▶ 运行'}
          </button>
          <button type="button" className="lvs-btn" disabled={!conn || !total || running}
                  onClick={stepOnce} title="前进一层并读取该层真实数值">
            下一步 ▸
          </button>
          <button type="button" className="lvs-btn" disabled={!conn || !Object.keys(vals).length}
                  onClick={reset} title="清空已取数值，回到未运行状态">重置</button>

          <span className="lvs-sep"/>
          <span className="lvs-dim">视图</span>
          <select className="lvs-sel" value={view} onChange={e => changeView(e.target.value)}
                  title={curView.hint + '（只改呈现方式，数值来源不变）'}>
            {VIEWS.map(v => <option key={v.id} value={v.id}>{v.label}</option>)}
          </select>

          <span className="lvs-sep"/>
          <label className="lvs-chk" title="每层取 3 个承重矩阵（Q / gate / down）→ 更快；精细模式取该层全部大矩阵 → 更全但更慢">
            <input type="checkbox" checked={fine} onChange={e => setFine(e.target.checked)}/> 精细取数
          </label>
          <label className="lvs-chk" title="闲置时缓慢自转">
            <input type="checkbox" checked={autoRot} onChange={e => setAutoRot(e.target.checked)}/> 自转
          </label>
          <span className="lvs-sep"/>
          <span className="lvs-dim">速度</span>
          <input className="lvs-range" type="range" min="0.25" max="3" step="0.25"
                 value={speed} onChange={e => setSpeed(Number(e.target.value))}/>
          <span className="lvs-mono">{speed.toFixed(2)}×</span>
          {loadingL >= 0 && <span className="lvs-loading">读取 L{loadingL} 真实权重…</span>}
          {err && <span className="lvs-err">{err}</span>}
        </div>

        {/* 图例（随视图换文案） */}
        <div className="lvs-legend">
          {(LEGEND[view] || LEGEND.tower).map(([k, txt]) => (
            <span key={k}><i className={k}/>{txt}</span>
          ))}
        </div>
      </div>

      {/* ── 右侧面板 ── */}
      <aside className="lvs-side">
        {modelSlot}

        {!conn && (
          <div className="lvs-empty">
            <b>未连接本地模型</b>
            <p>实时分析只读取<b>本机模型目录</b>的真实权重（config.json + *.safetensors）。
              先在上方「模型库 / 下载」选择或下载模型，再点「连接」。</p>
            <p className="dim2">未连接时舞台不绘制数值着色——只保留结构轮廓，避免把示意当成实验数据。</p>
          </div>
        )}

        {conn && !selLayer && (
          <>
            <div className="lvs-side-hd">运行说明</div>
            <ol className="lvs-steps">
              <li><b>运行</b>：自 L0 向上逐层读取该层承重矩阵的真实标量统计（safetensors 按 32×64 行块切片），
                每层点亮并保留余晖。</li>
              <li><b>下一步</b>：单步推进一层，停在该层；盘面外圈出现扫描环，右上角显示该层三个承重矩阵的 rms / mean|W| / max|W|。</li>
              <li>点击任意层盘 → 选中该层，格内升起<b>柱阵</b>（柱高=该格真实 rms），右侧列出该层参数表。</li>
              <li>底部「<b>视图</b>」下拉可切换 5 种呈现（光塔 / 螺旋梯 / 地层矩阵 / 星核 / 粒子星河）：
                只换摆法，数值口径完全一致；「地层矩阵」开「精细取数」可填满列。</li>
              <li>点击柱区或表格行可下钻到「参数热图 / 神经元级」做更细的分析。</li>
            </ol>

            <div className="lvs-side-hd">当前模型结构</div>
            <div className="lvs-kv">
              <span className="k">架构</span><span className="v mono">{conn.arch || '—'}</span>
              <span className="k">层数</span><span className="v mono">{cfg.layers ?? '—'}</span>
              <span className="k">d_model</span><span className="v mono">{cfg.hidden ?? '—'}</span>
              <span className="k">heads</span><span className="v mono">{cfg.heads ?? '—'} Q / {cfg.kv_heads ?? '—'} KV</span>
              <span className="k">vocab</span><span className="v mono">{cfg.vocab != null ? cfg.vocab.toLocaleString() : '—'}</span>
              <span className="k">参数量</span><span className="v mono">{fmtP(conn.total_params || 0)}</span>
              <span className="k">权重文件</span><span className="v mono">{(conn.files || []).length} 个 safetensors</span>
            </div>

            {scanState === 'loading' && <div className="lvs-side-note">层基线扫描中（down_proj 全层）…</div>}
            {scan && scan.layers && scan.layers.length > 0 && (
              <>
                <div className="lvs-side-hd">层基线 · down_proj rms<span>{scan.layers.length} 层</span></div>
                <div className="lvs-bars">
                  {scan.layers.map(x => {
                    const mx = Math.max(...scan.layers.map(y => y.rms)) || 1;
                    return (
                      <span key={x.layer} className="b" title={`L${x.layer} · rms ${fmtV(x.rms)}${x.sampled ? ' · 抽样' : ''}`}>
                        <i style={{ height: (8 + (x.rms / mx) * 46).toFixed(1) + 'px' }}/>
                      </span>
                    );
                  })}
                </div>
                <div className="lvs-side-note">柱高=各层 mlp.down_proj 真实 rms（服务端 32×64 行块统计）。</div>
              </>
            )}

            <div className="lvs-side-note">
              数值来源：<code className="mono">POST /api/live/weights/summary</code> —— safetensors 按行块切片读取，
              超限行域均匀抽样；<b>不整体加载权重</b>，因此秒级完成且不占显存。
              激活级实时前向（逐 token 残差流）需加载权重到 GPU，属下一期。
            </div>
          </>
        )}

        {selLayer && (
          <>
            <div className="lvs-side-hd">
              L{sel} · 参数读数
              <button type="button" className="lvs-x" onClick={() => setSel(-1)} title="返回运行说明">×</button>
            </div>
            <div className="lvs-kv">
              <span className="k">该层 tensor</span><span className="v mono">{selLayer.tensors.length}</span>
              <span className="k">该层参数</span>
              <span className="v mono">{fmtP(selLayer.tensors.reduce((a, t) => a + (t.params || 0), 0))}</span>
              <span className="k">已取数值</span>
              <span className="v mono">{selBag ? Object.keys(selBag).length : 0} / {selLayer.tensors.length}</span>
            </div>

            <div className="lvs-table">
              <div className="hd">
                <span>tensor</span><span>形状</span><span>rms</span><span>mean|W|</span>
              </div>
              {selRows.map(r => (
                <div className={'row' + (r.st ? '' : ' pend')} key={r.name} title={r.name + (r.st ? '' : ' · 未取数')}>
                  <span className="nm mono">{r.name.split('.').slice(-2).join('.')}</span>
                  <span className="sh mono">{(r.shape || []).join('×')}</span>
                  <span className="rms mono">{r.st ? fmtV(r.st.rms) : '待取'}</span>
                  <span className="ma mono">{r.st ? fmtV(r.st.mean_abs) : '—'}</span>
                </div>
              ))}
            </div>

            {!selBag && (
              <button type="button" className="lvs-btn primary" style={{ width: '100%', marginTop: 6 }}
                      onClick={() => fetchLayer(sel)}>
                读取该层全部真实数值
              </button>
            )}

            <div className="lvs-side-actions">
              <button type="button" className="lvs-btn" onClick={() => onOpenParam && onOpenParam(sel)}
                      title="该层权重分块统计热图">参数热图 →</button>
              <button type="button" className="lvs-btn" onClick={() => onOpenNeuron && onOpenNeuron(sel)}
                      title="该层内部 3D 全量展开">神经元级 →</button>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}