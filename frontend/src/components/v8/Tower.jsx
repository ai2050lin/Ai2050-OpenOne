/* 3D 层塔：一个模型的全部层。层板按 HiddenState 序号排列：最下面 hs[0] = 嵌入层，块 b 的层板 = 该块的输出 hs[b+1]，
   最上面是输出头（只作标识，不可选）。层数来自模型骨架。
   各 tab 自己决定显示什么（研究工作台 · 空间的 SpaceLens 按 tab 组装）：
     values  按 hs 序号的叠加量（长度 nLayers + 1，null = 无数据），颜色单一色相由浅到深、按最大值归一化；
     marks   按 hs 序号的边缘标记（如「未见词表上可读出」），peak 为峰值层；
     onPick  点击某层（参数为 hs 序号；allowEmbedding 为 false 时嵌入层不可选）。点击从不切换 tab。
     edges   跨层连线（神经元上下游电路）：{hs1, hs2, p1:[x,z], p2:[x,z], color}，x∈[-1,1]、z∈[-D,D]；
             起终点画端点圆点、沿线在 75% 处画方向箭头（上游→目标 / 目标→下游）。
     target  选中单元在层板上的位置 {hs, x, z}：红圈标记。
     modules 传入时，每层板面画两个模块盒（后半 = Attention、前半 = MLP），可点击（onPickModule(hs, mod)）；
             不传 = 与旧版像素级一致。盒上的文字只在选中层画，避免透视下多层标签撞字。
     arch    模型结构参数（/api/model/arch）：有机块时模块盒画成有厚度的实体，
             底面积与厚度都按参数量定（MLP 通常是注意力的 2.8–4.7 倍，一眼能看出来），
             嵌入层画成一块宽而厚的「词表板」。没有 arch 时退回原来的平面色带。
     selModule 当前选中的模块 {hs, mod}：加粗描边。
     height  画布高度（选中单元聚焦时更高）。
   零依赖 canvas 投影：左键拖动旋转、滚轮/按钮缩放。层板只是排列，不代表模型几何。 */
import { useEffect, useRef, useState } from 'react';

import { fmtP, hsLabel, seq } from './v8Util.js';

const DEG = Math.PI / 180;
const MARK = '#1baf7a';
const ZOOM_MAX = 4;
const INITIAL_VIEW = { yaw: -30 * DEG, pitch: 22 * DEG, zoom: 1, dx: 0, dy: 0 };

/* 层内两个模块：色与 v8 的分类色一致（蓝 = 注意力、橙 = MLP），与研究上「同一层是两条通路」对应。
   z 轴分前后两半：z<0 后半给注意力、z>0 前半给 MLP。36 层叠瓦时，两条带在透视下各自形成一列纵向色柱，
   这比在每层内画小方块更可读——方块会被层板的填充色吃掉对比度。
   wall / side 是同一色的两个暗调，用来画实体盒的侧壁：没有侧壁就只是贴纸，看不出「有参数」。 */
const MOD_BOXES = [
  { mod: 'attn', label: 'Attention', color: '#2a78d6', wall: '#1f5fae', side: '#17457f', z0: -0.54, z1: -0.12 },
  { mod: 'mlp', label: 'MLP', color: '#eb6834', wall: '#c2542a', side: '#9c4321', z0: 0.12, z1: 0.54 },
];
const EMB = { color: '#64748b', wall: '#4a5a6e', side: '#3a4859' };

function inside(pt, pts) {
  let c = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}

function gridShape(kind, count) {
  const cols = Math.min(count, kind === 'attn' ? 8 : 128);
  const rows = Math.ceil(count / cols);
  return { cols, rows, z: kind === 'attn' ? -.53 : .14, depth: .4 };
}

export function towerIndexPosition(kind, count, index) {
  const { cols, rows, z, depth } = gridShape(kind, count);
  return [-.94 + 1.88 * ((index % cols) + .5) / cols, z + depth * (Math.floor(index / cols) + .5) / rows];
}

export default function Tower({ nLayers, selected, onPick, values, marks, peak, valueLabel = '叠加量', markLabel = '', allowEmbedding = true, caption, edges, target, modules, selModule, onPickModule, arch, height = 600, focusKey = '', focusModules = false, cells, onPickIndex, panControls = false, fit = 1 }) {
  const cv = useRef(null), hits = useRef([]), drag = useRef(null);
  const cellGrid = useRef(null);
  const [view, setView] = useState(INITIAL_VIEW);
  const [pan, setPan] = useState(false);
  const [hover, setHover] = useState(null);
  const [w, setW] = useState(340);
  const H = height;
  useEffect(() => {
    const el = cv.current?.parentElement;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(([e]) => setW(Math.max(1, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  /* 选中单元（出现上下游连线）时自动放大聚焦一次：渲染期派生（props 变化时调 state 的官方模式） */
  const [prevFocus, setPrevFocus] = useState('');
  if (focusKey && focusKey !== prevFocus) {
    setPrevFocus(focusKey);
    setView(v => (v.zoom >= 1.9 ? v : { ...v, zoom: 1.9 }));
  }
  useEffect(() => {
    const c = cv.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = w * dpr; c.height = H * dpr;
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, H);
    const n = nLayers + 2, span = 4.2, s = Math.min(w * 0.3, H * 0.22) * fit * view.zoom;
    const yOf = i => -span / 2 + (span * i) / (n - 1);
    const P = (x, y, z) => {
      const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw), cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
      const x1 = x * cy - z * sy, z1 = x * sy + z * cy;
      return [w / 2 + x1 * s + view.dx, H / 2 - (y * cp - z1 * sp) * s + view.dy];
    };
    const nums = (values || []).filter(v => v != null);
    const max = nums.length ? Math.max(1e-9, ...nums) : 1;
    const markSet = new Set(marks || []);
    const out = [];
    /* 结构参数：有 arch 就按参数量定体积（底面积 ∝ √参数、厚度 ∝ 参数），没有就退回平面色带 */
    const am = arch && arch.available ? arch : null;
    const pmax = am ? Math.max(am.attn?.params || 0, am.mlp?.params || 0, 1) : 0;
    const modGeom = mb => {
      const par = am ? ((mb.mod === 'attn' ? am.attn : am.mlp) || {}).params || 0 : 0;
      return { par, hw: am ? 0.26 + 0.58 * Math.sqrt(par / pmax) : 0.84, ht: am ? 0.008 + 0.040 * (par / pmax) : 0 };
    };
    const quad = (pts, fill, alpha) => {
      ctx.beginPath(); pts.forEach(([a, c2], k) => (k ? ctx.lineTo(a, c2) : ctx.moveTo(a, c2))); ctx.closePath();
      ctx.globalAlpha = alpha; ctx.fillStyle = fill; ctx.fill(); ctx.globalAlpha = 1;
    };
    /* 一个有厚度的盒：侧壁三面 + 顶面在调用方画（顶面要带描边/文字） */
    const walls3 = (x0, x1, z0, z1, yy, ht, c, alpha) => {
      const b = [P(x0, yy, z0), P(x1, yy, z0), P(x1, yy, z1), P(x0, yy, z1)];
      const t = [P(x0, yy + ht, z0), P(x1, yy + ht, z0), P(x1, yy + ht, z1), P(x0, yy + ht, z1)];
      if (ht > 0.0006) {
        quad([b[2], b[3], t[3], t[2]], c.wall, alpha);           // 前壁（朝观察者）
        quad([b[0], b[3], t[3], t[0]], c.side, alpha * 0.94);    // 左端壁
        quad([b[1], b[2], t[2], t[1]], c.side, alpha * 0.84);    // 右端壁
      }
      return t;
    };
    /* 一个模块的实体。默认画成一块整体：36 层叠瓦时只有整体能读。
       选中这一层或这个模块时，拆成它的权重矩阵——注意力的 q/k/v/o 宽度差就是 GQA 的形状（k/v 很窄），
       MLP 的三块大致等宽。拆开后每一块单独可悬停，读到自己的形状与参数量。 */
    const drawModule = (mb, hs, yy, on, opts = {}) => {
      const { split = false, label = false } = opts;
      const { par, hw, ht } = modGeom(mb);
      const mats = am ? ((mb.mod === 'attn' ? am.attn : am.mlp).mats || []) : [];
      const clickable = hs != null && (hs > 0 || allowEmbedding);
      const face = on ? 0.92 : 0.74;
      const stroke = t => {
        ctx.beginPath(); t.forEach(([a, c2], k) => (k ? ctx.lineTo(a, c2) : ctx.moveTo(a, c2))); ctx.closePath();
        ctx.lineWidth = on ? 1.8 : 0.7; ctx.strokeStyle = on ? mb.color : 'rgba(255,255,255,0.85)'; ctx.stroke();
      };
      if (mats.length > 1 && split) {
        const tot = mats.reduce((acc, m) => acc + m.params, 0) || 1;
        const mh = Math.max(...mats.map(m => m.params));
        const gap = 0.022, inner = 2 * hw - gap * (mats.length - 1);
        let x = -hw;
        for (const m of mats) {
          const w = inner * (m.params / tot);
          const t = walls3(x, x + w, mb.z0, mb.z1, yy, ht * (0.5 + 0.5 * (m.params / mh)), mb, on ? 0.9 : 0.62);
          quad(t, mb.color, face); stroke(t);
          out.push({ hs: clickable ? hs : null, mod: mb.mod, pts: t, label: `${hsLabel(hs)} · ${mb.label} · ${m.name}`, v: null,
                     info: `${m.name} [${m.shape[0].toLocaleString()} × ${m.shape[1].toLocaleString()}] · ${fmtP(m.params)}` });
          x += w + gap;
        }
      } else {
        const t = walls3(-hw, hw, mb.z0, mb.z1, yy, ht, mb, on ? 0.9 : 0.62);
        quad(t, mb.color, face); stroke(t);
        out.push({ hs: clickable ? hs : null, mod: mb.mod, pts: t, label: `${hsLabel(hs)} · ${mb.label}`, v: null,
                   info: am ? `${(mb.mod === 'attn' ? am.attn : am.mlp).shape_text} · ${fmtP(par)} 参数/层` : '' });
      }
      if (label) {
        /* 标签写在盒子右侧外面：压在盒子上会挡住拆分出来的矩阵块，也容易被上面的层盖住。
           深色字加白色描边（halo），在明暗层板上都读得清。 */
        const c = P(hw + 0.06, yy + ht * 0.7, (mb.z0 + mb.z1) / 2);
        const txt = am ? `${mb.label} ${fmtP(par)}` : mb.label;
        ctx.font = '600 10.5px ui-sans-serif, system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(255,255,255,0.92)'; ctx.strokeText(txt, c[0], c[1] + 3.4);
        ctx.fillStyle = mb.color; ctx.fillText(txt, c[0], c[1] + 3.4);
      }
    };
    for (let i = 0; i < n; i += 1) {
      const y = yOf(i), D = 0.6, isHead = i === n - 1, hs = isHead ? null : i;
      const pts = [P(-1.2, y, -D), P(1.2, y, -D), P(1.2, y, D), P(-1.2, y, D)];
      ctx.beginPath(); pts.forEach(([a, c2], k) => (k ? ctx.lineTo(a, c2) : ctx.moveTo(a, c2))); ctx.closePath();
      const v = hs != null && values ? values[hs] : null, sel = hs != null && hs === selected;
      ctx.fillStyle = isHead ? 'rgba(148,163,184,0.28)' : v == null ? (hs === 0 ? 'rgba(148,163,184,0.22)' : 'rgba(241,245,249,0.9)') : seq(0.06 + (0.94 * v) / max);
      ctx.globalAlpha = 0.93; ctx.fill(); ctx.globalAlpha = 1;
      ctx.lineWidth = sel ? 2.4 : 0.8; ctx.strokeStyle = sel ? '#e34948' : '#94a3b8'; ctx.stroke();
      if (hs != null && markSet.has(hs)) {                  // 边缘标记：层板右前沿一段粗线
        const [a, b] = [P(0.72, y, D), P(1.2, y, D)];
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
        ctx.lineWidth = hs === peak ? 5 : 3.2; ctx.strokeStyle = MARK; ctx.stroke();
      }
      const clickable = hs != null && (hs > 0 || allowEmbedding);
      out.push({ hs: clickable ? hs : null, pts, v, label: isHead ? '输出头（unembedding）' : hsLabel(hs), mark: hs != null && markSet.has(hs), peak: hs === peak });
      /* 嵌入层：整张词表（vocab × d）画成一块宽而厚的板，顶面刻出词表行（一行一个 token）。
         它是三块结构里唯一不在「块」里的一个，位置就在最底层，不随选中层变化。 */
      if (am && hs === 0) {
        const eht = 0.058;
        walls3(-1.08, 1.08, -0.52, 0.52, y, eht, EMB, 0.68);
        const t = [P(-1.08, y + eht, -0.52), P(1.08, y + eht, -0.52), P(1.08, y + eht, 0.52), P(-1.08, y + eht, 0.52)];
        quad(t, EMB.color, 0.88);
        for (let k = 1; k < 24; k += 1) {                    // 词表行：细密竖纹，读作「一行一个 token」
          const x = -1.08 + (2.16 * k) / 24;
          const a = P(x, y + eht, -0.5), b2 = P(x, y + eht, 0.5);
          ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b2[0], b2[1]);
          ctx.lineWidth = 0.6; ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.stroke();
        }
        out.push({ hs: allowEmbedding ? 0 : null, pts: t, label: `词嵌入 ${am.embed.shape_text}`, v: null,
                   info: `${fmtP(am.embed.params)} 参数${am.embed.tied ? '（与输出头共享）' : ''}` });
      }
      /* 层内模块盒：Attn 后半面、MLP 前半面。后 push ⇒ 反转后优先命中，点盒不会误选整层。
         文字只在选中层画：多层同时画会在透视下撞字 */
      if (modules && hs != null && hs > 0 && !focusModules) {
        for (const mb of MOD_BOXES) {
          const on = !!(selModule && selModule.hs === hs && selModule.mod === mb.mod);
          drawModule(mb, hs, y, on, { split: sel || on, label: sel || on });
        }
      }
      if (isHead || hs === 0 || (hs - 1) % 5 === 0 || sel || hs === peak) {
        const lc = pts.slice().sort((a2, c2) => a2[0] - c2[0])[0];
        ctx.font = `${sel || hs === peak ? 600 : 400} 10.5px ui-sans-serif, system-ui, sans-serif`;
        ctx.fillStyle = sel ? '#e34948' : hs === peak ? MARK : '#475569'; ctx.textAlign = 'right';
        ctx.fillText(isHead ? '输出头' : hs === 0 ? '嵌入' : `块 ${hs - 1}${hs === peak ? ' 峰' : ''}`, lc[0] - 6, lc[1] + 3);
      }
    }
    if (focusModules && modules && selected > 0 && selected < n - 1) {
      for (const mb of MOD_BOXES) drawModule(mb, selected, yOf(selected), selModule?.mod === mb.mod, { split: false, label: true });
    }
    cellGrid.current = null;
    if (cells?.count > 0) {
      const { cols, rows, z, depth } = gridShape(cells.kind, cells.count);
      const y = yOf(cells.hs), origin = P(-.94, y, z), px = P(.94, y, z), pz = P(-.94, y, z + depth);
      const a = [px[0] - origin[0], px[1] - origin[1]], b = [pz[0] - origin[0], pz[1] - origin[1]];
      cellGrid.current = { origin, a, b, cols, rows, count: cells.count };
      const corner = [px[0] + b[0], px[1] + b[1]];
      quad([origin, px, corner, pz], cells.kind === 'attn' ? '#e6f0fc' : '#fff0e8', .96);
      ctx.fillStyle = cells.kind === 'attn' ? '#2a78d6' : '#ac6446';
      const size = cells.kind === 'attn' ? 4 : Math.max(.6, Math.min(1.9, view.zoom * .65));
      const overlay = cells.values?.length===cells.count ? cells.values : null;
      const colors = cells.colors?.length===cells.count ? cells.colors : null;
      const scale = overlay ? Math.max(1e-12,...overlay.map(Math.abs)) : 1;
      for (let index = 0; index < cells.count; index += 1) {
        const x = ((index % cols) + .5) / cols, zPos = (Math.floor(index / cols) + .5) / rows;
        if(colors)ctx.fillStyle=colors[index];
        else if(overlay)ctx.fillStyle=`rgba(${overlay[index]<0?'235,104,52':'42,120,214'},${.1+.85*Math.abs(overlay[index])/scale})`;
        ctx.fillRect(origin[0] + a[0] * x + b[0] * zPos - size / 2, origin[1] + a[1] * x + b[1] * zPos - size / 2, size, size);
      }
    }
    /* 跨层连线（上下游电路）：端点圆点 + 弧线 + 75% 处方向箭头，画在层板之上 */
    for (const e of (edges || [])) {
      const [x1, z1] = e.p1, [x2, z2] = e.p2;
      const a = P(x1, yOf(e.hs1), z1), b = P(x2, yOf(e.hs2), z2);
      const c = P((x1 + x2) / 2 * 0.72, (yOf(e.hs1) + yOf(e.hs2)) / 2, (z1 + z2) / 2 * 0.72);
      ctx.globalAlpha = 0.9;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.quadraticCurveTo(c[0], c[1], b[0], b[1]);
      ctx.lineWidth = e.width || 1.8; ctx.strokeStyle = e.color; ctx.setLineDash(e.dash || []); ctx.stroke(); ctx.setLineDash([]);
      const t = 0.75, u = 1 - t;
      const q = [u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2 * u * t * c[1] + t * t * b[1]];
      const tx = 2 * u * (c[0] - a[0]) + 2 * t * (b[0] - c[0]), ty = 2 * u * (c[1] - a[1]) + 2 * t * (b[1] - c[1]);
      const tl = Math.max(1e-6, Math.hypot(tx, ty)), nx = tx / tl, ny = ty / tl, ah = 6;
      ctx.beginPath();
      ctx.moveTo(q[0] - ah * nx - ah * 0.5 * -ny, q[1] - ah * ny - ah * 0.5 * nx);
      ctx.lineTo(q[0], q[1]);
      ctx.lineTo(q[0] - ah * nx + ah * 0.5 * -ny, q[1] - ah * ny + ah * 0.5 * nx);
      ctx.lineWidth = e.width || 1.8; ctx.strokeStyle = e.color; ctx.stroke();
      for (const p of [a, b]) { ctx.beginPath(); ctx.arc(p[0], p[1], 2.6, 0, Math.PI * 2); ctx.fillStyle = e.color; ctx.fill(); }
      ctx.globalAlpha = 1;
    }
    /* 选中的模块置顶重绘：36 层叠瓦时这一层盒子会被上面几层压住，只加粗描边根本看不见。
       在全部层画完之后补一遍，才真正「可见地选中」。 */
    if (!focusModules && modules && selModule && selModule.hs > 0 && selModule.hs < n - 1) {
      const y = yOf(selModule.hs);
      for (const mb of MOD_BOXES) {
        if (mb.mod !== selModule.mod) continue;
        drawModule(mb, selModule.hs, y, true, { split: true, label: true });
      }
    }
    if (target) {                                           // 选中单元：层板上的红圈标记
      const p = P(target.x, yOf(target.hs), target.z);
      ctx.beginPath(); ctx.arc(p[0], p[1], 5.5, 0, Math.PI * 2);
      ctx.lineWidth = 2; ctx.strokeStyle = '#e34948'; ctx.stroke();
      ctx.beginPath(); ctx.arc(p[0], p[1], 2, 0, Math.PI * 2); ctx.fillStyle = '#e34948'; ctx.fill();
    }
    hits.current = out.reverse();
  }, [nLayers, selected, values, marks, peak, allowEmbedding, view, w, H, edges, target, modules, selModule, arch, focusModules, cells, fit]);
  const locate = e => {
    const r = cv.current.getBoundingClientRect();
    const pt = [e.clientX - r.left, e.clientY - r.top];
    return { pt, hit: hits.current.find(h => inside(pt, h.pts)) };
  };
  function indexAt([x, y]) {
    const grid = cellGrid.current;
    if (!grid) return null;
    const { origin, a, b, cols, rows, count } = grid;
    const det = a[0] * b[1] - a[1] * b[0];
    if (Math.abs(det) < 1) return null;
    const dx = x - origin[0], dy = y - origin[1];
    const u = (dx * b[1] - dy * b[0]) / det, v = (a[0] * dy - a[1] * dx) / det;
    if (u < 0 || u >= 1 || v < 0 || v >= 1) return null;
    const index = Math.floor(v * rows) * cols + Math.floor(u * cols);
    return index < count ? index : null;
  }
  useEffect(() => {
    const c = cv.current;
    const wheel = e => { e.preventDefault(); setView(v => ({ ...v, zoom: Math.max(0.5, Math.min(ZOOM_MAX, v.zoom * (e.deltaY < 0 ? 1.1 : 0.91))) })); };
    c?.addEventListener('wheel', wheel, { passive: false });
    return () => c?.removeEventListener('wheel', wheel);
  }, []);
  const zoomBy = f => setView(v => ({ ...v, zoom: Math.max(0.5, Math.min(ZOOM_MAX, v.zoom * f)) }));
  return (
    <div className="v8-tower-canvas">
      <canvas ref={cv} style={{ width: w, height: H, touchAction: 'none' }} role="img" aria-label={caption || `${nLayers} 个块的层塔`}
        onPointerDown={e => {
          if (!e.isPrimary || ![0, 2].includes(e.button)) return;
          drag.current = { x: e.clientX, y: e.clientY, view, moved: false, pan: panControls && (pan || e.shiftKey || e.button === 2) };
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={e => {
          if (drag.current) {
            const d = drag.current, dx = e.clientX - d.x, dy = e.clientY - d.y;
            if (Math.abs(dx) + Math.abs(dy) <= 3 && !d.moved) return;
            d.moved = true;
            setHover(null);
            setView(d.pan ? { ...d.view, dx: d.view.dx + dx, dy: d.view.dy + dy }
              : { ...d.view, yaw: d.view.yaw + dx * 0.008, pitch: Math.max(-1.2, Math.min(1.2, d.view.pitch + dy * 0.008)) });
            return;
          }
          const { pt, hit } = locate(e);
          setHover(hit ? { ...hit, x: pt[0], y: pt[1] } : null);
        }}
        onPointerUp={e => {
          const d = drag.current; drag.current = null;
          if (!d || d.moved || e.button === 2) return;
          const { pt, hit } = locate(e);
          const cell = indexAt(pt);
          if (cell != null && onPickIndex) onPickIndex(cell);
          else if (hit?.mod && onPickModule) onPickModule(hit.hs, hit.mod);
          else if (hit?.hs != null) onPick?.(hit.hs);
        }}
        onPointerCancel={() => { drag.current = null; setHover(null); }}
        onPointerLeave={() => setHover(null)}
        onContextMenu={panControls ? e => e.preventDefault() : undefined} />
      {hover && (
        <div className="v8-tip" style={{ left: Math.max(0, Math.min(hover.x + 12, w - 250)), top: hover.y + 10 }}>
          {hover.label}
          {hover.info && <small>{hover.info}</small>}
          {hover.v != null && <small>{valueLabel} {Number(hover.v).toLocaleString()}</small>}
          {hover.mark && markLabel && <small>● {markLabel}{hover.peak ? '（峰值层）' : ''}</small>}
        </div>
      )}
      <div className="v8-tower-btns">
        {panControls && <button type="button" className="fw-tbtn ghost" aria-pressed={pan} onClick={() => setPan(value => !value)}>平移</button>}
        <button type="button" className="fw-tbtn ghost" onClick={() => zoomBy(1.3)}>放大</button>
        <button type="button" className="fw-tbtn ghost" onClick={() => zoomBy(1 / 1.3)}>缩小</button>
        <button type="button" className="fw-tbtn ghost v8-reset" onClick={() => { setView(INITIAL_VIEW); setPan(false); }}>重置视角</button>
      </div>
    </div>
  );
}
