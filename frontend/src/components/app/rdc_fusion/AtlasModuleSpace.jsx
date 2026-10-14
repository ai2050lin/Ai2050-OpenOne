/* AtlasModuleSpace · 三模块 3D 关系空间（module_space.v1 驱动；design/atlas_module_space3d_scheme_v1.md）
   数据：GET /api/ai-rnd/atlas/tree?schema=v2 → module_space 段（派生器确定性布局，人手禁写）。
   结构：三舱一塔三桥——KN 晶格舱 / AR 链舱 / GR 环形舱（舱体形态=组织假说示意，非模型真实几何）+
   中心机制竖井塔（MEC 楼层盘）+ 汇合核（next-token 条件化汇合示意）+ 三主干轴（内容/真值/形式）。
   桥线宽/虚实编码分离度 band（registry 角度值对预注册门 30° 派生：separable 细虚线 / mixed 粗实线），
   桥面节点 = 跨域特征（X1 FTR-05 / X2 FTR-07 / X3 FTR-08）。
   交互：左键旋转 / 右键平移 / 滚轮缩放 / 闲置自转；点击模板/桥/MEC 盘 → 内部详情面板，
   点击特征节点 → onPickFeature（与全景共用 FeatureDetail 抽屉，协议键不引入第二事实源）。
   诚实口径：派生示意布局；LIVE/DEMO 不伪造。相机/交互态走 ref（防 effect 重建渲染循环）。 */
import { useEffect, useRef, useState } from 'react';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

const E_COLOR = { E3_causal_scoped: '#b45309', E2_predictive: '#1e5c22', E2_held_out: '#177a53', E1_repeatable: '#24558a' };
const E_LABEL = { E3_causal_scoped: 'E3 因果', E2_predictive: 'E2 可预测', E2_held_out: 'E2h 留出/行为', E1_repeatable: 'E1 可重复' };
const ST_COLOR = { measured: '#1e5c22', done: '#1e5c22', collecting: '#24558a', demo: '#64748b', pending: '#9aa4b1' };
const ST_LABEL = { measured: '实测', done: '完成', collecting: '采集中', demo: '演示', pending: '待立项' };
const MOD_NAME = { KN: '知识网络', AR: '推理功能', GR: '语法机制' };
const MODELS = ['angle_4b_deg', 'angle_14b_deg', 'angle_glm4_deg'];
const MODEL_LBL = { angle_4b_deg: 'qwen3-4b', angle_14b_deg: 'qwen3-14b', angle_glm4_deg: 'glm4-9b' };

function quadBez(p1, c, p2, t) {
  const u = 1 - t;
  return [0, 1, 2].map(i => u * u * p1[i] + 2 * u * t * c[i] + t * t * p2[i]);
}

export default function AtlasModuleSpace({ sel, onPickFeature }) {
  const wrapRef = useRef(null);
  const cvRef = useRef(null);
  const [tree, setTree] = useState(null);
  const [err, setErr] = useState(null);
  const [modOn, setModOn] = useState({ KN: true, AR: true, GR: true });
  const [edgeOn, setEdgeOn] = useState({ a: true, c: true, x: true, b: true });
  const [colorMode, setColorMode] = useState('elevel');
  const [autoRot, setAutoRot] = useState(true);
  const [localSel, setLocalSel] = useState(null);        // {kind, ref}（tpl/bridge/mec 内部面板）
  const uiRef = useRef({});
  uiRef.current = { modOn, edgeOn, colorMode, autoRot, localSel, sel, tree };
  const cam = useRef({ yaw: 0.55, pitch: 0.36, dist: 37, tgt: [0, 2.4, 0], goal: null, lastAct: 0 });
  const drag = useRef(null);
  const hoverRef = useRef({ picks: [] });

  useEffect(() => {
    let dead = false;
    fetch(`${API_BASE}/api/ai-rnd/atlas/tree?schema=v2`)
      .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(d => { if (!dead) setTree(d.module_space ? d : null) || (d.module_space ? null : setErr('派生档缺少 module_space 段——重跑 python server/derive_atlas_tree.py')); })
      .catch(e => { if (!dead) setErr(String(e.message || e)); });
    return () => { dead = true; };
  }, []);

  /* join 表：模板/特征/机制（节点只带 ref id，详情从同档派生段 join——零重复事实源） */
  const joins = (() => {
    if (!tree) return null;
    const tpl = {}, ftr = {}, mec = {};
    (tree.modules || []).forEach(m => (m.templates || []).forEach(t => { tpl[t.id] = { ...t, mod: m.id }; }));
    (tree.modules || []).forEach(m => (m.families || []).forEach(f => (f.features || []).forEach(x => { ftr[x.id] = x; })));
    (tree.cross_links || []).forEach(x => (x.detail || []).forEach(f => { ftr[f.id] = f; }));
    (tree.mechanisms || []).forEach(m => { mec[m.id] = m; });
    const linksByTpl = {}, linksByFtr = {};
    (tree.links || []).forEach(l => {
      (linksByTpl[l.tpl] = linksByTpl[l.tpl] || []).push(l);
      (linksByFtr[l.ftr] = linksByFtr[l.ftr] || []).push(l);
    });
    return { tpl, ftr, mec, linksByTpl, linksByFtr };
  })();

  useEffect(() => {
    const cv = cvRef.current, wrap = wrapRef.current;
    if (!cv || !wrap || !tree) return undefined;
    const ctx = cv.getContext('2d');
    const ms = tree.module_space;
    const nodePos = {};                                    // ref → pos
    ms.nodes.forEach(n => { nodePos[n.ref] = n.pos; });
    const nodeKind = {};
    ms.nodes.forEach(n => { nodeKind[n.ref] = n.kind; });

    let raf = 0, W = 0, H = 0;
    const dprOf = () => window.devicePixelRatio || 1;
    const resize = () => {
      const dpr = dprOf();
      W = wrap.clientWidth; H = wrap.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr;
      cv.style.width = W + 'px'; cv.style.height = H + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const project = p => {
      const c = cam.current;
      const dx = p[0] - c.tgt[0], dy = p[1] - c.tgt[1], dz = p[2] - c.tgt[2];
      const cy = Math.cos(c.yaw), sy = Math.sin(c.yaw);
      const x1 = dx * cy - dz * sy, z1 = dx * sy + dz * cy;
      const cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
      const y2 = dy * cp - z1 * sp, z2 = dy * sp + z1 * cp;
      const per = 26 / (26 + z2 + c.dist);
      return [W / 2 + x1 * per * 14, H / 2 - y2 * per * 14, per];
    };
    const seg = (a, b, style, w, dash) => {
      const p1 = project(a), p2 = project(b);
      ctx.strokeStyle = style; ctx.lineWidth = w;
      ctx.setLineDash(dash || []); ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke(); ctx.setLineDash([]);
    };
    const ell = (c, yz, r, fill, stroke, lw) => {
      const p = project([c[0], yz, c[2]]);
      ctx.beginPath(); ctx.ellipse(p[0], p[1], r * p[2] * 14, r * p[2] * 14 * 0.32, 0, 0, Math.PI * 2);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1; ctx.stroke(); }
      return p;
    };
    const label = (p, txt, font, col, align) => {
      ctx.fillStyle = col || '#52616f'; ctx.font = font || '10px ui-monospace,Consolas,monospace';
      ctx.textAlign = align || 'left'; ctx.fillText(txt, p[0], p[1]); ctx.textAlign = 'left';
    };
    const nodeColor = n => {
      if (uiRef.current.colorMode === 'module') return (ms.modules[n.mod] || {}).color || '#8a93a3';
      return n.kind === 'tpl'
        ? (ST_COLOR[(joins.tpl[n.ref] || {}).st] || '#8a93a3')
        : (E_COLOR[n.band] || '#8a93a3');
    };

    const draw = () => {
      const ui = uiRef.current;
      const c = cam.current;
      if (c.goal) {
        const g = c.goal, L = (a, b) => a + (b - a) * 0.09;
        c.yaw = L(c.yaw, g.yaw); c.pitch = L(c.pitch, g.pitch); c.dist = L(c.dist, g.dist);
        c.tgt = c.tgt.map((v, i) => L(v, g.tgt[i]));
        if (Math.abs(c.yaw - g.yaw) < 0.002 && Math.abs(c.dist - g.dist) < 0.05) c.goal = null;
      } else if (!drag.current && ui.autoRot && performance.now() - c.lastAct > 2000) {
        c.yaw += 0.0022;
      }
      ctx.clearRect(0, 0, W, H);
      const picks = [];

      /* 地面参考网格 */
      for (let i = -16; i <= 16; i += 4) {
        seg([i, 0, -16], [i, 0, 16], '#e8edf3', 1);
        seg([-16, 0, i], [16, 0, i], '#e8edf3', 1);
      }
      /* 三主干轴 */
      (ms.trunks || []).forEach(t => {
        if (!ui.modOn[t.from]) return;
        const col = ms.modules[t.from].color;
        seg(t.p1, [0, 1.0, 0], col + '88', 4);
        label(project([t.p1[0] * 0.5, 1.5, t.p1[2] * 0.5]), t.axis, '10px system-ui', '#52616f', 'center');
      });
      /* 机制竖井塔 + 汇合核 */
      ctx.strokeStyle = 'rgba(124,58,237,0.5)'; ctx.lineWidth = 2;
      const b0 = project([-1.3, 0.4, 0]), b1 = project([1.3, 0.4, 0]);
      const t0 = project([-1.3, 5.7, 0]), t1 = project([1.3, 5.7, 0]);
      ctx.beginPath(); ctx.moveTo(b0[0], b0[1]); ctx.lineTo(t0[0], t0[1]);
      ctx.moveTo(b1[0], b1[1]); ctx.lineTo(t1[0], t1[1]); ctx.stroke();
      (ms.tower.slabs || []).forEach(sl => {
        const isE3 = (joins.mec[sl.id] || {}).evidence_level_floor === 'E3_causal_scoped';
        const p = ell([0, sl.y, 0], sl.y, 1.5,
          `rgba(124,58,237,${isE3 ? '0.22' : '0.12'})`, 'rgba(124,58,237,0.65)', 1);
        label([p[0] + 12, p[1] + 3], sl.id, '9px ui-monospace,monospace', '#6d28d9');
        picks.push({ x: p[0], y: p[1], r: 10, kind: 'mec', ref: sl.id });
      });
      const core = project(ms.tower.core);
      const grd = ctx.createRadialGradient(core[0], core[1], 2, core[0], core[1], 16);
      grd.addColorStop(0, 'rgba(124,58,237,0.55)'); grd.addColorStop(1, 'rgba(124,58,237,0.05)');
      ctx.fillStyle = grd; ctx.beginPath(); ctx.arc(core[0], core[1], 16, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = 'rgba(124,58,237,0.6)'; ctx.lineWidth = 1.2; ctx.stroke();
      label([core[0] - 18, core[1] - 20], '汇合核 = next-token 条件化汇合（示意）', '10px system-ui', '#6d28d9');
      /* 模块舱体 */
      Object.keys(ms.modules).forEach(mid => {
        if (!ui.modOn[mid]) return;
        const m = ms.modules[mid], cc = m.center, col = m.color;
        if (m.shape === 'lattice') {
          const h = m.half;
          const cs = [
            [cc[0] - h[0], cc[1] - h[1], cc[2] - h[2]], [cc[0] + h[0], cc[1] - h[1], cc[2] - h[2]],
            [cc[0] + h[0], cc[1] - h[1], cc[2] + h[2]], [cc[0] - h[0], cc[1] - h[1], cc[2] + h[2]],
            [cc[0] - h[0], cc[1] + h[1], cc[2] - h[2]], [cc[0] + h[0], cc[1] + h[1], cc[2] - h[2]],
            [cc[0] + h[0], cc[1] + h[1], cc[2] + h[2]], [cc[0] - h[0], cc[1] + h[1], cc[2] + h[2]]];
          [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
            .forEach(e => seg(cs[e[0]], cs[e[1]], col + '55', 1));
          [0.34, 0.67].forEach(f => {
            seg([cc[0] - h[0] + 2 * h[0] * f, cc[1] - h[1], cc[2] - h[2]],
              [cc[0] - h[0] + 2 * h[0] * f, cc[1] - h[1], cc[2] + h[2]], col + '22', 1);
            seg([cc[0] - h[0], cc[1] - h[1], cc[2] - h[2] + 2 * h[2] * f],
              [cc[0] + h[0], cc[1] - h[1], cc[2] - h[2] + 2 * h[2] * f], col + '22', 1);
          });
        } else if (m.shape === 'chain') {
          seg([6.9, 1.1, 6.7], [10.0, 1.8, 7.5], col + '88', 2);
          seg([10.0, 1.8, 7.5], [13.1, 2.5, 6.7], col + '88', 2);
          [[6.9, 1.1, 6.7], [10.0, 1.8, 7.5], [13.1, 2.5, 6.7]]
            .forEach(s => ell([s[0], s[1] - 0.25, s[2]], s[1] - 0.25, 1.1, col + '14', col + '66', 1));
          label(project([10, 0.6, 8.6]), '前提 → 门控 → 结论（链式假说示意）', '10px system-ui', '#9a6a52', 'center');
        } else {
          ctx.strokeStyle = col + '77'; ctx.lineWidth = 1.4; ctx.beginPath();
          for (let i = 0; i <= 40; i++) {
            const th = i / 40 * Math.PI * 2;
            const p = project([cc[0] + m.radius * Math.cos(th), cc[1], cc[2] + m.radius * Math.sin(th)]);
            if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
          }
          ctx.stroke();
          label(project([cc[0], 0.4, cc[2] + m.radius + 1.2]), '位置/顺序环（序列假说示意）', '10px system-ui', '#4d8577', 'center');
        }
        label(project([cc[0], m.shape === 'lattice' ? 4.6 : 4.2, cc[2]]),
          `${mid} ${MOD_NAME[mid]}`, '600 11px system-ui', col, 'center');
      });
      /* 桥（band 编码分离度：mixed 粗实线 / separable 细虚线） */
      (ms.bridges || []).forEach(br => {
        if (!ui.edgeOn.b) return;
        if (!ui.modOn[br.mods[0]] && !ui.modOn[br.mods[1]]) return;
        const mixed = br.band === 'mixed';
        ctx.strokeStyle = mixed ? '#7c3aed' : '#7c3aed66';
        ctx.lineWidth = mixed ? 3.2 : 1.4;
        if (!mixed) ctx.setLineDash([7, 4]);
        ctx.beginPath();
        for (let i = 0; i <= 28; i++) {
          const p = project(quadBez(br.p1, br.ctrl, br.p2, i / 28));
          if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke(); ctx.setLineDash([]);
        const mp = project(br.mid);
        const angTxt = MODELS.filter(k => br.angles && br.angles[k] != null)
          .map(k => br.angles[k].toFixed(1)).join('/');
        label([mp[0] - 24, mp[1] - 10],
          `${br.edge} ${br.type} · ${angTxt}°（${br.band === 'mixed' ? '弱混合' : br.band === 'separable' ? '可分' : br.band}）`,
          mixed ? '600 10.5px ui-monospace,monospace' : '10px ui-monospace,monospace',
          mixed ? '#6d28d9' : '#8a7bb5');
      });
      /* 舱内关联边 */
      (tree.links || []).forEach(l => {
        const t = joins.tpl[l.tpl];
        const tp = nodePos[l.tpl], fp = nodePos[l.ftr];
        if (!t || !tp || !fp) return;
        if (!ui.modOn[t.mod]) return;
        const fmod = nodeKind[l.ftr];
        if (fmod && fmod.indexOf && fmod.indexOf('X') === 0 && !ui.edgeOn.b) return;
        if (!ui.edgeOn[l.kind]) return;
        const isX = l.kind === 'x';
        ctx.strokeStyle = isX ? '#d97706cc' : (l.kind === 'a' ? ms.modules[t.mod].color + 'aa' : '#94a3b888');
        ctx.lineWidth = isX ? 1.8 : 1;
        if (l.kind === 'c' || isX) ctx.setLineDash([5, 4]);
        const p1 = project(tp), p2 = project(fp);
        ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke(); ctx.setLineDash([]);
      });
      /* 机制横梁（跨模块 MEC） */
      (ms.mech_beams || []).forEach(mb => {
        if (!ui.modOn[mb.mods[0]] && !ui.modOn[mb.mods[1]]) return;
        mb.mods.forEach(mid => {
          if (!ui.modOn[mid]) return;
          const cc = ms.modules[mid].center;
          seg([0, mb.y, 0], [cc[0] * 0.62, 2.6, cc[2] * 0.62], 'rgba(124,58,237,0.4)', 1.4, [6, 3]);
        });
      });
      /* 节点（深度排序：远→近） */
      const nodes = ms.nodes.filter(n => {
        if (n.kind === 'tpl') return ui.modOn[n.mod];
        return n.mod.indexOf('X') === 0 ? true : ui.modOn[n.mod];
      });
      const projd = nodes.map(n => ({ n, p: project(n.pos) }));
      projd.sort((a, b) => a.p[2] - b.p[2]);
      projd.forEach(q => {
        const n = q.n, p = q.p;
        const gap = Boolean(n.gap);
        const tpl = n.kind === 'tpl' ? joins.tpl[n.ref] : null;
        const r = (n.kind === 'tpl' ? 3.4 + Math.log10(((tpl || {}).prod || {}).cells + 1) * 1.4 : 3.2)
          * Math.max(0.55, p[2]);
        ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
        if (gap) {
          ctx.fillStyle = '#ffffff'; ctx.fill();
          ctx.strokeStyle = '#b91c1c'; ctx.lineWidth = 1.4; ctx.setLineDash([2, 2]); ctx.stroke(); ctx.setLineDash([]);
        } else {
          ctx.fillStyle = nodeColor(n); ctx.fill();
          ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke();
        }
        if (n.kind === 'tpl' && tpl && tpl.tm) label([p[0] + r + 3, p[1] + 3], tpl.tm, '9px ui-monospace,monospace', '#52616f');
        if (n.kind === 'ftr') label([p[0] + r + 3, p[1] + 3], n.ref, '9px ui-monospace,monospace', '#7a8494');
        if (gap) label([p[0] - 4, p[1] - r - 5], 'gap', '600 9px system-ui', '#b91c1c', 'right');
        if (ui.sel === n.ref) {
          ctx.beginPath(); ctx.arc(p[0], p[1], r + 5, 0, Math.PI * 2);
          ctx.strokeStyle = '#b45309'; ctx.lineWidth = 2; ctx.stroke();
        }
        if (ui.localSel && ui.localSel.ref === n.ref) {
          ctx.beginPath(); ctx.arc(p[0], p[1], r + 5, 0, Math.PI * 2);
          ctx.strokeStyle = '#b45309'; ctx.lineWidth = 2; ctx.stroke();
        }
        picks.push({ x: p[0], y: p[1], r: r + 3, kind: n.kind, ref: n.ref });
      });
      hoverRef.current.picks = picks;
      raf = requestAnimationFrame(draw);
    };

    const onDown = e => {
      drag.current = { btn: e.button, x: e.clientX, y: e.clientY, moved: 0 };
      cam.current.lastAct = performance.now(); cam.current.goal = null;
    };
    const onMove = e => {
      const d = drag.current;
      if (!d) return;
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      d.moved += Math.abs(dx) + Math.abs(dy);
      cam.current.lastAct = performance.now();
      if (d.btn === 2 || e.shiftKey) {
        const c = cam.current, s = c.dist * 0.0016;
        const cy = Math.cos(c.yaw), sy = Math.sin(c.yaw);
        c.tgt[0] -= dx * cy * s; c.tgt[2] -= dx * sy * s; c.tgt[1] += dy * s;
      } else {
        cam.current.yaw += dx * 0.0055;
        cam.current.pitch = Math.max(-1.3, Math.min(1.3, cam.current.pitch + dy * 0.0045));
      }
      d.x = e.clientX; d.y = e.clientY;
    };
    const onUp = e => {
      const d = drag.current;
      drag.current = null;
      if (!d || d.moved >= 5) return;
      const rect = cv.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      let best = null, bd = 14;
      hoverRef.current.picks.forEach(pk => {
        const dd = Math.hypot(pk.x - mx, pk.y - my) - pk.r;
        if (dd < bd) { bd = dd; best = pk; }
      });
      if (!best) return;
      if (best.kind === 'ftr') {
        if (onPickFeature) onPickFeature(best.ref);
        setLocalSel({ kind: 'ftr', ref: best.ref });
      } else {
        setLocalSel({ kind: best.kind, ref: best.ref });
      }
    };
    const onWheel = e => {
      e.preventDefault();
      cam.current.dist = Math.max(14, Math.min(80, cam.current.dist * Math.exp(e.deltaY * 0.001)));
      cam.current.lastAct = performance.now(); cam.current.goal = null;
    };
    const onCtx = e => e.preventDefault();

    cv.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    cv.addEventListener('wheel', onWheel, { passive: false });
    cv.addEventListener('contextmenu', onCtx);
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf); ro.disconnect();
      cv.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      cv.removeEventListener('wheel', onWheel);
      cv.removeEventListener('contextmenu', onCtx);
    };
  }, [tree, joins, onPickFeature]);

  if (err) return <div className="at-err">模块空间数据不可达（{err}）——启动 :5001 后重试；派生档缺失时在仓库根运行 python server/derive_atlas_tree.py。</div>;
  if (!tree) return <div className="at-note">加载三模块 3D 关系空间（module_space.v1）…</div>;

  const dSel = localSel;
  const bridge = dSel && dSel.kind === 'bridge' ? tree.module_space.bridges.find(b => b.edge === dSel.ref) : null;
  const tplD = dSel && dSel.kind === 'tpl' ? joins.tpl[dSel.ref] : null;
  const mecD = dSel && dSel.kind === 'mec' ? joins.mec[dSel.ref] : null;
  const ftrD = dSel && dSel.kind === 'ftr' ? joins.ftr[dSel.ref] : null;
  const kindCn = { a: 'a 实测', c: 'c 采集', x: 'x 跨模块' };

  return (
    <div className="atms-root">
      <div className="atms-stage" ref={wrapRef}>
        <canvas ref={cvRef} className="atms-cv"/>
        <div className="atms-cap">module_space.v1 · <b>组织假说示意（派生布局）</b> — 舱体形态≠模型真实几何 · 桥线宽=分离度（细=可分 粗=弱混合）· sha8 {tree.tree_sha256_8}</div>
        <div className="atms-toolbar">
          <b>模块显隐</b>
          <div className="atms-row">
            {Object.keys(MOD_NAME).map(mid => (
              <button key={mid} type="button" className={'fw-tbtn atms-' + mid.toLowerCase() + (modOn[mid] ? ' on' : '')}
                onClick={() => setModOn(o => ({ ...o, [mid]: !o[mid] }))}>{mid}</button>
            ))}
          </div>
          <b>边类型过滤</b>
          <div className="atms-row">
            {[['a', 'a 实测'], ['c', 'c 采集'], ['x', 'x 跨模块'], ['b', '桥']].map(([k, lb]) => (
              <button key={k} type="button" className={'fw-tbtn' + (edgeOn[k] ? ' on' : '')}
                onClick={() => setEdgeOn(o => ({ ...o, [k]: !o[k] }))}>{lb}</button>
            ))}
          </div>
          <b>视角 / 着色</b>
          <div className="atms-row">
            <button type="button" className={'fw-tbtn' + (autoRot ? ' on' : '')}
              onClick={() => setAutoRot(v => !v)}>⟳ 自转</button>
            <button type="button" className="fw-tbtn"
              onClick={() => { cam.current.goal = { yaw: 0.55, pitch: 0.36, dist: 37, tgt: [0, 2.4, 0] }; }}>复位</button>
            <button type="button" className={'fw-tbtn' + (colorMode === 'elevel' ? ' on' : '')}
              onClick={() => setColorMode('elevel')}>E级</button>
            <button type="button" className={'fw-tbtn' + (colorMode === 'module' ? ' on' : '')}
              onClick={() => setColorMode('module')}>模块色</button>
          </div>
          <b>图例</b>
          <div className="atms-legend">
            <span><i style={{ background: E_COLOR.E2_predictive }}/>E2</span>
            <span><i style={{ background: E_COLOR.E2_held_out }}/>E2h</span>
            <span><i style={{ background: E_COLOR.E1_repeatable }}/>E1</span>
            <span><i style={{ background: E_COLOR.E3_causal_scoped }}/>E3</span>
            <span><span className="atms-ln" style={{ borderTop: '2px solid #7c3aed' }}/>桥（粗=弱混合）</span>
            <span><span className="atms-ln" style={{ borderTop: '2px dashed #d97706' }}/>x 跨模块</span>
            <span><i style={{ border: '1.4px dashed #b91c1c', background: '#fff' }}/>缺口模板</span>
          </div>
        </div>
      </div>

      <div className="atms-detail">
        {!dSel && <span className="dim2">点击模板 / 桥节点 / MEC 楼层盘看详情；点击特征节点打开全景特征抽屉。</span>}
        {tplD && (() => {
          const lks = joins.linksByTpl[tplD.id] || [];
          return (
            <>
              <div className="atms-dh">
                <b className="mono">{tplD.id}</b><b>{tplD.name}</b>
                <span className="amv-st mono">{tplD.tm || '未注册'} · {ST_LABEL[tplD.st]}</span>
                {tplD.metric && <span className="amv-st mono">{tplD.metric}</span>}
                {tplD.gap && <span className="amv-st gap">缺口：{tplD.gap}</span>}
                <button type="button" className="amv-close" onClick={() => setLocalSel(null)}>×</button>
              </div>
              <div className="atms-dgrid">
                <div className="amv-dsec"><div className="amv-dsec-h">对象档案</div>
                  <div className="mono">{tplD.pat}</div><div>例：{tplD.ex}</div>
                  <div className="atms-chips"><span className="chip">{tplD.family} · prod {tplD.prod.cells} cells</span></div>
                </div>
                <div className="amv-dsec"><div className="amv-dsec-h">关联机制（{lks.length} 条）</div>
                  <div className="atms-chips">
                    {lks.length ? lks.map((l, i) => {
                      const f = joins.ftr[l.ftr];
                      return <span key={i} className={'chip' + (l.kind === 'x' ? ' atms-chip-x' : '')}>
                        {kindCn[l.kind]} → {l.ftr}（{f ? E_LABEL[f.evidence_level] : ''}）</span>;
                    }) : <span className="dim2">无锚定特征——缺口</span>}
                  </div>
                </div>
              </div>
            </>
          );
        })()}
        {bridge && (() => {
          const f = joins.ftr[bridge.ftr];
          return (
            <>
              <div className="atms-dh">
                <b className="mono">{bridge.edge}</b><b>{bridge.type} 跨域桥（{bridge.mods.join(' ↔ ')}）</b>
                <span className={'amv-st mono' + (bridge.band === 'mixed' ? ' gap' : '')}>
                  {bridge.band === 'mixed' ? '弱混合（唯一强耦合对）' : '几何可分'}</span>
                <button type="button" className="amv-close" onClick={() => setLocalSel(null)}>×</button>
              </div>
              <div className="atms-dgrid">
                <div className="amv-dsec"><div className="amv-dsec-h">分离度读数（主角角度 · registry 派生 · 门 {bridge.band_gate_deg}°）</div>
                  <div className="atms-angles">
                    {MODELS.filter(k => bridge.angles[k] != null).map(k => (
                      <div key={k} className="atms-angle"><b className="mono">{bridge.angles[k].toFixed(1)}°</b><span>{MODEL_LBL[k]}</span></div>
                    ))}
                  </div>
                </div>
                <div className="amv-dsec"><div className="amv-dsec-h">桥面证据特征</div>
                  <div className="mono">{bridge.ftr}</div>
                  <div className="atms-stmt">{f ? f.statement.slice(0, 220) + (f.statement.length > 220 ? '…' : '') : ''}</div>
                </div>
              </div>
            </>
          );
        })()}
        {mecD && (() => (
          <>
            <div className="atms-dh">
              <b className="mono">{mecD.id}</b><b>{mecD.name}</b>
              <span className="amv-st mono">底座 {mecD.evidence_level_floor}</span>
              <button type="button" className="amv-close" onClick={() => setLocalSel(null)}>×</button>
            </div>
            <div className="atms-dgrid">
              <div className="amv-dsec"><div className="amv-dsec-h">成员特征</div>
                <div className="atms-chips">{mecD.features.map(f => (
                  <span key={f.id} className="chip">{f.id}（{E_LABEL[f.evidence_level]}）</span>))}</div>
              </div>
              <div className="amv-dsec"><div className="amv-dsec-h">跨模块服务</div>
                <div className="atms-stmt"> {(tree.module_space.mech_beams.find(b => b.mec === mecD.id) || { mods: null }).mods
                  ? '横梁 → ' + (tree.module_space.mech_beams.find(b => b.mec === mecD.id).mods.join(' + ')) + '（图中紫色虚线梁）'
                  : '当前为模块内机制，无横梁'}</div>
              </div>
            </div>
          </>
        ))()}
        {ftrD && (
          <div className="atms-dh">
            <b className="mono">{ftrD.id}</b><span className="amv-st mono">{E_LABEL[ftrD.evidence_level]}</span>
            <span className="dim2">完整 statement 与证据链见全景底部特征抽屉（共用 FeatureDetail）</span>
            <button type="button" className="amv-close" onClick={() => setLocalSel(null)}>×</button>
          </div>
        )}
      </div>
    </div>
  );
}
