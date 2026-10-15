/* AtlasTemplateSpace · 语言模板 3D 关系空间（atlas_tree_v2 驱动；scheme §9 / mockup v1 移植）
   节点 = 语言模板（模块聚簇 + 黄金角派生布点，pos 来自派生器 v2，非模型真实几何——界面必注）；
   边三类（由 links 派生，非手工维护）：共享 FTR（灰）/ 同模板家族（紫）/ 跨模块桥（橙虚线）；
   交互：拖拽旋转 / 滚轮缩放 / 自动旋转 / 模块显隐 / 边标签开关；点击节点 → 下方三段详情面板。
   点击特征 → onPickFeature(fid) 联动 FeatureDetail 抽屉。 */
import { API_BASE } from '../../../config.js';
import { useEffect, useMemo, useRef, useState } from 'react';
import { LANG_TEMPLATES } from './distributedData.js';

const ECLASS = { E2_predictive: 'at-e2', E2_held_out: 'at-e2', E2_behavioral: 'at-e2', E3_causal_scoped: 'at-e3', E1_repeatable: 'at-e1' };
const ELABEL = { E2_predictive: 'E2', E2_held_out: 'E2h', E2_behavioral: 'E2b', E3_causal_scoped: 'E3', E1_repeatable: 'E1' };
const ST_NAME = { measured: '已实测', collecting: '采集中', demo: 'DEMO', pending: '待启动', done: '已完成' };
const KIND_NAME = { a: '实测锚定', c: '采集中', x: '跨模块桥' };
const GOLD = 2.399963229728653;

/* 三模块聚簇中心 = MOD_CENTERS（与派生器同源；仅模块环渲染用） */
const MOD_CENTER = { KN: [-2.6, 0, -1.15], AR: [2.6, 0, -1.15], GR: [0, 0.15, 2.45] };

export default function AtlasTemplateSpace({ onPickFeature }) {
  const [tree, setTree] = useState(null);
  const [err, setErr] = useState(null);
  const [autoRot, setAutoRot] = useState(true);
  const [showLbl, setShowLbl] = useState(true);
  const [modOff, setModOff] = useState({});
  const [sel, setSel] = useState(null);
  const canvasRef = useRef(null);
  const stateRef = useRef({ yaw: 0.65, pitch: 0.42, zoom: 1, drag: false, moved: 0, lx: 0, ly: 0 });
  const tpRef = useRef({});

  useEffect(() => {
    let dead = false;
    fetch(`${API_BASE}/api/ai-rnd/atlas/tree?schema=v2`)
      .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(d => { if (!dead) setTree(d); })
      .catch(e => { if (!dead) setErr(String(e.message || e)); });
    return () => { dead = true; };
  }, []);

  const templates = useMemo(() => (tree ? tree.modules.flatMap(m => m.templates) : []), [tree]);
  const modById = useMemo(() => {
    const mp = {};
    if (tree) tree.modules.forEach(m => { mp[m.id] = m; });
    return mp;
  }, [tree]);
  const ftrById = useMemo(() => {
    const fp = {};
    if (tree) {
      tree.modules.forEach(m => m.families.forEach(f => f.features.forEach(x => { fp[x.id] = x; })));
      tree.cross_links.forEach(c => c.features.forEach(fid => {
        if (!fp[fid]) fp[fid] = { id: fid, evidence_level: 'E2_predictive', statement: `交叉边 ${c.edge}（${c.type}）` };
      }));
    }
    return fp;
  }, [tree]);
  const edges = useMemo(() => {
    if (!tree) return [];
    const out = [];
    const ts = tree.modules.flatMap(m => m.templates);
    const lmap = {};
    tree.links.forEach(l => { (lmap[l.tpl] = lmap[l.tpl] || []).push(l); });
    for (let i = 0; i < ts.length; i++) {
      for (let j = i + 1; j < ts.length; j++) {
        const a = ts[i], b = ts[j];
        const al = lmap[a.id] || [], bl = lmap[b.id] || [];
        const af = al.filter(l => l.kind !== 'x').map(l => l.ftr);
        const bf = bl.filter(l => l.kind !== 'x').map(l => l.ftr);
        const shared = af.filter(f => bf.includes(f));
        if (shared.length) out.push({ a: a.id, b: b.id, type: 'shared', label: shared.join(' ') });
        if (a.family === b.family && a.family) out.push({ a: a.id, b: b.id, type: 'family', label: a.family });
        const ax = al.filter(l => l.kind === 'x').map(l => l.ftr);
        const bx = bl.filter(l => l.kind === 'x').map(l => l.ftr);
        const xs = ax.filter(f => bx.includes(f));
        if (xs.length) out.push({ a: a.id, b: b.id, type: 'xbridge', label: xs.join(' ') });
      }
    }
    return out;
  }, [tree]);
  const linksByTpl = useMemo(() => {
    const mp = {};
    (tree ? tree.links : []).forEach(l => { (mp[l.tpl] = mp[l.tpl] || []).push(l); });
    return mp;
  }, [tree]);

  useEffect(() => { setSel(null); }, [modOff]);

  /* 渲染循环 */
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !tree) return;
    const ctx = cv.getContext('2d');
    const st = stateRef.current;
    let raf = 0, W = 0, H = 0;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    const project = p => {
      const cy = Math.cos(st.yaw), sy = Math.sin(st.yaw), cp = Math.cos(st.pitch), sp = Math.sin(st.pitch);
      const x = p[0] * cy - p[2] * sy, z = p[0] * sy + p[2] * cy, y = p[1];
      const y2 = y * cp - z * sp, z2 = y * sp + z * cp;
      const f = 4.6, s = (f / (f + z2)) * Math.min(W, H) * 0.148 * st.zoom;
      return { x: W / 2 + x * s, y: H / 2 - y2 * s + 22, z: z2, s };
    };
    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      /* 地面网格 */
      for (let g = -4; g <= 4; g++) {
        const p1 = project([g, -0.95, -4]), p2 = project([g, -0.95, 4]);
        const p3 = project([-4, -0.95, g]), p4 = project([4, -0.95, g]);
        ctx.strokeStyle = g === 0 ? '#dfe4ea' : '#eef1f4';
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(p3.x, p3.y); ctx.lineTo(p4.x, p4.y); ctx.stroke();
      }
      /* 模块区域环 + 标签 */
      Object.entries(MOD_CENTER).forEach(([mid, c]) => {
        if (modOff[mid]) return;
        const m = modById[mid];
        ctx.beginPath();
        for (let k = 0; k <= 48; k++) {
          const a = k / 48 * Math.PI * 2, p = project([c[0] + 1.75 * Math.cos(a), -0.9, c[2] + 1.75 * Math.sin(a)]);
          k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y);
        }
        ctx.strokeStyle = (m ? m.color : '#888') + '55'; ctx.lineWidth = 1.2; ctx.stroke();
        ctx.fillStyle = (m ? m.color : '#888') + '0d'; ctx.fill();
        const lp = project([c[0], -0.9, c[2] + 1.95]);
        ctx.font = '700 12px "Segoe UI","Microsoft YaHei"'; ctx.textAlign = 'center';
        ctx.fillStyle = m ? m.color : '#888'; ctx.fillText(m ? m.name : mid, lp.x, lp.y);
      });
      /* 投影缓存 */
      const tp = tpRef.current;
      templates.forEach(t => { tp[t.id] = project(t.pos); });
      /* 边 */
      edges.forEach(e => {
        const a = templates.find(t => t.id === e.a), b = templates.find(t => t.id === e.b);
        if (modOff[a.mod] || modOff[b.mod]) return;
        const pa = tp[e.a], pb = tp[e.b];
        const hl = sel && (sel === e.a || sel === e.b);
        const dim = sel && !hl;
        const col = e.type === 'shared' ? '#64748b' : e.type === 'family' ? '#c084fc' : '#d85a30';
        ctx.strokeStyle = col;
        ctx.globalAlpha = dim ? 0.06 : hl ? 0.95 : (e.type === 'shared' ? 0.45 : 0.5);
        ctx.lineWidth = hl ? 2 : 1.1;
        ctx.setLineDash(e.type === 'xbridge' ? [5, 4] : []);
        ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
        if (showLbl && (hl || (!sel && e.type !== 'family'))) {
          ctx.font = '9.5px Consolas,monospace'; ctx.textAlign = 'center';
          ctx.fillStyle = dim ? '#cbd2da' : col;
          ctx.fillText(e.label, (pa.x + pb.x) / 2, (pa.y + pb.y) / 2 - 3);
        }
      });
      /* 节点（深度排序） */
      const nodes = templates.filter(t => !modOff[t.mod]).map(t => ({ t, p: tp[t.id] })).sort((a, b) => b.p.z - a.p.z);
      nodes.forEach(({ t, p }) => {
        const hlSel = sel === t.id;
        const dim = sel && !hlSel && !edges.some(e => (e.a === sel && e.b === t.id) || (e.b === sel && e.a === t.id));
        const r = (hlSel ? 9 : 6.5) * (0.75 + p.s / (Math.min(W, H) * 0.12));
        ctx.globalAlpha = dim ? 0.18 : 1;
        if (hlSel) { ctx.beginPath(); ctx.arc(p.x, p.y, r + 7, 0, 7); ctx.fillStyle = modById[t.mod].color + '33'; ctx.fill(); }
        ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 7);
        ctx.fillStyle = t.gap ? '#fef2f2' : modById[t.mod].color;
        ctx.fill();
        ctx.lineWidth = t.gap ? 1.6 : 1.2;
        ctx.strokeStyle = t.gap ? '#dc2626' : '#ffffff'; ctx.stroke();
        if (t.gap) { ctx.beginPath(); ctx.arc(p.x, p.y, 2.4, 0, 7); ctx.fillStyle = '#dc2626'; ctx.fill(); }
        ctx.font = (hlSel ? '700 ' : '') + '11px "Segoe UI","Microsoft YaHei"'; ctx.textAlign = 'center';
        ctx.fillStyle = dim ? '#c4ccd4' : '#1c2024';
        ctx.fillText(t.name, p.x, p.y - r - 5);
        ctx.globalAlpha = 1;
      });
    };
    const tick = () => {
      resize();
      draw();
      if (autoRot && !st.drag) st.yaw += 0.0028;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [tree, edges, sel, autoRot, showLbl, modOff, templates, modById]);

  /* 交互 */
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const st = stateRef.current;
    const onDown = e => { st.drag = true; st.moved = 0; st.lx = e.clientX; st.ly = e.clientY; };
    const onMove = e => {
      if (st.drag) {
        st.yaw += (e.clientX - st.lx) * 0.006;
        st.pitch = Math.max(-1.2, Math.min(1.35, st.pitch + (e.clientY - st.ly) * 0.006));
        st.moved += Math.abs(e.clientX - st.lx) + Math.abs(e.clientY - st.ly);
        st.lx = e.clientX; st.ly = e.clientY;
      }
    };
    const onUp = e => {
      if (st.drag && st.moved < 5) {
        const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
        let best = null, bd = 16;
        templates.forEach(t => {
          if (modOff[t.mod]) return;
          const p = tpRef.current[t.id];
          if (!p) return;
          const d = Math.hypot(p.x - mx, p.y - my);
          if (d < bd) { bd = d; best = t.id; }
        });
        setSel(best);
      }
      st.drag = false;
    };
    const onWheel = e => {
      e.preventDefault();
      st.zoom = Math.max(0.45, Math.min(2.6, st.zoom * (e.deltaY > 0 ? 0.92 : 1.09)));
    };
    cv.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    cv.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      cv.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      cv.removeEventListener('wheel', onWheel);
    };
  }, [templates, modOff]);

  if (err) return <div className="at-err">模板空间数据不可达（{err}）——运行 python server/derive_atlas_tree.py 后重试。</div>;
  if (!tree) return <div className="at-note">加载语言模板 3D 关系空间（atlas_tree_v2）…</div>;

  const selT = sel ? templates.find(t => t.id === sel) : null;
  const selMod = selT ? modById[selT.mod] : null;
  const demoOf = id => LANG_TEMPLATES.find(t => t.id === id);
  const rel = sel ? edges.filter(e => e.a === sel || e.b === sel) : [];

  return (
    <div className="ats-root">
      <div className="ats-toolbar">
        <button type="button" className={'amv-stbtn' + (autoRot ? ' on' : '')} onClick={() => setAutoRot(v => !v)}>⟳ 自动旋转</button>
        <button type="button" className="amv-stbtn" onClick={() => { stateRef.current.yaw = 0.65; stateRef.current.pitch = 0.42; stateRef.current.zoom = 1; }}>复位视角</button>
        <span className="ats-sep" />
        {tree.modules.map(m => (
          <button key={m.id} type="button" className={'ats-mchip' + (modOff[m.id] ? ' off' : '')} onClick={() => setModOff(v => ({ ...v, [m.id]: !v[m.id] }))}>
            <i style={{ background: m.color }} />{m.name}
          </button>
        ))}
        <span className="ats-sep" />
        <button type="button" className={'amv-stbtn' + (showLbl ? ' on' : '')} onClick={() => setShowLbl(v => !v)}>边标签</button>
        <span className="ats-hint">拖拽旋转 · 滚轮缩放 · 点击节点看测试详情</span>
      </div>
      <canvas ref={canvasRef} className="ats-cv" />
      <div className="ats-legend">
        <span className="li"><span className="sw" />共享编码机制（标签 = 共享 FTR）</span>
        <span className="li"><span className="sw fam" />同模板家族</span>
        <span className="li"><span className="sw x" />跨模块桥（经交叉特征）</span>
        <span className="li"><i className="gd" />缺口模板（无机制对应）</span>
        <span className="note">示意布局：模块聚簇 + 派生布点（黄金角），非模型真实几何 · tree v2 sha8 {tree.tree_sha256_8}</span>
      </div>

      {selT && (
        <div className="ats-detail">
          <div className="amv-detail-h" style={{ '--mc': selMod.color }}>
            <b>{selT.name}</b>
            <span className="mono pat">{selT.pat}</span>
            <span className={'amv-st ' + selT.st}>{ST_NAME[selT.st]}</span>
            <span className="amv-st mono">{selT.tm || '—'}</span>
            {selT.metric && <span className="amv-st mono">{selT.metric}</span>}
            {selT.gap && <span className="amv-st gap">缺口</span>}
            <button type="button" className="amv-close" onClick={() => setSel(null)}>×</button>
          </div>
          <div className="amv-detail-grid">
            <div className="amv-dsec">
              <h6>测试输入 · 最大激活示例</h6>
              {(() => {
                const d = demoOf(selT.id);
                if (d && d.demo && d.demo.act_lines) {
                  return d.demo.act_lines.map((line, i) => (
                    <div key={i} className="amv-tokline">
                      {line.map(([tx, v], j) => (
                        <span key={j} className="amv-tok"
                          style={v > 0 ? { background: `rgba(52,211,153,${v})`, color: v > 0.6 ? '#022c22' : '#064e3b' } : undefined}>{tx}</span>
                      ))}
                      {i === 0 && <span className="amv-toknote">（DEMO 激活示例 · 着色=强度）</span>}
                    </div>
                  ));
                }
                return <div className="amv-dempty">示例：{selT.ex}（token 级激活样例待封存 npz 锚定后接入）</div>;
              })()}
            </div>
            <div className="amv-dsec">
              <h6>生产与采集（{selT.tm || '未分配 TM'}）</h6>
              <div className="amv-kv"><span>归属模块</span><b style={{ color: selMod.color }}>{selMod.name}</b> ｜ 家族 <b>{selT.family}</b></div>
              <div className="amv-kv"><span>节点 / cells</span><b className="mono">{selT.prod.nodes} 节点 · {selT.prod.cells} cells</b></div>
              <div className="amv-kv"><span>聚合状态</span><b className="mono">{selT.prod.agg}</b></div>
              <div className="amv-kv"><span>关联模板</span><b>{rel.length ? rel.map(e => {
                const oid = e.a === sel ? e.b : e.a;
                return templates.find(t => t.id === oid).name;
              }).join(' · ') : '无（孤立节点）'}</b></div>
              {selT.gap && <div className="amv-gapbox">⚠ {selT.gap}——建议立项补测</div>}
            </div>
            <div className="amv-dsec">
              <h6>对应 LLM 编码机制（模板 → 特征关联）</h6>
              {linksByTpl[selT.id] && linksByTpl[selT.id].length ? linksByTpl[selT.id].map(l => {
                const f = ftrById[l.ftr];
                return (
                  <div key={l.ftr} className="amv-ftrrow" onClick={() => onPickFeature && onPickFeature(l.ftr)}>
                    <span className={'amv-dot ' + (ECLASS[(f || {}).evidence_level] || 'at-e1')} />
                    <span className="mono fid">{l.ftr}</span>
                    <span className={'amv-eb ' + (ECLASS[(f || {}).evidence_level] || 'at-e1')}>{ELABEL[(f || {}).evidence_level] || '?'}</span>
                    <span className="fst">{(f || {}).statement || ''}
                      <span className="kind">{KIND_NAME[l.kind]}</span>
                    </span>
                  </div>
                );
              }) : <div className="amv-dempty">无特征锚定——列入缺口清单</div>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
