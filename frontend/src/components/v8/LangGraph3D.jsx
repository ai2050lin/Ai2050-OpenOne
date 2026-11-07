/* 语言图谱 · 关系图（种子，2026-11-07 审定）。可拖动或用方向键旋转的 3D 视图，右侧是同一节点的二维详情和键盘查询。
   布局是人工展示：区域 = 语言目录三大类，纵向 = 抽象程度；距离不代表模型几何。
   边按依据区分浓淡：有材料 / 只有标签或目录 / 人工定义无材料。“模型里是否这样组织”尚未测量，界面写“未测”。 */
import { useMemo, useRef, useState } from 'react';

import { BASIS, EDGES, EDGE_TYPES, NODES, STATE_LABEL, ZONES, neighbors } from './data/lang_graph_seed.js';
import './client_guide.css';

const W = 760, H = 480, SCALE = 172;
function project(view, x, y, z) {
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw), cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  const X = x * cy + z * sy, Z = -x * sy + z * cy, Y = y * cp - Z * sp, D = y * sp + Z * cp, f = 3.4 / (3.4 + D);
  return { x: W / 2 + X * SCALE * f, y: H / 2 - Y * SCALE * f, d: D, f };
}

export default function LangGraph3D() {
  const [view, setView] = useState({ yaw: -0.3, pitch: 0.16 });
  const [sel, setSel] = useState('apple');
  const [zone, setZone] = useState('');
  const [focus, setFocus] = useState(false);
  const drag = useRef(null);
  const node = NODES.find(item => item.id === sel);
  const near = useMemo(() => neighbors(sel), [sel]);
  const visible = item => focus ? near.has(item.id) : !zone || item.zone === zone || item.also === zone;
  const pts = Object.fromEntries(NODES.map(item => [item.id, project(view, item.x, item.y, item.z)]));
  const rotate = (dYaw, dPitch) => setView(v => ({ yaw: v.yaw + dYaw, pitch: Math.max(-0.2, Math.min(1, v.pitch + dPitch)) }));
  const rel = EDGES.filter(([a, b]) => a === sel || b === sel).map(([a, b, t, basis]) => [NODES.find(item => item.id === (a === sel ? b : a)), t, basis]);
  const counts = Object.fromEntries(Object.keys(STATE_LABEL).map(k => [k, NODES.filter(item => item.st === k).length]));

  return <section className="v8-card cg-section lg3-card" aria-label="语言关系图">
    <h3>语言关系图 <span className="mp-tag ok">种子 · 2026-11-07 审定</span></h3>
    <p className="v8-note">词形、词义、操作和规则分开成节点。先看人定义的关系；模型内部是否同样组织，要等实测。</p>
    <div className="lg3-grid">
      <div>
        <div className="v6-toolbar" role="group" aria-label="类别筛选">
          {[['', '全部'], ...Object.entries(ZONES).map(([k, z]) => [k, z.name])].map(([k, label]) =>
            <button type="button" key={label} className="fw-tbtn" aria-pressed={!focus && zone === k} onClick={() => { setZone(k); setFocus(false); }}>{label}</button>)}
          <button type="button" className="fw-tbtn" aria-pressed={focus} onClick={() => setFocus(!focus)}>只看相邻</button>
        </div>
        <svg className="lg3-svg" viewBox={`0 0 ${W} ${H}`} tabIndex={0} role="img" aria-label="语言关系图三维视图，拖动或用方向键旋转；右侧可用列表查询"
          onPointerDown={e => { drag.current = { x: e.clientX, y: e.clientY, moved: 0 }; }}
          onPointerMove={e => { const d = drag.current; if (!d) return; const dx = e.clientX - d.x, dy = e.clientY - d.y; d.moved += Math.abs(dx) + Math.abs(dy); d.x = e.clientX; d.y = e.clientY; if (d.moved > 4) rotate(dx * 0.008, dy * 0.006); }}
          onPointerUp={() => { setTimeout(() => { drag.current = null; }, 0); }}
          onPointerLeave={() => { drag.current = null; }}
          onKeyDown={e => { const m = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, 0.08], ArrowDown: [0, -0.08] }[e.key]; if (m) { e.preventDefault(); rotate(...m); } }}>
          {!focus && Object.entries(ZONES).filter(([k]) => !zone || zone === k).map(([k, z]) => { const p = project(view, z.x, 1.18, 0);
            return <text key={k} x={p.x} y={p.y} fontSize="13" fontWeight="650" fill="#16202c" textAnchor="middle">{z.name}</text>; })}
          {EDGES.map(([a, b, t, basis]) => {
            const A = NODES.find(item => item.id === a), B = NODES.find(item => item.id === b);
            if (!visible(A) || !visible(B)) return null;
            const [, dash, color] = EDGE_TYPES[t], [label, opacity, width] = BASIS[basis], hot = a === sel || b === sel;
            return <line key={`${a}-${b}`} x1={pts[a].x} y1={pts[a].y} x2={pts[b].x} y2={pts[b].y} stroke={color}
              strokeWidth={width + (hot ? 0.9 : 0)} strokeOpacity={hot ? Math.min(1, opacity + 0.3) : opacity} strokeDasharray={dash || undefined}>
              <title>{`${A.name} — ${B.name} · ${EDGE_TYPES[t][0]} · ${label}`}</title></line>;
          })}
          {NODES.filter(visible).sort((a, b) => pts[b.id].d - pts[a.id].d).map(item => {
            const p = pts[item.id], c = ZONES[item.zone].color, on = item.id === sel, r = (on ? 10 : 7.5) * p.f;
            return <g key={item.id} className="lg3-node" onClick={() => { if (!drag.current || drag.current.moved <= 4) setSel(item.id); }}>
              <title>{`${item.name} · ${STATE_LABEL[item.st]}`}</title>
              <circle cx={p.x} cy={p.y} r={r + 7} fill="transparent"/>
              <circle cx={p.x} cy={p.y} r={r} fill={item.st === 'real' ? c : '#fff'} stroke={c} strokeWidth={on ? 3 : 2} strokeDasharray={item.st === 'new' ? '3 3' : undefined}/>
              <text x={p.x} y={p.y - r - 5} fontSize={11.5 * Math.max(0.86, p.f)} fontWeight={on ? 700 : 450} fill="#16202c" textAnchor="middle" paintOrder="stroke" stroke="#fff" strokeWidth="3">{item.name}</text>
            </g>;
          })}
        </svg>
        <p className="v8-note">拖动或用方向键旋转。纵向是抽象程度，三个区域是展示分类，距离不代表模型几何。</p>
        <div className="lg3-legend">
          <span><i className="lg3-dot full"/>已有项目材料 {counts.real}</span><span><i className="lg3-dot"/>目录已有，未测 {counts.cat}</span><span><i className="lg3-dot dash"/>需新增 {counts.new}</span>
        </div>
        <div className="lg3-legend">{Object.values(EDGE_TYPES).map(([label, dash, color]) => <span key={label}>
          <svg width="30" height="8" aria-hidden="true"><line x1="0" y1="4" x2="30" y2="4" stroke={color} strokeWidth="2" strokeDasharray={dash || undefined}/></svg>{label}</span>)}</div>
        <div className="lg3-legend"><b>边的依据</b>{Object.entries(BASIS).map(([key, [label, opacity, width]]) => <span key={key}>
          <svg width="30" height="8" aria-hidden="true"><line x1="0" y1="4" x2="30" y2="4" stroke="#33404f" strokeWidth={width} strokeOpacity={opacity}/></svg>{label} {EDGES.filter(e => e[3] === key).length}</span>)}</div>
      </div>
      <aside className="lg3-side">
        <h4>{node.name} <span className={'mp-tag' + (node.st === 'real' ? ' ok' : '')}>{STATE_LABEL[node.st]}</span></h4>
        <dl className="mp-join-scope"><dt>类别</dt><dd>{ZONES[node.zone].name}{node.also ? `，也属于${ZONES[node.also].name}` : ''}</dd><dt>类型</dt><dd>{node.kind}</dd><dt>目录</dt><dd>{node.cat || '无'}</dd></dl>
        <p className="v8-note">{node.note}</p>
        <h4>关系</h4>
        <ul className="lg3-rel">{rel.map(([m, t, basis]) => <li key={m.id}><button type="button" className="v6-linkbtn" onClick={() => setSel(m.id)}>{m.name}</button><small>{EDGE_TYPES[t][0]} · {BASIS[basis][0]}</small></li>)}</ul>
        <h4>模型里是否这样组织</h4>
        <p className="v8-note"><span className="mp-tag">未测</span> 以后显示：关系是否读得出、未见词能否复现、干预后回答是否改变。</p>
        <label className="v8-note">键盘查询 <select value={sel} onChange={e => setSel(e.target.value)} aria-label="选择语言节点">
          {NODES.map(item => <option key={item.id} value={item.id}>{ZONES[item.zone].name} · {item.name}</option>)}</select></label>
      </aside>
    </div>
  </section>;
}
