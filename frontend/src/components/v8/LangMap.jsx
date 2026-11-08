/* 空间 · 语言图谱 · 结构图谱（Phase 341，lang_map.v1）。
   四层：领域 → 子领域 → 语言项 → 词。数据来自中心只读端点 /api/lang/map 与 /api/lang/map/words/{id}（随版本发布、带 sha256）。
   3D 总览与 2D 树共用同一个选中对象；右侧是详情。只描述语言本身，不含模型信息；布局距离不代表语义距离或模型几何。 */
import { useEffect, useMemo, useRef, useState } from 'react';

import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { ExtraPackPrompt } from '../open/DataBar.jsx';
import './lang_map.css';

const COL = { KN: '#2a78d6', RS: '#d4622b', GR: '#2f8f5b' };
const WORD_CAP = 48;               // 3D 里一个语言项最多展开的词数；其余在右侧列表
const DOM_ANG = { KN: -Math.PI / 2, RS: -Math.PI / 2 + 2 * Math.PI / 3, GR: -Math.PI / 2 + 4 * Math.PI / 3 };
const SECTOR = 2 * Math.PI / 3 * 0.86;

/* 把接口数据摊平成节点表：ROOT、领域、子领域、语言项；词按需加入 */
function buildIndex(map) {
  const N = new Map();
  const root = { id: 'ROOT', name: '语言图谱', level: 0, kids: [] };
  N.set('ROOT', root);
  map.domains.forEach(d => {
    N.set(d.id, { id: d.id, name: d.name, desc: d.desc, level: 1, dom: d.id, parent: 'ROOT', kids: d.subs.map(s => s.id) });
    root.kids.push(d.id);
    d.subs.forEach((s, si) => {
      N.set(s.id, { id: s.id, name: s.name, level: 2, dom: d.id, parent: d.id, si, ns: d.subs.length, kids: s.items.map(i => i.id) });
      s.items.forEach((it, ii) => N.set(it.id, { ...it, level: 3, dom: d.id, parent: s.id, ii, ni: s.items.length, kids: [] }));
    });
  });
  return N;
}

function pos(N, n) {
  if (n.level === 0) return [0, 3.2, 0];
  const a0 = DOM_ANG[n.dom];
  if (n.level === 1) return [Math.cos(a0) * 1.1, 2.2, Math.sin(a0) * 1.1];
  const subAng = s => a0 - SECTOR / 2 + SECTOR * (s.si + 0.5) / s.ns;
  if (n.level === 2) { const a = subAng(n); return [Math.cos(a) * 2.4, 1.0, Math.sin(a) * 2.4]; }
  const item = n.level === 3 ? n : N.get(n.parent), s = N.get(item.parent), span = SECTOR / s.ns * 0.9;
  const ia = subAng(s) - span / 2 + span * (item.ii + 0.5) / item.ni;
  if (n.level === 3) return [Math.cos(ia) * 3.6, -0.3, Math.sin(ia) * 3.6];
  const shown = Math.min(n.total, WORD_CAP), ring = Math.floor(n.wi / 16), m = Math.min(16, shown - ring * 16);
  const a = ia + (n.wi % 16 - (m - 1) / 2) * 0.05;
  return [Math.cos(a) * (4.5 + ring * 0.45), -1.5 - ring * 0.35, Math.sin(a) * (4.5 + ring * 0.45)];
}

const pathOf = (N, id) => { const p = []; let n = N.get(id); while (n) { p.unshift(n); n = n.parent ? N.get(n.parent) : null; } return p; };

function Map3D({ N, sel, focus, words, onPick }) {
  const cv = useRef(null), view = useRef({ yaw: 0.5, pitch: 0.78, zoom: 1 }), pts = useRef([]), size = useRef({ W: 0, H: 0 });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const el = cv.current;
    const resize = () => { const r = el.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      size.current = { W: r.width, H: r.height }; el.width = r.width * dpr; el.height = r.height * dpr;
      el.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0); setTick(t => t + 1); };
    resize();
    const ro = new ResizeObserver(resize); ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const el = cv.current, ctx = el.getContext('2d'), { W, H } = size.current;
    if (!W) return;
    const { yaw, pitch, zoom } = view.current;
    const proj = ([x, y, z]) => {
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = x * cy - z * sy, z1 = x * sy + z * cy, y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      const s = Math.min(W, H) / 10.5 * zoom, f = 12 / (12 + z2);
      return [W / 2 + x1 * s * f, H * 0.5 - y1 * s * f, z2, f];
    };
    const ids = ['ROOT', ...N.get('ROOT').kids];
    for (const d of N.get('ROOT').kids) for (const s of N.get(d).kids) { ids.push(s); ids.push(...N.get(s).kids); }
    const wordNodes = (words || []).slice(0, WORD_CAP);
    const all = new Map(ids.map(id => [id, N.get(id)]));
    wordNodes.forEach(w => all.set(w.id, w));
    const P = new Map([...all].map(([id, n]) => [id, proj(pos(N, n))]));
    const sn = all.get(sel) || N.get('ROOT');
    const path = new Set(pathOf(N, sn.level === 4 ? sn.parent : sel).map(n => n.id));
    if (sn.level === 4) path.add(sel);
    const dim = n => focus && n.dom && n.dom !== focus;
    ctx.clearRect(0, 0, W, H);
    for (const [id, n] of all) {
      if (!n.parent || !P.has(n.parent)) continue;
      const a = P.get(id), b = P.get(n.parent), on = path.has(id);
      ctx.strokeStyle = on ? COL[n.dom] : '#c3cad5'; ctx.globalAlpha = dim(n) ? 0.12 : on ? 0.9 : 0.55; ctx.lineWidth = on ? 1.6 : 0.8;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    if (sn.level === 3) for (const t of sn.also || []) { if (!P.has(t)) continue;
      const a = P.get(sel), b = P.get(t); ctx.setLineDash([4, 4]); ctx.strokeStyle = '#d29a00'; ctx.globalAlpha = 0.9; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.setLineDash([]); }
    const fnEdges = sn.level === 4 ? sn.fn.map(t => [sel, t]) : sn.level === 3 ? wordNodes.flatMap(w => w.fn.map(t => [w.id, t])) : [];
    for (const [u, v] of fnEdges) { if (!P.has(u) || !P.has(v)) continue; const a = P.get(u), b = P.get(v);
      ctx.strokeStyle = '#7050a0'; ctx.globalAlpha = 0.75; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(a[0], a[1]);
      ctx.quadraticCurveTo((a[0] + b[0]) / 2, Math.max(a[1], b[1]) + 30, b[0], b[1]); ctx.stroke(); }
    ctx.globalAlpha = 1;
    const fnTargets = new Set(sn.level === 4 ? sn.fn : []);
    pts.current = [...P].map(([id, p]) => ({ id, p })).sort((u, v) => v.p[2] - u.p[2]);
    for (const { id, p } of pts.current) {
      const n = all.get(id), r = [9, 8, 6.5, 4.2, 3][n.level] * p[3] * zoom ** 0.5, c = n.level === 0 ? '#172033' : COL[n.dom];
      ctx.globalAlpha = dim(n) ? 0.15 : 1;
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      if (n.level === 3) {
        ctx.fillStyle = n.flags[3] ? c : n.flags[2] ? `${c}aa` : '#fff'; ctx.fill(); ctx.strokeStyle = c; ctx.lineWidth = 1.4; ctx.stroke();
        if (n.origin === 'new') { ctx.setLineDash([2, 2]); ctx.beginPath(); ctx.arc(p[0], p[1], r + 3, 0, Math.PI * 2); ctx.strokeStyle = '#a8670a'; ctx.stroke(); ctx.setLineDash([]); }
      } else { ctx.fillStyle = n.level === 4 && n.review !== 'reviewed' ? '#7050a0' : c; ctx.fill(); }
      if (id === sel) { ctx.beginPath(); ctx.arc(p[0], p[1], r + 5, 0, Math.PI * 2); ctx.strokeStyle = '#d29a00'; ctx.lineWidth = 2.5; ctx.stroke(); }
      const label = n.level <= 1 || id === sel || (n.level === 2 && (!focus || n.dom === focus))
        || (n.level === 3 && (path.has(n.parent) || fnTargets.has(id) || (sn.level === 3 && sn.also?.includes(n.parent))))
        || (n.level === 4 && n.wi < 24);
      if (label && !dim(n)) { ctx.fillStyle = n.level <= 1 ? '#172033' : '#3e4b60';
        ctx.font = `${n.level <= 1 ? 'bold 13px' : n.level === 2 ? '600 12px' : '11px'} "Microsoft YaHei",sans-serif`; ctx.fillText(n.name, p[0] + r + 3, p[1] + 4); }
    }
    ctx.globalAlpha = 1;
  }, [N, sel, focus, words, tick]);

  const drag = useRef(null);
  const down = e => { drag.current = { x: e.clientX, y: e.clientY, ...view.current, moved: false }; e.currentTarget.setPointerCapture(e.pointerId); };
  const move = e => { const d = drag.current; if (!d) return; const dx = e.clientX - d.x, dy = e.clientY - d.y; if (Math.abs(dx) + Math.abs(dy) > 4) d.moved = true;
    view.current = { ...view.current, yaw: d.yaw + dx * 0.006, pitch: Math.max(-0.2, Math.min(1.3, d.pitch + dy * 0.005)) }; setTick(t => t + 1); };
  const up = e => { const d = drag.current; drag.current = null; if (!d || d.moved) return;
    const r = e.currentTarget.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    let best = null, bd = 14; for (const { id, p } of pts.current) { const dd = Math.hypot(p[0] - x, p[1] - y); if (dd < bd) { bd = dd; best = id; } }
    if (best) onPick(best); };
  useEffect(() => {
    const el = cv.current;
    const wheel = e => { e.preventDefault(); view.current = { ...view.current, zoom: Math.max(0.6, Math.min(2.6, view.current.zoom * (e.deltaY < 0 ? 1.1 : 0.9))) }; setTick(t => t + 1); };
    el.addEventListener('wheel', wheel, { passive: false });
    return () => el.removeEventListener('wheel', wheel);
  }, []);
  return <div className="lm-3d">
    <canvas ref={cv} onPointerDown={down} onPointerMove={move} onPointerUp={up} aria-label="语言图谱 3D 总览：三个领域分成三个扇区，从上到下是领域、子领域、语言项和词" />
    <div className="lm-hint v8-note">拖动旋转 · 滚轮缩放 · 点节点选中；选中语言项后展开它的词</div>
    <div className="lm-legend v8-note"><span><i style={{ borderColor: '#c3cad5' }} />层级</span><span><i style={{ borderColor: '#d29a00', borderTopStyle: 'dashed' }} />交叉归属</span>
      <span><i style={{ borderColor: '#7050a0' }} />词 → 功能</span><span>实心 = 有测试模板 · 空心 = 只有定义与例句 · 虚线外圈 = 新增</span></div>
  </div>;
}

function Tree({ N, sel, words, onPick }) {
  const host = sel.includes('#') ? sel.split('#')[0] : sel;          // 选中的是词时，展开到它所在的语言项
  const open = new Set(pathOf(N, host).map(n => n.id));
  const fill = n => n.level === 3 ? (n.flags[3] ? COL[n.dom] : n.flags[2] ? `${COL[n.dom]}aa` : '#fff') : n.level ? COL[n.dom] : '#172033';
  const dot = n => <span className="lm-dot" style={{ background: fill(n), borderColor: n.level ? COL[n.dom] : '#172033' }} />;
  const node = id => {
    const n = N.get(id);
    if (n.level === 3) {
      const ws = id === host ? words : null;
      return <details key={id} open={open.has(id) || undefined}>
        <summary className={id === sel ? 'sel' : ''} onClick={e => { e.preventDefault(); onPick(id); }}>{dot(n)}{n.name}{n.origin === 'new' && <span className="lm-tag new">新</span>}<span className="lm-cnt">{n.n_words} 词</span></summary>
        {ws && ws.slice(0, 80).map(w => <div key={w.id} className={`lm-leaf${w.id === sel ? ' sel' : ''}`} onClick={() => onPick(w.id)}>{w.name}{w.en && <span className="lm-cnt">{w.en}</span>}</div>)}
        {ws && ws.length > 80 && <div className="lm-leaf v8-note">另 {ws.length - 80} 个词在右侧列表</div>}
      </details>;
    }
    return <details key={id} open={open.has(id) || n.level === 0 || undefined}>
      <summary className={id === sel ? 'sel' : ''} onClick={e => { e.preventDefault(); onPick(id); }}>{dot(n)}<b>{n.name}</b>
        <span className="lm-cnt">{n.level === 1 ? `${n.kids.reduce((a, s) => a + N.get(s).kids.length, 0)} 项` : n.level === 2 ? `${n.kids.length} 项` : ''}</span></summary>
      {n.kids.map(node)}
    </details>;
  };
  return <div className="lm-tree" role="tree" aria-label="语言图谱目录树">{node('ROOT')}</div>;
}

function Detail({ N, map, sel, words, wordsApi, onPick }) {
  const n = N.get(sel) || (words || []).find(w => w.id === sel);
  if (!n) return null;
  const c = n.level ? COL[n.dom] : '#172033';
  const crumb = pathOf(N, n.level === 4 ? n.parent : sel).concat(n.level === 4 ? [n] : []);
  const kidList = ids => <div className="lm-kids">{ids.map(k => { const m = N.get(k);
    return <button type="button" key={k} onClick={() => onPick(k)}><span><span className="lm-dot" style={{ background: COL[m.dom], borderColor: COL[m.dom] }} />{m.name}</span>
      <span className="v8-note">{m.level === 1 ? `${m.kids.length} 子领域` : m.level === 2 ? `${m.kids.length} 项` : `${m.n_words} 词`}</span></button>; })}</div>;
  return <aside className="lm-side" aria-label="语言图谱详情">
    <div className="lm-crumb">{crumb.map((p, i) => <span key={p.id}>{i > 0 && ' › '}<button type="button" onClick={() => onPick(p.id)}>{p.name}</button></span>)}</div>
    <h4 style={{ color: c }}>{n.name}{n.origin === 'new' && <span className="lm-tag new">新</span>}</h4>
    {n.level === 0 && <><p className="v8-note">四层：领域 → 子领域 → 语言项 → 词。共 {map.stats.subs} 个子领域、{map.stats.items} 个语言项、{map.stats.words} 个词，另有 {map.stats.dims} 个横切维度。</p>{kidList(n.kids)}</>}
    {n.level === 1 && <><p>{n.desc}</p>{kidList(n.kids)}</>}
    {n.level === 2 && kidList(n.kids)}
    {n.level === 3 && <>
      <div className="lm-ladder">{map.status.map((s, i) => <span key={s} className={n.flags[i] ? 'y' : ''}>{s}</span>)}</div>
      {n.example && <div className="lm-ex">{n.example}</div>}
      <p className="v8-note">ID {n.id} · {n.origin === 'catalog' ? '来自语言目录 v3' : '新增，待审定'}</p>
      {n.also.length > 0 && <p><span className="v8-note">也属于 </span>{n.also.map(a => <button type="button" className="lm-tag" key={a} onClick={() => onPick(a)}>{N.get(a)?.name || a}</button>)}</p>}
      {n.templates.length > 0 && <p><span className="v8-note">测试模板 </span>{n.templates.map(t => <span className="lm-tag" key={t}>{t}</span>)}</p>}
      {n.fn_words.length > 0 && <><p className="v8-note">表达这个功能的词</p>
        <div className="lm-chips">{n.fn_words.map(w => <button type="button" key={`${w.item}:${w.zh}`} onClick={() => onPick(w.item)}>{w.zh}</button>)}</div></>}
      {n.n_words > 0 ? <>
        <p className="v8-note">词层 · {n.n_words} 个{words?.some(w => w.review !== 'reviewed') ? ' · 含草稿，待审定' : ''}</p>
        {wordsApi.error ? <Empty>{wordsApi.error}<ExtraPackPrompt pack={wordsApi.missingPack} /></Empty>
          : !words ? <p className="v8-note">读取词层…</p>
            : <div className="lm-chips">{words.map(w => <button type="button" key={w.id} title={w.note || ''} onClick={() => onPick(w.id)}>{w.name}</button>)}</div>}
      </> : <p className="v8-note">还没有词层。</p>}
    </>}
    {n.level === 4 && <>
      <p>{n.en && <span className="lm-tag">{n.en}</span>}<span className={`lm-tag${n.review !== 'reviewed' ? ' draft' : ''}`}>{n.src}{n.review !== 'reviewed' ? ' · 待审定' : ''}</span></p>
      {n.note && <p className="v8-note">{n.note}</p>}
      {n.fn.length > 0 && <><p className="v8-note">表达的功能</p><div className="lm-chips">{n.fn.map(t => <button type="button" key={t} onClick={() => onPick(t)}>{N.get(t)?.name || t}</button>)}</div></>}
      <p className="v8-note">词、义项与 token 分开记录；同形不同义的词作为不同义项挂在不同位置。</p>
    </>}
  </aside>;
}

export default function LangMap() {
  const api = useApi('/api/lang/map');
  const map = api.data?.domains ? api.data : null;
  const N = useMemo(() => (map ? buildIndex(map) : null), [map]);
  const [sel, setSel] = useState('ROOT');
  const [focus, setFocus] = useState(null);
  const [mode, setMode] = useState('3d');
  const [q, setQ] = useState('');
  const itemId = sel.includes('#') ? sel.split('#')[0] : N?.get(sel)?.level === 3 ? sel : null;
  const wordsApi = useApi(itemId && N?.get(itemId)?.n_words ? `/api/lang/map/words/${itemId}` : null);
  const words = useMemo(() => {
    if (!itemId || !wordsApi.data?.words || wordsApi.data.item !== itemId) return null;
    const it = N.get(itemId), list = wordsApi.data.words;
    return list.map((w, wi) => ({ id: `${itemId}#${wi}`, name: w.zh, en: w.en, note: w.note, src: w.src, fn: w.fn, review: w.review,
      level: 4, dom: it.dom, parent: itemId, wi, total: list.length }));
  }, [itemId, wordsApi.data, N]);
  if (api.error) return <section className="v8-card"><Empty>语言图谱暂不可用。{api.error}<ExtraPackPrompt pack={api.missingPack} /></Empty></section>;
  if (!map) return <section className="v8-card"><p className="v8-note">读取语言图谱…</p></section>;
  const pick = id => { setSel(id); const n = N.get(id); if (n?.dom && focus && n.dom !== focus) setFocus(null); };
  const search = () => {
    const t = q.trim(); if (!t) return;
    const nodes = [...N.values()];
    const hit = nodes.find(n => n.level >= 1 && n.name === t)
      || nodes.flatMap(n => (n.fn_words || []).filter(w => w.zh === t).map(w => w.item))[0]
      || nodes.find(n => n.level >= 1 && n.name.includes(t));
    if (hit) pick(typeof hit === 'string' ? hit : hit.id);
  };
  return <div className="lm" aria-label="语言图谱结构">
    <div className="lm-tool">
      <div className="lm-doms">{map.domains.map(d => <button type="button" key={d.id} style={{ color: COL[d.id] }} aria-pressed={focus === d.id}
        onClick={() => { const next = focus === d.id ? null : d.id; setFocus(next); setSel(next || 'ROOT'); }}>
        <b>{d.name}</b><small>{d.subs.length} 子领域 · {d.subs.reduce((a, s) => a + s.items.length, 0)} 语言项 · {d.subs.reduce((a, s) => a + s.items.reduce((b, i) => b + i.n_words, 0), 0)} 词</small></button>)}</div>
      <div className="lm-search">
        <input value={q} onChange={e => setQ(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') search(); }} placeholder="搜索语言项或虚词，如 否定、把" aria-label="搜索语言图谱" />
        <div className="lm-seg" role="group" aria-label="视图">{[['3d', '3D 总览'], ['tree', '2D 树']].map(([id, label]) =>
          <button type="button" key={id} aria-pressed={mode === id} onClick={() => setMode(id)}>{label}</button>)}</div>
      </div>
    </div>
    <section className="v8-card lm-main">
      <div className="lm-view">{mode === '3d' ? <Map3D N={N} sel={sel} focus={focus} words={words} onPick={pick} /> : <Tree N={N} sel={sel} words={words} onPick={pick} />}</div>
      <Detail N={N} map={map} sel={sel} words={words} wordsApi={wordsApi} onPick={pick} />
    </section>
    <p className="v8-note">版本 {map.version} · {map.manifest?.created?.slice(0, 10)} · lang_map.json sha256 {map.manifest?.files?.['lang_map.json']?.slice(0, 12)}… · {map.note}</p>
  </div>;
}

/* 横切维度：语种、语码混用、方言、语域、修辞、表述变换等，作为每个节点的标签，不单独成域 */
export function LangMapDims() {
  const api = useApi('/api/lang/map');
  const dims = api.data?.dims;
  if (!dims) return <p className="v8-note">{api.error || '读取…'}</p>;
  return <div className="v8-weight-scroll"><table className="v6-table"><thead><tr><th>维度</th><th>例子</th><th>测试模板</th></tr></thead>
    <tbody>{dims.map(d => <tr key={d.id}><td>{d.name}<br /><span className="v8-note">{d.id}</span></td><td>{d.example}</td><td>{d.templates.join('、') || '—'}</td></tr>)}</tbody></table></div>;
}
