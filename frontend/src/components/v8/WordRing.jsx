/* 语言图谱 · 词汇关系环图（环形层级 + 属性枢纽）。
   外圈：每个词一个点，按“大类 → 类别 → 词”排在圆上，外侧是类别与大类弧带。
   圈内：属性（颜色、大小）是枢纽，每个词连向自己的属性。边数随词数线性增长，几千个词也能放下；
   全图只画点，选大类、搜索或点击时才显示词名与关系。数据只来自外部词表，不含模型信息。 */
import { useMemo, useState } from 'react';

import { CAT_ZH, COLOR_HEX, COLOR_ZH, SIZE_ZH, SOURCES, SUPERS, WORDS } from './data/langWords.js';
import './client_guide.css';

const W = 860, C = W / 2, R = 300;
const SUPER_COLOR = { animal: '#2a78d6', plant: '#1baf7a', food: '#c98a00', artifact: '#7050a0', nature: '#8a5a2b' };
const LABEL_LIMIT = 200;

function layout(words) {
  /* 类别之间空 1.5 个位置，大类之间空 4 个位置 */
  const slots = [];
  let pos = 0;
  words.forEach((w, i) => {
    const prev = words[i - 1];
    if (prev) pos += prev.sup !== w.sup ? 4 : prev.cat !== w.cat ? 1.5 : 1;
    slots.push(pos);
  });
  const total = pos + 4;
  return words.map((w, i) => ({ ...w, a: (2 * Math.PI * slots[i]) / total - Math.PI / 2 }));
}
const pt = (a, r) => [C + r * Math.cos(a), C + r * Math.sin(a)];
function arc(a0, a1, r0, r1) {
  const large = a1 - a0 > Math.PI ? 1 : 0;
  const [x0, y0] = pt(a0, r1), [x1, y1] = pt(a1, r1), [x2, y2] = pt(a1, r0), [x3, y3] = pt(a0, r0);
  return `M${x0},${y0} A${r1},${r1} 0 ${large} 1 ${x1},${y1} L${x2},${y2} A${r0},${r0} 0 ${large} 0 ${x3},${y3} Z`;
}
function groups(placed, key) {
  const out = [];
  for (const w of placed) {
    const last = out[out.length - 1];
    if (last && last.key === w[key]) { last.a1 = w.a; last.n += 1; } else out.push({ key: w[key], sup: w.sup, a0: w.a, a1: w.a, n: 1 });
  }
  return out;
}

export default function WordRing() {
  const [sup, setSup] = useState('');
  const [mode, setMode] = useState('color');
  const [query, setQuery] = useState('');
  const [sel, setSel] = useState(null);          // { type: 'word' | 'hub', id }
  const words = useMemo(() => layout(sup ? WORDS.filter(w => w.sup === sup) : WORDS), [sup]);
  const step = words.length > 1 ? Math.abs(words[1].a - words[0].a) : 0.1;
  const cats = groups(words, 'cat'), sups = groups(words, 'sup');
  const hubs = useMemo(() => {
    const keys = mode === 'color' ? Object.keys(COLOR_HEX).filter(c => words.some(w => w.color === c)) : mode === 'size' ? ['S', 'L'].filter(s => words.some(w => w.size === s)) : [];
    return keys.map((k, i) => {
      const a = (2 * Math.PI * i) / keys.length - Math.PI / 2, r = mode === 'size' ? 70 : 120;
      return { k, a, x: pt(a, r)[0], y: pt(a, r)[1], n: words.filter(w => (mode === 'color' ? w.color : w.size) === k).length };
    });
  }, [mode, words]);
  const hubOf = w => hubs.find(h => h.k === (mode === 'color' ? w.color : w.size));
  const q = query.trim().toLowerCase();
  const hits = q ? new Set(words.filter(w => w.zh.includes(q) || w.en.toLowerCase().includes(q)).map(w => w.id)) : null;
  const selWord = sel?.type === 'word' ? WORDS.find(w => w.id === sel.id) : null;
  const active = w => (sel?.type === 'word' && (w.id === sel.id || (mode === 'cat' ? w.cat === selWord.cat : hubOf(w) && hubOf(w) === hubOf(selWord)))) || (sel?.type === 'hub' && hubOf(w)?.k === sel.id);
  const anyActive = Boolean(sel);
  const showAll = words.length <= LABEL_LIMIT;

  const edges = mode === 'cat' ? [] : words.filter(hubOf).map(w => {
    const h = hubOf(w), [x, y] = pt(w.a, R - 6), [cx, cy] = pt(w.a, R * 0.55);
    const on = active(w);
    return <path key={w.id} d={`M${x},${y} Q${cx},${cy} ${h.x},${h.y}`} fill="none"
      stroke={mode === 'color' ? COLOR_HEX[w.color] : '#566373'} strokeWidth={on ? 1.6 : 0.8} strokeOpacity={anyActive ? (on ? 0.9 : 0.06) : 0.28}/>;
  });
  const sideList = selWord ? WORDS.filter(w => w.id !== selWord.id && (mode === 'size' ? w.size && w.size === selWord.size : mode === 'color' ? w.color && w.color === selWord.color : w.cat === selWord.cat))
    : sel?.type === 'hub' ? WORDS.filter(w => (mode === 'color' ? w.color : w.size) === sel.id) : [];

  return <section className="v8-card cg-section" aria-label="词汇关系环图">
    <h3>词汇关系环图</h3>
    <p className="v8-note">外圈每个点是一个词，按大类、类别排列；圈内的枢纽是属性，词连向自己的属性，同一枢纽上的词共享这一属性。
      共 {WORDS.length} 个词（{SOURCES.map(([s, n]) => `${s} ${n}`).join('、')}）。颜色与大小只在 ENT-XF 词表上有，是人工标注的常识属性。</p>
    <div className="v6-toolbar" role="group" aria-label="大类">
      {[['', '全部'], ...SUPERS].map(([k, label]) => <button type="button" key={k || 'all'} className="fw-tbtn" aria-pressed={sup === k}
        onClick={() => { setSup(k); setSel(null); }}>{label}{k ? ` ${WORDS.filter(w => w.sup === k).length}` : ''}</button>)}
    </div>
    <div className="v6-toolbar">
      <span className="v8-note">关系</span>
      {[['color', '颜色'], ['size', '大小'], ['cat', '只看类别']].map(([k, label]) => <button type="button" key={k} className="fw-tbtn" aria-pressed={mode === k}
        onClick={() => { setMode(k); setSel(null); }}>{label}</button>)}
      <input type="search" className="ring-search" placeholder="搜索词（中文或英文）" aria-label="搜索词" value={query}
        onChange={e => setQuery(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && hits?.size) setSel({ type: 'word', id: [...hits][0] }); }}/>
      {hits && <span className="v8-note">{hits.size} 个匹配，回车选中第一个</span>}
    </div>
    <div className="lg3-grid">
      <svg className="ring-svg" viewBox={`0 0 ${W} ${W}`} role="img" aria-label="词汇关系环图：外圈是词，圈内是属性枢纽">
        {sups.map(g => { const [lx, ly] = pt((g.a0 + g.a1) / 2, R + 52);
          return <g key={`s${g.key}${g.a0}`}><path d={arc(g.a0 - step / 2, g.a1 + step / 2, R + 30, R + 36)} fill={SUPER_COLOR[g.key]}/>
            <text x={lx} y={ly} fontSize="13" fontWeight="650" fill="#16202c" textAnchor="middle" dominantBaseline="middle">{SUPERS.find(s => s[0] === g.key)[1]}</text></g>; })}
        {cats.map(g => { const mid = (g.a0 + g.a1) / 2, [lx, ly] = pt(mid, R + 22);
          return <g key={`c${g.key}`}><path d={arc(g.a0 - step / 2, g.a1 + step / 2, R + 8, R + 16)} fill={SUPER_COLOR[g.sup]} fillOpacity="0.35"><title>{`${CAT_ZH[g.key]} · ${g.n} 个词`}</title></path>
            {!showAll && g.n >= 12 && <text x={lx} y={ly} fontSize="10" fill="#33404f" textAnchor="middle" dominantBaseline="middle">{CAT_ZH[g.key]}</text>}</g>; })}
        {edges}
        {hubs.map(h => <g key={h.k} className="ring-hub" onClick={() => setSel(sel?.type === 'hub' && sel.id === h.k ? null : { type: 'hub', id: h.k })}>
          <circle cx={h.x} cy={h.y} r={8 + Math.sqrt(h.n)} fill={mode === 'color' ? COLOR_HEX[h.k] : '#566373'} stroke="#fff" strokeWidth="2"
            strokeOpacity="1" opacity={sel?.type === 'hub' && sel.id !== h.k ? 0.35 : 1}><title>{`${mode === 'color' ? COLOR_ZH[h.k] : SIZE_ZH[h.k]} · ${h.n} 个词`}</title></circle>
          <text x={h.x} y={h.y + 13 + Math.sqrt(h.n)} fontSize="10.5" fill="#16202c" textAnchor="middle" dominantBaseline="hanging">{mode === 'color' ? COLOR_ZH[h.k] : SIZE_ZH[h.k]} {h.n}</text></g>)}
        {words.map(w => {
          const [x, y] = pt(w.a, R), on = active(w), hit = hits?.has(w.id), label = showAll || on || hit;
          const deg = (w.a * 180) / Math.PI, flip = deg > 90 && deg < 270;
          return <g key={w.id} className="ring-word" onClick={() => setSel(sel?.type === 'word' && sel.id === w.id ? null : { type: 'word', id: w.id })}>
            <circle cx={x} cy={y} r={sel?.id === w.id ? 5 : hit ? 4 : 3} fill={w.color && mode === 'color' ? COLOR_HEX[w.color] : '#fff'}
              stroke={hit ? '#e47a1c' : SUPER_COLOR[w.sup]} strokeWidth={hit || sel?.id === w.id ? 2 : 1.2} opacity={anyActive && !on ? 0.35 : 1}>
              <title>{`${w.zh} ${w.en} · ${CAT_ZH[w.cat]}`}</title></circle>
            {label && <text x={x} y={y} fontSize={showAll ? 8.5 : 10} fill="#16202c" dominantBaseline="middle"
              transform={`rotate(${flip ? deg + 180 : deg} ${x} ${y}) translate(${flip ? -7 : 7} 0)`} textAnchor={flip ? 'end' : 'start'}
              fontWeight={sel?.id === w.id ? 700 : 400}>{w.zh}</text>}
          </g>;
        })}
      </svg>
      <aside className="lg3-side">
        {selWord ? <>
          <h4>{selWord.zh} <small className="v8-note">{selWord.en}</small></h4>
          <dl className="mp-join-scope"><dt>大类</dt><dd>{SUPERS.find(s => s[0] === selWord.sup)[1]}</dd><dt>类别</dt><dd>{CAT_ZH[selWord.cat]}</dd>
            <dt>颜色</dt><dd>{selWord.color ? `${COLOR_ZH[selWord.color]}${selWord.alt.length ? `（也可：${selWord.alt.map(c => COLOR_ZH[c] || c).join('、')}）` : ''}` : '无标签'}</dd>
            <dt>大小</dt><dd>{selWord.size ? SIZE_ZH[selWord.size] : '无标签'}</dd><dt>来源</dt><dd>{selWord.source}</dd></dl>
          <h4>{mode === 'color' ? '同色' : mode === 'size' ? '同样大小' : '同类'}的词 · {sideList.length}</h4>
        </> : sel?.type === 'hub' ? <h4>{mode === 'color' ? `${COLOR_ZH[sel.id]}色` : SIZE_ZH[sel.id]}的词 · {sideList.length}</h4>
          : <p className="v8-note">点一个词看它的属性和同属性的词；点圈内的枢纽看共享这一属性的全部词。选一个大类后，环上会显示词名。</p>}
        {sideList.length > 0 && <div className="ring-list">{sideList.slice(0, 80).map(w =>
          <button type="button" className="v6-linkbtn" key={w.id} onClick={() => { if (sup && w.sup !== sup) setSup(''); setSel({ type: 'word', id: w.id }); }}>{w.zh}</button>)}
          {sideList.length > 80 && <span className="v8-note">等 {sideList.length} 个</span>}</div>}
        <h4>图例</h4>
        <div className="lg3-legend">{SUPERS.map(([k, label]) => <span key={k}><i className="lg3-dot" style={{ borderColor: SUPER_COLOR[k] }}/>{label}</span>)}</div>
        <p className="v8-note">点的边框是大类，填色是词表标注的颜色（空心表示没有颜色标签）。点之间的距离只为排版，没有含义。</p>
      </aside>
    </div>
  </section>;
}
