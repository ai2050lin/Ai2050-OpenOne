import { useState } from 'react';
import map from '../../../../shared/dist_protocol/lang_map/lang_map.json';
import { useWorkspaceResource } from './nativeObservationArchive.js';

const titleForDimension = '横切条件';
const normalize = text => text.normalize('NFKC').toLowerCase().trim();

function matches(item, path, words, linkState) {
  const plan = item.plan;
  const linked = Boolean(plan?.related?.length);
  if ((linkState === 'linked' && !linked) || (linkState === 'pending' && linked)) return false;
  const parts = [path, item.id, item.name, item.example];
  for (const pair of [plan, plan?.followup].filter(Boolean)) {
    parts.push(pair.factor, ...pair.variants.map(v => pair.prompt.replace('{x}', v.value) + ' ' + v.expected));
  }
  for (const ref of plan?.related || []) parts.push(ref.id, ref.reason);
  const haystack = normalize(parts.filter(Boolean).join(' '));
  return words.every(word => haystack.includes(word));
}

function QuestionPair({ pair }) {
  return <ol className="tr-question-pair" type="A">
    {pair.variants.map(({ value, expected }) => <li key={value}>
      <p>{pair.prompt.replace('{x}', value)}</p><p><span>参考答案</span> {expected}</p>
    </li>)}
  </ol>;
}

export default function ResearchDirections({ onRecord, onFamily, admin }) {
  const {data: questions, error} = useWorkspaceResource('direction-questions');
  const directions = map.domains.map(domain => ({ ...domain,
    subs: domain.subs.map(group => ({ ...group, items: group.items.map(item => ({
      ...item, plan: questions?.items?.[item.id],
    })) })),
  }));
  const dimensions = map.dims.map(item => ({ ...item, plan: questions?.dimensions?.[item.id] }));
  const [query, setQuery] = useState('');
  const [linkState, setLinkState] = useState('all');
  const [expanded, setExpanded] = useState(new Set());
  const [selectedId, setSelectedId] = useState(null);
  const words = normalize(query).split(/\s+/).filter(Boolean);
  const filtering = words.length > 0 || linkState !== 'all';
  const visible = new Map();
  function filterItems(items, path) {
    return items.filter(item => {
      if (!matches(item, path, words, linkState)) return false;
      visible.set(item.id, { item, path });
      return true;
    });
  }
  const roots = directions.map(domain => {
    const subs = domain.subs.map(group => ({ ...group,
      items: filterItems(group.items, domain.name + ' / ' + group.name),
    })).filter(group => group.items.length);
    return { ...domain, subs, count: subs.reduce((n, group) => n + group.items.length, 0) };
  }).filter(domain => domain.count);
  const mainCount = visible.size;
  const dimItems = filterItems(dimensions, titleForDimension);
  const selected = visible.get(selectedId) || (visible.size === 1 ? visible.values().next().value : null);
  function toggle(key) {
    setExpanded(current => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  }
  function branch(id, name, count, children) {
    // Searching opens matching paths; disclosure buttons retain ordinary keyboard behavior.
    const key = filtering ? JSON.stringify([query, linkState, id]) : id;
    const open = filtering ? !expanded.has(key) : expanded.has(key);
    return <li key={id}>
      <button type="button" className="tr-branch" aria-expanded={open} onClick={() => toggle(key)}>
        <span aria-hidden="true">{open ? '▾' : '▸'}</span> {name} <small>{count}</small>
      </button>
      {open && <ul>{children()}</ul>}
    </li>;
  }
  function leaves(items) {
    return items.map(item => <li key={item.id}>
      <button type="button" className="tr-leaf" aria-pressed={selected?.item.id === item.id}
        onClick={() => setSelectedId(item.id)}>{item.name}</button>
    </li>);
  }
  const plan = selected?.item.plan;
  return <section aria-label="研究方向与设计题">
    {error && <p className="v8-note">中心设计题暂不可用 · {error}</p>}
    <div className="tr-controls">
      <label>查找方向或题目<input type="search" value={query} maxLength={250} placeholder="例如 多步、青→白、主谓"
        onChange={event => setQuery(event.target.value)}/></label>
      <label>精选记录关联<select value={linkState} onChange={event => setLinkState(event.target.value)}>
        <option value="all">全部</option><option value="linked">已关联历史</option><option value="pending">待关联历史</option>
      </select></label>
    </div>
    <p className="tr-tree-count" role="status">{mainCount} 个研究项 · {dimItems.length} 个横切条件</p>
    <div className="tr-layout tr-directions">
      <nav aria-label="研究方向目录"><ul>{roots.map(domain =>
        branch(domain.id, domain.name, domain.count, () => domain.subs.map(group =>
          branch(group.id, group.name, group.items.length, () => leaves(group.items))
        ))
      )}{dimItems.length > 0 && branch('dimensions', titleForDimension, dimItems.length, () => leaves(dimItems))}</ul></nav>
      {selected ? <article className="tr-detail tr-question-detail" aria-label="方向与题目">
        <small>{selected.path} · {selected.item.id}</small><h4>{selected.item.name}</h4>
        {plan ? <><p>公开设计题 · 对照因素：{plan.factor}</p><QuestionPair pair={plan}/>
          {plan.followup && <details><summary>进一步区分两种解法</summary>
            <p>{plan.followup.factor}</p><QuestionPair pair={plan.followup}/><p>{plan.followup.interpretation}</p>
          </details>}</> : <p>该方向待配题。</p>}
        {plan?.related?.length ? <details><summary>相关历史记录（{plan.related.length}）</summary>
          {plan.related.map(ref => <div key={ref.id}><p>{ref.reason}</p><button type="button"
            onClick={() => onRecord(ref.id)}>打开 Phase {ref.id.replace('gf-', '')}</button></div>)}
        </details> : <p className="tr-tree-note">精选历史记录待关联。</p>}
        {plan?.centerFamily && admin && <button type="button" onClick={() => onFamily(plan.centerFamily)}>查询中心同族运行</button>}
      </article> : <p className="tr-tree-note">{visible.size
        ? '展开方向，选择研究项即可查看题目、参考答案与关联历史。'
        : '没有匹配项。可以更换关键词或放宽历史关联条件。'}</p>}
    </div>
  </section>;
}
