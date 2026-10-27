/* 图谱：先选「类型 → 类别 → 语言项」，再看该语言项在 LLM 内部对应的规则（每条规则来自一张 piece，带证据等级与来源）。
   两条路线分开显示：T = 模板对照（自顶向下），N = 原生神经元（自底向上，尚未接入）。等级只描述可读出证据。 */
import { useMemo, useState } from 'react';

import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { LANG_LABEL, LEVELS, itemLevel, levelMeta, matchLang, shortModel } from './v7Util.js';

export function LevelChip({ level, prefix, title, none }) {
  if (none) return <span className="v7-chip none" title={title}>{prefix && <em>{prefix}</em>}— 未接入</span>;
  const m = levelMeta(level);
  return (
    <span className="v7-chip" style={{ background: m.color, color: m.ink }} title={title || `等级 ${m.level}：${m.label}`}>
      {prefix && <em>{prefix}</em>}{m.sym} {m.label}
    </span>
  );
}

function TreeNav({ tree, model, lang, st, set }) {
  const [open, setOpen] = useState(() => Object.fromEntries(tree.types.map(t => [t.id, !t.collapsed])));
  const cats = tree.categories;
  const items = tree.items.filter(it => matchLang(it, lang));
  return (
    <nav className="v7-tree" aria-label="语言树">
      <button type="button" className={'v7-tree-all' + (!st.type ? ' on' : '')} onClick={() => set({ type: '' })}>全部语言项 <small>{items.length}</small></button>
      {tree.types.map(t => {
        const tItems = items.filter(it => it.type === t.id);
        const studied = tItems.filter(it => itemLevel(it, model) > 0).length;
        return (
          <div key={t.id} className="v7-tree-type">
            <div className={'v7-tree-row type' + (st.type === t.id ? ' on' : '')}>
              <button type="button" className="v7-caret" aria-expanded={open[t.id]} aria-label={`${open[t.id] ? '折叠' : '展开'}${t.name}`}
                      onClick={() => setOpen(o => ({ ...o, [t.id]: !o[t.id] }))}>{open[t.id] ? '▾' : '▸'}</button>
              <button type="button" className="v7-tree-name" onClick={() => set({ type: t.id })} title={t.desc}>
                {t.name}<small>{studied}/{tItems.length} 已有数据</small>
              </button>
            </div>
            {open[t.id] && cats.filter(c => c.type === t.id).map(c => {
              const cItems = tItems.filter(it => it.category === c.id);
              if (!cItems.length) return null;
              return (
                <div key={c.id} className="v7-tree-cat">
                  <div className={'v7-tree-row cat' + (st.type === c.id ? ' on' : '')}>
                    <button type="button" className="v7-tree-name" onClick={() => set({ type: c.id })}>{c.name}</button>
                  </div>
                  {cItems.map(it => {
                    const m = levelMeta(itemLevel(it, model));
                    return (
                      <button key={it.id} type="button" className={'v7-tree-item' + (st.item === it.id ? ' on' : '')}
                              onClick={() => set({ item: it.id })} title={`${it.name}：${m.label}`}>
                        <span className="sym" style={{ color: m.level ? m.color : 'var(--fw-text-3)' }} aria-hidden="true">{m.sym}</span>
                        <span className="nm">{it.name}</span>
                      </button>
                    );
                  })}
                </div>
              );
            })}
          </div>
        );
      })}
    </nav>
  );
}

function ItemCard({ it, model, on, onPick }) {
  const lv = itemLevel(it, model);
  return (
    <button type="button" className={'v7-card' + (on ? ' on' : '') + (lv ? '' : ' empty')} onClick={onPick}>
      <div className="v7-card-hd"><b>{it.name}</b><span className="mono">{it.id}</span></div>
      <p>{it.desc}</p>
      <div className="v7-card-routes">
        <LevelChip prefix="T" level={lv} title="路线 T（模板对照）在当前模型上的读出轴等级" />
        <LevelChip prefix="N" none title="路线 N（原生神经元经模板确认）尚未接入" />
      </div>
      <div className="v7-card-ft">
        {it.status.planned ? <span className="muted">还没有模式族</span>
          : <span className="muted">{it.status.pieces_T} 张拼图 · {it.status.models.length} 个模型</span>}
        {(it.status.languages_tested || []).map(l => <span key={l} className="v7-lang">{LANG_LABEL[l] || l}</span>)}
      </div>
    </button>
  );
}

function RuleRow({ r, piece, onResponse }) {
  return (
    <li className="v7-rule">
      <LevelChip level={r.level} />
      <div>
        <p>{r.text}</p>
        {piece && (
          <small className="muted">
            来源：<span className="mono">{piece.tm_id}</span> · 因子 <span className="mono">{piece.factor}</span>
            {' '}· 分片 {piece.source?.shards?.length || 0} · 分析缓存 <span className="mono" title={piece.source?.analysis_cache}>{(piece.source?.analysis_cache || '').slice(0, 10)}</span>
            {piece.gate_excluded_cells?.length > 0 && <> · 行为门排除 {piece.gate_excluded_cells.length} 格</>}
            {' '}<button type="button" className="v6-linkbtn" onClick={() => onResponse(piece)}>查看响应点云 →</button>
          </small>
        )}
      </div>
    </li>
  );
}

function ItemDetail({ itemId, model, set, onOpenSpace }) {
  const api = useApi(itemId ? `/api/lang/item/${itemId}` : null);
  const [allModels, setAllModels] = useState(false);
  if (!itemId) return <Empty>在左侧或中间选一个语言项，查看它在 LLM 内部对应的规则。</Empty>;
  if (api.error) return <Empty>该语言项的数据未接入：{api.error}</Empty>;
  const d = api.data;
  if (!d || d.pending) return <Empty>{d?.note || '读取中…'}</Empty>;
  const st = d.status;
  const pieces = Object.fromEntries((d.pieces || []).map(p => [p.id, p]));
  const rules = (d.rules || []).filter(r => allModels || !model || r.model === model);
  const models = Object.keys(st.by_model || {});
  const response = p => set({ page: 'space', view: 'response', tm: p.tm_id });
  return (
    <div className="v7-detail">
      <div className="v7-detail-hd">
        <h3>{d.item.name}</h3>
        <span className="mono">{d.item.id}</span>
      </div>
      <p className="v7-detail-desc">{d.item.desc}</p>
      {d.item.also_in?.length > 0 && <p className="v6-note">也属于：{d.item.also_in.join('、')}（交叉项只存一次）</p>}
      <div className="v7-detail-status">
        <div><small>路线 T · 模板对照</small><LevelChip level={st.read} /></div>
        <div><small>路线 N · 原生神经元</small><LevelChip none /></div>
        <div><small>控制轴（干预）</small><span className="v7-chip none">— 未检验</span></div>
      </div>
      {models.length > 0 && (
        <table className="v6-mini-table v7-model-table">
          <thead><tr><th>模型</th><th>读出轴</th><th>拼图</th><th>等级 3 的模式族</th></tr></thead>
          <tbody>
            {models.map(m => (
              <tr key={m} className={m === model ? 'cur' : ''}>
                <th title={m}>{shortModel(m)}{m === model && <em className="v6-tag">当前</em>}</th>
                <td><LevelChip level={st.by_model[m].read} /></td>
                <td className="num">{st.by_model[m].pieces}</td>
                <td>{(st.by_model[m].families || []).join('、') || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="v7-detail-actions">
        <button type="button" className="fw-tbtn primary" onClick={() => onOpenSpace(itemId)} disabled={!st.families.length}>在 3D 中查看</button>
        {st.planned && <button type="button" className="fw-tbtn" onClick={() => set({ page: 'progress' })}>未研究：去「进展 · 参与」领题</button>}
      </div>
      <h4 className="v7-h4">
        内部规则 <small>{rules.length} 条{model && !allModels ? ` · ${shortModel(model)}` : ' · 全部模型'}</small>
        {models.length > 1 && (
          <label className="v7-toggle"><input type="checkbox" checked={allModels} onChange={e => setAllModels(e.target.checked)} />显示全部模型</label>
        )}
      </h4>
      {rules.length ? (
        <ul className="v7-rules">{rules.map(r => <RuleRow key={r.piece} r={r} piece={pieces[r.piece]} onResponse={response} />)}</ul>
      ) : <Empty>{st.planned ? '这个语言项还没有模式族，暂无内部规则。' : '当前模型上还没有这个语言项的拼图。'}</Empty>}
      <details className="v7-criteria">
        <summary>等级判据（固定在 pieces.py，随结果返回）</summary>
        <ul>{LEVELS.map(l => <li key={l.level}><LevelChip level={l.level} /> {d.level_labels?.[l.level]}</li>)}</ul>
        <p className="v6-note">{d.criteria?.readable_rule}</p>
        <p className="v6-note">{d.criteria?.baseline_rule}</p>
        <p className="v6-note">等级只描述可读出证据：不表示该方向被模型使用，也不是计算单元；控制轴需要预注册干预。</p>
      </details>
    </div>
  );
}

export default function AtlasPage({ tree, model, lang, st, set, onOpenSpace }) {
  const scope = st.type;
  const items = useMemo(() => tree.items.filter(it => matchLang(it, lang)
    && (!scope || it.type === scope || it.category === scope)), [tree, lang, scope]);
  const scopeName = tree.types.find(t => t.id === scope)?.name || tree.categories.find(c => c.id === scope)?.name || '全部语言项';
  const counts = LEVELS.map(l => items.filter(it => itemLevel(it, model) === l.level).length);
  return (
    <div className="v7-atlas">
      <TreeNav tree={tree} model={model} lang={lang} st={st} set={set} />
      <section className="v7-center" aria-label="语言项">
        <div className="v7-center-hd">
          <h2>{scopeName}</h2>
          <div className="v7-counts" aria-label="当前模型上的等级分布">
            {LEVELS.map((l, i) => <span key={l.level} title={l.label}><LevelChip level={l.level} /> {counts[i]}</span>)}
          </div>
        </div>
        <p className="v6-note">等级为路线 T 在 {shortModel(model) || '—'} 上的读出轴证据；各模型分开计数，不合成百分比。{tree.policy}</p>
        {items.length ? (
          <div className="v7-cards">
            {items.map(it => <ItemCard key={it.id} it={it} model={model} on={st.item === it.id} onPick={() => set({ item: it.id })} />)}
          </div>
        ) : <Empty>这个范围里没有符合语种筛选的语言项。</Empty>}
        {tree.unmapped?.length > 0 && <p className="v6-note">另有 {tree.unmapped.length} 张拼图的因子还没有映射到语言项（不计入覆盖）。</p>}
      </section>
      <aside className="v7-side" aria-label="语言项详情">
        <ItemDetail key={st.item} itemId={st.item} model={model} set={set} onOpenSpace={onOpenSpace} />
      </aside>
    </div>
  );
}
