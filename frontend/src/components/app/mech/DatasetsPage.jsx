/* 数据集总览：三类材料各自的行为门与正确率，点题目或句子进入「例句」/「语言项」。
   ① 机制材料（冲突数据集）：每条事实的行为门（无上下文时模型知道该事实）与划分——来自机制案例报告；
   ② 行为广度（C-Eval / CMMLU 等考试题库）：各大类与科目的正确率——只做行为广度，不作机制材料（选项字母会污染读出）；
   ③ 最小对模板（语言项）：各大类语言项在所选模型上的读出等级——来自语言谱图（lang_atlas.v1）。
   全部只读本机数据包；没有数据时显示「未接入」。 */
import { useState } from 'react';

import { useApi } from '../rdc_fusion/api.js';
import { Empty } from '../rdc_fusion/ui.jsx';
import PanelBoundary from '../PanelBoundary.jsx';
import { itemLevel, levelMeta, shortModel } from '../../v8/v8Util.js';
import BenchCard from './BenchCard.jsx';
import { useMechCase, useMechIndex } from './mechData.js';
import { fmt, isConflict, schemaLabel } from './mechUtil.js';

function ConflictSet({ summary, onFact }) {
  const { report, error, loading } = useMechCase(summary.id);
  const [all, setAll] = useState(false);
  if (error) return <Empty>案例报告不在本机数据包里：{error}</Empty>;
  if (loading || !report) return <Empty>读取…</Empty>;
  const gate = report.gate || [];
  const shown = all ? gate : gate.slice(0, 12);
  return (
    <>
      <div className="wb9-tiles">
        <div><b>{report.n_gated ?? '—'} / {report.n_facts ?? '—'}</b><span>过行为门</span><small>无上下文时模型给出正确答案（间隔 &gt; 0）</small></div>
        <div><b>{report.n_discovery ?? '—'} / {report.n_heldout ?? '—'}</b><span>发现 / 留出</span><small>按材料 id 与主语哈希确定性切分</small></div>
        <div><b>{Object.keys(report.priority_table || {}).length}</b><span>线索条件</span><small>同一事实在不同线索下各成一句</small></div>
      </div>
      <table className="wb9-table">
        <thead><tr><th>主语</th><th>记忆答案</th><th>反事实（上下文）答案</th><th>划分</th><th>行为门间隔</th><th>行为门</th><th /></tr></thead>
        <tbody>{shown.map(g => (
          <tr key={g.subject} className={g.gated ? '' : 'muted'}>
            <td>{g.subject}</td><td>{g.answer}</td><td>{g.counterfactual}</td><td>{g.split === 'heldout' ? '留出' : '发现'}</td>
            <td>{fmt(g.gate_margin)}</td><td className={g.gated ? 'good' : 'bad'}>{g.gated ? '✓ 过门' : '✗ 未过门'}</td>
            <td>{g.gated && <button type="button" className="fw-tbtn ghost" onClick={() => onFact(summary.id, g.subject)}>例句 →</button>}</td>
          </tr>
        ))}</tbody>
      </table>
      {gate.length > 12 && <button type="button" className="fw-tbtn ghost" onClick={() => setAll(v => !v)}>{all ? '收起' : `显示全部 ${gate.length} 条`}</button>}
    </>
  );
}

function LangItems({ onItem }) {
  const treeApi = useApi('/api/lang/tree');
  const tree = treeApi.data && !treeApi.data.pending ? treeApi.data : null;
  const [model, setModel] = useState('');
  if (treeApi.error) return <Empty>未接入：语言谱图不在本机数据包里（{treeApi.error}）。</Empty>;
  if (!tree) return <Empty>{treeApi.data?.pending ? '语言谱图还在后台生成。' : '读取语言谱图…'}</Empty>;
  const models = tree.models || [];
  const m = models.includes(model) ? model : models[0] || '';
  return (
    <>
      <div className="wb9-toolbar">
        <label>模型 <select value={m} onChange={e => setModel(e.target.value)}>{models.map(x => <option key={x} value={x}>{shortModel(x)}</option>)}</select></label>
        <span className="wb9-legend">{[0, 1, 2, 3, 4].map(l => <span key={l}><i style={{ background: levelMeta(l).color }} />{levelMeta(l).sym} {levelMeta(l).label}</span>)}</span>
      </div>
      <div className="wb9-cats">
        {(tree.types || []).map(t => {
          const items = (tree.items || []).filter(it => it.type === t.id);
          if (!items.length) return null;
          const has = items.filter(it => itemLevel(it, m) >= 1).length;
          return (
            <div key={t.id} className="wb9-cat">
              <h4>{t.name}<small>{has} / {items.length} 条有数据</small></h4>
              <div className="wb9-items">
                {items.map(it => {
                  const lv = levelMeta(itemLevel(it, m));
                  return (
                    <button key={it.id} type="button" onClick={() => onItem(it.id, m)} title={`${it.name} · ${lv.label}`}
                            style={{ background: lv.color, color: lv.ink }}>{lv.sym} {it.name}</button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
      <p className="v6-note">等级只描述可读出证据（1 有数据 → 4 跨模型 / 跨族），由固定判据判定，不表示控制或解释。点语言项看行为门逐格结果、对照例句与编码层段。</p>
    </>
  );
}

export default function DatasetsPage({ onFact, onItem }) {
  const index = useMechIndex();
  const conflicts = index.cases.filter(c => isConflict(c) && !c.error);
  const sec = (title, sub, body) => (
    <section key={title} className="wb9-sec" aria-label={title}>
      <header><h3>{title}</h3><small>{sub}</small></header>
      <PanelBoundary name={title}>{body}</PanelBoundary>
    </section>
  );
  return (
    <div className="wb9-page">
      {conflicts.length
        ? conflicts.map(c => sec(`机制材料 · ${schemaLabel(c.schema)}（${c.set}）`, `${c.id}：每条事实一行，点「例句」看这一句`,
          <ConflictSet key={c.id} summary={c} onFact={onFact} />))
        : sec('机制材料 · 冲突数据集', '上下文与记忆冲突等', index.loading ? <Empty>读取…</Empty>
          : <Empty>未接入：还没有机制案例报告{index.error ? `（${index.error}）` : ''}。冲突材料的行为门随报告一起进数据包。</Empty>)}
      {sec('行为广度 · 考试题库', '各大类与科目的正确率；只看行为广度，不作机制材料', <BenchCard />)}
      {sec('最小对模板 · 语言项', '各大类语言项在所选模型上的读出等级', <LangItems onItem={onItem} />)}
    </div>
  );
}
