import { useState } from 'react';
import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';

const LABELS = { discovery: '发现集', confirm: '新句式确认', lockbox: '未见答案词盲测' };
const n3 = n => Number.isFinite(n) ? n.toFixed(3) : '—';

export default function AttributePrefixTransferReport({ doc }) {
  const [id, setId] = useState(doc.counterexamples[0] || '');
  const [casesOpen, setCasesOpen] = useState(false);
  const [split, position] = id.split(':');
  const index = Number(position);
  const cases = doc.cases[split];
  const sha = casesOpen ? cases?.chunks[Math.floor(index / cases.chunk_size)] : null;
  const source = useApi(sha ? `/api/research-library/artifacts/${sha}` : null);
  const row = source.data?.cases?.find(item => item.id === id);
  return <section aria-label="属性绑定留出结果">
    <h3>属性绑定与问题前写入</h3>
    <p>{doc.conclusion}</p>
    <div style={{ overflowX: 'auto' }}><table className="v6-table">
      <thead><tr><th>划分</th><th>原始首词正确率</th><th>预测 RMSE（基线 → KV）</th><th>备选首词（原始 → 编辑）</th><th>回放 / 恢复</th></tr></thead>
      <tbody>{Object.entries(doc.summary).map(([key, s]) => <tr key={key}>
        <td>{LABELS[key]} · {s.n} 条</td><td>{(s.clean_accuracy * 100).toFixed(1)}%</td>
        <td>{n3(s.baseline_rmse)} → {n3(s.prediction_rmse)} {key !== 'discovery' && `· ${s.prediction_gate ? '通过' : '未通过'}`}</td>
        <td>{s.alternates ? `${s.alternates.clean} → ${s.alternates.targeted} · ${s.causal_gate ? '通过' : '未通过'}` : '—'}</td>
        <td>{s.replay_exact == null ? '—' : `${s.replay_exact} / ${s.rescue_exact}（总数 ${s.n}）`}</td>
      </tr>)}</tbody>
    </table></div>
    {doc.capability && <details><summary>任务外能力保持复测</summary>
      {Object.entries(doc.capability.summary).map(([key, value]) => <p key={key}>
        {key === 'targeted' ? '定向写入' : '等范数对照'} · {value.n} 题 · {value.status === 'noninferior' ? '通过非劣效门' : '能力保持未证实'} · 净损失区间 {value.loss_interval.map(v => (v*100).toFixed(2)).join(' 至 ')} 个百分点
      </p>)}
      <p className="v8-note">使用既有选项题与固定写入偏移。完整逐题产物可按 SHA 查询。</p>
      <code>{doc.capability.artifact_sha256}</code>
    </details>}
    <details onToggle={e => setCasesOpen(e.currentTarget.open)}><summary>逐题记录与反例（{doc.counterexamples.length} 条反例索引）</summary>
      <label>样本 <select aria-label="属性绑定样本" value={id} onChange={e => setId(e.target.value)}>
        {Object.entries(doc.cases).map(([key, group]) => <optgroup key={key} label={LABELS[key]}>
          {Array.from({ length: group.n }, (_, i) => `${key}:${String(i).padStart(3, '0')}`).map(value =>
            <option key={value} value={value}>{value}{doc.counterexamples.includes(value) ? ' · 反例' : ''}</option>)}
        </optgroup>)}
      </select></label>
      {source.error ? <Empty>{source.error}</Empty> : row ? <>
        <p>签署答案 {row.expected}，备选 {row.contrast}。实际间隔 {n3(row.prediction.actual_gap)}，预测 {n3(row.prediction.predicted_gap)}。</p>
        <ul>{Object.entries(row.conditions).map(([key, value]) => <li key={key}>{key} · {value.text || '无输出'} · 间隔 {n3(value.gap)}</li>)}</ul>
        <details><summary>原始 token、锚点与缓存改动</summary><pre className="v8-artifact-json">{JSON.stringify(row, null, 2)}</pre></details>
      </> : <Empty>读取样本…</Empty>}
    </details>
    <details><summary>实验边界与原生数据索引</summary>
      <ul>{doc.limitations.map(text => <li key={text}>{text}</li>)}</ul>
      <p>原生归档保存 B35 问题前末位 V 的完整 BF16 数值。分片可在「全部产物版本」按 SHA 查询。</p>
      <ul>{Object.entries(doc.native).map(([key, value]) => <li key={key}>{LABELS[key]} · <code>{value}</code></li>)}</ul>
    </details>
  </section>;
}
