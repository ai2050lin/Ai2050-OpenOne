import { useState } from 'react';
import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';

const tabs = ['真实续写', '阻断与恢复', '反例与来源'];
const scalar = (x, places = 3) => Number.isFinite(x) ? x.toFixed(places) : '—';
const percent = x => `${scalar(100 * x, 1)}%`;
const caseNames = {
  '0-fact': '短段随原事实给码', '0-state': '短段随实发词给码',
  '1-fact': '长段随原事实给码', '1-state': '长段随实发词给码',
  fact_to_state: '长段改为跟随实发词', state_to_fact: '长段改为跟随原事实',
  address_and_value_give_different_code: '寻址阻断与写入阻断给出了不同答码',
  h18_restore_lowers_original_margin: '恢复 H18 后，原选码的优势反而下降',
  joint_changes_actual_code: '联合阻断改变了实际答码',
  key_only_recovers: '只恢复原生 Key 即恢复答码的个例',
  value_only_recovers: '只恢复原生 Value 即恢复答码的个例',
};
const arms = {
  sham: '原样运行', address: '阻断 State 寻址', value: '阻断 State 写入', key: '移除 Key 寻址方向',
  joint: '三路联合阻断', 'restore-h18': '恢复 H18', 'restore-parallel': '恢复两个并行头', 'restore-all': '原生全部恢复',
};
const doseNames = { address: '寻址', value: '写入', fact: '事实', book: '码本' };

function Cases({ cases }) {
  return cases.map(item => <details key={`${item.tag}:${item.id}`}>
    <summary>{caseNames[item.tag] || item.tag} · {item.id}</summary>
    <p>原事实 {item.fact}（{item.factual_code}） · 实发 State {item.state}（{item.emitted_state_code}）</p>
    {item.output && <pre className="v8-artifact-json">{item.output}</pre>}
    {item.paired_output && <pre className="v8-artifact-json">{item.paired_output}</pre>}
    {item.arms && <table className="v6-table"><thead><tr><th>干预</th><th>实答</th><th>P / Q logits</th></tr></thead>
      <tbody>{Object.entries(item.arms).map(([key, arm]) => <tr key={key}><td>{arms[key] || key}</td><td>{arm.code}</td>
        <td>{arm.logits.map(x => scalar(x, 2)).join(' / ')}</td></tr>)}</tbody></table>}
    <details><summary>原始提示</summary><pre className="v8-artifact-json">{item.input}</pre></details>
  </details>);
}

export default function KvLengthReport({ doc }) {
  const [tab, setTab] = useState(0);
  const evidence = useApi(tab === 2 ? `/api/research-library/artifacts/${doc.case_sha256}` : null);
  const { result: r, tests } = doc;
  const distance = tests.find(test => test.id === 'distance-preference');
  return <section aria-label="Phase 2952 实发 State 证据">
    <p><strong>同一提示和实发 State 下，延长指定中性段后，随实发词答码的优势下降 {scalar(-distance.mean)} logit。</strong></p>
    <p className="v8-note">Phase 2952 · Qwen3-14B 原始 BF16 · {r.complete} 个完整留出家族 · {r.intervention_families} 个干预家族</p>
    <div className="v6-toolbar" role="group" aria-label="Phase 2952 查看维度">{tabs.map((name, i) =>
      <button key={name} type="button" className="fw-tbtn" aria-pressed={tab === i} onClick={() => setTab(i)}>{name}</button>)}</div>
    {tab === 0 && <>
      <p>提示事实 → 模型实际生成 State → L25H18 读取 → 模型实际给出 P 或 Q。</p>
      <table className="v6-table"><thead><tr><th>中性段</th><th>State 到答码</th><th>跟随实发词</th><th>跟随原事实</th></tr></thead>
        <tbody>{['0', '1'].map(key => <tr key={key}><td>{key === '0' ? '短' : '长'}</td><td>{r.gap[key][0]} token</td>
          <td>{r.counts[key].state}</td><td>{r.counts[key].fact}</td></tr>)}</tbody></table>
      <p>统一张量形状后，{r.same_k} 对 K 和 {r.same_v} 对 V 逐位相同。L25H18 对 State 的注意力中位数从 {percent(r.attention.short)} 降到 {percent(r.attention.long)}。</p>
      <p className="v8-note">全词表首选与原贪心核验一致 {r.shape_top_matches}/{r.shape_top_total}。主距离终点预注册方向为增加，本轮实测为下降。</p>
    </>}
    {tab === 1 && <>
      <p>寻址阻断在答码位置移除通向 State 的读取。写入阻断保留注意力权重，只清零读取的 Value。</p>
      <p>三路联合阻断改变 {r.actual_joint_flips} 条实际答码。全部原生恢复后仍有 {r.all_restore_flips} 条改码。</p>
      <table className="v6-table"><thead><tr><th>留出检验</th><th>效果（logit）</th><th>99.375% 区间</th><th>冻结判定</th></tr></thead>
        <tbody>{tests.map(test => <tr key={test.id}><td>{test.label}</td><td>{scalar(test.mean)}</td>
          <td>[{test.ci.map(x => scalar(x)).join(', ')}]</td><td>{test.passed ? '通过' : '未过门'}</td></tr>)}</tbody></table>
      <details><summary>等幅随机对照的实际剂量</summary>
        {Object.entries(doc.dose).map(([key, dose]) => {
          const [path, pair] = key.split('-');
          return <p key={key}>{doseNames[path]}对照 {pair} · 目标 {scalar(dose.target[2])} · 对照 {scalar(dose.control[2])} · 偏差 p95 {percent(dose.p95)}</p>;
        })}
      </details>
    </>}
    {tab === 2 && <>
      {doc.limitations.map(line => <p className="v8-note" key={line}>{line}</p>)}
      <Cases cases={evidence.data?.cases || doc.highlights} />
      {evidence.error && <Empty>完整反例对象暂未读到。以上为报告内精选反例。</Empty>}
      <details><summary>冻结与原始文件 SHA-256</summary><pre className="v8-artifact-json">{JSON.stringify(doc.source_file_sha256, null, 2)}</pre></details>
    </>}
  </section>;
}
