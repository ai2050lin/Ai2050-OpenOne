import { useMainline } from './data/mainlineState.js';
/* 关联机制 · 控制台（Phase 335）：在 3D 空间下面。
   - 输入一段文字 → 本机后端做一次真实前向（/api/live/run，保存全部 37 层隐态），返回下一个词的预测；
     再按层取回最后一个位置的隐态（/api/traces/{id}/field），算每层的改写幅度 1 − cos(h_l, h_{l−1})，叠加到 3D。
   - 模式：信息流、逐层读出、注意力头、因素、对照、方法比较。后四项里标“案例”的数据来自 MA-H2 汇总。
   没有本机后端时诚实显示不可用，不做模拟。实时观察是未登记观察，不产生研究结论。 */
import { useState } from 'react';

import { AlgorithmPanel } from './AssocDeep.jsx';
import { LineChart } from './H2Results.jsx';
import { API_BASE } from '../app/rdc_fusion/api.js';
import { runOnce, useLiveBase } from './liveRun.js';


const MODES = [['flow', '信息流'], ['lens', '逐层读出'], ['heads', '注意力头'], ['factor', '因素'], ['method', '方法比较'], ['tests', '三类检验']];
const EXAMPLE_A = '资料：香蕉是蓝色的。\n问：按这份资料，香蕉通常是什么颜色的？';
const EXAMPLE_B = '资料：香蕉是蓝色的。\n问：在现实世界里，香蕉通常是什么颜色的？';

/* 三类检验（Phase 340）：每类与同信息预算的简单基线并列；数据来自冻结运行，不来自实时生成。未测就写未测。 */
export function TestsPanel() {
  const { CLAIMS, TESTS, METHOD_RESULTS } = useMainline();
  return <>
    <p className="ac-k">三类检验 · 主张 {CLAIMS[0].id}</p>
    <div className="v8-weight-scroll"><table className="v6-table">
      <thead><tr><th>检验</th><th>项目</th><th>结果</th></tr></thead>
      <tbody>{TESTS.flatMap(test => test.rows.map(([name, value], i) => <tr key={`${test.id}:${name}`}>
        {i === 0 && <td rowSpan={test.rows.length} title={test.desc}><b>{test.name}</b><br /><span className="v8-note">{test.stage}</span></td>}
        <td>{name}</td><td>{value == null ? <span className="mp-tag">未测</span> : value}</td></tr>))}</tbody>
    </table></div>
    <p className="v8-note">响应预测：只用允许的前缀和冻结规则预测未见材料的内部响应。输出预测：预测全词表分布、生成与具体错误。原生机制检验：交换对应因素的自然状态，查变化与不变项。结果只读冻结运行，左边的实时生成不计入。</p>
    <details className="mp-fold"><summary>已完成的范围内方法检验（与独立预测分开）</summary><div className="mp-fold-body">
      {METHOD_RESULTS.map(result => <p className="v8-note" key={result.run_id}><b>{result.run_id} · {result.label}：{result.value.toFixed(4)} {result.unit}</b><br />{result.scope}<br />
        <a href={`${API_BASE}/api/research-library/artifacts/${result.artifact_sha256}`} target="_blank" rel="noreferrer">回查原始报告</a> · <a href={`?lens=spatial&view=research&run=${result.run_id}`}>在 3D 空间定位组成员</a></p>)}
    </div></details>
  </>;
}

export function TopTokens({ top }) {
  return <ol className="ac-top">{top.slice(0, 8).map(t => <li key={t.token_id}><code>{JSON.stringify(t.text).slice(1, -1) || '␣'}</code>
    <i><b style={{ width: `${Math.round(t.probability * 100)}%` }} /></i><span>{(t.probability * 100).toFixed(1)}%</span></li>)}</ol>;
}

export default function AssocConsole({ mode, onMode, onLive, panels, children }) {
  const live = useLiveBase();
  const [textA, setTextA] = useState(EXAMPLE_A), [textB, setTextB] = useState(EXAMPLE_B);
  const [res, setRes] = useState({ a: null, b: null }), [busy, setBusy] = useState(false), [msg, setMsg] = useState('');
  const ready = Boolean(live.base) && live.status?.available !== false;
  const where = live.base ? (live.base === API_BASE ? '本中心' : live.base.replace(/^https?:\/\//, '')) : '';
  const status = !live.checked ? '正在查找本机模型…' : !ready ? '没有连接到可用的本机模型：生成需要桌面节点或本机后端。右侧的案例数据仍可查看。'
    : `本机后端 ${where}${live.status?.model_name ? ` · ${live.status.model_name}` : ''} · ${live.status?.status === 'ready' ? '模型已加载' : live.status?.status === 'loading' ? '模型加载中' : '模型未加载，生成时自动加载（占用 GPU）'}`;

  async function generate() {
    setBusy(true); setMsg('');
    try {
      const a = await runOnce(live.base, textA);
      const b = mode === 'compare' ? await runOnce(live.base, textB) : null;
      setRes({ a, b }); onLive?.(a);
    } catch (e) { setMsg(e.message); } finally { setBusy(false); }
  }

  const curves = [res.a && { name: mode === 'compare' ? '句子 A' : '本次输入', color: '#d29a00', values: res.a.curve },
    mode === 'compare' && res.b && { name: '句子 B', color: '#7050a0', values: res.b.curve }].filter(Boolean);

  return <section className="v8-card ac" aria-label="控制台">
    <header className="ac-head"><h3>控制台</h3>
      <div className="ac-modes" role="tablist" aria-label="观察与分析模式">{MODES.map(([id, label]) =>
        <button type="button" role="tab" key={id} aria-selected={mode === id} aria-pressed={mode === id} onClick={() => onMode(id)}>{label}</button>)}</div>
    </header>
    <div className="ac-body">
      <div className="ac-input">
        <label className="v8-note" htmlFor="ac-a">{mode === 'compare' ? '句子 A' : '输入一段文字'}</label>
        <textarea id="ac-a" rows={4} value={textA} onChange={e => setTextA(e.target.value)} />
        {mode === 'compare' && <><label className="v8-note" htmlFor="ac-b">句子 B（只改一处）</label>
          <textarea id="ac-b" rows={4} value={textB} onChange={e => setTextB(e.target.value)} /></>}
        <div className="ac-row">
          <button type="button" className="fw-tbtn primary" disabled={!ready || busy || !textA.trim() || (mode === 'compare' && !textB.trim())} onClick={generate}>
            {busy ? '生成中…' : mode === 'compare' ? '分别生成并对照' : '生成'}</button>
          <span className={`ac-dot ${ready ? 'ok' : 'off'}`} aria-hidden="true" /><span className="v8-note">{status}</span>
        </div>
        {msg && <p className="ac-err" role="alert">{msg}</p>}
        <p className="v8-note">实时生成是一次前向的未登记观察，不产生研究结论；轨迹按原生格式保存在本机。</p>
      </div>
      <div className="ac-result" aria-live="polite">
        {mode === 'flow' && (res.a ? <>
          <p className="ac-k">下一个词的预测 <span className="v8-note">· {res.a.tokens.length} 个 token · {res.a.elapsed} ms</span></p>
          <TopTokens top={res.a.top} />
          <LineChart label="每层改写幅度" yLabel="每层改写幅度 1 − cos(h_l, h_l−1)，最后一个位置；3D 中的金色圆点与此相同" series={curves} height={170} />
        </> : <div className="ac-empty"><p className="ac-k">信息流</p>
          <p className="v8-note">3D 现在显示的是案例 MA-H2 的汇总（2,520 条）：头的颜色是对资料颜色的注意力，彩线是五个因素。</p>
          <p className="v8-note">在左边输入一段文字并生成后，这里显示模型预测的下一个词，3D 主干上叠加本次输入每一层的改写幅度（金色圆点，越大改写越多）。</p></div>)}
        {mode === 'compare' && (res.a && res.b ? <>
          <div className="ac-two"><div><p className="ac-k">句子 A 的预测</p><TopTokens top={res.a.top} /></div><div><p className="ac-k">句子 B 的预测</p><TopTokens top={res.b.top} /></div></div>
          <LineChart label="两句的每层改写幅度" yLabel="每层改写幅度 1 − cos(h_l, h_l−1)" series={curves} height={170} />
        </> : <div className="ac-empty"><p className="ac-k">对照</p><p className="v8-note">两句只差一处（例如“按这份资料”和“在现实世界里”），分别生成后比较预测的下一个词和每层改写幅度。看到差别只是观察；要说明原因需要正式运行的修补实验。</p></div>)}
        {['lens', 'heads', 'factor'].includes(mode) && <>{panels[mode]}</>}
        {mode === 'method' && <AlgorithmPanel />}
        {mode === 'tests' && <TestsPanel />}
      </div>
    </div>
    {children}
  </section>;
}
