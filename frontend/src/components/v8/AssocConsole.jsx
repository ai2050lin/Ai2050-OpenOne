/* 关联机制 · 控制台（Phase 335）：在 3D 空间下面。
   - 输入一段文字 → 本机后端做一次真实前向（/api/live/run，保存全部 37 层隐态），返回下一个词的预测；
     再按层取回最后一个位置的隐态（/api/traces/{id}/field），算每层的改写幅度 1 − cos(h_l, h_{l−1})，叠加到 3D。
   - 模式：信息流、逐层读出、注意力头、因素、对照、方法比较。后四项里标“案例”的数据来自 MA-H2 汇总。
   没有本机后端时诚实显示不可用，不做模拟。实时观察是未登记观察，不产生研究结论。 */
import { useEffect, useState } from 'react';

import { API_BASE } from '../app/rdc_fusion/api.js';
import { AlgorithmPanel } from './AssocDeep.jsx';
import { LineChart } from './H2Results.jsx';
import { request } from './MechanismTraceExplorer.jsx';
import { rewriteCurve } from './data/assocFlow.js';

const MODES = [['flow', '信息流'], ['lens', '逐层读出'], ['heads', '注意力头'], ['factor', '因素'], ['compare', '对照'], ['method', '方法比较']];
const EXAMPLE_A = '资料：香蕉是蓝色的。\n问：按这份资料，香蕉通常是什么颜色的？';
const EXAMPLE_B = '资料：香蕉是蓝色的。\n问：在现实世界里，香蕉通常是什么颜色的？';

/* 找一个能做真实前向的后端：同源中心或本机节点 */
function useLiveBase() {
  const [state, setState] = useState({ base: null, status: null, checked: false });
  useEffect(() => {
    let dead = false;
    (async () => {
      for (const b of [...new Set([API_BASE, 'http://127.0.0.1:5001', 'http://127.0.0.1:5002'])]) {
        try { const s = await request(b, '/api/live/status'); if (!dead) setState({ base: b, status: s, checked: true }); return; } catch { /* 试下一个 */ }
      }
      if (!dead) setState({ base: null, status: null, checked: true });
    })();
    return () => { dead = true; };
  }, []);
  return state;
}

async function runOnce(base, text) {
  const d = await request(base, '/api/live/run', { method: 'POST', body: JSON.stringify({ text, max_tokens: 1024 }) });
  const pos = d.readout_positions.at(-1);
  const hidden = [];
  for (let l = 0; l <= d.n_layers; l++) {
    const f = await request(base, `/api/traces/${d.trace_id}/field?block=${l}&position=${pos}&boundary=${l ? 'hidden_state' : 'embedding'}`);
    hidden.push(f.values);
  }
  return { id: d.trace_id, tokens: d.tokens, top: d.output_top || [], curve: rewriteCurve(hidden), elapsed: d.elapsed_ms };
}

function TopTokens({ top }) {
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
      </div>
    </div>
    {children}
  </section>;
}
