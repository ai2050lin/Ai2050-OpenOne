import { useRef, useState } from 'react';
import { API_CONFIG } from '../../config/api';
const base = `${API_CONFIG.main.replace(/\/$/, '')}/api/research-assets`;

const labels = { actual_Wgate_jk: '真实 W_gate[j,k]', actual_Wup_jk: '真实 W_up[j,k]', actual_Wdown_kj: '真实 W_down[k,j]', embedding_coordinate: '词嵌入 E[token,k]', hidden_coordinate: 'HiddenState H[checkpoint,token,k]', normalized_mlp_input: '归一化 MLP 输入 x[k]', gate_unit: '实际 gate[j]', up_unit: '实际 up[j]', actual_product_unit: '实际乘积神经元 a[j]', full_mlp_output_coordinate: '整个 MLP 输出 m[k]', gate_input_term: 'gate 的单输入项 W[j,k]x[k]', up_input_term: 'up 的单输入项 W[j,k]x[k]', single_unit_down_term: '该神经元的 down 项 W[k,j]a[j]', gate_all_input_sum: 'gate 全部输入 FP64 重算', up_all_input_sum: 'up 全部输入 FP64 重算' };

export default function NativeMlpParameterInspector() {
  const [meta, setMeta] = useState(null), [data, setData] = useState(null), [error, setError] = useState(''), [busy, setBusy] = useState(false);
  const [choice, setChoice] = useState({ case: '', site: '23:6197', coordinate: '0', checkpoint: '24', token: '' });
  const request = useRef(0);
  async function load() {
    const id = ++request.current; setBusy(true); setError(''); setData(null);
    try {
      const response = await fetch(`${base}/native-mlp-cases`), result = await response.json();
      if (!response.ok) throw new Error(result.detail || '原生 MLP 尚未发布');
      if (id === request.current) { setMeta(result); setChoice((c) => ({ ...c, case: String(result.cases[0]?.case ?? '') })); }
    } catch (e) { if (id === request.current) setError(e.message); } finally { if (id === request.current) setBusy(false); }
  }
  async function query(event) {
    event.preventDefault(); const id = ++request.current; setBusy(true); setError(''); setData(null);
    const [layer, unit] = choice.site.split(':'); const p = new URLSearchParams({ case: choice.case, layer, unit, coordinate: choice.coordinate, checkpoint: choice.checkpoint });
    if (choice.token.trim()) p.set('token', choice.token);
    try {
      const response = await fetch(`${base}/native-mlp-parameter?${p}`), result = await response.json();
      if (!response.ok) throw new Error(result.detail || '原生参数查询失败');
      if (id === request.current) setData(result);
    } catch (e) { if (id === request.current) setError(e.message); } finally { if (id === request.current) setBusy(false); }
  }
  return <details style={{ marginTop: 14, borderTop: '1px solid #537184', paddingTop: 10 }}>
    <summary>真实中层 gate / up / down：单神经元、单权重、全部输入坐标（Phase 2670–2676）</summary>
    <p>固定五个候选神经元只是观察窗口；全部坐标背景在上方热力图中。k 可查询 0–2559，包含低值坐标；不搬运其他样本的场。</p>
    <p>编号从 0 开始。H0 是词嵌入；H1–H36 是 block 0–35 的输出，其中 H36 尚未经过模型末尾的 RMSNorm。MLP 的 neuron 编号与残差坐标 k 不是同一对象。</p>
    <button type="button" onClick={load} disabled={busy}>读取中层 MLP 展示例</button>
    {meta && <form onSubmit={query} style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10 }}>
      <label>样本 <select aria-label="中层MLP样本" value={choice.case} onChange={(e) => setChoice({ ...choice, case: e.target.value, token: '' })} style={{ maxWidth: 380 }}>{meta.cases.map((r) => <option key={r.case} value={r.case}>{r.label}</option>)}</select></label>
      <label>原生单元 <select aria-label="中层MLP单元" value={choice.site} onChange={(e) => setChoice({ ...choice, site: e.target.value })}>{meta.sites.map((s) => <option key={`${s.layer}:${s.unit}`} value={`${s.layer}:${s.unit}`}>layer {s.layer} · neuron {s.unit}</option>)}</select></label>
      <label>已做真实改动的标量 <select aria-label="中层MLP实测标量" value="" onChange={(e) => { const s = meta.validated_scalars[Number(e.target.value)]; if (s) setChoice({ ...choice, site: `${s.layer}:${s.unit}`, coordinate: String(s.coordinate) }); }}><option value="">选择可自动填入坐标</option>{meta.validated_scalars.map((s, i) => <option key={i} value={i}>L{s.layer} J{s.unit} · {s.kind} k={s.coordinate} · {s.control === 'low' ? '低值对照' : '普通对照'}</option>)}</select></label>
      {['coordinate', 'checkpoint', 'token'].map((k) => <label key={k}>{k} {k === 'token' ? '（空=边界）' : ''}<input aria-label={`中层MLP ${k}`} type="number" min="0" max={k === 'coordinate' ? 2559 : k === 'checkpoint' ? 36 : undefined} step="1" required={k !== 'token'} value={choice[k]} onChange={(e) => setChoice({ ...choice, [k]: e.target.value })} style={{ width: 75 }} /></label>)}
      <button type="submit" disabled={busy}>查询真实中层参数</button>
    </form>}
    {error && <p role="alert">{error}</p>}
    {data && <>
      <p>{data.case_id} · layer {data.layer}, neuron {data.unit} · token {data.token} ({data.token_string}) · k={data.coordinate}</p>
      <p>短答案自然输出：{data.natural_short.generated}；内容正确：{String(data.natural_short.content_correct)}。结构化答案自然输出：{data.natural_structured.generated}；内容正确：{String(data.natural_structured.content_correct)}。</p>
      <details><summary>实际短答案与结构化答案前缀（两种测量条件）</summary><pre style={{ whiteSpace: 'pre-wrap' }}>{data.prompt}</pre><pre style={{ whiteSpace: 'pre-wrap' }}>{data.fp_structured_prompt}</pre></details>
      <div style={{ overflowX: 'auto' }}><table style={{ fontVariantNumeric: 'tabular-nums' }}><tbody>{Object.entries(data.values).map(([k, v]) => <tr key={k}><td style={{ paddingRight: 20 }}>{labels[k] || k}</td><td>{String(v)}</td></tr>)}</tbody></table></div>
      <p>以下是同值 FP32、完整结构化答案所有 token 的局部导数，不是权重数值，也不是自由生成成功率。</p>
      <div style={{ overflowX: 'auto' }}><table style={{ borderSpacing: '16px 4px', fontVariantNumeric: 'tabular-nums' }}><thead><tr><th>单参数</th><th>完整分数</th><th>内容</th><th>格式</th><th>EOS</th></tr></thead><tbody>{Object.entries(data.scalar_sequence_derivatives).map(([kind, v]) => <tr key={kind}><td>{kind}</td>{['all', 'content', 'format', 'eos'].map((p) => <td key={p}>{String(v[p])}</td>)}</tr>)}</tbody></table></div>
      <p>单标量有限变化的效应可能接近浮点分辨率；no-op 一致不等于微小效应已准确分辨。这里保留预测误差，不把完成测量标为机制通过。</p>
      {data.actual_scalar_validation.length ? data.actual_scalar_validation.map((s) => <div key={s.site_index}>
        <p>实际改动：{s.kind} · {s.control} · 原权重 {String(s.original_weight)} · 向量 RMS {String(s.vector_rms)} · 本例 no-op 差 {String(data.scalar_noop_error)}</p>
        <div style={{ overflowX: 'auto' }}><table style={{ borderSpacing: '14px 3px' }}><thead><tr><th>剂量倍数</th><th>方向</th><th>实际 Δ权重</th><th>实测完整 Δ分数</th><th>导数预测</th><th>实测−预测</th></tr></thead><tbody>{s.effects.map((e, i) => <tr key={i}><td>{e.dose_scale}</td><td>{e.sign}</td><td>{String(e.actual_deltas[0])}</td><td>{String(e.effect)}</td><td>{String(e.predicted)}</td><td>{String(e.effect - e.predicted)}</td></tr>)}</tbody></table></div>
      </div>) : <p>此物理坐标有完整导数，但不在预先冻结的 30 个有限改动标量中；未做过的有限改动不补造数值。</p>}
      <details><summary>全部 token 的参数项（未截断、未 Top-K）</summary><div style={{ maxHeight: 300, overflow: 'auto' }}><table style={{ borderSpacing: '14px 3px' }}><thead><tr><th>分支 / token</th><th>gate</th><th>up</th><th>down</th></tr></thead><tbody>{['Y', 'N'].flatMap((b) => data.all_token_scalar_terms.gate[b].map((_, t) => <tr key={`${b}${t}`}><td>{b}/{t}</td>{['gate', 'up', 'down'].map((k) => <td key={k}>{String(data.all_token_scalar_terms[k][b][t])}</td>)}</tr>))}</tbody></table></div></details>
      <p style={{ color: '#e4c689' }}>{data.boundary}</p>
    </>}
  </details>;
}
