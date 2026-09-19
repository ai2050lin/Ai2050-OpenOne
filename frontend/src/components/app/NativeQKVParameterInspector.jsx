import { useRef, useState } from 'react';
import { API_CONFIG } from '../../config/api';

const base = `${API_CONFIG.main.replace(/\/$/, '')}/api/research-assets`;
const initial = { case: '0', layer: '0', kind: 'q', output_row: '0', input_coordinate: '947', token: '0', query_position: '1', source_token: '0', head: '0', head_coordinate: '0', checkpoint: '0', unit: '0', output_coordinate: '0' };

export default function NativeQKVParameterInspector() {
  const [meta, setMeta] = useState(null), [choice, setChoice] = useState(initial), [data, setData] = useState(null);
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [showInputs, setShowInputs] = useState(false);
  const request = useRef(0);
  function change(patch) { request.current += 1; setChoice((old) => ({ ...old, ...patch })); setData(null); setShowInputs(false); setError(''); setBusy(false); }
  async function load() {
    const id = ++request.current; setBusy(true); setError(''); setData(null);
    try {
      const response = await fetch(`${base}/native-qkv-cases`), obj = await response.json();
      if (!response.ok) throw new Error(obj.detail || 'QKV 图谱尚未就绪');
      if (id === request.current) { setMeta(obj); setChoice({ ...initial, case: String(obj.cases[0].case) }); }
    } catch (e) { if (id === request.current) setError(e.message); } finally { if (id === request.current) setBusy(false); }
  }
  async function query(event) {
    event.preventDefault(); const id = ++request.current; setBusy(true); setError(''); setData(null); setShowInputs(false);
    try {
      const response = await fetch(`${base}/native-qkv-parameter?${new URLSearchParams(choice)}`), obj = await response.json();
      if (!response.ok) throw new Error(obj.detail || 'QKV 参数查询失败');
      if (id === request.current) setData(obj);
    } catch (e) { if (id === request.current) setError(e.message); } finally { if (id === request.current) setBusy(false); }
  }
  const selected = meta?.cases.find((r) => String(r.case) === choice.case);
  const numbers = { output_row: ['投影输出 r', choice.kind === 'q' ? 4095 : 1023], input_coordinate: ['输入坐标 k', 2559],
    token: ['投影及 E/H token', (selected?.tokens || 1) - 1], source_token: ['P 来源 token', (selected?.tokens || 1) - 1],
    head: ['P 的 query head', 31], head_coordinate: ['P/V/Wo head 坐标 d', 127], checkpoint: ['H checkpoint', 36],
    unit: ['MLP 神经元 j', 9727], output_coordinate: ['MLP/Wo 输出坐标', 2559] };
  return <details style={{ marginTop: 14, borderTop: '1px solid #537184', paddingTop: 10 }}>
    <summary>Q/K/V 单参数 → head 归一化 / RoPE → 来源路由 → MLP 真实坐标</summary>
    <p>只读原始数据，不加载模型。可逐坐标查看真实词嵌入、HiddenState、学习权重与乘积项；128 前缀标量试验中的 16 个预定真值/v0 展示例，不是全部语言类型的自然生成确认。</p>
    <button type="button" onClick={load} disabled={busy}>读取 QKV 参数图谱</button>
    {meta && <form onSubmit={query} style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 9 }}>
      <label>样本 <select aria-label="QKV样本" value={choice.case} onChange={(e) => change({ case: e.target.value, token: '0', source_token: '0' })} style={{ maxWidth: 430 }}>{meta.cases.map((r) => <option key={r.case} value={r.case}>{r.label}</option>)}</select></label>
      <label>真实单标量测试 <select aria-label="QKV实测标量" value="" onChange={(e) => { const c = meta.controls[Number(e.target.value)]; if (e.target.value !== '' && c) change({ layer: String(c.layer), kind: c.kind, output_row: String(c.output_row), input_coordinate: String(c.input_coordinate), head: String(c.kind === 'q' ? c.head : c.head * 4), head_coordinate: String(c.head_coordinate) }); }}><option value="">选择 48 个实测普通/低值参数</option>{meta.controls.map((c, i) => <option key={i} value={i}>L{c.layer} W{c.kind}[{c.output_row},{c.input_coordinate}] {c.control}</option>)}</select></label>
      <label>层 <select aria-label="QKV层" value={choice.layer} onChange={(e) => change({ layer: e.target.value })}>{meta.layers.map((l) => <option key={l}>{l}</option>)}</select></label>
      <label>投影 <select aria-label="QKV投影" value={choice.kind} onChange={(e) => change({ kind: e.target.value, output_row: '0' })}>{['q', 'k', 'v'].map((k) => <option key={k}>{k}</option>)}</select></label>
      <label>P/MLP 查询 <select aria-label="QKV查询边界" value={choice.query_position} onChange={(e) => change({ query_position: e.target.value })}><option value="0">正文末尾</option><option value="1">任务末尾</option></select></label>
      {Object.entries(numbers).map(([key, [label, max]]) => <label key={key}>{label} <input aria-label={`QKV ${key}`} type="number" min="0" max={max} step="1" required value={choice[key]} onChange={(e) => change({ [key]: e.target.value })} style={{ width: 78 }} /></label>)}
      <button type="submit" disabled={busy}>查询 QKV 真实参数与坐标</button>
    </form>}
    {error && <p role="alert">{error}</p>}
    {data && <>
      <p>{data.case_id} · 实际返回参数：L{data.indices.layer} W{data.indices.kind}[{data.indices.output_row},{data.indices.input_coordinate}]；投影/E/H token {data.indices.token}（{data.indices.token_string}），H{data.indices.checkpoint}；P/MLP 查询 token {data.indices.query_token}；query head {data.indices.head} → KV head {data.indices.kv_head}；MLP j{data.indices.unit} → 输出坐标 {data.indices.output_coordinate}。</p>
      <p>实际自然输出：{data.natural.generated}。此处的 3,072 条改权重后生成轨迹，完整 token/文本均未变；不能把另一组前缀在固定256计算中的 argmax 变化称为自然生成翻转。</p>
      <details><summary>真实输入</summary><pre style={{ whiteSpace: 'pre-wrap' }}>{data.prompt}</pre></details>
      <p>下列 norm/RoPE 对应 {data.traces.headnorm.kind.toUpperCase()} 的 head {data.traces.headnorm.head}；由投影输出 r 决定。V 本身没有 headnorm，选择 V 时这里只展示对应 K 分支。P 的 query head 是单独指定的，不能混用两种 head 编号。</p>
      <div style={{ overflowX: 'auto' }}><table><tbody>{Object.entries(data.values).map(([k, v]) => <tr key={k}><td style={{ paddingRight: 18 }}>{k}</td><td>{String(v)}</td></tr>)}</tbody></table></div>
      <button type="button" onClick={() => setShowInputs((v) => !v)}>{showInputs ? '收起全部输入坐标' : '展示全部 2560 输入坐标（不筛选）'}</button>
      {showInputs && <div style={{ maxHeight: 350, overflow: 'auto' }}><table><thead><tr><th>k</th><th>真实 W[r,k]</th><th>原生归一化 x[token,k]</th><th>完整坐标乘积 W·x</th></tr></thead><tbody>{data.traces.projection_input.W.map((w, i) => <tr key={i}><td>{i}</td><td>{String(w)}</td><td>{String(data.traces.projection_input.x[i])}</td><td>{String(data.traces.projection_input.Wx[i])}</td></tr>)}</tbody></table></div>}
      <details><summary>全部 128 个 head 内坐标：线性投影 / gamma / 原生归一化 / RoPE</summary><div style={{ maxHeight: 350, overflow: 'auto' }}><table><thead><tr><th>d</th>{['linear', 'gamma', 'native_norm', 'native_RoPE'].map((k) => <th key={k}>{k}</th>)}</tr></thead><tbody>{data.traces.headnorm.linear.map((_, i) => <tr key={i}><td>{i}</td>{['linear', 'gamma', 'native_norm', 'native_RoPE'].map((k) => <td key={k}>{String(data.traces.headnorm[k][i])}</td>)}</tr>)}</tbody></table></div></details>
      <details><summary>当前 query head 的全部真实来源 token（包含原生零值）</summary><div style={{ maxHeight: 350, overflow: 'auto' }}><table><thead><tr><th>位置</th><th>token</th><th>因果可见</th><th>原生 P</th><th>全部128维 QK</th><th>V[d]</th><th>当前head全部维度写入项</th></tr></thead><tbody>{data.traces.all_source_tokens.map((r) => <tr key={r.position}><td>{r.position}</td><td>{r.token}</td><td>{String(r.causal_allowed)}</td>{['P', 'QK_all128_before_mask', 'V_d', 'selected_head_output_term'].map((k) => <td key={k}>{String(r[k])}</td>)}</tr>)}</tbody></table></div></details>
      <details><summary>同一真实标量的实測 ± 两剂量（全部结果保留）</summary>
        <p>按完整权重行 RMS 给绝对剂量，不等于普通与低值参数的相对改变量相同。局部预测不是完整下游预测；Q/K 局部预测误差仍需解释。</p>
        {data.scalar_effects.length === 0 ? <p>此地址有真实权重与计算项，但没有本轮有限改动测量。请选择上方实测标量，不补造结果。</p> : <div style={{ overflowX: 'auto' }}><table><thead><tr><th>剂量</th><th>有效 Δθ</th><th>实际 ΔP L1</th><th>局部 P 预测误差 L1</th><th>实际 head 变化 L1</th><th>head 预测误差 L1</th><th>完整词表概率 L1</th><th>固定串 ΔlogP</th><th>首token</th></tr></thead><tbody>{data.scalar_effects.map((e, i) => <tr key={i}><td>{e.sign}×{e.dose}</td><td>{String(e.effective_delta)}</td><td>{String(e.probability.actual_L1)}</td><td>{String(e.probability.prediction_error_L1)}</td><td>{String(e.head_output.actual_L1)}</td><td>{String(e.head_output.prediction_error_L1)}</td><td>{String(e.all_vocabulary_probability_L1)}</td><td>{String(e.fixed_baseline_sequence_logprob_change)}</td><td>{e.baseline_next_id} → {e.changed_next_id}</td></tr>)}</tbody></table></div>}
        <p>没有保存逐例改权重后的原始 P 场；热力图中的标量响应是明确标注的跨128前缀聚合，不伪装成单例结果。</p>
      </details>
      <p style={{ color: '#e4c689' }}>{data.boundary}</p>
    </>}
  </details>;
}
