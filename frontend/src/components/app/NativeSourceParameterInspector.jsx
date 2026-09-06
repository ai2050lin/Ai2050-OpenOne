import { useRef, useState } from 'react';
import { API_CONFIG } from '../../config/api';
const base = `${API_CONFIG.main.replace(/\/$/, '')}/api/research-assets`;
const labels = {
  embedding_coordinate: '词嵌入 E[token,k]', hidden_coordinate: 'HiddenState H[checkpoint,token,k]',
  actual_probability: '原生 Attention 概率 P', actual_Wo_k_hd: '真实 W_o[k,head·128+d]',
  actual_Wgate_jk: '真实 W_gate[j,k]', actual_Wup_jk: '真实 W_up[j,k]', actual_Wdown_kj: '真实 W_down[k,j]',
  conditional_head_source_gate_term: '指定 head/source/坐标的条件 gate 输入项（非删除效果）',
  conditional_head_source_up_term: '指定 head/source/坐标的条件 up 输入项（非删除效果）',
};

export default function NativeSourceParameterInspector() {
  const [meta, setMeta] = useState(null), [data, setData] = useState(null), [error, setError] = useState(''), [busy, setBusy] = useState(false);
  const [choice, setChoice] = useState({ caseKey: '', site: '23:6197', coordinate: '0', checkpoint: '24', source_token: '0', head: '0', head_coordinate: '0', query_position: '1', hidden_token: '' });
  const request = useRef(0);
  function changeChoice(patch) {
    // A response belongs to the exact requested physical indices, never to
    // whichever indices happen to be visible when a delayed response arrives.
    request.current += 1;
    setChoice((current) => ({ ...current, ...patch }));
    setData(null); setError(''); setBusy(false);
  }
  async function load() {
    const id = ++request.current; setBusy(true); setError(''); setData(null);
    try {
      const response = await fetch(`${base}/native-source-cases`), obj = await response.json();
      if (!response.ok) throw new Error(obj.detail || '来源图谱尚未发布');
      if (id === request.current) { setMeta(obj); setChoice((c) => ({ ...c, caseKey: `${obj.cases[0].dataset}:${obj.cases[0].case}` })); }
    } catch (e) { if (id === request.current) setError(e.message); } finally { if (id === request.current) setBusy(false); }
  }
  async function query(event) {
    event.preventDefault(); const id = ++request.current; setBusy(true); setError(''); setData(null);
    const [dataset, caseId] = choice.caseKey.split(':'), [layer, unit] = choice.site.split(':');
    const params = new URLSearchParams({ dataset, case: caseId, layer, unit, ...Object.fromEntries(Object.entries(choice).filter(([k, v]) => !['caseKey', 'site'].includes(k) && v !== '')) });
    try {
      const response = await fetch(`${base}/native-source-parameter?${params}`), obj = await response.json();
      if (!response.ok) throw new Error(obj.detail || '来源参数查询失败');
      if (id === request.current) setData(obj);
    } catch (e) { if (id === request.current) setError(e.message); } finally { if (id === request.current) setBusy(false); }
  }
  const selected = meta?.cases.find((r) => `${r.dataset}:${r.case}` === choice.caseKey);
  return <details style={{ marginTop: 14, borderTop: '1px solid #537184', paddingTop: 10 }}>
    <summary>来源 token → Q/K/V → head → 原生坐标 → 真实 MLP 权重（Phase 2677–2684）</summary>
    <p>在完整原坐标图谱上寻址，不搬运差分。来源位置的 K/V 已包含前文上下文；将某项分账到一个词，不等于这个词独占该语义。</p>
    <button type="button" disabled={busy} onClick={load}>读取来源坐标展示例</button>
    {meta && <form onSubmit={query} style={{ display: 'flex', flexWrap: 'wrap', gap: 9, marginTop: 10 }}>
      <label>样本 <select aria-label="来源坐标样本" value={choice.caseKey} onChange={(e) => changeChoice({ caseKey: e.target.value, source_token: '0', hidden_token: '' })} style={{ maxWidth: 440 }}>{meta.cases.map((r) => <option key={`${r.dataset}:${r.case}`} value={`${r.dataset}:${r.case}`}>{r.dataset} · {r.label}</option>)}</select></label>
      <label>原生单元 <select aria-label="来源坐标单元" value={choice.site} onChange={(e) => changeChoice({ site: e.target.value })}>{meta.sites.map((s) => <option key={`${s.layer}:${s.unit}`} value={`${s.layer}:${s.unit}`}>L{s.layer} J{s.unit}</option>)}</select></label>
      <label>真实单标量测试 <select aria-label="来源实测标量" value="" onChange={(e) => { const s = e.target.value === '' ? null : meta.scalar_controls?.[Number(e.target.value)]; if (s) changeChoice({ site: `${s.layer}:${s.unit}`, coordinate: String(s.coordinate) }); }}><option value="">选择普通/低值标量</option>{(meta.scalar_controls || []).map((s, i) => <option key={i} value={i}>L{s.layer} J{s.unit} {s.kind} k{s.coordinate} {s.control === 'low' ? '低值' : '普通'}</option>)}</select></label>
      <label>注意力/MLP 查询边界 <select aria-label="来源查询边界" value={choice.query_position} onChange={(e) => changeChoice({ query_position: e.target.value })}><option value="0">正文末尾（前缀控制）</option><option value="1">任务末尾</option></select></label>
      {Object.entries({ coordinate: ['残差坐标 k', 2559], checkpoint: ['H checkpoint', 36], source_token: ['来源 token', (selected?.tokens || 1) - 1], head: ['query head', 31], head_coordinate: ['head 内坐标 d', 127], hidden_token: ['E/H token（空=查询边界）', (selected?.tokens || 1) - 1] }).map(([k, [label, max]]) => <label key={k}>{label} <input aria-label={`来源 ${k}`} type="number" min="0" max={max} step="1" required={k !== 'hidden_token'} value={choice[k]} onChange={(e) => changeChoice({ [k]: e.target.value })} style={{ width: 76 }} /></label>)}
      <button type="submit" disabled={busy}>查询来源与真实参数</button>
    </form>}
    {error && <p role="alert">{error}</p>}
    {data && <>
      <p>{data.case_id} · 实际结果 L{data.layer} / J{data.unit} / k{data.coordinate} / H{data.checkpoint} / head 内 d{data.head_coordinate} · 查询 token {data.query_token} ({data.query_token_string})；E/H token {data.hidden_token} ({data.hidden_token_string})；来源 {data.source_token} ({data.source_token_string})，外部角色 {data.source_role}；query head {data.head} → KV head {data.kv_head}。</p>
      <p>自然输出：{data.natural.generated}；内容正确 {String(data.natural.content_correct)}，严格格式正确 {String(data.natural.strict_correct)}。</p>
      <details><summary>真实输入前缀</summary><pre style={{ whiteSpace: 'pre-wrap' }}>{data.prompt}</pre></details>
      <div style={{ overflowX: 'auto' }}><table style={{ fontVariantNumeric: 'tabular-nums' }}><tbody>{Object.entries(data.values).map(([k, v]) => <tr key={k}><td style={{ paddingRight: 18 }}>{labels[k] || k}</td><td>{String(v)}</td></tr>)}</tbody></table></div>
      <details><summary>全部来源 token 在当前残差坐标上的分账（未筛 Top-K）</summary><div style={{ maxHeight: 350, overflow: 'auto' }}><table><thead><tr><th>token</th><th>原 token</th><th>外部角色</th><th>全部 head 项</th><th>当前 head 项</th></tr></thead><tbody>{data.source_trace.map((r) => <tr key={r.token}><td>{r.token}</td><td>{r.token_string}</td><td>{r.role}</td><td>{String(r.all_heads_term)}</td><td>{String(r.selected_head_term)}</td></tr>)}</tbody></table></div></details>
      <details><summary>当前 head 全部 128 个物理坐标 Q / K / V / W_o</summary><div style={{ maxHeight: 350, overflow: 'auto' }}><table><thead><tr><th>d</th><th>Q</th><th>K</th><th>V</th><th>真实 W_o</th></tr></thead><tbody>{data.head_coordinate_trace.Q.map((_, i) => <tr key={i}><td>{i}</td>{['Q', 'K', 'V', 'Wo'].map((k) => <td key={k}>{String(data.head_coordinate_trace[k][i])}</td>)}</tr>)}</tbody></table></div></details>
      {data.numeric_scalar_validation && <details><summary>同前缀真实单权重改动：局部预测与完整输出概率分账</summary>
        <p>{data.numeric_scalar_validation.scope}</p>
        <p>实际 BF16 生成串：{data.numeric_scalar_validation.observed_text}；FP32 测量是对该串全部 token 强制前缀计分。无操作精确：{String(data.numeric_scalar_validation.noop_exact)}；12 矩阵恢复：{String(data.numeric_scalar_validation.all12_matrices_restored)}。</p>
        {data.numeric_scalar_validation.effects.length === 0 ? <p>这个坐标有原值/来源分账，但不是本轮 30 个有限改动标量之一。可使用上方“真实单标量测试”选择器，不补造测试结果。</p> : <div style={{ overflowX: 'auto' }}><table><thead><tr><th>分支</th><th>剂量</th><th>真实 Δθ</th><th>局部输出 L1</th><th>局部误差 L1</th><th>相对误差</th><th>完整串 ΔlogP</th><th>非 EOS</th><th>EOS</th><th>FP64 读出 ΔlogP</th></tr></thead><tbody>{data.numeric_scalar_validation.effects.map((c, i) => <tr key={i}><td>{c.kind}</td><td>{c.sign}×{c.scale}</td>{['actual_delta', 'local_actual_l1', 'local_error_l1', 'local_relative_l1', 'output_full_delta', 'output_nonEOS_delta', 'output_EOS_delta', 'output_full64_delta'].map((k) => <td key={k}>{c[k] === undefined || c[k] === null ? '未测/分母为零' : String(c[k])}</td>)}</tr>)}</tbody></table></div>}
      </details>}
      <p style={{ color: '#e4c689' }}>{data.boundary}</p>
    </>}
  </details>;
}
