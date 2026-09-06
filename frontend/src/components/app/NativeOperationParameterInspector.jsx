import { useRef, useState } from 'react';
import { API_CONFIG } from '../../config/api';

const base = `${API_CONFIG.main.replace(/\/$/, '')}/api/research-assets`;
const fmt = (v) => Number(v).toPrecision(8);
const fieldLabels = { embedding_token_hj: '词嵌入[token,hj]', hidden_checkpoint_token_hj: 'HiddenState[检查点,token,hj]', hidden_V_block_boundary_hj: 'V 所在 block 的输出边界[hj]', mlp_neuron_boundary_ak: 'MLP 中间单位[ak]' };

export default function NativeOperationParameterInspector({ apiPrefix = 'native-operation', caseCount = 32, title = '八操作匹配图谱：双读出、词嵌入、神经元与单 V 参数（Phase 2641–2647）' }) {
  const [selection, setSelection] = useState({ case: '0', layer: '0', j: '57', k: '32', hj: '0', ak: '0', checkpoint: '1', token: '' });
  const [cases, setCases] = useState([]), [data, setData] = useState(null), [error, setError] = useState(''), [busy, setBusy] = useState(false);
  const requestId = useRef(0);
  const change = (key, value) => setSelection((old) => ({ ...old, [key]: value }));
  async function inspect(event) {
    event.preventDefault(); const id = ++requestId.current; setBusy(true); setData(null); setError('');
    try {
      const params = new URLSearchParams(Object.fromEntries(Object.entries(selection).filter(([, value]) => value !== '')));
      const [response, optionResponse] = await Promise.all([fetch(`${base}/${apiPrefix}-parameter?${params}`), fetch(`${base}/${apiPrefix}-cases`)]);
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : '逐参数查询失败');
      const choices = optionResponse.ok ? (await optionResponse.json()).cases : [];
      if (id === requestId.current) { setData(payload); setCases(choices); }
    } catch (failure) { if (id === requestId.current) setError(failure.message || '读取失败'); }
    finally { if (id === requestId.current) setBusy(false); }
  }
  return <details style={{ marginTop: 14, borderTop: '1px solid #537184', paddingTop: 10 }}>
    <summary style={{ cursor: 'pointer' }}>{title}</summary>
    <p>保留 {caseCount} 个完整示例；全部坐标可查询。checkpoint 0 是词嵌入，1–36 是 block 输出；token 留空取输出边界。梯度表取边界，参数乘积覆盖全部 token。</p>
    <form onSubmit={inspect} style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      <label>case <input aria-label={`${apiPrefix} case`} list={`${apiPrefix}-case-options`} type="number" min="0" required value={selection.case} onChange={(e) => change('case', e.target.value)} style={{ width: 64 }} /></label>
      <datalist id={`${apiPrefix}-case-options`}>{cases.map((r) => <option key={r.case_index} value={r.case_index}>{r.case_id}</option>)}</datalist>
      <label>V 层 <select aria-label={`${apiPrefix} layer`} value={selection.layer} onChange={(e) => change('layer', e.target.value)}>{[0, 5, 17, 35].map((l) => <option key={l}>{l}</option>)}</select></label>
      {['j', 'k', 'hj', 'ak', 'checkpoint', 'token'].map((key) => <label key={key}>{key} <input aria-label={`${apiPrefix} ${key}`} type="number" min="0" step="1" required={key !== 'token'} value={selection[key]} onChange={(e) => change(key, e.target.value)} style={{ width: 62 }} /></label>)}
      <button type="submit" disabled={busy}>{busy ? '读取中…' : '查询真实标量'}</button>
    </form>
    {error && <p role="alert" style={{ color: '#ffb5ab' }}>{error}</p>}
    {data && <>
      <p>{data.case_id}<br />{data.text}{data.prefill && <><br />实验者给定的续写前缀：{data.prefill}</>}<br />实际生成：{data.generated}；目标：{data.target}；预定完整答案评分：{data.name_content_correct ? '通过' : '未通过'}。</p>
      <p>真实 V 权重 [{data.j},{data.k}] = {fmt(data.actual_weight)}；隐藏坐标 hj、MLP 单位 ak 与权重位置是不同对象。</p>
      <table style={{ width: '100%', fontVariantNumeric: 'tabular-nums' }}><thead><tr><th>原始数值</th><th>BF16</th><th>FP32</th></tr></thead><tbody>
        {Object.keys(data.fields.bf16).map((key) => <tr key={key}><td>{fieldLabels[key] || key}</td><td>{fmt(data.fields.bf16[key])}</td><td>{fmt(data.fields.fp32[key])}</td></tr>)}
      </tbody></table>
      {Object.entries(data.objectives).map(([obj, r]) => <p key={obj}>{obj === 'native' ? '冻结 BF16 原生输出对' : '外部固定读出对'}：{r.available ? <>IDs {r.output_ids.join(' − ')}；全 token 参数导数 {fmt(r.all_token_parameter_derivative)}；仅末位置 {fmt(r.last_token_only_derivative)}；HiddenState 伴随 {fmt(r.hidden_block_boundary_adjoint_hj)}；MLP 伴随 {fmt(r.mlp_boundary_adjoint_ak)}</> : '不可用（不以零代替）'}</p>)}
      <div style={{ overflow: 'auto', maxHeight: 250 }}><table style={{ whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums' }}>
        <thead><tr>{['token', 'V 输入[k]', 'V 输出[j]', '原生伴随[j]', '原生参数乘积', '固定读出伴随[j]', '固定读出参数乘积'].map((s) => <th key={s} style={{ padding: 4 }}>{s}</th>)}</tr></thead>
        <tbody>{data.tokens.map((t) => <tr key={t.position}><td>{t.position}/{t.token}</td><td>{fmt(t.V_input_k)}</td><td>{fmt(t.V_output_j)}</td>{['native', 'common'].flatMap((obj) => ['adjoint_j', 'parameter_product'].map((key) => <td key={`${obj}/${key}`} style={{ padding: 4 }}>{t[obj] ? fmt(t[obj][key]) : '不可用'}</td>))}</tr>)}</tbody>
      </table></div>
      <p style={{ color: '#e4c689' }}>FP32 会改变基线；共同读出由实验者指定。两种导数若使用相同输出对，相等是代数恒等，不是发现共同语义齿轮。</p>
      <p style={{ color: '#e4c689' }}>{data.boundary}</p>
    </>}
  </details>;
}
