import { useRef, useState } from 'react';
import { API_CONFIG } from '../../config/api';

const base = `${API_CONFIG.main.replace(/\/$/, '')}/api/research-assets`;
const modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'];
const fmt = (value) => Number(value).toPrecision(8);
const labels = { all_token_derivative: '全 token 参数导数', last_token_derivative: '仅末位置参数导数', embedding_last_token_hj: '末 token 词嵌入[hj]', hidden_block_output_hj: 'block 输出 HiddenState[hj]', hidden_adjoint_hj: 'HiddenState 伴随[hj]', mlp_neuron_ak: 'MLP 中间神经元[ak]', mlp_adjoint_ak: 'MLP 伴随[ak]' };

export default function NativePrecisionParameterInspector() {
  const [selection, setSelection] = useState({ frame: '20', layer: '0', module: 'v_proj', j: '0', k: '0', hj: '0', ak: '0' });
  const [frames, setFrames] = useState([]);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const requestId = useRef(0);
  const change = (key, value) => setSelection((old) => ({ ...old, [key]: value }));
  async function inspect(event) {
    event.preventDefault(); const id = ++requestId.current;
    setBusy(true); setError(''); setData(null);
    try {
      const [response, optionResponse] = await Promise.all([fetch(`${base}/native-precision-parameter?${new URLSearchParams(selection)}`), fetch(`${base}/native-precision-frames`)]);
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : '双精度查询失败');
      const options = optionResponse.ok ? (await optionResponse.json()).frames : [];
      if (id === requestId.current) { setData(payload); setFrames(options); }
    } catch (failure) { if (id === requestId.current) setError(failure.message || '读取失败'); }
    finally { if (id === requestId.current) setBusy(false); }
  }
  return <details style={{ marginTop: 14, borderTop: '1px solid #537184', paddingTop: 10 }}>
    <summary style={{ cursor: 'pointer' }}>同权重 BF16／FP32 逐坐标对照（Phase 2637–2640）</summary>
    <p>同一个权重、同一前缀；仅计算精度不同。j/k 指定矩阵位置，hj 指定隐藏坐标，ak 指定 MLP 神经元，三者不要混同。</p>
    <form onSubmit={inspect} style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      <label>frame <input aria-label="精度对照 frame" list="precision-frame-options" type="number" min="0" step="1" required value={selection.frame} onChange={(e) => change('frame', e.target.value)} style={{ width: 65 }} /></label>
      <datalist id="precision-frame-options">{frames.map((f) => <option key={f.frame_id} value={f.frame_id}>{f.case_id}</option>)}</datalist>
      <label>层 <select aria-label="精度对照 layer" value={selection.layer} onChange={(e) => change('layer', e.target.value)}>{[0, 5, 17, 35].map((l) => <option key={l} value={l}>{l}</option>)}</select></label>
      <label>矩阵 <select aria-label="精度对照 module" value={selection.module} onChange={(e) => change('module', e.target.value)}>{modules.map((m) => <option key={m}>{m}</option>)}</select></label>
      {['j', 'k', 'hj', 'ak'].map((key) => <label key={key}>{key} <input aria-label={`精度对照 ${key}`} type="number" min="0" step="1" required value={selection[key]} onChange={(e) => change(key, e.target.value)} style={{ width: 64 }} /></label>)}
      <button type="submit" disabled={busy}>{busy ? '读取中…' : '同位置对照'}</button>
    </form>
    {error && <p role="alert" style={{ color: '#ffb5ab' }}>{error}</p>}
    {data && <>
      <p>{data.case_id} · L{data.layer}/{data.module}[{data.j},{data.k}]<br />两种精度的真实权重数值相同：{fmt(data.actual_weight_jk_same_both_precisions)}</p>
      <table style={{ width: '100%', fontVariantNumeric: 'tabular-nums' }}><thead><tr><th>量</th><th>BF16</th><th>FP32</th></tr></thead><tbody>
        {Object.keys(data.values.bf16).map((key) => <tr key={key}><td>{labels[key] || key}</td><td>{fmt(data.values.bf16[key])}</td><td>{fmt(data.values.fp32[key])}</td></tr>)}
      </tbody></table>
      <p>所有 token 位置的参数乘积项如下；HiddenState／MLP 表格取当前边界位置。</p>
      <div style={{ overflow: 'auto', maxHeight: 250 }}><table style={{ whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums' }}>
        <thead><tr>{['位置 / token', '输入 BF16', '输入 FP32', '伴随 BF16', '伴随 FP32', '乘积 BF16', '乘积 FP32'].map((v) => <th key={v} style={{ padding: 4 }}>{v}</th>)}</tr></thead>
        <tbody>{data.tokens.map((t) => <tr key={t.position}><td>{t.position} / {t.token}</td>{['input_k', 'output_adjoint_j', 'product'].flatMap((key) => ['bf16', 'fp32'].map((p) => <td key={`${key}/${p}`} style={{ padding: 4 }}>{fmt(t[p][key])}</td>))}</tr>)}</tbody>
      </table></div>
      <p style={{ color: '#e4c689' }}>FP32 是同值权重的数值对照，不是恢复了预训练精度；它会改变基线状态。局部导数可验证，不代表已发现语义齿轮。</p>
    </>}
  </details>;
}
