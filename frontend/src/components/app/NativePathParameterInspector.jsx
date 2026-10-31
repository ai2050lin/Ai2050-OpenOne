import { useRef, useState } from 'react';
import { API_CONFIG } from '../../config/api';

const endpoint = `${API_CONFIG.main.replace(/\/$/, '')}/api/research-assets`;
const modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'];
const number = (value) => Number(value).toPrecision(8);

export default function NativePathParameterInspector() {
  const [selection, setSelection] = useState({ frame: '0', layer: '35', module: 'v_proj', j: '0', k: '0' });
  const [frames, setFrames] = useState([]);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const requestId = useRef(0);
  const change = (key, value) => setSelection((old) => ({ ...old, [key]: value }));
  async function inspect(event) {
    event.preventDefault();
    const id = ++requestId.current;
    setBusy(true); setError(''); setData(null);
    try {
      const [response, options] = await Promise.all([
        fetch(`${endpoint}/native-path-parameter?${new URLSearchParams(selection)}`),
        fetch(`${endpoint}/native-path-frames`),
      ]);
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : '参数路径查询失败');
      const choices = options.ok ? (await options.json()).frames : [];
      if (id === requestId.current) { setData(payload); setFrames(choices); }
    } catch (failure) {
      if (id === requestId.current) setError(failure.message || '无法读取原生路径');
    } finally { if (id === requestId.current) setBusy(false); }
  }
  return <details style={{ marginTop: 14, borderTop: '1px solid #537184', paddingTop: 10 }}>
    <summary style={{ cursor: 'pointer' }}>跨位置单权重：全部 token 逐项查询（Phase 2632–2635）</summary>
    <p>j 是投影输出坐标，k 是输入坐标；仅 down 的 k 对应 MLP 中间神经元。28 个矩阵、34 个真实生成前缀，按原始编号读取。</p>
    <form onSubmit={inspect} style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      <label>前缀 frame <input aria-label="原生路径 frame" list="native-path-frames" type="number" min="0" step="1" required value={selection.frame} onChange={(e) => change('frame', e.target.value)} style={{ width: 65 }} /></label>
      <datalist id="native-path-frames">{frames.map((f) => <option key={f.frame_id} value={f.frame_id}>{f.case_id} / step {f.step}{f.eos ? ' EOS' : ''}</option>)}</datalist>
      <label>层 <select aria-label="原生路径 layer" value={selection.layer} onChange={(e) => change('layer', e.target.value)}>{[0, 5, 17, 35].map((l) => <option key={l} value={l}>{l}</option>)}</select></label>
      <label>矩阵 <select aria-label="原生路径 module" value={selection.module} onChange={(e) => change('module', e.target.value)}>{modules.map((m) => <option key={m}>{m}</option>)}</select></label>
      {['j', 'k'].map((key) => <label key={key}>{key} <input aria-label={`原生路径 ${key}`} type="number" min="0" step="1" required value={selection[key]} onChange={(e) => change(key, e.target.value)} style={{ width: 64 }} /></label>)}
      <button type="submit" disabled={busy}>{busy ? '读取中…' : '逐项查询'}</button>
    </form>
    {error && <p role="alert" style={{ color: '#ffb5ab' }}>{error}</p>}
    {data && <>
      <p>{data.case_id} · step {data.step} {data.eos ? '（EOS）' : ''} · {data.chosen_token} / {data.runnerup_token}<br />W[{data.j},{data.k}], shape {data.shape.join(' × ')} · {data.token_count} 个 token</p>
      <p>真实权重：{number(data.values.actual_weight_jk)}<br />全 token 导数：{number(data.values.all_token_derivative)}<br />仅末位置：{number(data.values.last_token_only_derivative)}<br />其余位置求和：{number(data.values.earlier_token_sum)}</p>
      <p>以下保留全部位置，未选 Top-K；乘积相加得到此权重的局部导数。</p>
      <div style={{ overflow: 'auto', maxHeight: 270 }}><table style={{ fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' }}>
        <thead><tr>{['位置', 'token / ID', '输入[k]', '投影输出[j]', '输出伴随[j]', '乘积'].map((label) => <th key={label} style={{ padding: 4 }}>{label}</th>)}</tr></thead>
        <tbody>{data.tokens.map((t) => <tr key={t.position}><td>{t.position}</td><td>{t.token} / {t.token_id}</td>{['input_k', 'projection_output_j', 'output_adjoint_j', 'product'].map((key) => <td key={key} style={{ padding: 4 }}>{number(t[key])}</td>)}</tr>)}</tbody>
      </table></div>
      <p style={{ color: '#e4c689' }}>这是对模型自身两个输出 token 的局部敏感度；反传不描述 BF16 舍入跳变，也不等同于语义必要性或有限扰动的准确预测。</p>
    </>}
  </details>;
}
