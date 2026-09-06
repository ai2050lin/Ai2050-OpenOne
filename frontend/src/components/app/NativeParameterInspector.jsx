import { useRef, useState } from 'react';
import { API_CONFIG } from '../../config/api';
import NativePathParameterInspector from './NativePathParameterInspector';
import NativePrecisionParameterInspector from './NativePrecisionParameterInspector';
import NativeOperationParameterInspector from './NativeOperationParameterInspector';
import NativeSequenceParameterInspector from './NativeSequenceParameterInspector';
import NativeAtlasHeatmap from './NativeAtlasHeatmap';
import NativeMlpParameterInspector from './NativeMlpParameterInspector';
import NativeSourceParameterInspector from './NativeSourceParameterInspector';

const labels = {
  embedding_anchor_j: '词嵌入：锚点坐标 j',
  hidden_final_boundary_j: 'HiddenState：末层边界坐标 j',
  mlp_neuron_k: 'MLP 中间神经元 a[k]',
  actual_down_weight_jk: '真实权重 W_down[j,k]',
  down_weight_local_derivative_jk: '单 down 权重局部导数',
  gate_weight_local_derivative_kj: '单 gate 权重局部导数 [k,j]',
  up_weight_local_derivative_kj: '单 up 权重局部导数 [k,j]',
  single_neuron_delete_predicted_margin_change: '单神经元归零：解析 margin 变化',
  single_coordinate_delete_predicted_margin_change: '单坐标归零：解析 margin 变化',
};

export default function NativeParameterInspector() {
  const [selection, setSelection] = useState({ model: 'qwen4', case: '0', j: '0', k: '0', checkpoint: '', token: '' });
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [wide, setWide] = useState(false);
  const requestId = useRef(0);
  async function inspect(event) {
    event.preventDefault();
    const id = ++requestId.current;
    setBusy(true); setError(''); setData(null);
    try {
      const query = new URLSearchParams(Object.fromEntries(Object.entries(selection).filter(([, value]) => value !== '')));
      const response = await fetch(`${API_CONFIG.main.replace(/\/$/, '')}/api/research-assets/native-parameter?${query}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : '坐标查询失败');
      if (id === requestId.current) setData(payload);
    } catch (failure) {
      if (id === requestId.current) setError(failure.message || '无法连接原生参数数据');
    } finally { if (id === requestId.current) setBusy(false); }
  }
  return (
    <details style={{ position: 'absolute', bottom: 64, left: wide ? 16 : 'min(340px, 20vw)', width: wide ? 'min(1100px, calc(100% - 56px))' : 'min(510px, 72vw)', maxHeight: wide ? 'calc(100vh - 110px)' : '65vh', overflow: 'auto', zIndex: wide ? 150 : 24, color: '#dce7f5', background: '#101926f5', border: '1px solid #537184', borderRadius: 8, padding: 12, fontSize: 12 }}>
      <summary style={{ cursor: 'pointer' }}>原生坐标 → 神经元 → 单参数查询（已发布研究）</summary>
      <button type="button" aria-label="原生查询宽屏" aria-pressed={wide} onClick={() => setWide(!wide)} style={{ marginTop: 8 }}>{wide ? '还原紧凑查询视图' : '展开宽屏查询视图'}</button>
      <p>按原始编号查询全部坐标与最终 MLP 参数，无 Top-K 筛选。数据只读，不加载模型。</p>
      <form onSubmit={inspect} style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <label>模型 <select aria-label="原生参数模型" value={selection.model} onChange={(e) => setSelection({ ...selection, model: e.target.value })}>
          <option value="qwen4">Qwen3-4B</option><option value="qwen14">Qwen3-14B</option><option value="glm4">GLM4</option><option value="ds7">DS7B</option>
        </select></label>
        {['case', 'j', 'k'].map((key) => <label key={key}>{key} <input aria-label={`原生参数 ${key}`} type="number" min="0" step="1" required value={selection[key]} onChange={(e) => setSelection({ ...selection, [key]: e.target.value })} style={{ width: 66 }} /></label>)}
        {['checkpoint', 'token'].map((key) => <label key={key}>{key}（可选） <input aria-label={`全token ${key}`} type="number" min="0" step="1" value={selection[key]} onChange={(e) => setSelection({ ...selection, [key]: e.target.value })} style={{ width: 66 }} /></label>)}
        <button type="submit" disabled={busy}>{busy ? '读取中…' : '查询'}</button>
      </form>
      {error && <p role="alert" style={{ color: '#ffb5ab' }}>{error}</p>}
      {data && <>
        <p>{data.case_id} · layer {data.layer} · j={data.coordinate_j}, k={data.neuron_k}</p>
        <p>可选范围：case 0–{data.counts.cases - 1}；j 0–{data.counts.coordinates - 1}；k 0–{data.counts.neurons - 1}</p>
        <p style={{ maxHeight: 90, overflow: 'auto' }}>{data.prompt}</p>
        {data.full_token_coordinate !== undefined && <p>全token原场：checkpoint {data.full_token_checkpoint} / token {data.full_token_position} ({data.full_token_string}) / j {data.coordinate_j} = {Number(data.full_token_coordinate).toPrecision(9)}</p>}
        <table style={{ width: '100%', fontVariantNumeric: 'tabular-nums' }}><tbody>
          {Object.entries(data.values).map(([key, value]) => <tr key={key}><td>{labels[key] || key}</td><td style={{ textAlign: 'right' }}>{Number(value).toPrecision(9)}</td></tr>)}
        </tbody></table>
        <p style={{ color: '#e4c689' }}>{data.boundary}</p>
      </>}
      <NativeAtlasHeatmap />
      <NativeSourceParameterInspector />
      <NativeMlpParameterInspector />
      <NativePathParameterInspector />
      <NativePrecisionParameterInspector />
      <NativeOperationParameterInspector key="matched-operation" />
      <NativeOperationParameterInspector key="output-function" apiPrefix="native-output" caseCount={64} title="语言操作 × 输出功能：固定读出与逐参数图谱（Phase 2648–2654）" />
      <NativeSequenceParameterInspector key="full-sequence" />
      <NativeSequenceParameterInspector key="multi-sequence" multi />
    </details>
  );
}
