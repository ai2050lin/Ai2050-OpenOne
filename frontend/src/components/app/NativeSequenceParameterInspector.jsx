import { useRef, useState } from 'react';
import { API_CONFIG } from '../../config/api';

const base = `${API_CONFIG.main.replace(/\/$/, '')}/api/research-assets`;
const fmt = (v) => String(v);
const labels = { embedding: '词嵌入[token,hj]', hidden: 'H[checkpoint,token,hj]', mlp: 'MLP[layer,ak]', decision_hidden_boundary: '实际答案识别时刻 H[checkpoint,末token,hj]', decision_mlp_boundary: '实际答案识别时刻 MLP[layer,ak]' };

export default function NativeSequenceParameterInspector({ multi = false }) {
  const prefix = multi ? 'native-multitoken' : 'native-sequence';
  const [selection, setSelection] = useState({ case: '256', layer: '0', j: '57', k: '32', hj: '0', ak: '0', checkpoint: '1', token: '' });
  const [data, setData] = useState(null), [cases, setCases] = useState([]), [busy, setBusy] = useState(false), [error, setError] = useState('');
  const requestId = useRef(0);
  async function inspect(event) {
    event.preventDefault(); const id = ++requestId.current; setBusy(true); setData(null); setError('');
    try {
      const params = new URLSearchParams(Object.fromEntries(Object.entries(selection).filter(([, v]) => v !== '')));
      const [response, options] = await Promise.all([fetch(`${base}/${prefix}-parameter?${params}`), fetch(`${base}/${prefix}-cases`)]);
      const payload = await response.json(); if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : '参数读取失败');
      const choices = options.ok ? (await options.json()).cases : [];
      if (id === requestId.current) { setData(payload); setCases(choices); }
    } catch (e) { if (id === requestId.current) setError(e.message || '读取失败'); }
    finally { if (id === requestId.current) setBusy(false); }
  }
  return <details style={{ marginTop: 14, borderTop: '1px solid #537184', paddingTop: 10 }}>
    <summary style={{ cursor: 'pointer' }}>{multi ? '多 token 内容 / 格式 / EOS：全坐标与真实单参数（Phase 2662–2669）' : '完整规范答案 + EOS：逐坐标、逐神经元、逐参数（Phase 2655–2661）'}</summary>
    <p>64 个示例，H 检查点 0 为词嵌入，1–36 为 block 输出；全部物理坐标可查询。token 留空取原 prompt 边界。{multi ? '每条规范答案实际为 4 token，再预测 EOS；不是长句解释。' : '两条分支分别强制输入答案词，并预测结束 token。'}</p>
    <form onSubmit={inspect} style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
      {Object.keys(selection).map((key) => <label key={key}>{key} <input aria-label={`${prefix} ${key}`} type="number" min="0" step="1" required={key !== 'token'} list={key === 'case' ? `${prefix}-options` : undefined} value={selection[key]} onChange={(e) => setSelection((old) => ({ ...old, [key]: e.target.value }))} style={{ width: 66 }} /></label>)}
      <datalist id={`${prefix}-options`}>{cases.map((c) => <option key={c.case_index} value={c.case_index}>{c.case_id}</option>)}</datalist>
      <button type="submit" disabled={busy}>{busy ? '读取中…' : '查询真实参数'}</button>
    </form>
    {error && <p role="alert" style={{ color: '#ffb5ab' }}>{error}</p>}
    {data && <>
      <p>{data.case_id}<br />{data.text}<br />BF16 实际生成：{data.generated}；预定答案：{data.target}；整词判定：{data.content_correct ? '通过' : '未通过'}。</p>
      <p>FP32 logP(Y,EOS) − logP(N,EOS) = {fmt(data.sequence_contrast)}；{multi ? `内容部分 ${fmt(data.content_contrast)}；格式部分 ${fmt(data.format_contrast)}` : `首词部分 ${fmt(data.first_token_contrast)}`}；结束部分 {fmt(data.eos_contrast)}。</p>
      <p>真实 V[{data.layer}][{data.j},{data.k}] = {fmt(data.actual_weight)}；完整序列对该权重的导数 = {fmt(data.parameter_derivative)}。</p>
      {multi && <p>该参数的分项导数：{Object.entries(data.part_parameter_derivatives).map(([p, v]) => `${p} = ${fmt(v)}`).join('；')}。全部是同一多 token 指令下的原生计算。</p>}
      <div style={{ maxWidth: '100%', overflowX: 'auto' }}><table style={{ fontVariantNumeric: 'tabular-nums', borderSpacing: '10px 6px', whiteSpace: 'nowrap' }}><thead><tr><th>物理量</th><th>BF16</th><th>FP32</th></tr></thead><tbody>
        {Object.keys(data.fields.bf16).map((key) => <tr key={key}><td>{labels[key] || key}</td><td>{fmt(data.fields.bf16[key])}</td><td>{data.fields.fp32[key] === undefined ? '未采集' : fmt(data.fields.fp32[key])}</td></tr>)}
      </tbody></table></div>
      {data.branches.map((b) => <details key={b.label} style={{ marginTop: 10 }}>
        <summary>{b.label} 分支：{b.answer} + EOS；logP = {fmt(b.total_logprob)}；全 token 参数导数 = {fmt(b.derivative)}</summary>
        <p>预测目标 IDs：{b.target_ids.join(', ')}；各步 logP：{b.logprobs.map(fmt).join(', ')}。仅 prompt 末位置导数 {fmt(b.prompt_last_only)}；仅分支末位置 {fmt(b.branch_last_only)}。完整对比取 Y 导数减 N 导数。</p>
        <p>此分支 H[{data.layer + 1}, prompt 末位, {data.hj}] 伴随：{fmt(b.hidden_prompt_adjoint)}；MLP[{data.layer}, {data.ak}] 伴随：{fmt(b.mlp_prompt_adjoint)}。</p>
        {multi && <p>目标类别：{b.categories.join(', ')}；分项 logP：{Object.entries(b.part_logprobs).map(([p, v]) => `${p}=${fmt(v)}`).join('；')}；分项参数导数：{Object.entries(b.part_derivatives).map(([p, v]) => `${p}=${fmt(v)}`).join('；')}。</p>}
        <div style={{ maxHeight: 240, overflow: 'auto' }}><table style={{ whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums' }}><thead><tr>{['位置 / token', '输入[k]', 'V 输出[j]', '伴随[j]', '参数乘积', ...(multi ? ['内容乘积', '格式乘积', 'EOS 乘积'] : [])].map((s) => <th key={s}>{s}</th>)}</tr></thead><tbody>
          {b.tokens.map((t) => <tr key={t.position}><td>{t.position} / {t.token} ({t.token_id})</td>{['x', 'value', 'adjoint', 'product'].map((key) => <td key={key} style={{ padding: 4 }}>{fmt(t[key])}</td>)}{multi && ['content', 'format', 'eos'].map((p) => <td key={p} style={{ padding: 4 }}>{fmt(t.parts[p].product)}</td>)}</tr>)}
        </tbody></table></div>
      </details>)}
      <p style={{ color: '#e4c689' }}>{data.boundary}</p>
    </>}
  </details>;
}
