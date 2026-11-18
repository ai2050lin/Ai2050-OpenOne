import {useState} from 'react';
import {apiUrl} from '../../config.js';
import {ConditionAtlas} from './MlpConditionReport.jsx';
import {NativeArray} from './MechanismScene.jsx';

const n=x=>Number.isFinite(x)?x.toPrecision(4):'未测';
const names={broad:'广泛关联 B29/U17 → B30/U2',rare:'强响应 B33/U1521 → B34/U6097'};
const mode={upstream:'共同输入缩减',unit:'源神经元置零'};
function DoseCurve({rows}) {
  const points=rows.filter(r=>r.mode==='upstream').sort((a,b)=>a.dose-b.dose);
  const scale=Math.max(1e-9,...points.flatMap(r=>[Math.abs(r.target_delta_mean),Math.abs(r.control_delta_mean),...r.target_delta_interval.map(Math.abs)]));
  const x=d=>60+620*d,y=v=>110-85*v/scale;
  return <><svg viewBox="0 0 740 240" role="img" aria-label="共同输入干预与等范数对照的实测剂量曲线" className="v8-dose-chart">
    <line x1="60" x2="680" y1="110" y2="110" stroke="#9aa8b8"/>
    <text x="4" y="24">{n(scale)}</text><text x="4" y="114">0</text><text x="4" y="204">{n(-scale)}</text>
    {[['target_delta_mean','#3379b7'],['control_delta_mean','#bd8233']].map(([key,color])=><polyline key={key} fill="none" stroke={color} strokeWidth="2" points={points.map(r=>`${x(r.dose)},${y(r[key])}`).join(' ')}/>)}
    {points.map(r=><g key={r.dose}><line x1={x(r.dose)} x2={x(r.dose)} y1={y(r.target_delta_interval[0])} y2={y(r.target_delta_interval[1])} stroke="#3379b7" strokeWidth="3"/><circle cx={x(r.dose)} cy={y(r.target_delta_mean)} r="3" fill="#3379b7"/><text x={x(r.dose)-8} y="230">{r.dose}</text></g>)}
  </svg><p className="v8-note">横轴：缩减比例；纵轴：目标单元有符号变化。蓝色为干预及家族重采样区间，橙色为等范数对照；只连接实测剂量。</p></>;
}
function Prefix({sample}) {
  if(!sample)return null;
  const fields=[['R4','#3379b7'],['source_A','#bd8233'],['target_A','#758657']];
  const W=760,H=230,x=i=>45+i/Math.max(1,sample.R4.length-1)*690;
  return <><svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="完整前缀的原生强响应曲线" className="v8-dose-chart">
    <line x1="45" x2="735" y1="190" y2="190" stroke="#afb9c5"/>
    {fields.map(([key,color])=>{const m=Math.max(1e-12,...sample[key].map(Math.abs));return <polyline key={key} fill="none" stroke={color} strokeWidth="1.3" points={sample[key].map((v,i)=>`${x(i)},${190-Math.abs(v)/m*165}`).join(' ')}/>})}
    <text x="45" y="215">token 0</text><text x="640" y="215">{sample.R4.length-1}</text>
  </svg><p className="v8-note">蓝：R33 坐标4；橙：B33/U1521；绿：B34/U6097。各曲线按自身最大绝对值缩放，只用于对齐峰值；原值和正负号在下方查询。横轴为实际 token 位置。</p></>;
}
export default function MlpTriggerReport({doc,hideScene=false}) {
  const [tab,setTab]=useState('result'),[pair,setPair]=useState('rare'),[sampleIndex,setSampleIndex]=useState(0),[coordinate,setCoordinate]=useState(4);
  const o=doc.observations[pair],sample=doc.examples?.[sampleIndex],curves=doc.dose_curve.filter(r=>r.pair===pair);
  return <section aria-label="稀有触发与共同上游研究报告">
    <p><strong>强响应从哪里来？</strong> · Qwen3-4B BF16 · {doc.families} 个新家族、{doc.samples?.toLocaleString()} 条长文本</p>
    <p className="v8-note">真实单节点结果，正式机制证据尚待独立复核。扫描全部前缀；普查覆盖和成功上传不计为破解率。</p>
    <p>角色问答行为门：{doc.behavior.passed?'通过本轮句式检查':'未整体通过'}，最低句式准确率 {(100*Math.min(...Object.values(doc.behavior.per_frame))).toFixed(2)}%（门槛85%）。本报告检验数值触发与输入路径，不登记为属性语义机制。</p>
    <div className="v6-toolbar">{[['result','发现与边界'],['atlas','3D与原生响应'],['effects','干预与能力'],['evidence','反例与来源']].map(([k,v])=><button key={k} className="fw-tbtn" aria-pressed={tab===k} onClick={()=>setTab(k)}>{v}</button>)}</div>
    {tab==='result'&&<>
      {doc.conclusions?.map(s=><p key={s}>{s}</p>)}
      <div className="v6-toolbar">{Object.entries(names).map(([k,v])=><button className="fw-tbtn" key={k} onClick={()=>setPair(k)} aria-pressed={pair===k}>{v}</button>)}</div>
      <p>共同输入模型的留出误差 / 简单基线：{n(o.common_input_prediction.mse_ratio)}；胜出 {o.common_input_prediction.family_wins}/{doc.families} 个家族；{o.common_input_prediction.passed?'通过冻结响应预测门':'未通过冻结响应预测门'}。</p>
      <p>原始相关 {n(o.pearson)}；扣除共同输入预测后的残差相关 {n(o.residual_correlation)}。相关减少本身不能证明共同原因，需结合干预页。</p>
      <details><summary>长度、重复程度与时态的全部交叉条件</summary><table className="v6-table"><thead><tr><th>长度 / 背景 / 时态</th><th>出现强响应的文本</th><th>每篇事件数</th><th>前600位置事件数</th><th>R4能量中位数</th></tr></thead><tbody>{doc.triggers.map(r=><tr key={`${r.long}${r.repetition}${r.past}`}><td>{r.tokens_range.join('–')} / {r.repetition?'重复':'多样'} / {r.past?'was':'is'}</td><td>{r.any_event}/{r.n}</td><td>{n(r.mean_events)}</td><td>{n(r.first600_mean_events)}</td><td>{n(r.R4_energy_median)}</td></tr>)}</tbody></table><p>强响应定义：位置≥64 且 |B33/U1521|&gt;10。长度改变观察机会，必须同时看固定前600位置。</p></details>
      <details><summary>批处理敏感性与完整预测结果</summary><pre className="v8-artifact-json">{JSON.stringify({repeatability:doc.repeatability,observations:doc.observations,trigger_effects:doc.trigger_effects},null,2)}</pre></details>
    </>}
    {tab==='atlas'&&<>
      <p>原生编号与完整参数沿用同一模型快照。下方八层背景来自历史普查；本轮新增证据只覆盖四个标出的 MLP 单元及选定残差坐标。</p>
      <ConditionAtlas doc={doc} initialBlock={33} initialUnit={1521} hideScene={hideScene} historyLabel="历史条件卡 · MLP-CONDITIONAL-PATHS-v2"/>
      {hideScene&&<p>在成果库选择“定位卡与关联”，开启叠加到结构，即可查看本轮残差坐标及 MLP 单元的位置。</p>}
      <label>真实前缀实例 <select aria-label="触发研究实例" value={sampleIndex} onChange={e=>setSampleIndex(Number(e.target.value))}>{doc.examples?.map((v,i)=><option key={v.id} value={i}>{v.id}</option>)}</select></label>
      <Prefix sample={sample}/>
      {sample&&<><p>上游选定位置 {sample.position} · token「{sample.token}」；R4={n(sample.R4[sample.position])}，源 A={n(sample.source_A[sample.position])}，下游 A={n(sample.target_A[sample.position])}。</p>
        <details><summary>完整原生残差、逐位置数值及原文</summary><p>R33[{coordinate}] = {sample.residual[coordinate]}；采集边界为注意力之后、MLP归一化之前。</p><NativeArray values={sample.residual} selected={coordinate} onPick={setCoordinate} label="触发研究完整残差向量"/><pre className="v8-sample-text">{sample.prompt}</pre><pre className="v8-artifact-json">{JSON.stringify(sample,null,2)}</pre></details></>}
    </>}
    {tab==='effects'&&<>
      <div className="v6-toolbar">{Object.entries(names).map(([k,v])=><button className="fw-tbtn" key={k} onClick={()=>setPair(k)}>{v}</button>)}</div>
      <p>剂量表示将选定上游坐标缩减至零的比例。先将预测写入文件，再运行干预；随机对照匹配实际 BF16 扰动范数，容差1%。</p>
      <DoseCurve rows={curves}/>
      <table className="v6-table"><thead><tr><th>操作 / 剂量</th><th>目标变化 / 对照</th><th>有效对照</th><th>答案切换</th><th>首词改变</th><th>预测误差比</th></tr></thead><tbody>{curves.map(c=><tr key={`${c.mode}${c.dose}`}><td>{mode[c.mode]} / {c.dose}</td><td>{n(c.target_delta_mean)} / {n(c.control_delta_mean)}</td><td>{c.valid_controls}/{c.n}</td><td>{c.answer_switches}</td><td>{c.first_token_changes}</td><td>{n(c.forecast_mse_vs_nochange)}</td></tr>)}</tbody></table>
      {curves.filter(c=>c.mode==='upstream'&&c.dose===1).map(c=><p key={c.pair}>阻断单个源单元后，目标效应绝对量保留比 {n(c.unit_clamp_retained_l1)}；阻断整层 MLP 写出后 {n(c.mlp_clamp_retained_l1)}。大于1表示阻断后效应反而增大，不能解释为“超过100%的机制已破解”。</p>)}
      <p>{doc.restoration_checks}/{doc.families} 个家族取消编辑后逐位恢复完整词表 logits；这项检查验证实现正确，不替代机制救援。</p>
      <table className="v6-table"><thead><tr><th>能力检查</th><th>原对改错 / 原错改对</th><th>损失上界</th><th>2%门</th></tr></thead><tbody>{Object.entries(doc.capability).map(([k,v])=><tr key={k}><td>{k} · {v.n}题</td><td>{v.lost} / {v.gained}</td><td>{n(v.conservative_loss_upper)}</td><td>{v.passed?'通过':'未通过'}</td></tr>)}</tbody></table>
      <p className="v8-note">能力检查只覆盖历史选择题的最终位置编辑；不覆盖前缀峰值编辑后的开放生成，也不是通用能力保证。</p>
      <details><summary>完整剂量曲线、置信区间与副作用</summary><pre className="v8-artifact-json">{JSON.stringify({curves,capability:doc.capability},null,2)}</pre></details>
    </>}
    {tab==='evidence'&&<>
      {doc.counterexamples.map((c,i)=><details key={i}><summary>{names[c.pair]} · 家族{c.family} · {c.type==='prediction_error'?'响应预测误差':'干预预测误差'}</summary>{c.row&&<pre className="v8-sample-text">{c.row.prompt}</pre>}<pre className="v8-artifact-json">{JSON.stringify(c,null,2)}</pre></details>)}
      <details><summary>适用范围与未解决问题</summary>{doc.limits.map(s=><p key={s}>{s}</p>)}</details>
      {doc.native_math&&<details><summary>真实权重的代数分解（与预注册验收分开）</summary><pre className="v8-artifact-json">{JSON.stringify(doc.native_math,null,2)}</pre></details>}
      {doc.relative_read&&<details><summary>归一化与读入坐标的分工</summary><p>这是原生参数的描述性近似；数值误差不计为独立机制确认。</p><pre className="v8-artifact-json">{JSON.stringify(doc.relative_read,null,2)}</pre></details>}
      <p>模型快照 {doc.model_snapshot}；运行 {doc.run_id}。下列链接从当前服务器按内容哈希读取。</p>
      {Object.entries(doc.artifacts||{}).map(([label,sha])=><p key={label}><a href={apiUrl(`/api/research-library/artifacts/${sha}`)} target="_blank" rel="noreferrer">{label}</a></p>)}
    </>}
  </section>;
}
