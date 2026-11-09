import {useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';
import {TailScene} from './MlpTailAtlas.jsx';
import {ConditionDetails} from './MlpConditionCard.jsx';
import {NativeArray} from './MechanismScene.jsx';
const num=v=>Number.isFinite(v)?v.toPrecision(4):'未测';
const pct=v=>Number.isFinite(v)?`${(100*v).toFixed(1)}%`:'未测';
const pairs={broad:[29,17,30,2],rare:[33,1521,34,6097]};
const name={broad:'广泛关联候选',rare:'稀有强响应候选'};

function ConditionAtlas({doc}) {
  const path=b=>doc.condition_layers?.[b]?`/api/research-library/artifacts/${doc.condition_layers[b]}`:null;
  const b28=useApi(path(28)),b29=useApi(path(29)),b30=useApi(path(30)),b31=useApi(path(31)),
    b32=useApi(path(32)),b33=useApi(path(33)),b34=useApi(path(34)),b35=useApi(path(35));
  const parameter=useApi(doc.native_parameter_sha256?`/api/research-library/artifacts/${doc.native_parameter_sha256}`:null);
  const [block,setBlock]=useState(29),[unit,setUnit]=useState(17),[metric,setMetric]=useState('within_token_variance_fraction'),[coordinate,setCoordinate]=useState(0);
  const all=[b28,b29,b30,b31,b32,b33,b34,b35],blocks=[28,29,30,31,32,33,34,35];
  const layers=Object.fromEntries(blocks.map((b,i)=>[b,all[i].data]));
  const error=all.find(r=>r.error)?.error;
  const ready=blocks.every(b=>layers[b]?.model_snapshot===doc.model_snapshot&&layers[b]?.block===b);
  const chosen=Object.entries(pairs).find(([,p])=>(p[0]===block&&p[1]===unit)||(p[2]===block&&p[3]===unit));
  let relation=null;
  if(chosen&&Number.isFinite(doc.observations[chosen[0]].pearson)){
    const [src,u,dst,v]=chosen[1],incoming=block===dst,values=Array(9728).fill(null);
    values[incoming?u:v]=doc.observations[chosen[0]].pearson;
    relation={source:src,target:dst,direction:incoming?'incoming':'outgoing',values};
  }
  const weights=parameter.data?.parameters?.[`B${block}/U${unit}`];
  return <>
    <p>服务器条件响应图谱 · 全部 77,824 个原生编号 · 新受控实验涉及 4 个单元。板间距离仅用于排布。</p>
    <div className="v6-toolbar"><label>层 <select aria-label="条件图谱层" value={block} onChange={e=>setBlock(Number(e.target.value))}>{blocks.map(b=><option key={b} value={b}>B{b}</option>)}</select></label>
      <label>单元 <input aria-label="条件图谱单元" type="number" min="0" max="9727" value={unit} onChange={e=>setUnit(Math.max(0,Math.min(9727,Number(e.target.value)||0)))}/></label>
      <label>观察指标 <select aria-label="条件图谱指标" value={metric} onChange={e=>setMetric(e.target.value)}>{[['within_token_variance_fraction','同token内变化'],['sign_mix','正负写入并存'],['token_predictor_fvu','词身份预测误差'],['double_negative_energy','负门控与负value']].map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label>
      {Object.entries(pairs).map(([k,p])=><button key={k} className="fw-tbtn" onClick={()=>{setBlock(p[0]);setUnit(p[1])}}>{name[k]}</button>)}</div>
    {error?<p>服务器图谱读取失败：{error}</p>:ready?<TailScene blocks={blocks} layers={layers} metric={metric} block={block} unit={unit} relation={relation} allEdges={false} onPick={(b,u)=>{setBlock(b);setUnit(u)}}/>:<p>正在读取服务器 {all.filter(r=>r.data).length}/8 层条件记录…</p>}
    <p className="v8-note">连线只显示本轮已测单元对的共同响应；其余关系为未测，不填零。预测和因果验收见相应页签。</p>
    {ready&&<ConditionDetails data={layers[block]} block={block} unit={unit}/>}
    <details><summary>完整原生参数与复算定位</summary>{weights?Object.entries(weights).map(([k,v])=><div key={k}><p>{k} · 坐标 {coordinate} = {v.values[coordinate]}</p><NativeArray values={v.values} selected={coordinate} onPick={setCoordinate} label={`条件卡完整${k}参数`}/><details><summary>全部2560个BF16原值与来源哈希</summary><pre className="v8-artifact-json">{JSON.stringify(v)}</pre></details></div>):<p>本轮包携带4个受控单元的完整参数。当前单元的全参数仍从原生尾层档案查询：<a href={`?lens=spatial&view=graph&gview=internal&run=MLP-TAIL-B28-B35-ATLAS-v2&snapshot=${doc.model_snapshot}&tail_block=${block}&tail_unit=${unit}`}>打开原生档案</a>；原档案未部署时会明确显示缺件。</p>}</details>
  </>;
}

function Curve({rows}) {
  const finite=rows.filter(r=>Number.isFinite(r.target_delta_mean));
  const extent=Math.max(.000001,...finite.flatMap(r=>[Math.abs(r.target_delta_mean),Math.abs(r.control_delta_mean)]));
  const x=v=>65+v/2*550,y=v=>105-v/extent*75;
  return <svg viewBox="0 0 680 220" className="v8-dose-chart" role="img" aria-label="条件路径原生写入与等范数对照剂量曲线">
    <line x1="65" y1="105" x2="615" y2="105" stroke="#9aa8b8"/>
    <text x="8" y="30">{num(extent)}</text><text x="8" y="110">0</text><text x="8" y="180">{num(-extent)}</text>
    {[['target_delta_mean','#3279bd'],['control_delta_mean','#de793c']].map(([key,color])=><polyline key={key} fill="none" stroke={color} strokeWidth="2" points={finite.map(r=>`${x(r.dose)},${y(r[key])}`).join(' ')}/>)}
    {finite.map(r=><text key={r.dose} x={x(r.dose)} y="205">{r.dose}×</text>)}
  </svg>;
}

export default function MlpConditionReport({doc}) {
  const [tab,setTab]=useState('result'),[family,setFamily]=useState('broad'),[page,setPage]=useState(0);
  const obs=doc.observations[family],rows=doc.dose_curve.filter(r=>r.pair===family);
  const errors=doc.counterexamples.filter(r=>r.pair===family);
  const [src,u,dst,v]=pairs[family];
  const link=(b,i)=>`?lens=spatial&view=graph&gview=internal&run=MLP-TAIL-B28-B35-ATLAS-v2&snapshot=${doc.model_snapshot}&tail_block=${b}&tail_unit=${i}`;
  return <section aria-label="MLP条件响应与预测验证报告">
    <p><strong>条件响应与跨层预测</strong> · Qwen3-4B BF16 · 512 个新文本家族 × 16 种配平条件</p>
    <p className="v8-note">词身份、角色、证据位置和答案映射独立交叉。预测器先在发现集冻结，再执行留出与原模型干预。材料是四类角色任务的人工长文本；数字门通过仍需审计，不自动晋级为语义机制。</p>
    <div className="v6-toolbar">{Object.keys(pairs).map(k=><button className="fw-tbtn" key={k} aria-pressed={family===k} onClick={()=>{setFamily(k);setPage(0)}}>{name[k]}</button>)}</div>
    <p><a href={link(src,u)}>B{src}/U{u} 条件卡与参数</a> — <a href={link(dst,v)}>B{dst}/U{v} 条件卡与参数</a></p>
    <div className="v6-toolbar">{[['result','预测与条件'],['atlas','3D条件图谱'],['dose','干预与能力'],['errors',`反例（${errors.length}）`],['evidence','材料与证据']].map(([k,label])=><button className="fw-tbtn" key={k} aria-pressed={tab===k} onClick={()=>setTab(k)}>{label}</button>)}</div>
    {tab==='atlas'&&<ConditionAtlas doc={doc}/>}
    {tab==='result'&&<>
      <p>条件预测：<strong>{obs.conditional.passed?'通过冻结判据':'未通过冻结判据'}</strong>。相对条件基线的均方误差 {num(obs.conditional.relative_mse)}，在 {obs.conditional.family_wins}/{obs.conditional.families} 个家族上更好。</p>
      <p className="v8-note">预测使用同位置的源单元响应，不是只凭语言图谱独立预测整个模型。低于 1 表示优于基线。</p>
      <p>稀有条件：{obs.rare.positions} 个位置、{obs.rare.families} 个家族。{obs.rare.powered?'达到预注册分层样本门。':'未达到 128 家族分层门，不将未检出解释为不存在。'}</p>
      <table className="v6-table"><thead><tr><th>统计视角</th><th>数值</th></tr></thead><tbody><tr><td>Pearson</td><td>{num(obs.pearson)}</td></tr><tr><td>排序关联</td><td>{num(obs.spearman)}</td></tr><tr><td>削弱两端极值后的关联</td><td>{num(obs.winsorized)}</td></tr></tbody></table>
      <details><summary>各因素独立变化的响应</summary><pre className="v8-artifact-json">{JSON.stringify(obs.factor_contrasts,null,2)}</pre></details>
      <p>各新任务原模型正确率：{Object.entries(doc.behavior_by_frame).map(([k,v])=>`${k} ${pct(v)}`).join('；')}。</p>
    </>}
    {tab==='dose'&&<><Curve rows={rows}/><p className="v8-note">蓝：实际目标单元变化；橙：严格等范数对照。纵轴是有符号 A 变化，连线只连接实测剂量。完整取消编辑后恢复 {doc.restored_families}/512 个家族。</p>
      <table className="v6-table"><thead><tr><th>剂量</th><th>有效等范数</th><th>答案变化 / 对照</th><th>目标预测误差 / 不变基线</th><th>输出分布预测 KL / 不变基线</th></tr></thead><tbody>{rows.map(r=><tr key={r.dose}><td>{r.dose}×</td><td>{r.n}/512</td><td>{pct(r.switch.edited_rate)} / {pct(r.switch.control_rate)}</td><td>{num(r.target_prediction_mse)} / {num(r.target_nochange_mse)}</td><td>{num(r.distribution_prediction_KL)} / {num(r.distribution_nochange_KL)}</td></tr>)}</tbody></table>
      <p className="v8-note">干预预测以原始内部状态和输出为已知条件，不能算独立下一词预测。低余量与普通余量样本、救援与假恢复另行保留。</p>
      <details><summary>低余量分层、路径恢复及全部剂量</summary><pre className="v8-artifact-json">{JSON.stringify(rows,null,2)}</pre></details>
      <table className="v6-table"><thead><tr><th>能力保持编辑</th><th>原对改错</th><th>损失上界</th><th>2% 门</th></tr></thead><tbody>{Object.entries(doc.capability).map(([k,v])=><tr key={k}><td>{k}</td><td>{v.lost}/{v.n}</td><td>{pct(v.loss_upper)}</td><td>{v.passed?'通过':'未通过'}</td></tr>)}</tbody></table><p className="v8-note">能力保持只检验 1× 剂量、两个源单元的正负写入。已有题库共 2,200 题；不代表 2× 剂量或所有自由生成任务同样安全。</p></>}
    {tab==='errors'&&<><p>保留预测失败和未产生预期行为的样本，全部样本仍可从报告对象回查。</p>
      {doc.prediction_counterexamples.filter(r=>r.pair===family).map(r=><details key={r.family}><summary>预测反例 · 家族 {r.family} · 误差 {num(r.conditional_mse)} / 基线 {num(r.baseline_mse)}</summary><pre className="v8-sample-text">{r.prompt}</pre></details>)}
      {errors.slice(page*8,page*8+8).map(r=><details key={r.sample_id}><summary>{r.sample_id} · {r.reason==='norm mismatch'?'范数匹配失败':'没有预期答案变化'}</summary><pre className="v8-sample-text">{doc.counterexample_prompts[r.sample_id]}</pre><pre className="v8-artifact-json">{JSON.stringify(r.record,null,2)}</pre></details>)}
      <div className="v6-toolbar"><button disabled={!page} onClick={()=>setPage(page-1)}>上一页</button><span>{page+1} / {Math.max(1,Math.ceil(errors.length/8))}</span><button disabled={(page+1)*8>=errors.length} onClick={()=>setPage(page+1)}>下一页</button></div></>}
    {tab==='evidence'&&<><p className="v8-note">{doc.scope}</p><details><summary>适用范围、协议差异与未完成的复现</summary><p>冻结文字提到句式指示变量，实际冻结基线未包含该变量。结果按实际算法报告，本轮不据此晋级正式语义机制；独立节点复算待完成。</p><ul>{doc.interpretation_limits?.map(s=><li key={s}>{s}</li>)}</ul></details><details><summary>冻结材料、功效、代码及模型身份</summary><pre className="v8-artifact-json">{JSON.stringify(doc.preregistration,null,2)}</pre></details><details><summary>原始产物清单与预测承诺</summary><pre className="v8-artifact-json">{JSON.stringify({plan:doc.plan_sha256,predictor:doc.predictor_sha256,control_calibration:doc.control_calibration_sha256,capability_power:doc.capability_power_sha256,prior_failure:doc.prior_failure_sha256,manifest:doc.raw_manifest},null,2)}</pre></details></>}
  </section>;
}
