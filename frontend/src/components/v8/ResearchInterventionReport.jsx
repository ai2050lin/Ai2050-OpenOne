import { useState } from 'react';
import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { ExtraPackPrompt } from '../open/DataBar.jsx';
import NaturalRegisterReport from './NaturalRegisterReport.jsx';
import ExternalAiRegisterReport from './ExternalAiRegisterReport.jsx';
import EntityArbitrationReport from './EntityArbitrationReport.jsx';
import AtomicArbitrationReport from './AtomicArbitrationReport.jsx';
import NativeMlpReport from './NativeMlpReport.jsx';
import NativeFunctionReport from './NativeFunctionReport.jsx';
import MlpInventoryReport from './MlpInventoryReport.jsx';
import MlpCodeStudyReport from './MlpCodeStudyReport.jsx';
import MlpDictionaryReport from './MlpDictionaryReport.jsx';
import MlpParameterAuditReport from './MlpParameterAuditReport.jsx';
import MlpFidelityReport from './MlpFidelityReport.jsx';
import MlpFunctionalReport from './MlpFunctionalReport.jsx';
import MlpKvMediationReport from './MlpKvMediationReport.jsx';
import MlpEarlyReadReport from './MlpEarlyReadReport.jsx';
import MlpFactorSteerReport from './MlpFactorSteerReport.jsx';
import B35RecursiveReport from './B35RecursiveReport.jsx';
import AttributePrefixTransferReport from './AttributePrefixTransferReport.jsx';
import MainlineReport from './MainlineReport.jsx';
import MlpTailAtlas from './MlpTailAtlas.jsx';
import MlpB34ConfirmReport from './MlpB34ConfirmReport.jsx';
import MlpConditionReport from './MlpConditionReport.jsx';

const NI={noninferior:'通过非劣效门',inferior:'超过允许损失',inconclusive:'能力保持尚未证实',not_tested:'未测试'};
const pct=x=>Number.isFinite(x)?`${(100*x).toFixed(2)}%`:'—';
const num=x=>Number.isFinite(x)?x.toFixed(3):'—';
const points=x=>Number.isFinite(x)?(100*x).toFixed(2):'—';
const REASON={'keyed capability regression':'提供答案口径：原对改错','generated literal-content proxy failed':'生成内容字面代理未通过','paired attenuation below preregistered 0.5 nats':'成对条件差衰减未达 0.5 nats','no desired switch or nonpositive dose effect':'未发生目标翻转或效应方向不符'};

function DoseCurve({curve,description}){
  const rows=curve?.rows||[],ks=[...new Set(rows.map(r=>r.k))].sort((a,b)=>a-b);
  const finite=rows.filter(r=>Number.isFinite(r.alpha)&&Number.isFinite(r.shift));
  if(!finite.length)return <Empty>尚无真实剂量曲线。</Empty>;
  const W=720,H=260,L=65,R=20,T=20,B=45;
  const xmin=Math.min(0,...finite.map(r=>r.alpha)),xmax=Math.max(1,...finite.map(r=>r.alpha));
  const ymin=Math.min(0,...finite.map(r=>r.shift)),ymax=Math.max(.001,...finite.map(r=>r.shift));
  const x=v=>L+(v-xmin)/(xmax-xmin)*(W-L-R),y=v=>H-B-(v-ymin)/(ymax-ymin||1)*(H-T-B);
  const colors=['#99bce5','#5e94d0','#3273b8','#16477b'];
  return <><p className="v8-note">{description||'确认集 · 固定续写候选的对数优势变化（nats）。'}每条线对应每层编辑列数 k；不是能力提升曲线。α=1 移除所选投影，α&gt;1 改变其符号与强度。</p>
    <svg className="v8-dose-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="真实剂量响应曲线">
      {[0,.25,.5,.75,1].map(t=>{const v=ymin+t*(ymax-ymin);return <g key={t}><line x1={L} x2={W-R} y1={y(v)} y2={y(v)} stroke="#e2e8f0"/><text x={L-8} y={y(v)+4} textAnchor="end">{v.toFixed(2)}</text></g>})}
      <line x1={L} x2={W-R} y1={y(0)} y2={y(0)} stroke="#94a3b8" strokeDasharray="4 3"/>
      {[-1,0,1,2,4,6].filter(v=>v>=xmin&&v<=xmax).map(v=><text key={v} x={x(v)} y={H-23} textAnchor="middle">{v}</text>)}
      <text x={(L+W-R)/2} y={H-4} textAnchor="middle">编辑强度 α</text>
      {ks.map((k,i)=>{const points=[...finite.filter(r=>r.k===k),{alpha:0,shift:0}].sort((a,b)=>a.alpha-b.alpha);return <g key={k}><polyline fill="none" stroke={colors[i%colors.length]} strokeWidth="2" points={points.map(p=>`${x(p.alpha)},${y(p.shift)}`).join(' ')}/>{points.map(p=><circle key={p.alpha} cx={x(p.alpha)} cy={y(p.shift)} r="3" fill={colors[i%colors.length]}><title>{`k=${k}, α=${p.alpha}, Δ=${num(p.shift)} nats`}</title></circle>)}</g>})}
    </svg><div className="v6-toolbar">{ks.map((k,i)=><span key={k} style={{color:colors[i%colors.length]}}>● k={k}</span>)}</div>
    <details><summary>全部剂量与编辑范围</summary><table className="v6-table"><thead><tr><th>k / α</th><th>效应（nats）</th><th>指标达标比例</th><th>操作覆盖参数</th><th>最大局部相对范数</th></tr></thead><tbody>{rows.map(r=><tr key={r.id}><td>{r.k} / {r.alpha}</td><td>{num(r.shift)}</td><td>{pct(r.flip_fraction)}</td><td>{r.addressed_params?.toLocaleString()}</td><td>{pct(r.max_relative_frobenius)}</td></tr>)}</tbody></table></details>
  </>;
}

export default function ResearchInterventionReport({sha,planSha,capabilityInputsSha,rawFallback=false}){
  const report=useApi(sha?`/api/research-library/artifacts/${sha}`:null);
  const [tab,setTab]=useState('result'),[page,setPage]=useState(0);
  if(!sha)return null;
  if(report.error)return <Empty>{report.error}<ExtraPackPrompt pack={report.missingPack}/></Empty>;
  if(!report.data)return <Empty>读取研究闭环报告…</Empty>;
  const d=report.data;
  if(d.schema==='mlp_conditional_report.v1')return <MlpConditionReport doc={d}/>;
  if(d.schema==='mlp_tail_atlas_report.v1')return <MlpTailAtlas/>;
  if(['mlp_b34_confirm_report.v1','mlp_b31_authority_report.v1'].includes(d.schema))return <MlpB34ConfirmReport doc={d}/>;
  if(d.schema==='mainline_run_report.v1')return <MainlineReport doc={d}/>;
  if(d.schema==='research_native_function_report.v1')return <NativeFunctionReport doc={d}/>;
  if(d.schema==='mlp_reverse_inventory_report.v1')return <MlpInventoryReport doc={d}/>;
  if(d.schema==='mlp_atlas_code_report.v1')return <MlpCodeStudyReport doc={d}/>;
  if(d.schema==='mlp_dictionary_atlas_report.v1')return <MlpDictionaryReport doc={d}/>;
  if(d.schema==='mlp_parameter_audit_report.v1')return <MlpParameterAuditReport doc={d}/>;
  if(d.schema==='mlp_fidelity_report.v2')return <MlpFidelityReport doc={d}/>;
  if(d.schema==='mlp_functional_report.v1')return <MlpFunctionalReport doc={d}/>;
  if(d.schema==='mlp_kv_mediation_report.v1')return <MlpKvMediationReport doc={d}/>;
  if(d.schema==='mlp_early_read_route_report.v1')return <MlpEarlyReadReport doc={d}/>;
  if(d.schema==='mlp_factor_steer_report.v1')return <MlpFactorSteerReport doc={d}/>;
  if(d.schema==='b35_recursive_report.v1')return <B35RecursiveReport doc={d}/>;
  if(d.schema==='attribute_prefix_transfer_report.v1')return <AttributePrefixTransferReport doc={d}/>;
  if(d.schema==='research_native_mlp_long_report.v1')return <NativeMlpReport doc={d} planSha={planSha} capabilityInputsSha={capabilityInputsSha}/>;
  if(d.schema==='research_atomic_arbitration_report.v1')return <AtomicArbitrationReport doc={d}/>;
  if(d.schema==='research_entity_arbitration_report.v1')return <EntityArbitrationReport doc={d}/>;
  if(d.schema==='research_external_ai_register_report.v1')return <ExternalAiRegisterReport doc={d}/>;
  if(d.schema==='research_natural_register_report.v1')return <NaturalRegisterReport doc={d}/>;
  if(d.schema!=='research_intervention_report.v1')return rawFallback?<details><summary>归档研究报告 · {d.schema||'通用格式'}（完整原始记录）</summary><pre className="v8-artifact-json">{JSON.stringify(d,null,2)}</pre></details>:null;
  const r=d.result,cap=r.capability,errors=r.counterexamples||[];
  return <section className="v8-intervention-report" aria-label="真实权重编辑研究报告">
    <p className="v8-note">{r.run_id} · CUDA {r.env?.precision} · 快照 {r.model_snapshot.slice(0,16)}</p>
    <p><strong>{d.primary_name||'固定续写候选'}：{r.target_supported_in_scope?'支持范围内干预效应':'未通过预注册主判'}</strong> · {NI[cap.status]||cap.status}</p>
    <div className="v6-toolbar" role="group" aria-label="研究报告内容">{[['result','结果与能力'],['dose','剂量曲线'],['counterexamples',`反例（${errors.length}）`],['prereg','预注册与原始数据']].map(([k,t])=><button className="fw-tbtn" key={k} aria-pressed={tab===k} onClick={()=>setTab(k)}>{t}</button>)}</div>
    {tab==='result'&&<>
      <div className="v8-library-stats"><span><b>{pct(r.lockbox.target.flip_fraction)}</b>{d.primary_label||'最终留出翻转比例'}</span><span><b>{cap.n.toLocaleString()}</b>能力门正式题数</span><span><b>{points(cap.net_loss)}</b>净正确率损失（百分点）</span></div>
      <p className="v8-note">能力损失置信区间 [{points(cap.loss_interval[0])}, {points(cap.loss_interval[1])}] 个百分点；允许损失 {points(cap.margin)} 个百分点。只有区间上界不超过允许损失才通过。“未显著下降”不能自动通过。</p>
      <p className="v8-note">确认集预先选定主编辑 k={r.selected.k}、α={r.selected.alpha}；最终留出不用于重新选参。紧致度：{r.compactness.status==='observed_upper_bound'?'已找到可行编辑上界':'预算内未定位'}，不是已经证明的最小编辑。</p>
      {r.baseline_paired_gap&&<p className="v8-note">原模型成对语域条件差 {num(r.baseline_paired_gap.mean)} nats；主编辑衰减 {num(r.lockbox.target.shift.mean)} nats，区间 [{r.lockbox.target.shift.ci95.map(num).join(', ')}]。测量门：{r.baseline_readout_gate?'通过':'未通过'}。</p>}
      {d.verification?.generations&&<p className="v8-note">自然生成词表代理：口语标记 {d.verification.generations.base.colloquial_markers} → {d.verification.generations.edited.colloquial_markers}。尚未独立验收自然风格切换；句式和长度仍共变。</p>}
      {d.prior_measurement_audit&&<details><summary>上一版本的测量缺口与修订原因</summary><p className="v8-note">旧绝对分类在口语文本上的正确率为 0；全列编辑时随机列对照与目标相同。旧结果保持负结果，本版本改用事前冻结的成对条件差和稀疏主编辑。</p></details>}
      <details><summary>分数据源与负控制</summary><table className="v6-table"><thead><tr><th>来源</th><th>题数</th><th>原模型</th><th>编辑后</th><th>净损失</th></tr></thead><tbody>{Object.entries(r.capability_sources).map(([name,s])=><tr key={name}><td>{name}{name.includes('M3Exam')?'（仅参考，不入能力门）':''}</td><td>{s.n}</td><td>{pct(s.base_accuracy)}</td><td>{pct(s.edited_accuracy)}</td><td>{pct(s.net_loss)}</td></tr>)}</tbody></table><p className="v8-note">能力门针对冻结混合题池；不声称每个来源分别满足 2 个百分点非劣效。</p>{Object.entries(r.control_comparisons).map(([name,s])=><p className="v8-note" key={name}>相对 {name}：{num(s.mean)} nats，区间 [{s.ci95.map(num).join(', ')}]。</p>)}</details>
      <details><summary>自然生成与内容保持 · 逐条比较</summary><p className="v8-note">这些词表和字面内容指标属于代理观察，不能代替独立风格评审。相同输入的完整原模型／编辑后输出均保留。</p>{d.generations.base.map((b,i)=>{const e=d.generations.edited[i],input=d.inputs?.find(s=>s.id===b.id);return <details key={b.id}><summary>{b.id} · {b.split} · 内容代理 {b.literal_content_ok?'✓':'△'} → {e.literal_content_ok?'✓':'△'}</summary><p className="v8-note">应保留 {e.expected.item}、{e.expected.quantity}；生成 tokens {b.generated_tokens} → {e.generated_tokens}。</p>{input&&<details><summary>实际输入 · {input.tokens} tokens</summary><pre className="v8-sample-text">{input.prompt}</pre></details>}<pre className="v8-sample-text">原模型：{b.text}{'\n\n'}编辑后：{e.text}</pre></details>})}</details>
      <details><summary>范围与当前限制</summary>{d.limitations.map(s=><p className="v8-note" key={s}>{s}</p>)}</details>
    </>}
    {tab==='dose'&&<DoseCurve curve={d.dose_curve} description={d.dose_description}/>}
    {tab==='counterexamples'&&<><p className="v8-note">保留未翻转、能力退化与自然生成内容代理失败；这些记录不自动归因于“分布式容量”。</p>{errors.length?errors.slice(page*20,(page+1)*20).map((e,i)=><details key={page*20+i}><summary>{e.sample_id} · {REASON[e.reason]||e.reason}{e.source?.includes('M3Exam')?'（镜像参考题）':''}</summary><pre className="v8-sample-text">{JSON.stringify({...e,input:d.counterexample_inputs?.[e.sample_id]||d.inputs?.find(s=>s.id===e.sample_id)},null,2)}</pre></details>):<Empty>按预注册规则没有记录到反例；不表示所有语境都成立。</Empty>}<div className="v6-toolbar"><button className="fw-tbtn" disabled={!page} onClick={()=>setPage(page-1)}>上一组反例</button><span>{Math.min(page*20+1,errors.length)}–{Math.min((page+1)*20,errors.length)} / {errors.length}</span><button className="fw-tbtn" disabled={(page+1)*20>=errors.length} onClick={()=>setPage(page+1)}>下一组反例</button></div></>}
    {tab==='prereg'&&<><p className="v8-note">预注册 SHA256 {r.design_sha256}；原生 BF16 以 uint16 位模式无损保留。全场范围为所有层的两个读出位置，不是全部 token。</p><details><summary>冻结预注册</summary><pre className="v8-sample-text">{JSON.stringify(d.preregistration,null,2)}</pre></details><details><summary>原场与实际编辑清单</summary><pre className="v8-sample-text">{JSON.stringify({native:d.raw_manifest,edit:d.edit_manifest},null,2)}</pre></details><details><summary>完整研究报告 · 可回查哈希</summary><pre className="v8-sample-text">{JSON.stringify(d,null,2)}</pre></details></>}
  </section>;
}
