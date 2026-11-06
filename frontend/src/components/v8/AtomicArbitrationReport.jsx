import {useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';
const pct=v=>Number.isFinite(v)?`${(v*100).toFixed(2)}%`:'—';
const num=v=>Number.isFinite(v)?v.toFixed(4):'—';
const label={discovery:'发现集',unseen_combination:'熟悉实体 · 未见组合/句式',unseen_entity:'未见实体',length_panel:'512/2048长度面板',mlp:'候选32个MLP单元',random_mlp:'同块/RMS匹配随机单元',heads:'候选8个Head',random_heads:'同块随机Head'};
function CalibrationVersion({version:v}){
  const [open,setOpen]=useState(false);const data=useApi(open?`/api/research-library/artifacts/${v.artifact_sha256}`:null);
  return <details onToggle={e=>setOpen(e.currentTarget.open)}><summary>v{v.version} · {v.scope} · {v.passed?'通过':'未通过'}</summary><p className="v8-note">{v.cells.map(c=>`F${c.frame}/${c.scope} ${pct(c.acc)}`).join('；')}</p>
    {data.error?<p className="v8-note">{data.error}</p>:data.data?.result.rows.filter(r=>!r.correct).map(r=><details key={r.sample_id}><summary>{r.sample_id} · 应答{r.expected} / 实际{r.predicted}</summary><pre className="v8-sample-text">{data.data.corpus.items.find(i=>i.id===r.sample_id)?.prompt}</pre></details>)}
    <p className="v8-note">校准材料/完整逐条结果：{v.artifact_sha256}；未从旧版本删除失败样本。</p></details>;
}
function DosePlot({rows}){
  const full=rows.find(r=>r.condition==='mlp'&&r.alpha===1);if(!full)return null;
  const baseline=full.accuracy+full.loss.mean,points=[{alpha:0,accuracy:baseline},...rows.filter(r=>r.condition==='mlp')];
  const x=v=>55+v*470,y=v=>180-v*150;
  return <svg viewBox="0 0 600 220" role="img" aria-label="实际MLP消融剂量曲线" style={{width:'100%',maxWidth:650}}>
    {[0,.5,1].map(v=><g key={v}><line x1="55" x2="525" y1={y(v)} y2={y(v)} stroke="#dbe3ed"/><text x="10" y={y(v)+4}>{pct(v)}</text></g>)}
    <polyline fill="none" stroke="#2979bf" strokeWidth="2" points={points.map(p=>`${x(p.alpha)},${y(p.accuracy)}`).join(' ')}/>
    {points.map(p=><g key={p.alpha}><circle cx={x(p.alpha)} cy={y(p.accuracy)} r="4" fill="#2979bf"/><text x={x(p.alpha)} y="202" textAnchor="middle">{p.alpha}</text></g>)}
    {rows.filter(r=>r.condition==='random_mlp').map((p,i)=><circle key={i} cx={x(p.alpha)} cy={y(p.accuracy)} r="5" fill="#dd7843"><title>同块/RMS匹配随机组</title></circle>)}
    <text x="245" y="218">消融强度 α</text><text x="55" y="16">完整整数正确率 · 蓝：候选组 / 橙：随机组</text>
  </svg>;
}
export default function AtomicArbitrationReport({doc:d}){
  const [tab,setTab]=useState('result'),[page,setPage]=useState(0);const a=d.analysis,cap=d.capability;
  return <section aria-label="原子序数仲裁与未见组合真实报告"><p><strong>{a.behavior_gate?'有参考双范围行为门通过':'有参考双范围行为门未通过'} · {a.H2_supported?'MLP组效应达到预注册标准':'MLP组效应未达预注册标准'}</strong></p>
    <p className="v8-note">{d.run_id} · 本地Qwen3-4B / CUDA BF16。两范围都提供真实参考卡和冲突虚构记录；这是来源仲裁。无参考现实问答的校准失败另保留，不能将本轮通过解释为内部知识记忆已破解。</p>
    <div className="v8-library-stats"><span><b>{d.n_inputs}</b>完整整数自然生成</span><span><b>64 + 32</b>熟悉实体 / 新实体</span><span><b>{d.native.files.toLocaleString()}</b>原生文件逐一哈希核对</span></div>
    <div className="v6-toolbar">{[['result','行为门与路径'],['prediction','未见组合预测'],['trace','3D与完整参数'],['errors',`反例（${d.counterexamples.length}）`],['mlp','MLP主线方案'],['source','预注册与来源']].map(([k,t])=><button key={k} className="fw-tbtn" aria-pressed={tab===k} onClick={()=>setTab(k)}>{t}</button>)}</div>
    {tab==='result'&&<><table className="v6-table"><thead><tr><th>划分</th><th>范围</th><th>题数</th><th>完整整数正确率</th></tr></thead><tbody>{a.cell_statistics.map((r,i)=><tr key={i}><td>{label[r.split]}</td><td>{r.scope==='现实'?'现实（有参考）':r.scope}</td><td>{r.n}</td><td>{pct(r.accuracy)}</td></tr>)}</tbody></table><details><summary>完整保留的前三版校准失败</summary>{d.calibration_versions.map(v=><CalibrationVersion key={v.version} version={v}/>)}</details>
      <h3>真实激活消融与剂量</h3><p className="v8-note">α=0为原模型；α=1将所选单元在每个实际生成query处归零，权重不改。选择只用发现集；单元按首位数字的直接贡献选出，尚不能称“仲裁专属神经元”。</p>
      <DosePlot rows={a.dose_curve}/><table className="v6-table"><thead><tr><th>操作 / α</th><th>整数正确率</th><th>下降（百分点）</th><th>首步间隔衰减</th></tr></thead><tbody>{a.dose_curve.map((r,i)=><tr key={i}><td>{label[r.condition]} / {r.alpha}</td><td>{pct(r.accuracy)}</td><td>{(r.loss.mean*100).toFixed(2)}</td><td>{num(r.gap_attenuation.mean)} nats</td></tr>)}</tbody></table>
      {d.selection?.random_unit_rms_ratios&&<p className="v8-note">随机单元按同块/RMS近邻抽样，实际RMS比范围 [{num(Math.min(...d.selection.random_unit_rms_ratios))}, {num(Math.max(...d.selection.random_unit_rms_ratios))}]，并非严格等能量匹配。零观察错误的bootstrap区间会退化，不能证明总体效应严格为零。</p>}
      <p className="v8-note">能力保持：新操作重新实测 {cap.n} 道正式题，净损失 {(100*cap.net_loss).toFixed(2)} 个百分点，区间 [{cap.loss_interval.map(v=>(v*100).toFixed(2)).join(', ')}]；{cap.status==='noninferior'?'通过2个百分点非劣效门':cap.status==='inferior'?'超过允许损失':'尚未证实非劣效'}。重复题池，140参考题不计正式门；不代表全部能力分别保持。</p></>}
    {tab==='result'&&d.numeric_control&&<p className="v8-note">额外无冲突数值对照：原模型 {pct(d.numeric_control.accuracy.base)}，候选MLP组消融后 {pct(d.numeric_control.accuracy.mlp32)}，随机组 {pct(d.numeric_control.accuracy.random32)}。如果无冲突任务也受损，候选组可能支持数值答案而非仲裁专用开关。该对照在自身执行前登记，长度最多差1token，不更改主判。</p>}
    {tab==='prediction'&&<><p className="v8-note">使用256个熟悉实体的新数值组合/新句式。预测时只读取已学习的记忆节点、显式输入数值和范围，不调用原模型，不读取测试hidden/logits。真实输出仅用于事后评分。新实体行为门单列，不宣称图谱对新实体知识泛化。</p>
      <p><strong>完整词表下一token预测：{a.output_prediction.supported?'达到冻结预测判据':'未达到冻结预测判据'}</strong></p><p>图谱KL {num(a.output_prediction.graph_KL)}，范围盲基线KL {num(a.output_prediction.scope_blind_KL)}，相对降低 {pct(a.output_prediction.relative_reduction)}；top-1一致率 {pct(a.output_prediction.top1_agreement)}。</p>
      <p className="v8-note">该输出器是训练得到的条件混合代理，不是完整机制电路，也不预测完整整数的后续所有token。</p>
      {d.strong_baselines&&<p className="v8-note"><strong>事后强基线审查：</strong>直接按输入中的来源规则选择数字、使用发现集数字原型时，KL为 {num(d.strong_baselines.rule_KL)}；图谱混合器 {d.strong_baselines.graph_better_than_rule?'在本次事后比较中更好':'未证实优于这个强规则基线'}。这项检查不修改冻结判据；对弱基线的提升不能直接当作机制破解。</p>}
      {a.internal_prediction.map(r=><p key={r.space}>B{a.profile_block} · {r.dimensions}维{r.space==='mlp'?'完整MLP阵列':'HiddenState'}：相对实体×范围均值基线的标准化MSE降低 {pct(r.relative_reduction)}；{r.supported?'达到冻结预测判据':'未达到冻结预测判据'}。</p>)}<details><summary>预测公式、逐例KL与误差</summary><pre className="v8-artifact-json">{JSON.stringify({predictor:d.predictor,metrics:a.output_prediction,internal:a.internal_prediction},null,2)}</pre></details></>}
    {tab==='trace'&&<><p className="v8-note">原生坐标不做TopK截断；37层隐态、36块完整Head与MLP在声明读出位置保留。8例另存全token隐态/QKV和真实生成步，参数仍来自同一完整快照。</p>{d.examples.map(e=><p key={e.trace_id}><a href={e.url}>{e.entity} · {e.scope} · {e.length} tokens → 真实3D计算路径</a></p>)}<p className="v8-note">原场 {(d.native.bytes/2**30).toFixed(2)} GiB保存在本机；3D布局距离不是语义几何。</p></>}
    {tab==='errors'&&<><p className="v8-note">主任务错误、干预退化、预测偏差和能力退化都保留；不事后排除困难实体。</p>{d.counterexamples.slice(page*20,(page+1)*20).map((r,i)=><details key={i}><summary>{r.sample_id} · {r.reason}</summary><pre className="v8-artifact-json">{JSON.stringify(r,null,2)}</pre>{r.trace_id&&<a href={`?lens=spatial&view=live&trace=${r.trace_id}`}>回查原生响应</a>}</details>)}<button className="fw-tbtn" disabled={!page} onClick={()=>setPage(p=>p-1)}>上一页</button><span>{page+1}/{Math.max(1,Math.ceil(d.counterexamples.length/20))}</span><button className="fw-tbtn" disabled={(page+1)*20>=d.counterexamples.length} onClick={()=>setPage(p=>p+1)}>下一页</button></>}
    {tab==='mlp'&&<><p><strong>可以将原生MLP神经元作为近期主线；不能保证先把全部MLP破解，再研究其余模块。</strong></p>{d.mlp_plan.map((r,i)=><section key={i}><h3>{i+1}. {r.title}</h3><p className="v8-note">{r.detail}</p></section>)}<p className="v8-note">方案受Goodfire启发，其可解释性比例属于特定模型的初步审计。项目已有两单元工具检验与自然语料普查，尚不是全部350,208单元的机制解释。</p><a href="https://already-clean.goodfire.com/" target="_blank" rel="noreferrer">Goodfire原文</a></>}
    {tab==='source'&&<>{d.limitations.map((r,i)=><p className="v8-note" key={i}>{r}</p>)}<details><summary>冻结预注册与模型身份</summary><pre className="v8-artifact-json">{JSON.stringify({prereg:d.preregistration,freeze:d.freeze,native:d.native,sources:d.artifact_refs},null,2)}</pre></details><p><a href={d.reference_source} target="_blank" rel="noreferrer">NIH PubChem参考元素表</a></p></>}
  </section>;
}
