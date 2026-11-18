import {useMemo,useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';
import {apiUrl} from '../../config.js';
import {NativeArray} from './MechanismScene.jsx';
import './hiddenstate_components.css';
import './record_selection.css';

const num=x=>Number.isFinite(x)?x.toPrecision(4):'未检验';
const pct=x=>Number.isFinite(x)?`${(100*x).toFixed(2)}%`:'未检验';
const ci=x=>x?.frame_interval?.map(num).join(' 至 ')||'未检验';
const label=key=>{const [task,alphabet]=key.split(':');return `${{status:'按有效状态',time:'按最新日期',explicit:'按指定编号'}[task]} · ${alphabet==='digits'?'数字1/2':'字母A/B'}`;};
const effectReason=e=>{
  if(e.candidate==='joint')return '联合作用只作交互检验，不能据此确认定向中介链。';
  const reasons=[];
  if(!e.controls_valid)reasons.push('至少一个范数或倍率对照无效，停止该作用主张');
  if(e.mean_loss<.2)reasons.push('未达到0.2 nats的预设最小作用');
  if(e.random_excess?.frame_interval?.[0]<=0)reasons.push('未稳定超过随机等范数方向');
  if(e.native_excess?.frame_interval?.[0]<=0)reasons.push('未稳定超过原生等范数方向');
  return reasons.length?reasons.join('；'):'达到本轮范围内的作用门，尚不证明唯一性、必要性或完整机制。';
};

function Fields({doc}){
  const [branch,setBranch]=useState(doc.active_branches[0]||'status:digits'),[kind,setKind]=useState('hidden'),[layer,setLayer]=useState(0),[split,setSplit]=useState(1),[position,setPosition]=useState(0),[factor,setFactor]=useState(0),[index,setIndex]=useState(0);
  const key=String(kind==='hidden'?layer:kind==='units'?26:29),n=kind==='hidden'?2560:kind==='units'?9728:32,nf=kind==='heads'?1:2;
  const digest=doc.coordinate_artifacts?.[branch]?.[kind]?.[key],res=useApi(digest?`/api/research-library/artifacts/${digest}`:null);
  const values=useMemo(()=>{
    const d=res.data,a=d?.array;
    if(!a||d.run_id!==doc.run_id||d.model_snapshot!==doc.model_snapshot||d.branch!==branch||d.field!==kind||String(d.index)!==key||a.dtype!=='<f4'||a.encoding!=='base64'||a.shape?.join(',')!==`2,${nf},2,${n}`)return null;
    try{const raw=Uint8Array.from(atob(a.data),c=>c.charCodeAt(0));if(raw.length!==2*nf*2*n*4)return null;const v=new DataView(raw.buffer),offset=((split*nf+(kind==='heads'?0:factor))*2+position)*n;return Array.from({length:n},(_,i)=>v.getFloat32((offset+i)*4,true));}catch{return null;}
  },[res.data,doc.run_id,doc.model_snapshot,branch,kind,key,nf,n,split,factor,position]);
  return <>
    <div className="hs-controls">
      <label>任务分支 <select aria-label="来源编码分支" value={branch} onChange={e=>setBranch(e.target.value)}>{Object.keys(doc.behavior.confirm.branches).map(k=><option key={k} value={k}>{label(k)}{doc.active_branches.includes(k)?'':'（停止）'}</option>)}</select></label>
      <label>对象 <select aria-label="来源编码对象" value={kind} onChange={e=>{setKind(e.target.value);setIndex(0);}}><option value="hidden">全HiddenState坐标</option><option value="units">B26全部MLP单元</option><option value="heads">B29全部注意力头</option></select></label>
      {kind==='hidden'&&<label>层 <select aria-label="来源编码层" value={layer} onChange={e=>setLayer(Number(e.target.value))}>{Array.from({length:37},(_,i)=><option key={i} value={i}>{i?`B${i-1}残差`:'输入嵌入'}</option>)}</select></label>}
      <label>数据 <select aria-label="来源编码数据" value={split} onChange={e=>setSplit(Number(e.target.value))}><option value="0">校准集</option><option value="1">独立留出集</option></select></label>
      <label>读出位置 <select aria-label="来源编码位置" value={position} onChange={e=>setPosition(Number(e.target.value))}><option value="0">答案映射出现之前</option><option value="1">答案前缀之后</option></select></label>
      {kind!=='heads'&&<label>成分 <select aria-label="来源编码成分" value={factor} onChange={e=>setFactor(Number(e.target.value))}><option value="0">记录来源对照</option><option value="1">输出符号对照</option></select></label>}
    </div>
    {!doc.active_branches.includes(branch)?<p role="status">这一整支没有通过行为门，未继续机制检验；失败样本保留在“行为与反例”。</p>:!digest?<p role="status">该原生数组尚无可用产物。</p>:res.error?<p role="status">来源编码数组读取失败：{res.error}</p>:!res.data?<p>正在读取原生数组…</p>:!values?<p role="status">数据身份或形状不匹配，未显示坐标。</p>:<>
      <label>原生索引 <input aria-label="来源编码原生索引" type="number" min="0" max={n-1} value={index} onChange={e=>setIndex(Math.min(n-1,Math.max(0,Number(e.target.value)||0)))}/></label>
      <p>原生索引 {index} = {num(values[index])}</p><NativeArray values={values} selected={index} onPick={setIndex} label="来源编码完整原生阵列"/>
    </>}
    <p className="v8-note">按原生顺序显示全部坐标。来源对照为平均 s×激活，输出对照为平均 s×m×激活，s表示应选记录的位置，m表示答案映射方向；两者是平衡条件下的统计成分，不是独立语义模块。注意力头显示“应选来源−另一来源”的质量差。映射出现前，输出对照应严格为零。</p>
  </>;
}

export default function SourceOutputReport({doc}){
  const [tab,setTab]=useState('behavior'),[page,setPage]=useState(0);
  return <section className="hs-components-report record-selection-report" aria-label="记录来源与输出符号报告">
    <h3>是在选择记录，还是在生成答案符号？</h3>
    <p>固定上一轮的 B29 两个注意力头和 B26 的32个MLP单元，用 {doc.unique_inputs.toLocaleString()} 条新输入交叉改变选择规则与答案映射。候选没有重选，留出集没有按单条正确与否筛选。</p>
    <p role="status"><strong>{doc.active_branches.length} / 6 个整支通过校准及留出行为门。</strong> 未通过分支停止机制分析；本报告不升级为完整语言机制或独立复现证据。</p>
    {doc.summary.map((s,i)=><p key={i}>{s}</p>)}
    <div className="v6-toolbar">{[['behavior','行为与反例'],['fields','来源与输出成分'],['prediction','未见样本与基线'],['effects','作用与负对照']].map(([id,text])=><button className="fw-tbtn" key={id} aria-pressed={tab===id} onClick={()=>setTab(id)}>{text}</button>)}</div>
    {tab==='behavior'&&<>
      <p>状态和日期独立交叉；第三类任务直接指定记录编号。选中同一条记录时，答案可以是1、2、A或B。每支必须在全部“记录位置×映射”格及中英文格中，同时达到90%的候选严格正确率和完整词表首词正确率。</p>
      <table className="v6-table" aria-label="来源编码行为门"><thead><tr><th>任务</th><th>校准 n / 严格正确率</th><th>留出 n / 严格正确率</th><th>留出首词正确率</th><th>整支状态</th></tr></thead><tbody>{Object.entries(doc.behavior.confirm.branches).map(([k,v])=><tr key={k}><td>{label(k)}</td><td>{doc.behavior.calibration.branches[k].n} / {pct(doc.behavior.calibration.branches[k].accuracy)}</td><td>{v.n} / {pct(v.accuracy)}</td><td>{pct(v.top1)}</td><td>{doc.active_branches.includes(k)?'通过':'停止'}</td></tr>)}</tbody></table>
      <details><summary>全部逐格结果和失败原因</summary>{Object.entries(doc.behavior).map(([split,d])=><div key={split}><h4>{split==='calibration'?'校准':'留出'}</h4>{Object.entries(d.branches).map(([branch,b])=><details key={branch}><summary>{label(branch)} · {b.passed?'通过':'未通过'}</summary><table className="v6-table"><thead><tr><th>条件</th><th>n</th><th>严格正确率</th><th>首词正确率</th></tr></thead><tbody>{Object.entries(b.cells).map(([key,v])=><tr key={key}><td>{key}</td><td>{v.n}</td><td>{pct(v.accuracy)}</td><td>{pct(v.top1)}</td></tr>)}</tbody></table></details>)}</div>)}</details>
      {doc.failure_diagnostics&&<>
        <h4>首词失败的具体原因</h4>
        <p>下表仅对已保存的首选token做诊断。去除空格后相同的字母，与仍输出记录编号的错误分别计数；这不修改原行为门，也不把停止分支重新纳入机制分析。</p>
        <table className="v6-table" aria-label="来源编码首词诊断"><thead><tr><th>留出任务</th><th>n</th><th>去空格后代码正确</th><th>正确代码带额外空格</th><th>字母任务仍输出编号</th></tr></thead><tbody>{doc.failure_diagnostics.branches.filter(d=>d.split==='confirm').map(d=><tr key={d.branch}><td>{label(d.branch)}</td><td>{d.n}</td><td>{d.counts.correct_after_trimming_whitespace}</td><td>{d.counts.correct_code_with_extra_whitespace}</td><td>{d.counts.record_number_in_letter_task}</td></tr>)}</tbody></table>
      </>}
      <h4>保留的错误、并列和格式失败：{doc.failures.length} 条</h4>
      {doc.failures.slice(page*6,page*6+6).map(f=><details key={f.sample.id}><summary>{f.sample.id} · 正确答案优势 {num(f.score.target_margin)}</summary><pre className="v8-sample-text">{f.sample.prompt}</pre><p>应选记录 {f.sample.target_sign===1?'1':'2'}；答案符号 {f.answer}；实际首词 {JSON.stringify(f.predicted_text)}（token ID {f.score.argmax}）；候选分数 {f.score.pair_logits.map(num).join(' / ')}。{f.failure_type==='candidate_choice'?'候选选择错误或并列。':'候选间选对，但实际首词不满足作答要求。'}</p></details>)}
      <div className="v6-toolbar"><button className="fw-tbtn" disabled={!page} onClick={()=>setPage(page-1)}>上一页反例</button><span>{page+1} / {Math.max(1,Math.ceil(doc.failures.length/6))}</span><button className="fw-tbtn" disabled={(page+1)*6>=doc.failures.length} onClick={()=>setPage(page+1)}>下一页反例</button></div>
    </>}
    {tab==='fields'&&<>
      <p>在答案映射出现前，同一家族的四个映射变体具有逐位相同的前缀；这一点单独校验。能读出来源与能读出输出符号分别报告，探针准确不等于原模型一定使用该读法。</p>
      <Fields doc={doc}/>
      <details><summary>沿用的原生组及负对照索引</summary>{doc.candidates.map(c=><p key={c.id}>{c.id}：{c.indices.join(', ')}。同层对照：{c.controls.slice(0,2).map(x=>x.indices.join(', ')).join(' / ')}。</p>)}</details>
    </>}
    {tab==='prediction'&&<>
      <p>只在校准集拟合，在完整合格留出分支检验。原生8变量读出使用实际激活；256维词袋使用输入token；8格参照使用外部已知任务规则解析文本。后者是人工规则基线，不是发现的内部算法。嵌入层最后token恒定，其基线与常数预测相同。</p>
      {!doc.observations.length&&<p role="status">没有通过行为门的完整分支，未报告内部预测优势。</p>}
      {doc.observations.map(o=><div key={o.branch}><h4>{label(o.branch)} · {o.families} 个留出家族</h4><table className="v6-table" aria-label={`来源编码基线 ${o.branch}`}><thead><tr><th>方法</th><th>原模型logit差的均方误差</th></tr></thead><tbody>{Object.entries(o.mse).map(([k,v])=><tr key={k}><td>{{constant:'校准均值 / 最后token嵌入',parsed8:'文本规则解析后的8格均值',token_hash:'256维token词袋',native8:'固定原生组的8变量读出'}[k]}</td><td>{num(v)}</td></tr>)}</tbody></table><p>原生读出误差 / 规则参照误差：{num(o.native_over_parsed)}；误差改善的句式聚类区间：{ci(o.native_gain)}。</p><table className="v6-table" aria-label={`来源编码探针 ${o.branch}`}><thead><tr><th>B26固定32单元的线性读出</th><th>准确率</th></tr></thead><tbody>{Object.entries(o.probes).map(([key,v])=><tr key={key}><td>{key.startsWith('0')?'映射前':'答案前缀'} · {key.endsWith('target')?'记录来源':'答案符号'}</td><td>{pct(v.accuracy)}</td></tr>)}</tbody></table></div>)}
    </>}
    {tab==='effects'&&<>
      <p>按事先固定的家族、映射和位置选取 {doc.causal_samples} 条输入，保留其中所有原始错误。移除候选原生组，与同层同数量组及实际扰动范数匹配的方向对照比较。若某方向需要超过4倍放大或范数误差超过2%，停止相应对照主张，不剔除样本。</p>
      <p>预设作用门：平均正确答案优势下降至少0.2 nats，且超过两类等幅方向对照的99.1667%句式区间均为正，每个对照都须有效。校正按每个候选的六个任务分支计算，未保证所有辅助指标的全局错误率。</p>
      {!doc.effects.length&&<p role="status">行为门阻止了后续原模型干预，不能把缺失结果写成“没有作用”。</p>}
      {doc.effects.map(e=><details key={`${e.branch}:${e.candidate}`}><summary>{label(e.branch)} · {e.candidate} · 优势下降 {num(e.mean_loss)}</summary><table className="v6-table" aria-label={`来源编码作用 ${e.branch} ${e.candidate}`}><tbody>{[['样本 / 家族',`${e.n} / ${e.families}`],['正确答案优势下降区间',ci(e.margin_loss)],['超过随机等范数对照',ci(e.random_excess)],['超过原生方向等范数对照',ci(e.native_excess)],['无效原生方向对照',e.invalid_native_controls??'不适用'],['首词变化',e.top1_changes],['联合效应减去两个单独效应',ci(e.interaction)],['作用门',e.supported?'范围内支持；尚非完整机制':'未支持或不适用']].map(([key,value])=><tr key={key}><td>{key}</td><td>{value}</td></tr>)}</tbody></table><p>{effectReason(e)}</p></details>)}
      <details><summary>与直接按编号选择的任务比较</summary>{doc.specificity.map((s,i)=><p key={i}>{label(s.branch)} {s.candidate||''}：{s.status==='tested'?`效应差区间 ${ci(s.difference_vs_explicit)}；±0.1范围等效 ${s['equivalent_within_0.1']?'通过':'未通过'}`:'所需任务分支未通过行为门，无法比较'}。</p>)}</details>
      <p>B26早于B29，联合干预不证明“B29→B26”链条。副作用只在本轮三类任务内比较，尚未检验通用能力保持。恢复输出的逐位检查也不能替代机制结论。</p>
    </>}
    <details><summary>限制、审计与原始证据</summary><ul>{doc.limitations.map((s,i)=><li key={i}>{s}</li>)}</ul>{Object.entries(doc.artifacts||{}).map(([name,sha])=><p key={name}><a href={apiUrl(`/api/research-library/artifacts/${sha}`)} target="_blank" rel="noreferrer">{name}</a></p>)}<p>运行 {doc.run_id}；模型快照 {doc.model_snapshot}。原生数组的审计属于同节点技术复算。</p></details>
  </section>;
}
