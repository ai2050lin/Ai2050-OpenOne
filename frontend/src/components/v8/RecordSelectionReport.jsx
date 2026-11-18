import {useMemo,useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';
import {apiUrl} from '../../config.js';
import {NativeArray} from './MechanismScene.jsx';
import './hiddenstate_components.css';
import './record_selection.css';

const num=v=>Number.isFinite(v)?v.toPrecision(4):'未定义';
const pct=v=>Number.isFinite(v)?`${(100*v).toFixed(2)}%`:'未定义';
const interval=v=>v?.frame_interval?.map(num).join(' 至 ')||'未检验';
const names={ablation:'移除所选组',valid_source:'屏蔽有效记录的直接来源',obsolete_source:'屏蔽过时记录的直接来源'};

function Coordinates({doc}){
  const [kind,setKind]=useState('heads'),[block,setBlock]=useState(0),[split,setSplit]=useState(1),[term,setTerm]=useState(0),[index,setIndex]=useState(0);
  const key=kind==='heads'?'all':String(block),digest=doc.coordinate_artifacts?.[kind]?.[key];
  const resource=useApi(digest?`/api/research-library/artifacts/${digest}`:null),length=kind==='heads'?1152:9728;
  const values=useMemo(()=>{
    const d=resource.data,a=d?.array;
    if(!a||a.dtype!=='<f4'||a.encoding!=='base64'||a.shape?.join(',')!==`2,2,${length}`||d.model_snapshot!==doc.model_snapshot||d.run_id!==doc.run_id||d.field!==kind||String(d.index)!==key)return null;
    try{const raw=Uint8Array.from(atob(a.data),c=>c.charCodeAt(0));if(raw.length!==4*4*length)return null;const view=new DataView(raw.buffer);return Array.from({length},(_,i)=>view.getFloat32(((split*2+term)*length+i)*4,true));}catch{return null;}
  },[resource.data,doc.model_snapshot,doc.run_id,kind,key,length,split,term]);
  return <>
    <div className="hs-controls">
      <label>对象 <select aria-label="记录机制对象" value={kind} onChange={e=>{setKind(e.target.value);setIndex(0);setTerm(0);}}><option value="heads">全部36层 × 32个注意力头</option><option value="units">MLP原生单元</option></select></label>
      {kind==='units'&&<label>块 <select aria-label="记录机制层" value={block} onChange={e=>setBlock(Number(e.target.value))}>{Array.from({length:36},(_,i)=><option key={i} value={i}>B{i}</option>)}</select></label>}
      <label>数据 <select aria-label="记录机制数据划分" value={split} onChange={e=>setSplit(Number(e.target.value))}><option value="0">发现集合格配对</option><option value="1">留出集合格配对</option></select></label>
      <label>度量 <select aria-label="记录机制度量" value={term} onChange={e=>setTerm(Number(e.target.value))}>{(kind==='heads'?['有效记录在前：注意力质量差','有效记录在后：注意力质量差']:['激活的配对对照','局部写入对照']).map((v,i)=><option key={v} value={i}>{v}</option>)}</select></label>
    </div>
    {!digest?<p role="status">此版本缺少坐标产物。</p>:resource.error?<p role="status">记录机制坐标读取失败：{resource.error}</p>:!resource.data?<p>正在读取原生数值…</p>:!values?<p role="status">数据身份或形状不匹配，未显示坐标。</p>:<>
      <label>原生索引 <input aria-label="记录机制原生索引" type="number" min="0" max={length-1} value={index} onChange={e=>setIndex(Math.min(length-1,Math.max(0,Number(e.target.value)||0)))}/></label>
      <p>原生索引 {index} = {num(values[index])}{kind==='heads'?`（B${Math.floor(index/32)} / H${index%32}）`:`（B${block} / U${index}）`}</p>
      <NativeArray values={values} selected={index} onPick={setIndex} label="记录机制完整原生阵列"/>
    </>}
    <p className="v8-note">保留原生顺序与全部数值，未按强度排序。头的度量是“有效来源−过时来源”的注意力质量；MLP配对对照以记录1/2作符号。局部写入尚未包含后续层和最终归一化，不能当作最终因果贡献。失败条件的整组配对未进入这些均值。</p>
  </>;
}

export default function RecordSelectionReport({doc}){
  const [tab,setTab]=useState('behavior'),[candidate,setCandidate]=useState(doc.candidates[0].id),[method,setMethod]=useState('ablation'),[dose,setDose]=useState(1),[page,setPage]=useState(0);
  const effect=doc.effects.find(e=>e.candidate===candidate&&e.method===method&&e.dose===dose),selected=doc.candidates.find(c=>c.id===candidate);
  const matched=doc.matched_native_controls?.find(e=>e.candidate===candidate);
  const failedCells=Object.entries(doc.behavior.cells).filter(([,v])=>v.strict_accuracy<.9);
  return <section className="hs-components-report record-selection-report" aria-label="有效与过时记录机制报告">
    <h3>模型怎样选中有效记录？</h3>
    <p>本轮把问题拆成“选记录编号”和“读取记录内容”。这里只检验前者；颜色提取的两轮校准失败仍保留。共 {doc.unique_inputs.toLocaleString()} 条新输入，留出集 {doc.families.confirm} 个家族。</p>
    <p role="status"><strong>完整范围的行为门未通过；以下为受限候选与作用诊断，不作正式机制确认。</strong> 留出行为准确率 {pct(doc.behavior.strict_accuracy)}；{failedCells.length} 个条件格未达90%。合格配对 {doc.admission.eligible_families.length} / {doc.families.confirm}，排除 {doc.admission.excluded_families.length} 组，原始证据均保留。</p>
    <p><strong>目前能保留的拼图：</strong>所选头组在前后两种位置都更关注有效记录，但移除后的作用未稳定超过同层对照；所选MLP组对正确答案优势有小幅影响，合格样本中未造成首词答案翻转。组合读出的预测MSE为简单条件基线的 {num(doc.baselines.native_over_structural)} 倍，尚未建立更好的机制预测器。</p>
    <div className="v6-toolbar">{[['behavior','行为与反例'],['native','注意力头与MLP单元'],['prediction','简单基线比较'],['effects','实际作用与对照']].map(([id,label])=><button className="fw-tbtn" key={id} type="button" aria-pressed={tab===id} onClick={()=>setTab(id)}>{label}</button>)}</div>
    {tab==='behavior'&&<>
      <p>发现材料用于校准和选候选；留出材料使用新实体、新记录表述和新查询句式。有效记录有时在前、有时在后；过时内容分别与有效内容一致或冲突。答案只要求数字1或2，不能将本轮准确率解释成颜色任务准确率。</p>
      <table className="v6-table" aria-label="记录选择行为校准"><thead><tr><th>版本 / 检验</th><th>样本数</th><th>准确率</th><th>逐格门</th></tr></thead><tbody>{doc.calibration_history.map(r=><tr key={r.id}><td>{r.label}</td><td>{r.n}</td><td>{pct(r.accuracy)}</td><td>{r.passed?'通过':'未通过'}</td></tr>)}</tbody></table>
      <p>小试包含在完整发现集中，不重复计数。原失败不被扩大样本后的结果覆盖。通过条件是预设逐格90%，并非只看总体准确率；有限样本仍有不确定性。</p>
      <details><summary>留出集未通过的全部条件格</summary><table className="v6-table"><tbody>{failedCells.map(([k,v])=><tr key={k}><td>{k}（句式:有效位置:冲突:语言）</td><td>{v.n}</td><td>{pct(v.strict_accuracy)}</td></tr>)}</tbody></table></details>
      <details><summary>错误集中在哪些条件？</summary><p>以下为事后描述，分组有重叠；相关性尚不能证明错误原因。</p><table className="v6-table" aria-label="记录选择行为失败分组"><thead><tr><th>条件</th><th>样本数</th><th>错误数</th><th>并列</th></tr></thead><tbody>{(doc.behavior_error_strata||[]).filter(r=>['valid_position','language','conflict'].includes(r.field)).map(r=><tr key={`${r.field}:${r.value}`}><td>{r.field==='valid_position'?(r.value===-1?'有效在前':'有效在后'):r.field==='language'?(r.value===-1?'中文查询':'英文查询'):(r.value?'新旧冲突':'新旧一致')}</td><td>{r.n}</td><td>{r.errors}</td><td>{r.ties}</td></tr>)}</tbody></table></details>
      <h4>真实错误与反例（{doc.failures.length} 条）</h4>
      {doc.failures.slice(page*8,page*8+8).map(f=><details key={f.sample.id}><summary>{f.sample.id} · 正确答案优势 {num(f.score.target_margin)}</summary><pre className="v8-sample-text">{f.sample.prompt}</pre><p>实际候选分数 {f.score.choice_logits.map(num).join(' / ')}；要求的有效记录 {f.sample.valid_position===-1?'1':'2'}。</p></details>)}
      <div className="v6-toolbar"><button className="fw-tbtn" disabled={!page} onClick={()=>setPage(page-1)}>上一页反例</button><span>{page+1} / {Math.max(1,Math.ceil(doc.failures.length/8))}</span><button className="fw-tbtn" disabled={(page+1)*8>=doc.failures.length} onClick={()=>setPage(page+1)}>下一页反例</button></div>
      <details><summary>冻结方案、审计与原始证据</summary>{Object.entries(doc.artifacts||{}).map(([label,sha])=><p key={label}><a href={apiUrl(`/api/research-library/artifacts/${sha}`)} target="_blank" rel="noreferrer">{label}</a></p>)}<p>模型快照 {doc.model_snapshot}。单节点技术审计不等于独立科学复现。</p></details>
    </>}
    {tab==='native'&&<>
      <p>先在发现集前四个句式中选组，再由后四个句式选层；随后冻结。头组为2个头，单元组为32个MLP中间单元。候选由原生响应提出，未使用确认集重新排序。</p>
      {doc.candidates.map(c=><details key={c.id}><summary>{c.id} · {c.indices.length} 个原生对象 · 发现期 maxT p={num(c.validation_maxT_p)}</summary><p>索引：{c.indices.join(', ')}。候选选择性门：{c.discovery_support?'通过':'未通过'}；不是完整机制确认。</p><p>对照组 {c.controls.map(v=>v.indices.join(',')).join(' / ')}</p></details>)}
      <Coordinates doc={doc}/>
      <details><summary>候选在留出样本的观测</summary>{doc.observations.map(o=><p key={o.id}>{o.id}：均值 {num(o.mean)}；句式聚类区间 {interval(o.cluster_ci)}；双侧家族符号置换 p={num(o.two_sided_family_swap_p)}。{o.head_positions&&`有效在前/后的均值 ${o.head_positions.map(num).join(' / ')}`}</p>)}</details>
    </>}
    {tab==='prediction'&&<>
      <p>预测目标为原模型“记录1相对记录2”的logit优势。只在同一批 {doc.baselines.eligible_families} 个合格留出家族上比较。原生组读出需要实际模型激活，尚不是仅靠输入运行的独立算法。</p>
      <table className="v6-table" aria-label="记录选择预测基线"><thead><tr><th>方法</th><th>留出均方误差</th></tr></thead><tbody>{Object.entries(doc.baselines.mse).map(([k,v])=><tr key={k}><td>{{global_mean:'按查询语言取全局均值',parsed_32cell:'解析条件后的32格均值',native_group_readout:'固定原生组的线性读出',embedding_constant:'固定答案前token的嵌入常数'}[k]}</td><td>{num(v)}</td></tr>)}</tbody></table>
      <p>答案前最后一个token在全部材料中相同，因此该位置的输入嵌入相同；嵌入基线只输出发现集拟合的常数。上下文差异来自后续计算，不能据此给单个嵌入坐标命名语义。</p>
      <p>原生组误差 / 32格基线误差：{num(doc.baselines.native_over_structural)}。基线误差减去原生组误差的句式区间：{interval(doc.baselines.native_gain_ci)}。</p>
      <p>始终选第一条 / 最后一条的准确率均为50%；外部给定的状态标签解析器为100%。后者只是人工提供的规则，不能被当作发现了模型内部算法。</p>
      <p>补强的负对照：1,024次整家族符号置换、双侧统计量及候选组最大统计量校正；前缀位置尚不可见的条件保持逐位相同；单元和头对照在同层匹配数量与发现期写入能量。</p>
    </>}
    {tab==='effects'&&<>
      <div className="hs-controls"><label>候选 <select aria-label="记录作用候选" value={candidate} onChange={e=>{setCandidate(e.target.value);setMethod('ablation');}}>{doc.candidates.map(c=><option key={c.id}>{c.id}</option>)}</select></label>
        <label>操作 <select aria-label="记录作用方法" value={method} onChange={e=>{setMethod(e.target.value);setDose(1);}}>{Object.entries(names).filter(([k])=>k==='ablation'||selected.kind==='heads').map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label>
        <label>剂量 <select aria-label="记录作用剂量" value={dose} disabled={method!=='ablation'} onChange={e=>setDose(Number(e.target.value))}><option value="0.5">移除一半</option><option value="1">完整移除</option></select></label></div>
      {effect&&<>
        <p>{effect.n} 个合格留出家族。其余家族的操作结果保留为失败诊断，不计入下表。作用证据：受限诊断，未升级正式机制。</p>
        <table className="v6-table" aria-label="记录选择实际作用"><tbody>{[['正确答案优势下降',num(effect.mean_margin_loss)],['同层对照组的下降',num(effect.native_group_control_loss)],['等残差范数随机对照的下降',num(effect.matched_norm_control_loss)],['超过同层对照：句式区间',interval(effect.excess_native)],['超过等范数对照：句式区间',interval(effect.excess_norm)],['冲突减去一致条件的作用',interval(effect.conflict_minus_agreement)],['正确→错误 / 错误→正确',`${effect.first_correct_to_wrong} / ${effect.first_wrong_to_correct}`],['完整词表平均KL',num(effect.mean_KL)]].map(([k,v])=><tr key={k}><td>{k}</td><td>{v}</td></tr>)}</tbody></table>
        <p>失败原因：{effect.failures.join('；')}</p>
        {method==='ablation'&&dose===1&&matched&&<><h4>补强：真实对照组方向，匹配实际扰动幅度</h4><p>原始同层组的写入能量较小，因此另用其真实残差变化方向，缩放至候选组的实际扰动范数。这是方向对照，不是普通的单元组消融。</p><table className="v6-table" aria-label="记录选择等幅度原生方向对照"><tbody>{[['对照平均优势下降',num(matched.control_mean_loss)],['候选超过对照的句式区间',interval(matched.excess)],['缩放倍数：最小 / 中位 / 最大',matched.scale_min_median_max.map(num).join(' / ')],['实际范数最大相对误差',pct(matched.maximum_norm_error)]].map(([k,v])=><tr key={k}><td>{k}</td><td>{v}</td></tr>)}</tbody></table></>}
        {method!=='ablation'&&<p>只屏蔽所选头在答案前位置到对应文本的直接边，注意力会重新归一化；不排除早期层或其他位置已传播的信息。来源长度未匹配，因此不能证明来源必要性。</p>}
        <details><summary>作用最弱或方向相反的真实样本</summary>{effect.counterexamples.map(e=><details key={e.family}><summary>家族 {e.family} · 优势下降 {num(e.loss)}</summary><pre className="v8-sample-text">{e.prompt}</pre></details>)}</details>
      </>}
      <p>干预限于最后位置的原生头输出或MLP中间激活；未修改权重。恢复后输出逐位一致。这里只测同任务的一致/冲突副作用，尚未验证其他任务能力保持，也未验证头到MLP的因果中介链。</p>
    </>}
    <details><summary>结论边界与下一阶段</summary><ul>{doc.limitations.map((s,i)=><li key={i}>{s}</li>)}</ul><p>下一阶段应固定本轮候选，独立交叉检验状态词、时间顺序和记录内容，再验证“选源→绑定→读出”的组合与替代路径；不能继续靠改提示、换层和缩小条件取得局部过门。</p></details>
  </section>;
}
