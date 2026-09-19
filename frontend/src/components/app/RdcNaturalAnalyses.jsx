import {useEffect, useState} from 'react';

const API=(import.meta.env.VITE_RDC_CONSTRUCTION_API||'http://127.0.0.1:5004/api/rdc-construction')+'/questions';
const fmt=x=>x==null?'未定义':Number(x).toPrecision(6);
const kinds=[['readout','首步完整词表'],['prospective','预测器自身历史'],['learning','六运行训练效果'],['parameter','全部参数形成'],['training_bridge','零更新桥与训练轨迹']];
const Evidence=({data})=><details><summary>完整分析、配对区间与来源身份</summary><pre>{JSON.stringify(data,null,2)}</pre></details>;

function Readout({data,cohort,split}){
  const rows=data.summaries.filter(r=>r.split===split&&r.kind==='selected');
  return <><p>全部九条冻结路线，使用原始完整词表头。这里只评价回答起点，首 token 常为 JSON 格式；KL 或首 token 一致不等于正确完成答案。</p>
    <div className="prefix-table"><table><thead><tr><th>冻结路线</th><th>原分布 → 预测分布 KL</th><th>原首 token 一致率</th><th>首教师 token NLL</th><th>原生首教师 NLL</th></tr></thead><tbody>{rows.map(r=>{const s=r.summary[cohort];return <tr key={r.variant}><td>{r.variant}</td><td>{fmt(s.native_to_predicted_KL)}</td><td>{fmt(s.native_top1_agreement)}</td><td>{fmt(s.predicted_first_teacher_NLL)}</td><td>{fmt(s.native_first_teacher_NLL)}</td></tr>;})}</tbody></table></div>
    <p>selected 与各自 alpha0 的全部配对区间也保留在完整分析中，未按读出效果重新选择超参数。</p></>;
}

function Prospective({data,cohort}){
  return <><p>两条固定规则只计算前 13 个原生块，并消费自己刚生成的 token，随后用固定系数与原词表头预测。没有把原生晚层答案场或未来 token 输入规则。</p>
    <div className="prefix-table"><table><thead><tr><th>冻结规则</th><th>完整匹配且停止</th><th>原模型匹配且停止</th><th>JSON</th><th>EOS</th><th>128 上限</th><th>完整序列与原生一致</th></tr></thead><tbody>{data.rules.map(r=>{const s=r.summaries[cohort].means;return <tr key={r.variant}><td>{r.variant}</td><td>{fmt(s.complete_correct_and_stopped)}</td><td>{fmt(s.native_complete_correct_and_stopped)}</td><td>{fmt(s.strict_JSON)}</td><td>{fmt(s.natural_EOS)}</td><td>{fmt(s.cap_censored)}</td><td>{fmt(s.complete_sequence_matches_native)}</td></tr>;})}</tbody></table></div>
    <p>表中均为比例，不是题数。场误差仅在预定保留样本、双方尚有相同已消费前缀时成立；首次输出分叉的预测位置仍包含，之后不比较成“同条件误差”。完整答案按全部题目评分，非匹配也不自动等于推理错误。</p></>;
}

function Learning({data,cohort,split}){
  return <><p>只训练 Q4 block16 的全部 gate／up／down 参数，其余原权重固定。下表是六个实际 96 步终点转为 BF16 后的自然执行，不是 FP32 训练桥结果。真答案、同篇章整答案置乱、真实教师历史上的表面类质量各含两个固定种子。</p>
    <div className="prefix-table"><table><thead><tr><th>实际训练运行</th><th>完整教师 NLL</th><th>原生教师 NLL</th><th>后续教师 NLL</th><th>完整匹配且停止</th><th>原生完整匹配</th><th>问题差异变化 MSE</th></tr></thead><tbody>{data.runs.map(r=>{const s=r.summaries.find(x=>x.split===split)?.summary[cohort];return s&&<tr key={r.run}><td>{r.run}</td><td>{fmt(s.teacher_mean_NLL)}</td><td>{fmt(s.native_teacher_mean_NLL)}</td><td>{fmt(s.teacher_later_mean_NLL)}</td><td>{fmt(s.complete_correct_and_stopped)}</td><td>{fmt(s.native_complete_correct_and_stopped)}</td><td>{fmt(s.within_state_change_MSE)}</td></tr>;})}</tbody></table></div>
    <p>教师 NLL 越低越好，但它使用给定的真实答案历史；完整自由回答单独评分。状态变化大不等于机制改善。配对区间按篇章抽样，条件于这两个种子，不代表所有训练种子的总体不确定性，也不还原原预训练史。</p></>;
}

function Parameters({data}){
  return <><p>逐字核对全部 {data.parameters.toLocaleString()} 个目标参数的 FP32 训练终点及实际 GPU 转型的 BF16 部署终点，无 Top-K。原模型参数没有被覆盖。</p>
    <div className="prefix-table"><table><thead><tr><th>实际运行</th><th>FP32 更新 L2</th><th>BF16 更新 L2</th><th>FP32 改变字数</th><th>BF16 改变字数</th><th>实际转型误差 L2</th></tr></thead><tbody>{data.summaries.map(r=><tr key={r.run}><td>{r.run}</td><td>{fmt(r.FP32_update_L2)}</td><td>{fmt(r.BF16_update_L2)}</td><td>{r.FP32_changed_words}</td><td>{r.BF16_changed_words}</td><td>{fmt(r.actual_FP32_to_actual_BF16_rounding_L2)}</td></tr>)}</tbody></table></div>
    {['FP32','BF16'].map(precision=><details key={precision}><summary>{precision} · 六运行全部参数更新方向余弦</summary><div className="prefix-table"><table><thead><tr><th>运行</th>{data.runs.map((r,i)=><th key={r} title={r}>{i+1}</th>)}</tr></thead><tbody>{data[precision+'_direction_cosines'].map((row,i)=><tr key={data.runs[i]}><td>{i+1} · {data.runs[i]}</td>{row.map((v,j)=><td key={j}>{fmt(v)}</td>)}</tr>)}</tbody></table></div></details>)}
    <p>方向相近不等于同一语义功能；零更新方向的余弦未定义，不显示为“正交”。参数改变与语言行为改变必须分别核对。</p></>;
}

function TrainingBridge({data,cohort}){
  const base=data.baseline_summary;
  return <><p>补充精度诊断：全部 192 个原验证问题、48 篇，真实教师答案含 EOS。先重放原 BF16，再只安装零更新 FP32 block16 桥；没有优化器更新。该诊断在训练已开始后新增，不是新的确认材料。</p>
    <div className="prefix-table"><table data-testid="training-bridge-baselines"><thead><tr><th>零更新端点</th><th>完整教师 NLL</th><th>首教师 NLL</th><th>后续教师 NLL</th><th>相对原生教师场 MSE</th></tr></thead><tbody>{[['native','原生 BF16'],['untrained_bridge','零更新 FP32 桥']].map(([key,label])=><tr key={key}><td>{label}</td>{base[key][cohort].map((x,i)=><td key={i}>{fmt(x)}</td>)}<td>{fmt(key==='native'?0:base.teacher_state_MSE[cohort])}</td></tr>)}</tbody></table></div>
    <p>以下全部六运行 × 四个固定检查点仍在 FP32 桥中评价。差值为训练端点减零更新桥，负 NLL 差表示给定教师历史评分降低；不能与原生 BF16 自由生成效果混称。</p>
    <div className="prefix-table" style={{maxHeight:540}}><table data-testid="training-bridge-checkpoints"><thead><tr><th>实际训练运行</th><th>步</th><th>完整 NLL</th><th>首 NLL</th><th>后续 NLL</th><th>完整 NLL − 零更新桥</th><th>95% 下限</th><th>95% 上限</th><th>相对零更新桥教师场 MSE</th></tr></thead><tbody>{data.checkpoints.map(r=>{const p=r.paired_vs_untrained_bridge.answer_mean_NLL[cohort];return <tr key={r.run+r.step}><td>{r.run}</td><td>{r.step}</td>{r.summary[cohort].map((x,i)=><td key={i}>{fmt(x)}</td>)}<td>{fmt(p.mean_left_minus_right)}</td>{p.paired_context_bootstrap_95_percent_interval.map((x,i)=><td key={i}>{fmt(x)}</td>)}<td>{fmt(r.teacher_state_MSE_vs_untrained_bridge[cohort])}</td></tr>;})}</tbody></table></div>
    <p>区间按完整篇章配对抽样，属于事后新增诊断、未作多重比较校正；全部检查点均保留，不据此重新挑选终点。完整回执也保留相对原生基线、首步／后续指标的区间与全部坐标依据。</p></>;
}

export default function NaturalAnalyses({model,refresh,confirmationOpen}){
  const [kind,setKind]=useState('readout'),[scope,setScope]=useState('nonconfirmation'),[split,setSplit]=useState('diagnostic'),[cohort,setCohort]=useState('equal_cohort');
  const [response,setResponse]=useState(null),[error,setError]=useState(null);
  const local=['learning','parameter','training_bridge'].includes(kind);
  const validModel=!local||model==='qwen4',validScope=scope!=='confirmation'||(confirmationOpen&&!['parameter','training_bridge'].includes(kind));
  const id=JSON.stringify([model,kind,scope,refresh,confirmationOpen]);
  useEffect(()=>{let live=true;if(validModel&&validScope)fetch(API+'/analysis?'+new URLSearchParams({model,kind,scope})).then(async r=>{const value=await r.json();if(!r.ok)throw new Error(value.detail||r.statusText);return value;}).then(value=>{if(live){setResponse({id,value});setError(null);}}).catch(e=>live&&setError({id,text:e.message}));return()=>{live=false;};},[id,validModel,validScope]);
  const actual=validModel&&validScope&&response?.id===id?response.value:null;
  const data=actual?.complete?actual.result:null,effectiveSplit=scope==='confirmation'?'confirmation':split;
  return <div id="natural-complete-analyses"><h3>从状态预测到自行回答与训练形成：分项验收</h3>
    <div className="prefix-controls"><label>完整证据类型<select aria-label="完整证据类型" value={kind} onChange={e=>setKind(e.target.value)}>{kinds.map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label>
      <label>完整证据范围<select aria-label="完整证据范围" value={scope} onChange={e=>setScope(e.target.value)}><option value="nonconfirmation">非确认</option><option value="confirmation">封闭确认</option></select></label>
      {kind!=='parameter'&&<label>完整证据语料<select aria-label="完整证据语料" value={cohort} onChange={e=>setCohort(e.target.value)}>{['equal_cohort','drop','quoref'].map(x=><option key={x}>{x}</option>)}</select></label>}
      {scope==='nonconfirmation'&&['readout','learning'].includes(kind)&&<label>完整证据划分<select aria-label="完整证据划分" value={split} onChange={e=>setSplit(e.target.value)}>{['diagnostic','validation'].map(x=><option key={x}>{x}</option>)}</select></label>}</div>
    {!validModel?<p role="status">六运行训练只在 Q4 上登记；当前模型没有这项训练证据。</p>:!validScope?<p role="status">{kind==='parameter'?'参数终点不按确认材料重复计数，请查看非确认范围。':kind==='training_bridge'?'零更新桥仅覆盖原验证集，请查看非确认范围。':'确认集尚未解封，不读取其分析或结果。'}</p>:error?.id===id?<p role="alert" className="prefix-error">{error.text}</p>:!data?<p role="status">{actual?'待完成：尚无通过核对的分析回执。已编写或已通过资格检查的代码不算实验结果。':'正在读取当前范围的真实回执…'}</p>:<>
      {kind==='readout'&&<Readout data={data} cohort={cohort} split={effectiveSplit}/>}
      {kind==='prospective'&&<Prospective data={data} cohort={cohort}/>}
      {kind==='learning'&&<Learning data={data} cohort={cohort} split={effectiveSplit}/>}
      {kind==='parameter'&&<Parameters data={data}/>}
      {kind==='training_bridge'&&<TrainingBridge data={data} cohort={cohort}/>}
      <Evidence data={actual}/></>}
  </div>;
}
