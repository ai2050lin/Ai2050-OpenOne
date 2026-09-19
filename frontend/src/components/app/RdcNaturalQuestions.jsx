import {useEffect, useState} from 'react';
import {Field} from './RdcJointAtlas.jsx';
import NaturalAnalyses from './RdcNaturalAnalyses.jsx';
import SourceCoupling from './RdcSourceCoupling.jsx';

const API=(import.meta.env.VITE_RDC_CONSTRUCTION_API||'http://127.0.0.1:5004/api/rdc-construction')+'/questions';
const MODELS=['qwen4','qwen14','glm4'];
const fmt=x=>x==null?'未提交':Number(x).toPrecision(6);
async function get(path,params={}){
  const response=await fetch(API+path+'?'+new URLSearchParams(params));
  if(!response.ok)throw new Error((await response.json().catch(()=>({}))).detail||response.statusText);
  return response.json();
}
function Evidence({title,data}){return <details><summary>{title}</summary><pre>{JSON.stringify(data,null,2)}</pre></details>;}

function NativeParameterPath({model,question,refresh,enabled}){
  const [block,setBlock]=useState(12),[unit,setUnit]=useState(0),[output,setOutput]=useState(0),[result,setResult]=useState(null),[error,setError]=useState(null);
  const id=JSON.stringify([model,question,block,unit,output,refresh]);
  async function load(){try{setResult({id,value:await get('/parameter-path',{model,question,block,unit,output})});setError(null);}catch(e){setResult(null);setError({id,text:e.message});}}
  const value=result?.id===id?result.value:null;
  return <div id="natural-parameter-path"><h4>同一固定参数，四个真实问题：完整读取与写回项</h4>
    <p>手动选择原生单元和输出坐标；显示全部输入乘积与全部单元写回项，不按幅度挑选。gₖ=ΣᵢWg[k,i]xᵢ，uₖ=ΣᵢWu[k,i]xᵢ，aₖ=SiLU(gₖ)uₖ，写回ⱼ=ΣₖWd[j,k]aₖ。MLP 单元 k 与残差坐标 j 是不同空间。</p>
    <div className="prefix-controls">{[['路径 block',block,setBlock],['路径 MLP 单元 k',unit,setUnit],['路径输出坐标 j',output,setOutput]].map(([label,v,set])=><label key={label}>{label}<input aria-label={label} type="number" min={0} value={v} onChange={e=>set(Number(e.target.value))}/></label>)}<button disabled={!enabled} onClick={load}>读取自然问题的完整参数路径</button></div>
    {error?.id===id&&<p role="alert" className="prefix-error">{error.text}</p>}
    {value&&<><div className="prefix-table"><table><thead><tr><th>自然问题</th><th>原生 gate / up</th><th>原生单元乘积</th><th>该单元写回项</th><th>全部单元重算 − 原生写回</th></tr></thead><tbody>{value.records.map(r=><tr key={r.question_id}><td>{r.question}</td><td>{fmt(r.native_gate_BF16)} / {fmt(r.native_up_BF16)}</td><td>{fmt(r.native_product_BF16)}</td><td>{fmt(r.selected_unit_write_term_FP64)}</td><td>{fmt(r.write_FP64_minus_native_BF16)}</td></tr>)}</tbody></table></div>
      <p>CPU 用原始 BF16 参数和状态重算 FP64 加和；与原生 BF16 矩阵乘法／SiLU／舍入不逐位相同，上表明确列出差异。既有原生采集资格另验证了实际 GPU 同形状计算。这里不把某个单元命名为语义概念。</p>
      <h4>每题全部输入坐标乘积</h4><Field key={id+'input'} data={value.input_terms}/><h4>每题全部 MLP 单元写回项</h4><Field key={id+'write'} data={value.write_terms}/>
      <Evidence title="完整固定参数向量、标量地址、原始数值与源场身份" data={value}/></>}
  </div>;
}

function NativeFields({model,refresh}){
  const [split,setSplit]=useState('diagnostic'),[cohort,setCohort]=useState('all'),[question,setQuestion]=useState('');
  const [index,setIndex]=useState(null),[detail,setDetail]=useState(null),[field,setField]=useState(null),[failure,setFailure]=useState(null);
  const [mode,setMode]=useState('first_hidden'),[view,setView]=useState('raw'),[block,setBlock]=useState(12),[step,setStep]=useState(0),[start,setStart]=useState(0);
  const indexID=JSON.stringify([model,split,cohort,refresh]);
  useEffect(()=>{let alive=true;get('/samples',{model,split,cohort}).then(rows=>{if(alive){setIndex({id:indexID,rows});setFailure(null);}}).catch(e=>alive&&setFailure({id:indexID,text:e.message}));return()=>{alive=false;};},[indexID]);
  const options=index?.id===indexID?index.rows:[];
  const chosen=options.some(r=>r.question_id===question)?question:options.find(r=>r.captured)?.question_id||options[0]?.question_id||'';
  const selected=options.find(r=>r.question_id===chosen);
  const sourceID=JSON.stringify([model,chosen,refresh]),id=JSON.stringify([sourceID,mode,view,block,step,start]);
  async function load(){try{const [a,b]=await Promise.all([get('/sample',{model,question:chosen}),get('/field',{model,question:chosen,mode,view,block,step,start})]);setDetail({id:sourceID,value:a});setField({id,value:b});setFailure(null);}catch(e){setField(null);setFailure({id,text:e.message});}}
  const actual=detail?.id===sourceID?detail.value:null,shown=field?.id===id?field.value:null;
  const error=[id,indexID].includes(failure?.id)?failure.text:null;
  return <div id="natural-question-fields"><h3>同篇章四问题：真实语言 → 原生响应 → 自身历史</h3>
    <div className="prefix-controls">
      <label>自然材料划分<select aria-label="自然材料划分" value={split} onChange={e=>setSplit(e.target.value)}>{['diagnostic','validation','train','confirmation'].map(x=><option key={x}>{x}</option>)}</select></label>
      <label>自然语料<select aria-label="自然语料" value={cohort} onChange={e=>setCohort(e.target.value)}>{['all','drop','quoref'].map(x=><option key={x}>{x}</option>)}</select></label>
      <label className="prefix-wide">自然问题<select aria-label="自然问题" value={chosen} onChange={e=>setQuestion(e.target.value)}>{options.map(r=><option key={r.question_id} value={r.question_id}>{r.cohort} · 题{r.within_context_index+1} · {r.captured?'已提交':'待采集'} · {r.question}</option>)}</select></label>
      <label>自然场类型<select aria-label="自然场类型" value={mode} onChange={e=>setMode(e.target.value)}>
        <option value="first_hidden">首步全部层 + postnorm</option><option value="MLP">选定块全部 MLP 单元</option><option value="attention">block12 全部 attention 来源</option><option value="source_H12">篇章 H12 全坐标 / token 分页</option>
        <option value="history_postnorm">所有生成步 postnorm</option><option value="history_H12">所有生成步 H12</option><option value="history_read">所有生成步来源读出</option><option value="history_all_hidden">指定生成步全部层（仅预定材料）</option><option value="teacher_postnorm">教师给定历史 postnorm</option>
      </select></label>
      <label>自然场数值<select aria-label="自然场数值" value={view} onChange={e=>setView(e.target.value)}><option value="raw">原始值</option><option value="row_RMS">逐完整行 RMS 归一化</option></select></label>
      {mode==='MLP'&&<label>自然 MLP 块<input aria-label="自然 MLP 块" type="number" min={0} max={80} value={block} onChange={e=>setBlock(Number(e.target.value))}/></label>}
      {mode==='history_all_hidden'&&<label>自然生成步<input aria-label="自然生成步" type="number" min={0} max={127} value={step} onChange={e=>setStep(Number(e.target.value))}/></label>}
      {mode==='source_H12'&&<label>篇章 token 起点<input aria-label="篇章 token 起点" type="number" min={0} value={start} onChange={e=>setStart(Number(e.target.value))}/></label>}
      <button disabled={!selected?.captured} onClick={load}>读取自然问题与完整坐标</button>
    </div>
    <p>接口只开放完整提交的四题组。确认集未解封时不可查询；未保留的层／生成步返回缺失说明，不以其他数组代替。切换查询后旧图立即隐藏。</p>
    {error&&<p role="alert" className="prefix-error">{error}</p>}
    {actual&&<><blockquote style={{whiteSpace:'pre-wrap'}}>{actual.material.passage}</blockquote>
      <div className="prefix-table"><table><thead><tr><th>同篇章问题</th><th>原始标注（联合跨度）</th><th>模型完整自由回答</th><th>匹配且停止</th></tr></thead><tbody>{actual.siblings.map(({material:r,record:q})=><tr key={r.question_id}><td>{r.question}</td><td>{JSON.stringify(r.answer_annotations)}</td><td>{q.history?.generated_text??'训练题未采集自由历史'}</td><td>{q.history?String(q.history.score.whole_response_exact_and_stopped):'未测试'}</td></tr>)}</tbody></table></div>
      <p>完整文本匹配是保守指标，不是通用语义正确率；教师评分不是自由回答。实际首 token：{actual.record.statistics?.argmax??'见执行回执'}。字段在发出对应 token 之前采集。</p>
      <Evidence title="原始题目、实际完整输入/分词、逐题输出及执行身份" data={actual}/></>}
    {shown&&<p>{shown.axes}</p>}<Field key={id} data={shown}/>
    <NativeParameterPath model={model} question={chosen} refresh={refresh} enabled={Boolean(selected?.captured)}/>
  </div>;
}

function Aggregate({model,refresh,figures}){
  const [cohort,setCohort]=useState('drop'),[channel,setChannel]=useState('hidden'),[statistic,setStatistic]=useState('train_standardized_RMS'),[block,setBlock]=useState(12);
  const [field,setField]=useState(null),[failure,setFailure]=useState(null);
  const id=JSON.stringify([model,cohort,channel,statistic,block,refresh]);
  async function load(){try{setField({id,value:await get('/atlas',{model,cohort,channel,statistic,block})});setFailure(null);}catch(e){setField(null);setFailure({id,text:e.message});}}
  const actual=field?.id===id?field.value:null;
  return <div id="natural-question-aggregate"><h3>低幅背景也保留：全部原生坐标统计</h3>
    <p>诊断篇章每组四个真实问题，减去各自篇章均值，汇总问题相关变化。标准化仅使用对应语料训练集尺度；原始值与标准化值分开，坐标顺序不变。亮区不是已证明的语义模块。</p>
    <div className="prefix-controls"><label>图谱语料<select aria-label="图谱语料" value={cohort} onChange={e=>setCohort(e.target.value)}>{['drop','quoref'].map(x=><option key={x}>{x}</option>)}</select></label>
      <label>图谱坐标空间<select aria-label="图谱坐标空间" value={channel} onChange={e=>setChannel(e.target.value)}><option value="hidden">全部残差层 + postnorm</option><option value="MLP">选定块全部 MLP 单元</option></select></label>
      <label>图谱统计<select aria-label="图谱统计" value={statistic} onChange={e=>setStatistic(e.target.value)}><option value="train_standardized_RMS">问题变化 RMS / 训练坐标 SD</option><option value="within_RMS">问题变化原始 RMS</option><option value="mean">原始均值</option></select></label>
      {channel==='MLP'&&<label>图谱 MLP 块<input aria-label="图谱 MLP 块" type="number" min={0} max={80} value={block} onChange={e=>setBlock(Number(e.target.value))}/></label>}<button onClick={load}>读取自然语料全坐标图谱</button></div>
    {failure?.id===id&&<p role="alert" className="prefix-error">{failure.text}</p>}
    {actual&&<p>{actual.axes} · {actual.normalization_definition}</p>}<Field key={id} data={actual}/>
    {figures.map(f=><details key={f.sha256}><summary>{f.name} · 已实际查看的固定线性色标总览</summary><img src={API+'/figure?'+new URLSearchParams({model,name:f.name})} alt={model+' '+f.name+' 全原生坐标自然问题响应'} style={{width:'100%',height:'auto'}} loading="lazy"/><p>两语料共用色标 [0, {fmt(f.color_limits[1])}]；全部原生坐标，非 Top-K。交互图可采用不同显示映射，请勿直接比较颜色深浅。</p></details>)}
  </div>;
}

function RelationGeometry({model,refresh,summary}){
  const [cohort,setCohort]=useState('drop'),[metric,setMetric]=useState('excess'),[left,setLeft]=useState('H12'),[right,setRight]=useState('postnorm');
  const [matrix,setMatrix]=useState(null),[error,setError]=useState(null);const id=JSON.stringify([model,refresh,cohort,metric]);
  useEffect(()=>{let active=true;if(summary?.complete)get('/relation-geometry',{model,cohort,metric}).then(v=>active&&setMatrix({id,value:v})).catch(e=>active&&setError({id,text:e.message}));return()=>{active=false;};},[id,summary?.complete]);
  if(!summary?.complete)return null;
  const data=matrix?.id===id?matrix.value:null,labels=summary.labels;
  const value=data?.values?.[labels.indexOf(left)]?.[labels.indexOf(right)];
  const count=data?.valid_context_counts?.[labels.indexOf(left)]?.[labels.indexOf(right)];
  return <div id="natural-relation-geometry"><h3>四问题关系怎样跨层保留：全坐标关系矩阵</h3>
    <p>每个观察空间都使用全部原坐标，比较同篇章四题的中心化 Gram 关系。原始相似度可能因四题几何本身而偏高，所以同时计算全部 24 种问题身份置换的精确平均。下表“超出置换均值”不是语义解释比例；词面重叠、问题长度与残差接续也可能造成对应。</p>
    <div className="prefix-table"><table><thead><tr><th>观察空间 → postnorm</th><th>原始相似度</th><th>置换均值</th><th>超出置换均值 [描述性 95% 区间]</th></tr></thead><tbody>{summary.fixed_pairs.map(r=>{const a=r.by_cohort,b=r.paired_excess.equal_cohort;return <tr key={r.left}><td>{r.left}</td><td>{fmt((a.drop.similarity_mean+a.quoref.similarity_mean)/2)}</td><td>{fmt((a.drop.permutation_mean+a.quoref.permutation_mean)/2)}</td><td>{fmt(b.mean_left_minus_right)} [{b.paired_context_bootstrap_95_percent_interval.map(fmt).join(', ')}]</td></tr>;})}</tbody></table></div>
    <p>诊断集 96 篇章，整篇章配对抽样，两语料等权；区间未作多重比较校正。四题 Gram 的秩至多为 3 是样本数限制，不说明模型内部只有三维。首步关系复用不代表能跨生成步复用。</p>
    <div className="prefix-controls"><label>关系语料<select aria-label="关系语料" value={cohort} onChange={e=>setCohort(e.target.value)}>{['drop','quoref'].map(x=><option key={x}>{x}</option>)}</select></label>
      <label>关系指标<select aria-label="关系指标" value={metric} onChange={e=>setMetric(e.target.value)}>{['excess','similarity','permutation_mean'].map(x=><option key={x}>{x}</option>)}</select></label>
      <label>关系行空间<select aria-label="关系行空间" value={left} onChange={e=>setLeft(e.target.value)}>{labels.map(x=><option key={x}>{x}</option>)}</select></label>
      <label>关系列空间<select aria-label="关系列空间" value={right} onChange={e=>setRight(e.target.value)}>{labels.map(x=><option key={x}>{x}</option>)}</select></label></div>
    <p data-testid="relation-value">{data?`${left} → ${right}：${value==null?'未定义（零响应能量）':fmt(value)}；有效篇章 ${count??0}`:'正在读取当前关系矩阵…'}</p>
    {error?.id===id&&<p role="alert">{error.text}</p>}
    {summary.figures.map(f=><details key={f.sha256}><summary>{f.name} · 全部观察空间矩阵（已实际核图）</summary><img src={API+'/relation-figure?'+new URLSearchParams({model,name:f.name})} alt={model+' 全坐标关系矩阵 '+f.name} style={{width:'100%',height:'auto'}} loading="lazy"/><p>行列是观察空间，不是单个神经元；灰色表示未定义。两语料相同线性色标 [{f.color_limits.map(fmt).join(', ')}]，不重排节点。</p></details>)}
    <Evidence title="全部关系数值、节点索引与有效篇章数（非 Top-K）" data={data}/>
  </div>;
}

export default function NaturalQuestions({refresh}){
  const [overview,setOverview]=useState(null),[error,setError]=useState(''),[model,setModel]=useState('qwen4');
  useEffect(()=>{let alive=true;get('/overview').then(x=>{if(alive){setOverview(x);setError('');}}).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[refresh]);
  const item=overview?.models?.[model]||{},fit=item.fit_summaries?.filter(x=>x.split==='diagnostic'&&x.target==='postnorm'&&x.kind==='selected')||[];
  return <section id="construction-natural-questions"><h2>Phase 2748 · 自然篇章怎样随问题改变内部响应</h2>
    <p>冻结 400 篇章 × 每篇 4 题，共 1600 题；DROP 与 Quoref 分开报告。篇章级训练／验证／诊断／确认划分，避免把同篇四题随机拆开。确认集：{overview?.confirmation_open?'已通过冻结门':'尚未解封'}。</p>
    <div className="construction-cards">{MODELS.map(m=>{const x=overview?.models?.[m];return <div key={m}><strong>{m}</strong><span>{x?.progress?.first_prefix_questions??0} / 1344 非确认题</span><span>原生采集：{x?.native_complete?'已完成':'进行中或待执行'}</span><span>统计预测：{x?.fit?.all_passed?'已完成非确认分析':'待完成'}</span><small>未量化 BF16 · 三模型串行</small></div>;})}</div>
    <p className="prefix-warning">这是研究进度，不是机制破解声明。首 token 常由 JSON 格式主导；状态拟合、完整答案、训练形成和预测器自身历史分别验收。当前六运行训练：{overview?.training?.all_passed?'已提交最终训练回执':'尚未提交最终训练回执'}。</p>
    {error&&<p role="alert" className="prefix-error">{error}</p>}
    <div className="prefix-controls"><label>自然研究模型<select aria-label="自然研究模型" value={model} onChange={e=>setModel(e.target.value)}>{MODELS.map(m=><option key={m}>{m}</option>)}</select></label></div>
    <h3>完整自由生成：内容、格式与停止分开</h3>
    {item.behavior?.length?<div className="prefix-table"><table><thead><tr><th>划分 / 语料</th><th>题数</th><th>保守匹配且停止</th><th>JSON 格式</th><th>原生 EOS / 截断</th></tr></thead><tbody>{item.behavior.map(r=><tr key={r.split+r.cohort}><td>{r.split} / {r.cohort}</td><td>{r.questions}</td><td>{r.counts.exact_and_stopped}</td><td>{r.counts.strict_JSON_array}</td><td>{r.counts.natural_EOS} / {r.counts.cap_censored}</td></tr>)}</tbody></table></div>:<p>完整行为汇总尚未提交，不从进度或首 token 推算答案能力。</p>}
    <h3>未参与选择的篇章：全部坐标预测误差</h3><p>当前主规则：{item.primary_rule||'尚未选择'}。下表为两个语料等权平均；语境内 MSE 衡量四问题间变化，零变化误差为基线。统计预测不证明唯一的参数机制，也不等于能自行答题。</p>
    {fit.length?<div className="prefix-table"><table><thead><tr><th>冻结路线</th><th>整体 MSE</th><th>语境内 MSE</th><th>零变化 MSE</th><th>验证尺度归一目标</th></tr></thead><tbody>{fit.map(r=>{const s=r.summary.equal_cohort;return <tr key={r.variant}><td>{r.variant}</td><td>{fmt(s.absolute_MSE)}</td><td>{fmt(s.within_MSE)}</td><td>{fmt(s.zero_change_MSE)}</td><td>{fmt(s.normalized_selection_objective)}</td></tr>;})}</tbody></table></div>:<p>尚无已提交的冻结诊断结果。</p>}
    {item.native_history_prediction?.complete&&<div data-testid="natural-history-prediction"><h3>首步规则能否跨生成步复用？</h3>
      <p>新补充诊断：使用原模型真实历史上当时已知的早层输入，原规则保持冻结，未新增训练。不是预测器自行生成；因而这里出现的误差不能归因于预测器之前已生成错误 token。共核对 {item.native_history_prediction.native_generated_tokens} 个原生步骤，包含验证与诊断；下表只列诊断结果。</p>
      <div className="prefix-table"><table data-testid="natural-history-comparison"><thead><tr><th>生成索引分组</th><th>主规则 MSE</th><th>固定训练均值 MSE</th><th>零向量参考 MSE</th><th>置乱规则 MSE</th><th>早层查询标准化 RMS</th></tr></thead><tbody>{item.native_history_prediction.summaries.map(r=>{const p=item.native_history_prediction.paired_comparisons.find(x=>x.bin===r.bin)?.primary_minus_target_shuffle_control?.equal_cohort;const s=r.summary.equal_cohort;return <tr key={r.bin}><td>{r.bin}</td><td>{fmt(s.prediction_MSE)}</td><td>{fmt(s.training_mean_MSE)}</td><td>{fmt(s.native_norm_squared_mean)}</td><td>{fmt(p?.right_mean)}</td><td>{fmt(s.query_standardized_RMS)}</td></tr>;})}</tbody></table></div>
      <p>先在每题内部对相应步骤平均，再对篇章、语料等权平均。晚步分组仅包含仍在生成的题目，不能作为无条件时间曲线；查询尺度来自首步训练坐标，不是语义距离。固定训练均值也不是具备生成位置知识的强基线。完整后续复用失败时，首步的已成立结果仍保留。</p>
      <p>零向量参考误差等于已保存的原生 postnorm 坐标平方均值，现补充显示这一描述性参照，未新拟合或重选主规则。零向量不是模型真实归一化状态，也不是具有语言能力的预测器；它用于避免把胜过失配的首步均值误解为低误差。</p>
      <Evidence title="原生历史配对区间、全部分组及来源回执" data={item.native_history_prediction}/>
    </div>}
    <RelationGeometry model={model} refresh={refresh} summary={item.relation_geometry}/>
    <SourceCoupling model={model} refresh={refresh}/>
    <NaturalAnalyses model={model} refresh={refresh} confirmationOpen={overview?.confirmation_open}/>
    <NativeFields model={model} refresh={refresh}/><Aggregate model={model} refresh={refresh} figures={item.figures||[]}/>
    <Evidence title="自然研究实时进度与各阶段实际回执" data={overview}/>
  </section>;
}
