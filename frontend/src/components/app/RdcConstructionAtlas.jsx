import {useEffect, useState} from 'react';
import {Field} from './RdcJointAtlas.jsx';
import {RuntimeFields,HistoryFindings,SelfHistory} from './RdcRuntimeAtlas.jsx';
import FormationAtlas from './RdcFormationAtlas.jsx';
import NaturalQuestions from './RdcNaturalQuestions.jsx';
import './RdcPrefixAtlas.css';
import './RdcJointAtlas.css';
import './RdcConstructionAtlas.css';

const API=import.meta.env.VITE_RDC_CONSTRUCTION_API||'http://127.0.0.1:5004/api/rdc-construction';
const MODELS=['qwen4','qwen14','glm4'];
const number=x=>x==null?'未提交':Number(x).toLocaleString('zh-CN');
const precise=x=>x==null?'未提交':x!==0&&Math.abs(x)<1e-5?x.toExponential(4):x.toFixed(6);
const interval=x=>x?.interval95?`${precise(x.mean)} [${precise(x.interval95[0])}, ${precise(x.interval95[1])}]`:'未提交';
async function get(path, parameters={}){
  const r=await fetch(API+path+'?'+new URLSearchParams(parameters));
  if(!r.ok)throw new Error((await r.json().catch(()=>({}))).detail||r.statusText);
  return r.json();
}
function Evidence({title,data}){return <details><summary>{title}</summary><pre>{JSON.stringify(data??{},null,2)}</pre></details>;}
function ErrorLine({error}){return error?<p role="alert" className="prefix-error">{error}</p>:null;}

function Findings({summary}){
  const [model,setModel]=useState('qwen4');
  const parts=summary?.models?.[model]||{};
  const anova=parts.analysis?.anova_summaries?.filter(r=>r.query_language==='all'&&r.view==='per_vector_RMS')||[];
  const relation=parts.analysis?.native_relation_summary||[];
  const primary=parts.fit?.primary||[];
  const compilation=parts.compilation?.summaries?.filter(r=>r.family==='all')||[];
  return <section id="construction-findings"><h2>共同现象、候选规则与未见预测</h2>
    <p>这里分开显示状态交互、原生首位置判断与冻结预测结果。全坐标计数不是独立样本数；置信区间按共同语义组计算。</p>
    <div className="prefix-controls"><label>结果模型<select aria-label="结果模型" value={model} onChange={e=>setModel(e.target.value)}>{MODELS.map(m=><option key={m}>{m}</option>)}</select></label></div>
    <h3>同词袋关系：原生 B1 首位置，而非完整自由生成</h3>
    {model==='glm4'&&<p className="prefix-warning">本轮 GLM4 的 320 个原生首 token 均为换行符（ID 198）。此处的首位置正确数为零不代表完整回答能力为零；完整自然生成另行核对，不能把格式位置当作答案位置。</p>}
    {!relation.length?<p>关系分析尚未提交。</p>:<div className="prefix-table"><table><thead><tr><th>关系族</th><th>配对数</th><th>双边首 token 正确</th><th>答案方向分离度</th><th>早层 Q 变化 MSE</th></tr></thead><tbody>{relation.map(r=><tr key={r.family}><td>{r.family}</td><td>{r.pairs}</td><td>{r.B1_both_argmax_correct}/{r.pairs}</td><td>{interval(r.answer_direction_separation)}</td><td>{precise(r.early_Q_MSE_mean)}</td></tr>)}</tbody></table></div>}
    <p>肯定/否定方向按每一对的实际外部答案对齐。相同词袋只排除纯词袋解释，不能自动排除位置、局部序列或模板规则。</p>
    <h3>前缀 × 查询的条件交互：所有原生坐标</h3>
    {!anova.length?<p>交互分解尚未提交。</p>:<div className="prefix-table"><table><thead><tr><th>原生边界</th><th>总变化能量</th><th>前缀主效应</th><th>查询主效应</th><th>交互</th><th>交互占比</th></tr></thead><tbody>{anova.map(r=><tr key={r.boundary}><td>{r.boundary}</td><td>{precise(r.total_coordinate_variance_mean)}</td><td>{precise(r.prefix_main_effect_energy_mean)}</td><td>{precise(r.query_main_effect_energy_mean)}</td><td>{precise(r.interaction_energy_mean)}</td><td>{r.interaction_fraction==null?'无定义':(100*r.interaction_fraction).toFixed(3)+'%'}</td></tr>)}</tbody></table></div>}
    <p>表中每个完整向量先作 RMS 归一化，再分解 Y=μ+A+B+C。原值、语言内查询与全坐标能量均保留；接近零的负能量属于记录中的浮点抵消误差。C 是观测网格中的统计交互，不是已确认的语义模块。</p>
    <h3>新 case × 未见 query：提前预测关系变化</h3>
    {!primary.length?<p>全跨坐标预测尚未提交，不能将观察结果当成预测成功。</p>:<div className="prefix-table"><table><thead><tr><th>目标</th><th>零变化 MSE</th><th>真实配对拟合 MSE</th><th>零 − 真实</th><th>配对置乱 − 真实</th></tr></thead><tbody>{primary.map(r=><tr key={r.target}><td>{r.target}</td><td>{interval(r.zero_change_MSE)}</td><td>{interval(r.true_correspondence_MSE)}</td><td>{interval(r.zero_minus_true_MSE)}</td><td>{interval(r.shuffled_minus_true_MSE)}</td></tr>)}</tbody></table></div>}
    <p>此输入为真实查询经过第 0 块后的 H1，不含待预测后层。全矩阵解保留全部坐标；同输入逐坐标规则和置乱规则另行比较。统计拟合不等于提取了唯一因果机制。</p>
    {!!parts.compilation?.same_input_pair_prediction?.length&&<><h3>相同 H1 输入：跨坐标是否优于逐坐标</h3><div className="prefix-table"><table><thead><tr><th>目标</th><th>全矩阵 MSE</th><th>逐坐标 MSE</th><th>逐坐标 − 全矩阵</th></tr></thead><tbody>{parts.compilation.same_input_pair_prediction.map(r=><tr key={r.target}><td>{r.target}</td><td>{interval(r.full_matrix_MSE)}</td><td>{interval(r.same_input_diagonal_MSE)}</td><td>{interval(r.diagonal_minus_full_MSE)}</td></tr>)}</tbody></table></div></>}
    {!!compilation.length&&<><h3>绝对状态预测 → 原生 Q → 完整词表</h3><p>配对变化预测加到无前缀锚点，是额外的外推假设。绝对逐坐标模型单独拟合，不使用该锚点；置乱结果接近时不能归因为关系特异性。KL 越小越好，argmax 一致率不是语言答案正确率。</p><div className="prefix-table"><table><thead><tr><th>冻结规则</th><th>Hearly MSE</th><th>postnorm MSE</th><th>全部 Q MSE</th><th>KL（原生‖预测）</th><th>argmax 一致</th></tr></thead><tbody>{compilation.map(r=><tr key={r.variant}><td>{r.variant}</td><td>{precise(r.metrics.Hearly_MSE.mean)}</td><td>{precise(r.metrics.postnorm_MSE.mean)}</td><td>{precise(r.metrics.Q_MSE.mean)}</td><td>{interval(r.metrics.KL_native_to_prediction)}</td><td>{(100*r.metrics.argmax_agreement.mean).toFixed(3)}%</td></tr>)}</tbody></table></div></>}
    <Evidence title="全坐标、同输入逐坐标与完整词表的全部结果" data={{fit:parts.fit,diagonal:parts.diagonal,compilation:parts.compilation}}/>
  </section>;
}

function Sources({summary,refresh}){
  const [model,setModel]=useState('qwen4'),[sample,setSample]=useState(''),[index,setIndex]=useState(null);
  const [revision,setRevision]=useState(0);
  const [mode,setMode]=useState('queries'),[boundary,setBoundary]=useState('postnorm'),[query,setQuery]=useState(0),[block,setBlock]=useState('early'),[view,setView]=useState('raw');
  const [data,setData]=useState(null),[detail,setDetail]=useState(null),[pair,setPair]=useState(null),[interaction,setInteraction]=useState(null),[prediction,setPrediction]=useState(null),[error,setError]=useState('');
  useEffect(()=>{let alive=true;get('/samples',{model}).then(rows=>alive&&setIndex({model,rows})).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[model,refresh]);
  const options=index?.model===model?index.rows:[],chosen=options.some(r=>r.sample_id===sample)?sample:options[0]?.sample_id||'';
  const row=options.find(r=>r.sample_id===chosen),native=summary?.models?.[model]?.capture;
  const early=native?.early??(model==='qwen4'?12:13),actualBlock=block==='early'?early:0;
  const boundaries=['postnorm',...(native?.selected_query_boundaries||[0,1,early])];
  const id=JSON.stringify([model,chosen,mode,boundary,query,actualBlock,view,revision]);
  function change(setter,value){setter(value);setRevision(v=>v+1);}
  const sourceID=JSON.stringify([model,chosen]);
  async function load(){try{const [field,source]=await Promise.all([get('/field',{model,sample:chosen,mode,boundary,query,block:actualBlock,view}),get('/sample',{model,sample:chosen})]);setData({id,value:field});setDetail({id:sourceID,value:source});setError('');}catch(e){setData(null);setError(e.message);}}
  async function loadPair(){try{setPair({id,value:await get('/pair',{model,sample:chosen,boundary,query,view})});setError('');}catch(e){setPair(null);setError(e.message);}}
  async function loadInteraction(){try{setInteraction({id,value:await get('/interaction',{model,sample:chosen,boundary,view})});setError('');}catch(e){setInteraction(null);setError(e.message);}}
  async function predict(){try{setPrediction({id,value:await get('/prediction',{model,sample:chosen,query,target:boundary==='postnorm'?'postnorm':'Hearly'})});setError('');}catch(e){setPrediction(null);setError(e.message);}}
  return <section id="construction-fields"><h2>原始语句 → 查询构造 → 全坐标交互</h2>
    <div className="prefix-controls"><label>原生模型<select aria-label="原生模型" value={model} onChange={e=>change(setModel,e.target.value)}>{MODELS.map(m=><option key={m}>{m}</option>)}</select></label>
      <label className="prefix-wide">原始表达<select aria-label="原始表达" value={chosen} onChange={e=>change(setSample,e.target.value)}>{options.map(r=><option key={r.sample_id} value={r.sample_id}>{r.family} · case{r.case} · {r.language} · 世界{r.world} · {r.split} · {r.captured?'已采集':'待采集'}</option>)}</select></label>
      <label>原生字段<select aria-label="原生字段" value={mode} onChange={e=>change(setMode,e.target.value)}><option value="queries">100 查询完整 HiddenState</option><option value="prefix">前缀全部层锚点</option><option value="q_before_rope">实际 Q（RoPE 前）</option><option value="q_input">Q 投影输入</option><option value="q_projected">Q 投影原值</option><option value="attention">全部来源 attention</option><option value="attention_output">attention 写回</option></select></label>
      <label>查询边界<select aria-label="查询边界" value={boundary} onChange={e=>change(setBoundary,e.target.value)}>{boundaries.map(b=><option key={b}>{b}</option>)}</select></label>
      <label>实际 attention 块<select value={block} onChange={e=>change(setBlock,e.target.value)}><option value="early">早层 block{early}</option><option value="0">首块 block0</option></select></label>
      <label>查询编号<input aria-label="查询编号" type="number" min={0} max={99} value={query} onChange={e=>change(setQuery,Number(e.target.value))}/></label>
      <label>数值视图<select aria-label="数值视图" value={view} onChange={e=>change(setView,e.target.value)}><option value="raw">原始值</option><option value="RMS">完整向量 RMS</option></select></label>
      <button disabled={!row?.captured} onClick={load}>读取原生场</button><button disabled={!row?.captured} onClick={loadPair}>读取配对对照</button><button disabled={!row?.captured} onClick={loadInteraction}>读取条件交互</button><button disabled={!row?.captured} onClick={predict}>读取冻结关系预测</button>
    </div>
    <p>全部原生坐标保留。attention 的横轴是来源位置，不是残差坐标；Q 头分量、MLP 单元和标量参数索引分别标注。100 个后缀是诊断查询，不保证接到每段文本后都自然。</p>
    <ErrorLine error={error}/>
    {detail?.id===sourceID&&<><blockquote>{detail.value.material.original_text}</blockquote><p>外部答案：{detail.value.material.target}；原生完整输入共 {detail.value.material.prompt_ids.length} token。{detail.value.material.novelty}</p><Evidence title="实际模型输入、分词、查询及逐位数值核对" data={detail.value}/></>}
    {data?.id===id&&<p>{data.value.axes}</p>}<Field data={data?.id===id?data.value:null}/>
    {pair?.id===id&&<><h3>相同 token 多重集的两种关系</h3>{pair.value.materials.map(r=><blockquote key={r.sample_id}>世界{r.world}：{r.original_text}<br/>外部规则答案：{r.target}</blockquote>)}<Field data={pair.value.field}/><p>{pair.value.scope}</p></>}
    {interaction?.id===id&&<><h3>条件交互 C：不由两个主效应相加解释的部分</h3><p>{interaction.value.scope} · 分解前视图：{interaction.value.view_before_decomposition}</p><Field data={interaction.value.interaction}/><Field data={interaction.value.mean_terms}/></>}
    {prediction?.id===id&&<><h3>冻结 H1 算子的关系变化预测</h3><p>实际预测目标：{prediction.value.target}。选其他层时，此按钮仍查询冻结的 Hearly 目标，不声称任意层预测。</p><Field data={prediction.value}/><p>零变化误差：{precise(prediction.value.zero_change_MSE)}；来源/查询划分：{prediction.value.source_split}/{prediction.value.query_split}。</p></>}
  </section>;
}

function Parameters(){
  const [model,setModel]=useState('qwen4'),[block,setBlock]=useState(16),[unit,setUnit]=useState(0),[input,setInput]=useState(0),[inputR,setInputR]=useState(1),[output,setOutput]=useState(0);
  const [value,setValue]=useState(null),[meta,setMeta]=useState(null),[error,setError]=useState('');
  const id=JSON.stringify([model,block,unit,input,inputR,output]);
  async function load(){try{const [v,m]=await Promise.all([get('/parameter',{model,block,unit,input,input_r:inputR,output}),get('/parameters',{model,block})]);setValue({id,value:v});setMeta({id,value:m});setError('');}catch(e){setValue(null);setError(e.message);}}
  return <section id="construction-parameters"><h2>固定参数骨架：读取、条件与写回</h2><p>直接查询原 checkpoint 的完整 gate/up/down 向量。Γ[k,j,i,r]=Wd[j,k]Wg[k,i]Wu[k,r] 是精确的参数乘积表示，不是“一个参数对应一个概念”的证明，也不需要物化或截断巨大张量。</p>
    <div className="prefix-controls"><label>参数模型<select aria-label="参数模型" value={model} onChange={e=>setModel(e.target.value)}>{MODELS.map(m=><option key={m}>{m}</option>)}</select></label>
      {[["参数块",block,setBlock],["MLP 单元",unit,setUnit],["输入坐标 i",input,setInput],["输入坐标 r",inputR,setInputR],["输出坐标 j",output,setOutput]].map(([label,v,set])=><label key={label}>{label}<input aria-label={label} type="number" min={0} value={v} onChange={e=>set(Number(e.target.value))}/></label>)}<button onClick={load}>查询原生参数因子</button></div>
    <ErrorLine error={error}/>{value?.id===id&&<><p>Wg={precise(value.value.W_gate_k_i)}；Wu={precise(value.value.W_up_k_r)}；Wd={precise(value.value.W_down_j_k)}；Γ={precise(value.value.Gamma_k_j_i_r)}。</p><Field data={value.value.factors}/><p>{value.value.scope}</p></>}
    {meta?.id===id&&<Evidence title="所有原生参数地址及当前层完整参数表" data={meta.value}/>}
  </section>;
}

function AttentionParameters(){
  const [model,setModel]=useState('qwen4'),[block,setBlock]=useState(12),[head,setHead]=useState(0),[input,setInput]=useState(0),[inputR,setInputR]=useState(1),[output,setOutput]=useState(0),[qp,setQP]=useState(113),[kp,setKP]=useState(17);
  const [value,setValue]=useState(null),[error,setError]=useState('');
  const id=JSON.stringify([model,block,head,input,inputR,output,qp,kp]);
  async function load(){try{setValue({id,value:await get('/attention-parameter',{model,block,head,input,input_r:inputR,output,query_position:qp,key_position:kp})});setError('');}catch(e){setValue(null);setError(e.message);}}
  const selected=value?.id===id?value.value:null;
  return <section id="construction-attention-parameters"><h2>原生 QK / OV：位置与条件不能被省略</h2><p>QK 展示带 RoPE 和 norm 增益的分子系数；Qwen 还需要两个随输入变化的头 RMS 分母，GLM 则另含 Q/K/V 偏置。OV 是读取到写回的参数乘积。二者均不是语义边，也不是完整 attention 概率。</p>
    <div className="prefix-controls"><label>attention 参数模型<select aria-label="attention 参数模型" value={model} onChange={e=>setModel(e.target.value)}>{MODELS.map(m=><option key={m}>{m}</option>)}</select></label>
      {[["attention 参数块",block,setBlock],["查询头",head,setHead],["Q/V 输入坐标",input,setInput],["K 输入坐标",inputR,setInputR],["写回坐标",output,setOutput],["查询位置",qp,setQP],["来源位置",kp,setKP]].map(([label,v,set])=><label key={label}>{label}<input aria-label={label} type="number" min={0} value={v} onChange={e=>set(Number(e.target.value))}/></label>)}<button onClick={load}>读取 QK / OV 因子</button></div><ErrorLine error={error}/>
    {selected&&<><p>query head {selected.metadata.query_head} → KV head {selected.metadata.kv_head}；QK 分子系数 {precise(selected.qk_numerator_coefficient)}；OV 系数 {precise(selected.ov_coefficient)}。横轴：全部 {selected.metadata.head_dim} 个头分量，而非残差坐标。</p><Field data={selected.component_terms}/><Field data={selected.factors}/><p>{selected.scope}</p><Evidence title="精确参数地址、GQA 和条件边界" data={selected.metadata}/></>}
  </section>;
}

function NaturalInteractions({summary}){
  const natural=summary?.phase2746?.natural_interactions,scrutiny=summary?.phase2746?.natural_scrutiny;
  const global=natural?.reports?.filter(r=>r.layer==='postnorm'&&r.cohort==='all')||[];
  const matched=scrutiny?.matched_depth_reports?.filter(r=>r.weighting==='equal_document')||[];
  return <section id="construction-natural"><h2>自然语料：共同响应、条件交互与语料构成</h2><p>复用此前 10,000 个自然前缀、2,777 篇来源文档和 100 个固定查询，不计作新增模型样本。全部 2,560 个原生坐标参与分解；文档等权避免长文档重复窗口占据更大权重。交互占比不是“语义百分比”。</p>
    {!!global.length&&<div className="prefix-table"><table><thead><tr><th>权重</th><th>视图</th><th>末端交互 / 总变动</th><th>其中语料间查询差异</th></tr></thead><tbody>{global.map(r=>{const d=scrutiny?.cohort_decomposition?.find(x=>x.weighting===r.weighting&&x.view===r.view);return <tr key={r.weighting+r.view}><td>{r.weighting}</td><td>{r.view}</td><td>{(100*r.interaction_fraction).toFixed(4)}%</td><td>{d?`${(100*d.between_fraction_of_global_interaction).toFixed(4)}%`:'待核对'}</td></tr>;})}</tbody></table></div>}
    {!!matched.length&&<><h3>同一批 576 前缀的各层对照</h3><div className="prefix-table"><table><thead><tr><th>边界</th><th>视图</th><th>条件交互 / 总变动</th></tr></thead><tbody>{matched.map(r=><tr key={r.layer+r.view}><td>{r.layer}</td><td>{r.view}</td><td>{(100*r.interaction_fraction).toFixed(4)}%</td></tr>)}</tbody></table></div><p>不能把百万末端网格与较小的中间层样本直接当作同样材料的深度曲线；此处四个边界严格使用相同来源身份。最终 RMSNorm 前后是不同采集边界。</p></>}
    <Evidence title="全部语料、全坐标边际、权重与恒等式核对" data={{natural,scrutiny,parameter_structure:summary?.phase2746?.parameter_structure}}/>
  </section>;
}

function Archives({refresh}){
  const [prefix,setPrefix]=useState('capture/qwen4'),[offset,setOffset]=useState(0),[index,setIndex]=useState(null),[path,setPath]=useState(''),[arrays,setArrays]=useState(null),[name,setName]=useState('');
  const [row,setRow]=useState(0),[rowCount,setRowCount]=useState(32),[start,setStart]=useState(0),[count,setCount]=useState(8192),[field,setField]=useState(null),[error,setError]=useState('');
  const listID=JSON.stringify([prefix,offset,refresh]);
  useEffect(()=>{let alive=true;get('/archives',{prefix,offset,limit:100}).then(v=>alive&&setIndex({id:listID,value:v})).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[prefix,offset,listID]);
  const options=index?.id===listID?index.value.rows:[],chosen=options.some(r=>r.path===path)?path:options[0]?.path||'';
  useEffect(()=>{if(!chosen)return;let alive=true;get('/arrays',{path:chosen}).then(v=>alive&&setArrays({path:chosen,value:v})).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[chosen]);
  const fields=arrays?.path===chosen?arrays.value:[],arrayName=fields.some(r=>r.name===name)?name:fields[0]?.name||'';
  const id=JSON.stringify([chosen,arrayName,row,rowCount,start,count]);
  async function load(){try{setField({id,value:await get('/array',{path:chosen,name:arrayName,row_start:row,row_count:rowCount,start,count})});setError('');}catch(e){setField(null);setError(e.message);}}
  return <section id="construction-archives"><h2>完整原始数组：按索引分页，不按幅值筛选</h2><p>原场、查询 Q/K/V、来源权重、交互边际、全矩阵系数及全部单元对照均可回查。ZIP 按显示行读取，不持久缓存巨型数组。展示窗口不改变分析所用坐标。</p>
    <div className="prefix-controls"><label>档案前缀<input aria-label="档案前缀" value={prefix} onChange={e=>{setPrefix(e.target.value);setOffset(0);}}/></label><label className="prefix-wide">数值档案<select aria-label="数值档案" value={chosen} onChange={e=>setPath(e.target.value)}>{options.map(r=><option key={r.path}>{r.path}</option>)}</select></label><label className="prefix-wide">原生数组<select aria-label="原生数组" value={arrayName} onChange={e=>setName(e.target.value)}>{fields.map(r=><option key={r.name} value={r.name}>{r.name} · {r.shape?.join(' × ')}</option>)}</select></label>
      {[["起始行",row,setRow,0,10000000],["行数",rowCount,setRowCount,1,128],["起始坐标",start,setStart,0,10000000],["坐标数",count,setCount,1,32768]].map(([label,v,set,min,max])=><label key={label}>{label}<input aria-label={label} type="number" min={min} max={max} value={v} onChange={e=>set(Number(e.target.value))}/></label>)}<button disabled={!arrayName} onClick={load}>读取原始索引窗口</button></div>
    <div className="prefix-controls"><button disabled={!offset} onClick={()=>setOffset(Math.max(0,offset-100))}>上一页档案</button><span>{offset+1}–{Math.min(offset+100,index?.value?.total||0)} / {index?.value?.total||0}</span><button disabled={offset+100>=(index?.value?.total||0)} onClick={()=>setOffset(offset+100)}>下一页档案</button></div>
    <ErrorLine error={error}/>{field?.id===id&&<p>原始形状 {field.value.tensor_shape?.join(' × ')}；行 {field.value.row_start}–{field.value.row_end}/{field.value.total_rows}。{field.value.scale_scope}</p>}<Field data={field?.id===id?field.value:null}/>
  </section>;
}

function NativeLanguage({summary,refresh}){
  const [model,setModel]=useState('glm4'),[sample,setSample]=useState(''),[index,setIndex]=useState(null),[value,setValue]=useState(null),[field,setField]=useState(null),[error,setError]=useState('');
  useEffect(()=>{let alive=true;get('/samples',{model}).then(rows=>alive&&setIndex({model,rows})).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[model,refresh]);
  const options=index?.model===model?index.rows:[],chosen=options.some(r=>r.sample_id===sample)?sample:options[0]?.sample_id||'';
  const id=JSON.stringify([model,chosen]),reports=summary?.models?.[model]?.native_language?.summary||[];
  const review=summary?.models?.[model]?.native_language_review;
  async function load(){try{const result=await get('/native-language',{model,sample:chosen});setValue({id,value:result});setField(null);setError('');}catch(e){setValue(null);setError(e.message);}}
  async function loadField(){try{setField({id,value:await get('/native-language-field',{model,sample:chosen})});setError('');}catch(e){setField(null);setError(e.message);}}
  return <section id="construction-language"><h2>原模型自主历史：格式、内容与停止</h2><p>同一批 320 个表达：Q4 复用已有生成；Q14/GLM 新增原生 BF16 贪心生成，128 token 上限。不是提取算法生成，不把截断当错误答案，也不把首换行当语义失败。</p><div className="prefix-controls"><label>生成模型<select aria-label="生成模型" value={model} onChange={e=>setModel(e.target.value)}>{MODELS.map(m=><option key={m}>{m}</option>)}</select></label><label className="prefix-wide">生成表达<select aria-label="生成表达" value={chosen} onChange={e=>setSample(e.target.value)}>{options.map(r=><option key={r.sample_id} value={r.sample_id}>{r.family} · case{r.case} · {r.language} · 世界{r.world}</option>)}</select></label><button disabled={!chosen} onClick={load}>读取原模型完整输出</button><button disabled={!chosen||model==='qwen4'} onClick={loadField}>读取逐步全部坐标</button></div><ErrorLine error={error}/>
    {!!reports.length&&<div className="prefix-table"><table><thead><tr><th>语言族</th><th>正确且停止 / 表达</th><th>双边正确且停止 / 配对</th><th>解析错误</th><th>EOS 未解析</th><th>截断</th><th>B1/B8 首 token 不同</th></tr></thead><tbody>{reports.map(r=><tr key={r.family}><td>{r.family}</td><td>{r.correct_and_stopped}/{r.expressions}</td><td>{r.both_worlds_correct_and_stopped}/{r.pairs}</td><td>{r.parsed_wrong}</td><td>{r.unparsed_EOS}</td><td>{r.censored}</td><td>{r.first_B1_B8_disagreements}</td></tr>)}</tbody></table></div>}
    {review?.reviewed_rows&&<><p>补充事后审阅：全部 {review.reviewed_rows} 条未解析 EOS 中，另有 {review.additional_explicit_correct_conclusions} 条结论正确、{review.additional_explicit_wrong_conclusions} 条结论错误，均违反只答是/否格式。主评分仍为 {review.unchanged_primary_correct_and_stopped}/320；事后结论口径为 {review.secondary_conclusion_and_stopped_count}/320。不是新增独立评测，也不评价整段推理链。</p><Evidence title="逐条完整文本与事后审阅依据" data={review}/></>}
    {value?.id===id&&<><blockquote>{value.value.material.original_text}</blockquote><p>外部目标：{value.value.material.target}；正确且停止：{String(value.value.record.answer_scoring.parsed_and_stopped_correct)}；实际 token 数：{value.value.record.generated_ids.length}；复用旧轨迹：{String(value.value.reused_prior_result)}。</p><blockquote>{value.value.record.generated_text}</blockquote><Evidence title="完整实际 token、评分及数值形状" data={value.value}/></>}
    {field?.id===id&&<p>{field.value.axes}</p>}<Field data={field?.id===id?field.value:null}/>
  </section>;
}

function Training({summary,refresh}){
  const [options,setOptions]=useState([]),[samples,setSamples]=useState([]),[variant,setVariant]=useState('matched_natural_target_2742_r0p10'),[sample,setSample]=useState('');
  const [loaded,setLoaded]=useState(null),[error,setError]=useState('');
  useEffect(()=>{let alive=true;Promise.all([get('/norm-variants'),get('/samples',{model:'qwen4'})]).then(([v,s])=>{if(alive){setOptions(v);setSamples(s);}}).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[refresh]);
  const chosen=samples.some(r=>r.sample_id===sample)?sample:samples[0]?.sample_id||'';
  const id=JSON.stringify([variant,chosen]);
  const analysis=summary?.norm;
  const calibration=analysis?.calibration?.prospective_reports?.filter(r=>r.cohort==='all')||[];
  const comparisons=analysis?.direction_comparisons?.filter(r=>r.primary_radius&&r.seed==='paired_mean_of_two_fixed_seeds')||[];
  const behavior=analysis?.behavior?.filter(r=>r.family==='all')||[];
  async function load(){try{const [a,b,s]=await Promise.all([get('/norm-behavior',{variant,sample:chosen}),get('/norm-behavior',{variant:'native',sample:chosen}),get('/sample',{model:'qwen4',sample:chosen})]);setLoaded({id,value:{variant:a,native:b,material:s.material}});setError('');}catch(e){setLoaded(null);setError(e.message);}}
  return <section id="construction-training"><h2>真实训练方向：幅度、概率校准与自主历史</h2>
    <p>23 个条件含 5 个旧原生/训练基线，以及 18 个新位移/方向条件。新条件匹配实际部署的 BF16 参数位移；它们复用已有真实训练方向，不冒充新增训练。首 token、完整答案、格式、停止与生成历史分叉分别保存。</p>
    <p>已提交参数条件：{number(summary?.norm_progress?.variants_completed)} / 23；完整实验：{summary?.norm_execution?.all_passed?'已完成':'进行中或待执行'}。</p>
    {!!calibration.length&&<><h3>全条件结果：自然内容评分与原生生成分开</h3><div className="prefix-table"><table><thead><tr><th>参数条件</th><th>自然 NLL · 96 文档</th><th>校准后 NLL</th><th>完整正确且停止 / 320</th><th>相对原生改变的输出</th><th>截断数</th></tr></thead><tbody>{calibration.map(r=>{const b=behavior.find(x=>x.variant===r.variant);return <tr key={r.variant}><td>{r.variant}</td><td>{interval(r.raw_NLL)}</td><td>{interval(r.joint_calibrated_NLL)}</td><td>{b?.correct_and_stopped??'未提交'}</td><td>{b?.changed_sequences_vs_native??'未提交'}</td><td>{b?.censored??'未提交'}</td></tr>;})}</tbody></table></div></>}
    {!!comparisons.length&&<><h3>预定主比较：实际位移范数 0.10，两个固定种子配对平均</h3><div className="prefix-table"><table><thead><tr><th>真实方向 − 对照</th><th>自然原始 NLL 差</th><th>校准后 NLL 差</th><th>关系 NLL 差</th><th>完整生成成功率差</th></tr></thead><tbody>{comparisons.map(r=><tr key={r.control}><td>{r.control}</td><td>{interval(r.natural_minus_control.natural_raw)}</td><td>{interval(r.natural_minus_control.natural_joint_calibrated)}</td><td>{interval(r.natural_minus_control.conditional_NLL)}</td><td>{interval(r.natural_minus_control.complete_success)}</td></tr>)}</tbody></table></div><p>NLL 差为负才表示真实方向更好。区间按文档/语义组计算，不是两次训练的种子总体区间；已暴露的材料不冒充全新确认。校准只用于评分，没有用于这里的生成。</p></>}
    <h3>逐条原生输出回查</h3><div className="prefix-controls"><label className="prefix-wide">训练参数条件<select aria-label="训练参数条件" value={variant} onChange={e=>setVariant(e.target.value)}>{options.map(r=><option key={r.variant.name}>{r.variant.name}</option>)}</select></label><label className="prefix-wide">训练检验表达<select aria-label="训练检验表达" value={chosen} onChange={e=>setSample(e.target.value)}>{samples.map(r=><option key={r.sample_id} value={r.sample_id}>{r.family} · case{r.case} · {r.language} · 世界{r.world}</option>)}</select></label><button disabled={!chosen} onClick={load}>读取真实生成对照</button></div>
    <ErrorLine error={error}/>{loaded?.id===id&&<><blockquote>{loaded.value.material.original_text}</blockquote><p>外部目标：{loaded.value.material.target}。以下为同样 B8 分组条件下各自生成的历史，没有注入答案或使用概率校准。</p>{[['native',loaded.value.native],['selected',loaded.value.variant]].map(([label,item])=><div key={label}><h4>{label==='native'?'原模型':'所选实际参数条件'}</h4><blockquote>{item.record.generated_text}</blockquote><p>正确且停止：{String(item.record.answer_scoring.parsed_and_stopped_correct)}；首个分叉位置：{item.record.first_divergence_from_native_B8??'完全相同'}；实际 token 数：{item.record.generated_ids.length}。</p><Evidence title="完整 token 身份、评分与执行形状" data={item}/></div>)}</>}
    <Evidence title="位移匹配、全部坐标、自然评分与自身历史结果" data={{analysis,execution:summary?.norm_execution,progress:summary?.norm_progress}}/>
  </section>;
}

export default function RdcConstructionAtlas(){
  const [summary,setSummary]=useState(null),[refresh,setRefresh]=useState(0),[error,setError]=useState('');
  useEffect(()=>{let alive=true;get('/overview').then(v=>{if(alive){setSummary(v);setError('');}}).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[refresh]);
  return <main className="prefix-app construction-app"><header><div><small>RDC · PHASE 2745 → 2748 · 原生参数与自然问题</small><h1>语境怎样构造查询</h1><p>从固定读写关系，到条件交互、自然问题、未见预测与真实训练证据。</p></div><button onClick={()=>setRefresh(v=>v+1)}>刷新已提交结果</button></header>
    <nav><a href="/rdc-query">前阶段百万查询图谱</a><a href="/rdc-binding">关系绑定研究</a><a href="#construction-natural-questions">自然问题研究</a><a href="#construction-findings">实际结果</a><a href="#construction-fields">原生场</a><a href="#construction-parameters">参数因子</a><a href="#construction-runtime">真实生成全场</a><a href="#construction-history-results">历史条件规律</a><a href="#construction-self-history">自行续写</a><a href="#construction-archives">全部档案</a></nav>
    <ErrorLine error={error}/><section className="prefix-warning"><h2>研究状态，不是完成声明</h2><p>当前状态：{summary?.current_status||'读取状态中'}。观察、恒等式、统计预测与机制证据分开记录；尚未破解整个语言编码机制。</p><div className="construction-cards">{MODELS.map(m=><div key={m}><strong>{m}</strong><span>{number(summary?.models?.[m]?.captured_rows)} / 320 原生来源</span><span>{number((summary?.models?.[m]?.captured_rows||0)*100)} 查询端点</span><small>原始 BF16 · 不量化 · 模型串行</small></div>)}</div><p>本次续研不沿用此前自行设定的 6 小时 / 12 GiB 上限；保留物理安全检查和逐项消耗记录。全部材料划分、已暴露历史及待执行任务可回查。历史队列失败记录保留，不覆盖之后手动恢复的实际结果。</p><Evidence title="整合计划、授权与历史执行队列" data={{plan:summary?.plan,queue:summary?.queue,protocol:summary?.protocol}}/></section>
    <NaturalQuestions refresh={refresh}/>
    <Findings summary={summary}/><Sources summary={summary} refresh={refresh}/><Parameters/>
    <NativeLanguage summary={summary} refresh={refresh}/><Training summary={summary} refresh={refresh}/>
    <NaturalInteractions summary={summary}/><AttentionParameters/>
    <RuntimeFields refresh={refresh}/><HistoryFindings summary={summary}/><SelfHistory summary={summary} refresh={refresh}/>
    <FormationAtlas summary={summary} refresh={refresh}/><Archives refresh={refresh}/><section><h2>证据审查与理论边界</h2><p>保留有数据支持的关系敏感性、全坐标分账与有限预测；不将“滑齿”“纯语义基”“完美同构”预写为实验结论。RDC 名称不变，公式恒等式不等于新的普遍语言定律。</p><Evidence title="三附件 24 条审查与逐条证据来源" data={summary?.review}/><Evidence title="理论拼图、公式与验证回执" data={{theory:summary?.theory,verification:summary?.verification}}/></section>
  </main>;
}
