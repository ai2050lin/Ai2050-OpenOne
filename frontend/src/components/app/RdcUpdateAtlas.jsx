import {useEffect,useState} from 'react';
import {Field} from './RdcJointAtlas.jsx';
import './RdcPrefixAtlas.css';
import './RdcJointAtlas.css';
import './RdcLawAtlas.css';
import './RdcBindingAtlas.css';
import './RdcUpdateAtlas.css';

const API=import.meta.env.VITE_RDC_UPDATE_API||'http://127.0.0.1:5002/api/rdc-update';
async function get(path,params){const r=await fetch(API+path+(params?'?'+new URLSearchParams(params):''));if(!r.ok)throw new Error((await r.json().catch(()=>({}))).detail||r.statusText);return r.json();}
function ErrorLine({error}){return error?<p role="alert">{error}</p>:null;}
function Evidence({title,data,children}){return <details><summary>{title}</summary>{children}<pre>{JSON.stringify(data??{},null,2)}</pre></details>;}

function SourceFields(){
  const [model,setModel]=useState('qwen4'),[samples,setSamples]=useState([]),[cohort,setCohort]=useState('gum'),[split,setSplit]=useState('new_connected'),[sample,setSample]=useState('');
  const [mode,setMode]=useState('all_layers'),[anchor,setAnchor]=useState(0),[layer,setLayer]=useState(12),[view,setView]=useState('raw');
  const [field,setField]=useState(null),[detail,setDetail]=useState(null),[graph,setGraph]=useState(null),[error,setError]=useState('');
  const [block,setBlock]=useState(16),[unit,setUnit]=useState(0),[input,setInput]=useState(0),[output,setOutput]=useState(0),[scalar,setScalar]=useState(null);
  useEffect(()=>{let active=true;get('/samples',{model}).then(v=>{if(active){setSamples(v);setError('');}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[model]);
  const cohorts=[...new Set(samples.map(r=>r.cohort))].sort(),actualCohort=cohorts.includes(cohort)?cohort:cohorts[0]||'';
  const splits=[...new Set(samples.filter(r=>r.cohort===actualCohort).map(r=>r.split))],actualSplit=splits.includes(split)?split:splits[0]||'';
  const filtered=samples.filter(r=>r.cohort===actualCohort&&r.split===actualSplit);
  const chosen=filtered.some(r=>r.sample_id===sample)?sample:filtered[0]?.sample_id||'',row=filtered.find(r=>r.sample_id===chosen),actual=Math.min(anchor,(row?.anchors?.length||1)-1);
  const identity=JSON.stringify([model,chosen,mode,actual,layer,view]),sourceID=JSON.stringify([model,chosen]),graphID=JSON.stringify([chosen,actual]),scalarID=JSON.stringify([model,chosen,block,actual,unit,input,output]);
  async function query(){try{const [value,source]=await Promise.all([get('/field',{sample:chosen,model,mode,anchor:actual,layer,view}),get('/sample',{sample:chosen,model})]);setField({identity,value});setDetail({id:sourceID,value:source});setError('');}catch(e){setField(null);setError(e.message);}}
  async function queryGraph(){try{setGraph({id:graphID,value:await get('/graph',{sample:chosen,anchor:actual})});setError('');}catch(e){setGraph(null);setError(e.message);}}
  async function queryScalar(){try{setScalar({id:scalarID,value:await get('/scalar',{sample:chosen,block,anchor:actual,unit,input_coordinate:input,output_coordinate:output})});setError('');}catch(e){setScalar(null);setError(e.message);}}
  const d=detail?.id===sourceID?detail.value:null,s=scalar?.id===scalarID?scalar.value:null;
  return <section id="update-sources"><h2>外部语言材料与完整原生坐标</h2><p>自然语料、交错操作和五类双语材料保留同一家族的稳定身份。来源、层、token、坐标与 MLP 单元是不同的轴；同一个坐标编号跨模型不代表相同功能。</p>
    <div className="prefix-controls"><label>模型<select aria-label="模型" value={model} onChange={e=>{setModel(e.target.value);setSamples([]);setField(null);setDetail(null);}}>{['qwen4','qwen14','glm4'].map(m=><option key={m}>{m}</option>)}</select></label>
      <label>语言模式族<select value={actualCohort} onChange={e=>setCohort(e.target.value)}>{cohorts.map(c=><option key={c}>{c}</option>)}</select></label>
      <label>材料划分<select value={actualSplit} onChange={e=>setSplit(e.target.value)}>{splits.map(s0=><option key={s0}>{s0}</option>)}</select></label>
      <label className="prefix-wide">来源样本<select value={chosen} onChange={e=>setSample(e.target.value)}>{filtered.map(r=><option key={r.sample_id} value={r.sample_id}>{r.sample_id} · {r.tokens} tokens · {r.captured?'已采集':'未提交'}</option>)}</select></label>
      <label>场范围<select value={mode} onChange={e=>setMode(e.target.value)}><option value="all_layers">一个锚点的全部层</option><option value="sources">早期层全部来源 token</option><option value="fixture">预定全场样例：一层全部 token</option></select></label>
      <label>锚点<select value={actual} onChange={e=>setAnchor(Number(e.target.value))}>{row?.anchors?.map((p,i)=><option key={i} value={i}>token {p}</option>)}</select></label>
      <label>样例层 H<input type="number" min={0} max={80} value={layer} onChange={e=>setLayer(Number(e.target.value))}/></label>
      <label>数值视图<select aria-label="数值视图" value={view} onChange={e=>setView(e.target.value)}><option value="raw">原始值</option><option value="RMS">逐向量 RMS</option></select></label>
      <button disabled={!row?.captured} onClick={query}>读取完整坐标场</button><button disabled={!row?.captured||model!=='qwen4'} onClick={queryGraph}>读取候选有向来源图</button></div>
    <ErrorLine error={error}/>{!filtered.length&&<p>该模型没有登记的匹配材料；不会填入其他模型的数据。</p>}
    {model!=='qwen4'&&<p>大模型保持原始 BF16，使用带独立位置和掩码的批处理以减少卸载开销。批量与单样本数值不保证逐位相同；对应检查保存在当前模型的 shape_audit，原始记录注明执行形状。</p>}
    {d&&<><blockquote>{d.original_text||d.text}</blockquote><p>{d.source_group} · {d.novelty||d.scope}</p><Evidence title="原始材料、分词、外部关系和当前模型记录" data={d}/></>}
    <Field data={field?.identity===identity?field.value:null}/>
    {model==='qwen4'&&graph?.id===graphID&&<><p>下图的列是候选 head token，不是激活坐标。仅使用已知前缀与英文训练集拟合的读出；不是模型原生 attention。</p><Field data={graph.value}/></>}
    <h3>任意输入坐标 → MLP 单元 → 输出坐标</h3><p>以下标量路径对应 Qwen3-4B。大模型的完整单元与场保存在原始数组区，没有把 4B 参数冒充为大模型参数。</p>
    <div className="prefix-controls"><label>原生 block<select value={block} onChange={e=>setBlock(Number(e.target.value))}>{[6,16,35].map(b=><option key={b}>{b}</option>)}</select></label>
      {[['输入坐标',input,setInput,2559],['MLP 单元',unit,setUnit,9727],['输出坐标',output,setOutput,2559]].map(([label,value,setter,max])=><label key={label}>{label}<input aria-label={label} type="number" min={0} max={max} value={value} onChange={e=>setter(Number(e.target.value))}/></label>)}
      <button disabled={!row?.captured||model!=='qwen4'} onClick={queryScalar}>读取原生标量路径</button></div>
    {s&&<><pre>{JSON.stringify(s.chain,null,2)}</pre><Field data={s.input_terms}/><Field data={s.unit_terms}/><p>{s.scope}</p></>}
  </section>;
}

function NativePaths({refresh}){
  const [items,setItems]=useState([]),[sample,setSample]=useState(''),[anchor,setAnchor]=useState(0),[block,setBlock]=useState(16),[source,setSource]=useState(0),[unit,setUnit]=useState(0),[output,setOutput]=useState(0),[data,setData]=useState(null),[error,setError]=useState('');
  useEffect(()=>{let active=true;get('/samples').then(v=>active&&setItems(v.filter(r=>r.native_path))).catch(e=>active&&setError(e.message));return()=>{active=false;};},[refresh]);
  const chosen=items.some(r=>r.sample_id===sample)?sample:items[0]?.sample_id||'',row=items.find(r=>r.sample_id===chosen),actual=Math.min(anchor,(row?.anchors?.length||1)-1),identity=JSON.stringify([chosen,actual,block,source,unit,output]);
  async function query(){try{setData({identity,value:await get('/native-path',{sample:chosen,anchor:actual,block,source_position:source,unit,output_coordinate:output})});setError('');}catch(e){setData(null);setError(e.message);}}
  const r=data?.identity===identity?data.value:null;
  return <section id="update-native"><h2>来源 token → attention 写入 → 全 MLP 读写</h2><p>24 个预先声明的跨族样本。保留全部来源、32 个 attention 头、2560 个坐标和 9728 个 MLP 单元。这里使用已经观察到的 attention、归一化分母和门值核算计算来源；它不是提前预测，也不是唯一的因果归因。</p>
    <div className="prefix-controls"><label className="prefix-wide">原生路径样本<select value={chosen} onChange={e=>setSample(e.target.value)}>{items.map(r0=><option key={r0.sample_id} value={r0.sample_id}>{r0.cohort} · {r0.split} · {r0.sample_id}</option>)}</select></label>
      <label>路径 block<select value={block} onChange={e=>setBlock(Number(e.target.value))}>{[16,35].map(b=><option key={b}>{b}</option>)}</select></label>
      <label>路径锚点<select value={actual} onChange={e=>setAnchor(Number(e.target.value))}>{row?.anchors?.map((p,i)=><option key={i} value={i}>token {p}</option>)}</select></label>
      {[['来源 token 位置',source,setSource,row?.anchors?.[actual]||0],['路径 MLP 单元',unit,setUnit,9727],['路径输出坐标',output,setOutput,2559]].map(([label,value,setter,max])=><label key={label}>{label}<input type="number" min={0} max={max} value={value} onChange={e=>setter(Number(e.target.value))}/></label>)}
      <button disabled={!chosen} onClick={query}>核算原生来源路径</button></div><ErrorLine error={error}/>{!items.length&&<p>原生路径尚未提交；完成后点击页面顶部“刷新运行结果”。</p>}
    {r&&<><pre>{JSON.stringify(r.chain,null,2)}</pre><Field data={r.attention_write}/><Field data={r.unit_reads}/><Field data={r.MLP_write}/><p>{r.scope}</p></>}
  </section>;
}

function Gradient(){
  const [query,setQuery]=useState(0),[part,setPart]=useState('content'),[unit,setUnit]=useState(0),[input,setInput]=useState(0),[output,setOutput]=useState(0),[data,setData]=useState(null),[error,setError]=useState('');
  const identity=JSON.stringify([query,part,unit,input,output]);
  async function load(){try{setData({identity,value:await get('/gradient',{query,part,unit,input_coordinate:input,output_coordinate:output})});setError('');}catch(e){setData(null);setError(e.message);}}
  const r=data?.identity===identity?data.value:null;
  return <section id="update-gradient"><h2>全参数更新：内容与格式不等于独立模块</h2><p>完整词表损失 = 候选数字集合内损失 + 输出落入数字集合的格式损失。梯度相加，但两部分通常不正交。格式约束只针对声明的训练目标与局部导数，不能直接命名为“纯语义空间”。</p>
    <div className="prefix-controls"><label>梯度目标<select value={part} onChange={e=>setPart(e.target.value)}>{['content','format','full'].map(p=><option key={p}>{p}</option>)}</select></label>
      {[['混合程序 query',query,setQuery,767],['梯度 MLP 单元',unit,setUnit,9727],['梯度输入坐标',input,setInput,2559],['梯度输出坐标',output,setOutput,2559]].map(([label,value,setter,max])=><label key={label}>{label}<input type="number" min={0} max={max} value={value} onChange={e=>setter(Number(e.target.value))}/></label>)}<button onClick={load}>读取全部参数行因子</button></div>
    <ErrorLine error={error}/>{r&&<><p>{r.material.source_group} · {r.material.representation} · {r.material.split} · 监督目标 {r.material.target}</p><pre>{JSON.stringify(r.scalars,null,2)}</pre><Field data={r.input_terms}/><Field data={r.output_terms}/><p>{r.scope}</p></>}
  </section>;
}

function Archive(){
  const [areas,setAreas]=useState([]),[area,setArea]=useState('language_analysis'),[files,setFiles]=useState([]),[file,setFile]=useState(''),[headers,setHeaders]=useState([]),[name,setName]=useState('');
  const [row,setRow]=useState(0),[column,setColumn]=useState(0),[data,setData]=useState(null),[error,setError]=useState('');
  useEffect(()=>{get('/areas').then(setAreas).catch(e=>setError(e.message));},[]);
  useEffect(()=>{let active=true;get('/files',{area}).then(v=>{if(active){setFiles(v);setFile(v[0]?.file||'');setHeaders([]);setName('');setRow(0);setColumn(0);setError('');}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[area]);
  useEffect(()=>{let active=true;if(file)get('/arrays',{area,file}).then(v=>{if(active){setHeaders(v);setName(v[0]?.array||'');setRow(0);setColumn(0);setError('');}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[area,file]);
  const header=headers.find(h=>h.array===name),width=header?.shape?.at(-1)||1,identity=JSON.stringify([area,file,name,row,column]);
  async function query(){try{setData({identity,value:await get('/array',{area,file,name,row_start:row,row_count:37,start:column,count:Math.min(width,8192)})});setError('');}catch(e){setData(null);setError(e.message);}}
  return <section id="update-archive"><h2>完整数组与可重算档案</h2><p>所有保存的数值档案均可查询或下载。显示按原索引分页，不以 Top-K、阈值或降维删去低幅值背景。最后一轴可能是坐标、MLP 单元、词表、token 或样本，必须结合原张量名和形状阅读。</p>
    <div className="prefix-controls"><label className="prefix-wide">研究目录<select value={area} onChange={e=>{setArea(e.target.value);setFile('');setName('');setHeaders([]);}}>{areas.map(a=><option key={a.area} value={a.area}>{a.area||'(root)'} ({a.files})</option>)}</select></label>
      <label className="prefix-wide">数值档案<select value={file} onChange={e=>setFile(e.target.value)}>{files.map(f=><option key={f.file}>{f.file}</option>)}</select></label>
      <label>原张量<select value={name} onChange={e=>{setName(e.target.value);setRow(0);setColumn(0);}}>{headers.map(h=><option key={h.array} value={h.array}>{h.array} [{h.shape.join('×')}]</option>)}</select></label>
      <label>起始展平行<input type="number" min={0} value={row} onChange={e=>setRow(Number(e.target.value))}/></label><label>起始原生列<input type="number" min={0} value={column} onChange={e=>setColumn(Number(e.target.value))}/></label><button disabled={!name} onClick={query}>读取原序数组页</button>
      {file&&<a href={API+'/download?'+new URLSearchParams({area,file})}>下载完整 NPZ</a>}</div>
    <ErrorLine error={error}/>{data?.identity===identity&&<p>原张量 [{data.value.tensor_shape.join('×')}] · 当前行 [{data.value.row_start},{data.value.row_end}) / {data.value.total_rows}</p>}<Field data={data?.identity===identity?data.value:null}/>
  </section>;
}

function Behavior({refresh}){
  const [mode,setMode]=useState('own_history'),[items,setItems]=useState([]),[branch,setBranch]=useState('native'),[sample,setSample]=useState(''),[data,setData]=useState(null),[error,setError]=useState(''),[step,setStep]=useState(0);
  useEffect(()=>{let active=true;get('/behavior-index',{mode}).then(v=>{if(active){setItems(v);setBranch(v[0]?.branch||'native');setSample(v[0]?.sample_id||'');setError('');}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[mode,refresh]);
  const rows=items.filter(r=>r.branch===branch),chosen=rows.some(r=>r.sample_id===sample)?sample:rows[0]?.sample_id||'',identity=JSON.stringify([mode,branch,chosen]);
  async function query(){try{setData({identity,value:await get('/behavior',{mode,branch,sample:chosen})});setStep(0);setError('');}catch(e){setData(null);setError(e.message);}}
  const r=data?.identity===identity?data.value:null,actualStep=Math.min(step,(r?.generated_ids?.length||1)-1);
  return <section id="update-history"><h2>独立自身历史与原生同历史核对</h2><p>每个分支自行生成 token，不注入参考模型的 token、状态或 KV。原生同历史陪跑只做诊断，不回填分支。首 token、候选答案内评分、完整答案、停止和截断各自保留。</p>
    <div className="prefix-controls"><label>轨迹集合<select aria-label="轨迹集合" value={mode} onChange={e=>{setMode(e.target.value);setItems([]);setData(null);}}>{['own_history','same_history','long_answers','qwen4','qwen14','glm4'].map(m=><option key={m}>{m}</option>)}</select></label>
      <label>独立分支<select aria-label="独立分支" value={branch} onChange={e=>setBranch(e.target.value)}>{[...new Set(items.map(r0=>r0.branch))].map(b=><option key={b}>{b}</option>)}</select></label>
      <label className="prefix-wide">轨迹样本<select aria-label="轨迹样本" value={chosen} onChange={e=>setSample(e.target.value)}>{rows.map(r0=><option key={r0.sample_id} value={r0.sample_id}>{r0.cohort} · {r0.sample_id}</option>)}</select></label><button disabled={!chosen} onClick={query}>读取真实生成</button></div><ErrorLine error={error}/>
    {!rows.length&&<p>尚无匹配的已提交轨迹；不会用别的模型或分支代替。</p>}{r&&<><p>{r.material.source_group} · {r.material.original_text||r.material.text}</p><blockquote>{r.generated_text}</blockquote>
      <div className="update-facts"><span>终止答案解析正确：{String(r.answer_scoring?.conservative_final_correct??'未解析／不适用')}</span><span>答案正确且停止：{String(r.material.kind==='natural'?'不适用':r.answer_scoring?.parsed_and_stopped_correct)}</span><span>EOS：{String(r.answer_scoring?.EOS)}</span><span>长度上限截断：{String(r.answer_scoring?.censored)}</span><span>首次分叉：{r.first_divergence_step??'无／不适用'}</span></div><p>只解析完整短答或明确末尾答案标记；未解析和截断不等于内容答错。推理过程本身未评分。</p>
      <div className="update-facts"><span>格式包装复核正确：{String(r.format_aware_scoring?.conservative_final_correct??'未解析／不适用')}</span><span>复核答案正确且停止：{String(r.material.kind==='natural'?'不适用':r.format_aware_scoring?.parsed_and_stopped_correct)}</span><span>复核规则：{r.format_aware_scoring?.format_audit_method??'未提交'}</span></div><p>格式复核是同一次生成的次级事后审计：只补认明确末尾答案的 Markdown／公式包装或题目指定变量的完整值陈述。保留上面的原解析结果，不把评分修正当作模型能力提升。</p>
      {r.manual_terminal_adjudication&&<aside><p>残余终止答案人工复核：{r.manual_terminal_adjudication.answer??'仍不确定'} · 正确：{String(r.manual_terminal_adjudication.correct??'未定')}。这是主研究代理非盲、事后逐条复核，不改动两版解析器，也不评分推理过程。</p><blockquote>{r.manual_terminal_adjudication.quoted_terminal_text}</blockquote></aside>}
      <label className="update-step">已生成 token 步骤 {actualStep+1}/{r.generated_ids.length}<input type="range" min={0} max={r.generated_ids.length-1} value={actualStep} onChange={e=>setStep(Number(e.target.value))}/></label>
      <pre>{JSON.stringify(r.steps?.[actualStep]||{step:actualStep,token_id:r.generated_ids[actualStep]},null,2)}</pre>
      <Evidence title="当前已生成前缀 ID 与完整运行记录" data={{generated_prefix_ids:r.generated_ids.slice(0,actualStep+1),prompt_ids:r.generation_prompt_ids||r.prompt_ids,first:r.first,record:r}}/>
    </>}
  </section>;
}

export default function RdcUpdateAtlas(){
  const [summary,setSummary]=useState(null),[error,setError]=useState(''),[refresh,setRefresh]=useState(0);
  useEffect(()=>{let active=true;get('/overview').then(v=>{if(active){setSummary(v);setError('');}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[refresh]);
  return <main className="prefix-app law-app binding-app update-app"><header><small>RDC · DIRECTED RELATIONS / PARAMETER UPDATES / PREDICTIVE STATE</small><h1>关系、学习与自回归接续图谱</h1><p>共同现象 → 候选规律 → 原生参数结构 → 未见组合预测 → 训练形成证据</p><nav><a href="#update-evidence">证据与状态</a><a href="#update-sources">语言与全坐标场</a><a href="#update-native">来源读写</a><a href="#update-gradient">全参数梯度</a><a href="#update-archive">全部数组</a><a href="#update-history">自身历史</a><a href="#update-theory">统一拼图</a><a href="/rdc-binding">前轮图谱 ↗</a></nav></header>
    <ErrorLine error={error}/><section id="update-evidence" className="prefix-warning"><h2>四个 Phase，分别核对每一种证据</h2><p>“格式约束后的梯度”“看起来相近的 HiddenState”与“普适语义机制”不是同一件事。已观察到的局部参数效应、预测收益和失败条件分别保留；没有用新名称把有限结果升级为语言机制闭合。</p>
      <button onClick={()=>setRefresh(v=>v+1)}>刷新运行结果</button><div className="law-metrics"><div><b>{summary?.capture?.rows??'未提交'}</b><span>新自然与混合表达原生采集</span></div><div><b>{summary?.language?.rows??'未提交'}</b><span>五族中英关系表达</span></div><div><b>{summary?.middle?.runs?.length??'未提交'}</b><span>中层实际训练轨迹</span></div><div><b>{summary?.history?.trajectories??'进行中／未提交'}</b><span>独立自身历史轨迹</span></div></div>
      <Evidence title="附件审查：保留、收窄与纠错" data={summary?.contract?.corrections}/>
      <Evidence title="各结果实际提交状态、资源及最终审计" data={{completion:summary?.completion,scale:Object.fromEntries(Object.entries(summary?.scale||{}).map(([m,r])=>[m,{complete:!!r.timestamp,rows:r.rows,seconds:r.seconds,execution_shape:r.runtime?.execution_shape_protocol,shape_audit:r.execution_shape_audit}])),resources:summary?.resources,integrity:summary?.integrity}}/>
      <Evidence title="有向关系预测与同 RMS 新确认对照" data={{original:summary?.graph?.selected,confirmation:summary?.analysis,fresh:summary?.fresh_confirmation}}/>
      <Evidence title="参数效应预测、格式约束与原生中层训练" data={{forecast:summary?.forecast_audit,full_gradient_audit:summary?.autograd_audit,middle:summary?.middle}}/>
      <Evidence title="双语全坐标关系、正文锚点与词汇身份对照" data={{language:summary?.language,label_audit:summary?.language_logic,alignment_and_identity:summary?.language_identity,causal_prefix_numerics:summary?.causal_anchor,prefix_only_replay:summary?.causal_replay,prediction:summary?.language_prediction}}><p>“正文锚点”是正文内最后完整 token，可能是词而非句号，尚未读到问题与回答风格指令。同正文前缀在不同完整执行长度下有数值差异，不能解释成较晚指令的语义影响。仅前缀重放与等长度未来指令对照单独登记，预测器不重新拟合。词义和长距离角色材料在匹配当前 token 后没有可用对照；高相似度尚不能识别独立语义优势。这些诊断复用原语义组，不是新的独立确认。</p></Evidence>
      <Evidence title="三模型同批全坐标关系与执行形状对照" data={{matched_shape_analysis:summary?.scale_analysis,native_models:summary?.scale}}/>
    </section><SourceFields/><NativePaths refresh={refresh}/><Gradient/><Archive/><Behavior refresh={refresh}/>
    <section id="update-theory"><h2>RDC 统一理论、完整拼图与剩余边界</h2><p>恒等式、拟合模型、受限经验规律和未完成问题分别标记。有限阶矩的反例是合成状态证书；真实前缀实验另列，不能将两者混为模型遗忘的机制证明。</p>
      {!summary?.theory?.puzzles?.length&&<p>本轮完整理论账本尚未提交；不把待办写成已完成成果。</p>}
      <div className="prefix-table"><table><thead><tr><th>Phase</th><th>保留的核心拼图</th><th>硬边界</th><th>证据状态</th></tr></thead><tbody>{summary?.theory?.puzzles?.map((p,i)=><tr key={p.phase+'_'+i}><td>{p.phase}</td><td>{p.retained_puzzle}</td><td>{p.boundary}</td><td>{p.evidence_status}</td></tr>)}</tbody></table></div>
      {summary?.theory?.formulas?.map(f=><Evidence key={f.id} title={`${f.id} · ${f.kind}`} data={f}/>)}
      <Evidence title="有限矩边界、可达前缀探针和真实 KV 成本" data={{synthetic:summary?.moment_boundary,native:summary?.predictive_state}}/>
      <Evidence title="全部生成、同历史与来源核算结果" data={{own_history:summary?.history,formal_answer_scoring:summary?.answer_scoring,secondary_terminal_format_audit:summary?.terminal_format_audit,manual_residual_terminal_audit:summary?.manual_terminal_audit,long_answers:summary?.long_answers,same_history:summary?.same_history,native_paths:summary?.native_paths}}/>
    </section><section><h2>科学图：完整坐标、控制条件与实际误差</h2><p>二维图像缩放属于显示；数组、坐标索引、原值和分组信息保留。每张图标明原始值或归一化方式，不把图上距离当作真实语义几何。</p>{summary?.figures?.map(f=><figure key={f.path}><figcaption>{f.title}</figcaption><a href={API+'/figure/'+f.path} target="_blank" rel="noreferrer"><img loading="lazy" src={API+'/figure/'+f.path} alt={f.title}/></a><p>{f.scope}</p></figure>)}</section>
    <footer>仅以 GET 读取研究档案，不启动模型、修改参数或提交训练。原有 5001 应用与前轮路由保留；本页独立只读服务默认使用 5002。状态以实际结果与最终审计为准。</footer>
  </main>;
}
