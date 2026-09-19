import {useEffect,useMemo,useState} from 'react';
import {Field} from './RdcJointAtlas.jsx';
import './RdcPrefixAtlas.css';
import './RdcJointAtlas.css';
import './RdcLawAtlas.css';
import './RdcBindingAtlas.css';

const API='http://127.0.0.1:5001/api/rdc-binding';
async function get(path,params){const r=await fetch(API+path+(params?'?'+new URLSearchParams(params):''));if(!r.ok)throw new Error((await r.json().catch(()=>({}))).detail||r.statusText);return r.json();}
function ErrorLine({error}){return error?<p role="alert">{error}</p>:null;}
function Evidence({title,data,children}){return <details><summary>{title}</summary>{children}<pre>{JSON.stringify(data||{},null,2)}</pre></details>;}

function SourceFields({samples}){
  const [cohort,setCohort]=useState('gum'),[split,setSplit]=useState('connected_test'),[sample,setSample]=useState('');
  const [mode,setMode]=useState('all_layers'),[anchor,setAnchor]=useState(0),[layer,setLayer]=useState(12),[view,setView]=useState('raw');
  const [field,setField]=useState(null),[detail,setDetail]=useState(null),[roles,setRoles]=useState(null),[error,setError]=useState('');
  const [block,setBlock]=useState(16),[unit,setUnit]=useState(0),[input,setInput]=useState(0),[output,setOutput]=useState(0),[scalar,setScalar]=useState(null);
  const filtered=useMemo(()=>samples.filter(r=>r.cohort===cohort&&r.split===split),[samples,cohort,split]);
  const chosen=filtered.some(r=>r.sample_id===sample)?sample:filtered[0]?.sample_id||'',row=filtered.find(r=>r.sample_id===chosen);
  const actual=Math.min(anchor,(row?.anchors?.length||1)-1),identity=JSON.stringify([chosen,mode,actual,layer,view]);
  const pathID=JSON.stringify([chosen,block,actual,unit,input,output]);
  async function query(){try{const [value,source]=await Promise.all([get('/field',{sample:chosen,mode,anchor:actual,layer,view}),get('/sample',{sample:chosen})]);setField({identity,value});setDetail({id:chosen,source});setError('');}catch(e){setField(null);setError(e.message);}}
  async function roleQuery(){try{setRoles({id:chosen,value:await get('/roles',{sample:chosen})});setError('');}catch(e){setRoles(null);setError(e.message);}}
  async function scalarQuery(){try{setScalar({id:pathID,value:await get('/scalar',{sample:chosen,block,anchor:actual,unit,input_coordinate:input,output_coordinate:output})});setError('');}catch(e){setScalar(null);setError(e.message);}}
  return <section id="binding-source"><h2>真实输入 → 全坐标场 → 来源与参数</h2><p>自然语料发现集链接前轮原场；新连接组合、可执行程序与六步确认集保留独立身份。H12 的所有来源均保留。原生坐标、预测角色分数和 MLP 单元是不同的轴。</p><p>有符号确认集含 128 个文档来源位置、127 个不同模型输入；相同输入的复用已单独审计，不能视为独立重复实验。</p>
    <div className="prefix-controls"><label>语料族<select value={cohort} onChange={e=>setCohort(e.target.value)}>{[...new Set(samples.map(r=>r.cohort))].sort().map(c=><option key={c}>{c}</option>)}</select></label>
      <label>划分<select value={split} onChange={e=>setSplit(e.target.value)}>{['train','validation','test','connected_test','matched_test','depth_test','prospective_depth6','signed_connected','signed_matched'].map(s=><option key={s}>{s}</option>)}</select></label>
      <label className="prefix-wide">来源<select value={chosen} onChange={e=>setSample(e.target.value)}>{filtered.map(r=><option key={r.sample_id} value={r.sample_id}>{r.sample_id} · {r.tokens} tokens · {r.captured?'已采集':'待采集'}</option>)}</select></label>
      <label>场范围<select value={mode} onChange={e=>setMode(e.target.value)}><option value="all_layers">一个锚点的全部层</option><option value="H12_sources">全部 H12 来源 token</option><option value="fixture_all_tokens">预定样例：一层全部 token</option></select></label>
      <label>锚点<select value={actual} onChange={e=>setAnchor(Number(e.target.value))}>{row?.anchors?.map((p,i)=><option key={i} value={i}>token {p}</option>)}</select></label>
      <label>层 H<input type="number" min={0} max={36} value={layer} onChange={e=>setLayer(Number(e.target.value))}/></label>
      <label>数值<select value={view} onChange={e=>setView(e.target.value)}><option value="raw">原始值</option><option value="RMS">完整向量 RMS</option></select></label>
      <button disabled={!row?.captured} onClick={query}>读取全坐标场</button><button disabled={!row?.captured} onClick={roleQuery}>读取来源角色分数</button></div>
    {!filtered.length&&<p>该语料族／划分没有登记材料。</p>}<ErrorLine error={error}/>
    {detail?.id===chosen&&<><blockquote>{detail.source.original_text||detail.source.text}</blockquote><p>{detail.source.source_group} · {detail.source.novelty||detail.source.provenance}</p>
      <Evidence title="完整输入、外部关系及连接端点可见性" data={{token_ids:detail.source.prompt_ids,relations:detail.source.relations||detail.source.retrospective_ud,visibility:detail.source.connected_visibility}}/>
      {detail.source.program&&<Evidence title="独立解释器核对的程序与目标（仅评估标签）" data={{target:detail.source.target,program:detail.source.program,depth:detail.source.depth}}/>}</>}
    <Field data={field?.identity===identity?field.value:null}/>
    {roles?.id===chosen&&<><p>角色列：{roles.value.role_names.join(' · ')}。由训练前缀拟合，不是当前句子的 gold 依存树。</p><Field data={roles.value}/></>}
    <h3>任意坐标—MLP 单元—输出坐标</h3><div className="prefix-controls"><label>原生 block<select value={block} onChange={e=>setBlock(Number(e.target.value))}>{[6,16,35].map(b=><option key={b}>{b}</option>)}</select></label>
      {[['输入坐标',input,setInput,2559],['MLP 单元',unit,setUnit,9727],['输出坐标',output,setOutput,2559]].map(([label,value,setter,max])=><label key={label}>{label}<input type="number" min={0} max={max} value={value} onChange={e=>setter(Number(e.target.value))}/></label>)}<button disabled={!row?.captured} onClick={scalarQuery}>读取原生标量路径</button></div>
    {scalar?.id===pathID&&<><pre>{JSON.stringify(scalar.value.chain,null,2)}</pre><Field data={scalar.value.input_terms}/><Field data={scalar.value.unit_terms}/><p>{scalar.value.scope}</p></>}
  </section>;
}

function Archive(){
  const [areas,setAreas]=useState([]),[area,setArea]=useState('atlas/condition_profiles'),[files,setFiles]=useState([]),[file,setFile]=useState(''),[headers,setHeaders]=useState([]),[name,setName]=useState('');
  const [row,setRow]=useState(0),[column,setColumn]=useState(0),[data,setData]=useState(null),[error,setError]=useState('');
  useEffect(()=>{get('/areas').then(setAreas).catch(e=>setError(e.message));},[]);
  useEffect(()=>{let active=true;get('/files',{area}).then(v=>{if(active){setFiles(v);setFile(v[0]?.file||'');setHeaders([]);setName('');setRow(0);setColumn(0);setError('');}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[area]);
  useEffect(()=>{let active=true;if(file)get('/arrays',{area,file}).then(v=>{if(active){setHeaders(v);setName(v[0]?.array||'');setRow(0);setColumn(0);setError('');}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[area,file]);
  const header=headers.find(r=>r.array===name),width=header?.shape?.at(-1)||1,identity=JSON.stringify([area,file,name,row,column]);
  async function query(){try{setData({identity,value:await get('/array',{area,file,name,row_start:row,row_count:37,start:column,count:Math.min(width,8192)})});setError('');}catch(e){setData(null);setError(e.message);}}
  return <section id="binding-archive"><h2>完整数组与可重算档案</h2><p>全部已保存原场、预测系数、9728×9728 单元关系、训练增量及梯度因子均可查。数值按原序分页；不以 Top-K、投影或阈值删去低幅值背景。</p>
    <div className="prefix-controls"><label className="prefix-wide">研究目录<select value={area} onChange={e=>{setArea(e.target.value);setFile('');setHeaders([]);setName('');}}>{areas.map(a=><option key={a.area} value={a.area}>{a.area} ({a.files})</option>)}</select></label>
      <label className="prefix-wide">文件<select value={file} onChange={e=>setFile(e.target.value)}>{files.map(f=><option key={f.file}>{f.file}</option>)}</select></label>
      <label>原张量<select value={name} onChange={e=>{setName(e.target.value);setRow(0);setColumn(0);}}>{headers.map(h=><option key={h.array} value={h.array}>{h.array} [{h.shape.join('×')}]</option>)}</select></label>
      <label>起始展平行<input type="number" min={0} value={row} onChange={e=>setRow(Number(e.target.value))}/></label><label>起始原生列<input type="number" min={0} value={column} onChange={e=>setColumn(Number(e.target.value))}/></label><button disabled={!name} onClick={query}>读取原序数组页</button></div>
    <ErrorLine error={error}/>{data?.identity===identity&&<p>原张量 [{data.value.tensor_shape.join('×')}] · 行 [{data.value.row_start},{data.value.row_end}) · 共 {data.value.total_rows} 行</p>}<Field data={data?.identity===identity?data.value:null}/>
  </section>;
}

function Gradient(){
  const [query,setQuery]=useState(0),[part,setPart]=useState('full'),[unit,setUnit]=useState(0),[input,setInput]=useState(0),[output,setOutput]=useState(0),[data,setData]=useState(null),[error,setError]=useState('');
  const identity=JSON.stringify([query,part,unit,input,output]);
  async function load(){try{setData({identity,value:await get('/gradient',{query,part,unit,input_coordinate:input,output_coordinate:output})});setError('');}catch(e){setData(null);setError(e.message);}}
  const r=data?.identity===identity?data.value:null;
  return <section id="binding-gradient"><h2>7471 万参数：完整损失、内容与格式</h2><p>梯度使用当前样本的监督目标。精确分解 L全文词表 = L数字内内容 + L数字输出格式；两部分通常不正交，不能当成互不重叠的“语义百分比”。候选数字评分也不等于自然生成。</p>
    <div className="prefix-controls"><label>梯度目标<select value={part} onChange={e=>setPart(e.target.value)}><option value="full">完整词表 CE</option><option value="content">数字集合内内容</option><option value="format">输出数字格式</option></select></label>
      {[['程序 query',query,setQuery,767],['MLP 单元',unit,setUnit,9727],['输入坐标',input,setInput,2559],['输出坐标',output,setOutput,2559]].map(([label,value,setter,max])=><label key={label}>{label}<input type="number" min={0} max={max} value={value} onChange={e=>setter(Number(e.target.value))}/></label>)}<button onClick={load}>读取完整参数行梯度</button></div>
    <ErrorLine error={error}/>{r&&<><p>{r.material.source_group} · {r.material.representation} · {r.material.split} · 评估目标 {r.material.target}</p><p>{r.precision} 来源：{r.factor_archive}</p><pre>{JSON.stringify(r.scalars,null,2)}</pre><Field data={r.input_terms}/><Field data={r.output_terms}/><p>{r.scope}</p></>}
  </section>;
}

function Behavior(){
  const [mode,setMode]=useState('qwen4'),[items,setItems]=useState([]),[branch,setBranch]=useState('native'),[sample,setSample]=useState(''),[data,setData]=useState(null),[error,setError]=useState('');
  useEffect(()=>{let active=true;get('/behavior-index',{mode}).then(v=>{if(active){setItems(v);setBranch(v[0]?.branch||'native');setSample(v[0]?.sample_id||'');setError('');}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[mode]);
  const rows=items.filter(r=>r.branch===branch),chosen=rows.some(r=>r.sample_id===sample)?sample:rows[0]?.sample_id||'',identity=JSON.stringify([mode,branch,chosen]);
  async function load(){try{setData({identity,value:await get('/behavior',{mode,branch,sample:chosen})});setError('');}catch(e){setData(null);setError(e.message);}}
  const r=data?.identity===identity?data.value:null;
  return <section id="binding-behavior"><h2>原始输出与每条分支的自身历史</h2><p>短预算、长预算和参数更新轨迹分开保存；停止、格式与内容分开评价。没有匹配记录时显示待完成，不填入别的模型结果。oracle 分支使用正确答案，不是零样本方法。</p>
    <div className="prefix-controls"><label>结果集<select value={mode} onChange={e=>{setMode(e.target.value);setItems([]);setData(null);}}>{[['qwen4','4B 原生8token'],['qwen14','14B 原生8token'],['glm4','GLM 原生8token'],['binding','自然续写／绑定与中层训练'],['long_native','4B 原生128token'],['autonomous','4B 参数更新64token']].map(([v,l])=><option key={v} value={v}>{l}</option>)}</select></label>
      <label>独立分支<select value={branch} onChange={e=>setBranch(e.target.value)}>{[...new Set(items.map(r=>r.branch))].map(b=><option key={b}>{b}</option>)}</select></label>
      <label className="prefix-wide">来源<select value={chosen} onChange={e=>setSample(e.target.value)}>{rows.map(r=><option key={r.sample_id} value={r.sample_id}>{r.cohort} · {r.split} · {r.sample_id}</option>)}</select></label><button disabled={!chosen} onClick={load}>读取真实生成</button></div>
    <ErrorLine error={error}/>{!rows.length&&<p>该结果集／分支尚无已提交轨迹。</p>}{r&&<><p>{r.material.source_group} · {r.material.original_text||r.material.text}</p><blockquote>{r.generated}</blockquote>
      <p>严格数字匹配 {r.exact_correct==null?'不适用':String(r.exact_correct)} · 保守末答案 {r.conservative_final_digit??'未解析／不适用'} · EOS {String(r.eos)} · 上限截断 {String(r.token_limit_censored)}</p>
      <Evidence title="完整输入、每步 token、更新与精度记录" data={{prompt_ids:r.prompt_ids||r.material.prompt_ids,generated_ids:r.generated_ids,steps:r.steps,case_update:r.case_update,target:r.target,initial_evaluation:r.initial_evaluation,natural_first_target:r.first_evaluation_target_text,first_state_error:r.state_error_first,later_state_error:r.state_error_after_first}}/></>}
  </section>;
}

export default function RdcBindingAtlas(){
  const [summary,setSummary]=useState(null),[samples,setSamples]=useState([]),[error,setError]=useState('');
  useEffect(()=>{let active=true;Promise.all([get('/overview'),get('/samples')]).then(([s,r])=>{if(active){setSummary(s);setSamples(r);}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[]);
  return <main className="prefix-app law-app binding-app"><header><small>RDC · SOURCE BINDING / NATIVE TRAINING / CONTENT & FORMAT</small><h1>来源绑定与训练形成图谱</h1><p>共同现象 → 候选规律 → 原生参数结构 → 未见组合预测 → 训练形成证据</p><nav><a href="#binding-summary">证据边界</a><a href="#binding-source">材料与全场</a><a href="#binding-archive">完整数组</a><a href="#binding-gradient">参数梯度</a><a href="#binding-behavior">自身历史</a><a href="#binding-theory">统一拼图</a><a href="/rdc-law">前轮图谱 ↗</a></nav></header><ErrorLine error={error}/>
    <section id="binding-summary" className="prefix-warning"><h2>观察、预测、训练与行为分别核对</h2><p>来源联合核的收益有限，尚未稳定超过打乱角色的对照。中层连贯训练降低了部分自然预测损失，但准确率不普遍增加；乱序也并非完全不能学习。完整实验状态以各结果文件和最终审计为准。</p>
      <div className="law-metrics"><div><b>{summary?.material.discovery_rows??'—'}</b><span>严格英语自然发现窗口</span></div><div><b>128</b><span>新连接／匹配自然窗口</span></div><div><b>{summary?.middle.runs?.length??'待完成'}</b><span>真实中层训练轨迹</span></div><div><b>{summary?.autonomous.trajectories??'待完成'}</b><span>参数更新自身历史轨迹</span></div></div>
      <Evidence title="附件审查：保留、收窄与未证明主张" data={summary?.review}/><Evidence title="配对误差、连接端点、原模型与训练比较" data={summary?.analysis}/>
      <Evidence title="运行状态、资源与完整性" data={{scale:Object.fromEntries(Object.entries(summary?.scale||{}).map(([m,r])=>[m,{complete:!!r.timestamp,seconds:r.seconds}])),resources:summary?.resources,integrity:summary?.integrity}}/>
    </section><SourceFields samples={samples}/><Archive/><Gradient/><Behavior/>
    <section id="binding-theory"><h2>统一理论与完整核心拼图</h2><p>RDC 保留既有证据边界。恒等式、拟合候选、推广检查与未完成事项分别记录；没有用新名称把实验现象提升为新数学定理或 AGI 结论。</p>
      {!summary?.theory?.puzzles?.length&&<p>本轮完整理论总账尚待最终提交；不把待运行计划显示为已完成。</p>}
      <div className="prefix-table"><table><thead><tr><th>Phase</th><th>保留拼图</th><th>硬边界</th><th>状态</th></tr></thead><tbody>{summary?.theory?.puzzles?.map((p,i)=><tr key={p.phase+'_'+i}><td>{p.phase}</td><td>{p.retained_puzzle}</td><td>{p.boundary}</td><td>{p.evidence_status}</td></tr>)}</tbody></table></div>
      {summary?.theory?.formulas?.map(f=><Evidence key={f.id} title={`${f.id} · ${f.kind}`} data={f}/>)}
      <Evidence title="完整全参数结构、Alpha / Gamma 与中层训练" data={{native:summary?.native_bilinear,alpha:summary?.alpha,gamma:summary?.gamma,middle:summary?.middle}}/>
      <Evidence title="内容／格式分解与六步新组合" data={summary?.followup}/><Evidence title="自身历史和三模型行为分析" data={summary?.behavior_analysis}/>
      <Evidence title="来源矩碰撞反例、有符号对照与新自然确认" data={summary?.signed_source}><p>反例是使用实际探针的合成状态，不是两条真实语言轨迹。有符号对照修复该反例但未胜出旧验证集；新自然确认和六步程序结果另列，不重新按测试集选胜者。</p></Evidence>
    </section><section><h2>真实数据科学图</h2><p>原生索引保序。图像缩放只影响展示；所有数值可在上方数组库回查。</p>{summary?.figures?.map(f=><figure key={f.path}><figcaption>{f.title}</figcaption><a href={API+'/figure/'+f.path} target="_blank" rel="noreferrer"><img loading="lazy" src={API+'/figure/'+f.path} alt={f.title}/></a><p>{f.scope}</p></figure>)}</section>
    <footer>本页面只读取研究档案，不启动模型、修改权重或提交训练。</footer>
  </main>;
}
