import {useEffect, useMemo, useState} from 'react';
import {Field} from './RdcJointAtlas.jsx';
import './RdcPrefixAtlas.css';
import './RdcJointAtlas.css';
import './RdcOperatorAtlas.css';
import './RdcLawAtlas.css';

const API='http://127.0.0.1:5001/api/rdc-law';
const fmt=x=>x==null?'—':Number(x).toPrecision(5);
async function get(path,params){const r=await fetch(API+path+(params?'?'+new URLSearchParams(params):''));if(!r.ok)throw new Error((await r.json().catch(()=>({}))).detail||r.statusText);return r.json();}
function ErrorLine({error}){return error?<p role="alert">{error}</p>:null;}

function MaterialField({samples}){
  const [cohort,setCohort]=useState('gum'),[split,setSplit]=useState('train'),[sample,setSample]=useState('');
  const [mode,setMode]=useState('all_layers'),[anchor,setAnchor]=useState(0),[layer,setLayer]=useState(12),[view,setView]=useState('raw');
  const [result,setResult]=useState(null),[detail,setDetail]=useState(null),[error,setError]=useState('');
  const [block,setBlock]=useState(35),[unit,setUnit]=useState(0),[input,setInput]=useState(0),[output,setOutput]=useState(0),[scalar,setScalar]=useState(null);
  const filtered=useMemo(()=>samples.filter(r=>r.cohort===cohort&&r.split===split),[samples,cohort,split]);
  const chosen=filtered.some(r=>r.sample_id===sample)?sample:filtered[0]?.sample_id||'',row=filtered.find(r=>r.sample_id===chosen);
  const actualAnchor=Math.min(anchor,(row?.anchors?.length||1)-1),identity=JSON.stringify([chosen,mode,actualAnchor,layer,view]);
  const pathIdentity=JSON.stringify([chosen,block,actualAnchor,unit,input,output]);
  async function query(){try{const [value,source]=await Promise.all([get('/field',{sample:chosen,mode,anchor:actualAnchor,layer,view}),get('/sample',{sample:chosen})]);setResult({identity,value});setDetail({id:chosen,source});setError('');}catch(e){setResult(null);setError(e.message);}}
  async function path(){try{setScalar({identity:pathIdentity,value:await get('/scalar',{sample:chosen,block,anchor:actualAnchor,unit,input_coordinate:input,output_coordinate:output})});setError('');}catch(e){setScalar(null);setError(e.message);}}
  return <section id="law-fields"><h2>外部材料 → 全坐标响应 → 原生参数</h2><p>真实多体裁自然窗口和真人问题。全部 token 参与全层统计；所有锚点、所有 H12 来源向量持续保留。12 个预定主集样例另保留每层×每 token 原场。依存类型共现是事后标签，不是已证明的语义操作。</p>
    <div className="prefix-controls"><label>语料族<select value={cohort} onChange={e=>setCohort(e.target.value)}>{['gum','ewt','cmrc','squad_qa','cmrc_qa','hotpot_qa'].map(c=><option key={c}>{c}</option>)}</select></label>
      <label>冻结划分<select value={split} onChange={e=>setSplit(e.target.value)}>{['train','validation','test','confirmation'].map(s=><option key={s}>{s}</option>)}</select></label>
      <label className="prefix-wide">来源<select value={chosen} onChange={e=>setSample(e.target.value)}>{filtered.map(r=><option key={r.sample_id} value={r.sample_id}>{r.sample_id} · {r.full_field?'含全 token 原场':'锚点＋全部早层来源'} · {r.held_relation_combinations.join(' / ')}</option>)}</select></label>
      <label>场范围<select value={mode} onChange={e=>setMode(e.target.value)}><option value="all_layers">一个锚点的全部层</option><option value="all_tokens">一层的全部 token</option></select></label>
      <label>锚点<select value={actualAnchor} onChange={e=>setAnchor(Number(e.target.value))}>{row?.anchors?.map((p,i)=><option key={i} value={i}>{i} · token {p}</option>)}</select></label>
      <label>层 H<input type="number" min={0} max={36} value={layer} onChange={e=>setLayer(Number(e.target.value))}/></label>
      <label>数值视图<select value={view} onChange={e=>setView(e.target.value)}><option value="raw">原始值</option><option value="RMS">完整向量 RMS</option><option value="train_z">训练坐标 z</option></select></label><button disabled={!chosen} onClick={query}>读取全坐标场</button></div>
    {!filtered.length&&<p>此语料族／划分没有登记材料；不显示其他来源的旧结果。</p>}<ErrorLine error={error}/>
    {detail?.id===chosen&&<><blockquote>{detail.source.text}</blockquote><p>{detail.source.source_group} · {detail.source.novelty} · 体裁 {detail.source.genre}</p><details><summary>外部类型关系、输入身份与标签边界</summary><pre>{JSON.stringify({tokens:detail.source.prompt_ids,relations:detail.source.retrospective_graph||detail.source.graph,held:detail.source.held_relation_combinations,anchor_endpoint_visibility:detail.source.held_combination_visibility,prior_document:detail.source.prior_document_in_recent_campaigns},null,2)}</pre></details></>}
    <Field data={result?.identity===identity?result.value:null}/><h3>任意输入坐标—MLP 单元—输出坐标</h3>
    <div className="prefix-controls"><label>原生 block<select value={block} onChange={e=>setBlock(Number(e.target.value))}>{[6,16,35].map(b=><option key={b}>{b}</option>)}</select></label>
      {[['输入坐标',input,setInput,2559],['MLP 单元',unit,setUnit,9727],['输出坐标',output,setOutput,2559]].map(([label,value,setter,max])=><label key={label}>{label}<input type="number" min={0} max={max} value={value} onChange={e=>setter(Number(e.target.value))}/></label>)}<button disabled={!chosen} onClick={path}>核对全部读写项</button></div>
    {scalar?.identity===pathIdentity&&<><pre>{JSON.stringify(scalar.value.chain,null,2)}</pre><Field data={scalar.value.input_terms}/><Field data={scalar.value.unit_terms}/><p>{scalar.value.scope}</p></>}
  </section>;
}

function Arrays(){
  const [areas,setAreas]=useState([]),[area,setArea]=useState('atlas/joint_products'),[files,setFiles]=useState([]),[file,setFile]=useState(''),[headers,setHeaders]=useState([]),[name,setName]=useState('');
  const [row,setRow]=useState(0),[column,setColumn]=useState(0),[result,setResult]=useState(null),[error,setError]=useState('');
  useEffect(()=>{get('/areas').then(setAreas).catch(e=>setError(e.message));},[]);
  useEffect(()=>{let active=true;get('/files',{area}).then(v=>{if(active){setFiles(v);setFile(v[0]?.file||'');setHeaders([]);setName('');setRow(0);setColumn(0);setError('');}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[area]);
  useEffect(()=>{let active=true;if(file)get('/arrays',{area,file}).then(v=>{if(active){setHeaders(v);setName(v[0]?.array||'');setRow(0);setColumn(0);setError('');}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[area,file]);
  const header=headers.find(h=>h.array===name),width=header?.shape?.at(-1)||1,total=header?.shape?.slice(0,-1).reduce((a,b)=>a*b,1)||1;
  const identity=JSON.stringify([area,file,name,row,column]);
  async function query(){try{setResult({identity,value:await get('/array',{area,file,name,row_start:row,row_count:37,start:column,count:Math.min(width,8192)})});setError('');}catch(e){setResult(null);setError(e.message);}}
  return <section id="law-arrays"><h2>全部原场、单元、参数、预测与训练档案</h2><p>全部已保存 NPZ 可查，不限于高幅值坐标。按原顺序分页，宽度超过8192逐列翻页；前置轴展平行号明确显示。不同数组的列可能是坐标、单元、来源、参数或词表，不能混称同一空间。</p>
    <div className="prefix-controls"><label className="prefix-wide">研究目录<select value={area} onChange={e=>{setArea(e.target.value);setFile('');setHeaders([]);setName('');}}>{areas.map(a=><option key={a.area} value={a.area}>{a.area} ({a.files})</option>)}</select></label>
      <label className="prefix-wide">文件／条件画像<select value={file} onChange={e=>setFile(e.target.value)}>{files.map(f=><option key={f.file} value={f.file}>{f.label||f.file}</option>)}</select></label><label>原张量<select value={name} onChange={e=>{setName(e.target.value);setRow(0);setColumn(0);}}>{headers.map(h=><option key={h.array} value={h.array}>{h.array} [{h.shape.join('×')}]</option>)}</select></label>
      <label>起始展平行<input type="number" min={0} value={row} onChange={e=>setRow(Number(e.target.value))}/></label><label>起始原生列<input type="number" min={0} value={column} onChange={e=>setColumn(Number(e.target.value))}/></label><button disabled={!name} onClick={query}>读取原序数值页</button><span>总行 {total} · 原始宽度 {width}</span></div>
    <ErrorLine error={error}/>{result?.identity===identity&&<p>原张量 [{result.value.tensor_shape.join('×')}] · 行 [{result.value.row_start}, {result.value.row_end})</p>}<Field data={result?.identity===identity?result.value:null}/>
  </section>;
}

function Training(){
  const [panel,setPanel]=useState(0),[unit,setUnit]=useState(0),[input,setInput]=useState(0),[output,setOutput]=useState(0),[result,setResult]=useState(null),[error,setError]=useState('');
  const identity=JSON.stringify([panel,unit,input,output]);
  async function query(){try{setResult({identity,value:await get('/gradient',{panel_index:panel,unit,input_coordinate:input,output_coordinate:output})});setError('');}catch(e){setResult(null);setError(e.message);}}
  const r=result?.identity===identity?result.value:null;
  return <section id="law-training"><h2>真实训练：从一个损失到7471万原生参数</h2><p>这里查询真实全词表交叉熵梯度，不是外挂探针。三个梯度矩阵由完整外积精确表示；36次单步更新和四条64步轨迹可在上方数组库回查。训练需要已知语料目标；当前梯度不等于原预训练历史。</p>
    <div className="prefix-controls">{[['梯度 query 索引',panel,setPanel,287],['MLP 单元',unit,setUnit,9727],['输入坐标',input,setInput,2559],['输出坐标',output,setOutput,2559]].map(([label,value,setter,max])=><label key={label}>{label}<input type="number" min={0} max={max} value={value} onChange={e=>setter(Number(e.target.value))}/></label>)}<button onClick={query}>读取原生参数梯度</button></div><ErrorLine error={error}/>
    {r&&<><p>{r.query.sample_id} · token {r.query.position} · 实际目标 ID {r.query.target_id} · {r.query.split}</p><pre>{JSON.stringify(r.scalar_parameter_derivatives,null,2)}</pre><Field data={r.input_gradient_terms}/><Field data={r.output_gradient_terms}/><Field data={r.all_query_gradient_inner_products}/><p>{r.scope}</p></>}
  </section>;
}

function Behavior(){
  const [model,setModel]=useState('live'),[items,setItems]=useState([]),[branch,setBranch]=useState('native'),[sample,setSample]=useState(''),[result,setResult]=useState(null),[error,setError]=useState('');
  useEffect(()=>{let active=true;get('/behavior-index',{model}).then(v=>{if(active){setItems(v);setSample(v[0]?.sample_id||'');setBranch(v[0]?.branch||'native');setError('');}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[model]);
  const rows=items.filter(r=>r.branch===branch),chosen=rows.some(r=>r.sample_id===sample)?sample:rows[0]?.sample_id||'',identity=JSON.stringify([model,branch,chosen]);
  async function query(){try{setResult({identity,value:await get('/behavior',{model,branch,sample:chosen})});setError('');}catch(e){setResult(null);setError(e.message);}}
  const r=result?.identity===identity?result.value:null;
  return <section id="law-behavior"><h2>自身历史生成与三个模型的真实问答</h2><p>每条分支保留自己的已选 token 和 KV。真人 QA 按完整答案评分；自然续写没有唯一标准答案。14B/GLM是原生32token上限复查，主4B为48token；不能把长度上限未停止都当作语义错误。</p>
    <div className="prefix-controls"><label>行为数据<select value={model} onChange={e=>{setModel(e.target.value);setItems([]);setResult(null);}}><option value="live">4B 七条冻结分支</option><option value="own_history">自身历史的独立原生参照</option>{['qwen4','qwen14','glm4'].map(m=><option key={m}>{m}</option>)}</select></label>
      <label>运行分支<select value={branch} onChange={e=>setBranch(e.target.value)}>{[...new Set(items.map(i=>i.branch))].map(b=><option key={b}>{b}</option>)}</select></label><label className="prefix-wide">真实来源<select value={chosen} onChange={e=>setSample(e.target.value)}>{rows.map(x=><option key={x.sample_id} value={x.sample_id}>{x.cohort} · {x.sample_id}</option>)}</select></label><button disabled={!chosen} onClick={query}>读取实际输出和历史</button></div><ErrorLine error={error}/>
    {!rows.length&&<p>此模型／分支尚无已提交结果，不用其他模型或演示数据填充。</p>}{r&&<><p>{r.question||'自然续写'} · {r.source_group}</p>{r.answers&&<p>参考答案：{r.answers.map(a=>a.text).join(' / ')}</p>}<blockquote>{r.generated_text}</blockquote><p>完整规范字符串匹配 {r.normalized_full_EM==null?'不适用':String(r.normalized_full_EM)} · F1 {fmt(r.answer_F1)} · 原生 EOS {String(r.stopped_by_native_EOS)} · 重复4gram {fmt(r.repeated_4gram_fraction)}</p>
      <details><summary>实际输入、全部生成 token 和逐步记录</summary><pre>{r.actual_prompt||r.material.text}</pre><p>提示 token IDs：{r.prompt_ids.join(', ')}</p><p>生成 token IDs：{r.generated_ids.join(', ')}</p><pre>{JSON.stringify(r.steps||[],null,2)}</pre></details>{r.KV_comparisons&&<details><summary>同一自身历史下的全部层 KV 核对（不注入参照）</summary><p>主运行生成 IDs 精确复现：{String(r.main_rollout_IDs_exact)}</p><pre>{JSON.stringify(r.KV_comparisons,null,2)}</pre></details>}</>}
  </section>;
}

export default function RdcLawAtlas(){
  const [summary,setSummary]=useState(null),[samples,setSamples]=useState([]),[error,setError]=useState('');
  useEffect(()=>{let active=true;Promise.all([get('/overview'),get('/samples')]).then(([s,r])=>{if(active){setSummary(s);setSamples(r);}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[]);
  return <main className="prefix-app operator-app law-app"><header><small>RDC · FORMATION / OPERATION / COMPOSITION</small><h1>条件规律与训练形成图谱</h1><p>共同现象 → 候选规律 → 原生参数 → 未见组合预测。把观察、局部预测、训练变化和真实生成分开核对。</p><nav><a href="#law-summary">证据边界</a><a href="#law-fields">自然材料与全场</a><a href="#law-arrays">完整数组</a><a href="#law-training">训练参数</a><a href="#law-behavior">真实行为</a><a href="#law-theory">统一拼图</a><a href="/rdc-operator">前轮算子图谱 ↗</a></nav></header><ErrorLine error={error}/>
    <section id="law-summary" className="prefix-warning"><h2>当前能说什么，不能说什么</h2><p>历史条件改善了部分后层预测，未证明统一语义齿轮。连贯训练在主测试总体上优于乱序对照，但未优于原模型；在组合端点已出现的确认子集上，四组参数更新都变差。未见组合上的状态预测收益不能代替输出概率与语言能力提升。</p>
      <div className="law-metrics"><div><b>{summary?.material.main_rows??'—'}</b><span>主材料 / 六族</span></div><div><b>{summary?.material.confirmation_rows??'—'}</b><span>冻结确认材料</span></div><div><b>{summary?.atlas.profiles?.length??'—'}</b><span>完整条件画像</span></div><div><b>{summary?.deployment.trajectories??'待完成'}</b><span>已完成自身历史轨迹</span></div></div>
      <details><summary>18项审查与过度结论修正</summary><pre>{JSON.stringify(summary?.review||[],null,2)}</pre></details><details><summary>实际资源、阶段完成状态与审计</summary><pre>{JSON.stringify({resources:summary?.resources,scale:Object.fromEntries(Object.entries(summary?.scale||{}).map(([k,v])=>[k,{completed:!!v.timestamp,seconds:v.seconds}])),integrity:summary?.integrity},null,2)}</pre></details>
      <details><summary>组合窗口不等于当前前缀：126个端点已出现的锚点核查</summary><p>原组合窗口组共192个锚点，其中66个较早位置尚未出现组合的全部必要端点。126个端点已出现的锚点仍使用事后人工句法标签，不能等同于模型已执行语义组合；所有原始预测和划分均未修改。</p><pre>{JSON.stringify(summary?.combination_visibility||{},null,2)}</pre></details>
    </section><MaterialField samples={samples}/><Arrays/><Training/><Behavior/>
    <section id="law-theory"><h2>统一拼图与候选规律账本</h2><p>历史拼图保留适用范围；新观察、数学恒等式、拟合模型和尚未通过的假设分别登记。本轮不宣称新数学定理、完整语言闭合或AGI。</p>
      {!summary?.theory?.puzzles?.length&&<p>本轮完整理论总账尚未提交；已有阶段详见研究记录，不把计划显示为已完成证据。</p>}
      <div className="prefix-table"><table><thead><tr><th>Phase</th><th>保留的拼图</th><th>硬边界</th><th>证据状态</th></tr></thead><tbody>{summary?.theory?.puzzles?.map((p,i)=><tr key={p.phase+'_'+i}><td>{p.phase}</td><td>{p.retained_puzzle}</td><td>{p.boundary}</td><td>{p.evidence_status}</td></tr>)}</tbody></table></div>
      {summary?.theory?.formulas?.map(f=><details key={f.id}><summary>{f.id} · {f.kind}</summary><pre>{f.expression}</pre><p>{f.variables}</p><p>{f.evidence}</p></details>)}
      <details><summary>冻结预测、确认与真实参数训练的完整分层结果</summary><pre>{JSON.stringify({prediction:summary?.prediction,confirmation:summary?.confirmation,training:summary?.training},null,2)}</pre></details>
      <details><summary>七分支部署、三模型端点口径与配对结果</summary><pre>{JSON.stringify({deployment:summary?.deployment,paired:summary?.deployment_paired,scale:summary?.scale,own_history:summary?.history_followup},null,2)}</pre></details>
    </section><section><h2>真实数据科学图</h2><p>每张图链接实际数组与计算范围。缩略显示、标准化、非正交交叉项和条件权重不能混为语义贡献比例。</p>{summary?.figures?.map(f=><figure key={f.path}><figcaption>{f.title}</figcaption><a href={API+'/figure/'+f.path} target="_blank" rel="noreferrer"><img loading="lazy" src={API+'/figure/'+f.path} alt={f.title}/></a><p>{f.scope}</p></figure>)}</section>
  </main>;
}
