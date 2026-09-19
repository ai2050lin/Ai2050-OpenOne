import {useEffect, useMemo, useState} from 'react';
import {Field} from './RdcJointAtlas.jsx';
import './RdcPrefixAtlas.css';
import './RdcJointAtlas.css';
import './RdcOperatorAtlas.css';

const API='http://127.0.0.1:5001/api/rdc-operator';
const fmt=x=>x==null?'—':Number(x).toPrecision(5);
const evidenceLabels={inherited_scoped_evidence_not_rerun_this_campaign:'继承的有限域证据／本轮未重跑',new_observation_and_native_behavior:'新观察及原生行为',known_identity_plus_scoped_local_prediction:'已知恒等式校准＋有限域局部预测',new_compilation_behavior_and_numeric_audit:'新概率编译、行为与数值核对',same_goal_followup_with_explicit_reanalysis_scope:'同目标续研／复分析范围已注明'};
async function get(path,params) {
  const response=await fetch(API+path+(params?'?'+new URLSearchParams(params):''));
  if (!response.ok) throw new Error((await response.json().catch(()=>({}))).detail||response.statusText);
  return response.json();
}

function NativeField({samples}) {
  const [sample,setSample]=useState(''),[language,setLanguage]=useState('all'),[fullOnly,setFullOnly]=useState(true);
  const [mode,setMode]=useState('all_tokens'),[layer,setLayer]=useState(12),[anchor,setAnchor]=useState(0),[view,setView]=useState('raw');
  const [result,setResult]=useState(null),[detail,setDetail]=useState(null),[error,setError]=useState('');
  const [block,setBlock]=useState(6),[unit,setUnit]=useState(0),[input,setInput]=useState(0),[output,setOutput]=useState(0),[scalar,setScalar]=useState(null);
  const filtered=useMemo(()=>samples.filter(r=>r.committed&&(language==='all'||r.language===language)&&(!fullOnly||r.full_field)),[samples,language,fullOnly]);
  const currentSample=filtered.some(r=>r.sample_id===sample)?sample:filtered[0]?.sample_id||'';
  const identity=JSON.stringify([currentSample,mode,layer,anchor,view]);
  const scalarIdentity=JSON.stringify([currentSample,block,anchor,unit,input,output]);
  const row=filtered.find(r=>r.sample_id===currentSample);
  async function query() {
    try {const [value,source]=await Promise.all([get('/field',{sample:currentSample,mode,layer,anchor,view}),get('/sample',{sample:currentSample})]);setResult({identity,value});setDetail({sample:currentSample,source});setError('');}
    catch(e){setResult(null);setError(e.message);}
  }
  async function queryScalar() {
    try {setScalar({identity:scalarIdentity,value:await get('/scalar',{sample:currentSample,block,anchor,unit,input_coordinate:input,output_coordinate:output})});setError('');}
    catch(e){setScalar(null);setError(e.message);}
  }
  return <section id="operator-fields"><h2>原生坐标场与任意标量路径</h2>
    <p>每个自然窗口都扫描了全部 token、37 个层边界。所有来源保留两锚点完整场；预先指定的 16 个来源另保留全 token 原场，其他全 token 数值以统计和重算身份登记。</p>
    <div className="prefix-controls"><label>语言<select value={language} onChange={e=>setLanguage(e.target.value)}><option value="all">全部</option><option>en</option><option>zh</option></select></label>
      <label><input type="checkbox" checked={fullOnly} onChange={e=>setFullOnly(e.target.checked)}/>仅全 token 原场样例</label>
      <label>自然来源<select value={currentSample} onChange={e=>setSample(e.target.value)}>{filtered.map(r=><option key={r.sample_id}>{r.sample_id}</option>)}</select></label>
      <label>范围<select value={mode} onChange={e=>setMode(e.target.value)}><option value="all_tokens">一层全部 token</option><option value="all_layers">一个锚点全部层</option></select></label>
      <label>层边界 H<input type="number" min={0} max={36} value={layer} onChange={e=>setLayer(Number(e.target.value))}/></label>
      <label>锚点<select value={anchor} onChange={e=>setAnchor(Number(e.target.value))}>{[0,1].map(i=><option key={i} value={i}>{i} · token {row?.anchors?.[i]??'—'}</option>)}</select></label>
      <label>数值视图<select value={view} onChange={e=>setView(e.target.value)}><option value="raw">原值</option><option value="RMS">每行 RMS</option><option value="train_z">训练坐标 z</option></select></label>
      <button disabled={!currentSample} onClick={query}>读取完整原生坐标</button></div>
    {detail?.sample===currentSample&&<><blockquote>{detail.source.text}</blockquote><p>{detail.source.title} · {detail.source.source_group} · {detail.source.annotation_scope}</p></>}
    {error&&<p role="alert">{error}</p>}<Field data={result?.identity===identity?result.value:null}/>
    <h3>真实权重与全部乘积</h3><p>任选输入坐标、MLP 单元与输出坐标。显示全部 2560 个读取项和全部 9728 个写回项；单项可回查不等于该项就是一个概念。</p>
    <div className="prefix-controls"><label>block<select value={block} onChange={e=>setBlock(Number(e.target.value))}>{[6,16,34].map(b=><option key={b}>{b}</option>)}</select></label>
      {[['输入坐标',input,setInput,2559],['MLP 单元',unit,setUnit,9727],['输出坐标',output,setOutput,2559]].map(([name,value,setter,max])=><label key={name}>{name}<input type="number" min={0} max={max} value={value} onChange={e=>setter(Number(e.target.value))}/></label>)}
      <button disabled={!currentSample} onClick={queryScalar}>核对任意参数路径</button></div>
    {scalar?.identity===scalarIdentity&&<><p>实际 token {scalar.value.native_token} · block {scalar.value.block}</p><pre>{JSON.stringify(scalar.value.chain,null,2)}</pre><Field data={scalar.value.input_terms}/><Field data={scalar.value.unit_terms}/></>}
  </section>;
}

function ArrayBrowser() {
  const [areas,setAreas]=useState([]),[area,setArea]=useState('operators'),[files,setFiles]=useState([]),[file,setFile]=useState(''),[arrays,setArrays]=useState([]),[name,setName]=useState('');
  const [row,setRow]=useState(0),[column,setColumn]=useState(0),[data,setData]=useState(null),[error,setError]=useState('');
  useEffect(()=>{get('/areas').then(setAreas).catch(e=>setError(e.message));},[]);
  useEffect(()=>{let active=true;get('/files',{area}).then(x=>{if(active){setFiles(x);setFile(x[0]?.file||'');setArrays([]);setName('');setRow(0);setColumn(0);setError('');}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[area]);
  useEffect(()=>{let active=true;if(!file)return;get('/arrays',{area,file}).then(x=>{if(active){setArrays(x);setName(x[0]?.array||'');setRow(0);setColumn(0);setError('');}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[area,file]);
  const selected=arrays.find(a=>a.array===name),total=selected?selected.shape.slice(0,-1).reduce((a,b)=>a*b,1):0;
  const width=selected?.shape.at(-1)||2560,wide=width>16384,effectiveColumn=wide?column:0;
  const identity=JSON.stringify([area,file,name,row,effectiveColumn]);
  async function query(){try{setData({identity,value:await get('/array',{area,file,name,row_start:row,row_count:37,start:effectiveColumn,count:wide?8192:width})});setError('');}catch(e){setData(null);setError(e.message);}}
  return <section id="operator-arrays"><h2>全部因子、算子、矩阵和预测数组</h2><p>按目录→来源文件→原张量选取。原生残差坐标及不超画布限制的 MLP 单元完整显示；更宽的 MLP／词表按原序 8192 列分页，全部列均可查询和下载。大张量的前置轴也明确分页；这不是 Top-K 筛选。注意：attention 的列是来源位置，单元与残差坐标是不同空间。</p>
    <div className="prefix-controls"><label>研究目录<select value={area} onChange={e=>{setArea(e.target.value);setFile('');setArrays([]);setName('');setError('');}}>{areas.map(a=><option key={a.area} value={a.area}>{a.area} ({a.files})</option>)}</select></label>
      <label>来源文件<select value={file} onChange={e=>setFile(e.target.value)}>{files.map(f=><option key={f.file}>{f.file}</option>)}</select></label>
      <label>完整数组<select value={name} onChange={e=>{setName(e.target.value);setRow(0);setColumn(0);}}>{arrays.map(a=><option key={a.array} value={a.array}>{a.array} [{a.shape.join('×')}]</option>)}</select></label>
      <label>前置轴起始行<input type="number" min={0} max={Math.max(0,total-1)} value={row} onChange={e=>setRow(Number(e.target.value))}/></label><span>总行数 {total} · 原生列数 {width}</span>
      {wide&&<label>宽数组原序起始列<input type="number" min={0} max={width-1} step={8192} value={column} onChange={e=>setColumn(Number(e.target.value))}/></label>}<button disabled={!name} onClick={query}>读取原序坐标页</button></div>
    {error&&<p role="alert">{error}</p>}{data?.identity===identity&&<p>原张量 [{data.value.tensor_shape.join('×')}] · 行 [{data.value.row_start},{data.value.row_end})</p>}
    <Field data={data?.identity===identity?data.value:null}/>
  </section>;
}

function QA() {
  const [model,setModel]=useState('qwen4'),[scope,setScope]=useState('main'),[items,setItems]=useState([]),[id,setId]=useState(''),[result,setResult]=useState(null),[error,setError]=useState('');
  const identity=JSON.stringify([model,scope,id]);
  useEffect(()=>{let active=true;get('/qa-index',{model,scope}).then(x=>{if(active){setItems(x);setId(x[0]?.question_id||'');}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[model,scope]);
  async function query(){try{setResult({identity,value:await get('/qa',{model,scope,id})});setError('');}catch(e){setResult(null);setError(e.message);}}
  const r=result?.identity===identity?result.value:null;
  return <section id="operator-qa"><h2>真实问答、匹配与停止</h2><p>人类问题保持原文，多跳题保留全部可用原始段落，并非每题固定 10 段（Phase 2728 原始来源审计纠正）。完整字符串未匹配可能是同义表达、单位转换，也可能真错；不能全部标成“不懂”。支持句标注和答案没有输入生成过程。</p>
    <div className="prefix-controls"><label>问答模型<select value={model} onChange={e=>setModel(e.target.value)}>{['qwen4','qwen14','glm4'].map(x=><option key={x}>{x}</option>)}</select></label>
      <label>问答范围<select value={scope} onChange={e=>setScope(e.target.value)}><option>main</option><option>confirmation</option></select></label>
      <label>原始问题<select value={id} onChange={e=>setId(e.target.value)}>{items.map(x=><option key={x.question_id} value={x.question_id}>{x.language}/{x.question_type} · {x.normalized_full_EM?'字符串匹配':'字符串未匹配'} · {x.question}</option>)}</select></label><button disabled={!id} onClick={query}>读取真实答案与输入</button></div>
    {!items.length&&<p>此模型／范围没有已提交问题。14B 与 GLM 的各 64 题集中在 main，含原有确认来源和多跳材料，不另建 confirmation 问答批次。</p>}
    {error&&<p role="alert">{error}</p>}{r&&<><h3>{r.question}</h3><p>参考答案：{r.answers.map(a=>a.text).join(' / ')}</p><blockquote>{r.generated_text}</blockquote>
      <p>完整规范字符串匹配：{String(r.normalized_full_EM)} · F1 {fmt(r.answer_F1)} · 原生 EOS {String(r.stopped_by_native_EOS)} · 达到长度上限 {String(r.hit_48_token_limit)}</p>
      <p>内部 query 原场：{r.raw_query_field}（上方数组浏览器可读取每个原生坐标与来源）。</p>
      {r.typed_hyperedge&&<details><summary>外部问题—答案跨度—支持句—内部来源超边</summary><pre>{JSON.stringify(r.typed_hyperedge,null,2)}</pre></details>}
      <details><summary>展开实际模型输入、token 与评分边界</summary><pre>{r.actual_prompt}</pre><p>原生 token IDs：{r.prompt_ids.join(', ')}</p><p>{r.scoring_scope}</p></details></>}
  </section>;
}

function Autonomous() {
  const [items,setItems]=useState([]),[sample,setSample]=useState(''),[result,setResult]=useState(null),[error,setError]=useState('');
  useEffect(()=>{get('/generation-index').then(x=>{setItems(x);setSample(x[0]?.sample_id||'');}).catch(e=>setError(e.message));},[]);
  async function query(){try{setResult(await get('/generation',{sample}));setError('');}catch(e){setResult(null);setError(e.message);}}
  const r=result?.sample_id===sample?result:null;
  return <section id="operator-generation"><h2>各自历史的连续生成</h2><p>分支独立维护自己的 token 历史和 KV。三个近似 MLP 以外仍使用原生网络，首位置也保留原生计算；这不是完整提取出的独立语言模型。达到 48 token 上限不直接等于停止错误。</p>
    <div className="prefix-controls"><label>连续生成来源<select value={sample} onChange={e=>setSample(e.target.value)}>{items.map(x=><option key={x.sample_id}>{x.sample_id}</option>)}</select></label><button disabled={!sample} onClick={query}>对照真实连续生成</button></div>
    {error&&<p role="alert">{error}</p>}{r&&<><blockquote>{r.initial_text}</blockquote>{Object.entries(r.branches).map(([name,b])=><article key={name}><h3>{name}</h3><pre>{b.generated_text}</pre><p>token {b.token_count} · EOS {String(b.stopped_by_native_EOS)} · 重复四元组比例 {fmt(b.repeated_4gram_fraction)}{name!=='native'&&<> · 与原生首个分叉步 {b.first_branch_step_1based??'未分叉'}</>}</p>{b.steps&&<details><summary>相同已选 token 历史上的独立原生诊断（没有反馈刷新近似分支）</summary><pre>{JSON.stringify(b.steps,null,2)}</pre></details>}</article>)}</>}
  </section>;
}

function Ordering() {
  const [items,setItems]=useState([]),[id,setId]=useState(''),[result,setResult]=useState(null),[error,setError]=useState('');
  useEffect(()=>{get('/operation-index').then(x=>{setItems(x);setId(x[0]?.question_id||'');}).catch(e=>setError(e.message));},[]);
  async function query(){try{setResult(await get('/operation',{id}));setError('');}catch(e){setResult(null);setError(e.message);}}
  const r=result?.question_id===id?result:null;
  return <section><h2>自然问题与材料的顺序操作</h2><p>保持问题、完整材料与金标不变，只调换先问问题／先给材料。顺序同时改变位置与可用历史，不冒充只改变一个语义因素的实验。</p>
    <div className="prefix-controls"><label>顺序操作问题<select value={id} onChange={e=>setId(e.target.value)}>{items.map(x=><option key={x.question_id} value={x.question_id}>{x.language} · {x.question}</option>)}</select></label><button disabled={!id} onClick={query}>比较问题顺序与原生回答</button></div>
    {!items.length&&<p>顺序实验尚未提交。</p>}{error&&<p role="alert">{error}</p>}{r&&<><h3>{r.question}</h3><p>先材料：{r.context_first.generated_text} · F1 {fmt(r.context_first.answer_F1)}</p><p>先问题：{r.generated_text} · F1 {fmt(r.answer_F1)}</p><details><summary>完整层间变化、来源位置与实际输入</summary><pre>{JSON.stringify(r,null,2)}</pre></details></>}
  </section>;
}

function EvidenceResults({summary}) {
  const compiled=summary?.compilation;
  const records=compiled?.local?.confirmation||[];
  return <section><h2>新来源确认：局部误差与真正输出分开</h2>
    <p>下表是同形状 prefix 上只替换末端 query 的局部 MLP 写回，由真实剩余网络计算完整词表。K/J 的公式重算是数学校准，不等于语义算法已被提取。</p>
    <div style={{overflowX:'auto'}}><table><thead><tr><th>替换配置</th><th>锚点数</th><th>原生→近似 KL</th><th>校准后 KL</th><th>argmax 一致</th><th>后归一化场误差</th></tr></thead><tbody>{records.map(r=><tr key={r.name}><td>{r.name}</td><td>{r.anchors}</td><td>{fmt(r.raw_KL)}</td><td>{fmt(r.calibrated_KL)}</td><td>{fmt(r.argmax_agreement)}</td><td>{fmt(r.postnorm_relative_MSE)}</td></tr>)}</tbody></table></div>
    {!records.length&&<p>真实网络编译尚未提交。</p>}
    <details><summary>同来源配对增益与文章聚类区间（均值变化不自动等于稳定改善）</summary><pre>{JSON.stringify(summary?.paired_probability,null,2)}</pre></details>
    <h3>同题原生行为与后验响应分层</h3><p>下面的三模型结果使用相同 64 道真实问题和完整材料；聊天模板、分词与网络各自原生。答案匹配／未匹配只用于事后整理，完整层×坐标统计可在 behavior 目录查询，不作为在线输入或已验证的推理方向。</p>
    <div style={{overflowX:'auto'}}><table><thead><tr><th>原生模型</th><th>同题数</th><th>完整规范匹配</th><th>答案 F1</th><th>原生 EOS</th></tr></thead><tbody>{Object.entries(summary?.behavior?.matched64||{}).map(([key,r])=><tr key={key}><td>{key}</td><td>{r.questions}</td><td>{fmt(r.EM)}</td><td>{fmt(r.F1)}</td><td>{fmt(r.EOS)}</td></tr>)}</tbody></table></div>
    <details><summary>行为分组、输入长度混杂与配对来源区间</summary><pre>{JSON.stringify(summary?.behavior,null,2)}</pre></details>
    <h3>同规模材料与独立模型复查</h3><p>4B 匹配子集为复分析；14B/GLM 分别拟合，并在各自确认采集之前冻结。来源数与训练行数相同，仍不能把相同层比例或坐标索引当作同一个语义功能。</p>{['qwen4','qwen14','glm4'].map(key=>{const s=summary?.scale?.[key];return <article key={key}><h4>{key}</h4>{s?.sources?<><p>{s.sources} 来源 · {s.tokens} 个本模型 token · 原生宽度 {s.native_width} · 本模型 block {s.own_blocks.join(', ')}；坐标索引不作跨模型语义对齐。</p><details><summary>完整复查数据与边界</summary><pre>{JSON.stringify(s,null,2)}</pre></details></>:<p>尚未完成；不显示模拟结果。</p>}</article>;})}
    <details><summary>跨模型配对区间与全部非精确字符端点</summary><pre>{JSON.stringify(summary?.matched_scale_audit,null,2)}</pre></details>
    <details><summary>顺序操作：答案相同是否意味着完整响应相同</summary><pre>{JSON.stringify(summary?.operation_response_agreement,null,2)}</pre></details>
    <details><summary>总体输出坐标矩阵与按文章留出的敏感度估计（非未来语言预测）</summary><pre>{JSON.stringify(summary?.metric_population?{geometry:summary.metric_population.geometry,summaries:summary.metric_population.summaries,paired_heldout:summary.metric_population.paired_heldout,limits:summary.metric_population.limits}:null,null,2)}</pre></details>
    <details><summary>完整确认、条件对照、微分检验与共享/交互项统计</summary><pre>{JSON.stringify({confirmation:summary?.confirmation,calculus:summary?.calculus,structure:summary?.structure,QA_atlas:summary?.QA_atlas,autonomous:compiled?.autonomous,metric_followup:summary?.metric_followup,metric_paired:summary?.metric_paired},null,2)}</pre></details>
    {summary?.identity_audit?.timestamp&&<><h3>追加纠错：前缀可用性与数值含义</h3><p>完整 token 审计发现 4 个未完成汉字字节片段被提前标成“其”指代线索；4096 个算子锚点均不受影响。纠正统计单独保存，不覆盖旧数组。原事件字段 event_onset_counts 实际统计最大整块变化幅值的位置，不能读成事件起始层；大幅变化也可能是反向写回。</p><details><summary>原始审计与全坐标更正依据</summary><pre>{JSON.stringify({identity:summary.identity_audit,correction:summary.cue_correction},null,2)}</pre></details></>}
  </section>;
}

function Theory({value}) {
  return <section id="operator-theory"><h2>RDC、核心拼图与未闭合部分</h2><p>理论名称保留为“条件化输出场闭合理论”。架构恒等式、经验预测规则和历史部分接口分开登记；本轮没有新增已经证明的一般语言闭合定理。</p>
    {value?.puzzles?.length?<><p>{value.scope}</p><div style={{overflowX:'auto'}}><table><thead><tr><th>Phase</th><th>保留的拼图</th><th>证据边界</th><th>证据状态</th></tr></thead><tbody>{value.puzzles.map(r=><tr key={r.phase}><td>{r.phase}</td><td>{r.retained_puzzle}</td><td>{r.boundary}</td><td>{evidenceLabels[r.evidence_status]||r.evidence_status}</td></tr>)}</tbody></table></div>
      <details><summary>统一接口、已知计算公式与已测试的局部规则</summary>{value.formulas.map(f=><article key={f.id}><h3>{f.id} · {f.kind}</h3><pre>{f.expression}</pre><p>{f.variables}</p><p>{f.evidence}</p></article>)}</details><details><summary>三图谱连接与尚未解释的部分</summary><pre>{JSON.stringify(value.global_atlas_interface,null,2)}</pre></details></>:<p>完整交付审计后更新理论索引；不以计划替代已验证结论。</p>}
  </section>;
}

export default function RdcOperatorAtlas() {
  const [summary,setSummary]=useState(null),[samples,setSamples]=useState([]),[error,setError]=useState(''),[revision,setRevision]=useState(0);
  async function refresh(){try{const [s,r]=await Promise.all([get('/overview'),get('/samples')]);setSummary(s);setSamples(r);setRevision(v=>v+1);setError('');}catch(e){setError(e.message);}}
  useEffect(()=>{let active=true;Promise.all([get('/overview'),get('/samples')]).then(([s,r])=>{if(active){setSummary(s);setSamples(r);}}).catch(e=>active&&setError(e.message));return()=>{active=false;};},[]);
  const reports=(summary?.operators?.reports||[]).filter(r=>r.split==='test'&&r.stratum==='all'&&(summary.choices[String(r.block)]===r.name||r.name==='frozen_gate_global'||r.name==='FP32_native_oracle'));
  return <main className="prefix-app operator-app"><header><p className="prefix-kicker">RDC · 原生条件坐标算子</p><h1>普通语言位置，完整坐标，真实输出</h1>
    <p>外部来源／问题／关系 ↔ 内部响应 ↔ 真实参数作用。观察、局部近似与机制证据分别标注；尚未证明一般语言闭合。</p>
    <nav><a href="/rdc-joint">历史联合图谱</a> · <a href="/rdc-relation">关系图谱</a> · <a href="/rdc-prefix">前缀图谱</a> · <button onClick={refresh}>刷新已提交研究结果</button></nav></header>
    <nav aria-label="本页研究查询"><a href="#operator-fields">原场与标量路径</a><a href="#operator-arrays">所有原生数组</a><a href="#operator-qa">自然问答</a><a href="#operator-generation">连续生成</a><a href="#operator-theory">理论与拼图</a><a href="#operator-figures">科学图</a></nav>
    {error&&<p role="alert">{error}</p>}
    <section><h2>当前证据与覆盖</h2><p>材料已冻结 {summary?.material?.sources??'—'} 个自然窗口；主采集 {summary?.capture?.main?.tokens??'—'} token，确认采集 {summary?.capture?.confirmation?.tokens??'未完成'} token。全部原生坐标不是“每份原场都永久落盘”，两者在下方分开。</p>
      <p>主问答 {summary?.QA?.qwen4?.main?.sources??'待执行'} 题；完整规范字符串匹配 {fmt(summary?.QA?.qwen4?.main?.normalized_full_EM)}，原生停止 {fmt(summary?.QA?.qwen4?.main?.native_EOS_fraction)}。这些分数不等于所有语义问题的正确率。</p>
      <p>选定局部算子只接收当前层可得的完整归一化输入 x 和前缀条件。复制完整原生公式的数值校准行是核对基准，不是新提取算法。</p>
      <div style={{overflowX:'auto'}}><table><thead><tr><th>block</th><th>完整坐标算子</th><th>相对 MSE</th><th>原值 MSE</th><th>方向余弦</th></tr></thead><tbody>{reports.map(r=><tr key={r.block+r.name}><td>{r.block}</td><td>{r.name}</td><td>{fmt(r.relative_MSE)}</td><td>{fmt(r.raw_MSE)}</td><td>{fmt(r.cosine_mean)}</td></tr>)}</tbody></table></div>
      <details><summary>查看审查结论、公式边界和阶段状态</summary><pre>{JSON.stringify({corrections:summary?.review?.corrections,plan:summary?.plan,integrity:summary?.integrity},null,2)}</pre></details>
    </section>
    <EvidenceResults summary={summary}/><NativeField samples={samples}/><ArrayBrowser key={'arrays'+revision}/><QA key={'qa'+revision}/><Ordering key={'order'+revision}/><Autonomous key={'auto'+revision}/><Theory value={summary?.theory}/>
    <section id="operator-figures"><h2>全坐标科学图</h2>{(summary?.figures||[]).map(f=><figure key={f.path}><img src={API+'/figure/'+f.path} alt={f.title} loading="lazy" style={{maxWidth:'100%'}}/><figcaption>{f.title} · {f.scope}</figcaption></figure>)}{!summary?.figures?.length&&<p>图表尚未提交。未用演示图代替真实实验。</p>}</section>
  </main>;
}
