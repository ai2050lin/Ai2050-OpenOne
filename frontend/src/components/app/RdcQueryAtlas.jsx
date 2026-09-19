import {useEffect,useState} from 'react';
import {Field} from './RdcJointAtlas.jsx';
import './RdcPrefixAtlas.css';
import './RdcJointAtlas.css';
import './RdcLawAtlas.css';
import './RdcBindingAtlas.css';
import './RdcUpdateAtlas.css';
import './RdcQueryAtlas.css';

const API=import.meta.env.VITE_RDC_QUERY_API||'http://127.0.0.1:5003/api/rdc-query';
async function get(path,params){const r=await fetch(API+path+(params?'?'+new URLSearchParams(params):''));if(!r.ok)throw new Error((await r.json().catch(()=>({}))).detail||r.statusText);return r.json();}
function ErrorLine({error}){return error?<p role="alert">{error}</p>:null;}
function Evidence({title,data,children}){return <details><summary>{title}</summary>{children}<pre>{JSON.stringify(data??{},null,2)}</pre></details>;}
const fmt=x=>x==null?'未提交':Number(x).toLocaleString('zh-CN');
const precision=x=>x!==0&&Math.abs(x)<1e-6?x.toExponential(3):x.toFixed(6);
const statistic=r=>r?.interval95?`${precision(r.mean)} [${precision(r.interval95[0])}, ${precision(r.interval95[1])}]`:'未提交';

function Findings({summary}){
 const prediction=summary?.prediction_analysis,formation=summary?.formation_history_analysis;
 const rows=Object.entries(prediction?.primary_coordinate_test?.candidate_mse||{}),late=formation?.late?.filter(r=>r.representation==='all')||[];
 const trained=formation?.formation?.paired_reports?.filter(r=>r.kind==='natural_content'&&r.cohort==='all'&&r.checkpoint==='deployed_BF16')||[];
 const calibration=summary?.identifiability?.calibration?.prospective_reports?.filter(r=>r.cohort==='all')||[];
 const confirmation=summary?.followup?.summary?.find(r=>r.cohort==='all'&&r.query_split==='unseen_query');
 const relations=summary?.identifiability?.relations?.matched_pair_reports?.filter(r=>r.variant==='native')||[];
 const identityBehavior=summary?.identifiability?.behavior?.summary?.filter(r=>r.family==='all')||[];
 return <section id="query-findings"><h2>实际结果：预测增量、学习与行为分开看</h2><p>下表来自已经提交的结果，不按有利方向挑选。区间为来源文档／语义组 bootstrap 的 95% 区间，并非坐标之间的独立重复。</p>
  {!rows.length&&<p>预测分析尚未提交。</p>}{!!rows.length&&<><h3>192 留出前缀 · 170 文档 · 20 未见查询</h3><div className="prefix-table"><table><thead><tr><th>冻结候选</th><th>全坐标 postnorm MSE</th><th>完整词表 KL</th></tr></thead><tbody>{rows.map(([name,mse])=><tr key={name}><td>{name}</td><td>{statistic(mse)}</td><td>{statistic(prediction.primary_vocabulary_test?.KL_native_to_prediction?.[name])}</td></tr>)}</tbody></table></div><p>相对 uniform 和 shuffled-values 的预定门：坐标 MSE {prediction.prespecified_candidate_qualification.MSE_ordered_vs_uniform_and_shuffle?'通过':'未通过'}；完整词表 KL {prediction.prespecified_candidate_qualification.KL_ordered_vs_uniform_and_shuffle?'通过':'未通过'}。这不是对整个语言机制的通过判定。</p></>}
  {!!trained.length&&<><h3>真实中层训练：实际 BF16 部署后的自然内容 NLL</h3><div className="prefix-table"><table><thead><tr><th>抽样种子</th><th>真实标签 − 原生</th><th>置乱标签 − 原生</th><th>真实 − 置乱</th></tr></thead><tbody>{trained.map(r=><tr key={r.seed}><td>{r.seed}</td><td>{statistic(r.natural_minus_baseline_NLL)}</td><td>{statistic(r.permuted_minus_baseline_NLL)}</td><td>{statistic(r.natural_minus_permuted_NLL)}</td></tr>)}</tbody></table></div><p>负值表示 NLL 降低。相同每步更新范数不等于累计参数位移相同；NLL 下降不能单独证明学到了关系机制。</p></>}
  {!!late.length&&<><h3>同一批 96 表达 · 24 语义组 · 自身历史生成</h3><div className="prefix-table"><table><thead><tr><th>分支</th><th>正确且 EOS</th><th>未解析 EOS</th><th>截断</th><th>解析错误</th><th>平均 token</th><th>下步 KV 改变／可比较</th></tr></thead><tbody>{late.map(r=><tr key={r.branch}><td>{r.branch}</td><td>{r.correct_and_stopped}/{r.expressions}</td><td>{r.unparsed_EOS}</td><td>{r.censored}</td><td>{r.parsed_wrong}</td><td>{r.mean_tokens.toFixed(2)}</td><td>{r.next_KV_changed}/{r.next_KV_compared}</td></tr>)}</tbody></table></div><p>形式、终值、停止、长度和历史分叉是不同指标；没有逐步判定完整推理文本。</p></>}
  {!!calibration.length&&<><h3>概率校准强对照 · 96 预留文档的新内容位置</h3><div className="prefix-table"><table><thead><tr><th>实际参数变体</th><th>原 NLL</th><th>各自 validation 校准后 NLL</th><th>校准后相对原生差</th></tr></thead><tbody>{calibration.map(r=><tr key={r.variant}><td>{r.variant}</td><td>{statistic(r.raw_NLL)}</td><td>{statistic(r.joint_calibrated_NLL)}</td><td>{statistic(r.joint_minus_original_joint)}</td></tr>)}</tbody></table></div><p>温度／训练词频先验只在 validation 选择；没有用于这里记录的自由生成，也不能替代参数位移匹配。</p></>}
  {confirmation&&<><h3>独立确认 · 96 预留文档 · 20 未见查询</h3><div className="prefix-table"><table><thead><tr><th>候选／信息条件</th><th>postnorm MSE</th><th>完整词表 KL</th></tr></thead><tbody>{Object.entries(confirmation.postnorm_MSE).map(([name,value])=><tr key={name}><td>{name}</td><td>{statistic(value)}</td><td>{statistic(confirmation.full_vocab_KL[name])}</td></tr>)}</tbody></table></div><p>文档配额为 GUM 19、EWT 39、CMRC 38。有序规则相对打乱值的优势区间跨过零；额外观察 query H12 是更晚的信息条件，不是原前缀预测器的胜出。</p></>}
  {!!relations.length&&<><h3>严格相同 token 多重集 · 原生关系分离</h3><div className="prefix-table"><table><thead><tr><th>关系族</th><th>配对数</th><th>答案方向分离度</th><th>两边首 argmax 均正确</th></tr></thead><tbody>{relations.map(r=><tr key={r.family}><td>{r.family}</td><td>{r.token_matched_pairs}</td><td>{statistic(r.answer_aligned_yes_probability_separation)}</td><td>{r.both_first_argmax_answers_correct}</td></tr>)}</tbody></table></div><p>正分离可反对纯词袋解释，但不能排除位置或浅层序列规律；二候选条件概率与自由生成分开报告。</p></>}
  {!!identityBehavior.length&&<><h3>严格配对 · 五个参数变体的自身历史生成</h3><div className="prefix-table"><table><thead><tr><th>参数变体</th><th>正确且 EOS／320</th><th>两边均正确／160</th><th>未解析 EOS</th><th>截断</th><th>相对原生成功率变化</th></tr></thead><tbody>{identityBehavior.map(r=><tr key={r.variant}><td>{r.variant}</td><td>{r.correct_and_stopped}</td><td>{r.paired_token_matched_both_correct}</td><td>{r.unparsed_EOS}</td><td>{r.censored}</td><td>{statistic(r.paired_success_change_vs_native)}</td></tr>)}</tbody></table></div><p>全部采用相同 B8 分组，最多 128 token；没有将校准概率或正确答案用于生成。参数变体来自已完成的训练，不是本阶段新增训练。</p></>}
 </section>;
}

function Sources({refresh}){
 const [scope,setScope]=useState('natural'),[model,setModel]=useState('qwen4'),[cohort,setCohort]=useState('gum'),[split,setSplit]=useState('test'),[offset,setOffset]=useState(0);
 const [storedIndex,setIndex]=useState({rows:[],total:0}),[sample,setSample]=useState(''),[mode,setMode]=useState('queries'),[layer,setLayer]=useState(12),[view,setView]=useState('raw');
 const [data,setData]=useState(null),[detail,setDetail]=useState(null),[error,setError]=useState(''),[query,setQuery]=useState(0),[candidate,setCandidate]=useState(4),[target,setTarget]=useState(2),[pred,setPred]=useState(null);
 const actualModel=scope==='scale'?model:'qwen4',actualCohort=['transfer','identifiability'].includes(scope)?'':cohort,actualSplit=['scale','identifiability'].includes(scope)?'':split;
 const indexID=JSON.stringify([scope,actualModel,actualCohort,actualSplit,offset,refresh]),index=storedIndex.id===indexID?storedIndex:{rows:[],total:0};
 useEffect(()=>{let alive=true;get('/samples',{scope,model:actualModel,cohort:actualCohort,split:actualSplit,offset,limit:100}).then(v=>{if(alive){setIndex({...v,id:indexID});setError('');}}).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[scope,actualModel,actualCohort,actualSplit,offset,indexID]);
 const chosen=index.rows.some(r=>r.sample_id===sample)?sample:index.rows[0]?.sample_id||'',row=index.rows.find(r=>r.sample_id===chosen);
 const identity=JSON.stringify([scope,actualModel,chosen,mode,layer,view]),sourceID=JSON.stringify([scope,actualModel,chosen]),predID=JSON.stringify([chosen,query,candidate,target]);
 async function load(){try{const [field,source]=await Promise.all([get('/field',{sample:chosen,scope,model:actualModel,mode,layer,view}),get('/sample',{sample:chosen,scope,model:actualModel})]);setData({identity,value:field});setDetail({id:sourceID,value:source});setError('');}catch(e){setData(null);setError(e.message);}}
 async function predict(){try{setPred({id:predID,value:await get('/prediction',{sample:chosen,query,candidate,target})});setError('');}catch(e){setPred(null);setError(e.message);}}
 return <section id="query-sources"><h2>自然历史 × 固定查询 × 完整坐标</h2><p>100 个探针是明确的中英文字符串，不保证是单 token，也不保证接在每段历史后都自然。后续状态采用原模型 BF16 位模式无损保存，低幅值坐标未删除。</p>
  <div className="prefix-controls"><label>材料范围<select aria-label="材料范围" value={scope} onChange={e=>{setScope(e.target.value);setOffset(0);setData(null);}}><option value="natural">万条自然前缀</option><option value="transfer">配对文字与代码</option><option value="scale">三模型匹配材料</option><option value="followup">独立后续确认</option><option value="identifiability">严格词汇身份配对</option></select></label>
   {scope==='scale'&&<label>查询模型<select aria-label="查询模型" value={model} onChange={e=>setModel(e.target.value)}>{['qwen4','qwen14','glm4'].map(m=><option key={m}>{m}</option>)}</select></label>}
   {!['transfer','identifiability'].includes(scope)&&<label>自然语料<select aria-label="自然语料" value={cohort} onChange={e=>{setCohort(e.target.value);setOffset(0);}}>{['gum','ewt','cmrc'].map(c=><option key={c}>{c}</option>)}</select></label>}
   {!['scale','identifiability'].includes(scope)&&<label>文档划分<select aria-label="文档划分" value={split} onChange={e=>{setSplit(e.target.value);setOffset(0);}}>{['train','validation','test','mixed_holdout','confirmation'].map(s=><option key={s}>{s}</option>)}</select></label>}
   <label className="prefix-wide">查询来源<select aria-label="查询来源" value={chosen} onChange={e=>setSample(e.target.value)}>{index.rows.map(r=><option key={r.sample_id} value={r.sample_id}>{r.sample_id} · {r.tokens} tokens · {r.detail?'详细场 · ':''}{r.captured?'已提交':'等待采集'}</option>)}</select></label>
   <label>显示对象<select aria-label="显示对象" value={mode} onChange={e=>setMode(e.target.value)}><option value="queries">100 查询全部坐标</option><option value="layers">前缀锚点全部层</option><option value="sources">详细样本 H12 全来源</option><option value="fixture">预定全场样例单层</option><option value="statistics">全词表统计矩阵</option></select></label>
   <label>样例层<input type="number" min={0} max={80} value={layer} onChange={e=>setLayer(Number(e.target.value))}/></label><label>场数值视图<select aria-label="场数值视图" value={view} onChange={e=>setView(e.target.value)}><option value="raw">原始值</option><option value="RMS">逐行 RMS</option></select></label><button disabled={!row?.captured} onClick={load}>读取完整查询场</button></div>
  <div className="query-pagination"><button disabled={offset===0} onClick={()=>setOffset(Math.max(0,offset-100))}>上一页来源</button><span>{offset+1}–{Math.min(offset+100,index.total)} / {index.total}；按冻结材料顺序分页</span><button disabled={offset+100>=index.total} onClick={()=>setOffset(offset+100)}>下一页来源</button></div>
  <ErrorLine error={error}/>{detail?.id===sourceID&&<><blockquote>{detail.value.text||detail.value.original_text}</blockquote><p>{detail.value.source_group} · {detail.value.historical_document_exposure?'历史研究曾涉及同一文档；本轮窗口不同':'本轮登记的来源身份'}</p><Evidence title="文本、分词、外部事件与类型化关系" data={detail.value}/></>}
  {data?.identity===identity&&data.value.axes&&<p>{data.value.axes}</p>}<Field data={data?.identity===identity?data.value:null}/>
  <h3>真正后续状态只作目标，不进入预测输入</h3><p>五个候选在相同训练/验证划分下比较：已观察前缀 H12/KV、已知查询的独立响应和原生 block12 参数进入提取器。完整后层状态和答案不能作为预测输入。query-only 只是附加特征不读取历史来源，共同解码器仍接收前缀 H12；它不是“无信息”对照。</p>
  <div className="prefix-controls"><label>查询编号<input aria-label="查询编号" type="number" min={0} max={99} value={query} onChange={e=>setQuery(Number(e.target.value))}/></label><label>候选规则<select value={candidate} onChange={e=>setCandidate(Number(e.target.value))}>{['query-only','uniform','quadratic','shuffled-values','ordered-softmax'].map((n,i)=><option key={n} value={i}>{n}</option>)}</select></label><label>预测目标<select value={target} onChange={e=>setTarget(Number(e.target.value))}>{['raw H24','raw H36','postnorm'].map((n,i)=><option key={n} value={i}>{n}</option>)}</select></label><button disabled={scope!=='natural'||!row?.detail||!row?.captured} onClick={predict}>读取全坐标预测对照</button></div>
  <Field data={scope==='natural'&&pred?.id===predID?pred.value:null}/>
 </section>;
}

function Ordered({refresh}){
 const [items,setItems]=useState([]),[path,setPath]=useState(''),[block,setBlock]=useState(16),[s,setS]=useState(0),[r,setR]=useState(0),[unit,setUnit]=useState(0),[input,setInput]=useState(0),[output,setOutput]=useState(0),[data,setData]=useState(null),[error,setError]=useState('');
 useEffect(()=>{get('/path-index').then(setItems).catch(e=>setError(e.message));},[refresh]);
 const normalized=items.map(x=>({...x,path:x.path.replaceAll('\\','/')})),chosen=normalized.some(x=>x.path===path)?path:normalized[0]?.path||'',row=normalized.find(x=>x.path===chosen),n=row?.audits?.[0]?.visible_sources||0,identity=JSON.stringify([chosen,block,s,r,unit,input,output]);
 async function load(){try{setData({identity,value:await get('/ordered-path',{path:chosen,block,source_s:s,source_r:r,unit,input_coordinate:input,output_coordinate:output})});setError('');}catch(e){setData(null);setError(e.message);}}
 const v=data?.identity===identity?data.value:null;
 return <section id="query-ordered"><h2>有序来源对 → 单元 → 标量参数</h2><p>门控分支的来源 s 与 up 分支的来源 r 分开记录。交换这两个来源可以改变单项；但全部来源的双重求和会抵消反对称部分。此处是带余项的原生计算核算，不是“一个单元就是一个概念”。</p>
  <div className="prefix-controls"><label className="prefix-wide">有序路径记录<select aria-label="有序路径记录" value={chosen} onChange={e=>{setPath(e.target.value);setS(0);setR(0);}}>{normalized.map(x=><option key={x.path} value={x.path}>{x.sample_id} · {x.mode} · {x.step??x.probe_index}</option>)}</select></label><label>路径层<select value={block} onChange={e=>setBlock(Number(e.target.value))}>{[16,35].map(b=><option key={b}>{b}</option>)}</select></label>
   {[['门控来源 s',s,setS,n],['up 来源 r',r,setR,n],['有序 MLP 单元',unit,setUnit,9727],['有序输入坐标',input,setInput,2559],['有序输出坐标',output,setOutput,2559]].map(([label,value,setter,max])=><label key={label}>{label}<input aria-label={label} type="number" min={0} max={max} value={value} onChange={e=>setter(Number(e.target.value))}/></label>)}<button disabled={!chosen} onClick={load}>读取有序参数路径</button></div>
  <ErrorLine error={error}/>{v&&<><pre>{JSON.stringify(v.chain,null,2)}</pre><p>{v.scope}</p><Field data={v.ordered_matrix}/><Field data={v.all_units}/><Field data={v.input_terms}/><Field data={v.native_output_terms}/></>}
 </section>;
}

function Events({refresh}){
 const [rows,setRows]=useState([]),[sample,setSample]=useState(''),[anchor,setAnchor]=useState(0),[data,setData]=useState(null),[error,setError]=useState('');
 useEffect(()=>{get('/event-index').then(setRows).catch(e=>setError(e.message));},[refresh]);
 const chosen=rows.some(r=>r.sample_id===sample)?sample:rows[0]?.sample_id||'',row=rows.find(r=>r.sample_id===chosen),actual=Math.min(anchor,(row?.anchors.length||1)-1),identity=JSON.stringify([chosen,actual]);
 async function load(){try{setData({identity,value:await get('/event',{sample:chosen,anchor:actual})});setError('');}catch(e){setData(null);setError(e.message);}}
 const v=data?.identity===identity?data.value:null;
 return <section id="query-events"><h2>真实生成时间上的全层与查询响应</h2><p>保留同一轨迹的固定时间点，以及输出文本中首次明确变量值和答案标记。步骤 j 的状态预测第 j 个生成 token；文字中出现变量不自动证明模型内部执行了符号程序。</p>
  <div className="prefix-controls"><label className="prefix-wide">事件轨迹<select aria-label="事件轨迹" value={chosen} onChange={e=>{setSample(e.target.value);setAnchor(0);}}>{rows.map(r=><option key={r.sample_id} value={r.sample_id}>{r.representation} · {r.sample_id}</option>)}</select></label><label>实际时间锚点<select aria-label="实际时间锚点" value={actual} onChange={e=>setAnchor(Number(e.target.value))}>{row?.anchors.map((j,i)=><option key={i} value={i}>生成步 {j}</option>)}</select></label><button disabled={!row?.captured} onClick={load}>读取时间锚点图谱</button></div><ErrorLine error={error}/>
  {v&&<><p>当前记录的原生生成步：{v.anchor_step}。{v.scope}</p><Evidence title="完整生成文本与外部事件时间" data={v.record.events}><blockquote>{v.native_generated_text}</blockquote></Evidence><pre>{JSON.stringify(v.record.steps[v.anchor_step],null,2)}</pre><Field data={v.layer_field}/><Field data={v.query_field}/></>}
 </section>;
}

function Behavior({refresh}){
 const [mode,setMode]=useState('late'),[branch,setBranch]=useState('native'),[storedRows,setRows]=useState({rows:[]}),[sample,setSample]=useState(''),[data,setData]=useState(null),[step,setStep]=useState(0),[error,setError]=useState('');
 const branches=mode==='late'?['native','fixed128_digit','entropy_digit','entropy_letter','terminal_marker_digit']:['native','code_identity','mapped_code'],actualBranch=branches.includes(branch)?branch:'native';
 const rowID=JSON.stringify([mode,actualBranch,refresh]),rows=storedRows.id===rowID?storedRows.rows:[];
 useEffect(()=>{let alive=true;get('/behavior-index',{mode,branch:actualBranch}).then(v=>alive&&setRows({id:rowID,rows:v})).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[mode,actualBranch,rowID]);
 const chosen=rows.some(r=>r.sample_id===sample)?sample:rows[0]?.sample_id||'',identity=JSON.stringify([mode,actualBranch,chosen]);
 async function load(){try{setData({identity,value:await get('/behavior',{mode,branch:actualBranch,sample:chosen})});setStep(0);setError('');}catch(e){setData(null);setError(e.message);}}
 const v=data?.identity===identity?data.value:null,j=Math.min(step,(v?.generated_ids.length||1)-1);
 return <section id="query-history"><h2>晚期偏置与跨表达注入：各自历史，各自评分</h2><p>数字偏置同时作用于 1–8，不提供正确答案；高熵不是“废话”的定义。当前读出修改不会改变已计算的缓存，但选出不同 token 后，后续历史和 KV 可以分叉。</p>
  <div className="prefix-controls"><label>实验集合<select aria-label="实验集合" value={mode} onChange={e=>setMode(e.target.value)}><option value="late">晚期类别偏置</option><option value="injection">文字与代码响应注入</option></select></label><label>真实分支<select aria-label="真实分支" value={actualBranch} onChange={e=>setBranch(e.target.value)}>{branches.map(b=><option key={b}>{b}</option>)}</select></label><label className="prefix-wide">生成样本<select aria-label="生成样本" value={chosen} onChange={e=>setSample(e.target.value)}>{rows.map(r=><option key={r.sample_id} value={r.sample_id}>{r.representation} · {r.sample_id}</option>)}</select></label><button disabled={!chosen} onClick={load}>读取自身历史生成</button></div><ErrorLine error={error}/>
  {v&&<><blockquote>{v.generated_text}</blockquote><div className="update-facts"><span>正确且停止：{String(v.answer_scoring.parsed_and_stopped_correct)}</span><span>EOS：{String(v.answer_scoring.EOS)}</span><span>上限截断：{String(v.answer_scoring.censored)}</span><span>已解析末答：{v.answer_scoring.conservative_final_answer??'未解析'}</span></div><p>未解析、截断、明确答错分别保留；推理链本身未评分。</p>
    <label className="update-step">生成步 {j+1}/{v.generated_ids.length}<input type="range" min={0} max={v.generated_ids.length-1} value={j} onChange={e=>setStep(Number(e.target.value))}/></label><pre>{JSON.stringify(v.steps[j],null,2)}</pre><Evidence title="干预触发、数值边界、KV 核对与原评分" data={{intervention:v.intervention,cache:v.cache_audit,next_KV:v.own_history_next_KV_comparison,scoring:v.answer_scoring,first_divergence:v.first_token_divergence_from_native??v.first_divergence}}/></>}
 </section>;
}

function Identity({refresh}){
 const [rows,setRows]=useState([]),[sample,setSample]=useState(''),[variant,setVariant]=useState('native'),[mode,setMode]=useState('queries'),[data,setData]=useState(null),[error,setError]=useState('');
 useEffect(()=>{let alive=true;get('/identity-index').then(v=>alive&&setRows(v)).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[refresh]);
 const chosen=rows.some(r=>r.sample_id===sample)?sample:rows[0]?.sample_id||'',identity=JSON.stringify([chosen,variant,mode]),v=data?.id===identity?data.value:null;
 async function load(){try{setData({id:identity,value:await get('/identity-pair',{sample:chosen,variant,mode})});setError('');}catch(e){setData(null);setError(e.message);}}
 return <section id="query-identity"><h2>相同 token，不同关系：训练效果是否可识别</h2><p>每对实际 token 的种类和次数、问题文本都相同，关系顺序或角色分配改变。原生 100 查询可在上方材料范围读取；这里对比同批形状的 6 查询、全部 MLP 单元及实际训练参数变体。另用留出材料选择概率校准，检查损失下降是否需要关系学习的解释。</p>
  {!rows.length&&<p>下一完整阶段尚未提交材料；不会用演示数据填充。</p>}<div className="prefix-controls"><label className="prefix-wide">身份配对材料<select aria-label="身份配对材料" value={chosen} onChange={e=>setSample(e.target.value)}>{rows.map(r=><option key={r.sample_id} value={r.sample_id}>{r.family} · {r.language} · world {r.world} · {r.sample_id}</option>)}</select></label>
   <label>实际训练变体<select aria-label="实际训练变体" value={variant} onChange={e=>setVariant(e.target.value)}>{['native','natural_target_2742','within_cohort_permuted_target_2742','natural_target_2743','within_cohort_permuted_target_2743'].map(x=><option key={x}>{x}</option>)}</select></label>
   <label>身份对照字段<select aria-label="身份对照字段" value={mode} onChange={e=>setMode(e.target.value)}><option value="queries">配对 6 查询 × 全坐标</option><option value="units">两层 gate/up/activation 全单元</option><option value="layers">指定层原始坐标</option></select></label><button disabled={!chosen} onClick={load}>读取严格身份对照</button></div><ErrorLine error={error}/>
  {v&&<><p>{v.scope}</p>{v.material.map((r,i)=><div key={r.sample_id}><h3>World {r.world} · {r.target}</h3><blockquote>{r.original_text}</blockquote><p>该原生变体的真实生成：</p><blockquote>{v.own_history[i]?.generated_text||'尚未提交生成'}</blockquote><Evidence title="明确关系、原始分词、实际输出与评分" data={{material:r,capture:v.capture[i],behavior:v.own_history[i]}}/></div>)}<Field data={v.field}/></>}
 </section>;
}

function Archive(){
 const [prefix,setPrefix]=useState('capture/fields/'),[index,setIndex]=useState({rows:[],total:0}),[offset,setOffset]=useState(0),[path,setPath]=useState(''),[headers,setHeaders]=useState([]),[name,setName]=useState(''),[row,setRow]=useState(0),[column,setColumn]=useState(0),[data,setData]=useState(null),[error,setError]=useState('');
 useEffect(()=>{let alive=true;get('/archives',{prefix,offset,limit:100}).then(v=>{if(alive){setIndex(v);setPath(v.rows[0]?.path||'');setHeaders([]);}}).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[prefix,offset]);
 useEffect(()=>{let alive=true;if(path)get('/arrays',{path}).then(v=>{if(alive){setHeaders(v);setName(v[0]?.array||'');setRow(0);setColumn(0);}}).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[path]);
 const identity=JSON.stringify([path,name,row,column]);
 async function load(){try{setData({identity,value:await get('/array',{path,name,row_start:row,row_count:37,start:column,count:8192})});setError('');}catch(e){setData(null);setError(e.message);}}
 return <section id="query-archive"><h2>所有原始数组：按原索引分页查询</h2><p>完整档案可下载。最后一轴可能是坐标、MLP 单元、词表、来源、查询或参数；不能只凭数组形状把它们混称“神经元”。档案存在不等于所属 Phase 已通过验证。</p>
  <div className="prefix-controls"><label>档案路径前缀<input aria-label="档案路径前缀" value={prefix} onChange={e=>{setPrefix(e.target.value);setOffset(0);}}/></label><label className="prefix-wide">数值档案<select aria-label="数值档案" value={path} onChange={e=>setPath(e.target.value)}>{index.rows.map(r=><option key={r.path}>{r.path}</option>)}</select></label><label>原张量<select aria-label="原张量" value={name} onChange={e=>{setName(e.target.value);setRow(0);setColumn(0);}}>{headers.map(h=><option key={h.array} value={h.array}>{h.array} [{h.shape.join('×')}]</option>)}</select></label><label>展平起始行<input type="number" min={0} value={row} onChange={e=>setRow(Number(e.target.value))}/></label><label>原生起始列<input type="number" min={0} value={column} onChange={e=>setColumn(Number(e.target.value))}/></label><button disabled={!name} onClick={load}>读取原序数值页</button>{path&&<a href={API+'/download?'+new URLSearchParams({path})}>下载完整档案</a>}</div>
  <div className="query-pagination"><button disabled={!offset} onClick={()=>setOffset(Math.max(0,offset-100))}>上一页档案</button><span>{offset+1}–{Math.min(offset+100,index.total)} / {index.total}</span><button disabled={offset+100>=index.total} onClick={()=>setOffset(offset+100)}>下一页档案</button></div><ErrorLine error={error}/>{data?.identity===identity&&<p>原张量 [{data.value.tensor_shape.join('×')}]；展平行 {data.value.row_start}–{data.value.row_end} / {data.value.total_rows}</p>}<Field data={data?.identity===identity?data.value:null}/>
 </section>;
}

export default function RdcQueryAtlas(){
 const [summary,setSummary]=useState(null),[error,setError]=useState(''),[refresh,setRefresh]=useState(0);
 useEffect(()=>{let alive=true;get('/overview').then(v=>{if(alive){setSummary(v);setError('');}}).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[refresh]);
 return <main className="prefix-app law-app binding-app update-app query-app"><header><small>RDC · NATIVE QUERY RESPONSE / ORDERED SOURCES / ACTUAL EVENT TIME</small><h1>条件查询与有序来源图谱</h1><p>共同现象 → 可执行候选 → 原生参数 → 未见材料预测 → 实际训练与生成</p><nav><a href="#query-evidence">证据状态</a><a href="#query-findings">实际结果</a><a href="#query-sources">百万查询场</a><a href="#query-ordered">有序参数路径</a><a href="#query-events">生成时间</a><a href="#query-history">自身历史测试</a><a href="#query-identity">严格身份对照</a><a href="#query-archive">全部数组</a><a href="#query-theory">RDC 与完整拼图</a><a href="/rdc-update">前轮研究 ↗</a></nav></header>
  <ErrorLine error={error}/><section id="query-evidence" className="prefix-warning"><h2>先发现结构，分别验证每一种主张</h2><p>查询响应是有限实验记录；有序分支是可追溯计算因子；预测与训练收益必须胜过具体对照。没有把它们直接改名为“流形同构”或“语言机制已闭合”。</p><button onClick={()=>setRefresh(v=>v+1)}>刷新查询研究结果</button>
   <div className="law-metrics"><div><b>{fmt(summary?.committed_natural_prefixes)}</b><span>已提交自然前缀</span></div><div><b>{fmt(summary?.committed_query_endpoints)}</b><span>完整查询端点</span></div><div><b>{fmt(summary?.material?.source_documents)}</b><span>来源文档；窗口不独立</span></div><div><b>{fmt(summary?.formation?.runs?.length)}</b><span>真实／打乱目标训练轨迹</span></div></div>
   <Evidence title="附件审查与严格结论边界" data={summary?.contract}/><Evidence title="完成状态、原始精度与有限资源账本" data={{completion:summary?.completion,resources:summary?.resources,ledger:summary?.ledger,verification:summary?.final}}/>
   <Evidence title="百万响应、可达历史对与真实生成事件" data={{atlas:summary?.atlas,pairs:summary?.pairs,events:summary?.events}}/>
   <Evidence title="同输入预算候选：全坐标误差与全词表误差" data={{rules:summary?.rules,vocabulary:summary?.vocabulary}}/>
   <Evidence title="文字与代码配对映射及一次性注入" data={{mapping:summary?.transfer,vocabulary:summary?.transfer_vocabulary,injection:summary?.injection}}/>
   <Evidence title="实际形成、晚期偏置和三模型核对" data={{formation:summary?.formation,late:summary?.late,scale:summary?.scale}}/>
   <Evidence title="高跨模型相关的探索性查询身份对照" data={summary?.theory?.query_identity_control}/>
   <Evidence title="中英文分组之后：语言内部完整查询矩阵" data={summary?.theory?.query_language_control}/>
   <Evidence title="严格身份配对与概率校准：实际续研" data={summary?.identifiability}/>
   <Evidence title="总体拟合是否预测了关系变化：零变化强基线" data={summary?.theory?.identifiability_pair_change_control}/>
   <Evidence title="下一完整阶段资源估计（不是科学问题已解决）" data={{admission:summary?.next_stage_admission,after_identifiability:summary?.continuation_after_2744}}/>
  </section><Findings summary={summary}/><Sources refresh={refresh}/><Ordered refresh={refresh}/><Events refresh={refresh}/><Behavior refresh={refresh}/><Identity refresh={refresh}/><Archive/>
  <section id="query-theory"><h2>RDC 主体、累计拼图与尚未解决的问题</h2><p>定义、恒等式、拟合模型、经验规律和独立确认分别标记。有限查询相近不意味着任意未来等价；继续训练不等于重建预训练形成史。</p>
   {!summary?.theory?.puzzles?.length&&<p>本轮完整理论汇总尚未提交，不能用计划填充已完成结果。</p>}
   <div className="prefix-table"><table><thead><tr><th>Phase</th><th>保留拼图</th><th>边界</th><th>证据状态</th></tr></thead><tbody>{summary?.theory?.puzzles?.map((p,i)=><tr key={p.phase+'_'+i}><td>{p.phase}</td><td>{p.retained_puzzle}</td><td>{p.boundary}</td><td>{p.evidence_status}</td></tr>)}</tbody></table></div>
   {summary?.theory?.formulas?.map(f=><Evidence key={f.id} title={`${f.id} · ${f.kind}`} data={f}/>)}<Evidence title="同目标自动续研：实际独立确认与资源准入" data={summary?.followup}/>
  </section><section><h2>全坐标科学图与对照结果</h2><p>原坐标顺序不变，原始值与归一化值分开。展示缩放不是用于定义主干的降维；每张图均可回查数值。</p>{summary?.figures?.map(f=><figure key={f.path}><figcaption>{f.title}</figcaption><a href={API+'/figure/'+f.path} target="_blank" rel="noreferrer"><img loading="lazy" src={API+'/figure/'+f.path} alt={f.title}/></a><p>{f.scope}</p></figure>)}</section>
  <footer>独立 GET 只读服务默认端口 5003；不加载模型或写入参数。既有 5001、5002 和旧研究页面保持不变。原始数值与重算配置持续保留。</footer>
 </main>;
}
