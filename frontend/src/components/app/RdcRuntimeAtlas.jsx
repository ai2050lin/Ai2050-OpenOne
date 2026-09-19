import {useEffect,useState} from 'react';
import {Field} from './RdcJointAtlas.jsx';

const API=import.meta.env.VITE_RDC_CONSTRUCTION_API||'http://127.0.0.1:5004/api/rdc-construction';
const fmt=x=>x==null?'未提交':Number(x).toFixed(6);
async function get(path,args={}){const r=await fetch(API+path+'?'+new URLSearchParams(args));if(!r.ok)throw new Error((await r.json().catch(()=>({}))).detail||r.statusText);return r.json();}
function Evidence({title,data}){return <details><summary>{title}</summary><pre>{JSON.stringify(data??{},null,2)}</pre></details>;}
const short=name=>name==='training_current_input_token_H36_mean'?'当前输入 token 训练均值':name.endsWith('__direct_complete_coordinate_H36')?'直接预测 H36':name.endsWith('__predicted_H35_general_Q35_native_remainder')?'一般 Q + 原生余部':name;

export function RuntimeFields({refresh}){
  const [cohort,setCohort]=useState('discovery'),[family,setFamily]=useState(''),[sample,setSample]=useState(''),[index,setIndex]=useState(null);
  const [step,setStep]=useState(0),[mode,setMode]=useState('units'),[field,setField]=useState('product'),[block,setBlock]=useState(12),[view,setView]=useState('raw');
  const [data,setData]=useState(null),[detail,setDetail]=useState(null),[links,setLinks]=useState(null),[error,setError]=useState('');
  useEffect(()=>{let alive=true;get('/runtime-samples',{cohort}).then(value=>alive&&setIndex({cohort,value})).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[cohort,refresh]);
  const all=index?.cohort===cohort?index.value:[],families=[...new Set(all.map(r=>r.family))],options=all.filter(r=>!family||r.family===family);
  const chosen=options.some(r=>r.sample_id===sample)?sample:options[0]?.sample_id||'';
  const id=JSON.stringify([cohort,chosen,step,mode,field,block,view]),sourceID=JSON.stringify([cohort,chosen]);
  const names=mode==='coordinates'?['attention_input','attention_write','pre_MLP','MLP_input','MLP_write']:['gate','up','product'];
  const selectedField=names.includes(field)?field:names[0];
  async function load(){try{const [a,b]=await Promise.all([get('/runtime-field',{cohort,sample:chosen,step,mode,field:selectedField,block,view}),get('/runtime',{cohort,sample:chosen})]);setData({id,value:a});setDetail({id:sourceID,value:b});setError('');}catch(e){setData(null);setError(e.message);}}
  async function loadLinks(){try{setLinks({id,value:await get('/mechanism-case',{cohort,sample:chosen,step,block})});setError('');}catch(e){setLinks(null);setError(e.message);}}
  return <section id="construction-runtime"><h2>原生自身历史：全层、全坐标、全部单元</h2>
    <p>不是附加诊断查询。每一行来自实际连续生成；字段在第 t 步产生、随后读出第 t 个 token。MLP/来源全场保留前8步，所有真实生成步保留全部 H；不存在的后续场明确返回缺失。</p>
    <div className="prefix-controls"><label>运行材料<select aria-label="运行材料" value={cohort} onChange={e=>{setCohort(e.target.value);setFamily('');}}><option value="discovery">896 既有表达的新运行谱</option><option value="confirmation">512 新文档 / 新组合确认</option></select></label>
      <label>运行语言族<select aria-label="运行语言族" value={family} onChange={e=>setFamily(e.target.value)}><option value="">全部语言族</option>{families.map(f=><option key={f}>{f}</option>)}</select></label>
      <label className="prefix-wide">运行表达<select aria-label="运行表达" value={chosen} onChange={e=>setSample(e.target.value)}>{options.map(r=><option key={r.sample_id} value={r.sample_id}>{r.family} · {r.language} · {r.sample_id}</option>)}</select></label>
      <label>真实生成步<input aria-label="真实生成步" type="number" min={0} max={127} value={step} onChange={e=>setStep(Number(e.target.value))}/></label>
      <label>运行场类型<select aria-label="运行场类型" value={mode} onChange={e=>setMode(e.target.value)}><option value="units">每层全部 MLP 单元</option><option value="coordinates">每层全部残差坐标</option><option value="hidden">H0..H36 全场</option><option value="postnorm">最终 postnorm</option><option value="attention">全头全来源 attention</option><option value="query">全头 Q（RoPE 前）</option></select></label>
      <label>运行场字段<select aria-label="运行场字段" value={selectedField} onChange={e=>setField(e.target.value)}>{names.map(n=><option key={n}>{n}</option>)}</select></label>
      <label>来源 / Q 的块<input aria-label="运行参数块" type="number" min={0} max={35} value={block} onChange={e=>setBlock(Number(e.target.value))}/></label>
      <label>运行数值视图<select aria-label="运行数值视图" value={view} onChange={e=>setView(e.target.value)}><option value="raw">原始值</option><option value="RMS">完整行 RMS</option></select></label><button disabled={!chosen} onClick={load}>读取自身历史全场</button><button disabled={!chosen} onClick={loadLinks}>关联参数、传播与输出证据</button></div>
    {error&&<p role="alert" className="prefix-error">{error}</p>}
    {detail?.id===sourceID&&<><blockquote>{detail.value.material.original_text}</blockquote><p>实际输出：</p><blockquote>{detail.value.record.generated_text}</blockquote><p>生成步 {detail.value.record.generated_ids.length}；EOS {String(detail.value.record.EOS)}；达到上限 {String(detail.value.record.censored)}。{detail.value.material.novelty}</p><Evidence title="实际输入、token 身份、保存边界与证据地址" data={detail.value}/></>}
    {data?.id===id&&<p>{data.value.axes} 本步输出 token ID：{data.value.emitted_token_id}。</p>}<Field data={data?.id===id?data.value:null}/>
    {links?.id===id&&<Evidence title="同一表达 / 步的参数地址、全场、尾网络效应与冻结预测" data={links.value}/>}
  </section>;
}

export function HistoryFindings({summary}){
  const h=summary?.phase2746?.history_prediction,confirmation=h?.confirmation;
  const [cohort,setCohort]=useState('confirmation'),[subset,setSubset]=useState('natural'),[first,setFirst]=useState(false);
  const source=cohort==='confirmation'?confirmation?.analysis:h?.analysis;
  const names=[h?.frozen?.autonomous_direct_baseline_route,h?.frozen?.autonomous_primary_native_constrained_route,'training_current_input_token_H36_mean'];
  const report=source?.reports?.find(r=>r.subset===subset&&(first?r.horizon==='first_position':r.horizon!=='first_position'));
  const pairs=source?.pair_reports?.find(r=>r.subset==='all');
  const answers=source?.controlled_first_answers?.results?.filter(r=>names.includes(r.route))||[];
  const reuse=summary?.phase2746?.runtime_reuse?.reports||[];
  const differential=summary?.phase2746?.differential;
  const diffrows=differential?.reports?.filter(r=>r.family==='all'&&r.direction===0&&r.epsilon===0.01)||[];
  return <section id="construction-history-results"><h2>条件运行规律：候选预测与适用边界</h2>
    <h3>全单元相邻步相关：移除共同输出阶段均值</h3><p>同一批896表达、相同前三步；表格只列两个层的全9728单元中位数，完整36层数组仍保留。去掉阶段均值不等于去掉词汇、位置、历史等全部混杂；不同层的同一索引不自动是同一功能。</p>
    {!!reuse.length&&<div className="prefix-table"><table><thead><tr><th>语言族</th><th>块</th><th>混合阶段相关中位数</th><th>阶段内相关中位数</th></tr></thead><tbody>{reuse.flatMap(r=>r.summaries.filter(v=>v.view==='raw'&&[0,35].includes(v.block)).map(v=><tr key={r.family+v.block}><td>{r.family}</td><td>{v.block}</td><td>{fmt(v.median_product_pooled_correlation)}</td><td>{fmt(v.median_product_within_step_correlation)}</td></tr>))}</tbody></table></div>}
    <h3>完整余下网络传播：原生精度的有限变化不能混作光滑导数</h3><p>{differential?.native_tails??'未提交'} 个原生尾网络、{differential?.direction_epsilon_comparisons??'未提交'} 个方向/尺度比较；下表为固定稠密方向、ε=0.01、同一51来源组的误差均值。JVP/VJP 是局部计算关系，不是语言定理。</p>
    {!!diffrows.length&&<div className="prefix-table"><table><thead><tr><th>入口</th><th>FP32 中心差分相对误差</th><th>BF16 有限变化相对误差</th><th>忽略中间网络读出对照</th></tr></thead><tbody>{diffrows.map(r=><tr key={r.start}><td>H{r.start}</td><td>{fmt(r.measures.postnorm_central.source_group_mean.mean)}</td><td>{fmt(r.measures.BF16_postnorm_central.source_group_mean.mean)}</td><td>{fmt(r.measures.bare_readout_control.source_group_mean.mean)}</td></tr>)}</tbody></table></div>}
    <h3>冻结规则：真实早期状态 + 可用过去 KV → 当前末端</h3><p>输入为完整 H0/H12 和 B35(H12;过去KV) 的7680坐标，禁止使用当前真实H35/H36/Q35或答案。直接H36与一般Q路径都在旧验证集固定λ=0.001；原生约束一般Q只是在该类路线内选中，并非所有误差指标最优。</p>
    <div className="prefix-controls"><label>预测检验材料<select aria-label="预测检验材料" value={cohort} onChange={e=>setCohort(e.target.value)}><option value="confirmation">192新文档 + 320新表达</option><option value="discovery">旧暴露材料的拟合保留组</option></select></label><label>预测报告类别<select aria-label="预测报告类别" value={subset} onChange={e=>setSubset(e.target.value)}><option value="natural">自然语料</option><option value="controlled">受控语言族</option><option value="all">全部语言族等权</option></select></label><label>预测位置<select aria-label="预测位置" value={first?'first':'three'} onChange={e=>setFirst(e.target.value==='first')}><option value="three">前三个实际位置</option><option value="first">首个位置</option></select></label></div>
    {cohort==='confirmation'&&<p>新文档仅 EWT25 + CMRC167；没有新的 GUM 文档，不冒充三语料确认。五类新表达共160对完整token多重集一致。自然读出忠实度不是内容正确率。</p>}
    <div className="prefix-table"><table><thead><tr><th>冻结路径</th><th>全状态 MSE</th><th>完整词表 KL</th><th>原生 token 一致率</th><th>KL − 直接路径：95%组区间</th></tr></thead><tbody>{report?.results?.filter(r=>names.includes(r.route)).map(r=><tr key={r.route}><td>{short(r.route)}</td><td>{fmt(r.metrics.postnorm_MSE.mean)}</td><td>{fmt(r.metrics.KL_reference_prediction.mean)}</td><td>{fmt(r.metrics.argmax_agrees_original_native.mean)}</td><td>{r.metrics.KL_reference_prediction.minus_direct_interval95.map(fmt).join(' → ')}</td></tr>)}</tbody></table></div>
    <p>分层 source-group 区间；每个语料/关系族等权，不把坐标、世界、翻译或生成步当成独立案例。新确认中源V置乱特征与原特征收益接近，不能声称精确来源配对已被证明关键。</p>
    <h3>关系变化与真正的答案：不能用句号/EOS的一致率代替</h3><div className="prefix-table"><table><thead><tr><th>路径</th><th>关系变化 MSE</th><th>变化 MSE − 零变化</th><th>首 token 外部答案正确率</th><th>两候选条件正确率</th></tr></thead><tbody>{answers.map(r=>{const p=pairs?.results?.find(v=>v.route===r.route);return <tr key={r.route}><td>{short(r.route)}</td><td>{fmt(p?.metrics.pair_change_MSE.mean)}</td><td>{fmt(p?.metrics.pair_MSE_minus_zero.mean)}</td><td>{fmt(r.metrics.first_token_gold_match.mean)}</td><td>{fmt(r.metrics.candidate_conditional_gold_match.mean)}</td></tr>;})}</tbody></table></div>
    <Evidence title="全部冻结路线、置乱对照、配对区间与数值底噪地址" data={h}/><Evidence title="全部单元相关与完整跨层微分证据" data={{reuse:summary?.phase2746?.runtime_reuse,differential}}/>
    <details><summary>全原生坐标科学图：原值、RMS、有限尺度与自治边界</summary>{summary?.phase2746?.figures?.figures?.map(r=><figure key={r.path}><img style={{width:'100%',height:'auto'}} loading="lazy" src={API+'/figure?stage=2746&name='+encodeURIComponent(r.path.split('/').pop().replace('.png',''))} alt={r.caption}/><figcaption>{r.caption}</figcaption></figure>)}</details>
  </section>;
}

export function SelfHistory({summary,refresh}){
  const [index,setIndex]=useState([]),[sample,setSample]=useState(''),[route,setRoute]=useState('frozen_generalQ_native'),[field,setField]=useState('postnorm');
  const [value,setValue]=useState(null),[map,setMap]=useState(null),[error,setError]=useState('');
  useEffect(()=>{let alive=true;get('/runtime-samples',{cohort:'confirmation'}).then(v=>alive&&setIndex(v)).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[refresh]);
  const chosen=index.some(r=>r.sample_id===sample)?sample:index[0]?.sample_id||'',id=JSON.stringify([chosen,route,field]);
  const c=summary?.phase2746?.history_prediction?.confirmation;
  const report=c?.autonomous_analysis?.reports?.find(r=>r.subset==='controlled'),natural=c?.autonomous_analysis?.reports?.find(r=>r.subset==='natural');
  async function load(){try{const v=await get('/self-history',{sample:chosen});setValue({sample:chosen,value:v});setError('');}catch(e){setValue(null);setError(e.message);}}
  async function loadMap(){try{setMap({id,value:await get('/self-history-field',{sample:chosen,route,field})});setError('');}catch(e){setMap(null);setError(e.message);}}
  return <section id="construction-self-history"><h2>自行延续历史：预测器能否继续走下去</h2>
    <p>已提交 {c?.autonomous_progress?.rows??0} /512 表达。一次原生前缀初始化后，各分支使用自己的token和KV，不再注入原模型未来状态。原生B1基线也采用相同前缀/单步执行形状；较早的B8原生续写只是另列数值参照。</p>
    {report&&<div className="prefix-table"><table><thead><tr><th>实际自身历史路径</th><th>受控正确且停止</th><th>自然完整轨迹与原模型一致</th><th>自然至少8次连续同token重复</th></tr></thead><tbody>{report.routes.map(r=>{const n=natural?.routes?.find(v=>v.route===r.route);return <tr key={r.route}><td>{r.route}</td><td>{fmt(r.metrics.correct_and_stopped.mean)}</td><td>{fmt(n?.metrics.full_sequence_equal_native.mean)}</td><td>{fmt(n?.metrics.contains_run_at_least8.mean)}</td></tr>;})}</tbody></table></div>}
    {report&&<p>自然重复不是完整语义质量评分，但这里显示明确退化。标准化输入RMS未越过训练最大值不能保证条件或方向仍在训练分布内。原生B1/B8有{c.autonomous_analysis.natural_B1_B8_full_sequence_disagreements}篇自然轨迹、{c.autonomous_analysis.controlled_B1_B8_full_sequence_disagreements}条受控输出不同，因此候选的比较基线固定为同形状B1。</p>}
    <div className="prefix-controls"><label className="prefix-wide">自身历史表达<select aria-label="自身历史表达" value={chosen} onChange={e=>setSample(e.target.value)}>{index.map(r=><option key={r.sample_id} value={r.sample_id}>{r.family} · {r.language} · {r.sample_id}</option>)}</select></label><button disabled={!chosen} onClick={load}>读取三个自身历史分支</button>
      <label>自身历史路径<select aria-label="自身历史路径" value={route} onChange={e=>setRoute(e.target.value)}><option value="native_B1_cache">原模型 B1</option><option value="frozen_direct">直接 H36 预测</option><option value="frozen_generalQ_native">一般 Q + 原生余部</option></select></label>
      <label>自身历史场<select aria-label="自身历史场" value={field} onChange={e=>setField(e.target.value)}>{['postnorm','H0','H12','H35','H36','Q35','candidate'].map(f=><option key={f}>{f}</option>)}</select></label><button disabled={!chosen} onClick={loadMap}>读取全部逐步坐标</button></div>
    {error&&<p role="alert" className="prefix-error">{error}</p>}
    {value?.sample===chosen&&<><blockquote>{value.value.material.original_text}</blockquote><p>外部目标：{value.value.material.target??'自然续写无唯一参考答案'}。当前H35及H36在候选中来自预测；H36显示线性头输出，不等同一般Q分支实际编译的末层输出。</p><div className="construction-output-grid">{Object.entries(value.value.records).map(([name,r])=><article key={name}><h3>{name}</h3>{!r.generated_ids?<p>主结果尚未提交，不以试运行代替。</p>:<><blockquote>{r.generated_text||'仅特殊 token'}</blockquote><p>{r.generated_ids.length}步；EOS {String(r.EOS)}；正确且停止 {r.answer_scoring?String(r.answer_scoring.parsed_and_stopped_correct):'不适用'}。</p></>}</article>)}</div><Evidence title="完整自身历史与数值/源状态检查" data={value.value}/></>}
    {map?.id===id&&<p>{map.value.axes}</p>}<Field data={map?.id===id?map.value:null}/><Evidence title="自身历史完整主实验及分析状态" data={{result:c?.autonomous_result,analysis:c?.autonomous_analysis}}/>
  </section>;
}
