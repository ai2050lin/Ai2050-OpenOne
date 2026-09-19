import {useEffect,useState} from 'react';
import {Field} from './RdcJointAtlas.jsx';
import FormationProgramEvidence from './RdcFormationProgramEvidence.jsx';
import './RdcFormationEvidence.css';

const API=import.meta.env.VITE_RDC_CONSTRUCTION_API||'http://127.0.0.1:5004/api/rdc-construction';
async function get(path,params={}){const r=await fetch(API+path+'?'+new URLSearchParams(params));if(!r.ok)throw new Error((await r.json()).detail||r.statusText);return r.json();}

export default function FormationEvidence({data,refresh}){
  const [figure,setFigure]=useState(''),[model,setModel]=useState('qwen4'),[variant,setVariant]=useState('native');
  const [ownList,setOwnList]=useState({id:'',rows:[]}),[ownSample,setOwnSample]=useState(''),[ownField,setOwnField]=useState('postnorm'),[step,setStep]=useState(0);
  const [propList,setPropList]=useState([]),[propSample,setPropSample]=useState(''),[propRun,setPropRun]=useState('true_token_2747');
  const [propField,setPropField]=useState('hidden'),[route,setRoute]=useState('full_prefix');
  const [ownLoaded,setOwnLoaded]=useState(null),[propLoaded,setPropLoaded]=useState(null),[error,setError]=useState(''),[busy,setBusy]=useState('');
  const ownRun=model==='qwen4'?variant:'native';const listId=JSON.stringify([model,ownRun]);
  useEffect(()=>{let alive=true;get('/formation-own-samples',{model,variant:ownRun}).then(rows=>alive&&setOwnList({id:listId,rows})).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[model,ownRun,listId,refresh]);
  useEffect(()=>{let alive=true;get('/formation-propagation-samples').then(r=>alive&&setPropList(r)).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[refresh]);
  const ownRows=ownList.id===listId?ownList.rows:[];
  const os=ownRows.some(r=>r.sample_id===ownSample)?ownSample:ownRows[0]?.sample_id||'';
  const ps=propList.some(r=>r.sample_id===propSample)?propSample:propList[0]?.sample_id||'';
  const ownId=JSON.stringify([model,ownRun,os,ownField,step]);const propId=JSON.stringify([ps,propRun,propField,route]);
  const own=ownLoaded?.id===ownId?ownLoaded.value:null;const prop=propLoaded?.id===propId?propLoaded.value:null;
  const selected=(data?.figures||[]).find(r=>r.name===figure)||(data?.figures||[])[0];
  async function loadOwn(){setBusy('own');try{setOwnLoaded({id:ownId,value:await get('/formation-own-field',{model,variant:ownRun,sample:os,field:ownField,step})});setError('');}catch(e){setOwnLoaded(null);setError(e.message);}finally{setBusy('');}}
  async function loadProp(){setBusy('prop');try{setPropLoaded({id:propId,value:await get('/formation-propagation-field',{sample:ps,run:propRun,field:propField,route})});setError('');}catch(e){setPropLoaded(null);setError(e.message);}finally{setBusy('');}}
  return <div id="formation-evidence">
    <h3>已核验的形成与完整前缀图谱</h3>
    <p>总体改善与分族反例同时保留。FP32 平滑传播、原生 BF16 有限变化及自身历史是不同实验对象，不把近似公式当作已经破解的语言编码器。</p>
    {selected&&<><label>形成研究图<select aria-label="形成研究图" value={selected.name} onChange={e=>setFigure(e.target.value)}>{(data?.figures||[]).map(r=><option key={r.name} value={r.name}>{r.name}</option>)}</select></label>
      <figure><img key={selected.name} src={API+'/figure?stage=2747&name='+encodeURIComponent(selected.name)} alt={selected.name} style={{maxWidth:'100%',height:'auto'}}/><figcaption>{selected.caption}</figcaption></figure></>}
    <section id="formation-propagation-view"><h3>完整前缀：单表达、单方向、全部跨层坐标</h3>
      <div className="prefix-controls"><label className="prefix-wide">传播表达<select aria-label="传播表达" value={ps} onChange={e=>setPropSample(e.target.value)}>{propList.map(r=><option key={r.sample_id} value={r.sample_id}>{r.family} · {r.sample_id}</option>)}</select></label>
        <label>传播方向<select aria-label="传播方向" value={propRun} onChange={e=>setPropRun(e.target.value)}>{(data?.runs||[]).map(r=><option key={r.run}>{r.run}</option>)}</select></label>
        <label>传播原生场<select aria-label="传播原生场" value={propField} onChange={e=>setPropField(e.target.value)}>{['hidden','Q','gate','up','product','MLP_input','MLP_write'].map(v=><option key={v}>{v}</option>)}</select></label>
        <label>传播路径<select aria-label="传播路径" value={route} onChange={e=>setRoute(e.target.value)}><option value="full_prefix">完整前缀</option><option value="last_position_only">仅末位置初始变化</option><option value="difference">两路径之差</option></select></label>
        <button disabled={!ps||Boolean(busy)} onClick={loadProp}>{busy==='prop'?'正在读取20层…':'读取完整传播场'}</button></div>
      {prop&&<><blockquote>{prop.material.original_text}</blockquote><p>{prop.axes}</p><p>{prop.boundary}</p></>}<Field data={prop}/>
    </section>
    <section id="formation-own-view"><h3>真实自身历史：逐生成步完整坐标</h3>
      <p>每步完整 postnorm；预先选定44条表达另存每步全部H边界。自然续写没有唯一标准答案；EOS、达到长度上限及受控题完整答案分别报告。跨模型 token 数不能直接当作相同长度。</p>
      <ul>{(data?.own_runs||[]).map(r=><li key={r.model+r.variant}>{r.model} / {r.variant}：{r.complete?'已完成':`已提交 ${r.progress?.pilot?'先导 ':''}${r.progress?.expressions||0} / ${r.progress?.total||512}`}
        {!r.complete&&r.wave_progress?.stage&&`；最近记录：波次 ${r.wave_progress.wave+1}，${r.wave_progress.stage==='first_B1'?'已完成首前缀评分':`自身生成第 ${r.wave_progress.own_step} 步，当前波次已生成 ${r.wave_progress.generated_tokens} token`}`}
        {r.scoring_audit&&`；${r.scoring_audit.controlled_expressions} 条冻结评分精确复算，已停止但未解析 ${r.scoring_audit.EOS_unparsed_count} 条（不自动判为语义错误）`}
        {r.terminal_review&&`；另列事后非盲终结论核查 ${r.terminal_review.supplemental_terminal_correct_and_stopped}/320，冻结 ${r.terminal_review.primary_correct_and_stopped}/320 不回写，不评分整个推理链`}</li>)}</ul>
      <div className="prefix-controls"><label>自身历史模型<select aria-label="自身历史模型" value={model} onChange={e=>setModel(e.target.value)}>{['qwen4','qwen14','glm4'].map(v=><option key={v}>{v}</option>)}</select></label>
        <label>自身历史参数<select aria-label="自身历史参数" disabled={model!=='qwen4'} value={ownRun} onChange={e=>setVariant(e.target.value)}><option value="native">原生 BF16</option>{(data?.runs||[]).map(r=><option key={r.run}>{r.run}</option>)}</select></label>
        <label className="prefix-wide">自身历史表达<select aria-label="自身历史表达" value={os} onChange={e=>setOwnSample(e.target.value)}>{ownRows.map(r=><option key={r.sample_id} value={r.sample_id}>{r.family} · {r.full_hidden_collected?'含全H':'逐步postnorm'} · {r.sample_id}</option>)}</select></label>
        <label>自身历史场<select aria-label="自身历史场" value={ownField} onChange={e=>setOwnField(e.target.value)}><option value="postnorm">全部生成步 postnorm</option><option value="all_hidden">指定生成步的全部H</option></select></label>
        <label>自身生成步<input aria-label="自身生成步" type="number" min="0" max="255" value={step} onChange={e=>setStep(Number(e.target.value))}/></label>
        <button disabled={!os||Boolean(busy)} onClick={loadOwn}>{busy==='own'?'正在读取…':'读取真实自身历史'}</button></div>
      {!ownRows.length&&<p>该模型/参数组合尚无正式提交轨迹，不显示先导或其他组合来代替。</p>}
      {own&&<><blockquote>{own.record.actual_input}</blockquote><p>真实生成：{own.record.generated_text}</p><p>EOS：{String(own.record.EOS)}；达到上限：{String(own.record.censored)}；B1/B8首token：{own.record.first_shape.B1_argmax}/{own.record.first_shape.B8_argmax}</p><p>{own.boundary}</p></>}
      {own?.record.answer_scoring&&<aside data-testid="own-language-scoring">冻结解析答案：{own.record.answer_scoring.conservative_final_answer??'未解析（不能据此认定内容错误）'}；目标：{own.record.target}；冻结内容正确且停止：{String(own.record.answer_scoring.parsed_and_stopped_correct)}；严格只答目标格式：{String(own.record.answer_scoring.strict_answer_only)}。内容、格式和停止分别记录。</aside>}
      {own?.supplemental_terminal_review&&<aside data-testid="own-terminal-review">另列事后非盲终结论核查：{own.supplemental_terminal_review.supplemental_terminal_correct_and_stopped?'终结论正确且停止':'终结论错误'}；目标：{own.supplemental_terminal_review.target}。原文：{own.supplemental_terminal_review.exact_terminal_quote} 原冻结评分不变，严格只答目标格式仍不通过；不认定整个推理链正确。</aside>}
      <Field data={own}/>
    </section>
    <FormationProgramEvidence data={data?.program_own} refresh={refresh}/>
    {error&&<p role="alert" className="prefix-error">{error}</p>}
  </div>;
}
