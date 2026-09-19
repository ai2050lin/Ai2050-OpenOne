import {useEffect,useState} from 'react';
import {Field} from './RdcJointAtlas.jsx';

const API=import.meta.env.VITE_RDC_CONSTRUCTION_API||'http://127.0.0.1:5004/api/rdc-construction';
const BRANCHES={native:'原生读出',code_identity:'直接使用代码响应',mapped_code:'代码映射',shuffled_map:'错误配对映射',mapped_digit_bias:'映射 + 全数字偏置',mapped_letter_bias:'映射 + 全字母偏置'};
async function get(path,params){const r=await fetch(API+path+'?'+new URLSearchParams(params));if(!r.ok)throw new Error((await r.json()).detail||r.statusText);return r.json();}

export default function FormationProgramEvidence({data,refresh}){
  const [branch,setBranch]=useState('native'),[sample,setSample]=useState(''),[field,setField]=useState('postnorm');
  const [listing,setListing]=useState({branch:'',rows:[]}),[loaded,setLoaded]=useState(null),[error,setError]=useState(''),[busy,setBusy]=useState(false);
  useEffect(()=>{let alive=true;get('/formation-program-samples',{branch}).then(rows=>alive&&setListing({branch,rows})).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[branch,refresh]);
  const rows=listing.branch===branch?listing.rows:[];
  const sid=rows.some(r=>r.sample_id===sample)?sample:rows[0]?.sample_id||'';
  const id=JSON.stringify([branch,sid,field]),value=loaded?.id===id?loaded.value:null;
  async function load(){setBusy(true);try{setLoaded({id,value:await get('/formation-program-field',{branch,sample:sid,field})});setError('');}catch(e){setLoaded(null);setError(e.message);}finally{setBusy(false);}}
  return <section id="formation-program-view">
    <h3>文字—代码：一次读出替换后的真实自身历史</h3>
    <p>输入代码响应是额外已观察信息。六条路径均只在第一步调整读出，原生前缀和 KV 保持不变，后续使用各自生成的 token。数字概率、格式前缀、完整答案和停止分别判断。</p>
    <p>{data?.complete?'192条正式轨迹已完成':`已提交${data?.progress?.pilot?'先导':''} ${data?.progress?.completed||0} / ${data?.progress?.total||192} 条轨迹；正式查询只显示已提交的正式记录。`}</p>
    {!!data?.summary?.length&&<ul>{data.summary.map(r=><li key={r.branch}>{BRANCHES[r.branch]}：冻结解析正确且停止 {r.parsed_and_stopped_correct}/{r.groups}，EOS {r.EOS}，达到上限 {r.censored}</li>)}</ul>}
    {data?.terminal_review?.all_passed&&<p>补充事后非盲终值核查（主研究代理）：完整检查全部 {data.terminal_review.complete_residual_EOS_unparsed_set_size} 条已停止但冻结解析未识别的输出。
      直接代码路径的终值正确且停止为 {data.terminal_review.summary.find(r=>r.branch==='code_identity')?.supplemental_terminal_correct_and_stopped}/32；其余路径不变。原冻结评分和格式失败记录保留，补充核查不评判整条推理链。</p>}
    <div className="prefix-controls">
      <label>程序读出路径<select aria-label="程序读出路径" value={branch} onChange={e=>setBranch(e.target.value)}>{Object.entries(BRANCHES).map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label>
      <label className="prefix-wide">程序表达<select aria-label="程序表达" value={sid} onChange={e=>setSample(e.target.value)}>{rows.map(r=><option key={r.sample_id} value={r.sample_id}>深度 {r.depth} · {r.sample_id}</option>)}</select></label>
      <label>程序原生场<select aria-label="程序原生场" value={field} onChange={e=>setField(e.target.value)}><option value="postnorm">每步原生 postnorm</option><option value="first_readout">第一次实际读出向量</option><option value="first_hidden">第一步全部 H</option><option value="final_hidden">最后一步全部 H</option></select></label>
      <button disabled={!sid||busy} onClick={load}>{busy?'正在读取…':'读取程序自身历史'}</button>
    </div>
    {!rows.length&&<p>所选路径尚无正式记录；不以先导结果或其他路径代替。</p>}
    {value&&<><p>首次与原生 token 分叉：{value.record.first_divergence_from_native??'无'}；正确且停止：{String(value.record.answer_scoring.parsed_and_stopped_correct)}；EOS：{String(value.record.answer_scoring.EOS)}；达到上限：{String(value.record.answer_scoring.censored)}</p>
      <pre style={{whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{value.record.generated_text}</pre><p>{value.axes}</p><p>{value.boundary}</p></>}
    {value?.supplemental_terminal_review&&<aside data-testid="program-terminal-review"><p>本例事后非盲终值核查：末句明确给出正确请求变量值，已停止；冻结解析器未覆盖该表达，严格“只输出数字”的格式要求仍未通过。</p>
      <blockquote>{value.supplemental_terminal_review.exact_terminal_quote}</blockquote></aside>}
    <Field data={value}/>{error&&<p role="alert" className="prefix-error">{error}</p>}
  </section>;
}
